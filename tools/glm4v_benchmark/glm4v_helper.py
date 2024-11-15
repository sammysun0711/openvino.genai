from pathlib import Path
import types
from typing import Optional, Tuple, Union, List
import gc
import openvino as ov
from openvino.runtime import opset13
import nncf
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, TextIteratorStreamer
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from typing import Optional, Tuple, Union, List, Dict, Any
from transformers import __version__ as transformers_version
from transformers.generation.utils import  GenerationConfig, ModelOutput
import time
from threading import Thread, Event
import os
import shutil

def _chatglm_transformer_forward(
            self,
            input_ids: torch.LongTensor = None,
            images: torch.Tensor = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """take care of image_encode, position_ids and (attention_mask = None is fine)"""

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, seq_length, _ = inputs_embeds.shape

        if self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(batch_size=batch_size, device=input_ids.device,
                                                  dtype=inputs_embeds.dtype)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.pre_seq_len)),
                                            attention_mask], dim=-1)
        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                if self.training:
                    raise NotImplementedError("GLM4v edge series don't support training.")
                    # https://github.com/THUDM/GLM-4/issues/264
                    new_input_ids, new_attention_mask = [], []
                    for i in range(len(input_ids)):
                        input_id = input_ids[i].tolist()
                        boi_token_pos, eoi_token_pos = input_id.index(self.config.boi_token_id), input_id.index(self.config.eoi_token_id)
                        assert eoi_token_pos - boi_token_pos == 2

                        new_attention_mask.append(torch.cat(
                            (attention_mask[i, :boi_token_pos + 1], torch.ones(num_patches).to(attention_mask.device),
                             attention_mask[i, eoi_token_pos:])))

                        new_input_ids.append(torch.cat(
                            (input_ids[i, :boi_token_pos + 1], input_ids[i, -1].repeat(num_patches),
                             input_ids[i, eoi_token_pos:])))

                    attention_mask = torch.stack(new_attention_mask, dim=0)
                    input_ids = torch.stack(new_input_ids, dim=0)
                    inputs_embeds = self.embedding(input_ids)

                full_attention_mask = self.get_masks(inputs_embeds, past_key_values, padding_mask=attention_mask)
                
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)

        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]

        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
        )
        lm_logits = self.output_layer(hidden_states)
        lm_logits = lm_logits.to(torch.float32)
        # if not return_dict:
        output = (lm_logits,) + tuple(v for v in [presents, all_hidden_states, all_self_attentions] if v is not None)
        return output


def model_has_state(ov_model: ov.Model):
    return len(ov_model.get_sinks()) > 0


def model_has_input_output_name(ov_model: ov.Model, name: str):
    """
    Helper function for checking that model has specified input or output name

    Parameters:
      ov_model (ov.Model):
      name (str):
          name of input or output

    Returns:
      True if input or output with requested name exists else False
    """
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])


def fuse_cache_reorder(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    gather_dim: int,
):
    """
    Fuses reored_cache during generate cycle into ov.Model. Used with stateful models, because we can not modify model state directly.

    Adds a new beam_idx parameter and Gather op per each kv-cache input in a given model.
    Should be run before make_stateful. Implements optimumum's _reorder_cache
    inside the model in the beginning of each iteration.
    Gather works along given gather_dim dimension that may vary from model to model.
    KV-cache inputs are identified based on names in key_value_input_names.
    Append the new beam_idx parameter to not_kv_inputs.

    Parameters:
      ov_model (`ov.Model`):
          openvino model for processing
      not_kv_inputs (`List[str]`):
          list of input nodes in model that not related to past key values
      key_value_input_names (`List[str]`):
          list of names for key value input layers
      gather_dim (int):
          dimension for gathering cache during reorder pass
    """

    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("inputs_embeds").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})  # why list is not accepted?
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()


def build_state_initializer(ov_model: ov.Model, batch_dim: int):
    """
    Build initialization ShapeOf Expression for all ReadValue ops

    Parameters:
      ov_model (ov.Model):
          openvino model
      batch_dim (int):
          index of dimension corresponding to batch size
    """
    input_ids = ov_model.input("inputs_embeds")
    batch = opset13.gather(
        opset13.shape_of(input_ids, output_type="i64"),
        opset13.constant([0]),
        opset13.constant(0),
    )
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [(opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim) for dim in dims]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()


def make_stateful(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    key_value_output_names: List[str],
    batch_dim: int,
    num_attention_heads: int,
    num_beams_and_batch: int = None,
):
    """
    Hides kv-cache inputs and outputs inside the model as variables.

    Parameters:
        ov_model (ov.Model):
            openvino model
        not_kv_inputs (`List[str]`):
            list of input nodes in model that not related to past key values
        key_value_input_names (`List[str]`):
            list of names for key value input layers
        key_value_output_names (`List[str]`):
            list of names for key value input layers
        batch_dim (int):
            index of batch dimension in key value layers
        num_attention_heads (int):
            number of attention heads for batch dimension initialization
        num_beams_an_batch (int):
            precalculated number of beams and batch for shapes initialization
    """
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}

    if num_beams_and_batch is not None:
        # Set batch size for input_ids and attention mask to avoid dynamic dimension got propagated from the end of the model back to ReadValue
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:  # == 1 for beam_index
                shape[0] = num_beams_and_batch
                input.get_node().set_partial_shape(shape)
    for kv_name_pair in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_name_pair[0]] = kv_name_pair[1]
        if num_beams_and_batch is not None:
            input = ov_model.input(kv_name_pair[0])
            shape = input.get_partial_shape()
            shape[batch_dim] = num_beams_and_batch * num_attention_heads
            input.get_node().set_partial_shape(shape)

    if num_beams_and_batch is not None:
        # Re-validation model if shapes are altered above
        ov_model.validate_nodes_and_infer_types()

    apply_make_stateful_transformation(ov_model, input_output_map)
    if num_beams_and_batch is None:
        build_state_initializer(ov_model, batch_dim)


def patch_stateful(ov_model):
    key_value_input_names = [key.get_any_name() for key in ov_model.inputs[2:-1]]
    key_value_output_names = [key.get_any_name() for key in ov_model.outputs[1:]]
    not_kv_inputs = [input for input in ov_model.inputs if not any(name in key_value_input_names for name in input.get_names())]
    if not key_value_input_names or not key_value_output_names:
        return
    batch_dim = 0
    num_attention_heads = 1

    fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, batch_dim)
    make_stateful(
        ov_model,
        not_kv_inputs,
        key_value_input_names,
        key_value_output_names,
        batch_dim,
        num_attention_heads,
        None,
    )


core = ov.Core()


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()

def copy_all_needed_file(model_dir, output_dir):
    for filename in os.listdir(model_dir):
        # Check if the file ends with .py
        if filename.endswith('.py') or "tokenizer_config" in filename:
            # Construct full file path
            src_file = os.path.join(model_dir, filename)
            dest_file = os.path.join(output_dir, filename)
            # Copy the file
            shutil.copy(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")

def convert_glm4v_model(model_dir, output_dir, quantization_config):
    model_name = Path(model_dir).name
    output_dir = Path(output_dir)

    lang_model_path = output_dir / "language_model.xml"
    image_embed_path = output_dir / "image_embed.xml"
    embed_token_path = output_dir / "embed_token.xml"

    if all(
        [
            lang_model_path.exists(),
            image_embed_path.exists(),
            embed_token_path.exists(),
        ]
    ):
        print(f"✅ {model_name} model already converted. You can find results in {output_dir}")
        return
    print(f"⌛ {model_name} conversion started. Be patient, it may takes some time.")
    print("⌛ Load Original model")
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float32)
    model.config.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    copy_all_needed_file(model_dir, output_dir)

    print("✅ Original model successfully loaded")

    if not embed_token_path.exists():
        print("⌛ Convert Input embedding model")
        ov_model = ov.convert_model(
            model.transformer.embedding,
            example_input=torch.ones([1, 10], dtype=torch.int64),
        )
        ov.save_model(ov_model, embed_token_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Input embedding model successfully converted")

    vision_embed_tokens = model.transformer.vision
    if not image_embed_path.exists():
        print("⌛ Convert Image embedding model")
        # vision_embed_tokens.forward = vision_embed_tokens.vit
        ov_model = ov.convert_model(vision_embed_tokens, example_input=torch.ones([1, 3, 672, 672]))
        ov.save_model(ov_model, image_embed_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("✅ Image embedding model successfully converted")

    if not lang_model_path.exists():
        print("⌛ Convert Language model")
        
        def _glm4_core_attention_forward(self, query_layer, key_layer, value_layer, attention_mask):
            causal_mask = torch.zeros_like(attention_mask, dtype=torch.float32)
            causal_mask.masked_fill_(attention_mask, float("-inf"))
            context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer, causal_mask)
            context_layer = context_layer.transpose(1, 2).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.reshape(*new_context_layer_shape)
            return context_layer
        
        
        input_ids = torch.zeros([2, 2], dtype=torch.int64)
        inputs_embeds = torch.zeros([2, 2,2048], dtype=torch.float32)
        
        
        pkv = model.transformer(
            input_ids=input_ids,
            attention_mask=torch.ones((2, 2), dtype=torch.int64),
        )[1]
        
        model.transformer._orig_forward = model.transformer.forward
        model.transformer.forward = types.MethodType(_chatglm_transformer_forward, model.transformer)
        for block in model.transformer.encoder.layers:
            block.self_attention.core_attention._orig_forward = block.self_attention.core_attention.forward
            block.self_attention.core_attention.forward = types.MethodType(_glm4_core_attention_forward,
                block.self_attention.core_attention,
            )
        model.transformer.config.torchscript = True
        model_inputs = ["position_ids","attention_mask"]
        model_outputs = ["logits"]
        for idx in range(len(pkv)):
            model_inputs.extend([f"past_key_values.{idx}.key", f"past_key_values.{idx}.value"])
            model_outputs.extend([f"present.{idx}.key", f"present.{idx}.value"])
        model_inputs.append("inputs_embeds")
        position_ids = torch.tensor([[2, 3], [2, 3]])
        input_shape = ov.PartialShape([-1, -1, -1])
        ov_model = ov.convert_model(
            model.transformer,
            example_input={
                "inputs_embeds": inputs_embeds,
                "position_ids": position_ids,
                "attention_mask": torch.ones([2, 4], dtype=torch.int64),
                "past_key_values": pkv,
            },
        )

        for input, input_name in zip(ov_model.inputs, model_inputs):
            input.get_tensor().set_names({input_name})

        for output, output_name in zip(ov_model.outputs, model_outputs):
            output.get_tensor().set_names({output_name})
        patch_stateful(ov_model)
        print("✅ Language model successfully converted")

        if quantization_config is not None:
            print(f"⌛ Weights compression with {quantization_config['mode']} mode started")
            ov_model = nncf.compress_weights(ov_model, **quantization_config)
            print("✅ Weights compression finished")

        ov.save_model(ov_model, lang_model_path)
        del ov_model
        cleanup_torchscript_cache()
        del model
        gc.collect()
        print(f"✅ {model_name} model conversion finished. You can find results in {output_dir}")


def is_empty(images_list: Optional[List[List[torch.Tensor]]]):
    if images_list is None or len(images_list) == 0:
        return True
    for image_list in images_list:
        if image_list is not None:
            return False
    return True

class OvGLM4v(GenerationMixin):
    def __init__(self, model_dir, device, llm_times=[], image_embed_t=[], embed_token_t=[]):
        model_dir = Path(model_dir)
        self.model = core.read_model(model_dir / "language_model.xml")
        self.image_embed = core.compile_model(model_dir / "image_embed.xml", device)
        self.embed_token = core.compile_model(model_dir / "embed_token.xml", device)
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}
        # compiled_model = core.compile_model(self.model, device, config={"INFERENCE_PRECISION_HINT": "fp16"})
        compiled_model = core.compile_model(self.model, device)

        self.request = compiled_model.create_infer_request()
        self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.main_input_name = "input_ids"
        self.device = torch.device("cpu")
        self.num_pkv = 2
        self._supports_cache_class = False
        self.next_beam_idx = None
        self._past_length = None
        self.hd_transform_order = "glb_sub"

        self.llm_times = llm_times
        self.image_embed_t = image_embed_t
        self.embed_token_t = embed_token_t
        
    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True

    def __call__(
        self,
        input_ids: torch.LongTensor = None,
        images: torch.Tensor = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return self.forward(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: torch.Tensor = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """take care of image_encode, position_ids and (attention_mask = None is fine)"""
        # generate mode with past_key_values. the image features are already mapped
        if past_key_values is None:
            # not allow for inputs_embeds, because we want to process image feature
            assert input_ids is not None and inputs_embeds is None, f"{input_ids} {inputs_embeds}"
            self.request.reset_state()
            self.llm_times.clear()
            self.embed_token_t.clear()

            self.next_beam_idx = np.arange(input_ids.shape[0], dtype=int)
            self._past_length = 0
            if not is_empty(images):  # multi-modality
                image_size: int = self.config.vision_config['image_size']
                patch_size: int = self.config.vision_config['patch_size']
                num_patches = (image_size // patch_size // 2) ** 2
                assert len(input_ids) == len(images), f"{len(input_ids)} {len(images)}"

                start = time.perf_counter()
                inputs_embeds = torch.from_numpy(self.embed_token(input_ids)[0])
                self.embed_token_t.append((time.perf_counter() - start)*1000)
                
                images = images.to(dtype=inputs_embeds.dtype)
                start = time.perf_counter()
                images_features = torch.from_numpy(self.image_embed(images)[0])
                self.image_embed_t.append((time.perf_counter() - start)*1000)
                if position_ids is None:
                    position_ids = self.get_position_ids(input_ids, device=inputs_embeds.device)
                new_input_embeds, new_position_ids = [], []

                for i in range(len(input_ids)):
                    input_id = input_ids[i].tolist()
                    boi_token_pos, eoi_token_pos = input_id.index(self.config.boi_token_id), input_id.index(
                        self.config.eoi_token_id)
                    assert eoi_token_pos - boi_token_pos == 2
                    new_input_embeds.append(torch.cat(
                        (inputs_embeds[i, :boi_token_pos], images_features[i].to(inputs_embeds.device),
                         inputs_embeds[i, eoi_token_pos + 1:])))
                    new_position_ids.append(torch.arange(new_input_embeds[-1].shape[0]))
                inputs_embeds = torch.stack(new_input_embeds, dim=0)
                position_ids = torch.stack(new_position_ids, dim=0)
        if inputs_embeds is None:
            start = time.perf_counter()
            inputs_embeds = self.embed_token(input_ids)[0]
            self.embed_token_t.append((time.perf_counter() - start)*1000)
        inputs = {}
        inputs["inputs_embeds"] = inputs_embeds
        inputs["attention_mask"] = attention_mask
        inputs["position_ids"] = position_ids
        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = self.next_beam_idx if self.next_beam_idx is not None else np.arange(inputs_embeds.shape[0], dtype=int)

        start = time.perf_counter()
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        self.llm_times.append((time.perf_counter() - start)*1000)

        logits = self.request.get_tensor("logits").data
        logits = torch.from_numpy(logits).to(self.device)
        past_key_values = ((),)
        self._past_length += inputs["inputs_embeds"].shape[1]

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def _reorder_cache(self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """
        self.next_beam_idx = np.array(beam_idx)  # save beam_idx to be used as an input in the next iteration
        return past_key_values

    def _get_past_length(self, past_key_values=None):
        if past_key_values is None:
            return 0
        return self._past_length
    
    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        if int(transformers_version.split('.')[1]) >= 44:
            assert not standardize_cache_format
            _, cache = self._extract_past_from_model_output(
                outputs
            )
            model_kwargs['past_key_values'] = cache
        else:
            cache = self._extract_past_from_model_output(
                outputs, standardize_cache_format
            )

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_id], dim=-1
            )

        model_kwargs["is_first_forward"] = False
        return model_kwargs
    
    def chat_stream(self, image, query, tokenizer, generate_kwargs):
        #prompt=text
        inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True, add_special_tokens=False)  # chat mode
        inputs = inputs.to("cpu")

        #self.model_inputs = self.tokenizer(prompt, return_tensors="pt")
        #self.model_inputs.pop("token_type_ids", None)
        streamer = TextIteratorStreamer(tokenizer,
                                        skip_prompt=True,
                                        skip_special_tokens=True)
        """
        generate_kwargs = dict(self.model_inputs,
                               streamer=streamer,
                               max_new_tokens=self.args.max_new_tokens,
                               do_sample=self.args.do_sample,
                               top_p=self.args.top_p,
                               temperature=self.args.temperature,
                               top_k=self.args.top_k,
                               repetition_penalty=self.args.repetition_penalty,
                               eos_token_id=self.tokenizer.eos_token_id)
        """
        generate_kwargs["streamer"]=streamer
        stream_complete = Event()

        def generate_and_signal_complete():
            """
            genration function for single thread
            """
            self.generate(**inputs, **generate_kwargs)
            stream_complete.set()

        # t = Thread(target=self.ov_model.generate, kwargs=generate_kwargs)
        t = Thread(target=generate_and_signal_complete)
        t.start()

        for new_text in streamer:
            yield new_text
            
    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            images: Optional[torch.Tensor] = None,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            is_first_forward: bool = True,
            **kwargs
    ) -> dict:
        # only last token for input_ids if past is not None
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids, device=input_ids.device)
        if attention_mask is not None:
            image_size: int = self.config.vision_config['image_size']
            patch_size: int = self.config.vision_config['patch_size']
            num_patches = (image_size // patch_size // 2) ** 2
            new_attention_masks = []

            # if not image, use this default id
            eoi_token_pos = 2
            boi_token_pos = 0

            for i in range(len(input_ids)):
                input_id = input_ids[i].tolist()
                if not is_empty(images):
                    boi_token_pos, eoi_token_pos = input_id.index(self.config.boi_token_id), input_id.index(
                        self.config.eoi_token_id)
                assert eoi_token_pos - boi_token_pos == 2
                new_attention_masks.append(torch.cat(
                    (attention_mask[i, :boi_token_pos + 1], attention_mask.new_ones(num_patches),
                     attention_mask[i, eoi_token_pos:])
                ))
            attention_mask = torch.stack(new_attention_masks, dim=0)
        if not is_first_forward:
            if past_key_values is not None:
                position_ids = position_ids[..., -1:]
                input_ids = input_ids[:, -1:]
                
        return {
            "input_ids": input_ids,
            "images": images,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def vision_embed_tokens(
        self,
        image: torch.LongTensor,
    ) -> torch.FloatTensor:
        vit_output = torch.from_numpy(self.image_embed(image)[0])
        return torch.from_numpy(self.img_projection(vit_output)[0])
        