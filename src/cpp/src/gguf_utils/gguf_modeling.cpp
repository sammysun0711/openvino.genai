// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <stdexcept>

#include <openvino/openvino.hpp>
#include "openvino/runtime/core.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/genai/tokenizer.hpp"

#include "gguf_utils/building_blocks.hpp"
#include "gguf_utils/gguf_modeling.hpp"


using namespace ov;
using namespace ov::op::v13;
using namespace ov::op;

namespace {

auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

// Also valid for other models, e.g. SmolLMs
// CVS-166108: Adding shared_embedding as true by default based on following two reason:
// 1. For optimum-cli converted LLM OpenVINO IR, original input embedding weight will be reused for last make_lm_head layer
//    Which can reduce both model size on disk and runtime memory usage via storing only single embeddding consts
//    (e.g. Qwen2.5-7B-Instruct-Q4_0 token_embd.weight & output.weight shape [3584, 152064]
// 2. For some GGUF model that contains both token_embd.weight & output.weight, e.g. Qwen2.5-3B-Instruct Q4_0
//    meet accuracy issue on MTL/LNL GPU due to use both token_embd.weight & output.weight in OpenVINO IR.
// WA Known issue: Qwen2.5-3B-Instruct-Q4_K_M meet accuracy issue on MTL/LNL CPU if only re-used token_embd.weight

std::shared_ptr<ov::Model> create_language_model(
    const std::map<std::string, GGUFMetaData>& configs,
    std::unordered_map<std::string, ov::Tensor>& consts,
    std::unordered_map<std::string, gguf_tensor_type>& qtypes,
    bool shared_embedding = false) {
    // Create input parameters
    auto input_ids = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    set_name(input_ids, "input_ids");

    auto attention_mask = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    set_name(attention_mask, "attention_mask");

    auto position_ids = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::PartialShape{-1, -1});
    set_name(position_ids, "position_ids");

    auto beam_idx = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i32, ov::PartialShape{-1});
    set_name(beam_idx, "beam_idx");

    // Create embedding layer
    auto [inputs_embeds, embeddings] = make_embedding(
        "model.embed_tokens",
        input_ids->output(0),
        consts,
        qtypes.at("model.embed_tokens.qtype"));

    auto hidden_states = inputs_embeds;

    // Initialize RoPE
    auto rope_const = init_rope(
        std::get<int>(configs.at("head_size")),
        std::get<int>(configs.at("max_position_embeddings")),
        std::get<float>(configs.at("rope_freq_base")));

    // Get input shape components
    auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(input_ids);
    auto batch_axis = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{1}, 0);
    auto batch_size = std::make_shared<ov::op::v8::Gather>(
        input_shape, batch_axis, batch_axis);

    auto hidden_dim = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{1}, 3);

    // Process layers
    ov::SinkVector sinks;
    ov::Output<ov::Node> causal_mask;
    std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> cos_sin_cached;
    std::shared_ptr<ov::Node> output_shape = nullptr;

    for (int i = 0; i < std::get<int>(configs.at("layer_num")); ++i) {
        auto [new_hidden, layer_sinks, new_mask, new_cos_sin, new_shape] = layer(
            configs,
            consts,
            qtypes,
            i,
            hidden_states,
            attention_mask,
            causal_mask,
            position_ids,
            rope_const,
            beam_idx,
            batch_size,
            hidden_dim,
            cos_sin_cached,
            output_shape);

        hidden_states = new_hidden;
        causal_mask = new_mask;
        cos_sin_cached = new_cos_sin;
        output_shape = new_shape;

        sinks.insert(sinks.end(), layer_sinks.begin(), layer_sinks.end());
    }

    // Final layer norm
    auto final_norm = make_rms_norm(
        "model.norm",
        hidden_states,
        consts,
        std::get<float>(configs.at("rms_norm_eps")));

    // LM head
    auto embed_out = make_lm_head(
        "lm_head",
        final_norm,
        consts,
        embeddings,
        qtypes.at("lm_head.qtype"),
        shared_embedding);

    // Create results
    auto logits = std::make_shared<ov::op::v0::Result>(embed_out);
    set_name(logits, "logits");

    // Create model
    ov::ParameterVector inputs{input_ids, attention_mask, position_ids, beam_idx};
    auto model = std::make_shared<ov::Model>(ov::OutputVector({logits->output(0)}), sinks, inputs);

    // Set runtime options
    if (std::get<int>(configs.at("file_type")) == 1 || std::get<int>(configs.at("file_type")) == 0) {
        model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    }
    model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});
    // CVS-166554: Dynamic quatnization enabled by default with gourp size 32 on MTL platfrom cause the runtime issue
    // Apply WA to disable dynamic quantization with rt_info to fix GPU plugin issue on MTL
    model->set_rt_info(0, {"runtime_options", ov::hint::dynamic_quantization_group_size.name()});

    return model;
}

} // namespace

ov::Core core_with_extension() {
    ov::Core core;
    const char* ov_tokenizer_path = "/home/openvino/workspaces/AIGC/openvino.genai/build/openvino_genai/libopenvino_tokenizers.so";
    OPENVINO_ASSERT(ov_tokenizer_path, "openvino_tokenizers path is not set");
    core.add_extension(ov_tokenizer_path);
    return core;
}

ov::Core get_core_singleton() {
    static ov::Core core = core_with_extension();
    return core;
}
/*
def add_ragged_dimension(input_node: list[ov.Output]) -> list[ov.Output]:
    shape = opset.shape_of(input_node[0])
    batch_size = opset.gather(shape, as_node(0), as_node(0))
    ragged_begins = opset.range(as_node(0), batch_size, as_node(1), output_type="i32").outputs()
    ragged_ends = opset.range(
        as_node(1), opset.add(batch_size, make_constant_node(1, Type.i64)), as_node(1), output_type="i32"
    ).outputs()
    return ragged_begins + ragged_ends + input_node
*/
/*
std::tuple<ov::Output<ov::Node>, ov::Output<ov::Node>> add_ragged_dimension(const std::vector<ov::Output<ov::Node>>& input_node) {
    auto shape = std::make_shared<ov::op::v3::ShapeOf>(input_node[0]);
    auto batch_size = std::make_shared<ov::op::v8::Gather>(shape, ov::op::v0::Constant(ov::element::i64, {1}, 0), ov::op::v0::Constant(ov::element::i64, {1}, 0));
    auto ragged_begins = std::make_shared<ov::op::v4::Range>(ov::op::v0::Constant(ov::element::i32, {1}, 0), batch_size, ov::op::v0::Constant(ov::element::i32, {1}, 1));
    auto ragged_ends = std::make_shared<ov::op::v4::Range>(ov::op::v0::Constant(ov::element::i32, {1}, 1), std::make_shared<ov::op::v1::Add>(batch_size, ov::op::v0::Constant(ov::element::i32, {1}, 1)), ov::op::v0::Constant(ov::element::i32, {1}, 1));

    return {ragged_begins->outputs()[0], ragged_ends->outputs()[0]};
}
*/

std::vector<ov::Output<ov::Node>> add_ragged_dimension(const std::vector<ov::Output<ov::Node>>& input_node) {
    std::vector<ov::Output<ov::Node>> output_nodes;
    auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(input_node[0]);
    auto const_0 = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{}, 0);
    auto const_1 = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{}, 0);
    auto batch_size = std::make_shared<ov::op::v8::Gather>(
        input_shape, const_0, const_0);

    auto ragged_begins = std::make_shared<ov::op::v4::Range>(const_0, batch_size, const_1, ov::element::i32);
    auto ragged_add = std::make_shared<ov::op::v1::Add>(batch_size, const_1);
    auto ragged_ends = std::make_shared<ov::op::v4::Range>(const_1, ragged_add, const_1, ov::element::i32);

    output_nodes.push_back(ragged_begins->outputs()[0]);
    output_nodes.push_back(ragged_ends->outputs()[0]);
    
    for (auto& node : input_node) {
        output_nodes.push_back(node);
    }
    
    return output_nodes;
}

bool is_special(int token_type){
    return token_type == 3 || token_type == 4;
}

std::shared_ptr<ov::Model> create_tokenizer_model(
    const std::map<std::string, GGUFMetaData>& configs,
    std::unordered_map<std::string, ov::Tensor>& consts) { 
    std::cout << "Create tokenizer model called\n";
    std::string model_type = std::get<std::string>(configs.at("tokenizer.model"));
    std::cout << "model_type " << model_type << "\n";
    std::string pre = std::get<std::string>(configs.at("tokenizer.pre"));
    std::cout << "pre " << pre << "\n";
    int eos_token_id = std::get<int>(configs.at("tokenizer.eos_token_id"));
    std::cout << "eos_token_id " << eos_token_id << "\n";
    int padding_token_id = std::get<int>(configs.at("tokenizer.padding_token_id"));
    std::cout << "padding_token_id " << padding_token_id << "\n";
    int bos_token_id = std::get<int>(configs.at("tokenizer.bos_token_id"));
    std::cout << "bos_token_id " << bos_token_id << "\n";
    auto add_bos_token = configs.at("tokenizer.add_bos_token");
    //std::cout << "add_bos_token " << add_bos_token << "\n";
    std::string chat_template = std::get<std::string>(configs.at("tokenizer.chat_template"));
    std::cout << "chat_template " << chat_template << "\n";

    std::vector<std::string> tokens = std::get<std::vector<std::string>>(configs.at("tokenizer.tokens"));
    std::cout << "tokens[0]: " << tokens[0] << "\n";

    //std::vector<int> tokens_type = std::get<std::vector<int>>(configs.at("tokenizer.tokens_type"));
    //std::cout << "token_type[0]" << token_type[0] << "\n";

    std::vector<std::string> merges = std::get<std::vector<std::string>>(configs.at("tokenizer.merges"));
    std::cout << "merges[0]: " << merges[0] << "\n";

    std::cout << "Extract meta data finished\n";

    ov::Core core = core_with_extension();

    // 1 string tensor
    // tokenizer_inputs = [ov.op.Parameter(Type.string, ov.PartialShape(["?"]))]
    auto tokenizer_inputs = std::make_shared<ov::op::v0::Parameter>(
        ov::element::string, ov::PartialShape{-1});
    set_name(tokenizer_inputs, "tokenizer_inputs");

    // 3 tensors: begins[i32], ends[i32], chars[u8]
    // outputs = opset.string_tensor_unpack(tokenizer_inputs[0]).outputs()
    auto string_tensor_unpacked = std::make_shared<ov::op::v15::StringTensorUnpack>(
        tokenizer_inputs
    )->outputs();

    std::cout << "outputs size: " << string_tensor_unpacked.size() << "\n";
    std::cout << "outputs[0] type: " << string_tensor_unpacked[0].get_element_type() << "\n";
    std::cout << "outputs[0] shape: " << string_tensor_unpacked[0].get_partial_shape() << "\n";
    std::cout << "outputs[1] type: " << string_tensor_unpacked[1].get_element_type() << "\n";
    std::cout << "outputs[1] shape: " << string_tensor_unpacked[1].get_partial_shape() << "\n";
    std::cout << "outputs[2] type: " << string_tensor_unpacked[2].get_element_type() << "\n";
    std::cout << "outputs[2] shape: " << string_tensor_unpacked[2].get_partial_shape() << "\n";

    // 5 tensors: ragged_begins[i32], ragged_ends[i32], begins[i32], ends[i32], chars[u8]
    // outputs = add_ragged_dimension(outputs)
    // auto [ragged_begin, ragged_end] = add_ragged_dimension(outputs->outputs());
    auto ragged_output = add_ragged_dimension(string_tensor_unpacked);
    std::cout << "ragged_output size: " << ragged_output.size() << "\n";
    std::cout << "outputs[0] type: " << ragged_output[0].get_element_type() << "\n";
    std::cout << "outputs[0] shape: " << ragged_output[0].get_partial_shape() << "\n";
    std::cout << "outputs[1] type: " << ragged_output[1].get_element_type() << "\n";
    std::cout << "outputs[1] shape: " << ragged_output[1].get_partial_shape() << "\n";
    std::cout << "outputs[2] type: " << ragged_output[2].get_element_type() << "\n";
    std::cout << "outputs[2] shape: " << ragged_output[2].get_partial_shape() << "\n";
    std::cout << "outputs[3] type: " << ragged_output[3].get_element_type() << "\n";
    std::cout << "outputs[3] shape: " << ragged_output[3].get_partial_shape() << "\n";
    std::cout << "outputs[4] type: " << ragged_output[4].get_element_type() << "\n";
    std::cout << "outputs[4] shape: " << ragged_output[4].get_partial_shape() << "\n";
    /*
    special_tokens = [
        token
        for token, token_type in zip(tokenizer_config["tokens"], tokenizer_config["token_type"])
        if is_special(token_type)
    ]
    special_tokens_re = "|".join(special_tokens)
    */
    /*
    std::vector<std::int> special_tokens;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (is_special(tokens_type[i])) {
            special_tokens.push_back(tokens[i]);
        }
    }
    */
    return nullptr;
}

std::shared_ptr<ov::Model> create_from_gguf(const std::string& model_path) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Loading and unpacking model from: " << model_path << std::endl;
    auto [config, consts, qtypes] = load_gguf(model_path);
    auto load_finish_time = std::chrono::high_resolution_clock::now();
    std::cout << "Loading and unpacking model done. Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(load_finish_time - start_time).count() << "ms" << std::endl;
    std::cout << "Start generating OV model..." << std::endl;
    
    std::shared_ptr<ov::Model> model;
    std::shared_ptr<ov::Model> tokenizer;

    const std::string model_arch = std::get<std::string>(config.at("architecture"));
    if (!model_arch.compare("llama") || !model_arch.compare("qwen2")) {
        model = create_language_model(config, consts, qtypes);
        tokenizer = create_tokenizer_model(config, consts);
    } else {
        OPENVINO_THROW("Unsupported model architecture '", model_arch, "'");
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - load_finish_time).count();
    std::cout << "Model generation done. Time: " << duration << "ms" << std::endl;
    return model;
}
