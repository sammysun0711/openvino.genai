import argparse
import math
import logging
import time
from pathlib import Path
from typing import Union, Dict, Any, List

import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from tqdm.autonotebook import trange
from transformers import AutoTokenizer, AutoModel, AutoConfig
import openvino as ov
from openvino.runtime.passes import Manager, MatcherPass, WrapType, Matcher
from openvino.runtime import opset10 as ops
from openvino.runtime import Core, Type, get_version
from openvino_tokenizers import convert_tokenizer
from openvino import convert_model
from openvino.preprocess import PrePostProcessor

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.pydantic_v1 import BaseModel, Extra, Field
from langchain.schema.embeddings import Embeddings


logger = logging.getLogger(__name__)

DEFAULT_QUERY_BGE_INSTRUCTION_EN = (
    "Represent this question for searching relevant passages: "
)
DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："

packed_layername_tensor_dict_list = [{"name": "aten::mul/Multiply"}]


class ReplaceTensor(MatcherPass):
    def __init__(self, packed_layername_tensor_dict_list):
        MatcherPass.__init__(self)
        self.model_changed = False

        param = WrapType("opset10.Multiply")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            if root is None:
                return False
            for y in packed_layername_tensor_dict_list:
                root_name = root.get_friendly_name()
                if root_name.find(y["name"]) != -1:
                    max_fp16 = np.array([[[[-np.finfo(np.float16).max]]]]).astype(np.float32)
                    new_tenser = ops.constant(max_fp16, Type.f32, name="Constant_4431")
                    root.set_arguments([root.input_value(0).node, new_tenser])
                    packed_layername_tensor_dict_list.remove(y)

            return True

        self.register_matcher(Matcher(param, "ReplaceTensor"), callback)


class OVBgeEmbeddings(BaseModel, Embeddings):
    """
    OpenVINO BGE embedding models.
    modify from OpenVINOBgeEmbeddings of langchain_community.embeddings 
    """

    ov_model: Any
    """OpenVINO model object."""
    tokenizer: Any
    """Tokenizer for embedding model."""
    model_dir: str
    """Path to store models."""
    model_kwargs: Dict[str, Any]
    """Keyword arguments passed to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method of the model."""
    query_instruction: str = DEFAULT_QUERY_BGE_INSTRUCTION_EN
    """Instruction to use for embedding query."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)
        ov_config = {"CACHE_DIR": "model_cache"}
        core =ov.Core()
        model_path = Path(fp16_model_dir) / "openvino_model.xml"
        self.ov_model = core.compile_model(model_path, self.model_kwargs["device"], ov_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        
        if "-zh" in self.model_dir:
            self.query_instruction = DEFAULT_QUERY_BGE_INSTRUCTION_ZH

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  # Object has no len() method
            return 1
        # Empty string or list of ints
        elif len(text) == 0 or isinstance(text[0], int):
            return len(text)
        else:
            # Sum of length of individual strings
            return sum([len(t) for t in text])

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 4,
        show_progress_bar: bool = None,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = False,
        max_length: int = 512,
    ) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings.

        :param sentences: the sentences to embed.
        :param batch_size: the batch size used for the computation.
        :param show_progress_bar: Whether to output a progress bar when encode sentences.
        :param convert_to_numpy: Whether the output should be a list of numpy vectors. If False, it is a list of PyTorch tensors.
        :param convert_to_tensor: Whether the output should be one large tensor. Overwrites `convert_to_numpy`.
        :param normalize_embeddings: Whether to normalize returned vectors to have length 1. In that case,
            the faster dot-product (util.dot_score) instead of cosine similarity can be used.
        :param max_length: max length used to truncate input with tokenizer

        :return: By default, a 2d numpy array with shape [num_inputs, output_dimension] is returned. If only one string
            input is provided, then the output is a 1d array with shape [output_dimension]. If `convert_to_tensor`, a
            torch Tensor is returned instead.
        """

        if convert_to_tensor:
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort(
            [-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            features = self.tokenizer(
                sentences_batch, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
            # print(features)
            input_ft = features['input_ids'].numpy()
            token_type_ids = features['token_type_ids'].numpy()
            attention_mask = features['attention_mask'].numpy()
            inputs={"input_ids":input_ft, "token_type_ids":token_type_ids, "attention_mask":attention_mask}

            # print("OV model raw input ", input_ft)
            out_features = self.ov_model(inputs)
            # print("OV model raw outputs:  ", out_features)
            embeddings = out_features[0][:, 0]
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(
                    torch.from_numpy(embeddings), p=2, dim=1)

            # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
            if convert_to_numpy:
                embeddings = embeddings.cpu()

            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx]
                          for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            if len(all_embeddings):
                all_embeddings = torch.stack(all_embeddings)
            else:
                all_embeddings = torch.Tensor()
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy()
                                        for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = [t.replace("\n", " ") for t in texts]
        embeddings = self.encode(texts, **self.encode_kwargs)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        start = time.time()
        text = text.replace("\n", " ")
        embedding = self.encode(
            self.query_instruction + text, **self.encode_kwargs
        )
        print(f"openvino embeddings time cost is: {time.time()-start} ms")
        return embedding.tolist()


def convert_to_fp16(fp16_model_dir, model_name_or_path):

    config = AutoConfig.from_pretrained(model_name_or_path)
    print(config)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path, config = config)


    # 3 inputs with dynamic axis [batch_size, sequence_length] and type int64
    inputs_info = [([1, 512], ov.Type.i64)] * 3
    ov_model = convert_model(
         model,
         example_input=tokenizer("test", return_tensors="pt").data,
         input=inputs_info,
    )

    static_shape = [1,512]
    # change dynamic shape to static shape
    name_to_shape = dict()
    for input_obj in ov_model.inputs:
        # shape = input_obj.get_partial_shape()            
        # input may have no name, in such case use map based on input index or port instead
        if len(input_obj.get_names()) != 0:
            name_to_shape[input_obj.get_any_name()] = static_shape
    ov_model.reshape(name_to_shape)

    # avoid generating nan, copy from notebook llm-rag-langchain
    manager = Manager()
    manager.register_pass(ReplaceTensor(packed_layername_tensor_dict_list))
    manager.run_passes(ov_model)

    ov.save_model(ov_model, Path(fp16_model_dir) / "openvino_model.xml", compress_to_fp16=True)
    
    tokenizer.save_pretrained(fp16_model_dir)
    ov_tokenizer = convert_tokenizer(tokenizer, with_detokenizer=False)
    ov.save_model(ov_tokenizer, Path(fp16_model_dir) / "openvino_tokenizer.xml")


fp16_model_dir = "bge-small-zh-v1.5" 
model_name_or_path = "BAAI/bge-small-zh-v1.5"

if ((Path(fp16_model_dir) / "openvino_model.xml").is_file()):
    print("--- bge-small-zh-v1.5 exists")
else:
    print("--- Convert pytorch model to OpenVINO FP16 model")
    convert_to_fp16(fp16_model_dir, model_name_or_path)

print("--- Run model on NPU and compare PT embedding results with OV embeddings results")
# # Init Pytorch BGE Embeddings w/ Langchain
do_norm = True

pt_embeddings = HuggingFaceBgeEmbeddings(model_name = model_name_or_path,
                                                model_kwargs={'device': 'cpu'}
                                                ,encode_kwargs={'normalize_embeddings': do_norm})

queries = ["您好", 
           "请介绍下清华大学", 
           "晚上睡不着怎么办？",
           "若我有一亿美元，在人工智能盛行的今天，我怎样投资才能收益最大化？",
           "糕点商店里原本有三种蛋糕：草莓奶油蛋糕，巧克力椰蓉蛋糕，和红丝绒布朗尼蛋糕。如名字所描述的那样，每种蛋糕都有两种成分：草莓奶油蛋糕包含草莓和奶油两个成分，巧克力椰蓉蛋糕包含巧克力和椰蓉两种成分，红丝绒布朗尼蛋糕包含红丝绒和布朗尼两种成分。在蛋糕制作完成后，往往每一种成分的材料都会有所剩余。为了减少浪费，商店常常会把多出来的成分两两搭配，做成新的小商品卖出去。比如草莓和巧克力可以做成草莓味巧克力酱，布朗尼和椰蓉可以做成布朗尼椰蓉饼干。以此类推可知，如果所有的成分都可以两两组合，那么最终商店能做出哪些小商品出来？",
           "桌子有左中右3个抽屉；张三，李四，王五，赵六都看到桌子上有一袋巧克力。张三让李四和王五出门后，在赵六面前把这袋巧克力放进了右抽屉；王五回来后，张三让赵六出门去找李四，并在王五面前从左抽屉拿出一盒饼干放进中抽屉里；等李四和赵六返回，张三又让王五和赵六出去买酱油，等二人走后，他告诉李四刚才已将一盒饼干放进中抽屉；张三等了很久，发现王五和赵六还没回来，就派李四去寻找，可最后只有王五和李四回来了。王五告诉张三，一开始他们没有找到卖酱油的店，所以只好分头去买，后来赵六走丢了；回来的路上，王五碰上了李四，两人便先赶了回来。于是，张三让两人出门去找赵六；为防再次走丢，张三叮嘱李四和王五要时刻同行，就算酱油买不到，也要找回赵六。结果，李四和王五在外面找到了赵六，发现他已经买了酱油。三人觉得张三从来不出门跑腿，十分气愤，讨论并达成共识，回去见到张三后，不要告诉他买到了酱油的事情，并让王五把酱油藏到自己的背包里。等三人一同回来后，他们按照计划谎称没有买到酱油，并希望张三以后买东西也要一同出门，不能偷懒，张三答应了。当大家最后站在桌子前，四人分别写下自己知道的物品清单和物品所在位置。问，这四人写下的物品和位置信息是否一致，为什么？",
           "折纸的过程看似简单，其实想要做好，还是需要一套很复杂的工艺。以折一支玫瑰花为例，我们可以将整个折纸过程分成三个阶段，即：创建栅格折痕，制作立体基座，完成花瓣修饰。首先是创建栅格折痕：这一步有点像我们折千纸鹤的第一步，即通过对称州依次对折，然后按照长和宽两个维度，依次进行多等分的均匀折叠；最终在两个方向上的折痕会交织成一套完整均匀的小方格拼接图案；这些小方格就组成了类似二维坐标系的参考系统，使得我们在该平面上，通过组合临近折痕的方式从二维小方格上折叠出三维的高台或凹陷，以便于接下来的几座制作过程。需要注意的是，在建立栅格折痕的过程中，可能会出现折叠不对成的情况，这种错误所带来的后果可能是很严重的，就像是蝴蝶效应，一开始只是毫厘之差，最后可能就是天壤之别。然后是制作立体基座：在这一步，我们需要基于栅格折痕折出对称的三维高台或凹陷。从对称性分析不难发现，玫瑰花会有四个周对称的三维高台和配套凹陷。所以，我们可以先折出四分之一的凹陷和高台图案，然后以这四分之一的部分作为摸板，再依次折出其余三个部分的重复图案。值得注意的是，高台的布局不仅要考虑长和宽这两个唯独上的规整衬度和对称分布，还需要同时保证高这个维度上的整齐。与第一阶段的注意事项类似，请处理好三个维度上的所有折角，确保它们符合计划中所要求的那种布局，以免出现三维折叠过程中的蝴蝶效应；为此，我们常常会在折叠第一个四分之一图案的过程中，与成品玫瑰花进行反复比较，以便在第一时间排除掉所有可能的错误。最后一个阶段是完成花瓣修饰。在这个阶段，我们往往强调一个重要名词，叫用心折叠。这里的用心已经不是字面上的认真这个意思，而是指通过我们对于大自然中玫瑰花外型的理解，借助自然的曲线去不断修正花瓣的形状，以期逼近现实中的玫瑰花瓣外形。请注意，在这个阶段的最后一步，我们需要通过拉扯已经弯折的四个花瓣，来调整玫瑰花中心的绽放程度。这个过程可能会伴随玫瑰花整体结构的崩塌，所以，一定要控制好调整的力道，以免出现不可逆的后果。最终，经过三个阶段的折叠，我们会得到一支栩栩如生的玫瑰花冠。如果条件允许，我们可以在一根拉直的铁丝上缠绕绿色纸条，并将玫瑰花冠插在铁丝的一段。这样，我们就得到了一支手工玫瑰花。总之，通过创建栅格折痕，制作立体基座，以及完成花瓣修饰，我们从二维的纸面上创作出了一支三维的花朵。这个过程虽然看似简单，但它确实我们人类借助想象力和常见素材而创作出的艺术品。请赏析以上内容的精妙之处。"]


ov_version = get_version()
# Init OV BGE Embeddings w/ Langchain
embedding_device = 'NPU'
print(f"Run OV BGE embedding on {embedding_device} with {ov_version}")
encode_kwargs = {'normalize_embeddings': do_norm, "max_length": 512}
embedding_model_kwargs = {"device": embedding_device}
#print(embedding_model_kwargs.keys())

ov_embeddings = OVBgeEmbeddings(
    model_dir=fp16_model_dir,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=encode_kwargs,
)

for query in queries:
    #print("Input query: ", query)
    pt_embeddings_results = pt_embeddings.embed_query(query)
    #print(pt_embeddings_results)
    # print("PT embedding shape: ", np.array(pt_embeddings_results).shape)
    ov_embeddings_results = ov_embeddings.embed_query(query)
    #print("OV embedings shape: ", np.array(ov_embeddings_results).shape)
    #print("OV embedings data: ", np.array(ov_embeddings_results))
    mses = ((np.array(pt_embeddings_results)-np.array(ov_embeddings_results))**2).mean(axis=0)
    print("Mean square error between PT embedding results and OV embeddings results: ", mses)

    distance = 0
    norm = 0
    for i in range(0,10):
        distance+=math.pow(pt_embeddings_results[i]-ov_embeddings_results[i],2)
        norm+=math.pow(pt_embeddings_results[i],2)
        diff = math.pow(distance/norm,0.5)
        # print("Diff between pt_embeddings_results and ov_embeddings_results: ", diff)


