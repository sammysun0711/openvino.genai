import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, TextIteratorStreamer
from threading import Thread
import argparse
from glm4v_helper import OvGLM4v
import time
import numpy
import sys
sys.path.append("..")
from llm_bench.llm_bench_utils.memory_profile import MemConsumption

seed = 42
torch.manual_seed(seed)          
torch.cuda.manual_seed(seed)      
torch.cuda.manual_seed_all(seed)
numpy.random.seed(seed)

parser = argparse.ArgumentParser('glm4v ov convert tool', add_help=True, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model_dir', type=str, required=True, help='Directory where the model is stored')
parser.add_argument('--image_path', type=str, default="bird.png", help='Image path')
parser.add_argument('--image_size', type=int, default=672, required=False, help='Image size, default is 672 from default config of glm4v')
parser.add_argument('--device', type=str, default="GPU", required=True, help='Device to run inference on, default is "CPU"')
parser.add_argument('--prompt', type=str, default='描述这张图片', help='Prompt, default is "描述这张图片"')
parser.add_argument('--num_count', type=int, default=5, help='Number of infers, default is 5')

args = parser.parse_args()
model_dir = args.model_dir
image_path = args.image_path
device = args.device
prompt = args.prompt
num_count = args.num_count
image_size = args.image_size

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

query = prompt
image = Image.open(image_path).convert('RGB')
image = image.resize((image_size, image_size))

llm_times=[]
input_token_length = []
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

mem_consumption = MemConsumption()
mem_consumption.start_collect_mem_consumption_thread()        
model = OvGLM4v(model_dir, device, llm_times=llm_times, input_token_length=input_token_length)

gen_kwargs = {"max_new_tokens": 128, 
              "do_sample": True, 
              "top_k": 50,
              "top_p": 0.95, 
              "eos_token_id": [ 59246,59253,59255]}
tokens_len = 10

"""
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0]))
"""
with torch.no_grad():
    # for text in model.chat_stream(image, query, tokenizer, tokens_len, gen_kwargs):
    #     print(text, end='', flush=True)
    model.chat_stream(image, query, tokenizer, tokens_len, gen_kwargs)
print("\n-------------------------warmup finished-------------------------")

first_token_t = []
avg_token_t = []
start_time = time.time()
NC = num_count
max_rss_mem, max_shared_mem, max_uss_mem = None, None, None
pipeline_latency = []
for i in range(NC):
    """
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        res = tokenizer.decode(outputs[0])
    """

    mem_consumption.start_collect_memory_consumption()
    print("== Genereate output: ")
    start_time = time.time()
    with torch.no_grad():
        # for text in model.chat_stream(image, query, tokenizer, tokens_len, gen_kwargs):
        #     print(text, end='', flush=True)
        model.chat_stream(image, query, tokenizer, tokens_len, gen_kwargs)
    end_time = time.time()
    pipeline_latency.append((end_time - start_time))
    mem_consumption.end_collect_momory_consumption()

    print("\n--------------------------------------------")
    if len(llm_times) > 1:
        first_token_t.append(llm_times[0])
        avg_token = sum(llm_times[1:]) / (len(llm_times) - 1)
        avg_token_t.append(avg_token)
        print(f"First input token size: ", input_token_length[-1])
        print(f"VLM Model First token latency: {llm_times[0]:.2f} ms, Output len: {len(llm_times)}, Average 2nd+ token latency: {avg_token:.2f} ms")
        max_rss_mem, max_shared_mem, max_uss_mem =  mem_consumption.get_max_memory_consumption()
        print("max_rss_mem: {:.2f} MB".format(max_rss_mem))
        mem_consumption.clear_max_memory_consumption()

print("--------------------------------------------")
print("")
print(f"== Performance metrics from {NC} times run: ")
print(f"First input token size: ", input_token_length[0])
avg_token_ft = sum(first_token_t) / len(first_token_t)
print(f"Average VLM first token latency: {avg_token_ft:.2f} ms")
avg_token_av = sum(avg_token_t) / len(avg_token_t)
print(f"Average VLM 2nd+ token latency: {avg_token_av:.2f} ms")
avg_token_rate = 1000 / avg_token_av
print(f"Average VLM token rate: {avg_token_rate:.2f} t/s")
print("Max RSS Memory Usage: {:.2f} MB".format(max_rss_mem))
print(f'Average E2E pipeline inference time of {NC} iteration: {sum(pipeline_latency)/NC:.2f} s')
print("")
print("--------------------------------------------")

mem_consumption.end_collect_mem_consumption_thread()
