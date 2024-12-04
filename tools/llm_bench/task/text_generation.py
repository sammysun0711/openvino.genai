# -*- coding: utf-8 -*-
# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import time
import datetime
import logging as log
import llm_bench_utils.ov_utils
import llm_bench_utils.pt_utils
import llm_bench_utils.model_utils as model_utils
import numpy as np
import hashlib
import llm_bench_utils.metrics_print as metrics_print
import llm_bench_utils.output_csv
from transformers import set_seed
import llm_bench_utils.output_json
import llm_bench_utils.output_file
import llm_bench_utils.gen_output_data as gen_output_data
import llm_bench_utils.parse_json_data as parse_json_data
import threading, queue

FW_UTILS = {'pt': llm_bench_utils.pt_utils, 'ov': llm_bench_utils.ov_utils}

DEFAULT_OUTPUT_TOKEN_SIZE = 512
def simple_streamer(subword):
    print(subword, end='', flush=True)
    # Return flag corresponds whether generation should be stopped.
    # False means continue generation.
    return False

# class StreamerBase:
#     """
    
#         Base class for streamers. In order to use inherit from from this class and implement put, and methods.
#     """
#     def __init__(self) -> None:
#         ...
#     def end(self) -> None:
#         """
#         End is called at the end of generation. It can be used to flush cache if your own streamer has one
#         """
#     def put(self, token: int) -> bool:
#         """
#         Put is called every time new token is decoded. Returns a bool flag to indicate whether generation should be stopped, if return true generation stops
#         """
import openvino_genai
class IterableStreamer(openvino_genai.StreamerBase):
    """
    A custom streamer class for handling token streaming and detokenization with buffering.
    
    Attributes:
        tokenizer (Tokenizer): The tokenizer used for encoding and decoding tokens.
        tokens_cache (list): A buffer to accumulate tokens for detokenization.
        text_queue (Queue): A synchronized queue for storing decoded text chunks.
        print_len (int): The length of the printed text to manage incremental decoding.
    """

    def __init__(self, tokenizer, tokens_len = 1):
        """
        Initializes the IterableStreamer with the given tokenizer.
        
        Args:
            tokenizer (Tokenizer): The tokenizer to use for encoding and decoding tokens.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.tokens_cache = []
        self.text_queue = queue.Queue()
        self.print_len = 0
        self.tokens_len = tokens_len

    def __iter__(self):
        """
        Returns the iterator object itself.
        """
        return self

    def __next__(self):
        """
        Returns the next value from the text queue.
        
        Returns:
            str: The next decoded text chunk.
        
        Raises:
            StopIteration: If there are no more elements in the queue.
        """
        value = self.text_queue.get()  # get() will be blocked until a token is available.
        if value is None:
            raise StopIteration
        return value

    def get_stop_flag(self):
        """
        Checks whether the generation process should be stopped.
        
        Returns:
            bool: Always returns False in this implementation.
        """
        return False

    def put_word(self, word: str):
        """
        Puts a word into the text queue.
        
        Args:
            word (str): The word to put into the queue.
        """
        self.text_queue.put(word)

    def put(self, token_id: int) -> bool:
        """
        Processes a token and manages the decoding buffer. Adds decoded text to the queue.
        
        Args:
            token_id (int): The token_id to process.
        
        Returns:
            bool: True if generation should be stopped, False otherwise.
        """        
        self.tokens_cache.append(token_id)
        if len(self.tokens_cache) % self.tokens_len == 0:
            text = self.tokenizer.decode(self.tokens_cache)

            word = ''
            if len(text) > self.print_len and '\n' == text[-1]:
                # Flush the cache after the new line symbol.
                word = text[self.print_len:]            
                self.tokens_cache = []
                self.print_len = 0
            elif len(text) >= 3 and text[-3:] == chr(65533):
                # Don't print incomplete text.
                pass
            elif len(text) > self.print_len:
                # It is possible to have a shorter text after adding new token.
                # Print to output only if text lengh is increaesed.
                word = text[self.print_len:]
                self.print_len = len(text)
            self.put_word(word)        

            if self.get_stop_flag():
                # When generation is stopped from streamer then end is not called, need to call it here manually.
                self.end()
                return True  # True means stop  generation
            else:
                return False  # False means continue generation
        else:
            return False

    def end(self):
        """
        Flushes residual tokens from the buffer and puts a None value in the queue to signal the end.
        """
        text = self.tokenizer.decode(self.tokens_cache)
        if len(text) > self.print_len:
            word = text[self.print_len:]
            self.put_word(word)
            self.tokens_cache = []
            self.print_len = 0
        self.put_word(None)

def run_text_generation(input_text, num, model, tokenizer, args, iter_data_list, md5_list,
                        prompt_index, bench_hook, model_precision, proc_id, mem_consumption, tokens_len):
    set_seed(args['seed'])
    input_text_list = [input_text] * args['batch_size']
    if args["output_dir"] is not None and num == 0:
        for bs_index, in_text in enumerate(input_text_list):
            llm_bench_utils.output_file.output_input_text(in_text, args, model_precision, prompt_index, bs_index, proc_id)
    tok_encode_start = time.perf_counter()
    input_data = tokenizer(input_text_list, return_tensors='pt')
    tok_encode_end = time.perf_counter()
    tok_encode_time = (tok_encode_end - tok_encode_start) * 1000
    input_data.pop('token_type_ids', None)
    # Remove `token_type_ids` from inputs
    input_tokens = input_data['input_ids'] if 'input_ids' in input_data else input_data
    input_token_size = input_tokens[0].numel()
    if args['batch_size'] > 1:
        out_str = '[warm-up]' if num == 0 else '[{}]'.format(num)
        out_str += " Batch_size={}, ".format(args['batch_size'])
        out_str += 'all input token size after padding: {} * {}, '.format(input_token_size, args['batch_size'])
        if args['infer_count'] is not None:
            out_str += 'all max_output_token_size: {} * {}'.format(args['infer_count'], args['batch_size'])
        log.info(out_str)

    max_rss_mem_consumption = ''
    max_uss_mem_consumption = ''
    max_shared_mem_consumption = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start_collect_memory_consumption()
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']
    start = time.perf_counter()
    if args['infer_count'] is not None and args['end_token_stopping'] is False:
        model.generation_config.eos_token_id = None
        model.config.eos_token_id = None
        result = model.generate(
            **input_data,
            max_new_tokens=int(max_gen_tokens),
            num_beams=args['num_beams'],
            use_cache=True,
            eos_token_id=None,
            do_sample=False
        )
    else:
        result = model.generate(
            **input_data,
            max_new_tokens=int(max_gen_tokens),
            num_beams=args['num_beams'],
            use_cache=True,
            do_sample=False
        )
    end = time.perf_counter()
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.end_collect_momory_consumption()
        max_rss_mem_consumption, max_shared_mem_consumption, max_uss_mem_consumption = mem_consumption.get_max_memory_consumption()
        mem_consumption.clear_max_memory_consumption()

    generation_time = end - start
    tok_decode_start = time.perf_counter()
    generated_text = tokenizer.batch_decode(result)
    tok_decode_end = time.perf_counter()
    tok_decode_time = (tok_decode_end - tok_decode_start) * 1000
    # Only text_gen need to minus length of input_data, because generated_text may include input_text
    num_tokens = 0
    result_md5_list = []
    for bs_idx in range(args['batch_size']):
        if 'sum' not in args['model_name'] and result[bs_idx][:input_token_size].equal(input_tokens[bs_idx]):
            generated_token_size = len(result[bs_idx]) - input_tokens[bs_idx].numel()
        else:
            generated_token_size = len(result[bs_idx])
        # Encoder-decoder models expect the `decoder_input_ids` to start with a special token
        # When counting the output length, subtract 1. The last token does not participate in inference.
        if model.config.is_encoder_decoder and result[bs_idx][0] == model.config.decoder_start_token_id:
            generated_token_size = generated_token_size - 1
        num_tokens += generated_token_size
        """
        if generated_token_size > max_gen_tokens:
            log.error('Output token size is over max output token size!')
        """
        result_text = generated_text[bs_idx]
        if args["output_dir"] is not None:
            llm_bench_utils.output_file.output_gen_text(result_text, args, model_precision, prompt_index, num, bs_idx, proc_id)
        result_md5_list.append(hashlib.new("md5", result_text.encode(), usedforsecurity=False).hexdigest())
    if len(md5_list[num]) == 0:
        md5_list[num] = {prompt_index : result_md5_list}
    else:
        md5_list[num][prompt_index] = result_md5_list
    per_token_time = ""
    if num_tokens > 0:
        per_token_time = generation_time * 1000 / (num_tokens / args['batch_size'])
    else:
        log.warning("No generated tokens")
    tm_list = []
    tm_infer_list = []
    if bench_hook is not None:
        tm_list = bench_hook.get_time_list()
        log.debug('latency of all tokens:')
        [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_list)]
        tm_infer_list = bench_hook.get_time_infer_list()
        log.debug('latency of all infers:')
        [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_infer_list)]
        if args['num_beams'] == 1 and generated_token_size != len(tm_infer_list):
            log.warning(f'Output token size({generated_token_size}) is not equal to infer count({len(tm_infer_list)})')
    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        in_size=input_token_size * args['batch_size'],
        infer_count=len(tm_infer_list),
        out_size=num_tokens,
        gen_time=generation_time,
        latency=per_token_time,
        res_md5=result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        prompt_idx=prompt_index,
        tokenization_time=(tok_encode_time, tok_decode_time)
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        num,
        iter_data,
        tm_list,
        tm_infer_list,
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        tokenization_time=(tok_encode_time, tok_decode_time),
        batch_size=args['batch_size'],
        prompt_idx=prompt_index
    )
    if num > 0:
        prev_md5 = md5_list[num - 1][prompt_index]
        if result_md5_list != prev_md5:
            log.warning(f"[{num}] Prompt[{prompt_index}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {num - 1} iteration {prev_md5}")
            metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)
            if not args.get("use_cb", False):
                if num == 1:
                    # if the device is CPU, throw exception
                    if args['devices'].lower().startswith('cpu') is True:
                        assert (result_md5_list == prev_md5)
                else:
                    # throw exception
                    assert (result_md5_list == prev_md5)
    else:
        metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)
    if bench_hook is not None:
        bench_hook.clear_time_list()
        bench_hook.clear_time_infer_list()


def run_text_generation_genai(input_text, num, model, tokenizer, args, iter_data_list, md5_list, prompt_index,
                              streamer, model_precision, proc_id, mem_consumption, tokens_len):
    set_seed(args['seed'])
    input_text_list = [input_text] * args['batch_size']
    if args["output_dir"] is not None and num == 0:
        for bs_index, in_text in enumerate(input_text_list):
            llm_bench_utils.output_file.output_input_text(in_text, args, model_precision, prompt_index, bs_index, proc_id)
    max_rss_mem_consumption = ''
    max_uss_mem_consumption = ''
    max_shared_mem_consumption = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start_collect_memory_consumption()
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']
    tokenizer = model.get_tokenizer()
    input_data = tokenizer.encode(input_text_list)
    num_input_tokens = input_data.input_ids.shape[1]
    if args['batch_size'] > 1:
        out_str = '[warm-up]' if num == 0 else '[{}]'.format(num)
        out_str += " Batch_size={}, ".format(args['batch_size'])
        out_str += 'all input token size after padding: {} * {}, '.format(num_input_tokens, args['batch_size'])
        if args['infer_count'] is not None:
            out_str += 'all max_output_token_size: {} * {}'.format(args['infer_count'], args['batch_size'])
        log.info(out_str)

    text_print_streamer = IterableStreamer(model.get_tokenizer(), tokens_len)
    def token_printer():
        # Getting next elements from iterable will be blocked until a new token is available.
        for word in text_print_streamer:
            print(word, end='', flush=True)
    printer_thread = threading.Thread(target=token_printer, daemon=True)
    printer_thread.start()

    start = time.perf_counter()
    #generation_result = model.generate(input_text_list, max_new_tokens=max_gen_tokens, num_beams=args["num_beams"], do_sample=False)
    # generation_result = model.generate(input_text_list, max_new_tokens=max_gen_tokens, num_beams=args["num_beams"], do_sample=False, streamer=simple_streamer)
    generation_result = model.generate(input_text_list, max_new_tokens=max_gen_tokens, num_beams=args["num_beams"], do_sample=False, streamer=text_print_streamer)
    printer_thread.join()
    end = time.perf_counter()
    print("\n")
    generated_text = generation_result.texts
    perf_metrics = generation_result.perf_metrics

    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.end_collect_momory_consumption()
        max_rss_mem_consumption, max_shared_mem_consumption, max_uss_mem_consumption = mem_consumption.get_max_memory_consumption()
        mem_consumption.clear_max_memory_consumption()

    generation_time = end - start
    generated_tokens = [tokenizer.encode(text).input_ids.data for text in generated_text]
    # Only text_gen need to minus length of input_data, because generated_text may include input_text
    num_tokens = 0
    result_md5_list = []
    for bs_idx in range(args['batch_size']):
        generated_text_len = generated_tokens[bs_idx].shape[-1]
        num_tokens += generated_text_len
        """
        if generated_text_len > max_gen_tokens:
            log.error('Output token size is over max output token size!')
        """
        result_text = generated_text[bs_idx]
        if args["output_dir"] is not None:
            llm_bench_utils.output_file.output_gen_text(result_text, args, model_precision, prompt_index, num, bs_idx, proc_id)
        result_md5_list.append(hashlib.new("md5", result_text.encode(), usedforsecurity=False).hexdigest())
    if len(md5_list[num]) == 0:
        md5_list[num] = {prompt_index : result_md5_list}
    else:
        md5_list[num][prompt_index] = result_md5_list
    per_token_time = ""
    if num_tokens > 0:
        per_token_time = generation_time * 1000 / (num_tokens / args['batch_size'])
    else:
        log.warning("No generated tokens")
    first_token_time = (perf_metrics.get_ttft().mean - perf_metrics.raw_metrics.tokenization_durations[-1] / 1000) / args["batch_size"]
    second_tokens_durations = (
        np.array(perf_metrics.raw_metrics.m_new_token_times[1:])
        - np.array(perf_metrics.raw_metrics.m_new_token_times[:-1]) / args["batch_size"]
    ).tolist()

    tm_list = np.array([first_token_time] + second_tokens_durations) / 1000
    log.debug('latency of all tokens:')
    [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_list)]
    tokenization_time = (
        np.mean(perf_metrics.raw_metrics.tokenization_durations) / 1000,
        np.mean(perf_metrics.raw_metrics.detokenization_durations) / 1000
    )
    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        in_size=num_input_tokens * args['batch_size'],
        infer_count=len(tm_list),
        out_size=num_tokens,
        gen_time=generation_time,
        latency=per_token_time,
        res_md5=result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        prompt_idx=prompt_index,
        tokenization_time=tokenization_time
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        num,
        iter_data,
        tm_list.tolist(),
        [],
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        tokenization_time=tokenization_time,
        batch_size=args['batch_size'],
        prompt_idx=prompt_index
    )

    print("== Generating output: ", generated_text[0])
    if num > 0:
        prev_md5 = md5_list[num - 1][prompt_index]
        if result_md5_list != prev_md5:
            log.warning(f"[{num}] Prompt[{prompt_index}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {num - 1} iteration {prev_md5}")
            metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)
            if not args.get("use_cb", False):
                if num == 1:
                    # if the device is CPU, throw exception
                    if args['devices'].lower().startswith('cpu') is True:
                        assert (result_md5_list == prev_md5)
                else:
                    # throw exception
                    assert (result_md5_list == prev_md5)
    else:
        metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)


def run_text_generation_genai_with_stream(input_text, num, model, tokenizer, args, iter_data_list, md5_list,
                                          prompt_index, streamer, model_precision, proc_id, mem_consumption, tokens_len):
    set_seed(args['seed'])
    input_text_list = [input_text] * args['batch_size']
    if args["output_dir"] is not None and num == 0:
        for bs_index, in_text in enumerate(input_text_list):
            llm_bench_utils.output_file.output_input_text(in_text, args, model_precision, prompt_index, bs_index, proc_id)
    pipe_tokenizer = model.get_tokenizer()
    tok_encode_start = time.perf_counter()
    input_data = pipe_tokenizer.encode(input_text_list)
    tok_encode_end = time.perf_counter()
    input_token_size = input_data.input_ids.shape[1]
    tok_encode_time = (tok_encode_end - tok_encode_start) * 1000
    if args['batch_size'] > 1:
        out_str = '[warm-up]' if num == 0 else '[{}]'.format(num)
        out_str += " Batch_size={}, ".format(args['batch_size'])
        out_str += 'all input token size after padding: {} * {}, '.format(input_token_size, args['batch_size'])
        if args['infer_count'] is not None:
            out_str += 'all max_output_token_size: {} * {}'.format(args['infer_count'], args['batch_size'])
        log.info(out_str)
    max_rss_mem_consumption = ''
    max_uss_mem_consumption = ''
    max_shared_mem_consumption = ''
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.start_collect_memory_consumption()
    max_gen_tokens = DEFAULT_OUTPUT_TOKEN_SIZE if args['infer_count'] is None else args['infer_count']
    streamer.reset()
    start = time.perf_counter()
    generated_tokens = model.generate(input_data, max_new_tokens=max_gen_tokens, num_beams=args["num_beams"], streamer=streamer, do_sample=False).tokens
    end = time.perf_counter()
    if (args['mem_consumption'] == 1 and num == 0) or args['mem_consumption'] == 2:
        mem_consumption.end_collect_momory_consumption()
        max_rss_mem_consumption, max_shared_mem_consumption, max_uss_mem_consumption = mem_consumption.get_max_memory_consumption()
        mem_consumption.clear_max_memory_consumption()
    generation_time = end - start
    tok_decode_start = time.perf_counter()
    generated_text = pipe_tokenizer.decode(generated_tokens)
    tok_decode_end = time.perf_counter()
    tok_decode_time = (tok_decode_end - tok_decode_start) * 1000
    # Only text_gen need to minus length of input_data, because generated_text may include input_text
    num_tokens = 0
    result_md5_list = []
    for bs_idx in range(args['batch_size']):
        generated_text_len = len(generated_tokens[bs_idx])
        num_tokens += generated_text_len
        """
        if generated_text_len > max_gen_tokens:
            log.error('Output token size is over max output token size!')
        """
        result_text = generated_text[bs_idx]
        if args["output_dir"] is not None:
            llm_bench_utils.output_file.output_gen_text(result_text, args, model_precision, prompt_index, num, bs_idx, proc_id)
        result_md5_list.append(hashlib.new("md5", result_text.encode(), usedforsecurity=False).hexdigest())
    if len(md5_list[num]) == 0:
        md5_list[num] = {prompt_index : result_md5_list}
    else:
        md5_list[num][prompt_index] = result_md5_list
    per_token_time = ""
    if num_tokens > 0:
        per_token_time = generation_time * 1000 / (num_tokens / args['batch_size'])
    else:
        log.warning("No generated tokens")
    tm_list = streamer.get_time_list()
    log.debug('latency of all tokens:')
    [log.debug('[{}]{:.4f}'.format(idx, tm)) for idx, tm in enumerate(tm_list)]
    iter_data = gen_output_data.gen_iterate_data(
        iter_idx=num,
        in_size=input_token_size * args['batch_size'],
        infer_count=len(tm_list),
        out_size=num_tokens,
        gen_time=generation_time,
        latency=per_token_time,
        res_md5=result_md5_list,
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        prompt_idx=prompt_index,
        tokenization_time=(tok_encode_time, tok_decode_time)
    )
    iter_data_list.append(iter_data)
    metrics_print.print_metrics(
        iter_num=num,
        iter_data=iter_data,
        tms=tm_list,
        tms_infer=[],
        warm_up=(num == 0),
        max_rss_mem=max_rss_mem_consumption,
        max_shared_mem=max_shared_mem_consumption,
        max_uss_mem=max_uss_mem_consumption,
        tokenization_time=(tok_encode_time, tok_decode_time),
        batch_size=args['batch_size'],
        prompt_idx=prompt_index
    )
    if num > 0:
        prev_md5 = md5_list[num - 1][prompt_index]
        if result_md5_list != prev_md5:
            log.warning(f"[{num}] Prompt[{prompt_index}]'s md5 {result_md5_list} "
                        f"is different from md5 of the {num - 1} iteration {prev_md5}")
            metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)
            if not args.get("use_cb", False):
                if num == 1:
                    # if the device is CPU, throw exception
                    if args['devices'].lower().startswith('cpu') is True:
                        assert (result_md5_list == prev_md5)
                else:
                    # throw exception
                    assert (result_md5_list == prev_md5)
    else:
        metrics_print.print_generated(num, warm_up=(num == 0), generated=generated_text[0], prompt_idx=prompt_index)
    streamer.reset()


def run_text_generation_benchmark(model_path, framework, device, args, num_iters, tokens_len, mem_consumption):
    model, tokenizer, pretrain_time, bench_hook, use_genai = FW_UTILS[framework].create_text_gen_model(model_path, device, **args)
    model_precision = model_utils.get_model_precision(model_path.parts)
    iter_data_list = []
    md5_list = {num : {} for num in range(num_iters + 1)}
    input_text_list = get_text_prompt(args)
    if args['prompt_index'] is None:
        prompt_idx_list = [prompt_idx for prompt_idx, input_text in enumerate(input_text_list)]
        text_list = input_text_list
    else:
        prompt_idx_list = []
        text_list = []
        for i in args['prompt_index']:
            if 0 <= i < len(input_text_list):
                text_list.append(input_text_list[i])
                prompt_idx_list.append(i)
    if len(input_text_list) == 0:
        raise RuntimeError('==Failure prompts is empty ==')
    log.info(f"Numbeams: {args['num_beams']}, benchmarking iter nums(exclude warm-up): {num_iters}, "
             f'prompt nums: {len(text_list)}, prompt idx: {prompt_idx_list}')

    # if num_iters == 0, just output warm-up data
    if not use_genai:
        text_gen_fn = run_text_generation
    elif bench_hook is not None:
        text_gen_fn = run_text_generation_genai_with_stream
    else:
        text_gen_fn = run_text_generation_genai

    proc_id = os.getpid()
    iter_timestamp = model_utils.init_timestamp(num_iters, text_list, prompt_idx_list)
    if args['subsequent'] is False:
        for num in range(num_iters + 1):
            for idx, input_text in enumerate(text_list):
                p_idx = prompt_idx_list[idx]
                if num == 0:
                    log.info(f'[warm-up][P{p_idx}] Input text: {input_text}')
                iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
                text_gen_fn(input_text, num, model, tokenizer, args, iter_data_list, md5_list,
                            p_idx, bench_hook, model_precision, proc_id, mem_consumption, tokens_len)
                iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
                prefix = '[warm-up]' if num == 0 else '[{}]'.format(num)
                log.info(f"{prefix}[P{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")
    else:
        for idx, input_text in enumerate(text_list):
            p_idx = prompt_idx_list[idx]
            for num in range(num_iters + 1):
                if num == 0:
                    log.info(f'[warm-up][P{p_idx}] Input text: {input_text}')
                iter_timestamp[num][p_idx]['start'] = datetime.datetime.now().isoformat()
                text_gen_fn(input_text, num, model, tokenizer, args, iter_data_list, md5_list,
                            prompt_idx_list[idx], bench_hook, model_precision, proc_id, mem_consumption)
                iter_timestamp[num][p_idx]['end'] = datetime.datetime.now().isoformat()
                prefix = '[warm-up]' if num == 0 else '[{}]'.format(num)
                log.info(f"{prefix}[P{p_idx}] start: {iter_timestamp[num][p_idx]['start']}, end: {iter_timestamp[num][p_idx]['end']}")

    metrics_print.print_average(iter_data_list, prompt_idx_list, args['batch_size'], True)

    return iter_data_list, pretrain_time, iter_timestamp


def get_text_prompt(args):
    text_list = []
    output_data_list, is_json_data = model_utils.get_param_from_file(args, 'prompt')
    if is_json_data is True:
        text_param_list = parse_json_data.parse_text_json_data(output_data_list)
        if len(text_param_list) > 0:
            for text in text_param_list:
                text_list.append(text)
    else:
        text_list.append(output_data_list[0])
    return text_list
