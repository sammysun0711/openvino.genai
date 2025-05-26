#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompt')
    args = parser.parse_args()

    device = 'CPU'  # GPU can be used as well
    # pipe = openvino_genai.LLMPipeline(args.model_dir, device)
    scheduler_confg = openvino_genai.SchedulerConfig()
    generation_config = openvino_genai.GenerationConfig()
    generation_config.max_new_tokens = 1000
    props = {}
    # ... adjust scheduler and geneartion configs as necessary ..

    scheduler_confg.use_cache_eviction = False  # enables_cache_evction

    # optional - configure eviction parameters
    scheduler_confg.cache_eviction_config = openvino_genai.CacheEvictionConfig(
        start_size=32,
        recent_size=128,
        max_cache_size=672,
        aggregation_mode=openvino_genai.AggregationMode.NORM_SUM)

    print("Init pipeline ...\n")
    pipe = openvino_genai.ContinuousBatchingPipeline(args.model_dir,
                                                     scheduler_confg,
                                                     device,
                                                     {},
                                                     props
                                                     )
    # config = openvino_genai.GenerationConfig()
    # config.max_new_tokens = 100

    # print(pipe.generate(args.prompt, config))
    prompt_list = [args.prompt] * 5
    print("Start generation ...\n")
    result = pipe.generate(prompt_list, [generation_config] * len(prompt_list))


if '__main__' == __name__:
    main()
