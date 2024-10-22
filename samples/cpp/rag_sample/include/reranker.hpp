// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef _RERANKER
#define _RERANKER

#include <chrono>
#include <fstream>
#include <iostream>
#include <openvino/openvino.hpp>
#include <openvino/runtime/tensor.hpp>
#include <sstream>

#include "json.hpp"

class Reranker {
public:
    ov::Core core;
    ov::InferRequest reranker_model;
    ov::InferRequest reranker_tokenizer;
    
    std::string SPECIAL_TOKENS_SEP="</s></s> ";
    uint64_t PAD_VALUE = 1;
    Reranker() = default;
    ~Reranker() = default;

    void init(std::string reranker_path, std::string device);
    std::vector<std::string> compress_documents(std::string query, std::vector<std::string> documents, int top_k);

private:
    std::vector<std::vector<ov::Tensor>> tokenize(std::string query, std::vector<std::string> documents);
    std::vector<int> rerank(std::vector<std::vector<ov::Tensor>> tokenizer_outputs, int top_k);
    inline ov::Tensor padding_for_input_shape(ov::Tensor input, ov::Shape shape, uint64_t padding_number);
    std::vector<ov::Tensor> prepare_reranker_input(std::vector<std::vector<ov::Tensor>> tokenizer_outputs);
};

#endif