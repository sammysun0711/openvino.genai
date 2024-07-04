// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef _EMBEDDINGS
#define _EMBEDDINGS

#include <chrono>
#include <fstream>
#include <iostream>
#include <openvino/openvino.hpp>
#include <openvino/runtime/tensor.hpp>
#include <sstream>

#include "json.hpp"

class Embeddings {
public:
    ov::Core core;
    ov::InferRequest embedding_model;
    ov::InferRequest tokenizer;
    size_t BATCH_SIZE;

    Embeddings() = default;
    ~Embeddings() = default;

    void init(std::string bert_path, std::string device);

    std::vector<std::vector<float>> encode_queries(std::vector<std::string> queries);

private:
    std::vector<float> encode_query(std::string query);
    std::vector<ov::Tensor> tokenize(std::string prompt);

    inline ov::Tensor convert_inttensor_to_floattensor(ov::Tensor itensor);
    inline ov::Tensor padding_for_fixed_input_shape(ov::Tensor input, ov::Shape shape);
};

#endif
