// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "reranker.hpp"

void Reranker::init(std::string reranker_path, std::string device){
    std::string reranker_model_path = (std::filesystem::path(reranker_path) / "openvino_model.xml").string();
    std::string reranker_tokenizer_path = (std::filesystem::path(reranker_path) / "openvino_tokenizer.xml").string();

    core.add_extension(OPENVINO_TOKENIZERS_PATH);
    try {
        reranker_model = core.compile_model(reranker_model_path, device).create_infer_request();
        std::cout << "Load reranker model successed\n";
        reranker_tokenizer = core.compile_model(reranker_tokenizer_path, "CPU").create_infer_request();
        std::cout << "Load tokenizer model successed\n";
        std::cout << "Init reranker models successed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }
}

std::vector<std::vector<ov::Tensor>> Reranker::tokenize(std::string query, std::vector<std::string> documents){
    std::vector<std::vector<ov::Tensor>> tokenizer_outputs;
    for (auto& doc: documents)
    {
        std::string tokenizer_input = query + SPECIAL_TOKENS_SEP + doc;
        auto tokenizer_input_tensor = ov::Tensor{ov::element::string, {1}, &tokenizer_input};
        reranker_tokenizer.set_input_tensor(tokenizer_input_tensor);
        reranker_tokenizer.infer();
        std::vector<ov::Tensor> tokenizer_output;

        ov::Tensor input_ids(reranker_tokenizer.get_tensor("input_ids").get_element_type(), reranker_tokenizer.get_tensor("input_ids").get_shape());
        reranker_tokenizer.get_tensor("input_ids").copy_to(input_ids);
        ov::Tensor attention_mask(reranker_tokenizer.get_tensor("attention_mask").get_element_type(), reranker_tokenizer.get_tensor("attention_mask").get_shape());
        reranker_tokenizer.get_tensor("attention_mask").copy_to(attention_mask);
        tokenizer_output.push_back(input_ids);
        tokenizer_output.push_back(attention_mask);

        tokenizer_outputs.push_back(tokenizer_output);
    }
    return tokenizer_outputs;
}

inline ov::Tensor Reranker::padding_for_input_shape(ov::Tensor input, ov::Shape shape, uint64_t padding_number) {
    ov::Tensor padded_input = ov::Tensor{ov::element::i64, shape};
    std::fill_n(padded_input.data<uint64_t>(), padded_input.get_size(), padding_number);
    std::copy_n(input.data<uint64_t>(), input.get_size(), padded_input.data<uint64_t>());
    return padded_input;
}

std::vector<ov::Tensor> Reranker::prepare_reranker_input(std::vector<std::vector<ov::Tensor>> tokenizer_outputs) {

    //find max length
    uint64_t max_length = 0;
    ov::Shape max_shape = {1, max_length};
    for (auto& tokenizer_output: tokenizer_outputs)
    {   
        uint64_t length = tokenizer_output[0].get_shape()[-1];
        if (max_length < length) {
            max_length = length;
            max_shape = {1, max_length};
        }  
    }

    ov::Tensor input_ids_multi{ov::element::i64, ov::Shape{tokenizer_outputs.size(), max_length}};
    uint64_t* input_ids_multi_data = input_ids_multi.data<uint64_t>();
    ov::Tensor attention_mask_multi{ov::element::i64, ov::Shape{tokenizer_outputs.size(), max_length}};
    uint64_t* attention_mask_multi_data = attention_mask_multi.data<uint64_t>();

    for (size_t i = 0; i < tokenizer_outputs.size(); i++)
    {   
        ov::Tensor padded_input_ids = padding_for_input_shape(tokenizer_outputs[i][0], max_shape, PAD_VALUE);
        ov::Tensor padded_attention_mask = padding_for_input_shape(tokenizer_outputs[i][1], max_shape, 0);
        std::copy_n(padded_input_ids.data<uint64_t>(), padded_input_ids.get_size(), input_ids_multi_data + i * max_length);
        std::copy_n(padded_attention_mask.data<uint64_t>(), padded_attention_mask.get_size(), attention_mask_multi_data + i * max_length);
    }
    return {input_ids_multi, attention_mask_multi};
}

bool compare(const std::pair<float, int>& i, const std::pair<float, int>& j) {
    return i.first > j.first;
}

std::vector<int> Reranker::rerank(std::vector<std::vector<ov::Tensor>> tokenizer_outputs, int top_k){

    std::vector<ov::Tensor> reranker_input = prepare_reranker_input(tokenizer_outputs);
    reranker_model.set_tensor("input_ids",reranker_input[0]);
    reranker_model.set_tensor("attention_mask",reranker_input[1]);
    reranker_model.infer();
    ov::Tensor res = reranker_model.get_output_tensor(0);
    float* output_buffer = res.data<float>();

    auto shape = res.get_shape();

    std::vector<std::pair<float, int>> scores_ids;


    for (size_t j = 0; j < res.get_size(); j++) {
        float score = 1.0 / (1.0 + exp(- output_buffer[j]));
        std::cout << "DEBUG: score output: "  << j << ": " << score << std::endl;
        std::pair<float, int> score_id= std::make_pair(score, j);
        scores_ids.push_back(score_id);
    }
    
    std::sort(scores_ids.begin(), scores_ids.end(), compare);
    std::vector<int> res_ids;
    int k = 0;
    for (auto & score_id: scores_ids) {
        if (k<top_k && score_id.first > 0.5) {
            res_ids.push_back(score_id.second);
        }
    }

    return res_ids;
}

std::vector<std::string> Reranker::compress_documents(std::string query, std::vector<std::string> documents, int top_k){
    std::vector<std::vector<ov::Tensor>> tokenizer_outputs = tokenize(query, documents);
    std::cout << "DEBUG: tokenizer_outputs success " << std::endl;
    std::vector<int> ids = rerank(tokenizer_outputs, top_k);
    std::vector<std::string> good_documents;
    for (auto id: ids) {
        std::cout << "DEBUG: id " << id << std::endl;
        good_documents.push_back(documents[id]);
    }
    return good_documents;
}