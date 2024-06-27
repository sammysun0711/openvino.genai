#ifndef _EMBEDDINGS
#define _EMBEDDINGS

#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <openvino/runtime/tensor.hpp>
#include <openvino/openvino.hpp>
#include "json.hpp"   

#include "state.hpp"

class Embeddings{
    public:
        ov::Core core;
        ov::InferRequest embedding_model;
        ov::InferRequest tokenizer;
        size_t BATCH_SIZE;

        Embeddings() = default;
        ~Embeddings() = default;

        void init(std::string bert_path, std::string device);
        std::vector<std::vector<std::vector<float>>> encode_queries(std::vector<std::string> queries);

    private:
        
        
        std::vector<std::vector<float>> encode_query(std::string query);
        std::vector<ov::Tensor> tokenize(std::string prompt);

        inline ov::Tensor convert_inttensor_to_floattensor(ov::Tensor itensor);
        inline ov::Tensor padding_for_fixed_input_shape(ov::Tensor input, ov::Shape shape);
};

#endif


