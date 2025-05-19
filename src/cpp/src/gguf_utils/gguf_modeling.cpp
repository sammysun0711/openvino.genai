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
#include "openvino/pass/serialize.hpp"

#include "gguf_utils/building_blocks.hpp"
#include "gguf_utils/gguf_modeling.hpp"
#ifdef __cplusplus
extern "C" {
#endif

#include "deps/sha1/sha1.h"
#ifdef __cplusplus
}
#endif

using namespace ov;
using namespace ov::op::v13;
using namespace ov::op;

#define HASH_TYPE_SHA1_STR   "sha1"



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

void save_openvino_model(const std::shared_ptr<ov::Model>& model, const std::string& save_path, bool compress_to_fp16) {
    try {
        auto serialize_start_time = std::chrono::high_resolution_clock::now();
        ov::save_model(model, save_path, compress_to_fp16);
        auto serialize_finish_time = std::chrono::high_resolution_clock::now();
        auto serialize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(serialize_finish_time - serialize_start_time).count();
        std::cout << "Save generated OpenVINO model to: " << save_path << " done. Time: " << serialize_duration << " ms\n";
    }
    catch (const ov::Exception& e) {
        std::cerr << "[Warning] Exception during model serialization: " << e.what() << std::endl;
    }
}

void compute_hash_sha1(const std::unordered_map<std::string, ov::Tensor>& consts){

    // sha1 init
    SHA1_CTX sha1_model_hash_ctx;

    SHA1Init(&sha1_model_hash_ctx);

    for (const auto& pair : consts) {
        const std::string& key = pair.first;
        const ov::Tensor& tensor = pair.second;

        // Per Layer Hash
        char result[21]; // sha1 outputs 20 bytes
        SHA1( result, (const char *)tensor.data(), tensor.get_size());

        char hex_result[41] = {0};
        for (int  offset = 0; offset < 20; offset++) {
            snprintf( ( hex_result + (2*offset)), sizeof(hex_result) - (2*offset), "%02x", result[offset]&0xff);
        }

        //printf("%-8s  %-s  %s\n", HASH_TYPE_SHA1_STR, hex_result, key.c_str());
    }
}
/*
void compute_hash_xxhash(const std::unordered_map<std::string, ov::Tensor>& consts){

    // xxh64 init
    XXH64_state_t* xxh64_model_hash_state = NULL;
    xxh64_model_hash_state = XXH64_createState();
    if (xxh64_model_hash_state==NULL) {
        abort();
    }

    XXH64_hash_t const seed = 0;
    if (XXH64_reset(xxh64_model_hash_state, seed) == XXH_ERROR) {
        abort();
    }

    for (const auto& pair : consts) {
        const std::string& key = pair.first;
        const ov::Tensor& tensor = pair.second;

        // Per Layer Hash
        XXH64_hash_t hash = XXH64(tensor.data(), tensor.get_size(), 0);

        char hex_result[17];
        for (int  offset = 0; offset < 8; offset++) {
            unsigned int shift_bits_by = (8 * (8 - offset - 1));
            snprintf( ( hex_result + (2*offset)), sizeof(hex_result) - (2*offset), "%02x", (unsigned char) (hash >> shift_bits_by)&0xff);
        }

        printf("%-8s  %-s  %s\n", HASH_TYPE_XXH64_STR, hex_result, key.c_str());
    }
}
*/

std::shared_ptr<ov::Model> create_from_gguf(const std::string& model_path, const ov::AnyMap& properties) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "Loading and unpacking model from: " << model_path << std::endl;
    auto [config, consts, qtypes] = load_gguf(model_path);

    auto load_finish_time = std::chrono::high_resolution_clock::now();
    std::cout << "Loading and unpacking model done. Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(load_finish_time - start_time).count() << "ms" << std::endl;

    auto hash_start_time = std::chrono::high_resolution_clock::now();
    compute_hash_sha1(consts);
    auto hash_finish_time = std::chrono::high_resolution_clock::now();
    std::cout << "Compute GGUF hash with SHA1 done. Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(hash_finish_time - hash_start_time).count() << "ms" << std::endl;

    std::cout << "Start generating OpenVINO model..." << std::endl;

    std::shared_ptr<ov::Model> model;

    const std::string model_arch = std::get<std::string>(config.at("architecture"));
    if (!model_arch.compare("llama") || !model_arch.compare("qwen2")) {
        model = create_language_model(config, consts, qtypes);
        if (properties.find(ov::cache_dir.name()) != properties.end()) {
            std::string cache_dir = properties.at(ov::cache_dir.name()).as<std::string>();
            if (!cache_dir.empty()) {
                std::filesystem::path model_cache_dir(cache_dir);
                std::filesystem::path gguf_model_path(model_path);
                std::filesystem::path save_path = model_cache_dir / (gguf_model_path.stem().string() + "_openvino_model.xml");
                save_openvino_model(model, save_path.string(), true);
            }
        }
    } else {
        OPENVINO_THROW("Unsupported model architecture '", model_arch, "'");
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - load_finish_time).count();
    std::cout << "Model generation done. Time: " << duration << "ms" << std::endl;

    return model;
}
