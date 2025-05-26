// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"
#include "filesystem"
int main(int argc, char* argv[]) try {
    std::cout << "Init\n";
    if (2 > argc)
        //throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<PROMPT>\" <TOKENIZER_PATH>");
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<PROMPT>\"");
    
    std::string models_path = argv[1];
    std::string prompt = argv[2];
    //std::string tokenizer_path = argv[3];
    std::string device = "CPU";  // GPU can be used as well
    //ov::genai::Tokenizer tokenizer(tokenizer_path);
    ov::AnyMap properties;
    properties["ATTENTION_BACKEND"] = "SDPA";
    //std::filesystem::path model_path(models_path);
    /*
    if (std::filesystem::exists(model_path.parent_path() / "openvino_model.xml")) {
        models_path = model_path.parent_path();
    } else {
        models_path = model_path;
    }
    properties[ov::cache_dir.name()] = "model_cache";
    properties[ov::genai::enable_save_ov_model.name()] = true;
    
    ov::genai::LLMPipeline pipe(models_path, tokenizer, device, properties);
    */
    std::cout << "Create pipeline start\n";
    ov::genai::LLMPipeline pipe(models_path, device, properties);
    std::cout << "Create pipeline end\n";
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    std::cout << "Generating text with prompt: " << prompt << std::endl;
    std::string result = pipe.generate(prompt, config);
    std::cout << result << std::endl;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
