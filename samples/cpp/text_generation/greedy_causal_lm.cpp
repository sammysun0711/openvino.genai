// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (4 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <TOKENIZER_DIR> \"<PROMPT>\"");

    std::string models_path = argv[1];
    std::string prompt = argv[2];
    std::string device = "CPU";  // GPU can be used as well
    std::string tokenizer_path = argv[3];
    ov::genai::Tokenizer tokenizer(tokenizer_path);
    ov::genai::LLMPipeline pipe(models_path, tokenizer, device);
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
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
