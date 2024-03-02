#include "iostream"
#include "openvino_backend_api.h"
#include <chrono>
#include <thread>

std::string tiktoken_path = "C:\\llm\\qwen_cpp\\openvino.genai\\llm\\qwen_cpp\\Qwen-7B-Chat-NNCF_INT4_ALL_LAYERS_STATEFUL\\qwen.tiktoken";
qwen::QwenConfig config;
std::unique_ptr<qwen::QwenTokenizer> tokenizer = std::make_unique<qwen::QwenTokenizer>(tiktoken_path, config);
auto text_streamer = std::make_shared<qwen::TextStreamer>(std::cout, tokenizer.get());

void callback(int32_t *new_token_id, bool *stop_generation)
{
    if (!*stop_generation)
    {
        //std::cout << *new_token_id << "\n";
        text_streamer->put({*new_token_id});
    }
}

int main()
{
    params default_params;
    std::string model_path = "C:\\llm\\qwen_cpp\\openvino.genai\\llm\\qwen_cpp\\Qwen-7B-Chat-NNCF_INT4_ALL_LAYERS_STATEFUL\\modified_openvino_model.xml";
    std::string tokenizer_path = "C:\\llm\\qwen_cpp\\openvino.genai\\llm\\qwen_cpp\\Qwen-7B-Chat-NNCF_INT4_ALL_LAYERS_STATEFUL\\qwen.tiktoken";
    std::string prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is OpenVINO?<|im_end|>\n<|im_start|>assistant\n";
    default_params.tokenizer_path = tokenizer_path;
    char *cstr = new char[model_path.length() + 1];
    std::strcpy(cstr, model_path.c_str());

    // Initialize OpenVINO API Interface
    auto openvino_api_interface = openvino_backend::api_interface(default_params);

    auto status = openvino_api_interface.api_status();
    // Load LLM model
    openvino_api_interface.api_loadmodel(cstr, 0);

    status = openvino_api_interface.api_status();

    // Unload LLM model
    openvino_api_interface.api_unloadmodel();
    status = openvino_api_interface.api_status();
    std::cout << "Sleep 5s to check if GPU memory released after model unload\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    openvino_api_interface.api_loadmodel(cstr, 0);
    status = openvino_api_interface.api_status();

    // Load Tokenizer
    openvino_api_interface.api_loadtokenizer(tokenizer_path);

    // Non-stream generation
    std::cout << "Input prompt: " << prompt << "\n";
    std::string response = openvino_api_interface.api_Generate(prompt, default_params);
    std::cout << "Generated response: " << response << "\n";
    status = openvino_api_interface.api_status();

    // Get performance statistics
    auto llm_perf_statistic = openvino_api_interface.get_performance_statistics();

    // Reset
    openvino_api_interface.api_Reset();
    status = openvino_api_interface.api_status();

    // Stream generation
    std::cout << "Input prompt: " << prompt << "\n";
    openvino_api_interface.api_Generate(prompt, default_params, callback);
    text_streamer->end();
    status = openvino_api_interface.api_status();
}