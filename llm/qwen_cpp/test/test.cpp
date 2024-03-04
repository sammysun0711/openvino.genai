#include "iostream"
#include "openvino_backend_api.h"
#include <chrono>
#include <thread>

params default_params;
// Initialize OpenVINO API Interface
auto openvino_api_interface = openvino_backend::api_interface(default_params);
qwen::QwenConfig config;
std::string tokenizer_path = "Qwen-7B-Chat-NNCF_INT4\\qwen.tiktoken";
std::unique_ptr<qwen::QwenTokenizer> tokenizer = std::make_unique<qwen::QwenTokenizer>(tokenizer_path, config);
auto text_streamer = std::make_shared<qwen::TextStreamer>(std::cout, tokenizer.get());

void callback(int32_t *new_token_id, bool *stop_generation)
{
    if (!*stop_generation)
    {
        // std::cout << *new_token_id << "\n";
        text_streamer->put({*new_token_id});
    }
}
void task1(const std::string &prompt, const params &params, void (*api_callback)(int32_t *new_token_id, bool *_stop_generation))
{
    openvino_api_interface.api_Generate(prompt, default_params, callback);
}

int main()
{
    try
    {
        // params default_params;
        std::string model_path = "Qwen-7B-Chat-NNCF_INT4\\openvino_model.xml";
        // std::string tokenizer_path = "Qwen-7B-Chat-NNCF_INT4\\qwen.tiktoken";
        std::string prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is OpenVINO?<|im_end|>\n<|im_start|>assistant\n";
        char *cstr = new char[model_path.length() + 1];
        std::strcpy(cstr, model_path.c_str());

        auto status = openvino_api_interface.api_status();
        // Load LLM model
        openvino_api_interface.api_loadmodel(cstr, 0);

        status = openvino_api_interface.api_status();

        // Unload LLM model
        auto success = openvino_api_interface.api_unloadmodel();
        status = openvino_api_interface.api_status();
        std::cout << "Sleep 5s to check if GPU memory released after model unload\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        openvino_api_interface.api_loadmodel(cstr, 0);
        status = openvino_api_interface.api_status();

        // Load Tokenizer
        openvino_api_interface.api_loadtokenizer(tokenizer_path);

        // Unload Tokenizer
        success = openvino_api_interface.api_unloadtokenizer();
        openvino_api_interface.api_loadtokenizer(tokenizer_path);

        // Non-stream generation
        std::cout << "Input prompt: " << prompt << "\n";
        std::string response = openvino_api_interface.api_Generate(prompt, default_params);
        std::cout << "Generated response: " << response << "\n";
        status = openvino_api_interface.api_status();

        // Get performance statistics
        auto llm_perf_statistic = openvino_api_interface.get_performance_statistics();

        // API Reset
        openvino_api_interface.api_Reset();
        status = openvino_api_interface.api_status();

        // Stream generation
        std::cout << "Input prompt: " << prompt << "\n";
        // openvino_api_interface.api_Generate(prompt, default_params, callback);
        std::cout << "Start stream generate with new thread.\n";
        std::thread t1(task1, prompt, default_params, callback);
        std::cout << "Sleep 1s in main thread after stream generation started. \n";
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        // openvino_api_interface.api_stop();
        t1.join();

        text_streamer->end();
        status = openvino_api_interface.api_status();

        // Get performance statistics
        llm_perf_statistic = openvino_api_interface.get_performance_statistics();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << "\n";
        exit(EXIT_FAILURE);
    }
    return 0;
}
