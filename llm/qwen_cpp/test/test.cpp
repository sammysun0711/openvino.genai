#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include "openvino_backend_api.h"

ov_params default_params;
// 参数初始化
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
        // std::string content = tokenizer->decode({*new_token_id});
    }
}
void task1(const std::string &prompt, const ov_params &params, void (*api_callback)(int32_t *new_token_id, bool *_stop_generation))
{
    openvino_api_interface.api_Generate(prompt, default_params, callback);
}

int main()
{
    try
    {
        std::string model_path = "Qwen-7B-Chat-NNCF_INT4\\openvino_model.xml";
        std::string user_prompt = "What is OpenVINO?";
        // std::string user_prompt = "请介绍下清华大学";
        std::string system_message = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>";
        std::ostringstream oss_prompt;
        oss_prompt << system_message << "\n<|im_start|>user\n"
                   << user_prompt << "<|im_end|>\n<|im_start|>assistant\n";
        std::string prompt = oss_prompt.str();
        char *cstr = new char[model_path.length() + 1];
        std::strcpy(cstr, model_path.c_str());

        // 获取状态
        auto status = openvino_api_interface.api_status();

        // 通过本地路径加载模型
        openvino_api_interface.api_loadmodel(cstr, 0);
        status = openvino_api_interface.api_status();

        // 卸载模型
        auto success = openvino_api_interface.api_unloadmodel();
        status = openvino_api_interface.api_status();
        std::cout << "Sleep 5s to check if GPU memory released after model unload\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));

        // 通过内存buffer加载模型
        std::string encrypt_model_path = default_params.model_path;
        std::string encrypt_weights_path = std::regex_replace(model_path, std::regex(".xml"), ".bin");
        std::ifstream model_file(encrypt_model_path, std::ios::in | std::ios::binary);
        std::ifstream weights_file(encrypt_weights_path, std::ios::in | std::ios::binary);
        // User can add file decryption of model_file and weights_file in memory here
        std::vector<uint8_t> model_buffer((std::istreambuf_iterator<char>(model_file)), std::istreambuf_iterator<char>());
        std::vector<uint8_t> weights_buffer((std::istreambuf_iterator<char>(weights_file)), std::istreambuf_iterator<char>());
        openvino_api_interface.api_loadmodel(&model_buffer, &weights_buffer, 0);
        status = openvino_api_interface.api_status();

        // 通过路径加载Tokenizer
        openvino_api_interface.api_loadtokenizer(tokenizer_path);

        // 卸载Tokenizer
        success = openvino_api_interface.api_unloadtokenizer();
        openvino_api_interface.api_loadtokenizer(tokenizer_path);

        // 非流式生成接口
        std::cout << "Input prompt: " << prompt << "\n";
        std::string response = openvino_api_interface.api_Generate(prompt, default_params);
        std::cout << "Generated response: " << response << "\n";
        status = openvino_api_interface.api_status();

        // 获取性能数据
        auto llm_perf_statistic = openvino_api_interface.get_performance_statistics();

        // 环境复位
        openvino_api_interface.api_Reset();
        status = openvino_api_interface.api_status();

        // 流式生成接口
        std::cout << "Input prompt: " << prompt << "\n";
        openvino_api_interface.api_Generate(prompt, default_params, callback);
        llm_perf_statistic = openvino_api_interface.get_performance_statistics();
        openvino_api_interface.api_Reset();
        status = openvino_api_interface.api_status();

        // 子线程调用流式生成， 主线程调用api_stop()实现停止生成
        std::cout << "\nStart stream generate with new thread.\n";
        std::thread t1(task1, prompt, default_params, callback);
        std::cout << "Sleep 1s in main thread after stream generation started.\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        std::cout << "\nCall api_stop() to stop stream generation.\n";
        openvino_api_interface.api_stop();
        t1.join();

        text_streamer->end();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << "\n";
        exit(EXIT_FAILURE);
    }
    return 0;
}
