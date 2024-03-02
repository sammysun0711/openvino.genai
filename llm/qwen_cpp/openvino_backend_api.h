#include "openvino/openvino.hpp"
#include <openvino/runtime/properties.hpp>
#include "openvino/runtime/intel_gpu/properties.hpp"
#include <chrono>
#include "qwen.h"
#include <functional>
#include <numeric>
#include <algorithm>

struct params
{
    std::string model_path = "openvino_model.xml";
    std::string tokenizer_path = "qwen.tiktoken";
    int32_t n_ctx = 2048;
    int32_t n_predict = 512;
    bool do_sample = true;
    int32_t top_k = 40;
    float top_p = 0.90;
    float temperature = 0.20;
    float repeat_penalty = 1.10;
    int32_t repeat_last_n = 32;
    int32_t seed = -1;
    std::string model_cache_dir = "openvino_cache";
    std::string device = "GPU";
};

struct PerformanceStatistic
{
    // LLM Model
    double llm_load_duration = 0.0;
    double llm_unload_duration = 0.0;
    double llm_cancel_duration = 0.0;

    // Tokenizer
    double tokenizer_load_duration = 0.0;

    // Generation
    double llm_first_infer_duration = 0.0;
    double llm_prompt_evaluation_speed = 0.0;
    double llm_generate_next_token_duration = 0.0;
    double llm_average_token_per_second = 0.0;
    int input_token_num = 0;
    int generated_token_num = 0;
};

namespace openvino_backend
{
    // 流式接口回调函数
    // auto api_callback = [](Tokenize token, void *lambda);
    /*auto api_callback = [](int32_t token_id){
        std::cout << "API callback called called \n";
        std::cout << "token_id: " << token_id;
    };

    std::function<int32_t token_id> api_callback { std::cout << "token_id: " << token_id; };

    void print_token(int32_t i)
    {
        std::cout << i << '\n';
    }

    std::function<void()> api_callback = [](int32_t token_id)
    //{ print_token(token_id); };

    int32_t api_callback = [](const int32_t token_id)
    {
        std::cout << "API callback called called \n";
        std::cout << "token_id: " << token_id;
    };
    */

    enum status
    {
        init = 0,      // Init parameters
        loaded = 1,    // Model loaded or reset
        unloaded = 2,  // Unload model -> Release model and tokenizer
        inference = 3, // Running generation
        error = -1     // General error
    };

    class api_interface
    {
        /*
        void api_callback(int32_t *new_token_id, bool *new_token_available)
        {
            //*new_token_id = _new_token_id;
            //*new_token_available = _new_token_available;
            std::cout << *new_token_id << "\n";
            std::cout << *new_token_available << "\n";
        }
        */

    public:
        // 参数初始化
        api_interface(const params &params);
        ~api_interface();

        // 加载模型
        void api_loadmodel(char *buffer, int thread_num);

        // Load tokenizer with tokenizer path
        void api_loadtokenizer(std::string tokenizer_path);

        // Load tokenizer passed with pointer
        void api_loadtokenizer(std::shared_ptr<qwen::QwenTokenizer> tokenizer_ptr);

        // 流式接口
        bool api_Generate(const std::string &prompt, const params &params, void(*api_callback)(int32_t *new_token_id, bool *_stop_generation));

        // 非流式接口 (Raw prompt)
        std::string api_Generate(const std::string &prompt, const params &params);

        // 环境复位
        void api_Reset();

        // 卸载模型
        bool api_unloadmodel();

        // 获取状态
        int api_status();

        // 停止生成
        bool api_stop();

        const size_t BATCH_SIZE = 1;

        int32_t generate_first_token(std::vector<int> &input_ids, const params &params);

        int32_t generate_next_token(int32_t input_token, std::vector<int32_t> history_ids, const params &params);

        PerformanceStatistic get_performance_statistics();

    private:
        ov::Core _core;
        std::string _device = "GPU";
        std::string _model_cache_dir = "openvino_model_cache";
        std::unique_ptr<ov::InferRequest> _infer_request = nullptr;
        ov::AnyMap _device_config = {};
        PerformanceStatistic _perf_statistic;
        qwen::QwenConfig _tokenizer_config;
        std::shared_ptr<qwen::QwenTokenizer> _tokenizer = nullptr;
        size_t _vocab_size;
        int32_t _seed;
        status _api_status;
        int32_t _new_token_id = _tokenizer_config.im_end_id;
        bool _stop_generation = false;
    }; // api_interface
} // namespace openvino_backend
