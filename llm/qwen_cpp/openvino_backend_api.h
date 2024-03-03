#include "openvino/openvino.hpp"
#include <openvino/runtime/properties.hpp>
#include "openvino/runtime/intel_gpu/properties.hpp"
#include <chrono>
#include "qwen.h"
#include <functional>
#include <numeric>
#include <algorithm>

#if defined(_WIN32) || defined(__CYGWIN__)
#define OPENVINO_CORE_IMPORTS __declspec(dllimport)
#define OPENVINO_CORE_EXPORTS __declspec(dllexport)
#define _OPENVINO_HIDDEN_METHOD
#elif defined(__GNUC__) && (__GNUC__ >= 4) || defined(__clang__)
#define OPENVINO_CORE_IMPORTS __attribute__((visibility("default")))
#define OPENVINO_CORE_EXPORTS __attribute__((visibility("default")))
#define _OPENVINO_HIDDEN_METHOD __attribute__((visibility("hidden")))
#else
#define OPENVINO_CORE_IMPORTS
#define OPENVINO_CORE_EXPORTS
#define _OPENVINO_HIDDEN_METHOD
#endif

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

namespace openvino_backend
{
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
        // Example for callback function used in stream generation case
        void callback(int32_t *new_token_id, bool *stop_generation)
        {
            if (!*stop_generation)
            {
                std::cout << *new_token_id << "\n";
            }
        }
        */

    public:
        // 参数初始化
        OPENVINO_CORE_EXPORTS api_interface(const params &params);
        OPENVINO_CORE_EXPORTS ~api_interface();

        // 加载模型
        OPENVINO_CORE_EXPORTS void api_loadmodel(char *buffer, int thread_num);

        // Load tokenizer with tokenizer path
        OPENVINO_CORE_EXPORTS void api_loadtokenizer(std::string tokenizer_path);

        // Load tokenizer passed with pointer
        OPENVINO_CORE_EXPORTS void api_loadtokenizer(std::shared_ptr<qwen::QwenTokenizer> tokenizer_ptr);

        // 流式接口
        OPENVINO_CORE_EXPORTS bool api_Generate(const std::string &prompt, const params &params, void (*api_callback)(int32_t *new_token_id, bool *_stop_generation));

        // 非流式接口 (Raw prompt)
        OPENVINO_CORE_EXPORTS std::string api_Generate(const std::string &prompt, const params &params);

        // 环境复位
        OPENVINO_CORE_EXPORTS void api_Reset();

        // 卸载模型
        OPENVINO_CORE_EXPORTS bool api_unloadmodel();

        // 获取状态
        OPENVINO_CORE_EXPORTS int api_status();

        // 停止生成
        OPENVINO_CORE_EXPORTS bool api_stop();

        // Get performance statistic
        OPENVINO_CORE_EXPORTS PerformanceStatistic get_performance_statistics();

        int32_t generate_first_token(std::vector<int> &input_ids, const params &params);

        int32_t generate_next_token(int32_t input_token, std::vector<int32_t> history_ids, const params &params);

    private:
        ov::Core _core;
        std::string _device = "GPU";
        const size_t BATCH_SIZE = 1;
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
