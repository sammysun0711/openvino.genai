// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>
#include <openvino_extensions/strings.hpp>
#include <chrono>
#include <vector>
#include <openvino/runtime/properties.hpp>
#include "openvino/runtime/intel_gpu/properties.hpp"


namespace {
std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest&& tokenizer, std::string_view prompt) {
    constexpr size_t BATCH_SIZE = 1;
    ov::Tensor destination = tokenizer.get_input_tensor();
    openvino_extensions::pack_strings(std::array<std::string_view, BATCH_SIZE>{prompt}, destination);
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

void print_token(ov::InferRequest& detokenizer, int32_t out_token) {
    constexpr size_t BATCH_SIZE = 1;
    ov::Tensor inp = detokenizer.get_input_tensor();
    inp.set_shape({BATCH_SIZE, 1});
    inp.data<int32_t>()[0] = out_token;
    detokenizer.infer();
    std::cout << openvino_extensions::unpack_strings(detokenizer.get_output_tensor()).front() << std::flush;
}
}

int main(int argc, char* argv[]) try {
    if (argc != 6) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <openvino_model.xml> <tokenizer.xml> <detokenizer.xml> '<prompt>' '<device>'");
    }
    ov::Core core;
    std::cout << "init core works\n";
    //std::string extension_path = "C:\\Users\\S590\\Documents\\code\\openvino.genai\\build\\llm\\cpp\\Release\\\user_ov_extensions.dll";
    //core.add_extension(extension_path);
    core.add_extension(USER_OV_EXTENSIONS_PATH);  // USER_OV_EXTENSIONS_PATH is defined in root CMakeLists.txt

    ov::AnyMap device_config = {};
    if (argv[5] == "CPU") {
        device_config[ov::cache_dir.name()] = "llm-cache";
        device_config[ov::hint::scheduling_core_type.name()] = ov::hint::SchedulingCoreType::PCORE_ONLY;
        device_config[ov::hint::enable_hyper_threading.name()] = false;
        device_config[ov::hint::enable_cpu_pinning.name()] = true;
        device_config[ov::hint::inference_precision.name()] = "f32";
        device_config[ov::enable_profiling.name()] = false;
    }

    if (argv[5] == "GPU") {
        device_config[ov::cache_dir.name()] = "llm-cache";
        device_config[ov::intel_gpu::hint::queue_throttle.name()] = ov::intel_gpu::hint::ThrottleLevel::MEDIUM;
        device_config[ov::intel_gpu::hint::queue_priority.name()] = ov::hint::Priority::MEDIUM;
        device_config[ov::intel_gpu::hint::host_task_priority.name()] = ov::hint::Priority::HIGH;
        device_config[ov::hint::enable_cpu_pinning.name()] = true;
        device_config[ov::enable_profiling.name()] = false;
    }


    std::cout << "Load OpenVINO extension\n";
    auto [input_ids, attention_mask] = tokenize(core.compile_model(argv[2], "CPU").create_infer_request(), argv[4]);
    std::cout << "Init tokenize works\n";
    ov::InferRequest detokenizer = core.compile_model(argv[3], "CPU").create_infer_request();
    std::cout << "Init detokenize works\n";
    auto read_model_start = std::chrono::high_resolution_clock::now();
    std::shared_ptr<ov::Model> model = core.read_model(argv[1]);
    auto read_model_stop = std::chrono::high_resolution_clock::now();
    auto read_model_duration = std::chrono::duration_cast<std::chrono::milliseconds>(read_model_stop - read_model_start);
    std::cout << "Read llm model time: " << read_model_duration.count() / 1000.0 << " s\n";
    constexpr size_t BATCH_SIZE = 1;
    /*
    std::map<size_t, ov::PartialShape> shapes = {
        {0, ov::PartialShape{
            BATCH_SIZE, -1
        }},
        {1, ov::PartialShape{
            BATCH_SIZE, -1
        }},
        //{2, ov::PartialShape{
	//    BATCH_SIZE, -1
        //}}
    };
    */
    std::map<size_t, ov::PartialShape> shapes;
    std::vector<ov::Output<ov::Node>> inputs = model->inputs();
    //std::cout << "inputs.size(): " << inputs.size() << std::endl;
    /*
    for (size_t idx = 0; idx < inputs.size(); ++idx) {
	std::cout << "idx: "  << idx << "\n";  
        ov::PartialShape shape = inputs.at(idx).get_partial_shape();
        shape[0] = BATCH_SIZE;
        shapes.emplace(idx, shape);
    }*/

    //std::cout << "get model inputs works\n";
    for (size_t idx = 0; idx < inputs.size(); ++idx) { 
        ov::PartialShape shape = inputs.at(idx).get_partial_shape();
        shape[0] = BATCH_SIZE;
        shapes.emplace(idx, shape);
    }
    //std::cout << "get inputs shape works\n";
    model->reshape(shapes);
    //std::cout << "model reshape works\n";
    auto compile_model_start = std::chrono::high_resolution_clock::now();
    //ov::InferRequest ireq = core.compile_model(model, "CPU", ov::cache_dir("llm-cache")).create_infer_request();
    ov::InferRequest ireq = core.compile_model(model, argv[5], device_config).create_infer_request();
    auto compile_model_stop = std::chrono::high_resolution_clock::now();
    auto compile_model_duration = std::chrono::duration_cast<std::chrono::milliseconds>(compile_model_stop - compile_model_start);
    std::cout << "Compile llm model on " << argv[5] << " time: " << compile_model_duration.count() / 1000.0 << " s\n";
    for (size_t idx = 2; idx < inputs.size(); ++idx) {
	    //std::cout << "idx: " << idx << "\n"; 
	    //std::cout << "inputs.at(idx).get_partial_shape().get_min_shape() " << inputs.at(idx).get_partial_shape().get_min_shape() << "\n";
        ireq.get_input_tensor(idx).set_shape(inputs.at(idx).get_partial_shape().get_min_shape());
    }
    //std::cout << "get input tensor shape works\n";
    ireq.get_tensor("input_ids").set_shape(input_ids.get_shape());  // TODO: replace with ireq.set_tensor("input_ids", input_ids); after it's fixed
    ireq.get_tensor("attention_mask").set_shape(input_ids.get_shape());
    //ireq.set_tensor("input_ids", input_ids);
    std::copy_n(input_ids.data<const int64_t>(), input_ids.get_size(), ireq.get_tensor("input_ids").data<int64_t>());
    std::fill_n(ireq.get_tensor("attention_mask").data<int64_t>(), input_ids.get_size(), 1);
    //ireq.get_tensor("position_ids").set_shape(input_ids.get_shape());
    //std::iota(ireq.get_tensor("position_ids").data<int64_t>(), ireq.get_tensor("position_ids").data<int64_t>() + ireq.get_tensor("position_ids").get_size(), 0);

    //std::cout << "set input works\n";
    int output_tokens_count = 0;
    auto first_infer_start = std::chrono::high_resolution_clock::now(); 
    ireq.infer();
    auto first_infer_stop = std::chrono::high_resolution_clock::now();
    auto first_infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(first_infer_stop - first_infer_start);
    output_tokens_count+=1;
    //std::cout << "first inference time: " << first_infer_duration.count() / 1000.0 << " s\n";

    size_t vocab_size = ireq.get_tensor("logits").get_shape().back();
    //std::cout << "vocab_size: " << vocab_size << "\n";
    float* logits = ireq.get_tensor("logits").data<float>() + (input_ids.get_size() - 1) * vocab_size;
    int32_t out_token = int32_t(std::max_element(logits, logits + vocab_size) - logits);
    //std::cout << "first output token " << out_token << "\n";

    ireq.get_tensor("input_ids").set_shape({BATCH_SIZE, 1});
    //ireq.get_tensor("position_ids").set_shape({BATCH_SIZE, 1});
    constexpr int32_t SPECIAL_EOS_TOKEN = 2;  // There's no way to extract the value from the detokenizer for now
    
    std::vector<double> second_tokens_latency;
    auto second_infer_start = std::chrono::high_resolution_clock::now();
    auto second_infer_stop = std::chrono::high_resolution_clock::now();
    auto second_infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(second_infer_stop - second_infer_start);
    while (out_token != SPECIAL_EOS_TOKEN) {
	    //std::cout << "ireq.get_tensor(attention_mask).get_size() before: " << ireq.get_tensor("attention_mask").get_size() << std::endl;
        ireq.get_tensor("input_ids").data<int64_t>()[0] = out_token;
        ireq.get_tensor("attention_mask").set_shape({BATCH_SIZE, ireq.get_tensor("attention_mask").get_shape()[1] + 1});
	    //std::cout << "ireq.get_tensor(attention_mask).get_size() after: " << ireq.get_tensor("attention_mask").get_size() << std::endl;
        std::fill_n(ireq.get_tensor("attention_mask").data<int64_t>(), ireq.get_tensor("attention_mask").get_size(), 1);
        //ireq.get_tensor("position_ids").data<int64_t>()[0] = ireq.get_tensor("attention_mask").get_size() - 2;
        for (size_t idx = 2; idx < inputs.size(); ++idx) {
            ireq.set_input_tensor(idx, ireq.get_output_tensor(idx - 1));
        }
	    second_infer_start = std::chrono::high_resolution_clock::now();
        ireq.start_async();
        print_token(detokenizer, out_token);
	    //std::cout << "print token works\n";
        ireq.wait();
	    second_infer_stop = std::chrono::high_resolution_clock::now();
	    second_infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(second_infer_stop - second_infer_start);
	    second_tokens_latency.push_back(second_infer_duration.count());
	    //std::cout << count << "th infer works\n";
        logits = ireq.get_tensor("logits").data<float>();
        out_token = int32_t(std::max_element(logits, logits + vocab_size) - logits);
	    //std::cout << "out_token: " << out_token << "\n";
        output_tokens_count += 1;
    }
    std::cout << '\n';
    /*
    for (auto & t : second_tokens_latency) {
        std::cout << "latency: " << t << " ms\n";
    }
    */
    double sum = std::accumulate(second_tokens_latency.begin()+1, second_tokens_latency.end(), 0.0);
    double avg_second_latency = sum / second_tokens_latency.size();
    std::cout << "Input num tokens: " << input_ids.get_size() << ", output num tokens: " << output_tokens_count << "\n";
    std::cout << "First token latency: " << first_infer_duration.count() / 1000.0 << " s\n";
    std::cout << "Second token latency: " << second_tokens_latency[0] << " ms\n";
    std::cout << "Average 2nd+ latency per token: " << avg_second_latency << " ms\n";
    std::cout << "Average 2nd+ tokens per seconds: " << std::setprecision(3) << 1000.0 / avg_second_latency << " tokens/s \n";

    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
