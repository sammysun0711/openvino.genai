
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <regex>
#include <random>
#include <openvino/openvino.hpp>
#include <openvino/runtime/properties.hpp>
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/serialize.hpp"
#include "sampling.hpp"

#ifdef _WIN32
#include <codecvt>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

const std::string sentences[] =
{
    "What is OpenVINO?",
    "If I have 100 million dollars, what kinds of projects should I invest to maximize my benefits in background of a growing number of artificial intelligence technologies?",
    "Originally, There were three types of cake in the cake store: Strawberry Cream Cake, Chocolate Coconut Cake, and Red Velvet Brownie Cake. Customer number is large enough so that no cake would be left every day when the store close. As the name suggested, each cake has two ingredients: Strawberry Cream Cake with strawberries and cream, Chocolate Coconut Cake with chocolate and coconut, and Red Velvet Brownie Cake with red velvet and brownie. Different ingredients can be compatibly mixed with each other without any issue. After the cake is made, there are often some leftover materials for each ingredient. In order to reduce waste, the store often combine the extra ingredients in pairs to make new small gifts to gain extra sales. For example, strawberries and chocolate can be mixed to create strawberry-flavored chocolate sauce, and brownies and shredded coconut can be mixed to create brownie coconut cookies. Only two ingredients can be mixed, and mixture with more than two ingredients can cost a lot of time and will not be adopted. In order to decrease the problem complexity, the store will also prevent from careful decorations or other difficult steps as in procedure of making cakes, so that time cost can be omited. By analogy, if all the ingredients can be combined in pairs, what small products can the store make in the end?",
    "There is a table, which contains three drawers: left drawer, middle drawer and right drawer; Tom Ethan, Elbert Alex, Jack Johnson, and Mario Thompson all saw a bag of chocolates on the table. Tom Ethan asked Elbert Alex and Jack Johnson to go out, and after that, he put the bag of chocolates in the right drawer in front of Mario Thompson; after Jack Johnson came back, Tom Ethan asked Mario Thompson to go out to find Elbert Alex, and took it from the left drawer in front of Jack Johnson. Then He take out a box of biscuits and put them in the middle drawer; when Elbert Alex and Mario Thompson returned, Tom Ethan asked Jack Johnson and Mario Thompson to go out to buy a bottle of soy sauce. Tom Ethan waited for a long time, and found that Jack Johnson and Mario Thompson had not returned, so he sent Elbert Alex to look for them, but in the end only Jack Johnson and Elbert Alex came back. Jack Johnson told Tom Ethan that at first they could not find any shop that is providing soy sauce, so they had to separate to search other shops, which is why Mario Thompson got lost; on the way back, Jack Johnson ran into Elbert Alex, and they rushed back first. Therefore, Tom Ethan asked them to go out to find Mario Thompson again; in order to prevent getting lost again, Tom Ethan told Elbert Alex and Jack Johnson to walk together at all time, and even if they could not get the soy sauce, they had to find and get back with Mario Thompson. As a result, Elbert Alex and Jack Johnson found Mario Thompson outside and found that he had bought a bottle of soy sauce. The three felt that Tom Ethan never went out to do anthing but they are busy all the time. So they were very angry. They discussed and made a conclusion. After going back to see Tom Ethan, they should not tell him about the soy sauce they bought, and asked Jack Johnson to hide the soy sauce in his backpack. After the three of them came back together, they pretended to claim that they did not foudn and bought soy sauce according to the plan, and hoped that Tom Ethan would go out together to buy things in the future, and he should not be so lazy. Tom Ethan agreed and felt sory about that. When everyone finally stood in front of the table, the four of them wrote down the list of items they knew and the location of the items. So the question is: is the information writen by these four people consistent, and why?",
    "The process of Origami seems simple at the first glance, but in fact, it still requires a very complicated process to do it well. Taking folding a rose as an example, we can divide the entire process into three stages, including: firstly creating a grid of creases, secondly making a three-dimensional base, and thirdly finishing petal decoration. The first step is to create a grid of creases: this step is a bit like the first step of folding a gift of thousand-paper-crane. That is to say, we can fold the paper in half (or namedly equal-folds) through the symmetrical axis, and repeat such step in the other symmetrical axis. And then apply multiple equal-folds in sequence relative to each smaller rectangle divided by the two creases; After that, the creases in each direction will interweave into a complete set of uniform small square splicing patterns; these small squares form a reference space similar to a two-dimensional coordinate system, allowing us to combine adjacent creases on the plane from Three-dimensional high platforms or depressions are folded on the two-dimensional small squares to facilitate the next steps of folding. It should be noted that, in the process of creating grid creases, there may be rare cases when the folds are not aligned. The consequences of this error can be very serious. And just like the butterfly effect, it is only a slight difference at the beginning , and in the end it may generate a disaster world which is completely different from plan. Anyway, let's continue. The second step is make the three-dimensional base: In this step, we need to fold a set of symmetrical three-dimensional high platforms or depressions based on the grid creases. From the symmetry analysis, it is not difficult to find that the rose will have four symmetrical three-dimensional high platforms and supporting depressions. Therefore, we can firstly fold out a quarter of the depression and plateau patterns, which would help build a base to compose into a complex 3D structure. And then, we use this quarter as a template, and fold out the repeating patterns on the remaining three parts of the whole structure in turn. It is worth noting that the layout of the high platform not only needs to consider the regular contrast and symmetrical distribution of the length and width, but also needs to ensure the orderliness of the height dimension. This is very important, since we will never go back to this step after all parts were made, and you would better start from first step if you make anything wrong in the this step. Similar to the precautions in the first stage, please handle all the corners in three dimensions to ensure that they conform to the layout required in the plan, which would help us avoid the butterfly effect and increase the robustness in the process of three-dimensional folding. Just like building a skyscrapper in the real world, people usually take a lot of time when building the base but soon get finished when extending the structure after that. Time is worth to cost in the base, but would be saved in the future after you succeed in base. Anyway, let's continue. During the first quarter of the pattern, repeated comparisons with the finished rose were made to eliminate any possible errors in the first place. The final stage is to finish the petal grooming. At this stage, we often emphasize an important term called folding-by-heart. The intention here is no longer literally serious, but focus is moved to our understanding of the shape of a rose in nature, and we usually use natural curves to continuously correct the shape of petals in order to approach the shape of rose petals in reality. One more comment: this is also the cause of randomness to the art, which can be generated differently by different people. Recall that rose should be adjusted close to reality, so in the last step of this stage, we need to open the bloom in the center of the rose, by pulling on the four petals that have been bent. This process may be accompanied by the collapse of the overall structure of the rose, so we should be very careful to save strength of adjustment, and it must be well controlled to avoid irreversible consequences. Ultimately, after three stages of folding, we end up with a crown of rose with a similar shape close to reality. If condition is permited, we can wrap a green paper strip twisted on a straightened iron wire, and insert the rose crown we just created onto one side of the iron wire. In this way, we got a hand-made rose with a green stem. We can also repeat the steps above to increase the number of rose, so that it can be made into a cluster. Different color of rose is usually more attractive and can be considered as a better plan of gift to your friend. In summary, by creating a grid of creases, making a three-dimensional base, and finishing with petals, we created a three-dimensional rose from a two-dimensional paper. Although this process may seem simple, it is indeed a work of art created by us humans with the help of imagination and common materials. At last, Please comment to assess the above content.",
};

namespace {

struct Args {
    std::string ov_model_path = "openvino_model.xml";
    std::string token_model_path = "tokenizer.xml";
    std::string detoken_model_path = "detokenizer.xml";
    std::string device = "GPU";
    bool reduce_logits = false;
    bool do_sample = false;
    int top_k = 0;
    float top_p = 0.7;
    float temp = 0.95;
    float repeat_penalty = 1.0;
    int output_fixed_len = 0;
    bool force_max_generation = false;
    bool enable_sdpa = false;
    bool enable_dynamic_quantization = false;
    bool use_stateless = false;
    std::string cache_dir = "llm-cache";
};

static void usage(const std::string& prog) {
    std::cout << "Usage: " << prog << " [options]\n"
        << "\n"
        << "options:\n"
        << "  -h, --help              show this help message and exit\n"
        << "  -m, --model PATH        Phi-3 OpenVINO model path (default: openvino_model.xml)\n"
        << "  -token PATH             Tokenizer model path (default: tokenizer.xml)\n"
        << "  -detoken PATH           DeTokenizer model path (default: detokenizer.xml)\n"
        << "  -d, --device            Device (default: GPU)\n"
        << "  --reduce_logits         Reduce_logits (default: False)\n"
        << "  --do_sample             Search (default: False)\n"
        << "  --top_k N               top-k sampling (default: 0)\n"
        << "  --top_p N               top-p sampling (default: 0.7)\n"
        << "  --temp N                temperature (default: 0.95)\n"
        << "  --repeat_penalty N      penalize repeat sequence of tokens (default: 1.0, 1.0 = disabled)\n"
        << "  --output_fixed_len N    set output fixed lenth (default: 0, output lenth is determined by the model)\n"
	<< "  --force_max_generation  force llm to generate until fixed length \n"
        << "  --cache_dir             Cache directory to store model cache, default is 'llm_cache', set to '' to disable model cache\n"
        << "  --enable_sdpa           Enable SDPA Optimization (default: false)\n"
        << "  --enable_dynamic_quantization Enable Dynamic quantization optimizaton\n"
        << "  --use_stateless         Use statelsss model text generation, (default: false\n";
}

static Args parse_args(const std::vector<std::string>& argv) {
    Args args;

    for (size_t i = 1; i < argv.size(); i++) {
        const std::string& arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        }
        else if (arg == "-m" || arg == "--model") {
            args.ov_model_path = argv[++i];
        }
        else if (arg == "-token") {
            args.token_model_path = argv[++i];
        }
        else if (arg == "-detoken") {
            args.detoken_model_path = argv[++i];
        }
        else if (arg == "-d" || arg == "--device") {
            args.device = argv[++i];
        }
        else if (arg == "--reduce_logits") {
            args.reduce_logits = true;
        }
        else if (arg == "--do_sample") {
            args.do_sample = true;
        }
	else if (arg == "--force_max_generation") {
            args.force_max_generation = true;
        }
        else if (arg == "--top_k") {
            args.top_k = std::stoi(argv[++i]);
        }
        else if (arg == "--top_p") {
            args.top_p = std::stof(argv[++i]);
        }
        else if (arg == "--temp") {
            args.temp = std::stof(argv[++i]);
        }
        else if (arg == "--repeat_penalty") {
            args.repeat_penalty = std::stof(argv[++i]);
        }
        else if (arg == "--output_fixed_len") {
            args.output_fixed_len = std::stoi(argv[++i]);
        }
	else if (arg == "--cache_dir") {
            args.cache_dir = argv[++i];
        }
	else if (arg == "--enable_sdpa") {
            args.enable_sdpa = true;
        }
        else if (arg == "--enable_dynamic_quantization") {
            args.enable_dynamic_quantization = true;
        }
	 else if (arg == "--use_stateless")
	{
	    args.use_stateless = true;
	}
	else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    return args;
}

static Args parse_args(int argc, char** argv) {
    std::vector<std::string> argv_vec;
    argv_vec.reserve(argc);

#ifdef _WIN32
    LPWSTR* wargs = CommandLineToArgvW(GetCommandLineW(), &argc);

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(converter.to_bytes(wargs[i]));
    }

    LocalFree(wargs);
#else
    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(argv[i]);
    }
#endif

    return parse_args(argv_vec);
}

int64_t get_out_token_id(const std::vector<int>& input_ids, float* logits, size_t vocab_size, Args args) {
    int64_t out_token;

    // logits pre-process
    if (args.repeat_penalty != 1.f) {
        sampling_repetition_penalty(logits, logits + vocab_size, input_ids, args.repeat_penalty);
    }

    if (args.do_sample)
    {
        if (args.temp > 0) {
            sampling_temperature(logits, logits + vocab_size, args.temp);
        }

        std::vector<TokenIdScore> token_scores(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            token_scores[i] = TokenIdScore(i, logits[i]);
        }

        // top_k sampling
        if (0 < args.top_k && args.top_k < (int)token_scores.size()) {
            sampling_top_k(token_scores.data(), token_scores.data() + args.top_k,
                token_scores.data() + token_scores.size());
            token_scores.resize(args.top_k);
        }

        // top_p sampling
        if (0.f < args.top_p && args.top_p < 1.f) {
            auto pos = sampling_top_p(token_scores.data(), token_scores.data() + token_scores.size(), args.top_p);
            token_scores.resize(pos - token_scores.data());
        }

        // sample next token
        sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());
        for (size_t i = 0; i < token_scores.size(); i++) {
            logits[i] = token_scores[i].score;
        }

        thread_local std::random_device rd;
        thread_local std::mt19937 gen(rd());

        std::discrete_distribution<> dist(logits, logits + token_scores.size());
        out_token = token_scores[dist(gen)].id;
    }
    else {
        out_token = std::max_element(logits, logits + vocab_size) - logits;
    }

    return out_token;
}

std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string && prompt) {
    constexpr size_t BATCH_SIZE = 1;
    tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt});
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

std::string detokenize(ov::InferRequest& detokenizer, std::vector<int64_t>& tokens) {
    constexpr size_t BATCH_SIZE = 1;
    detokenizer.set_input_tensor(ov::Tensor{ov::element::i64, {BATCH_SIZE, tokens.size()}, tokens.data()});
    detokenizer.infer();
    return detokenizer.get_output_tensor().data<std::string>()[0];
}

// The following reasons require TextStreamer to keep a cache of previous tokens:
// detokenizer removes starting ' '. For example detokenize(tokenize(" a")) == "a",
// but detokenize(tokenize("prefix a")) == "prefix a"
// 1 printable token may consist of 2 token ids: detokenize(incomplete_token_idx) == "�"
struct TextStreamer {
    ov::InferRequest detokenizer;
    std::vector<int64_t> token_cache;
    size_t print_len = 0;

    void put(int64_t token) {
        token_cache.push_back(token);
        std::string text = detokenize(detokenizer, token_cache);
        if (!text.empty() && '\n' == text.back()) {
            // Flush the cache after the new line symbol
            std::cout << std::string_view{text.data() + print_len, text.size() - print_len};
            token_cache.clear();
            print_len = 0;
            return;
        }
        if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
            // Don't print incomplete text
            return;
        }
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
        print_len = text.size();
    }

    void end() {
        std::string text = detokenize(detokenizer, token_cache);
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << '\n';
        token_cache.clear();
        print_len = 0;
    }
};

static double get_duration_ms_until_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
}

/*@brief Insert slice transformation matches following graph, start from logits (Results) to search along root->parent-> grandparent node,
 * then insert slice between Reshape (grandparent node) and Matmul to keep only last dim of matmul first input, first input shape reduced
 * from [1, seq_len, 4096] to [1, 1,4096]. Therefore, after graph transformation, we can reduce matmul computation
 * from [1, seq_len, 4096] * [1, 4096, 151936] = [1, seq_len, 151936] to [1,1,4096]*[4096,151936] = [1,1,151936]
 *
 * Original graph
 *         +----------+            +----------+
 *         |  Reshape |            | Constant |
 *         +----------+            +----------+
 *              |                       |
 *              -----------    ----------
 *                        |    |
 *                        v    v
 *                      +--------+
 *                      | MatMul |
 *                      +--------+
 *                          |
 *                          v
 *                     +----------+
 *                     |  logits  |
 *                     +----------+
 *
 * Modified graph after insert slice:
 *
 *         +----------+            +----------+
 *         |  Reshape |            | Constant |
 *         +----------+            +----------+
 *              |                       |
 *         +----------+                 |
 *         |  Slice   |                 |
 *         +----------+                 |
 *              |                       |
 *              -----------    ----------
 *                        |    |
 *                        v    v
 *                      +--------+
 *                      | MatMul |
 *                      +--------+
 *                          |
 *                          v
 *                     +----------+
 *                     |  logits  |
 *                     +----------+
*/

class InsertSlice : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertSlice", "0");
    explicit InsertSlice() {
        auto label = ov::pass::pattern::wrap_type<ov::op::v0::Result>();
        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            auto root = std::dynamic_pointer_cast<ov::op::v0::Result>(m.get_match_root());
            if (!root) {
                return false;
            }
            std::string root_name = root->get_friendly_name();
            if (root->get_output_partial_shape(0).size() == 3) {
                std::cout << "Find target root node name: " << root_name << "\n";
                auto parent = root->input_value(0).get_node_shared_ptr();
                std::cout << "Find parent node name: " << parent->get_friendly_name() << "\n";

                //llama2
                auto grand_parent = parent->input_value(0).get_node_shared_ptr();
                std::cout << "Find grandparent node name: " << grand_parent->get_friendly_name() << "\n";

                ov::Output<ov::Node> grand_parent_output = parent->get_input_source_output(0); // parent->get_input_source_output(0);
                std::set<ov::Input<ov::Node>> consumers = grand_parent_output.get_target_inputs();
                auto partial_shape = grand_parent_output.get_partial_shape().get_min_shape();
                int32_t dims = static_cast<int32_t>(partial_shape[2]);
		    
                std::vector<int32_t> start_v = { 0, -1, 0 };
                std::vector<int32_t> stop_v = { 1, -2, dims };
                std::vector<int32_t> step_v = { 1, -1, 1 };

                std::cout << "Original reshape node output shape:" << grand_parent_output.get_partial_shape() << std::endl;
                auto starts = ov::op::v0::Constant::create(ov::element::i32,
                    ov::Shape{ 3 },
                    start_v);
                auto stop = ov::op::v0::Constant::create(ov::element::i32,
                    ov::Shape{ 3 },
                    stop_v);
                auto step = ov::op::v0::Constant::create(ov::element::i32,
                    ov::Shape{ 3 },
                    step_v);
                auto slice = std::make_shared<ov::opset13::Slice>(grand_parent, starts, stop, step); //data, starts, ends, steps
                std::cout << "After insert slice node, output shape" << slice->output(0).get_partial_shape() << std::endl;
                for (auto consumer : consumers) {
                    consumer.replace_source_output(slice->output(0));
                }
                register_new_node(slice);
            }

            return true;
            };
        // Register pattern with Parameter operation as a pattern root node
        auto m = std::make_shared<ov::pass::pattern::Matcher>(label, "InsertSlice");
        // Register Matcher
        register_matcher(m, callback);
    }
};

}

int main(int argc, char* argv[]) try {

    Args args = parse_args(argc, argv);

    std::cout << ov::get_openvino_version() << std::endl;
	
    ov::Core core;
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in root CMakeLists.txt
    auto startTime = Time::now();
    auto tokenizer_model = core.read_model(args.token_model_path);
    ov::InferRequest tokenizer = core.compile_model(tokenizer_model, "CPU").create_infer_request();
    auto input_ids = tokenizer.get_tensor("input_ids");
    auto attention_mask = tokenizer.get_tensor("attention_mask");
    ov::InferRequest detokenizer = core.compile_model(args.detoken_model_path, "CPU").create_infer_request();
    auto duration_ms = get_duration_ms_until_now(startTime);
    std::cout << "Load Phi-3 tokenizer took " << duration_ms << " ms" << std::endl;
    std::string device = args.device;
    constexpr size_t BATCH_SIZE = 1;

    ov::AnyMap device_config = {};
    if (device.find("CPU") != std::string::npos) {
        device_config[ov::cache_dir.name()] = args.cache_dir;
        device_config[ov::hint::scheduling_core_type.name()] = ov::hint::SchedulingCoreType::PCORE_ONLY;
        device_config[ov::hint::enable_hyper_threading.name()] = false;
        device_config[ov::hint::enable_cpu_pinning.name()] = true;
        device_config[ov::enable_profiling.name()] = false;
    }

    if (device.find("GPU") != std::string::npos) {
        device_config[ov::cache_dir.name()] = args.cache_dir;
        device_config[ov::intel_gpu::hint::queue_throttle.name()] = ov::intel_gpu::hint::ThrottleLevel::MEDIUM;
        device_config[ov::intel_gpu::hint::queue_priority.name()] = ov::hint::Priority::MEDIUM;
        device_config[ov::intel_gpu::hint::host_task_priority.name()] = ov::hint::Priority::HIGH;
        device_config[ov::hint::enable_cpu_pinning.name()] = true;
        device_config[ov::enable_profiling.name()] = false;
        device_config[ov::intel_gpu::hint::enable_sdpa_optimization.name()] = args.enable_sdpa;
	if (args.enable_dynamic_quantization){
	    size_t group_size = 128;
	    device_config[ov::hint::dynamic_quantization_group_size.name()] = group_size;
	}
    }

    double total_time = 0;
    int count = 0;
    double first_time;
    
    // Read OpenVINO Model
    if (args.reduce_logits) {
        startTime = Time::now();
        std::shared_ptr<ov::Model> model = core.read_model(args.ov_model_path);
        duration_ms = get_duration_ms_until_now(startTime);
        std::cout << "Read minicpm Model took " << duration_ms << " ms" << std::endl;

        std::cout << "######## [Model Graph Optimization] Step 2: Insert slice node after reshape to reduce logits operation ########\n";
        ov::pass::Manager manager;
        manager.register_pass<InsertSlice>();
        manager.run_passes(model);

        std::string modifiled_file = std::regex_replace(args.ov_model_path, std::regex("openvino_model"), "modified_openvino_model");
        std::cout << "Save modified model in " << modifiled_file << "\n";
        ov::serialize(model, modifiled_file);

        ov::CompiledModel compilemodel = core.compile_model(modifiled_file, device, device_config);

        return 0;
    }

    //Compile model
    startTime = Time::now();
    ov::CompiledModel compilemodel = core.compile_model(args.ov_model_path, device, device_config);
    ov::InferRequest ireq = compilemodel.create_infer_request();
    duration_ms = get_duration_ms_until_now(startTime);
    std::cout << "Compile LLM model took " << duration_ms << " ms" << std::endl;

    std::cout << "Model inputs: \n";
    auto model_inputs = compilemodel.inputs();
    /*
    for (size_t idx = 0; idx < model_inputs.size(); ++idx) {
        std::cout << "idx: " << idx << ", name: " << model_inputs[idx].get_any_name() << "\n";
    }

    std::cout << "Model outputs: \n";
    auto model_outputs = compilemodel.outputs();
    for (size_t idx = 0; idx < model_outputs.size(); ++idx) {
        std::cout << "idx: " << idx << ", name: " << model_outputs[idx].get_any_name() << "\n";
    }*/


    auto inputs = compilemodel.inputs();
    TextStreamer text_streamer{ std::move(detokenizer) };
	
    // input length, output length, first time, other time
    std::vector<std::tuple<size_t, size_t, double, double>> perf_records;

    for (std::string input_text : sentences) {
        total_time = 0;
        count = 0;
	std::string system_message = "<|system|>\nYou are a helpful assistant.<|end|>";
        std::ostringstream oss_prompt;
        oss_prompt << system_message << "\n<|user|>\n"
                   << input_text << "<|end|>\n<|assistant|>\n";
        std::string prompt_text = oss_prompt.str();
        std::cout << " #### sentence: index " << prompt_text << std::endl;
        tokenize(tokenizer, prompt_text.c_str());
        input_ids = tokenizer.get_tensor("input_ids");
        attention_mask = tokenizer.get_tensor("attention_mask");
	auto input_len = input_ids.get_size();
        std::cout << "input lenghth " << input_ids.get_size() << std::endl;
	    
        std::vector<int> output_ids;
        output_ids.reserve(input_ids.get_size());
        for (size_t idx = 0; idx < input_ids.get_size(); ++idx) {
            output_ids.emplace_back(((int)input_ids.data<const int64_t>()[idx]));
        }

        if (args.use_stateless)
        {
            for (size_t idx = 3; idx < model_inputs.size(); ++idx)
            {
              ireq.get_input_tensor(idx).set_shape({BATCH_SIZE, 32, 0, 96});
            }
        }	
        ireq.set_tensor("input_ids", input_ids);
        ireq.set_tensor("attention_mask", attention_mask);
	ireq.get_tensor("position_ids").set_shape(input_ids.get_shape());
        std::iota(ireq.get_tensor("position_ids").data<int64_t>(), ireq.get_tensor("position_ids").data<int64_t>() + ireq.get_tensor("position_ids").get_size(), 0);

	if (!args.use_stateless)
        {
            ireq.get_tensor("beam_idx").set_shape({ BATCH_SIZE });
            ireq.get_tensor("beam_idx").data<int32_t>()[0] = 0;

	    for (auto &&state : ireq.query_state()){
                state.reset();
            }
	}

        startTime = Time::now();
        ireq.infer();
        duration_ms = get_duration_ms_until_now(startTime);
        std::cout << "First token took " << duration_ms << " ms" << std::endl;
        first_time = duration_ms;

        int64_t sequence_len = ireq.get_tensor("logits").get_shape().at(1) - 1;
        size_t vocab_size = ireq.get_tensor("logits").get_shape().back();
        float* logits = ireq.get_tensor("logits").data<float>() + sequence_len * vocab_size;

        int64_t out_token = get_out_token_id(output_ids, logits, vocab_size, args);
        output_ids.emplace_back(((int)out_token));

        ireq.get_tensor("input_ids").set_shape({ BATCH_SIZE, 1 });
        ireq.get_tensor("position_ids").set_shape({ BATCH_SIZE, 1 });

	// Get the runtime info from the tokenizer model that we read earlier
	auto rt_info = tokenizer_model->get_rt_info(); //Get the runtime info for the model
	int64_t SPECIAL_EOS_TOKEN;
	
	if (rt_info.count("eos_token_id") > 0) { //check if the runtime information has a valid EOS token ID
		SPECIAL_EOS_TOKEN = rt_info["eos_token_id"].as<int64_t>();
	} else {
		throw std::runtime_error("EOS token ID not found in model's runtime information.");
	}
	while (out_token != SPECIAL_EOS_TOKEN) {
            startTime = Time::now();
            ireq.get_tensor("input_ids").data<int64_t>()[0] = out_token;
            ireq.get_tensor("attention_mask").set_shape({ BATCH_SIZE, ireq.get_tensor("attention_mask").get_shape()[1] + 1 });
            std::fill_n(ireq.get_tensor("attention_mask").data<int64_t>(), ireq.get_tensor("attention_mask").get_size(), 1);
            ireq.get_tensor("position_ids").data<int64_t>()[0] = ireq.get_tensor("attention_mask").get_size() - 2;
	    if (args.use_stateless)
            {
              for (size_t idx = 3; idx < model_inputs.size(); ++idx)
              {
		//std::cout << "idx: " << idx << ", name: " << model_inputs[idx].get_any_name() << "\n";

                ireq.set_input_tensor(idx, ireq.get_output_tensor(idx - 2));
              }
            }
            ireq.start_async();
            ireq.wait();
            duration_ms = get_duration_ms_until_now(startTime);
            count += 1;
            total_time += duration_ms;

            text_streamer.put(out_token);
            logits = ireq.get_tensor("logits").data<float>();

            out_token = get_out_token_id(output_ids, logits, vocab_size, args);
            output_ids.emplace_back(((int)out_token));

            if (args.output_fixed_len > 0) {
                if(count >= (args.output_fixed_len - 1))
                    break;
            } 
            else {
                if (out_token == SPECIAL_EOS_TOKEN) {
                    break;
                }
            }
        }
        
        text_streamer.end();

        if (count > 0) {
	    double avg_time = total_time / count;
            std::cout << "Other Avg inference took total " << total_time << " ms token num " << count << " first " << first_time << " ms " << " avg " << total_time / (count) << " ms" << std::endl;
	    perf_records.push_back({input_len, count, first_time, avg_time});
        }
    }
    std::cout << "input id, input token len, out token len, first token time, average time" << std::endl;
    size_t index = 0;
    for (auto i : perf_records) {
        std::cout << index << ", "<< std::get<0>(i) << ", "<< std::get<1>(i) << ", "<< std::get<2>(i) << ", "<< std::get<3>(i) << std::endl;
	index++;
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
