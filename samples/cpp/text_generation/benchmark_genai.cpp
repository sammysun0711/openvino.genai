// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cxxopts.hpp>

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    cxxopts::Options options("benchmark_vanilla_genai", "Help command");

    options.add_options()("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>())(
        "p,prompt",
        "Prompt",
        cxxopts::value<std::string>()->default_value("The Sky is blue because"))(
        "nw,num_warmup",
        "Number of warmup iterations",
        cxxopts::value<size_t>()->default_value(std::to_string(
            1)))("n,num_iter", "Number of iterations", cxxopts::value<size_t>()->default_value(std::to_string(3)))(
        "mt,max_new_tokens",
        "Maximal number of new tokens",
        cxxopts::value<size_t>()->default_value(std::to_string(
            20)))("d,device", "device", cxxopts::value<std::string>()->default_value("CPU"))("h,help", "Print usage");

    cxxopts::ParseResult result;
    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cout << e.what() << "\n\n";
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    std::string prompt = result["prompt"].as<std::string>();
    const std::string models_path = result["model"].as<std::string>();
    std::string device = result["device"].as<std::string>();
    size_t num_warmup = result["num_warmup"].as<size_t>();
    size_t num_iter = result["num_iter"].as<size_t>();

    /*
    prompt =
        "<s>[INST] <<SYS>> A chat between a curious user and an artificial intelligence assistant. The assistant gives "
        "helpful, detailed, and polite answers to the user's questions. <</SYS>> You will act as a Christian, and "
        "fully summarize following text:\nSometimes it's nice to take a minute in the pew by yourself beforehand. You "
        "have this beautiful church probably almost all to yourself. Can you feel its energy resonating through you? "
        "Can you feel the majesty of the Lord's kingdom and how you're a part of it? Take a moment to kneel and pray "
        "with your head down and hands clasped together. Reflect on your faith and how you feel currently. Think about "
        "how you've been responding to God's call and how you've been living in the light of his love. When the priest "
        "is ready for you, of course. You'll probably see him there by his lonesome or someone else walk out just "
        "before you. Sit down either across from him or behind the screen -- it's totally up to you whether or not you "
        "prefer to remain anonymous. He won't treat you any differently either way. Make the sign of the cross upon "
        "his prompt, saying, \"Bless me, Father, for I have sinned. It has been 10 years since my last confession.\" "
        "This is your standard, traditional phrasing. However, if you just sit down and say hello, that's fine, too. "
        "The priest knows what he's doing. The Byzantine Rite is a bit different. The priest may sit to your side and "
        "put his epitrachelion on your head. He may then also do the Prayer of Absolution. But the idea remains the "
        "exact same -- just go wherever he takes you. Once you sit down and you've made the sign of the cross, just "
        "sit back and follow the priest's lead. He'll ask you how long it's been since your last confession (if you "
        "don't voluntarily offer that information), how you are feeling, maybe how your faith is going, and then ask "
        "you what sins you would like to talk about with him and God. It's just a casual conversation! Do not fret. "
        "There is absolutely zero pressure on your part. Again, as long as you come there with the intention of "
        "leaving with a clean heart, you're more than welcome in the church. There is no wrong way to go about "
        "confession! This part is intimidating, but think about it this way: the priest you're talking to has probably "
        "heard just about everything before. Whatever you have to say will not blow his mind. So when he asks, start "
        "rattling them off, from the most serious to the least. If he asks any questions, answer them, but do not feel "
        "the need to go into detail. A simple, \"I did so and so,\" will suffice. Your priest is going to be very "
        "understanding. If you don't remember the exact timeframe, that's fine. If you don't remember your motivation, "
        "that's fine. All your priest cares about is that you're being as honest as possible and that your heart is in "
        "the right place. He'll talk you through everything, possibly asking about your intentions, but mainly just "
        "letting you know that God loves you, sin and all. If he has any ideas to bring you closer to God, he may "
        "suggest them at this juncture. He's there to help, after all. He will then ask you to make an Act of "
        "Contrition. That goes like this: My God, I am sorry for my sins with all my heart.In choosing to do wrong and "
        "failing to do good,I have sinned against You whom I should loveabove all things. I firmly intend, with your "
        "help,to do penance, to sin no more, andto avoid whatever leads me to sin.Our Savior Jesus Christ suffered and "
        "died for us.In his name, my God, have mercy.If you are a Roman Catholic, your act of contrition will go like "
        "this: Oh my God, I am very sorry for having offended thee. But most of all, because they offend you, my God, "
        "who is all good and deserving of all my love. I firmly resolve with the help of thy grace, to sin no more, "
        "and to avoid the near occasion of sin. Amen. Don't worry! It won't be anything huge. Take the absolution to "
        "heart -- you now have a brand new, clean slate to work with. \"Penance\" is your expression of regret and "
        "repentance, showing God that you're truly sorry and that you wish for nothing more than to be forgiven. "
        "Thanks. [/INST]";
    */
    prompt =
        "The process of Origami seems simple at the first glance, but in fact, it still requires a very complicated "
        "process to do it well. Taking folding a rose as an example, we can divide the entire process into three "
        "stages, including: firstly creating a grid of creases, secondly making a three-dimensional base, and thirdly "
        "finishing petal decoration. The first step is to create a grid of creases: this step is a bit like the first "
        "step of folding a gift of thousand-paper-crane. That is to say, we can fold the paper in half (or namedly "
        "equal-folds) through the symmetrical axis, and repeat such step in the other symmetrical axis. And then apply "
        "multiple equal-folds in sequence relative to each smaller rectangle divided by the two creases; After that, "
        "the creases in each direction will interweave into a complete set of uniform small square splicing patterns; "
        "these small squares form a reference space similar to a two-dimensional coordinate system, allowing us to "
        "combine adjacent creases on the plane from Three-dimensional high platforms or depressions are folded on the "
        "two-dimensional small squares to facilitate the next steps of folding. It should be noted that, in the "
        "process of creating grid creases, there may be rare cases when the folds are not aligned. The consequences of "
        "this error can be very serious. And just like the butterfly effect, it is only a slight difference at the "
        "beginning , and in the end it may generate a disaster world which is completely different from plan. Anyway, "
        "let's continue. The second step is make the three-dimensional base: In this step, we need to fold a set of "
        "symmetrical three-dimensional high platforms or depressions based on the grid creases. From the symmetry "
        "analysis, it is not difficult to find that the rose will have four symmetrical three-dimensional high "
        "platforms and supporting depressions. Therefore, we can firstly fold out a quarter of the depression and "
        "plateau patterns, which would help build a base to compose into a complex 3D structure. And then, we use this "
        "quarter as a template, and fold out the repeating patterns on the remaining three parts of the whole "
        "structure in turn. It is worth noting that the layout of the high platform not only needs to consider the "
        "regular contrast and symmetrical distribution of the length and width, but also needs to ensure the "
        "orderliness of the height dimension. This is very important, since we will never go back to this step after "
        "all parts were made, and you would better start from first step if you make anything wrong in the this step. "
        "Similar to the precautions in the first stage, please handle all the corners in three dimensions to ensure "
        "that they conform to the layout required in the plan, which would help us avoid the butterfly effect and "
        "increase the robustness in the process of three-dimensional folding. Just like building a skyscrapper in the "
        "real world, people usually take a lot of time when building the base but soon get finished when extending the "
        "structure after that. Time is worth to cost in the base, but would be saved in the future after you succeed "
        "in base. Anyway, let's continue. During the first quarter of the pattern, repeated comparisons with the "
        "finished rose were made to eliminate any possible errors in the first place. The final stage is to finish the "
        "petal grooming. At this stage, we often emphasize an important term called folding-by-heart. The intention "
        "here is no longer literally serious, but focus is moved to our understanding of the shape of a rose in "
        "nature, and we usually use natural curves to continuously correct the shape of petals in order to approach "
        "the shape of rose petals in reality. One more comment: this is also the cause of randomness to the art, which "
        "can be generated differently by different people. Recall that rose should be adjusted close to reality, so in "
        "the last step of this stage, we need to open the bloom in the center of the rose, by pulling on the four "
        "petals that have been bent. This process may be accompanied by the collapse of the overall structure of the "
        "rose, so we should be very careful to save strength of adjustment, and it must be well controlled to avoid "
        "irreversible consequences. Ultimately, after three stages of folding, we end up with a crown of rose with a "
        "similar shape close to reality. If condition is permited, we can wrap a green paper strip twisted on a "
        "straightened iron wire, and insert the rose crown we just created onto one side of the iron wire. In this "
        "way, we got a hand-made rose with a green stem. We can also repeat the steps above to increase the number of "
        "rose, so that it can be made into a cluster. Different color of rose is usually more attractive and can be "
        "considered as a better plan of gift to your friend. In summary, by creating a grid of creases, making a "
        "three-dimensional base, and finishing with petals, we created a three-dimensional rose from a two-dimensional "
        "paper. Although this process may seem simple, it is indeed a work of art created by us humans with the help "
        "of imagination and common materials. At last, Please comment to assess the above content.";
    ov::genai::GenerationConfig config;
    config.max_new_tokens = result["max_new_tokens"].as<size_t>();

    config.max_new_tokens = 512;
    num_iter = 3;
    ov::genai::LLMPipeline pipe(models_path, device);
    std::cout << "********************************************************* Warmup start !!! "
                 "**********************************************\n";
    for (size_t i = 0; i < num_warmup; i++) {
        ov::genai::DecodedResults res = pipe.generate(prompt, config);
        ov::genai::PerfMetrics metrics = res.perf_metrics;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Load time: " << metrics.get_load_time() << " ms" << std::endl;
        std::cout << "Generate time: " << metrics.get_generate_duration().mean << " ± "
                  << metrics.get_generate_duration().std << " ms" << std::endl;
        std::cout << "Tokenization time: " << metrics.get_tokenization_duration().mean << " ± "
                  << metrics.get_tokenization_duration().std << " ms" << std::endl;
        std::cout << "Detokenization time: " << metrics.get_detokenization_duration().mean << " ± "
                  << metrics.get_detokenization_duration().std << " ms" << std::endl;
        std::cout << "TTFT: " << metrics.get_ttft().mean << " ± " << metrics.get_ttft().std << " ms" << std::endl;
        std::cout << "TPOT: " << metrics.get_tpot().mean << " ± " << metrics.get_tpot().std << " ms/token "
                  << std::endl;
        std::cout << "Throughput: " << metrics.get_throughput().mean << " ± " << metrics.get_throughput().std
                  << " tokens/s" << std::endl;
        std::cout << "Average Output tokens counter: " << metrics.get_num_generated_tokens() << std::endl;
        std::cout << "Average Input tokens counter: " << metrics.get_num_input_tokens() << std::endl;
    }
    std::cout << "********************************************************* Warmup end !!! "
                 "**********************************************\n";

    ov::genai::DecodedResults res = pipe.generate(prompt, config);
    ov::genai::PerfMetrics metrics = res.perf_metrics;

    std::cout << "Run num_iter: " << num_iter << "\n";

    for (size_t i = 0; i < num_iter - 1; i++) {
        std::cout << "============================================================ Generation iteration " << i
                  << " start !!! ============================================================\n";
        res = pipe.generate(prompt, config);
        metrics = metrics + res.perf_metrics;
        std::cout << "============================================================ Generation iteration " << i
                  << " end !!! ============================================================\n";
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Load time: " << metrics.get_load_time() << " ms" << std::endl;
    std::cout << "Generate time: " << metrics.get_generate_duration().mean << " ± "
              << metrics.get_generate_duration().std << " ms" << std::endl;
    std::cout << "Tokenization time: " << metrics.get_tokenization_duration().mean << " ± "
              << metrics.get_tokenization_duration().std << " ms" << std::endl;
    std::cout << "Detokenization time: " << metrics.get_detokenization_duration().mean << " ± "
              << metrics.get_detokenization_duration().std << " ms" << std::endl;
    std::cout << "TTFT: " << metrics.get_ttft().mean << " ± " << metrics.get_ttft().std << " ms" << std::endl;
    std::cout << "TPOT: " << metrics.get_tpot().mean << " ± " << metrics.get_tpot().std << " ms/token " << std::endl;
    std::cout << "Throughput: " << metrics.get_throughput().mean << " ± " << metrics.get_throughput().std << " tokens/s"
              << std::endl;
    std::cout << "Average Output tokens counter: " << metrics.get_num_generated_tokens() / num_iter << std::endl;
    std::cout << "Average Input tokens counter: " << metrics.get_num_input_tokens() / num_iter << std::endl;

    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}
