// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (4 != argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <DRAFT_MODEL_DIR> '<PROMPT>'");
    }

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 200;
    // Speculative decoding generation parameters like `num_assistant_tokens` and `assistant_confidence_threshold` are
    // mutually excluded add parameter to enable speculative decoding to generate `num_assistant_tokens` candidates by
    // draft_model per iteration
    config.num_assistant_tokens = 5;
    // add parameter to enable speculative decoding to generate candidates by draft_model while candidate probability is
    // higher than `assistant_confidence_threshold` config.assistant_confidence_threshold = 0.4;

    std::string main_model_path = argv[1];
    std::string draft_model_path = argv[2];
    std::string prompt = argv[3];

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

    // User can run main and draft model on different devices.
    // Please, set device for main model in `LLMPipeline` constructor and in in `ov::genai::draft_model` for draft.
    std::string main_device = "CPU", draft_device = "CPU";

    ov::genai::LLMPipeline pipe(main_model_path, main_device, ov::genai::draft_model(draft_model_path, draft_device));

    auto streamer = [](std::string subword) {
        std::cout << subword << std::flush;
        return ov::genai::StreamingStatus::RUNNING;
    };

    // Since the streamer is set, the results will
    // be printed each time a new token is generated.
    // pipe.generate(prompt, config, streamer);
    int num_warmup = 1;
    int num_iter = 3;
    std::cout << "********************************************************* Warmup start !!! "
                 "**********************************************\n";
    for (size_t i = 0; i < num_warmup; i++)
        pipe.generate(prompt, config);

    std::cout << "********************************************************* Warmup end !!! "
                 "**********************************************\n";
    ov::genai::DecodedResults res = pipe.generate(prompt, config);
    ov::genai::PerfMetrics metrics = res.perf_metrics;

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
