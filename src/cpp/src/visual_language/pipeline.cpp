// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <optional>
#include <random>

#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/text_streamer.hpp"

#include "visual_language/vlm_config.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "visual_language/embedding_model.hpp"

#include "sampler.hpp"
#include "utils.hpp"
#include "lm_encoding.hpp"

using namespace ov::genai;

namespace {

} // namespace

namespace ov::genai {

const ModelsMap::mapped_type& get_model_weights_pair(const ModelsMap& models_map, const std::string& key) {
    auto it = models_map.find(key);
    if (it != models_map.end()) {
        return it->second;
    }
    OPENVINO_THROW("Model with key '", key, "' not found in models map.");
}

}

class ov::genai::VLMPipeline::VLMPipelineImpl {
public:
    // A config to follow for LLM input construction.
    VLMConfig m_vlm_config;
    // A config to follow for text generation.
    GenerationConfig m_generation_config;
    // A tokenizer encoding a prompt.
    Tokenizer m_tokenizer;
    // A model to compute token embeddings.
    // Input shape: [N, conversation length].
    // Output shape: [1, conversation length, hidden_size].
    EmbeddingsModel m_embedding;
    // A language model used to generate a response.
    // Input shapes: inputs_embeds[N, conversation length, hidden_size],
    // position_ids[N, conversation length], beam_idx[N].
    // Output shape: logits[N, conversation length, vocab_size].
    ov::InferRequest m_language;
    // True if chat mode is activated to save conversation
    // history between generate() calls.
    bool m_is_chat_conversation;
    // InputsEmbedder
    std::shared_ptr<InputsEmbedder> m_inputs_embedder;
    // Load pipeline time
    float m_load_time_ms = 0;
    // Axis num in kv cache from m_language model, which contains information about history len
    size_t m_kv_cache_seq_length_axis = 2;
    // Component for applying sampling to lm outputs
    Sampler m_sampler;

    VLMPipelineImpl(
        const std::filesystem::path& models_dir,
        const std::string& device,
        const ov::AnyMap& properties
    ) :
        m_vlm_config{
            utils::from_config_json_if_exists<VLMConfig>(
                models_dir, "config.json"
            )
        },
        m_generation_config{
            utils::from_config_json_if_exists<GenerationConfig>(
                models_dir, "generation_config.json"
            )
        },
        m_is_chat_conversation{false} {

        std::cout << "*************************** VLMPipelineImpl initialization called *************************** \n "  << "\n";
        m_inputs_embedder = std::make_shared<InputsEmbedder>(
            m_vlm_config, models_dir, device, properties);

        m_tokenizer = m_inputs_embedder->get_tokenizer();
        m_embedding = m_inputs_embedder->get_embedding_model();

        auto compiled_language_model = utils::singleton_core().compile_model(
            models_dir / "openvino_language_model.xml", device, properties
        );
        ov::genai::utils::print_compiled_model_properties(compiled_language_model, "VLM language model");
        auto language_model = compiled_language_model.get_runtime_model();
        m_kv_cache_seq_length_axis = ov::genai::utils::get_kv_axes_pos(language_model).seq_len;

        m_language = compiled_language_model.create_infer_request();

        m_language.get_tensor("attention_mask").set_shape({1, 0});

        // If eos_token_id was not provided, take value
        if (m_generation_config.eos_token_id == -1) {
            m_generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());
        }

        m_sampler.set_tokenizer(m_tokenizer);
        m_sampler.set_seed(m_generation_config.rng_seed);
    }

    VLMPipelineImpl(
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap& properties,
        const ov::genai::GenerationConfig& generation_config
    ) :
        m_vlm_config{
            utils::from_config_json_if_exists<VLMConfig>(
                config_dir_path, "config.json"
            )
        },
        m_generation_config{generation_config},
        m_is_chat_conversation{false} {
        
        m_inputs_embedder = std::make_shared<InputsEmbedder>(
            m_vlm_config, models_map, tokenizer, config_dir_path, device, properties);

        m_tokenizer = m_inputs_embedder->get_tokenizer();
        m_embedding = m_inputs_embedder->get_embedding_model();

        auto m_language_pair = get_model_weights_pair(models_map, "language");
        m_language = utils::singleton_core().compile_model(
            m_language_pair.first, m_language_pair.second, device, properties
        ).create_infer_request();

        m_language.get_tensor("attention_mask").set_shape({1, 0});

        // If eos_token_id was not provided, take value
        if (m_generation_config.eos_token_id == -1) {
            m_generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());
        }

        m_sampler.set_tokenizer(m_tokenizer);
        m_sampler.set_seed(m_generation_config.rng_seed);
    }

    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& rgbs,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) {
        std::cout << "------------------------- VLMDecodedResults generate with stream generation called ------------------------- \n";
        auto generate_start_time = std::chrono::steady_clock::now();
        VLMPerfMetrics perf_metrics;
        auto& raw_counters = perf_metrics.raw_metrics;
        auto& raw_vlm_counters = perf_metrics.vlm_raw_metrics;

        if (!m_is_chat_conversation) {
            m_language.reset_state();
            m_language.get_tensor("attention_mask").set_shape({1, 0});
        }

        // If stop_token_ids were not provided, take value from default m_generation_config
        if (generation_config.stop_token_ids.empty())
            generation_config.stop_token_ids = m_generation_config.stop_token_ids;

        // If eos_token_id was not provided, take value from default m_generation_config
        if (generation_config.eos_token_id == -1)
            generation_config.set_eos_token_id(m_generation_config.eos_token_id);
        generation_config.validate();
        
        m_inputs_embedder->set_stop_token_ids(generation_config.stop_token_ids);

        m_inputs_embedder->set_apply_chat_template_status(generation_config.apply_chat_template);

        auto start_get_inputs_embeds = std::chrono::steady_clock::now();
        std::cout << "[m_inputs_embedder infer]\n";
        ov::Tensor inputs_embeds = m_inputs_embedder->get_inputs_embeds(prompt, rgbs, perf_metrics);
        auto end_get_inputs_embeds = std::chrono::steady_clock::now();

        auto to_remove_from_hist = m_inputs_embedder->get_num_tokens_to_remove_from_hist();
        ov::genai::utils::trim_kv_cache(m_language, to_remove_from_hist, m_kv_cache_seq_length_axis, std::nullopt);

        std::vector<SequenceGroup::Ptr> requests;
        size_t request_id = 0;
        size_t block_size = 1; // not used

        size_t history_size = m_language.get_tensor("attention_mask").get_shape().at(1) - to_remove_from_hist;
        size_t inputs_embeds_size = inputs_embeds.get_shape().at(1);

        KVCacheState& kv_cache_state = m_inputs_embedder->get_kv_cache_state();
        std::vector<int64_t> tokenized_history = kv_cache_state.get_state();
        ov::Tensor prompt_ids(ov::element::i64, { history_size + inputs_embeds_size });
        OPENVINO_ASSERT(prompt_ids.get_size() >= tokenized_history.size(), "Prompt ids size is less than tokenized history size");
        std::fill_n(prompt_ids.data<int64_t>(), prompt_ids.get_size(), m_tokenizer.get_pad_token_id());
        std::copy(tokenized_history.begin(), tokenized_history.end(), prompt_ids.data<int64_t>());

        SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(request_id, prompt_ids, generation_config, block_size);
        requests.push_back(sequence_group);

        std::shared_ptr<StreamerBase> streamer_ptr = ov::genai::utils::create_streamer(streamer, m_tokenizer);

        OPENVINO_ASSERT(streamer_ptr == nullptr || generation_config.num_return_sequences == 1 &&
            (generation_config.is_greedy_decoding() || generation_config.is_multinomial()),
            "Currently streaming is possible only with batch size=1 and only for greedy or multinomial decoding");

        ov::Tensor new_atten_mask = ov::Tensor{ov::element::i64, { 1, history_size + inputs_embeds_size }};
        std::fill_n(new_atten_mask.data<int64_t>(), new_atten_mask.get_size(), 1);

        ov::Tensor position_ids;
        std::optional<int64_t> rope_delta;
        std::tie(position_ids, rope_delta) = m_inputs_embedder->get_position_ids(inputs_embeds_size, history_size);

        if (m_sampler.get_seed() != generation_config.rng_seed) {
            m_sampler.set_seed(generation_config.rng_seed);
        }
        std::cout << "=========================================\n";
        std::cout << "get_lm_encoded_results called\n";
        std::cout << "inputs_embeds size: " << inputs_embeds_size << "\n";
        std::cout << "inputs_embeds.get_shape(): [" << inputs_embeds.get_shape()[0] << ", " << inputs_embeds.get_shape()[1] << "]\n";
        std::cout << "new_atten_mask.get_shape(): [" << new_atten_mask.get_shape()[0] << ", " << new_atten_mask.get_shape()[1] << "]\n";
        std::cout << "position_ids.get_shape(): [" << position_ids.get_shape()[0] << ", " << position_ids.get_shape()[1] << "]\n";
        ov::genai::utils::GenerationFinishInfo finish_info = ov::genai::get_lm_encoded_results(m_language, inputs_embeds, new_atten_mask, streamer_ptr, m_sampler, requests,
                                                                                               position_ids, kv_cache_state, m_embedding, rope_delta);

        ov::genai::EncodedResults& encoded_result = finish_info.results;

        auto decode_start_time = std::chrono::steady_clock::now();
        VLMDecodedResults decoded;
        for (size_t idx = 0; idx < encoded_result.tokens.size(); ++idx) {
            decoded.texts.push_back(m_tokenizer.decode(encoded_result.tokens.at(idx)));
            decoded.scores.push_back(encoded_result.scores.at(idx));
        }
        auto decode_end_time = std::chrono::steady_clock::now();

        std::string decoded_results = decoded.texts.at(0);
        if (m_is_chat_conversation)
            m_inputs_embedder->update_chat_history(decoded_results);
        else
            kv_cache_state.reset_state();

        auto generate_end_time = std::chrono::steady_clock::now();
        decoded.perf_metrics = encoded_result.perf_metrics;

        // Common perf metrics
        auto& res_raw_counters = decoded.perf_metrics.raw_metrics;
        decoded.perf_metrics.num_input_tokens = prompt_ids.get_size();
        decoded.perf_metrics.load_time = m_load_time_ms;
        res_raw_counters.generate_durations.emplace_back(PerfMetrics::get_microsec(generate_end_time - generate_start_time));
        res_raw_counters.detokenization_durations.emplace_back(PerfMetrics::get_microsec(decode_end_time - decode_start_time));
        res_raw_counters.tokenization_durations.insert(res_raw_counters.tokenization_durations.end(), raw_counters.tokenization_durations.begin(), raw_counters.tokenization_durations.end());
        
        // VLM specific perf metrics
        decoded.perf_metrics.vlm_raw_metrics.prepare_embeddings_durations.emplace_back(PerfMetrics::get_microsec(end_get_inputs_embeds - start_get_inputs_embeds));

        // Evaluate statistics
        decoded.perf_metrics.m_evaluated = false;
        decoded.perf_metrics.evaluate_statistics(generate_start_time);

        return decoded;
    }

    VLMDecodedResults generate(
        const std::string& prompt,
        const ov::AnyMap& config_map
    ) {
        std::cout << "=========================  VLMDecodedResults generate called ========================= \n";
        auto image = config_map.find(ov::genai::image.name());
        auto images = config_map.find(ov::genai::images.name());
        OPENVINO_ASSERT(
            config_map.end() == image || config_map.end() == images,
            "Only one property can be set: image of images."
        );
        std::vector<ov::Tensor> rgbs;
        if (config_map.end() != image) {
            rgbs = {image->second.as<ov::Tensor>()};
        } if (config_map.end() != images) {
            if (images->second.is<std::vector<ov::Tensor>>()) {
                rgbs = images->second.as<std::vector<ov::Tensor>>();
            }
            else if (images->second.is<ov::Tensor>()){
                rgbs = {images->second.as<ov::Tensor>()};
            }
            else {
                OPENVINO_THROW("Unknown images type.");
            }
        }
        ov::genai::OptionalGenerationConfig config_arg = utils::get_config_from_map(config_map);
        GenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
        config.update_generation_config(config_map);

        return generate(
            prompt,
            rgbs,
            config,
            utils::get_streamer_from_map(config_map)
        );
    }

    void start_chat(const std::string& system_message) {
        m_is_chat_conversation = true;
        bool have_state = 0 != m_language.get_tensor("attention_mask").get_size();
        if (have_state) {
            // Resetting state may be slow.
            m_language.reset_state();
            // Since if is already introduced, move all resetting here.
            m_language.get_tensor("attention_mask").set_shape({0, 0});
        }
        m_inputs_embedder->start_chat(system_message);
    }

    void finish_chat() {
        m_is_chat_conversation = false;
        // Resetting state may be slow.
        m_language.reset_state();
        m_language.get_tensor("attention_mask").set_shape({0, 0});
        // clear all chat history
        m_inputs_embedder->finish_chat();
    }

    Tokenizer get_tokenizer() const {
        return m_tokenizer;
    }

    void set_chat_template(const std::string& new_template) {
        OPENVINO_ASSERT(!m_is_chat_conversation, "Chat template cannot be changed once start_chat() is called. Please, finish current chat via finish_chat()");
        m_tokenizer.set_chat_template(new_template);
    }

    GenerationConfig get_generation_config() const {
        return m_generation_config;
    }

    void set_generation_config(const GenerationConfig& new_config) {
        int64_t default_eos_token_id = m_generation_config.eos_token_id;
        auto default_stop_token_ids = m_generation_config.stop_token_ids;
        m_generation_config = new_config;

        // If stop_token_ids were not provided, take value from default config
        if (m_generation_config.stop_token_ids.empty())
            m_generation_config.stop_token_ids = default_stop_token_ids;
        // if eos_token_id was not provided in config forward from default config
        if (m_generation_config.eos_token_id == -1)
            m_generation_config.set_eos_token_id(default_eos_token_id);

        m_generation_config.validate();
    }
};

VLMPipeline::VLMPipeline(
    const std::filesystem::path& models_dir,
    const std::string& device,
    const ov::AnyMap& properties
) {
    std::cout << "############################ VLMPipeline::VLMPipeline called ############################\n";
    auto start_time = std::chrono::steady_clock::now();
    m_pimpl = std::make_unique<VLMPipelineImpl>(models_dir, device, properties);
    auto stop_time = std::chrono::steady_clock::now();
    m_pimpl->m_load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
}

VLMPipeline::VLMPipeline(
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap& properties,
    const ov::genai::GenerationConfig& generation_config
) {
    auto start_time = std::chrono::steady_clock::now();
    m_pimpl = std::make_unique<VLMPipelineImpl>(models_map, tokenizer, config_dir_path, device, properties, generation_config);
    auto stop_time = std::chrono::steady_clock::now();
    m_pimpl->m_load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
}

ov::genai::VLMPipeline::~VLMPipeline() = default;

VLMDecodedResults VLMPipeline::generate(
    const std::string& prompt,
    const std::vector<ov::Tensor>& rgbs,
    const GenerationConfig& generation_config,
    const StreamerVariant& streamer
) {
    std::cout << "=== VLMPipeline::generate Multi image path\n";
    return m_pimpl->generate(prompt, rgbs, generation_config, streamer);
}

VLMDecodedResults VLMPipeline::generate(
    const std::string& prompt,
    const ov::Tensor& rgb,
    const GenerationConfig& generation_config,
    const StreamerVariant& streamer
) {
    std::cout << "=== VLMPipeline::generate Single image path\n";
    return m_pimpl->generate(prompt, {rgb}, generation_config, streamer);
}

VLMDecodedResults VLMPipeline::generate(
    const std::string& prompt,
    const ov::AnyMap& config_map
) {
    return m_pimpl->generate(prompt, config_map);
}

void VLMPipeline::start_chat(const std::string& system_message) {
    m_pimpl->start_chat(system_message);
}

void VLMPipeline::finish_chat() {
    m_pimpl->finish_chat();
}

void VLMPipeline::set_chat_template(const std::string& new_template) {
    m_pimpl->set_chat_template(new_template);
}

Tokenizer VLMPipeline::get_tokenizer() const {
    return m_pimpl->get_tokenizer();
}

GenerationConfig VLMPipeline::get_generation_config() const {
    return m_pimpl->get_generation_config();
}

void VLMPipeline::set_generation_config(const GenerationConfig& new_config) {
    m_pimpl->set_generation_config(new_config);
}
