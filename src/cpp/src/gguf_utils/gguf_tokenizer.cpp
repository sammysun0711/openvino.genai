// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <cstdint>

#include "gguf_tokenizer.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"

#ifdef _WIN32
#    define NOMINMAX
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

using namespace ov::op;

using FactoryCreateType = ov::OutputVector (*)(const std::string& op_type,
                                               const ov::OutputVector& inputs,
                                               const ov::AnyMap& attributes);

constexpr int32_t MAX_LENGTH = 8192;
constexpr float VOCAB_SIZE_CACHE_PROPORTION = 0.2f;
constexpr int32_t MIN_CACHE_CAPACITY = 20'000;

namespace ov {
namespace genai {
bool is_gguf_model(const std::filesystem::path& file_path) {
    return file_path.extension() == ".gguf";
}

std::map<std::string, GGUFMetaData> tokenizer_config_from_meta(
    const std::unordered_map<std::string, GGUFMetaData>& metadata) {
    std::map<std::string, GGUFMetaData> tokenizer_config;

    const std::string prefix = "tokenizer.";
    for (const auto& [key, value] : metadata) {
        std::cout << "key: " << key << "\n";
        if (key.compare(0, prefix.size(), prefix) == 0) {
            size_t last_dot = key.find_last_of('.');
            // Extract the last part after "."
            std::string sub_key = (last_dot != std::string_view::npos) ? std::string(key.substr(last_dot + 1)) : key;
            tokenizer_config[sub_key] = value;
        }
    }

    return tokenizer_config;
}

std::shared_ptr<void> load_shared_object(const std::filesystem::path& path) {
#ifdef _WIN32
    HMODULE handle = LoadLibraryW(path.wstring().c_str());
    if (!handle) {
        throw std::runtime_error("Failed to load shared object: " + path.string());
    }

    return std::shared_ptr<void>(handle, [](void* h) {
        if (h)
            FreeLibrary(static_cast<HMODULE>(h));
    });
#else
    void* handle = dlopen(path.c_str(), RTLD_LAZY);
    if (!handle) {
        throw std::runtime_error("Failed to load shared object: " + path.string() + "\n" + dlerror());
    }

    return std::shared_ptr<void>(handle, [](void* h) {
        if (h)
            dlclose(h);
    });
#endif
}

void* get_symbol(const std::shared_ptr<void>& shared_object, const char* symbolName) {
    if (!shared_object || !symbolName) {
        throw std::invalid_argument("Null shared object or symbol name.");
    }

#ifdef _WIN32
    HMODULE handle = static_cast<HMODULE>(shared_object.get());
    void* symbol = reinterpret_cast<void*>(GetProcAddress(handle, symbolName));
    if (!symbol) {
        throw std::runtime_error("Failed to find symbol: " + std::string(symbolName));
    }
    return symbol;
#else
    void* handle = shared_object.get();
    // Clear existing errors
    dlerror();
    void* symbol = dlsym(handle, symbolName);
    const char* error = dlerror();
    if (error) {
        throw std::runtime_error("Failed to find symbol: " + std::string(symbolName) + "\n" + error);
    }
    return symbol;
#endif
}

ov::OutputVector add_ragged_dimension(const ov::OutputVector& inputs) {
    auto input_shape = std::make_shared<v3::ShapeOf>(inputs[0], element::i32);
    auto const_zero = std::make_shared<v0::Constant>(element::i32, Shape{}, 0);
    auto const_one = std::make_shared<v0::Constant>(element::i32, Shape{}, 1);
    auto batch_size = std::make_shared<v8::Gather>(input_shape, const_zero, const_zero);
    auto batch_size_plus_one = std::make_shared<v1::Add>(batch_size, const_one);
    auto ragged_begins = std::make_shared<v4::Range>(const_zero, batch_size, const_one, element::i32)->output(0);
    auto ragged_ends = std::make_shared<v4::Range>(const_one, batch_size_plus_one, const_one, element::i32)->output(0);

    ov::OutputVector res = ov::OutputVector{ragged_begins, ragged_ends};
    res.insert(res.end(), inputs.begin(), inputs.end());
    return res;
}

bool is_special_token(int32_t token_type) {
    return token_type == 3 || token_type == 4;
}

std::string join_special_tokens(const std::vector<std::string>& special_tokens) {
    std::ostringstream oss;
    for (size_t i = 0; i < special_tokens.size(); ++i) {
        if (i > 0)
            oss << "|";
        oss << special_tokens[i];
    }
    return oss.str();
}

std::vector<std::string> get_split_regex(const std::string& pre) {
    // taken from
    // https://github.com/ggml-org/llama.cpp/blob/8551c44d840a7db50adb958ccaf464dc3ded82e7/src/llama-vocab.cpp#L279
    // TODO: complete for other archs
    static std::unordered_map<std::string, std::vector<std::string>> regex_map = {
        {"qwen2",
         {
             // original regex from tokenizer.json
             // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|
             // ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
             "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
             "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
         }},
        {"smollm",
         {
             "\\p{N}",
             "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
         }},
    };

    if (regex_map.count(pre)) {
        return regex_map.at(pre);
    }

    std::vector<std::string> default_regex_exprs = {
        "[\\p{P}\\$\\+<=>\\^~\\|]+",
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        "\\p{N}+",
        "[0-9][0-9][0-9]",
    };

    return default_regex_exprs;
}

ov::OutputVector create_string_constant(const std::vector<std::string>& input_strings) {
    std::vector<int32_t> begins{};
    std::vector<int32_t> ends{};
    std::vector<uint8_t> chars{};

    int32_t offset = 0;

    for (const auto& input_string : input_strings) {
        auto len = static_cast<int32_t>(input_string.size());
        begins.push_back(offset);
        offset += len;
        ends.push_back(offset);
        chars.insert(chars.end(), input_string.begin(), input_string.end());
    }

    auto const_begins = std::make_shared<v0::Constant>(element::i32, ov::Shape{begins.size()}, begins)->output(0);
    auto const_ends = std::make_shared<v0::Constant>(element::i32, ov::Shape{ends.size()}, ends)->output(0);
    auto const_chars = std::make_shared<v0::Constant>(element::u8, ov::Shape{chars.size()}, chars)->output(0);

    return ov::OutputVector{const_begins, const_ends, const_chars};
}

ov::OutputVector create_string_constant(const std::vector<std::vector<uint8_t>>& input_strings) {
    std::vector<int32_t> begins{};
    std::vector<int32_t> ends{};
    std::vector<uint8_t> chars{};

    int32_t offset = 0;

    for (const auto& input_string : input_strings) {
        auto len = static_cast<int32_t>(input_string.size());
        begins.push_back(offset);
        offset += len;
        ends.push_back(offset);
        chars.insert(chars.end(), input_string.begin(), input_string.end());
    }

    auto const_begins = std::make_shared<v0::Constant>(element::i32, ov::Shape{begins.size()}, begins)->output(0);
    auto const_ends = std::make_shared<v0::Constant>(element::i32, ov::Shape{ends.size()}, ends)->output(0);
    auto const_chars = std::make_shared<v0::Constant>(element::u8, ov::Shape{chars.size()}, chars)->output(0);

    return ov::OutputVector{const_begins, const_ends, const_chars};
}

const std::unordered_map<std::string, uint8_t>& unicode_to_bytes() {
    static const std::unordered_map<std::string, uint8_t> map = []() {
        std::vector<uint8_t> bs;

        // Range: '!' (33) to '~' (126)
        for (uint8_t i = static_cast<uint8_t>('!'); i <= static_cast<uint8_t>('~'); ++i) {
            bs.push_back(i);
        }

        // Range: '¡' (161) to '¬' (172)
        for (uint8_t i = 0xA1; i <= 0xAC; ++i) {
            bs.push_back(i);
        }

        // Range: '®' (174) to 'ÿ' (255)
        for (int32_t i = 0xAE; i <= 0xFF; ++i) {
            bs.push_back(static_cast<uint8_t>(i));
        }

        std::vector<int32_t> cs;
        cs.reserve(bs.size());
        for (uint8_t byte : bs) {
            cs.push_back(static_cast<int32_t>(byte));
        }

        int32_t n = 0;
        for (int32_t b = 0; b < 256; ++b) {
            uint8_t byte = static_cast<uint8_t>(b);
            if (std::find(bs.begin(), bs.end(), byte) == bs.end()) {
                bs.push_back(byte);
                cs.push_back(256 + n);
                ++n;
            }
        }

        std::unordered_map<std::string, uint8_t> result;
        for (size_t i = 0; i < cs.size(); ++i) {
            int32_t cp = cs[i];
            std::string utf8_char;

            if (cp <= 0x7F) {
                utf8_char += static_cast<char>(cp);
            } else if (cp <= 0x7FF) {
                utf8_char += static_cast<char>(0xC0 | ((cp >> 6) & 0x1F));
                utf8_char += static_cast<char>(0x80 | (cp & 0x3F));
            } else if (cp <= 0xFFFF) {
                utf8_char += static_cast<char>(0xE0 | ((cp >> 12) & 0x0F));
                utf8_char += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                utf8_char += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
                utf8_char += static_cast<char>(0xF0 | ((cp >> 18) & 0x07));
                utf8_char += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                utf8_char += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                utf8_char += static_cast<char>(0x80 | (cp & 0x3F));
            }

            result[utf8_char] = bs[i];
        }

        return result;
    }();

    return map;
}

int utf8_char_length(unsigned char lead_byte) {
    if ((lead_byte & 0b10000000) == 0)
        return 1;  // 0xxxxxxx (ASCII)
    else if ((lead_byte & 0b11100000) == 0b11000000)
        return 2;  // 110xxxxx
    else if ((lead_byte & 0b11110000) == 0b11100000)
        return 3;  // 1110xxxx
    else if ((lead_byte & 0b11111000) == 0b11110000)
        return 4;  // 11110xxx
    else
        return -1;  // Invalid
}

std::vector<std::string> split_utf8_chars(const std::string& input) {
    std::vector<std::string> result;
    size_t i = 0;

    while (i < input.size()) {
        unsigned char lead = static_cast<unsigned char>(input[i]);
        int len = utf8_char_length(lead);
        if (len <= 0 || i + len > input.size()) {
            std::cerr << "Invalid UTF-8 sequence at byte index " << i << std::endl;
            break;  // Stop on error
        }
        result.emplace_back(input.substr(i, len));
        i += len;
    }

    return result;
}

std::vector<uint8_t> apply_unicode_to_bytes(const std::string& token) {
    auto bytes_encoder = unicode_to_bytes();

    std::vector<uint8_t> res{};
    bool return_original = false;
    auto unicode_chars = split_utf8_chars(token);

    for (const auto& uni_char : unicode_chars) {
        if (bytes_encoder.count(uni_char)) {
            res.push_back(bytes_encoder.at(uni_char));
        } else {
            return_original = true;
            break;
        }
    }

    if (return_original) {
        std::vector<uint8_t> bytes(token.begin(), token.end());
        return bytes;
    }
    return res;
}

std::vector<std::vector<uint8_t>> parse_bbpe_vocab(const std::vector<std::string>& vocab) {
    std::vector<std::vector<uint8_t>> res;
    int32_t iter = 0;
    for (const auto& token : vocab) {
        res.push_back(apply_unicode_to_bytes(token));
    }
    return res;
}

ov::OutputVector parse_bbpe_config(const std::map<std::string, GGUFMetaData>& tokenizer_config,
                                   ov::OutputVector inputs,
                                   const FactoryCreateType& create_func) {
    // 1. Parse vocab and add as input
    std::vector<std::string> vocab_from_config{};
    if (auto val = std::get_if<std::vector<std::string>>(&tokenizer_config.at("tokens"))) {
        vocab_from_config = *val;
    }
    auto vocab = parse_bbpe_vocab(vocab_from_config);
    auto vocab_const = create_string_constant(vocab);

    inputs.insert(inputs.end(), vocab_const.begin(), vocab_const.end());

    // 2. Parse merges
    std::vector<std::string> merges{};
    if (auto val = std::get_if<std::vector<std::string>>(&tokenizer_config.at("merges"))) {
        merges = *val;
    }

    std::vector<std::vector<uint8_t>> left_merges;
    std::vector<std::vector<uint8_t>> right_merges;

    for (const auto& merge : merges) {
        size_t space = merge.find(' ');
        std::string left = merge.substr(0, space);
        std::string right = merge.substr(space + 1);

        left_merges.push_back(apply_unicode_to_bytes(left));
        right_merges.push_back(apply_unicode_to_bytes(right));
    }

    auto left_merges_const = create_string_constant(left_merges);
    auto right_merges_const = create_string_constant(right_merges);

    inputs.insert(inputs.end(), left_merges_const.begin(), left_merges_const.end());
    inputs.insert(inputs.end(), right_merges_const.begin(), right_merges_const.end());

    // 3. Extract special tokens
    ov::Tensor token_types{};
    std::vector<std::string> tokens{};
    if (auto val = std::get_if<std::vector<std::string>>(&tokenizer_config.at("tokens"))) {
        tokens = *val;
    }
    if (auto val = std::get_if<ov::Tensor>(&tokenizer_config.at("token_type"))) {
        token_types = *val;
    }

    std::vector<std::vector<uint8_t>> special_tokens;
    std::vector<int32_t> special_token_indices;

    for (size_t i = 0; i < vocab.size(); ++i) {
        if (is_special_token(token_types.data<int32_t>()[i])) {
            special_tokens.push_back(vocab[i]);
            special_token_indices.push_back(static_cast<int32_t>(i));
        }
    }

    auto const_special_tokens = create_string_constant(special_tokens);
    inputs.insert(inputs.end(), const_special_tokens.begin(), const_special_tokens.end());
    ov::Output<ov::Node> const_special_token_indices =
        std::make_shared<v0::Constant>(element::i32, ov::Shape{special_token_indices.size()}, special_token_indices);
    inputs.push_back(const_special_token_indices);

    // 4. Build BPETokenizer node
    std::string unk_token = "";
    if (auto it = tokenizer_config.find("unknown_token_id");
        it != tokenizer_config.end() && std::holds_alternative<ov::Tensor>(it->second)) {
        const auto& tensor = std::get<ov::Tensor>(it->second);
        uint32_t unknown_token_id = tensor.data<uint32_t>()[0];
        unk_token = vocab_from_config[unknown_token_id];
    }

    int32_t cache_capacity =
        std::max<int32_t>(static_cast<int32_t>(vocab.size() * VOCAB_SIZE_CACHE_PROPORTION), MIN_CACHE_CAPACITY);

    std::map<std::string, ov::Any> attributes = {
        {"unk_token", unk_token},
        {"fuse_unk", true},
        {"suffix_indicator", std::string("")},
        {"end_suffix", std::string("")},
        {"byte_fallback", true},
        {"cache_capacity", cache_capacity},
    };

    return create_func("BPETokenizer", inputs, attributes);
}

/**
 * @brief This function aim to patch the chat template stored in gguf model to 
 * be consistent with chat template stored in the original tokenizer_config.json of huggingface models.
 * If certain mismatched pattern found, then the pattern will be replaced with a specific substring.
 * Otherwise, the original chat template is returned.
 * Current this function is used to patch the chat template for Qwen2.5 models, but the logic can be extended to other models
 * 
 *
 * Example: The function finds the substring for Qwen2.5:
 * "{{\"name\": <function-name>, \"arguments\": <args-json-object>}}"
 * in the input string (str1_content) and replaces it with:
 * "{\"name\": <function-name>, \"arguments\": <args-json-object>}"
 *
 * This assumes that "<function-name>" and "<args-json-object>" are literal
 * parts of the substring to be found.
 *
 * @param chat_template A string contains original chat template stored in gguf models
 * @return patched_chat_template A new string contains updated chat template with the specific replacement made if pattern matched.
 */
std::string patch_chat_template(const std::string& chat_template) {
    std::string patched_chat_template = chat_template;

    // Define the exact pattern to find in orignal chat_template
    // Using C++ raw string literals (R"(...)") to correctly represent the literal content,
    const std::string qwen2_5_substring_to_find =
        R"({{\"name\": <function-name>, \"arguments\": <args-json-object>}})";

    // Define the exact replacement substring for str2
    const std::string qwen2_5_replacement_substring =
        R"({\"name\": <function-name>, \"arguments\": <args-json-object>})";

    // Find the position of the substring to be replaced
    size_t pos = patched_chat_template.find(qwen2_5_substring_to_find);

    if (pos != std::string::npos) {
        // Substring found, perform the replacement
        patched_chat_template.replace(pos, qwen2_5_substring_to_find.length(), qwen2_5_replacement_substring);
    }

    return patched_chat_template;
}

std::tuple<std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>, std::map<std::string, GGUFMetaData>>
create_tokenizer_from_config(const std::shared_ptr<void>& shared_object_ov_tokenizers,
                             const std::filesystem::path& gguf_model_path) {
    auto gguf_metadata = std::get<0>(get_gguf_data(gguf_model_path.string()));
    auto tokenizer_config = tokenizer_config_from_meta(gguf_metadata);

    auto tokenizer_input = std::make_shared<v0::Parameter>(element::string, PartialShape{Dimension::dynamic()});

    FactoryCreateType create_func =
        reinterpret_cast<FactoryCreateType>(get_symbol(shared_object_ov_tokenizers, "create_tokenizer_node"));

    OutputVector outputs = create_func("StringTensorUnpack", {tokenizer_input}, {});

    // Add ragged dimension (you need to define this)
    outputs = add_ragged_dimension(outputs);

    // Special token filtering
    std::vector<std::string> tokens, special_tokens;
    ov::Tensor token_types;
    if (auto val = std::get_if<std::vector<std::string>>(&tokenizer_config.at("tokens"))) {
        tokens = *val;
    }
    if (auto val = std::get_if<ov::Tensor>(&tokenizer_config.at("token_type"))) {
        token_types = *val;
    }

    const auto token_types_data = token_types.data<int32_t>();

    for (size_t i = 0; i < tokens.size(); ++i) {
        if (is_special_token(token_types_data[i])) {
            special_tokens.push_back(tokens[i]);
        }
    }

    std::string special_tokens_re = join_special_tokens(special_tokens);

    ov::Tensor ov_special_tokens(ov::element::u8, {special_tokens_re.size()});
    std::memcpy(ov_special_tokens.data<uint8_t>(), special_tokens_re.data(), special_tokens_re.size());
    auto const_special_tokens = std::make_shared<v0::Constant>(ov_special_tokens);

    ov::OutputVector inputs_to_split = outputs;
    inputs_to_split.push_back(const_special_tokens->output(0));
    outputs = create_func("SpecialTokensSplit", inputs_to_split, {});

    // no normalization steps
    // Regex Split

    std::string pre{};
    if (auto val = std::get_if<std::string>(&tokenizer_config.at("pre"))) {
        pre = *val;
    }

    auto split_res = get_split_regex(pre);

    for (const auto& split_re : split_res) {
        std::string split_behaviour = "isolate";

        ov::Tensor ov_split_re(ov::element::u8, {split_re.size()});
        std::memcpy(ov_split_re.data<uint8_t>(), split_re.data(), split_re.size());
        auto const_ov_split_re = std::make_shared<v0::Constant>(ov_split_re);

        outputs.push_back(const_ov_split_re->output(0));
        outputs =
            create_func("RegexSplit", outputs, {{"behaviour", split_behaviour}, {"invert", false}, {"max_splits", -1}});
    }

    std::string model{};
    if (auto val = std::get_if<std::string>(&tokenizer_config.at("model"))) {
        model = *val;
    }

    ov::OutputVector bbpe_inputs(outputs.begin(), outputs.begin() + 5);
    outputs = parse_bbpe_config(tokenizer_config, bbpe_inputs, create_func);

    ov::Output<ov::Node> max_length = std::make_shared<v0::Constant>(element::i32, ov::Shape{}, MAX_LENGTH);
    ov::Output<ov::Node> ends_minus_begins = std::make_shared<v1::Subtract>(outputs[1], outputs[0]);
    max_length = std::make_shared<v1::Minimum>(ends_minus_begins, max_length);
    outputs[0] = std::make_shared<v1::Subtract>(outputs[1], max_length)->output(0);

    // Left padding
    ends_minus_begins = std::make_shared<v1::Subtract>(outputs[1], outputs[0]);
    auto reduce_axis = std::make_shared<v0::Constant>(element::i32, ov::Shape{1}, 0);
    ov::Output<ov::Node> max_length_batch = std::make_shared<v1::ReduceMax>(ends_minus_begins, reduce_axis, false);

    ov::OutputVector inputs_for_ragged_to_dense = outputs;
    inputs_for_ragged_to_dense.push_back(max_length_batch);
    ov::Output<ov::Node> const_zero_for_rg = std::make_shared<v0::Constant>(element::i32, ov::Shape{}, 0);
    inputs_for_ragged_to_dense.push_back(const_zero_for_rg);

    outputs =
        create_func("RaggedToDense", inputs_for_ragged_to_dense, {{"pad_right", false}, {"pad_max_length", false}});

    // Convert output types
    outputs[0] = std::make_shared<v0::Convert>(outputs[0], element::i64)->output(0);
    outputs[1] = std::make_shared<v0::Convert>(outputs[1], element::i64)->output(0);
    outputs[0].get_tensor().add_names({"input_ids"});
    outputs[1].get_tensor().add_names({"attention_mask"});
    
    auto tokenizer = std::make_shared<Model>(outputs, ParameterVector{tokenizer_input}, "tokenizer");

    std::string chat_template = "";
    if (auto val = std::get_if<std::string>(&tokenizer_config.at("chat_template"))) {
        chat_template = *val;
    }
    std::string patched_chat_template = patch_chat_template(chat_template);
    tokenizer->set_rt_info(patched_chat_template, "chat_template");
    /*
    std::cout << "config_chat_template: " << config_chat_template << std::endl;

    tokenizer->set_rt_info(config_chat_template, "chat_template");

    std::set<std::string> chat_template = {
    "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"};

    const ov::AnyMap& rt_info = tokenizer->get_rt_info();
    
    auto iter = rt_info.find("chat_template");
    std::string original_chat_template = "";
    if (rt_info.end() != iter) {
        std::cout << "Find chat_template in rt_info\n";
        original_chat_template = iter->second.as<std::string>();
    }

    std::cout << "original chat template: " << original_chat_template << std::endl;
    
    ov::save_model(tokenizer, "tokenizer_origin.xml", false);
    */
    
    //ov::save_model(tokenizer, "tokenizer_patched.xml", false);

    // DETOKENIZER model
    auto detokenizer_input =
        std::make_shared<v0::Parameter>(element::i64, PartialShape{Dimension::dynamic(), Dimension::dynamic()});

    OPENVINO_ASSERT(model == "gpt2");
    auto vocab = parse_bbpe_vocab(tokens);
    ov::OutputVector const_vocab = create_string_constant(vocab);
    OutputVector detokenizer_outputs = {detokenizer_input};
    detokenizer_outputs.insert(detokenizer_outputs.end(), const_vocab.begin(), const_vocab.end());

    std::vector<int32_t> special_token_ids;
    for (size_t i = 0; i < token_types.get_size(); ++i) {
        if (is_special_token(token_types.data<int32_t>()[i]))
            special_token_ids.push_back(static_cast<int32_t>(i));
    }

    // vocab decoder
    auto special_ids_const =
        std::make_shared<v0::Constant>(element::i32, ov::Shape{special_token_ids.size()}, special_token_ids);
    auto const_zero = std::make_shared<v0::Constant>(element::i32, ov::Shape{1}, 0);
    auto const_one = std::make_shared<v0::Constant>(element::i32, ov::Shape{1}, 1);
    int32_t int32_max_value = std::numeric_limits<int32_t>::max();
    auto const_int32_max = std::make_shared<v0::Constant>(element::i32, ov::Shape{1}, int32_max_value);
    auto sliced_skips =
        std::make_shared<v8::Slice>(special_ids_const, const_zero, const_int32_max, const_one)->outputs();
    detokenizer_outputs.insert(detokenizer_outputs.end(), sliced_skips.begin(), sliced_skips.end());

    // Decode
    detokenizer_outputs = create_func("VocabDecoder", detokenizer_outputs, {});
    ov::OutputVector inputs_for_fused_ragged(detokenizer_outputs.begin(), detokenizer_outputs.end() - 1);
    auto outputs_fused_ragged = create_func("FuzeRagged", inputs_for_fused_ragged, {});
    outputs_fused_ragged.insert(outputs_fused_ragged.end(), detokenizer_outputs.end() - 1, detokenizer_outputs.end());
    auto packed_output = create_func("StringTensorPack", outputs_fused_ragged, {});
    packed_output[0].get_tensor().add_names({"string_output"});
    auto detokenizer = std::make_shared<Model>(packed_output, ParameterVector{detokenizer_input}, "detokenizer");

    return {tokenizer, detokenizer, tokenizer_config};
}

}  // namespace genai
}  // namespace ov
