// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
/*
#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (3 != argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<PROMPT>'");
    }

    std::string models_path = argv[1];
    std::string prompt = argv[2];

    std::string device = "CPU";  // GPU can be used as well
    ov::genai::LLMPipeline pipe(models_path, device);

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    config.do_sample = true;
    config.top_p = 0.9;
    config.top_k = 30;
    auto streamer = [](std::string subword) {
        std::cout << subword << std::flush;
        return false;
    };

    // Since the streamer is set, the results will
    // be printed each time a new token is generated.
    pipe.generate(prompt, config, streamer);
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
*/

// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/streamer_base.hpp"
#include <iostream>
#include <queue>
/*
class IterableStreamer: public ov::genai::StreamerBase {
public:
    IterableStreamer(ov::genai::Tokenizer tokenizer){
        _tokenizer = tokenizer;
    }

    std::string _next(){
        std::string value = "";
        if (_queue.size() > 0){
            value = _queue.front();
        }
        else{
            std::cout << "_queue.size() is empty!\n";
        }
        return value;
    }

    bool _get_stop_flag(){
        return false;
    }

    void put_word(std::string word){
        _queue.push(word);
    }


    bool put(int64_t token) override {
        // Custom decoding/tokens processing logic.

        // Returns a flag whether generation should be stopped, if true generation stops.
        // return false;
        
        _tokens_cache.push_back(token);
        std::string text = _tokenizer.decode(_tokens_cache);
        std::string word = "";
        if ((text.size() > _print_len) && ('\n' == text.back())){
            // Flush the cache after the new line symbol.
            word = text.substr(_print_len);
            _tokens_cache = {};
            _print_len = 0;
        }

        else if ((text.size() >=3) && (text.substr(-3).c_str() == (char *)65533)){
            // Don't print incomplete text.
            std::cout << "Don't print incomplete text.\n";
        }
        else if (text.size() > _print_len){
            // It is possible to have a shorter text after adding new token.
            // Print to output only if text length is increaesed.
            word = text.substr(_print_len);
            _print_len = text.size();
        }
        put_word(word);

        if (_get_stop_flag()){
            // When generation is stopped from streamer then end is not called, need to call it here manually.
            end();
            return true;   // True means stop  generation
        }
        else{
            return false;  // False means continue generation
        }

    };

    void end() override {
        
        //Flushes residual tokens from the buffer and puts a None value in the queue to signal the end.
        
        // Custom finalization logic.
        
        std::string text = _tokenizer.decode(_tokens_cache);
        if (text.size() > _print_len){
            std::string word = text.substr(_print_len);
            put_word(word);
            _tokens_cache = {};
            _print_len = 0;
        }
        put_word("");
        
    };

    //std::function<bool(std::string)> on_finalized_subword_callback = [](std::string words)->bool { return false; };

public:
    ov::genai::Tokenizer _tokenizer;
    std::vector<int64_t> _tokens_cache = {};
    std::queue<std::string> _queue;
    int _print_len = 0;

};
*/

/*
class ChunkStreamer: public IterableStreamer {

public:
    ChunkStreamer(ov::genai::Tokenizer tokenizer, int tokens_len){
        //IterableStreamer(tokenizer);
        _tokens_len = tokens_len;
    };

    bool put(int64_t token){
        if (((_tokens_cache.size() + 1) % _tokens_len) != 0){
            _tokens_cache.push_back(token);
            return false;
        }
        return put(token);
    }
        
private:
    int _tokens_len = 0;


};
*/
/*
int main(int argc, char* argv[]) try {
    if (3 != argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<PROMPT>'");
    }

    std::string models_path = argv[1];
    std::string prompt = argv[2];

    std::string device = "CPU";  // GPU can be used as well
    ov::genai::LLMPipeline pipe(models_path, device);

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    config.do_sample = true;
    config.top_p = 0.9;
    config.top_k = 30;
    
    auto streamer = [](std::string subword) {
        std::cout << subword << std::flush;
        return false;
    };
    
    // Since the streamer is set, the results will
    // be printed each time a new token is generated.
    IterableStreamer iterable_streamer = IterableStreamer(pipe.get_tokenizer());
    auto lambda_func = [&iterable_streamer](std::string subword) {
        while(iterable_streamer._next()!=std::string("")){
            std::cout<< "Hi\n"; 
        }
        
        //for (auto subword: &iterable_streamer){
        //    std::cout << subword << std::flush;
        //}

        return false;
    };

    //ChunkStreamer text_print_streamer = ChunkStreamer(pipe.get_tokenizer());
    //pipe.generate(prompt, config, ov::genai::streamer(lambda_func));
    pipe.generate(prompt, config, ov::genai::streamer(lambda_func));
    //pipe.generate(prompt, config, ov::genai::streamer(streamer));

} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
*/
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include <iostream>

class CustomStreamer: public ov::genai::StreamerBase {
public:
    CustomStreamer(ov::genai::Tokenizer tokenizer){
        _tokenizer = tokenizer;
    }
        std::string _next(){
        std::string value = "";
        if (_queue.size() > 0){
            value = _queue.front();
        }
        else{
            std::cout << "_queue.size() is empty!\n";
        }
        return value;
    }

    bool _get_stop_flag(){
        return false;
    }

    void put_word(std::string word){
        _queue.push(word);
    }

    /*
    bool put(int64_t token) {
        // Custom decoding/tokens processing logic.

        // Returns a flag whether generation should be stopped, if true generation stops.
        return false;
    };
    */
       bool put(int64_t token) override {
        // Custom decoding/tokens processing logic.

        // Returns a flag whether generation should be stopped, if true generation stops.
        // return false;
        
        _tokens_cache.push_back(token);
        std::string text = _tokenizer.decode(_tokens_cache);
        std::string word = "";
        if ((text.size() > _print_len) && ('\n' == text.back())){
            // Flush the cache after the new line symbol.
            word = text.substr(_print_len);
            _tokens_cache = {};
            _print_len = 0;
        }

        else if ((text.size() >=3) && (text.substr(-3).c_str() == (char *)65533)){
            // Don't print incomplete text.
            std::cout << "Don't print incomplete text.\n";
        }
        else if (text.size() > _print_len){
            // It is possible to have a shorter text after adding new token.
            // Print to output only if text length is increaesed.
            word = text.substr(_print_len);
            _print_len = text.size();
        }
        put_word(word);

        if (_get_stop_flag()){
            // When generation is stopped from streamer then end is not called, need to call it here manually.
            end();
            return true;   // True means stop  generation
        }
        else{
            return false;  // False means continue generation
        }

    };
    /*
    void end() {
        // Custom finalization logic.
        std::cout << "end" << std::endl;
    };
    */
    void end() override {
        
        //Flushes residual tokens from the buffer and puts a None value in the queue to signal the end.
        
        // Custom finalization logic.
        
        std::string text = _tokenizer.decode(_tokens_cache);
        if (text.size() > _print_len){
            std::string word = text.substr(_print_len);
            put_word(word);
            _tokens_cache = {};
            _print_len = 0;
        }
        put_word("");
        
    };


private:
   ov::genai::Tokenizer _tokenizer;
   std::vector<int64_t> _tokens_cache = {};
   std::queue<std::string> _queue;
   int _print_len = 0;

};


int main(int argc, char* argv[]) {
    std::string models_path = argv[1];
    ov::genai::LLMPipeline pipe(models_path, "CPU");

    CustomStreamer custom_streamer(pipe.get_tokenizer());
    
    /*
    auto lambda_func = [&custom_streamer](std::string subword) {
        //std::cout << subword << std::flush;
        custom_streamer.put_word(subword);
        std::cout << custom_streamer._next() << std::endl;
        return false;
    };
    */
    /*
    auto lambda_func = [&custom_streamer](int64_t token_id) {
        //std::cout << subword << std::flush;
        custom_streamer.put(token_id);
        std::cout << custom_streamer._next() << std::endl;
        return false;
    };
    */

    std::cout << pipe.generate("The Sun is yellow because", ov::genai::max_new_tokens(15), ov::genai::streamer(lambda_func));
}
