//#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include "openvino_backend_api.h"
#include "json.hpp"    

using json = nlohmann::ordered_json;

int get_status(openvino_backend::api_interface &llm_api_interface){ 
    return llm_api_interface.api_status();
}
void load_model(openvino_backend::api_interface &llm_api_interface, char *cstr){ 
    llm_api_interface.api_loadmodel(cstr, 0);
}
bool unload_model(openvino_backend::api_interface &llm_api_interface){ 
    return llm_api_interface.api_unloadmodel();
}

void load_tokenizer(openvino_backend::api_interface &llm_api_interface, std::string tokenizer_path) {
    llm_api_interface.api_loadtokenizer(tokenizer_path);
}

bool init_llm_backend(openvino_backend::api_interface &llm_api_interface, std::string model_path, std::string tokenizer_path){
    char *cstr = new char[model_path.length() + 1];
    std::strcpy(cstr, model_path.c_str());
    load_model(llm_api_interface, cstr);
    load_tokenizer(llm_api_interface, tokenizer_path);
    return true;
}

std::string non_stream_generation(openvino_backend::api_interface &llm_api_interface, std::string prompt, ov_params default_params){ 
    return llm_api_interface.api_Generate(prompt, default_params);
};

void reset_llm_backend(openvino_backend::api_interface &llm_api_interface){
    llm_api_interface.api_Reset();
}

std::string apply_chat_template(std::string user_prompt){
    std::string system_message = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>";
    std::ostringstream oss_prompt;
    oss_prompt << system_message << "\n<|im_start|>user\n"
                << user_prompt << "<|im_end|>\n<|im_start|>assistant\n";
    std::string processed_prompt = oss_prompt.str();

    return processed_prompt;
}
/*
std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string&& prompt) {
    constexpr size_t BATCH_SIZE = 1;
    //auto input_tensor = ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt};

    //tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt});
    std::cout << "Init input tensor works\n";
    tokenizer.set_input_tensor(input_tensor);
    std::cout << "Set input tensor works\n";
    tokenizer.infer();
    std::cout << "Tokenizer infer works\n";
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}*/
std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string prompt) {
    constexpr size_t BATCH_SIZE = 1;
    auto input_tensor = ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt};
    std::cout << "prompt: " << prompt;
    //auto input_tensor = ov::Tensor{ov::element::string, {prompt.length() + 1}, &prompt};
    auto shape = input_tensor.get_shape();
    std::cout << "input tensor shape: [ ";
    for(auto s: shape){
        std::cout << s << " ";
    }
    std::cout << "]\n";
    std::cout << "Init input tensor works\n";
    tokenizer.set_input_tensor(input_tensor);
    std::cout << "Set input tensor works\n";
    tokenizer.infer();
    std::cout << "Tokenizer infer works\n";
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

void run_bert_embedding(std::string query){
    std::string bert_path = "bge-large-zh-v1.5/openvino_model.xml";
    std::string bert_tokenizer_path = "bge-large-zh-v1.5/openvino_tokenizer.xml";
    ov::Core core;
    core.add_extension(OPENVINO_TOKENIZERS_PATH);  // OPENVINO_TOKENIZERS_PATH is defined in CMakeLists.txt
    //Read the tokenizer model information from the file to later get the runtime information
    ov::InferRequest embedding_model = core.compile_model(bert_path, "CPU").create_infer_request();
    std::cout << "Load embedding model successed\n";
    auto tokenizer_model = core.read_model(bert_tokenizer_path);
    ov::InferRequest tokenizer = core.compile_model(tokenizer_model, "CPU").create_infer_request();
    std::cout << "Load tokenizer model successed\n";
    //char *temp_query = new char[query.length() + 1];
    //std::strcpy(temp_query, query.c_str());
    auto [input_ids, attention_mask] = tokenize(tokenizer, query);
    std::cout << "tokenize encode successed\n";
    auto seq_len = input_ids.get_size();

    // Initialize inputs
    embedding_model.set_tensor("input_ids", input_ids);
    embedding_model.set_tensor("attention_mask", attention_mask);
    ov::Tensor position_ids = embedding_model.get_tensor("position_ids");
    position_ids.set_shape(input_ids.get_shape());
    std::iota(position_ids.data<int64_t>(), position_ids.data<int64_t>() + seq_len, 0);
    constexpr size_t BATCH_SIZE = 1;
    embedding_model.infer();
    std::cout << "embedding infer successed\n";
}

int main(){
    // HTTP Server
    //httplib::Server svr;
    std::cout << "start\n";
    std::unique_ptr<httplib::Server> svr;
    svr.reset(new httplib::Server());
    std::cout << "Init http server\n";
    svr->set_default_headers({{"Server", "openvino.genai.rag_poc"}});

    // CORS preflight
    svr->Options(R"(.*)", [](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin",      req.get_header_value("Origin"));
        //res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods",     "POST");
        res.set_header("Access-Control-Allow-Headers",     "*");
        return res.set_content("", "application/json; charset=utf-8");
    });

    ov_params default_params;
    std::cout << "Init openvino backend api\n";
    openvino_backend::api_interface llm_api_inferface = openvino_backend::api_interface(default_params);

    std::string model_path = "Qwen-7B-Chat-NNCF_INT4/openvino_model.xml";
    std::string tokenizer_path = "Qwen-7B-Chat-NNCF_INT4/qwen.tiktoken";
    //std::string user_prompt = "What is OpenVINO?";
    //std::string documents_chunk = "Create LLM-powered Chatbot using OpenVINO\n\nIn the rapidly evolving world of artificial intelligence (AI), chatbots have emerged as powerful tools for businesses to enhance customer interactions and streamline operations. \nLarge Language Models (LLMs) are artificial intelligence systems that can understand and generate human language. They use deep learning algorithms and massive amounts of data to learn the nuances of language and produce coherent and relevant responses.\nWhile a decent intent-based chatbot can answer basic, one-touch inquiries like order management, FAQs, and policy questions, LLM chatbots can tackle more complex, multi-touch questions. LLM enables chatbots to provide support in a conversational manner, similar to how humans do, through contextual memory. Leveraging the capabilities of Language Models, chatbots are becoming increasingly intelligent, capable of understanding and responding to human language with remarkable accuracy.";
    std::string documents_chunk = "test";
    bool status = init_llm_backend(llm_api_inferface, model_path, tokenizer_path);
    std::cout << "Init RAG Model Sever Status: " << status << "\n";
    run_bert_embedding(documents_chunk);
 
    const auto handle_completions = [&llm_api_inferface, &default_params](const httplib::Request & req, httplib::Response & res) {
	    res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
	    std::cout << "req.body: " << req.body << "\n";
	    std::string prompt = req.body;
	    std::string processed_prompt = apply_chat_template(prompt);
	    std::cout << "processed_prompt: " << processed_prompt << "\n";
	    std::string response = non_stream_generation(llm_api_inferface, processed_prompt, default_params);
	    reset_llm_backend(llm_api_inferface);
	    res.set_content(response, "text/plain");
	    //json data = json::parse(req.body);
	    //std::string s = data.dump();
	    //std::cout << "data: " << s << "\n";
    };
    //svr->Get ("/health",    handle_health);
    svr->Post("/completions", handle_completions);
    //svr->Post("/embeddings",          handle_embeddings);
    svr->listen("0.0.0.0", 8080);
}
