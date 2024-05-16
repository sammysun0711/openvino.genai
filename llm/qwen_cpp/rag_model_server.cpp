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
        
    bool status = init_llm_backend(llm_api_inferface, model_path, tokenizer_path);
    std::cout << "Init RAG Model Sever Status: " << status << "\n";
 
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
