//#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include "openvino_backend_api.h"
#include "json.hpp"    

int get_status(openvino_backend::api_interface &llm_api_inferface){ 
    return llm_api_inferface.api_status();
}
void load_model(openvino_backend::api_interface &llm_api_inferface, char *cstr){ 
    llm_api_inferface.api_loadmodel(cstr, 0);
}
bool unload_model(openvino_backend::api_interface &llm_api_inferface){ 
    return llm_api_inferface.api_unloadmodel();
}

void load_tokenizer(openvino_backend::api_interface &llm_api_inferface, std::string tokenizer_path) {
    llm_api_inferface.api_loadtokenizer(tokenizer_path);
}

bool init_llm_backend(openvino_backend::api_interface &llm_api_inferface, std::string model_path, std::string tokenizer_path){
    char *cstr = new char[model_path.length() + 1];
    std::strcpy(cstr, model_path.c_str());
    load_model(llm_api_inferface, cstr);
    load_tokenizer(llm_api_inferface, tokenizer_path);
    return true;
}

std::string non_stream_generation(openvino_backend::api_interface &llm_api_inferface, std::string prompt, ov_params default_params){ 
    return llm_api_inferface.api_Generate(prompt, default_params);
};

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
    httplib::Server svr;
    svr.set_default_headers({{"Server", "rag_poc"}});

    // CORS preflight
    /*
    svr.Options(R"(.*)", [](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin",      req.get_header_value("Origin"));
        //res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods",     "POST");
        res.set_header("Access-Control-Allow-Headers",     "*");
        return res.set_content("", "application/json; charset=utf-8");
    });
    */

    ov_params default_params;
    openvino_backend::api_interface llm_api_inferface = openvino_backend::api_interface(default_params);

    std::string model_path = "Qwen-7B-Chat-NNCF_INT4/openvino_model.xml";
    std::string tokenizer_path = "Qwen-7B-Chat-NNCF_INT4/qwen.tiktoken";
    std::string user_prompt = "What is OpenVINO?";
        
    bool success = init_llm_backend(llm_api_inferface, model_path, tokenizer_path);
    std::cout << "Init RAG Model Sever Status: " << success << "\n";

    svr.Get("/Generate", [&](const httplib::Request &req, httplib::Response &res) {
	    //std::string user_prompt = req.body;
        std::string processed_prompt = apply_chat_template(user_prompt);
        std::string response = non_stream_generation(llm_api_inferface, processed_prompt, default_params);
        res.set_content(response, "text/plain");
    });

    svr.listen("0.0.0.0", 8080);
}
