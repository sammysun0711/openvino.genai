// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
// #include <windows.h>
#include "httplib.h"
#include "iostream"
#include "json.hpp"

using json = nlohmann::json;

bool fileExists(const std::string& path) {
    std::ifstream file(path.c_str());
    return file.good();
}

void custom_sleep(int seconds) {
#ifdef _WIN32
    Sleep(seconds * 1000);
#else
    sleep(seconds);
#endif
}

static auto usage() -> void {
    std::cout << "Usage: "
              << " [options]\n"
              << "\n"
              << "options:\n"
              << "  help     \n"
              << "  init_embeddings     \n"
              << "  embeddings          \n"
              << "  db_retrieval          \n"
              << "  db_retrieval_llm          \n"
              << "  embeddings_unload          \n"
              << "  llm_init            \n"
              << "  llm         \n"
              << "  llm_unload            \n"
              << "  health_cheak            \n"
              << "  exit         \n";
}

bool check_vaild_sentence(std::string sentence) {
    if (sentence == "\n" && sentence[0] == '\n') {
        std::cout << "Invalid sentence\n";
        return false;
    }
    int iCount = std::count(sentence.begin(), sentence.end(), '\n');
    if (iCount > 1) {
        std::cout << "Invalid sentence iCount>1\n";
        return false;
    }

    return true;
}

int main() {
    // HTTP
    std::cout << "Init client \n";
    httplib::Client cli("http://127.0.0.1:7890");
    cli.set_default_headers({{"Client", "openvino.genai.rag_sample"}});
    std::string command;
    std::cout << "Init client finished\n";
    usage();
    bool status = true;
    cli.set_connection_timeout(100, 0);  // set default timeout to 100 seconds
    cli.set_read_timeout(100, 0);  // set default timeout to 100 seconds    
    cli.set_write_timeout(100, 0);   // set default timeout to 100 seconds

    while (std::cin >> command && command != "exit") {
        if (command == "help") {
            usage();
        } else if (command == "init_embeddings") {
            auto init_embeddings = cli.Post("/embeddings_init", "", "");
            if (init_embeddings->status == httplib::StatusCode::OK_200) {
                std::cout << init_embeddings->body << "\n";
            } else {
                std::cout << "Init embeddings failed\n";
                std::cout << "Status: " << httplib::status_message(init_embeddings->status) << std::endl;
            }
        } else if (command == "embeddings") {
            std::cout << "This is the unit test for embeddings: \n";
            std::cout << "Path of test json file: \n";
            std::string path;
            std::getline(std::cin, path);
            if (path == "Stop!") 
                break;
            while (true) {
                getline(std::cin, path);
                if (path.length() != 0) {
                    if (path == "exit")
                        break;
                    if (fileExists(path)) {
                        std::cout << "Succeed to read the json file." << std::endl;
                    } else {
                        std::cout << "Failed to read json file" << std::endl;
                    }  
                         
                    std::cout << "path: " << path << "\n";

                    std::ifstream f(path);
                    json data = json::parse(f);
                    auto embeddings = cli.Post("/embeddings", data.dump(), "application/json");
                    if (embeddings->status == httplib::StatusCode::OK_200) {
                        std::cout << embeddings->body << "\n";
                    } else {
                        std::cout << "Embeddings failed\n";
                        std::cout << "Status: " << httplib::status_message(embeddings->status) << std::endl;
                    }
                }
            }
        } else if (command == "llm_init") {
            auto llm_init = cli.Post("/llm_init", "", "");
            if (llm_init->status == httplib::StatusCode::OK_200) {
                std::cout << llm_init->body << "\n";
            } else {
                std::cout << "Init llm failed\n";
                std::cout << "Status: " << httplib::status_message(llm_init->status) << std::endl;
            }

        } else if (command == "llm") {
            std::string user_prompt;
            std::cout << "Enter your prompt: ";
            while (true) {
                getline(std::cin, user_prompt);
                if (user_prompt.length() != 0) {
                    if (user_prompt == "exit")
                        break;
                    auto completions = cli.Post("/completions", user_prompt, "text/plain");
                    custom_sleep(1);
                    if (completions->status == httplib::StatusCode::OK_200) {
                        std::cout << "completions->body: " << completions->body << "\n";
                    } else {
                        std::cout << "Completions failed\n";
                        std::cout << "Status: " << httplib::status_message(completions->status) << std::endl;
                    }
                    std::cout << "Enter your prompt: ";
                }
            }
        } else if (command == "db_retrieval") {
            std::string query_prompt;
            std::cout << "Enter your prompt for DB: ";
            while (true) {
                getline(std::cin, query_prompt);
                if (query_prompt.length() != 0) {
                    if (query_prompt == "exit")
                        break;
                    auto db_retrieval = cli.Post("/db_retrieval", query_prompt, "text/plain");
                    custom_sleep(1);
                    if (db_retrieval->status == httplib::StatusCode::OK_200) {
                        std::cout << "db_retrieval->body: " << db_retrieval->body << "\n";
                    } else {
                        std::cout << "db_retrieval failed\n";
                        std::cout << "Status: " << httplib::status_message(db_retrieval->status) << std::endl;
                    }
                }
            }
        } else if (command == "db_retrieval_llm") {
            std::string query_prompt;
            std::cout << "Enter your prompt for DB: ";
            while (true) {
                getline(std::cin, query_prompt);
                if (query_prompt.length() != 0) {
                    if (query_prompt == "exit")
                        break;
                    auto db_retrieval = cli.Post("/db_retrieval_llm", query_prompt, "text/plain");
                    custom_sleep(1);
                    if (db_retrieval->status == httplib::StatusCode::OK_200) {
                        std::cout << "db_retrieval->body: " << db_retrieval->body << "\n";
                    } else {
                        std::cout << "db_retrieval failed\n";
                        std::cout << "Status: " << httplib::status_message(db_retrieval->status) << std::endl;
                    }
                }
            }
        } else if (command == "llm_unload") {
            auto llm_unload = cli.Post("/llm_unload", "", "");
            if (llm_unload->status == httplib::StatusCode::OK_200) {
                std::cout << "Unload llm success\n";
            } else {
                std::cout << "Unload llm failed\n";
                std::cout << "Status: " << httplib::status_message(llm_unload->status) << std::endl;
            }
        } else if (command == "embeddings_unload") {
            auto embeddings_unload = cli.Post("/embeddings_unload", "", "");
            if (embeddings_unload->status == httplib::StatusCode::OK_200) {
                std::cout << "Unload embeddings success\n";
            } else {
                std::cout << "Unload embeddings failed\n";
                std::cout << "Status: " << httplib::status_message(embeddings_unload->status) << std::endl;
            }
        } else if (command == "health_cheak") {
            auto health = cli.Post("/health", "", "");
            if (health->status == httplib::StatusCode::OK_200) {
                std::cout << "health: " << health->body << "\n";
            } else {
                std::cout << "health_cheak failed\n";
                std::cout << "Status: " << httplib::status_message(health->status) << std::endl;
            }
        }

        else {
            std::cerr << "Unknown argument: " << std::endl;
            usage();
        }
    }

    return 0;
}
