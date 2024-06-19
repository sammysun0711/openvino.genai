#include <unistd.h>

#include "httplib.h"
#include "iostream"
#include "json.hpp"
using json = nlohmann::json;

static auto usage() -> void {
    std::cout << "Usage: " << " [options]\n"
              << "\n"
              << "options:\n"
              << "  help     \n"
              << "  init_embeddings     \n"
              << "  embeddings          \n"
              << "  llm_init            \n"
              << "  llm         \n"
              << "  exit         \n";
}

bool check_vaild_sentence(std::string sentence) {
    if (sentence == "\n" and sentence[0] == '\n') {
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
    httplib::Client cli("http://0.0.0.0:8080");
    cli.set_default_headers({{"Client", "openvino.genai.rag_sample"}});
    std::string command;
    std::cout << "Init client finished\n";
    bool status = true;
    while (std::cin >> command && command != "exit") {
        if (command == "help") {
            usage();
        } else if (command == "init_embeddings") {
            auto init_embeddings = cli.Post("/embeddings_init", "", "");
            if (init_embeddings->status == 200) {
                std::cout << "Init embeddings success\n";
            } else {
                std::cout << "Init embeddings failed\n";
            }
        } else if (command == "embeddings") {
            std::cout << "load json\n";
            std::ifstream f("../../../../samples/cpp/rag_sample/"
                            "document_data.json");
            json data = json::parse(f);
            auto embeddings = cli.Post("/embeddings", data.dump(), "application/json");
            if (embeddings->status == 200) {
                std::cout << "Embeddings success\n";
            } else {
                std::cout << "Embeddings failed\n";
            }
        } else if (command == "llm_init") {
            auto llm_init = cli.Post("/llm_init", "", "");
            if (llm_init->status == 200) {
                std::cout << "Init llm success\n";
            } else {
                std::cout << "Init llm failed\n";
            }

        } else if (command == "llm") {
            std::string user_prompt;
            std::cout << "Enter your prompt: ";
            while (std::cin >> command && command != "exit") {
                getline(std::cin, user_prompt);

                if (check_vaild_sentence(command + user_prompt)) {
                    auto completions = cli.Post("/completions", command + user_prompt, "text/plain");
                    if (completions->status == 200) {
                        std::cout << "completions->body: " << completions->body << "\n";
                    } else {
                        std::cout << "Completions failed\n";
                    }
                    std::cout << "Enter your prompt: ";
                }
            }

        } else {
            std::cerr << "Unknown argument: " << std::endl;
            usage();
        }
    }

    return 0;
}
