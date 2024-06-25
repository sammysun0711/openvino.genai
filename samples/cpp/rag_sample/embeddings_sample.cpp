// #include "embeddings.hpp"
#include "httplib.h"
#include "json.hpp"
#include "util.hpp"
#include "handle_master.hpp"
using json = nlohmann::json;



int main(int argc, char** argv) {
    // std::cout << "load json\n";

    // std::ifstream f("C:/Users/openvino/ys/xiake_genai/openvino.genai/samples/cpp/rag_sample/document_data.json");
    // json data = json::parse(f);
    // std::vector<std::string> inputs;
    // for (std::string elem : data["data"])
    //     inputs.push_back(elem);
    // std::shared_ptr<Embeddings> emb_ptr = std::make_shared<Embeddings>("C:/Users/openvino/ys/xiake_genai/models/bge-large-zh-v1.5", "CPU");
    // std::vector<std::vector<std::vector<float>>> res = emb_ptr->encode_queries(inputs);
    // emb_ptr.reset();
    

    std::unique_ptr<httplib::Server> svr;
    svr.reset(new httplib::Server());
    std::cout << "Init http server" << std::endl;
        svr->Options(R"(.*)", [](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        res.set_header("Access-Control-Allow-Methods", "POST");
        res.set_header("Access-Control-Allow-Headers", "*");
        return res.set_content("", "application/json; charset=utf-8");
    });

    std::shared_ptr<Test> test;
    // const auto test_init = [&test](const httplib::Request& req_embedding,
    //                                                 httplib::Response& res_embedding) {
    //     test = std::make_shared<Test>();
    //     test->init(1,2);
    // };

    // const auto test = [&test](const httplib::Request& req_embedding,
    //                                                 httplib::Response& res_embedding) {
    //     test->printA();
    //     test->printB();
    // };
    HandleMaster handle_master;

    auto test_init = handle_master.get_test_init(test);
    auto test_handler = handle_master.get_test(test);
    svr->Post("/embeddings_init", test_init);
    svr->Post("/embeddings", test_handler);

    svr->listen("127.0.0.1", 7890);
} 