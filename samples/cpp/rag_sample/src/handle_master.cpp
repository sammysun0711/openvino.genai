#include "handle_master.hpp"

using json = nlohmann::json;

std::function<void(const httplib::Request&, httplib::Response&)> HandleMaster::get_handle_llm_init(
    util::ServerContext& server_context_ref) {
    const auto handle_llm_init = [&server_context_ref](const httplib::Request& req, httplib::Response& res) {
        if (server_context_ref.llm_state == State::STOPPED || server_context_ref.llm_state == State::ERR) {
            server_context_ref.llm_pointer =
                std::make_shared<ov::genai::LLMPipeline>(server_context_ref.args.llm_model_path,
                                                         server_context_ref.args.llm_device);
            server_context_ref.llm_pointer->start_chat();
            server_context_ref.llm_state = State::IDLE;
            res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
            res.set_content("Init llm success.", "text/plain");

        } else {
            res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
            res.set_content("Cannot init llm, cause llm is already be initialized.", "text/plain");
        }
    };
    return handle_llm_init;
}

std::function<void(const httplib::Request&, httplib::Response&)> HandleMaster::get_handle_llm(
    util::ServerContext& server_context_ref) {
    const auto handle_llm = [&server_context_ref](const httplib::Request& req_llm, httplib::Response& res_llm) {
        if (server_context_ref.llm_state == State::IDLE) {
            res_llm.set_header("Access-Control-Allow-Origin", req_llm.get_header_value("Origin"));
            std::cout << "req_llm.body: " << req_llm.body << "\n";
            std::string prompt = req_llm.body;
            server_context_ref.llm_state = State::RUNNING;
            auto config = server_context_ref.llm_pointer->get_generation_config();
            config.max_new_tokens = server_context_ref.args.max_new_tokens;
            std::string response = server_context_ref.llm_pointer->generate(prompt, config);
            server_context_ref.llm_state = State::IDLE;
            std::cout << "response: " << response << "\n";
            res_llm.set_content(response, "text/plain");
        } else {
            res_llm.set_header("Access-Control-Allow-Origin", req_llm.get_header_value("Origin"));
            res_llm.set_content(
                "Cannot do llm chat, cause llm inferrequest is now not initialized or busy, check the stats of llm.",
                "text/plain");
        }
    };
    return handle_llm;
}

std::function<void(const httplib::Request&, httplib::Response&)> HandleMaster::get_handle_llm_unload(
    util::ServerContext& server_context_ref) {
    const auto handle_llm_unload = [&server_context_ref](const httplib::Request& req, httplib::Response& res) {
        server_context_ref.llm_pointer->finish_chat();
        server_context_ref.llm_pointer.reset();
        server_context_ref.llm_state = State::STOPPED;
    };
    return handle_llm_unload;
}

std::function<void(const httplib::Request&, httplib::Response&)> HandleMaster::get_handle_embeddings_init(
    util::ServerContext& server_context_ref) {
    const auto handle_embeddings_init = [&server_context_ref](const httplib::Request& req_embedding,
                                                              httplib::Response& res_embedding) {
        if (server_context_ref.embedding_state == State::STOPPED || server_context_ref.embedding_state == State::ERR) {
            server_context_ref.embedding_pointer = std::make_shared<Embeddings>();
            server_context_ref.embedding_pointer->init(server_context_ref.args.embedding_model_path,
                                                       server_context_ref.args.embedding_device);
            server_context_ref.embedding_state = State::IDLE;
            res_embedding.set_header("Access-Control-Allow-Origin", req_embedding.get_header_value("Origin"));
            res_embedding.set_content("Init embeddings success.", "text/plain");
        } else {
            res_embedding.set_header("Access-Control-Allow-Origin", req_embedding.get_header_value("Origin"));
            res_embedding.set_content("Cannot init embeddings, cause embeddings is already be initialized.",
                                      "text/plain");
        }
    };

    return handle_embeddings_init;
}

std::function<void(const httplib::Request&, httplib::Response&)> HandleMaster::get_handle_embeddings(
    util::ServerContext& server_context_ref) {
    const auto handle_embeddings = [&server_context_ref](const httplib::Request& req_embedding,
                                                         httplib::Response& res_embedding) {
        if (server_context_ref.embedding_state == State::IDLE) {
            res_embedding.set_header("Access-Control-Allow-Origin", req_embedding.get_header_value("Origin"));
            json json_file = json::parse(req_embedding.body);
            std::cout << "get json_file successed\n";
            std::vector<std::string> inputs;
            for (auto& elem : json_file["data"])
                inputs.push_back(elem);
            std::cout << "get inputs successed\n";
            server_context_ref.embedding_state = State::RUNNING;
            std::vector<std::vector<std::vector<float>>> res =
                server_context_ref.embedding_pointer->encode_queries(inputs);
            server_context_ref.embedding_state = State::IDLE;
            res_embedding.set_content("Embeddings success", "text/plain");
        } else {
            res_embedding.set_header("Access-Control-Allow-Origin", req_embedding.get_header_value("Origin"));
            res_embedding.set_content("Cannot do embeddings, cause embeddings inferrequest is now not initialized or busy, check "
                                      "the stats of embeddings.",
                                      "text/plain");
        }
    };

    return handle_embeddings;
}

std::function<void(const httplib::Request&, httplib::Response&)> HandleMaster::get_handle_embeddings_unload(
    util::ServerContext& server_context_ref) {
    const auto handle_embeddings_unload = [&server_context_ref](const httplib::Request& req_embedding,
                                                                httplib::Response& res_embedding) {
        server_context_ref.embedding_pointer.reset();
        server_context_ref.embedding_state = State::STOPPED;
    };

    return handle_embeddings_unload;
}

std::function<void(const httplib::Request&, httplib::Response&)> HandleMaster::get_handle_health(
    util::ServerContext& server_context_ref) {
    const auto handle_health = [&server_context_ref](const httplib::Request& req_health,
                                                     httplib::Response& res_health) {
        res_health.set_header("Access-Control-Allow-Origin", req_health.get_header_value("Origin"));

        std::string response_emb;
        switch (server_context_ref.embedding_state) {
        case (State::STOPPED):
            response_emb = "STOPPED";
            break;
        case (State::IDLE):
            response_emb = "IDLE";
            break;
        case (State::RUNNING):
            response_emb = "RUNNING";
            break;
        case (State::ERR):
            response_emb = "ERROR";
            break;
        }

        std::string response_llm;
        switch (server_context_ref.llm_state) {
        case (State::STOPPED):
            response_llm = "STOPPED";
            break;
        case (State::IDLE):
            response_llm = "IDLE";
            break;
        case (State::RUNNING):
            response_llm = "RUNNING";
            break;
        case (State::ERR):
            response_llm = "ERROR";
            break;
        }

        std::string response = "embedding_state: " + response_emb + "   llm_state: " + response_llm;
        // std::cout << "embeddings state: " << response << "\n";
        res_health.set_content(response, "text/plain");
    };

    return handle_health;
}
