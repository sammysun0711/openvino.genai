#ifndef _HANDLE_MASTER
#define _HANDLE_MASTER

#include <chrono>
#include <fstream>
#include <iostream>
#include <openvino/openvino.hpp>
#include <sstream>
#include <variant>

#include "embeddings.hpp"
#include "httplib.h"
#include "json.hpp"
#include "util.hpp"
#include "state.hpp"

using json = nlohmann::json;
using HandleInput = std::variant<int, std::shared_ptr<Embeddings>, std::shared_ptr<ov::genai::LLMPipeline>>;

class HandleMaster {
public:
    HandleMaster() = default;
    ~HandleMaster() = default;

    // std::function<void(const httplib::Request&, httplib::Response&)> get_handle(std::string handle_name,
    //                                                                             HandleInput handle_type,
    //                                                                             util::Args args);

    std::function<void(const httplib::Request&, httplib::Response&)> get_handle_llm_init(
        util::ServerContest& server_contest_ref,
        util::Args args);

    std::function<void(const httplib::Request&, httplib::Response&)> get_handle_llm(
        util::ServerContest& server_contest_ref,
        util::Args args);

    std::function<void(const httplib::Request&, httplib::Response&)> get_handle_llm_unload(
        util::ServerContest& server_contest_ref);

    std::function<void(const httplib::Request&, httplib::Response&)> get_handle_embeddings_init(
        util::ServerContest& server_contest_ref,
        util::Args args);

    std::function<void(const httplib::Request&, httplib::Response&)> get_handle_embeddings(
        util::ServerContest& server_contest_ref);

    std::function<void(const httplib::Request&, httplib::Response&)> get_handle_embeddings_unload(
        util::ServerContest& server_contest_ref);

    std::function<void(const httplib::Request&, httplib::Response&)> get_handle_health(
        util::ServerContest& server_contest_ref);


};

#endif