//#define CPPHTTPLIB_OPENSSL_SUPPORT
#include "httplib.h"

int main(){
    // HTTP
    httplib::Client cli("http://localhost:8080");
    const httplib::Headers headers = { {"Accept", "application/json"}, };
    cli.set_default_headers(headers);

    // HTTPS
    //httplib::Client cli("https://cpp-httplib-server.yhirose.repl.co");
    std::string user_prompt = "What is OpenVINO?";

    auto res = cli.Get("/Generate");
    //auto res = svr.Get("/Generate", [&](const httplib::Request &, httplib::Response &res) {
    //    std::string processed_prompt = apply_prompt_template(user_prompt);
    //    std::string response = non_stream_generation(llm_api_inferface, processed_prompt, default_params);
    //    res.set_content(response, "text/plain");
    //});
    //std::string body = user_prompt;
    /*
    auto res = cli.Post("/Generate", body.size(),[](size_t offset, size_t length, DataSink &sink) {
	sink.write(body.data() + offset, length);
	return true; // return 'false' if you want to cancel the request.
    },
    "text/plain"); 
    */
    /*
    auto res = cli.Post("/Generate", [&](const httplib::Request &req, httplib::Response &) {
        //res.set_content(user_prompt, "text/plain");
	std::cout << req.body << "\n";
        //req.body = user_prompt;
    });
    */
    //auto res = cli.Get("/Generate", user_prompt, "text/plain");
    //{"user_prompt": user_prompt,"n_predict": 128}
    std::cout << "res->status: " << res->status << "\n";
    std::cout << "res->body: " << res->body << "\n";
    return 0;
}
