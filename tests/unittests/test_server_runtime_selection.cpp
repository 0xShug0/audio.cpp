// Public legacy handlers reject every explicit slots field before touching state.
// Exercise frontend forwarding too; no test hook in the original runtime is needed.
#include "runtime.h"
#include "engine/framework/io/json.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        minitts::server::ServerConfig config;
        config.backend = engine::core::BackendType::Cpu;
        config.ui_management = true;
        minitts::server::ServerState state(config, std::filesystem::current_path());
        for (const std::string value : {"1", "2", "null", "false", "\"4\""}) {
            minitts::server::HttpRequest request;
            request.method = "POST";
            request.path = "/v1/models/load";
            request.body = "{\"slots\":" + value + "}";
            for (bool forwarded : {false, true}) {
                const auto response = forwarded ? state.forward_to_core(request) : state.handle(request);
                if (response.status != 400 ||
                    response.body.find("slots requires --parallel-jobs") == std::string::npos) {
                    throw std::runtime_error("legacy registration did not reject explicit slots");
                }
            }
        }
        minitts::server::HttpRequest list;
        list.method = "GET";
        list.path = "/v1/models";
        const auto response = state.handle(list);
        if (response.status != 200 ||
            !engine::io::json::parse(response.body).require("data").as_array().empty() ||
            response.body.find("slots") != std::string::npos) {
            throw std::runtime_error("rejected registration changed legacy model state");
        }
        std::cout << "legacy registration and frontend forwarding boundary passed\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
