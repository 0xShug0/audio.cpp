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

        // Live speech needs `input` only for a model that speaks it. The entries
        // are lazy and the streams are never run, so no model loads and the
        // body is never read.
        minitts::server::ServerConfig live_config;
        live_config.backend = engine::core::BackendType::Cpu;
        live_config.ui_enabled = false;
        for (const std::string task : {"tts", "s2s", "vc"}) {
            minitts::server::ServerModelConfig model;
            model.id = task;
            model.path = std::filesystem::temp_directory_path() / ("audiocpp-live-input-" + task);
            model.family = "live_input_fixture";
            model.task = task;
            model.mode = "streaming";
            model.lazy = true;
            live_config.models.push_back(model);
        }
        minitts::server::ServerState live(live_config, std::filesystem::current_path());
        for (const std::string task : {"tts", "s2s", "vc"}) {
            for (const std::string input : {"", "&input="}) {
                std::istream pcm(nullptr);
                minitts::server::HttpRequest request;
                request.method = "POST";
                request.path = "/v1/audio/speech/live";
                request.query = "model=" + task + input;
                request.body_stream = &pcm;
                const auto response = live.handle(request);
                const bool rejected = response.status == 400 &&
                    response.body.find("live speech requires an 'input' query parameter") != std::string::npos;
                if (rejected != (task == "tts") || (!rejected && (response.status != 200 || !response.stream_body))) {
                    throw std::runtime_error("legacy live speech without input: " + task + " got HTTP " +
                                             std::to_string(response.status) + ": " + response.body);
                }
            }
        }
        std::cout << "legacy live speech input requirement passed\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
