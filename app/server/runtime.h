#pragma once

#include "config.h"
#include "http.h"

#include "engine/framework/io/json.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session.h"

#include <filesystem>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace minitts::server {

class ServerState final : public IHttpHandler {
public:
    ServerState(ServerConfig config, std::filesystem::path request_base);

    HttpResponse handle(const HttpRequest & request) override;

private:
    struct LoadedModel {
        ServerModelConfig config;
        engine::runtime::TaskSpec task;
        std::unique_ptr<engine::runtime::ILoadedVoiceModel> model;
        std::unique_ptr<engine::runtime::IVoiceTaskSession> session;
        engine::runtime::IOfflineVoiceTaskSession * offline = nullptr;
        std::mutex mutex;
    };

    void load_models();
    void ensure_model_loaded_locked(LoadedModel & model);
    LoadedModel & require_model(const engine::io::json::Value & body);
    struct TimedTaskResult;
    TimedTaskResult run_model(LoadedModel & model, const engine::runtime::TaskRequest & request);
    HttpResponse handle_speech(const std::string & body_text);
    HttpResponse handle_transcription(const std::string & body_text);
    HttpResponse handle_generic_run(const std::string & body_text);
    std::string models_json() const;

    ServerConfig config_;
    std::filesystem::path request_base_;
    std::vector<std::unique_ptr<LoadedModel>> models_;
    std::unordered_map<std::string, size_t> model_index_;
};

}  // namespace minitts::server
