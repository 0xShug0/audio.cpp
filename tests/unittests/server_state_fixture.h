#pragma once

// Define Mock server, model, and model loaders to use as test fixtures
// when testing ServerState specific code that doesn't require a model
// actually loaded, nor a server started and bound to any addresses

#include "runtime.h"
#include "engine/framework/runtime/registry.h"

#include <chrono>
#include <filesystem>
#include <functional>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace rt = engine::runtime;
namespace srv = minitts::server;


namespace {

// lambda for test-supplied model run behavior
using RunFunction = std::function<rt::TaskResult (const rt::TaskRequest &)>;

inline constexpr const char * kModelFamily = "test-server-fixture";

// Minimal mock Offline Session
class MockSession final : public rt::IOfflineVoiceTaskSession {
public:
    MockSession(const rt::TaskSpec & task, std::shared_ptr<RunFunction> run)
        : task_(task), run_(std::move(run)) {}

    std::string family() const override { return kModelFamily; }
    rt::VoiceTaskKind task_kind() const override { return task_.task; }
    rt::RunMode run_mode() const override { return task_.mode; }
    void prepare(const rt::SessionPreparationRequest &) override {}
    rt::TaskResult run(const rt::TaskRequest & request) override { return (*run_)(request); }

private:
    rt::TaskSpec task_;
    std::shared_ptr<RunFunction> run_;
};

// Minimal mock offline model
class MockModel final : public rt::ILoadedVoiceModel {
public:
    explicit MockModel(std::shared_ptr<RunFunction> run)
        : run_(std::move(run)) {
        metadata_.family = kModelFamily;
    }

    const rt::ModelMetadata & metadata() const noexcept override { return metadata_; }
    const rt::CapabilitySet & capabilities() const noexcept override { return capabilities_; }
    std::unique_ptr<rt::IVoiceTaskSession> create_task_session(const rt::TaskSpec & task, const rt::SessionOptions &) const {
        return std::make_unique<MockSession>(task, run_);
    }

private:
    std::shared_ptr<RunFunction> run_;
    rt::ModelMetadata metadata_;
    rt::CapabilitySet capabilities_;
};

// Mock model loader so ServerState can "load" our mock model without any actual weights
class MockLoader final : public rt::IVoiceModelLoader {
public:
    explicit MockLoader(std::shared_ptr<RunFunction> run) : run_(std::move(run)) {}

    std::string family() const override { return kModelFamily; }
    bool can_load(const rt::ModelLoadRequest &) const override { return true; }
    rt::ModelInspection inspect(const rt::ModelLoadRequest &) const override { return {}; }
    std::unique_ptr<rt::ILoadedVoiceModel> load(const rt::ModelLoadRequest &) const override {
        return std::make_unique<MockModel>(run_);
    }

private:
    std::shared_ptr<RunFunction> run_;
};
} // namespace

// Test-only friend class of runtime::ServerState, needed
// to modify its model loader to use the mock loader.
// Declared in the minitts::server namespace originally
namespace minitts::server {

class ServerRuntimeTestAccess {
public:
    static void set_registry_factory(ServerState & state, const std::shared_ptr<MockLoader> & loader) {
        state.registry_factory_ = [loader] {
            rt::ModelRegistry registry;
            registry.register_loader(loader);
            return registry;
        };
    }

    static void add(ServerState & state, ServerModelConfig config) {
        auto model = state.make_model(std::move(config));
        state.model_index_.emplace(model->config.id, state.models_.size());
        state.models_.push_back(std::move(model));
    }
};
} // namespace minitts::server
using Access = srv::ServerRuntimeTestAccess;

namespace {
// Test fixture: test a real ServerState object using a lazily loaded MockModel, so
// no weights are loaded, no actual server is launched.
class Fixture {
public:
    explicit Fixture(RunFunction run, std::string cors = "*")
        : root_(make_temp_root()),
          run_(std::make_shared<RunFunction>(std::move(run))),
          loader_(std::make_shared<MockLoader>(run_)) {
        srv::ServerConfig config;
        config.backend = engine::core::BackendType::Cpu;
        config.ui_enabled = false;
        config.ui_management = true; // needed to ensure the model loader will use our mock model
        config.idle_unload_ms = 0;
        config.max_loaded_models = 1;
        config.min_free_memory_mb = 0;
        config.cors_origins = cors;
        state_ = std::make_unique<srv::ServerState>(std::move(config), root_);
        Access::set_registry_factory(*state_, loader_);
    }
    Fixture(const Fixture &) = delete;
    ~Fixture() {
        // Destroy ServerState before cleaning up temp directory
        state_.reset();
        std::error_code ec;
        std::filesystem::remove_all(root_, ec);
    }
    Fixture & operator= (const Fixture &) = delete;

    void add(const std::string & id, const std::string & mode = "offline") {
        // ServerState's load_model needs a fake file in place when looking up model
        // contracts for the fake model
        std::ofstream(root_ / id) << "controlled fixture";
        // Now create and add the model, already "loaded"
        srv::ServerModelConfig model;
        model.id = id;
        model.path = root_ / id;
        model.family = loader_->family();
        model.mode = mode;
        model.lazy = true;
        Access::add(*state_, std::move(model));
    }

    // Test convenience methods
    srv::HttpResponse handle(const srv::HttpRequest & request) {
        return state_->handle(request);
    }

    srv::HttpResponse post(const std::string & path, std::string body, const std::string origin = "http://server-state-test") {
        srv::HttpRequest request;
        request.method = "POST";
        request.path = path;
        request.body = std::move(body);
        request.headers["content-type"] = "application/json";
        request.headers["origin"] = origin;
        return handle(request);
    }

private:
    static std::filesystem::path make_temp_root() {
        const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
        auto root = std::filesystem::temp_directory_path() /
            ("audio-cpp-server-state-test-" + std::to_string(ticks));
        std::filesystem::create_directories(root);
        return root;
    }

    std::filesystem::path root_;
    std::shared_ptr<RunFunction> run_;
    std::shared_ptr<MockLoader> loader_;
    std::unique_ptr<srv::ServerState> state_;
};

} // namespace
