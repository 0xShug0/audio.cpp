#pragma once

#ifdef SERVER_STATE_FIXTURE_PARALLEL
    #include "parallel_runtime.h"
    #include "engine/framework/audio/wav_writer.h"

    #include <condition_variable>
    #include <future>
    #include <iostream>
    #include <map>
    #include <sstream>
    #include <tuple>
#else
    #include "runtime.h"
    #include "engine/framework/runtime/registry.h"

    #include <filesystem>
    #include <memory>
    #include <utility>
#endif

#include <chrono>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>

namespace rt = engine::runtime;
namespace srv = minitts::server;

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) { throw std::runtime_error(message); }
}

// lambda for test-supplied model run behavior
using RunFunction = std::function<rt::TaskResult (const rt::TaskRequest &)>;

} // namespace


// Define Mock server, model, and model loaders to use as test fixtures
// when testing ServerState specific code that doesn't require a model
// actually loaded, nor a server started and bound to any addresses
#ifdef SERVER_STATE_FIXTURE_PARALLEL
    using namespace std::chrono_literals;
    using engine::io::json::parse;

    namespace {
    template<class F> bool observe(F f) {
        const auto deadline = std::chrono::steady_clock::now() + 2s;
        while (!f()) {
            if (std::chrono::steady_clock::now() >= deadline) { return false; }
            std::this_thread::yield();
        }
        return true;
    }
    struct Gate {
        std::mutex mutex;
        std::condition_variable cv;
        bool enabled = false, entered = false, released = false;
        void arm() { std::lock_guard<std::mutex> lock(mutex); enabled = true; entered = released = false; }
        void wait() {
            std::unique_lock<std::mutex> lock(mutex);
            if (!enabled) { return; }
            entered = true;
            cv.notify_all();
            // A failing test must still drain its futures and release server state.
            cv.wait_for(lock, 5s, [&] { return released; });
        }
        bool seen() { std::lock_guard<std::mutex> lock(mutex); return entered; }
        void release() { std::lock_guard<std::mutex> lock(mutex); released = true; cv.notify_all(); }
    };
    struct Control {
        Gate load, run, destroy;
        std::atomic<int> loads{0}, destructions{0}, runs{0};
        std::atomic<int> sessions{0}, clones{0};
        std::atomic<bool> fail_load{false}, null_session{false}, fail_prepare{false}, fail_run{false}, fail_clone{false};
        std::atomic<bool> premature_model_destruction{false};
    };
    class Session final : public rt::IOfflineVoiceTaskSession,
                          public rt::IBatchedOfflineVoiceTaskSession,
                          public rt::IStreamingVoiceTaskSession,
                          public rt::IParallelVoiceTaskSessionFactory {
    public:
        Session(std::shared_ptr<Control> c, rt::TaskSpec task, std::string revision, size_t capacity = 4, std::shared_ptr<RunFunction> run = {})
            : control_(std::move(c)), run_(std::move(run)), task_(task), revision_(std::move(revision)), capacity_(capacity) { ++control_->sessions; }
        ~Session() override { --control_->sessions; }
        std::string family() const override { return "lifecycle_fixture"; }
        rt::VoiceTaskKind task_kind() const override { return task_.task; }
        rt::RunMode run_mode() const override { return task_.mode; }
        void prepare(const rt::SessionPreparationRequest &) override {
            if (control_->fail_prepare) { throw std::runtime_error("controlled preparation failure"); }
        }
        size_t parallel_session_capacity() const noexcept override { return capacity_; }
        std::unique_ptr<rt::IVoiceTaskSession> create_parallel_session() const override {
            if (++control_->clones == 2 && control_->fail_clone) {
                throw std::runtime_error("controlled partial pool failure");
            }
            return std::make_unique<Session>(control_, task_, revision_, capacity_, run_);
        }
        rt::TaskResult run(const rt::TaskRequest & request) override {
            if (run_ && *run_) {
                return (*run_)(request);
            } else {
                return run_default(request);
            }
        }

        rt::TaskResult run_default(const rt::TaskRequest & request) {
            control_->run.wait();
            ++control_->runs;
            if (control_->fail_run) { throw std::runtime_error("controlled inference failure"); }
            if (task_.task == rt::VoiceTaskKind::Asr && !request.audio_input) {
                throw std::runtime_error("replacement ASR requires audio input");
            }
            rt::TaskResult result;
            result.text_output = rt::Transcript{revision_, "en"};
            result.audio_output = rt::AudioBuffer{16000, 1, {0.1f, -0.1f}};
            return result;
        }
        std::vector<rt::TaskResult> run_batch(const std::vector<rt::TaskRequest> & requests) override {
            std::vector<rt::TaskResult> results;
            for (const auto & request : requests) { results.push_back(run(request)); }
            return results;
        }
        rt::StreamingPolicy streaming_policy() const override {
            rt::StreamingPolicy policy;
            policy.input = rt::StreamingInputKind::None;
            policy.output = rt::StreamingOutputKind::PullEvents;
            return policy;
        }
        void start_stream(const rt::TaskRequest & request) override { result_ = run(request); emitted_ = false; }
        std::optional<rt::StreamEvent> next_stream_event() override {
            if (emitted_) { return {}; }
            emitted_ = true;
            rt::StreamEvent event;
            event.audio_output = result_.audio_output;
            event.partial_text = result_.text_output;
            event.is_final = true;
            return event;
        }
        void reset() override { emitted_ = false; }
        rt::StreamEvent process_audio_chunk(const rt::AudioChunk &) override { return {}; }
        rt::TaskResult finalize() override { return result_; }
    private:
        std::shared_ptr<Control> control_;
        std::shared_ptr<RunFunction> run_;
        rt::TaskSpec task_;
        std::string revision_;
        size_t capacity_;
        rt::TaskResult result_;
        bool emitted_ = false;
    };
    class Model final : public rt::ILoadedVoiceModel {
    public:
        Model(std::shared_ptr<Control> c, std::string revision, std::shared_ptr<RunFunction> run = {})
             : control_(std::move(c)), run_(std::move(run)), revision_(std::move(revision)) {
            metadata_.family = "lifecycle_fixture";
        }
        ~Model() override {
            control_->destroy.wait();
            if (control_->sessions != 0) { control_->premature_model_destruction = true; }
            ++control_->destructions;
        }
        const rt::ModelMetadata & metadata() const noexcept override { return metadata_; }
        const rt::CapabilitySet & capabilities() const noexcept override { return capabilities_; }
        std::unique_ptr<rt::IVoiceTaskSession> create_task_session(const rt::TaskSpec & task, const rt::SessionOptions & options) const override {
            if (control_->null_session) { return nullptr; }
            const auto cap = options.options.find("fixture_capacity");
            return std::make_unique<Session>(control_, task, revision_,
                cap == options.options.end() ? 4 : std::stoul(cap->second),
                run_);
        }
    private:
        std::shared_ptr<Control> control_;
        std::shared_ptr<RunFunction> run_;
        std::string revision_;
        rt::ModelMetadata metadata_;
        rt::CapabilitySet capabilities_;
    };
    class Loader final : public rt::IVoiceModelLoader {
    public:
        explicit Loader(std::shared_ptr<RunFunction> run = {}): run_(std::move(run)) { }

        std::map<std::string, std::shared_ptr<Control>> controls;
        std::string family() const override { return "lifecycle_fixture"; }
        bool can_load(const rt::ModelLoadRequest &) const override { return true; }
        rt::ModelInspection inspect(const rt::ModelLoadRequest &) const override { return {}; }
        std::unique_ptr<rt::ILoadedVoiceModel> load(const rt::ModelLoadRequest & request) const override {
            const auto c = controls.at(request.model_path.filename().string());
            ++c->loads;
            c->load.wait();
            if (c->fail_load) { throw std::runtime_error("controlled load failure"); }
            return std::make_unique<Model>(c, request.model_path.filename().string(), run_);
        }

    private:
        std::shared_ptr<RunFunction> run_;
    };
    std::string quote(const std::string & text) { return engine::io::json::stringify_string(text); }
    srv::HttpResponse post(srv::ParallelServerState & state, const std::string & path, std::string body, const std::string & origin = "") {
        srv::HttpRequest request;
        request.method = "POST"; request.path = path; request.body = std::move(body);
        request.headers["content-type"] = "application/json";
        if (!origin.empty()) request.headers["origin"] = origin;
        return state.handle(request);
    }
    srv::HttpResponse run(srv::ParallelServerState & state, const std::string & id) {
        return post(state, "/v1/tasks/run", "{\"model\":" + quote(id) + ",\"text\":\"old request\"}");
    }
    void success(const srv::HttpResponse & response) {
        require(response.status == 200, "HTTP " + std::to_string(response.status) + ": " + response.body);
    }
    class Writer final : public srv::HttpStreamWriter {
    public:
        std::string output;
        bool disconnected = false;
        void write(std::string_view s) override {
            if (disconnected) { throw std::runtime_error("client disconnected"); }
            output.append(s);
        }
    };
    }

    namespace minitts::server {
    class ServerRuntimeTestAccess {
    public:
        static void registry(ParallelServerState & state, const std::shared_ptr<Loader> & loader) {
            state.registry_factory_ = [loader] {
                rt::ModelRegistry registry; registry.register_loader(loader); return registry;
            };
            state.config_.ui_management = true;
        }
        static void add(ParallelServerState & state, ServerModelConfig config) {
            auto model = state.make_model(std::move(config));
            state.model_index_.emplace(model->registered_id, state.models_.size());
            state.models_.push_back(std::move(model));
        }
        static auto & model(ParallelServerState & state, const std::string & id) {
            return *state.models_.at(state.model_index_.at(id));
        }
        static ModelSlots::State slots(ParallelServerState & state, const std::string & id) { return model(state, id).busy.state(); }
        static bool resident(ParallelServerState & state, const std::string & id) { return model(state, id).resident(); }
        static bool loaded(ParallelServerState & state, const std::string & id) { return model(state, id).loaded.load(); }
        static auto block_metadata(ParallelServerState & state, const std::string & id) {
            return std::unique_lock<std::shared_mutex>(model(state, id).metadata_mutex);
        }
        static auto block_bulk_publication(ParallelServerState & state) {
            return std::unique_lock<std::mutex>(state.bulk_admission_mutex_);
        }
    };
    }
    using Access = srv::ServerRuntimeTestAccess;

    namespace {
    struct Fixture {
        std::filesystem::path root;
        std::shared_ptr<RunFunction> run_;
        std::shared_ptr<Loader> loader;
        std::unique_ptr<srv::ParallelServerState> state;
        int count;
        Fixture(int slots, int limit = 0, RunFunction run = nullptr, const std::string cors = "*")
             : run_(std::make_shared<RunFunction>(std::move(run))),
               loader(std::make_shared<Loader>(std::move(run_))),
               count(slots) {
            root = std::filesystem::temp_directory_path() /
                ("audiocpp-lifecycle-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            std::filesystem::create_directories(root);
            srv::ServerConfig config; config.backend = engine::core::BackendType::Cpu;
            config.ui_enabled = false; config.max_loaded_models = limit; config.busy_timeout_ms = 0;
            config.cors_origins = cors;
            state = std::make_unique<srv::ParallelServerState>(config, root);
            Access::registry(*state, loader);
        }
        ~Fixture() {
            for (const auto & [name, c] : loader->controls) {
                c->load.release(); c->run.release(); c->destroy.release();
            }
            state.reset();
            std::error_code ec; std::filesystem::remove_all(root, ec);
        }
        std::shared_ptr<Control> asset(const std::string & name) {
            std::ofstream(root / name) << "controlled fixture";
            auto c = std::make_shared<Control>(); loader->controls.emplace(name, c); return c;
        }
        std::shared_ptr<Control> add(const std::string & id, const std::string & mode = "offline") {
            auto c = asset(id);
            srv::ServerModelConfig model;
            model.id = id; model.path = root / id; model.family = loader->family();
            model.slots = count; model.lazy = true; model.mode = mode;
            Access::add(*state, model); return c;
        }
        auto queued(const std::string & id, int n, bool management = false) {
            return observe([&] { const auto s = Access::slots(*state, id);
                return (management ? s.waiting_management : s.waiting_requests) == n; });
        }
        srv::HttpResponse replace(const std::string & id, const std::string & asset, const std::string & task = "tts") {
            return post(*state, "/v1/models/load", "{\"id\":" + quote(id) + ",\"path\":" +
                quote((root / asset).generic_string()) + ",\"family\":\"lifecycle_fixture\",\"task\":" +
                quote(task) + ",\"slots\":" + std::to_string(count) + "}");
        }
    };

    } // namespace
#else // Non-parallel, ServerState fixture
    namespace {

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
#endif
