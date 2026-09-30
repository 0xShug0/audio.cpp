// Run the actual server handlers with tiny controlled loaders. Barriers force
// lifecycle overlaps without GPU timing, sleeps in production, or large models.
#include "runtime.h"
#include "engine/framework/audio/wav_writer.h"
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <sstream>

using namespace std::chrono_literals;
namespace rt = engine::runtime;
namespace srv = minitts::server;
using engine::io::json::parse;

namespace {
void require(bool value, const std::string & message) {
    if (!value) { throw std::runtime_error(message); }
}
void concurrent_json_parsing() {
    std::promise<void> start;
    auto ready = start.get_future().share();
    std::vector<std::future<void>> workers;
    for (int worker = 0; worker < 8; ++worker) {
        workers.push_back(std::async(std::launch::async, [ready, worker] {
            ready.wait();
            for (int wave = 0; wave < 200; ++wave) {
                const std::string label = std::to_string(worker) + ":" + std::to_string(wave);
                const auto value = parse("{\"text\":\"" + label + "\",\"requests\":[null,true,1.5,{\"text\":\"hello\"}]}");
                require(value.require("text").as_string() == label &&
                        value.require("requests").as_array().size() == 4,
                        "concurrent JSON parse mixed request contents");
                require(parse(engine::io::json::stringify(value)).require("text").as_string() == label,
                        "concurrent JSON round-trip changed request contents");
                bool rejected = false;
                try { (void)parse("{\"text\":"); }
                catch (const std::runtime_error & error) {
                    rejected = std::string(error.what()).find("failed to parse json at byte ") == 0;
                }
                require(rejected, "concurrent invalid JSON lost its own error position");
            }
        }));
    }
    start.set_value();
    for (auto & worker : workers) { worker.get(); }
}
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
};
class Session final : public rt::IOfflineVoiceTaskSession,
                      public rt::IBatchedOfflineVoiceTaskSession,
                      public rt::IStreamingVoiceTaskSession,
                      public rt::IParallelVoiceTaskSessionFactory {
public:
    Session(std::shared_ptr<Control> c, rt::TaskSpec task, std::string revision)
        : control_(std::move(c)), task_(task), revision_(std::move(revision)) {}
    std::string family() const override { return "lifecycle_fixture"; }
    rt::VoiceTaskKind task_kind() const override { return task_.task; }
    rt::RunMode run_mode() const override { return task_.mode; }
    void prepare(const rt::SessionPreparationRequest &) override {}
    size_t parallel_session_capacity() const noexcept override { return 4; }
    std::unique_ptr<rt::IVoiceTaskSession> create_parallel_session() const override {
        return std::make_unique<Session>(control_, task_, revision_);
    }
    rt::TaskResult run(const rt::TaskRequest & request) override {
        control_->run.wait();
        ++control_->runs;
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
    rt::TaskSpec task_;
    std::string revision_;
    rt::TaskResult result_;
    bool emitted_ = false;
};
class Model final : public rt::ILoadedVoiceModel {
public:
    Model(std::shared_ptr<Control> c, std::string revision) : control_(std::move(c)), revision_(std::move(revision)) {
        metadata_.family = "lifecycle_fixture";
    }
    ~Model() override { control_->destroy.wait(); ++control_->destructions; }
    const rt::ModelMetadata & metadata() const noexcept override { return metadata_; }
    const rt::CapabilitySet & capabilities() const noexcept override { return capabilities_; }
    std::unique_ptr<rt::IVoiceTaskSession> create_task_session(const rt::TaskSpec & task, const rt::SessionOptions &) const override {
        return std::make_unique<Session>(control_, task, revision_);
    }
private:
    std::shared_ptr<Control> control_;
    std::string revision_;
    rt::ModelMetadata metadata_;
    rt::CapabilitySet capabilities_;
};
class Loader final : public rt::IVoiceModelLoader {
public:
    std::map<std::string, std::shared_ptr<Control>> controls;
    std::string family() const override { return "lifecycle_fixture"; }
    bool can_load(const rt::ModelLoadRequest &) const override { return true; }
    rt::ModelInspection inspect(const rt::ModelLoadRequest &) const override { return {}; }
    std::unique_ptr<rt::ILoadedVoiceModel> load(const rt::ModelLoadRequest & request) const override {
        const auto c = controls.at(request.model_path.filename().string());
        ++c->loads;
        c->load.wait();
        return std::make_unique<Model>(c, request.model_path.filename().string());
    }
};
std::string quote(const std::string & text) { return engine::io::json::stringify_string(text); }
srv::HttpResponse post(srv::ServerState & state, const std::string & path, std::string body) {
    srv::HttpRequest request;
    request.method = "POST"; request.path = path; request.body = std::move(body);
    request.headers["content-type"] = "application/json";
    return state.handle(request);
}
srv::HttpResponse run(srv::ServerState & state, const std::string & id) {
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
    static void registry(ServerState & state, const std::shared_ptr<Loader> & loader) {
        state.registry_factory_ = [loader] {
            rt::ModelRegistry registry; registry.register_loader(loader); return registry;
        };
        state.config_.ui_management = true;
    }
    static void add(ServerState & state, ServerModelConfig config) {
        auto model = state.make_model(std::move(config));
        state.model_index_.emplace(model->registered_id, state.models_.size());
        state.models_.push_back(std::move(model));
    }
    static auto & model(ServerState & state, const std::string & id) {
        return *state.models_.at(state.model_index_.at(id));
    }
    static ModelSlots::State slots(ServerState & state, const std::string & id) { return model(state, id).busy.state(); }
    static bool resident(ServerState & state, const std::string & id) { return model(state, id).resident(); }
    static bool loaded(ServerState & state, const std::string & id) { return model(state, id).loaded.load(); }
    static auto block_metadata(ServerState & state, const std::string & id) {
        return std::unique_lock<std::shared_mutex>(model(state, id).metadata_mutex);
    }
};
}
using Access = srv::ServerRuntimeTestAccess;

namespace {
struct Fixture {
    std::filesystem::path root;
    std::shared_ptr<Loader> loader = std::make_shared<Loader>();
    std::unique_ptr<srv::ServerState> state;
    int count;
    Fixture(int slots, int limit = 0) : count(slots) {
        root = std::filesystem::temp_directory_path() /
            ("audiocpp-lifecycle-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        srv::ServerConfig config; config.backend = engine::core::BackendType::Cpu;
        config.ui_enabled = false; config.max_loaded_models = limit; config.busy_timeout_ms = 0;
        state = std::make_unique<srv::ServerState>(config, root);
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

void older_work_before_management(int count, bool reconfigure) {
    Fixture f(count); auto c = f.add("shared"); f.asset("replacement");
    success(run(*f.state, "shared"));
    c->run.arm();
    std::vector<std::future<srv::HttpResponse>> active;
    for (int i = 0; i < count; ++i) { active.push_back(std::async(std::launch::async, [&] { return run(*f.state, "shared"); })); }
    require(observe([&] { return Access::slots(*f.state, "shared").active == count; }), "active work not established");
    auto old = std::async(std::launch::async, [&] { return run(*f.state, "shared"); });
    const bool old_queued = f.queued("shared", 1);
    auto manager = std::async(std::launch::async, [&] {
        return reconfigure ? f.replace("shared", "replacement", "asr") :
            post(*f.state, "/v1/models/unload", "{\"id\":\"shared\"}");
    });
    const bool manager_queued = f.queued("shared", 1, true);
    c->run.release();
    for (auto & a : active) { success(a.get()); }
    const auto result = old.get(); const auto managed = manager.get();
    require(old_queued && manager_queued, "ordered overlaps not established");
    success(result); success(managed);
    require(result.body.find("shared") != std::string::npos, "older request used replacement state");
    require(c->loads == 1, "queued request reloaded the old model");
    require(Access::resident(*f.state, "shared") == reconfigure, "incorrect final residency");
    const auto s = Access::slots(*f.state, "shared");
    require(s.active == 0 && s.waiting_requests == 0 && s.waiting_management == 0, "ordered drain leaked leases");
}

void bulk_releases_idle_first(int count, bool hold_metadata) {
    Fixture f(count); auto slow = f.add("slow"); auto idle = f.add("idle");
    success(run(*f.state, "idle"));
    slow->load.arm();
    auto loading = std::async(std::launch::async, [&] { return run(*f.state, "slow"); });
    require(observe([&] { return slow->load.seen(); }), "slow initial load did not start");
    std::optional<std::unique_lock<std::shared_mutex>> metadata;
    if (hold_metadata) { metadata.emplace(Access::block_metadata(*f.state, "slow")); }
    auto bulk = std::async(std::launch::async, [&] {
        return post(*f.state, "/v1/tasks/unload_all_models", "{}");
    });
    const bool idle_freed = observe([&] { return idle->destructions == 1; });
    const bool older_loading = loading.wait_for(0ms) == std::future_status::timeout;
    // A full status response may wait for this model's metadata, but must not
    // hold the global registration lock while doing so.
    std::future<srv::HttpResponse> listing;
    std::promise<void> listing_started;
    bool listing_blocked = true;
    if (hold_metadata) {
        auto started = listing_started.get_future();
        listing = std::async(std::launch::async, [&] {
            srv::HttpRequest request; request.method = "GET"; request.path = "/v1/models";
            listing_started.set_value();
            return f.state->handle(request);
        });
        started.wait();
        listing_blocked = listing.wait_for(20ms) == std::future_status::timeout;
    }
    srv::HttpRequest voices; voices.method = "GET"; voices.path = "/v1/audio/voices"; voices.query = "model=idle";
    auto independent = std::async(std::launch::async, [&] { return f.state->handle(voices); });
    const bool unrelated_progress = independent.wait_for(200ms) == std::future_status::ready;
    metadata.reset(); slow->load.release();
    if (listing.valid()) { success(listing.get()); }
    success(independent.get()); success(loading.get()); success(bulk.get());
    require(idle_freed && older_loading, "bulk retained idle model behind a blocked initial load");
    require(listing_blocked && unrelated_progress,
            "bulk/status held the global registry while waiting for model metadata");
    require(!Access::resident(*f.state, "idle") && !Access::resident(*f.state, "slow"), "bulk failed to drain residents");
}

void retiring_resident(int count, int limit) {
    Fixture f(count, limit); auto retiring = f.add("retiring"); auto keeper = f.add("keeper"); auto incoming = f.add("incoming");
    success(run(*f.state, "retiring"));
    if (limit == 2) { success(run(*f.state, "keeper")); }
    retiring->destroy.arm();
    auto unload = std::async(std::launch::async, [&] { return post(*f.state, "/v1/models/unload", "{\"id\":\"retiring\"}"); });
    require(observe([&] { return retiring->destroy.seen(); }), "retirement did not start");
    const bool released = !Access::resident(*f.state, "retiring") && !Access::loaded(*f.state, "retiring");
    auto target = std::async(std::launch::async, [&] { return run(*f.state, "incoming"); });
    const bool concurrent = target.wait_for(500ms) == std::future_status::ready;
    retiring->destroy.release();
    success(target.get()); success(unload.get());
    require(released && concurrent, "retiring model blocked incoming load");
    if (limit == 2) {
        require(Access::resident(*f.state, "keeper") && keeper->loads == 1 && keeper->destructions == 0,
                "retiring model caused unnecessary eviction");
    }
    require(incoming->loads == 1, "incoming model retried instead of loading once");
}

void overlapping_managers(int count) {
    Fixture f(count); auto c = f.add("shared"); f.asset("first"); f.asset("last");
    c->run.arm();
    auto active = std::async(std::launch::async, [&] { return run(*f.state, "shared"); });
    require(observe([&] { return c->run.seen(); }), "manager test did not start inference");
    auto first = std::async(std::launch::async, [&] { return f.replace("shared", "first"); });
    const bool first_queued = f.queued("shared", 1, true);
    auto last = std::async(std::launch::async, [&] { return f.replace("shared", "last"); });
    const bool last_queued = f.queued("shared", 2, true);
    c->run.release();
    success(active.get()); success(first.get()); success(last.get());
    require(first_queued && last_queued, "management order was not established");
    const auto final = run(*f.state, "shared"); success(final);
    require(final.body.find("last") != std::string::npos, "later manager lost to earlier one");
}

void deferred_stream_ownership(int count, bool disconnect) {
    Fixture f(count); auto c = f.add("stream", "streaming"); f.asset("replacement");
    auto response = post(*f.state, "/v1/audio/speech", "{\"model\":\"stream\",\"input\":\"hello\",\"stream\":true,\"response_format\":\"pcm\"}");
    success(response); require(bool(response.stream_body), "missing streaming callback");
    auto manager = std::async(std::launch::async, [&] { return f.replace("stream", "replacement"); });
    const bool queued = f.queued("stream", 1, true);
    const bool protected_state = manager.wait_for(0ms) == std::future_status::timeout;
    Writer writer; writer.disconnected = disconnect;
    bool threw = false;
    try { response.stream_body(writer); } catch (const std::runtime_error &) { threw = true; }
    require(threw == disconnect, "unexpected stream callback result");
    response.stream_body = {}; // transport drops the callback at exchange end
    success(manager.get());
    require(queued && protected_state, "management replaced state before the deferred stream ran");
    require(c->runs == 1 && c->loads == 1, "stream used a different configuration");
    if (!disconnect) { require(writer.output.find("speech.audio.done") != std::string::npos, "stream returned replacement output"); }
    const auto s = Access::slots(*f.state, "stream");
    require(s.active == 0 && s.waiting_management == 0, "stream completion/disconnect leaked ownership");
}

void deferred_batch_ownership(int count, bool generic = false) {
    Fixture f(count); auto c = f.add("batch"); f.asset("replacement");
    const auto wav_path = f.root / "input.wav";
    engine::audio::write_pcm16_wav(wav_path, 16000, 1, {0.1f, -0.1f});
    std::ifstream wav(wav_path, std::ios::binary);
    const std::string audio{std::istreambuf_iterator<char>(wav), std::istreambuf_iterator<char>()};
    srv::HttpRequest request;
    request.method = "POST"; request.path = "/v1/batches/transcriptions";
    request.headers["content-type"] = "multipart/form-data; boundary=test_boundary";
    request.body = "--test_boundary\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nbatch\r\n";
    for (int i = 0; i < 2; ++i) {
        request.body += "--test_boundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"input.wav\"\r\n"
                        "Content-Type: audio/wav\r\n\r\n" + audio + "\r\n";
    }
    request.body += "--test_boundary--\r\n";
    if (generic) {
        request.path = "/v1/tasks/batch";
        request.headers["content-type"] = "application/json";
        request.body = "{\"model\":\"batch\",\"requests\":[{\"text\":\"first\"},{\"text\":\"second\"}]}";
    }
    auto response = f.state->handle(request); success(response);
    require(bool(response.stream_body), "missing native batch callback");
    auto manager = std::async(std::launch::async, [&] { return f.replace("batch", "replacement"); });
    const bool queued = f.queued("batch", 1, true);
    const bool protected_state = manager.wait_for(0ms) == std::future_status::timeout;
    Writer writer; response.stream_body(writer); response.stream_body = {};
    success(manager.get());
    require(queued && protected_state && c->loads == 1 && c->runs == 2,
            "batch validation/execution crossed a configuration transition");
    require(writer.output.find(generic ? "task.batch.done" : "batch.transcription.done") != std::string::npos &&
            writer.output.find("\"text\":\"batch\"") != std::string::npos,
            "native batch returned replacement output");
    require(Access::slots(*f.state, "batch").active == 0, "native batch leaked a lease");
}
void generic_batches_hold_independent_slots(int count) {
    Fixture f(count); auto c = f.add("batch"); f.asset("replacement");
    const std::string body = "{\"model\":\"batch\",\"requests\":[{\"text\":\"first\"},{\"text\":\"second\"}]}";
    auto first = post(*f.state, "/v1/tasks/batch", body);
    auto second = post(*f.state, "/v1/tasks/batch", body);
    success(first); success(second);
    require(Access::slots(*f.state, "batch").active == 2, "generic batches did not lease independent slots");
    auto manager = std::async(std::launch::async, [&] { return f.replace("batch", "replacement"); });
    const bool queued = f.queued("batch", 1, true);
    Writer one, two;
    first.stream_body(one); first.stream_body = {};
    const bool retained = Access::slots(*f.state, "batch").active == 1 &&
        manager.wait_for(0ms) == std::future_status::timeout;
    second.stream_body(two); second.stream_body = {};
    success(manager.get());
    require(queued && retained && c->runs == 4, "generic batch management did not wait for both callbacks");
    require(one.output.find("task.batch.done") != std::string::npos &&
            two.output.find("task.batch.done") != std::string::npos &&
            one.output.find("\"text\":\"batch\"") != std::string::npos &&
            two.output.find("\"text\":\"batch\"") != std::string::npos,
            "generic batches used replacement state");
    require(Access::slots(*f.state, "batch").active == 0, "generic batch callbacks stranded ownership");
}

void generic_batch_legacy_and_invalid_input() {
    Fixture f(1); f.add("batch");
    auto response = post(*f.state, "/v1/tasks/batch",
        "{\"model\":\"batch\",\"requests\":[{\"text\":\"hello\"}]}");
    success(response); Writer writer; response.stream_body(writer); response.stream_body = {};
    require(writer.output.find("task.batch.done") != std::string::npos, "legacy generic batch failed");
    require(post(*f.state, "/v1/tasks/batch", "{\"model\":\"batch\",\"requests\":[]}").status == 400,
            "empty generic batch was accepted");
    require(Access::slots(*f.state, "batch").active == 0, "legacy generic batch leaked ownership");
}

void rejected_preparation_releases_lease(int count) {
    Fixture f(count); f.add("shared");
    bool rejected = false;
    try {
        const auto invalid = post(*f.state, "/v1/tasks/run",
            "{\"model\":\"shared\",\"audio\":" + quote((f.root / "missing.wav").generic_string()) + "}");
        rejected = invalid.status >= 400;
    } catch (const std::runtime_error &) {
        // The transport converts general handler exceptions into HTTP errors.
        rejected = true;
    }
    require(rejected, "missing input was accepted");
    require(Access::slots(*f.state, "shared").active == 0, "failed preparation leaked a bound lease");
    success(run(*f.state, "shared"));
}
}

int main() {
    try {
        concurrent_json_parsing();
        generic_batch_legacy_and_invalid_input();
        for (int count : {2, 4}) {
            older_work_before_management(count, false);
            older_work_before_management(count, true);
            overlapping_managers(count);
            bulk_releases_idle_first(count, false);
            bulk_releases_idle_first(count, true);
            retiring_resident(count, 1);
            retiring_resident(count, 2);
            deferred_stream_ownership(count, false);
            deferred_stream_ownership(count, true);
            deferred_batch_ownership(count);
            deferred_batch_ownership(count, true);
            generic_batches_hold_independent_slots(count);
            rejected_preparation_releases_lease(count);
            std::cout << "PASS real handlers: slots=" << count
                      << " queued unload/reconfiguration, manager order, bulk release, global lock,"
                         " resident limits 1/2, deferred stream/disconnect\n";
        }
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
