#include "engine/framework/runtime/session_pool.h"
#include "model_slots.h"

#include <future>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace engine::runtime;
using minitts::server::ModelSlots;

void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}
template<class Fn> void rejects(Fn fn) {
    bool rejected = false;
    try { fn(); } catch (const std::exception &) { rejected = true; }
    require(rejected, "invalid pool was accepted");
}

class Legacy : public IOfflineVoiceTaskSession {
public:
    std::string family() const override { return "test-family"; }
    VoiceTaskKind task_kind() const override { return VoiceTaskKind::Tts; }
    RunMode run_mode() const override { return RunMode::Offline; }
    void prepare(const SessionPreparationRequest &) override {}
    TaskResult run(const TaskRequest &) override { return {}; }
};

enum class Fault { None, Null, Throw, Family, Task, Mode, Interface };
struct State {
    std::shared_ptr<const int> weights = std::make_shared<const int>(42);
    int clones = 0;
    size_t capacity = 4;
    Fault fault = Fault::None;
    std::vector<int> destroyed;
};

class Offline final : public IOfflineVoiceTaskSession,
                      public IBatchedOfflineVoiceTaskSession,
                      public IParallelVoiceTaskSessionFactory {
public:
    explicit Offline(std::shared_ptr<State> state, int id = 0) : state(std::move(state)), id(id) {}
    ~Offline() override { state->destroyed.push_back(id); }
    std::string family() const override { return family_name; }
    VoiceTaskKind task_kind() const override { return task; }
    RunMode run_mode() const override { return mode; }
    void prepare(const SessionPreparationRequest &) override {}
    size_t parallel_session_capacity() const noexcept override { return state->capacity; }
    std::unique_ptr<IVoiceTaskSession> create_parallel_session() const override {
        const int next = ++state->clones;
        auto result = std::make_unique<Offline>(state, next);
        // Fail after one successful clone to exercise partial construction.
        if (next == 2) {
            switch (state->fault) {
                case Fault::Null: return nullptr;
                case Fault::Throw: throw std::runtime_error("allocation failed");
                case Fault::Family: result->family_name = "wrong-family"; break;
                case Fault::Task: result->task = VoiceTaskKind::Asr; break;
                case Fault::Mode: result->mode = RunMode::Streaming; break;
                case Fault::Interface: return std::make_unique<Legacy>(); // no batch interface
                case Fault::None: break;
            }
        }
        return result;
    }
    TaskResult run(const TaskRequest & request) override {
        TaskResult out;
        out.text_output = Transcript{request.text_input->text + ":" + std::to_string(++calls), ""};
        return out;
    }
    std::vector<TaskResult> run_batch(const std::vector<TaskRequest> & requests) override {
        std::vector<TaskResult> results;
        for (const auto & request : requests) { results.push_back(run(request)); }
        return results;
    }
    std::shared_ptr<State> state;
    int id;
    int calls = 0;
    std::string family_name = "test-family";
    VoiceTaskKind task = VoiceTaskKind::Tts;
    RunMode mode = RunMode::Offline;
};

class Streaming final : public IStreamingVoiceTaskSession, public IParallelVoiceTaskSessionFactory {
public:
    std::string family() const override { return "different-test-family"; }
    VoiceTaskKind task_kind() const override { return VoiceTaskKind::Asr; }
    RunMode run_mode() const override { return RunMode::Streaming; }
    void prepare(const SessionPreparationRequest &) override {}
    size_t parallel_session_capacity() const noexcept override { return 4; }
    std::unique_ptr<IVoiceTaskSession> create_parallel_session() const override {
        return std::make_unique<Streaming>();
    }
    void reset() override { chunks = 0; }
    StreamEvent process_audio_chunk(const AudioChunk &) override { ++chunks; return {}; }
    TaskResult finalize() override {
        TaskResult result;
        result.text_output = Transcript{std::to_string(chunks), ""};
        return result;
    }
    int chunks = 0;
};

void test_pool_validation_and_lifetime() {
    VoiceTaskSessionPool legacy(std::make_unique<Legacy>(), 1);
    require(legacy.capacity() == 1 && legacy.size() == 1, "legacy default changed");
    rejects([] { VoiceTaskSessionPool pool(std::make_unique<Legacy>(), 2); });
    rejects([] { VoiceTaskSessionPool pool(nullptr, 1); });
    for (size_t count : {size_t{0}, kMaxParallelSessions + 1}) {
        rejects([&] { VoiceTaskSessionPool pool(std::make_unique<Legacy>(), count); });
    }
    rejects([&] { legacy.at(1); });
    for (auto fault : {Fault::Null, Fault::Throw, Fault::Family, Fault::Task, Fault::Mode, Fault::Interface}) {
        auto state = std::make_shared<State>(); state->fault = fault;
        rejects([&] { VoiceTaskSessionPool pool(std::make_unique<Offline>(state), 4); });
        require(state->clones == 2 && state->destroyed.back() == 0, "failed clone outlived primary");
        // A failure must not poison the next load.
        state->fault = Fault::None;
        { VoiceTaskSessionPool retry(std::make_unique<Offline>(state), 4); }
        require(state->destroyed.back() == 0, "successful pool destroyed primary first");
    }
    for (size_t capacity : {size_t{0}, size_t{1}, size_t{2}}) {
        auto state = std::make_shared<State>(); state->capacity = capacity;
        rejects([&] { VoiceTaskSessionPool pool(std::make_unique<Offline>(state), 3); });
        require(state->clones == 0, "capacity rejection allocated clones");
    }
}

void test_offline_and_native_batches() {
    auto state = std::make_shared<State>();
    VoiceTaskSessionPool pool(std::make_unique<Offline>(state), 4);
    ModelSlots slots; slots.configure(4);
    std::promise<void> release;
    auto gate = release.get_future().share();
    std::vector<std::future<bool>> jobs;
    for (int i = 0; i < 4; ++i) {
        // Lease before launching to guarantee all four sessions participate.
        auto lease = slots.acquire_run(1000, "test");
        jobs.push_back(std::async(std::launch::async, [&, i, lease = std::move(lease)] {
            gate.wait();
            auto & session = dynamic_cast<Offline &>(pool.at(lease.slot()));
            require(session.state->weights.get() == state->weights.get(), "weights were duplicated");
            TaskRequest request; request.text_input = Transcript{std::to_string(i), ""};
            auto results = pool.get<IBatchedOfflineVoiceTaskSession>(lease.slot())->run_batch({request, request});
            return results[0].text_output->text == std::to_string(i) + ":1" &&
                   results[1].text_output->text == std::to_string(i) + ":2";
        }));
    }
    require(slots.active() == 4 && !slots.try_acquire(), "batch lease did not protect session");
    release.set_value();
    for (auto & job : jobs) { require(job.get(), "cross-slot batch state contamination"); }
    require(slots.active() == 0, "finished batch retained its lease");
}

void test_streaming_state_and_error_release() {
    VoiceTaskSessionPool pool(std::make_unique<Streaming>(), 2);
    ModelSlots slots; slots.configure(2);
    {
        auto first = slots.acquire_run(1000, "stream");
        auto second = slots.acquire_run(1000, "stream");
        auto * a = pool.get<IStreamingVoiceTaskSession>(first.slot());
        auto * b = pool.get<IStreamingVoiceTaskSession>(second.slot());
        a->start_stream({}); b->start_stream({});
        a->process_audio_chunk({}); b->process_audio_chunk({}); a->process_audio_chunk({});
        require(a->finish_stream().text_output->text == "2", "first stream state contaminated");
        require(b->finish_stream().text_output->text == "1", "second stream state contaminated");
        require(!slots.try_acquire(), "stream lease did not prevent unload");
    }
    try {
        auto lease = slots.acquire_run(1000, "stream");
        auto * stream = pool.get<IStreamingVoiceTaskSession>(lease.slot());
        stream->start_stream({}); stream->process_audio_chunk({});
        throw std::runtime_error("client disconnected");
    } catch (const std::runtime_error &) {}
    auto lease = slots.acquire_run(1000, "stream");
    auto * stream = pool.get<IStreamingVoiceTaskSession>(lease.slot());
    stream->start_stream({});
    require(stream->finish_stream().text_output->text == "0", "next stream retained failed stream state");
}
} // namespace

int main() {
    try {
        test_pool_validation_and_lifetime();
        test_offline_and_native_batches();
        test_streaming_state_and_error_release();
        std::cout << "PASS session pool validation, rollback, lifetime, shared weights, batch and stream isolation\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
