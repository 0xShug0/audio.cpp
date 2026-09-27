#include "engine/framework/runtime/session_pool.h"
#include "model_slots.h"
#include "audited_model_slots.h"

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

void test_loaded_model_factory() {
    int created = 0;
    VoiceTaskSessionPool pool(std::make_unique<Legacy>(), 2, [&] {
        ++created;
        return std::make_unique<Legacy>();
    }, 2);
    require(created == 1 && pool.size() == 2 && pool.capacity() == 2,
            "loaded model factory did not honor its audited capacity");
    require(&pool.at(0) != &pool.at(1), "loaded model fallback reused mutable session state");
    // An explicit adapter's backend/mode rejection must not be bypassed.
    auto state = std::make_shared<State>();
    state->capacity = 1;
    rejects([&] {
        VoiceTaskSessionPool rejected(std::make_unique<Offline>(state), 2, [&] {
            ++created;
            return std::make_unique<Legacy>();
        }, 16);
    });
    require(created == 1, "fallback bypassed explicit capacity restriction");
    rejects([] {
        VoiceTaskSessionPool null_clone(std::make_unique<Legacy>(), 2,
            [] { return std::unique_ptr<IVoiceTaskSession>{}; }, 2);
    });
    rejects([] {
        VoiceTaskSessionPool wrong_interface(std::make_unique<Legacy>(), 2,
            [] { return std::make_unique<Streaming>(); }, 2);
    });
    rejects([] {
        VoiceTaskSessionPool unvalidated_count(std::make_unique<Legacy>(), 3,
            [] { return std::make_unique<Legacy>(); }, 2);
    });
    rejects([] {
        VoiceTaskSessionPool unaudited_factory(std::make_unique<Legacy>(), 2,
            [] { return std::make_unique<Legacy>(); });
    });
}

void test_audited_model_policy() {
    using engine::core::BackendType;
    using minitts::server::audited_slot_capacity;
    auto capacity = [](std::string_view family, VoiceTaskKind task,
                       BackendType backend = BackendType::Cuda, RunMode mode = RunMode::Offline) {
        return audited_slot_capacity(family, task, backend, mode);
    };
    require(capacity("bs_roformer", VoiceTaskKind::SourceSeparation) == 4, "audited separation capacity wrong");
    require(capacity("qwen3_asr", VoiceTaskKind::Asr) == 4, "audited ASR capacity wrong");
    require(capacity("cosyvoice3", VoiceTaskKind::VoiceCloning) == 4, "audited clone capacity wrong");
    require(capacity("qwen3_tts", VoiceTaskKind::Tts) == 3, "four-slot parity failure not restricted");
    require(capacity("rvc", VoiceTaskKind::VoiceConversion) == 3, "four-slot request failure not restricted");
    require(capacity("controlfoley", VoiceTaskKind::AudioGeneration) == 2, "VRAM-limited family not restricted");
    require(capacity("cosyvoice3", VoiceTaskKind::Tts) == 1, "untested task enabled");
    require(capacity("bs_roformer", VoiceTaskKind::SourceSeparation, BackendType::Cpu) == 1, "CPU fallback enabled");
    require(capacity("bs_roformer", VoiceTaskKind::SourceSeparation, BackendType::Vulkan) == 4, "validated Vulkan separation disabled");
    require(capacity("index_tts2", VoiceTaskKind::Tts, BackendType::Vulkan) == 3, "Vulkan VRAM ceiling wrong");
    require(capacity("yue2", VoiceTaskKind::AudioGeneration, BackendType::Vulkan) == 3, "Vulkan allocation failure count enabled");
    require(capacity("zipvoice", VoiceTaskKind::VoiceCloning, BackendType::Vulkan) == 2, "unstable Vulkan counts enabled");
    for (const auto & entry : minitts::server::kAuditedVulkanOfflineModels) {
        require(capacity(entry.family, entry.task, BackendType::Vulkan) == entry.capacity, "Vulkan table entry not admitted");
        require(capacity(entry.family, entry.task, BackendType::Vulkan, RunMode::Streaming) == 1, "Vulkan streaming fallback enabled");
        require(capacity(entry.family, entry.task, BackendType::Cpu) == 1, "CPU fallback enabled by Vulkan audit");
    }
    for (const auto & entry : minitts::server::kAuditedCudaOfflineModels) {
        require(capacity(entry.family, entry.task) == entry.capacity, "CUDA capacity changed");
    }
    require(capacity("firered_audio", VoiceTaskKind::VoiceCloning, BackendType::Vulkan) == 2, "shared FireRed Vulkan weights disabled");
    require(capacity("heartmula", VoiceTaskKind::AudioGeneration, BackendType::Vulkan) == 2, "shared HeartMuLa Vulkan weights disabled");
    for (const auto family : {"auk", "controlfoley", "glm_tts", "inflect_v2",
                              "mel_band_roformer", "moss_tts_v15", "outetts", "personaplex", "sheetsage2", "stable_audio"}) {
        for (const auto & entry : minitts::server::kAuditedCudaOfflineModels) {
            if (entry.family == family) {
                require(capacity(entry.family, entry.task, BackendType::Vulkan) == 1, "unvalidated Vulkan family enabled");
            }
        }
    }
    require(capacity("cosyvoice3", VoiceTaskKind::Tts, BackendType::Vulkan) == 1, "untested Vulkan task enabled");
    require(capacity("unknown-model", VoiceTaskKind::Tts, BackendType::Vulkan) == 1, "unknown Vulkan family enabled");
    require(capacity("qwen3_asr", VoiceTaskKind::Asr, BackendType::Cuda, RunMode::Streaming) == 1, "streaming fallback enabled");
    require(capacity("unknown-model", VoiceTaskKind::Tts) == 1, "unknown family enabled");
    require(capacity("f5_tts", VoiceTaskKind::Tts) == 2, "revalidated CUDA family disabled");
    require(capacity("chatterbox", VoiceTaskKind::VoiceCloning) == 2, "validated CUDA slots disabled");
    require(capacity("chatterbox_turbo", VoiceTaskKind::Tts) == 2, "validated CUDA slots disabled");
    require(capacity("mms_forced_aligner", VoiceTaskKind::Alignment) == 2, "validated CUDA slots disabled");
    require(capacity("seed_vc", VoiceTaskKind::VoiceConversion) == 2, "validated CUDA slots disabled");
    require(capacity("vevo2", VoiceTaskKind::VoiceConversion) == 2, "validated CUDA slots disabled");
    require(capacity("ace_step", VoiceTaskKind::AudioGeneration) == 2, "validated CUDA slots disabled");
    require(capacity("ace_step", VoiceTaskKind::AudioGeneration, BackendType::Vulkan) == 1, "unaudited ACE-Step Vulkan slots enabled");
    require(capacity("dramabox", VoiceTaskKind::Tts) == 1, "VRAM-blocked family enabled");
    require(capacity("liveavatar", VoiceTaskKind::AudioGeneration) == 1, "VRAM-blocked family enabled");
    require(capacity("audiosr", VoiceTaskKind::SpeechToSpeech) == 2, "guarded AudioSR CUDA slots disabled");
    require(capacity("miocodec", VoiceTaskKind::VoiceConversion) == 2, "deterministic MioCodec CUDA slots disabled");
    require(capacity("moss_tts_local", VoiceTaskKind::Tts) == 2, "revalidated CUDA slots disabled");
    require(capacity("moss_tts_local", VoiceTaskKind::Tts, BackendType::Vulkan) == 1, "unaudited Vulkan slots enabled");
    require(capacity("moss_ttsd", VoiceTaskKind::Tts) == 2, "revalidated CUDA slots disabled");
    require(capacity("moss_ttsd", VoiceTaskKind::Tts, BackendType::Vulkan) == 1, "unaudited Vulkan slots enabled");
    require(capacity("miotts", VoiceTaskKind::Tts) == 2, "revalidated CUDA slots disabled");
    require(capacity("miotts", VoiceTaskKind::Tts, BackendType::Vulkan) == 1, "unaudited Vulkan slots enabled");
    require(capacity("neutts", VoiceTaskKind::Tts) == 2, "revalidated CUDA slots disabled");
    require(capacity("neutts", VoiceTaskKind::Tts, BackendType::Vulkan) == 1, "unaudited Vulkan slots enabled");
    require(capacity("vieneu_v3_turbo", VoiceTaskKind::Tts) == 2, "revalidated CUDA slots disabled");
    require(capacity("vieneu_v3_turbo", VoiceTaskKind::Tts, BackendType::Vulkan) == 1, "unaudited Vulkan slots enabled");
    require(capacity("minimax_h3", VoiceTaskKind::AudioGeneration) == 1, "hardware-blocked family enabled");
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
        test_loaded_model_factory();
        test_audited_model_policy();
        test_streaming_state_and_error_release();
        std::cout << "PASS session pool validation, rollback, lifetime, shared weights, batch and stream isolation\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
