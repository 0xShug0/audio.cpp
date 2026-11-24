// Test the opaque C handles against a tiny recording session. Including the
// facade here exposes its private handle definitions only to this test binary;
// no test hook or extra symbol is added to the public C API.
#include "../../src/capi/audiocpp.cpp"
#include "../../app/workflow/execution.h"
#include "../../app/workflow/workflow.h"
#include "../../app/recipe/cli_recipe.h"
#include "test_assert.h"
#include "engine/framework/audio/wav_writer.h"

#include <chrono>
#include <fstream>
#include <iostream>

namespace {
using engine::test::require;
using engine::test::require_eq;
using Options = std::unordered_map<std::string, std::string>;

struct Record {
    std::vector<Options> prepared, executed;
    bool fail_prepare = false, fail_run = false;
};

class RecordingSession final : public rt::IOfflineVoiceTaskSession,
                               public rt::IStreamingVoiceTaskSession {
public:
    RecordingSession(std::shared_ptr<Record> record, rt::RunMode mode)
        : record_(std::move(record)), mode_(mode) {}
    std::string family() const override { return "seed_fixture"; }
    rt::VoiceTaskKind task_kind() const override { return rt::VoiceTaskKind::Tts; }
    rt::RunMode run_mode() const override { return mode_; }
    void prepare(const rt::SessionPreparationRequest & request) override {
        record_->prepared.push_back(request.options);
        if (record_->fail_prepare) throw std::runtime_error("controlled preparation failure");
        prepared_options_ = request.options;
    }
    rt::TaskResult run(const rt::TaskRequest & request) override {
        record_->executed.push_back(request.options);
        if (record_->fail_run) throw std::runtime_error("controlled execution failure");
        rt::TaskResult out;
        const auto seed = prepared_options_.find("seed");
        out.text_output = rt::Transcript{seed == prepared_options_.end() ? "omitted" : seed->second, "en"};
        return out;
    }
    rt::StreamingPolicy streaming_policy() const override { return {}; }
    void start_stream(const rt::TaskRequest & request) override { (void)run(request); }
    void reset() override {}
    rt::StreamEvent process_audio_chunk(const rt::AudioChunk &) override { return {}; }
    rt::TaskResult finalize() override { return {}; }
private:
    std::shared_ptr<Record> record_;
    rt::RunMode mode_;
    Options prepared_options_;
};

class Model final : public rt::ILoadedVoiceModel {
public:
    explicit Model(std::shared_ptr<Record> record) : record_(std::move(record)) { metadata_.family = "seed_fixture"; }
    const rt::ModelMetadata & metadata() const noexcept override { return metadata_; }
    const rt::CapabilitySet & capabilities() const noexcept override { return capabilities_; }
    std::unique_ptr<rt::IVoiceTaskSession> create_task_session(const rt::TaskSpec & task, const rt::SessionOptions &) const override {
        return std::make_unique<RecordingSession>(record_, task.mode);
    }
private:
    std::shared_ptr<Record> record_;
    rt::ModelMetadata metadata_;
    rt::CapabilitySet capabilities_;
};

class Loader final : public rt::IVoiceModelLoader {
public:
    explicit Loader(std::shared_ptr<Record> record) : record_(std::move(record)) {}
    std::string family() const override { return "seed_fixture"; }
    bool can_load(const rt::ModelLoadRequest &) const override { return true; }
    rt::ModelInspection inspect(const rt::ModelLoadRequest &) const override { return {}; }
    std::unique_ptr<rt::ILoadedVoiceModel> load(const rt::ModelLoadRequest &) const override {
        return std::make_unique<Model>(record_);
    }
private:
    std::shared_ptr<Record> record_;
};

uint32_t resolved(const Options & options) {
    const auto value = rt::parse_u32_option(options, {"seed"});
    require(value && *value <= 2147483647U, "seed resolved within common range");
    return *value;
}

size_t draws = 0;
void capture(engine::debug::LogLevel, const char * text, void *) {
    if (std::string(text).find("request.seed ") != std::string::npos) ++draws;
    std::cerr << text;
}

audiocpp_session handle(const std::shared_ptr<Record> & record, rt::RunMode mode) {
    audiocpp_session out;
    out.mode = mode;
    out.session = std::make_unique<RecordingSession>(record, mode);
    return out;
}

void capi_prepare_and_run() {
    auto record = std::make_shared<Record>();
    auto session = handle(record, rt::RunMode::Offline);
    std::unique_ptr<audiocpp_request> request(audiocpp_request_create());
    require(audiocpp_request_set_option(request.get(), "seed", "-1") == AUDIOCPP_OK, "set sentinel");
    const auto before = draws;
    require(audiocpp_session_prepare(&session, request.get()) == AUDIOCPP_OK, "explicit prepare");
    const auto prepared = resolved(record->prepared.back());
    audiocpp_result * result = nullptr;
    require(audiocpp_session_run(&session, request.get(), &result) == AUDIOCPP_OK, "run prepared request");
    audiocpp_result_free(result);
    require_eq(resolved(record->executed.back()), prepared, "prepare/run share seed");
    require_eq(draws, before + 1, "prepare/run draw once");
    require_eq(request->request.options.at("seed"), std::string("-1"), "raw sentinel restored");
    require(audiocpp_session_run(&session, request.get(), &result) == AUDIOCPP_OK, "repeat raw request");
    audiocpp_result_free(result);
    require_eq(draws, before + 2, "new invocation draws again, irrespective of collisions");
    require(record->prepared.back() == record->executed.back(), "second invocation prepare/run aligned");

    require(audiocpp_session_prepare(&session, request.get()) == AUDIOCPP_OK, "prepare before mutation");
    const auto changed = draws;
    require(audiocpp_request_set_text(request.get(), "changed", nullptr) == AUDIOCPP_OK, "request mutation");
    require(audiocpp_session_run(&session, request.get(), &result) == AUDIOCPP_OK, "mutated request run");
    audiocpp_result_free(result);
    require_eq(draws, changed + 1, "mutation invalidates pending resolution");
    require_eq(request->request.options.at("seed"), std::string("-1"), "mutated request stays raw");

    require(audiocpp_session_prepare(&session, request.get()) == AUDIOCPP_OK, "prepare before failure");
    record->fail_run = true;
    require(audiocpp_session_run(&session, request.get(), &result) != AUDIOCPP_OK, "controlled failure propagated");
    require_eq(request->request.options.at("seed"), std::string("-1"), "exception restores sentinel");
    record->fail_run = false;
    const auto failed = draws;
    require(audiocpp_session_run(&session, request.get(), &result) == AUDIOCPP_OK, "retry after failure");
    audiocpp_result_free(result);
    require_eq(draws, failed + 1, "failed invocation cannot retain pending seed");
    record->fail_prepare = true;
    require(audiocpp_session_prepare(&session, request.get()) != AUDIOCPP_OK, "explicit prepare failure propagated");
    record->fail_prepare = false;
    const auto prepare_failed = draws;
    require(audiocpp_session_run(&session, request.get(), &result) == AUDIOCPP_OK, "run after failed preparation");
    audiocpp_result_free(result);
    require_eq(draws, prepare_failed + 1, "failed preparation cannot retain seed");

    require(audiocpp_session_prepare(&session, request.get()) == AUDIOCPP_OK, "prepare before fixed seed");
    require(audiocpp_request_set_option(request.get(), "seed", "42") == AUDIOCPP_OK, "set explicit seed");
    const auto fixed = draws;
    require(audiocpp_session_run(&session, request.get(), &result) == AUDIOCPP_OK, "explicit request run");
    audiocpp_result_free(result);
    require_eq(record->executed.back().at("seed"), std::string("42"), "explicit seed untouched");
    require_eq(draws, fixed, "explicit seed never draws");
    require(audiocpp_session_run(&session, nullptr, &result) == AUDIOCPP_OK, "omitted request");
    audiocpp_result_free(result);
    require(record->executed.back().empty(), "omitted seed stays absent");
}

void capi_identity_stream_and_sessions() {
    auto record = std::make_shared<Record>();
    auto stream = handle(record, rt::RunMode::Streaming);
    auto other = handle(record, rt::RunMode::Offline);
    std::unique_ptr<audiocpp_request> request(audiocpp_request_create());
    require(audiocpp_request_set_option(request.get(), "seed", "-1") == AUDIOCPP_OK, "set stream seed");
    require(audiocpp_session_prepare(&stream, request.get()) == AUDIOCPP_OK, "prepare stream");
    const auto stream_seed = resolved(record->prepared.back());
    require(audiocpp_session_prepare(&other, request.get()) == AUDIOCPP_OK, "prepare independent session");
    const auto other_seed = resolved(record->prepared.back());
    const auto before = draws;
    require(audiocpp_stream_start(&stream, request.get()) == AUDIOCPP_OK, "start prepared stream");
    require_eq(resolved(record->executed.back()), stream_seed, "stream retains its own prepared seed");
    audiocpp_result * result = nullptr;
    require(audiocpp_session_run(&other, request.get(), &result) == AUDIOCPP_OK, "run independent session");
    audiocpp_result_free(result);
    require_eq(resolved(record->executed.back()), other_seed, "other session retains own seed");
    require_eq(draws, before, "independent pending resolutions do not redraw");
    require(audiocpp_stream_finish(&stream, &result) == AUDIOCPP_OK, "finish stream");
    audiocpp_result_free(result);
    require(audiocpp_session_prepare(&stream, request.get()) == AUDIOCPP_OK, "prepare before reset");
    require(audiocpp_stream_reset(&stream) == AUDIOCPP_OK, "reset stream");
    const auto reset = draws;
    require(audiocpp_stream_start(&stream, request.get()) == AUDIOCPP_OK, "stream after reset");
    require_eq(draws, reset + 1, "reset invalidates pending resolution");
    require(audiocpp_stream_reset(&stream) == AUDIOCPP_OK, "stop test stream");
    require(audiocpp_session_prepare(&other, request.get()) == AUDIOCPP_OK, "prepare request to release");
    request.reset();
    request.reset(audiocpp_request_create());
    require(audiocpp_request_set_option(request.get(), "seed", "-1") == AUDIOCPP_OK, "new handle seed");
    const auto released = draws;
    require(audiocpp_session_run(&other, request.get(), &result) == AUDIOCPP_OK, "new request after freed handle");
    audiocpp_result_free(result);
    require_eq(draws, released + 1, "freed request identity never matches new handle");
}

void application_dispatch(const std::filesystem::path & root) {
    auto record = std::make_shared<Record>();
    RecordingSession session(record, rt::RunMode::Offline);
    minitts::app::AppBatchRequest batch;
    for (const char * value : {"-1", "42", "-1", ""}) {
        rt::TaskRequest request;
        if (*value) request.options["seed"] = value;
        batch.requests.push_back({std::to_string(batch.requests.size()), std::move(request)});
    }
    const auto before = draws;
    const auto results = minitts::app::run_offline_batch(session, session, batch, minitts::app::AudioMergeMode::None);
    require_eq(draws, before + 2, "batch resolves only sentinel items");
    require(record->prepared.front() == record->executed.front(), "batch first preparation aligned");
    require_eq(record->prepared.size(), size_t{4}, "each item after randomization gets matching preparation");
    require(record->prepared == record->executed, "mixed batch preparation matches every invocation");
    require_eq(results.results[1].result.text_output->text, std::string("42"), "fixed item restores prepared seed");
    require_eq(results.results[3].result.text_output->text, std::string("omitted"), "omitted item cannot inherit random preparation");
    resolved(record->executed[0]); resolved(record->executed[2]);
    require_eq(record->executed[1].at("seed"), std::string("42"), "fixed batch neighbor preserved");
    require(record->executed[3].empty(), "omitted batch neighbor preserved");

    rt::ModelRegistry registry;
    registry.register_loader(std::make_shared<Loader>(record));
    minitts::app::CliRecipeOptions config;
    config.model_paths["tts"] = root;
    config.model_families["tts"] = "seed_fixture";
    minitts::app::CliRecipeModelExecutor executor(registry, std::move(config));
    rt::TaskRequest recipe;
    recipe.options = minitts::app::options_for_recipe_role({{"seed", "42"}, {"tts.seed", "-1"}}, "tts", {"tts"}, true);
    (void)executor.run("tts", {rt::VoiceTaskKind::Tts, rt::RunMode::Offline}, recipe);
    resolved(record->executed.back());
    require(record->prepared.back() == record->executed.back(), "recipe preparation aligned after role overrides");

    const auto workflow = root / "workflow.json";
    std::ofstream(workflow) << "{\"steps\":[{\"id\":\"one\",\"type\":\"model\",\"family\":\"seed_fixture\",\"model\":\".\",\"task\":\"tts\",\"request\":{\"text\":\"hello\",\"seed\":-1}}]}";
    minitts::app::WorkflowRunOptions options;
    options.workflow_path = workflow;
    options.output_dir = root / "outputs";
    options.backend.type = engine::core::BackendType::Cpu;
    minitts::app::run_json_workflow(registry, options);
    resolved(record->executed.back());
    require(record->prepared.back() == record->executed.back(), "workflow preparation aligned");
}
void mixed_batch_seed_restoration(const std::filesystem::path & root) {
    for (const auto & seeds : {std::vector<std::string>{"42", "42", "42"},
                              std::vector<std::string>{"42", "-1", "42"},
                              std::vector<std::string>{"-1", "42", "42"}}) {
        auto record = std::make_shared<Record>();
        RecordingSession session(record, rt::RunMode::Offline);
        minitts::app::AppBatchRequest batch;
        for (const auto & seed : seeds) {
            rt::TaskRequest request;
            request.options["seed"] = seed;
            batch.requests.push_back({std::to_string(batch.requests.size()), std::move(request)});
        }
        const auto results = minitts::app::run_offline_batch(session, session, batch, minitts::app::AudioMergeMode::None);
        require_eq(results.results.back().result.text_output->text, std::string("42"), "final output uses restored prepared seed");
        if (seeds.front() == "42") {
            require_eq(results.results.front().result.text_output->text, results.results.back().result.text_output->text,
                       "fixed first/last outputs match across an intervening random item");
        }
        require_eq(record->prepared.size(), seeds[1] == "42" && seeds[0] == "42" ? size_t{1} : size_t{3},
                   "fixed-only batches preserve prepare-once; randomized batches stay request-aligned");
    }

    auto record = std::make_shared<Record>();
    rt::ModelRegistry registry;
    registry.register_loader(std::make_shared<Loader>(record));
    std::filesystem::create_directories(root / "seeds");
    std::ofstream(root / "seeds/01.txt") << "42";
    std::ofstream(root / "seeds/02.txt") << "-1";
    std::ofstream(root / "seeds/03.txt") << "42";
    minitts::app::WorkflowRunOptions options;
    options.workflow_path = root / "mixed.json";
    options.output_dir = root / "mixed-output";
    options.backend.type = engine::core::BackendType::Cpu;
    std::ofstream(options.workflow_path) << R"({"steps":[
        {"id":"seeds","type":"batch_inputs","text_dir":"seeds"},
        {"id":"each","type":"model","foreach":"${seeds.items}","model":".","family":"seed_fixture","task":"tts",
         "request":{"text":"test","options":{"seed":"${item.text}"}}}]})";
    minitts::app::run_json_workflow(registry, options);
    require_eq(record->executed.size(), size_t{3}, "workflow tested all mixed items");
    require(record->prepared == record->executed, "workflow cannot retain middle item's prepared seed");
}

void workflow_loops_and_recipe_stream(const std::filesystem::path & root) {
    auto record = std::make_shared<Record>();
    rt::ModelRegistry registry;
    registry.register_loader(std::make_shared<Loader>(record));
    minitts::app::WorkflowRunOptions options;
    options.workflow_path = root / "loops.json";
    options.output_dir = root / "outputs";
    options.backend.type = engine::core::BackendType::Cpu;
    std::filesystem::create_directories(root / "texts");
    std::ofstream(root / "texts/one.txt") << "one";
    std::ofstream(root / "texts/two.txt") << "two";
    std::ofstream(options.workflow_path) << R"({"steps":[
        {"id":"inputs","type":"batch_inputs","text_dir":"texts"},
        {"id":"each","type":"model","foreach":"${inputs.items}","model":".","family":"seed_fixture","task":"tts",
         "request":{"text":"${item.text}","seed":-1}}]})";
    const auto before = draws;
    minitts::app::run_json_workflow(registry, options);
    require_eq(draws, before + 2, "foreach resolves each expanded invocation");
    require_eq(record->prepared.size(), size_t{2}, "foreach prepares each fresh seed");
    require(record->prepared == record->executed, "foreach preparation/execution options aligned");
    record->prepared.clear(); record->executed.clear();

    engine::audio::write_pcm16_wav(root / "input.wav", 16000, 1, std::vector<float>(32000, 0.1f));
    std::ofstream(options.workflow_path) << R"({"steps":[
        {"id":"chunks","type":"chunked_model","model":".","family":"seed_fixture","task":"tts","chunk_seconds":1,
         "request":{"text":"test","audio":"input.wav","seed":-1}}]})";
    const auto chunked = draws;
    minitts::app::run_json_workflow(registry, options);
    require_eq(draws, chunked + 2, "workflow chunks resolve once per model invocation");
    require_eq(record->executed.size(), size_t{2}, "workflow used two audio chunks");
    require(record->prepared == record->executed, "workflow chunk preparation/execution seeds aligned");

    minitts::app::CliRecipeOptions config;
    config.model_paths["tts"] = root;
    config.model_families["tts"] = "seed_fixture";
    minitts::app::CliRecipeModelExecutor executor(registry, std::move(config));
    rt::TaskRequest request;
    request.options = {{"seed", "-1"}};
    const auto streaming = draws;
    (void)executor.stream("tts", {rt::VoiceTaskKind::Tts, rt::RunMode::Streaming}, request, {});
    require_eq(draws, streaming + 1, "streaming recipe resolves once");
    require(record->prepared.back() == record->executed.back(), "streaming recipe prepares resolved request");
}

} // namespace

int main(int argc, char ** argv) {
    engine::debug::configure_logging({true});
    engine::debug::set_log_callback(capture, nullptr);
    const auto start = std::chrono::steady_clock::now();
    const auto root = std::filesystem::temp_directory_path() /
        ("audiocpp-seed-" + std::to_string(start.time_since_epoch().count()));
    std::filesystem::create_directories(root);
    try {
        const bool batch_only = argc > 1 && std::string(argv[1]) == "batch-seed";
        const bool additional_only = batch_only || (argc > 1 && std::string(argv[1]) == "loops");
        if (!additional_only) {
            capi_prepare_and_run();
            capi_identity_stream_and_sessions();
            application_dispatch(root);
        }
        if (!batch_only) workflow_loops_and_recipe_stream(root);
        mixed_batch_seed_restoration(root);
        std::filesystem::remove_all(root);
        engine::debug::timing_log_scalar("session.wall_ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        std::cout << "request seed dispatch passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::filesystem::remove_all(root);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
