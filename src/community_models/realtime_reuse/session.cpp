#include "engine/community_models/realtime_reuse/session.h"
#include "engine/community_models/realtime_reuse/frontend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

namespace engine::models::realtime_reuse {
namespace {
class Session final : public runtime::RuntimeSessionBase,
                      public runtime::IOfflineVoiceTaskSession,
                      public runtime::IStreamingVoiceTaskSession {
public:
    Session(runtime::TaskSpec task, runtime::SessionOptions options, std::shared_ptr<const Assets> assets,
        std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), contract_(std::move(contract)) {
        runtime::validate_spec_backed_session_options(options, *contract_, "realtime_reuse", "Real-time RE-USE");
        if (task.task != runtime::VoiceTaskKind::SpeechToSpeech) throw std::runtime_error("Real-time RE-USE supports the s2s task");
        Config config;
        config.exit_layer = runtime::parse_int_option(options.options, {"realtime_reuse.exit_layer"}).value_or(8);
        config.lookahead_frames = runtime::parse_int_option(options.options, {"realtime_reuse.lookahead_frames"}).value_or(0);
        const auto storage = runtime::find_option(options.options, {"realtime_reuse.weight_type"}).value_or("native");
        runtime_ = std::make_unique<Runtime>(std::move(assets), execution_context(), assets::parse_tensor_storage_type(storage), config);
    }
    std::string family() const override { return "realtime_reuse"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }
    void prepare(const runtime::SessionPreparationRequest &) override { mark_prepared(); }
    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("Real-time RE-USE run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Real-time RE-USE");
        if (!request.audio_input || request.audio_input->channels != 1 || request.audio_input->samples.empty())
            throw std::runtime_error("Real-time RE-USE requires nonempty mono audio");
        const auto started = std::chrono::steady_clock::now();
        const auto & input = *request.audio_input;
        runtime::TaskResult result;
        result.audio_output = runtime::AudioBuffer{input.sample_rate, 1,
            restore(*runtime_, input.samples, input.sample_rate, execution_context().config().threads)};
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }
    runtime::StreamingPolicy streaming_policy() const override {
        return {runtime::StreamingInputKind::AudioChunks, runtime::StreamingOutputKind::FinalResult, 0, .02};
    }
    void start_stream(const runtime::TaskRequest & request) override {
        require_prepared("Real-time RE-USE start_stream");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Real-time RE-USE");
        reset();
        started_ = std::chrono::steady_clock::now();
        active_ = true;
    }
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override {
        if (!active_) throw std::runtime_error("Real-time RE-USE stream has not started");
        if (chunk.channels != 1) throw std::runtime_error("Real-time RE-USE streaming requires mono audio");
        if (!frontend_) {
            sample_rate_ = chunk.sample_rate;
            frontend_ = std::make_unique<StreamingFrontend>(*runtime_, sample_rate_, execution_context().config().threads);
        }
        if (chunk.sample_rate != sample_rate_) throw std::runtime_error("Real-time RE-USE sample rate changed within a stream");
        runtime::StreamEvent event;
        auto output = frontend_->push(chunk.samples, false);
        if (!output.empty()) {
            collected_.insert(collected_.end(), output.begin(), output.end());
            event.audio_output = runtime::AudioBuffer{sample_rate_, 1, std::move(output)};
        }
        return event;
    }
    void set_stream_event_sink(runtime::StreamEventCallback sink) override { sink_ = std::move(sink); }
    runtime::TaskResult finalize() override {
        if (!active_ || !frontend_) throw std::runtime_error("Real-time RE-USE requires audio before finalization");
        auto tail = frontend_->push({}, true);
        collected_.insert(collected_.end(), tail.begin(), tail.end());
        if (sink_) {
            runtime::StreamEvent event;
            if (!tail.empty()) event.audio_output = runtime::AudioBuffer{sample_rate_, 1, std::move(tail)};
            event.is_final = true;
            sink_(event);
        }
        active_ = false;
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started_));
        runtime::TaskResult result;
        result.audio_output = runtime::AudioBuffer{sample_rate_, 1, std::move(collected_)};
        return result;
    }
    void reset() override {
        frontend_.reset();
        collected_.clear();
        sample_rate_ = 0;
        active_ = false;
    }
private:
    runtime::TaskSpec task_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<Runtime> runtime_;
    std::unique_ptr<StreamingFrontend> frontend_;
    runtime::StreamEventCallback sink_;
    std::vector<float> collected_;
    int sample_rate_ = 0;
    bool active_ = false;
    std::chrono::steady_clock::time_point started_;
};
}
std::shared_ptr<runtime::IVoiceModelLoader> make_realtime_reuse_loader() {
    runtime::SpecBackedVoiceModelConfig<Assets> config;
    config.family = "realtime_reuse";
    config.load_assets = load_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const Assets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<Session>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}
} // namespace engine::models::realtime_reuse
