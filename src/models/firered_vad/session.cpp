#include "engine/models/firered_vad/session.h"
#include "engine/models/firered_vad/runtime.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <cmath>
#include <limits>

namespace engine::models::firered_vad {
namespace {

struct Assets { assets::ResourceBundle resources; };

class Session final : public runtime::RuntimeSessionBase,
                      public runtime::IOfflineVoiceTaskSession,
                      public runtime::IStreamingVoiceTaskSession {
public:
    Session(const runtime::SessionOptions & options, runtime::RunMode mode,
            std::shared_ptr<const assets::TensorSource> source,
            std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), mode_(mode), contract_(std::move(contract)),
          runtime_(std::move(source), execution_context()) {
        if (mode_ == runtime::RunMode::Streaming && !runtime_.causal())
            throw std::runtime_error("FireRed VAD streaming requires the Stream-VAD checkpoint");
    }
    std::string family() const override { return "firered_vad"; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::Vad; }
    runtime::RunMode run_mode() const override { return mode_; }
    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "FireRed VAD");
        if (!request.audio || request.audio->sample_rate != 16000 || request.audio->channels != 1)
            throw std::runtime_error("FireRed VAD requires 16000 Hz mono audio");
        options_ = parse_options(request.options);
        mark_prepared();
    }
    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("FireRed VAD run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "FireRed VAD");
        if (!request.audio_input) throw std::runtime_error("FireRed VAD requires audio_input");
        const auto started = std::chrono::steady_clock::now();
        auto result = runtime_.detect(*request.audio_input, parse_options(request.options));
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }
    runtime::StreamingPolicy streaming_policy() const override {
        return {runtime::StreamingInputKind::AudioChunks, runtime::StreamingOutputKind::FinalResult, 2560, 0.0};
    }
    void start_stream(const runtime::TaskRequest & request) override {
        require_prepared("FireRed VAD stream");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "FireRed VAD");
        options_ = parse_options(request.options);
        reset();
    }
    void reset() override { runtime_.reset(options_); }
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override {
        require_prepared("FireRed VAD stream");
        const auto started = std::chrono::steady_clock::now();
        auto event = runtime_.process(chunk);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return event;
    }
    runtime::TaskResult finalize() override { return runtime_.finalize(); }

private:
    DetectionOptions parse_options(const std::unordered_map<std::string, std::string> & values) const {
        const auto ms_frames = [&](const char * name, int default_frames) {
            const auto value = runtime::parse_int_option(values, {name});
            return value ? static_cast<int>((static_cast<int64_t>(*value) + 5) / 10) : default_frames;
        };
        const auto sec_frames = [&](const char * name, int default_frames) {
            const auto value = runtime::parse_finite_float_option(values, {name});
            if (!value) return default_frames;
            const double frames = std::round(*value * 100.0f);
            if (frames < 1 || frames > std::numeric_limits<int>::max())
                throw std::runtime_error(std::string(name) + " is outside the supported frame range");
            return static_cast<int>(frames);
        };
        DetectionOptions result;
        const auto threshold = values.find("threshold");
        result.threshold = threshold == values.end() ? (runtime_.causal() ? 0.5 : 0.4) : std::stod(threshold->second);
        result.min_speech_frames = ms_frames("min_speech_duration_ms", runtime_.causal() ? 8 : 20);
        result.smooth_window_frames = ms_frames("smooth_window_ms", 5);
        result.max_speech_frames = sec_frames("max_speech_duration_sec", 2000);
        result.min_silence_frames = ms_frames("min_silence_duration_ms", 20);
        result.merge_silence_frames = ms_frames("merge_silence_duration_ms", 0);
        result.extend_speech_frames = ms_frames("speech_pad_ms", 0);
        result.pad_start_frames = ms_frames("speech_start_pad_ms", 5);
        result.chunk_frames = sec_frames("audio_chunk_duration_sec", 30000);
        return result;
    }
    runtime::RunMode mode_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    FireRedDfsmnRuntime runtime_;
    DetectionOptions options_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_firered_vad_loader() {
    runtime::SpecBackedVoiceModelConfig<Assets> config;
    config.family = "firered_vad";
    config.load_assets = [](const std::filesystem::path & path) {
        auto result = std::make_shared<Assets>();
        result->resources = model_spec::load_resource_bundle_for_family(path, "firered_vad");
        return result;
    };
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                              std::shared_ptr<const Assets> assets,
                              std::shared_ptr<const model_spec::ModelContract> contract) {
        if (task.task != runtime::VoiceTaskKind::Vad) throw std::runtime_error("FireRed VAD supports the VAD task");
        runtime::validate_spec_backed_session_options(options, *contract, "firered_vad", "FireRed VAD");
        return std::make_unique<Session>(options, task.mode, assets->resources.open_tensor_source("weights"), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::firered_vad
