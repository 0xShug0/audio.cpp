#include "engine/community_models/earshot_vad/session.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/filesystem.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::community_models::earshot_vad {
namespace {

constexpr const char * kFamily = "earshot_vad";
constexpr const char * kDescription = "Earshot VAD (native C++ port), 16 kHz, 16 ms frames.";

runtime::CapabilitySet earshot_capabilities() {
    runtime::CapabilitySet out;
    out.supported_tasks = {
        {runtime::VoiceTaskKind::Vad, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
    };
    out.supports_timestamps = true;
    return out;
}

runtime::ModelMetadata earshot_metadata() {
    runtime::ModelMetadata metadata;
    metadata.family = kFamily;
    metadata.variant = "16k";
    metadata.description = kDescription;
    metadata.weight_candidates = {kWeightFile};
    return metadata;
}

int64_t to_input_sample(int64_t sample_16k, int input_rate) {
    if (input_rate == kSampleRate) {
        return sample_16k;
    }
    return static_cast<int64_t>(std::llround(static_cast<double>(sample_16k) * input_rate / kSampleRate));
}

runtime::SpeechSegment make_segment(int64_t start, int64_t end, float confidence) {
    runtime::SpeechSegment segment;
    segment.span = {start, std::max(start, end)};
    segment.confidence = confidence;
    return segment;
}

class EarshotVADSession final
    : public runtime::IOfflineVoiceTaskSession
    , public runtime::IStreamingVoiceTaskSession {
public:
    EarshotVADSession(runtime::TaskSpec task, const runtime::SessionOptions & options,
                      std::shared_ptr<const EarshotWeights> weights)
        : task_(std::move(task)),
          threads_(std::max(1, options.backend.threads)),
          default_config_(earshot_config_from_options(options.options)),
          stream_config_(default_config_),
          weights_(std::move(weights)),
          detector_(weights_),
          segmenter_(default_config_) {
        if (task_.task != runtime::VoiceTaskKind::Vad) {
            throw std::runtime_error("Earshot VAD only supports VoiceTaskKind::Vad");
        }
    }

    std::string family() const override { return kFamily; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        (void)earshot_config_from_options(request.options, &default_config_);
        prepared_ = true;
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("run()");
        if (task_.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("Earshot VAD offline run called on non-offline session");
        }
        if (!request.audio_input.has_value()) {
            throw std::runtime_error("Earshot VAD offline run requires audio_input");
        }
        const auto & audio = *request.audio_input;
        if (audio.sample_rate <= 0 || audio.channels <= 0) {
            throw std::runtime_error("Earshot VAD requires a positive sample rate and channel count");
        }
        if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
            throw std::runtime_error("Earshot VAD input size is not divisible by channel count");
        }
        const auto config = earshot_config_from_options(request.options, &default_config_);
        const auto started = std::chrono::steady_clock::now();

        std::vector<float> converted;
        const std::vector<float> * mono = &audio.samples;
        if (audio.sample_rate != kSampleRate || audio.channels != 1) {
            converted = audio::convert_interleaved_audio_to_mono_linear_resampled(
                audio.samples, audio.sample_rate, audio.channels, kSampleRate);
            mono = &converted;
        }
        const int64_t input_frames = static_cast<int64_t>(audio.samples.size()) / audio.channels;
        const int64_t length = static_cast<int64_t>(mono->size());

        const auto scores = score_earshot_frames(weights_, mono->data(), length, threads_);
        debug::timing_log_scalar("earshot_vad.offline.score_ms", debug::elapsed_ms(started));
        const auto segments = segment_earshot_scores(scores, config);

        runtime::TaskResult result;
        result.speech_segments.reserve(segments.size());
        const int64_t scored_frames = static_cast<int64_t>(scores.size());
        for (const auto & segment : segments) {
            const int64_t start = segment.start_frame * kFrameSize;
            // A segment still open at the end also covers the unscored tail samples.
            const int64_t end = segment.end_frame >= scored_frames ? length : segment.end_frame * kFrameSize;
            result.speech_segments.push_back(make_segment(
                std::min(input_frames, to_input_sample(start, audio.sample_rate)),
                std::min(input_frames, to_input_sample(end, audio.sample_rate)),
                segment.confidence));
        }
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

    runtime::StreamingPolicy streaming_policy() const override {
        runtime::StreamingPolicy policy;
        policy.input = runtime::StreamingInputKind::AudioChunks;
        policy.output = runtime::StreamingOutputKind::FinalResult;
        policy.preferred_audio_chunk_samples = kFrameSize;
        return policy;
    }

    void start_stream(const runtime::TaskRequest & request) override {
        require_prepared("start_stream()");
        stream_config_ = earshot_config_from_options(request.options, &default_config_);
        reset_stream();
    }

    void reset() override {
        require_prepared("reset()");
        reset_stream();
    }

    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override {
        require_prepared("process_audio_chunk()");
        if (task_.mode != runtime::RunMode::Streaming) {
            throw std::runtime_error("Earshot VAD process_audio_chunk called on non-streaming session");
        }
        if (chunk.sample_rate != kSampleRate) {
            throw std::runtime_error("Earshot VAD streaming requires 16000 Hz audio chunks");
        }
        if (chunk.channels != 1) {
            throw std::runtime_error("Earshot VAD streaming requires mono audio chunks");
        }
        if (!stream_started_) {
            stream_origin_ = chunk.start_sample;
            stream_started_ = true;
        } else if (chunk.start_sample != stream_origin_ + received_) {
            throw std::runtime_error("Earshot VAD streaming chunks must be contiguous");
        }
        const auto started = std::chrono::steady_clock::now();
        received_ += static_cast<int64_t>(chunk.samples.size());

        runtime::StreamEvent event;
        for (const float sample : chunk.samples) {
            pending_.push_back(sample);
            if (static_cast<int64_t>(pending_.size()) == kFrameSize) {
                push_frame(event);
                pending_.clear();
            }
        }
        debug::timing_log_scalar("earshot_vad.streaming.chunk_ms", debug::elapsed_ms(started));
        return event;
    }

    runtime::TaskResult finalize() override {
        require_prepared("finalize()");
        if (task_.mode != runtime::RunMode::Streaming) {
            throw std::runtime_error("Earshot VAD finalize called on non-streaming session");
        }
        runtime::TaskResult result;
        result.speech_segments = streamed_segments_;
        if (const auto last = segmenter_.finish()) {
            result.speech_segments.push_back(
                make_segment(stream_origin_ + last->start_frame * kFrameSize, stream_origin_ + received_, last->confidence));
        }
        return result;
    }

private:
    void require_prepared(const char * operation) const {
        if (!prepared_) {
            throw std::runtime_error(std::string("Earshot VAD ") + operation + " called before prepare()");
        }
    }

    void reset_stream() {
        detector_.reset();
        segmenter_ = EarshotSegmenter(stream_config_);
        pending_.clear();
        pending_.reserve(kFrameSize);
        streamed_segments_.clear();
        stream_started_ = false;
        stream_origin_ = 0;
        received_ = 0;
    }

    void push_frame(runtime::StreamEvent & event) {
        const float score = detector_.predict(pending_.data());
        const auto update = segmenter_.push(score);
        if (update.started_frame) {
            event.voice_activity.push_back({
                runtime::VoiceActivityEvent::Kind::SpeechStart,
                stream_origin_ + *update.started_frame * kFrameSize,
                score,
                std::nullopt,
            });
        }
        if (update.finished) {
            const auto segment = make_segment(
                stream_origin_ + update.finished->start_frame * kFrameSize,
                stream_origin_ + update.finished->end_frame * kFrameSize,
                update.finished->confidence);
            streamed_segments_.push_back(segment);
            event.voice_activity.push_back({
                runtime::VoiceActivityEvent::Kind::SpeechEnd,
                segment.span.end_sample,
                score,
                segment,
            });
        }
    }

    runtime::TaskSpec task_;
    int threads_ = 1;
    EarshotSegmenterConfig default_config_;
    EarshotSegmenterConfig stream_config_;
    std::shared_ptr<const EarshotWeights> weights_;
    EarshotDetector detector_;
    EarshotSegmenter segmenter_;
    std::vector<float> pending_;
    std::vector<runtime::SpeechSegment> streamed_segments_;
    bool prepared_ = false;
    bool stream_started_ = false;
    int64_t stream_origin_ = 0;
    int64_t received_ = 0;
};

class EarshotVADLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    explicit EarshotVADLoadedModel(std::shared_ptr<const EarshotWeights> weights)
        : metadata_(earshot_metadata()),
          capabilities_(earshot_capabilities()),
          weights_(std::move(weights)) {}

    const runtime::ModelMetadata & metadata() const noexcept override { return metadata_; }
    const runtime::CapabilitySet & capabilities() const noexcept override { return capabilities_; }

    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override {
        return std::make_unique<EarshotVADSession>(task, options, weights_);
    }

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const EarshotWeights> weights_;
};

std::vector<runtime::NamedAsset> discover_weight_assets(const runtime::ModelLoadRequest & request) {
    if (engine::io::is_existing_file(request.model_path)) {
        return {{"default", std::filesystem::weakly_canonical(request.model_path)}};
    }
    return runtime::discover_named_assets(std::filesystem::weakly_canonical(request.model_path), {kWeightFile});
}

class EarshotVADLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override { return kFamily; }
    std::vector<std::string> family_aliases() const override { return {"earshot"}; }
    runtime::CapabilitySet advertised_capabilities() const override { return earshot_capabilities(); }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family() && *request.family_hint != "earshot") {
            return false;
        }
        try {
            const auto assets = resolve_earshot_assets(request.model_path);
            // Without a family hint, only claim files carrying the Earshot weight name.
            return request.family_hint.has_value() || assets.checkpoint_path.filename() == kWeightFile;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        if (request.config_id.has_value()) {
            throw std::runtime_error("Earshot VAD does not expose selectable config assets");
        }
        const auto assets = resolve_earshot_assets(request.model_path);
        runtime::ModelInspection inspection;
        inspection.model_root = assets.model_root;
        inspection.metadata = earshot_metadata();
        inspection.capabilities = earshot_capabilities();
        inspection.discovered_weights = discover_weight_assets(request);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        return load_earshot_vad_model(request);
    }
};

}  // namespace

std::unique_ptr<runtime::ILoadedVoiceModel> load_earshot_vad_model(const runtime::ModelLoadRequest & request) {
    if (request.config_id.has_value()) {
        throw std::runtime_error("Earshot VAD does not expose selectable config assets");
    }
    const auto weights = discover_weight_assets(request);
    const auto * selected = runtime::select_named_asset(weights, request.weight_id, "weight");
    const auto assets = resolve_earshot_assets(selected != nullptr ? selected->path : request.model_path);
    return std::make_unique<EarshotVADLoadedModel>(load_earshot_weights_cached(assets.checkpoint_path));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_earshot_vad_loader() {
    return std::make_shared<EarshotVADLoader>();
}

}  // namespace engine::community_models::earshot_vad
