#include "engine/community_models/parakeet_tdt/bark3.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/chunking.h"
#include "engine/framework/runtime/errors.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::community_models::parakeet_tdt {
runtime::GraphOptimizationOptions bark3_encoder_graph_options(runtime::GraphOptimizationBackend backend) {
    auto options = runtime::graph_optimization_options_for_backend(backend);
    if (backend == runtime::GraphOptimizationBackend::Cpu) {
        // Allocation happens after optimization. ggml needs the view nodes to
        // release their backing buffers at the correct last-use boundary.
        options.elide_metadata_only_ops = false;
        options.elide_noop_nodes = false;
    }
    return options;
}
namespace {
constexpr int kSampleRate = 16000;

void validate_chunk_options(const std::unordered_map<std::string, std::string> & options) {
    try {
        const auto mode = engine::audio::parse_audio_chunk_mode(options);
        if (mode != engine::audio::AudioChunkMode::Auto && mode != engine::audio::AudioChunkMode::Fixed &&
            mode != engine::audio::AudioChunkMode::Vad && mode != engine::audio::AudioChunkMode::None)
            throw runtime::InvalidRequestError("Bark audio_chunk_mode must be auto, fixed, vad, or none");
        if (const auto duration = engine::audio::parse_audio_chunk_seconds_override(options)) {
            if (!std::isfinite(*duration) || *duration < 0.001f)
                throw runtime::InvalidRequestError("Bark audio_chunk_duration_sec must be finite and at least 0.001");
        }
    } catch (const std::runtime_error & error) {
        throw runtime::InvalidRequestError(error.what());
    }
}

runtime::SessionPreparationRequest preparation_with_prefix(runtime::SessionPreparationRequest request) {
    validate_chunk_options(request.options);
    if (request.audio && request.audio->max_input_samples > 0 &&
        request.audio->sample_rate > 0 && request.audio->channels > 0) {
        const long double frames = request.audio->max_input_samples / request.audio->channels;
        const long double samples = std::ceil(frames * kSampleRate / request.audio->sample_rate);
        if (samples > std::numeric_limits<int64_t>::max() - kBark3InputPrefixSamples)
            throw runtime::InvalidRequestError("Bark input capacity is too large");
        request.audio = runtime::AudioPreparationContract{
            kSampleRate, 1, static_cast<int64_t>(samples) + kBark3InputPrefixSamples};
    }
    return request;
}

}  // namespace

void remove_bark3_prefix_timestamps(std::vector<runtime::WordTimestamp> & words, int64_t original_samples) {
    const auto shift = [original_samples](int64_t value) {
        return std::min(original_samples, value <= kBark3InputPrefixSamples ? int64_t(0) : value - kBark3InputPrefixSamples);
    };
    for (auto & word : words) {
        word.span.start_sample = shift(word.span.start_sample);
        word.span.end_sample = std::max(word.span.start_sample, shift(word.span.end_sample));
    }
}

namespace {
void validate_chunk_mode(const runtime::TaskRequest & request) {
    validate_chunk_options(request.options);
    const auto mode = request.options.find("audio_chunk_mode");
    if (mode != request.options.end() && mode->second == "vad")
        throw runtime::InvalidRequestError("Bark VAD chunking is an offline operation");
}

class BarkOffline final : public runtime::IOfflineVoiceTaskSession {
public:
    explicit BarkOffline(std::unique_ptr<runtime::IOfflineVoiceTaskSession> inner) : inner_(std::move(inner)) {
        if (!inner_) throw std::invalid_argument("Bark requires an offline delegate");
    }
    std::string family() const override { return inner_->family(); }
    runtime::VoiceTaskKind task_kind() const override { return inner_->task_kind(); }
    runtime::RunMode run_mode() const override { return inner_->run_mode(); }
    void prepare(const runtime::SessionPreparationRequest & request) override {
        prepared_ = false;
        inner_->prepare(preparation_with_prefix(request));
        prepared_ = true;
    }
    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        if (!prepared_) throw std::logic_error("Bark requires prepare() before run()");
        validate_chunk_options(request.options);
        if (!request.audio_input || request.audio_input->samples.empty() ||
            request.audio_input->sample_rate <= 0 || request.audio_input->channels <= 0 ||
            request.audio_input->samples.size() % request.audio_input->channels != 0)
            throw runtime::InvalidRequestError("Bark requires nonempty audio with a valid sample rate and channel layout");
        auto padded = request;
        auto & audio = *padded.audio_input;
        if (audio.sample_rate != kSampleRate || audio.channels != 1) {
            audio.samples = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
                audio.samples, audio.sample_rate, audio.channels, kSampleRate);
            audio.sample_rate = kSampleRate;
            audio.channels = 1;
        }
        const auto original_samples = static_cast<int64_t>(audio.samples.size());
        audio.samples.insert(audio.samples.begin(), kBark3InputPrefixSamples, 0.f);
        auto result = inner_->run(padded);
        remove_bark3_prefix_timestamps(result.word_timestamps, original_samples);
        return result;
    }
private:
    std::unique_ptr<runtime::IOfflineVoiceTaskSession> inner_;
    bool prepared_ = false;
};

class BarkStreaming final : public runtime::IStreamingVoiceTaskSession {
public:
    explicit BarkStreaming(std::unique_ptr<runtime::IStreamingVoiceTaskSession> inner) : inner_(std::move(inner)) {
        if (!inner_) throw std::invalid_argument("Bark requires a streaming delegate");
        inner_->set_stream_event_sink([this](const runtime::StreamEvent & event) {
            if (sink_) {
                auto shifted = event;
                remove_bark3_prefix_timestamps(shifted.word_timestamps, received_);
                sink_(shifted);
            }
        });
    }
    std::string family() const override { return inner_->family(); }
    runtime::VoiceTaskKind task_kind() const override { return inner_->task_kind(); }
    runtime::RunMode run_mode() const override { return inner_->run_mode(); }
    runtime::StreamingPolicy streaming_policy() const override { return inner_->streaming_policy(); }
    void prepare(const runtime::SessionPreparationRequest & request) override {
        if (active_) throw std::logic_error("Cannot prepare an active Bark stream");
        prepared_ = false;
        inner_->prepare(preparation_with_prefix(request));
        prepared_ = true;
    }
    void set_stream_event_sink(runtime::StreamEventCallback sink) override { sink_ = std::move(sink); }
    void start_stream(const runtime::TaskRequest & request) override {
        if (!prepared_) throw std::logic_error("Bark requires prepare() before streaming");
        active_ = false;
        received_ = 0;
        validate_chunk_mode(request);
        inner_->start_stream(request);
        active_ = true;
    }
    void reset() override {
        active_ = false;
        received_ = 0;
        inner_->reset();
    }
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override {
        if (!active_) throw std::logic_error("Bark stream is not active");
        if (chunk.sample_rate != kSampleRate || chunk.channels != 1 || chunk.start_sample != received_)
            throw runtime::InvalidRequestError("Bark streaming requires contiguous mono 16 kHz chunks");
        if (chunk.samples.empty()) return {};
        auto padded = chunk;
        if (received_ == 0) {
            padded.samples.insert(padded.samples.begin(), kBark3InputPrefixSamples, 0.f);
        } else {
            padded.start_sample += kBark3InputPrefixSamples;
        }
        received_ += static_cast<int64_t>(chunk.samples.size());
        try {
            auto event = inner_->process_audio_chunk(padded);
            remove_bark3_prefix_timestamps(event.word_timestamps, received_);
            return event;
        } catch (...) {
            active_ = false;
            throw;
        }
    }
    std::optional<runtime::StreamEvent> next_stream_event() override {
        auto event = inner_->next_stream_event();
        if (event) remove_bark3_prefix_timestamps(event->word_timestamps, received_);
        return event;
    }
    runtime::TaskResult finalize() override {
        if (!active_) throw std::logic_error("Bark stream is not active");
        active_ = false;
        if (received_ == 0) throw runtime::InvalidRequestError("Bark requires nonempty streamed audio");
        auto result = inner_->finalize();
        remove_bark3_prefix_timestamps(result.word_timestamps, received_);
        return result;
    }
private:
    std::unique_ptr<runtime::IStreamingVoiceTaskSession> inner_;
    runtime::StreamEventCallback sink_;
    int64_t received_ = 0;
    bool prepared_ = false;
    bool active_ = false;
};
}  // namespace

std::unique_ptr<runtime::IOfflineVoiceTaskSession> wrap_bark3_offline(
    std::unique_ptr<runtime::IOfflineVoiceTaskSession> session) {
    return std::make_unique<BarkOffline>(std::move(session));
}
std::unique_ptr<runtime::IStreamingVoiceTaskSession> wrap_bark3_streaming(
    std::unique_ptr<runtime::IStreamingVoiceTaskSession> session) {
    return std::make_unique<BarkStreaming>(std::move(session));
}
}  // namespace engine::community_models::parakeet_tdt
