#include "engine/community_models/voice_isolator/session.h"

#include "engine/community_models/voice_isolator/assets.h"
#include "engine/community_models/voice_isolator/runtime.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/audio/streaming_resampler.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <chrono>
#include <stdexcept>

namespace engine::models::voice_isolator {
namespace {

constexpr int kSampleRate = 48000;
using Storage = assets::TensorStorageType;

Options parse_options(const std::unordered_map<std::string, std::string> & options) {
    Options out;
    out.lookahead_frames = runtime::parse_int_option(options, {"voice_isolator.lookahead_frames"}).value_or(-1);
    out.block_frames = runtime::parse_int_option(options, {"voice_isolator.block_frames"}).value_or(4);
    out.chunk_seconds =
        runtime::parse_finite_float_option(options, {"voice_isolator.audio_chunk_duration_sec"}).value_or(0.0F);
    out.overlap_seconds =
        runtime::parse_finite_float_option(options, {"voice_isolator.audio_chunk_overlap_sec"}).value_or(0.0F);
    out.weight_storage = runtime::parse_tensor_storage_option(options, "voice_isolator.weight_type", Storage::Native,
        {Storage::Native, Storage::F32, Storage::BF16});
    out.attention_cache_storage = runtime::parse_tensor_storage_option(options, "voice_isolator.attention_cache_type",
        Storage::Native, {Storage::Native, Storage::F32, Storage::F16, Storage::Q8_0, Storage::Q5_0, Storage::Q4_0});
    return out;
}

class Session final : public runtime::RuntimeSessionBase,
                      public runtime::IOfflineVoiceTaskSession,
                      public runtime::IStreamingVoiceTaskSession {
public:
    Session(runtime::TaskSpec task, runtime::SessionOptions options, std::shared_ptr<const Assets> assets,
        std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), contract_(std::move(contract)) {
        runtime::validate_spec_backed_session_options(options, *contract_, "voice_isolator", "Voice Isolator");
        if (task.task != runtime::VoiceTaskKind::SpeechToSpeech) {
            throw std::runtime_error("Voice Isolator supports the s2s task");
        }
        const bool streaming = task.mode == runtime::RunMode::Streaming;
        const auto checkpoint = open_checkpoint(*assets, streaming);
        const auto parsed = parse_options(options.options);
        block_frames_ = parsed.block_frames;
        model_ = Model::load(checkpoint.config_json, *checkpoint.tensors, execution_context(), parsed);
        if (streaming && !model_.streams()) {
            throw std::runtime_error(
                "This Voice Isolator checkpoint has no streaming mode: use the realtime checkpoint");
        }
    }

    std::string family() const override { return "voice_isolator"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }
    void prepare(const runtime::SessionPreparationRequest &) override { mark_prepared(); }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("Voice Isolator run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_,
            "Voice Isolator");
        if (!request.audio_input || request.audio_input->channels < 1 || request.audio_input->samples.empty()) {
            throw std::runtime_error("Voice Isolator requires nonempty input audio");
        }
        const auto started = std::chrono::steady_clock::now();
        const auto & input = *request.audio_input;
        auto mono = input.channels == 1 ? input.samples
            : audio::mixdown_interleaved_to_mono_average(input.samples, input.channels);
        if (input.sample_rate != kSampleRate) {
            mono = audio::resample_mono_soxr_or_linear(mono, input.sample_rate, kSampleRate, {});
        }
        auto output = model_.isolate_mono_48k(mono);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        runtime::TaskResult result;
        result.audio_output = runtime::AudioBuffer{output.sample_rate, 1, std::move(output.samples)};
        return result;
    }

    runtime::StreamingPolicy streaming_policy() const override {
        // One block of 10 ms frames, so a chunk runs as a single graph.
        return {runtime::StreamingInputKind::AudioChunks, runtime::StreamingOutputKind::FinalResult,
            block_frames_ * (kSampleRate / 100), 0.01 * static_cast<double>(block_frames_)};
    }

    void start_stream(const runtime::TaskRequest & request) override {
        require_prepared("Voice Isolator start_stream");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_,
            "Voice Isolator");
        started_ = std::chrono::steady_clock::now();
        reset();
        if (!stream_) {
            stream_ = model_.create_streaming_session();
        }
        stream_active_ = true;
    }

    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override {
        if (!stream_active_) {
            throw std::runtime_error("Voice Isolator stream has not started");
        }
        if (chunk.channels < 1 || chunk.sample_rate <= 0) {
            throw std::runtime_error("Voice Isolator streaming requires audio with a sample rate and channels");
        }
        if (stream_rate_ == 0) {
            stream_rate_ = chunk.sample_rate;
            if (stream_rate_ != kSampleRate) {
                resampler_ = std::make_unique<audio::StreamingSoxrResampler>(stream_rate_, kSampleRate);
            }
        }
        if (chunk.sample_rate != stream_rate_) {
            throw std::runtime_error("Voice Isolator sample rate changed within a stream");
        }
        auto mono = chunk.channels == 1 ? chunk.samples
            : audio::mixdown_interleaved_to_mono_average(chunk.samples, chunk.channels);
        runtime::StreamEvent event;
        auto output = stream_->push(resampler_ ? resampler_->process(mono) : mono);
        if (!output.empty()) {
            collected_.insert(collected_.end(), output.begin(), output.end());
            event.audio_output = runtime::AudioBuffer{kSampleRate, 1, std::move(output)};
        }
        return event;
    }

    void set_stream_event_sink(runtime::StreamEventCallback sink) override { sink_ = std::move(sink); }

    runtime::TaskResult finalize() override {
        if (!stream_active_) {
            throw std::runtime_error("Voice Isolator stream has not started");
        }
        std::vector<float> tail;
        if (resampler_) {
            tail = stream_->push(resampler_->process({}, true));
        }
        const auto rest = stream_->flush();
        tail.insert(tail.end(), rest.begin(), rest.end());
        collected_.insert(collected_.end(), tail.begin(), tail.end());
        if (sink_) {
            runtime::StreamEvent event;
            if (!tail.empty()) {
                event.audio_output = runtime::AudioBuffer{kSampleRate, 1, std::move(tail)};
            }
            event.is_final = true;
            sink_(event);
        }
        stream_active_ = false;
        resampler_.reset();
        stream_rate_ = 0;
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started_));
        runtime::TaskResult result;
        result.audio_output = runtime::AudioBuffer{kSampleRate, 1, std::move(collected_)};
        collected_.clear();
        return result;
    }

    void reset() override {
        stream_active_ = false;
        if (stream_) {
            stream_->reset();
        }
        resampler_.reset();
        stream_rate_ = 0;
        collected_.clear();
    }

private:
    runtime::TaskSpec task_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    Model model_;
    int64_t block_frames_ = 4;
    std::unique_ptr<StreamingSession> stream_;
    bool stream_active_ = false;
    // Streams at other rates are resampled to 48 kHz as they arrive.
    std::unique_ptr<audio::StreamingSoxrResampler> resampler_;
    int stream_rate_ = 0;
    runtime::StreamEventCallback sink_;
    std::vector<float> collected_;
    std::chrono::steady_clock::time_point started_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_voice_isolator_loader() {
    runtime::SpecBackedVoiceModelConfig<Assets> config;
    config.family = "voice_isolator";
    config.load_assets = load_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                                std::shared_ptr<const Assets> assets,
                                std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<Session>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::voice_isolator
