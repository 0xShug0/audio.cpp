#include "engine/models/micro_wake_word/session.h"

#include "engine/models/micro_wake_word/frontend.h"
#include "engine/models/micro_wake_word/runtime.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <stdexcept>

namespace engine::models::micro_wake_word {
namespace {

struct Assets {
    assets::ResourceBundle resources;
    io::json::Value config;
};

class Session final : public runtime::RuntimeSessionBase,
                      public runtime::IOfflineVoiceTaskSession,
                      public runtime::IStreamingVoiceTaskSession {
public:
    Session(const runtime::SessionOptions & options, runtime::RunMode mode,
            std::shared_ptr<const Assets> assets,
            std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), mode_(mode), assets_(std::move(assets)),
          contract_(std::move(contract)),
          network_(assets_->resources.open_tensor_source("weights"), assets_->config,
                   execution_context()) {}

    std::string family() const override { return "micro_wake_word"; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::WakeWord; }
    runtime::RunMode run_mode() const override { return mode_; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(
            request.options, request.option_arrays, *contract_, "microWakeWord");
        if (!request.audio || request.audio->sample_rate != 16000 || request.audio->channels != 1) {
            throw std::runtime_error("microWakeWord requires 16000 Hz mono audio");
        }
        configure(request.options);
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("microWakeWord run");
        runtime::validate_spec_backed_request_options(
            request.options, request.option_arrays, *contract_, "microWakeWord");
        if (!request.audio_input) throw std::runtime_error("microWakeWord requires audio_input");
        configure(request.options);
        reset_state();
        const auto started = std::chrono::steady_clock::now();
        append_audio(request.audio_input->samples, 0);
        process_available();
        runtime::TaskResult result;
        result.speech_segments = detections_;
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

    runtime::StreamingPolicy streaming_policy() const override {
        return {runtime::StreamingInputKind::AudioChunks,
                runtime::StreamingOutputKind::PullEvents, 480, 0.0};
    }

    void start_stream(const runtime::TaskRequest & request) override {
        require_prepared("microWakeWord stream");
        runtime::validate_spec_backed_request_options(
            request.options, request.option_arrays, *contract_, "microWakeWord");
        configure(request.options);
        reset_state();
        stream_started_ = true;
    }

    void reset() override { reset_state(); }

    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override {
        require_prepared("microWakeWord stream");
        if (!stream_started_) throw std::runtime_error("microWakeWord stream was not started");
        if (chunk.sample_rate != 16000 || chunk.channels != 1) {
            throw std::runtime_error("microWakeWord requires 16000 Hz mono audio chunks");
        }
        if (chunk.start_sample != received_samples_) {
            throw std::runtime_error("microWakeWord requires contiguous streaming audio chunks");
        }
        const auto started = std::chrono::steady_clock::now();
        const size_t before = detections_.size();
        append_audio(chunk.samples, chunk.start_sample);
        process_available();
        runtime::StreamEvent event;
        for (size_t i = before; i < detections_.size(); ++i) {
            runtime::VoiceActivityEvent wake;
            wake.kind = runtime::VoiceActivityEvent::Kind::SpeechSegment;
            wake.sample = detections_[i].span.end_sample;
            wake.probability = detections_[i].confidence;
            wake.segment = detections_[i];
            event.voice_activity.push_back(std::move(wake));
        }
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return event;
    }

    runtime::TaskResult finish_stream() override {
        if (!stream_started_) throw std::runtime_error("microWakeWord stream was not started");
        stream_started_ = false;
        runtime::TaskResult result;
        result.speech_segments = detections_;
        return result;
    }

    runtime::TaskResult finalize() override { return finish_stream(); }

private:
    void configure(const std::unordered_map<std::string, std::string> & options) {
        threshold_ = runtime::parse_finite_float_option(options, {"threshold"})
                         .value_or(network_.config().probability_cutoff);
        smoothing_window_ = runtime::parse_int_option(options, {"sliding_window_size"})
                                .value_or(network_.config().sliding_window_size);
        if (threshold_ < 0.0F || threshold_ > 1.0F || smoothing_window_ < 1) {
            throw std::runtime_error("invalid microWakeWord threshold or sliding window size");
        }
    }

    void reset_state() {
        frontend_.reset();
        network_.reset();
        pending_audio_.clear();
        pending_features_.clear();
        probabilities_.clear();
        detections_.clear();
        received_samples_ = 0;
        processed_samples_ = 0;
        active_ = false;
        stream_started_ = false;
    }

    void append_audio(const std::vector<float> & samples, int64_t start_sample) {
        if (start_sample != received_samples_) {
            throw std::runtime_error("microWakeWord audio is not contiguous");
        }
        if (!std::all_of(samples.begin(), samples.end(), [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error("microWakeWord requires finite audio samples");
        }
        pending_audio_.insert(pending_audio_.end(), samples.begin(), samples.end());
        received_samples_ += static_cast<int64_t>(samples.size());
    }

    void process_available() {
        constexpr size_t hop = 160;
        std::vector<int16_t> pcm(hop);
        size_t consumed = 0;
        // Keep the same one-sample lookahead as the upstream feature generator.
        while (pending_audio_.size() - consumed > hop) {
            for (size_t i = 0; i < hop; ++i) {
                const float scaled = std::clamp(pending_audio_[consumed + i] * 32768.0F,
                                                -32768.0F, 32767.0F);
                pcm[i] = static_cast<int16_t>(scaled);
            }
            auto row = frontend_.process_hop(pcm.data(), pcm.size());
            consumed += hop;
            processed_samples_ += static_cast<int64_t>(hop);
            if (!row.empty()) {
                pending_features_.insert(pending_features_.end(), row.begin(), row.end());
                const size_t required = static_cast<size_t>(network_.config().input_frames * 40);
                if (pending_features_.size() == required) {
                    observe(network_.infer(pending_features_));
                    pending_features_.clear();
                } else if (pending_features_.size() > required) {
                    throw std::runtime_error("microWakeWord frontend/model frame cadence is inconsistent");
                }
            }
        }
        pending_audio_.erase(pending_audio_.begin(), pending_audio_.begin() + static_cast<std::ptrdiff_t>(consumed));
    }

    void observe(float probability) {
        probabilities_.push_back(probability);
        while (probabilities_.size() > static_cast<size_t>(smoothing_window_)) probabilities_.pop_front();
        if (probabilities_.size() != static_cast<size_t>(smoothing_window_)) return;
        double sum = 0.0;
        for (float value : probabilities_) sum += value;
        const float average = static_cast<float>(sum / probabilities_.size());
        if (average > threshold_ && !active_) {
            runtime::SpeechSegment detection;
            detection.span.end_sample = processed_samples_;
            detection.span.start_sample = std::max<int64_t>(
                0, processed_samples_ - static_cast<int64_t>(network_.config().input_frames) * 160);
            detection.confidence = average;
            detection.text = network_.config().wake_phrase;
            detections_.push_back(std::move(detection));
            active_ = true;
        } else if (average <= threshold_) {
            active_ = false;
        }
    }

    runtime::RunMode mode_;
    std::shared_ptr<const Assets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    MicroSpeechFrontend frontend_;
    Runtime network_;
    std::vector<float> pending_audio_;
    std::vector<float> pending_features_;
    std::deque<float> probabilities_;
    std::vector<runtime::SpeechSegment> detections_;
    int64_t received_samples_ = 0;
    int64_t processed_samples_ = 0;
    float threshold_ = 0.5F;
    int smoothing_window_ = 5;
    bool active_ = false;
    bool stream_started_ = false;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_micro_wake_word_loader() {
    runtime::SpecBackedVoiceModelConfig<Assets> config;
    config.family = "micro_wake_word";
    config.load_assets = [](const std::filesystem::path & path) {
        auto result = std::make_shared<Assets>();
        result->resources = model_spec::load_resource_bundle_for_family(path, "micro_wake_word");
        result->config = result->resources.parse_json("config");
        return result;
    };
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const Assets> assets,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        if (task.task != runtime::VoiceTaskKind::WakeWord) {
            throw std::runtime_error("microWakeWord supports the wake-word task");
        }
        runtime::validate_spec_backed_session_options(
            options, *contract, "micro_wake_word", "microWakeWord");
        return std::make_unique<Session>(options, task.mode, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::micro_wake_word
