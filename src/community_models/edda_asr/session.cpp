#include "engine/community_models/edda_asr/model.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/partial_text.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace engine::community_models::edda_asr {
namespace {

constexpr const char * kFamily = "edda_asr";
constexpr int64_t kSampleRate = 16000;

class EddaSession final : public runtime::RuntimeSessionBase,
    public runtime::IOfflineVoiceTaskSession, public runtime::IStreamingVoiceTaskSession {
public:
    EddaSession(const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const EddaAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), assets_(std::move(assets)), contract_(std::move(contract)), mode_(task.mode) {
        runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "Edda");
        const auto type = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"edda_asr.weight_type", "weight_type"}).value_or("native"));
        const bool drafter = runtime::parse_bool_option(
            runtime::find_option(options.options, {"edda_asr.drafter", "drafter"}).value_or("true"), "drafter");
        weights_ = load_weights(*assets_, execution_context(), type, drafter);
        runtime_ = std::make_unique<EddaRuntime>(*assets_, *weights_, execution_context());
        trace(debug::LogLevel::Info, kFamily, std::string("drafter=") +
            (weights_->drafter ? "loaded" : assets_->drafter ? "disabled" : "absent"));
    }
    std::string family() const override { return kFamily; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::Asr; }
    runtime::RunMode run_mode() const override { return mode_; }
    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(request.options, *contract_, "Edda");
        mark_prepared();
    }
    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("Edda run");
        begin(request);
        while (next_chunk_ < spans_.size()) {
            decode_chunk();
        }
        return finalize();
    }
    runtime::StreamingPolicy streaming_policy() const override {
        runtime::StreamingPolicy policy;
        policy.input = runtime::StreamingInputKind::None;
        policy.output = runtime::StreamingOutputKind::PullEvents;
        return policy;
    }
    void start_stream(const runtime::TaskRequest & request) override {
        require_prepared("Edda start_stream");
        begin(request);
    }
    std::optional<runtime::StreamEvent> next_stream_event() override {
        if (!started_) {
            throw std::runtime_error("Edda stream has not started");
        }
        if (next_chunk_ == spans_.size()) {
            return std::nullopt;
        }
        decode_chunk();
        runtime::StreamEvent event;
        auto delta = publisher_.publish(result_.text_output->text);
        if (!delta.empty()) {
            event.partial_text = runtime::Transcript{std::move(delta), language_};
        }
        return event;
    }
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk &) override {
        throw std::runtime_error("Edda streaming requires complete audio; output is committed per audio chunk");
    }
    runtime::TaskResult finalize() override {
        if (!started_ || next_chunk_ != spans_.size()) {
            throw std::runtime_error("Edda finalize requires all audio chunks to finish");
        }
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(start_time_));
        auto result = std::move(result_);
        reset();
        return result;
    }
    void reset() override {
        started_ = false;
        next_chunk_ = 0;
        audio_.clear();
        spans_.clear();
        publisher_.reset();
        result_ = {};
    }

private:
    void begin(const runtime::TaskRequest & request) {
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Edda");
        if (!request.audio_input) {
            throw std::runtime_error("Edda requires audio input");
        }
        reset();
        start_time_ = std::chrono::steady_clock::now();
        language_ = runtime::find_option(request.options, {"language"}).value_or("da");
        prompt_ = assets_->prompt(language_);
        options_ = {};
        options_.max_tokens = runtime::parse_positive_i64_option(request.options, {"max_tokens"},
            assets_->max_positions - static_cast<int64_t>(prompt_.size()));
        options_.speculative = runtime::parse_bool_option(
            runtime::find_option(request.options, {"speculative"}).value_or("true"), "speculative");
        options_.use_selector = runtime::parse_bool_option(
            runtime::find_option(request.options, {"draft_selector"}).value_or("true"), "draft_selector");
        options_.block = runtime::parse_positive_i64_option(request.options, {"draft_block"},
            assets_->drafter ? assets_->drafter->block : 16);
        options_.num_beams = runtime::parse_positive_i64_option(request.options, {"num_beams"}, 1);
        options_.length_penalty = runtime::parse_finite_float_option(request.options, {"length_penalty"}).value_or(1.0f);
        if (assets_->drafter && (options_.block < 2 || options_.block > assets_->drafter->block)) {
            throw std::runtime_error("Edda draft_block must be between 2 and " + std::to_string(assets_->drafter->block));
        }
        const auto & audio = *request.audio_input;
        audio_ = audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
        if (audio.sample_rate != kSampleRate) {
            audio::SoxrResampleOptions resample;
            resample.output_length_policy = audio::SoxrOutputLengthPolicy::ExactExpected;
            resample.require_full_input = true;
            auto resampled = audio::try_resample_mono_soxr(audio_, audio.sample_rate, kSampleRate, resample);
            if (!resampled) {
                throw std::runtime_error("Edda needs SOXR for non-16-kHz input");
            }
            audio_ = std::move(*resampled);
        }
        if (audio_.empty()) {
            audio_.resize(kSampleRate / 2, 0.0f);
        }
        const auto chunk_mode = audio::parse_audio_chunk_mode(request.options);
        const double chunk_seconds = runtime::parse_finite_float_option(
            request.options, {"audio_chunk_duration_sec"}).value_or(30.0F);
        if (chunk_seconds < 1.0 || chunk_seconds > 30.0) {
            throw std::runtime_error("Edda audio_chunk_duration_sec must be between 1 and 30 seconds");
        }
        const auto window = static_cast<int64_t>(std::llround(chunk_seconds * kSampleRate));
        const auto samples = static_cast<int64_t>(audio_.size());
        if (chunk_mode == audio::AudioChunkMode::Auto || chunk_mode == audio::AudioChunkMode::Silence) {
            const auto trigger = chunk_mode == audio::AudioChunkMode::Auto ? 30 * kSampleRate : window;
            spans_ = audio::plan_silence_audio_chunks(audio_, {window, trigger});
        } else if (chunk_mode == audio::AudioChunkMode::Fixed) {
            for (const auto & span : audio::plan_audio_chunks(audio_.size(), {window, window})) {
                spans_.push_back({span.output_start_sample, span.output_start_sample + span.valid_samples});
            }
        } else if (chunk_mode == audio::AudioChunkMode::None) {
            if (samples > 30 * kSampleRate) {
                throw std::runtime_error("Edda audio_chunk_mode=none requires audio of at most 30 seconds");
            }
            spans_.push_back({0, samples});
        } else {
            throw std::runtime_error("Edda audio_chunk_mode must be auto, silence, fixed, or none");
        }
        result_.text_output = runtime::Transcript{"", language_};
        started_ = true;
        trace(debug::LogLevel::Info, kFamily, "language=" + language_ + " chunks=" + std::to_string(spans_.size()) +
            " speculative=" + (options_.speculative && weights_->drafter && options_.num_beams == 1 ? "true" : "false") +
            " num_beams=" + std::to_string(options_.num_beams));
    }
    void decode_chunk() {
        const auto [begin, end] = spans_.at(next_chunk_);
        const std::vector<float> samples(audio_.begin() + begin, audio_.begin() + end);
        const auto decoded = runtime_->transcribe(samples, prompt_, options_);
        const auto text = io::trim_ascii_whitespace(assets_->decode_text(decoded.tokens));
        if (!text.empty()) {
            if (!result_.text_output->text.empty()) {
                result_.text_output->text += ' ';
            }
            result_.text_output->text += text;
        }
        const auto rate = decoded.drafted > 0 ? static_cast<double>(decoded.tokens.size() + 1) / decoded.target_passes : 1.0;
        trace(debug::LogLevel::Info, kFamily, "chunk=" + std::to_string(next_chunk_) +
            " tokens=" + std::to_string(decoded.tokens.size()) + " passes=" + std::to_string(decoded.target_passes) +
            " tokens_per_pass=" + std::to_string(rate) + " accepted=" + std::to_string(decoded.accepted) + "/" +
            std::to_string(decoded.drafted) + " eos=" + (decoded.reached_eos ? "true" : "false"));
        debug::timing_log_scalar("edda.encoder_ms", decoded.encoder_ms);
        debug::timing_log_scalar("edda.decoder_ms", decoded.decoder_ms);
        debug::timing_log_scalar("edda.drafter_ms", decoded.drafter_ms);
        ++next_chunk_;
    }

    std::shared_ptr<const EddaAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<EddaWeights> weights_;
    std::unique_ptr<EddaRuntime> runtime_;
    runtime::RunMode mode_;
    std::string language_;
    std::vector<int32_t> prompt_;
    DecodeOptions options_;
    std::vector<float> audio_;
    std::vector<runtime::TimeSpan> spans_;
    runtime::PartialTextPublisher publisher_;
    runtime::TaskResult result_;
    size_t next_chunk_ = 0;
    bool started_ = false;
    std::chrono::steady_clock::time_point start_time_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_edda_asr_loader() {
    runtime::SpecBackedVoiceModelConfig<EddaAssets> config;
    config.family = kFamily;
    config.load_assets = load_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const EddaAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
        if (task.task != runtime::VoiceTaskKind::Asr) {
            throw std::runtime_error("Edda supports ASR");
        }
        return std::make_unique<EddaSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::community_models::edda_asr
