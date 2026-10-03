#include "engine/models/owsm_ctc/model.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace engine::models::owsm_ctc {
namespace {

class OWSMCTCV4Session final : public runtime::RuntimeSessionBase,
                            public runtime::IOfflineVoiceTaskSession {
public:
    OWSMCTCV4Session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options,
        std::shared_ptr<const OWSMCTCV4Assets> assets,
        std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), assets_(std::move(assets)), contract_(std::move(contract)) {
        runtime::validate_spec_backed_session_options(options, *contract_, "owsm_ctc", "OWSM-CTC v4");
        if (task.task != runtime::VoiceTaskKind::Asr || task.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("OWSM-CTC v4 requires an offline ASR session");
        }
        const auto type = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"owsm_ctc.weight_type"}).value_or("native"));
        weights_ = load_owsm_ctc_weights(*assets_, execution_context(), type);
        ctc_runtime_ = std::make_unique<OWSMCTCV4EBranchformerRuntime>(*assets_, *weights_, execution_context());
    }

    std::string family() const override { return "owsm_ctc"; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::Asr; }
    runtime::RunMode run_mode() const override { return runtime::RunMode::Offline; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(request.options, *contract_, "OWSM-CTC v4");
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("OWSM-CTC v4 run");
        runtime::validate_spec_backed_request_options(request.options, *contract_, "OWSM-CTC v4");
        if (!request.audio_input) {
            throw std::runtime_error("OWSM-CTC v4 requires audio input");
        }
        const auto started = std::chrono::steady_clock::now();
        auto language = runtime::find_option(request.options, {"language"}).value_or("eng");
        const auto target = runtime::find_option(request.options, {"target_language"});
        auto language_token = language == "auto" ? -1 : assets_->token_id("<" + language + ">");
        const auto task_token = target.has_value()
            ? assets_->token_id("<st_" + *target + ">")
            : assets_->token_id("<asr>");
        const auto & input = *request.audio_input;
        auto mono = audio::convert_interleaved_audio_to_mono_linear_resampled(
            input.samples, input.sample_rate, input.channels, input.sample_rate);
        if (input.sample_rate != 16000) {
            const auto resampled = audio::try_resample_mono_soxr(mono, input.sample_rate, 16000, {});
            if (!resampled) {
                throw std::runtime_error("OWSM-CTC v4 requires SOXR for resampling; provide 16 kHz audio");
            }
            mono = *resampled;
        }
        if (mono.empty()) {
            throw std::runtime_error("OWSM-CTC v4 audio input is empty");
        }

        const auto mode = audio::parse_audio_chunk_mode(request.options);
        if (mode != audio::AudioChunkMode::Auto && mode != audio::AudioChunkMode::Fixed &&
            mode != audio::AudioChunkMode::None) {
            throw std::runtime_error("OWSM-CTC v4 supports auto, fixed, and none audio chunking");
        }
        if (mode == audio::AudioChunkMode::None &&
            mono.size() > static_cast<size_t>(assets_->config.max_audio_samples)) {
            throw std::runtime_error("OWSM-CTC v4 audio exceeds 30 seconds; use audio_chunk_mode=auto");
        }
        if (language_token < 0) {
            language_token = assets_->token_id("<nolang>");
        }
        std::vector<int32_t> frames;
        if (mono.size() <= static_cast<size_t>(assets_->config.max_audio_samples) &&
            mode != audio::AudioChunkMode::Fixed) {
            frames = ctc_runtime_->frame_tokens(mono, language_token, task_token);
        } else {
            constexpr int64_t context_samples = 32000;
            constexpr int64_t buffer_samples = 480000;
            constexpr int64_t hop_samples = buffer_samples - 2 * context_samples;
            constexpr int64_t buffer_frames = 375;
            constexpr int64_t context_frames = 25;
            std::vector<float> padded(static_cast<size_t>(context_samples), 0.0F);
            padded.insert(padded.end(), mono.begin(), mono.end());
            padded.resize(padded.size() + context_samples, 0.0F);
            const auto spans = audio::plan_audio_chunks(static_cast<int64_t>(padded.size()),
                {buffer_samples, hop_samples});
            for (const auto & span : spans) {
                std::vector<float> window(static_cast<size_t>(buffer_samples));
                audio::copy_planar_chunk(window, padded, 1, static_cast<int64_t>(padded.size()),
                    span, {buffer_samples, hop_samples});
                auto tokens = ctc_runtime_->frame_tokens(window, language_token, task_token);
                frames.insert(frames.end(), tokens.begin() + context_frames,
                    tokens.begin() + buffer_frames - context_frames);
                if (span.valid_samples < buffer_samples) {
                    break;
                }
            }
            const auto wanted = static_cast<size_t>(std::nearbyint(static_cast<double>(mono.size()) / 1280.0));
            if (frames.size() > wanted) {
                frames.resize(wanted);
            }
        }
        std::vector<int32_t> tokens;
        int32_t previous = -1;
        for (const auto token : frames) {
            if (token != previous && token != assets_->config.blank_id) {
                tokens.push_back(token);
            }
            previous = token;
        }
        runtime::TaskResult result;
        result.text_output = runtime::Transcript{assets_->decode_visible(tokens), target.value_or(language)};
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

private:
    std::shared_ptr<const OWSMCTCV4Assets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<OWSMCTCV4Weights> weights_;
    std::unique_ptr<OWSMCTCV4EBranchformerRuntime> ctc_runtime_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_owsm_ctc_loader() {
    runtime::SpecBackedVoiceModelConfig<OWSMCTCV4Assets> config;
    config.family = "owsm_ctc";
    config.load_assets = load_owsm_ctc_assets;
    config.create_session = [](
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options,
        std::shared_ptr<const OWSMCTCV4Assets> assets,
        std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<OWSMCTCV4Session>(
            task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::owsm_ctc
