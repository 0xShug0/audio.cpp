#include "engine/models/canary_asr/model.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::canary_asr {
namespace {

class CanarySession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    CanarySession(const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const CanaryAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), assets_(std::move(assets)), contract_(std::move(contract)) {
        runtime::validate_spec_backed_session_options(options, *contract_, "canary_asr", "Canary ASR");
        if (task.task != runtime::VoiceTaskKind::Asr || task.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("Canary ASR requires an offline ASR session");
        }
        const auto type = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"canary_asr.weight_type"}).value_or("native"));
        weights_ = load_canary_weights(*assets_, execution_context(), type);
        runtime_ = std::make_unique<CanaryRuntime>(*assets_, *weights_, execution_context());
    }

    std::string family() const override { return "canary_asr"; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::Asr; }
    runtime::RunMode run_mode() const override { return runtime::RunMode::Offline; }
    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(request.options, *contract_, "Canary ASR");
        mark_prepared();
    }
    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("Canary run");
        runtime::validate_spec_backed_request_options(request.options, *contract_, "Canary ASR");
        if (!request.audio_input) {
            throw std::runtime_error("Canary ASR requires audio input");
        }
        const auto language = runtime::find_option(request.options, {"language"}).value_or("en");
        const auto target = runtime::find_option(request.options, {"target_language"}).value_or(language);
        for (const auto & code : {language, target}) {
            if (code != "en" && code != "de" && code != "es" && code != "fr") {
                throw std::runtime_error("Canary supports en, de, es and fr");
            }
        }
        const bool pnc = runtime::parse_bool_option(runtime::find_option(request.options, {"pnc"}).value_or("true"), "pnc");
        const std::vector<std::string> prompt_text = {"<|startofcontext|>", "<|startoftranscript|>",
            "<|emo:undefined|>", "<|" + language + "|>", "<|" + target + "|>", pnc ? "<|pnc|>" : "<|nopnc|>",
            "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>"};
        std::vector<int32_t> prompt;
        for (const auto & token : prompt_text) {
            prompt.push_back(assets_->special_token(token));
        }
        const auto & audio = *request.audio_input;
        const auto mono = audio::convert_interleaved_audio_to_mono_linear_resampled(audio.samples, audio.sample_rate, audio.channels, 16000);
        const auto mode = audio::parse_audio_chunk_mode(request.options);
        if (mode != audio::AudioChunkMode::Auto && mode != audio::AudioChunkMode::Fixed && mode != audio::AudioChunkMode::None) {
            throw std::runtime_error("Canary supports auto, fixed and none audio chunking");
        }
        const float seconds = audio::parse_audio_chunk_seconds_override(request.options).value_or(40.0f);
        if (!std::isfinite(seconds) || seconds <= 0.0f || seconds > 40.0f) {
            throw std::runtime_error("Canary audio_chunk_duration_sec must be positive and at most 40");
        }
        const int64_t chunk_samples = mode == audio::AudioChunkMode::None ? static_cast<int64_t>(mono.size())
            : static_cast<int64_t>(seconds * 16000);
        auto chunks = audio::plan_audio_chunks(static_cast<int64_t>(mono.size()), {chunk_samples, chunk_samples});
        // Per-feature sample variance needs at least two 10 ms frames in each chunk.
        if (chunks.size() > 1 && chunks.back().valid_samples < 320) {
            auto & tail = chunks.back();
            auto & previous = chunks[chunks.size() - 2];
            const int64_t transfer = 320 - tail.valid_samples;
            if (previous.valid_samples - transfer < 320) {
                throw std::runtime_error("Canary chunk duration is too short to retain a 20 ms final chunk");
            }
            previous.valid_samples -= transfer;
            tail.copy_start_sample -= transfer;
            tail.output_start_sample -= transfer;
            tail.valid_samples += transfer;
        }
        trace(debug::LogLevel::Info, "canary_asr", "chunks=" + std::to_string(chunks.size()) +
            " chunk_samples=" + std::to_string(chunk_samples));
        runtime::TaskResult result;
        result.text_output = runtime::Transcript{"", target};
        const int64_t max_tokens = runtime::parse_i64_option(request.options, {"max_tokens"}).value_or(0);
        if (max_tokens < 0) {
            throw std::runtime_error("Canary max_tokens must not be negative");
        }
        for (const auto & chunk : chunks) {
            const auto start = mono.begin() + chunk.copy_start_sample;
            const std::vector<float> samples(start, start + chunk.valid_samples);
            const auto tokens = runtime_->transcribe(samples, prompt, max_tokens);
            // NeMo AggregateTokenizer joins pieces across language vocabularies before replacing metaspace.
            std::string text;
            for (const auto id : tokens) {
                text += assets_->vocabulary.at(static_cast<size_t>(id)).text;
            }
            const std::string metaspace = "\xe2\x96\x81";
            for (size_t at = 0; (at = text.find(metaspace, at)) != std::string::npos; ++at) {
                text.replace(at, metaspace.size(), " ");
            }
            const auto first = text.find_first_not_of(' ');
            text = first == std::string::npos ? "" : text.substr(first);
            if (!result.text_output->text.empty() && !text.empty()) {
                result.text_output->text += ' ';
            }
            result.text_output->text += text;
            trace(debug::LogLevel::Info, "canary_asr", "chunk=" + std::to_string(chunk.index) +
                " samples=" + std::to_string(chunk.valid_samples) + " tokens=" + std::to_string(tokens.size()));
        }
        return result;
    }

private:
    std::shared_ptr<const CanaryAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<CanaryWeights> weights_;
    std::unique_ptr<CanaryRuntime> runtime_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_canary_asr_loader() {
    runtime::SpecBackedVoiceModelConfig<CanaryAssets> config;
    config.family = "canary_asr";
    config.load_assets = load_canary_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const CanaryAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<CanarySession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::canary_asr
