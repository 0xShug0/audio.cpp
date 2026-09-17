#include "engine/models/cohere_asr/model.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::cohere_asr {
namespace {

class CohereSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    CohereSession(const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const CohereAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), assets_(std::move(assets)), contract_(std::move(contract)) {
        runtime::validate_spec_backed_session_options(options, *contract_, "cohere_asr", "Cohere ASR");
        if (task.task != runtime::VoiceTaskKind::Asr || task.mode != runtime::RunMode::Offline) {
            throw std::runtime_error("Cohere ASR requires an offline ASR session");
        }
        const auto type = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"cohere_asr.weight_type"}).value_or("native"));
        weights_ = load_cohere_weights(*assets_, execution_context(), type);
        runtime_ = std::make_unique<CohereRuntime>(*assets_, *weights_, execution_context());
    }

    std::string family() const override { return "cohere_asr"; }
    runtime::VoiceTaskKind task_kind() const override { return runtime::VoiceTaskKind::Asr; }
    runtime::RunMode run_mode() const override { return runtime::RunMode::Offline; }
    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(request.options, *contract_, "Cohere ASR");
        mark_prepared();
    }
    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("Cohere run");
        runtime::validate_spec_backed_request_options(request.options, *contract_, "Cohere ASR");
        if (!request.audio_input) {
            throw std::runtime_error("Cohere ASR requires audio input");
        }
        const auto language = runtime::find_option(request.options, {"language"}).value_or("en");
        const std::vector<std::string> languages = {"en", "fr", "de", "es", "it", "pt", "nl", "pl", "el", "ar", "ja", "zh", "vi", "ko"};
        if (std::find(languages.begin(), languages.end(), language) == languages.end()) {
            throw std::runtime_error("Unsupported Cohere transcription language: " + language);
        }
        const bool pnc = runtime::parse_bool_option(runtime::find_option(request.options, {"pnc"}).value_or("true"), "pnc");
        const std::vector<std::string> prompt_text = {"<|startofcontext|>", "<|startoftranscript|>",
            "<|emo:undefined|>", "<|" + language + "|>", "<|" + language + "|>", pnc ? "<|pnc|>" : "<|nopnc|>",
            "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>"};
        std::string prompt_string;
        for (const auto & token : prompt_text) {
            prompt_string += token;
        }
        const auto prompt = tokenizers::tokenize_sentencepiece(assets_->vocabulary, prompt_string);
        const auto & input_audio = *request.audio_input;
        const auto mono = audio::convert_interleaved_audio_to_mono_linear_resampled(
            input_audio.samples, input_audio.sample_rate, input_audio.channels, 16000);
        const auto mode = audio::parse_audio_chunk_mode(request.options);
        const float seconds = audio::parse_audio_chunk_seconds_override(request.options).value_or(35.0f);
        if (!std::isfinite(seconds) || seconds < 0.04f || seconds > 35.0f) {
            throw std::runtime_error("Cohere audio_chunk_duration_sec must be between 0.04 and 35");
        }
        const int64_t chunk_samples = static_cast<int64_t>(seconds * 16000);
        std::vector<runtime::TimeSpan> chunks;
        if (mode == audio::AudioChunkMode::Auto || mode == audio::AudioChunkMode::QuietEnergy) {
            chunks = audio::plan_quiet_energy_audio_chunks(mono, {chunk_samples, std::min<int64_t>(80000, chunk_samples / 2), 1600});
        } else if (mode == audio::AudioChunkMode::Fixed || mode == audio::AudioChunkMode::None) {
            const int64_t size = mode == audio::AudioChunkMode::None ? static_cast<int64_t>(mono.size()) : chunk_samples;
            for (const auto & chunk : audio::plan_audio_chunks(static_cast<int64_t>(mono.size()), {size, size})) {
                chunks.push_back({chunk.copy_start_sample, chunk.copy_start_sample + chunk.valid_samples});
            }
        } else {
            throw std::runtime_error("Cohere supports auto, quiet_energy, fixed and none audio chunking");
        }
        // Retain a tiny final tail without producing a chunk too short for sample variance.
        if (chunks.size() > 1 && chunks.back().end_sample - chunks.back().start_sample < 320) {
            auto & tail = chunks.back();
            auto & previous = chunks[chunks.size() - 2];
            const int64_t transfer = 320 - (tail.end_sample - tail.start_sample);
            if (previous.end_sample - previous.start_sample - transfer < 320) {
                throw std::runtime_error("Cohere chunk duration leaves less than 20 ms in the final chunks");
            }
            tail.start_sample -= transfer;
            previous.end_sample -= transfer;
        }
        const int64_t max_tokens = runtime::parse_i64_option(request.options, {"max_tokens"}).value_or(256);
        if (max_tokens < 1 || max_tokens > 1014) {
            throw std::runtime_error("Cohere max_tokens must be between 1 and 1014");
        }
        trace(debug::LogLevel::Info, "cohere_asr", "chunks=" + std::to_string(chunks.size()));
        runtime::TaskResult result;
        result.text_output = runtime::Transcript{"", language};
        for (size_t start = 0; start < chunks.size(); start += 8) {
            const size_t count = std::min<size_t>(8, chunks.size() - start);
            std::vector<std::vector<float>> batch;
            for (size_t i = 0; i < count; ++i) {
                const auto & span = chunks[start + i];
                batch.emplace_back(mono.begin() + span.start_sample, mono.begin() + span.end_sample);
            }
            auto generated = runtime_->transcribe(batch, prompt, max_tokens);
            for (size_t i = 0; i < count; ++i) {
                const auto & chunk = chunks[start + i];
                auto & tokens = generated[i];
                tokens.erase(std::remove_if(tokens.begin(), tokens.end(), [&](int32_t id) {
                    const auto & piece = assets_->vocabulary.at(static_cast<size_t>(id));
                    return piece.type == tokenizers::SentencePieceType::Control ||
                        piece.type == tokenizers::SentencePieceType::Unknown ||
                        piece.text == "<pad>" ||
                        (piece.text.size() >= 4 && piece.text.compare(0, 2, "<|") == 0);
                }), tokens.end());
                const auto text = tokenizers::decode_sentencepiece(assets_->vocabulary, tokens);
                if (!result.text_output->text.empty() && !text.empty() && language != "ja" && language != "zh") {
                    result.text_output->text += ' ';
                }
                result.text_output->text += text;
                trace(debug::LogLevel::Info, "cohere_asr", "start=" + std::to_string(chunk.start_sample) +
                    " end=" + std::to_string(chunk.end_sample) + " tokens=" + std::to_string(tokens.size()));
            }
        }
        return result;
    }

private:
    std::shared_ptr<const CohereAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<CohereWeights> weights_;
    std::unique_ptr<CohereRuntime> runtime_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_cohere_asr_loader() {
    runtime::SpecBackedVoiceModelConfig<CohereAssets> config;
    config.family = "cohere_asr";
    config.load_assets = load_cohere_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const CohereAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<CohereSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::cohere_asr
