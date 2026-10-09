#include "engine/models/index_echo/text_translation.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/io/text.h"
#include "engine/framework/sampling/hf_sampler.h"
#include "engine/framework/tokenizers/llama_bpe.h"
#include "engine/framework/modules/transformers/qwen35_decoder_runtime.h"
#include "engine/models/index_echo/connector.h"
#include "engine/framework/modules/speech_encoders/qwen3_audio_encoder_runtime.h"

#include <algorithm>
#include <chrono>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::index_echo {
namespace {

using Clock = std::chrono::steady_clock;

std::string instruction(std::string_view language) {
    if (language == "en") {
        return "For each sentence, output three lines: the [MM:SS.CC-MM:SS.CC] timestamp, the transcript, then the English translation.";
    }
    if (language == "es") {
        return "For each sentence, output three lines: the [MM:SS.CC-MM:SS.CC] timestamp, the transcript, then the Spanish translation.";
    }
    if (language == "ja") {
        return "For each sentence, output three lines: the [MM:SS.CC-MM:SS.CC] timestamp, the transcript, then the Japanese translation.";
    }
    throw std::runtime_error("Index-Echo-S2TT target_language must be en, es, or ja");
}

std::string dubbing_instruction(std::string_view language) {
    if (language == "en") {
        return "把这段语音翻译成英文，先输出原文，换行输出英文译文。";
    }
    if (language == "es") {
        return "把这段语音翻译成西班牙语，先输出原文，换行输出西班牙语译文。";
    }
    if (language == "ja") {
        return "把这段语音翻译成日语，先输出原文，换行输出日语译文。";
    }
    if (language == "zh") {
        return "把这段语音翻译成中文，先输出原文，换行输出中文译文。";
    }
    throw std::runtime_error("Index-Echo-S2ST target_language must be en, es, ja, or zh");
}

std::string prompt_text(
    int64_t audio_tokens,
    std::string_view target_language,
    std::string_view context,
    std::string_view glossary) {
    std::string audio_block = "<|audio_start|>";
    for (int64_t i = 0; i < audio_tokens; ++i) {
        audio_block += "<|audio_pad|>";
    }
    audio_block += "<|audio_end|>\n";
    if (!context.empty()) {
        audio_block += "[Context]\n";
        audio_block += context;
        audio_block += "\n\n";
    }
    if (!glossary.empty()) {
        audio_block += "[Glossary]\n";
        audio_block += glossary;
        audio_block += "\n\n";
    }
    return "<|im_start|>user\n" + audio_block + instruction(target_language) +
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

std::string dubbing_prompt_text(int64_t audio_tokens, std::string_view target_language) {
    std::string audio_block = "<|audio_start|>";
    for (int64_t i = 0; i < audio_tokens; ++i) {
        audio_block += "<|audio_pad|>";
    }
    audio_block += "<|audio_end|>";
    return "<|im_start|>user\n" + audio_block + "\n" + dubbing_instruction(target_language) +
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

int32_t required_token_id(const tokenizers::LlamaBpeTokenizer & tokenizer, const char * token) {
    const auto id = tokenizer.find_token_id(token);
    if (!id.has_value()) {
        throw std::runtime_error(std::string("Index-Echo tokenizer is missing ") + token);
    }
    return *id;
}

}  // namespace

class IndexEchoQwen3OmniAuTQwen35TranslationRuntime::Impl {
public:
    Impl(std::shared_ptr<const IndexEchoAssets> assets,
         core::ExecutionContext & execution,
         assets::TensorStorageType storage_type)
        : assets_(std::move(assets)),
          frontend_({16000, 400, 160, 128, audio::STFTFamily::Default}),
          audio_encoder_(assets_->qwen3_omni_audio->model_weights,
              assets_->qwen3_omni_audio->config.audio_encoder,
              execution, 128ull * 1024ull * 1024ull, storage_type,
              assets_->qwen3_omni_audio->config.hf_transformers_layout),
          connector_(assets_, execution, storage_type),
          decoder_config_([&] {
              auto config = assets_->qwen35_config;
              config.round_bf16_activations = storage_type == assets::TensorStorageType::Native &&
                  assets::tensor_storage_type_for_dtype(assets_->qwen35_weights->require_metadata(
                      config.weight_prefix + ".embed_tokens.weight").dtype) == assets::TensorStorageType::BF16;
              return config;
          }()),
          qwen35_(assets_->qwen35_weights, decoder_config_, execution, 512ull * 1024ull * 1024ull,
                  128ull * 1024ull * 1024ull, storage_type),
          dubbing_(assets_->resources.has_file("mapper_config")) {
        tokenizers::LlamaBpeTokenizerSpec spec;
        spec.tokenizer_json_path = assets_->resources.require_file("tokenizer");
        spec.tokenizer_config_path = assets_->resources.require_file("tokenizer_config");
        spec.pre_type = tokenizers::LlamaBpePreTokenizer::Qwen35;
        tokenizer_ = tokenizers::load_llama_bpe_tokenizer(spec);
        audio_pad_id_ = required_token_id(*tokenizer_, "<|audio_pad|>");
        im_end_id_ = required_token_id(*tokenizer_, "<|im_end|>");
        eos_id_ = required_token_id(*tokenizer_, "<|endoftext|>");
    }

    std::string translate_window(
        const runtime::AudioBuffer & audio,
        std::string_view target_language,
        std::string_view context,
        std::string_view glossary,
        int64_t max_new_tokens,
        float temperature,
        uint32_t seed) {
        if (max_new_tokens <= 0) {
            throw std::runtime_error("Index-Echo max_text_tokens must be positive");
        }
        const auto audio_started = Clock::now();
        if (audio.samples.empty() || audio.sample_rate <= 0 || audio.channels <= 0) {
            throw std::runtime_error("Index-Echo requires non-empty audio with valid format");
        }
        const auto mono = audio::convert_interleaved_audio_to_mono_linear_resampled(
            audio.samples, audio.sample_rate, audio.channels, 16000);
        const auto mel = frontend_.compute(mono);
        qwen3_asr::Qwen3ASRAudioFeatures features;
        features.values = std::move(mel.values);
        features.frames = mel.frames;
        features.mel_bins = mel.mel_bins;
        features.attention_mask.assign(static_cast<size_t>(features.frames), 1);
        features.encoder_tokens = modules::qwen3_audio_encoder_token_count(features.frames);
        const auto encoded = audio_encoder_.encode(features);
        auto connected = connector_.connect(encoded);
        if (decoder_config_.round_bf16_activations) {
            core::round_f32_to_bf16_in_place(connected.values);
        }
        engine::debug::timing_log_scalar("index_echo.audio_conditioning_ms", engine::debug::elapsed_ms(audio_started));

        const auto token_ids = tokenizer_->encode(
            dubbing_ ? dubbing_prompt_text(connected.tokens, target_language)
                     : prompt_text(connected.tokens, target_language, context, glossary),
            true);
        int64_t audio_positions = 0;
        auto embeddings = qwen35_.token_embedding(token_ids);
        const size_t hidden = static_cast<size_t>(connected.hidden_size);
        for (size_t i = 0; i < token_ids.size(); ++i) {
            if (token_ids[i] != audio_pad_id_) {
                continue;
            }
            std::copy_n(
                connected.values.begin() + static_cast<std::ptrdiff_t>(audio_positions * connected.hidden_size),
                hidden,
                embeddings.begin() + static_cast<std::ptrdiff_t>(i * hidden));
            ++audio_positions;
        }
        if (audio_positions != connected.tokens) {
            throw std::runtime_error("Index-Echo prompt audio slots do not match encoder output");
        }
        const int64_t prompt_steps = static_cast<int64_t>(token_ids.size());
        if (prompt_steps + max_new_tokens + 1 > decoder_config_.decode_cache_steps) {
            throw std::runtime_error("Index-Echo prompt and max_text_tokens exceed decoder cache capacity");
        }
        auto decode = qwen35_.create_decode_session(prompt_steps + max_new_tokens + 1);
        const auto prefill_started = Clock::now();
        auto prefill = decode->prefill_embeddings(embeddings, prompt_steps);
        std::vector<float> hidden_state(
            prefill.hidden.end() - static_cast<std::ptrdiff_t>(hidden), prefill.hidden.end());
        engine::debug::timing_log_scalar("index_echo.text_prefill_ms", engine::debug::elapsed_ms(prefill_started));

        const auto decode_started = Clock::now();
        std::vector<int32_t> generated;
        std::mt19937 rng(seed);
        sampling::HfSampler sampler;
        sampling::HfSamplerScratch scratch;
        sampling::HfSamplingOptions sampling_options;
        sampling_options.do_sample = temperature > 0.0F;
        if (sampling_options.do_sample) {
            sampling_options.temperature = temperature;
        }
        for (int64_t step = 0; step < max_new_tokens; ++step) {
            const auto logits = qwen35_.lm_head(hidden_state);
            const int32_t token = sampler.sample(
                logits, generated, sampling_options, scratch, rng, nullptr,
                "Index-Echo text sampler");
            if (token == im_end_id_ || token == eos_id_) {
                break;
            }
            generated.push_back(token);
            hidden_state = decode->run_embedding_step(qwen35_.token_embedding({token}));
        }
        engine::debug::timing_log_scalar("index_echo.text_decode_ms", engine::debug::elapsed_ms(decode_started));
        return engine::io::trim_ascii_whitespace(tokenizer_->decode(generated, true));
    }

private:
    std::shared_ptr<const IndexEchoAssets> assets_;
    audio::WhisperLogMelExtractor frontend_;
    modules::Qwen3AudioEncoderRuntime audio_encoder_;
    IndexEchoAudioConnectorRuntime connector_;
    modules::Qwen35DecoderConfig decoder_config_;
    modules::Qwen35DecoderRuntime qwen35_;
    std::shared_ptr<tokenizers::LlamaBpeTokenizer> tokenizer_;
    int32_t audio_pad_id_ = 0;
    int32_t im_end_id_ = 0;
    int32_t eos_id_ = 0;
    bool dubbing_ = false;
};

IndexEchoQwen3OmniAuTQwen35TranslationRuntime::IndexEchoQwen3OmniAuTQwen35TranslationRuntime(
    std::shared_ptr<const IndexEchoAssets> assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, weight_storage_type)) {}

IndexEchoQwen3OmniAuTQwen35TranslationRuntime::~IndexEchoQwen3OmniAuTQwen35TranslationRuntime() = default;

std::string IndexEchoQwen3OmniAuTQwen35TranslationRuntime::translate_window(
    const runtime::AudioBuffer & audio,
    std::string_view target_language,
    std::string_view context,
    std::string_view glossary,
    int64_t max_new_tokens,
    float temperature,
    uint32_t seed) {
    return impl_->translate_window(audio, target_language, context, glossary, max_new_tokens, temperature, seed);
}

}  // namespace engine::models::index_echo
