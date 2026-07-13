#include "engine/models/chatterbox/vc.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace engine::models::chatterbox {
namespace {

void validate_audio(const runtime::AudioBuffer & audio, const char * role) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error(std::string("Chatterbox VC ") + role + " audio sample rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error(std::string("Chatterbox VC ") + role + " audio channel count must be positive");
    }
    if (audio.samples.empty()) {
        throw std::runtime_error(std::string("Chatterbox VC ") + role + " audio must not be empty");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error(std::string("Chatterbox VC ") + role + " audio samples must be divisible by channels");
    }
}

runtime::AudioBuffer trim_mono_audio(
    runtime::AudioBuffer audio,
    int64_t max_samples) {
    if (audio.channels != 1) {
        throw std::runtime_error("Chatterbox VC trim expects mono audio");
    }
    if (max_samples > 0 && static_cast<int64_t>(audio.samples.size()) > max_samples) {
        audio.samples.resize(static_cast<size_t>(max_samples));
    }
    return audio;
}

}  // namespace

struct ChatterboxVcComponent::State {
    explicit State(engine::core::BackendConfig backend) : s3_cache(backend) {}
    S3GenSessionCache s3_cache;
};

runtime::AudioBuffer load_chatterbox_vc_audio_mono(
    const std::string & path,
    int sample_rate) {
    return runtime::AudioBuffer{
        sample_rate,
        1,
        engine::audio::read_wav_f32_as_mono_linear_resampled(std::filesystem::path(path), sample_rate),
    };
}

runtime::AudioBuffer normalize_chatterbox_vc_audio_mono(
    const runtime::AudioBuffer & audio,
    int sample_rate) {
    validate_audio(audio, "input");
    return runtime::AudioBuffer{
        sample_rate,
        1,
        engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
            audio.samples,
            audio.sample_rate,
            audio.channels,
            sample_rate),
    };
}

ChatterboxVcComponent::ChatterboxVcComponent(
    engine::models::chatterbox::S3TokenizerComponent tokenizer_component,
    engine::models::chatterbox::CAMPPlusEncoderComponent speaker_encoder,
    std::shared_ptr<const S3FlowEncoderWeights> flow_encoder_weights,
    std::shared_ptr<const S3FlowDecoderWeights> flow_decoder_weights,
    engine::models::chatterbox::HiFTVocoderComponent vocoder,
    ChatterboxPromptPrepConfig prompt_prep_config,
    const engine::core::ExecutionContext & execution_context,
    bool mem_saver)
    : tokenizer_(std::move(tokenizer_component)),
      speaker_encoder_(std::move(speaker_encoder)),
      flow_encoder_weights_(std::move(flow_encoder_weights)),
      flow_decoder_weights_(std::move(flow_decoder_weights)),
      vocoder_(std::move(vocoder)),
      prompt_prep_config_(prompt_prep_config),
      execution_context_(&execution_context),
      mem_saver_(mem_saver),
      state_(std::make_shared<State>(execution_context.config())) {
    if (!flow_encoder_weights_ || !flow_decoder_weights_) {
        throw std::runtime_error("ChatterboxVcComponent requires S3 flow weights");
    }
}

ChatterboxVoiceConversionOutputs ChatterboxVcComponent::convert(
    const runtime::AudioBuffer & source_audio,
    const runtime::AudioBuffer & target_voice,
    const ChatterboxVoiceConversionConfig & config) const {
    if (config.num_inference_steps <= 0) {
        throw std::runtime_error("Chatterbox VC num_inference_steps must be positive");
    }
    if (!(config.s3gen_cfg_rate >= 0.0F)) {
        throw std::runtime_error("Chatterbox VC s3gen_cfg_rate must be non-negative");
    }
    validate_audio(source_audio, "source");
    validate_audio(target_voice, "target");

    ChatterboxVoiceConversionOutputs outputs;
    const auto source_16k = normalize_chatterbox_vc_audio_mono(source_audio, 16000);
    const auto target_24k = trim_mono_audio(
        normalize_chatterbox_vc_audio_mono(target_voice, 24000),
        prompt_prep_config_.decoder_condition_samples);
    const auto target_16k = normalize_chatterbox_vc_audio_mono(target_24k, 16000);

    const auto target_ref_start = std::chrono::steady_clock::now();
    auto target_ref = tokenizer_.embed_reference_from_wavs(
        *speaker_encoder_.weights(),
        target_24k,
        target_16k);
    outputs.target_ref_ms = engine::debug::elapsed_ms(target_ref_start);

    const auto source_tokenizer_start = std::chrono::steady_clock::now();
    auto source_tokens = tokenizer_.tokenize(source_16k, std::nullopt);
    outputs.source_tokenizer_ms = engine::debug::elapsed_ms(source_tokenizer_start);
    outputs.source_speech_token_count = source_tokens.token_count;
    outputs.source_speech_tokens = source_tokens.tokens;

    const auto s3gen_start = std::chrono::steady_clock::now();
    S3GenTimingBreakdown timing;
    const auto generated = compute_s3gen_inference(
        state_->s3_cache,
        *flow_encoder_weights_,
        *flow_decoder_weights_,
        vocoder_,
        target_ref,
        outputs.source_speech_tokens,
        outputs.source_speech_token_count,
        config.num_inference_steps,
        config.s3gen_cfg_rate,
        true,
        {},
        config.seed,
        config.seed,
        execution_context_ != nullptr ? execution_context_->config() : engine::core::BackendConfig{},
        &timing);
    outputs.s3gen_ms = engine::debug::elapsed_ms(s3gen_start);
    outputs.s3gen_timing = timing;
    outputs.s3gen_token2mel_ms = timing.token2mel_ms;
    outputs.s3gen_vocoder_ms = timing.vocoder_ms;
    outputs.waveform = generated.waveform;
    outputs.samples = generated.samples;

    if (mem_saver_) {
        state_->s3_cache.release_runtime_graphs();
        vocoder_.release_runtime_cache();
    }

    engine::debug::timing_log_scalar("chatterbox.vc.target_ref_ms", outputs.target_ref_ms);
    engine::debug::timing_log_scalar("chatterbox.vc.source_tokenizer_ms", outputs.source_tokenizer_ms);
    engine::debug::timing_log_scalar("chatterbox.vc.s3gen.total_ms", outputs.s3gen_ms);
    engine::debug::timing_log_scalar("chatterbox.vc.s3gen.token2mel.total_ms", outputs.s3gen_token2mel_ms);
    engine::debug::timing_log_scalar("chatterbox.vc.s3gen.vocoder_ms", outputs.s3gen_vocoder_ms);
    engine::debug::trace_log_scalar("chatterbox.vc.source_speech_tokens", outputs.source_speech_token_count);
    return outputs;
}

}  // namespace engine::models::chatterbox
