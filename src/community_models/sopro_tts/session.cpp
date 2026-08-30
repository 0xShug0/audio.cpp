#include "engine/community_models/sopro_tts/session.h"

#include "engine/community_models/sopro_tts/acoustic.h"
#include "engine/community_models/sopro_tts/reference.h"
#include "engine/community_models/sopro_tts/semantic_encoder.h"
#include "engine/community_models/sopro_tts/semantic_lm.h"
#include "engine/community_models/sopro_tts/speaker_encoder.h"
#include "engine/community_models/sopro_tts/text_tokenizer.h"
#include "engine/community_models/sopro_tts/vocoder.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::sopro_tts {
namespace {

constexpr const char * kFamily = "sopro_tts";
constexpr size_t kWeightContextBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kGraphArenaBytes = 1024ull * 1024ull * 1024ull;
// SoproTTS.DECODE_CONTEXT_FRAMES: mel frames of prompt fed to the vocoder so
// its convolutions start warm, then dropped from the output.
constexpr int64_t kDecodeContextFrames = 32;

std::shared_ptr<const SoproTTSAssets> require_assets(std::shared_ptr<const SoproTTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Sopro session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("Sopro session requires a model contract");
    }
    return contract;
}

const runtime::AudioBuffer * reference_audio(const runtime::TaskRequest & request) {
    if (request.voice.has_value() && request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value()) {
        return &*request.voice->speaker->audio;
    }
    return request.audio_input.has_value() ? &*request.audio_input : nullptr;
}

std::vector<float> to_mono_24k(const runtime::AudioBuffer & audio, int target_rate) {
    if (audio.samples.empty()) {
        throw std::runtime_error("Sopro reference audio is empty");
    }
    const int channels = std::max(1, audio.channels);
    std::vector<float> mono = channels == 1
        ? audio.samples
        : engine::audio::mixdown_interleaved_to_mono_average(audio.samples, channels);
    if (audio.sample_rate > 0 && audio.sample_rate != target_rate) {
        mono = engine::audio::resample_mono_torchaudio_sinc_hann(mono, audio.sample_rate, target_rate);
    }
    // sopro.audio.to_mono_resampled clamps before anything else touches it.
    for (auto & value : mono) {
        value = std::min(1.0F, std::max(-1.0F, value));
    }
    return mono;
}

std::unique_ptr<runtime::IVoiceTaskSession> create_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const SoproTTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<SoproTTSSession>(task, options, std::move(assets), std::move(contract));
}

}  // namespace

SoproTTSSession::SoproTTSSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const SoproTTSAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))) {
    runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "Sopro");
    if (const auto value = runtime::find_option(options.options, {"language"})) {
        default_language_ = *value;
    }
    const auto matmul_storage = runtime::parse_tensor_storage_option(
        options.options,
        "matmul_weight_type",
        assets::TensorStorageType::F32,
        {assets::TensorStorageType::Native,
         assets::TensorStorageType::F32,
         assets::TensorStorageType::F16,
         assets::TensorStorageType::BF16,
         assets::TensorStorageType::Q8_0});
    const auto conv_storage = runtime::parse_tensor_storage_option(
        options.options,
        "conv_weight_type",
        assets::TensorStorageType::F32,
        {assets::TensorStorageType::Native,
         assets::TensorStorageType::F32,
         assets::TensorStorageType::F16});

    core::ExecutionContext & execution = execution_context();
    tokenizer_ = std::make_unique<SoproTextTokenizer>(assets_->tokenizer_path);
    speaker_encoder_ = std::make_unique<SoproSpeakerEncoderRuntime>(
        *assets_, execution, kWeightContextBytes, kGraphArenaBytes, matmul_storage, conv_storage);
    semantic_encoder_ = std::make_unique<SoproSemanticEncoderRuntime>(
        *assets_, execution, kWeightContextBytes, kGraphArenaBytes, matmul_storage, conv_storage);
    vocoder_ = std::make_unique<SoproVocoderRuntime>(
        *assets_, execution, kWeightContextBytes, kGraphArenaBytes, matmul_storage, conv_storage);
    semantic_lm_ = std::make_unique<SoproSemanticLMRuntime>(
        *assets_, execution, kGraphArenaBytes, kGraphArenaBytes, kWeightContextBytes, matmul_storage);
    acoustic_ = std::make_unique<SoproAcousticRuntime>(
        *assets_, execution, kWeightContextBytes, kGraphArenaBytes, matmul_storage, conv_storage);
    reference_builder_ = std::make_unique<SoproReferenceBuilder>(
        *assets_, *speaker_encoder_, *semantic_encoder_, *vocoder_);
}

SoproTTSSession::~SoproTTSSession() = default;

std::string SoproTTSSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind SoproTTSSession::task_kind() const {
    return runtime::VoiceTaskKind::Tts;
}

runtime::RunMode SoproTTSSession::run_mode() const {
    return task_.mode;
}

void SoproTTSSession::prepare(const runtime::SessionPreparationRequest & request) {
    runtime::validate_spec_backed_request_options(request.options, *contract_, "Sopro");
    mark_prepared();
}

SoproRequestOptions SoproTTSSession::parse_options(const runtime::TaskRequest & request) const {
    const auto & defaults = assets_->config.generation;
    SoproRequestOptions out;
    out.language = default_language_;
    out.temperature = defaults.temperature;
    out.top_p = defaults.top_p;
    out.top_k = defaults.top_k;
    out.steps = defaults.steps;
    out.max_seconds = defaults.max_seconds;
    out.min_seconds = defaults.min_seconds;
    out.max_segment_chars = defaults.max_segment_chars;
    out.ref_seconds = defaults.ref_seconds;

    if (const auto value = runtime::find_option(request.options, {"language"})) {
        out.language = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"temperature"})) {
        out.temperature = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"top_p"})) {
        out.top_p = *value;
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"top_k"})) {
        out.top_k = *value;
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"num_inference_steps"})) {
        out.steps = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"max_seconds"})) {
        out.max_seconds = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"min_seconds"})) {
        out.min_seconds = *value;
    }
    if (const auto value = runtime::parse_i64_option(request.options, {"text_chunk_size"})) {
        out.max_segment_chars = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"ref_seconds"})) {
        out.ref_seconds = *value;
    }
    if (const auto value = runtime::parse_u64_option(request.options, {"seed"})) {
        out.seed = *value;
        out.has_seed = true;
    }
    if (!out.has_seed) {
        out.seed = runtime::random_u64_seed();
    }
    if (out.steps < 1) {
        throw std::runtime_error("Sopro num_inference_steps must be positive");
    }
    if (out.max_segment_chars < 1) {
        throw std::runtime_error("Sopro text_chunk_size must be positive");
    }
    if (out.max_seconds <= 0.0F) {
        throw std::runtime_error("Sopro max_seconds must be positive");
    }
    if (out.ref_seconds <= 0.0F) {
        throw std::runtime_error("Sopro ref_seconds must be positive");
    }
    // language_tag() rejects anything outside the four supported languages, so
    // fail before any weights are touched.
    (void) language_tag(out.language);
    return out;
}

runtime::TaskResult SoproTTSSession::run(const runtime::TaskRequest & request) {
    require_prepared("Sopro run");
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Sopro run requires an offline session");
    }
    runtime::validate_spec_backed_request_options(request.options, *contract_, "Sopro");
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("Sopro requires non-empty text input");
    }
    const runtime::AudioBuffer * reference = reference_audio(request);
    if (reference == nullptr || reference->samples.empty()) {
        throw std::runtime_error(
            "Sopro requires reference voice audio (voice preset or voice_ref) for zero-shot cloning");
    }
    const auto options = parse_options(request);
    const auto & config = assets_->config;
    const auto sample_rate = static_cast<int>(config.sample_rate);
    const int64_t token_samples = config.semantic_encoder.token_samples_24k;
    const int64_t hop_ratio = config.hop_ratio();
    const int64_t n_mels = config.model.acoustic_mel_n_mels;
    const int64_t vocoder_hop = vocoder_->hop_length();

    std::mt19937_64 rng(options.seed);
    const auto reference_audio24 = to_mono_24k(*reference, sample_rate);

    const auto reference_start = std::chrono::steady_clock::now();
    const auto voice = reference_builder_->build(reference_audio24, options.ref_seconds, rng);
    engine::debug::timing_log_scalar(
        "sopro_tts.reference.prepare_ms",
        engine::debug::elapsed_ms(reference_start, std::chrono::steady_clock::now()));
    if (voice.semantic_tokens.empty() || voice.mel_frames <= 0) {
        throw std::runtime_error("Sopro reference audio produced no semantic tokens");
    }

    // _steps(): one semantic token per token_samples output samples.
    const auto steps_for = [&](float seconds) {
        return std::max<int64_t>(
            1,
            static_cast<int64_t>(std::ceil(
                static_cast<double>(seconds) * static_cast<double>(sample_rate) /
                static_cast<double>(token_samples))));
    };

    const auto style_count = std::min<int64_t>(
        config.generation.style_tokens, static_cast<int64_t>(voice.semantic_tokens.size()));
    const std::vector<int32_t> style_tokens(
        voice.semantic_tokens.begin(),
        voice.semantic_tokens.begin() + static_cast<ptrdiff_t>(style_count));
    const auto prompt_budget = config.generation.prompt_tokens;
    std::vector<int32_t> carry;
    if (prompt_budget > 0) {
        const auto count = std::min<int64_t>(
            prompt_budget, static_cast<int64_t>(voice.semantic_tokens.size()));
        carry.assign(
            voice.semantic_tokens.begin(),
            voice.semantic_tokens.begin() + static_cast<ptrdiff_t>(count));
    }

    SoproSemanticLMOptions lm_options;
    lm_options.max_steps = steps_for(options.max_seconds);
    lm_options.min_steps = steps_for(options.min_seconds);
    lm_options.temperature = options.temperature;
    lm_options.top_p = options.top_p;
    lm_options.top_k = options.top_k;

    std::vector<std::vector<float>> parts;
    for (const auto & segment : split_text(request.text_input->text, options.max_segment_chars)) {
        const auto text_ids = tokenizer_->encode(segment, options.language);
        const auto lm_start = std::chrono::steady_clock::now();
        const auto tokens = semantic_lm_->generate(text_ids, style_tokens, carry, lm_options, rng);
        engine::debug::timing_log_scalar(
            "sopro_tts.semantic_lm.generate_ms",
            engine::debug::elapsed_ms(lm_start, std::chrono::steady_clock::now()));
        engine::debug::trace_log_scalar(
            "sopro_tts.semantic_lm.tokens", static_cast<int64_t>(tokens.size()));
        if (tokens.empty()) {
            continue;
        }
        if (prompt_budget > 0) {
            const auto count = std::min<int64_t>(prompt_budget, static_cast<int64_t>(tokens.size()));
            carry.assign(tokens.end() - static_cast<ptrdiff_t>(count), tokens.end());
        }

        SoproAcousticRequest acoustic;
        acoustic.semantic_tokens = voice.semantic_tokens;
        acoustic.semantic_tokens.insert(
            acoustic.semantic_tokens.end(), tokens.begin(), tokens.end());
        acoustic.cond_vec = voice.cond_vec;
        acoustic.prompt_mel = voice.mel;
        acoustic.prompt_frames = voice.mel_frames;
        acoustic.total_frames =
            voice.mel_frames + static_cast<int64_t>(tokens.size()) * hop_ratio;
        acoustic.steps = options.steps;
        acoustic.seed = rng();
        const auto acoustic_start = std::chrono::steady_clock::now();
        const auto mel = acoustic_->solve(acoustic);
        engine::debug::timing_log_scalar(
            "sopro_tts.acoustic.solve_ms",
            engine::debug::elapsed_ms(acoustic_start, std::chrono::steady_clock::now()));

        // Denormalise and hand the vocoder a short prompt run-up so its
        // convolution state matches the reference, then drop that run-up.
        const int64_t context = std::min(kDecodeContextFrames, voice.mel_frames);
        const int64_t begin = voice.mel_frames - context;
        const int64_t decode_frames = acoustic.total_frames - begin;
        std::vector<float> decode_mel(static_cast<size_t>(n_mels * decode_frames), 0.0F);
        for (int64_t c = 0; c < n_mels; ++c) {
            const float mean = config.model.acoustic_mel_mean[static_cast<size_t>(c)];
            const float scale = config.model.acoustic_mel_std[static_cast<size_t>(c)];
            const float * source = mel.data() + static_cast<size_t>(c * acoustic.total_frames + begin);
            float * target = decode_mel.data() + static_cast<size_t>(c * decode_frames);
            for (int64_t t = 0; t < decode_frames; ++t) {
                target[t] = source[t] * scale + mean;
            }
        }
        auto wav = vocoder_->decode(decode_mel, decode_frames);
        const int64_t skip = context * vocoder_hop;
        const int64_t target_length = static_cast<int64_t>(tokens.size()) * token_samples;
        if (static_cast<int64_t>(wav.size()) <= skip) {
            continue;
        }
        const int64_t end = std::min<int64_t>(static_cast<int64_t>(wav.size()), skip + target_length);
        parts.emplace_back(
            wav.begin() + static_cast<ptrdiff_t>(skip), wav.begin() + static_cast<ptrdiff_t>(end));
    }

    runtime::TaskResult result;
    runtime::AudioBuffer audio;
    audio.sample_rate = sample_rate;
    audio.channels = 1;
    if (parts.empty()) {
        audio.samples.assign(static_cast<size_t>(token_samples), 0.0F);
        result.audio_output = std::move(audio);
        return result;
    }

    // Level-match once over the whole utterance, then trim and cross-fade the
    // segment joins (SoproTTS.synthesize).
    std::vector<float> concatenated;
    for (const auto & part : parts) {
        concatenated.insert(concatenated.end(), part.begin(), part.end());
    }
    const float gain = audio_ops::match_gain(concatenated, sample_rate);
    std::vector<std::vector<float>> trimmed;
    trimmed.reserve(parts.size());
    for (size_t index = 0; index < parts.size(); ++index) {
        auto part = parts[index];
        for (auto & value : part) {
            value *= gain;
        }
        part = index == 0
            ? audio_ops::trim_lead(part, sample_rate)
            : audio_ops::trim_lead(
                  part, sample_rate, audio_ops::kSegmentLeadSeconds, audio_ops::kSegmentSkipSeconds);
        trimmed.push_back(audio_ops::trim_trail(part, sample_rate));
    }
    auto out = audio_ops::join_segments(std::move(trimmed), sample_rate);
    audio_ops::soft_limit(out);
    audio_ops::fade_edges(out, sample_rate, false, true, audio_ops::kFinalFadeSeconds);
    audio.samples = std::move(out);
    result.audio_output = std::move(audio);
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_sopro_tts_loader() {
    runtime::SpecBackedVoiceModelConfig<SoproTTSAssets> config;
    config.family = kFamily;
    // The upstream repo and the model card both call the family "sopro"; keep
    // the short spelling working as a --family hint.
    config.aliases = {"sopro", "sopro_v2", "sopro_v2_turbo"};
    config.load_assets = load_sopro_tts_assets;
    config.create_session = create_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::community_models::sopro_tts
