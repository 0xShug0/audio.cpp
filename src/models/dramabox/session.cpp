#include "engine/models/dramabox/session.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/dramabox/latent_state.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::dramabox {
namespace {

using Clock = std::chrono::steady_clock;

constexpr const char * kDefaultNegativePrompt =
    "worst quality, inconsistent, robotic, distorted, noise, static, muffled, unclear, unnatural, monotone";
constexpr const char * kFamily = "dramabox";
constexpr float kPi = 3.14159265358979323846F;
constexpr size_t kMaxPromptConditioningCacheEntries = 32;
constexpr size_t kMaxReferenceCacheEntries = 8;

std::shared_ptr<const DramaBoxAssets> require_assets(std::shared_ptr<const DramaBoxAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("DramaBox session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("DramaBox session requires a model contract");
    }
    return contract;
}

uint64_t mix_reference_hash(uint64_t hash, uint64_t value) {
    constexpr uint64_t kPrime = 1099511628211ull;
    hash ^= value;
    hash *= kPrime;
    return hash;
}

uint64_t inline_reference_audio_key(const runtime::AudioBuffer & audio) {
    uint64_t hash = 14695981039346656037ull;
    hash = mix_reference_hash(hash, static_cast<uint64_t>(static_cast<uint32_t>(audio.sample_rate)));
    hash = mix_reference_hash(hash, static_cast<uint64_t>(static_cast<uint32_t>(audio.channels)));
    hash = mix_reference_hash(hash, static_cast<uint64_t>(audio.samples.size()));
    for (const float sample : audio.samples) {
        uint32_t bits = 0;
        std::memcpy(&bits, &sample, sizeof(bits));
        hash = mix_reference_hash(hash, bits);
    }
    return hash;
}

std::optional<uint64_t> inline_reference_cache_key(
    const DramaBoxRequest & parsed,
    const runtime::TaskRequest & request) {
    if (!parsed.target_voice.empty() || !parsed.has_inline_target_voice ||
        !request.voice.has_value() ||
        !request.voice->speaker.has_value() ||
        !request.voice->speaker->audio.has_value()) {
        return std::nullopt;
    }
    return inline_reference_audio_key(*request.voice->speaker->audio);
}

DramaBoxPerfMode parse_perf_mode(const std::string & value) {
    if (value == "off" || value == "exact" || value == "standard") {
        return DramaBoxPerfMode::Exact;
    }
    if (value == "flash_attention") {
        return DramaBoxPerfMode::FlashAttention;
    }
    throw std::runtime_error("Invalid dramabox.perf_mode: " + value);
}

bool mem_saver_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"dramabox.mem_saver"})) {
        return runtime::parse_bool_option(*value, "dramabox.mem_saver");
    }
    return false;
}

DramaBoxRequest parse_dramabox_request(const runtime::TaskRequest & request, const DramaBoxConfig & config) {
    DramaBoxRequest out;
    if (request.text_input.has_value()) {
        out.prompt = request.text_input->text;
    }
    if (out.prompt.empty()) {
        throw std::runtime_error("DramaBox requires a non-empty prompt");
    }
    if (const auto value = runtime::find_option(request.options, {"negative_prompt"})) {
        out.negative_prompt = *value;
    }
    out.steps = runtime::parse_int_option(request.options, {"num_inference_steps"})
        .value_or(config.diffusion_steps);
    if (out.steps <= 0) {
        throw std::runtime_error("DramaBox num_inference_steps must be positive");
    }
    out.cfg_scale = runtime::parse_float_option(request.options, {"guidance_scale"})
        .value_or(config.default_cfg_scale);
    out.spatio_temporal_guidance_scale =
        runtime::parse_float_option(request.options, {"spatio_temporal_guidance_scale"})
            .value_or(config.default_spatio_temporal_guidance_scale);
    out.duration_scale = runtime::parse_float_option(request.options, {"duration_scale"})
        .value_or(config.default_duration_scale);
    out.duration_sec = runtime::parse_float_option(request.options, {"duration_sec"})
        .value_or(0.0F);
    out.reference_duration_sec = runtime::parse_float_option(request.options, {"reference_duration_sec"})
        .value_or(config.default_reference_duration_sec);
    out.audio_chunk_threshold_sec = runtime::parse_float_option(request.options, {"audio_chunk_threshold_sec"})
        .value_or(out.audio_chunk_threshold_sec);
    out.audio_chunk_duration_sec = runtime::parse_float_option(request.options, {"audio_chunk_duration_sec"})
        .value_or(out.audio_chunk_duration_sec);
    out.cross_fade_duration_sec = runtime::parse_float_option(request.options, {"cross_fade_duration_sec"})
        .value_or(out.cross_fade_duration_sec);
    out.seed = runtime::parse_u32_option(request.options, {"seed"}).value_or(static_cast<uint32_t>(out.seed));
    if (const auto value = runtime::find_option(request.options, {"guidance_rescale"})) {
        if (*value != "auto") {
            out.guidance_rescale = runtime::parse_float_option(request.options, {"guidance_rescale"}).value();
        }
    }
    if (const auto value = runtime::find_option(request.options, {"target_voice"})) {
        out.target_voice = *value;
    } else if (request.voice.has_value() &&
               request.voice->speaker.has_value() &&
               request.voice->speaker->audio.has_value()) {
        out.has_inline_target_voice = true;
    }
    if (out.reference_duration_sec <= 0.0F) {
        throw std::runtime_error("DramaBox reference_duration_sec must be positive");
    }
    if (out.duration_sec < 0.0F) {
        throw std::runtime_error("DramaBox duration_sec must be non-negative");
    }
    if (out.audio_chunk_threshold_sec <= 0.0F || out.audio_chunk_duration_sec <= 0.0F ||
        out.cross_fade_duration_sec < 0.0F) {
        throw std::runtime_error("DramaBox chunking options must be positive, with non-negative cross_fade_duration_sec");
    }
    return out;
}

const std::string & negative_prompt_for_request(const DramaBoxRequest & parsed) {
    if (!parsed.negative_prompt.empty()) {
        return parsed.negative_prompt;
    }
    static const std::string default_negative_prompt(kDefaultNegativePrompt);
    return default_negative_prompt;
}

runtime::AudioBuffer read_reference_audio(const DramaBoxRequest & parsed, const runtime::TaskRequest & request) {
    if (!parsed.target_voice.empty()) {
        const auto wav = audio::read_wav_f32(parsed.target_voice);
        if (wav.sample_rate <= 0 || wav.channels <= 0 || wav.samples.empty()) {
            throw std::runtime_error("DramaBox target_voice WAV is empty");
        }
        return runtime::AudioBuffer{wav.sample_rate, wav.channels, wav.samples};
    }
    if (parsed.has_inline_target_voice &&
        request.voice.has_value() &&
        request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value()) {
        return *request.voice->speaker->audio;
    }
    throw std::runtime_error("DramaBox reference audio was requested but no audio was available");
}

runtime::AudioBuffer equal_power_crossfade_concat(
    const runtime::AudioBuffer & previous,
    const runtime::AudioBuffer & next,
    float cross_fade_duration_sec) {
    if (previous.sample_rate != next.sample_rate) {
        throw std::runtime_error("DramaBox long-form chunk sample-rate mismatch");
    }
    if (previous.channels <= 0 || next.channels <= 0 || previous.samples.empty() || next.samples.empty()) {
        throw std::runtime_error("DramaBox long-form received empty chunk audio");
    }
    const int channels = std::max(previous.channels, next.channels);
    auto prev_planar = audio::deinterleave_to_planar_channels(previous.samples, previous.channels);
    auto next_planar = audio::deinterleave_to_planar_channels(next.samples, next.channels);
    const int64_t prev_frames = static_cast<int64_t>(prev_planar.size()) / previous.channels;
    const int64_t next_frames = static_cast<int64_t>(next_planar.size()) / next.channels;
    if (previous.channels != channels) {
        std::vector<float> expanded(static_cast<size_t>(channels * prev_frames), 0.0F);
        for (int c = 0; c < channels; ++c) {
            std::copy_n(
                prev_planar.data() + static_cast<std::ptrdiff_t>((c % previous.channels) * prev_frames),
                static_cast<size_t>(prev_frames),
                expanded.data() + static_cast<std::ptrdiff_t>(c * prev_frames));
        }
        prev_planar = std::move(expanded);
    }
    if (next.channels != channels) {
        std::vector<float> expanded(static_cast<size_t>(channels * next_frames), 0.0F);
        for (int c = 0; c < channels; ++c) {
            std::copy_n(
                next_planar.data() + static_cast<std::ptrdiff_t>((c % next.channels) * next_frames),
                static_cast<size_t>(next_frames),
                expanded.data() + static_cast<std::ptrdiff_t>(c * next_frames));
        }
        next_planar = std::move(expanded);
    }
    int64_t fade_samples = static_cast<int64_t>(
        std::llround(static_cast<double>(cross_fade_duration_sec) * static_cast<double>(previous.sample_rate)));
    fade_samples = std::max<int64_t>(1, std::min<int64_t>(fade_samples, std::min(prev_frames, next_frames)));
    const int64_t out_frames = prev_frames + next_frames - fade_samples;
    std::vector<float> out(static_cast<size_t>(channels * out_frames), 0.0F);
    for (int c = 0; c < channels; ++c) {
        const float * prev = prev_planar.data() + static_cast<std::ptrdiff_t>(c * prev_frames);
        const float * nxt = next_planar.data() + static_cast<std::ptrdiff_t>(c * next_frames);
        float * dst = out.data() + static_cast<std::ptrdiff_t>(c * out_frames);
        std::copy_n(prev, static_cast<size_t>(prev_frames - fade_samples), dst);
        for (int64_t i = 0; i < fade_samples; ++i) {
            const float t = fade_samples <= 1 ? 1.0F : static_cast<float>(i) / static_cast<float>(fade_samples - 1);
            const float fade_out = std::cos(t * kPi * 0.5F);
            const float fade_in = std::sin(t * kPi * 0.5F);
            dst[prev_frames - fade_samples + i] = prev[prev_frames - fade_samples + i] * fade_out + nxt[i] * fade_in;
        }
        std::copy_n(nxt + static_cast<std::ptrdiff_t>(fade_samples), static_cast<size_t>(next_frames - fade_samples), dst + static_cast<std::ptrdiff_t>(prev_frames));
    }
    return runtime::AudioBuffer{
        previous.sample_rate,
        channels,
        audio::interleave_planar_channels(out, channels, out_frames),
    };
}

}  // namespace

DramaBoxSession::DramaBoxSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const DramaBoxAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(std::move(options)),
      task_(task),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      prompt_conditioning_cache_(kMaxPromptConditioningCacheEntries),
      negative_conditioning_cache_(1),
      reference_latents_(kMaxReferenceCacheEntries) {
    runtime::validate_spec_backed_session_options(this->options(), *contract_, kFamily, "DramaBox");
    mem_saver_ = mem_saver_from_options(this->options());
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("DramaBox currently supports offline sessions");
    }
    if (task_.task != runtime::VoiceTaskKind::Tts &&
        task_.task != runtime::VoiceTaskKind::VoiceCloning &&
        task_.task != runtime::VoiceTaskKind::AudioGeneration) {
        throw std::runtime_error("DramaBox supports Tts, VoiceCloning, and AudioGeneration tasks");
    }
    if (const auto it = this->options().options.find("dramabox.perf_mode"); it != this->options().options.end()) {
        perf_mode_ = parse_perf_mode(it->second);
    }
    tokenizer_ = std::make_unique<DramaBoxGemmaTokenizer>(assets_);
    gemma_prompt_ = std::make_unique<DramaBoxGemma3PromptRuntime>(
        execution_context(),
        assets_,
        engine::assets::TensorStorageType::Native,
        engine::assets::TensorStorageType::Native,
        3);
    prompt_connector_ = std::make_unique<DramaBoxPromptConnectorRuntime>(
        execution_context(),
        assets_,
        engine::assets::TensorStorageType::Native,
        3,
        perf_mode_);
    dit_ = std::make_unique<DramaBoxDitRuntime>(
        execution_context(),
        assets_,
        engine::assets::TensorStorageType::Native,
        perf_mode_);
    audio_encoder_ = std::make_unique<DramaBoxAudioVaeEncoderRuntime>(
        execution_context(),
        assets_,
        engine::assets::TensorStorageType::Native);
    audio_decoder_ = std::make_unique<DramaBoxAudioVaeDecoderRuntime>(
        execution_context(),
        assets_,
        engine::assets::TensorStorageType::Native);
    vocoder_ = std::make_unique<DramaBoxVocoderRuntime>(
        execution_context(),
        assets_,
        engine::assets::TensorStorageType::Native);
}

DramaBoxSession::~DramaBoxSession() = default;

std::string DramaBoxSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind DramaBoxSession::task_kind() const {
    return task_.task;
}

runtime::RunMode DramaBoxSession::run_mode() const {
    return task_.mode;
}

void DramaBoxSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.text.has_value() && !request.voice.has_value() && request.options.empty()) {
        mark_prepared();
        return;
    }
    runtime::validate_spec_backed_request_options(request.options, *contract_, "DramaBox");
    runtime::TaskRequest task_request;
    task_request.text_input = request.text;
    task_request.voice = request.voice;
    task_request.options = request.options;
    const auto parsed = parse_dramabox_request(task_request, assets_->config);
    if (mem_saver_) {
        mark_prepared();
        return;
    }
    const double duration = parsed.duration_sec > 0.0F
        ? static_cast<double>(parsed.duration_sec)
        : estimate_dramabox_duration_seconds(parsed.prompt, parsed.duration_scale);
    int64_t target_tokens = dramabox_target_latent_shape(duration, assets_->config).token_count();
    const bool long_form = duration > parsed.audio_chunk_threshold_sec;
    std::vector<DramaBoxPromptChunk> prepared_chunks;
    if (long_form) {
        prepared_chunks = chunk_prompt_for_duration(
            parsed.prompt,
            parsed.audio_chunk_threshold_sec,
            parsed.audio_chunk_duration_sec,
            parsed.duration_scale);
        if (prepared_chunks.empty()) {
            throw std::runtime_error("DramaBox long-form chunker produced no chunks");
        }
        target_tokens = dramabox_target_latent_shape(
            estimate_dramabox_duration_seconds(prepared_chunks.front().text, parsed.duration_scale),
            assets_->config)
                            .token_count();
    }
    int64_t ref_tokens = 0;
    if (!parsed.target_voice.empty() || parsed.has_inline_target_voice) {
        const auto & config = assets_->config.audio_vae;
        const int64_t ref_mel_frames =
            static_cast<int64_t>(std::llround(static_cast<double>(parsed.reference_duration_sec) *
                                              static_cast<double>(config.sample_rate) /
                                              static_cast<double>(config.hop_length))) + 1;
        ref_tokens = std::max<int64_t>(1, (ref_mel_frames + 3) / 4);
        audio_encoder_->prepare(1, ref_mel_frames);
    }
    const bool cfg_enabled = parsed.cfg_scale > 1.0F;
    const bool stg_enabled = parsed.spatio_temporal_guidance_scale > 0.0F;
    const int64_t branch_count = 1 + (cfg_enabled ? 1 : 0) + (stg_enabled ? 1 : 0);
    const int64_t prompt_batch = cfg_enabled ? 2 : 1;
    gemma_prompt_->prepare(prompt_batch);
    prompt_connector_->prepare(prompt_batch);
    const int64_t context_tokens = assets_->config.gemma.prompt_max_length;
    dit_->prepare(
        branch_count,
        target_tokens + ref_tokens,
        context_tokens,
        stg_enabled,
        ref_tokens);
    audio_decoder_->prepare(1, target_tokens);
    vocoder_->prepare(std::max<int64_t>(target_tokens * 4 - 3, 1));
    mark_prepared();
}

runtime::TaskResult DramaBoxSession::run(const runtime::TaskRequest & request) {
    require_prepared("DramaBox run");
    runtime::validate_spec_backed_request_options(request.options, *contract_, "DramaBox");
    const auto wall_start = Clock::now();
    const auto parsed = parse_dramabox_request(request, assets_->config);
    engine::debug::trace_log_scalar("dramabox.request.steps", parsed.steps);
    engine::debug::trace_log_scalar("dramabox.request.cfg_scale", static_cast<double>(parsed.cfg_scale));
    engine::debug::trace_log_scalar(
        "dramabox.request.spatio_temporal_guidance_scale",
        static_cast<double>(parsed.spatio_temporal_guidance_scale));
    const double duration = parsed.duration_sec > 0.0F
        ? static_cast<double>(parsed.duration_sec)
        : estimate_dramabox_duration_seconds(parsed.prompt, parsed.duration_scale);
    engine::debug::trace_log_scalar(
        "dramabox.request.long_form",
        duration > parsed.audio_chunk_threshold_sec);
    runtime::TaskResult result;
    if (duration > parsed.audio_chunk_threshold_sec) {
        const auto chunks = chunk_prompt_for_duration(
            parsed.prompt,
            parsed.audio_chunk_threshold_sec,
            parsed.audio_chunk_duration_sec,
            parsed.duration_scale);
        if (chunks.empty()) {
            throw std::runtime_error("DramaBox long-form chunker produced no chunks");
        }
        engine::debug::trace_log_scalar("dramabox.long_form.chunks", static_cast<int64_t>(chunks.size()));
        std::optional<runtime::AudioBuffer> combined;
        for (size_t index = 0; index < chunks.size(); ++index) {
            auto chunk_request = parsed;
            chunk_request.prompt = chunks[index].text;
            chunk_request.duration_sec = 0.0F;
            engine::debug::trace_log_scalar("dramabox.long_form.chunk_est_seconds", chunks[index].estimated_duration_seconds);
            auto audio = generate_audio(chunk_request, request);
            combined = combined.has_value()
                ? std::optional<runtime::AudioBuffer>(
                      equal_power_crossfade_concat(*combined, audio, parsed.cross_fade_duration_sec))
                : std::optional<runtime::AudioBuffer>(std::move(audio));
        }
        result.audio_output = std::move(*combined);
    } else {
        result.audio_output = generate_audio(parsed, request);
    }
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return result;
}

DramaBoxConditioningEncoding DramaBoxSession::prompt_conditioning(const std::string & prompt) {
    const int64_t sequence_length = assets_->config.gemma.prompt_max_length;
    const PromptCacheKey prompt_key{prompt, sequence_length};
    if (auto * cached = prompt_conditioning_cache_.find(prompt_key)) {
        engine::debug::trace_log_scalar("dramabox.prompt.cache_hit", true);
        return *cached;
    }
    engine::debug::trace_log_scalar("dramabox.prompt.cache_hit", false);
    const auto tokens = tokenizer_->encode(std::vector<std::string>{prompt});
    const auto prompt_features = gemma_prompt_->encode(tokens);
    auto conditioning = prompt_connector_->encode(prompt_features);
    prompt_conditioning_cache_.put(prompt_key, conditioning);
    return conditioning;
}

DramaBoxConditioningEncoding DramaBoxSession::prompt_conditioning_for_guidance(
    const std::string & prompt,
    const std::string & negative_prompt,
    bool cfg_enabled) {
    if (!cfg_enabled) {
        return prompt_conditioning(prompt);
    }
    const int64_t sequence_length = assets_->config.gemma.prompt_max_length;
    const PromptCacheKey prompt_key{prompt, sequence_length};
    const PromptCacheKey negative_key{negative_prompt, sequence_length};
    const auto * cached_prompt = prompt_conditioning_cache_.find(prompt_key);
    const auto * cached_negative = negative_conditioning_cache_.find(negative_key);
    const bool prompt_hit = cached_prompt != nullptr;
    const bool negative_shape_hit = cached_negative != nullptr;
    engine::debug::trace_log_scalar("dramabox.prompt.cache_hit", prompt_hit);
    engine::debug::trace_log_scalar("dramabox.prompt.negative_cache_hit", negative_shape_hit);
    if (prompt_hit && negative_shape_hit) {
        return join_positive_negative_conditioning(*cached_prompt, *cached_negative);
    }
    if (!prompt_hit && !negative_shape_hit) {
        const auto tokens = tokenizer_->encode(std::vector<std::string>{prompt, negative_prompt});
        const auto prompt_features = gemma_prompt_->encode(tokens);
        auto conditioning = prompt_connector_->encode(prompt_features);
        if (conditioning.batch != 2) {
            throw std::runtime_error("DramaBox batched CFG prompt conditioning returned unexpected batch size");
        }
        auto positive = select_conditioning_batch(conditioning, 0);
        negative_conditioning_cache_.put(negative_key, select_conditioning_batch(conditioning, 1));
        prompt_conditioning_cache_.put(prompt_key, std::move(positive));
        return conditioning;
    }
    DramaBoxConditioningEncoding positive;
    if (prompt_hit) {
        positive = *cached_prompt;
    } else {
        const auto tokens = tokenizer_->encode(std::vector<std::string>{prompt});
        const auto prompt_features = gemma_prompt_->encode(tokens);
        positive = prompt_connector_->encode(prompt_features);
        prompt_conditioning_cache_.put(prompt_key, positive);
    }
    const DramaBoxConditioningEncoding * negative = cached_negative;
    if (!negative_shape_hit) {
        const auto tokens = tokenizer_->encode(std::vector<std::string>{negative_prompt});
        const auto prompt_features = gemma_prompt_->encode(tokens);
        negative_conditioning_cache_.put(negative_key, prompt_connector_->encode(prompt_features));
        negative = negative_conditioning_cache_.find(negative_key);
    }
    return join_positive_negative_conditioning(positive, *negative);
}

DramaBoxEncodedReferenceLatents DramaBoxSession::encode_reference_latents(
    const DramaBoxRequest & parsed,
    const runtime::TaskRequest & request) {
    const auto inline_key = inline_reference_cache_key(parsed, request);
    const ReferenceCacheKey reference_key{parsed.target_voice.string(), inline_key, parsed.reference_duration_sec};
    const bool cacheable_reference = !parsed.target_voice.empty() || inline_key.has_value();
    if (cacheable_reference) {
        if (auto * cached = reference_latents_.find(reference_key)) {
            engine::debug::trace_log_scalar("dramabox.reference.cache_hit", true);
            return *cached;
        }
    }
    engine::debug::trace_log_scalar("dramabox.reference.cache_hit", false);
    const auto ref_audio = read_reference_audio(parsed, request);
    int64_t ref_mel_frames = 0;
    const auto ref_mel = reference_log_mel(
        ref_audio,
        assets_->config,
        parsed.reference_duration_sec,
        execution_context().config().threads,
        ref_mel_frames);
    auto latents = audio_encoder_->encode(ref_mel, 1, ref_mel_frames);
    if (cacheable_reference) {
        reference_latents_.put(reference_key, latents);
    }
    return latents;
}

runtime::AudioBuffer DramaBoxSession::generate_audio(const DramaBoxRequest & parsed, const runtime::TaskRequest & request) {
    const double duration = parsed.duration_sec > 0.0F
        ? static_cast<double>(parsed.duration_sec)
        : estimate_dramabox_duration_seconds(parsed.prompt, parsed.duration_scale);
    const auto latent_shape = dramabox_target_latent_shape(duration, assets_->config);
    engine::debug::trace_log_scalar("dramabox.request.latent_tokens", latent_shape.token_count());
    auto state = create_dramabox_initial_state(latent_shape, assets_->config);
    const int64_t target_tokens = state.tokens;
    int64_t ref_tokens = 0;
    if (!parsed.target_voice.empty() || parsed.has_inline_target_voice) {
        const auto reference_start = Clock::now();
        const auto ref_latent = encode_reference_latents(parsed, request);
        ref_tokens = ref_latent.tokens;
        append_reference_latents(state, ref_latent, assets_->config);
        engine::debug::timing_log_scalar("dramabox.reference.total_ms", engine::debug::elapsed_ms(reference_start, Clock::now()));
        if (mem_saver_) {
            const auto release_start = Clock::now();
            audio_encoder_->release_runtime_state();
            engine::debug::timing_log_scalar(
                "dramabox.audio_vae_encoder.release_runtime_ms",
                engine::debug::elapsed_ms(release_start, Clock::now()));
        }
    }
    const auto rng_policy = engine::sampling::resolve_torch_cuda_sampling_policy(
        execution_context().backend_type(),
        execution_context().config().device,
        "dramabox.rng",
        "DramaBox",
        engine::sampling::TorchCudaSamplingPolicyFailureMode::StrictCuda);
    auto noise = engine::sampling::generate_torch_cuda_tensor_iterator_randn(
        state.latent.size(),
        static_cast<uint64_t>(parsed.seed),
        0,
        rng_policy,
        engine::sampling::TorchRandnPrecision::BFloat16);
    for (size_t i = 0; i < state.latent.size(); ++i) {
        state.latent[i] = noise[i] * state.denoise_mask[i] + state.clean_latent[i] * (1.0F - state.denoise_mask[i]);
    }

    const bool cfg_enabled = parsed.cfg_scale > 1.0F;
    const bool stg_enabled = parsed.spatio_temporal_guidance_scale > 0.0F;
    const int64_t branch_count = 1 + (cfg_enabled ? 1 : 0) + (stg_enabled ? 1 : 0);
    engine::debug::trace_log_scalar("dramabox.request.branch_count", branch_count);
    engine::debug::trace_log_scalar("dramabox.request.cfg_enabled", cfg_enabled);
    engine::debug::trace_log_scalar("dramabox.request.stg_enabled", stg_enabled);
    const auto prompt_start = Clock::now();
    auto conditioning = prompt_conditioning_for_guidance(parsed.prompt, negative_prompt_for_request(parsed), cfg_enabled);
    engine::debug::timing_log_scalar("dramabox.prompt.total_ms", engine::debug::elapsed_ms(prompt_start, Clock::now()));
    if (mem_saver_) {
        const auto release_start = Clock::now();
        prompt_connector_->release_runtime_state();
        gemma_prompt_->release_runtime_state();
        engine::debug::timing_log_scalar(
            "dramabox.prompt.release_runtime_ms",
            engine::debug::elapsed_ms(release_start, Clock::now()));
    }

    const auto sampler_start = Clock::now();
    const auto sigmas = make_dramabox_ltx2_sigmas(parsed.steps, 128);
    std::vector<float> branch_conditioning_features =
        make_branch_conditioning_features(conditioning, branch_count, cfg_enabled, stg_enabled);
    DramaBoxConditioningEncoding branch_conditioning;
    branch_conditioning.batch = branch_count;
    branch_conditioning.tokens = conditioning.tokens;
    branch_conditioning.hidden_size = conditioning.hidden_size;
    branch_conditioning.features = std::move(branch_conditioning_features);
    std::vector<float> rope_cos;
    std::vector<float> rope_sin;
    make_dramabox_audio_rope_repeated(
        state.positions,
        branch_count,
        state.tokens,
        assets_->config,
        rope_cos,
        rope_sin);
    std::vector<float> timestep_mask(static_cast<size_t>(state.tokens), 0.0F);
    for (int64_t token = 0; token < state.tokens; ++token) {
        timestep_mask[static_cast<size_t>(token)] = state.denoise_mask[static_cast<size_t>(token * 128)];
    }
    dit_->prepare_static_inputs(
        branch_count,
        state.tokens,
        stg_enabled,
        ref_tokens,
        branch_conditioning,
        rope_cos,
        rope_sin,
        timestep_mask);
    const float guidance_rescale =
        parsed.guidance_rescale < 0.0F ? auto_rescale_for_cfg(parsed.cfg_scale) : parsed.guidance_rescale;
    std::vector<float> sigma_values(1, 0.0F);
    std::vector<float> sigma_features;
    DramaBoxDitInputs dit_inputs;
    dit_inputs.batch = branch_count;
    dit_inputs.tokens = state.tokens;
    dit_inputs.stg_enabled = stg_enabled;
    dit_inputs.ref_tokens = ref_tokens;
    dit_inputs.latent = &state.latent;
    dit_inputs.sigma_features = &sigma_features;
    std::vector<float> guided_cond;
    std::vector<float> guided_pred;
    for (int64_t step = 0; step < parsed.steps; ++step) {
        const float sigma = sigmas[static_cast<size_t>(step)];
        const float sigma_next = sigmas[static_cast<size_t>(step + 1)];
        const float scaled_sigma = sigma * static_cast<float>(assets_->config.transformer.timestep_scale_multiplier);
        sigma_values[0] = scaled_sigma;
        fill_dramabox_timestep_features(sigma_values, sigma_features);
        auto velocity = dit_->forward(dit_inputs);
        guided_prediction_from_velocity(
            velocity,
            state,
            branch_count,
            sigma,
            parsed.cfg_scale,
            parsed.spatio_temporal_guidance_scale,
            guidance_rescale,
            cfg_enabled,
            stg_enabled,
            guided_cond,
            guided_pred);
        post_process_and_euler_step(state, guided_pred, sigma, sigma_next);
    }
    if (mem_saver_) {
        const auto release_start = Clock::now();
        dit_->release_runtime_state();
        engine::debug::timing_log_scalar(
            "dramabox.dit.release_runtime_ms",
            engine::debug::elapsed_ms(release_start, Clock::now()));
    }
    engine::debug::timing_log_scalar("dramabox.sampler.total_ms", engine::debug::elapsed_ms(sampler_start, Clock::now()));

    const auto decode_start = Clock::now();
    std::vector<float> target_latent(static_cast<size_t>(target_tokens * 128), 0.0F);
    std::copy_n(state.latent.data(), static_cast<size_t>(target_latent.size()), target_latent.data());
    if (target_tokens > 514) {
        for (int64_t d = 0; d < 128; ++d) {
            const float left = target_latent[static_cast<size_t>(511 * 128 + d)];
            const float right = target_latent[static_cast<size_t>(514 * 128 + d)];
            target_latent[static_cast<size_t>(512 * 128 + d)] = left * (2.0F / 3.0F) + right * (1.0F / 3.0F);
            target_latent[static_cast<size_t>(513 * 128 + d)] = left * (1.0F / 3.0F) + right * (2.0F / 3.0F);
        }
    }
    const auto mel = audio_decoder_->decode_to_device(target_latent, 1, target_tokens);
    const auto audio = vocoder_->synthesize(mel);
    if (mem_saver_) {
        const auto release_start = Clock::now();
        vocoder_->release_runtime_state();
        audio_decoder_->release_runtime_state();
        engine::debug::timing_log_scalar(
            "dramabox.decode.release_runtime_ms",
            engine::debug::elapsed_ms(release_start, Clock::now()));
    }
    engine::debug::timing_log_scalar("dramabox.decode.total_ms", engine::debug::elapsed_ms(decode_start, Clock::now()));

    runtime::TaskResult result;
    runtime::AudioBuffer buffer;
    buffer.sample_rate = static_cast<int>(audio.sample_rate);
    buffer.channels = static_cast<int>(audio.channels);
    buffer.samples = engine::audio::interleave_planar_channels(audio.waveform, static_cast<int>(audio.channels), audio.samples);
    return buffer;
}

namespace {

std::unique_ptr<runtime::IVoiceTaskSession> create_dramabox_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const DramaBoxAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<DramaBoxSession>(
        task,
        options,
        std::move(assets),
        std::move(contract));
}

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_dramabox_loader() {
    runtime::SpecBackedVoiceModelConfig<DramaBoxAssets> config;
    config.family = kFamily;
    config.load_assets = load_dramabox_assets;
    config.create_session = create_dramabox_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::dramabox
