// xtts_v2 TTS session
#include "engine/community_models/xtts_v2/session.h"
#include "engine/community_models/xtts_v2/gpt.h"
#include "engine/community_models/xtts_v2/vocoder.h"
#include "engine/community_models/xtts_v2/tokenizer_text.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace engine::models::xtts_v2 {

struct XttSv2TtsSession::Impl {
    std::shared_ptr<const XttSv2Assets> assets;
    std::shared_ptr<const engine::model_spec::ModelContract> contract;
    XttSv2GptEngine gpt;
    XttSv2Vocoder vocoder;
    runtime::SessionPreparationRequest preparation;
    bool weights_loaded = false;
    struct VoiceConditionals {
        std::vector<float> cond;       // 32 x 1024 conditioning latents
        std::vector<float> spk;        // 512-dim speaker embedding
    };
    // Canonical mono 22050 Hz PCM is the key, preserving stereo/mixdown equivalence.
    runtime::CacheSlots<std::vector<float>, VoiceConditionals> conditionals_cache;
};

XttSv2TtsSession::XttSv2TtsSession(const runtime::TaskSpec & task,
    std::shared_ptr<const XttSv2Assets> assets, const runtime::SessionOptions & options,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options), impl_(std::make_unique<Impl>()) {
    if (task.task != runtime::VoiceTaskKind::Tts || task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("XTTS v2 supports offline TTS only");
    }
    impl_->assets = std::move(assets);
    impl_->contract = std::move(contract);
    runtime::validate_spec_backed_session_options(options, *impl_->contract, "xtts_v2", "XTTS v2");
    const auto slots = runtime::parse_i64_option(options.options, {"xtts_v2.conditionals_cache_slots"}).value_or(1);
    if (slots < 0 || static_cast<uint64_t>(slots) > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("xtts_v2.conditionals_cache_slots must be non-negative and fit size_t");
    }
    impl_->conditionals_cache.set_capacity(static_cast<size_t>(slots));
}

XttSv2TtsSession::~XttSv2TtsSession() = default;

void XttSv2TtsSession::prepare(const runtime::SessionPreparationRequest & request) {
    auto & c = *impl_;
    runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *c.contract, "XTTS v2");
    if (!c.weights_loaded) {
        c.gpt.load(*c.assets, execution_context());
        c.vocoder.load(*c.assets, execution_context());
        c.weights_loaded = true;
    }
    c.preparation = request;
    mark_prepared();
}

runtime::TaskResult XttSv2TtsSession::run(const runtime::TaskRequest & request) {
    const auto wall_start = std::chrono::steady_clock::now();
    require_prepared("run");
    auto & c = *impl_;
    runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *c.contract, "XTTS v2");
    auto effective_options = c.preparation.options;
    for (const auto & option : request.options) effective_options.insert_or_assign(option.first, option.second);
    const auto & text_input = request.text_input ? request.text_input : c.preparation.text;
    if (!text_input || text_input->text.empty()) throw std::runtime_error("XTTS v2 requires text input");
    const auto text_chunk_size = engine::text::parse_text_chunk_size_override(effective_options).value_or(256);
    const auto text_chunk_mode = engine::text::parse_text_chunk_mode_override(effective_options)
        .value_or(engine::text::TextChunkMode::Default);
    std::string lang = runtime::find_option(effective_options, {"language"}).value_or("en");
    if (!text_input->language.empty()) lang = text_input->language;
    if (xtts_v2_language_index(lang.c_str()) < 0) throw std::runtime_error("XTTS v2 unsupported language: " + lang);
    const float temperature = runtime::parse_finite_float_option(effective_options, {"temperature"}).value_or(0.75f);
    const int top_k = runtime::parse_int_option(effective_options, {"top_k"}).value_or(50);
    const float top_p = runtime::parse_finite_float_option(effective_options, {"top_p"}).value_or(0.85f);
    const float repetition_penalty = runtime::parse_finite_float_option(effective_options, {"repetition_penalty"}).value_or(10.f);
    const int max_mel_tokens = runtime::parse_int_option(effective_options, {"max_mel_tokens"}).value_or(c.assets->hyper_params.max_mel_tokens);
    if (temperature < 0 || top_k < 0 || top_p < 0 || top_p > 1 || repetition_penalty < 1 ||
        max_mel_tokens < 1 || max_mel_tokens > static_cast<int>(c.assets->hyper_params.max_mel_tokens)) {
        throw std::runtime_error("XTTS v2 invalid sampling options; max_mel_tokens must fit the model context");
    }

    const auto seed = runtime::parse_u32_option(effective_options, {"seed"});
    std::mt19937 rng(seed ? *seed : runtime::random_u32_seed());
    // Reference audio (voice cloning requires it)
    const runtime::AudioBuffer* ref_buf = nullptr;
    const auto & voice = request.voice ? request.voice : c.preparation.voice;
    if (voice && voice->speaker && voice->speaker->audio && !voice->speaker->audio->samples.empty()) {
        ref_buf = &*voice->speaker->audio;
    } else if (request.audio_input.has_value() && !request.audio_input->samples.empty()) {
        ref_buf = &*request.audio_input;
    }
    if (ref_buf == nullptr) {
        throw std::runtime_error("xtts_v2: voice cloning requires a reference WAV (--voice-ref)");
    }

    // The same canonical reference can reuse conditioning across text chunks
    // and across requests, up to the configured number of voice slots.
    auto ref_pcm = engine::audio::convert_interleaved_audio_to_mono_torchaudio_sinc_hann_resampled(
        ref_buf->samples, ref_buf->sample_rate, ref_buf->channels, 22050);

    const auto * conditionals = c.conditionals_cache.find(ref_pcm);
    const bool cache_valid = conditionals != nullptr;
    const bool will_evict = !cache_valid && c.conditionals_cache.capacity() > 0 &&
        c.conditionals_cache.size() >= c.conditionals_cache.capacity();
    engine::debug::trace_log_scalar("xtts_v2.reference.cache_hit", cache_valid);
    Impl::VoiceConditionals prepared_conditionals;
    if (!cache_valid) {
        const auto conditioning_start = std::chrono::steady_clock::now();
        if (!c.gpt.compute_conditioning(ref_pcm.data(), (int)ref_pcm.size(), prepared_conditionals.cond)) {
            throw std::runtime_error("xtts_v2: conditioning encoder failed on reference audio");
        }
        engine::debug::timing_log_scalar("xtts_v2.conditioning_ms", engine::debug::elapsed_ms(conditioning_start, std::chrono::steady_clock::now()));
        const auto speaker_start = std::chrono::steady_clock::now();
        // Reference truncates the audio to max_ref_length=30 s before the speaker
        // encoder (xtts.py get_conditioning_latents).
        const int max_ref_ns = 22050 * 30;
        const int spk_ns = (int)std::min<size_t>(ref_pcm.size(), (size_t)max_ref_ns);
        prepared_conditionals.spk = c.vocoder.speaker_embed(ref_pcm.data(), spk_ns);
        engine::debug::timing_log_scalar("xtts_v2.speaker_ms", engine::debug::elapsed_ms(speaker_start, std::chrono::steady_clock::now()));
        if (prepared_conditionals.spk.empty()) throw std::runtime_error("xtts_v2: speaker encoder failed on reference audio");
        c.conditionals_cache.put(std::move(ref_pcm), prepared_conditionals);
        conditionals = &prepared_conditionals;
    }
    engine::debug::trace_log_scalar("xtts_v2.reference.cache_slots", static_cast<int64_t>(c.conditionals_cache.capacity()));
    engine::debug::trace_log_scalar("xtts_v2.reference.cache_entries", static_cast<int64_t>(c.conditionals_cache.size()));
    engine::debug::trace_log_scalar("xtts_v2.reference.cache_evicted", will_evict);
    const auto& cond_latents = conditionals->cond;
    const auto& spk_emb = conditionals->spk;

    // Long-form support: split oversized text on sentence boundaries with the
    // shared framework chunker and synthesize each chunk independently.
    auto text_request = request;
    text_request.text_input = text_input;
    text_request.options = std::move(effective_options);
    const auto chunks = runtime::chunk_text_request(text_request, text_chunk_size, text_chunk_mode);
    engine::debug::trace_log_scalar("xtts_v2.text_chunk_size", text_chunk_size);
    engine::debug::trace_log_scalar("xtts_v2.text_chunk_mode", engine::text::text_chunk_mode_name(text_chunk_mode));
    engine::debug::trace_log_scalar("xtts_v2.text_chunk_count", static_cast<int64_t>(chunks.size()));
    std::vector<float> waveform;
    for (size_t ci = 0; ci < chunks.size(); ci++) {
        const std::string chunk_norm = normalize_xtts_v2_text(chunks[ci].text_input->text, lang.c_str());
        auto text_tokens = tokenize_xtts_v2(&c.assets->vocab, chunk_norm, lang.c_str());
        if (text_tokens.empty()) {
            throw std::runtime_error("xtts_v2: tokenizer produced no tokens for text chunk");
        }
        // xTTS v2 text framing: [START]=261 ... [STOP]=0
        text_tokens.insert(text_tokens.begin(), c.assets->hyper_params.start_text_token);
        text_tokens.push_back(c.assets->hyper_params.stop_text_token);

        engine::debug::trace_log_i32("xtts_v2.text_tokens", {static_cast<int64_t>(text_tokens.size())}, text_tokens);

        // Generate codes and capture the per-position post-norm hidden states
        // during decoding; these are exactly what a teacher-forced second pass
        // (extract_latents) would recompute, so the second pass is skipped.
        std::vector<float> decode_hiddens;   // [start_mel, c1, c2, ...] x 1024
        const auto generate_start = std::chrono::steady_clock::now();
        auto mel_codes = c.gpt.generate_mel_codes(
            text_tokens, cond_latents, temperature, top_k, top_p,
            repetition_penalty, max_mel_tokens, rng, &decode_hiddens);
        engine::debug::timing_log_scalar("xtts_v2.generate_ms", engine::debug::elapsed_ms(generate_start, std::chrono::steady_clock::now()));
        engine::debug::trace_log_scalar("xtts_v2.generated_tokens", static_cast<int64_t>(mel_codes.size()));
        if (mel_codes.empty()) {
            throw std::runtime_error("xtts_v2: mel code generation produced no output");
        }

        // Upstream forward pads the generated sequence by five positions and
        // then slices those five positions off: N generated codes map to N
        // latents. The AR pass already captured the same causal hidden states.
        const int D = c.assets->hyper_params.d_model;
        const int T_mel = static_cast<int>(mel_codes.size());
        if (decode_hiddens.size() != static_cast<size_t>(T_mel) * D) {
            throw std::runtime_error("XTTS v2 generated codes and captured latents have different lengths");
        }
        std::vector<float> latents_cm(decode_hiddens.size());
        for (int t = 0; t < T_mel; ++t) {
            for (int ch = 0; ch < D; ++ch) {
                latents_cm[static_cast<size_t>(ch) * T_mel + t] = decode_hiddens[static_cast<size_t>(t) * D + ch];
            }
        }
        engine::debug::trace_log_i32("xtts_v2.mel_codes", {T_mel}, mel_codes);
        const auto vocoder_start = std::chrono::steady_clock::now();
        auto chunk_wave = c.vocoder.generate(latents_cm.data(), T_mel,
                                             spk_emb.empty() ? nullptr : spk_emb.data());
        engine::debug::timing_log_scalar("xtts_v2.vocoder_ms", engine::debug::elapsed_ms(vocoder_start, std::chrono::steady_clock::now()));
        if (chunk_wave.empty()) {
            throw std::runtime_error("xtts_v2: vocoder produced no output");
        }
        // Amplitude normalization per chunk (matches reference output scaling)
        float max_amp = 0;
        for (float s : chunk_wave) max_amp = std::max(max_amp, std::abs(s));
        if (max_amp > 1.0f) {
            const float scale = 0.8f / max_amp;
            for (float& s : chunk_wave) s *= scale;
        }
        // Small inter-chunk pause keeps sentence boundaries natural.
        if (ci > 0) waveform.insert(waveform.end(), (size_t)(24000 * 0.12f), 0.0f);
        waveform.insert(waveform.end(), chunk_wave.begin(), chunk_wave.end());
    }

    runtime::TaskResult result;
    result.audio_output = runtime::AudioBuffer{24000, 1, std::move(waveform)};
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, std::chrono::steady_clock::now()));
    return result;
}

} // namespace engine::models::xtts_v2
