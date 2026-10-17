// xtts_v2 TTS session
#include "engine/community_models/xtts_v2/session.h"
#include "engine/community_models/xtts_v2/gpt.h"
#include "engine/community_models/xtts_v2/vocoder.h"
#include "engine/community_models/xtts_v2/tokenizer_text.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/text/chunking.h"
#include <stdexcept>
#include <chrono>

#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <chrono>
#include <omp.h>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace engine::models::xtts_v2 {
namespace {

static std::string get_opt(const runtime::SessionPreparationRequest& req, const std::string& key,
                           const std::string& def) {
    auto it = req.options.find(key);
    return (it != req.options.end()) ? it->second : def;
}

template<typename T>
static T parse_opt(const std::string& s, T def) {
    try {
        if constexpr (std::is_same_v<T, int>) return std::stoi(s);
        else if constexpr (std::is_same_v<T, float>) return std::stof(s);
        else return def;
    } catch (...) { return def; }
}

} // namespace

struct XttSv2TtsSession::Impl {
    std::shared_ptr<const XttSv2Assets> assets;
    XttSv2GptEngine gpt;
    XttSv2Vocoder vocoder;
    int n_threads = 4;
    float temperature = 0.75f;
    int top_k = 50;
    float top_p = 0.85f;
    float repetition_penalty = 10.0f;
    int max_mel_tokens = 605;
    std::string language = "en";
    bool prepared = false;
    // Session-owned conditioning cache: keyed by the reference audio content so
    // repeated requests with the same voice reuse the conditioning graphs.
    struct RefCache {
        std::vector<float> pcm;        // reference at 22050 Hz
        std::vector<float> cond;       // 32 x 1024 conditioning latents
        std::vector<float> spk;        // 512-dim speaker embedding
    } ref_cache;
};

XttSv2TtsSession::XttSv2TtsSession(std::shared_ptr<const XttSv2Assets> assets, const runtime::SessionOptions& session_options)
    : impl_(std::make_unique<Impl>()) {
    impl_->assets = std::move(assets);
    impl_->n_threads = session_options.backend.threads > 0 ? session_options.backend.threads : impl_->n_threads;
}

XttSv2TtsSession::~XttSv2TtsSession() = default;

void XttSv2TtsSession::prepare(const runtime::SessionPreparationRequest& request) {
    auto& c = *impl_;
    c.temperature = parse_opt<float>(get_opt(request, "temperature", "0.75"), 0.75f);
    c.top_k = parse_opt<int>(get_opt(request, "top_k", "50"), 50);
    c.top_p = parse_opt<float>(get_opt(request, "top_p", "0.85"), 0.85f);
    c.repetition_penalty = parse_opt<float>(get_opt(request, "repetition_penalty", "10.0"), 10.0f);
    c.max_mel_tokens = parse_opt<int>(get_opt(request, "max_mel_tokens", "605"), 605);
    c.language = get_opt(request, "language", "en");

    if (!c.gpt.load(c.assets->gpt_gguf_path, c.assets->vocoder_gguf_path, c.n_threads, false)) {
        throw std::runtime_error("xtts_v2: failed to load GPT weights from " + c.assets->gpt_gguf_path.string());
    }
    if (!c.vocoder.load(c.assets->vocoder_gguf_path, c.n_threads, false)) {
        throw std::runtime_error("xtts_v2: failed to load vocoder weights from " + c.assets->vocoder_gguf_path.string());
    }
    c.prepared = true;
}

runtime::TaskResult XttSv2TtsSession::run(const runtime::TaskRequest& request) {
    auto& c = *impl_;
    if (!c.prepared) {
        throw std::runtime_error("xtts_v2: session used before successful prepare()");
    }

    std::string text;
    if (request.text_input) text = request.text_input->text;
    if (text.empty()) throw std::runtime_error("xtts_v2: request has no text input");

    std::string lang = c.language;
    if (request.text_input && !request.text_input->language.empty())
        lang = request.text_input->language;

    // Reference audio (voice cloning requires it)
    const runtime::AudioBuffer* ref_buf = nullptr;
    if (request.voice.has_value() && request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value() && !request.voice->speaker->audio->samples.empty()) {
        ref_buf = &*request.voice->speaker->audio;
    } else if (request.audio_input.has_value() && !request.audio_input->samples.empty()) {
        ref_buf = &*request.audio_input;
    }
    if (ref_buf == nullptr) {
        throw std::runtime_error("xtts_v2: voice cloning requires a reference WAV (--voice-ref)");
    }

    // Resample the reference to 22.05 kHz once; reuse cached conditioning when the
    // reference content is unchanged across requests (frequent-graph reuse).
    const auto xtts_cache_t0 = std::chrono::steady_clock::now();
    std::vector<float> ref_pcm;
    if (ref_buf->sample_rate != 22050) {
        ref_pcm = engine::audio::resample_mono_torchaudio_sinc_hann(
            ref_buf->samples, ref_buf->sample_rate, 22050);
    } else {
        ref_pcm = ref_buf->samples;
    }

    bool cache_valid = (ref_pcm.size() == c.ref_cache.pcm.size()) &&
                       std::equal(ref_pcm.begin(), ref_pcm.end(), c.ref_cache.pcm.begin());
    if (!cache_valid) {
        std::vector<float> cond, spk;
        if (!c.gpt.compute_conditioning(ref_pcm.data(), (int)ref_pcm.size(), cond)) {
            throw std::runtime_error("xtts_v2: conditioning encoder failed on reference audio");
        }
        // Reference truncates the audio to max_ref_length=30 s before the speaker
        // encoder (xtts.py get_conditioning_latents).
        const int max_ref_ns = 22050 * 30;
        const int spk_ns = (int)std::min<size_t>(ref_pcm.size(), (size_t)max_ref_ns);
        spk = c.vocoder.speaker_embed(ref_pcm.data(), spk_ns);
        if (spk.empty()) throw std::runtime_error("xtts_v2: speaker encoder failed on reference audio");
        if (getenv("XTTS_TIMING")) fprintf(stderr, "xtts_timing: cond+spk (cache miss) = %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now()-xtts_cache_t0).count());
        c.ref_cache = Impl::RefCache{std::move(ref_pcm), std::move(cond), std::move(spk)};
    }
    const auto& cond_latents = c.ref_cache.cond;
    const auto& spk_emb = c.ref_cache.spk;

    // Long-form support: split oversized text on sentence boundaries with the
    // shared framework chunker and synthesize each chunk independently.
    const std::vector<std::string> chunks = engine::text::split_text_chunks(text, 240);
    std::vector<float> waveform;
    for (size_t ci = 0; ci < chunks.size(); ci++) {
        const std::string chunk_norm = normalize_xtts_v2_text(chunks[ci], lang.c_str());
        auto text_tokens = tokenize_xtts_v2(&c.gpt.vocab(), chunk_norm, lang.c_str());
        if (text_tokens.empty()) {
            throw std::runtime_error("xtts_v2: tokenizer produced no tokens for text chunk");
        }
        // xTTS v2 text framing: [START]=261 ... [STOP]=0
        text_tokens.insert(text_tokens.begin(), 261);
        text_tokens.push_back(0);

        // Generate codes and capture the per-position post-norm hidden states
        // during decoding; these are exactly what a teacher-forced second pass
        // (extract_latents) would recompute, so the second pass is skipped.
        std::vector<float> decode_hiddens;   // [start_mel, c1, c2, ...] x 1024
        const auto xtts_chunk_t0 = std::chrono::steady_clock::now();
        auto mel_codes = c.gpt.generate_mel_codes(
            text_tokens, cond_latents, c.temperature, c.top_k, c.top_p,
            c.repetition_penalty, c.max_mel_tokens, &decode_hiddens);
        if (mel_codes.empty()) {
            throw std::runtime_error("xtts_v2: mel code generation produced no output");
        }

        // Latent frame mapping (identical to the reference gpt.py
        // forward(return_latent=True) slice): the vocoder takes the frames at
        // absolute positions [Tpre, Tpre + T_mel) where Tpre is the first mel
        // code position, and the final 5 latent frames are dropped.
        const int mel_len = (int)mel_codes.size() + 2;   // [start] + codes + [stop]
        int T_mel = mel_len - 5;
        if (T_mel <= 0) T_mel = mel_len;
        std::vector<float> latents_cm;
        const int n_hid = (int)decode_hiddens.size() / 1024;   // 1 + n_codes
        if (n_hid >= T_mel) {
            // decode_hiddens[0] is the start_mel position, which is exactly
            // where the reference slice begins (off = n_cond + text = Tpre-1).
            latents_cm.resize((size_t)1024 * T_mel);
            for (int t = 0; t < T_mel; t++)
                for (int ch = 0; ch < 1024; ch++)
                    latents_cm[(size_t)ch * T_mel + t] = decode_hiddens[(size_t)t * 1024 + ch];
        } else {
            // Fallback: degenerate short generations lack enough captured frames.
            std::vector<int32_t> mel_seq;
            mel_seq.reserve(mel_codes.size() + 2);
            mel_seq.push_back(1024);
            mel_seq.insert(mel_seq.end(), mel_codes.begin(), mel_codes.end());
            mel_seq.push_back(1025);
            auto latents = c.gpt.extract_latents(text_tokens, mel_seq, cond_latents);
            if (latents.empty()) {
                throw std::runtime_error("xtts_v2: latent extraction failed");
            }
            const int off = (int)latents.size() / 1024 - mel_len;
            latents_cm.resize((size_t)1024 * T_mel);
            for (int t = 0; t < T_mel; t++)
                for (int ch = 0; ch < 1024; ch++)
                    latents_cm[(size_t)ch * T_mel + t] = latents[(size_t)(off + t) * 1024 + ch];
        }

        const auto xtts_t2 = std::chrono::steady_clock::now();
        auto chunk_wave = c.vocoder.generate(latents_cm.data(), T_mel,
                                             spk_emb.empty() ? nullptr : spk_emb.data());
        const auto xtts_t3 = std::chrono::steady_clock::now();
        if (getenv("XTTS_TIMING")) fprintf(stderr, "xtts_timing: chunk gen+latents=%.2f vocoder=%.2f\n",
            std::chrono::duration<double>(xtts_t2-xtts_chunk_t0).count(),
            std::chrono::duration<double>(xtts_t3-xtts_t2).count());
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
    return result;
}

} // namespace engine::models::xtts_v2
