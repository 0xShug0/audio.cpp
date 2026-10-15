// xtts_v2 TTS session
#include "engine/community_models/xtts_v2/session.h"
#include "engine/community_models/xtts_v2/gpt.h"
#include "engine/community_models/xtts_v2/vocoder.h"
#include "engine/community_models/xtts_v2/tokenizer_text.h"

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
    if (!c.prepared) return runtime::TaskResult{};

    // Get text input
    std::string text;
    if (request.text_input) {
        text = request.text_input->text;
    }
    if (text.empty()) return runtime::TaskResult{};

    // Language from request or default
    std::string lang = c.language;
    if (request.text_input && !request.text_input->language.empty())
        lang = request.text_input->language;

    // Normalize text
    text = normalize_xtts_v2_text(text, lang.c_str());
    fprintf(stderr, "xtts_v2: normalized text: '%s'\n", text.c_str());

    // Tokenize
    auto text_tokens = tokenize_xtts_v2(&c.gpt.vocab(), text, lang.c_str());
    if (text_tokens.empty()) return runtime::TaskResult{};
    // xTTS v2 text framing: [START]=261 ... [STOP]=0
    text_tokens.insert(text_tokens.begin(), 261);
    text_tokens.push_back(0);
    fprintf(stderr, "xtts_v2: %zu text tokens (framed):", text_tokens.size());
    if (getenv("XTTS_DEBUG")) for (int t : text_tokens) fprintf(stderr, " %d", t);
    fprintf(stderr, "\n");

    // Get reference audio (from voice condition, fall back to audio_input)
    std::vector<float> cond_latents;
    std::vector<float> spk_emb;
    std::vector<float> ref_pcm;
    int ref_sr = 22050;

    const runtime::AudioBuffer* ref_buf = nullptr;
    if (request.voice.has_value() && request.voice->speaker.has_value() &&
        request.voice->speaker->audio.has_value() && !request.voice->speaker->audio->samples.empty()) {
        ref_buf = &*request.voice->speaker->audio;
    } else if (request.audio_input.has_value() && !request.audio_input->samples.empty()) {
        ref_buf = &*request.audio_input;
    }

    auto t0 = std::chrono::steady_clock::now();
    if (ref_buf) {
        ref_pcm = ref_buf->samples;
        ref_sr = ref_buf->sample_rate;

        // Resample to 22050 if needed
        if (ref_sr != 22050) {
            int sr_in = ref_sr;
            int sr_out = 22050;
            double ratio = (double)sr_in / sr_out;
            int out_len = (int)(ref_pcm.size() / ratio + 0.5);
            std::vector<float> resampled(out_len);
            for (int i = 0; i < out_len; i++) {
                double src = i * ratio;
                int idx = (int)src;
                double frac = src - idx;
                if (idx + 1 < (int)ref_pcm.size())
                    resampled[i] = (float)((1.0 - frac) * ref_pcm[idx] + frac * ref_pcm[idx + 1]);
                else resampled[i] = ref_pcm[idx];
            }
            ref_pcm = std::move(resampled);
        }

        fprintf(stderr, "xtts_v2: computing conditioning (%d samples @ 22050 Hz)\n", (int)ref_pcm.size());
        if (!c.gpt.compute_conditioning(ref_pcm.data(), (int)ref_pcm.size(), cond_latents))
            return runtime::TaskResult{};
        auto t0b = std::chrono::steady_clock::now();
        if (getenv("XTTS_TIMING")) fprintf(stderr, "xtts_timing:   cond = %.2f s\n", std::chrono::duration<double>(t0b - t0).count());

        // Speaker embedding from 16kHz-resampled reference audio.
        // Reference truncates the audio to max_ref_length=30 s BEFORE the speaker
        // encoder (xtts.py get_conditioning_latents), so match that here.
        {
            const int max_ref_ns = 22050 * 30;
            int spk_ns = (int)std::min<size_t>(ref_pcm.size(), (size_t)max_ref_ns);
            spk_emb = c.vocoder.speaker_embed(ref_pcm.data(), spk_ns);
        }
        auto t0c = std::chrono::steady_clock::now();
        if (getenv("XTTS_TIMING")) fprintf(stderr, "xtts_timing:   spk  = %.2f s\n", std::chrono::duration<double>(t0c - t0b).count());
        if (spk_emb.empty()) fprintf(stderr, "xtts_v2: WARNING speaker embedding failed, using zeros\n");
        if (getenv("XTTS_DUMP_SPK")) {
            FILE* f = fopen(getenv("XTTS_DUMP_SPK"), "wb");
            if (f) { fwrite(spk_emb.data(), 4, spk_emb.size(), f); fclose(f); }
        }
        else {
            float rms = 0; for (auto v : spk_emb) rms += v * v;
            double vsum = 0; for (float v : spk_emb) vsum += v;
            fprintf(stderr, "xtts_v2: speaker embedding ok (512 dims, rms=%.4f sum=%.6f v0=%.6f)\n", sqrtf(rms / spk_emb.size()), vsum, spk_emb.empty() ? 0.f : spk_emb[0]);
        }
    }
    if (getenv("XTTS_NOCOND")) { fprintf(stderr, "xtts_dbg: XTTS_NOCOND set - zeroing cond latents\n"); cond_latents.clear(); }
    auto t1 = std::chrono::steady_clock::now();
    if (getenv("XTTS_TIMING")) fprintf(stderr, "xtts_timing: omp max threads = %d\n", omp_get_max_threads());
    if (getenv("XTTS_TIMING")) fprintf(stderr, "xtts_timing: cond+spk = %.2f s\n", std::chrono::duration<double>(t1 - t0).count());
    if (getenv("XTTS_SPKSCALE")) {
        float sc = atof(getenv("XTTS_SPKSCALE"));
        for (auto& v : spk_emb) v *= sc;
        fprintf(stderr, "xtts_dbg: spk_emb scaled by %s\n", getenv("XTTS_SPKSCALE"));
    }
    // DEBUG: inject reference cond latent (32x1024 f32) / speaker embedding (512 f32)
    // from raw files (e.g. extracted from speakers_xtts.pth) to isolate pipeline stages.
    if (getenv("XTTS_INJECT_COND")) {
        FILE* f = fopen(getenv("XTTS_INJECT_COND"), "rb");
        if (f) {
            cond_latents.resize(32 * 1024);
            size_t rd = fread(cond_latents.data(), 4, 32 * 1024, f);
            fclose(f);
            fprintf(stderr, "xtts_dbg: injected cond latents (%zu floats, rms=%.4f)\n", rd,
                    [&]{ double s = 0; for (auto v : cond_latents) s += (double)v * v; return sqrt(s / cond_latents.size()); }());
        }
    }
    if (getenv("XTTS_INJECT_SPK")) {
        FILE* f = fopen(getenv("XTTS_INJECT_SPK"), "rb");
        if (f) {
            spk_emb.resize(512);
            size_t rd = fread(spk_emb.data(), 4, 512, f);
            fclose(f);
            fprintf(stderr, "xtts_dbg: injected speaker embedding (%zu floats)\n", rd);
        }
    }

    // Generate mel codes
    fprintf(stderr, "xtts_v2: generating mel codes\n");
    auto t2 = std::chrono::steady_clock::now();
    auto mel_codes = c.gpt.generate_mel_codes(
        text_tokens, cond_latents, c.temperature, c.top_k, c.top_p,
        c.repetition_penalty, c.max_mel_tokens);
    if (mel_codes.empty()) return runtime::TaskResult{};
    if (getenv("XTTS_FORCE_CODES")) {
        mel_codes.clear();
        const char* p = getenv("XTTS_FORCE_CODES");
        while (*p) {
            char* end = nullptr; long v = strtol(p, &end, 10);
            if (end == p) break;
            mel_codes.push_back((int32_t)v);
            p = (*end == ',') ? end + 1 : end;
        }
        fprintf(stderr, "xtts_dbg: FORCED %zu codes\n", mel_codes.size());
    }
    fprintf(stderr, "xtts_v2: %zu mel codes\n", mel_codes.size());
    if (getenv("XTTS_DEBUG")) {
        fprintf(stderr, "xtts_dbg: codes =");
        for (int c : mel_codes) fprintf(stderr, " %d", c);
        fprintf(stderr, "\n");
    }

    // Extract GPT latents: reference feeds [start_mel=1024] + codes + [stop_mel=1025]
    // with mel_pos 0..N+1, and vocodes ALL N+2 latent positions
    std::vector<int32_t> mel_seq;
    mel_seq.reserve(mel_codes.size() + 2);
    mel_seq.push_back(1024);
    mel_seq.insert(mel_seq.end(), mel_codes.begin(), mel_codes.end());
    mel_seq.push_back(1025);
    fprintf(stderr, "xtts_v2: extracting GPT latents\n");
    auto t3 = std::chrono::steady_clock::now();
    if (getenv("XTTS_TIMING")) fprintf(stderr, "xtts_timing: mel_code_gen = %.2f s (%d codes)\n", std::chrono::duration<double>(t3 - t2).count(), (int)mel_codes.size());
    auto latents = c.gpt.extract_latents(text_tokens, mel_seq, cond_latents);
    if (latents.empty()) return runtime::TaskResult{};
    if (getenv("XTTS_DUMP_LATENTS")) {
        FILE* f = fopen(getenv("XTTS_DUMP_LATENTS"), "wb");
        if (f) { fwrite(latents.data(), 4, latents.size(), f); fclose(f); }
        fprintf(stderr, "xtts_dbg: dumped %zu latents\n", latents.size());
    }

    // Reference gpt.py forward(return_latent=True):
    //   enc = last_hidden_state[:, offset:]        (offset = n_cond, skip conditioning)
    //   return enc[:, -second_len:][:, :-5]        (mel part = last N+2 positions, drop last 5)
    // gpt_extract returns ALL positions [cond | text | mel]; take the LAST mel_seq.size()
    // positions (the mel part), then drop the final 5 (sub = -5).
    int mel_len = (int)mel_seq.size();             // [start, c1..cN, stop] = N+2
    int T_mel = mel_len - 5;                       // reference: mel_logits[:, :-5]
    if (T_mel <= 0) T_mel = mel_len;               // degenerate short sequences: keep all
    int off = (int)latents.size() / 1024 - mel_len;  // skip cond+text positions
    fprintf(stderr, "xtts_v2: vocoding (%d of %zu latent frames, mel offset %d)\n", T_mel, mel_seq.size(), off);
    // gpt_extract returns [T][d_model] (time-major); vocoder expects [d_model][T] (channel-major)
    std::vector<float> latents_cm((size_t)1024 * T_mel);
    for (int t = 0; t < T_mel; t++)
        for (int ch = 0; ch < 1024; ch++)
            latents_cm[(size_t)ch * T_mel + t] = latents[(size_t)(off + t) * 1024 + ch];
    if (getenv("XTTS_DEBUG")) {
        double s2 = 0; float mn = 1e30f, mx = -1e30f;
        for (float v : latents_cm) { s2 += (double)v * v; if (v < mn) mn = v; if (v > mx) mx = v; }
        fprintf(stderr, "xtts_dbg: vocoder input latents rms=%.4f min=%.4f max=%.4f (n=%zu)\n",
                sqrt(s2 / latents_cm.size()), mn, mx, latents_cm.size());
        // per-frame RMS of first few frames (should be ~1 if LayerNorm'd)
        for (int t = 0; t < 3 && t < T_mel; t++) {
            double fs = 0;
            for (int ch = 0; ch < 1024; ch++) { float v = latents_cm[(size_t)ch * T_mel + t]; fs += (double)v * v; }
            fprintf(stderr, "xtts_dbg:   frame %d rms=%.4f\n", t, sqrt(fs / 1024));
        }
    }
    auto t4 = std::chrono::steady_clock::now();
    if (getenv("XTTS_TIMING")) fprintf(stderr, "xtts_timing: extract = %.2f s\n", std::chrono::duration<double>(t4 - t3).count());
    auto waveform = c.vocoder.generate(latents_cm.data(), T_mel, spk_emb.empty() ? nullptr : spk_emb.data());
    auto t5 = std::chrono::steady_clock::now();
    if (getenv("XTTS_TIMING")) fprintf(stderr, "xtts_timing: vocoder = %.2f s\n", std::chrono::duration<double>(t5 - t4).count());
    if (waveform.empty()) return runtime::TaskResult{};

    // Amplitude normalization
    float max_amp = 0;
    for (float s : waveform) max_amp = std::max(max_amp, std::abs(s));
    if (max_amp > 0) { float scale = 0.8f / max_amp; for (float& s : waveform) s *= scale; }

    // Build result
    runtime::TaskResult result;
    result.audio_output = runtime::AudioBuffer{24000, 1, std::move(waveform)};
    fprintf(stderr, "xtts_v2: output %.2f seconds (%d samples)\n",
            (float)result.audio_output->samples.size() / 24000.0f,
            (int)result.audio_output->samples.size());
    return result;
}

} // namespace engine::models::xtts_v2
