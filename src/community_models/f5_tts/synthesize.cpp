#include "engine/community_models/f5_tts/synthesize.h"

#include "cpu_graph_compute.h"

#include "engine/community_models/f5_tts/runtime.h"

#include "cpu_graph_compute.h"

#include "engine/framework/core/backend.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "engine/framework/assets/tensor_source.h"

#include <algorithm>
#include <map>
#include <chrono>
#include <random>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace engine::models::f5_tts {
namespace {

// ---- mel filterbank (librosa-compatible, htk-free slaney, 24 kHz) ----------
constexpr int kSampleRate = 24000;
constexpr int kNfft = 1024;
constexpr int kHop = 256;
constexpr int kNMel = 100;
constexpr float kFMin = 0.0F;
constexpr float kFMax = 12000.0F;

// torchaudio defaults used by F5: htk mel scale, norm=None (no slaney
// normalization), power=1 (magnitude), f_max=sample_rate/2.
float hz_to_mel_htk(float hz) { return 2595.0F * std::log10(1.0F + hz / 700.0F); }
float mel_to_hz_htk(float mel) { return 700.0F * (std::pow(10.0F, mel / 2595.0F) - 1.0F); }

const std::vector<float> & mel_filterbank() {
    static std::vector<float> fb;  // [n_freqs, kNMel] like torch fb (freq-major)
    static std::once_flag once;
    std::call_once(once, [] {
        const int n_freqs = kNfft / 2 + 1;
        const float m_min = hz_to_mel_htk(kFMin);
        const float m_max = hz_to_mel_htk(kFMax);
        std::vector<float> mels(kNMel + 2);
        for (int i = 0; i < kNMel + 2; ++i) {
            mels[i] = mel_to_hz_htk(m_min + (m_max - m_min) * i / (kNMel + 1));
        }
        fb.assign(static_cast<size_t>(n_freqs) * kNMel, 0.0F);
        for (int m = 0; m < kNMel; ++m) {
            const float lo = mels[m];
            const float mid = mels[m + 1];
            const float hi = mels[m + 2];
            for (int f = 0; f < n_freqs; ++f) {
                const float freq = static_cast<float>(f) * kSampleRate / kNfft;
                if (freq <= lo || freq >= hi) continue;
                const float w = freq <= mid
                    ? (freq - lo) / (mid - lo)
                    : (hi - freq) / (hi - mid);
                fb[static_cast<size_t>(f) * kNMel + m] = w;  // freq-major like torch
            }
        }
    });
    return fb;
}

void fft_inplace(std::vector<float> & re, std::vector<float> & im, bool inverse) {
    const size_t n = re.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const float ang = static_cast<float>(2.0 * M_PI / static_cast<double>(len)) * (inverse ? 1 : -1);
        for (size_t i = 0; i < n; i += len) {
            for (size_t k = 0; k < len / 2; ++k) {
                const float wr = std::cos(ang * static_cast<float>(k));
                const float wi = std::sin(ang * static_cast<float>(k));
                const size_t a = i + k;
                const size_t b = i + k + len / 2;
                const float vr = re[b] * wr - im[b] * wi;
                const float vi = re[b] * wi + im[b] * wr;
                const float ur = re[a];
                const float ui = im[a];
                re[a] = ur + vr;
                im[a] = ui + vi;
                re[b] = ur - vr;
                im[b] = ui - vi;
            }
        }
    }
    if (inverse) {
        for (size_t i = 0; i < n; ++i) {
            re[i] /= static_cast<float>(n);
            im[i] /= static_cast<float>(n);
        }
    }
}

// log-mel with reflect-padded center STFT (librosa semantics), log clamp 1e-5.
std::vector<float> compute_mel(const std::vector<float> & wav) {
    const int n_freqs = kNfft / 2 + 1;
    // torchaudio center=True: n_frames = 1 + floor(len / hop)
    const int frames = std::max(1, 1 + static_cast<int>(wav.size() / kHop));
    std::vector<float> hann(kNfft);
    for (int i = 0; i < kNfft; ++i) {
        hann[i] = 0.5F * (1.0F - std::cos(2.0F * static_cast<float>(M_PI) * i / kNfft));
    }
    const auto & fb = mel_filterbank();
    std::vector<float> mel(static_cast<size_t>(kNMel) * frames);
    std::vector<float> re(kNfft), im(kNfft);
    std::vector<float> spec_pow(n_freqs);
    for (int t = 0; t < frames; ++t) {
        std::fill(re.begin(), re.end(), 0.0F);
        std::fill(im.begin(), im.end(), 0.0F);
        // torchaudio center=True alignment: frame t center = t*hop, achieved by
        // starting one hop earlier than librosa's default convention.
        const int start = (t - 2) * kHop;
        for (int i = 0; i < kNfft; ++i) {
            int r = start + i;
            // torch reflect pad: wav[-k] = wav[k], wav[L+k] = wav[L-2-k]
            if (r < 0) r = -r;
            if (r >= static_cast<int>(wav.size())) r = 2 * static_cast<int>(wav.size()) - 2 - r;
            r = std::clamp(r, 0, static_cast<int>(wav.size()) - 1);
            re[i] = wav[r] * hann[i];
        }
        fft_inplace(re, im, false);
        for (int f = 0; f < n_freqs; ++f) {
            spec_pow[f] = std::sqrt(re[f] * re[f] + im[f] * im[f]);  // power=1 magnitude
        }
        for (int m = 0; m < kNMel; ++m) {
            float acc = 0.0F;
            for (int f = 0; f < n_freqs; ++f) {
                acc += fb[static_cast<size_t>(f) * kNMel + m] * spec_pow[f];
            }
            mel[static_cast<size_t>(m) * frames + t] = std::log(std::max(acc, 1e-5F));
        }
    }
    return mel;  // [mel, frames] feature-fastest memory
}

std::vector<float> resample(const std::vector<float> & in, int sr_in, int sr_out) {
    if (sr_in == sr_out || in.empty()) return in;
    const double ratio = static_cast<double>(sr_out) / sr_in;
    const size_t out_n = static_cast<size_t>(static_cast<double>(in.size()) * ratio);
    std::vector<float> out(out_n);
    for (size_t i = 0; i < out_n; ++i) {
        const double pos = static_cast<double>(i) / ratio;
        const size_t i0 = static_cast<size_t>(pos);
        const size_t i1 = std::min(i0 + 1, in.size() - 1);
        const double frac = pos - i0;
        out[i] = static_cast<float>(in[i0] * (1 - frac) + in[i1] * frac);
    }
    return out;
}

std::unordered_map<std::string, int32_t> load_vocab(const std::string & dir) {
    std::unordered_map<std::string, int32_t> map;
    std::ifstream f(dir + "/vocab.txt", std::ios::binary);
    if (!f) throw std::runtime_error("cannot open vocab.txt in " + dir);
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<std::string> lines;
    std::string cur;
    for (size_t i = 0; i < content.size();) {
        if (content[i] == '\n') {
            lines.push_back(cur);
            cur.clear();
            ++i;
            continue;
        }
        size_t len = 1;
        const auto c = static_cast<unsigned char>(content[i]);
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        cur.append(content, i, len);
        i += len;
    }
    if (!cur.empty()) lines.push_back(cur);
    for (size_t i = 0; i < lines.size(); ++i) {
        if (!lines[i].empty()) map.emplace(lines[i], static_cast<int32_t>(i));
    }
    return map;
}

const char * dialect_token(const std::string & dialect) {
    if (dialect == "MSA") return "\xE2\x91\xA0";
    if (dialect == "SAU") return "\xE2\x91\xA1";
    if (dialect == "UAE") return "\xE2\x91\xA2";
    if (dialect == "ALG") return "\xE2\x91\xA3";
    if (dialect == "IRQ") return "\xE2\x91\xA4";
    if (dialect == "EGY") return "\xE2\x91\xA5";
    if (dialect == "MAR") return "\xE2\x91\xA6";
    if (dialect == "OMN") return "\xE2\x91\xA7";
    if (dialect == "TUN") return "\xE2\x91\xA8";
    if (dialect == "LEV") return "\xE2\x91\xA9";
    if (dialect == "SDN") return "\xE2\x91\xAA";
    if (dialect == "LBY") return "\xE2\x91\xAB";
    return "\xE2\x93\xAA";  // ⓪ UNK
}

std::vector<std::string> utf8_chars(const std::string & s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        size_t len = 1;
        const auto c = static_cast<unsigned char>(s[i]);
        if (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        out.emplace_back(s, i, len);
        i += len;
    }
    return out;
}

// Python: if ref_text ends with a single-byte char (ASCII), a space is
// appended so ref and gen text do not fuse into one token stream.
std::string apply_ref_trailing_space(std::string ref_text) {
    if (!ref_text.empty() && (static_cast<unsigned char>(ref_text.back()) & 0x80) == 0) {
        ref_text += " ";
    }
    return ref_text;
}

// Habibi prompt assembly: dialect token + 〈ref_text + gen_chunk〉.
std::string assemble_chunk_text(
    const std::string & dialect, const std::string & ref_text, const std::string & chunk) {
    return std::string(dialect_token(dialect)) + "\xE3\x80\x88" + ref_text + chunk + "\xE3\x80\x89";
}

std::vector<int32_t> tokenize_text(
    const std::unordered_map<std::string, int32_t> & vocab, const std::string & s) {
    std::vector<int32_t> ids;
    for (const auto & ch : utf8_chars(s)) {
        const auto it = vocab.find(ch);
        ids.push_back(it != vocab.end() ? it->second : 0);
    }
    return ids;
}

struct Rng {
    uint64_t state;
    explicit Rng(uint64_t seed) : state(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next_u64() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }
    float next_f32() {
        return static_cast<float>(static_cast<double>(next_u64() >> 11) / 9007199254740992.0);
    }
    float normal() {
        // Box-Muller
        float u1 = std::max(next_f32(), 1e-7F);
        float u2 = next_f32();
        return std::sqrt(-2.0F * std::log(u1)) * std::cos(2.0F * static_cast<float>(M_PI) * u2);
    }
};

// Python F5 uses Empirically Pruned Step Sampling (EPSS) for low NFE:
// non-uniform grids on a 1/32 quantum (get_epss_timesteps); uniform
// linspace otherwise. The sway transform is applied on top either way.
std::vector<float> sway_timesteps(int steps, float coef) {
    static const std::map<int, std::vector<int>> kEpss = {
        {5, {0, 2, 4, 8, 16, 32}},
        {6, {0, 2, 4, 6, 8, 16, 32}},
        {7, {0, 2, 4, 6, 8, 16, 24, 32}},
        {10, {0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32}},
        {12, {0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32}},
        {16, {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32}},
    };
    std::vector<float> t;
    const auto it = kEpss.find(steps);
    if (steps == 32 || it == kEpss.end()) {
        t.resize(static_cast<size_t>(steps) + 1);
        for (int i = 0; i <= steps; ++i) {
            t[static_cast<size_t>(i)] = static_cast<float>(i) / steps;
        }
    } else {
        for (const int v : it->second) t.push_back(static_cast<float>(v) / 32.0F);
    }
    for (auto & v : t) {
        v = v + coef * (std::cos(static_cast<float>(M_PI) / 2 * v) - 1 + v);
    }
    return t;
}

// ---- vocos vocoder ----------------------------------------------------------
// Loads vocos.safetensors and decodes [frames][100] log-mel -> waveform.
struct VocosWeights {
    std::shared_ptr<const assets::TensorSource> source;
    std::vector<float> embed_w;   // [512, 100, 7] torch
    std::vector<float> embed_b;   // [512]
    std::vector<float> input_nw, input_nb;  // backbone.norm (pre-blocks)
    struct Block {
        std::vector<float> dw_w, dw_b, n_w, n_b, p1w, p1b, p2w, p2b, gamma;
    };
    std::vector<Block> blocks;    // 8
    std::vector<float> final_nw, final_nb;  // [512]
    std::vector<float> head_w, head_b;      // [1026, 512], [1026]
};

const VocosWeights & load_vocos_once(const std::string & path) {
    static std::unordered_map<std::string, VocosWeights> cache;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto it = cache.find(path); it != cache.end()) return it->second;
    VocosWeights v;
    v.source = assets::open_tensor_source(path);
    if (std::filesystem::path(path).extension() == ".gguf") {
        // GGUF packages store the vocoder under the "vocos" namespace
        v.source = assets::make_prefixed_tensor_source(v.source, "vocos");
    }
    const auto f32 = [&](const char * n) { return v.source->require_f32(n); };
    v.embed_w = f32("backbone.embed.weight");
    v.embed_b = f32("backbone.embed.bias");
    v.input_nw = f32("backbone.norm.weight");
    v.input_nb = f32("backbone.norm.bias");
    v.blocks.resize(8);
    for (int i = 0; i < 8; ++i) {
        const std::string p = "backbone.convnext." + std::to_string(i) + ".";
        auto & b = v.blocks[i];
        b.dw_w = f32((p + "dwconv.weight").c_str());
        b.dw_b = f32((p + "dwconv.bias").c_str());
        b.n_w = f32((p + "norm.weight").c_str());
        b.n_b = f32((p + "norm.bias").c_str());
        b.p1w = f32((p + "pwconv1.weight").c_str());
        b.p1b = f32((p + "pwconv1.bias").c_str());
        b.p2w = f32((p + "pwconv2.weight").c_str());
        b.p2b = f32((p + "pwconv2.bias").c_str());
        b.gamma = f32((p + "gamma").c_str());
    }
    v.final_nw = f32("backbone.final_layer_norm.weight");
    v.final_nb = f32("backbone.final_layer_norm.bias");
    v.head_w = f32("head.out.weight");
    v.head_b = f32("head.out.bias");
    return cache.emplace(path, std::move(v)).first->second;
}

// CPU decode (pure host math, no graph): frames x 100 -> samples
std::vector<float> vocos_decode(const std::string & vocos_path, const std::vector<float> & mel_rows) {
    const auto & v = load_vocos_once(vocos_path);
    const int T = static_cast<int>(mel_rows.size()) / kNMel;
    const int D = 512;
    const int IM = 1536;

    // embed conv k7 pad 3 groups 1: out[t, o] = sum_i sum_k w[o, i, k] * x[t+k-3, i] + b[o]
    std::vector<float> h(static_cast<size_t>(T) * D);
    for (int t = 0; t < T; ++t) {
        for (int o = 0; o < D; ++o) {
            float acc = v.embed_b[o];
            const float * w = v.embed_w.data() + static_cast<size_t>(o) * kNMel * 7;
            for (int k = 0; k < 7; ++k) {
                const int tt = t + k - 3;
                if (tt < 0 || tt >= T) continue;
                const float * x = mel_rows.data() + static_cast<size_t>(tt) * kNMel;
                for (int i = 0; i < kNMel; ++i) {
                    acc += w[static_cast<size_t>(i) * 7 + k] * x[i];
                }
            }
            h[static_cast<size_t>(t) * D + o] = acc;
        }
    }

    // input layernorm (backbone.norm) after embed — VocosBackbone.forward applies
    // norm BEFORE the convnext stack (easy to miss; verified against torch hook).
    for (int t = 0; t < T; ++t) {
        float * x = h.data() + static_cast<size_t>(t) * D;
        float mu = 0, var = 0;
        for (int c = 0; c < D; ++c) mu += x[c];
        mu /= D;
        for (int c = 0; c < D; ++c) {
            const float d = x[c] - mu;
            var += d * d;
        }
        var /= D;
        const float inv = 1.0F / std::sqrt(var + 1e-6F);
        for (int c = 0; c < D; ++c) {
            x[c] = (x[c] - mu) * inv * v.input_nw[c] + v.input_nb[c];
        }
    }

    // 8 convnext blocks
#ifdef F5_MEL_TEST
    { std::ofstream f("/tmp/cpp_vocos_embed.bin", std::ios::binary);
      f.write(reinterpret_cast<const char*>(h.data()), h.size() * 4); }
#endif
    for (int bi = 0; bi < 8; ++bi) {
        const auto & B = v.blocks[bi];
        std::vector<float> dw(static_cast<size_t>(T) * D);
        for (int t = 0; t < T; ++t) {
            for (int c = 0; c < D; ++c) {
                float acc = B.dw_b[c];
                const float * w = B.dw_w.data() + static_cast<size_t>(c) * 7;
                for (int k = 0; k < 7; ++k) {
                    const int tt = t + k - 3;
                    if (tt < 0 || tt >= T) continue;
                    acc += w[k] * h[static_cast<size_t>(tt) * D + c];
                }
                dw[static_cast<size_t>(t) * D + c] = acc;
            }
        }
        // ln -> gelu -> pw1 -> pw2 -> gamma -> +residual
        std::vector<float> ln_buf(static_cast<size_t>(T) * D);
        for (int t = 0; t < T; ++t) {
            const float * x = dw.data() + static_cast<size_t>(t) * D;
            float mu = 0, var = 0;
            for (int c = 0; c < D; ++c) mu += x[c];
            mu /= D;
            for (int c = 0; c < D; ++c) {
                const float d = x[c] - mu;
                var += d * d;
            }
            var /= D;
            const float inv = 1.0F / std::sqrt(var + 1e-6F);
            for (int c = 0; c < D; ++c) {
                ln_buf[static_cast<size_t>(t) * D + c] = (x[c] - mu) * inv * B.n_w[c] + B.n_b[c];
            }
        }
        std::vector<float> mid(static_cast<size_t>(T) * IM);
        for (int t = 0; t < T; ++t) {
            for (int o = 0; o < IM; ++o) {
                const float * w = B.p1w.data() + static_cast<size_t>(o) * D;
                const float * x = ln_buf.data() + static_cast<size_t>(t) * D;
                float acc = B.p1b[o];
                for (int c = 0; c < D; ++c) acc += w[c] * x[c];
                // gelu exact
                const float xg = acc;
                const float k0 = 0.7978845608028654F;
                const float inner = k0 * xg * (1.0F + 0.044715F * xg * xg);
                acc = 0.5F * xg * (1.0F + std::tanh(inner));
                mid[static_cast<size_t>(t) * IM + o] = acc;
            }
        }
        for (int t = 0; t < T; ++t) {
            for (int o = 0; o < D; ++o) {
                const float * w = B.p2w.data() + static_cast<size_t>(o) * IM;
                const float * x = mid.data() + static_cast<size_t>(t) * IM;
                float acc = B.p2b[o];
                for (int c = 0; c < IM; ++c) acc += w[c] * x[c];
                h[static_cast<size_t>(t) * D + o] += B.gamma[o] * acc;
            }
        }
#ifdef F5_MEL_TEST
        if (bi == 0) { std::ofstream f("/tmp/cpp_vocos_blk0.bin", std::ios::binary);
          f.write(reinterpret_cast<const char*>(h.data()), h.size() * 4); }
#endif
    }

    // final layernorm
#ifdef F5_MEL_TEST
    {
        std::ofstream f("/tmp/cpp_vocos_h.bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(h.data()), h.size() * 4);
    }
#endif
    for (int t = 0; t < T; ++t) {
        float * x = h.data() + static_cast<size_t>(t) * D;
        float mu = 0, var = 0;
        for (int c = 0; c < D; ++c) mu += x[c];
        mu /= D;
        for (int c = 0; c < D; ++c) {
            const float d = x[c] - mu;
            var += d * d;
        }
        var /= D;
        const float inv = 1.0F / std::sqrt(var + 1e-6F);
        for (int c = 0; c < D; ++c) {
            x[c] = (x[c] - mu) * inv * v.final_nw[c] + v.final_nb[c];
        }
    }
#ifdef F5_MEL_TEST
    { std::ofstream f("/tmp/cpp_vocos_fln.bin", std::ios::binary);
      f.write(reinterpret_cast<const char*>(h.data()), h.size() * 4); }
#endif

    // head: [T, 1026] -> (mag 513, phase 513)
    const int n_freqs = kNfft / 2 + 1;
    std::vector<float> spec(static_cast<size_t>(T) * (2 * n_freqs));
    for (int t = 0; t < T; ++t) {
        const float * x = h.data() + static_cast<size_t>(t) * D;
        for (int o = 0; o < 2 * n_freqs; ++o) {
            const float * w = v.head_w.data() + static_cast<size_t>(o) * D;
            float acc = v.head_b[o];
            for (int c = 0; c < D; ++c) acc += w[c] * x[c];
            spec[static_cast<size_t>(t) * (2 * n_freqs) + o] = acc;
        }
    }

#ifdef F5_MEL_TEST
    { std::ofstream f("/tmp/cpp_head_spec.bin", std::ios::binary);
      f.write(reinterpret_cast<const char*>(spec.data()), spec.size() * 4); }
#endif
    // ISTFT (center, hann) — build full spectrum frames then overlap-add
    const int out_len = (T - 1) * kHop + kNfft;
    std::vector<float> out(static_cast<size_t>(out_len), 0.0F);
    std::vector<float> wsum(static_cast<size_t>(out_len), 0.0F);
    std::vector<float> hann(kNfft);
    for (int i = 0; i < kNfft; ++i) {
        hann[i] = 0.5F * (1.0F - std::cos(2.0F * static_cast<float>(M_PI) * i / kNfft));
    }
    std::vector<float> re(kNfft), im(kNfft);
    for (int t = 0; t < T; ++t) {
        const float * row = spec.data() + static_cast<size_t>(t) * (2 * n_freqs);
        std::fill(re.begin(), re.end(), 0.0F);
        std::fill(im.begin(), im.end(), 0.0F);
        for (int f = 0; f < n_freqs; ++f) {
            const float mag = std::min(std::exp(row[f]), 100.0F);
            const float ph = row[n_freqs + f];
            re[f] = mag * std::cos(ph);
            im[f] = mag * std::sin(ph);
            if (f > 0 && f < n_freqs - 1) {
                re[kNfft - f] = re[f];
                im[kNfft - f] = -im[f];
            }
        }
        re[n_freqs - 1] = im[n_freqs - 1] = 0.0F;  // nyquist bin real
        fft_inplace(re, im, true);
        const int start = t * kHop;
        for (int i = 0; i < kNfft; ++i) {
            out[static_cast<size_t>(start + i)] += re[i] * hann[i];
            wsum[static_cast<size_t>(start + i)] += hann[i] * hann[i];
        }
    }
#ifdef F5_MEL_TEST
    { std::ofstream f("/tmp/cpp_istft_raw.bin", std::ios::binary);
      f.write(reinterpret_cast<const char*>(out.data()), out.size() * 4); }
#endif
    std::vector<float> audio(static_cast<size_t>(T - 1) * kHop + 1);
    const int keep = static_cast<int>(audio.size());
    for (int i = 0; i < keep; ++i) {
        const size_t idx = i + kNfft / 2;  // center: drop first half-frame
        audio[static_cast<size_t>(i)] =
            idx < out.size() && wsum[idx] > 1e-8F ? out[idx] / wsum[idx] : 0.0F;
    }
    return audio;
}

}  // namespace

// ---- GPU vocoder (ggml graph; CUDA or CPU via F5ComputeDevice) ----
// Same math as vocos_decode above (verified vs torch at mel-corr 0.9963):
// embed conv k7 -> input LN -> 8x ConvNeXt (dwconv7, LN, pw1+GELU, pw2,
// gamma, residual) -> final LN -> head linear [T,1026]; the ISTFT tail runs
// on the host (O(n) overlap-add, ~minor vs the backbone matmuls).
namespace {

ggml_tensor * vlin(
    ggml_context * ctx,
    ggml_tensor * w,    // ne [in, out] (torch [out, in] order)
    ggml_tensor * b,    // ne [out]
    ggml_tensor * x) {  // ne [in, T]
    auto * out = ggml_mul_mat(ctx, w, x);
    auto * b2 = ggml_reshape_2d(ctx, b, ggml_nelements(b), 1);
    auto * r = ggml_repeat(ctx, b2, out);
    return ggml_add(ctx, out, r);
}

ggml_tensor * vln(
    ggml_context * ctx,
    ggml_tensor * x,          // ne [D, T]
    ggml_tensor * w, ggml_tensor * b) {  // ne [D]
    auto * n = ggml_norm(ctx, x, 1e-6F);
    auto * w2 = ggml_reshape_2d(ctx, w, ggml_nelements(w), 1);
    auto * b2 = ggml_reshape_2d(ctx, b, ggml_nelements(b), 1);
    return ggml_add(
        ctx, ggml_mul(ctx, n, ggml_repeat(ctx, w2, n)), ggml_repeat(ctx, b2, n));
}

// depthwise k7 pad3 over ne [C, T] (c fastest, time = ne1); wk host data
// (c*7+k), bias leaf [C]
ggml_tensor * leaf_zero_ret(
    ggml_context * ctx,
    int64_t c,
    int64_t t,
    const std::function<void(ggml_tensor *, size_t)> & leaf_zero) {
    auto * z = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c, t);
    leaf_zero(z, static_cast<size_t>(c) * t * sizeof(float));
    return z;
}

ggml_tensor * vdw7(
    ggml_context * ctx,
    ggml_tensor * x,
    const float * wk_host,
    ggml_tensor * b,
    const std::function<void(ggml_tensor *, const void *, size_t)> & leaf_write,
    const std::function<void(ggml_tensor *, size_t)> & leaf_zero) {
    const int64_t C = x->ne[0];
    const int64_t T = x->ne[1];
    auto * xp = ggml_concat(
        ctx,
        ggml_concat(ctx, leaf_zero_ret(ctx, C, 3, leaf_zero), x, 1),
        leaf_zero_ret(ctx, C, 3, leaf_zero),
        1);  // [C, T+6]
    ggml_tensor * out = nullptr;
    for (int k = 0; k < 7; ++k) {
        std::vector<float> wk(C);
        for (int c = 0; c < C; ++c) {
            wk[c] = wk_host[static_cast<size_t>(c) * 7 + k];
        }
        auto * wk_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, 1);
        leaf_write(wk_t, wk.data(), wk.size() * sizeof(float));
        auto * v = ggml_view_2d(
            ctx, xp, C, T, xp->nb[1], static_cast<size_t>(k) * xp->nb[1]);
        auto * term = ggml_mul(ctx, v, wk_t);  // [C,T] * [C,1] broadcast
        out = out == nullptr ? term : ggml_add(ctx, out, term);
    }
    auto * b2 = ggml_reshape_2d(ctx, b, ggml_nelements(b), 1);
    return ggml_add(ctx, out, ggml_repeat(ctx, b2, out));
}

struct VocosGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_backend_buffer_t io_buffer = nullptr;
    ggml_tensor * mel = nullptr;   // ne [100, T] (m fastest)
    ggml_tensor * spec = nullptr;  // ne [1026, T] (o fastest; row t at t*1026)
};

std::vector<float> vocos_decode_gpu(
    const std::string & vocos_path,
    const std::vector<float> & mel_rows,
    const F5ComputeDevice & dev) {
    const auto & v = load_vocos_once(vocos_path);
    const int T = static_cast<int>(mel_rows.size()) / kNMel;
    const int D = 512;
    const int IM = 1536;
    const int n_freqs = kNfft / 2 + 1;

    // one backend per cuda device (or cpu); leaked at exit (CUDA driver
    // shutdown cannot be ordered before buffer frees in static destruction)
    struct BackendOwnerV {
        ggml_backend_t value = nullptr;
    };
    static auto * owners = new std::map<int, BackendOwnerV>();
    const int key = dev.use_cuda ? dev.device : -1;
    ggml_backend_t backend = nullptr;
    if (dev.use_cuda) {
        auto ob = owners->find(key);
        if (ob == owners->end()) {
            core::BackendConfig cfg{core::BackendType::Cuda, dev.device, 1};
            ob = owners->emplace(key, BackendOwnerV{core::init_backend(cfg)}).first;
        }
        backend = ob->second.value;
    } else {
        auto ob = owners->find(key);
        if (ob == owners->end()) {
            core::BackendConfig cfg{core::BackendType::Cpu, 0, std::max(1, dev.threads)};
            ob = owners->emplace(key, BackendOwnerV{core::init_backend(cfg)}).first;
        }
        backend = ob->second.value;
    }
    const bool is_cuda = dev.use_cuda;

    // graph cache per (T, device)
    static auto * cache = new std::map<std::pair<int, int>, std::unique_ptr<VocosGraph>>();
    const auto ckey = std::make_pair(T, key);
    auto it = cache->find(ckey);
    if (it == cache->end()) {
        auto g = std::make_unique<VocosGraph>();
        const size_t ctx_bytes = 256ULL << 20;
        g->ctx = ggml_init({ctx_bytes, nullptr, is_cuda});
        ggml_context * ctx = g->ctx;
        std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> pending;
        const auto leaf_write = [&](ggml_tensor * t, const void * src, size_t bytes) {
            if (!is_cuda) {
                std::memcpy(t->data, src, bytes);
            } else {
                const auto * b = static_cast<const uint8_t *>(src);
                pending.emplace_back(t, std::vector<uint8_t>(b, b + bytes));
            }
        };
        const auto leaf_zero = [&](ggml_tensor * t, size_t bytes) {
            if (!is_cuda) {
                std::memset(t->data, 0, bytes);
            } else {
                pending.emplace_back(t, std::vector<uint8_t>(bytes, 0));
            }
        };
        auto leaf_f32v = [&](const std::vector<float> & src) {
            auto * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, static_cast<int64_t>(src.size()));
            leaf_write(t, src.data(), src.size() * sizeof(float));
            return t;
        };

        // mel leaf: ne [100, T], m fastest — mel_rows buffer is exactly that
        auto * mel = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kNMel, T);
        // weight leaves (torch order == needed ggml order, see notes)
        auto * embed_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, kNMel, D);
        leaf_write(embed_w, v.embed_w.data(), v.embed_w.size() * sizeof(float));
        auto * embed_b = leaf_f32v(v.embed_b);
        ggml_tensor * var_h = nullptr;
        auto * input_nw = leaf_f32v(v.input_nw);
        auto * input_nb = leaf_f32v(v.input_nb);
        auto * final_nw = leaf_f32v(v.final_nw);
        auto * final_nb = leaf_f32v(v.final_nb);
        auto * head_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 2 * n_freqs);
        leaf_write(head_w, v.head_w.data(), v.head_w.size() * sizeof(float));
        auto * head_b = leaf_f32v(v.head_b);

        // embed conv (dense k7, pad 3) via im2col + matmul on time-fastest rows
        auto * t_fast = ggml_cont(ctx, ggml_transpose(ctx, mel));  // ne [T, 100]
        {
            // im2col: weight ne [k, cin, cout]; cols ne [cin*k, T]
            auto * in3 = ggml_reshape_3d(ctx, t_fast, T, kNMel, 1);
            auto * cols = ggml_im2col(
                ctx, embed_w, in3, 1, 1, 3, 0, 1, 1, false, GGML_TYPE_F32);
            auto * w2 = ggml_reshape_2d(ctx, embed_w, kNMel * 7, D);
            auto * y = ggml_mul_mat(ctx, w2, cols);  // [D, T] feature-fastest
            auto * b2_ = ggml_reshape_2d(ctx, embed_b, D, 1);
            auto * yb = ggml_add(ctx, y, ggml_repeat(ctx, b2_, y));
            var_h = ggml_cont(ctx, yb);
        }
        auto * h = var_h;

        h = vln(ctx, h, input_nw, input_nb);
        ggml_set_name(h, "v_after_input_ln");

        for (int bi = 0; bi < 8; ++bi) {
            const auto & B = v.blocks[bi];
            auto * dwb = leaf_f32v(B.dw_b);
            auto * nw = leaf_f32v(B.n_w);
            auto * nb = leaf_f32v(B.n_b);
            auto * p1w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, IM);
            leaf_write(p1w, B.p1w.data(), B.p1w.size() * sizeof(float));
            auto * p1b = leaf_f32v(B.p1b);
            auto * p2w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, IM, D);
            leaf_write(p2w, B.p2w.data(), B.p2w.size() * sizeof(float));
            auto * p2b = leaf_f32v(B.p2b);
            auto * gam = leaf_f32v(B.gamma);

            auto * dw = vdw7(ctx, h, B.dw_w.data(), dwb, leaf_write, leaf_zero);
            ggml_set_name(dw, "v_dw");
            auto * ln = vln(ctx, dw, nw, nb);
            auto * mid = vlin(ctx, p1w, p1b, ln);
            mid = ggml_gelu(ctx, mid);  // exact erf
            auto * up = vlin(ctx, p2w, p2b, mid);  // [512, T]
            auto * g2 = ggml_reshape_2d(ctx, gam, D, 1);
            up = ggml_mul(ctx, up, ggml_repeat(ctx, g2, up));
            h = ggml_add(ctx, h, up);
        }

        h = vln(ctx, h, final_nw, final_nb);
        auto * spec = vlin(ctx, head_w, head_b, h);  // ne [1026, T]

        g->mel = mel;
        g->spec = spec;
        g->graph = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(g->graph, spec);
        core::validate_backend_graph_supported(backend, g->graph, "f5_vocos");
        if (is_cuda) {
            g->io_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            for (auto & leaf : pending) {
                ggml_backend_tensor_set(leaf.first, leaf.second.data(), 0, leaf.second.size());
            }
            g->gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (g->gallocr == nullptr || !ggml_gallocr_reserve(g->gallocr, g->graph) ||
                !ggml_gallocr_alloc_graph(g->gallocr, g->graph)) {
                throw std::runtime_error("vocos CUDA graph alloc failed");
            }
        }
        it = cache->emplace(ckey, std::move(g)).first;
    }
    VocosGraph & g = *it->second;

    // upload mel + compute
    if (is_cuda) {
        ggml_backend_tensor_set(g.mel, mel_rows.data(), 0, mel_rows.size() * sizeof(float));
    } else {
        std::memcpy(g.mel->data, mel_rows.data(), mel_rows.size() * sizeof(float));
    }
    const auto status = is_cuda
        ? core::compute_backend_graph(backend, g.graph, nullptr, "f5_vocos")
        : f5_cpu_graph_compute(g.ctx, g.graph,
              dev.threads > 0 ? dev.threads
                              : static_cast<int>(std::thread::hardware_concurrency()));
    if (is_cuda) {
        ggml_backend_synchronize(backend);
    }
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("vocos graph compute failed");
    }
    // spec: ne [1026, T] o-fastest == row t at t*1026 — same as host layout
    std::vector<float> spec(static_cast<size_t>(T) * 2 * n_freqs);
    if (is_cuda) {
        ggml_backend_tensor_get(g.spec, spec.data(), 0, spec.size() * sizeof(float));
    } else {
        std::memcpy(spec.data(), ggml_get_data(g.spec), spec.size() * sizeof(float));
    }

    // host ISTFT tail (identical to vocos_decode)
    const int out_len = (T - 1) * kHop + kNfft;
    std::vector<float> out(static_cast<size_t>(out_len), 0.0F);
    std::vector<float> wsum(static_cast<size_t>(out_len), 0.0F);
    std::vector<float> hann(kNfft);
    for (int i = 0; i < kNfft; ++i) {
        hann[i] = 0.5F * (1.0F - std::cos(2.0F * static_cast<float>(M_PI) * i / kNfft));
    }
    std::vector<float> re(kNfft), im(kNfft);
    for (int t = 0; t < T; ++t) {
        const float * row = spec.data() + static_cast<size_t>(t) * (2 * n_freqs);
        std::fill(re.begin(), re.end(), 0.0F);
        std::fill(im.begin(), im.end(), 0.0F);
        for (int f = 0; f < n_freqs; ++f) {
            const float mag = std::min(std::exp(row[f]), 100.0F);
            const float ph = row[n_freqs + f];
            re[f] = mag * std::cos(ph);
            im[f] = mag * std::sin(ph);
            if (f > 0 && f < n_freqs - 1) {
                re[kNfft - f] = re[f];
                im[kNfft - f] = -im[f];
            }
        }
        re[n_freqs - 1] = im[n_freqs - 1] = 0.0F;
        fft_inplace(re, im, true);
        const int start = t * kHop;
        for (int i = 0; i < kNfft; ++i) {
            out[static_cast<size_t>(start + i)] += re[i] * hann[i];
            wsum[static_cast<size_t>(start + i)] += hann[i] * hann[i];
        }
    }
    std::vector<float> audio(static_cast<size_t>(T - 1) * kHop + 1);
    const int keep = static_cast<int>(audio.size());
    for (int i = 0; i < keep; ++i) {
        const size_t idx = static_cast<size_t>(i) + kNfft / 2;
        audio[static_cast<size_t>(i)] =
            idx < out.size() && wsum[idx] > 1e-8F ? out[idx] / wsum[idx] : 0.0F;
    }
    return audio;
}

}  // namespace

namespace {

// UTF-8-aware sentence splitting: returns byte offsets that never split a
// multi-byte character. Break after . ! ? ؛ ، and newlines.
static bool is_break_byte_here(const std::string & text, size_t i, size_t * len) {
    const unsigned char c = static_cast<unsigned char>(text[i]);
    if (c == '.' || c == '!' || c == '?' || c == '\n') {
        *len = 1;
        return true;
    }
    // Arabic semicolon U+061B (D8 9B) and comma U+060C (D8 8C)
    if (c == 0xD8 && i + 1 < text.size()) {
        const unsigned char c2 = static_cast<unsigned char>(text[i + 1]);
        if (c2 == 0x9B || c2 == 0x8C) {
            *len = 2;
            return true;
        }
    }
    return false;
}

static size_t utf8_char_count(const std::string & s) {
    size_t n = 0;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        i += c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : 4;
        ++n;
    }
    return n;
}

// Split text so each chunk has at most max_chars UTF-8 characters,
// preferring sentence/clause boundaries.
std::vector<std::string> chunk_text(const std::string & text, size_t max_chars) {
    std::vector<std::string> chunks;
    if (utf8_char_count(text) <= max_chars) {
        chunks.push_back(text);
        return chunks;
    }
    // collect sentence pieces [start, end) breaking AFTER break chars
    std::vector<std::pair<size_t, size_t>> pieces;
    size_t piece_start = 0;
    for (size_t i = 0; i < text.size();) {
        size_t blen = 0;
        if (is_break_byte_here(text, i, &blen)) {
            pieces.emplace_back(piece_start, i + blen);
            piece_start = i + blen;
            i += blen;
            continue;
        }
        const unsigned char c = static_cast<unsigned char>(text[i]);
        i += c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : 4;
    }
    if (piece_start < text.size()) {
        pieces.emplace_back(piece_start, text.size());
    }
    // split every piece into <= max_chars slices (oversize sentences too),
    // snapping each cut to a WORD boundary: a mid-word slice makes the next
    // chunk start mid-word right after the reference prompt, and the model
    // drops the straddled word (observed: "لن |"يحتفظ" split). Falls back to
    // a hard cut only when the window contains no space at all.
    std::vector<std::pair<size_t, size_t>> slices;
    for (const auto & [ps, pe] : pieces) {
        size_t cs = ps;
        while (cs < pe) {
            size_t cnt = 0, ce = cs;
            while (ce < pe && cnt < max_chars) {
                const unsigned char c = static_cast<unsigned char>(text[ce]);
                ce += c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : 4;
                ++cnt;
            }
            if (ce < pe) {
                // back off to just after the last space inside the window
                for (size_t k = ce; k > cs + 1; --k) {
                    if (text[k - 1] == ' ') {
                        ce = k;
                        break;
                    }
                }
            }
            slices.emplace_back(cs, ce);
            cs = ce;
        }
    }
    for (const auto & [ss, se] : slices) {
        const std::string piece = text.substr(ss, se - ss);
        const size_t pc = utf8_char_count(piece);
        if (!chunks.empty()) {
            const std::string & prev = chunks.back();
            const size_t prev_c = utf8_char_count(prev);
            // merge when it fits, or when the piece is tiny (< 12 chars):
            // tiny chunks destabilize the sampler (observed NaN on 5 chars).
            // Absorbing beyond the size budget overflows the duration
            // estimate into the 1024-frame cap -> clipped trailing words, so
            // the allowance stays within the chunk-sizing safety margin.
            const bool tiny = pc < 12;
            const bool fits = prev_c + pc <= max_chars;
            const bool absorb = tiny && prev_c + pc <= max_chars + 4;
            if (fits || absorb) {
                chunks.back() = prev + piece;
                continue;
            }
        }
        chunks.push_back(piece);
    }
    return chunks;
}

struct ChunkResult {
    std::vector<float> gen_mel_rows;  // [gen][100]
    int gen_frames = 0;
    int duration_real = 0;
};

// Trim head/tail silence from a chunk's generated mel rows. The model parks
// unused duration slack as long pauses at chunk EDGES (observed: 1.5-3s
// pauses mid-text with a fast-paced reference), which lands between random
// word pairs once chunks are concatenated. Speech log-mel row means sit far
// above the silence floor (speech ~-0.5, pauses ~-6..-8), so -4.0 separates
// them robustly. A short head/tail is kept for natural word spacing; a chunk
// ending on sentence-final punctuation keeps a longer tail pause.
void trim_chunk_mel_silence(
    std::vector<float> & rows, int & gen_frames, bool sentence_final_tail) {
    if (gen_frames <= 0) return;
    const float kThresh = -4.0F;
    const auto row_mean = [&](int t) {
        const float * r = rows.data() + static_cast<size_t>(t) * kNMel;
        float acc = 0.0F;
        for (int m = 0; m < kNMel; ++m) acc += r[m];
        return acc / kNMel;
    };
    int first = 0;
    while (first < gen_frames && row_mean(first) < kThresh) ++first;
    first = std::max(0, first - 9);  // keep ~0.1s lead-in
    int last = gen_frames - 1;
    while (last >= first && row_mean(last) < kThresh) --last;
    if (last < first) return;  // all-silent chunk (pathological): leave as-is
    const int tail_keep = sentence_final_tail ? 47 : 24;  // ~0.5s / ~0.26s
    last = std::min(gen_frames - 1, last + tail_keep);
    const int keep = last - first + 1;
    if (keep >= gen_frames) return;
    std::vector<float> trimmed(static_cast<size_t>(keep) * kNMel);
    std::memcpy(trimmed.data(), rows.data() + static_cast<size_t>(first) * kNMel,
                trimmed.size() * sizeof(float));
    rows = std::move(trimmed);
    gen_frames = keep;
}


// Total mel-frame budget for one CFM pass (ref + generated). 2048 allows
// sentence-scale chunks (fewer seams, splits land on clause boundaries —
// the 1024 budget forced ~57-char mid-word slices that dropped straddled
// words) at ~6 GiB peak on an RTX 3090 (measured). F5_FRAME_BUDGET
// overrides for tuning.
int frame_budget() {
    static const int budget = [] {
        const char * env = std::getenv("F5_FRAME_BUDGET");
        if (env != nullptr) {
            const int v = std::atoi(env);
            if (v >= 256 && v <= 8192) return v;
        }
        return 2048;
    }();
    return budget;
}

// One CFM pass for a single chunk: the original pipeline verbatim.
ChunkResult synthesize_chunk(
    const std::string & model_path,
    const F5SynthesisRequest & request,
    const std::vector<float> & ref_mel_cols,  // [100][ref_frames]
    int ref_frames,
    int ref_voiced_frames,
    const std::vector<int32_t> & chunk_ids,
    const std::string & chunk_text_string,
    const std::string & chunk_ref_text,
    F5ComputeDevice & dev,
    uint32_t seed,
    std::vector<float> * out_final_latent_rows,
    double duration_slack = 1.0,
    int tail_pad = 0) {
    const F5Architecture arch;

    // Pacing in CHARACTERS (Arabic is 2 bytes/char; byte-based pacing
    // underestimates duration ~1.8x). The reference's speaking rate sets the
    // expectation, bounded by a normal-speech ceiling so an unusually slow
    // reference cannot drag generated speech into a compressed clamp.
    const int gen_chars = static_cast<int>(utf8_char_count(chunk_text_string));
    const int ref_chars = std::max(1, static_cast<int>(utf8_char_count(chunk_ref_text)));
    const double ref_rate = static_cast<double>(ref_voiced_frames) / ref_chars;  // frames/char
    // ceiling only guards pathological refs (e.g. mostly silence); the
    // reference's speaking rate drives pacing, so slow refs stay slow
    constexpr double kMaxFramesPerChar = 93.75 / 2.5;  // >= 2.5 chars/s floor... (frames/char ceiling)
    const double rate = std::max(std::min(ref_rate, kMaxFramesPerChar), 93.75 / 14.0);
    float local_speed = request.speed;
    if (gen_chars < 10) local_speed = 0.3F;
    // duration_slack (>1 for chunked long-form): the rate estimate has zero
    // tolerance for run-to-run pace variance; mid-text chunk tails end on
    // whole words, and any undershoot clips the last word (observed: "النفط",
    // "فقط" dropped at chunk tails). Excess frames become a short tail pause.
    int duration = ref_frames + static_cast<int>(rate * gen_chars * duration_slack / local_speed);
    // tail_pad (>0 for single-chunk short text): without it a slightly slow
    // sampled pace runs out of frames and the final phonemes are clipped
    // (observed: "أين اللون الأحمر؟" -> "الأخر"). ~0.2s is proportionally
    // negligible for long text and ends as silence, not slower speech.
    duration += tail_pad;
    duration = std::max(duration, gen_chars + 1);
    // per-chunk safety cap: a single chunk never exceeds the graph budget;
    // longer inputs are split upstream by chunk_text instead of truncated.
    const int kChunkFrameCap = frame_budget();
    if (duration > kChunkFrameCap) duration = kChunkFrameCap;
    const int duration_real = duration;
    duration = (duration + 63) / 64 * 64;  // graph bucket reuse

    std::vector<float> cond(static_cast<size_t>(duration) * kNMel, 0.0F);
    for (int t = 0; t < ref_frames && t < duration; ++t) {
        for (int m = 0; m < kNMel; ++m) {
            cond[static_cast<size_t>(t) * kNMel + m] =
                ref_mel_cols[static_cast<size_t>(m) * ref_frames + t];
        }
    }
    Rng rng(seed);
    std::vector<float> y(static_cast<size_t>(duration) * kNMel);
    for (auto & val : y) val = rng.normal();

    const auto ts = sway_timesteps(request.steps, request.sway_sampling_coef);
    for (size_t i = 0; i + 1 < ts.size(); ++i) {
        const float t = ts[i];
        const float dt = ts[i + 1] - ts[i];
        const auto pair = f5_dit_forward_cfg(
            model_path, y, cond, chunk_ids, t, duration, arch, &dev);
        const auto & v_cond = pair.first;
        const auto & v_null = pair.second;
        for (size_t k = 0; k < y.size(); ++k) {
            const float v = v_cond[k] + (v_cond[k] - v_null[k]) * request.cfg_strength;
            y[k] += dt * v;
        }
    }
    for (int t = 0; t < ref_frames && t < duration; ++t) {
        for (int m = 0; m < kNMel; ++m) {
            y[static_cast<size_t>(t) * kNMel + m] = cond[static_cast<size_t>(t) * kNMel + m];
        }
    }
    if (out_final_latent_rows != nullptr) {
        *out_final_latent_rows = y;
    }
    ChunkResult out;
    const int gen_frames = std::max(0, duration_real - ref_frames);
    out.gen_frames = gen_frames;
    out.duration_real = duration_real;
    out.gen_mel_rows.resize(static_cast<size_t>(gen_frames) * kNMel);
    for (int t = 0; t < gen_frames; ++t) {
        for (int m = 0; m < kNMel; ++m) {
            out.gen_mel_rows[static_cast<size_t>(t) * kNMel + m] =
                y[static_cast<size_t>(ref_frames + t) * kNMel + m];
        }
    }
    return out;
}

}  // namespace

F5SynthesisResult f5_synthesize(
    const std::string & model_path,
    const std::string & vocos_path,
    const F5SynthesisRequest & request) {
    const auto t0 = std::chrono::steady_clock::now();
    F5SynthesisResult result;

    const std::string dir = std::filesystem::path(model_path).parent_path().string();
    const auto vocab = load_vocab(dir);

    // ref audio -> 24k mono -> normalize to the training RMS -> mel
    // (Python F5: audio *= target_rms / rms when rms < target_rms; the
    // generated wave is scaled back at the end. Skipping this feeds the model
    // a conditioning mel far below its training distribution.)
    auto ref24 = resample(request.ref_audio, request.ref_sample_rate, kSampleRate);
    double ref_rms = 0.0;
    for (const auto v : ref24) ref_rms += double(v) * v;
    ref_rms = std::sqrt(ref_rms / std::max<size_t>(1, ref24.size()));
    constexpr double kTargetRms = 0.1;
    float ref_gain = 1.0F;
    if (std::getenv("F5_NO_RMS_NORM") == nullptr &&
        ref_rms > 0.0 && ref_rms < kTargetRms) {
        ref_gain = static_cast<float>(kTargetRms / ref_rms);
        for (auto & v : ref24) v *= ref_gain;
    }
    auto ref_mel = compute_mel(ref24);
    int ref_frames = static_cast<int>(ref_mel.size()) / kNMel;
    // The reference may use at most half the frame budget so a meaningful
    // generation budget remains. CRITICAL: the ref audio and ref_text must
    // stay aligned — truncating the audio while keeping the full transcript
    // makes the model speak the UNSAMPLED remainder of the transcript into
    // the generated region (observed: EGY ref leaked "استخدمه هيعجبك اوي"
    // when its 7.84s audio was cut to 5.46s).
    const int kMaxRefFrames = frame_budget() / 2;
    if (ref_frames > kMaxRefFrames) {
        std::fprintf(stderr,
            "F5-TTS: reference audio is %.1fs (%d frames) — truncated to %d frames (half the "
            "frame budget). The transcript tail will leak into the output; use a reference "
            "shorter than %.1fs or raise F5_FRAME_BUDGET.\n",
            ref_frames / 93.75, ref_frames, kMaxRefFrames, kMaxRefFrames / 93.75);
        std::vector<float> trimmed(static_cast<size_t>(kMaxRefFrames) * kNMel);
        for (int t = 0; t < kMaxRefFrames; ++t) {
            for (int m = 0; m < kNMel; ++m) {
                trimmed[static_cast<size_t>(m) * kMaxRefFrames + t] =
                    ref_mel[static_cast<size_t>(m) * ref_frames + t];
            }
        }
        ref_mel = std::move(trimmed);
        ref_frames = kMaxRefFrames;
    }

    // Voiced ref frames for the pacing rate: ref_frames/ref_chars counts the
    // reference's internal pauses as speech time, so a dramatic/paused
    // reference (the EGY sample) inflates the frames-per-char rate and the
    // model parks the excess duration as long mid-text pauses. Count only
    // frames whose log-mel row mean is above the silence floor.
    int ref_voiced_frames = 0;
    for (int t = 0; t < ref_frames; ++t) {
        float acc = 0.0F;
        for (int m = 0; m < kNMel; ++m) {
            acc += ref_mel[static_cast<size_t>(m) * ref_frames + t];
        }
        if (acc / kNMel > -4.0F) ++ref_voiced_frames;
    }
    ref_voiced_frames = std::max(1, ref_voiced_frames);

    F5ComputeDevice dev;
    dev.use_cuda = request.use_cuda;
    dev.device = request.cuda_device;
    dev.threads = request.threads;

    // ---- chunk long texts instead of truncating: each chunk is sized by
    // the DURATION budget (frames/char from the reference, capped) so no
    // chunk ever needs clamping; chunk N+1 is conditioned on the tail of
    // chunk N (voice and prosody continuity across seams) ----
    const int ref_chars0 = std::max<int>(1, static_cast<int>(utf8_char_count(request.ref_text)));
    const double rate0 = std::max(
        std::min(static_cast<double>(ref_voiced_frames) / ref_chars0, 93.75 / 2.5),
        93.75 / 14.0);
    const int gen_budget = frame_budget() - ref_frames;  // frames a chunk may generate
    // chars per chunk: budget / rate with a safety margin, so the duration
    // estimate NEVER engages the 1024-frame cap (cap = compressed speech).
    // The absolute floor is small: a slow reference legitimately means short
    // chunks (its 5.5 s reference eats most of the frame budget).
    size_t chars_per_chunk = std::max<size_t>(
        12, static_cast<size_t>(gen_budget / rate0 * 0.92));
    if (std::getenv("F5_SINGLE_CHUNK") != nullptr) {
        chars_per_chunk = 1u << 30;  // debug: never chunk
    }
    const auto chunks = chunk_text(request.text, chars_per_chunk);
    std::vector<float> all_rows;
    const std::string ref_text = apply_ref_trailing_space(request.ref_text);
    // Seed semantics match python F5 (seed=None -> fresh randomness every
    // run): an unspecified seed draws a random base seed per request, so a
    // sampling accident (e.g. a swallowed word like "هرمز") can be re-rolled
    // instead of being replayed identically on every request. fixed_seed
    // keeps deterministic per-chunk seeds (seed + chunk_index).
    const uint32_t base_seed = request.fixed_seed
        ? request.seed
        : (static_cast<uint32_t>(std::random_device{}()) ^
           static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count()));
    // Every chunk is conditioned on the ORIGINAL reference (vanilla F5
    // chunking semantics): chained references drift the voice and decay
    // the energy chunk over chunk (observed: two voices + fade to silence).
    for (size_t ci = 0; ci < chunks.size(); ++ci) {
        const std::string full = assemble_chunk_text(request.dialect, ref_text, chunks[ci]);
        const auto chunk_ids = tokenize_text(vocab, full);
        auto out = synthesize_chunk(
            model_path, request, ref_mel, ref_frames, ref_voiced_frames, chunk_ids,
            chunks[ci], ref_text, dev,
            base_seed + static_cast<uint32_t>(ci),
            nullptr, chunks.size() > 1 ? 1.20 : 1.0,
            chunks.size() > 1 ? 0 : 20);
        if (chunks.size() > 1) {
            const char last_ch = chunks[ci].empty() ? ' ' : chunks[ci].back();
            const bool sent_final = last_ch == '.' || last_ch == '!' || last_ch == '?';
            trim_chunk_mel_silence(out.gen_mel_rows, out.gen_frames, sent_final);
        }
        all_rows.insert(all_rows.end(), out.gen_mel_rows.begin(), out.gen_mel_rows.end());
    }

    result.audio = request.use_cuda
        ? vocos_decode_gpu(vocos_path, all_rows, dev)
        : vocos_decode(vocos_path, all_rows);
    // undo the reference normalization on the output (Python F5 parity)
    if (ref_gain != 1.0F) {
        double out_rms = 0.0;
        for (const auto v : result.audio) out_rms += double(v) * v;
        out_rms = std::sqrt(out_rms / std::max<size_t>(1, result.audio.size()));
        if (out_rms > 0.0 && out_rms < kTargetRms) {
            const float undo = static_cast<float>(out_rms / kTargetRms);
            for (auto & v : result.audio) v *= undo;
        }
    }
    result.sample_rate = kSampleRate;
    result.generation_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    return result;
}


#ifdef F5_MEL_TEST
std::vector<float> f5_test_mel(const std::vector<float> & wav) { return compute_mel(wav); }
std::vector<float> f5_test_vocos(const std::string & vp, const std::vector<float> & mel) { return vocos_decode(vp, mel); }
std::vector<float> f5_test_vocos_gpu(const std::string & vp, const std::vector<float> & mel, const F5ComputeDevice & dev) { return vocos_decode_gpu(vp, mel, dev); }
std::vector<int32_t> f5_test_token_ids(
    const std::string & model_path,
    const std::string & dialect,
    const std::string & ref_text,
    const std::string & gen_text) {
    const std::string dir = std::filesystem::path(model_path).parent_path().string();
    const auto vocab = load_vocab(dir);
    const std::string full = assemble_chunk_text(
        dialect, apply_ref_trailing_space(ref_text), gen_text);
    return tokenize_text(vocab, full);
}
#endif

}  // namespace engine::models::f5_tts
