// xtts_v2 GPT engine -- GPT-2 AR + ConditioningEncoder + Perceiver resampler
// MSVC's <cmath> only exposes M_PI with _USE_MATH_DEFINES; define it first.
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include "engine/community_models/xtts_v2/gpt.h"
#include "engine/community_models/xtts_v2/assets.h"
#include "engine/community_models/xtts_v2/tokenizer_text.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <cmath>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
static inline int xtts_ctz32(unsigned v) { unsigned long r; _BitScanForward(&r, v); return (int)r; }
#else
static inline int xtts_ctz32(unsigned v) { return __builtin_ctz(v); }
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::xtts_v2 {
namespace {

struct WL { ggml_context* ctx = nullptr; ggml_backend_buffer_t buf = nullptr;
            std::map<std::string, ggml_tensor*> tensors; };
static bool load_gguf(const char* path, ggml_backend_t be, WL& wl) {
    gguf_init_params gp = {true, &wl.ctx};
    gguf_context* gu = gguf_init_from_file(path, gp);
    if (!gu || !wl.ctx) { fprintf(stderr, "xtts_v2: GGUF load failed: %s\n", path); if (gu) gguf_free(gu); return false; }
    auto buf = ggml_backend_alloc_ctx_tensors(wl.ctx, be);
    if (!buf) { gguf_free(gu); ggml_free(wl.ctx); wl.ctx = nullptr; return false; }
    wl.buf = buf;
    FILE* fp = fopen(path, "rb");
    if (!fp) { ggml_backend_buffer_free(buf); gguf_free(gu); ggml_free(wl.ctx); wl.ctx = nullptr; return false; }
    size_t doff = gguf_get_data_offset(gu);
    for (ggml_tensor* t = ggml_get_first_tensor(wl.ctx); t; t = ggml_get_next_tensor(wl.ctx, t)) {
        int64_t tid = gguf_find_tensor(gu, ggml_get_name(t));
        if (tid < 0) continue;
        size_t off = gguf_get_tensor_offset(gu, tid), nbytes = ggml_nbytes(t);
        std::vector<uint8_t> tbuf(nbytes);
        if (fseek(fp, doff + off, SEEK_SET) != 0 || fread(tbuf.data(), 1, nbytes, fp) != nbytes) break;
        ggml_backend_tensor_set(t, tbuf.data(), 0, nbytes);
    }
    fclose(fp); gguf_free(gu);
    for (ggml_tensor* t = ggml_get_first_tensor(wl.ctx); t; t = ggml_get_next_tensor(wl.ctx, t))
        wl.tensors[ggml_get_name(t)] = t;
    return true;
}
static ggml_tensor* T(const std::map<std::string, ggml_tensor*>& m, const char* n) {
    auto it = m.find(n); return (it != m.end()) ? it->second : nullptr;
}
static ggml_tensor* T(const std::map<std::string, ggml_tensor*>& m, const std::string& n) { return T(m, n.c_str()); }
static ggml_tensor* req(const std::map<std::string, ggml_tensor*>& m, const char* n) {
    auto* t = T(m, n); if (!t) fprintf(stderr, "xtts: missing %s\n", n); return t;
}
static uint32_t kv_u32(gguf_context* g, const char* k, uint32_t d) {
    int ki = gguf_find_key(g, k); if (ki < 0 || gguf_get_kv_type(g, ki) != GGUF_TYPE_UINT32) return d;
    return static_cast<const uint32_t*>(gguf_get_arr_data(g, ki))[0];
}

static void h_ln(float* o, const float* i, const float* g, const float* b, int d) {
    float m = 0; for (int j = 0; j < d; j++) m += i[j]; m /= d;
    float v = 0; for (int j = 0; j < d; j++) { float f = i[j] - m; v += f * f; }
    float inv = 1.0f / sqrtf(v / d + 1e-5f);
    for (int j = 0; j < d; j++) { float x = (i[j] - m) * inv; if (g) x *= g[j]; if (b) x += b[j]; o[j] = x; }
}
// GroupNorm32 over [C, T] layout: G groups of C/G channels each; stats over group channels x all T.
// gamma/beta are per-channel [C]. x is [T][C] row-major (our internal layout).
static void h_gn32(float* x, const float* g, const float* b, int C, int T, int G) {
    int cg = C / G;
    std::vector<float> mean(G), var(G);
    // GroupNorm stats per group over (C/G channels) x T positions:
    for (int gr = 0; gr < G; gr++) {
        double s = 0, s2 = 0; int n = 0;
        for (int t = 0; t < T; t++) for (int c = gr * cg; c < (gr + 1) * cg; c++) {
            double v = x[(size_t)t * C + c]; s += v; s2 += v * v; n++;
        }
        mean[gr] = (float)(s / n); var[gr] = (float)(s2 / n - (s / n) * (s / n));
        float inv = 1.0f / sqrtf(var[gr] + 1e-5f);
        for (int t = 0; t < T; t++) for (int c = gr * cg; c < (gr + 1) * cg; c++) {
            float& xv = x[(size_t)t * C + c];
            xv = (xv - mean[gr]) * inv * g[c] + b[c];
        }
    }
}
static void h_c1d(float* o, const float* W, const float* X, const float* b, int ni, int no) {
#pragma omp parallel for schedule(static)
    for (int m = 0; m < no; m++) {
        float s = b ? b[m] : 0;
        const float* Wm = W + m * ni;
        for (int n = 0; n < ni; n++) s += Wm[n] * X[n];
        o[m] = s;
    }
}
static float gelu_t(float x) { return 0.5f * x * (1.0f + tanhf(x * 0.79788456f + 0.044715f * x * x * x)); }
static void get_emb(ggml_tensor* w, int t, float* o, int d) {
    int vocab = (int)w->ne[1];
    if (t < 0 || t >= vocab) { std::fill(o, o + d, 0.0f); return; }
    size_t co = (size_t)t * w->nb[1];
    size_t row_bytes = w->nb[1];
    if (w->type == GGML_TYPE_F32) ggml_backend_tensor_get(w, o, co, d * sizeof(float));
    else {
      std::vector<uint8_t> tmp(row_bytes);
      ggml_backend_tensor_get(w, tmp.data(), co, row_bytes);
      const auto* tt = ggml_get_type_traits(w->type); if (tt && tt->to_float) tt->to_float(tmp.data(), o, d); }
}
// Converts a GGML tensor to row-major [out_features][in_features] float vector.
// Handles both 3D conv1d weights [kernel=1, in_ch, out_ch] and 2D linear weights [in, out].
// GGML linear weight: ne[0]=in_features, ne[1]=out_features, data is row-major [out][in]
//   (i.e. already [out][in] when read contiguously - no transpose needed).
// GGML conv1d weight (kernel=1): ne[0]=1, ne[1]=in_ch, ne[2]=out_ch,
//   data layout is [out_ch][in_ch] contiguous - also already [out][in].
// In both cases the raw data is already [out][in] row-major, so just read it flat.
static std::vector<float> tconv1d(ggml_tensor* t) {
    if (!t) return {};
    size_t total = ggml_nelements(t);
    std::vector<float> r(total);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, r.data(), 0, total * sizeof(float));
    } else {
        size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> tmp(nbytes);
        ggml_backend_tensor_get(t, tmp.data(), 0, nbytes);
        const auto* tt = ggml_get_type_traits(t->type);
        if (tt && tt->to_float) tt->to_float(tmp.data(), r.data(), total);
    }
    return r;
}
static std::vector<float> comp_mel(const float* pcm, int ns, int sr, int nfft, int hop, int win, int nm, float fmin, float fmax) {
    std::vector<float> hann(win); for (int i = 0; i < win; i++) hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / win));
    // torchaudio MelSpectrogram defaults: mel_scale="htk", norm="slaney" (area norm only),
    // center=True with pad_mode="reflect". HTK mel: 2595*log10(1+f/700).
    int nf = nfft / 2 + 1;
    // torchaudio torch.stft: win_length(400) window is CENTER-padded to n_fft (1848 zeros each side)
    int wpad = (nfft - win) / 2;
    auto hz_to_mel = [](float f) { return 2595.0f * log10f(1.0f + f / 700.0f); };
    auto mel_to_hz = [](float m) { return 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f); };
    float fm1 = hz_to_mel(fmin), fm2 = hz_to_mel(fmax); std::vector<float> mp(nm + 2);
    for (int i = 0; i <= nm + 1; i++) mp[i] = mel_to_hz(fm1 + (fm2 - fm1) * i / (nm + 1));
    std::vector<float> fb(nm * nf, 0);
    for (int m = 0; m < nm; m++) {
        float enorm = 2.0f / (mp[m + 2] - mp[m] + 1e-10f);  // Slaney norm: 2/(f[m+2]-f[m]) in Hz
        for (int f = 0; f < nf; f++) {
            float fq = (float)sr * f / nfft;
            if (fq >= mp[m] && fq <= mp[m + 2]) fb[m * nf + f] = (fq <= mp[m + 1] ? (fq - mp[m]) / (mp[m + 1] - mp[m] + 1e-12f) : (mp[m + 2] - fq) / (mp[m + 2] - mp[m + 1] + 1e-12f)) * enorm;
        }
    }
    // center=True, reflect padding of nfft/2 on both sides (torch.nn.functional.pad reflect:
    // edge sample NOT repeated; indices reflect into the valid region)
    int pad = nfft / 2;
    std::vector<float> pd(2 * pad + ns, 0);
    for (int i = 1; i <= pad; i++) {
        int li = i, ri = i;                       // reflection distance
        while (li >= ns) li = 2 * (ns - 1) - li;  // fold if pad > ns-1 (not expected here)
        while (ri >= ns) ri = 2 * (ns - 1) - ri;
        pd[pad - i] = pcm[li];                    // left: pcm[i] mirrored
        pd[pad + ns - 1 + i] = pcm[ns - 1 - ri];  // right: pcm[ns-1-i] mirrored
    }
    std::memcpy(pd.data() + pad, pcm, ns * sizeof(float));
    int T2 = (int)((pd.size() - nfft) / hop) + 1; if (T2 <= 0) return {};
    std::vector<float> mel(nm * T2, 0);
    // Precompute bit-reversal permutation and twiddles for radix-2 FFT (nfft is a power of 2)
    const int levels = xtts_ctz32((unsigned)nfft);           // log2(nfft)
    std::vector<unsigned> brev(nfft);
    for (unsigned i = 0; i < (unsigned)nfft; i++) { unsigned r = 0; for (int b = 0; b < levels; b++) if (i & (1u << b)) r |= 1u << (levels - 1 - b); brev[i] = r; }
    std::vector<float> cos_t(nfft / 2), sin_t(nfft / 2);
    for (int i = 0; i < nfft / 2; i++) { float a = -2.0f * (float)M_PI * i / nfft; cos_t[i] = cosf(a); sin_t[i] = sinf(a); }
    std::vector<float> wwin(nfft, 0.0f);
    for (int n = wpad; n < wpad + win; n++) wwin[n] = hann[n - wpad];  // centered window
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T2; t++) {
        std::vector<float> re(nf), im(nf), fr(nfft), fi(nfft);
        // windowed frame -> in-place radix-2 FFT
        for (int n = 0; n < nfft; n++) { unsigned b = brev[n]; fr[b] = pd[(size_t)t * hop + n] * wwin[n]; fi[b] = 0; }
        for (int len = 2; len <= nfft; len <<= 1) {
            int half = len >> 1, step = nfft / len;
            for (int i = 0; i < nfft; i += len) {
                for (int j = 0; j < half; j++) {
                    float wr = cos_t[j * step], wi = sin_t[j * step];
                    float ur = fr[i + j], ui = fi[i + j];
                    float vr = fr[i + j + half] * wr - fi[i + j + half] * wi;
                    float vi = fr[i + j + half] * wi + fi[i + j + half] * wr;
                    fr[i + j] = ur + vr; fi[i + j] = ui + vi;
                    fr[i + j + half] = ur - vr; fi[i + j + half] = ui - vi;
                }
            }
        }
        for (int k = 0; k < nf; k++) { re[k] = fr[k]; im[k] = fi[k]; }
            for (int m = 0; m < nm; m++) { float s = 0; for (int f = 0; f < nf; f++) s += fb[m * nf + f] * (re[f] * re[f] + im[f] * im[f]);
            mel[m * T2 + t] = logf(std::max(s, 1e-5f)); } }  // torch.log(torch.clamp(mel, min=1e-5))
    return mel;
}
struct GB {
    ggml_tensor* l1w, *l1b, *qkw, *qkb, *pwo, *pwb, *l2w, *l2b, *fcw, *fcb, *fpw, *fpb;
};
static void read_tensor_weights(ggml_tensor* t, std::vector<float>& out) {
    if (!t) return;
    size_t n = ggml_nelements(t);
    out.resize(n);
    if (t->type == GGML_TYPE_F32)
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    else {
        size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> tmp(nbytes);
        ggml_backend_tensor_get(t, tmp.data(), 0, nbytes);
        const auto* tt = ggml_get_type_traits(t->type);
        if (tt && tt->to_float) tt->to_float(tmp.data(), out.data(), n);
    }
}
struct PercLayer {
    ggml_tensor* to_q = nullptr; std::vector<float> to_q_c;
    ggml_tensor* to_kv = nullptr; std::vector<float> to_kv_c;
    ggml_tensor* to_out = nullptr; std::vector<float> to_out_c;
    ggml_tensor* ffn_w = nullptr; std::vector<float> ffn_w_c;   // layers.N.1.0.weight
    ggml_tensor* ffn_b = nullptr; std::vector<float> ffn_b_c;   // layers.N.1.0.bias
    ggml_tensor* ffn2_w = nullptr; std::vector<float> ffn2_w_c; // layers.N.1.2.weight
    ggml_tensor* ffn2_b = nullptr; std::vector<float> ffn2_b_c; // layers.N.1.2.bias
    ggml_tensor* ln_w = nullptr; std::vector<float> ln_w_c;     // norm.weight (pre-attention LN)
    ggml_tensor* ln_b = nullptr; std::vector<float> ln_b_c;
};
struct GM { ggml_tensor* te, *me, *tp, *mp; std::vector<GB> bl;
    PercLayer perc[2];
    ggml_tensor* perc_latents = nullptr; std::vector<float> perc_latents_c;
    ggml_tensor* perc_norm_gamma = nullptr; std::vector<float> perc_norm_gamma_c;
    ggml_tensor* lfw, *lfb, *fnw, *fnb, *lmw, *lmb;
    ggml_tensor* ciw, *cib; std::vector<float> ciwc;
    std::vector<float> cib_data;
    std::vector<float> cqwc[6], cpwc[6], cqwb_c[6], cpwb_c[6], cgww_c[6], cgwb_c[6];
    const float* cqwb[6] = {}, *cpwb[6] = {}, *cgww[6] = {}, *cgwb[6] = {};
    ggml_tensor* pl, *png;
};
static bool xtts_bind_tensors(GM& m, const std::map<std::string, ggml_tensor*>& ts, const XttSv2HyperParams& hp) {
    m.te = req(ts, "gpt.text_embedding.weight"); m.me = req(ts, "gpt.mel_embedding.weight");
    m.tp = req(ts, "gpt.text_pos.weight"); m.mp = req(ts, "gpt.mel_pos.weight");
    m.bl.resize(hp.n_layers);
    for (uint32_t i = 0; i < hp.n_layers; i++) { auto& b = m.bl[i]; char buf2[128];
        auto f = [&](int idx, const char* s) { snprintf(buf2, sizeof(buf2), "gpt.gpt.h.%d.%s", idx, s); return std::string(buf2); };
        b.l1w = req(ts, f(i,"ln_1.weight").c_str()); b.l1b = T(ts, f(i,"ln_1.bias"));
        b.qkw = req(ts, f(i,"attn.c_attn.weight").c_str()); b.qkb = T(ts, f(i,"attn.c_attn.bias"));
        b.pwo = req(ts, f(i,"attn.c_proj.weight").c_str()); b.pwb = T(ts, f(i,"attn.c_proj.bias"));
        b.l2w = req(ts, f(i,"ln_2.weight").c_str()); b.l2b = T(ts, f(i,"ln_2.bias"));
        b.fcw = req(ts, f(i,"mlp.c_fc.weight").c_str()); b.fcb = T(ts, f(i,"mlp.c_fc.bias"));
        b.fpw = req(ts, f(i,"mlp.c_proj.weight").c_str()); b.fpb = T(ts, f(i,"mlp.c_proj.bias"));
    }
    m.lfw = T(ts, "gpt.gpt.ln_f.weight"); m.lfb = T(ts, "gpt.gpt.ln_f.bias");
    m.fnw = req(ts, "gpt.final_norm.weight"); m.fnb = T(ts, "gpt.final_norm.bias");
    m.lmw = req(ts, "gpt.mel_head.weight"); m.lmb = T(ts, "gpt.mel_head.bias");
    m.ciw = T(ts, "gpt.cond_enc.init.weight"); m.cib = T(ts, "gpt.cond_enc.init.bias");
    m.ciwc = tconv1d(m.ciw);
    std::vector<float> cib_c;
    if (m.cib) read_tensor_weights(m.cib, cib_c);
    m.cib_data = std::move(cib_c);
    for (int i = 0; i < 6; i++) { char buf2[128];
        snprintf(buf2, sizeof(buf2), "gpt.cond_enc.attn.%d.qkv.weight", i); m.cqwc[i] = tconv1d(T(ts, buf2));
        snprintf(buf2, sizeof(buf2), "gpt.cond_enc.attn.%d.qkv.bias", i); if (auto* bt = T(ts, buf2)) { m.cqwb_c[i].clear(); read_tensor_weights(bt, m.cqwb_c[i]); m.cqwb[i] = m.cqwb_c[i].data(); }
        snprintf(buf2, sizeof(buf2), "gpt.cond_enc.attn.%d.proj_out.weight", i); m.cpwc[i] = tconv1d(T(ts, buf2));
        snprintf(buf2, sizeof(buf2), "gpt.cond_enc.attn.%d.proj_out.bias", i); if (auto* bt = T(ts, buf2)) { read_tensor_weights(bt, m.cpwb_c[i]); m.cpwb[i] = m.cpwb_c[i].data(); }
        snprintf(buf2, sizeof(buf2), "gpt.cond_enc.attn.%d.norm.weight", i); if (auto* bt = T(ts, buf2)) { read_tensor_weights(bt, m.cgww_c[i]); m.cgww[i] = m.cgww_c[i].data(); }
        snprintf(buf2, sizeof(buf2), "gpt.cond_enc.attn.%d.norm.bias", i); if (auto* bt = T(ts, buf2)) { read_tensor_weights(bt, m.cgwb_c[i]); m.cgwb[i] = m.cgwb_c[i].data(); } }
    m.pl = T(ts, "gpt.perc.latents"); m.png = T(ts, "gpt.perc.norm.gamma");
    if (m.pl) read_tensor_weights(m.pl, m.perc_latents_c);
    if (m.png) read_tensor_weights(m.png, m.perc_norm_gamma_c);
    for (int i = 0; i < 2; i++) { auto& p = m.perc[i]; char buf2[128];
        auto f = [&](int idx, const char* s) { snprintf(buf2, sizeof(buf2), "gpt.perc.layers.%d.0.%s", idx, s); return std::string(buf2); };
        auto g = [&](int idx, const char* s) { snprintf(buf2, sizeof(buf2), "gpt.perc.layers.%d.1.%s", idx, s); return std::string(buf2); };
        p.to_q = T(ts, f(i, "to_q.weight")); p.to_kv = T(ts, f(i, "to_kv.weight")); p.to_out = T(ts, f(i, "to_out.weight"));
        p.ffn_w = T(ts, g(i, "0.weight")); p.ffn_b = T(ts, g(i, "0.bias"));
        p.ffn2_w = T(ts, g(i, "2.weight")); p.ffn2_b = T(ts, g(i, "2.bias"));
        snprintf(buf2, sizeof(buf2), "gpt.perc.layers.%d.0.norm_weight", i); p.ln_w = T(ts, buf2);
        snprintf(buf2, sizeof(buf2), "gpt.perc.layers.%d.0.norm_bias", i); p.ln_b = T(ts, buf2);
        if (p.to_q) read_tensor_weights(p.to_q, p.to_q_c);
        if (p.to_kv) read_tensor_weights(p.to_kv, p.to_kv_c);
        if (p.to_out) read_tensor_weights(p.to_out, p.to_out_c);
        if (p.ffn_w) read_tensor_weights(p.ffn_w, p.ffn_w_c);
        if (p.ffn_b) read_tensor_weights(p.ffn_b, p.ffn_b_c);
        if (p.ffn2_w) read_tensor_weights(p.ffn2_w, p.ffn2_w_c);
        if (p.ffn2_b) read_tensor_weights(p.ffn2_b, p.ffn2_b_c);
        if (p.ln_w) read_tensor_weights(p.ln_w, p.ln_w_c);
        if (p.ln_b) read_tensor_weights(p.ln_b, p.ln_b_c);
    }
    return m.te && m.me && m.fnw && m.lmw;
}
// Conditioning encoder: conv1d init + 6 x (LN -> multi-head self-attention -> residual).
// QKV weight layout after tconv1d: [3*hp, hp] row-major, rows 0..hp-1=Q, hp..2hp-1=K, 2hp..3hp-1=V.
static std::vector<float> cond_enc_fwd(const GM& m, const float* mel, int T, int hp) {
    if (m.ciwc.empty()) return {};
    // ciwc is [oc=hp][ic=mel_bands] row-major after tconv1d transpose
    int ic = m.ciwc.size() / hp, oc = hp;
    if (ic != 80) { fprintf(stderr, "xtts: cond_enc unexpected mel bands %d\n", ic); return {}; }
    std::vector<float> out(T * oc, 0.f);
    // Init conv1d (kernel size 1): out[t] = W * mel[t] + b
    if (!m.cib_data.empty()) {
        for (int t = 0; t < T; t++) for (int c = 0; c < oc && c < (int)m.cib_data.size(); c++)
            out[t * oc + c] += m.cib_data[c]; }
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) for (int c = 0; c < oc; c++) {
        float s = 0; for (int k = 0; k < ic; k++) s += m.ciwc[c * ic + k] * mel[t * ic + k];
        out[t * oc + c] += s; }
    // Reference: ConditioningEncoder(80, model_dim, num_attn_heads=gpt_n_heads=16)
    // (config.json gpt_n_heads=16; the GGUF key xtts.gpt.cond_enc_heads=4 is a converter error)
    const int n_heads = 16, hd = hp / n_heads;
    for (int l = 0; l < 6; l++) {
        if (m.cqwc[l].empty() || m.cpwc[l].empty()) return {};
        const float* W = m.cqwc[l].data();
        const float* B = m.cqwb[l];
        // AttentionBlock (Tortoise): x_normed = GN32(x); out = x_normed + proj_out(attn(qkv(x_normed)))
        // GroupNorm32(32 groups) over [C, T] - stats across channels-in-group AND time.
        h_gn32(out.data(), m.cgww[l], m.cgwb[l], hp, T, 32);
        // QKV row layout (reference QKVAttention): qkv.reshape(bs*h, 3*ch, T).split(ch, dim=1)
        // => per-head interleaved rows: head h owns rows [h*3*ch, (h+1)*3*ch):
        //    q_h = rows h*3ch .. h*3ch+ch, k_h = +ch .. +2ch, v_h = +2ch .. +3ch
        // With hp=1024 channels, n_heads=16: ch=64, head block stride 192.
        const int chh = hp / n_heads, blk = 3 * chh;  // 64, 192
        // K/V projections for all positions from the normed stream (per-head row ranges)
        std::vector<float> Kall((size_t)T * hp), Vall((size_t)T * hp);
#pragma omp parallel for schedule(static)
        for (int t2 = 0; t2 < T; t2++) {
            const float* xt2 = &out[(size_t)t2 * hp];
            for (int h = 0; h < n_heads; h++) {
                const float* Wk = W + (size_t)(h * blk + chh) * hp;
                const float* Wv = W + (size_t)(h * blk + 2 * chh) * hp;
                const float* Bk = B ? B + h * blk + chh : nullptr;
                const float* Bv = B ? B + h * blk + 2 * chh : nullptr;
                for (int r = 0; r < chh; r++) {
                    float sk = Bk ? Bk[r] : 0.f, sv = Bv ? Bv[r] : 0.f;
                    const float* Wkr = Wk + (size_t)r * hp;
                    const float* Wvr = Wv + (size_t)r * hp;
                    for (int c = 0; c < hp; c++) { float xv = xt2[c]; sk += Wkr[c] * xv; sv += Wvr[c] * xv; }
                    Kall[(size_t)t2 * hp + h * chh + r] = sk;
                    Vall[(size_t)t2 * hp + h * chh + r] = sv;
                }
            }
        }
        std::vector<float> attn_out((size_t)T * hp, 0.f);
#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) {
            const float* xt = &out[t * hp];
            std::vector<float> Q(hp), att(T), out_v(hp);
            for (int h = 0; h < n_heads; h++) {
                const float* Bq = B ? B + h * blk : nullptr;
                for (int r = 0; r < chh; r++) {
                    float s = Bq ? Bq[r] : 0.f;
                    const float* Wr = W + (size_t)(h * blk + r) * hp;
                    for (int c = 0; c < hp; c++) s += Wr[c] * xt[c];
                    Q[h * chh + r] = s;
                }
            }
            // Per-head attention (reference QKVAttention): each head has its own
            // [T, T] score matrix, softmaxed independently over keys.
            // scale = ch**-0.25 applied to BOTH q and k => effective ch**-0.5 on the product.
            const float scale = 1.0f / sqrtf(sqrtf((float)chh));
            for (int h = 0; h < n_heads; h++) {
                const float* qh = &Q[h * hd];
                // scores for head h
                float mx = -1e30f;
                for (int t2 = 0; t2 < T; t2++) {
                    const float* kh = &Kall[(size_t)t2 * hp + h * hd];
                    float sc = 0;
                    for (int j = 0; j < hd; j++) sc += qh[j] * kh[j];
                    att[t2] = sc * scale * scale;   // q*scale, k*scale => product * scale^2
                    if (att[t2] > mx) mx = att[t2];
                }
                float sa = 0;
                for (int t2 = 0; t2 < T; t2++) { att[t2] = expf(att[t2] - mx); sa += att[t2]; }
                for (int t2 = 0; t2 < T; t2++) att[t2] /= sa;
                // weighted sum of this head's values
                float* oh = &out_v[h * hd];
                for (int j = 0; j < hd; j++) oh[j] = 0;
                for (int t2 = 0; t2 < T; t2++) {
                    float a = att[t2];
                    if (a < 1e-10f) continue;
                    const float* v2 = &Vall[(size_t)t2 * hp + h * hd];
                    for (int j = 0; j < hd; j++) oh[j] += a * v2[j];
                }
            }
            // Output projection
            const float* Wp = m.cpwc[l].data();
            const float* Bp = m.cpwb[l];
            for (int r = 0; r < hp; r++) {
                float s = Bp ? Bp[r] : 0.f;
                const float* Wr = Wp + (size_t)r * hp;
                for (int c = 0; c < hp; c++) s += Wr[c] * out_v[c];
                attn_out[(size_t)t * hp + r] = s;
            }
        }
        // Residual: out = normed_out + attn_out (normed stream IS the residual base)
        for (int t = 0; t < T; t++) for (int j = 0; j < hp; j++) out[t * hp + j] += attn_out[t * hp + j];
    }
    return out;
}
// Perceiver resampler - host-side implementation.
// Structure (from Coqui xTTS v2 PerceiverResampler):
//   latents [32, 1024] learned queries
//   x 2 layers of: latents = latents + to_out(attention(to_q(latents), to_kv(context)))
//                  latents = latents + ffn2(gelu(ffn1(norm(latents))))
//   final: RMSNorm with gamma
// The GGUF has no per-layer norm weights, so pre-norms are parameter-free RMSNorm.
// Weight layout after tconv1d flat-read ([out][in] row-major):
//   to_q   [512][1024] : Q = W_q * latent   (inner dim 512 = 8 heads * 64)
//   to_kv  [1024][1024]: [K;V] = W_kv * ctx (K = rows 0..511, V = rows 512..1023)
//   to_out [1024][512] : out = W_o * attn
// GPT-2 uses LayerNorm (mean+var), not RMSNorm. ggml_norm does mean/var normalization;
// Perceiver resampler, corrected to match the reference (perceiver_encoder.py):
//   latents = repeat(learned latents [32, 1024])
//   x = proj_context(context)            (Identity when dim_context == dim)
//   for each layer:
//     latents = attn(latents, x) + latents          (K/V from cat(latents, x)!)
//     latents = ff(latents) + latents               (GEGLU: gelu(gate)*value)
//   out = RMSNorm(latents) * sqrt(dim) * gamma
// Attention: 8 heads, head_dim 64, inner 512. to_q: [512,1024], to_kv: [1024,1024],
// to_out: [1024,512] (all nn.Linear [out][in] row-major in GGUF).
static std::vector<float> perceiver_fwd(const GM& m, const float* cl, int CL, int hp, int chunk_idx = 0) {
    const int N = 32;          // number of latent queries
    const int D = hp;          // 1024 latent/context dim
    const int inner = m.perc[0].to_q ? (int)(m.perc[0].to_q->ne[1]) : 512;
    const int heads = 8, hd = inner / heads;
    if (m.perc_latents_c.empty() || (int)m.perc_latents_c.size() < N * D) return {};
    const int ffn_dim = m.perc[0].ffn_w_c.empty() ? 0 : (int)(m.perc[0].ffn_w_c.size() / D);
    if (ffn_dim == 0) return {};
    const int gelu_dim = ffn_dim / 2;

    std::vector<float> lat((size_t)N * D);
    std::copy(m.perc_latents_c.begin(), m.perc_latents_c.begin() + (size_t)N * D, lat.begin());

    // KV computed over [latents; context] = (N + CL) positions
    const int KT = N + CL;
    std::vector<float> kv_in((size_t)KT * D);
    std::vector<float> q((size_t)N * inner), k((size_t)KT * inner), v((size_t)KT * inner);
    std::vector<float> attn_out((size_t)N * inner), ffn_h(ffn_dim), ffn_out((size_t)N * D);

    for (int l = 0; l < 2; l++) {
        auto& p = m.perc[l];
        if (p.to_q_c.empty() || p.to_kv_c.empty() || p.to_out_c.empty()) return {};
        // KV input: [latents; context]
        for (int t = 0; t < N; t++) std::memcpy(&kv_in[(size_t)t * D], &lat[(size_t)t * D], D * sizeof(float));
        for (int t = 0; t < CL; t++) std::memcpy(&kv_in[(size_t)(N + t) * D], &cl[(size_t)t * D], D * sizeof(float));
        // to_kv: [1024 inner*2, D] row-major; K = rows 0..inner-1, V = rows inner..2*inner-1
#pragma omp parallel for schedule(static)
        for (int t = 0; t < KT; t++) {
            const float* xt = &kv_in[(size_t)t * D];
            for (int r = 0; r < inner; r++) {
                float sk = 0, sv = 0;
                const float* Wk = p.to_kv_c.data() + (size_t)r * D;
                const float* Wv = p.to_kv_c.data() + (size_t)(inner + r) * D;
                for (int c = 0; c < D; c++) { float xv = xt[c]; sk += Wk[c] * xv; sv += Wv[c] * xv; }
                k[(size_t)t * inner + r] = sk;
                v[(size_t)t * inner + r] = sv;
            }
        }
        // Q from raw latents
#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            const float* xn = &lat[(size_t)n * D];
            for (int r = 0; r < inner; r++) {
                float s = 0; const float* W = p.to_q_c.data() + (size_t)r * D;
                for (int c = 0; c < D; c++) s += W[c] * xn[c];
                q[(size_t)n * inner + r] = s;
            }
        }
        // Multi-head attention: N queries attend to KT keys
#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            for (int h = 0; h < heads; h++) {
                const float* qh = &q[(size_t)n * inner + h * hd];
                float mx = -1e30f;
                std::vector<float> att(KT);
                for (int t = 0; t < KT; t++) {
                    const float* kh = &k[(size_t)t * inner + h * hd];
                    float sc = 0; for (int j = 0; j < hd; j++) sc += qh[j] * kh[j];
                    sc /= sqrtf((float)hd);
                    att[t] = sc; if (sc > mx) mx = sc;
                }
                float sa = 0;
                for (int t = 0; t < KT; t++) { att[t] = expf(att[t] - mx); sa += att[t]; }
                for (int t = 0; t < KT; t++) att[t] /= sa;
                float* oh = &attn_out[(size_t)n * inner + h * hd];
                for (int j = 0; j < hd; j++) oh[j] = 0;
                for (int t = 0; t < KT; t++) {
                    float a = att[t]; if (a < 1e-10f) continue;
                    const float* vh = &v[(size_t)t * inner + h * hd];
                    for (int j = 0; j < hd; j++) oh[j] += a * vh[j];
                }
            }
        }
        // to_out projection [D][inner] + residual
#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            for (int r = 0; r < D; r++) {
                float s = 0; const float* W = p.to_out_c.data() + (size_t)r * inner;
                for (int c = 0; c < inner; c++) s += W[c] * attn_out[(size_t)n * inner + c];
                lat[(size_t)n * D + r] += s;
            }
        }
        // FFN: Linear(dim, dim_inner*2) -> GEGLU -> Linear(dim_inner, dim), residual
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            std::vector<float> ffn_h(ffn_dim);
            const float* xn = &lat[(size_t)n * D];
            for (int r = 0; r < ffn_dim; r++) {
                float s = 0; const float* W = p.ffn_w_c.data() + (size_t)r * D;
                for (int c = 0; c < D; c++) s += W[c] * xn[c];
                ffn_h[r] = s + p.ffn_b_c[r];
            }
            // GEGLU: chunk into [value=first half, gate=second half]; out = gelu(gate) * value
            // F.gelu default = exact erf gelu: gelu(x) = 0.5*x*(1+erf(x/sqrt(2)))
            for (int r = 0; r < gelu_dim; r++) {
                float val = ffn_h[r], gate = ffn_h[gelu_dim + r];
                ffn_h[r] = 0.5f * gate * (1.0f + erff(gate * 0.70710678118654752440f)) * val;
            }
            for (int r = 0; r < D; r++) {
                float s = 0; const float* W = p.ffn2_w_c.data() + (size_t)r * gelu_dim;
                for (int c = 0; c < gelu_dim; c++) s += W[c] * ffn_h[c];
                lat[(size_t)n * D + r] += s + p.ffn2_b_c[r];
            }
        }
    }
    // Final RMSNorm: normalize(x) * sqrt(D) * gamma
    {
        for (int n = 0; n < N; n++) {
            float* x = &lat[(size_t)n * D];
            float ss = 0; for (int j = 0; j < D; j++) ss += x[j] * x[j];
            float inv = 1.0f / (sqrtf(ss) + 1e-12f);  // 1/||x||_2, then *sqrt(D)*gamma
            for (int j = 0; j < D; j++) x[j] = x[j] * inv * sqrtf((float)D) * (m.perc_norm_gamma_c.empty() ? 1.f : m.perc_norm_gamma_c[j]);
        }
    }
    lat.resize((size_t)N * D);
    return lat;
}
// scale by weight and add bias for the full affine LayerNorm.
static ggml_tensor* ggml_layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    auto* n = ggml_norm(ctx, x, 1e-5f);
    auto* out = ggml_mul(ctx, n, ggml_repeat(ctx, w, n));
    if (b) out = ggml_add(ctx, out, ggml_repeat(ctx, b, out));
    return out;
}

// ---------------------------------------------------------------------------
// GGML-native GPT-2 autoregressive decode.
//
//  * Weights are used in-place from the backend buffer in whatever type the GGUF
//    stores (F32/F16/Q8_0/...); no dequantized host copies.  ggml's mul_mat
//    kernels do the GEMM/GEMV (vectorized, quantized vec_dot, multithreaded).
//  * The KV cache is a persistent F16 backend tensor per layer/K/V, laid out
//    [HD, n_ctx, H] so flash_attn_ext reads it directly.  New K/V rows are
//    written with ggml_set_rows (position index is a graph *input*, so the graph
//    does not change from step to step).
//  * Attention runs over a window that is padded up to a multiple of KV_BUCKET
//    and masked with an additive F16 mask (-inf outside the causal range).  The
//    graph is only rebuilt when the window crosses a bucket boundary (every 256
//    steps), not per token.
//  * Intermediates are planned by ggml_gallocr (buffers reused across layers).
//  * The lm_head (ln_f -> final_norm -> mel_head) is fused into the graph and,
//    for the prefix, only evaluated on the last position.
// ---------------------------------------------------------------------------
static constexpr int KV_BUCKET = 256;
static int round_up_i(int x, int m) { return (x + m - 1) / m * m; }

// Persistent KV cache. Kept across gpt_gen calls: the buffer is zeroed once at
// allocation, afterwards it only ever holds finite values from earlier chunks,
// and anything outside the causal window is masked to -inf, so no re-zeroing.
struct GptKV {
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor*> k, v;     // per layer, [HD, n_ctx, H] F16
    int n_ctx = 0;
    void release() {
        if (buf) { ggml_backend_buffer_free(buf); buf = nullptr; }
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
        k.clear(); v.clear(); n_ctx = 0;
    }
    ~GptKV() { release(); }
    bool ensure(ggml_backend_t be, const XttSv2HyperParams& hp, int need) {
        if (n_ctx >= need && buf) return true;
        release();
        const int nc = round_up_i(need, KV_BUCKET);
        ggml_init_params ip = { ggml_tensor_overhead() * (size_t)(2 * hp.n_layers + 8), nullptr, true };
        ctx = ggml_init(ip);
        if (!ctx) return false;
        for (uint32_t l = 0; l < hp.n_layers; l++) {
            k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hp.head_dim, nc, hp.n_heads));
            v.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hp.head_dim, nc, hp.n_heads));
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        if (!buf) { release(); return false; }
        ggml_backend_buffer_clear(buf, 0);
        n_ctx = nc;
        return true;
    }
};

// One compiled graph for a fixed (n_tok, n_kv) shape.
struct GptGraph {
    std::vector<uint8_t> meta;
    ggml_context* ctx = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_tensor *emb = nullptr, *idx = nullptr, *mask = nullptr, *logits = nullptr;
    ggml_tensor *hidden = nullptr;   // post ln_f+final_norm last-position state (pre-logits)
    int n_tok = 0, n_kv = 0;
    void release() {
        if (galloc) { ggml_gallocr_free(galloc); galloc = nullptr; }
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
        gf = nullptr; emb = idx = mask = logits = hidden = nullptr; n_tok = n_kv = 0;
    }
    ~GptGraph() { release(); }
};

static bool gpt_build_graph(GptGraph& g, ggml_backend_t be, const GM& m, const XttSv2HyperParams& hp,
                            GptKV& kv, int n_tok, int n_kv) {
    g.release();
    const int D = (int)hp.d, H = (int)hp.n_heads, HD = (int)hp.head_dim, L = (int)hp.n_layers;
    const size_t max_nodes = 4096;
    g.meta.resize(ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false) + (1u << 20));
    ggml_init_params ip = { g.meta.size(), g.meta.data(), true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) return false;
    ggml_context* c = g.ctx;
    g.gf = ggml_new_graph_custom(c, max_nodes, false);
    g.n_tok = n_tok; g.n_kv = n_kv;

    g.emb = ggml_new_tensor_2d(c, GGML_TYPE_F32, D, n_tok);   ggml_set_name(g.emb, "gpt_emb");  ggml_set_input(g.emb);
    g.idx = ggml_new_tensor_1d(c, GGML_TYPE_I64, n_tok);      ggml_set_name(g.idx, "kv_idx");   ggml_set_input(g.idx);
    g.mask = ggml_new_tensor_2d(c, GGML_TYPE_F16, n_kv, n_tok); ggml_set_name(g.mask, "kq_mask"); ggml_set_input(g.mask);

    const size_t fs = sizeof(float);
    ggml_tensor* x = g.emb;
    for (int l = 0; l < L; l++) {
        const GB& b = m.bl[l];
        ggml_tensor* cur = ggml_layer_norm(c, x, b.l1w, b.l1b);
        ggml_tensor* qkv = ggml_mul_mat(c, b.qkw, cur);                    // [3D, T]
        if (b.qkb) qkv = ggml_add(c, qkv, b.qkb);
        // Per-token layout is [Q(D) | K(D) | V(D)], each head-major [h*HD + d].
        ggml_tensor* q  = ggml_view_3d(c, qkv, HD, n_tok, H, qkv->nb[1], (size_t)HD * fs, 0);
        ggml_tensor* ks = ggml_view_3d(c, qkv, HD, n_tok, H, qkv->nb[1], (size_t)HD * fs, (size_t)D * fs);
        ggml_tensor* vs = ggml_view_3d(c, qkv, HD, n_tok, H, qkv->nb[1], (size_t)HD * fs, (size_t)2 * D * fs);
        // Append K/V rows at the positions given by g.idx. These must be expanded
        // into the graph BEFORE the attention node that reads the cache.
        ggml_build_forward_expand(g.gf, ggml_set_rows(c, kv.k[l], ks, g.idx));
        ggml_build_forward_expand(g.gf, ggml_set_rows(c, kv.v[l], vs, g.idx));
        ggml_tensor* kc = ggml_view_3d(c, kv.k[l], HD, n_kv, H, kv.k[l]->nb[1], kv.k[l]->nb[2], 0);
        ggml_tensor* vc = ggml_view_3d(c, kv.v[l], HD, n_kv, H, kv.v[l]->nb[1], kv.v[l]->nb[2], 0);
        ggml_tensor* a = ggml_flash_attn_ext(c, q, kc, vc, g.mask, 1.0f / sqrtf((float)HD), 0.0f, 0.0f);
        a = ggml_reshape_2d(c, a, D, n_tok);                               // [h*HD+d, t] already contiguous
        a = ggml_mul_mat(c, b.pwo, a);
        if (b.pwb) a = ggml_add(c, a, b.pwb);
        x = ggml_add(c, x, a);

        cur = ggml_layer_norm(c, x, b.l2w, b.l2b);
        cur = ggml_mul_mat(c, b.fcw, cur);
        if (b.fcb) cur = ggml_add(c, cur, b.fcb);
        cur = ggml_gelu(c, cur);                                            // gelu_new == tanh approx
        cur = ggml_mul_mat(c, b.fpw, cur);
        if (b.fpb) cur = ggml_add(c, cur, b.fpb);
        x = ggml_add(c, x, cur);
    }
    // Only the last position feeds the sampler. ln_f then final_norm (HF GPT2Model
    // applies ln_f itself, then the XTTS head applies final_norm again), then mel_head.
    ggml_tensor* last = n_tok > 1 ? ggml_view_2d(c, x, D, 1, x->nb[1], (size_t)(n_tok - 1) * x->nb[1]) : x;
    last = ggml_layer_norm(c, last, m.lfw, m.lfb);
    last = ggml_layer_norm(c, last, m.fnw, m.fnb);
    g.hidden = last; ggml_set_name(last, "mel_hidden"); ggml_set_output(last);
    ggml_tensor* lg = ggml_mul_mat(c, m.lmw, last);                         // [vocab, 1]
    if (m.lmb) lg = ggml_add(c, lg, m.lmb);
    g.logits = lg; ggml_set_name(lg, "mel_logits"); ggml_set_output(lg);
    ggml_build_forward_expand(g.gf, lg);

    g.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!g.galloc || !ggml_gallocr_alloc_graph(g.galloc, g.gf)) { g.release(); return false; }
    return true;
}

// Causal mask for query rows at absolute positions pos0..pos0+n_tok-1 over n_kv keys.
static void gpt_fill_mask(GptGraph& g, int pos0) {
    std::vector<ggml_fp16_t> mk((size_t)g.n_kv * g.n_tok);
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int t = 0; t < g.n_tok; t++)
        for (int j = 0; j < g.n_kv; j++)
            mk[(size_t)t * g.n_kv + j] = (j <= pos0 + t) ? zero : ninf;
    ggml_backend_tensor_set(g.mask, mk.data(), 0, mk.size() * sizeof(ggml_fp16_t));
}

// GPT-2 AR generation. Sequence layout:
//   [cond_latents (no pos emb)] -> [text tokens + text_pos] -> [start_mel + mel_pos] -> [mel codes + mel_pos]
// Returns generated mel codes (including the stop token if it was sampled, as before).
static std::vector<int> gpt_gen(ggml_backend_t be, GptKV& kv, const GM& m, const XttSv2HyperParams& hp,
        const std::vector<int>& tids, const std::vector<float>& cl, float temp, int top_k, float top_p, float rep_pen,
        int max_n, int n_threads, std::vector<float>* hiddens_out = nullptr) {
    std::vector<int> codes;
    const int D = (int)hp.d, V = (int)hp.mel_vocab;
    if (ggml_backend_is_cpu(be) && n_threads > 0) ggml_backend_cpu_set_n_threads(be, n_threads);

    // --- Prefix embeddings: [cond_latents | text+pos | start_mel+mel_pos0] ---
    const int ncl = (int)(cl.size() / hp.d);
    const int Tpre = ncl + (int)tids.size() + 1;
    if (!kv.ensure(be, hp, Tpre + max_n + 1)) { fprintf(stderr, "xtts: KV cache allocation failed\n"); return codes; }
    std::vector<float> E((size_t)Tpre * D, 0.f), tmp(D);
    {
        int p = 0;
        for (int c = 0; c < ncl; c++, p++)
            std::memcpy(&E[(size_t)p * D], &cl[(size_t)c * D], D * sizeof(float));
        for (int i = 0; i < (int)tids.size(); i++, p++) {
            get_emb(m.te, tids[i], &E[(size_t)p * D], D);
            if (i < (int)m.tp->ne[1]) {
                get_emb(m.tp, i, tmp.data(), D);
                for (int j = 0; j < D; j++) E[(size_t)p * D + j] += tmp[j];
            }
        }
        get_emb(m.me, (int)hp.start_mel_token, &E[(size_t)p * D], D);
        if ((int)m.mp->ne[1] > 0) {
            get_emb(m.mp, 0, tmp.data(), D);
            for (int j = 0; j < D; j++) E[(size_t)p * D + j] += tmp[j];
        }
    }

    std::vector<float> logits(V, 0.f);
    auto run = [&](GptGraph& g, const float* emb, int pos0, std::vector<float>* hid_out = nullptr) -> bool {
        std::vector<int64_t> idx(g.n_tok);
        for (int i = 0; i < g.n_tok; i++) idx[i] = pos0 + i;
        ggml_backend_tensor_set(g.emb, emb, 0, (size_t)g.n_tok * D * sizeof(float));
        ggml_backend_tensor_set(g.idx, idx.data(), 0, idx.size() * sizeof(int64_t));
        if (ggml_backend_graph_compute(be, g.gf) != GGML_STATUS_SUCCESS) return false;
        ggml_backend_tensor_get(g.logits, logits.data(), 0, (size_t)V * sizeof(float));
        if (hid_out) {
            const size_t prev = hid_out->size();
            hid_out->resize(prev + D);
            ggml_backend_tensor_get(g.hidden, hid_out->data() + prev, 0, (size_t)D * sizeof(float));
        }
        return true;
    };

    // --- Prefill (one pass over the whole prefix) ---
    {
        GptGraph pg;
        if (!gpt_build_graph(pg, be, m, hp, kv, Tpre, round_up_i(Tpre, KV_BUCKET))) { fprintf(stderr, "xtts: prefill graph build failed\n"); return codes; }
        gpt_fill_mask(pg, 0);
        if (!run(pg, E.data(), 0, hiddens_out)) { fprintf(stderr, "xtts: prefill compute failed\n"); return codes; }
    }
    
    // --- Decode ---
    GptGraph dg;                    // (n_tok=1, n_kv=bucket) graph, rebuilt only at bucket boundaries
    int seq_pos = Tpre;             // absolute position of the NEXT token
    int mel_pos = 1;                // start_mel consumed mel position 0 during prefill
    std::vector<float> scaled, me(D), mp(D);
    for (int g = 0; g < max_n; g++) {
        if ((int)codes.size() >= max_n) break;
        // ---- sample ----
        scaled = logits;
        if (temp > 0) for (auto& l2 : scaled) l2 /= temp;
        if (rep_pen != 1.0f && !codes.empty()) {
            // HF-style: penalize ALL distinct previously generated tokens
            std::vector<char> seen(V, 0);
            for (int tok : codes) if (tok >= 0 && tok < (int)scaled.size() && !seen[tok]) {
                seen[tok] = 1;
                scaled[tok] *= scaled[tok] > 0 ? 1.0f / rep_pen : rep_pen;
            }
        }
        float mx = *std::max_element(scaled.begin(), scaled.end());
        if (!std::isfinite(mx)) {
            fprintf(stderr, "xtts_v2: non-finite logits at mel step %d (pos %d), aborting generation\n", g, seq_pos);
            break;
        }
        float sa = 0; for (auto& l2 : scaled) { l2 = expf(l2 - mx); sa += l2; }
        for (auto& l2 : scaled) l2 /= sa;
        if (top_k > 0 && top_k < (int)scaled.size()) {
            std::vector<float> ls(scaled);
            std::partial_sort(ls.begin(), ls.begin() + top_k, ls.end(), std::greater<>());
            float thr = ls[top_k - 1];
            for (auto& l2 : scaled) if (l2 < thr) l2 = 0;
            float sa2 = 0; for (auto& l2 : scaled) sa2 += l2;
            if (sa2 > 0) for (auto& l2 : scaled) l2 /= sa2;
        }
        if (top_p < 1.0f) {
            std::vector<int> idx(scaled.size());
            for (int i = 0; i < (int)idx.size(); i++) idx[i] = i;
            std::sort(idx.begin(), idx.end(), [&](int a, int b) { return scaled[a] > scaled[b]; });
            float cum = 0; std::vector<bool> keep(scaled.size(), false);
            for (int i : idx) { if (scaled[i] > 0) { keep[i] = true; cum += scaled[i]; } if (cum >= top_p) break; }
            for (int i = 0; i < (int)scaled.size(); i++) if (!keep[i]) scaled[i] = 0;
            float sa2 = 0; for (auto& l2 : scaled) sa2 += l2;
            if (sa2 > 0) for (auto& l2 : scaled) l2 /= sa2;
        }
        double r = (double)rand() / RAND_MAX; float cum = 0; int st = 0;
        for (int j = 0; j < (int)scaled.size(); j++) { cum += scaled[j]; if (r <= cum) { st = j; break; } }
        codes.push_back(st);
        if (st == (int)hp.stop_mel_token) break;

        // ---- next token: mel_embedding(st) + mel_pos_embedding(mel_pos) ----
        get_emb(m.me, st, me.data(), D);
        std::fill(mp.begin(), mp.end(), 0.f);
        if (mel_pos < (int)m.mp->ne[1]) get_emb(m.mp, mel_pos, mp.data(), D);
        for (int j = 0; j < D; j++) me[j] += mp[j];
        mel_pos++;

        const int want_kv = std::min(kv.n_ctx, round_up_i(seq_pos + 1, KV_BUCKET));
        if (!dg.gf || dg.n_kv != want_kv) {
            if (!gpt_build_graph(dg, be, m, hp, kv, 1, want_kv)) { fprintf(stderr, "xtts: decode graph build failed\n"); break; }
        }
        // Refill the whole mask every step: gallocr may reassign the mask tensor's
        // buffer region between computes, so incremental single-element updates are
        // not durable. The refill is O(n_kv) F16 bytes (~512 B) - negligible.
        gpt_fill_mask(dg, seq_pos);
        if (!run(dg, me.data(), seq_pos, hiddens_out)) { fprintf(stderr, "xtts: decode compute failed\n"); break; }
        seq_pos++;
    }
    return codes;
}
// GPT-2 latent extraction via GGML graph
static std::vector<float> gpt_extract(ggml_context* ctx, ggml_backend_t be, const GM& m, const XttSv2HyperParams& hp,
        const std::vector<int>& tids, const std::vector<int>& mel_codes, const std::vector<float>& cl, int n_threads = 4) {
    int n_cond = (int)cl.size() / hp.d;
    int T = n_cond + (int)tids.size() + (int)mel_codes.size();
    fprintf(stderr, "xtts_v2: extracting GPT latents (%d cond + %d text + %d mel = %d tokens)\n",
            n_cond, (int)tids.size(), (int)mel_codes.size(), T);
    // Sequence layout: [cond_latents (no pos emb) | text_emb + text_pos | mel_emb + mel_pos]
    std::vector<float> embs(T * hp.d, 0);
    int idx = 0;
    for (int c = 0; c < (int)n_cond; c++, idx++)
        for (int j = 0; j < (int)hp.d; j++) embs[idx * hp.d + j] = cl[c * hp.d + j];
    std::vector<float> pe(hp.d);
    for (int i = 0; i < (int)tids.size(); i++, idx++) {
        get_emb(m.te, tids[i], &embs[idx * hp.d], hp.d);
        if (i < (int)m.tp->ne[1]) {
            get_emb(m.tp, i, pe.data(), hp.d);
            for (int j = 0; j < (int)hp.d; j++) embs[idx * hp.d + j] += pe[j];
        }
    }
    for (int i = 0; i < (int)mel_codes.size(); i++, idx++) {
        get_emb(m.me, mel_codes[i], &embs[idx * hp.d], hp.d);
        if (i < (int)m.mp->ne[1]) {
            get_emb(m.mp, i, pe.data(), hp.d);
            for (int j = 0; j < (int)hp.d; j++) embs[idx * hp.d + j] += pe[j];
        }
    }
    ggml_cgraph* gr = ggml_new_graph_custom(ctx, 8192, false);
    char nm[32];
    auto* emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.d, T);
    ggml_set_name(emb, "gpt_emb"); ggml_set_input(emb);
    { ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(be, ggml_nbytes(emb));
      emb->buffer = buf; emb->data = ggml_backend_buffer_get_base(buf);
      memcpy(emb->data, embs.data(), ggml_nbytes(emb)); }
    auto* x = emb;
    // Shared causal mask for all layers: [T keys, T queries], additive F16.
    // Element (key j, query i) = 0 if j <= i else -INF (decoder-only causal attention).
    ggml_tensor* causal_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, T, T);
    ggml_set_name(causal_mask, "attn_causal_mask");
    ggml_set_input(causal_mask);
    std::vector<ggml_fp16_t> mdata((size_t)T * T);
    for (int i = 0; i < T; i++)
        for (int j = 0; j < T; j++)
            mdata[(size_t)i * T + j] = ggml_fp32_to_fp16(j <= i ? 0.0f : -65504.0f);
    { ggml_backend_buffer_t mbuf = ggml_backend_alloc_buffer(be, ggml_nbytes(causal_mask));
      causal_mask->buffer = mbuf; causal_mask->data = ggml_backend_buffer_get_base(mbuf);
      memcpy(causal_mask->data, mdata.data(), ggml_nbytes(causal_mask));
    }
    for (int l = 0; l < (int)hp.n_layers; l++) { auto& bl = m.bl[l];
        auto* ln = ggml_layer_norm(ctx, x, bl.l1w, bl.l1b);
        snprintf(nm, sizeof(nm), "ln%d", l); ggml_set_name(ln, nm);
        auto* qkw = ggml_view_tensor(ctx, bl.qkw);
        auto* qkv = ggml_mul_mat(ctx, qkw, ln);
        if (bl.qkb) { auto* qkb = ggml_view_tensor(ctx, bl.qkb); qkv = ggml_add(ctx, qkv, ggml_repeat(ctx, qkb, qkv)); }
        qkv = ggml_reshape_3d(ctx, qkv, hp.head_dim, hp.n_heads * 3, T);
        // ggml_flash_attn_ext expects q: [n_embd, n_batch(T), n_head], k/v: [n_embd, n_kv(T), n_head_kv]
        // i.e. TIME in ne[1] and HEADS in ne[2]. The fused qkv output is [3*D, T]; a view with
        // ne={hd, T, nh} and nb1=nb2 of the reshaped [hd, 3*nh, T] parent maps (d, t, h) ->
        // d + h*hd + t*3*D, which is exactly the PyTorch [3, nh, hd] head-major split.
        auto* q = ggml_view_3d(ctx, qkv, hp.head_dim, T, hp.n_heads, qkv->nb[2], qkv->nb[1], 0);
        auto* k = ggml_view_3d(ctx, qkv, hp.head_dim, T, hp.n_heads, qkv->nb[2], qkv->nb[1], (size_t)hp.n_heads * hp.head_dim * ggml_type_size(qkv->type));
        auto* v = ggml_view_3d(ctx, qkv, hp.head_dim, T, hp.n_heads, qkv->nb[2], qkv->nb[1], 2 * (size_t)hp.n_heads * hp.head_dim * ggml_type_size(qkv->type));
        auto* a = ggml_flash_attn_ext(ctx, q, k, v, causal_mask, 1.0f / sqrtf(hp.head_dim), 0, 0);
        snprintf(nm, sizeof(nm), "at%d", l); ggml_set_name(a, nm);
        // flash_attn_ext result is [n_embd_v, n_head, n_batch] ("permuted" layout) whose
        // contiguous flat layout is ALREADY [t][h*hd + d] = the merged [D, T] we need.
        auto* po = ggml_reshape_3d(ctx, a, hp.head_dim * hp.n_heads, T, 1);
        auto* pwo = ggml_view_tensor(ctx, bl.pwo);
        po = ggml_mul_mat(ctx, pwo, po);
        if (bl.pwb) { auto* pwb = ggml_view_tensor(ctx, bl.pwb); po = ggml_add(ctx, po, ggml_repeat(ctx, pwb, po)); }
        snprintf(nm, sizeof(nm), "po%d", l); ggml_set_name(po, nm);
        x = ggml_add(ctx, x, po);
        auto* ln2 = ggml_layer_norm(ctx, x, bl.l2w, bl.l2b);
        snprintf(nm, sizeof(nm), "n2%d", l); ggml_set_name(ln2, nm);
        auto* fcw = ggml_view_tensor(ctx, bl.fcw);
        auto* fc = ggml_mul_mat(ctx, fcw, ln2);
        if (bl.fcb) { auto* fcb = ggml_view_tensor(ctx, bl.fcb); fc = ggml_add(ctx, fc, ggml_repeat(ctx, fcb, fc)); }
        // HF GPT2 activation_function = "gelu_new" = tanh approximation = ggml_gelu (not gelu_erf)
        fc = ggml_gelu(ctx, fc); snprintf(nm, sizeof(nm), "fc%d", l); ggml_set_name(fc, nm);
        auto* fpw = ggml_view_tensor(ctx, bl.fpw);
        auto* fp = ggml_mul_mat(ctx, fpw, fc);
        if (bl.fpb) { auto* fpb = ggml_view_tensor(ctx, bl.fpb); fp = ggml_add(ctx, fp, ggml_repeat(ctx, fpb, fp)); }
        snprintf(nm, sizeof(nm), "fp%d", l); ggml_set_name(fp, nm);
        x = ggml_add(ctx, x, fp); snprintf(nm, sizeof(nm), "xo%d", l); ggml_set_name(x, nm);
    }
    // Double norm: HF GPT2Model applies ln_f internally, then lm_head applies final_norm
    auto* lnf = ggml_layer_norm(ctx, x, m.lfw, m.lfb);
    snprintf(nm, sizeof(nm), "lnf_out"); ggml_set_name(lnf, nm);
    auto* out = ggml_layer_norm(ctx, lnf, m.fnw, m.fnb);
    ggml_set_name(out, "gpt_lat_out"); ggml_build_forward_expand(gr, out);
    // Simple plan/compute pattern (no sched, no gallocr): tensors created with
    // noalloc=false live in the ctx pool; graph compute writes intermediates there.
    // Only the weight tensors live in the backend buffer; graph data flows via ctx pool.
    {
        struct ggml_cplan plan = ggml_graph_plan(gr, n_threads, nullptr);
        std::vector<uint8_t> work(plan.work_size + 64);
        plan.work_data = work.data();
        ggml_graph_compute(gr, &plan);
    }
    std::vector<float> res(T * hp.d);
    memcpy(res.data(), out->data, T * hp.d * sizeof(float));
    return res;
}

}  // anonymous namespace

struct XttSv2GptEngine::Impl {
    WL gpt_wl;
    GM model;
    XttSv2HyperParams hp;
    XttSv2BpeVocab vocab;
    ggml_backend_t backend = nullptr;
    ggml_context* graph_ctx = nullptr;
    GptKV kv;                       // persistent F16 KV cache (reused across chunks)
    int n_threads = 4;
    std::vector<float> mel_norms;   // conditioning mel normalization (80 values)
};

XttSv2GptEngine::~XttSv2GptEngine() {
    if (impl_) { impl_->kv.release();
        if (impl_->graph_ctx) ggml_free(impl_->graph_ctx);
        if (impl_->backend) ggml_backend_free(impl_->backend);
        if (impl_->gpt_wl.ctx) ggml_free(impl_->gpt_wl.ctx);
        if (impl_->gpt_wl.buf) ggml_backend_buffer_free(impl_->gpt_wl.buf);
        delete impl_; impl_ = nullptr; }
}

bool XttSv2GptEngine::load(const std::filesystem::path& gpt_path, const std::filesystem::path& /*voc_path*/, int n_threads, bool verbose) {
    if (impl_) { delete impl_; impl_ = nullptr; }
    impl_ = new Impl(); impl_->n_threads = n_threads;
    impl_->backend = ggml_backend_cpu_init();
    if (!impl_->backend) return false;
    const std::string gpt_path_s = gpt_path.string();  // wchar_t-safe on Windows
    if (!load_gguf(gpt_path_s.c_str(), impl_->backend, impl_->gpt_wl)) {
        ggml_backend_free(impl_->backend); delete impl_; impl_ = nullptr; return false; }
    if (verbose) fprintf(stderr, "xtts_v2: loaded %d tensors from GPT GGUF\n", (int)impl_->gpt_wl.tensors.size());
    gguf_context* gu = gguf_init_from_file(gpt_path_s.c_str(), {true, nullptr});
    if (gu) {
        impl_->hp.n_layers = kv_u32(gu, "xtts.gpt.n_layers", 30);
        impl_->hp.d = kv_u32(gu, "xtts.gpt.d_model", 1024);
        impl_->hp.d_model = impl_->hp.d;
        impl_->hp.n_heads = kv_u32(gu, "xtts.gpt.heads", 16);
        impl_->hp.head_dim = (uint32_t)(impl_->hp.d / impl_->hp.n_heads);
        impl_->hp.ffn = kv_u32(gu, "xtts.gpt.ffn_dim", 4096);
        impl_->hp.ff_dim = impl_->hp.ffn;
        impl_->hp.text_vocab = kv_u32(gu, "xtts.gpt.number_text_tokens", 6681);
        impl_->hp.text_vocab_size = impl_->hp.text_vocab;
        impl_->hp.mel_vocab = kv_u32(gu, "xtts.gpt.number_mel_codes", 1026);
        impl_->hp.mel_vocab_size = impl_->hp.mel_vocab;
        impl_->hp.max_mel_tokens = (int)kv_u32(gu, "xtts.gpt.max_mel_tokens", 605);
        impl_->hp.start_mel_token = (int)kv_u32(gu, "xtts.gpt.start_mel_token", 1024);
        impl_->hp.stop_mel_token = (int)kv_u32(gu, "xtts.gpt.stop_mel_token", 1025);
        impl_->hp.cond_latents = (int)kv_u32(gu, "xtts.gpt.condition_num_latent", 32);
        impl_->hp.code_stride = (int)kv_u32(gu, "xtts.gpt.code_stride_len", 1024);
        gguf_free(gu);
    }
    if (!xtts_bind_tensors(impl_->model, impl_->gpt_wl.tensors, impl_->hp)) return false;
    if (!load_xtts_v2_tokenizer(&impl_->vocab, gpt_path)) {
        impl_->vocab.merges.resize(impl_->hp.text_vocab);
    }
    impl_->graph_ctx = ggml_init({1 << 24, nullptr, false});
    // Conditioning mel stats live next to the GGUF (mel_norms.bin, 80 F32).
    std::filesystem::path mn = gpt_path.parent_path() / "mel_norms.bin";
    if (FILE* f = fopen(mn.string().c_str(), "rb")) {
        impl_->mel_norms.resize(80);
        size_t rd = fread(impl_->mel_norms.data(), 4, 80, f);
        fclose(f);
        if (rd != 80) impl_->mel_norms.clear();
        else fprintf(stderr, "xtts_v2: mel_norms loaded from %s (80 values)\n", mn.string().c_str());
    }
    return true;
}

const XttSv2HyperParams& XttSv2GptEngine::hyper_params() const { return impl_->hp; }
const XttSv2BpeVocab& XttSv2GptEngine::vocab() const { return impl_->vocab; }

bool XttSv2GptEngine::compute_conditioning(const float* ref_pcm, int ref_ns_in, std::vector<float>& out) {
    if (!impl_) return false;
    auto& m = impl_->model;
    auto& hp = impl_->hp;
    // Reference get_gpt_cond_latents(audio, sr, length=gpt_cond_len=30, chunk_length=gpt_cond_chunk_len=6):
    //   audio = audio[:, :22050*length]           (truncate to 30 s)
    //   for each 4-second chunk (skip chunks < 0.33 s):
    //       mel_chunk = wav_to_mel_cloning(chunk, n_fft=2048, hop=256, win=1024)
    //       style_emb = conditioning_encoder + perceiver  -> [32, 1024]
    //   cond_latent = mean(style_embs)
    int ref_ns = ref_ns_in;
    int max_ns = 22050 * 30;   // gpt_cond_len = 30
    if (ref_ns > max_ns) ref_ns = max_ns;
    const int chunk_ns = 22050 * 6;    // gpt_cond_chunk_len = 6 s
    const int min_chunk_ns = (int)(22050 * 0.33f);
    const std::vector<float>& mel_norms = impl_->mel_norms;
    if (mel_norms.size() != 80) fprintf(stderr, "xtts_v2: WARNING mel_norms not found next to the GGUF, conditioning uses 1.0\n");
    // Process 6-second chunks: mel -> cond encoder -> perceiver -> [32,1024]; average.
    // Chunks are independent; compute them in parallel (chunk-level parallelism
    // dominates; the inner OpenMP pragmas serialize inside the outer region).
    struct ChunkRange { int cs, cn; };
    std::vector<ChunkRange> ranges;
    for (int cs = 0; cs < ref_ns; cs += chunk_ns) {
        int cn = std::min(chunk_ns, ref_ns - cs);
        if (cn < min_chunk_ns) continue;    // reference skips chunks < 0.33 s
        ranges.push_back({cs, cn});
    }
    if (ranges.empty()) return false;
    std::vector<std::vector<float>> chunk_latents(ranges.size());
    const int d = hp.d, nlat = hp.cond_latents > 0 ? hp.cond_latents : 32;
    for (int ci = 0; ci < (int)ranges.size(); ci++) {
        const int cs = ranges[ci].cs, cn = ranges[ci].cn;
        std::vector<float> mel = comp_mel(ref_pcm + cs, cn, 22050, 2048, 256, 1024, 80, 0, 8000);
        if (mel.empty()) continue;
        const int T = (int)mel.size() / 80;
        if (mel_norms.size() == 80)
            for (int mch = 0; mch < 80; mch++) for (int t = 0; t < T; t++)
                mel[(size_t)mch * T + t] /= mel_norms[mch];
        // comp_mel returns channel-major [80][T]; cond_enc_fwd expects time-major [T][80]
        std::vector<float> melT((size_t)T * 80);
        for (int mch = 0; mch < 80; mch++) for (int t = 0; t < T; t++)
            melT[(size_t)t * 80 + mch] = mel[(size_t)mch * T + t];
        auto cl = cond_enc_fwd(m, melT.data(), T, hp.d);
        if (cl.empty()) continue;
        // perceiver resampler: cond encoder output [T,1024] -> [32,1024]
        auto pl = perceiver_fwd(m, cl.data(), (int)cl.size() / hp.d, hp.d, ci);
        if (!pl.empty()) chunk_latents[ci] = std::move(pl);
    }
    std::vector<float> acc((size_t)nlat * d, 0.0f);
    int n_chunks = 0;
    for (auto& pl : chunk_latents) {
        if (pl.empty()) continue;
        for (size_t i = 0; i < acc.size(); i++) acc[i] += pl[i];
        n_chunks++;
    }
    if (n_chunks == 0) return false;
    for (auto& v : acc) v /= (float)n_chunks;
    out = std::move(acc);
    return true;
}
std::vector<int32_t> XttSv2GptEngine::generate_mel_codes(const std::vector<int32_t>& text_tokens, const std::vector<float>& cond_latents,
        float temperature, int top_k, float top_p, float rep_penalty, int max_tokens,
        std::vector<float>* decode_hiddens) {
    if (!impl_) return {};
    auto codes = gpt_gen(impl_->backend, impl_->kv, impl_->model, impl_->hp,
        std::vector<int>(text_tokens.begin(), text_tokens.end()), cond_latents,
        temperature, top_k, top_p, rep_penalty, max_tokens, impl_->n_threads, decode_hiddens);
    return std::vector<int32_t>(codes.begin(), codes.end());
}
std::vector<float> XttSv2GptEngine::extract_latents(const std::vector<int32_t>& text_tokens, const std::vector<int32_t>& mel_codes,
        const std::vector<float>& cond_latents) {
    if (!impl_) return {};
    ggml_context* ctx = ggml_init({(size_t)1 << 33, nullptr, false}); // 8GB for large graph
    auto latents = gpt_extract(ctx, impl_->backend, impl_->model, impl_->hp,
        std::vector<int>(text_tokens.begin(), text_tokens.end()),
        std::vector<int>(mel_codes.begin(), mel_codes.end()), cond_latents, impl_->n_threads);
    ggml_free(ctx); return latents;
}

}  // namespace engine::models::xtts_v2
