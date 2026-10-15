// xtts_v2 GPT engine -- GPT-2 AR + ConditioningEncoder + Perceiver resampler
#include "engine/models/xtts_v2/gpt.h"
#include "engine/models/xtts_v2/assets.h"
#include "engine/models/xtts_v2/tokenizer_text.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <cmath>
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
    if (t < 0 || t >= vocab) { std::fill(o, o + d, 0); return; }
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
    size_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2];
    size_t total = ne0 * ne1 * (ne2 ? ne2 : 1);
    std::vector<float> r(total);
    ggml_backend_tensor_get(t, r.data(), 0, total * sizeof(float));
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
    if (getenv("XTTS_DUMP_FB")) {
        FILE* ff = fopen("/tmp/fb_cpp.txt", "w");
        for (int k = 0; k < 6; k++) {
            fprintf(ff, "bin%d f=%.2f:", k, (float)sr * k / nfft);
            for (int m = 0; m < 4; m++) fprintf(ff, " %.5f", fb[m * nf + k]);
            fprintf(ff, "\n");
        }
        float s0 = 0; for (int k = 0; k < nf; k++) s0 += fb[0 * nf + k];
        fprintf(ff, "mel0 sum: %f\n", s0);
        fclose(ff);
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
    const int levels = __builtin_ctz((unsigned)nfft);           // log2(nfft)
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
        if (getenv("XTTS_DUMP_PWR") && t == 65) {
            FILE* ff = fopen("/tmp/pwr_cpp.bin", "wb");
            if (ff) { for (int f = 0; f < nf; f++) { float p = re[f]*re[f] + im[f]*im[f]; fwrite(&p, 4, 1, ff); } fclose(ff); }
            FILE* fg = fopen("/tmp/seg_cpp.bin", "wb");
            if (fg) { fwrite(pd.data() + (size_t)t * hop, 4, nfft, fg); fclose(fg); }
        }
        for (int m = 0; m < nm; m++) { float s = 0; for (int f = 0; f < nf; f++) s += fb[m * nf + f] * (re[f] * re[f] + im[f] * im[f]);
            mel[m * T2 + t] = logf(std::max(s, 1e-5f)); } }  // torch.log(torch.clamp(mel, min=1e-5))
    return mel;
}
struct GB {
    ggml_tensor* l1w, *l1b, *qkw, *qkb, *pwo, *pwb, *l2w, *l2b, *fcw, *fcb, *fpw, *fpb;
    std::vector<float> l1w_c, l1b_c, qkw_c, qkb_c, pwo_c, pwb_c, l2w_c, l2b_c, fcw_c, fcb_c, fpw_c, fpb_c;
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
    std::vector<float> lfw_c, lfb_c, lmw_c, lmb_c, fnw_c, fnb_c;
    ggml_tensor* ciw, *cib; std::vector<float> ciwc;
    std::vector<float> cib_data;
    std::vector<float> cqwc[6], cpwc[6], cqwb_c[6], cpwb_c[6], cgww_c[6], cgwb_c[6];
    const float* cqwb[6] = {}, *cpwb[6] = {}, *cgww[6] = {}, *cgwb[6] = {};
    ggml_tensor* pl, *png;
};
static bool xtts_bind_tensors(GM& m, const std::map<std::string, ggml_tensor*>& ts, const XttSv2HyperParams& hp) {
    m.te = req(ts, "gpt.text_embedding.weight"); m.me = req(ts, "gpt.mel_embedding.weight");
    m.tp = req(ts, "gpt.text_pos.weight"); m.mp = req(ts, "gpt.mel_pos.weight");
    if (getenv("XTTS_DEBUG")) {
        auto dmp = [](ggml_tensor* t, const char* nm2) {
            if (!t) { fprintf(stderr, "xtts_dbg: %s MISSING\n", nm2); return; }
            fprintf(stderr, "xtts_dbg: %s ne=[%lld,%lld] type=%s\n", nm2, (long long)t->ne[0], (long long)t->ne[1], ggml_type_name(t->type));
        };
        dmp(m.te, "text_embedding"); dmp(m.me, "mel_embedding"); dmp(m.tp, "text_pos"); dmp(m.mp, "mel_pos");
        dmp(m.bl.empty() ? nullptr : m.bl[0].qkw, "h0.c_attn.w");
        dmp(m.bl.empty() ? nullptr : m.bl[0].fcw, "h0.c_fc.w");
        dmp(m.bl.empty() ? nullptr : m.bl[0].fpw, "h0.c_proj.w");
    }
    m.bl.resize(hp.n_layers);
    for (uint32_t i = 0; i < hp.n_layers; i++) { auto& b = m.bl[i]; char buf2[128];
        auto f = [&](int idx, const char* s) { snprintf(buf2, sizeof(buf2), "gpt.gpt.h.%d.%s", idx, s); return std::string(buf2); };
        b.l1w = req(ts, f(i,"ln_1.weight").c_str()); b.l1b = T(ts, f(i,"ln_1.bias"));
        b.qkw = req(ts, f(i,"attn.c_attn.weight").c_str()); b.qkb = T(ts, f(i,"attn.c_attn.bias"));
        b.pwo = req(ts, f(i,"attn.c_proj.weight").c_str()); b.pwb = T(ts, f(i,"attn.c_proj.bias"));
        b.l2w = req(ts, f(i,"ln_2.weight").c_str()); b.l2b = T(ts, f(i,"ln_2.bias"));
        b.fcw = req(ts, f(i,"mlp.c_fc.weight").c_str()); b.fcb = T(ts, f(i,"mlp.c_fc.bias"));
        b.fpw = req(ts, f(i,"mlp.c_proj.weight").c_str()); b.fpb = T(ts, f(i,"mlp.c_proj.bias"));
        read_tensor_weights(b.l1w, b.l1w_c); read_tensor_weights(b.l1b, b.l1b_c);
        read_tensor_weights(b.qkw, b.qkw_c); read_tensor_weights(b.qkb, b.qkb_c);
        read_tensor_weights(b.pwo, b.pwo_c); read_tensor_weights(b.pwb, b.pwb_c);
        read_tensor_weights(b.l2w, b.l2w_c); read_tensor_weights(b.l2b, b.l2b_c);
        read_tensor_weights(b.fcw, b.fcw_c); read_tensor_weights(b.fcb, b.fcb_c);
        read_tensor_weights(b.fpw, b.fpw_c); read_tensor_weights(b.fpb, b.fpb_c); }
    m.lfw = T(ts, "gpt.gpt.ln_f.weight"); m.lfb = T(ts, "gpt.gpt.ln_f.bias");
    m.fnw = req(ts, "gpt.final_norm.weight"); m.fnb = T(ts, "gpt.final_norm.bias");
    m.lmw = req(ts, "gpt.mel_head.weight"); m.lmb = T(ts, "gpt.mel_head.bias");
    if (getenv("XTTS_DEBUG") && !m.bl.empty() && m.bl[0].qkw && m.bl[0].fcw && m.bl[0].fpw && m.lmw) {
        fprintf(stderr, "xtts_dbg: h0.c_attn.w ne=[%lld,%lld] type=%s | h0.c_fc.w ne=[%lld,%lld] | h0.c_proj.w ne=[%lld,%lld] | mel_head ne=[%lld,%lld]\n",
                (long long)m.bl[0].qkw->ne[0], (long long)m.bl[0].qkw->ne[1], ggml_type_name(m.bl[0].qkw->type),
                (long long)m.bl[0].fcw->ne[0], (long long)m.bl[0].fcw->ne[1],
                (long long)m.bl[0].fpw->ne[0], (long long)m.bl[0].fpw->ne[1],
                (long long)m.lmw->ne[0], (long long)m.lmw->ne[1]);
    }
    read_tensor_weights(m.lfw, m.lfw_c); read_tensor_weights(m.lfb, m.lfb_c);
    read_tensor_weights(m.lmw, m.lmw_c); read_tensor_weights(m.lmb, m.lmb_c);
    if (getenv("XTTS_DEBUG")) {
        fprintf(stderr, "xtts_dbg: lmw_c[0..7]:");
        for (int i = 0; i < 8; i++) fprintf(stderr, " %.4f", m.lmw_c[i]);
        fprintf(stderr, " | lmw_c[1024..1031]:");
        for (int i = 0; i < 8; i++) fprintf(stderr, " %.4f", m.lmw_c[1024 + i]);
        fprintf(stderr, " | lmb[0..3]:");
        for (int i = 0; i < 4; i++) fprintf(stderr, " %.4f", m.lmb_c[i]);
        fprintf(stderr, " | lmw ne=[%lld,%lld]\n", (long long)m.lmw->ne[0], (long long)m.lmw->ne[1]);
    }
    if (m.fnw) read_tensor_weights(m.fnw, m.fnw_c);
    if (m.fnb) read_tensor_weights(m.fnb, m.fnb_c);
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
        if (getenv("XTTS_DUMP_PERC") && l == 0) {
            char pq[512], pk[512], pv[512];
            snprintf(pq, sizeof(pq), "%s.c%d.L0.q", getenv("XTTS_DUMP_PERC"), chunk_idx);
            snprintf(pk, sizeof(pk), "%s.c%d.L0.k", getenv("XTTS_DUMP_PERC"), chunk_idx);
            snprintf(pv, sizeof(pv), "%s.c%d.L0.v", getenv("XTTS_DUMP_PERC"), chunk_idx);
            if (FILE* f = fopen(pq, "wb")) { fwrite(q.data(), 4, q.size(), f); fclose(f); }
            if (FILE* f = fopen(pk, "wb")) { fwrite(k.data(), 4, k.size(), f); fclose(f); }
            if (FILE* f = fopen(pv, "wb")) { fwrite(v.data(), 4, v.size(), f); fclose(f); }
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
        if (getenv("XTTS_DUMP_PERC") && l == 0) {
            char pa[512];
            snprintf(pa, sizeof(pa), "%s.c%d.L0.attn_out", getenv("XTTS_DUMP_PERC"), chunk_idx);
            FILE* fa = fopen(pa, "wb"); if (fa) { fwrite(attn_out.data(), 4, attn_out.size(), fa); fclose(fa); }
            char pl[512];
            snprintf(pl, sizeof(pl), "%s.c%d.L0.lat_after_attn", getenv("XTTS_DUMP_PERC"), chunk_idx);
            FILE* fl = fopen(pl, "wb"); if (fl) { fwrite(lat.data(), 4, lat.size(), fl); fclose(fl); }
        }
        // FFN: Linear(dim, dim_inner*2) -> GEGLU -> Linear(dim_inner, dim), residual
        std::vector<float> ffn_h1_dump;
        if (getenv("XTTS_DUMP_PERC") && l == 0) ffn_h1_dump.resize((size_t)N * ffn_dim);
#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; n++) {
            std::vector<float> ffn_h(ffn_dim);
            const float* xn = &lat[(size_t)n * D];
            for (int r = 0; r < ffn_dim; r++) {
                float s = 0; const float* W = p.ffn_w_c.data() + (size_t)r * D;
                for (int c = 0; c < D; c++) s += W[c] * xn[c];
                ffn_h[r] = s + p.ffn_b_c[r];
            }
            if (getenv("XTTS_DUMP_PERC") && l == 0)
                std::copy(ffn_h.begin(), ffn_h.end(), ffn_h1_dump.begin() + (size_t)n * ffn_dim);
            // GEGLU: chunk into [value=first half, gate=second half]; out = gelu(gate) * value
            // F.gelu default = exact erf gelu: gelu(x) = 0.5*x*(1+erf(x/sqrt(2)))
            for (int r = 0; r < gelu_dim; r++) {
                float val = ffn_h[r], gate = ffn_h[gelu_dim + r];
                ffn_h[r] = 0.5f * gate * (1.0f + erff(gate * 0.70710678118654752440f)) * val;
            }
            if (getenv("XTTS_DUMP_PERC") && l == 0 && n == 0) {
                char pg[512];
                snprintf(pg, sizeof(pg), "%s.c%d.L0.ffn_geglu", getenv("XTTS_DUMP_PERC"), chunk_idx);
                FILE* fg = fopen(pg, "wb"); if (fg) { fwrite(ffn_h.data(), 4, gelu_dim, fg); fclose(fg); }
                fprintf(stderr, "xtts_dbg: perc L0 n0 GEGLU: val[0..2]=%.3f,%.3f,%.3f gate[0..2]=%.3f,%.3f,%.3f gelu_out[0..2]=%.3f,%.3f,%.3f gelu_dim=%d ffn_dim=%d\n",
                        ffn_h[0], ffn_h[1], ffn_h[2], ffn_h[gelu_dim], ffn_h[gelu_dim+1], ffn_h[gelu_dim+2],
                        0.5f*ffn_h[0]*(1.0f+erff(ffn_h[gelu_dim]*0.70710678f)), 0.5f*ffn_h[1]*(1.0f+erff(ffn_h[gelu_dim+1]*0.70710678f)), 0.5f*ffn_h[2]*(1.0f+erff(ffn_h[gelu_dim+2]*0.70710678f)), gelu_dim, ffn_dim);
            }
            for (int r = 0; r < D; r++) {
                float s = 0; const float* W = p.ffn2_w_c.data() + (size_t)r * gelu_dim;
                for (int c = 0; c < gelu_dim; c++) s += W[c] * ffn_h[c];
                lat[(size_t)n * D + r] += s + p.ffn2_b_c[r];
            }
        }
        if (getenv("XTTS_DUMP_PERC") && l == 0) {
            char pf[512], ph[512];
            snprintf(pf, sizeof(pf), "%s.c%d.L0.lat_after_ffn", getenv("XTTS_DUMP_PERC"), chunk_idx);
            snprintf(ph, sizeof(ph), "%s.c%d.L0.ffn_h1", getenv("XTTS_DUMP_PERC"), chunk_idx);
            FILE* ff2 = fopen(pf, "wb"); if (ff2) { fwrite(lat.data(), 4, lat.size(), ff2); fclose(ff2); }
            FILE* fh = fopen(ph, "wb"); if (fh) { fwrite(ffn_h1_dump.data(), 4, ffn_h1_dump.size(), fh); fclose(fh); }
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
    if (getenv("XTTS_DUMP_PERC")) {
        char pf[512];
        snprintf(pf, sizeof(pf), "%s.c%d.final", getenv("XTTS_DUMP_PERC"), chunk_idx);
        FILE* ff = fopen(pf, "wb"); if (ff) { fwrite(lat.data(), 4, lat.size(), ff); fclose(ff); }
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

// GPT-2 AR generation with proper KV cache and separate position embeddings.
// Sequence layout: [cond_latents (no pos emb)] -> [text tokens + text_pos] -> [start_mel + mel_pos] -> [mel codes + mel_pos]
// Returns the generated mel codes (excluding start token, excluding stop token).
static std::vector<int> gpt_gen(ggml_context* ctx, ggml_backend_t be, const GM& m, const XttSv2HyperParams& hp,
        const std::vector<int>& tids, const std::vector<float>& cl, float temp, int top_k, float top_p, float rep_pen, int max_n) {
    (void)ctx; (void)be;  // host-side implementation; kept for signature compatibility
    std::vector<int> codes;
    const int H = hp.n_heads, HD = hp.head_dim, D = hp.d;
    const float scale = 1.0f / sqrtf((float)HD);
    const int max_pos = 4096;
    // KV cache: [layer][pos][head][dim] for K and V separately
    std::vector<std::vector<float>> k_cache(hp.n_layers), v_cache(hp.n_layers);
    for (int l = 0; l < (int)hp.n_layers; l++) {
        k_cache[l].assign((size_t)max_pos * H * HD, 0.f);
        v_cache[l].assign((size_t)max_pos * H * HD, 0.f);
    }
    std::vector<float> emb(D), qkv(D * 3), xln(D), attn_out(D), h2(hp.ffn), ff_out(D);
    int seq_pos = 0;  // absolute position in the sequence (cache index)


    // Batched forward of T token embeddings at positions seq_pos..seq_pos+T-1 (GEMM form).
    // Fills K/V caches; writes post-final-LN hidden states to out [T x D]. Much faster than
    // T sequential forward_token calls (one weight pass for the whole prefix).
    auto forward_batch = [&](const std::vector<float>& E, int T, std::vector<float>& out) {
        const size_t L = hp.n_layers;
        out.assign((size_t)T * D, 0.f);
        std::vector<float> R = E;                       // residual stream [T x D]
        std::vector<float> Xn((size_t)T * D);           // normed stream
        std::vector<float> QKV((size_t)T * 3 * D);
        std::vector<float> AT((size_t)T * D);           // attention merged output
        std::vector<float> FF((size_t)T * D);
        std::vector<float> H2((size_t)T * hp.ffn);
        for (int l = 0; l < (int)L; l++) {
            auto& bl = m.bl[l];
            // LayerNorm (row-wise)
#pragma omp parallel for schedule(static)
            for (int t = 0; t < T; t++)
                h_ln(&Xn[(size_t)t * D], &R[(size_t)t * D], bl.l1w_c.data(), bl.l1b_c.data(), D);
            // QKV GEMM: QKV[t] = Xn[t] @ W^T + b  (W [3D x D] row-major), flat parallel
#pragma omp parallel for schedule(static)
            for (int tm = 0; tm < T * 3 * D; tm++) {
                int t = tm / (3 * D), m = tm % (3 * D);
                float s = bl.qkb_c.empty() ? 0.f : bl.qkb_c[m];
                const float* Wm = bl.qkw_c.data() + (size_t)m * D;
                const float* x = &Xn[(size_t)t * D];
                for (int n = 0; n < D; n++) s += Wm[n] * x[n];
                QKV[(size_t)tm] = s;
            }
            // Store K,V into cache rows seq_pos..seq_pos+T-1
            for (int t = 0; t < T; t++) {
                std::memcpy(k_cache[l].data() + (size_t)(seq_pos + t) * H * HD, &QKV[(size_t)t * 3 * D] + D, D * sizeof(float));
                std::memcpy(v_cache[l].data() + (size_t)(seq_pos + t) * H * HD, &QKV[(size_t)t * 3 * D] + 2 * D, D * sizeof(float));
            }
            // Causal multi-head attention (queries = rows, keys/vals = cache rows 0..seq_pos+T-1)
#pragma omp parallel for schedule(static)
            for (int t = 0; t < T; t++) {
                const int Tc = seq_pos + t + 1;   // causal: query t sees positions 0..seq_pos+t
                for (int h = 0; h < H; h++) {
                    const float* qh = &QKV[(size_t)t * 3 * D] + h * HD;
                    std::vector<float> sc(Tc);
                    float mx = -1e30f;
                    for (int s = 0; s < Tc; s++) {
                        const float* kh = k_cache[l].data() + (size_t)s * H * HD + h * HD;
                        float v = 0; for (int j = 0; j < HD; j++) v += qh[j] * kh[j];
                        sc[s] = v * scale; if (sc[s] > mx) mx = sc[s];
                    }
                    float sa = 0;
                    for (int s = 0; s < Tc; s++) { sc[s] = expf(sc[s] - mx); sa += sc[s]; }
                    for (int s = 0; s < Tc; s++) sc[s] /= sa;
                    float* oh = &AT[(size_t)t * D] + h * HD;
                    for (int j = 0; j < HD; j++) oh[j] = 0;
                    for (int s = 0; s < Tc; s++) {
                        float a = sc[s]; if (a < 1e-10f) continue;
                        const float* vh = v_cache[l].data() + (size_t)s * H * HD + h * HD;
                        for (int j = 0; j < HD; j++) oh[j] += a * vh[j];
                    }
                }
            }
            // Output projection + residual (flat GEMM over T*D outputs)
#pragma omp parallel for schedule(static)
            for (int tm = 0; tm < T * D; tm++) {
                int t = tm / D, m = tm % D;
                float s = bl.pwb_c.empty() ? 0.f : bl.pwb_c[m];
                const float* Wm = bl.pwo_c.data() + (size_t)m * D;
                const float* x = &AT[(size_t)t * D];
                for (int n = 0; n < D; n++) s += Wm[n] * x[n];
                R[(size_t)tm] += s;
            }
            // FFN pre-norm (row-wise)
#pragma omp parallel for schedule(static)
            for (int t = 0; t < T; t++)
                h_ln(&Xn[(size_t)t * D], &R[(size_t)t * D], bl.l2w_c.data(), bl.l2b_c.data(), D);
            // FFN up-proj + gelu (flat GEMM over T*ffn outputs)
#pragma omp parallel for schedule(static)
            for (int tm = 0; tm < T * (int)hp.ffn; tm++) {
                int t = tm / (int)hp.ffn, m = tm % (int)hp.ffn;
                float s = bl.fcb_c.empty() ? 0.f : bl.fcb_c[m];
                const float* Wm = bl.fcw_c.data() + (size_t)m * D;
                const float* x = &Xn[(size_t)t * D];
                for (int n = 0; n < D; n++) s += Wm[n] * x[n];
                H2[(size_t)tm] = gelu_t(s);
            }
            // FFN down-proj + residual (flat GEMM over T*D outputs)
#pragma omp parallel for schedule(static)
            for (int tm = 0; tm < T * D; tm++) {
                int t = tm / D, m = tm % D;
                float s = bl.fpb_c.empty() ? 0.f : bl.fpb_c[m];
                const float* Wm = bl.fpw_c.data() + (size_t)m * (int)hp.ffn;
                const float* x = &H2[(size_t)t * hp.ffn];
                for (int n = 0; n < (int)hp.ffn; n++) s += Wm[n] * x[n];
                R[(size_t)tm] += s;
            }
        }
        // Final LN (double: ln_f + final_norm via lfw then fnw)
#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) {
            std::vector<float> tmp(D);
            h_ln(&tmp[0], &R[(size_t)t * D], m.lfw_c.data(), m.lfb_c.data(), D);
            h_ln(&out[(size_t)t * D], &tmp[0], m.fnw_c.data(), m.fnb_c.data(), D);
        }
        seq_pos += T;
    };

    // Forward one token embedding at position seq_pos; appends K,V to cache.
    // Returns the pre-logits hidden state (post final LN) in emb_out.
    static double t_qkv = 0, t_attn = 0, t_proj = 0, t_ffn = 0, t_logits = 0;
    auto forward_token = [&](const std::vector<float>& input_emb, std::vector<float>& emb_out) {
        if (seq_pos >= max_pos) { fprintf(stderr, "xtts: seq overflow %d\n", seq_pos); return; }
        std::vector<float> residual(D);
        std::copy(input_emb.begin(), input_emb.end(), residual.begin());
        for (int l = 0; l < (int)hp.n_layers; l++) {
            auto& bl = m.bl[l];
            // Pre-norm (LayerNorm with bias)
            h_ln(xln.data(), residual.data(), bl.l1w_c.data(), bl.l1b_c.data(), D);
            // QKV projection: [3D, D] row-major -> qkv[0..D-1]=Q, [D..2D-1]=K, [2D..3D-1]=V
            auto ta0 = omp_get_wtime();
            h_c1d(qkv.data(), bl.qkw_c.data(), xln.data(), bl.qkb_c.data(), D, D * 3);
            t_qkv += omp_get_wtime() - ta0;
            // Store K,V for this position in cache
            float* kdst = k_cache[l].data() + (size_t)seq_pos * H * HD;
            float* vdst = v_cache[l].data() + (size_t)seq_pos * H * HD;
            memcpy(kdst, qkv.data() + D, D * sizeof(float));
            memcpy(vdst, qkv.data() + 2 * D, D * sizeof(float));
            // Multi-head causal attention over positions 0..seq_pos
            auto ta1 = omp_get_wtime();
            std::vector<float> attn(H * HD, 0.f);
            int Tc = seq_pos + 1;
            std::vector<float> scores(Tc);
            for (int h = 0; h < H; h++) {
                const float* qh = qkv.data() + h * HD;
                float mx = -1e30f;
                for (int t = 0; t < Tc; t++) {
                    const float* kh = k_cache[l].data() + (size_t)t * H * HD + h * HD;
                    float sc = 0;
                    for (int j = 0; j < HD; j++) sc += qh[j] * kh[j];
                    sc *= scale;
                    scores[t] = sc;
                    if (sc > mx) mx = sc;
                }
                float sa = 0;
                for (int t = 0; t < Tc; t++) { scores[t] = expf(scores[t] - mx); sa += scores[t]; }
                for (int t = 0; t < Tc; t++) scores[t] /= sa;
                float* oh = &attn[h * HD];
                for (int j = 0; j < HD; j++) oh[j] = 0;
                for (int t = 0; t < Tc; t++) {
                    float a = scores[t];
                    if (a < 1e-10f) continue;
                    const float* vh = v_cache[l].data() + (size_t)t * H * HD + h * HD;
                    for (int j = 0; j < HD; j++) oh[j] += a * vh[j];
                }
            }
            t_attn += omp_get_wtime() - ta1;
            // Output projection + residual
            auto ta2 = omp_get_wtime();
            h_c1d(ff_out.data(), bl.pwo_c.data(), attn.data(), bl.pwb_c.data(), D, D);
            t_proj += omp_get_wtime() - ta2;
            for (int j = 0; j < D; j++) residual[j] += ff_out[j];
            // FFN with pre-norm
            h_ln(xln.data(), residual.data(), bl.l2w_c.data(), bl.l2b_c.data(), D);
            auto tf0 = omp_get_wtime();
            h_c1d(h2.data(), bl.fcw_c.data(), xln.data(), bl.fcb_c.data(), D, hp.ffn);
            for (int j = 0; j < (int)hp.ffn; j++) h2[j] = gelu_t(h2[j]);
            h_c1d(ff_out.data(), bl.fpw_c.data(), h2.data(), bl.fpb_c.data(), hp.ffn, D);
            t_ffn += omp_get_wtime() - tf0;
            for (int j = 0; j < D; j++) residual[j] += ff_out[j];
        }
        // Final LN (HF GPT2Model applies ln_f; then lm_head applies final_norm AGAIN - double norm!)
        h_ln(emb_out.data(), residual.data(), m.lfw_c.data(), m.lfb_c.data(), D);
        // Debug: capture post-ln_f / pre-final_norm hidden at this position
        if (getenv("XTTS_DUMP_HIDDEN")) {
            static FILE* hf = nullptr;
            if (!hf) { hf = fopen(getenv("XTTS_DUMP_HIDDEN"), "wb"); }
            fwrite(emb_out.data(), 4, D, hf);
            fflush(hf);
        }
        std::vector<float> tmp(D);
        std::memcpy(tmp.data(), emb_out.data(), D * sizeof(float));
        h_ln(emb_out.data(), tmp.data(), m.fnw_c.data(), m.fnb_c.data(), D);
        if (getenv("XTTS_DUMP_NORMED")) {
            static FILE* nf = nullptr;
            if (!nf) { nf = fopen(getenv("XTTS_DUMP_NORMED"), "wb"); }
            fwrite(emb_out.data(), 4, D, nf);
            fflush(nf);
        }
        seq_pos++;
    };

    // --- Prefix: [cond_latents (no pos emb) | text tokens + text_pos] ---
    if (getenv("XTTS_DUMP_COND") && !cl.empty()) {
        FILE* f = fopen(getenv("XTTS_DUMP_COND"), "wb");
        if (f) { fwrite(cl.data(), 4, cl.size(), f); fclose(f); }
    }
    // --- AR mel code generation ---
    // Start token 1024, stop token 1025. Mel positions start at 0.
    std::vector<float> emb_last(D, 0), logits(hp.mel_vocab, 0);
    int mel_pos = 1;   // start_mel consumed pos 0 during prefix
    // --- Build prefix embedding matrix: [cond_latents | text+pos | start_mel+melpos0] ---
    int Tpre = (int)(cl.size() / hp.d) + (int)tids.size() + 1;
    {
        std::vector<float> E((size_t)Tpre * D, 0.f);
        int p = 0;
        int ncl = (int)cl.size() / hp.d;
        for (int c = 0; c < ncl; c++, p++)
            std::memcpy(&E[(size_t)p * D], &cl[(size_t)c * D], D * sizeof(float));
        for (int i = 0; i < (int)tids.size(); i++, p++) {
            get_emb(m.te, tids[i], &E[(size_t)p * D], D);
            if (i < (int)m.tp->ne[1]) {
                std::vector<float> tp(D, 0);
                get_emb(m.tp, i, tp.data(), D);
                for (int j = 0; j < D; j++) E[(size_t)p * D + j] += tp[j];
            }
        }
        get_emb(m.me, 1024, &E[(size_t)p * D], D);          // start mel token
        if ((int)m.mp->ne[1] > 0) {
            std::vector<float> mp(D, 0);
            get_emb(m.mp, 0, mp.data(), D);
            for (int j = 0; j < D; j++) E[(size_t)p * D + j] += mp[j];
        }
        std::vector<float> hidden;
        forward_batch(E, Tpre, hidden);
        std::memcpy(emb_last.data(), &hidden[(size_t)(Tpre - 1) * D], D * sizeof(float));
        h_c1d(logits.data(), m.lmw_c.data(), emb_last.data(), m.lmb_c.data(), D, hp.mel_vocab);
    }
    if (getenv("XTTS_DUMP_LOGITS")) {
        FILE* f = fopen(getenv("XTTS_DUMP_LOGITS"), "wb");
        if (f) { fwrite(logits.data(), 4, logits.size(), f); fclose(f); }
        // self-check: recompute logits from the normed emb_last directly
        {
            float mx2 = -1e30f; int bi = -1;
            for (int m2 = 0; m2 < (int)hp.mel_vocab; m2++) {
                float s2 = m.lmb_c[m2];
                const float* Wm = m.lmw_c.data() + (size_t)m2 * D;
                for (int n2 = 0; n2 < D; n2++) s2 += Wm[n2] * emb_last[n2];
                if (s2 > mx2) { mx2 = s2; bi = m2; }
            }
            fprintf(stderr, "xtts_dbg: selfcheck argmax=%d val=%.3f | dumped argmax=%d val=%.3f | emb_last[0..4]:",
                    bi, mx2, (int)std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())), *std::max_element(logits.begin(), logits.end()));
            for (int j = 0; j < 5; j++) fprintf(stderr, " %.4f", emb_last[j]);
            fprintf(stderr, "\n");
        }
    }
    for (int g = 0; g < max_n; g++) {
        if ((int)codes.size() >= max_n) break;
        // Sample from logits
        std::vector<float> scaled(logits);
        if (temp > 0) for (auto& l2 : scaled) l2 /= temp;
        if (rep_pen != 1.0f && !codes.empty()) {
            // HF-style: penalize ALL distinct previously generated tokens
            std::vector<char> seen(hp.mel_vocab, 0);
            for (int tok : codes) if (tok >= 0 && tok < (int)scaled.size() && !seen[tok]) {
                seen[tok] = 1;
                scaled[tok] *= scaled[tok] > 0 ? 1.0f / rep_pen : rep_pen;
            }
        }
        // softmax
        float mx = *std::max_element(scaled.begin(), scaled.end());
        float sa = 0; for (auto& l2 : scaled) { l2 = expf(l2 - mx); sa += l2; }
        for (auto& l2 : scaled) l2 /= sa;
        // top-k filter (on probabilities)
        if (top_k > 0 && top_k < (int)scaled.size()) {
            std::vector<float> ls(scaled);
            std::partial_sort(ls.begin(), ls.begin() + top_k, ls.end(), std::greater<>());
            float thr = ls[top_k - 1];
            for (auto& l2 : scaled) if (l2 < thr) l2 = 0;
            float sa2 = 0; for (auto& l2 : scaled) sa2 += l2;
            if (sa2 > 0) for (auto& l2 : scaled) l2 /= sa2;
        }
        // top-p filter
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
        if (getenv("XTTS_DEBUG") && (g < 20 || g % 100 == 0)) {
            // Top-5 logits for diagnostics
            std::vector<std::pair<float,int>> top5;
            for (int j = 0; j < (int)scaled.size(); j++) if (scaled[j] > 0) top5.push_back({scaled[j], j});
            std::sort(top5.begin(), top5.end(), std::greater<>());
            fprintf(stderr, "xtts_dbg: step %d code=%d p=%.3f top:", g, st, scaled[st]);
            for (int k = 0; k < 3 && k < (int)top5.size(); k++) fprintf(stderr, " %d:%.3f", top5[k].second, top5[k].first);
            fprintf(stderr, " maxlogit=%.2f\n", mx);
        }
        if (st == 1025) break;  // stop token
        // Next token embedding: mel_embedding(st) + mel_pos_embedding(mel_pos)
        std::vector<float> me(D, 0), mp(D, 0), dummy(D);
        get_emb(m.me, st, me.data(), D);
        if (mel_pos < (int)m.mp->ne[1]) get_emb(m.mp, mel_pos, mp.data(), D);
        for (int j = 0; j < D; j++) me[j] += mp[j];
        mel_pos++;
        forward_token(me, emb_last);
        auto tl1 = omp_get_wtime();
        h_c1d(logits.data(), m.lmw_c.data(), emb_last.data(), m.lmb_c.data(), D, hp.mel_vocab);
        t_logits += omp_get_wtime() - tl1;
    }
    if (getenv("XTTS_TIMING")) if (getenv("XTTS_TIMING")) fprintf(stderr, "xtts_timing:   gen breakdown qkv=%.2f attn=%.2f proj=%.2f ffn=%.2f logits=%.2f s\n", t_qkv, t_attn, t_proj, t_ffn, t_logits);
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
    if (getenv("XTTS_DUMP_INPUT_EMBS")) {
        FILE* f = fopen(getenv("XTTS_DUMP_INPUT_EMBS"), "wb");
        if (f) { fwrite(embs.data(), 4, embs.size(), f); fclose(f); }
        fprintf(stderr, "xtts_dbg: dumped %zu input embs (T=%d)\n", embs.size(), T);
    }
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
    if (getenv("XTTS_DUMP_LNF")) ggml_build_forward_expand(gr, lnf);
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
    if (getenv("XTTS_DUMP_LNF")) {
        std::vector<float> x0((size_t)T * hp.d);
        memcpy(x0.data(), lnf->data, (size_t)T * hp.d * sizeof(float));
        FILE* f = fopen(getenv("XTTS_DUMP_LNF"), "wb");
        if (f) { fwrite(x0.data(), 4, (size_t)T * hp.d, f); fclose(f); }
        double s0 = 0; for (int j = 0; j < (int)hp.d; j++) s0 += (double)x0[j] * x0[j];
        fprintf(stderr, "xtts_dbg: dumped lnf_out (T=%d) pos0 rms=%.4f\n", T, std::sqrt(s0 / (double)hp.d));
    }
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
    int n_threads = 4;
};

XttSv2GptEngine::~XttSv2GptEngine() {
    if (impl_) { if (impl_->graph_ctx) ggml_free(impl_->graph_ctx);
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
    if (!load_gguf(gpt_path.c_str(), impl_->backend, impl_->gpt_wl)) {
        ggml_backend_free(impl_->backend); delete impl_; impl_ = nullptr; return false; }
    if (verbose) fprintf(stderr, "xtts_v2: loaded %d tensors from GPT GGUF\n", (int)impl_->gpt_wl.tensors.size());
    gguf_context* gu = gguf_init_from_file(gpt_path.c_str(), {true, nullptr});
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
    // Per-channel mel normalization (clips_mel_norms.pth / mel_stats.pth, 80 float32 values, log-domain means)
    static std::vector<float> mel_norms;
    if (mel_norms.empty()) {
        FILE* f = fopen("/data4/work/TTS/weights/gguf/mel_norms.bin", "rb");
        if (f) { mel_norms.resize(80); size_t rd = fread(mel_norms.data(), 4, 80, f); fclose(f);
                 if (rd != 80) mel_norms.clear(); }
        if (mel_norms.size() == 80) fprintf(stderr, "xtts_v2: mel_norms loaded (80 values, [0]=%.3f)\n", mel_norms[0]);
        else fprintf(stderr, "xtts_v2: WARNING mel_norms not loaded, using 1.0\n");
    }
    // Process 4-second chunks: mel -> cond encoder -> perceiver -> [32,1024]; average.
    std::vector<float> acc((size_t)hp.cond_latents * hp.d, 0.0f);
    if (acc.empty()) acc.resize((size_t)32 * 1024);
    int n_chunks = 0;
    for (int cs = 0; cs < ref_ns; cs += chunk_ns) {
        int cn = std::min(chunk_ns, ref_ns - cs);
        if (cn < min_chunk_ns) continue;    // reference skips chunks < 0.33 s
        auto tm0 = std::chrono::steady_clock::now();
        std::vector<float> mel = comp_mel(ref_pcm + cs, cn, 22050, 2048, 256, 1024, 80, 0, 8000);
        if (mel.empty()) continue;
        auto tm1 = std::chrono::steady_clock::now();
        int T = (int)mel.size() / 80;
        if (mel_norms.size() == 80)
            for (int mch = 0; mch < 80; mch++) for (int t = 0; t < T; t++)
                mel[(size_t)mch * T + t] /= mel_norms[mch];
        // comp_mel returns channel-major [80][T]; cond_enc_fwd expects time-major [T][80]
        std::vector<float> melT((size_t)T * 80);
        for (int mch = 0; mch < 80; mch++) for (int t = 0; t < T; t++)
            melT[(size_t)t * 80 + mch] = mel[(size_t)mch * T + t];
        auto tm2 = std::chrono::steady_clock::now();
        auto cl = cond_enc_fwd(m, melT.data(), T, hp.d);
        if (cl.empty()) return false;
        auto tm3 = std::chrono::steady_clock::now();
        if (getenv("XTTS_TIMING")) fprintf(stderr, "xtts_timing:   chunk %d: mel=%.2f cond_enc=%.2f", n_chunks, std::chrono::duration<double>(tm1 - tm0).count(), std::chrono::duration<double>(tm3 - tm2).count());
        if (getenv("XTTS_DUMP_PERC")) {
            char ps[512], pm[512], pc[512];
            snprintf(ps, sizeof(ps), "%s.c%d.style.bin", getenv("XTTS_DUMP_PERC"), n_chunks);
            snprintf(pm, sizeof(pm), "%s.c%d.mel.bin", getenv("XTTS_DUMP_PERC"), n_chunks);
            snprintf(pc, sizeof(pc), "%s.c%d.melT.bin", getenv("XTTS_DUMP_PERC"), n_chunks);
            FILE* fs = fopen(ps, "wb"); if (fs) { fwrite(cl.data(), 4, cl.size(), fs); fclose(fs); }
            FILE* fm = fopen(pm, "wb"); if (fm) { fwrite(mel.data(), 4, mel.size(), fm); fclose(fm); }
            FILE* fc = fopen(pc, "wb"); if (fc) { fwrite(melT.data(), 4, melT.size(), fc); fclose(fc); }
            fprintf(stderr, "xtts_dbg: chunk %d T=%d ns=%d dumped mel/style\n", n_chunks, T, cn);
        }
        // perceiver resampler: cond encoder output [T,1024] -> [32,1024]
        ggml_context* ctx = ggml_init({1 << 24, nullptr, false});
        auto tm4 = std::chrono::steady_clock::now();
        auto pl = perceiver_fwd(m, cl.data(), (int)cl.size() / hp.d, hp.d, n_chunks);
        ggml_free(ctx);
        if (pl.empty()) return false;
        auto tm5 = std::chrono::steady_clock::now();
        fprintf(stderr, " perceiver=%.2f s\n", std::chrono::duration<double>(tm5 - tm4).count());
        if (getenv("XTTS_DEBUG")) {
            float mn = pl[0], mx = pl[0], sum = 0;
            for (auto v : pl) { mn = std::min(mn, v); mx = std::max(mx, v); sum += v * v; }
            fprintf(stderr, "xtts_dbg: chunk %d (%d ms) latents n=%d min=%.4f max=%.4f rms=%.4f\n",
                    n_chunks, cn * 1000 / 22050, (int)pl.size(), mn, mx, sqrtf(sum / pl.size()));
        }
        for (size_t i = 0; i < acc.size(); i++) acc[i] += pl[i];
        n_chunks++;
    }
    if (n_chunks == 0) return false;
    for (auto& v : acc) v /= (float)n_chunks;
    if (getenv("XTTS_DEBUG")) {
        float mn = acc[0], mx = acc[0], sum = 0;
        for (auto v : acc) { mn = std::min(mn, v); mx = std::max(mx, v); sum += v * v; }
        fprintf(stderr, "xtts_dbg: cond latents (mean of %d chunks) n=%d min=%.4f max=%.4f rms=%.4f\n",
                n_chunks, (int)acc.size(), mn, mx, sqrtf(sum / acc.size()));
    }
    out = std::move(acc);
    return true;
}
std::vector<int32_t> XttSv2GptEngine::generate_mel_codes(const std::vector<int32_t>& text_tokens, const std::vector<float>& cond_latents,
        float temperature, int top_k, float top_p, float rep_penalty, int max_tokens) {
    if (!impl_) return {};
    ggml_context* ctx = ggml_init({1 << 22, nullptr, false});
    auto codes = gpt_gen(ctx, impl_->backend, impl_->model, impl_->hp,
        std::vector<int>(text_tokens.begin(), text_tokens.end()), cond_latents,
        temperature, top_k, top_p, rep_penalty, max_tokens);
    ggml_free(ctx); return std::vector<int32_t>(codes.begin(), codes.end());
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
