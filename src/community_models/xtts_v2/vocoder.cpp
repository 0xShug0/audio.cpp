// xtts_v2 vocoder -- HiFi-GAN + ResNet speaker encoder
// Ported from CrispASR xtts_voc.cpp for audio.cpp framework.
// MSVC's <cmath> only exposes M_PI with _USE_MATH_DEFINES; define it before
// any include (framework headers may pull <cmath> in transitively).
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include "engine/community_models/xtts_v2/vocoder.h"
#include "engine/framework/core/backend.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace engine::models::xtts_v2 {
namespace {

static ggml_tensor* T(const std::map<std::string, ggml_tensor*>& m, const std::string& name) {
    auto it = m.find(name);
    return (it != m.end()) ? it->second : nullptr;
}

static ggml_tensor* conv1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w,
    ggml_tensor* b, int stride, int padding, int dilation) {
    ggml_tensor* y = ggml_conv_1d(ctx, w, x, stride, padding, dilation);
    if (b) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, (int)b->ne[0]));
    return y;
}

static ggml_tensor* conv_t1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w,
    ggml_tensor* b, int stride, int padding) {
    ggml_tensor* y = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);
    if (padding > 0) {
        int64_t Tt = y->ne[0] - 2 * padding;
        y = ggml_cont(ctx, ggml_view_2d(ctx, y, Tt, y->ne[1], y->nb[1], padding * ggml_element_size(y)));
    }
    if (b) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, (int)b->ne[0]));
    return y;
}

static ggml_tensor* resblock(ggml_context* ctx, ggml_tensor* x,
    const std::map<std::string, ggml_tensor*>& ts, const std::string& p,
    int ks, const std::vector<int>& dil, float sl) {
    for (int d = 0; d < (int)dil.size(); d++) {
        ggml_tensor* r = x;
        x = ggml_leaky_relu(ctx, x, sl, false);
        x = conv1d(ctx, x, T(ts, p + ".c1." + std::to_string(d)),
                   T(ts, p + ".c1." + std::to_string(d) + ".bias"),
                   1, dil[d] * (ks - 1) / 2, dil[d]);
        x = ggml_leaky_relu(ctx, x, sl, false);
        x = conv1d(ctx, x, T(ts, p + ".c2." + std::to_string(d)),
                   T(ts, p + ".c2." + std::to_string(d) + ".bias"), 1, (ks - 1) / 2, 1);
        x = ggml_add(ctx, x, r);
    }
    return x;
}

static ggml_tensor* bn4d(ggml_context* ctx, ggml_tensor* x, const std::string& pfx) {
    // placeholder - actual BN inline
    return x;
}

// HiFi-GAN graph
static ggml_cgraph* build_hifigan(ggml_context* ctx0,
    const std::map<std::string, ggml_tensor*>& ts, int T_mel, bool use_spk, ggml_tensor* spk_in) {
    ggml_cgraph* gf = ggml_new_graph(ctx0);
    // T_mel here is the ALREADY-upsampled frame count (host-side linear interp above);
    // the graph input is [frames, 1024] channel-major.
    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_mel, 1024);
    ggml_set_name(x, "voc_input"); ggml_set_input(x);
    x = conv1d(ctx0, x, T(ts, "hifigan_decoder.dec.conv_pre.weight"),
               T(ts, "hifigan_decoder.dec.conv_pre.bias"), 1, 3, 1);
    if (use_spk) {
        auto* c = T(ts, "hifigan_decoder.dec.cond_layer.weight");
        auto* s = spk_in;  // shared input tensor for all 5 conditioning points
        auto* p = ggml_cont(ctx0, ggml_transpose(ctx0,
            ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, c, (int)c->ne[1], (int)c->ne[2]),
                         ggml_reshape_2d(ctx0, s, 512, 1))));
        if (auto* cb = T(ts, "hifigan_decoder.dec.cond_layer.bias"))
            p = ggml_add(ctx0, p, ggml_reshape_2d(ctx0, cb, 1, (int)ggml_nelements(cb)));
        x = ggml_add(ctx0, x, ggml_repeat(ctx0, p, x));
    }
    int rates[4] = {8,8,2,2}, ks[4] = {16,16,4,4}, rks[3] = {3,7,11};
    std::vector<int> dil = {1,3,5};
    for (int i = 0; i < 4; i++) {
        x = ggml_leaky_relu(ctx0, x, 0.1f, false);
        std::string u = "hifigan_decoder.dec.ups." + std::to_string(i);
        x = conv_t1d(ctx0, x, T(ts, u), T(ts, u + ".bias"), rates[i], (ks[i] - rates[i]) / 2);
        if (use_spk) {
            std::string cw = "hifigan_decoder.dec.conds." + std::to_string(i) + ".weight";
            auto* w = T(ts, cw);
            auto* s = spk_in;  // same shared input tensor
            auto* p = ggml_cont(ctx0, ggml_transpose(ctx0,
                ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, w, (int)w->ne[1], (int)w->ne[2]),
                             ggml_reshape_2d(ctx0, s, 512, 1))));
            if (auto* cb = T(ts, cw + ".bias"))
                p = ggml_add(ctx0, p, ggml_reshape_2d(ctx0, cb, 1, (int)ggml_nelements(cb)));
            x = ggml_add(ctx0, x, ggml_repeat(ctx0, p, x));
        }
        ggml_tensor* rs = nullptr;
        for (int j = 0; j < 3; j++) {
            std::string rp = "hifigan_decoder.dec.resb." + std::to_string(i * 3 + j);
            auto* rb = resblock(ctx0, x, ts, rp, rks[j], dil, 0.1f);
            rs = rs ? ggml_add(ctx0, rs, rb) : rb;
        }
        x = ggml_scale(ctx0, rs, 1.0f / 3);
    }
    x = ggml_leaky_relu(ctx0, x, 0.1f, false);
    x = conv1d(ctx0, x, T(ts, "hifigan_decoder.dec.conv_post.weight"),
               T(ts, "hifigan_decoder.dec.conv_post.bias"), 1, 3, 1);
    x = ggml_tanh(ctx0, x);
    ggml_set_name(x, "voc_output"); ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

// BN apply for conv2d: (x - rm) / sqrt(rv) * w + b
static ggml_tensor* apply_bn4d(ggml_context* ctx, ggml_tensor* x,
    ggml_tensor* rm, ggml_tensor* rv, ggml_tensor* w, ggml_tensor* b) {
    auto* rm4 = ggml_reshape_4d(ctx, rm, 1, 1, (int)x->ne[2], 1);
    auto* rv4 = ggml_reshape_4d(ctx, rv, 1, 1, (int)x->ne[2], 1);
    auto* w4 = ggml_reshape_4d(ctx, w, 1, 1, (int)x->ne[2], 1);
    auto* b4 = ggml_reshape_4d(ctx, b, 1, 1, (int)x->ne[2], 1);
    auto* m = ggml_repeat(ctx, rm4, x);
    // BN eps (1e-5) is pre-folded into .rv tensors at load time
    auto* v = ggml_repeat(ctx, ggml_sqrt(ctx, rv4), x);
    return ggml_add(ctx, ggml_mul(ctx, ggml_div(ctx, ggml_sub(ctx, x, m), v),
                   ggml_repeat(ctx, w4, x)), ggml_repeat(ctx, b4, x));
}

static ggml_tensor* se_blk(ggml_context* ctx, ggml_tensor* x, int C,
    ggml_tensor* f0w, ggml_tensor* f0b, ggml_tensor* f2w, ggml_tensor* f2b) {
    int64_t sp = x->ne[0] * x->ne[1];
    ggml_tensor* xr = ggml_reshape_3d(ctx, x, sp, C, 1);
    ggml_tensor* mn = ggml_scale(ctx, ggml_sum_rows(ctx, xr), 1.0f / sp);
    ggml_tensor* mT = ggml_cont(ctx, ggml_transpose(ctx, mn)); // [C,1,1]
    ggml_tensor* h = ggml_relu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, f0w, mT), f0b));
    ggml_tensor* g = ggml_sigmoid(ctx, ggml_add(ctx, ggml_mul_mat(ctx, f2w, h), f2b));
    ggml_tensor* g4 = ggml_reshape_4d(ctx, g, 1, 1, C, 1);
    return ggml_mul(ctx, x, ggml_repeat(ctx, g4, x));
}

static ggml_tensor* se_basic(ggml_context* ctx, ggml_tensor* x,
    const std::map<std::string, ggml_tensor*>& ts, const std::string& p,
    int C, bool ds) {
    ggml_tensor* res = x;
    // reference: conv1 -> relu -> bn1
    ggml_tensor* h = ggml_conv_2d(ctx, T(ts, p + ".conv1.weight"), x,
        ds ? 2 : 1, ds ? 2 : 1, 1, 1, 1, 1);
    h = ggml_relu(ctx, h);
    h = apply_bn4d(ctx, h, T(ts, p + ".bn1.rm"), T(ts, p + ".bn1.rv"),
                   T(ts, p + ".bn1.weight"), T(ts, p + ".bn1.bias"));
    h = ggml_conv_2d(ctx, T(ts, p + ".conv2.weight"), h, 1, 1, 1, 1, 1, 1);
    h = apply_bn4d(ctx, h, T(ts, p + ".bn2.rm"), T(ts, p + ".bn2.rv"),
                   T(ts, p + ".bn2.weight"), T(ts, p + ".bn2.bias"));
    h = se_blk(ctx, h, C, T(ts, p + ".se.fc.0.weight"), T(ts, p + ".se.fc.0.bias"),
               T(ts, p + ".se.fc.2.weight"), T(ts, p + ".se.fc.2.bias"));
    if (ds) {
        ggml_tensor* dc = ggml_conv_2d(ctx, T(ts, p + ".downsample.0.weight"), res, 2, 2, 0, 0, 1, 1);
        if (T(ts, p + ".downsample.1.weight")) {
            dc = apply_bn4d(ctx, dc, T(ts, p + ".downsample.1.rm"), T(ts, p + ".downsample.1.rv"),
                           T(ts, p + ".downsample.1.weight"), T(ts, p + ".downsample.1.bias"));
        }
        res = dc;
    }
    // reference: out += residual; out = relu(out)
    return ggml_relu(ctx, ggml_add(ctx, h, res));
}

// Speaker encoder graph
static ggml_cgraph* build_spk_enc(ggml_context* ctx0,
    const std::map<std::string, ggml_tensor*>& ts, int T_mel) {
    ggml_cgraph* gf = ggml_new_graph(ctx0);
    // GGML conv_2d expects feature maps as ne=[W(time), H(mels), C, N]
    // data layout: element (t, ch) at t + ch*T_mel -> matches mel[ch*Tm + t]
    ggml_tensor* x = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, T_mel, 64, 1, 1);
    ggml_set_name(x, "speaker_mel"); ggml_set_input(x);
    // reference stem: conv1 -> relu -> bn1
    ggml_tensor* h = ggml_conv_2d(ctx0, T(ts, "hifigan_decoder.se.conv1.weight"), x, 1, 1, 1, 1, 1, 1);
    h = ggml_add(ctx0, h, ggml_repeat(ctx0, ggml_reshape_4d(ctx0, T(ts, "hifigan_decoder.se.conv1.bias"), 1, 1, (int)h->ne[2], 1), h));
    h = ggml_relu(ctx0, h);
    h = apply_bn4d(ctx0, h, T(ts, "hifigan_decoder.se.bn1.rm"), T(ts, "hifigan_decoder.se.bn1.rv"),
                   T(ts, "hifigan_decoder.se.bn1.weight"), T(ts, "hifigan_decoder.se.bn1.bias"));
    int lf[4] = {32,64,128,256}, lb[4] = {3,4,6,3};
    for (int ly = 0; ly < 4; ly++) {
        for (int bl = 0; bl < lb[ly]; bl++) {
            std::string pfx = "hifigan_decoder.se.layer" + std::to_string(ly + 1) + "." + std::to_string(bl);
            h = se_basic(ctx0, h, ts, pfx, lf[ly], (bl == 0 && ly > 0));
        }
    }
    int Ws = (int)h->ne[0], Hs = (int)h->ne[1], Cs = (int)h->ne[2];
    // Flatten conv output to [Ws, Cs*Hs]: row r = c*Hs + h (PyTorch reshape(B, -1, T) order).
    // GGML element (w,h,c) at w + h*Ws + c*Ws*Hs == 2D view (w, r=c*Hs+h) at w + r*Ws. Exact match.
    ggml_tensor* xf = ggml_reshape_2d(ctx0, h, Ws, Cs * Hs);
    // ASP attention: Conv1d(2048,128,1) -> ReLU -> BN1d(128) -> Conv1d(128,2048,1) -> softmax(time)
    ggml_tensor* a1 = conv1d(ctx0, xf, T(ts, "hifigan_decoder.se.attention.0.weight"),
                             T(ts, "hifigan_decoder.se.attention.0.bias"), 1, 0, 1);
    a1 = ggml_relu(ctx0, a1);
    // BN1d over channels: a1 is [Ws, 128]; normalize per channel over time
    {
        auto* rm = T(ts, "hifigan_decoder.se.attention.2.rm");
        auto* rv = T(ts, "hifigan_decoder.se.attention.2.rv");
        auto* w  = T(ts, "hifigan_decoder.se.attention.2.weight");
        auto* b  = T(ts, "hifigan_decoder.se.attention.2.bias");
        auto* rm2 = ggml_reshape_2d(ctx0, rm, 1, (int)a1->ne[1]);
        auto* rv2 = ggml_reshape_2d(ctx0, rv, 1, (int)a1->ne[1]);
        auto* w2  = ggml_reshape_2d(ctx0, w, 1, (int)a1->ne[1]);
        auto* b2  = ggml_reshape_2d(ctx0, b, 1, (int)a1->ne[1]);
        auto* mn = ggml_repeat(ctx0, rm2, a1);
        auto* sd = ggml_repeat(ctx0, ggml_sqrt(ctx0, rv2), a1);
        a1 = ggml_add(ctx0, ggml_mul(ctx0, ggml_div(ctx0, ggml_sub(ctx0, a1, mn), sd), w2), b2);
    }
    ggml_tensor* a2 = conv1d(ctx0, a1, T(ts, "hifigan_decoder.se.attention.3.weight"),
                             T(ts, "hifigan_decoder.se.attention.3.bias"), 1, 0, 1);
    // softmax over time (ne[0]); w sums to 1 per channel row
    a2 = ggml_soft_max(ctx0, a2);
    // mu = sum_t x*w  [2048], sg = sqrt(clamp(sum_t x^2*w - mu^2))  [2048]
    ggml_tensor* xw = ggml_mul(ctx0, xf, a2);
    int CH = Cs * Hs;
    ggml_tensor* mu = ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, xw), CH, 1);   // [2048, 1]
    ggml_tensor* x2w = ggml_mul(ctx0, ggml_mul(ctx0, xf, xf), a2);
    ggml_tensor* var = ggml_sub(ctx0, ggml_reshape_2d(ctx0, ggml_sum_rows(ctx0, x2w), CH, 1),
                                ggml_mul(ctx0, mu, mu));
    ggml_tensor* sg = ggml_sqrt(ctx0, ggml_clamp(ctx0, var, 1e-5f, 1e30f));
    ggml_tensor* asp = ggml_concat(ctx0, mu, sg, 0);                 // [4096, 1]
    ggml_tensor* emb = ggml_mul_mat(ctx0, T(ts, "hifigan_decoder.se.fc.weight"), asp);
    if (T(ts, "hifigan_decoder.se.fc.bias"))
        emb = ggml_add(ctx0, emb, T(ts, "hifigan_decoder.se.fc.bias"));
    // L2 normalize
    ggml_tensor* nm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_mul(ctx0, emb, emb)));
    emb = ggml_div(ctx0, emb, nm);
    ggml_set_name(emb, "speaker_embedding");
    ggml_build_forward_expand(gf, emb);
    return gf;
}

// GGUF weight loader
struct WL { ggml_context* ctx = nullptr; ggml_backend_buffer_t buf = nullptr;
            std::map<std::string, ggml_tensor*> tensors; };

static bool load_gguf(const char* path, ggml_backend_t be, WL& wl) {
    gguf_init_params gp = {true, &wl.ctx};
    gguf_context* gu = gguf_init_from_file(path, gp);
    if (!gu || !wl.ctx) { if (gu) gguf_free(gu); return false; }
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
    // Fold BatchNorm eps (1e-5) into running_var tensors: PyTorch eval does sqrt(var + eps)
    for (ggml_tensor* t = ggml_get_first_tensor(wl.ctx); t; t = ggml_get_next_tensor(wl.ctx, t)) {
        std::string nm = ggml_get_name(t);
        if (nm.size() > 3 && nm.compare(nm.size() - 3, 3, ".rv") == 0 && t->type == GGML_TYPE_F32) {
            int64_t n = ggml_nelements(t);
            std::vector<float> tmp(n);
            ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(float));
            for (int64_t i = 0; i < n; i++) tmp[i] += 1e-5f;
            ggml_backend_tensor_set(t, tmp.data(), 0, n * sizeof(float));
        }
    }
    for (ggml_tensor* t = ggml_get_first_tensor(wl.ctx); t; t = ggml_get_next_tensor(wl.ctx, t))
        wl.tensors[ggml_get_name(t)] = t;
    return true;
}

// Mel spectrogram (simple DFT-based, for conditioning only)
// Speaker-encoder mel: hamming window, centered frames, power mel, NO log
// (matches torchaudio MelSpectrogram with hamming_window + PreEmphasis(0.97) applied upstream)
static std::vector<float> compute_mel_hamming(const float* pcm, int ns, int sr, int nfft,
    int hop, int win, int nm, float fmin, float fmax) {
    // torch.hamming_window(win) default: PERIODIC (denominator win-1)
    std::vector<float> ham(win);
    for (int i = 0; i < win; i++) ham[i] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i / (win - 1));
    // torchaudio center=True, pad_mode="reflect": reflect padding of nfft/2 on both sides
    int pad = nfft / 2;
    std::vector<float> padded(2 * pad + ns, 0);
    for (int i = 1; i <= pad; i++) {
        int li = i, ri = i;
        while (li >= ns) li = 2 * (ns - 1) - li;
        while (ri >= ns) ri = 2 * (ns - 1) - ri;
        padded[pad - i] = pcm[li];
        padded[pad + ns - 1 + i] = pcm[ns - 1 - ri];
    }
    std::memcpy(padded.data() + pad, pcm, ns * sizeof(float));
    int T = (int)((padded.size() - nfft) / hop) + 1;
    if (T <= 0) return {};
    int nf = nfft / 2 + 1;
    auto mscl = [](float f){ return 1127.01048f * logf(1.0f + f / 700.0f); };
    float fmin_m = mscl(fmin), fmax_m = mscl(fmax);
    std::vector<float> mpts(nm + 2);
    for (int i = 0; i <= nm + 1; i++)
        mpts[i] = 700.0f * (expf((fmin_m + (fmax_m - fmin_m) * i / (nm + 1)) / 1127.01048f) - 1.0f);
    std::vector<float> fb(nm * nf, 0);
    for (int m = 0; m < nm; m++) for (int f = 0; f < nf; f++) {
        float freq = (float)sr * f / nfft;
        if (freq >= mpts[m] && freq <= mpts[m + 2])
            fb[m * nf + f] = freq <= mpts[m + 1] ?
                (freq - mpts[m]) / (mpts[m + 1] - mpts[m] + 1e-12f) :
                (mpts[m + 2] - freq) / (mpts[m + 2] - mpts[m + 1] + 1e-12f);
    }
    std::vector<float> mel(nm * T, 0);
    std::vector<float> frame(nfft);
    // torchaudio centers the (short) window inside the n_fft frame
    const int wpad = (nfft - win) / 2;
    for (int t = 0; t < T; t++) {
        for (int n = 0; n < nfft; n++) {
            int wi = n - wpad;
            frame[n] = padded[(size_t)t * hop + n] * ((wi >= 0 && wi < win) ? ham[wi] : 0.0f);
        }
        // radix-2 FFT (nfft=512), iterative Cooley-Tukey: requires bit-reversed input order
        std::vector<float> re(frame), im(nfft, 0);
        for (int i = 1, j = 0; i < nfft; i++) {
            int bit = nfft >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
        }
        for (int len = 2; len <= nfft; len <<= 1) {
            int half = len >> 1;
            float wr = cosf(-2.0f * (float)M_PI / len), wi = sinf(-2.0f * (float)M_PI / len);
            for (int i = 0; i < nfft; i += len) {
                float cr = 1, ci = 0;
                for (int j = 0; j < half; j++) {
                    int a = i + j, b = i + j + half;
                    float tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
                    re[b] = re[a] - tr; im[b] = im[a] - ti;
                    re[a] += tr; im[a] += ti;
                    float nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
                }
            }
        }
        for (int m = 0; m < nm; m++) {
            float s = 0;
            for (int f = 0; f < nf; f++) s += fb[m * nf + f] * (re[f] * re[f] + im[f] * im[f]);
            mel[m * T + t] = s;
        }
    }
    return mel;
}

static std::vector<float> compute_mel(const float* pcm, int ns, int sr, int nfft,
    int hop, int win, int nm, float fmin, float fmax, bool log_flag) {
    std::vector<float> hann(win);
    for (int i = 0; i < win; i++) hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / win));
    int nf = nfft / 2 + 1;
    auto mscl = [](float f){ return 1127.01048f * logf(1.0f + f / 700.0f); };
    float fmin_m = mscl(fmin), fmax_m = mscl(fmax);
    std::vector<float> mpts(nm + 2);
    for (int i = 0; i <= nm + 1; i++)
        mpts[i] = 700.0f * (expf((fmin_m + (fmax_m - fmin_m) * i / (nm + 1)) / 1127.01048f) - 1.0f);
    std::vector<float> fb(nm * nf, 0);
    for (int m = 0; m < nm; m++) for (int f = 0; f < nf; f++) {
        float freq = (float)sr * f / nfft;
        if (freq >= mpts[m] && freq <= mpts[m + 2])
            fb[m * nf + f] = freq <= mpts[m + 1] ?
                (freq - mpts[m]) / (mpts[m + 1] - mpts[m] + 1e-12f) :
                (mpts[m + 2] - freq) / (mpts[m + 2] - mpts[m + 1] + 1e-12f);
    }
    std::vector<float> padded(ns + nfft - 1, 0);
    std::memcpy(padded.data() + (nfft - 1) / 2, pcm, ns * sizeof(float));
    int T = (int)((padded.size() - nfft) / hop) + 1;
    if (T <= 0) return {};
    std::vector<float> mel(nm * T, 0);
    for (int t = 0; t < T; t++) {
        std::vector<float> re(nf, 0), im(nf, 0);
        for (int k = 0; k < nf; k++) for (int n = 0; n < nfft; n++) {
            float a = -2.0f * (float)M_PI * k * n / nfft;
            re[k] += padded[t * hop + n] * (n < win ? hann[n] : 0.0f) * cosf(a);
            im[k] += padded[t * hop + n] * (n < win ? hann[n] : 0.0f) * sinf(a);
        }
        for (int m = 0; m < nm; m++) {
            float s = 0;
            for (int f = 0; f < nf; f++) s += fb[m * nf + f] * (re[f] * re[f] + im[f] * im[f]);
            mel[m * T + t] = log_flag ? logf(s + 1e-7f) : s;
        }
    }
    return mel;
}

} // namespace

struct XttSv2Vocoder::Impl {
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
    std::vector<uint8_t> compute_meta;
    int n_threads = 4;
    ~Impl() {
        if (ctx_w) ggml_free(ctx_w);
        if (buf_w) ggml_backend_buffer_free(buf_w);
        if (backend_cpu) ggml_backend_free(backend_cpu);
    }
};

XttSv2Vocoder::~XttSv2Vocoder() {
    if (impl_) { delete impl_; impl_ = nullptr; }
}

bool XttSv2Vocoder::load(const std::filesystem::path& path, const core::BackendConfig& backend_config) {
    n_threads_ = backend_config.threads;
    if (impl_) { delete impl_; impl_ = nullptr; }
    impl_ = new Impl();
    auto& c = *impl_;
    c.backend_cpu = core::init_backend(backend_config);
    if (!c.backend_cpu) return false;
    core::set_backend_threads(c.backend_cpu, backend_config.threads);
    WL wl;
    if (!load_gguf(path.string().c_str(), c.backend_cpu, wl)) {
        fprintf(stderr, "xtts_v2: failed to load vocoder GGUF: %s\n", path.string().c_str());
        return false;
    }
    c.ctx_w = wl.ctx;
    c.tensors = std::move(wl.tensors);
    { gguf_context* g = gguf_init_from_file(path.string().c_str(), {true, 0});
      if (g) {
          int ki = gguf_find_key(g, "xtts.mel_stats");
          if (ki >= 0 && gguf_get_kv_type(g, ki) == GGUF_TYPE_FLOAT32) {
              int n = gguf_get_arr_n(g, ki);
              const float* d = static_cast<const float*>(gguf_get_arr_data(g, ki));
              mel_stats_.assign(d, d + n);
          }
          gguf_free(g);
      }
    }
    fprintf(stderr, "xtts_v2: vocoder loaded from %s (%zu tensors)\n",
            path.string().c_str(), c.tensors.size());
    return true;
}

std::vector<float> XttSv2Vocoder::generate(const float* latent, int T, const float* spk_emb) const {
    if (!impl_ || !latent || T <= 0) return {};
    auto& c = *impl_;
    // Reference HifiDecoder.forward: TWO chained linear interpolates (align_corners=False):
    //   Stage 1) scale_factor = ar_mel_length_compression / output_hop_length = 1024/256 = 4
    //   Stage 2) scale_factor = output_sample_rate / input_sample_rate = 24000/22050
    // PyTorch align_corners=False: output len = floor(input * scale),
    //   src_pos = (dst + 0.5) / scale - 0.5
    // These must NOT be fused into one: the intermediate discrete length matters.
    const double s1 = 4.0;
    const double s2 = 24000.0 / 22050.0;
    int T1 = (int)(T * s1);
    std::vector<float> stage1(1024 * T1, 0);
    for (int ch = 0; ch < 1024; ch++)
        for (int to = 0; to < T1; to++) {
            float pos = (float)((to + 0.5) / s1 - 0.5);
            if (pos < 0) pos = 0;
            if (pos > T - 1) pos = T - 1;
            int t0 = (int)pos, t1 = std::min(t0 + 1, T - 1);
            float f = pos - t0;
            stage1[(size_t)ch * T1 + to] = latent[(size_t)ch * T + t0] * (1.0f - f) + latent[(size_t)ch * T + t1] * f;
        }
    int Tu = (int)(T1 * s2);
    std::vector<float> ups(1024 * Tu, 0);
    for (int ch = 0; ch < 1024; ch++)
        for (int to = 0; to < Tu; to++) {
            float pos = (float)((to + 0.5) / s2 - 0.5);
            if (pos < 0) pos = 0;
            if (pos > T1 - 1) pos = T1 - 1;
            int t0 = (int)pos, t1 = std::min(t0 + 1, T1 - 1);
            float f = pos - t0;
            ups[(size_t)ch * Tu + to] = stage1[(size_t)ch * T1 + t0] * (1.0f - f) + stage1[(size_t)ch * T1 + t1] * f;
        }
    size_t need = 128 * 1024 * 1024;
    if (c.compute_meta.size() < need) c.compute_meta.resize(need);
    ggml_context* ctx0 = ggml_init({need, c.compute_meta.data(), true});
    // Single shared speaker_emb input tensor (used by cond_layer + 4 conds.N blocks)
    ggml_tensor* spk_t = nullptr;
    if (spk_emb) {
        spk_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 512);
        ggml_set_name(spk_t, "speaker_emb"); ggml_set_input(spk_t);
    }
    ggml_cgraph* gf = build_hifigan(ctx0, c.tensors, Tu, spk_emb != nullptr, spk_t);
    // Allocate input tensors (tracked for cleanup)
    ggml_backend_buffer_t vin_buf = nullptr;
    ggml_backend_buffer_t spk_buf = nullptr;
    { ggml_tensor* vin = ggml_graph_get_tensor(gf, "voc_input");
      vin_buf = ggml_backend_alloc_buffer(c.backend_cpu, ggml_nbytes(vin));
      vin->buffer = vin_buf;
      vin->data = (void*)ggml_backend_buffer_get_base(vin_buf);
      ggml_backend_tensor_set(vin, ups.data(), 0, ggml_nbytes(vin)); }
    if (spk_t) {
        spk_buf = ggml_backend_alloc_buffer(c.backend_cpu, ggml_nbytes(spk_t));
        spk_t->buffer = spk_buf;
        spk_t->data = (void*)ggml_backend_buffer_get_base(spk_buf);
        ggml_backend_tensor_set(spk_t, spk_emb, 0, 512 * sizeof(float));
    }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c.backend_cpu));
    if (!ga || !ggml_gallocr_reserve(ga, gf)) {
        ggml_backend_buffer_free(vin_buf); ggml_backend_buffer_free(spk_buf);
        if (ga) ggml_gallocr_free(ga); ggml_free(ctx0); return {}; }
    if (!ggml_gallocr_alloc_graph(ga, gf) ||
        ggml_backend_graph_compute(c.backend_cpu, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(vin_buf); ggml_backend_buffer_free(spk_buf);
        ggml_gallocr_free(ga); ggml_free(ctx0); return {};
    }
    ggml_tensor* out = ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1);
    int64_t Ta = out->ne[0];
    if (Ta <= 0) {
        ggml_backend_buffer_free(vin_buf); ggml_backend_buffer_free(spk_buf);
        ggml_gallocr_free(ga); ggml_free(ctx0); return {}; }
    std::vector<float> result(Ta);
    ggml_backend_tensor_get(out, result.data(), 0, Ta * sizeof(float));
    ggml_backend_buffer_free(vin_buf);
    ggml_backend_buffer_free(spk_buf);
    ggml_gallocr_free(ga); ggml_free(ctx0);
    return result;
}

std::vector<float> XttSv2Vocoder::speaker_embed(const float* ref_pcm, int n_samples) const {
    if (!impl_ || !ref_pcm || n_samples <= 0) return {};
    auto& c = *impl_;
    // Resample 22050->16000
    std::vector<float> pcm16;
    { double r = 22050.0 / 16000.0; int ol = (int)std::round(n_samples / r);
      pcm16.resize(ol);
      for (int i = 0; i < ol; i++) {
          double s = i * r; int idx = (int)s; double f = s - idx;
          if (idx + 1 < n_samples) pcm16[i] = (float)((1.0 - f) * ref_pcm[idx] + f * ref_pcm[idx + 1]);
          else if (idx < n_samples) pcm16[i] = ref_pcm[idx];
      }
    }
    // Reference torch_spec: PreEmphasis(0.97) -> MelSpectrogram(16k, n_fft=512, win=400,
    // hop=160, hamming, n_mels=64, power=2.0), then ResNetSpeakerEncoder log_input=True:
    // x = (x + 1e-6).log(), then InstanceNorm1d(64)
    std::vector<float> pre(pcm16.size());
    for (int i = (int)pcm16.size() - 1; i > 0; i--) pre[i] = pcm16[i] - 0.97f * pcm16[i - 1];
    if (!pcm16.empty()) pre[0] = pcm16[0];
    auto mel = compute_mel_hamming(pre.data(), (int)pre.size(), 16000, 512, 160, 400, 64, 0, 8000);
    for (auto& v : mel) v = logf(v + 1e-6f); // LOGFIX
    int Tm = (int)mel.size() / 64;
    // InstanceNorm1d(64): per-channel zero-mean unit-variance over time
    for (int ch = 0; ch < 64; ch++) {
        float* row = mel.data() + (size_t)ch * Tm;
        double mean = 0; for (int t = 0; t < Tm; t++) mean += row[t];
        mean /= Tm;
        double var = 0; for (int t = 0; t < Tm; t++) var += (row[t] - mean) * (row[t] - mean);
        var /= Tm;
        float inv = (float)(1.0 / sqrt(var + 1e-5));
        for (int t = 0; t < Tm; t++) row[t] = (row[t] - (float)mean) * inv;
    }
    size_t need = 64 * 1024 * 1024;
    if (c.compute_meta.size() < need) c.compute_meta.resize(need);
    ggml_context* ctx0 = ggml_init({need, c.compute_meta.data(), true});
    ggml_cgraph* gf = build_spk_enc(ctx0, c.tensors, Tm);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(c.backend_cpu));
    if (!ga || !ggml_gallocr_reserve(ga, gf)) { if (ga) ggml_gallocr_free(ga); ggml_free(ctx0); return {}; }
    if (!ggml_gallocr_alloc_graph(ga, gf)) { ggml_gallocr_free(ga); ggml_free(ctx0); return {}; }
    ggml_tensor* mel_in = ggml_graph_get_tensor(gf, "speaker_mel");
    ggml_backend_tensor_set(mel_in, mel.data(), 0, mel.size() * sizeof(float));
    if (ggml_backend_graph_compute(c.backend_cpu, gf) != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(ga); ggml_free(ctx0); return {};
    }
    ggml_tensor* out = ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1);
    std::vector<float> result(512);
    ggml_backend_tensor_get(out, result.data(), 0, 512 * sizeof(float));
    ggml_gallocr_free(ga); ggml_free(ctx0);
    return result;
}

} // namespace engine::models::xtts_v2

