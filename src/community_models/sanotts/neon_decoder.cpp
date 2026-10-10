#include "neon_decoder.h"

#include <stdexcept>

#if defined(__aarch64__) && defined(__ARM_NEON)

#include <arm_neon.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <thread>

namespace engine::models::sanotts {
namespace {

// Weights repacked once so the inner loops read one sequential stream.
//   wt: [co4 / 4][cin * K][4]    (K > 1 only: one 4-channel block after another)
//   wb: [co4 / 4][cin][4]        (K == 1 only: one 4-channel block after another)
struct PackedConv {
    int cin = 0;
    int cout = 0;
    int kernel = 0;
    int co4 = 0;
    std::vector<float> wt;
    std::vector<float> wb;
    std::vector<float> bias;
};

PackedConv pack_conv(const float * w, const float * b, int cin, int cout, int kernel) {
    PackedConv c;
    c.cin = cin;
    c.cout = cout;
    c.kernel = kernel;
    c.co4 = (cout + 3) & ~3;
    c.bias.assign(static_cast<size_t>(c.co4), 0.0F);
    std::copy(b, b + cout, c.bias.begin());
    if (kernel == 1) {
        // written in order, reading 4 weight rows at a time (padding channels stay 0)
        c.wb.assign(static_cast<size_t>(c.co4) * cin, 0.0F);
        float * dst = c.wb.data();
        for (int ob = 0; ob < c.co4 / 4; ++ob) {
            for (int ci = 0; ci < cin; ++ci, dst += 4) {
                for (int j = 0; j < 4 && ob * 4 + j < cout; ++j) {
                    dst[j] = w[static_cast<size_t>(ob * 4 + j) * cin + ci];
                }
            }
        }
        return c;
    }
    // K > 1: [co4 / 4][cin * K][4], so one output block's weights are read as one sequential stream
    const int rows = cin * kernel;
    c.wt.assign(static_cast<size_t>(rows) * c.co4, 0.0F);
    float * dst = c.wt.data();
    for (int ob = 0; ob < c.co4 / 4; ++ob) {
        for (int r = 0; r < rows; ++r, dst += 4) {
            for (int j = 0; j < 4 && ob * 4 + j < cout; ++j) {
                dst[j] = w[static_cast<size_t>(ob * 4 + j) * rows + r];
            }
        }
    }
    return c;
}

/** Reusable spin barrier (C++17 has no std::barrier). A decode crosses ~40-70 of these, each
 *  a few microseconds apart, so sleeping waiters (futex wake-ups) cost more than the work in
 *  between: measured on a Pixel 9 Linux VM, a mutex/condvar barrier made 2 threads slower than
 *  1. Waiters spin briefly, then yield so oversubscribed hosts still make progress. */
class Barrier {
public:
    explicit Barrier(int count) : count_(count), left_(count) {}
    void wait() {
        const int generation = generation_.load(std::memory_order_acquire);
        if (left_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            left_.store(count_, std::memory_order_relaxed);
            generation_.fetch_add(1, std::memory_order_release);
            return;
        }
        for (int spin = 0; generation_.load(std::memory_order_acquire) == generation; ++spin) {
            if (spin < 4096) {
                __asm__ __volatile__("yield");   // AArch64 spin-loop hint
            } else {
                std::this_thread::yield();
            }
        }
    }

private:
    const int count_;
    std::atomic<int> left_;
    std::atomic<int> generation_{0};
};

void split_range(int n, int tid, int nt, int & lo, int & hi) {
    lo = static_cast<int>(static_cast<int64_t>(n) * tid / nt);
    hi = static_cast<int>(static_cast<int64_t>(n) * (tid + 1) / nt);
}

// 4 output channels x 16 frames per register block.
#define SANOTTS_FMA16(xq0, xq1, xq2, xq3, wv)                                                   \
    acc[0][0] = vfmaq_laneq_f32(acc[0][0], xq0, wv, 0); acc[0][1] = vfmaq_laneq_f32(acc[0][1], xq1, wv, 0); \
    acc[0][2] = vfmaq_laneq_f32(acc[0][2], xq2, wv, 0); acc[0][3] = vfmaq_laneq_f32(acc[0][3], xq3, wv, 0); \
    acc[1][0] = vfmaq_laneq_f32(acc[1][0], xq0, wv, 1); acc[1][1] = vfmaq_laneq_f32(acc[1][1], xq1, wv, 1); \
    acc[1][2] = vfmaq_laneq_f32(acc[1][2], xq2, wv, 1); acc[1][3] = vfmaq_laneq_f32(acc[1][3], xq3, wv, 1); \
    acc[2][0] = vfmaq_laneq_f32(acc[2][0], xq0, wv, 2); acc[2][1] = vfmaq_laneq_f32(acc[2][1], xq1, wv, 2); \
    acc[2][2] = vfmaq_laneq_f32(acc[2][2], xq2, wv, 2); acc[2][3] = vfmaq_laneq_f32(acc[2][3], xq3, wv, 2); \
    acc[3][0] = vfmaq_laneq_f32(acc[3][0], xq0, wv, 3); acc[3][1] = vfmaq_laneq_f32(acc[3][1], xq1, wv, 3); \
    acc[3][2] = vfmaq_laneq_f32(acc[3][2], xq2, wv, 3); acc[3][3] = vfmaq_laneq_f32(acc[3][3], xq3, wv, 3);

// Writes one 4-channel x 16-frame result block: channel-major (out[o * T + t]) or, with
// rows_ld > 0, frame rows (out[t * rows_ld + o]). Only the first m frames / no channels are real.
inline void store_block(const float32x4_t (&acc)[4][4], float * out, int rows_ld, int T, int t0, int m,
                        int o0, int no) {
    float tmp[4][16];
    for (int j = 0; j < 4; ++j) {
        for (int q = 0; q < 4; ++q) {
            vst1q_f32(tmp[j] + 4 * q, acc[j][q]);
        }
    }
    if (rows_ld > 0) {
        for (int t = 0; t < m; ++t) {
            for (int j = 0; j < no; ++j) {
                out[static_cast<size_t>(t0 + t) * rows_ld + o0 + j] = tmp[j][t];
            }
        }
    } else {
        for (int j = 0; j < no; ++j) {
            std::memcpy(out + static_cast<size_t>(o0 + j) * T + t0, tmp[j], sizeof(float) * m);
        }
    }
}

inline float32x4_t exp4(float32x4_t x) {
    x = vmaxq_f32(x, vdupq_n_f32(-87.0F));
    const float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.44269504088896341F)));
    float32x4_t q = vfmsq_f32(x, n, vdupq_n_f32(0.693145751953125F));
    q = vfmsq_f32(q, n, vdupq_n_f32(1.428606765330187045e-06F));
    float32x4_t p = vdupq_n_f32(1.9875691500e-4F);
    p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3F), p, q);
    p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3F), p, q);
    p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2F), p, q);
    p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1F), p, q);
    p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1F), p, q);
    p = vfmaq_f32(vaddq_f32(q, vdupq_n_f32(1.0F)), vmulq_f32(p, q), q);
    const int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(e));
}

/** Exact-erf GELU (nn.GELU()), erf by Abramowitz-Stegun 7.1.26, |error| < 1.5e-7. */
void gelu_range(float * x, size_t lo, size_t hi) {
    size_t i = lo;
    for (; i + 4 <= hi; i += 4) {
        const float32x4_t v = vld1q_f32(x + i);
        const float32x4_t z = vmulq_f32(v, vdupq_n_f32(0.70710678118654752F));
        const float32x4_t az = vabsq_f32(z);
        const float32x4_t t = vdivq_f32(
            vdupq_n_f32(1.0F), vfmaq_f32(vdupq_n_f32(1.0F), az, vdupq_n_f32(0.3275911F)));
        float32x4_t p = vfmaq_f32(vdupq_n_f32(-1.453152027F), t, vdupq_n_f32(1.061405429F));
        p = vfmaq_f32(vdupq_n_f32(1.421413741F), p, t);
        p = vfmaq_f32(vdupq_n_f32(-0.284496736F), p, t);
        p = vfmaq_f32(vdupq_n_f32(0.254829592F), p, t);
        float32x4_t er = vfmsq_f32(
            vdupq_n_f32(1.0F), vmulq_f32(p, t), exp4(vnegq_f32(vmulq_f32(az, az))));
        er = vbslq_f32(vcltq_f32(z, vdupq_n_f32(0.0F)), vnegq_f32(er), er);
        vst1q_f32(x + i, vmulq_f32(vmulq_f32(vdupq_n_f32(0.5F), v), vaddq_f32(vdupq_n_f32(1.0F), er)));
    }
    for (; i < hi; ++i) {
        x[i] = 0.5F * x[i] * (1.0F + std::erf(x[i] * 0.70710678118654752F));
    }
}

}  // namespace

struct SanoTtsNeonDecoder::Impl {
    // dimensions from SanoTtsConfig
    int hidden = 0;        // acoustic_hidden
    int proj_in = 0;       // acoustic_hidden + 3 frame features
    int embed_in = 0;      // acoustic_hidden + 1 ("inside" row) + noise_channels
    int depth = 0;         // acoustic_depth
    int dim = 0;
    int pw_hidden = 0;
    int blocks = 0;
    int noise = 0;
    int dw_kernel = 0;
    int out_dim = 0;       // n_fft + 2

    PackedConv proj;
    std::vector<PackedConv> frame_conv0, frame_conv1;
    std::vector<std::vector<float>> frame_scale;
    PackedConv embed;      // acoustic.output folded into decoder.embed, plus noise_adapter
    std::vector<float> norm_w, norm_b, final_w, final_b;
    struct Block {
        std::vector<float> dw_w, dw_b, norm_w, norm_b, gamma;
        PackedConv pw1, pw2;
    };
    std::vector<Block> decoder;
    PackedConv head;

    // work buffers, kept between calls
    std::vector<float> in, f, h, d, xs, e, hid, mean, inv, xpk;

    // per call
    int frames = 0, frames16 = 0, padded = 0;

    void conv_part(const PackedConv & c, const float * src, float * out, int rows_ld,
                   int tid, int nt, Barrier & bar);
    void layer_norm_part(const float * src, const std::vector<float> & g, const std::vector<float> & b,
                         float * out, int tid, int nt, Barrier & bar);
    void worker(const float * context, const float * feats, const float * noise_in, float * out,
                int tid, int nt, Barrier & bar);
};

// This thread's share of conv c over input rows src (cin, T) -> out (cout, T), or, with
// rows_ld > 0, frame rows out[t * rows_ld + o]. Threads split the 16-frame blocks; ends with a
// barrier (K > 1 also waits once after packing its padded input rows).
void SanoTtsNeonDecoder::Impl::conv_part(const PackedConv & c, const float * src, float * out,
                                         int rows_ld, int tid, int nt, Barrier & bar) {
    const int T = frames;
    const int cin = c.cin;
    const int K = c.kernel;
    const int pad = K / 2;
    const int co4 = c.co4;
    float * xp = xpk.data();
    int lo = 0, hi = 0;
    if (K == 1) {
        // Threads split the FRAMES: each packs its own 16-frame blocks and computes every output
        // channel for a block while that block (cin x 16 floats) is still in L1. With the
        // channel split, every thread re-read the whole packed input (3.4 MB for heart's pw2 on a
        // 1500-frame chunk, more than the A72's 1 MB L2) once per channel block.
        split_range(frames16 / 16, tid, nt, lo, hi);
        for (int tb = lo; tb < hi; ++tb) {
            const int t0 = tb * 16;
            const int m = std::min(16, T - t0);
            float * blk = xp + static_cast<size_t>(tb) * cin * 16;
            for (int ci = 0; ci < cin; ++ci) {
                float * dst = blk + static_cast<size_t>(ci) * 16;
                std::memcpy(dst, src + static_cast<size_t>(ci) * T + t0, sizeof(float) * m);
                if (m < 16) {
                    std::memset(dst + m, 0, sizeof(float) * (16 - m));
                }
            }
            for (int ob = 0; ob < co4 / 4; ++ob) {
                const int o0 = ob * 4;
                const int no = std::min(4, c.cout - o0);
                float32x4_t acc[4][4];
                for (int j = 0; j < 4; ++j) {
                    const float32x4_t bj = vdupq_n_f32(c.bias[o0 + j]);
                    for (int q = 0; q < 4; ++q) {
                        acc[j][q] = bj;
                    }
                }
                const float * xq = blk;
                const float * wq = c.wb.data() + static_cast<size_t>(ob) * cin * 4;
                for (int ci = 0; ci < cin; ++ci, xq += 16, wq += 4) {
                    const float32x4_t wv = vld1q_f32(wq);
                    const float32x4_t x0 = vld1q_f32(xq), x1 = vld1q_f32(xq + 4);
                    const float32x4_t x2 = vld1q_f32(xq + 8), x3 = vld1q_f32(xq + 12);
                    SANOTTS_FMA16(x0, x1, x2, x3, wv)
                }
                store_block(acc, out, rows_ld, T, t0, m, o0, no);
            }
        }
        bar.wait();
        return;
    }
    // K > 1: every thread packs some zero-padded input rows (width `padded`), then all wait
    split_range(cin, tid, nt, lo, hi);
    for (int ci = lo; ci < hi; ++ci) {
        float * row = xp + static_cast<size_t>(ci) * padded;
        std::memset(row, 0, sizeof(float) * padded);
        std::memcpy(row + pad, src + static_cast<size_t>(ci) * T, sizeof(float) * T);
    }
    bar.wait();
    // 2. K > 1: threads split the 16-frame blocks; each computes every output-channel block for its
    //    frames, so the inputs it touches (cin rows x (16 + K - 1) floats) stay in L1.
    split_range(frames16 / 16, tid, nt, lo, hi);
    for (int tb = lo; tb < hi; ++tb) {
        const int t0 = tb * 16;
        const int m = std::min(16, T - t0);
        for (int ob = 0; ob < co4 / 4; ++ob) {
            const int o0 = ob * 4;
            const int no = std::min(4, c.cout - o0);
            float32x4_t acc[4][4];
            for (int j = 0; j < 4; ++j) {
                const float32x4_t bj = vdupq_n_f32(c.bias[o0 + j]);
                for (int q = 0; q < 4; ++q) {
                    acc[j][q] = bj;
                }
            }
            const float * wp = c.wt.data() + static_cast<size_t>(ob) * cin * K * 4;
            for (int ci = 0; ci < cin; ++ci) {
                const float * xc = xp + static_cast<size_t>(ci) * padded + t0;
                for (int kk = 0; kk < K; ++kk, wp += 4) {
                    const float32x4_t wv = vld1q_f32(wp);
                    const float32x4_t x0 = vld1q_f32(xc + kk), x1 = vld1q_f32(xc + kk + 4);
                    const float32x4_t x2 = vld1q_f32(xc + kk + 8), x3 = vld1q_f32(xc + kk + 12);
                    SANOTTS_FMA16(x0, x1, x2, x3, wv)
                }
            }
            store_block(acc, out, rows_ld, T, t0, m, o0, no);
        }
    }
    bar.wait();
}

// LayerNorm (eps 1e-6, as runtime.cpp) over the channels of this thread's frames.
void SanoTtsNeonDecoder::Impl::layer_norm_part(const float * src, const std::vector<float> & g,
                                               const std::vector<float> & b, float * out,
                                               int tid, int nt, Barrier & bar) {
    const int T = frames;
    const int C = dim;
    int lo = 0, hi = 0;
    split_range(T, tid, nt, lo, hi);
    float * m = mean.data();
    float * v = inv.data();
    for (int t = lo; t < hi; ++t) {
        m[t] = 0.0F;
        v[t] = 0.0F;
    }
    for (int c = 0; c < C; ++c) {
        for (int t = lo; t < hi; ++t) {
            m[t] += src[static_cast<size_t>(c) * T + t];
        }
    }
    for (int t = lo; t < hi; ++t) {
        m[t] /= static_cast<float>(C);
    }
    for (int c = 0; c < C; ++c) {
        for (int t = lo; t < hi; ++t) {
            const float dd = src[static_cast<size_t>(c) * T + t] - m[t];
            v[t] += dd * dd;
        }
    }
    for (int t = lo; t < hi; ++t) {
        v[t] = 1.0F / std::sqrt(v[t] / static_cast<float>(C) + 1.0e-6F);
    }
    for (int c = 0; c < C; ++c) {
        for (int t = lo; t < hi; ++t) {
            out[static_cast<size_t>(c) * T + t] =
                (src[static_cast<size_t>(c) * T + t] - m[t]) * v[t] * g[c] + b[c];
        }
    }
    bar.wait();
}

void SanoTtsNeonDecoder::Impl::worker(const float * context, const float * feats,
                                      const float * noise_in, float * out, int tid, int nt,
                                      Barrier & bar) {
    const int T = frames;
    const size_t TT = static_cast<size_t>(T);
    int lo = 0, hi = 0;
    // inputs: [context | feats] -> in; f rows hidden.. = 1 ("inside" row) then noise
    split_range(proj_in + 1 + noise, tid, nt, lo, hi);
    for (int row = lo; row < hi; ++row) {
        if (row < hidden) {
            std::memcpy(in.data() + row * TT, context + row * TT, sizeof(float) * TT);
        } else if (row < proj_in) {
            std::memcpy(in.data() + row * TT, feats + (row - hidden) * TT, sizeof(float) * TT);
        } else if (row == proj_in) {
            std::fill_n(f.data() + static_cast<size_t>(hidden) * TT, TT, 1.0F);
        } else {
            const int n = row - proj_in - 1;
            std::memcpy(f.data() + (hidden + 1 + n) * TT, noise_in + n * TT, sizeof(float) * TT);
        }
    }
    bar.wait();

    // frame-stage acoustic blocks: f = proj(in); f += scale * conv(silu(conv(f)))
    conv_part(proj, in.data(), f.data(), 0, tid, nt, bar);
    for (int i = 0; i < depth; ++i) {
        conv_part(frame_conv0[i], f.data(), h.data(), 0, tid, nt, bar);
        split_range(hidden * T, tid, nt, lo, hi);
        for (int k = lo; k < hi; ++k) {
            h[k] = h[k] / (1.0F + std::exp(-h[k]));
        }
        bar.wait();
        conv_part(frame_conv1[i], h.data(), d.data(), 0, tid, nt, bar);
        const auto & s = frame_scale[i];
        split_range(hidden, tid, nt, lo, hi);
        for (int c = lo; c < hi; ++c) {
            const float sc = s.size() == 1 ? s[0] : s[c];
            float * fc = f.data() + c * TT;
            const float * dc = d.data() + c * TT;
            for (int t = 0; t < T; ++t) {
                fc[t] += sc * dc[t];
            }
        }
        bar.wait();
    }

    // ConvNeXt decoder (channel-major here; LayerNorm runs over channels per frame)
    conv_part(embed, f.data(), d.data(), 0, tid, nt, bar);
    layer_norm_part(d.data(), norm_w, norm_b, xs.data(), tid, nt, bar);
    const int dpad = dw_kernel / 2;
    for (const auto & blk : decoder) {
        split_range(dim, tid, nt, lo, hi);
        for (int c = lo; c < hi; ++c) {        // depthwise conv
            const float * xc = xs.data() + c * TT;
            float * ec = e.data() + c * TT;
            std::fill_n(ec, TT, blk.dw_b[c]);
            for (int k = 0; k < dw_kernel; ++k) {
                const float wk = blk.dw_w[static_cast<size_t>(c) * dw_kernel + k];
                const int a0 = std::max(0, dpad - k);
                const int b0 = std::min(T, T + dpad - k);
                for (int t = a0; t < b0; ++t) {
                    ec[t] += wk * xc[t + k - dpad];
                }
            }
        }
        bar.wait();
        layer_norm_part(e.data(), blk.norm_w, blk.norm_b, d.data(), tid, nt, bar);
        conv_part(blk.pw1, d.data(), hid.data(), 0, tid, nt, bar);
        const int quads = pw_hidden * T / 4;
        split_range(quads, tid, nt, lo, hi);
        gelu_range(hid.data(), static_cast<size_t>(lo) * 4,
                   hi == quads ? static_cast<size_t>(pw_hidden) * TT : static_cast<size_t>(hi) * 4);
        bar.wait();
        conv_part(blk.pw2, hid.data(), e.data(), 0, tid, nt, bar);
        split_range(dim, tid, nt, lo, hi);
        for (int c = lo; c < hi; ++c) {
            const float g = blk.gamma[c];
            float * xc = xs.data() + c * TT;
            const float * ec = e.data() + c * TT;
            for (int t = 0; t < T; ++t) {
                xc[t] += g * ec[t];
            }
        }
        bar.wait();
    }
    layer_norm_part(xs.data(), final_w, final_b, e.data(), tid, nt, bar);
    conv_part(head, e.data(), out, out_dim, tid, nt, bar);
}

bool SanoTtsNeonDecoder::available() { return true; }

SanoTtsNeonDecoder::SanoTtsNeonDecoder(const SanoTtsConfig & config, const WeightReader & read,
                                       int threads)
    : impl_(std::make_unique<Impl>()) {
    auto & p = *impl_;
    p.hidden = static_cast<int>(config.acoustic_hidden);
    p.proj_in = p.hidden + 3;
    p.noise = static_cast<int>(config.noise_channels);
    p.embed_in = p.hidden + 1 + p.noise;
    p.depth = static_cast<int>(config.acoustic_depth);
    p.dim = static_cast<int>(config.dim);
    p.pw_hidden = static_cast<int>(config.pw_hidden);
    p.blocks = static_cast<int>(config.blocks);
    p.dw_kernel = static_cast<int>(config.dw_kernel);
    p.out_dim = static_cast<int>(config.n_fft + 2);
    const int mels = static_cast<int>(config.mels);
    const int ak = static_cast<int>(config.acoustic_kernel);
    const int ek = static_cast<int>(config.embed_kernel);
    if (p.hidden <= 0 || p.dim <= 0 || p.pw_hidden <= 0 || p.blocks <= 0 || mels <= 0 ||
        ak > 33 || ek > 33 || p.dw_kernel <= 0) {
        throw std::runtime_error("sanoTTS NEON decoder: unsupported config");
    }

    auto expect = [&](const std::string & name, size_t size) {
        auto v = read(name);
        if (v.size() != size) {
            throw std::runtime_error("sanoTTS NEON decoder: unexpected size for " + name);
        }
        return v;
    };

    {
        const auto w = expect("acoustic.frame_input_proj.weight", static_cast<size_t>(p.hidden) * p.proj_in);
        const auto b = expect("acoustic.frame_input_proj.bias", p.hidden);
        p.proj = pack_conv(w.data(), b.data(), p.proj_in, p.hidden, 1);
    }
    for (int i = 0; i < p.depth; ++i) {
        const std::string q = "acoustic.frame_blocks." + std::to_string(i);
        const size_t n = static_cast<size_t>(p.hidden) * p.hidden * ak;
        const auto w0 = expect(q + ".net.0.weight", n);
        const auto b0 = expect(q + ".net.0.bias", p.hidden);
        const auto w2 = expect(q + ".net.2.weight", n);
        const auto b2 = expect(q + ".net.2.bias", p.hidden);
        p.frame_conv0.push_back(pack_conv(w0.data(), b0.data(), p.hidden, p.hidden, ak));
        p.frame_conv1.push_back(pack_conv(w2.data(), b2.data(), p.hidden, p.hidden, ak));
        auto s = read(q + ".scale");
        if (s.size() != 1 && s.size() != static_cast<size_t>(p.hidden)) {
            throw std::runtime_error("sanoTTS NEON decoder: unexpected size for " + q + ".scale");
        }
        p.frame_scale.push_back(std::move(s));
    }

    // acoustic.output (1x1, hidden -> mels) has no nonlinearity before decoder.embed, so fold it
    // in: embed(output(f)) = embed'(f) with one extra input row of ones (zero-padded like f) for
    // output's bias. noise_adapter takes the last rows, its bias joins embed's.
    {
        const auto wo = expect("acoustic.output.weight", static_cast<size_t>(mels) * p.hidden);
        const auto bo = expect("acoustic.output.bias", mels);
        const auto we = expect("decoder.embed.weight", static_cast<size_t>(p.dim) * mels * ek);
        const auto be = expect("decoder.embed.bias", p.dim);
        const auto wn = expect("decoder.noise_adapter.weight", static_cast<size_t>(p.dim) * p.noise * ek);
        const auto bn = expect("decoder.noise_adapter.bias", p.dim);
        std::vector<float> w(static_cast<size_t>(p.dim) * p.embed_in * ek, 0.0F);
        std::vector<float> b(p.dim);
        const int h1 = p.hidden + 1;                  // folded rows + the bias row
        // Output channels are independent: split them across `threads` (same result for any count).
        auto fold_range = [&](int o_lo, int o_hi) {
            std::vector<double> acc(static_cast<size_t>(ek) * h1);
            for (int o = o_lo; o < o_hi; ++o) {
                // acc[k][c] = sum over m (in order) of embed[o][m][k] * output[m][c], c == hidden -> bias
                std::fill(acc.begin(), acc.end(), 0.0);
                for (int m = 0; m < mels; ++m) {
                    const float * wrow = wo.data() + static_cast<size_t>(m) * p.hidden;
                    for (int k = 0; k < ek; ++k) {
                        const double e = we[(static_cast<size_t>(o) * mels + m) * ek + k];
                        double * ak = acc.data() + static_cast<size_t>(k) * h1;
                        for (int c = 0; c < p.hidden; ++c) {
                            ak[c] += e * wrow[c];
                        }
                        ak[p.hidden] += e * bo[m];
                    }
                }
                for (int k = 0; k < ek; ++k) {
                    for (int c = 0; c < h1; ++c) {
                        w[(static_cast<size_t>(o) * p.embed_in + c) * ek + k] =
                            static_cast<float>(acc[static_cast<size_t>(k) * h1 + c]);
                    }
                    for (int c = 0; c < p.noise; ++c) {
                        w[(static_cast<size_t>(o) * p.embed_in + h1 + c) * ek + k] =
                            wn[(static_cast<size_t>(o) * p.noise + c) * ek + k];
                    }
                }
                b[o] = static_cast<float>(static_cast<double>(be[o]) + bn[o]);
            }
        };
        const int nt = std::clamp(threads, 1, std::max(1, p.dim));
        std::vector<std::thread> pool;
        for (int t = 1; t < nt; ++t) {
            pool.emplace_back(fold_range, p.dim * t / nt, p.dim * (t + 1) / nt);
        }
        fold_range(0, p.dim / nt);
        for (auto & t : pool) {
            t.join();
        }
        p.embed = pack_conv(w.data(), b.data(), p.embed_in, p.dim, ek);
    }
    p.norm_w = expect("decoder.norm.weight", p.dim);
    p.norm_b = expect("decoder.norm.bias", p.dim);
    for (int i = 0; i < p.blocks; ++i) {
        const std::string q = "decoder.blocks." + std::to_string(i);
        Impl::Block blk;
        blk.dw_w = expect(q + ".dwconv.weight", static_cast<size_t>(p.dim) * p.dw_kernel);
        blk.dw_b = expect(q + ".dwconv.bias", p.dim);
        blk.norm_w = expect(q + ".norm.weight", p.dim);
        blk.norm_b = expect(q + ".norm.bias", p.dim);
        const auto w1 = expect(q + ".pwconv1.weight", static_cast<size_t>(p.pw_hidden) * p.dim);
        const auto b1 = expect(q + ".pwconv1.bias", p.pw_hidden);
        const auto w2 = expect(q + ".pwconv2.weight", static_cast<size_t>(p.dim) * p.pw_hidden);
        const auto b2 = expect(q + ".pwconv2.bias", p.dim);
        blk.pw1 = pack_conv(w1.data(), b1.data(), p.dim, p.pw_hidden, 1);
        blk.pw2 = pack_conv(w2.data(), b2.data(), p.pw_hidden, p.dim, 1);
        blk.gamma = expect(q + ".gamma", p.dim);
        p.decoder.push_back(std::move(blk));
    }
    p.final_w = expect("decoder.final_norm.weight", p.dim);
    p.final_b = expect("decoder.final_norm.bias", p.dim);
    {
        const auto w = expect("decoder.head.weight", static_cast<size_t>(p.out_dim) * p.dim);
        const auto b = expect("decoder.head.bias", p.out_dim);
        p.head = pack_conv(w.data(), b.data(), p.dim, p.out_dim, 1);
    }
}

SanoTtsNeonDecoder::~SanoTtsNeonDecoder() = default;

std::vector<float> SanoTtsNeonDecoder::decode(const std::vector<float> & context,
                                              const std::vector<float> & feats,
                                              const std::vector<float> & noise,
                                              int64_t frames, int threads) {
    auto & p = *impl_;
    const auto T = static_cast<size_t>(frames);
    if (frames < 1 || frames > (1 << 24) || context.size() != p.hidden * T ||
        feats.size() != 3 * T || noise.size() != p.noise * T) {
        throw std::runtime_error("sanoTTS NEON decoder: invalid input sizes");
    }
    p.frames = static_cast<int>(frames);
    p.frames16 = (p.frames + 15) & ~15;
    p.padded = p.frames16 + 32;   // room for kernels up to 33 wide
    const size_t max_cin = static_cast<size_t>(
        std::max({p.proj_in, p.embed_in, p.hidden, p.dim, p.pw_hidden}));
    auto grow = [](std::vector<float> & v, size_t n) {
        if (v.size() < n) {
            v.resize(n);
        }
    };
    grow(p.in, static_cast<size_t>(p.proj_in) * T);
    grow(p.f, static_cast<size_t>(p.embed_in) * T);
    grow(p.h, static_cast<size_t>(p.hidden) * T);
    grow(p.d, static_cast<size_t>(std::max(p.hidden, p.dim)) * T);
    grow(p.xs, static_cast<size_t>(p.dim) * T);
    grow(p.e, static_cast<size_t>(p.dim) * T);
    grow(p.hid, static_cast<size_t>(p.pw_hidden) * T);
    grow(p.mean, T);
    grow(p.inv, T);
    grow(p.xpk, max_cin * static_cast<size_t>(p.padded));

    std::vector<float> out(T * static_cast<size_t>(p.out_dim));
    const int nt = std::clamp(threads, 1, 16);
    Barrier bar(nt);
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(nt - 1));
    for (int tid = 1; tid < nt; ++tid) {
        pool.emplace_back([&, tid] {
            p.worker(context.data(), feats.data(), noise.data(), out.data(), tid, nt, bar);
        });
    }
    p.worker(context.data(), feats.data(), noise.data(), out.data(), 0, nt, bar);
    for (auto & t : pool) {
        t.join();
    }
    return out;
}

}  // namespace engine::models::sanotts

#else  // no AArch64 NEON

namespace engine::models::sanotts {

struct SanoTtsNeonDecoder::Impl {};

bool SanoTtsNeonDecoder::available() { return false; }

SanoTtsNeonDecoder::SanoTtsNeonDecoder(const SanoTtsConfig &, const WeightReader &, int) {
    throw std::runtime_error(
        "sanoTTS cpu_decoder=neon needs an AArch64 build with NEON; use cpu_decoder=ggml");
}

SanoTtsNeonDecoder::~SanoTtsNeonDecoder() = default;

std::vector<float> SanoTtsNeonDecoder::decode(const std::vector<float> &,
                                              const std::vector<float> &,
                                              const std::vector<float> &, int64_t, int) {
    throw std::runtime_error("sanoTTS NEON decoder is not available in this build");
}

}  // namespace engine::models::sanotts

#endif
