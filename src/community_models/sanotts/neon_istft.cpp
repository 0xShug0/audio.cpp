#include "engine/community_models/sanotts/neon_istft.h"

#include <stdexcept>

#if defined(__aarch64__) && defined(__ARM_NEON)

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>

namespace engine::models::sanotts {
namespace {

constexpr int kFft = 1024;
constexpr int kHop = 256;
constexpr int kBins = kFft / 2 + 1;       // 513
constexpr int kOutDim = 2 * kBins;        // 1026
constexpr double kPi = 3.141592653589793238462643383279502884;

// ---- float32 exp / sincos, about 1 ulp --------------------------------------

inline float32x4_t exp4(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0f)), vdupq_n_f32(88.0f));
    float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.44269504088896341f)));
    float32x4_t q = vfmsq_f32(x, n, vdupq_n_f32(0.693145751953125f));
    q = vfmsq_f32(q, n, vdupq_n_f32(1.428606765330187045e-06f));
    float32x4_t p = vdupq_n_f32(1.9875691500e-4f);
    p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), p, q);
    p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), p, q);
    p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), p, q);
    p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), p, q);
    p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), p, q);
    p = vfmaq_f32(vaddq_f32(q, vdupq_n_f32(1.0f)), vmulq_f32(p, q), q);
    int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(e));
}

inline void sincos4(float32x4_t x, float32x4_t * s, float32x4_t * c) {
    float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(0.63661977236758134f)));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(1.5703125f));
    r = vfmsq_f32(r, n, vdupq_n_f32(4.837512969970703125e-4f));
    r = vfmsq_f32(r, n, vdupq_n_f32(7.54978995489188216e-8f));
    float32x4_t r2 = vmulq_f32(r, r);
    float32x4_t ps = vfmaq_f32(vdupq_n_f32(8.3321608736e-3f), r2, vdupq_n_f32(-1.9515295891e-4f));
    ps = vfmaq_f32(vdupq_n_f32(-1.6666654611e-1f), r2, ps);
    float32x4_t sn = vfmaq_f32(r, vmulq_f32(r2, r), ps);
    float32x4_t pc = vfmaq_f32(vdupq_n_f32(-1.388731625493765e-3f), r2, vdupq_n_f32(2.443315711809948e-5f));
    pc = vfmaq_f32(vdupq_n_f32(4.166664568298827e-2f), r2, pc);
    float32x4_t cs = vfmaq_f32(vfmsq_f32(vdupq_n_f32(1.0f), r2, vdupq_n_f32(0.5f)), vmulq_f32(r2, r2), pc);
    uint32x4_t q = vreinterpretq_u32_s32(vcvtq_s32_f32(n));
    uint32x4_t swap = vtstq_u32(q, vdupq_n_u32(1));
    float32x4_t S = vbslq_f32(swap, cs, sn), C = vbslq_f32(swap, sn, cs);
    *s = vbslq_f32(vtstq_u32(q, vdupq_n_u32(2)), vnegq_f32(S), S);
    *c = vbslq_f32(vtstq_u32(vaddq_u32(q, vdupq_n_u32(1)), vdupq_n_u32(2)), vnegq_f32(C), C);
}

// ---- 1024-point real inverse FFT, four frames at once (lane = frame) ----------

struct Tables {
    float c2r_cos[256], c2r_sin[256];
    uint16_t rev888[512];
    float tw_s1[8][14];
    float tw_s2[64][14];
    Tables() {
        for (int k = 0; k < 256; ++k) {
            const double a = 2.0 * kPi * k / 1024.0;
            c2r_cos[k] = static_cast<float>(std::cos(a));
            c2r_sin[k] = static_cast<float>(std::sin(a));
        }
        for (int a = 0; a < 8; ++a)
            for (int b = 0; b < 8; ++b)
                for (int c = 0; c < 8; ++c)
                    rev888[(c * 8 + b) * 8 + a] = static_cast<uint16_t>((a * 8 + b) * 8 + c);
        for (int j = 0; j < 8; ++j)
            for (int m = 1; m <= 7; ++m) {
                const double ang = 2.0 * kPi * (j * m) / 64.0;
                tw_s1[j][2 * (m - 1)] = static_cast<float>(std::cos(ang));
                tw_s1[j][2 * (m - 1) + 1] = static_cast<float>(std::sin(ang));
            }
        for (int j = 0; j < 64; ++j)
            for (int m = 1; m <= 7; ++m) {
                const double ang = 2.0 * kPi * (j * m) / 512.0;
                tw_s2[j][2 * (m - 1)] = static_cast<float>(std::cos(ang));
                tw_s2[j][2 * (m - 1) + 1] = static_cast<float>(std::sin(ang));
            }
    }
};

const Tables & tables() {
    static const Tables t;
    return t;
}

inline void rad8_inv(
    float32x4_t x0r, float32x4_t x0i, float32x4_t x1r, float32x4_t x1i,
    float32x4_t x2r, float32x4_t x2i, float32x4_t x3r, float32x4_t x3i,
    float32x4_t x4r, float32x4_t x4i, float32x4_t x5r, float32x4_t x5i,
    float32x4_t x6r, float32x4_t x6i, float32x4_t x7r, float32x4_t x7i,
    float32x4_t * yr, float32x4_t * yi, int sp) {
    const float32x4_t kS = vdupq_n_f32(0.7071067811865475244f);
    float32x4_t e0r = vaddq_f32(x0r, x4r), e0i = vaddq_f32(x0i, x4i);
    float32x4_t e1r = vsubq_f32(x0r, x4r), e1i = vsubq_f32(x0i, x4i);
    float32x4_t e2r = vaddq_f32(x2r, x6r), e2i = vaddq_f32(x2i, x6i);
    float32x4_t e3r = vsubq_f32(x2r, x6r), e3i = vsubq_f32(x2i, x6i);
    float32x4_t E0r = vaddq_f32(e0r, e2r), E0i = vaddq_f32(e0i, e2i);
    float32x4_t E2r = vsubq_f32(e0r, e2r), E2i = vsubq_f32(e0i, e2i);
    float32x4_t E1r = vsubq_f32(e1r, e3i), E1i = vaddq_f32(e1i, e3r);
    float32x4_t E3r = vaddq_f32(e1r, e3i), E3i = vsubq_f32(e1i, e3r);

    float32x4_t o0r = vaddq_f32(x1r, x5r), o0i = vaddq_f32(x1i, x5i);
    float32x4_t o1r = vsubq_f32(x1r, x5r), o1i = vsubq_f32(x1i, x5i);
    float32x4_t o2r = vaddq_f32(x3r, x7r), o2i = vaddq_f32(x3i, x7i);
    float32x4_t o3r = vsubq_f32(x3r, x7r), o3i = vsubq_f32(x3i, x7i);
    float32x4_t O0r = vaddq_f32(o0r, o2r), O0i = vaddq_f32(o0i, o2i);
    float32x4_t O2r = vsubq_f32(o0r, o2r), O2i = vsubq_f32(o0i, o2i);
    float32x4_t O1r = vsubq_f32(o1r, o3i), O1i = vaddq_f32(o1i, o3r);
    float32x4_t O3r = vaddq_f32(o1r, o3i), O3i = vsubq_f32(o1i, o3r);

    yr[0 * sp] = vaddq_f32(E0r, O0r); yi[0 * sp] = vaddq_f32(E0i, O0i);
    yr[4 * sp] = vsubq_f32(E0r, O0r); yi[4 * sp] = vsubq_f32(E0i, O0i);
    float32x4_t u1r = vmulq_f32(vsubq_f32(O1r, O1i), kS), u1i = vmulq_f32(vaddq_f32(O1r, O1i), kS);
    yr[1 * sp] = vaddq_f32(E1r, u1r); yi[1 * sp] = vaddq_f32(E1i, u1i);
    yr[5 * sp] = vsubq_f32(E1r, u1r); yi[5 * sp] = vsubq_f32(E1i, u1i);
    yr[2 * sp] = vsubq_f32(E2r, O2i); yi[2 * sp] = vaddq_f32(E2i, O2r);
    yr[6 * sp] = vaddq_f32(E2r, O2i); yi[6 * sp] = vsubq_f32(E2i, O2r);
    float32x4_t u3r = vmulq_f32(vaddq_f32(O3r, O3i), kS), u3i = vmulq_f32(vsubq_f32(O3r, O3i), kS);
    yr[3 * sp] = vsubq_f32(E3r, u3r); yi[3 * sp] = vaddq_f32(E3i, u3i);
    yr[7 * sp] = vaddq_f32(E3r, u3r); yi[7 * sp] = vsubq_f32(E3i, u3i);
}

void ifft512_888(const Tables & tb, float32x4_t yr[512], float32x4_t yi[512]) {
#define SANO_CM(idx, k)                                                                  \
    const float32x4_t c##k = vdupq_n_f32(tw[2 * (k - 1)]);                               \
    const float32x4_t s##k = vdupq_n_f32(tw[2 * (k - 1) + 1]);                           \
    const float32x4_t a##k##r = yr[idx], a##k##i = yi[idx];                              \
    const float32x4_t x##k##r = vfmsq_f32(vmulq_f32(a##k##r, c##k), a##k##i, s##k);      \
    const float32x4_t x##k##i = vfmaq_f32(vmulq_f32(a##k##i, c##k), a##k##r, s##k);
    for (int b = 0; b < 512; b += 8) {
        rad8_inv(yr[b + 0], yi[b + 0], yr[b + 1], yi[b + 1], yr[b + 2], yi[b + 2], yr[b + 3], yi[b + 3],
                 yr[b + 4], yi[b + 4], yr[b + 5], yi[b + 5], yr[b + 6], yi[b + 6], yr[b + 7], yi[b + 7],
                 yr + b, yi + b, 1);
    }
    for (int b = 0; b < 512; b += 64) {
        rad8_inv(yr[b + 0], yi[b + 0], yr[b + 8], yi[b + 8], yr[b + 16], yi[b + 16], yr[b + 24], yi[b + 24],
                 yr[b + 32], yi[b + 32], yr[b + 40], yi[b + 40], yr[b + 48], yi[b + 48], yr[b + 56], yi[b + 56],
                 yr + b, yi + b, 8);
        for (int j = 1; j < 8; ++j) {
            const float * tw = tb.tw_s1[j];
            const int i = b + j;
            SANO_CM(i + 8, 1) SANO_CM(i + 16, 2) SANO_CM(i + 24, 3) SANO_CM(i + 32, 4)
            SANO_CM(i + 40, 5) SANO_CM(i + 48, 6) SANO_CM(i + 56, 7)
            rad8_inv(yr[i], yi[i], x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
                     yr + i, yi + i, 8);
        }
    }
    rad8_inv(yr[0], yi[0], yr[64], yi[64], yr[128], yi[128], yr[192], yi[192],
             yr[256], yi[256], yr[320], yi[320], yr[384], yi[384], yr[448], yi[448], yr, yi, 64);
    for (int j = 1; j < 64; ++j) {
        const float * tw = tb.tw_s2[j];
        SANO_CM(j + 64, 1) SANO_CM(j + 128, 2) SANO_CM(j + 192, 3) SANO_CM(j + 256, 4)
        SANO_CM(j + 320, 5) SANO_CM(j + 384, 6) SANO_CM(j + 448, 7)
        rad8_inv(yr[j], yi[j], x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
                 yr + j, yi + j, 64);
    }
#undef SANO_CM
}

/** zr/zi: bins 0..512 for four frames (lane = frame); bins 0 and 512 must be zero.
 *  Writes nf (1..4) frames of 1024 samples, already scaled by 1/1024. */
void c2r1024_soa4(const Tables & tb, const float32x4_t zr[513], const float32x4_t zi[513],
                  float * framed_out, int nf) {
    float32x4_t yr[512], yi[512];
    const float32x4_t kScale = vdupq_n_f32(0.5f / 512.0f);
    yr[0] = vdupq_n_f32(0.0f);
    yi[0] = vdupq_n_f32(0.0f);
    const float32x4_t k2 = vdupq_n_f32(1.0f / 512.0f);
    yr[tb.rev888[256]] = vmulq_f32(zr[256], k2);
    yi[tb.rev888[256]] = vnegq_f32(vmulq_f32(zi[256], k2));
    for (int k = 1; k < 256; ++k) {
        const int mk = 512 - k;
        const float32x4_t zkr = zr[k], zki = zi[k], zmr = zr[mk], zmi = zi[mk];
        const float32x4_t er = vaddq_f32(zkr, zmr), ei = vsubq_f32(zki, zmi);
        const float32x4_t dr = vsubq_f32(zkr, zmr), di = vaddq_f32(zki, zmi);
        const float32x4_t c = vdupq_n_f32(tb.c2r_cos[k]), s = vdupq_n_f32(tb.c2r_sin[k]);
        const float32x4_t xor_ = vfmsq_f32(vmulq_f32(dr, c), di, s);
        const float32x4_t xoi_ = vfmaq_f32(vmulq_f32(dr, s), di, c);
        const int d1 = tb.rev888[k], d2 = tb.rev888[mk];
        yr[d1] = vmulq_f32(vsubq_f32(er, xoi_), kScale);
        yi[d1] = vmulq_f32(vaddq_f32(ei, xor_), kScale);
        yr[d2] = vmulq_f32(vaddq_f32(er, xoi_), kScale);
        yi[d2] = vmulq_f32(vsubq_f32(xor_, ei), kScale);
    }
    ifft512_888(tb, yr, yi);
    float tmp[4][4];
    for (int m = 0; m < 512; m += 2) {
        const float32x4_t v0 = yr[m], v1 = yi[m], v2 = yr[m + 1], v3 = yi[m + 1];
        const float32x4_t t00 = vtrn1q_f32(v0, v1), t01 = vtrn2q_f32(v0, v1);
        const float32x4_t t10 = vtrn1q_f32(v2, v3), t11 = vtrn2q_f32(v2, v3);
        const float32x4_t f0 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t00), vreinterpretq_f64_f32(t10)));
        const float32x4_t f1 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t01), vreinterpretq_f64_f32(t11)));
        const float32x4_t f2 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t00), vreinterpretq_f64_f32(t10)));
        const float32x4_t f3 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t01), vreinterpretq_f64_f32(t11)));
        if (nf == 4) {
            vst1q_f32(framed_out + 0 * kFft + 2 * m, f0);
            vst1q_f32(framed_out + 1 * kFft + 2 * m, f1);
            vst1q_f32(framed_out + 2 * kFft + 2 * m, f2);
            vst1q_f32(framed_out + 3 * kFft + 2 * m, f3);
        } else {
            vst1q_f32(tmp[0], f0);
            vst1q_f32(tmp[1], f1);
            vst1q_f32(tmp[2], f2);
            for (int f = 0; f < nf; ++f) vst1q_f32(framed_out + f * kFft + 2 * m, vld1q_f32(tmp[f]));
        }
    }
}

/** Log magnitude / phase rows of up to four frames -> SoA4 complex spectrum. */
void spectrum_block(const float * spectrum, int64_t t0, int nf, float32x4_t zr[513], float32x4_t zi[513]) {
    alignas(16) float re[4][512];
    alignas(16) float im[4][512];
    const float32x4_t k100 = vdupq_n_f32(100.0f);
    for (int q = 0; q < 4; ++q) {
        if (q < nf) {
            const float * row = spectrum + static_cast<size_t>(t0 + q) * kOutDim;
            for (int fq = 0; fq < 512; fq += 4) {
                const float32x4_t mag = vminq_f32(exp4(vld1q_f32(row + fq)), k100);
                float32x4_t s, c;
                sincos4(vld1q_f32(row + kBins + fq), &s, &c);
                vst1q_f32(re[q] + fq, vmulq_f32(mag, c));
                vst1q_f32(im[q] + fq, vmulq_f32(mag, s));
            }
            re[q][0] = 0.0f;
            im[q][0] = 0.0f;
        } else {
            std::memset(re[q], 0, sizeof re[q]);
            std::memset(im[q], 0, sizeof im[q]);
        }
    }
    for (int fq = 0; fq < 512; fq += 4) {
        for (int pass = 0; pass < 2; ++pass) {
            const float (*src)[512] = pass == 0 ? re : im;
            float32x4_t * dst = pass == 0 ? zr : zi;
            const float32x4_t r0 = vld1q_f32(src[0] + fq), r1 = vld1q_f32(src[1] + fq);
            const float32x4_t r2 = vld1q_f32(src[2] + fq), r3 = vld1q_f32(src[3] + fq);
            const float32x4_t t0v = vtrn1q_f32(r0, r1), t1v = vtrn2q_f32(r0, r1);
            const float32x4_t t2v = vtrn1q_f32(r2, r3), t3v = vtrn2q_f32(r2, r3);
            dst[fq + 0] = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t0v), vreinterpretq_f64_f32(t2v)));
            dst[fq + 1] = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t1v), vreinterpretq_f64_f32(t3v)));
            dst[fq + 2] = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t0v), vreinterpretq_f64_f32(t2v)));
            dst[fq + 3] = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t1v), vreinterpretq_f64_f32(t3v)));
        }
    }
    zr[512] = vdupq_n_f32(0.0f);
    zi[512] = vdupq_n_f32(0.0f);
}

template <typename F>
void run_parallel(int nt, F && fn) {
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(std::max(0, nt - 1)));
    for (int t = 1; t < nt; ++t) pool.emplace_back(fn, t);
    fn(0);
    for (auto & th : pool) th.join();
}

}  // namespace

struct SanoTtsNeonIstft::Impl {
    std::unique_ptr<float[]> framed;   // frames x 1024, reused between calls
    size_t capacity = 0;
};

bool SanoTtsNeonIstft::available(int64_t n_fft, int64_t hop_length) {
    return n_fft == kFft && hop_length == kHop;
}

SanoTtsNeonIstft::SanoTtsNeonIstft() : impl_(std::make_unique<Impl>()) {}
SanoTtsNeonIstft::~SanoTtsNeonIstft() = default;

std::vector<float> SanoTtsNeonIstft::compute(
    const std::vector<float> & spectrum,
    int64_t frames,
    const std::vector<float> & window,
    int threads) {
    if (frames < 2 || static_cast<int64_t>(spectrum.size()) != frames * kOutDim ||
        static_cast<int64_t>(window.size()) != kFft) {
        throw std::runtime_error("sanoTTS NEON iSTFT: invalid input sizes");
    }
    const Tables & tb = tables();
    auto & s = *impl_;
    const size_t need = static_cast<size_t>(frames) * kFft;
    if (s.capacity < need) {
        s.framed.reset(new float[need]);   // intentionally uninitialised
        s.capacity = need;
    }
    float * framed = s.framed.get();
    const float * w = window.data();
    const int64_t blocks = (frames + 3) / 4;
    const int nt = (frames >= 24) ? std::clamp(threads, 1, 8) : 1;

    // 1. spectrum -> four frames of inverse FFT at a time
    run_parallel(nt, [&](int tid) {
        const int64_t lo = blocks * tid / nt, hi = blocks * (tid + 1) / nt;
        float32x4_t zr[513], zi[513];
        for (int64_t b = lo; b < hi; ++b) {
            const int64_t t0 = b * 4;
            const int nf = static_cast<int>(std::min<int64_t>(4, frames - t0));
            spectrum_block(spectrum.data(), t0, nf, zr, zi);
            c2r1024_soa4(tb, zr, zi, framed + static_cast<size_t>(t0) * kFft, nf);
        }
    });

    // 2. window, overlap-add and divide by the window envelope, per 256-sample output block.
    //    Output sample s sits at overlap-add position s + n_fft / 2; block b uses frames b-1 .. b+2.
    float inv_env[kHop];
    for (int j = 0; j < kHop; ++j) {
        inv_env[j] = 1.0f / (w[j] * w[j] + w[j + 256] * w[j + 256] + w[j + 512] * w[j + 512] + w[j + 768] * w[j + 768]);
    }
    const int64_t out_blocks = frames - 1;
    std::vector<float> out(static_cast<size_t>(out_blocks) * kHop);
    run_parallel(nt, [&](int tid) {
        const int64_t lo = out_blocks * tid / nt, hi = out_blocks * (tid + 1) / nt;
        for (int64_t b = lo; b < hi; ++b) {
            float * dst = out.data() + static_cast<size_t>(b) * kHop;
            if (b >= 1 && b + 2 <= frames - 1) {
                const float * f0 = framed + static_cast<size_t>(b - 1) * kFft + 768;
                const float * f1 = framed + static_cast<size_t>(b) * kFft + 512;
                const float * f2 = framed + static_cast<size_t>(b + 1) * kFft + 256;
                const float * f3 = framed + static_cast<size_t>(b + 2) * kFft;
                for (int j = 0; j < kHop; j += 4) {
                    float32x4_t acc = vmulq_f32(vld1q_f32(f0 + j), vld1q_f32(w + 768 + j));
                    acc = vfmaq_f32(acc, vld1q_f32(f1 + j), vld1q_f32(w + 512 + j));
                    acc = vfmaq_f32(acc, vld1q_f32(f2 + j), vld1q_f32(w + 256 + j));
                    acc = vfmaq_f32(acc, vld1q_f32(f3 + j), vld1q_f32(w + j));
                    vst1q_f32(dst + j, vmulq_f32(acc, vld1q_f32(inv_env + j)));
                }
            } else {
                const int64_t flo = std::max<int64_t>(0, b - 1), fhi = std::min<int64_t>(frames - 1, b + 2);
                for (int j = 0; j < kHop; ++j) {
                    float acc = 0.0f, env = 0.0f;
                    for (int64_t f = flo; f <= fhi; ++f) {
                        const int64_t o = (b + 2 - f) * kHop + j;
                        acc += w[o] * framed[static_cast<size_t>(f) * kFft + o];
                        env += w[o] * w[o];
                    }
                    dst[j] = acc / env;
                }
            }
        }
    });
    return out;
}

}  // namespace engine::models::sanotts

#else  // no AArch64 NEON

namespace engine::models::sanotts {

struct SanoTtsNeonIstft::Impl {};

bool SanoTtsNeonIstft::available(int64_t, int64_t) { return false; }
SanoTtsNeonIstft::SanoTtsNeonIstft() = default;
SanoTtsNeonIstft::~SanoTtsNeonIstft() = default;

std::vector<float> SanoTtsNeonIstft::compute(const std::vector<float> &, int64_t,
                                             const std::vector<float> &, int) {
    throw std::runtime_error("sanoTTS NEON iSTFT is not available in this build");
}

}  // namespace engine::models::sanotts

#endif
