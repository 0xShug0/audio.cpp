#include "engine/community_models/sanotts/neon_decoder.h"

#include <stdexcept>

#if defined(__aarch64__) && defined(__ARM_NEON)

#include <arm_neon.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>

namespace engine::models::sanotts {
namespace {

constexpr int kFft = 1024;
constexpr int kHop = 256;
constexpr float kDcBlockPole = 0.9973F;
constexpr double kPi = 3.141592653589793238462643383279502884;

// Weights packed once for 8-output-channel x 12-frame FP32 NEON microkernels:
//   wf32: [co8 / 8][cin * kernel][8]
//   bias: [co8]
struct PackedConv {
    int cin = 0;
    int cout = 0;
    int kernel = 0;
    int co8 = 0;
    std::vector<float> wf32;
    std::vector<float> bias;
};

PackedConv pack_conv_scaled(const float * w, const float * b, int cin, int cout, int kernel,
                            const float * scale_vec = nullptr, size_t scale_len = 0) {
    PackedConv c;
    c.cin = cin;
    c.cout = cout;
    c.kernel = kernel;
    c.co8 = (cout + 7) & ~7;
    const int rows = cin * kernel;
    c.bias.assign(static_cast<size_t>(c.co8), 0.0F);
    c.wf32.assign(static_cast<size_t>(rows) * c.co8, 0.0F);
    auto out_scale = [&](int o) -> float {
        if (scale_vec == nullptr || scale_len == 0) {
            return 1.0F;
        }
        return scale_len == 1 ? scale_vec[0] : scale_vec[o];
    };
    for (int o = 0; o < cout; ++o) {
        c.bias[o] = b[o] * out_scale(o);
    }
    float * dst = c.wf32.data();
    for (int ob = 0; ob < c.co8 / 8; ++ob) {
        for (int r = 0; r < rows; ++r, dst += 8) {
            for (int j = 0; j < 8 && ob * 8 + j < cout; ++j) {
                const int o = ob * 8 + j;
                dst[j] = w[static_cast<size_t>(o) * rows + r] * out_scale(o);
            }
        }
    }
    return c;
}

// Paired head weights for bins 1..511 (128 groups of 4 bins = 8 outputs: logmag[0..3], phase[0..3]).
// Bins 0 and 512 are always zeroed in sanoTTS and skipped completely.
struct HeadPaired {
    int cin = 0;
    int ng = 128;
    std::vector<float> wf32;  // [ng][cin][8]
    std::vector<float> bias;  // [ng][8]
};

HeadPaired pack_head_paired(const float * w, const float * b, int cin) {
    HeadPaired hp;
    hp.cin = cin;
    hp.ng = (511 + 3) / 4;  // 128
    hp.bias.assign(static_cast<size_t>(hp.ng) * 8, 0.0F);
    hp.wf32.assign(static_cast<size_t>(hp.ng) * cin * 8, 0.0F);
    for (int g = 0; g < hp.ng; ++g) {
        const int fq0 = 1 + g * 4;
        for (int j = 0; j < 4; ++j) {
            const int fq = fq0 + j;
            if (fq < 512) {
                hp.bias[g * 8 + j] = b[fq];
                hp.bias[g * 8 + 4 + j] = b[513 + fq];
                for (int ci = 0; ci < cin; ++ci) {
                    hp.wf32[(static_cast<size_t>(g) * cin + ci) * 8 + j] =
                        w[static_cast<size_t>(fq) * cin + ci];
                    hp.wf32[(static_cast<size_t>(g) * cin + ci) * 8 + 4 + j] =
                        w[static_cast<size_t>(513 + fq) * cin + ci];
                }
            }
        }
    }
    return hp;
}

/** Reusable spin barrier (C++17 has no std::barrier). Waiters spin briefly, then yield so
 *  oversubscribed hosts still make progress. */
class Barrier {
public:
    explicit Barrier(int count) : count_(count), left_(count) {}
    void wait() {
        if (count_ <= 1) {
            return;
        }
        const int generation = generation_.load(std::memory_order_acquire);
        if (left_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            left_.store(count_, std::memory_order_relaxed);
            generation_.fetch_add(1, std::memory_order_release);
            return;
        }
        for (int spin = 0; generation_.load(std::memory_order_acquire) == generation; ++spin) {
            if (spin < 4096) {
                __asm__ __volatile__("yield");
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

/** Persistent hybrid worker pool owned by one decoder instance: spins briefly between
 *  back-to-back calls, then sleeps on a condition variable (0% idle CPU). */
class WorkerPool {
public:
    WorkerPool() = default;
    ~WorkerPool() {
        if (workers_.empty()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_.store(true, std::memory_order_seq_cst);
            gen_.fetch_add(1, std::memory_order_seq_cst);
        }
        cv_.notify_all();
        for (auto & t : workers_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    WorkerPool(const WorkerPool &) = delete;
    WorkerPool & operator=(const WorkerPool &) = delete;

    template <typename Fn>
    void run(int nt, Fn && fn) {
        if (nt <= 1) {
            fn(0, 1);
            return;
        }
        ensure(nt);
        const int nw = static_cast<int>(workers_.size());
        fn_ = [&](int tid, int count) { fn(tid, count); };
        active_nt_.store(nt, std::memory_order_relaxed);
        done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_seq_cst);
        if (sleeping_.load(std::memory_order_seq_cst) > 0) {
            std::lock_guard<std::mutex> lock(mu_);
            cv_.notify_all();
        }
        fn(0, nt);
        for (int spin = 0; done_.load(std::memory_order_acquire) < nw; ++spin) {
            if (spin < 8192) {
                __asm__ __volatile__("yield");
            } else {
                std::this_thread::yield();
            }
        }
    }

private:
    void ensure(int nt) {
        while (static_cast<int>(workers_.size()) < nt - 1) {
            const int tid = static_cast<int>(workers_.size()) + 1;
            const int start_gen = gen_.load(std::memory_order_acquire);
            workers_.emplace_back([this, tid, start_gen] { worker_loop(tid, start_gen); });
        }
    }

    void worker_loop(int tid, int my_gen) {
        for (;;) {
            int g = gen_.load(std::memory_order_acquire);
            if (g == my_gen) {
                for (int spin = 0; spin < 2048; ++spin) {
                    g = gen_.load(std::memory_order_acquire);
                    if (g != my_gen) {
                        break;
                    }
                    __asm__ __volatile__("yield");
                }
                if (g == my_gen) {
                    sleeping_.fetch_add(1, std::memory_order_seq_cst);
                    g = gen_.load(std::memory_order_seq_cst);
                    if (g == my_gen) {
                        std::unique_lock<std::mutex> lock(mu_);
                        cv_.wait(lock, [&] {
                            return gen_.load(std::memory_order_acquire) != my_gen ||
                                   stop_.load(std::memory_order_acquire);
                        });
                        g = gen_.load(std::memory_order_acquire);
                    }
                    sleeping_.fetch_sub(1, std::memory_order_seq_cst);
                }
            }
            if (stop_.load(std::memory_order_acquire)) {
                return;
            }
            my_gen = g;
            const int nt = active_nt_.load(std::memory_order_acquire);
            if (tid < nt) {
                fn_(tid, nt);
            }
            done_.fetch_add(1, std::memory_order_release);
        }
    }

    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::thread> workers_;
    std::atomic<int> gen_{0};
    std::atomic<int> done_{0};
    std::atomic<int> active_nt_{1};
    std::atomic<int> sleeping_{0};
    std::atomic<bool> stop_{false};
    std::function<void(int, int)> fn_;
};

inline void split_range(int n, int tid, int nt, int & lo, int & hi) {
    lo = static_cast<int>(static_cast<int64_t>(n) * tid / nt);
    hi = static_cast<int>(static_cast<int64_t>(n) * (tid + 1) / nt);
}

// ---- Fast NEON Math Primitives (~1 ulp) -------------------------------------

inline float32x4_t exp4(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0F)), vdupq_n_f32(88.0F));
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

inline void sincos4(float32x4_t x, float32x4_t * s, float32x4_t * c) {
    const float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(0.63661977236758134F)));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(1.5703125F));
    r = vfmsq_f32(r, n, vdupq_n_f32(4.837512969970703125e-4F));
    r = vfmsq_f32(r, n, vdupq_n_f32(7.54978995489188216e-8F));
    const float32x4_t r2 = vmulq_f32(r, r);
    float32x4_t ps = vfmaq_f32(vdupq_n_f32(8.3321608736e-3F), r2, vdupq_n_f32(-1.9515295891e-4F));
    ps = vfmaq_f32(vdupq_n_f32(-1.6666654611e-1F), r2, ps);
    const float32x4_t sn = vfmaq_f32(r, vmulq_f32(r2, r), ps);
    float32x4_t pc = vfmaq_f32(vdupq_n_f32(-1.388731625493765e-3F), r2, vdupq_n_f32(2.443315711809948e-5F));
    pc = vfmaq_f32(vdupq_n_f32(4.166664568298827e-2F), r2, pc);
    const float32x4_t cs = vfmaq_f32(vfmsq_f32(vdupq_n_f32(1.0F), r2, vdupq_n_f32(0.5F)), vmulq_f32(r2, r2), pc);
    const uint32x4_t q = vreinterpretq_u32_s32(vcvtq_s32_f32(n));
    const uint32x4_t swap = vtstq_u32(q, vdupq_n_u32(1));
    const float32x4_t S = vbslq_f32(swap, cs, sn), C = vbslq_f32(swap, sn, cs);
    *s = vbslq_f32(vtstq_u32(q, vdupq_n_u32(2)), vnegq_f32(S), S);
    *c = vbslq_f32(vtstq_u32(vaddq_u32(q, vdupq_n_u32(1)), vdupq_n_u32(2)), vnegq_f32(C), C);
}

inline float32x4_t rcp4(float32x4_t x) {
    float32x4_t r = vrecpeq_f32(x);
    r = vmulq_f32(r, vrecpsq_f32(x, r));
    return vmulq_f32(r, vrecpsq_f32(x, r));
}

inline float32x4_t silu4(float32x4_t v) {
    return vmulq_f32(v, rcp4(vaddq_f32(vdupq_n_f32(1.0F), exp4(vnegq_f32(v)))));
}

/** Exact-erf GELU (nn.GELU()), erf by Abramowitz-Stegun 7.1.26, |error| < 1.5e-7. */
inline float32x4_t gelu4(float32x4_t v) {
    const float32x4_t z = vmulq_f32(v, vdupq_n_f32(0.70710678118654752F));
    const float32x4_t az = vabsq_f32(z);
    const float32x4_t k1 = vdupq_n_f32(1.0F);
    const float32x4_t t = rcp4(vfmaq_f32(k1, az, vdupq_n_f32(0.3275911F)));
    float32x4_t p = vfmaq_f32(vdupq_n_f32(-1.453152027F), t, vdupq_n_f32(1.061405429F));
    p = vfmaq_f32(vdupq_n_f32(1.421413741F), p, t);
    p = vfmaq_f32(vdupq_n_f32(-0.284496736F), p, t);
    p = vfmaq_f32(vdupq_n_f32(0.254829592F), p, t);
    float32x4_t er = vfmsq_f32(k1, vmulq_f32(p, t), exp4(vnegq_f32(vmulq_f32(az, az))));
    er = vbslq_f32(vcltq_f32(z, vdupq_n_f32(0.0F)), vnegq_f32(er), er);
    return vmulq_f32(vmulq_f32(vdupq_n_f32(0.5F), v), vaddq_f32(k1, er));
}

// ============================================================================
// Radix-8x8x8 1024-pt Real iFFT (4 frames at once) + Cached Hann Window Tables
// ============================================================================
struct IstftTables {
    float c2r_cos[256], c2r_sin[256];
    uint16_t rev888[512];
    float tw_s1[8][14];
    float tw_s2[64][14];
    float win[kFft];
    float win_s[kFft];  // win[o] * inv_env[o % 256] for interior 4-frame OLA

    IstftTables() {
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
        for (int i = 0; i < kFft; ++i) {
            win[i] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) / 1024.0));
        }
        for (int j = 0; j < kHop; ++j) {
            const float inv = 1.0F / (win[j] * win[j] + win[j + 256] * win[j + 256] +
                                      win[j + 512] * win[j + 512] + win[j + 768] * win[j + 768]);
            win_s[j] = win[j] * inv;
            win_s[j + 256] = win[j + 256] * inv;
            win_s[j + 512] = win[j + 512] * inv;
            win_s[j + 768] = win[j + 768] * inv;
        }
    }
};

const IstftTables & istft_tables() {
    static const IstftTables t;
    return t;
}

inline void rad8_inv(
    float32x4_t x0r, float32x4_t x0i, float32x4_t x1r, float32x4_t x1i,
    float32x4_t x2r, float32x4_t x2i, float32x4_t x3r, float32x4_t x3i,
    float32x4_t x4r, float32x4_t x4i, float32x4_t x5r, float32x4_t x5i,
    float32x4_t x6r, float32x4_t x6i, float32x4_t x7r, float32x4_t x7i,
    float32x4_t * yr, float32x4_t * yi, int sp) {
    const float32x4_t kS = vdupq_n_f32(0.7071067811865475244F);
    const float32x4_t e0r = vaddq_f32(x0r, x4r), e0i = vaddq_f32(x0i, x4i);
    const float32x4_t e1r = vsubq_f32(x0r, x4r), e1i = vsubq_f32(x0i, x4i);
    const float32x4_t e2r = vaddq_f32(x2r, x6r), e2i = vaddq_f32(x2i, x6i);
    const float32x4_t e3r = vsubq_f32(x2r, x6r), e3i = vsubq_f32(x2i, x6i);
    const float32x4_t E0r = vaddq_f32(e0r, e2r), E0i = vaddq_f32(e0i, e2i);
    const float32x4_t E2r = vsubq_f32(e0r, e2r), E2i = vsubq_f32(e0i, e2i);
    const float32x4_t E1r = vsubq_f32(e1r, e3i), E1i = vaddq_f32(e1i, e3r);
    const float32x4_t E3r = vaddq_f32(e1r, e3i), E3i = vsubq_f32(e1i, e3r);
    const float32x4_t o0r = vaddq_f32(x1r, x5r), o0i = vaddq_f32(x1i, x5i);
    const float32x4_t o1r = vsubq_f32(x1r, x5r), o1i = vsubq_f32(x1i, x5i);
    const float32x4_t o2r = vaddq_f32(x3r, x7r), o2i = vaddq_f32(x3i, x7i);
    const float32x4_t o3r = vsubq_f32(x3r, x7r), o3i = vsubq_f32(x3i, x7i);
    const float32x4_t O0r = vaddq_f32(o0r, o2r), O0i = vaddq_f32(o0i, o2i);
    const float32x4_t O2r = vsubq_f32(o0r, o2r), O2i = vsubq_f32(o0i, o2i);
    const float32x4_t O1r = vsubq_f32(o1r, o3i), O1i = vaddq_f32(o1i, o3r);
    const float32x4_t O3r = vaddq_f32(o1r, o3i), O3i = vsubq_f32(o1i, o3r);
    yr[0 * sp] = vaddq_f32(E0r, O0r); yi[0 * sp] = vaddq_f32(E0i, O0i);
    yr[4 * sp] = vsubq_f32(E0r, O0r); yi[4 * sp] = vsubq_f32(E0i, O0i);
    const float32x4_t u1r = vmulq_f32(vsubq_f32(O1r, O1i), kS), u1i = vmulq_f32(vaddq_f32(O1r, O1i), kS);
    yr[1 * sp] = vaddq_f32(E1r, u1r); yi[1 * sp] = vaddq_f32(E1i, u1i);
    yr[5 * sp] = vsubq_f32(E1r, u1r); yi[5 * sp] = vsubq_f32(E1i, u1i);
    yr[2 * sp] = vsubq_f32(E2r, O2i); yi[2 * sp] = vaddq_f32(E2i, O2r);
    yr[6 * sp] = vaddq_f32(E2r, O2i); yi[6 * sp] = vsubq_f32(E2i, O2r);
    const float32x4_t u3r = vmulq_f32(vaddq_f32(O3r, O3i), kS), u3i = vmulq_f32(vsubq_f32(O3r, O3i), kS);
    yr[3 * sp] = vsubq_f32(E3r, u3r); yi[3 * sp] = vaddq_f32(E3i, u3i);
    yr[7 * sp] = vaddq_f32(E3r, u3r); yi[7 * sp] = vsubq_f32(E3i, u3i);
}

void c2r1024_soa4(const IstftTables & tb, const float32x4_t zr[513], const float32x4_t zi[513],
                  float * framed_out, int nf) {
    float32x4_t yr[512], yi[512];
    const float32x4_t kScale = vdupq_n_f32(0.5F / 512.0F);
    yr[0] = vdupq_n_f32(0.0F); yi[0] = vdupq_n_f32(0.0F);
    const float32x4_t k2 = vdupq_n_f32(1.0F / 512.0F);
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
#define SANO_CM(idx, k)                                                                  \
    const float32x4_t c##k = vdupq_n_f32(tw[2 * (k - 1)]);                               \
    const float32x4_t s##k = vdupq_n_f32(tw[2 * (k - 1) + 1]);                           \
    const float32x4_t a##k##r = yr[idx], a##k##i = yi[idx];                              \
    const float32x4_t x##k##r = vfmsq_f32(vmulq_f32(a##k##r, c##k), a##k##i, s##k);      \
    const float32x4_t x##k##i = vfmaq_f32(vmulq_f32(a##k##i, c##k), a##k##r, s##k);
    for (int b = 0; b < 512; b += 8) {
        rad8_inv(yr[b+0],yi[b+0],yr[b+1],yi[b+1],yr[b+2],yi[b+2],yr[b+3],yi[b+3],
                 yr[b+4],yi[b+4],yr[b+5],yi[b+5],yr[b+6],yi[b+6],yr[b+7],yi[b+7], yr+b, yi+b, 1);
    }
    for (int b = 0; b < 512; b += 64) {
        rad8_inv(yr[b+0],yi[b+0],yr[b+8],yi[b+8],yr[b+16],yi[b+16],yr[b+24],yi[b+24],
                 yr[b+32],yi[b+32],yr[b+40],yi[b+40],yr[b+48],yi[b+48],yr[b+56],yi[b+56], yr+b, yi+b, 8);
        for (int j = 1; j < 8; ++j) {
            const float * tw = tb.tw_s1[j]; const int i = b + j;
            SANO_CM(i+8,1) SANO_CM(i+16,2) SANO_CM(i+24,3) SANO_CM(i+32,4) SANO_CM(i+40,5) SANO_CM(i+48,6) SANO_CM(i+56,7)
            rad8_inv(yr[i],yi[i],x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i, yr+i, yi+i, 8);
        }
    }
    rad8_inv(yr[0],yi[0],yr[64],yi[64],yr[128],yi[128],yr[192],yi[192],
             yr[256],yi[256],yr[320],yi[320],yr[384],yi[384],yr[448],yi[448], yr, yi, 64);
    for (int j = 1; j < 64; ++j) {
        const float * tw = tb.tw_s2[j];
        SANO_CM(j+64,1) SANO_CM(j+128,2) SANO_CM(j+192,3) SANO_CM(j+256,4) SANO_CM(j+320,5) SANO_CM(j+384,6) SANO_CM(j+448,7)
        rad8_inv(yr[j],yi[j],x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i, yr+j, yi+j, 64);
    }
#undef SANO_CM
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
            vst1q_f32(tmp[0], f0); vst1q_f32(tmp[1], f1); vst1q_f32(tmp[2], f2);
            for (int f = 0; f < nf; ++f) vst1q_f32(framed_out + f * kFft + 2 * m, vld1q_f32(tmp[f]));
        }
    }
}

// 4-way unrolled DC blocker (single-pass over [0..n), so output is bit-identical for any thread count).
void dc_block_4way(float * out, size_t n) {
    const float a1 = kDcBlockPole;
    const float a2 = a1 * a1;
    const float a3 = a2 * a1;
    const float a4 = a2 * a2;
    float x1 = 0.0F;
    float y1 = 0.0F;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const float v0 = out[i], v1 = out[i + 1], v2 = out[i + 2], v3 = out[i + 3];
        const float d0 = v0 - x1, d1 = v1 - v0, d2 = v2 - v1, d3 = v3 - v2;
        x1 = v3;
        const float t0 = d0;
        const float t1 = a1 * d0 + d1;
        const float t2 = a1 * t1 + d2;
        const float t3 = a1 * t2 + d3;
        out[i]     = t0 + a1 * y1;
        out[i + 1] = t1 + a2 * y1;
        out[i + 2] = t2 + a3 * y1;
        y1 = t3 + a4 * y1;
        out[i + 3] = y1;
    }
    for (; i < n; ++i) {
        const float v = out[i];
        const float y = v - x1 + a1 * y1;
        x1 = v;
        y1 = y;
        out[i] = y;
    }
}

// ============================================================================
// Exact FP32 8x12 NEON Microkernels
// ============================================================================
#define SANOTTS_FMA_8x12(x0, x1, x2, w0, w1) do { \
    a00 = vfmaq_laneq_f32(a00, x0, w0, 0); a01 = vfmaq_laneq_f32(a01, x1, w0, 0); a02 = vfmaq_laneq_f32(a02, x2, w0, 0); \
    a10 = vfmaq_laneq_f32(a10, x0, w0, 1); a11 = vfmaq_laneq_f32(a11, x1, w0, 1); a12 = vfmaq_laneq_f32(a12, x2, w0, 1); \
    a20 = vfmaq_laneq_f32(a20, x0, w0, 2); a21 = vfmaq_laneq_f32(a21, x1, w0, 2); a22 = vfmaq_laneq_f32(a22, x2, w0, 2); \
    a30 = vfmaq_laneq_f32(a30, x0, w0, 3); a31 = vfmaq_laneq_f32(a31, x1, w0, 3); a32 = vfmaq_laneq_f32(a32, x2, w0, 3); \
    a40 = vfmaq_laneq_f32(a40, x0, w1, 0); a41 = vfmaq_laneq_f32(a41, x1, w1, 0); a42 = vfmaq_laneq_f32(a42, x2, w1, 0); \
    a50 = vfmaq_laneq_f32(a50, x0, w1, 1); a51 = vfmaq_laneq_f32(a51, x1, w1, 1); a52 = vfmaq_laneq_f32(a52, x2, w1, 1); \
    a60 = vfmaq_laneq_f32(a60, x0, w1, 2); a61 = vfmaq_laneq_f32(a61, x1, w1, 2); a62 = vfmaq_laneq_f32(a62, x2, w1, 2); \
    a70 = vfmaq_laneq_f32(a70, x0, w1, 3); a71 = vfmaq_laneq_f32(a71, x1, w1, 3); a72 = vfmaq_laneq_f32(a72, x2, w1, 3); \
} while (0)

#define SANOTTS_DECL_ACC_8x12(bptr) \
    float32x4_t a00 = vdupq_n_f32((bptr)[0]), a01 = a00, a02 = a00; \
    float32x4_t a10 = vdupq_n_f32((bptr)[1]), a11 = a10, a12 = a10; \
    float32x4_t a20 = vdupq_n_f32((bptr)[2]), a21 = a20, a22 = a20; \
    float32x4_t a30 = vdupq_n_f32((bptr)[3]), a31 = a30, a32 = a30; \
    float32x4_t a40 = vdupq_n_f32((bptr)[4]), a41 = a40, a42 = a40; \
    float32x4_t a50 = vdupq_n_f32((bptr)[5]), a51 = a50, a52 = a50; \
    float32x4_t a60 = vdupq_n_f32((bptr)[6]), a61 = a60, a62 = a60; \
    float32x4_t a70 = vdupq_n_f32((bptr)[7]), a71 = a70, a72 = a70;

__attribute__((always_inline)) inline void store_cm_8x12(float * out, int T, int t0, int m, int o0, int no,
    float32x4_t a00,float32x4_t a01,float32x4_t a02,float32x4_t a10,float32x4_t a11,float32x4_t a12,
    float32x4_t a20,float32x4_t a21,float32x4_t a22,float32x4_t a30,float32x4_t a31,float32x4_t a32,
    float32x4_t a40,float32x4_t a41,float32x4_t a42,float32x4_t a50,float32x4_t a51,float32x4_t a52,
    float32x4_t a60,float32x4_t a61,float32x4_t a62,float32x4_t a70,float32x4_t a71,float32x4_t a72) {
    if (m == 12 && no == 8) {
        const size_t TT = static_cast<size_t>(T); float * p = out + static_cast<size_t>(o0) * TT + t0;
        vst1q_f32(p+0*TT,a00); vst1q_f32(p+0*TT+4,a01); vst1q_f32(p+0*TT+8,a02);
        vst1q_f32(p+1*TT,a10); vst1q_f32(p+1*TT+4,a11); vst1q_f32(p+1*TT+8,a12);
        vst1q_f32(p+2*TT,a20); vst1q_f32(p+2*TT+4,a21); vst1q_f32(p+2*TT+8,a22);
        vst1q_f32(p+3*TT,a30); vst1q_f32(p+3*TT+4,a31); vst1q_f32(p+3*TT+8,a32);
        vst1q_f32(p+4*TT,a40); vst1q_f32(p+4*TT+4,a41); vst1q_f32(p+4*TT+8,a42);
        vst1q_f32(p+5*TT,a50); vst1q_f32(p+5*TT+4,a51); vst1q_f32(p+5*TT+8,a52);
        vst1q_f32(p+6*TT,a60); vst1q_f32(p+6*TT+4,a61); vst1q_f32(p+6*TT+8,a62);
        vst1q_f32(p+7*TT,a70); vst1q_f32(p+7*TT+4,a71); vst1q_f32(p+7*TT+8,a72);
        return;
    }
    float tmp[8][12];
    vst1q_f32(tmp[0],a00); vst1q_f32(tmp[0]+4,a01); vst1q_f32(tmp[0]+8,a02);
    vst1q_f32(tmp[1],a10); vst1q_f32(tmp[1]+4,a11); vst1q_f32(tmp[1]+8,a12);
    vst1q_f32(tmp[2],a20); vst1q_f32(tmp[2]+4,a21); vst1q_f32(tmp[2]+8,a22);
    vst1q_f32(tmp[3],a30); vst1q_f32(tmp[3]+4,a31); vst1q_f32(tmp[3]+8,a32);
    vst1q_f32(tmp[4],a40); vst1q_f32(tmp[4]+4,a41); vst1q_f32(tmp[4]+8,a42);
    vst1q_f32(tmp[5],a50); vst1q_f32(tmp[5]+4,a51); vst1q_f32(tmp[5]+8,a52);
    vst1q_f32(tmp[6],a60); vst1q_f32(tmp[6]+4,a61); vst1q_f32(tmp[6]+8,a62);
    vst1q_f32(tmp[7],a70); vst1q_f32(tmp[7]+4,a71); vst1q_f32(tmp[7]+8,a72);
    for (int j = 0; j < no; ++j) std::memcpy(out + static_cast<size_t>(o0 + j) * T + t0, tmp[j], sizeof(float) * m);
}

__attribute__((always_inline)) inline void add_store_cm_8x12(float * out, int T, int t0, int m, int o0, int no,
    float32x4_t a00,float32x4_t a01,float32x4_t a02,float32x4_t a10,float32x4_t a11,float32x4_t a12,
    float32x4_t a20,float32x4_t a21,float32x4_t a22,float32x4_t a30,float32x4_t a31,float32x4_t a32,
    float32x4_t a40,float32x4_t a41,float32x4_t a42,float32x4_t a50,float32x4_t a51,float32x4_t a52,
    float32x4_t a60,float32x4_t a61,float32x4_t a62,float32x4_t a70,float32x4_t a71,float32x4_t a72) {
    if (m == 12 && no == 8) {
        const size_t TT = static_cast<size_t>(T); float * p = out + static_cast<size_t>(o0) * TT + t0;
#define SANOTTS_ADD3(r, v0, v1, v2) do { float * q = p + static_cast<size_t>(r) * TT; vst1q_f32(q, vaddq_f32(vld1q_f32(q), v0)); vst1q_f32(q+4, vaddq_f32(vld1q_f32(q+4), v1)); vst1q_f32(q+8, vaddq_f32(vld1q_f32(q+8), v2)); } while (0)
        SANOTTS_ADD3(0,a00,a01,a02); SANOTTS_ADD3(1,a10,a11,a12); SANOTTS_ADD3(2,a20,a21,a22); SANOTTS_ADD3(3,a30,a31,a32);
        SANOTTS_ADD3(4,a40,a41,a42); SANOTTS_ADD3(5,a50,a51,a52); SANOTTS_ADD3(6,a60,a61,a62); SANOTTS_ADD3(7,a70,a71,a72);
#undef SANOTTS_ADD3
        return;
    }
    float tmp[8][12];
    vst1q_f32(tmp[0],a00); vst1q_f32(tmp[0]+4,a01); vst1q_f32(tmp[0]+8,a02);
    vst1q_f32(tmp[1],a10); vst1q_f32(tmp[1]+4,a11); vst1q_f32(tmp[1]+8,a12);
    vst1q_f32(tmp[2],a20); vst1q_f32(tmp[2]+4,a21); vst1q_f32(tmp[2]+8,a22);
    vst1q_f32(tmp[3],a30); vst1q_f32(tmp[3]+4,a31); vst1q_f32(tmp[3]+8,a32);
    vst1q_f32(tmp[4],a40); vst1q_f32(tmp[4]+4,a41); vst1q_f32(tmp[4]+8,a42);
    vst1q_f32(tmp[5],a50); vst1q_f32(tmp[5]+4,a51); vst1q_f32(tmp[5]+8,a52);
    vst1q_f32(tmp[6],a60); vst1q_f32(tmp[6]+4,a61); vst1q_f32(tmp[6]+8,a62);
    vst1q_f32(tmp[7],a70); vst1q_f32(tmp[7]+4,a71); vst1q_f32(tmp[7]+8,a72);
    for (int j = 0; j < no; ++j) { float * q = out + static_cast<size_t>(o0 + j) * T + t0; for (int t = 0; t < m; ++t) q[t] += tmp[j][t]; }
}

__attribute__((always_inline)) inline void store_tile_8x12(float * dst_tile, int o0,
    float32x4_t a00,float32x4_t a01,float32x4_t a02,float32x4_t a10,float32x4_t a11,float32x4_t a12,
    float32x4_t a20,float32x4_t a21,float32x4_t a22,float32x4_t a30,float32x4_t a31,float32x4_t a32,
    float32x4_t a40,float32x4_t a41,float32x4_t a42,float32x4_t a50,float32x4_t a51,float32x4_t a52,
    float32x4_t a60,float32x4_t a61,float32x4_t a62,float32x4_t a70,float32x4_t a71,float32x4_t a72) {
    float * dp = dst_tile + static_cast<size_t>(o0) * 12;
    vst1q_f32(dp+0*12,a00); vst1q_f32(dp+0*12+4,a01); vst1q_f32(dp+0*12+8,a02);
    vst1q_f32(dp+1*12,a10); vst1q_f32(dp+1*12+4,a11); vst1q_f32(dp+1*12+8,a12);
    vst1q_f32(dp+2*12,a20); vst1q_f32(dp+2*12+4,a21); vst1q_f32(dp+2*12+8,a22);
    vst1q_f32(dp+3*12,a30); vst1q_f32(dp+3*12+4,a31); vst1q_f32(dp+3*12+8,a32);
    vst1q_f32(dp+4*12,a40); vst1q_f32(dp+4*12+4,a41); vst1q_f32(dp+4*12+8,a42);
    vst1q_f32(dp+5*12,a50); vst1q_f32(dp+5*12+4,a51); vst1q_f32(dp+5*12+8,a52);
    vst1q_f32(dp+6*12,a60); vst1q_f32(dp+6*12+4,a61); vst1q_f32(dp+6*12+8,a62);
    vst1q_f32(dp+7*12,a70); vst1q_f32(dp+7*12+4,a71); vst1q_f32(dp+7*12+8,a72);
}

__attribute__((always_inline)) inline void store_tr4x4(float * dst, size_t ld, float32x4_t r0, float32x4_t r1, float32x4_t r2, float32x4_t r3) {
    const float32x4_t t00 = vtrn1q_f32(r0, r1), t01 = vtrn2q_f32(r0, r1);
    const float32x4_t t10 = vtrn1q_f32(r2, r3), t11 = vtrn2q_f32(r2, r3);
    vst1q_f32(dst+0*ld, vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t00), vreinterpretq_f64_f32(t10))));
    vst1q_f32(dst+1*ld, vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t01), vreinterpretq_f64_f32(t11))));
    vst1q_f32(dst+2*ld, vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t00), vreinterpretq_f64_f32(t10))));
    vst1q_f32(dst+3*ld, vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t01), vreinterpretq_f64_f32(t11))));
}

__attribute__((always_inline)) inline void store_rows_8x12(float * out, int rows_ld, int t0, int m, int o0, int no,
    float32x4_t a00,float32x4_t a01,float32x4_t a02,float32x4_t a10,float32x4_t a11,float32x4_t a12,
    float32x4_t a20,float32x4_t a21,float32x4_t a22,float32x4_t a30,float32x4_t a31,float32x4_t a32,
    float32x4_t a40,float32x4_t a41,float32x4_t a42,float32x4_t a50,float32x4_t a51,float32x4_t a52,
    float32x4_t a60,float32x4_t a61,float32x4_t a62,float32x4_t a70,float32x4_t a71,float32x4_t a72) {
    if (m == 12 && no == 8) {
        const size_t ld = static_cast<size_t>(rows_ld); float * d0 = out + static_cast<size_t>(t0) * ld + o0;
        store_tr4x4(d0+0*ld, ld, a00, a10, a20, a30); store_tr4x4(d0+4*ld, ld, a01, a11, a21, a31); store_tr4x4(d0+8*ld, ld, a02, a12, a22, a32);
        float * d4 = d0 + 4;
        store_tr4x4(d4+0*ld, ld, a40, a50, a60, a70); store_tr4x4(d4+4*ld, ld, a41, a51, a61, a71); store_tr4x4(d4+8*ld, ld, a42, a52, a62, a72);
        return;
    }
    float tmp[8][12];
    vst1q_f32(tmp[0],a00); vst1q_f32(tmp[0]+4,a01); vst1q_f32(tmp[0]+8,a02);
    vst1q_f32(tmp[1],a10); vst1q_f32(tmp[1]+4,a11); vst1q_f32(tmp[1]+8,a12);
    vst1q_f32(tmp[2],a20); vst1q_f32(tmp[2]+4,a21); vst1q_f32(tmp[2]+8,a22);
    vst1q_f32(tmp[3],a30); vst1q_f32(tmp[3]+4,a31); vst1q_f32(tmp[3]+8,a32);
    vst1q_f32(tmp[4],a40); vst1q_f32(tmp[4]+4,a41); vst1q_f32(tmp[4]+8,a42);
    vst1q_f32(tmp[5],a50); vst1q_f32(tmp[5]+4,a51); vst1q_f32(tmp[5]+8,a52);
    vst1q_f32(tmp[6],a60); vst1q_f32(tmp[6]+4,a61); vst1q_f32(tmp[6]+8,a62);
    vst1q_f32(tmp[7],a70); vst1q_f32(tmp[7]+4,a71); vst1q_f32(tmp[7]+8,a72);
    for (int t = 0; t < m; ++t) { float * row = out + static_cast<size_t>(t0 + t) * rows_ld + o0; for (int j = 0; j < no; ++j) row[j] = tmp[j][t]; }
}

__attribute__((noinline)) void gemm_8x12_cm(const float * xblk, int cin, const PackedConv & c, float * out, int T, int t0, int m, bool add_res) {
    for (int ob = 0; ob < c.co8 / 8; ++ob) {
        const int o0 = ob * 8, no = std::min(8, c.cout - o0);
        SANOTTS_DECL_ACC_8x12(c.bias.data() + o0)
        const float * wp = c.wf32.data() + static_cast<size_t>(ob) * cin * 8, * xp = xblk;
        int ci = 0;
        for (; ci + 2 <= cin; ci += 2, xp += 24, wp += 16) {
            const float32x4_t x0 = vld1q_f32(xp), x1 = vld1q_f32(xp+4), x2 = vld1q_f32(xp+8), w0 = vld1q_f32(wp), w1 = vld1q_f32(wp+4);
            SANOTTS_FMA_8x12(x0, x1, x2, w0, w1);
            const float32x4_t x3 = vld1q_f32(xp+12), x4 = vld1q_f32(xp+16), x5 = vld1q_f32(xp+20), w2 = vld1q_f32(wp+8), w3 = vld1q_f32(wp+12);
            SANOTTS_FMA_8x12(x3, x4, x5, w2, w3);
        }
        for (; ci < cin; ++ci, xp += 12, wp += 8) {
            const float32x4_t x0 = vld1q_f32(xp), x1 = vld1q_f32(xp+4), x2 = vld1q_f32(xp+8), w0 = vld1q_f32(wp), w1 = vld1q_f32(wp+4);
            SANOTTS_FMA_8x12(x0, x1, x2, w0, w1);
        }
        if (add_res) add_store_cm_8x12(out, T, t0, m, o0, no, a00,a01,a02,a10,a11,a12,a20,a21,a22,a30,a31,a32,a40,a41,a42,a50,a51,a52,a60,a61,a62,a70,a71,a72);
        else store_cm_8x12(out, T, t0, m, o0, no, a00,a01,a02,a10,a11,a12,a20,a21,a22,a30,a31,a32,a40,a41,a42,a50,a51,a52,a60,a61,a62,a70,a71,a72);
    }
}

__attribute__((noinline)) void gemm_8x12_to_tile_gelu(const float * xblk, int cin, const PackedConv & c, float * dst_tile) {
    for (int ob = 0; ob < c.co8 / 8; ++ob) {
        const int o0 = ob * 8;
        SANOTTS_DECL_ACC_8x12(c.bias.data() + o0)
        const float * wp = c.wf32.data() + static_cast<size_t>(ob) * cin * 8, * xp = xblk;
        int ci = 0;
        for (; ci + 2 <= cin; ci += 2, xp += 24, wp += 16) {
            const float32x4_t x0 = vld1q_f32(xp), x1 = vld1q_f32(xp+4), x2 = vld1q_f32(xp+8), w0 = vld1q_f32(wp), w1 = vld1q_f32(wp+4);
            SANOTTS_FMA_8x12(x0, x1, x2, w0, w1);
            const float32x4_t x3 = vld1q_f32(xp+12), x4 = vld1q_f32(xp+16), x5 = vld1q_f32(xp+20), w2 = vld1q_f32(wp+8), w3 = vld1q_f32(wp+12);
            SANOTTS_FMA_8x12(x3, x4, x5, w2, w3);
        }
        for (; ci < cin; ++ci, xp += 12, wp += 8) {
            const float32x4_t x0 = vld1q_f32(xp), x1 = vld1q_f32(xp+4), x2 = vld1q_f32(xp+8), w0 = vld1q_f32(wp), w1 = vld1q_f32(wp+4);
            SANOTTS_FMA_8x12(x0, x1, x2, w0, w1);
        }
        store_tile_8x12(dst_tile, o0, a00,a01,a02,a10,a11,a12,a20,a21,a22,a30,a31,a32,a40,a41,a42,a50,a51,a52,a60,a61,a62,a70,a71,a72);
    }
    const int total = c.cout * 12;
    int i = 0;
    for (; i + 8 <= total; i += 8) { vst1q_f32(dst_tile+i, gelu4(vld1q_f32(dst_tile+i))); vst1q_f32(dst_tile+i+4, gelu4(vld1q_f32(dst_tile+i+4))); }
    for (; i < total; i += 4) vst1q_f32(dst_tile+i, gelu4(vld1q_f32(dst_tile+i)));
}

__attribute__((noinline)) void gemm_8x12_rows(const float * xblk, int cin, const PackedConv & c, float * out, int rows_ld, int t0, int m) {
    for (int ob = 0; ob < c.co8 / 8; ++ob) {
        const int o0 = ob * 8, no = std::min(8, c.cout - o0);
        SANOTTS_DECL_ACC_8x12(c.bias.data() + o0)
        const float * wp = c.wf32.data() + static_cast<size_t>(ob) * cin * 8, * xp = xblk;
        int ci = 0;
        for (; ci + 2 <= cin; ci += 2, xp += 24, wp += 16) {
            const float32x4_t x0 = vld1q_f32(xp), x1 = vld1q_f32(xp+4), x2 = vld1q_f32(xp+8), w0 = vld1q_f32(wp), w1 = vld1q_f32(wp+4);
            SANOTTS_FMA_8x12(x0, x1, x2, w0, w1);
            const float32x4_t x3 = vld1q_f32(xp+12), x4 = vld1q_f32(xp+16), x5 = vld1q_f32(xp+20), w2 = vld1q_f32(wp+8), w3 = vld1q_f32(wp+12);
            SANOTTS_FMA_8x12(x3, x4, x5, w2, w3);
        }
        for (; ci < cin; ++ci, xp += 12, wp += 8) {
            const float32x4_t x0 = vld1q_f32(xp), x1 = vld1q_f32(xp+4), x2 = vld1q_f32(xp+8), w0 = vld1q_f32(wp), w1 = vld1q_f32(wp+4);
            SANOTTS_FMA_8x12(x0, x1, x2, w0, w1);
        }
        store_rows_8x12(out, rows_ld, t0, m, o0, no, a00,a01,a02,a10,a11,a12,a20,a21,a22,a30,a31,a32,a40,a41,a42,a50,a51,a52,a60,a61,a62,a70,a71,a72);
    }
}

__attribute__((noinline)) void head_group_8x12_f32(
    const float * ln_blk, int cin, const float * wp, const float * bp, float out8x12[8][12]) {
    SANOTTS_DECL_ACC_8x12(bp)
    const float * xp = ln_blk;
    int ci = 0;
    for (; ci + 2 <= cin; ci += 2, xp += 24, wp += 16) {
        const float32x4_t x0 = vld1q_f32(xp), x1 = vld1q_f32(xp+4), x2 = vld1q_f32(xp+8), w0 = vld1q_f32(wp), w1 = vld1q_f32(wp+4);
        SANOTTS_FMA_8x12(x0, x1, x2, w0, w1);
        const float32x4_t x3 = vld1q_f32(xp+12), x4 = vld1q_f32(xp+16), x5 = vld1q_f32(xp+20), w2 = vld1q_f32(wp+8), w3 = vld1q_f32(wp+12);
        SANOTTS_FMA_8x12(x3, x4, x5, w2, w3);
    }
    for (; ci < cin; ++ci, xp += 12, wp += 8) {
        const float32x4_t x0 = vld1q_f32(xp), x1 = vld1q_f32(xp+4), x2 = vld1q_f32(xp+8), w0 = vld1q_f32(wp), w1 = vld1q_f32(wp+4);
        SANOTTS_FMA_8x12(x0, x1, x2, w0, w1);
    }
    store_tile_8x12(out8x12[0], 0, a00,a01,a02,a10,a11,a12,a20,a21,a22,a30,a31,a32,a40,a41,a42,a50,a51,a52,a60,a61,a62,a70,a71,a72);
}

inline void pack_win_k5(const float * src, int cin, int T, int t0, float * win) {
    if (t0 >= 2 && t0 + 14 <= T) {
        for (int ci = 0; ci < cin; ++ci) {
            const float * s = src + static_cast<size_t>(ci) * T + t0 - 2; float * d = win + static_cast<size_t>(ci) * 16;
            vst1q_f32(d, vld1q_f32(s)); vst1q_f32(d+4, vld1q_f32(s+4)); vst1q_f32(d+8, vld1q_f32(s+8)); vst1q_f32(d+12, vld1q_f32(s+12));
        }
    } else {
        for (int ci = 0; ci < cin; ++ci) {
            const float * s = src + static_cast<size_t>(ci) * T; float * d = win + static_cast<size_t>(ci) * 16;
            for (int j = 0; j < 16; ++j) { const int idx = t0 - 2 + j; d[j] = (idx >= 0 && idx < T) ? s[idx] : 0.0F; }
        }
    }
}

inline void pack_win_k7(const float * src, int cin, int T, int t0, float * win) {
    if (t0 >= 3 && t0 + 17 <= T) {
        for (int ci = 0; ci < cin; ++ci) {
            const float * s = src + static_cast<size_t>(ci) * T + t0 - 3; float * d = win + static_cast<size_t>(ci) * 20;
            vst1q_f32(d, vld1q_f32(s)); vst1q_f32(d+4, vld1q_f32(s+4)); vst1q_f32(d+8, vld1q_f32(s+8)); vst1q_f32(d+12, vld1q_f32(s+12)); vst1q_f32(d+16, vld1q_f32(s+16));
        }
    } else {
        for (int ci = 0; ci < cin; ++ci) {
            const float * s = src + static_cast<size_t>(ci) * T; float * d = win + static_cast<size_t>(ci) * 20;
            for (int j = 0; j < 20; ++j) { const int idx = t0 - 3 + j; d[j] = (idx >= 0 && idx < T) ? s[idx] : 0.0F; }
        }
    }
}

inline void pack_win_general(const float * src, int cin, int K, int stride, int T, int t0, float * win) {
    const int pad = K / 2, width = 12 + K - 1;
    for (int ci = 0; ci < cin; ++ci) {
        const float * s = src + static_cast<size_t>(ci) * T; float * d = win + static_cast<size_t>(ci) * stride;
        for (int j = 0; j < width; ++j) { const int idx = t0 - pad + j; d[j] = (idx >= 0 && idx < T) ? s[idx] : 0.0F; }
        for (int j = width; j < stride; ++j) d[j] = 0.0F;
    }
}

__attribute__((noinline)) void conv_k5_tile(const PackedConv & c, const float * win, float * out, int T, int t0, int m, int mode) {
    const int cin = c.cin;
    for (int ob = 0; ob < c.co8 / 8; ++ob) {
        const int o0 = ob * 8, no = std::min(8, c.cout - o0);
        SANOTTS_DECL_ACC_8x12(c.bias.data() + o0)
        const float * wp = c.wf32.data() + static_cast<size_t>(ob) * cin * 5 * 8, * xp = win;
        for (int ci = 0; ci < cin; ++ci, xp += 16) {
            for (int kk = 0; kk < 5; ++kk, wp += 8) {
                const float32x4_t x0 = vld1q_f32(xp+kk), x1 = vld1q_f32(xp+kk+4), x2 = vld1q_f32(xp+kk+8), w0 = vld1q_f32(wp), w1 = vld1q_f32(wp+4);
                SANOTTS_FMA_8x12(x0, x1, x2, w0, w1);
            }
        }
        if (mode == 1) {
            alignas(16) float tmp[8][12];
            store_tile_8x12(tmp[0], 0, a00,a01,a02,a10,a11,a12,a20,a21,a22,a30,a31,a32,a40,a41,a42,a50,a51,a52,a60,a61,a62,a70,a71,a72);
            for (int j = 0; j < no; ++j) {
                const float32x4_t v0 = silu4(vld1q_f32(tmp[j])), v1 = silu4(vld1q_f32(tmp[j]+4)), v2 = silu4(vld1q_f32(tmp[j]+8));
                float * dst = out + static_cast<size_t>(o0 + j) * T + t0;
                if (m == 12) { vst1q_f32(dst, v0); vst1q_f32(dst+4, v1); vst1q_f32(dst+8, v2); }
                else { float row[12]; vst1q_f32(row, v0); vst1q_f32(row+4, v1); vst1q_f32(row+8, v2); std::memcpy(dst, row, sizeof(float) * m); }
            }
        } else if (mode == 2) add_store_cm_8x12(out, T, t0, m, o0, no, a00,a01,a02,a10,a11,a12,a20,a21,a22,a30,a31,a32,a40,a41,a42,a50,a51,a52,a60,a61,a62,a70,a71,a72);
        else store_cm_8x12(out, T, t0, m, o0, no, a00,a01,a02,a10,a11,a12,a20,a21,a22,a30,a31,a32,a40,a41,a42,a50,a51,a52,a60,a61,a62,a70,a71,a72);
    }
}

__attribute__((noinline)) void conv_k7_to_tile(const PackedConv & c, const float * win7, float * dst_tile) {
    const int cin = c.cin;
    for (int ob = 0; ob < c.co8 / 8; ++ob) {
        const int o0 = ob * 8;
        SANOTTS_DECL_ACC_8x12(c.bias.data() + o0)
        const float * wp = c.wf32.data() + static_cast<size_t>(ob) * cin * 7 * 8, * xp = win7;
        for (int ci = 0; ci < cin; ++ci, xp += 20) {
            for (int kk = 0; kk < 7; ++kk, wp += 8) {
                const float32x4_t x0 = vld1q_f32(xp+kk), x1 = vld1q_f32(xp+kk+4), x2 = vld1q_f32(xp+kk+8), w0 = vld1q_f32(wp), w1 = vld1q_f32(wp+4);
                SANOTTS_FMA_8x12(x0, x1, x2, w0, w1);
            }
        }
        store_tile_8x12(dst_tile, o0, a00,a01,a02,a10,a11,a12,a20,a21,a22,a30,a31,a32,a40,a41,a42,a50,a51,a52,a60,a61,a62,a70,a71,a72);
    }
}

__attribute__((noinline)) void conv_k_general_tile(const PackedConv & c, const float * win, int stride, float * out, float * dst_tile, int T, int t0, int m, int mode) {
    const int cin = c.cin, K = c.kernel;
    for (int ob = 0; ob < c.co8 / 8; ++ob) {
        const int o0 = ob * 8, no = std::min(8, c.cout - o0);
        SANOTTS_DECL_ACC_8x12(c.bias.data() + o0)
        const float * wp = c.wf32.data() + static_cast<size_t>(ob) * cin * K * 8;
        for (int ci = 0; ci < cin; ++ci) {
            const float * xp = win + static_cast<size_t>(ci) * stride;
            for (int kk = 0; kk < K; ++kk, wp += 8) {
                const float32x4_t x0 = vld1q_f32(xp+kk), x1 = vld1q_f32(xp+kk+4), x2 = vld1q_f32(xp+kk+8), w0 = vld1q_f32(wp), w1 = vld1q_f32(wp+4);
                SANOTTS_FMA_8x12(x0, x1, x2, w0, w1);
            }
        }
        if (dst_tile != nullptr) store_tile_8x12(dst_tile, o0, a00,a01,a02,a10,a11,a12,a20,a21,a22,a30,a31,a32,a40,a41,a42,a50,a51,a52,a60,a61,a62,a70,a71,a72);
        else if (mode == 1) {
            alignas(16) float tmp[8][12];
            store_tile_8x12(tmp[0], 0, a00,a01,a02,a10,a11,a12,a20,a21,a22,a30,a31,a32,a40,a41,a42,a50,a51,a52,a60,a61,a62,a70,a71,a72);
            for (int j = 0; j < no; ++j) {
                const float32x4_t v0 = silu4(vld1q_f32(tmp[j])), v1 = silu4(vld1q_f32(tmp[j]+4)), v2 = silu4(vld1q_f32(tmp[j]+8));
                float * dst = out + static_cast<size_t>(o0 + j) * T + t0;
                if (m == 12) { vst1q_f32(dst, v0); vst1q_f32(dst+4, v1); vst1q_f32(dst+8, v2); }
                else { float row[12]; vst1q_f32(row, v0); vst1q_f32(row+4, v1); vst1q_f32(row+8, v2); std::memcpy(dst, row, sizeof(float) * m); }
            }
        } else if (mode == 2) add_store_cm_8x12(out, T, t0, m, o0, no, a00,a01,a02,a10,a11,a12,a20,a21,a22,a30,a31,a32,a40,a41,a42,a50,a51,a52,a60,a61,a62,a70,a71,a72);
        else store_cm_8x12(out, T, t0, m, o0, no, a00,a01,a02,a10,a11,a12,a20,a21,a22,a30,a31,a32,a40,a41,a42,a50,a51,a52,a60,a61,a62,a70,a71,a72);
    }
}

inline void ln_to_tile12(const float * src, int C, int T, int t0, int m, const float * g, const float * b, float * out_blk) {
    const float inv_c = 1.0F / static_cast<float>(C);
    if (m == 12) {
        const float32x4_t kInv = vdupq_n_f32(inv_c), kEps = vdupq_n_f32(1.0e-6F), k1 = vdupq_n_f32(1.0F);
        float32x4_t s0 = vdupq_n_f32(0.0F), s1 = s0, s2 = s0;
        for (int c = 0; c < C; ++c) { const float * p = src + static_cast<size_t>(c) * T + t0; s0 = vaddq_f32(s0, vld1q_f32(p)); s1 = vaddq_f32(s1, vld1q_f32(p+4)); s2 = vaddq_f32(s2, vld1q_f32(p+8)); }
        const float32x4_t m0 = vmulq_f32(s0, kInv), m1 = vmulq_f32(s1, kInv), m2 = vmulq_f32(s2, kInv);
        float32x4_t v0 = vdupq_n_f32(0.0F), v1 = v0, v2 = v0;
        for (int c = 0; c < C; ++c) {
            const float * p = src + static_cast<size_t>(c) * T + t0;
            const float32x4_t d0 = vsubq_f32(vld1q_f32(p), m0), d1 = vsubq_f32(vld1q_f32(p+4), m1), d2 = vsubq_f32(vld1q_f32(p+8), m2);
            v0 = vfmaq_f32(v0, d0, d0); v1 = vfmaq_f32(v1, d1, d1); v2 = vfmaq_f32(v2, d2, d2);
        }
        const float32x4_t r0 = vdivq_f32(k1, vsqrtq_f32(vfmaq_f32(kEps, v0, kInv)));
        const float32x4_t r1 = vdivq_f32(k1, vsqrtq_f32(vfmaq_f32(kEps, v1, kInv)));
        const float32x4_t r2 = vdivq_f32(k1, vsqrtq_f32(vfmaq_f32(kEps, v2, kInv)));
        for (int c = 0; c < C; ++c) {
            const float * p = src + static_cast<size_t>(c) * T + t0; float * d = out_blk + static_cast<size_t>(c) * 12;
            const float32x4_t bv = vdupq_n_f32(b[c]); const float gc = g[c];
            vst1q_f32(d, vfmaq_n_f32(bv, vmulq_f32(vsubq_f32(vld1q_f32(p), m0), r0), gc));
            vst1q_f32(d+4, vfmaq_n_f32(bv, vmulq_f32(vsubq_f32(vld1q_f32(p+4), m1), r1), gc));
            vst1q_f32(d+8, vfmaq_n_f32(bv, vmulq_f32(vsubq_f32(vld1q_f32(p+8), m2), r2), gc));
        }
        return;
    }
    for (int t = 0; t < m; ++t) {
        float sm = 0.0F; for (int c = 0; c < C; ++c) sm += src[static_cast<size_t>(c) * T + t0 + t]; sm *= inv_c;
        float sv = 0.0F; for (int c = 0; c < C; ++c) { const float dd = src[static_cast<size_t>(c) * T + t0 + t] - sm; sv += dd * dd; }
        const float iv = 1.0F / std::sqrt(sv * inv_c + 1.0e-6F);
        for (int c = 0; c < C; ++c) out_blk[static_cast<size_t>(c) * 12 + t] = (src[static_cast<size_t>(c) * T + t0 + t] - sm) * iv * g[c] + b[c];
    }
    for (int c = 0; c < C; ++c) std::memset(out_blk + static_cast<size_t>(c) * 12 + m, 0, sizeof(float) * (12 - m));
}

inline void ln_from_tile12(const float * in_blk, int C, const float * g, const float * b, float * out, int T, int t0, int m) {
    const float inv_c = 1.0F / static_cast<float>(C);
    const float32x4_t kInv = vdupq_n_f32(inv_c), kEps = vdupq_n_f32(1.0e-6F), k1 = vdupq_n_f32(1.0F);
    float32x4_t s0 = vdupq_n_f32(0.0F), s1 = s0, s2 = s0;
    for (int c = 0; c < C; ++c) { const float * p = in_blk + static_cast<size_t>(c) * 12; s0 = vaddq_f32(s0, vld1q_f32(p)); s1 = vaddq_f32(s1, vld1q_f32(p+4)); s2 = vaddq_f32(s2, vld1q_f32(p+8)); }
    const float32x4_t m0 = vmulq_f32(s0, kInv), m1 = vmulq_f32(s1, kInv), m2 = vmulq_f32(s2, kInv);
    float32x4_t v0 = vdupq_n_f32(0.0F), v1 = v0, v2 = v0;
    for (int c = 0; c < C; ++c) {
        const float * p = in_blk + static_cast<size_t>(c) * 12;
        const float32x4_t d0 = vsubq_f32(vld1q_f32(p), m0), d1 = vsubq_f32(vld1q_f32(p+4), m1), d2 = vsubq_f32(vld1q_f32(p+8), m2);
        v0 = vfmaq_f32(v0, d0, d0); v1 = vfmaq_f32(v1, d1, d1); v2 = vfmaq_f32(v2, d2, d2);
    }
    const float32x4_t r0 = vdivq_f32(k1, vsqrtq_f32(vfmaq_f32(kEps, v0, kInv)));
    const float32x4_t r1 = vdivq_f32(k1, vsqrtq_f32(vfmaq_f32(kEps, v1, kInv)));
    const float32x4_t r2 = vdivq_f32(k1, vsqrtq_f32(vfmaq_f32(kEps, v2, kInv)));
    if (m == 12) {
        for (int c = 0; c < C; ++c) {
            const float * p = in_blk + static_cast<size_t>(c) * 12; float * d = out + static_cast<size_t>(c) * T + t0;
            const float32x4_t bv = vdupq_n_f32(b[c]); const float gc = g[c];
            vst1q_f32(d, vfmaq_n_f32(bv, vmulq_f32(vsubq_f32(vld1q_f32(p), m0), r0), gc));
            vst1q_f32(d+4, vfmaq_n_f32(bv, vmulq_f32(vsubq_f32(vld1q_f32(p+4), m1), r1), gc));
            vst1q_f32(d+8, vfmaq_n_f32(bv, vmulq_f32(vsubq_f32(vld1q_f32(p+8), m2), r2), gc));
        }
    } else {
        float tmp[12];
        for (int c = 0; c < C; ++c) {
            const float * p = in_blk + static_cast<size_t>(c) * 12;
            const float32x4_t bv = vdupq_n_f32(b[c]); const float gc = g[c];
            vst1q_f32(tmp, vfmaq_n_f32(bv, vmulq_f32(vsubq_f32(vld1q_f32(p), m0), r0), gc));
            vst1q_f32(tmp+4, vfmaq_n_f32(bv, vmulq_f32(vsubq_f32(vld1q_f32(p+4), m1), r1), gc));
            vst1q_f32(tmp+8, vfmaq_n_f32(bv, vmulq_f32(vsubq_f32(vld1q_f32(p+8), m2), r2), gc));
            std::memcpy(out + static_cast<size_t>(c) * T + t0, tmp, sizeof(float) * m);
        }
    }
}

struct FrontStage {
    int vocab = 0, hidden = 0, feat_rows = 0, depth = 0, kernel = 0;
    bool has_out = false;
    std::vector<float> emb;
    PackedConv proj;
    std::vector<PackedConv> b0, b1;
    PackedConv out_proj;
};

}  // namespace

struct SanoTtsNeonDecoder::Impl {
    int hidden = 0, proj_in = 0, embed_in = 0, depth = 0, ak = 0, ek = 0;
    int dim = 0, pw_hidden = 0, blocks = 0, noise = 0, dw_kernel = 0, out_dim = 0;
    int n_fft = 0, hop_length = 0;

    PackedConv proj;
    std::vector<PackedConv> frame_conv0, frame_conv1;
    PackedConv embed;
    std::vector<float> norm_w, norm_b, final_w, final_b;
    struct Block {
        std::vector<float> dw_w, dw_b, norm_w, norm_b;
        PackedConv pw1, pw2;
    };
    std::vector<Block> decoder;
    PackedConv head;
    HeadPaired head_paired;

    bool frontend_ready = false;
    FrontStage dur_stage, aco_stage;
    std::vector<float> front_h, front_a;

    std::vector<float> f, h, xs, e, scratch;
    std::unique_ptr<float[]> framed;
    size_t framed_cap = 0;
    size_t per_thread_scratch = 0;

    WorkerPool pool;
    int frames = 0, nb12 = 0;

    void run_front(const FrontStage & st, const std::vector<int32_t> & tokens,
                   const std::vector<float> & feats, float * out, int threads);
    void worker(const float * context, const float * feats, const float * noise_in, float * out,
                bool fused_audio, int tid, int nt, Barrier & bar);
};

void SanoTtsNeonDecoder::Impl::run_front(const FrontStage & st, const std::vector<int32_t> & tokens,
                                         const std::vector<float> & feats, float * out, int threads) {
    if (!frontend_ready) {
        throw std::runtime_error("sanoTTS NEON front-end not initialized");
    }
    if (tokens.empty() || tokens.size() > (1U << 24)) {
        throw std::runtime_error("sanoTTS NEON front-end: invalid token count");
    }
    const int n = static_cast<int>(tokens.size());
    if (feats.size() != static_cast<size_t>(st.feat_rows) * static_cast<size_t>(n)) {
        throw std::runtime_error("sanoTTS NEON front-end: invalid feature size");
    }
    for (int32_t tok : tokens) {
        if (tok < 0 || tok >= st.vocab) {
            throw std::runtime_error("sanoTTS NEON front-end: token id out of range");
        }
    }

    const int H = st.hidden, rows = st.feat_rows, cin0 = H + rows;
    const int nb = (n + 11) / 12;
    const int nt = (n < 48 || threads <= 1) ? 1 : std::clamp(threads, 1, 16);
    const size_t need_buf = static_cast<size_t>(H) * static_cast<size_t>(n);
    if (front_h.size() < need_buf) front_h.resize(need_buf);
    if (front_a.size() < need_buf) front_a.resize(need_buf);
    const int k_stride = st.kernel == 5 ? 16 : ((12 + st.kernel - 1 + 3) & ~3);
    const size_t pt = ((std::max(static_cast<size_t>(cin0) * 12, static_cast<size_t>(H) * k_stride)) + 15) & ~static_cast<size_t>(15);
    if (scratch.size() < static_cast<size_t>(nt) * pt) scratch.resize(static_cast<size_t>(nt) * pt);

    float * hbuf = st.has_out ? front_h.data() : out;
    float * abuf = front_a.data();
    Barrier bar(nt);
    pool.run(nt, [&](int tid, int count) {
        int lo = 0, hi = 0;
        split_range(nb, tid, count, lo, hi);
        float * tbuf = scratch.data() + static_cast<size_t>(tid) * pt;
        for (int tb = lo; tb < hi; ++tb) {
            const int t0 = tb * 12, m = std::min(12, n - t0);
            for (int t = 0; t < m; ++t) {
                const int tok = tokens[t0 + t];
                const float * erow = st.emb.data() + static_cast<size_t>(tok) * H;
                for (int c = 0; c < H; ++c) tbuf[static_cast<size_t>(c) * 12 + t] = erow[c];
                for (int c = 0; c < rows; ++c) tbuf[static_cast<size_t>(H + c) * 12 + t] = feats[static_cast<size_t>(c) * n + t0 + t];
            }
            if (m < 12) {
                for (int c = 0; c < cin0; ++c) std::memset(tbuf + static_cast<size_t>(c) * 12 + m, 0, sizeof(float) * (12 - m));
            }
            gemm_8x12_cm(tbuf, cin0, st.proj, hbuf, n, t0, m, false);
        }
        bar.wait();
        for (int i = 0; i < st.depth; ++i) {
            if (st.kernel == 5) {
                for (int tb = lo; tb < hi; ++tb) { const int t0 = tb * 12, m = std::min(12, n - t0); pack_win_k5(hbuf, H, n, t0, tbuf); conv_k5_tile(st.b0[i], tbuf, abuf, n, t0, m, 1); }
                bar.wait();
                for (int tb = lo; tb < hi; ++tb) { const int t0 = tb * 12, m = std::min(12, n - t0); pack_win_k5(abuf, H, n, t0, tbuf); conv_k5_tile(st.b1[i], tbuf, hbuf, n, t0, m, 2); }
                bar.wait();
            } else {
                for (int tb = lo; tb < hi; ++tb) { const int t0 = tb * 12, m = std::min(12, n - t0); pack_win_general(hbuf, H, st.kernel, k_stride, n, t0, tbuf); conv_k_general_tile(st.b0[i], tbuf, k_stride, abuf, nullptr, n, t0, m, 1); }
                bar.wait();
                for (int tb = lo; tb < hi; ++tb) { const int t0 = tb * 12, m = std::min(12, n - t0); pack_win_general(abuf, H, st.kernel, k_stride, n, t0, tbuf); conv_k_general_tile(st.b1[i], tbuf, k_stride, hbuf, nullptr, n, t0, m, 2); }
                bar.wait();
            }
        }
        if (st.has_out) {
            const float b0 = st.out_proj.bias[0];
            const float * w = st.out_proj.wf32.data();
            for (int tb = lo; tb < hi; ++tb) {
                const int t0 = tb * 12, m = std::min(12, n - t0);
                if (m == 12) {
                    float32x4_t a0 = vdupq_n_f32(b0), a1 = a0, a2 = a0;
                    for (int ci = 0; ci < H; ++ci) {
                        const float * s = hbuf + static_cast<size_t>(ci) * n + t0;
                        const float wv = w[static_cast<size_t>(ci) * 8];
                        a0 = vfmaq_n_f32(a0, vld1q_f32(s), wv);
                        a1 = vfmaq_n_f32(a1, vld1q_f32(s + 4), wv);
                        a2 = vfmaq_n_f32(a2, vld1q_f32(s + 8), wv);
                    }
                    vst1q_f32(out + t0, a0); vst1q_f32(out + t0 + 4, a1); vst1q_f32(out + t0 + 8, a2);
                } else {
                    for (int t = 0; t < m; ++t) {
                        float acc = b0;
                        for (int ci = 0; ci < H; ++ci) acc += w[static_cast<size_t>(ci) * 8] * hbuf[static_cast<size_t>(ci) * n + t0 + t];
                        out[t0 + t] = acc;
                    }
                }
            }
        }
    });
}

void SanoTtsNeonDecoder::Impl::worker(const float * context, const float * feats,
                                      const float * noise_in, float * out, bool fused_audio,
                                      int tid, int nt, Barrier & bar) {
    const int T = frames;
    const size_t TT = static_cast<size_t>(T);
    float * tbuf = scratch.data() + static_cast<size_t>(tid) * per_thread_scratch;
    int lo12 = 0, hi12 = 0;
    split_range(nb12, tid, nt, lo12, hi12);

    // 1. Fused input pack + frame_input_proj -> f[0..hidden-1], plus inside-row 1s + noise -> f[hidden..]
    for (int tb = lo12; tb < hi12; ++tb) {
        const int t0 = tb * 12, m = std::min(12, T - t0);
        float * xblk = tbuf;
        if (m == 12) {
            for (int ci = 0; ci < hidden; ++ci) {
                const float * s = context + static_cast<size_t>(ci) * TT + t0; float * d = xblk + static_cast<size_t>(ci) * 12;
                vst1q_f32(d, vld1q_f32(s)); vst1q_f32(d+4, vld1q_f32(s+4)); vst1q_f32(d+8, vld1q_f32(s+8));
            }
            for (int ci = 0; ci < 3; ++ci) {
                const float * s = feats + static_cast<size_t>(ci) * TT + t0; float * d = xblk + static_cast<size_t>(hidden + ci) * 12;
                vst1q_f32(d, vld1q_f32(s)); vst1q_f32(d+4, vld1q_f32(s+4)); vst1q_f32(d+8, vld1q_f32(s+8));
            }
            const float32x4_t ones = vdupq_n_f32(1.0F);
            float * ins = f.data() + static_cast<size_t>(hidden) * TT + t0;
            vst1q_f32(ins, ones); vst1q_f32(ins+4, ones); vst1q_f32(ins+8, ones);
            for (int n = 0; n < noise; ++n) {
                const float * s = noise_in + static_cast<size_t>(n) * TT + t0; float * d = f.data() + static_cast<size_t>(hidden + 1 + n) * TT + t0;
                vst1q_f32(d, vld1q_f32(s)); vst1q_f32(d+4, vld1q_f32(s+4)); vst1q_f32(d+8, vld1q_f32(s+8));
            }
        } else {
            for (int ci = 0; ci < hidden; ++ci) { float * d = xblk + static_cast<size_t>(ci) * 12; std::memcpy(d, context + static_cast<size_t>(ci) * TT + t0, sizeof(float) * m); std::memset(d + m, 0, sizeof(float) * (12 - m)); }
            for (int ci = 0; ci < 3; ++ci) { float * d = xblk + static_cast<size_t>(hidden + ci) * 12; std::memcpy(d, feats + static_cast<size_t>(ci) * TT + t0, sizeof(float) * m); std::memset(d + m, 0, sizeof(float) * (12 - m)); }
            std::fill_n(f.data() + static_cast<size_t>(hidden) * TT + t0, m, 1.0F);
            for (int n = 0; n < noise; ++n) std::memcpy(f.data() + static_cast<size_t>(hidden + 1 + n) * TT + t0, noise_in + static_cast<size_t>(n) * TT + t0, sizeof(float) * m);
        }
        gemm_8x12_cm(xblk, proj_in, proj, f.data(), T, t0, m, false);
    }
    bar.wait();

    // 2. Acoustic frame blocks: f += pre_scaled_conv1(silu(conv0(f)))
    const int ak_stride = (12 + ak - 1 + 3) & ~3;
    for (int i = 0; i < depth; ++i) {
        if (ak == 5) {
            for (int tb = lo12; tb < hi12; ++tb) { const int t0 = tb * 12, m = std::min(12, T - t0); pack_win_k5(f.data(), hidden, T, t0, tbuf); conv_k5_tile(frame_conv0[i], tbuf, h.data(), T, t0, m, 1); }
            bar.wait();
            for (int tb = lo12; tb < hi12; ++tb) { const int t0 = tb * 12, m = std::min(12, T - t0); pack_win_k5(h.data(), hidden, T, t0, tbuf); conv_k5_tile(frame_conv1[i], tbuf, f.data(), T, t0, m, 2); }
            bar.wait();
        } else {
            for (int tb = lo12; tb < hi12; ++tb) { const int t0 = tb * 12, m = std::min(12, T - t0); pack_win_general(f.data(), hidden, ak, ak_stride, T, t0, tbuf); conv_k_general_tile(frame_conv0[i], tbuf, ak_stride, h.data(), nullptr, T, t0, m, 1); }
            bar.wait();
            for (int tb = lo12; tb < hi12; ++tb) { const int t0 = tb * 12, m = std::min(12, T - t0); pack_win_general(h.data(), hidden, ak, ak_stride, T, t0, tbuf); conv_k_general_tile(frame_conv1[i], tbuf, ak_stride, f.data(), nullptr, T, t0, m, 2); }
            bar.wait();
        }
    }

    // 3. Fused embed -> L1 emb_blk -> LayerNorm -> xs
    {
        const int ek_stride = ek == 7 ? 20 : ((12 + ek - 1 + 3) & ~3);
        float * win = tbuf, * emb_blk = win + static_cast<size_t>(embed_in) * ek_stride;
        for (int tb = lo12; tb < hi12; ++tb) {
            const int t0 = tb * 12, m = std::min(12, T - t0);
            if (ek == 7) { pack_win_k7(f.data(), embed_in, T, t0, win); conv_k7_to_tile(embed, win, emb_blk); }
            else { pack_win_general(f.data(), embed_in, ek, ek_stride, T, t0, win); conv_k_general_tile(embed, win, ek_stride, nullptr, emb_blk, T, t0, m, 0); }
            ln_from_tile12(emb_blk, dim, norm_w.data(), norm_b.data(), xs.data(), T, t0, m);
        }
        bar.wait();
    }

    // 4. ConvNeXt blocks: vectorized dwconv -> mega-fused L1 (LN -> PW1+GELU -> PW2(gamma)+xs)
    {
        int clo = 0, chi = 0;
        split_range(dim, tid, nt, clo, chi);
        const int dpad = dw_kernel / 2;
        float * ln_blk = tbuf, * hid_blk = ln_blk + static_cast<size_t>(dim) * 12;
        for (const auto & blk : decoder) {
            if (dw_kernel == 7) {
                for (int c = clo; c < chi; ++c) {
                    const float * xc = xs.data() + static_cast<size_t>(c) * TT;
                    float * ec = e.data() + static_cast<size_t>(c) * TT;
                    const float * dw = blk.dw_w.data() + static_cast<size_t>(c) * 7;
                    const float db = blk.dw_b[c];
                    int t = 0;
                    const int e0 = std::min(T, 3);
                    for (; t < e0; ++t) { float acc = db; for (int k = 0; k < 7; ++k) { const int j = t + k - 3; if (j >= 0 && j < T) acc += dw[k] * xc[j]; } ec[t] = acc; }
                    if (T >= 12) {
                        const float32x4_t bv = vdupq_n_f32(db);
                        const float w0=dw[0], w1=dw[1], w2=dw[2], w3=dw[3], w4=dw[4], w5=dw[5], w6=dw[6];
                        for (; t + 9 <= T; t += 4) {
                            const float32x4_t a = vld1q_f32(xc+t-3), b = vld1q_f32(xc+t+1), c2 = vld1q_f32(xc+t+5);
                            float32x4_t acc = vfmaq_n_f32(bv, a, w0);
                            acc = vfmaq_n_f32(acc, vextq_f32(a, b, 1), w1); acc = vfmaq_n_f32(acc, vextq_f32(a, b, 2), w2);
                            acc = vfmaq_n_f32(acc, vextq_f32(a, b, 3), w3); acc = vfmaq_n_f32(acc, b, w4);
                            acc = vfmaq_n_f32(acc, vextq_f32(b, c2, 1), w5); acc = vfmaq_n_f32(acc, vextq_f32(b, c2, 2), w6);
                            vst1q_f32(ec + t, acc);
                        }
                    }
                    for (; t < T; ++t) { float acc = db; for (int k = 0; k < 7; ++k) { const int j = t + k - 3; if (j >= 0 && j < T) acc += dw[k] * xc[j]; } ec[t] = acc; }
                }
            } else {
                for (int c = clo; c < chi; ++c) {
                    const float * xc = xs.data() + static_cast<size_t>(c) * TT; float * ec = e.data() + static_cast<size_t>(c) * TT;
                    std::fill_n(ec, TT, blk.dw_b[c]);
                    for (int k = 0; k < dw_kernel; ++k) {
                        const float wk = blk.dw_w[static_cast<size_t>(c) * dw_kernel + k];
                        const int a0 = std::max(0, dpad - k), b0 = std::min(T, T + dpad - k);
                        for (int t = a0; t < b0; ++t) ec[t] += wk * xc[t + k - dpad];
                    }
                }
            }
            bar.wait();

            for (int tb = lo12; tb < hi12; ++tb) {
                const int t0 = tb * 12, m = std::min(12, T - t0);
                ln_to_tile12(e.data(), dim, T, t0, m, blk.norm_w.data(), blk.norm_b.data(), ln_blk);
                gemm_8x12_to_tile_gelu(ln_blk, dim, blk.pw1, hid_blk);
                gemm_8x12_cm(hid_blk, pw_hidden, blk.pw2, xs.data(), T, t0, m, true);
            }
            bar.wait();
        }
    }

    // 5a. Spectrum output path (decode())
    if (!fused_audio) {
        for (int tb = lo12; tb < hi12; ++tb) {
            const int t0 = tb * 12, m = std::min(12, T - t0);
            ln_to_tile12(xs.data(), dim, T, t0, m, final_w.data(), final_b.data(), tbuf);
            gemm_8x12_rows(tbuf, dim, head, out, out_dim, t0, m);
        }
        return;
    }

    // 5b. Fused final_norm -> paired head (bins 1..511) -> exp4/sincos4 -> radix-8x8x8 iFFT
    const IstftTables & tb_istft = istft_tables();
    float * fr = framed.get();
    {
        alignas(16) float out8x12[8][12];
        float32x4_t zr0[513], zi0[513], zr1[513], zi1[513], zr2[513], zi2[513];
        const float32x4_t k100 = vdupq_n_f32(100.0F);
        for (int tb = lo12; tb < hi12; ++tb) {
            const int t0 = tb * 12, m = std::min(12, T - t0);
            ln_to_tile12(xs.data(), dim, T, t0, m, final_w.data(), final_b.data(), tbuf);
            for (int g = 0; g < head_paired.ng; ++g) {
                const int fq0 = 1 + g * 4;
                head_group_8x12_f32(tbuf, dim, head_paired.wf32.data() + static_cast<size_t>(g) * dim * 8,
                                    head_paired.bias.data() + g * 8, out8x12);
                for (int off = 0; off < 4; ++off) {
                    const int fq = fq0 + off;
                    if (fq >= 512) break;
                    float32x4_t s0, cc0;
                    const float32x4_t mg0 = vminq_f32(exp4(vld1q_f32(out8x12[off])), k100);
                    sincos4(vld1q_f32(out8x12[4 + off]), &s0, &cc0);
                    zr0[fq] = vmulq_f32(mg0, cc0); zi0[fq] = vmulq_f32(mg0, s0);
                    if (m > 4) {
                        float32x4_t s1, cc1;
                        const float32x4_t mg1 = vminq_f32(exp4(vld1q_f32(out8x12[off] + 4)), k100);
                        sincos4(vld1q_f32(out8x12[4 + off] + 4), &s1, &cc1);
                        zr1[fq] = vmulq_f32(mg1, cc1); zi1[fq] = vmulq_f32(mg1, s1);
                    }
                    if (m > 8) {
                        float32x4_t s2, cc2;
                        const float32x4_t mg2 = vminq_f32(exp4(vld1q_f32(out8x12[off] + 8)), k100);
                        sincos4(vld1q_f32(out8x12[4 + off] + 8), &s2, &cc2);
                        zr2[fq] = vmulq_f32(mg2, cc2); zi2[fq] = vmulq_f32(mg2, s2);
                    }
                }
            }
            c2r1024_soa4(tb_istft, zr0, zi0, fr + static_cast<size_t>(t0) * kFft, std::min(m, 4));
            if (m > 4) c2r1024_soa4(tb_istft, zr1, zi1, fr + static_cast<size_t>(t0 + 4) * kFft, std::min(m - 4, 4));
            if (m > 8) c2r1024_soa4(tb_istft, zr2, zi2, fr + static_cast<size_t>(t0 + 8) * kFft, m - 8);
        }
    }
    bar.wait();

    // 6. Direct NEON 4-frame-gather OLA (in the same pool, zero extra thread spawns)
    const int out_blocks = T - 1;
    int blo = 0, bhi = 0;
    split_range(out_blocks, tid, nt, blo, bhi);
    const float * w = tb_istft.win;
    const float * ws = tb_istft.win_s;
    for (int b = blo; b < bhi; ++b) {
        float * dst = out + static_cast<size_t>(b) * kHop;
        if (b >= 1 && b + 2 <= T - 1) {
            const float * f0 = fr + static_cast<size_t>(b - 1) * kFft + 768;
            const float * f1 = fr + static_cast<size_t>(b) * kFft + 512;
            const float * f2 = fr + static_cast<size_t>(b + 1) * kFft + 256;
            const float * f3 = fr + static_cast<size_t>(b + 2) * kFft;
            for (int j = 0; j < kHop; j += 4) {
                float32x4_t acc = vmulq_f32(vld1q_f32(f0 + j), vld1q_f32(ws + 768 + j));
                acc = vfmaq_f32(acc, vld1q_f32(f1 + j), vld1q_f32(ws + 512 + j));
                acc = vfmaq_f32(acc, vld1q_f32(f2 + j), vld1q_f32(ws + 256 + j));
                acc = vfmaq_f32(acc, vld1q_f32(f3 + j), vld1q_f32(ws + j));
                vst1q_f32(dst + j, acc);
            }
        } else {
            const int flo = std::max(0, b - 1), fhi = std::min(T - 1, b + 2);
            for (int j = 0; j < kHop; ++j) {
                float acc = 0.0F, env = 0.0F;
                for (int fk = flo; fk <= fhi; ++fk) {
                    const int o = (b + 2 - fk) * kHop + j;
                    acc += w[o] * fr[static_cast<size_t>(fk) * kFft + o];
                    env += w[o] * w[o];
                }
                dst[j] = acc / env;
            }
        }
    }
    bar.wait();

    // 7. Single-pass 4-way unrolled DC blocker on tid 0: ~60 us per sentence and
    //    guarantees bit-for-bit identical output across any thread count.
    if (tid == 0) {
        dc_block_4way(out, static_cast<size_t>(out_blocks) * kHop);
    }
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
    p.ak = static_cast<int>(config.acoustic_kernel);
    p.ek = static_cast<int>(config.embed_kernel);
    p.dim = static_cast<int>(config.dim);
    p.pw_hidden = static_cast<int>(config.pw_hidden);
    p.blocks = static_cast<int>(config.blocks);
    p.dw_kernel = static_cast<int>(config.dw_kernel);
    p.n_fft = static_cast<int>(config.n_fft);
    p.hop_length = static_cast<int>(config.hop_length);
    p.out_dim = p.n_fft + 2;
    const int mels = static_cast<int>(config.mels);
    if (p.hidden <= 0 || p.dim <= 0 || p.pw_hidden <= 0 || p.blocks <= 0 || mels <= 0 ||
        p.ak <= 0 || p.ak > 33 || p.ek <= 0 || p.ek > 33 || p.dw_kernel <= 0) {
        throw std::runtime_error("sanoTTS NEON decoder: unsupported config");
    }

    auto expect = [&](const std::string & name, size_t size) {
        auto v = read(name);
        if (v.size() != size) throw std::runtime_error("sanoTTS NEON decoder: unexpected size for " + name);
        return v;
    };

    {
        const auto w = expect("acoustic.frame_input_proj.weight", static_cast<size_t>(p.hidden) * p.proj_in);
        const auto b = expect("acoustic.frame_input_proj.bias", p.hidden);
        p.proj = pack_conv_scaled(w.data(), b.data(), p.proj_in, p.hidden, 1);
    }
    for (int i = 0; i < p.depth; ++i) {
        const std::string q = "acoustic.frame_blocks." + std::to_string(i);
        const size_t n = static_cast<size_t>(p.hidden) * p.hidden * p.ak;
        const auto w0 = expect(q + ".net.0.weight", n), b0 = expect(q + ".net.0.bias", p.hidden);
        const auto w2 = expect(q + ".net.2.weight", n), b2 = expect(q + ".net.2.bias", p.hidden);
        const auto s = read(q + ".scale");
        if (s.size() != 1 && s.size() != static_cast<size_t>(p.hidden)) {
            throw std::runtime_error("sanoTTS NEON decoder: unexpected size for " + q + ".scale");
        }
        p.frame_conv0.push_back(pack_conv_scaled(w0.data(), b0.data(), p.hidden, p.hidden, p.ak));
        p.frame_conv1.push_back(pack_conv_scaled(w2.data(), b2.data(), p.hidden, p.hidden, p.ak, s.data(), s.size()));
    }

    {
        const auto wo = expect("acoustic.output.weight", static_cast<size_t>(mels) * p.hidden);
        const auto bo = expect("acoustic.output.bias", mels);
        const auto we = expect("decoder.embed.weight", static_cast<size_t>(p.dim) * mels * p.ek);
        const auto be = expect("decoder.embed.bias", p.dim);
        const auto wn = expect("decoder.noise_adapter.weight", static_cast<size_t>(p.dim) * p.noise * p.ek);
        const auto bn = expect("decoder.noise_adapter.bias", p.dim);
        std::vector<float> w(static_cast<size_t>(p.dim) * p.embed_in * p.ek, 0.0F);
        std::vector<float> b(p.dim);
        const int h1 = p.hidden + 1;
        auto fold_range = [&](int o_lo, int o_hi) {
            std::vector<double> acc(static_cast<size_t>(p.ek) * h1);
            for (int o = o_lo; o < o_hi; ++o) {
                std::fill(acc.begin(), acc.end(), 0.0);
                for (int m = 0; m < mels; ++m) {
                    const float * wrow = wo.data() + static_cast<size_t>(m) * p.hidden;
                    for (int k = 0; k < p.ek; ++k) {
                        const double e = we[(static_cast<size_t>(o) * mels + m) * p.ek + k];
                        double * ak_acc = acc.data() + static_cast<size_t>(k) * h1;
                        for (int c = 0; c < p.hidden; ++c) ak_acc[c] += e * wrow[c];
                        ak_acc[p.hidden] += e * bo[m];
                    }
                }
                for (int k = 0; k < p.ek; ++k) {
                    for (int c = 0; c < h1; ++c) w[(static_cast<size_t>(o) * p.embed_in + c) * p.ek + k] = static_cast<float>(acc[static_cast<size_t>(k) * h1 + c]);
                    for (int c = 0; c < p.noise; ++c) w[(static_cast<size_t>(o) * p.embed_in + h1 + c) * p.ek + k] = wn[(static_cast<size_t>(o) * p.noise + c) * p.ek + k];
                }
                b[o] = static_cast<float>(static_cast<double>(be[o]) + bn[o]);
            }
        };
        const int nt = std::clamp(threads, 1, std::max(1, p.dim));
        std::vector<std::thread> pool;
        for (int t = 1; t < nt; ++t) pool.emplace_back(fold_range, p.dim * t / nt, p.dim * (t + 1) / nt);
        fold_range(0, p.dim / nt);
        for (auto & t : pool) t.join();
        p.embed = pack_conv_scaled(w.data(), b.data(), p.embed_in, p.dim, p.ek);
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
        const auto gamma = expect(q + ".gamma", p.dim);
        blk.pw1 = pack_conv_scaled(w1.data(), b1.data(), p.dim, p.pw_hidden, 1);
        blk.pw2 = pack_conv_scaled(w2.data(), b2.data(), p.pw_hidden, p.dim, 1, gamma.data(), gamma.size());
        p.decoder.push_back(std::move(blk));
    }
    p.final_w = expect("decoder.final_norm.weight", p.dim);
    p.final_b = expect("decoder.final_norm.bias", p.dim);
    {
        const auto w = expect("decoder.head.weight", static_cast<size_t>(p.out_dim) * p.dim);
        const auto b = expect("decoder.head.bias", p.out_dim);
        p.head = pack_conv_scaled(w.data(), b.data(), p.dim, p.out_dim, 1);
        if (p.n_fft == kFft && p.hop_length == kHop) {
            p.head_paired = pack_head_paired(w.data(), b.data(), p.dim);
            (void)istft_tables();
        }
    }

    const size_t dim8 = static_cast<size_t>((p.dim + 7) & ~7);
    const size_t pw8 = static_cast<size_t>((p.pw_hidden + 7) & ~7);
    const size_t ak_stride = static_cast<size_t>(p.ak == 5 ? 16 : ((12 + p.ak - 1 + 3) & ~3));
    const size_t ek_stride = static_cast<size_t>(p.ek == 7 ? 20 : ((12 + p.ek - 1 + 3) & ~3));
    const size_t need = std::max({
        static_cast<size_t>(p.proj_in) * 12,
        static_cast<size_t>(p.hidden) * ak_stride,
        static_cast<size_t>(p.embed_in) * ek_stride + dim8 * 12,
        static_cast<size_t>(p.dim) * 12 + pw8 * 12,
    });
    p.per_thread_scratch = (need + 15) & ~static_cast<size_t>(15);
}

SanoTtsNeonDecoder::~SanoTtsNeonDecoder() = default;

void SanoTtsNeonDecoder::init_frontend(const SanoTtsConfig & config, const WeightReader & read) {
    auto & p = *impl_;
    auto expect = [&](const std::string & name, size_t size) {
        auto v = read(name);
        if (v.size() != size) throw std::runtime_error("sanoTTS NEON front-end: unexpected size for " + name);
        return v;
    };
    auto load_stage = [&](FrontStage & st, const std::string & emb_name, const std::string & proj_name,
                          const std::string & blk_prefix, const char * out_name,
                          int vocab, int hidden, int feat_rows, int depth, int kernel) {
        if (vocab <= 0 || hidden <= 0 || depth < 0 || kernel <= 0 || kernel > 33) {
            throw std::runtime_error("sanoTTS NEON front-end: unsupported config");
        }
        st.vocab = vocab; st.hidden = hidden; st.feat_rows = feat_rows; st.depth = depth; st.kernel = kernel;
        st.has_out = (out_name != nullptr);
        st.emb = expect(emb_name, static_cast<size_t>(vocab) * hidden);
        const auto pw = expect(proj_name + ".weight", static_cast<size_t>(hidden) * (hidden + feat_rows));
        const auto pb = expect(proj_name + ".bias", hidden);
        st.proj = pack_conv_scaled(pw.data(), pb.data(), hidden + feat_rows, hidden, 1);
        st.b0.clear(); st.b1.clear();
        for (int i = 0; i < depth; ++i) {
            const std::string q = blk_prefix + std::to_string(i);
            const size_t n = static_cast<size_t>(hidden) * hidden * kernel;
            const auto w0 = expect(q + ".net.0.weight", n), b0 = expect(q + ".net.0.bias", hidden);
            const auto w2 = expect(q + ".net.2.weight", n), b2 = expect(q + ".net.2.bias", hidden);
            const auto sc = read(q + ".scale");
            if (sc.size() != 1 && sc.size() != static_cast<size_t>(hidden)) {
                throw std::runtime_error("sanoTTS NEON front-end: unexpected size for " + q + ".scale");
            }
            st.b0.push_back(pack_conv_scaled(w0.data(), b0.data(), hidden, hidden, kernel));
            st.b1.push_back(pack_conv_scaled(w2.data(), b2.data(), hidden, hidden, kernel, sc.data(), sc.size()));
        }
        if (st.has_out) {
            const auto ow = expect(std::string(out_name) + ".weight", hidden);
            const auto ob = expect(std::string(out_name) + ".bias", 1);
            st.out_proj = pack_conv_scaled(ow.data(), ob.data(), hidden, 1, 1);
        }
    };
    load_stage(p.dur_stage, "duration.embedding.weight", "duration.input_proj", "duration.blocks.",
               "duration.output", static_cast<int>(config.vocab_size), static_cast<int>(config.duration_hidden),
               3, static_cast<int>(config.duration_depth), static_cast<int>(config.duration_kernel));
    load_stage(p.aco_stage, "acoustic.embedding.weight", "acoustic.token_input_proj", "acoustic.token_blocks.",
               nullptr, static_cast<int>(config.vocab_size), static_cast<int>(config.acoustic_hidden),
               2, static_cast<int>(config.acoustic_token_depth), static_cast<int>(config.acoustic_kernel));
    p.frontend_ready = true;
}

bool SanoTtsNeonDecoder::has_frontend() const { return impl_->frontend_ready; }
bool SanoTtsNeonDecoder::can_decode_audio() const {
    return impl_->n_fft == kFft && impl_->hop_length == kHop;
}

std::vector<float> SanoTtsNeonDecoder::predict_duration(const std::vector<int32_t> & tokens,
                                                        const std::vector<float> & feats, int threads) {
    auto & p = *impl_;
    std::vector<float> out(tokens.size());
    p.run_front(p.dur_stage, tokens, feats, out.data(), threads);
    return out;
}

std::vector<float> SanoTtsNeonDecoder::predict_acoustic(const std::vector<int32_t> & tokens,
                                                        const std::vector<float> & feats, int threads) {
    auto & p = *impl_;
    std::vector<float> out(static_cast<size_t>(p.aco_stage.hidden) * tokens.size());
    p.run_front(p.aco_stage, tokens, feats, out.data(), threads);
    return out;
}

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
    p.nb12 = (p.frames + 11) / 12;
    const int nt = std::clamp(threads, 1, 16);

    auto grow = [](std::vector<float> & v, size_t n) { if (v.size() < n) v.resize(n); };
    grow(p.f, static_cast<size_t>(p.embed_in) * T);
    grow(p.h, static_cast<size_t>(p.hidden) * T);
    grow(p.xs, static_cast<size_t>(p.dim) * T);
    grow(p.e, static_cast<size_t>(p.dim) * T);
    grow(p.scratch, static_cast<size_t>(nt) * p.per_thread_scratch);

    std::vector<float> out(T * static_cast<size_t>(p.out_dim));
    Barrier bar(nt);
    p.pool.run(nt, [&](int tid, int count) {
        p.worker(context.data(), feats.data(), noise.data(), out.data(), false, tid, count, bar);
    });
    return out;
}

std::vector<float> SanoTtsNeonDecoder::decode_audio(const std::vector<float> & context,
                                                    const std::vector<float> & feats,
                                                    const std::vector<float> & noise,
                                                    int64_t frames, int threads) {
    auto & p = *impl_;
    const auto T = static_cast<size_t>(frames);
    if (frames < 2 || frames > (1 << 24) || !can_decode_audio() ||
        context.size() != p.hidden * T || feats.size() != 3 * T || noise.size() != p.noise * T) {
        throw std::runtime_error("sanoTTS NEON decoder: invalid decode_audio input sizes");
    }
    p.frames = static_cast<int>(frames);
    p.nb12 = (p.frames + 11) / 12;
    const int nt = std::clamp(threads, 1, 16);

    auto grow = [](std::vector<float> & v, size_t n) { if (v.size() < n) v.resize(n); };
    grow(p.f, static_cast<size_t>(p.embed_in) * T);
    grow(p.h, static_cast<size_t>(p.hidden) * T);
    grow(p.xs, static_cast<size_t>(p.dim) * T);
    grow(p.e, static_cast<size_t>(p.dim) * T);
    grow(p.scratch, static_cast<size_t>(nt) * p.per_thread_scratch);
    const size_t need_fr = T * kFft;
    if (p.framed_cap < need_fr) {
        p.framed.reset(new float[need_fr]);
        p.framed_cap = need_fr;
    }

    std::vector<float> samples((T - 1) * kHop);
    Barrier bar(nt);
    p.pool.run(nt, [&](int tid, int count) {
        p.worker(context.data(), feats.data(), noise.data(), samples.data(), true, tid, count, bar);
    });
    return samples;
}

}  // namespace engine::models::sanotts

#else  // no AArch64 NEON

namespace engine::models::sanotts {
struct SanoTtsNeonDecoder::Impl {};
bool SanoTtsNeonDecoder::available() { return false; }
SanoTtsNeonDecoder::SanoTtsNeonDecoder(const SanoTtsConfig &, const WeightReader &, int) {
    throw std::runtime_error("sanotts.exec_plan.decoder: impl neon needs an AArch64 build with NEON; select ggml instead");
}
SanoTtsNeonDecoder::~SanoTtsNeonDecoder() = default;
void SanoTtsNeonDecoder::init_frontend(const SanoTtsConfig &, const WeightReader &) {
    throw std::runtime_error("sanoTTS NEON front-end is not available in this build");
}
bool SanoTtsNeonDecoder::has_frontend() const { return false; }
bool SanoTtsNeonDecoder::can_decode_audio() const { return false; }
std::vector<float> SanoTtsNeonDecoder::predict_duration(const std::vector<int32_t> &, const std::vector<float> &, int) {
    throw std::runtime_error("sanoTTS NEON front-end is not available in this build");
}
std::vector<float> SanoTtsNeonDecoder::predict_acoustic(const std::vector<int32_t> &, const std::vector<float> &, int) {
    throw std::runtime_error("sanoTTS NEON front-end is not available in this build");
}
std::vector<float> SanoTtsNeonDecoder::decode(const std::vector<float> &, const std::vector<float> &, const std::vector<float> &, int64_t, int) {
    throw std::runtime_error("sanoTTS NEON decoder is not available in this build");
}
std::vector<float> SanoTtsNeonDecoder::decode_audio(const std::vector<float> &, const std::vector<float> &, const std::vector<float> &, int64_t, int) {
    throw std::runtime_error("sanoTTS NEON decoder is not available in this build");
}
}  // namespace engine::models::sanotts

#endif
