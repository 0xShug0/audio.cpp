// Native C++ port of the Earshot VAD network (https://github.com/pykeio/earshot,
// MIT OR Apache-2.0). The arithmetic follows upstream src/lib.rs and
// src/default_predictor.rs; scores match the Rust crate up to float rounding
// (FFT order, logf, mel sum order). The integer layers are exact.
//
// The float kernels live in kernels.inc and are compiled once per instruction
// set (baseline SSE2 or NEON, AVX2, AVX-512) and picked at run time. They apply
// the same operations to every value in the same order, so all instruction
// sets, compilers and thread counts give the same bits.

#include "engine/community_models/earshot_vad/runtime.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string_view>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define EARSHOT_SSE2 1
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
// AVX2 and AVX-512 kernels are compiled for their targets and picked at run time.
#define EARSHOT_X86_DISPATCH 1
#endif
#elif defined(__ARM_NEON) || defined(_M_ARM64)
#include <arm_neon.h>
#define EARSHOT_NEON 1
#endif

#if defined(EARSHOT_X86_DISPATCH) && defined(__clang__)
#define EARSHOT_PUSH_AVX2 _Pragma("clang attribute push(__attribute__((target(\"avx2\"))), apply_to = function)")
#define EARSHOT_PUSH_AVX512 \
    _Pragma("clang attribute push(__attribute__((target(\"avx2,avx512f,avx512bw,avx512vnni\"))), apply_to = function)")
#define EARSHOT_POP_TARGET _Pragma("clang attribute pop")
#elif defined(EARSHOT_X86_DISPATCH) && defined(__GNUC__)
#define EARSHOT_PUSH_AVX2 _Pragma("GCC push_options") _Pragma("GCC target(\"avx2\")")
#define EARSHOT_PUSH_AVX512 _Pragma("GCC push_options") _Pragma("GCC target(\"avx2,avx512f,avx512bw,avx512vnni\")")
#define EARSHOT_POP_TARGET _Pragma("GCC pop_options")
#else
// MSVC compiles AVX intrinsics without target flags.
#define EARSHOT_PUSH_AVX2
#define EARSHOT_PUSH_AVX512
#define EARSHOT_POP_TARGET
#endif

namespace engine::community_models::earshot_vad {
namespace {

constexpr float kPreEmphasis = 0.97f;
constexpr float kPowerScale = 1.0f / (32768.0f * 32768.0f);
constexpr int64_t kDepthwiseCols = kMels - 2;          // 38
constexpr int64_t kPooledCols = (kDepthwiseCols - 3) / 2 + 1;  // 18
constexpr int64_t kChannels = 16;
constexpr int64_t kLayer2Cols = 9;
constexpr int64_t kLayer3Cols = 5;
constexpr int64_t kRnnInput = kLayer3Cols * kChannels;  // 80
constexpr int64_t kHidden = 64;
constexpr int64_t kMelsPadded = 48;  // whole vectors of any width
constexpr int32_t kSqrtHalfBits = 0x3f3504f3;  // bits of sqrt(1/2)

// ---------------------------------------------------------------- weights

template <typename T>
void copy_tensor(
    const assets::TensorSource & source,
    std::string_view name,
    std::string_view dtype,
    size_t count,
    T * out) {
    const auto raw = source.require_tensor_data(name);
    if (raw.metadata.dtype != dtype) {
        throw std::runtime_error(
            "Earshot VAD tensor " + std::string(name) + " has dtype " + raw.metadata.dtype +
            ", expected " + std::string(dtype));
    }
    if (raw.bytes.size() != count * sizeof(T)) {
        throw std::runtime_error("Earshot VAD tensor size mismatch: " + std::string(name));
    }
    std::memcpy(out, raw.bytes.data(), raw.bytes.size());
}

template <typename T, size_t N>
void copy_tensor(const assets::TensorSource & source, std::string_view name, std::string_view dtype, std::array<T, N> & out) {
    copy_tensor(source, name, dtype, N, out.data());
}

EarshotWeights load_earshot_weights(const std::filesystem::path & checkpoint_path) {
    const auto source = assets::open_tensor_source(checkpoint_path);
    EarshotWeights weights;
    copy_tensor(*source, "norm_weight", "F32", weights.norm_weight);
    copy_tensor(*source, "layer1_kernel", "F32", weights.layer1_kernel);
    copy_tensor(*source, "layer1_weight", "F32", weights.layer1_weight);
    copy_tensor(*source, "layer1_bias", "F32", weights.layer1_bias);
    copy_tensor(*source, "layer2_kernel", "F32", weights.layer2_kernel);
    copy_tensor(*source, "layer2_weight", "F32", weights.layer2_weight);
    copy_tensor(*source, "layer2_bias", "F32", weights.layer2_bias);
    copy_tensor(*source, "layer3_kernel", "F32", weights.layer3_kernel);
    copy_tensor(*source, "layer3_weight", "F32", weights.layer3_weight);
    for (int64_t oc = 0; oc < kChannels; ++oc) {
        for (int64_t c = 0; c < kChannels; ++c) {
            weights.layer2_weight_t[c * kChannels + oc] = weights.layer2_weight[oc * kChannels + c];
            weights.layer3_weight_t[c * kChannels + oc] = weights.layer3_weight[oc * kChannels + c];
        }
        for (int64_t kw = 0; kw < 3; ++kw) {
            weights.layer2_kernel_t[kw * kChannels + oc] = weights.layer2_kernel[oc * 3 + kw];
            weights.layer3_kernel_t[kw * kChannels + oc] = weights.layer3_kernel[oc * 3 + kw];
        }
    }
    const auto pair_layout = [](const std::vector<int16_t> & rows, int64_t in_dim) {
        std::vector<int16_t> pairs(rows.size());
        for (int64_t r = 0; r < 2 * kHidden; ++r) {
            for (int64_t i = 0; i < in_dim; ++i) {
                pairs[((i / 2) * 2 * kHidden + r) * 2 + (i % 2)] = rows[r * in_dim + i];
            }
        }
        return pairs;
    };
    weights.rnn1_weight.resize(2 * kHidden * kRnnInput);
    copy_tensor(*source, "rnn1_weight", "I16", weights.rnn1_weight.size(), weights.rnn1_weight.data());
    weights.rnn1_pairs = pair_layout(weights.rnn1_weight, kRnnInput);
    weights.rnn2_weight.resize(2 * kHidden * kHidden);
    copy_tensor(*source, "rnn2_weight", "I16", weights.rnn2_weight.size(), weights.rnn2_weight.data());
    weights.rnn2_pairs = pair_layout(weights.rnn2_weight, kHidden);
    copy_tensor(*source, "output_weight", "I16", weights.output_weight);
    copy_tensor(*source, "hann_window", "F32", weights.hann_window);
    copy_tensor(*source, "mel_start", "I32", weights.mel_start);
    copy_tensor(*source, "mel_length", "I32", weights.mel_length);

    // Each band's coefficients are zero-padded to whole groups of eight, so the
    // mel sums run eight lanes at a time; padded bins must still be computed ones.
    int64_t total = 0;
    int64_t padded = 0;
    for (int64_t band = 0; band < kMels; ++band) {
        const int32_t start = weights.mel_start[band];
        const int32_t length = weights.mel_length[band];
        const int32_t groups = (length + 7) / 8;
        if (start < 0 || length <= 0 || start + 8 * groups > kBins) {
            throw std::runtime_error("Earshot VAD mel band is out of range");
        }
        weights.mel_offset[band] = static_cast<int32_t>(padded);
        weights.mel_groups[band] = groups;
        weights.mel_bins = std::max(weights.mel_bins, start + 8 * groups);
        total += length;
        padded += 8 * groups;
    }
    const auto coeffs = source->require_tensor_data("mel_coeffs");
    if (coeffs.metadata.dtype != "F32" || coeffs.bytes.size() != static_cast<size_t>(total) * sizeof(float)) {
        throw std::runtime_error("Earshot VAD mel_coeffs does not match mel_length");
    }
    std::vector<float> flat(static_cast<size_t>(total));
    std::memcpy(flat.data(), coeffs.bytes.data(), coeffs.bytes.size());
    weights.mel_coeffs.assign(static_cast<size_t>(padded), 0.0f);
    for (int64_t band = 0, src = 0; band < kMels; src += weights.mel_length[band], ++band) {
        std::copy_n(flat.begin() + src, weights.mel_length[band], weights.mel_coeffs.begin() + weights.mel_offset[band]);
    }
    return weights;
}

std::string cache_key(const std::filesystem::path & path) {
    std::error_code ec;
    const auto canonical = std::filesystem::weakly_canonical(path, ec);
    return ec ? path.lexically_normal().string() : canonical.string();
}

// ---------------------------------------------------------------- network

// Fixed-size real FFT: the 1024-point transform of a real window is computed
// as a 512-point complex FFT of (even, odd) sample pairs, radix-2, split
// real/imaginary arrays so every butterfly loop vectorizes. No allocation or
// locking, so it is cheap per frame and safe to run on many threads.
constexpr int64_t kHalfFft = kFftSize / 2;  // 512

struct FftTables {
    std::array<uint16_t, kHalfFft> bitrev{};
    std::array<float, kHalfFft - 1> twiddle_re{};  // stage h uses entries [h - 1, 2h - 1)
    std::array<float, kHalfFft - 1> twiddle_im{};
    std::array<float, kHalfFft + 16> post_re{};    // exp(-2 pi i k / 1024), k <= 512
    std::array<float, kHalfFft + 16> post_im{};
};

const FftTables & fft_tables() {
    static const FftTables tables = [] {
        FftTables t;
        constexpr double kPi = 3.14159265358979323846;
        for (int64_t n = 0; n < kHalfFft; ++n) {
            int64_t r = 0;
            for (int64_t bit = 1, v = n; bit < kHalfFft; bit <<= 1, v >>= 1) {
                r = (r << 1) | (v & 1);
            }
            t.bitrev[n] = static_cast<uint16_t>(r);
        }
        for (int64_t h = 1; h < kHalfFft; h <<= 1) {
            for (int64_t j = 0; j < h; ++j) {
                t.twiddle_re[h - 1 + j] = static_cast<float>(std::cos(-kPi * j / h));
                t.twiddle_im[h - 1 + j] = static_cast<float>(std::sin(-kPi * j / h));
            }
        }
        for (int64_t k = 0; k <= kHalfFft; ++k) {
            t.post_re[k] = static_cast<float>(std::cos(-2.0 * kPi * k / kFftSize));
            t.post_im[k] = static_cast<float>(std::sin(-2.0 * kPi * k / kFftSize));
        }
        return t;
    }();
    return tables;
}

// Exact int16 dot product; n is a multiple of 8.
inline int32_t dot_i16(const int16_t * a, const int16_t * b, int64_t n) {
#if defined(EARSHOT_SSE2)
    __m128i acc0 = _mm_setzero_si128();
    __m128i acc1 = _mm_setzero_si128();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm_add_epi32(acc0, _mm_madd_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i)),
                                                  _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i))));
        acc1 = _mm_add_epi32(acc1, _mm_madd_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i + 8)),
                                                  _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i + 8))));
    }
    for (; i < n; i += 8) {
        acc0 = _mm_add_epi32(acc0, _mm_madd_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i)),
                                                  _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i))));
    }
    __m128i acc = _mm_add_epi32(acc0, acc1);
    acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, _MM_SHUFFLE(1, 0, 3, 2)));
    acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(acc);
#elif defined(EARSHOT_NEON)
    int32x4_t acc0 = vdupq_n_s32(0);
    int32x4_t acc1 = vdupq_n_s32(0);
    for (int64_t i = 0; i < n; i += 8) {
        const int16x8_t x = vld1q_s16(a + i);
        const int16x8_t y = vld1q_s16(b + i);
        acc0 = vmlal_s16(acc0, vget_low_s16(x), vget_low_s16(y));
        acc1 = vmlal_s16(acc1, vget_high_s16(x), vget_high_s16(y));
    }
    const int32x4_t acc = vaddq_s32(acc0, acc1);
    return vgetq_lane_s32(acc, 0) + vgetq_lane_s32(acc, 1) + vgetq_lane_s32(acc, 2) + vgetq_lane_s32(acc, 3);
#else
    int32_t sum = 0;
    for (int64_t i = 0; i < n; ++i) {
        sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }
    return sum;
#endif
}

// minGRU layers. Q15 inputs times Q9 weights give Q24 accumulators; row L1 norms
// of the shipped weights keep every sum far below INT32_MAX. The matrix product
// depends only on the layer input, never on the layer's own state, so it runs
// batched over frames; only the elementwise update (the scan) is sequential.
//
// gemm: out[f, r] = dot(x[f], w[r]) for 128 rows; x is [frames, in_dim] and out is
// [frames, 128]. `rows` is the [128, in_dim] layout, `pairs` the [in_dim / 2, 128, 2] one.
using GemmFn = void (*)(const int16_t * x, int64_t frames, int64_t in_dim, const int16_t * rows,
                        const int16_t * pairs, int32_t * out);
// scan: h = combine(h, acc[f]) for each frame in order, for the hidden units of
// groups [first, last) (eight units each; every unit is independent of the
// others). state holds all 64 values. When out is not null, the group g values
// after frame f go to out[g * group_stride + f * 8 + (0..7)].
using ScanFn = void (*)(const int32_t * acc, int64_t frames, int first, int last, int16_t * state, int16_t * out,
                        int64_t group_stride);
constexpr int kScanGroups = kHidden / 8;

void gemm_baseline(const int16_t * x, int64_t frames, int64_t in_dim, const int16_t * rows,
                   const int16_t * /*pairs*/, int32_t * out) {
    for (int64_t f = 0; f < frames; ++f) {
        for (int64_t r = 0; r < 2 * kHidden; ++r) {
            out[f * 2 * kHidden + r] = dot_i16(x + f * in_dim, rows + r * in_dim, in_dim);
        }
    }
}

void scan_baseline(const int32_t * acc, int64_t frames, int first, int last, int16_t * state, int16_t * out,
                   int64_t group_stride) {
    for (int64_t f = 0; f < frames; ++f) {
        const int32_t * a = acc + f * 2 * kHidden;
        for (int64_t i = 8 * first; i < 8 * last; ++i) {
            // Gate x4, clamped to [0, 1] in Q24; the shift wraps like upstream's i32 `<<`.
            const int32_t shifted = static_cast<int32_t>(static_cast<uint32_t>(a[kHidden + i]) << 2);
            const int64_t gate = std::clamp<int32_t>(shifted, 0, 16777215);
            const int64_t y = (16777215 - gate) * (static_cast<int64_t>(state[i]) * 512) +
                              gate * static_cast<int64_t>(a[i]);
            state[i] = static_cast<int16_t>(std::clamp<int64_t>(y >> 33, -32768, 32767));
        }
        if (out != nullptr) {
            for (int g = first; g < last; ++g) {
                std::memcpy(out + g * group_stride + f * 8, state + 8 * g, 8 * sizeof(int16_t));
            }
        }
    }
}

// Cephes expf (relative error below 2e-7) from plain float operations, so the
// sigmoid rounds the same everywhere instead of following the C library.
float exp_portable(float x) {
    const float n = std::floor(x * 1.44269504088896341f + 0.5f);
    const float r = (x - n * 0.693359375f) - n * -2.12194440e-4f;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = p * (r * r) + r + 1.0f;
    return std::ldexp(p, static_cast<int>(n));
}

float output_score(const EarshotWeights & w, const int16_t * h1, const int16_t * h2) {
    const int32_t out = dot_i16(h1, w.output_weight.data(), kHidden) +
                        dot_i16(h2, w.output_weight.data() + kHidden, kHidden);
    // |x| < 128, so exp_portable never sees a NaN; huge values give inf and score 0.
    const float x = static_cast<float>(out) / 16777216.0f;
    return 1.0f / (1.0f + exp_portable(-x));
}

struct Kernels {
    const char * name;
    void (*features)(const EarshotWeights & w, const float * windowed, float * out);
    void (*conv)(const EarshotWeights & w, const float * context, int16_t * out);
    void (*emphasis_window)(const float * src, const float * hann, float * out);
    GemmFn gemm;
    ScanFn scan;
};

// ------------------------------------------------ baseline (SSE2, NEON or scalar)

namespace baseline {
#include "kernels.inc"
}  // namespace baseline

const Kernels kBaselineKernels = {
    "baseline",
    baseline::window_to_features,
    baseline::conv_stack,
    baseline::emphasis_window,
    gemm_baseline,
    scan_baseline,
};

#if defined(EARSHOT_X86_DISPATCH)
// ------------------------------------------------ AVX2

EARSHOT_PUSH_AVX2
namespace avx2 {

struct V8 {
    static constexpr int64_t kWidth = 8;
    __m256 v;
    static V8 load(const float * p) { return {_mm256_loadu_ps(p)}; }
    void store(float * p) const { _mm256_storeu_ps(p, v); }
    static V8 set1(float x) { return {_mm256_set1_ps(x)}; }
    static V8 zero() { return {_mm256_setzero_ps()}; }
    // std::max(a, b): returns a unless a < b.
    static V8 max_of(const V8 & a, const V8 & b) { return {_mm256_max_ps(b.v, a.v)}; }
    static V8 neg(const V8 & a) { return {_mm256_xor_ps(a.v, _mm256_set1_ps(-0.0f))}; }
    static V8 reverse(const V8 & a) { return {_mm256_permutevar8x32_ps(a.v, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0))}; }
    static void split_log2(const V8 & x, V8 & m, V8 & e) {
        const __m256i bits = _mm256_sub_epi32(_mm256_castps_si256(x.v), _mm256_set1_epi32(kSqrtHalfBits));
        e.v = _mm256_cvtepi32_ps(_mm256_srai_epi32(bits, 23));
        m.v = _mm256_castsi256_ps(
            _mm256_add_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(0x007fffff)), _mm256_set1_epi32(kSqrtHalfBits)));
    }
    friend V8 operator+(const V8 & a, const V8 & b) { return {_mm256_add_ps(a.v, b.v)}; }
    friend V8 operator-(const V8 & a, const V8 & b) { return {_mm256_sub_ps(a.v, b.v)}; }
    friend V8 operator*(const V8 & a, const V8 & b) { return {_mm256_mul_ps(a.v, b.v)}; }
};

inline void quantize_q15(const V8 & x, int16_t * out) {
    const __m256 v = _mm256_min_ps(_mm256_max_ps(x.v, _mm256_set1_ps(-32768.0f)), _mm256_set1_ps(32767.0f));
    __m256i t = _mm256_cvttps_epi32(v);
    const __m256 frac = _mm256_sub_ps(v, _mm256_cvtepi32_ps(t));
    t = _mm256_sub_epi32(t, _mm256_castps_si256(_mm256_cmp_ps(frac, _mm256_set1_ps(0.5f), _CMP_GE_OQ)));
    t = _mm256_add_epi32(t, _mm256_castps_si256(_mm256_cmp_ps(frac, _mm256_set1_ps(-0.5f), _CMP_LE_OQ)));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(out),
                     _mm_packs_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1)));
}

// DIT stages h = 4 and h = 8 over one 16-value block per step, regrouping the
// 128-bit halves in registers instead of running them four lanes wide. Same
// arithmetic as fft_stage_pair<V4>(re, im, 4).
// a, b = a + b * w, a - b * w for split complex vectors.
inline void butterfly8(__m256 & ar, __m256 & ai, __m256 & br, __m256 & bi, const __m256 & wr, const __m256 & wi) {
    const __m256 vr = _mm256_sub_ps(_mm256_mul_ps(br, wr), _mm256_mul_ps(bi, wi));
    const __m256 vi = _mm256_add_ps(_mm256_mul_ps(br, wi), _mm256_mul_ps(bi, wr));
    br = _mm256_sub_ps(ar, vr);
    bi = _mm256_sub_ps(ai, vi);
    ar = _mm256_add_ps(ar, vr);
    ai = _mm256_add_ps(ai, vi);
}

inline void fft_stage48(float * re, float * im) {
    const auto & t = fft_tables();
    const __m256 w4r = _mm256_broadcast_ps(reinterpret_cast<const __m128 *>(t.twiddle_re.data() + 3));
    const __m256 w4i = _mm256_broadcast_ps(reinterpret_cast<const __m128 *>(t.twiddle_im.data() + 3));
    const __m256 w8r = _mm256_loadu_ps(t.twiddle_re.data() + 7);
    const __m256 w8i = _mm256_loadu_ps(t.twiddle_im.data() + 7);
    for (int64_t base = 0; base < kHalfFft; base += 16) {
        const __m256 xr = _mm256_loadu_ps(re + base);
        const __m256 yr = _mm256_loadu_ps(re + base + 8);
        const __m256 xi = _mm256_loadu_ps(im + base);
        const __m256 yi = _mm256_loadu_ps(im + base + 8);
        // h = 4 pairs the halves of each 8-value block.
        __m256 ar = _mm256_permute2f128_ps(xr, yr, 0x20);
        __m256 br = _mm256_permute2f128_ps(xr, yr, 0x31);
        __m256 ai = _mm256_permute2f128_ps(xi, yi, 0x20);
        __m256 bi = _mm256_permute2f128_ps(xi, yi, 0x31);
        butterfly8(ar, ai, br, bi, w4r, w4i);
        // h = 8 pairs the two 8-value blocks.
        __m256 cr = _mm256_permute2f128_ps(ar, br, 0x20);
        __m256 dr = _mm256_permute2f128_ps(ar, br, 0x31);
        __m256 ci = _mm256_permute2f128_ps(ai, bi, 0x20);
        __m256 di = _mm256_permute2f128_ps(ai, bi, 0x31);
        butterfly8(cr, ci, dr, di, w8r, w8i);
        _mm256_storeu_ps(re + base, cr);
        _mm256_storeu_ps(re + base + 8, dr);
        _mm256_storeu_ps(im + base, ci);
        _mm256_storeu_ps(im + base + 8, di);
    }
}

// bitrev_radix4() with several 4x4 blocks per register, one per 128-bit
// chunk. The b of each chunk is chosen so that chunk s of a group lands at
// output position rev5(b) * 4 = (L * m + s) * 4, making every store one
// contiguous full-width write. Same arithmetic as bitrev_radix4().
// Row a of the bitrev_radix4() layout for two b, as real and imaginary parts.
inline void load_bitrev_row(const float * windowed, int64_t a, int64_t b0, int64_t b1, V8 & r, V8 & i) {
    const __m256 l0 = _mm256_loadu_ps(windowed + 2 * (a * 128 + b0 * 4));
    const __m256 l1 = _mm256_loadu_ps(windowed + 2 * (a * 128 + b1 * 4));
    const __m256 lo = _mm256_permute2f128_ps(l0, l1, 0x20);
    const __m256 hi = _mm256_permute2f128_ps(l0, l1, 0x31);
    r.v = _mm256_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0));
    i.v = _mm256_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 1, 3, 1));
}

// 4x4 transpose within each 128-bit half.
inline void transpose_halves(V8 & a, V8 & b, V8 & c, V8 & d) {
    const __m256 t0 = _mm256_shuffle_ps(a.v, b.v, 0x44);
    const __m256 t2 = _mm256_shuffle_ps(a.v, b.v, 0xEE);
    const __m256 t1 = _mm256_shuffle_ps(c.v, d.v, 0x44);
    const __m256 t3 = _mm256_shuffle_ps(c.v, d.v, 0xEE);
    a.v = _mm256_shuffle_ps(t0, t1, 0x88);
    b.v = _mm256_shuffle_ps(t0, t1, 0xDD);
    c.v = _mm256_shuffle_ps(t2, t3, 0x88);
    d.v = _mm256_shuffle_ps(t2, t3, 0xDD);
}

inline void bitrev_radix4_wide(const float * windowed, float * re, float * im) {
    const auto & t = fft_tables();
    const V8 zero = V8::zero();
    for (int64_t m = 0; m < 16; ++m) {
        const int64_t b0 = t.bitrev[8 * m] / 4;
        const int64_t b1 = t.bitrev[8 * m + 4] / 4;
        V8 r0, i0, r1, i1, r2, i2;
        load_bitrev_row(windowed, 0, b0, b1, r0, i0);
        load_bitrev_row(windowed, 1, b0, b1, r1, i1);
        load_bitrev_row(windowed, 2, b0, b1, r2, i2);
        const V8 t0r = r0 + r2;
        const V8 t0i = i0 + i2;
        const V8 t1r = r0 - r2;
        const V8 t1i = i0 - i2;
        const V8 t2r = r1 + zero;
        const V8 t2i = i1 + zero;
        const V8 t3r = r1 - zero;
        const V8 t3i = i1 - zero;
        V8 g0r = t0r + t2r;
        V8 g0i = t0i + t2i;
        V8 g2r = t0r - t2r;
        V8 g2i = t0i - t2i;
        V8 g1r = t1r + t3i;
        V8 g1i = t1i - t3r;
        V8 g3r = t1r - t3i;
        V8 g3i = t1i + t3r;
        transpose_halves(g0r, g1r, g2r, g3r);
        transpose_halves(g0i, g1i, g2i, g3i);
        const int64_t offset = 8 * m;
        g0r.store(re + offset);
        g0i.store(im + offset);
        g1r.store(re + 256 + offset);
        g1i.store(im + 256 + offset);
        g2r.store(re + 128 + offset);
        g2i.store(im + 128 + offset);
        g3r.store(re + 384 + offset);
        g3i.store(im + 384 + offset);
    }
}

// mel_sum() from kernels.inc with one 8-lane accumulator; same order.
inline float mel_sum(const float * bins, const float * coeffs, int32_t groups) {
    __m256 sum = _mm256_setzero_ps();
    for (int32_t g = 0; g < groups; ++g) {
        sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(bins + 8 * g), _mm256_loadu_ps(coeffs + 8 * g)));
    }
    const __m128 folded = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    const __m128 pairs = _mm_add_ps(folded, _mm_movehl_ps(folded, folded));
    return _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps(pairs, pairs, 1)));
}

using VF = V8;
#define EARSHOT_WIDE_VF 1
#define EARSHOT_STAGE48 1
#define EARSHOT_MEL8 1
#include "kernels.inc"
#undef EARSHOT_MEL8
#undef EARSHOT_STAGE48
#undef EARSHOT_WIDE_VF

// Four rows per pass over the [128, in_dim] layout; in_dim is a multiple of 16.
void gemm(const int16_t * x, int64_t frames, int64_t in_dim, const int16_t * rows, const int16_t * /*pairs*/,
          int32_t * out) {
    for (int64_t f = 0; f < frames; ++f) {
        const int16_t * xf = x + f * in_dim;
        for (int64_t r = 0; r < 2 * kHidden; r += 4) {
            const int16_t * w0 = rows + r * in_dim;
            __m256i a0 = _mm256_setzero_si256();
            __m256i a1 = _mm256_setzero_si256();
            __m256i a2 = _mm256_setzero_si256();
            __m256i a3 = _mm256_setzero_si256();
            for (int64_t i = 0; i < in_dim; i += 16) {
                const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(xf + i));
                a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(xv, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w0 + i))));
                a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(xv, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w0 + in_dim + i))));
                a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(xv, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w0 + 2 * in_dim + i))));
                a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(xv, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w0 + 3 * in_dim + i))));
            }
            const __m256i sums = _mm256_hadd_epi32(_mm256_hadd_epi32(a0, a1), _mm256_hadd_epi32(a2, a3));
            const __m128i total = _mm_add_epi32(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
            _mm_storeu_si128(reinterpret_cast<__m128i *>(out + f * 2 * kHidden + r), total);
        }
    }
}

// scan_baseline, eight hidden values per vector. The state stays in registers
// as int32; the int64 products use _mm256_mul_epi32 on even and odd lanes.
template <int N>
void scan_groups(const int32_t * acc, int64_t frames, int first, int16_t * state, int16_t * out, int64_t group_stride) {
    __m256i h[N];
    for (int i = 0; i < N; ++i) {
        const int g = first + i;
        h[i] = _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(state + 8 * g)));
    }
    const __m256i one_q24 = _mm256_set1_epi32(16777215);
    const __m256i zero = _mm256_setzero_si256();
    for (int64_t f = 0; f < frames; ++f) {
        const int32_t * a = acc + f * 2 * kHidden;
        for (int i = 0; i < N; ++i) {
            const int g = first + i;
            const __m256i cand = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + 8 * g));
            const __m256i pre = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + kHidden + 8 * g));
            const __m256i gate = _mm256_min_epi32(_mm256_max_epi32(_mm256_slli_epi32(pre, 2), zero), one_q24);
            const __m256i keep = _mm256_sub_epi32(one_q24, gate);
            const __m256i hs = _mm256_slli_epi32(h[i], 9);
            const __m256i even = _mm256_add_epi64(_mm256_mul_epi32(keep, hs), _mm256_mul_epi32(gate, cand));
            const __m256i odd = _mm256_add_epi64(
                _mm256_mul_epi32(_mm256_srli_epi64(keep, 32), _mm256_srli_epi64(hs, 32)),
                _mm256_mul_epi32(_mm256_srli_epi64(gate, 32), _mm256_srli_epi64(cand, 32)));
            // High dwords of the int64 sums are y >> 32; one more shift gives y >> 33,
            // and the saturating pack is the clamp to int16.
            const __m256i y = _mm256_srai_epi32(_mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA), 1);
            const __m128i packed = _mm_packs_epi32(_mm256_castsi256_si128(y), _mm256_extracti128_si256(y, 1));
            h[i] = _mm256_cvtepi16_epi32(packed);
            if (out != nullptr) {
                _mm_storeu_si128(reinterpret_cast<__m128i *>(out + g * group_stride + f * 8), packed);
            }
        }
    }
    for (int i = 0; i < N; ++i) {
        _mm_storeu_si128(reinterpret_cast<__m128i *>(state + 8 * (first + i)),
                         _mm_packs_epi32(_mm256_castsi256_si128(h[i]), _mm256_extracti128_si256(h[i], 1)));
    }
}

void scan(const int32_t * acc, int64_t frames, int first, int last, int16_t * state, int16_t * out,
          int64_t group_stride) {
    switch (last - first) {
        case 8: return scan_groups<8>(acc, frames, first, state, out, group_stride);
        case 4: return scan_groups<4>(acc, frames, first, state, out, group_stride);
        case 2: return scan_groups<2>(acc, frames, first, state, out, group_stride);
        default:
            for (int g = first; g < last; ++g) {
                scan_groups<1>(acc, frames, g, state, out, group_stride);
            }
    }
}

}  // namespace avx2
EARSHOT_POP_TARGET

const Kernels kAvx2Kernels = {
    "avx2",
    avx2::window_to_features,
    avx2::conv_stack,
    avx2::emphasis_window,
    avx2::gemm,
    avx2::scan,
};

// ------------------------------------------------ AVX-512 (F, BW, VNNI)

EARSHOT_PUSH_AVX512
namespace avx512 {

struct V16 {
    static constexpr int64_t kWidth = 16;
    __m512 v;
    static V16 load(const float * p) { return {_mm512_loadu_ps(p)}; }
    void store(float * p) const { _mm512_storeu_ps(p, v); }
    static V16 set1(float x) { return {_mm512_set1_ps(x)}; }
    static V16 zero() { return {_mm512_setzero_ps()}; }
    static V16 max_of(const V16 & a, const V16 & b) { return {_mm512_max_ps(b.v, a.v)}; }
    static V16 neg(const V16 & a) {
        return {_mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a.v), _mm512_set1_epi32(INT32_MIN)))};
    }
    static V16 reverse(const V16 & a) {
        return {_mm512_permutexvar_ps(_mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0), a.v)};
    }
    static void split_log2(const V16 & x, V16 & m, V16 & e) {
        const __m512i bits = _mm512_sub_epi32(_mm512_castps_si512(x.v), _mm512_set1_epi32(kSqrtHalfBits));
        e.v = _mm512_cvtepi32_ps(_mm512_srai_epi32(bits, 23));
        m.v = _mm512_castsi512_ps(
            _mm512_add_epi32(_mm512_and_si512(bits, _mm512_set1_epi32(0x007fffff)), _mm512_set1_epi32(kSqrtHalfBits)));
    }
    friend V16 operator+(const V16 & a, const V16 & b) { return {_mm512_add_ps(a.v, b.v)}; }
    friend V16 operator-(const V16 & a, const V16 & b) { return {_mm512_sub_ps(a.v, b.v)}; }
    friend V16 operator*(const V16 & a, const V16 & b) { return {_mm512_mul_ps(a.v, b.v)}; }
};

inline void quantize_q15(const V16 & x, int16_t * out) {
    const __m512 v = _mm512_min_ps(_mm512_max_ps(x.v, _mm512_set1_ps(-32768.0f)), _mm512_set1_ps(32767.0f));
    __m512i t = _mm512_cvttps_epi32(v);
    const __m512 frac = _mm512_sub_ps(v, _mm512_cvtepi32_ps(t));
    const __m512i one = _mm512_set1_epi32(1);
    t = _mm512_mask_add_epi32(t, _mm512_cmp_ps_mask(frac, _mm512_set1_ps(0.5f), _CMP_GE_OQ), t, one);
    t = _mm512_mask_sub_epi32(t, _mm512_cmp_ps_mask(frac, _mm512_set1_ps(-0.5f), _CMP_LE_OQ), t, one);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out), _mm512_cvtsepi32_epi16(t));
}

inline void butterfly16(__m512 & ar, __m512 & ai, __m512 & br, __m512 & bi, const __m512 & wr, const __m512 & wi) {
    const __m512 vr = _mm512_sub_ps(_mm512_mul_ps(br, wr), _mm512_mul_ps(bi, wi));
    const __m512 vi = _mm512_add_ps(_mm512_mul_ps(br, wi), _mm512_mul_ps(bi, wr));
    br = _mm512_sub_ps(ar, vr);
    bi = _mm512_sub_ps(ai, vi);
    ar = _mm512_add_ps(ar, vr);
    ai = _mm512_add_ps(ai, vi);
}

// DIT stages h = 4 and h = 8 over two 16-value blocks per step, regrouping
// 128-bit chunks with shuffle_f32x4. Same arithmetic as fft_stage_pair<V4>(re, im, 4).
inline void fft_stage48(float * re, float * im) {
    const auto & t = fft_tables();
    const __m512 w4r = _mm512_broadcast_f32x4(_mm_loadu_ps(t.twiddle_re.data() + 3));
    const __m512 w4i = _mm512_broadcast_f32x4(_mm_loadu_ps(t.twiddle_im.data() + 3));
    // Chunks (w8[0..3], w8[0..3], w8[4..7], w8[4..7]) for the (X0, Y0, X1, Y1) order below.
    const __m512 t8r = _mm512_loadu_ps(t.twiddle_re.data() + 7);
    const __m512 t8i = _mm512_loadu_ps(t.twiddle_im.data() + 7);
    const __m512 w8r = _mm512_shuffle_f32x4(t8r, t8r, _MM_SHUFFLE(1, 1, 0, 0));
    const __m512 w8i = _mm512_shuffle_f32x4(t8i, t8i, _MM_SHUFFLE(1, 1, 0, 0));
    constexpr int kEven = _MM_SHUFFLE(2, 0, 2, 0);
    constexpr int kOdd = _MM_SHUFFLE(3, 1, 3, 1);
    for (int64_t base = 0; base < kHalfFft; base += 32) {
        // Chunks of four: X = (X0, X1, X2, X3), Y = (Y0, Y1, Y2, Y3).
        const __m512 xr = _mm512_loadu_ps(re + base);
        const __m512 yr = _mm512_loadu_ps(re + base + 16);
        const __m512 xi = _mm512_loadu_ps(im + base);
        const __m512 yi = _mm512_loadu_ps(im + base + 16);
        // h = 4: (X0, X2, Y0, Y2) with (X1, X3, Y1, Y3).
        __m512 ar = _mm512_shuffle_f32x4(xr, yr, kEven);
        __m512 br = _mm512_shuffle_f32x4(xr, yr, kOdd);
        __m512 ai = _mm512_shuffle_f32x4(xi, yi, kEven);
        __m512 bi = _mm512_shuffle_f32x4(xi, yi, kOdd);
        butterfly16(ar, ai, br, bi, w4r, w4i);
        // h = 8: (X0, Y0, X1, Y1) with (X2, Y2, X3, Y3).
        __m512 cr = _mm512_shuffle_f32x4(ar, br, kEven);
        __m512 dr = _mm512_shuffle_f32x4(ar, br, kOdd);
        __m512 ci = _mm512_shuffle_f32x4(ai, bi, kEven);
        __m512 di = _mm512_shuffle_f32x4(ai, bi, kOdd);
        butterfly16(cr, ci, dr, di, w8r, w8i);
        // Back to (X0, X1, X2, X3) and (Y0, Y1, Y2, Y3).
        _mm512_storeu_ps(re + base, _mm512_shuffle_f32x4(cr, dr, kEven));
        _mm512_storeu_ps(re + base + 16, _mm512_shuffle_f32x4(cr, dr, kOdd));
        _mm512_storeu_ps(im + base, _mm512_shuffle_f32x4(ci, di, kEven));
        _mm512_storeu_ps(im + base + 16, _mm512_shuffle_f32x4(ci, di, kOdd));
    }
}

// bitrev_radix4() with several 4x4 blocks per register, one per 128-bit
// chunk. The b of each chunk is chosen so that chunk s of a group lands at
// output position rev5(b) * 4 = (L * m + s) * 4, making every store one
// contiguous full-width write. Same arithmetic as bitrev_radix4().
// Eight floats at lo and eight at hi as one vector.
inline __m512 load_pair(const float * lo, const float * hi) {
    return _mm512_castpd_ps(_mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_castps_pd(_mm256_loadu_ps(lo))),
                                               _mm256_castps_pd(_mm256_loadu_ps(hi)), 1));
}

// Row a of the bitrev_radix4() layout for four b, as real and imaginary parts.
inline void load_bitrev_row(const float * windowed, int64_t a, const int64_t * b, V16 & r, V16 & i) {
    const __m512i even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i odd = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    const float * row = windowed + 2 * (a * 128);
    const __m512 l0 = load_pair(row + 8 * b[0], row + 8 * b[1]);
    const __m512 l1 = load_pair(row + 8 * b[2], row + 8 * b[3]);
    r.v = _mm512_permutex2var_ps(l0, even, l1);
    i.v = _mm512_permutex2var_ps(l0, odd, l1);
}

// 4x4 transpose within each 128-bit chunk.
inline void transpose_chunks(V16 & a, V16 & b, V16 & c, V16 & d) {
    const __m512 t0 = _mm512_shuffle_ps(a.v, b.v, 0x44);
    const __m512 t2 = _mm512_shuffle_ps(a.v, b.v, 0xEE);
    const __m512 t1 = _mm512_shuffle_ps(c.v, d.v, 0x44);
    const __m512 t3 = _mm512_shuffle_ps(c.v, d.v, 0xEE);
    a.v = _mm512_shuffle_ps(t0, t1, 0x88);
    b.v = _mm512_shuffle_ps(t0, t1, 0xDD);
    c.v = _mm512_shuffle_ps(t2, t3, 0x88);
    d.v = _mm512_shuffle_ps(t2, t3, 0xDD);
}

inline void bitrev_radix4_wide(const float * windowed, float * re, float * im) {
    const auto & t = fft_tables();
    const V16 zero = V16::zero();
    for (int64_t m = 0; m < 8; ++m) {
        const int64_t b[4] = {t.bitrev[16 * m] / 4, t.bitrev[16 * m + 4] / 4, t.bitrev[16 * m + 8] / 4,
                              t.bitrev[16 * m + 12] / 4};
        V16 r0, i0, r1, i1, r2, i2;
        load_bitrev_row(windowed, 0, b, r0, i0);
        load_bitrev_row(windowed, 1, b, r1, i1);
        load_bitrev_row(windowed, 2, b, r2, i2);
        const V16 t0r = r0 + r2;
        const V16 t0i = i0 + i2;
        const V16 t1r = r0 - r2;
        const V16 t1i = i0 - i2;
        const V16 t2r = r1 + zero;
        const V16 t2i = i1 + zero;
        const V16 t3r = r1 - zero;
        const V16 t3i = i1 - zero;
        V16 g0r = t0r + t2r;
        V16 g0i = t0i + t2i;
        V16 g2r = t0r - t2r;
        V16 g2i = t0i - t2i;
        V16 g1r = t1r + t3i;
        V16 g1i = t1i - t3r;
        V16 g3r = t1r - t3i;
        V16 g3i = t1i + t3r;
        transpose_chunks(g0r, g1r, g2r, g3r);
        transpose_chunks(g0i, g1i, g2i, g3i);
        const int64_t offset = 16 * m;
        g0r.store(re + offset);
        g0i.store(im + offset);
        g1r.store(re + 256 + offset);
        g1i.store(im + 256 + offset);
        g2r.store(re + 128 + offset);
        g2i.store(im + 128 + offset);
        g3r.store(re + 384 + offset);
        g3i.store(im + 384 + offset);
    }
}

// mel_sum() from kernels.inc with one 8-lane accumulator; same order.
inline float mel_sum(const float * bins, const float * coeffs, int32_t groups) {
    __m256 sum = _mm256_setzero_ps();
    for (int32_t g = 0; g < groups; ++g) {
        sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(bins + 8 * g), _mm256_loadu_ps(coeffs + 8 * g)));
    }
    const __m128 folded = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    const __m128 pairs = _mm_add_ps(folded, _mm_movehl_ps(folded, folded));
    return _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps(pairs, pairs, 1)));
}

using VF = V16;
#define EARSHOT_WIDE_VF 1
#define EARSHOT_STAGE48 1
#define EARSHOT_MEL8 1
#include "kernels.inc"
#undef EARSHOT_MEL8
#undef EARSHOT_STAGE48
#undef EARSHOT_WIDE_VF

// VNNI over the [in_dim / 2, 128, 2] layout: one broadcast input pair times
// 16 rows per instruction, two frames per pass so each weight load is shared.
void gemm(const int16_t * x, int64_t frames, int64_t in_dim, const int16_t * /*rows*/, const int16_t * pairs,
          int32_t * out) {
    constexpr int kParts = 2 * kHidden / 16;
    const int64_t pair_count = in_dim / 2;
    int64_t f = 0;
    for (; f + 2 <= frames; f += 2) {
        const int16_t * x0 = x + f * in_dim;
        const int16_t * x1 = x0 + in_dim;
        __m512i a[kParts];
        __m512i b[kParts];
        for (int j = 0; j < kParts; ++j) {
            a[j] = _mm512_setzero_si512();
            b[j] = _mm512_setzero_si512();
        }
        for (int64_t p = 0; p < pair_count; ++p) {
            int32_t v0;
            int32_t v1;
            std::memcpy(&v0, x0 + 2 * p, sizeof(v0));
            std::memcpy(&v1, x1 + 2 * p, sizeof(v1));
            const __m512i s0 = _mm512_set1_epi32(v0);
            const __m512i s1 = _mm512_set1_epi32(v1);
            const int16_t * wp = pairs + p * 4 * kHidden;
            for (int j = 0; j < kParts; ++j) {
                const __m512i wv = _mm512_loadu_si512(wp + j * 32);
                a[j] = _mm512_dpwssd_epi32(a[j], s0, wv);
                b[j] = _mm512_dpwssd_epi32(b[j], s1, wv);
            }
        }
        for (int j = 0; j < kParts; ++j) {
            _mm512_storeu_si512(out + f * 2 * kHidden + j * 16, a[j]);
            _mm512_storeu_si512(out + (f + 1) * 2 * kHidden + j * 16, b[j]);
        }
    }
    for (; f < frames; ++f) {
        const int16_t * x0 = x + f * in_dim;
        __m512i a[kParts];
        for (int j = 0; j < kParts; ++j) {
            a[j] = _mm512_setzero_si512();
        }
        for (int64_t p = 0; p < pair_count; ++p) {
            int32_t v0;
            std::memcpy(&v0, x0 + 2 * p, sizeof(v0));
            const __m512i s0 = _mm512_set1_epi32(v0);
            const int16_t * wp = pairs + p * 4 * kHidden;
            for (int j = 0; j < kParts; ++j) {
                a[j] = _mm512_dpwssd_epi32(a[j], s0, _mm512_loadu_si512(wp + j * 32));
            }
        }
        for (int j = 0; j < kParts; ++j) {
            _mm512_storeu_si512(out + f * 2 * kHidden + j * 16, a[j]);
        }
    }
}

}  // namespace avx512
EARSHOT_POP_TARGET

const Kernels kAvx512Kernels = {
    "avx512",
    avx512::window_to_features,
    avx512::conv_stack,
    avx512::emphasis_window,
    avx512::gemm,
    avx2::scan,
};

struct CpuFeatures {
    bool avx2 = false;
    bool avx512 = false;  // F, BW and VNNI, with OS support for the ZMM state
};

CpuFeatures detect_cpu() {
    CpuFeatures cpu;
#if defined(_MSC_VER)
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) {
        return cpu;
    }
    __cpuid(info, 1);
    const bool osxsave = (info[2] & (1 << 27)) != 0 && (info[2] & (1 << 28)) != 0;
    if (!osxsave) {
        return cpu;
    }
    const unsigned long long xcr0 = _xgetbv(0);
    __cpuidex(info, 7, 0);
    cpu.avx2 = (info[1] & (1 << 5)) != 0 && (xcr0 & 0x6) == 0x6;
    const bool f = (info[1] & (1 << 16)) != 0;
    const bool bw = (info[1] & (1 << 30)) != 0;
    const bool vnni = (info[2] & (1 << 11)) != 0;
    cpu.avx512 = cpu.avx2 && f && bw && vnni && (xcr0 & 0xE6) == 0xE6;
#else
    __builtin_cpu_init();
    cpu.avx2 = __builtin_cpu_supports("avx2");
    cpu.avx512 = cpu.avx2 && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                 __builtin_cpu_supports("avx512vnni");
#endif
    return cpu;
}
#endif  // EARSHOT_X86_DISPATCH

// The widest supported kernel set. AUDIOCPP_EARSHOT_ISA=baseline|avx2|avx512 caps
// it (tests use this); AUDIOCPP_EARSHOT_BASELINE=1 is the same as baseline.
const Kernels & kernels() {
    static const Kernels * selected = [] {
#if defined(EARSHOT_X86_DISPATCH)
        const CpuFeatures cpu = detect_cpu();
        std::string_view cap = "avx512";
        if (const char * value = std::getenv("AUDIOCPP_EARSHOT_ISA")) {
            cap = value;
        }
        if (std::getenv("AUDIOCPP_EARSHOT_BASELINE") != nullptr) {
            cap = "baseline";
        }
        if (cpu.avx512 && cap == "avx512") {
            return &kAvx512Kernels;
        }
        if (cpu.avx2 && (cap == "avx512" || cap == "avx2")) {
            return &kAvx2Kernels;
        }
#endif
        return &kBaselineKernels;
    }();
    return *selected;
}

}  // namespace

// ---------------------------------------------------------------- assets

EarshotAssetPaths resolve_earshot_assets(const std::filesystem::path & model_path) {
    EarshotAssetPaths paths;
    if (engine::io::is_existing_file(model_path)) {
        paths.checkpoint_path = std::filesystem::weakly_canonical(model_path);
        paths.model_root = paths.checkpoint_path.parent_path();
        return paths;
    }
    if (!engine::io::is_existing_directory(model_path)) {
        throw std::runtime_error("Earshot VAD model path does not exist: " + model_path.string());
    }
    const auto checkpoint = model_path / kWeightFile;
    if (!engine::io::is_existing_file(checkpoint)) {
        throw std::runtime_error("Earshot VAD weights not found: " + checkpoint.string());
    }
    paths.model_root = std::filesystem::weakly_canonical(model_path);
    paths.checkpoint_path = std::filesystem::weakly_canonical(checkpoint);
    return paths;
}

std::shared_ptr<const EarshotWeights> load_earshot_weights_cached(const std::filesystem::path & checkpoint_path) {
    static std::mutex mutex;
    static std::unordered_map<std::string, std::weak_ptr<const EarshotWeights>> cache;
    const auto key = cache_key(checkpoint_path);
    std::lock_guard<std::mutex> lock(mutex);
    if (const auto it = cache.find(key); it != cache.end()) {
        if (auto existing = it->second.lock()) {
            return existing;
        }
    }
    auto loaded = std::make_shared<const EarshotWeights>(load_earshot_weights(checkpoint_path));
    cache[key] = loaded;
    return loaded;
}

// ---------------------------------------------------------------- options

EarshotSegmenterConfig earshot_config_from_options(
    const std::unordered_map<std::string, std::string> & options,
    const EarshotSegmenterConfig * fallback) {
    EarshotSegmenterConfig config = fallback != nullptr ? *fallback : EarshotSegmenterConfig{};
    const auto frames_from_ms = [](int ms, const char * name) {
        if (ms < 0) {
            throw std::runtime_error(std::string("Earshot VAD ") + name + " must be >= 0");
        }
        return static_cast<int>((static_cast<int64_t>(ms) * kSampleRate / 1000 + kFrameSize - 1) / kFrameSize);
    };
    const auto probability = [](float value, const char * name) {
        if (!(value >= 0.0f && value <= 1.0f)) {
            throw std::runtime_error(std::string("Earshot VAD ") + name + " must be in [0, 1]");
        }
        return value;
    };
    if (const auto value = runtime::parse_finite_float_option(options, {"earshot_vad.threshold", "threshold"})) {
        config.threshold = probability(*value, "threshold");
    }
    if (const auto value = runtime::parse_finite_float_option(options, {"earshot_vad.neg_threshold", "neg_threshold"})) {
        config.neg_threshold = probability(*value, "neg_threshold");
    }
    if (const auto value =
            runtime::parse_finite_float_option(options, {"earshot_vad.smoothing_factor", "smoothing_factor"})) {
        if (!(*value >= 0.0f && *value < 1.0f)) {
            throw std::runtime_error("Earshot VAD smoothing_factor must be in [0, 1)");
        }
        config.smoothing_factor = *value;
    }
    if (const auto value = runtime::parse_int_option(options, {"earshot_vad.speech_pad_ms", "speech_pad_ms"})) {
        config.preroll_frames = frames_from_ms(*value, "speech_pad_ms");
    }
    if (const auto value =
            runtime::parse_int_option(options, {"earshot_vad.min_speech_duration_ms", "min_speech_duration_ms"})) {
        config.min_duration_frames = frames_from_ms(*value, "min_speech_duration_ms");
    }
    if (const auto value =
            runtime::parse_int_option(options, {"earshot_vad.min_silence_duration_ms", "min_silence_duration_ms"})) {
        config.end_delay_frames = std::max(1, frames_from_ms(*value, "min_silence_duration_ms"));
    }
    if (const auto value = runtime::parse_int_option(options, {"earshot_vad.start_delay_ms", "start_delay_ms"})) {
        config.start_delay_frames = std::max(1, frames_from_ms(*value, "start_delay_ms"));
    }
    if (const auto value =
            runtime::find_option(options, {"earshot_vad.keep_trailing_segment", "keep_trailing_segment"})) {
        config.keep_trailing_segment = runtime::parse_bool_option(*value, "keep_trailing_segment");
    }
    return config;
}

// ---------------------------------------------------------------- detector

struct EarshotDetector::State {
    // The last 768 input samples plus the one before them (zeros at the start,
    // like the offline path), so emphasis_window() can build the window.
    std::array<float, kWindowSize + 1> history{};
    std::array<float, kMels * kContextFrames> context{};
    std::array<int16_t, 2 * kHidden> hidden{};         // layer 1 then layer 2
    std::array<float, kWindowSize> windowed{};
};

EarshotDetector::EarshotDetector(std::shared_ptr<const EarshotWeights> weights)
    : weights_(std::move(weights)),
      state_(std::make_unique<State>()) {
    if (!weights_) {
        throw std::runtime_error("Earshot VAD detector requires weights");
    }
}

EarshotDetector::~EarshotDetector() = default;
EarshotDetector::EarshotDetector(EarshotDetector &&) noexcept = default;
EarshotDetector & EarshotDetector::operator=(EarshotDetector &&) noexcept = default;

void EarshotDetector::reset() {
    *state_ = State{};
}

float EarshotDetector::predict(const float * frame) {
    const auto & w = *weights_;
    const Kernels & k = kernels();
    auto & s = *state_;
    std::memmove(s.history.data(), s.history.data() + kFrameSize, (kWindowSize + 1 - kFrameSize) * sizeof(float));
    std::memcpy(s.history.data() + (kWindowSize + 1 - kFrameSize), frame, kFrameSize * sizeof(float));
    k.emphasis_window(s.history.data() + 1, w.hann_window.data(), s.windowed.data());
    std::memmove(s.context.data(), s.context.data() + kMels, (kContextFrames - 1) * kMels * sizeof(float));
    k.features(w, s.windowed.data(), s.context.data() + (kContextFrames - 1) * kMels);
    int16_t rnn_input[kRnnInput];
    k.conv(w, s.context.data(), rnn_input);
    int32_t acc[2 * kHidden];
    int16_t * h1 = s.hidden.data();
    int16_t * h2 = s.hidden.data() + kHidden;
    k.gemm(rnn_input, 1, kRnnInput, w.rnn1_weight.data(), w.rnn1_pairs.data(), acc);
    k.scan(acc, 1, 0, kScanGroups, h1, nullptr, 0);
    k.gemm(h1, 1, kHidden, w.rnn2_weight.data(), w.rnn2_pairs.data(), acc);
    k.scan(acc, 1, 0, kScanGroups, h2, nullptr, 0);
    return output_score(w, h1, h2);
}

const char * earshot_kernel_name() {
    return kernels().name;
}

std::vector<float> score_earshot_frames(
    const std::shared_ptr<const EarshotWeights> & weights,
    const float * samples,
    int64_t sample_count,
    int threads) {
    if (!weights) {
        throw std::runtime_error("Earshot VAD scoring requires weights");
    }
    const auto & w = *weights;
    const Kernels & k = kernels();
    const int64_t frame_count = sample_count / kFrameSize;
    std::vector<float> scores(static_cast<size_t>(frame_count), 0.0f);
    if (frame_count == 0) {
        return scores;
    }
    threads = std::max(1, threads);
    using Clock = std::chrono::steady_clock;
    double frontend_ms = 0.0;
    double conv_ms = 0.0;
    double rnn_ms = 0.0;

    // Blocks bound the scratch memory. Within a block the front end, the convs
    // and both matrix products run in parallel over frames; only the two scans
    // run in order. Feature rows of frame f sit at row f - block + 2, so a
    // frame's three-row context starts at its own index; the first block's two
    // leading rows stay zero, like upstream's zeroed buffers.
    constexpr int64_t kBlock = 4096;
    constexpr int64_t kTile = 32;
    const int64_t block_frames = std::min(kBlock, frame_count);
    std::vector<float> features(static_cast<size_t>((block_frames + 2) * kMels), 0.0f);
    std::vector<int16_t> rnn_inputs(static_cast<size_t>(block_frames * kRnnInput));
    std::vector<int32_t> acc(static_cast<size_t>(block_frames * 2 * kHidden));
    // Hidden states are group-major ([group, frame, 8]) so scan threads never
    // share a cache line; group_stride is one group's span.
    const int64_t group_stride = block_frames * 8;
    std::vector<int16_t> h1(static_cast<size_t>(block_frames * kHidden));
    std::vector<int16_t> h2(static_cast<size_t>(block_frames * kHidden));
    const auto gather_hidden = [group_stride](const std::vector<int16_t> & groups, int64_t frame, int16_t * row) {
        for (int g = 0; g < kScanGroups; ++g) {
            std::memcpy(row + 8 * g, groups.data() + g * group_stride + frame * 8, 8 * sizeof(int16_t));
        }
    };
    std::array<int16_t, kHidden> state1{};
    std::array<int16_t, kHidden> state2{};

    for (int64_t block = 0; block < frame_count; block += kBlock) {
        const int64_t count = std::min(kBlock, frame_count - block);
        const int64_t tiles = (count + kTile - 1) / kTile;
        const bool parallel = threads > 1 && count >= 256;
        // Scan parts: contiguous group ranges, a power of two up to eight.
        int parts = 1;
        while (parallel && parts * 2 <= std::min(threads, kScanGroups)) {
            parts *= 2;
        }
        const int part_groups = kScanGroups / parts;
        auto mark = Clock::now();
        const auto lap = [&mark](double & total) {
            const auto now = Clock::now();
            total += std::chrono::duration<double, std::milli>(now - mark).count();
            mark = now;
        };
#ifdef _OPENMP
#pragma omp parallel num_threads(threads) if (parallel)
#endif
        {
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (int64_t i = 0; i < count; ++i) {
                // Pre-emphasis only needs the previous sample, so each window
                // recomputes it and frames are independent.
                const int64_t frame = block + i;
                alignas(64) float windowed[kWindowSize];
                const int64_t first = (frame - (kContextFrames - 1)) * kFrameSize;
                if (first > 0) {
                    k.emphasis_window(samples + first, w.hann_window.data(), windowed);
                } else {
                    for (int64_t j = 0; j < kWindowSize; ++j) {
                        const int64_t g = first + j;
                        float emphasised = 0.0f;
                        if (g >= 0) {
                            const float sample = samples[g] * 32768.0f;
                            const float prev = g > 0 ? samples[g - 1] * 32768.0f : 0.0f;
                            emphasised = sample - kPreEmphasis * prev;
                        }
                        windowed[j] = emphasised * w.hann_window[j];
                    }
                }
                k.features(w, windowed, features.data() + (i + 2) * kMels);
            }
#ifdef _OPENMP
#pragma omp master
#endif
            lap(frontend_ms);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (int64_t t = 0; t < tiles; ++t) {
                const int64_t begin = t * kTile;
                const int64_t n = std::min(kTile, count - begin);
                for (int64_t i = begin; i < begin + n; ++i) {
                    k.conv(w, features.data() + i * kMels, rnn_inputs.data() + i * kRnnInput);
                }
                k.gemm(rnn_inputs.data() + begin * kRnnInput, n, kRnnInput, w.rnn1_weight.data(),
                       w.rnn1_pairs.data(), acc.data() + begin * 2 * kHidden);
            }
#ifdef _OPENMP
#pragma omp master
#endif
            lap(conv_ms);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (int part = 0; part < parts; ++part) {
                k.scan(acc.data(), count, part * part_groups, (part + 1) * part_groups, state1.data(), h1.data(),
                       group_stride);
            }
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (int64_t t = 0; t < tiles; ++t) {
                const int64_t begin = t * kTile;
                const int64_t n = std::min(kTile, count - begin);
                int16_t rows[kTile * kHidden];
                for (int64_t i = 0; i < n; ++i) {
                    gather_hidden(h1, begin + i, rows + i * kHidden);
                }
                k.gemm(rows, n, kHidden, w.rnn2_weight.data(), w.rnn2_pairs.data(), acc.data() + begin * 2 * kHidden);
            }
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (int part = 0; part < parts; ++part) {
                k.scan(acc.data(), count, part * part_groups, (part + 1) * part_groups, state2.data(), h2.data(),
                       group_stride);
            }
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for (int64_t i = 0; i < count; ++i) {
                int16_t r1[kHidden];
                int16_t r2[kHidden];
                gather_hidden(h1, i, r1);
                gather_hidden(h2, i, r2);
                scores[static_cast<size_t>(block + i)] = output_score(w, r1, r2);
            }
        }
        lap(rnn_ms);
        // The last two feature rows are the next block's leading context.
        std::copy(features.begin() + count * kMels, features.begin() + (count + 2) * kMels, features.begin());
    }
    if (debug::timing_log_enabled()) {
        debug::timing_log_scalar("earshot_vad.offline.frames", frame_count);
        debug::timing_log_scalar("earshot_vad.offline.frontend_ms", frontend_ms);
        debug::timing_log_scalar("earshot_vad.offline.conv_ms", conv_ms);
        debug::timing_log_scalar("earshot_vad.offline.rnn_ms", rnn_ms);
    }
    return scores;
}

// ---------------------------------------------------------------- segmenter

EarshotSegmenter::EarshotSegmenter(EarshotSegmenterConfig config)
    : config_(config) {
    config_.start_delay_frames = std::max(1, config_.start_delay_frames);
    config_.end_delay_frames = std::max(1, config_.end_delay_frames);
}

EarshotSegmenter::Update EarshotSegmenter::push(float raw_score) {
    Update update;
    smoothed_ = config_.smoothing_factor * smoothed_ + (1.0f - config_.smoothing_factor) * raw_score;
    const bool is_voice = state_ == State::Voice ? smoothed_ >= config_.neg_threshold : smoothed_ >= config_.threshold;
    switch (state_) {
        case State::Silence:
            if (is_voice) {
                state_ = State::PendingStart;
                voice_frames_ = 1;
                score_sum_ = raw_score;
                score_count_ = 1;
            }
            break;
        case State::PendingStart:
            if (is_voice) {
                ++voice_frames_;
                score_sum_ += raw_score;
                ++score_count_;
                if (voice_frames_ >= config_.start_delay_frames) {
                    state_ = State::Voice;
                    // Upstream formula; the clamp only matters for tiny preroll values,
                    // which would otherwise start the segment after the current frame.
                    start_ = std::clamp<int64_t>(n_frames_ + voice_frames_ - config_.preroll_frames, 0, n_frames_);
                    silence_frames_ = 0;
                    start_reported_ = false;
                }
            } else {
                state_ = State::Silence;
            }
            break;
        case State::Voice:
            score_sum_ += raw_score;
            ++score_count_;
            silence_frames_ = is_voice ? 0 : silence_frames_ + 1;
            if (silence_frames_ >= config_.end_delay_frames) {
                state_ = State::Silence;
                if (n_frames_ > start_ && n_frames_ - start_ >= config_.min_duration_frames) {
                    update.finished = EarshotFrameSegment{
                        start_, n_frames_, static_cast<float>(score_sum_ / static_cast<double>(score_count_))};
                }
                start_reported_ = false;
            }
            break;
    }
    ++n_frames_;
    // Announce the start as soon as the segment would be reported if the input ended now
    // (the finish() rule); a later close by silence can only make it longer.
    if (state_ == State::Voice && !start_reported_ && n_frames_ > start_ &&
        n_frames_ - start_ >= config_.min_duration_frames) {
        start_reported_ = true;
        update.started_frame = start_;
    }
    return update;
}

std::optional<EarshotFrameSegment> EarshotSegmenter::finish() {
    std::optional<EarshotFrameSegment> out;
    if (state_ == State::Voice && n_frames_ > start_ && n_frames_ - start_ >= config_.min_duration_frames &&
        (config_.keep_trailing_segment || silence_frames_ >= config_.end_delay_frames)) {
        out = EarshotFrameSegment{
            start_, n_frames_, static_cast<float>(score_sum_ / static_cast<double>(std::max<int64_t>(1, score_count_)))};
    }
    state_ = State::Silence;
    start_reported_ = false;
    return out;
}

std::vector<EarshotFrameSegment> segment_earshot_scores(
    const std::vector<float> & scores,
    const EarshotSegmenterConfig & config) {
    EarshotSegmenter segmenter(config);
    std::vector<EarshotFrameSegment> segments;
    for (const float score : scores) {
        if (auto update = segmenter.push(score); update.finished) {
            segments.push_back(*update.finished);
        }
    }
    if (auto last = segmenter.finish()) {
        segments.push_back(*last);
    }
    return segments;
}

}  // namespace engine::community_models::earshot_vad
