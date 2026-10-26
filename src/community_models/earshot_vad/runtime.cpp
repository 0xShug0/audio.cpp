// Native C++ port of the Earshot VAD network (https://github.com/pykeio/earshot,
// MIT OR Apache-2.0). The arithmetic follows upstream src/lib.rs and
// src/default_predictor.rs operation for operation, so scores match the Rust
// crate up to FFT rounding.

#include "engine/community_models/earshot_vad/runtime.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/options.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif
#define EARSHOT_AVX2_DISPATCH 1
#endif
#elif defined(__ARM_NEON) || defined(_M_ARM64)
#include <arm_neon.h>
#define EARSHOT_NEON 1
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
    }
    weights.rnn1_weight.resize(128 * kRnnInput);
    copy_tensor(*source, "rnn1_weight", "I16", weights.rnn1_weight.size(), weights.rnn1_weight.data());
    weights.rnn2_weight.resize(128 * kHidden);
    copy_tensor(*source, "rnn2_weight", "I16", weights.rnn2_weight.size(), weights.rnn2_weight.data());
    copy_tensor(*source, "output_weight", "I16", weights.output_weight);
    copy_tensor(*source, "hann_window", "F32", weights.hann_window);
    copy_tensor(*source, "mel_start", "I32", weights.mel_start);
    copy_tensor(*source, "mel_length", "I32", weights.mel_length);

    int64_t total = 0;
    for (int64_t band = 0; band < kMels; ++band) {
        const int32_t start = weights.mel_start[band];
        const int32_t length = weights.mel_length[band];
        if (start < 0 || length <= 0 || start + length > kBins) {
            throw std::runtime_error("Earshot VAD mel band is out of range");
        }
        weights.mel_offset[band] = static_cast<int32_t>(total);
        total += length;
    }
    const auto coeffs = source->require_tensor_data("mel_coeffs");
    if (coeffs.metadata.dtype != "F32" || coeffs.bytes.size() != static_cast<size_t>(total) * sizeof(float)) {
        throw std::runtime_error("Earshot VAD mel_coeffs does not match mel_length");
    }
    weights.mel_coeffs.resize(static_cast<size_t>(total));
    std::memcpy(weights.mel_coeffs.data(), coeffs.bytes.data(), coeffs.bytes.size());
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
    std::array<float, kHalfFft + 1> post_re{};     // exp(-2 pi i k / 1024)
    std::array<float, kHalfFft + 1> post_im{};
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

// |X[k]|^2 for k in [0, 512] of the 1024-point DFT of `windowed` (768 samples,
// zero-padded).
void power_spectrum(const float * windowed, float * power) {
    const auto & t = fft_tables();
    alignas(16) float re[kHalfFft];
    alignas(16) float im[kHalfFft];
    for (int64_t n = 0; n < kWindowSize / 2; ++n) {
        re[t.bitrev[n]] = windowed[2 * n];
        im[t.bitrev[n]] = windowed[2 * n + 1];
    }
    for (int64_t n = kWindowSize / 2; n < kHalfFft; ++n) {
        re[t.bitrev[n]] = 0.0f;
        im[t.bitrev[n]] = 0.0f;
    }
    // Stages h = 1 and h = 2 as one exact radix-4 pass (twiddles 1 and -i).
    for (int64_t base = 0; base < kHalfFft; base += 4) {
        const float t0r = re[base] + re[base + 1];
        const float t0i = im[base] + im[base + 1];
        const float t1r = re[base] - re[base + 1];
        const float t1i = im[base] - im[base + 1];
        const float t2r = re[base + 2] + re[base + 3];
        const float t2i = im[base + 2] + im[base + 3];
        const float t3r = re[base + 2] - re[base + 3];
        const float t3i = im[base + 2] - im[base + 3];
        re[base] = t0r + t2r;
        im[base] = t0i + t2i;
        re[base + 2] = t0r - t2r;
        im[base + 2] = t0i - t2i;
        re[base + 1] = t1r + t3i;
        im[base + 1] = t1i - t3r;
        re[base + 3] = t1r - t3i;
        im[base + 3] = t1i + t3r;
    }
    for (int64_t h = 4; h < kHalfFft; h <<= 1) {
        const float * wr = t.twiddle_re.data() + (h - 1);
        const float * wi = t.twiddle_im.data() + (h - 1);
        for (int64_t base = 0; base < kHalfFft; base += 2 * h) {
            float * __restrict ar = re + base;
            float * __restrict ai = im + base;
            float * __restrict br = re + base + h;
            float * __restrict bi = im + base + h;
#if defined(EARSHOT_SSE2)
            for (int64_t j = 0; j < h; j += 4) {
                const __m128 xr = _mm_load_ps(br + j);
                const __m128 xi = _mm_load_ps(bi + j);
                const __m128 cr = _mm_loadu_ps(wr + j);
                const __m128 ci = _mm_loadu_ps(wi + j);
                const __m128 vr = _mm_sub_ps(_mm_mul_ps(xr, cr), _mm_mul_ps(xi, ci));
                const __m128 vi = _mm_add_ps(_mm_mul_ps(xr, ci), _mm_mul_ps(xi, cr));
                const __m128 ur = _mm_load_ps(ar + j);
                const __m128 ui = _mm_load_ps(ai + j);
                _mm_store_ps(ar + j, _mm_add_ps(ur, vr));
                _mm_store_ps(ai + j, _mm_add_ps(ui, vi));
                _mm_store_ps(br + j, _mm_sub_ps(ur, vr));
                _mm_store_ps(bi + j, _mm_sub_ps(ui, vi));
            }
#else
            for (int64_t j = 0; j < h; ++j) {
                const float vr = br[j] * wr[j] - bi[j] * wi[j];
                const float vi = br[j] * wi[j] + bi[j] * wr[j];
                const float ur = ar[j];
                const float ui = ai[j];
                ar[j] = ur + vr;
                ai[j] = ui + vi;
                br[j] = ur - vr;
                bi[j] = ui - vi;
            }
#endif
        }
    }
    // Split Z = FFT512(even + i * odd) into the real-input spectrum X.
    const float dc_re = re[0];
    const float dc_im = im[0];
    power[0] = (dc_re + dc_im) * (dc_re + dc_im);
    power[kHalfFft] = (dc_re - dc_im) * (dc_re - dc_im);
    for (int64_t k = 1; k < kHalfFft; ++k) {
        const float zr = re[k];
        const float zi = im[k];
        const float cr = re[kHalfFft - k];
        const float ci = -im[kHalfFft - k];
        const float er = 0.5f * (zr + cr);
        const float ei = 0.5f * (zi + ci);
        const float odd_re = 0.5f * (zi - ci);
        const float odd_im = -0.5f * (zr - cr);
        const float xr = er + (t.post_re[k] * odd_re - t.post_im[k] * odd_im);
        const float xi = ei + (t.post_re[k] * odd_im + t.post_im[k] * odd_re);
        power[k] = xr * xr + xi * xi;
    }
}

// Mel log-energies of one Hann-windowed frame, then the RMS norm applied by
// DefaultPredictor::normalize.
void window_to_features(const EarshotWeights & w, const float * windowed, float * out) {
    float power[kBins];
    power_spectrum(windowed, power);
    for (int64_t band = 0; band < kMels; ++band) {
        const float * coeffs = w.mel_coeffs.data() + w.mel_offset[band];
        const float * bins = power + w.mel_start[band];
        float value = 0.0f;
        for (int32_t k = 0; k < w.mel_length[band]; ++k) {
            value += (bins[k] * kPowerScale) * coeffs[k];
        }
        out[band] = std::log(value + 1e-20f);
    }
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < kMels; ++i) {
        sum_sq += out[i] * out[i];
    }
    const float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(kMels));
    for (int64_t i = 0; i < kMels; ++i) {
        out[i] = w.norm_weight[i] * out[i] * inv_rms;
    }
}

// Depthwise 3-tap conv (stride 2, pad 1) over [16, InCols] for output column ox.
template <int64_t InCols>
void depthwise_column(const float * input, const std::array<float, 48> & kernel, int64_t ox, float * dw) {
    for (int64_t c = 0; c < kChannels; ++c) {
        float sum = 0.0f;
        for (int64_t kw = 0; kw < 3; ++kw) {
            const int64_t ix = ox * 2 + kw - 1;
            if (ix < 0 || ix >= InCols) {
                continue;
            }
            sum += input[c * InCols + ix] * kernel[c * 3 + kw];
        }
        dw[c] = sum;
    }
}

// Pointwise 16x16 conv. Each output sums its inputs in upstream order; keeping
// outputs innermost lets the loop vectorize without reordering any sum.
void pointwise(const float * dw, const std::array<float, 256> & weight_t, float * acc) {
#if defined(EARSHOT_SSE2)
    static_assert(kChannels == 16, "SSE pointwise assumes 16 channels");
    __m128 a0 = _mm_setzero_ps();
    __m128 a1 = _mm_setzero_ps();
    __m128 a2 = _mm_setzero_ps();
    __m128 a3 = _mm_setzero_ps();
    for (int64_t c = 0; c < kChannels; ++c) {
        const __m128 x = _mm_set1_ps(dw[c]);
        const float * row = weight_t.data() + c * kChannels;
        a0 = _mm_add_ps(a0, _mm_mul_ps(x, _mm_loadu_ps(row)));
        a1 = _mm_add_ps(a1, _mm_mul_ps(x, _mm_loadu_ps(row + 4)));
        a2 = _mm_add_ps(a2, _mm_mul_ps(x, _mm_loadu_ps(row + 8)));
        a3 = _mm_add_ps(a3, _mm_mul_ps(x, _mm_loadu_ps(row + 12)));
    }
    _mm_storeu_ps(acc, a0);
    _mm_storeu_ps(acc + 4, a1);
    _mm_storeu_ps(acc + 8, a2);
    _mm_storeu_ps(acc + 12, a3);
#else
    for (int64_t oc = 0; oc < kChannels; ++oc) {
        acc[oc] = 0.0f;
    }
    for (int64_t c = 0; c < kChannels; ++c) {
        const float x = dw[c];
        const float * row = weight_t.data() + c * kChannels;
        for (int64_t oc = 0; oc < kChannels; ++oc) {
            acc[oc] += x * row[oc];
        }
    }
#endif
}

// Conv stack over a [3, 40] feature context; writes the Q15 [5, 16] RNN input.
void conv_stack(const EarshotWeights & w, const float * context, int16_t * out) {
    float row[kDepthwiseCols] = {};
    for (int64_t kh = 0; kh < 3; ++kh) {
        for (int64_t kw = 0; kw < 3; ++kw) {
            const float k = w.layer1_kernel[kh * 3 + kw];
            const float * src = context + kh * kMels + kw;
            int64_t ox = 0;
#if defined(EARSHOT_SSE2)
            const __m128 kv = _mm_set1_ps(k);
            for (; ox + 4 <= kDepthwiseCols; ox += 4) {
                _mm_storeu_ps(row + ox, _mm_add_ps(_mm_loadu_ps(row + ox), _mm_mul_ps(_mm_loadu_ps(src + ox), kv)));
            }
#endif
            for (; ox < kDepthwiseCols; ++ox) {
                row[ox] += src[ox] * k;
            }
        }
    }

    float layer1[kChannels * kPooledCols];
    for (int64_t c = 0; c < kChannels; ++c) {
        float scaled[kDepthwiseCols];
        const float scale = w.layer1_weight[c];
        const float bias = w.layer1_bias[c];
        for (int64_t ox = 0; ox < kDepthwiseCols; ++ox) {
            scaled[ox] = (row[ox] * scale) + bias;
        }
        // Zero-initialised max pool (kernel 3, stride 2) doubles as ReLU.
        float * cell = layer1 + c * kPooledCols;
        for (int64_t q = 0; q < kPooledCols; ++q) {
            cell[q] = std::max(std::max(std::max(0.0f, scaled[q * 2]), scaled[q * 2 + 1]), scaled[q * 2 + 2]);
        }
    }

    float layer2[kChannels * kLayer2Cols];
    for (int64_t ox = 0; ox < kLayer2Cols; ++ox) {
        float dw[kChannels];
        float acc[kChannels];
        depthwise_column<kPooledCols>(layer1, w.layer2_kernel, ox, dw);
        pointwise(dw, w.layer2_weight_t, acc);
        for (int64_t oc = 0; oc < kChannels; ++oc) {
            layer2[oc * kLayer2Cols + ox] = std::max(acc[oc] + w.layer2_bias[oc], 0.0f);
        }
    }

    for (int64_t ox = 0; ox < kLayer3Cols; ++ox) {
        float dw[kChannels];
        float acc[kChannels];
        depthwise_column<kLayer2Cols>(layer2, w.layer3_kernel, ox, dw);
        pointwise(dw, w.layer3_weight_t, acc);
        // Transposed output, no bias or ReLU.
        for (int64_t oc = 0; oc < kChannels; ++oc) {
            const float q = std::clamp(std::round(acc[oc] * 32768.0f), -32768.0f, 32767.0f);
            out[ox * kChannels + oc] = static_cast<int16_t>(q);
        }
    }
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

// Q15 input x Q9 weights -> Q24 accumulators. Row L1 norms of the shipped
// weights keep every sum far below INT32_MAX.
// out[r] = dot(x, w[r]) for `rows` rows of `in_dim` int16 weights.
using MatVecI16 = void (*)(const int16_t * x, const int16_t * w, int64_t in_dim, int64_t rows, int32_t * out);

void matvec_i16_baseline(const int16_t * x, const int16_t * w, int64_t in_dim, int64_t rows, int32_t * out) {
    for (int64_t r = 0; r < rows; ++r) {
        out[r] = dot_i16(x, w + r * in_dim, in_dim);
    }
}

#if defined(EARSHOT_AVX2_DISPATCH)
#if defined(__GNUC__) || defined(__clang__)
#define EARSHOT_TARGET_AVX2 __attribute__((target("avx2")))
#else
#define EARSHOT_TARGET_AVX2
#endif

// Four rows per pass; in_dim is a multiple of 16 and rows a multiple of 4.
EARSHOT_TARGET_AVX2 void matvec_i16_avx2(
    const int16_t * x, const int16_t * w, int64_t in_dim, int64_t rows, int32_t * out) {
    for (int64_t r = 0; r < rows; r += 4) {
        const int16_t * w0 = w + r * in_dim;
        __m256i a0 = _mm256_setzero_si256();
        __m256i a1 = _mm256_setzero_si256();
        __m256i a2 = _mm256_setzero_si256();
        __m256i a3 = _mm256_setzero_si256();
        for (int64_t i = 0; i < in_dim; i += 16) {
            const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x + i));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(xv, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w0 + i))));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(xv, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w0 + in_dim + i))));
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(xv, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w0 + 2 * in_dim + i))));
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(xv, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(w0 + 3 * in_dim + i))));
        }
        const __m256i sums = _mm256_hadd_epi32(_mm256_hadd_epi32(a0, a1), _mm256_hadd_epi32(a2, a3));
        const __m128i total = _mm_add_epi32(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out + r), total);
    }
}

bool cpu_has_avx2() {
#if defined(_MSC_VER) && !defined(__clang__)
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) {
        return false;
    }
    __cpuid(info, 1);
    const bool os_avx = (info[2] & (1 << 27)) != 0 && (info[2] & (1 << 28)) != 0;
    __cpuidex(info, 7, 0);
    const bool avx2 = (info[1] & (1 << 5)) != 0;
    return os_avx && avx2 && (_xgetbv(0) & 6) == 6;
#else
    return __builtin_cpu_supports("avx2");
#endif
}
#endif

MatVecI16 matvec_i16() {
#if defined(EARSHOT_AVX2_DISPATCH)
    // AUDIOCPP_EARSHOT_BASELINE=1 forces the SSE2 kernel (used by tests).
    static const MatVecI16 selected =
        (cpu_has_avx2() && std::getenv("AUDIOCPP_EARSHOT_BASELINE") == nullptr) ? matvec_i16_avx2 : matvec_i16_baseline;
    return selected;
#else
    return matvec_i16_baseline;
#endif
}

template <int64_t InDim>
void mingru(const int16_t * features, int16_t * hidden, const int16_t * weight, MatVecI16 matvec) {
    static_assert(InDim % 16 == 0, "matvec_i16 needs a multiple of 16");
    int32_t acc[2 * kHidden];
    matvec(features, weight, InDim, 2 * kHidden, acc);
    for (int64_t i = 0; i < kHidden; ++i) {
        // Gate x4, clamped to [0, 1] in Q24; the shift wraps like upstream's i32 `<<`.
        const int32_t shifted = static_cast<int32_t>(static_cast<uint32_t>(acc[kHidden + i]) << 2);
        const int64_t gate = std::clamp<int32_t>(shifted, 0, 16777215);
        const int64_t y = (16777215 - gate) * (static_cast<int64_t>(hidden[i]) * 512) +
                          gate * static_cast<int64_t>(acc[i]);
        hidden[i] = static_cast<int16_t>(std::clamp<int64_t>(y >> 33, -32768, 32767));
    }
}

float recurrent_step(const EarshotWeights & w, const int16_t * rnn_input, int16_t * state) {
    const MatVecI16 matvec = matvec_i16();
    mingru<kRnnInput>(rnn_input, state, w.rnn1_weight.data(), matvec);
    mingru<kHidden>(state, state + kHidden, w.rnn2_weight.data(), matvec);
    const int32_t out = dot_i16(state, w.output_weight.data(), 2 * kHidden);
    const float x = static_cast<float>(out) / 16777216.0f;
    return 1.0f / (1.0f + std::exp(-x));
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
    float prev_sample = 0.0f;
    std::array<float, kWindowSize> ring{};             // pre-emphasised samples
    std::array<float, kMels * kContextFrames> context{};
    std::array<int16_t, 2 * kHidden> hidden{};
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
    auto & s = *state_;
    std::memmove(s.ring.data(), s.ring.data() + kFrameSize, (kWindowSize - kFrameSize) * sizeof(float));
    float * tail = s.ring.data() + (kWindowSize - kFrameSize);
    for (int64_t i = 0; i < kFrameSize; ++i) {
        const float sample = frame[i] * 32768.0f;
        tail[i] = sample - kPreEmphasis * s.prev_sample;
        s.prev_sample = sample;
    }
    for (int64_t i = 0; i < kWindowSize; ++i) {
        s.windowed[i] = s.ring[i] * w.hann_window[i];
    }
    std::memmove(s.context.data(), s.context.data() + kMels, (kContextFrames - 1) * kMels * sizeof(float));
    window_to_features(w, s.windowed.data(), s.context.data() + (kContextFrames - 1) * kMels);
    int16_t rnn_input[kRnnInput];
    conv_stack(w, s.context.data(), rnn_input);
    return recurrent_step(w, rnn_input, s.hidden.data());
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
    const int64_t frame_count = sample_count / kFrameSize;
    std::vector<float> scores(static_cast<size_t>(frame_count), 0.0f);
    if (frame_count == 0) {
        return scores;
    }
    threads = std::max(1, threads);
    const auto started = std::chrono::steady_clock::now();

    // Features for frame f use the 768-sample window ending at frame f; the
    // two leading context rows stay zero, like upstream's zeroed buffers.
    // Pre-emphasis only needs the previous sample, so each window recomputes it
    // and frames are independent until the recurrent layers.
    std::vector<float> features(static_cast<size_t>((frame_count + 2) * kMels), 0.0f);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1 && frame_count >= 256)
#endif
    for (int64_t frame = 0; frame < frame_count; ++frame) {
        alignas(16) float windowed[kWindowSize];
        const int64_t first = (frame - (kContextFrames - 1)) * kFrameSize;
        int64_t i = 0;
        if (first > 0) {
            // Whole window and its previous sample are inside the signal.
            const float * src = samples + first;
#if defined(EARSHOT_SSE2)
            const __m128 scale = _mm_set1_ps(32768.0f);
            const __m128 coeff = _mm_set1_ps(kPreEmphasis);
            for (; i < kWindowSize; i += 4) {
                const __m128 sample = _mm_mul_ps(_mm_loadu_ps(src + i), scale);
                const __m128 prev = _mm_mul_ps(_mm_loadu_ps(src + i - 1), scale);
                const __m128 emphasised = _mm_sub_ps(sample, _mm_mul_ps(coeff, prev));
                _mm_store_ps(windowed + i, _mm_mul_ps(emphasised, _mm_loadu_ps(w.hann_window.data() + i)));
            }
#endif
            for (; i < kWindowSize; ++i) {
                const float emphasised = src[i] * 32768.0f - kPreEmphasis * (src[i - 1] * 32768.0f);
                windowed[i] = emphasised * w.hann_window[i];
            }
        } else {
            for (; i < kWindowSize; ++i) {
                const int64_t g = first + i;
                float emphasised = 0.0f;
                if (g >= 0) {
                    const float sample = samples[g] * 32768.0f;
                    const float prev = g > 0 ? samples[g - 1] * 32768.0f : 0.0f;
                    emphasised = sample - kPreEmphasis * prev;
                }
                windowed[i] = emphasised * w.hann_window[i];
            }
        }
        window_to_features(w, windowed, features.data() + (frame + 2) * kMels);
    }

    const auto frontend_done = std::chrono::steady_clock::now();
    std::vector<int16_t> rnn_inputs(static_cast<size_t>(frame_count * kRnnInput));
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) if (threads > 1 && frame_count >= 256)
#endif
    for (int64_t frame = 0; frame < frame_count; ++frame) {
        conv_stack(w, features.data() + frame * kMels, rnn_inputs.data() + frame * kRnnInput);
    }

    const auto conv_done = std::chrono::steady_clock::now();
    std::array<int16_t, 2 * kHidden> hidden{};
    for (int64_t frame = 0; frame < frame_count; ++frame) {
        scores[static_cast<size_t>(frame)] = recurrent_step(w, rnn_inputs.data() + frame * kRnnInput, hidden.data());
    }
    if (debug::timing_log_enabled()) {
        const auto ms = [](auto from, auto to) { return std::chrono::duration<double, std::milli>(to - from).count(); };
        const auto done = std::chrono::steady_clock::now();
        debug::timing_log_scalar("earshot_vad.offline.frames", frame_count);
        debug::timing_log_scalar("earshot_vad.offline.frontend_ms", ms(started, frontend_done));
        debug::timing_log_scalar("earshot_vad.offline.conv_ms", ms(frontend_done, conv_done));
        debug::timing_log_scalar("earshot_vad.offline.rnn_ms", ms(conv_done, done));
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
