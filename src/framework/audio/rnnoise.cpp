#include "engine/framework/audio/rnnoise.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/fft.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <complex>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440084436210484904
#endif

namespace engine::audio {
namespace {

constexpr int kRnnoiseSampleRate = 48000;
constexpr int kRnnoiseFrameSize = 480;
constexpr int kRnnoiseWindowSize = 960;
constexpr int kRnnoiseFreqSize = 481;
constexpr int kRnnoiseBands = 32;
constexpr int kRnnoiseFeatures = 65;
constexpr int kRnnoisePitchMinPeriod = 60;
constexpr int kRnnoisePitchMaxPeriod = 768;
constexpr int kRnnoisePitchFrameSize = 960;
constexpr int kRnnoisePitchBufSize = kRnnoisePitchMaxPeriod + kRnnoisePitchFrameSize;
constexpr int64_t kRnnoiseSequenceChunkFrames = 64;
constexpr size_t kRnnoiseFrameGraphContextBytes = 8ull * 1024ull * 1024ull;
constexpr size_t kRnnoiseSequenceGraphContextBytes = 32ull * 1024ull * 1024ull;

const std::array<int, kRnnoiseBands + 2> kRnnoiseEband20ms = {
    0, 2, 4, 6, 8, 10, 12, 15, 18, 21, 24, 28, 32, 36, 41, 47, 53,
    60, 68, 77, 87, 98, 110, 124, 140, 157, 176, 198, 223, 251, 282,
    317, 356, 400};

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct BackendDeleter {
    void operator()(ggml_backend * backend) const noexcept {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

struct Conv1dWeights {
    std::vector<float> weight;
    std::vector<float> bias;
    int64_t out_channels = 0;
    int64_t in_channels = 0;
    int64_t kernel = 0;
};

struct DenseGraphWeights {
    modules::LinearWeights linear;
    int64_t out_features = 0;
    int64_t in_features = 0;
};

struct GruGraphWeights {
    DenseGraphWeights input;
    DenseGraphWeights recurrent;
    int64_t input_size = 0;
    int64_t hidden_size = 0;
};

struct CausalConvGraphWeights {
    DenseGraphWeights linear;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel = 0;
};

struct RnnoiseConvGraphWeights {
    CausalConvGraphWeights streaming;
};

void require_size(const std::vector<float> & values, int64_t expected, const std::string & name) {
    if (expected < 0 || static_cast<int64_t>(values.size()) != expected) {
        throw std::runtime_error("RNNoise tensor size mismatch: " + name);
    }
}

Conv1dWeights load_conv1d(
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel) {
    Conv1dWeights weights;
    weights.weight = source.require_f32(prefix + ".weight", {out_channels, in_channels, kernel});
    weights.bias = source.require_f32(prefix + ".bias", {out_channels});
    weights.out_channels = out_channels;
    weights.in_channels = in_channels;
    weights.kernel = kernel;
    return weights;
}

DenseGraphWeights load_dense_graph(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_features,
    int64_t in_features) {
    DenseGraphWeights weights;
    weights.linear.weight = store.load_f32_tensor(source, prefix + ".weight", {out_features, in_features});
    weights.linear.bias = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    weights.out_features = out_features;
    weights.in_features = in_features;
    return weights;
}

DenseGraphWeights load_dense_graph_names(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & weight_name,
    const std::string & bias_name,
    int64_t out_features,
    int64_t in_features) {
    DenseGraphWeights weights;
    weights.linear.weight = store.load_f32_tensor(source, weight_name, {out_features, in_features});
    weights.linear.bias = store.load_f32_tensor(source, bias_name, {out_features});
    weights.out_features = out_features;
    weights.in_features = in_features;
    return weights;
}

std::vector<float> flatten_causal_conv_weight(const Conv1dWeights & conv) {
    std::vector<float> flattened(static_cast<size_t>(conv.out_channels * conv.in_channels * conv.kernel));
    for (int64_t out = 0; out < conv.out_channels; ++out) {
        for (int64_t in = 0; in < conv.in_channels; ++in) {
            for (int64_t k = 0; k < conv.kernel; ++k) {
                const size_t source = static_cast<size_t>((out * conv.in_channels + in) * conv.kernel + k);
                const size_t target = static_cast<size_t>(out * conv.in_channels * conv.kernel + k * conv.in_channels + in);
                flattened[target] = conv.weight[source];
            }
        }
    }
    return flattened;
}

CausalConvGraphWeights make_causal_conv_graph(core::BackendWeightStore & store, Conv1dWeights conv) {
    CausalConvGraphWeights weights;
    weights.in_channels = conv.in_channels;
    weights.out_channels = conv.out_channels;
    weights.kernel = conv.kernel;
    weights.linear.linear.weight = store.make_f32(
        core::TensorShape::from_dims({conv.out_channels, conv.in_channels * conv.kernel}),
        flatten_causal_conv_weight(conv));
    weights.linear.linear.bias = store.make_f32(core::TensorShape::from_dims({conv.out_channels}), std::move(conv.bias));
    weights.linear.out_features = conv.out_channels;
    weights.linear.in_features = conv.in_channels * conv.kernel;
    return weights;
}

RnnoiseConvGraphWeights make_rnnoise_conv_graph(core::BackendWeightStore & store, Conv1dWeights conv) {
    RnnoiseConvGraphWeights weights;
    weights.streaming = make_causal_conv_graph(store, conv);
    return weights;
}

GruGraphWeights load_gru_graph(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t input_size,
    int64_t hidden_size) {
    GruGraphWeights weights;
    weights.input = load_dense_graph_names(
        store,
        source,
        prefix + ".weight_ih_l0",
        prefix + ".bias_ih_l0",
        hidden_size * 3,
        input_size);
    weights.recurrent = load_dense_graph_names(
        store,
        source,
        prefix + ".weight_hh_l0",
        prefix + ".bias_hh_l0",
        hidden_size * 3,
        hidden_size);
    weights.input_size = input_size;
    weights.hidden_size = hidden_size;
    return weights;
}

void shift_append(std::vector<float> & memory, const float * input, int64_t channels, int64_t kernel) {
    const int64_t state_frames = kernel - 1;
    if (state_frames <= 0) {
        return;
    }
    require_size(memory, channels * state_frames, "conv memory");
    for (int64_t frame = 1; frame < state_frames; ++frame) {
        std::copy(
            memory.begin() + static_cast<std::ptrdiff_t>(frame * channels),
            memory.begin() + static_cast<std::ptrdiff_t>((frame + 1) * channels),
            memory.begin() + static_cast<std::ptrdiff_t>((frame - 1) * channels));
    }
    std::copy(
        input,
        input + channels,
        memory.begin() + static_cast<std::ptrdiff_t>((state_frames - 1) * channels));
}

std::array<float, kRnnoiseFrameSize> rnnoise_half_window() {
    std::array<float, kRnnoiseFrameSize> window{};
    for (int i = 0; i < kRnnoiseFrameSize; ++i) {
        const double inner = std::sin(0.5 * M_PI * (static_cast<double>(i) + 0.5) / kRnnoiseFrameSize);
        window[static_cast<size_t>(i)] = static_cast<float>(std::sin(0.5 * M_PI * inner * inner));
    }
    return window;
}

void apply_window(std::vector<float> & x) {
    static const auto window = rnnoise_half_window();
    for (int i = 0; i < kRnnoiseFrameSize; ++i) {
        x[static_cast<size_t>(i)] *= window[static_cast<size_t>(i)];
        x[static_cast<size_t>(kRnnoiseWindowSize - 1 - i)] *= window[static_cast<size_t>(i)];
    }
}

void compute_band_energy(float * band_energy, const std::complex<float> * x) {
    std::array<float, kRnnoiseBands + 2> sum{};
    for (int band = 0; band < kRnnoiseBands + 1; ++band) {
        const int start = kRnnoiseEband20ms[static_cast<size_t>(band)];
        const int size = kRnnoiseEband20ms[static_cast<size_t>(band + 1)] - start;
        for (int j = 0; j < size; ++j) {
            const float frac = static_cast<float>(j) / static_cast<float>(size);
            const float energy = std::norm(x[static_cast<size_t>(start + j)]);
            sum[static_cast<size_t>(band)] += (1.0f - frac) * energy;
            sum[static_cast<size_t>(band + 1)] += frac * energy;
        }
    }
    sum[1] = (sum[0] + sum[1]) * 2.0f / 3.0f;
    sum[kRnnoiseBands] = (sum[kRnnoiseBands] + sum[kRnnoiseBands + 1]) * 2.0f / 3.0f;
    for (int band = 0; band < kRnnoiseBands; ++band) {
        band_energy[static_cast<size_t>(band)] = sum[static_cast<size_t>(band + 1)];
    }
}

void compute_band_corr(float * band_corr, const std::complex<float> * x, const std::complex<float> * p) {
    std::array<float, kRnnoiseBands + 2> sum{};
    for (int band = 0; band < kRnnoiseBands + 1; ++band) {
        const int start = kRnnoiseEband20ms[static_cast<size_t>(band)];
        const int size = kRnnoiseEband20ms[static_cast<size_t>(band + 1)] - start;
        for (int j = 0; j < size; ++j) {
            const float frac = static_cast<float>(j) / static_cast<float>(size);
            const auto & xv = x[static_cast<size_t>(start + j)];
            const auto & pv = p[static_cast<size_t>(start + j)];
            const float corr = xv.real() * pv.real() + xv.imag() * pv.imag();
            sum[static_cast<size_t>(band)] += (1.0f - frac) * corr;
            sum[static_cast<size_t>(band + 1)] += frac * corr;
        }
    }
    sum[1] = (sum[0] + sum[1]) * 2.0f / 3.0f;
    sum[kRnnoiseBands] = (sum[kRnnoiseBands] + sum[kRnnoiseBands + 1]) * 2.0f / 3.0f;
    for (int band = 0; band < kRnnoiseBands; ++band) {
        band_corr[static_cast<size_t>(band)] = sum[static_cast<size_t>(band + 1)];
    }
}

void interp_band_gain(float * gains, const float * band_gains) {
    std::fill(gains, gains + kRnnoiseFreqSize, 0.0f);
    for (int band = 1; band < kRnnoiseBands; ++band) {
        const int start = kRnnoiseEband20ms[static_cast<size_t>(band)];
        const int size = kRnnoiseEband20ms[static_cast<size_t>(band + 1)] - start;
        for (int j = 0; j < size; ++j) {
            const float frac = static_cast<float>(j) / static_cast<float>(size);
            gains[start + j] = (1.0f - frac) * band_gains[band - 1] + frac * band_gains[band];
        }
    }
    for (int j = 0; j < kRnnoiseEband20ms[1]; ++j) {
        gains[j] = band_gains[0];
    }
    for (int j = kRnnoiseEband20ms[kRnnoiseBands]; j < kRnnoiseEband20ms[kRnnoiseBands + 1]; ++j) {
        gains[j] = band_gains[kRnnoiseBands - 1];
    }
}

void rnnoise_dct(float * out, const float * input) {
    constexpr float scale = 0.30151134457776363f;  // sqrt(2/22), from upstream RNNoise.
    for (int i = 0; i < kRnnoiseBands; ++i) {
        double sum = 0.0;
        for (int j = 0; j < kRnnoiseBands; ++j) {
            float basis = std::cos((static_cast<float>(j) + 0.5f) * static_cast<float>(i) * static_cast<float>(M_PI) / kRnnoiseBands);
            if (i == 0) {
                basis *= static_cast<float>(M_SQRT1_2);
            }
            sum += static_cast<double>(input[static_cast<size_t>(j)]) * static_cast<double>(basis);
        }
        out[static_cast<size_t>(i)] = static_cast<float>(sum) * scale;
    }
}

float inner_product(const float * x, const float * y, int count) {
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
        sum += static_cast<double>(x[i]) * static_cast<double>(y[i]);
    }
    return static_cast<float>(sum);
}

void autocorr(const float * x, float * ac, int lag, int count) {
    const int fast_count = count - lag;
    for (int k = 0; k <= lag; ++k) {
        ac[k] = inner_product(x, x + k, fast_count);
        double tail = 0.0;
        for (int i = k + fast_count; i < count; ++i) {
            tail += static_cast<double>(x[i]) * static_cast<double>(x[i - k]);
        }
        ac[k] += static_cast<float>(tail);
    }
}

void lpc(float * coeffs, const float * ac, int order) {
    std::fill(coeffs, coeffs + order, 0.0f);
    float error = ac[0];
    if (ac[0] == 0.0f) {
        return;
    }
    for (int i = 0; i < order; ++i) {
        double rr = 0.0;
        for (int j = 0; j < i; ++j) {
            rr += static_cast<double>(coeffs[j]) * static_cast<double>(ac[i - j]);
        }
        rr += ac[i + 1] / 8.0;
        const float r = static_cast<float>(-8.0 * rr / error);
        coeffs[i] = r / 8.0f;
        for (int j = 0; j < (i + 1) / 2; ++j) {
            const float a = coeffs[j];
            const float b = coeffs[i - 1 - j];
            coeffs[j] = a + r * b;
            coeffs[i - 1 - j] = b + r * a;
        }
        error -= r * r * error;
        if (error < 0.001f * ac[0]) {
            break;
        }
    }
}

void fir5(const float * input, const float * coeffs, float * output, int count, float * mem) {
    float m0 = mem[0];
    float m1 = mem[1];
    float m2 = mem[2];
    float m3 = mem[3];
    float m4 = mem[4];
    for (int i = 0; i < count; ++i) {
        const float sum = input[i] + coeffs[0] * m0 + coeffs[1] * m1 + coeffs[2] * m2 + coeffs[3] * m3 + coeffs[4] * m4;
        m4 = m3;
        m3 = m2;
        m2 = m1;
        m1 = m0;
        m0 = input[i];
        output[i] = sum;
    }
    mem[0] = m0;
    mem[1] = m1;
    mem[2] = m2;
    mem[3] = m3;
    mem[4] = m4;
}

void pitch_downsample(float * const * input, float * output, int count, int channels) {
    for (int i = 1; i < count / 2; ++i) {
        output[i] = 0.25f * input[0][2 * i - 1] + 0.5f * input[0][2 * i] + 0.25f * input[0][2 * i + 1];
    }
    output[0] = 0.25f * input[0][1] + 0.5f * input[0][0];
    if (channels == 2) {
        for (int i = 1; i < count / 2; ++i) {
            output[i] += 0.25f * input[1][2 * i - 1] + 0.5f * input[1][2 * i] + 0.25f * input[1][2 * i + 1];
        }
        output[0] += 0.25f * input[1][1] + 0.5f * input[1][0];
    }

    std::array<float, 5> ac{};
    std::array<float, 4> lpc_coeffs{};
    std::array<float, 5> mem{};
    std::array<float, 5> fir_coeffs{};
    autocorr(output, ac.data(), 4, count / 2);
    ac[0] *= 1.0001f;
    for (int i = 1; i <= 4; ++i) {
        ac[static_cast<size_t>(i)] -= ac[static_cast<size_t>(i)] * (0.008f * i) * (0.008f * i);
    }
    lpc(lpc_coeffs.data(), ac.data(), 4);
    float tmp = 1.0f;
    for (int i = 0; i < 4; ++i) {
        tmp *= 0.9f;
        lpc_coeffs[static_cast<size_t>(i)] *= tmp;
    }
    fir_coeffs[0] = lpc_coeffs[0] + 0.8f;
    fir_coeffs[1] = lpc_coeffs[1] + 0.8f * lpc_coeffs[0];
    fir_coeffs[2] = lpc_coeffs[2] + 0.8f * lpc_coeffs[1];
    fir_coeffs[3] = lpc_coeffs[3] + 0.8f * lpc_coeffs[2];
    fir_coeffs[4] = 0.8f * lpc_coeffs[3];
    fir5(output, fir_coeffs.data(), output, count / 2, mem.data());
}

void pitch_xcorr(const float * x, const float * y, float * xcorr, int count, int max_pitch) {
    for (int pitch = 0; pitch < max_pitch; ++pitch) {
        xcorr[pitch] = inner_product(x, y + pitch, count);
    }
}

void find_best_pitch(const float * xcorr, const float * y, int count, int max_pitch, int * best_pitch) {
    float syy = 1.0f;
    std::array<float, 2> best_num{-1.0f, -1.0f};
    std::array<float, 2> best_den{0.0f, 0.0f};
    best_pitch[0] = 0;
    best_pitch[1] = 1;
    for (int j = 0; j < count; ++j) {
        syy += y[j] * y[j];
    }
    for (int i = 0; i < max_pitch; ++i) {
        if (xcorr[i] > 0.0f) {
            const float scaled = xcorr[i] * 1.0e-12f;
            const float num = scaled * scaled;
            if (num * best_den[1] > best_num[1] * syy) {
                if (num * best_den[0] > best_num[0] * syy) {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best_pitch[1] = best_pitch[0];
                    best_num[0] = num;
                    best_den[0] = syy;
                    best_pitch[0] = i;
                } else {
                    best_num[1] = num;
                    best_den[1] = syy;
                    best_pitch[1] = i;
                }
            }
        }
        syy += y[i + count] * y[i + count] - y[i] * y[i];
        syy = std::max(1.0f, syy);
    }
}

void pitch_search(const float * x_lp, float * y, int count, int max_pitch, int * pitch) {
    const int lag = count + max_pitch;
    std::array<float, kRnnoisePitchFrameSize / 4> x_lp4{};
    std::array<float, (kRnnoisePitchFrameSize + kRnnoisePitchMaxPeriod) / 4> y_lp4{};
    std::array<float, kRnnoisePitchMaxPeriod / 2> xcorr{};
    std::array<int, 2> best_pitch{};
    for (int j = 0; j < count / 4; ++j) {
        x_lp4[static_cast<size_t>(j)] = x_lp[2 * j];
    }
    for (int j = 0; j < lag / 4; ++j) {
        y_lp4[static_cast<size_t>(j)] = y[2 * j];
    }
    pitch_xcorr(x_lp4.data(), y_lp4.data(), xcorr.data(), count / 4, max_pitch / 4);
    find_best_pitch(xcorr.data(), y_lp4.data(), count / 4, max_pitch / 4, best_pitch.data());
    std::fill(xcorr.begin(), xcorr.end(), 0.0f);
    for (int i = 0; i < max_pitch / 2; ++i) {
        if (std::abs(i - 2 * best_pitch[0]) > 2 && std::abs(i - 2 * best_pitch[1]) > 2) {
            continue;
        }
        xcorr[static_cast<size_t>(i)] = std::max(-1.0f, inner_product(x_lp, y + i, count / 2));
    }
    find_best_pitch(xcorr.data(), y, count / 2, max_pitch / 2, best_pitch.data());
    int offset = 0;
    if (best_pitch[0] > 0 && best_pitch[0] < (max_pitch / 2) - 1) {
        const float a = xcorr[static_cast<size_t>(best_pitch[0] - 1)];
        const float b = xcorr[static_cast<size_t>(best_pitch[0])];
        const float c = xcorr[static_cast<size_t>(best_pitch[0] + 1)];
        if ((c - a) > 0.7f * (b - a)) {
            offset = 1;
        } else if ((a - c) > 0.7f * (b - c)) {
            offset = -1;
        }
    }
    *pitch = 2 * best_pitch[0] - offset;
}

float compute_pitch_gain(float xy, float xx, float yy) {
    return xy / std::sqrt(1.0f + xx * yy);
}

float remove_doubling(float * x, int max_period, int min_period, int count, int * pitch, int prev_period, float prev_gain) {
    static constexpr std::array<int, 16> second_check = {0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2};
    const int min_period0 = min_period;
    max_period /= 2;
    min_period /= 2;
    *pitch /= 2;
    prev_period /= 2;
    count /= 2;
    x += max_period;
    if (*pitch >= max_period) {
        *pitch = max_period - 1;
    }

    int best_pitch = *pitch;
    int t0 = *pitch;
    float xx = 0.0f;
    float xy = 0.0f;
    for (int i = 0; i < count; ++i) {
        xx += x[i] * x[i];
        xy += x[i] * x[i - t0];
    }
    std::array<float, kRnnoisePitchMaxPeriod + 1> yy_lookup{};
    yy_lookup[0] = xx;
    float yy = xx;
    for (int i = 1; i <= max_period; ++i) {
        yy += x[-i] * x[-i] - x[count - i] * x[count - i];
        yy_lookup[static_cast<size_t>(i)] = std::max(0.0f, yy);
    }
    yy = yy_lookup[static_cast<size_t>(t0)];
    float best_xy = xy;
    float best_yy = yy;
    const float g0 = compute_pitch_gain(xy, xx, yy);
    float gain = g0;
    for (int k = 2; k <= 15; ++k) {
        const int t1 = (2 * t0 + k) / (2 * k);
        if (t1 < min_period) {
            break;
        }
        int t1b = 0;
        if (k == 2) {
            t1b = t1 + t0 > max_period ? t0 : t0 + t1;
        } else {
            t1b = (2 * second_check[static_cast<size_t>(k)] * t0 + k) / (2 * k);
        }
        float xy1 = 0.0f;
        float xy2 = 0.0f;
        for (int i = 0; i < count; ++i) {
            xy1 += x[i] * x[i - t1];
            xy2 += x[i] * x[i - t1b];
        }
        xy = 0.5f * (xy1 + xy2);
        yy = 0.5f * (yy_lookup[static_cast<size_t>(t1)] + yy_lookup[static_cast<size_t>(t1b)]);
        const float g1 = compute_pitch_gain(xy, xx, yy);
        float cont = 0.0f;
        if (std::abs(t1 - prev_period) <= 1) {
            cont = prev_gain;
        } else if (std::abs(t1 - prev_period) <= 2 && 5 * k * k < t0) {
            cont = 0.5f * prev_gain;
        }
        float thresh = std::max(0.3f, 0.7f * g0 - cont);
        if (t1 < 3 * min_period) {
            thresh = std::max(0.4f, 0.85f * g0 - cont);
        } else if (t1 < 2 * min_period) {
            thresh = std::max(0.5f, 0.9f * g0 - cont);
        }
        if (g1 > thresh) {
            best_xy = xy;
            best_yy = yy;
            best_pitch = t1;
            gain = g1;
        }
    }
    best_xy = std::max(0.0f, best_xy);
    float pg = best_yy <= best_xy ? 1.0f : best_xy / (best_yy + 1.0f);
    std::array<float, 3> xcorr{};
    for (int k = 0; k < 3; ++k) {
        xcorr[static_cast<size_t>(k)] = inner_product(x, x - (best_pitch + k - 1), count);
    }
    int offset = 0;
    if ((xcorr[2] - xcorr[0]) > 0.7f * (xcorr[1] - xcorr[0])) {
        offset = 1;
    } else if ((xcorr[0] - xcorr[2]) > 0.7f * (xcorr[1] - xcorr[2])) {
        offset = -1;
    }
    if (pg > gain) {
        pg = gain;
    }
    *pitch = 2 * best_pitch + offset;
    if (*pitch < min_period0) {
        *pitch = min_period0;
    }
    return pg;
}

void pitch_filter(
    std::complex<float> * x,
    const std::complex<float> * p,
    const float * ex,
    const float * ep,
    const float * exp,
    const float * gains) {
    std::array<float, kRnnoiseBands> r{};
    std::array<float, kRnnoiseFreqSize> rf{};
    std::array<float, kRnnoiseBands> new_energy{};
    std::array<float, kRnnoiseBands> norm{};
    std::array<float, kRnnoiseFreqSize> normf{};
    for (int band = 0; band < kRnnoiseBands; ++band) {
        if (exp[band] > gains[band]) {
            r[static_cast<size_t>(band)] = 1.0f;
        } else {
            const float corr2 = exp[band] * exp[band];
            const float gain2 = gains[band] * gains[band];
            r[static_cast<size_t>(band)] = std::sqrt(std::min(1.0f, std::max(0.0f, corr2 * (1.0f - gain2) / (0.001f + gain2 * (1.0f - corr2)))));
        }
        r[static_cast<size_t>(band)] *= std::sqrt(ex[band] / (1.0e-8f + ep[band]));
    }
    interp_band_gain(rf.data(), r.data());
    for (int i = 0; i < kRnnoiseFreqSize; ++i) {
        x[i] += rf[static_cast<size_t>(i)] * p[i];
    }
    compute_band_energy(new_energy.data(), x);
    for (int band = 0; band < kRnnoiseBands; ++band) {
        norm[static_cast<size_t>(band)] = std::sqrt(ex[band] / (1.0e-8f + new_energy[static_cast<size_t>(band)]));
    }
    interp_band_gain(normf.data(), norm.data());
    for (int i = 0; i < kRnnoiseFreqSize; ++i) {
        x[i] *= normf[static_cast<size_t>(i)];
    }
}

void biquad(float * y, float * mem, const float * x, const float * b, const float * a, int count) {
    for (int i = 0; i < count; ++i) {
        const float xi = x[i];
        const float yi = x[i] + mem[0];
        mem[0] = mem[1] + (b[0] * xi - a[0] * yi);
        mem[1] = b[1] * xi - a[1] * yi;
        y[i] = yi;
    }
}

struct RnnoiseAnalyzedAudioFrame {
    std::array<float, kRnnoiseFeatures> features{};
    std::array<float, kRnnoiseBands> ex{};
    std::array<float, kRnnoiseBands> ep{};
    std::array<float, kRnnoiseBands> exp{};
    std::array<std::complex<float>, kRnnoiseFreqSize> x{};
    std::array<std::complex<float>, kRnnoiseFreqSize> p{};
    bool silence = false;
};

struct RnnoiseAudioDspState {
    std::vector<float> analysis_mem;
    std::vector<float> synthesis_mem;
    std::vector<float> pitch_buf;
    std::vector<float> highpass_mem;
    std::vector<float> last_gains;
    std::vector<std::complex<float>> delayed_x;
    std::vector<std::complex<float>> delayed_p;
    std::vector<float> delayed_ex;
    std::vector<float> delayed_ep;
    std::vector<float> delayed_exp;
    std::vector<float> windowed;
    std::vector<float> pitch_frame;
    std::vector<float> synthesized;
    float last_gain = 0.0f;
    int last_period = 0;

    RnnoiseAudioDspState()
        : analysis_mem(static_cast<size_t>(kRnnoiseFrameSize), 0.0f),
          synthesis_mem(static_cast<size_t>(kRnnoiseFrameSize), 0.0f),
          pitch_buf(static_cast<size_t>(kRnnoisePitchBufSize), 0.0f),
          highpass_mem(2, 0.0f),
          last_gains(static_cast<size_t>(kRnnoiseBands), 0.0f),
          delayed_x(static_cast<size_t>(kRnnoiseFreqSize)),
          delayed_p(static_cast<size_t>(kRnnoiseFreqSize)),
          delayed_ex(static_cast<size_t>(kRnnoiseBands), 0.0f),
          delayed_ep(static_cast<size_t>(kRnnoiseBands), 0.0f),
          delayed_exp(static_cast<size_t>(kRnnoiseBands), 0.0f),
          windowed(static_cast<size_t>(kRnnoiseWindowSize), 0.0f),
          pitch_frame(static_cast<size_t>(kRnnoiseWindowSize), 0.0f),
          synthesized(static_cast<size_t>(kRnnoiseWindowSize), 0.0f) {}
};

void reset_audio_dsp_state(RnnoiseAudioDspState & state) {
    std::fill(state.analysis_mem.begin(), state.analysis_mem.end(), 0.0f);
    std::fill(state.synthesis_mem.begin(), state.synthesis_mem.end(), 0.0f);
    std::fill(state.pitch_buf.begin(), state.pitch_buf.end(), 0.0f);
    std::fill(state.highpass_mem.begin(), state.highpass_mem.end(), 0.0f);
    std::fill(state.last_gains.begin(), state.last_gains.end(), 0.0f);
    std::fill(state.delayed_x.begin(), state.delayed_x.end(), std::complex<float>{});
    std::fill(state.delayed_p.begin(), state.delayed_p.end(), std::complex<float>{});
    std::fill(state.delayed_ex.begin(), state.delayed_ex.end(), 0.0f);
    std::fill(state.delayed_ep.begin(), state.delayed_ep.end(), 0.0f);
    std::fill(state.delayed_exp.begin(), state.delayed_exp.end(), 0.0f);
    std::fill(state.windowed.begin(), state.windowed.end(), 0.0f);
    std::fill(state.pitch_frame.begin(), state.pitch_frame.end(), 0.0f);
    std::fill(state.synthesized.begin(), state.synthesized.end(), 0.0f);
    state.last_gain = 0.0f;
    state.last_period = 0;
}

RnnoiseAnalyzedAudioFrame analyze_rnnoise_audio_frame(RnnoiseAudioDspState & state, const float * input) {
    static const std::array<float, 2> a_hp = {-1.99599f, 0.99600f};
    static const std::array<float, 2> b_hp = {-2.0f, 1.0f};
    auto fft = get_real_fft_plan(static_cast<size_t>(kRnnoiseWindowSize));
    std::array<float, kRnnoiseFrameSize> highpassed{};
    std::array<float, kRnnoiseWindowSize> window_buffer{};
    std::array<float, kRnnoiseWindowSize> pitch_window{};
    std::array<float, kRnnoiseBands> log_energy{};
    RnnoiseAnalyzedAudioFrame frame;

    biquad(highpassed.data(), state.highpass_mem.data(), input, b_hp.data(), a_hp.data(), kRnnoiseFrameSize);

    std::copy(state.analysis_mem.begin(), state.analysis_mem.end(), window_buffer.begin());
    std::copy(highpassed.begin(), highpassed.end(), window_buffer.begin() + kRnnoiseFrameSize);
    std::copy(highpassed.begin(), highpassed.end(), state.analysis_mem.begin());
    std::copy(window_buffer.begin(), window_buffer.end(), state.windowed.begin());
    apply_window(state.windowed);
    fft->forward(
        {static_cast<size_t>(kRnnoiseWindowSize)},
        {static_cast<std::ptrdiff_t>(sizeof(float))},
        {static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))},
        0,
        state.windowed.data(),
        frame.x.data());
    compute_band_energy(frame.ex.data(), frame.x.data());

    std::move(
        state.pitch_buf.begin() + kRnnoiseFrameSize,
        state.pitch_buf.end(),
        state.pitch_buf.begin());
    std::copy(
        highpassed.begin(),
        highpassed.end(),
        state.pitch_buf.begin() + static_cast<std::ptrdiff_t>(kRnnoisePitchBufSize - kRnnoiseFrameSize));
    std::array<float, kRnnoisePitchBufSize / 2> pitch_buf{};
    float * pitch_input = state.pitch_buf.data();
    pitch_downsample(&pitch_input, pitch_buf.data(), kRnnoisePitchBufSize, 1);
    int pitch_index = 0;
    pitch_search(
        pitch_buf.data() + kRnnoisePitchMaxPeriod / 2,
        pitch_buf.data(),
        kRnnoisePitchFrameSize,
        kRnnoisePitchMaxPeriod - 3 * kRnnoisePitchMinPeriod,
        &pitch_index);
    pitch_index = kRnnoisePitchMaxPeriod - pitch_index;
    const float pitch_gain = remove_doubling(
        pitch_buf.data(),
        kRnnoisePitchMaxPeriod,
        kRnnoisePitchMinPeriod,
        kRnnoisePitchFrameSize,
        &pitch_index,
        state.last_period,
        state.last_gain);
    state.last_period = pitch_index;
    state.last_gain = pitch_gain;
    for (int i = 0; i < kRnnoiseWindowSize; ++i) {
        pitch_window[static_cast<size_t>(i)] =
            state.pitch_buf[static_cast<size_t>(kRnnoisePitchBufSize - kRnnoiseWindowSize - pitch_index + i)];
    }
    std::copy(pitch_window.begin(), pitch_window.end(), state.pitch_frame.begin());
    apply_window(state.pitch_frame);
    fft->forward(
        {static_cast<size_t>(kRnnoiseWindowSize)},
        {static_cast<std::ptrdiff_t>(sizeof(float))},
        {static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))},
        0,
        state.pitch_frame.data(),
        frame.p.data());
    compute_band_energy(frame.ep.data(), frame.p.data());
    compute_band_corr(frame.exp.data(), frame.x.data(), frame.p.data());
    for (int band = 0; band < kRnnoiseBands; ++band) {
        frame.exp[static_cast<size_t>(band)] =
            frame.exp[static_cast<size_t>(band)] /
            std::sqrt(0.001f + frame.ex[static_cast<size_t>(band)] * frame.ep[static_cast<size_t>(band)]);
    }
    rnnoise_dct(frame.features.data() + kRnnoiseBands, frame.exp.data());
    frame.features[2 * kRnnoiseBands] = 0.01f * (static_cast<float>(pitch_index) - 300.0f);

    float energy = 0.0f;
    float log_max = -2.0f;
    float follow = -2.0f;
    for (int band = 0; band < kRnnoiseBands; ++band) {
        float ly = std::log10(1.0e-2f + frame.ex[static_cast<size_t>(band)]);
        ly = std::max(log_max - 7.0f, std::max(follow - 1.5f, ly));
        log_max = std::max(log_max, ly);
        follow = std::max(follow - 1.5f, ly);
        energy += frame.ex[static_cast<size_t>(band)];
        log_energy[static_cast<size_t>(band)] = ly;
    }
    frame.silence = energy < 0.04f;
    if (!frame.silence) {
        rnnoise_dct(frame.features.data(), log_energy.data());
        frame.features[0] -= 12.0f;
        frame.features[1] -= 4.0f;
    }
    return frame;
}

void synthesize_rnnoise_audio_frame(
    RnnoiseAudioDspState & state,
    const RnnoiseAnalyzedAudioFrame & frame,
    const float * frame_gains,
    float * output) {
    auto fft = get_real_fft_plan(static_cast<size_t>(kRnnoiseWindowSize));
    std::array<float, kRnnoiseBands> gains{};
    std::array<float, kRnnoiseFreqSize> frequency_gains{};
    std::array<float, kRnnoiseWindowSize> inverse_buffer{};

    if (!frame.silence) {
        std::copy(frame_gains, frame_gains + kRnnoiseBands, gains.begin());
        pitch_filter(
            state.delayed_x.data(),
            state.delayed_p.data(),
            state.delayed_ex.data(),
            state.delayed_ep.data(),
            state.delayed_exp.data(),
            gains.data());
        for (int band = 0; band < kRnnoiseBands; ++band) {
            constexpr float alpha = 0.6f;
            gains[static_cast<size_t>(band)] = std::max(gains[static_cast<size_t>(band)], alpha * state.last_gains[static_cast<size_t>(band)]);
            state.last_gains[static_cast<size_t>(band)] = std::min(
                1.0f,
                gains[static_cast<size_t>(band)] *
                    (state.delayed_ex[static_cast<size_t>(band)] + 1.0e-3f) /
                    (frame.ex[static_cast<size_t>(band)] + 1.0e-3f));
        }
        interp_band_gain(frequency_gains.data(), gains.data());
        for (int i = 0; i < kRnnoiseFreqSize; ++i) {
            state.delayed_x[static_cast<size_t>(i)] *= frequency_gains[static_cast<size_t>(i)];
        }
    }

    fft->inverse(
        {static_cast<size_t>(kRnnoiseWindowSize)},
        {static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))},
        {static_cast<std::ptrdiff_t>(sizeof(float))},
        0,
        state.delayed_x.data(),
        inverse_buffer.data(),
        1.0f / static_cast<float>(kRnnoiseWindowSize));
    std::copy(inverse_buffer.begin(), inverse_buffer.end(), state.synthesized.begin());
    apply_window(state.synthesized);
    for (int i = 0; i < kRnnoiseFrameSize; ++i) {
        output[i] = state.synthesized[static_cast<size_t>(i)] + state.synthesis_mem[static_cast<size_t>(i)];
    }
    std::copy(
        state.synthesized.begin() + kRnnoiseFrameSize,
        state.synthesized.end(),
        state.synthesis_mem.begin());

    std::copy(frame.x.begin(), frame.x.end(), state.delayed_x.begin());
    std::copy(frame.p.begin(), frame.p.end(), state.delayed_p.begin());
    std::copy(frame.ex.begin(), frame.ex.end(), state.delayed_ex.begin());
    std::copy(frame.ep.begin(), frame.ep.end(), state.delayed_ep.begin());
    std::copy(frame.exp.begin(), frame.exp.end(), state.delayed_exp.begin());
}

}  // namespace

class RnnoiseSequenceGraph;

struct RnnoiseWeights {
    ~RnnoiseWeights();

    RnnoiseConfig config;
    std::filesystem::path source_path;
    std::unique_ptr<ggml_backend, BackendDeleter> backend;
    core::BackendType backend_type = core::BackendType::Cpu;
    std::shared_ptr<core::BackendWeightStore> store;
    RnnoiseConvGraphWeights conv1;
    RnnoiseConvGraphWeights conv2;
    GruGraphWeights gru1;
    GruGraphWeights gru2;
    GruGraphWeights gru3;
    DenseGraphWeights dense_out;
    DenseGraphWeights vad_dense;
    mutable std::unique_ptr<RnnoiseSequenceGraph> sequence_graph;
};

class RnnoiseFrameGraph {
public:
    explicit RnnoiseFrameGraph(const RnnoiseWeights & weights) : weights_(weights) {
        ggml_init_params params{};
        params.mem_size = kRnnoiseFrameGraphContextBytes;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ctx_.reset(ggml_init(params));
        if (!ctx_) {
            throw std::runtime_error("failed to create RNNoise graph context");
        }

        core::ModuleBuildContext build_ctx{ctx_.get(), "rnnoise.frame", weights_.backend_type};
        features_ = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({kRnnoiseFeatures})).tensor;
        conv1_memory_ = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({kRnnoiseFeatures * 2})).tensor;
        conv2_memory_ = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({weights_.config.conv1_channels * 2})).tensor;
        gru1_hidden_in_ = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({weights_.config.gru_size})).tensor;
        gru2_hidden_in_ = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({weights_.config.gru_size})).tensor;
        gru3_hidden_in_ = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({weights_.config.gru_size})).tensor;
        ones_ = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({weights_.config.gru_size})).tensor;

        auto features = core::wrap_tensor(features_, core::TensorShape::from_dims({kRnnoiseFeatures}));
        auto conv1_memory = core::wrap_tensor(conv1_memory_, core::TensorShape::from_dims({kRnnoiseFeatures * 2}));
        auto conv2_memory = core::wrap_tensor(conv2_memory_, core::TensorShape::from_dims({weights_.config.conv1_channels * 2}));
        auto gru1_hidden = core::wrap_tensor(gru1_hidden_in_, core::TensorShape::from_dims({weights_.config.gru_size}));
        auto gru2_hidden = core::wrap_tensor(gru2_hidden_in_, core::TensorShape::from_dims({weights_.config.gru_size}));
        auto gru3_hidden = core::wrap_tensor(gru3_hidden_in_, core::TensorShape::from_dims({weights_.config.gru_size}));
        auto ones = core::wrap_tensor(ones_, core::TensorShape::from_dims({weights_.config.gru_size}));

        auto conv1_input = modules::ConcatModule({0}).build(build_ctx, conv1_memory, features);
        auto conv1_output = causal_conv(build_ctx, conv1_input, weights_.conv1.streaming);
        auto conv2_input = modules::ConcatModule({0}).build(build_ctx, conv2_memory, conv1_output);
        auto conv2_output = causal_conv(build_ctx, conv2_input, weights_.conv2.streaming);
        auto next_gru1 = gru(build_ctx, conv2_output, gru1_hidden, ones, weights_.gru1);
        auto next_gru2 = gru(build_ctx, next_gru1, gru2_hidden, ones, weights_.gru2);
        auto next_gru3 = gru(build_ctx, next_gru2, gru3_hidden, ones, weights_.gru3);
        auto head = modules::ConcatModule({0}).build(build_ctx, conv2_output, next_gru1);
        head = modules::ConcatModule({0}).build(build_ctx, head, next_gru2);
        head = modules::ConcatModule({0}).build(build_ctx, head, next_gru3);
        auto gain_logits = modules::LinearModule({weights_.dense_out.in_features, weights_.dense_out.out_features, true})
                               .build(build_ctx, head, weights_.dense_out.linear);
        auto vad_logits = modules::LinearModule({weights_.vad_dense.in_features, weights_.vad_dense.out_features, true})
                              .build(build_ctx, head, weights_.vad_dense.linear);
        gains_ = sigmoid(build_ctx, gain_logits).tensor;
        vad_ = sigmoid(build_ctx, vad_logits).tensor;
        conv1_output_ = conv1_output.tensor;
        conv2_output_ = conv2_output.tensor;
        gru1_hidden_out_ = next_gru1.tensor;
        gru2_hidden_out_ = next_gru2.tensor;
        gru3_hidden_out_ = next_gru3.tensor;

        graph_ = ggml_new_graph_custom(ctx_.get(), 4096, false);
        ggml_build_forward_expand(graph_, gains_);
        ggml_build_forward_expand(graph_, vad_);
        ggml_build_forward_expand(graph_, conv1_output_);
        ggml_build_forward_expand(graph_, conv2_output_);
        ggml_build_forward_expand(graph_, gru1_hidden_out_);
        ggml_build_forward_expand(graph_, gru2_hidden_out_);
        ggml_build_forward_expand(graph_, gru3_hidden_out_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), weights_.backend.get());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate RNNoise graph buffer");
        }
        plan_ = core::create_backend_graph_plan_if_host(weights_.backend.get(), graph_);
        std::vector<float> ones_values(static_cast<size_t>(weights_.config.gru_size), 1.0f);
        ggml_backend_tensor_set(ones_, ones_values.data(), 0, ones_values.size() * sizeof(float));
    }

    ~RnnoiseFrameGraph() {
        if (plan_ != nullptr) {
            auto * backend = weights_.backend.get();
            core::free_backend_graph_plan(backend, plan_);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    void run(
        const float * features,
        const std::vector<float> & conv1_memory,
        const std::vector<float> & conv2_memory,
        const std::vector<float> & gru1_hidden,
        const std::vector<float> & gru2_hidden,
        const std::vector<float> & gru3_hidden,
        std::vector<float> & conv1_output,
        std::vector<float> & conv2_output,
        std::vector<float> & next_gru1,
        std::vector<float> & next_gru2,
        std::vector<float> & next_gru3,
        std::vector<float> & gains,
        float & vad) {
        ggml_backend_tensor_set(features_, features, 0, sizeof(float) * kRnnoiseFeatures);
        ggml_backend_tensor_set(conv1_memory_, conv1_memory.data(), 0, conv1_memory.size() * sizeof(float));
        ggml_backend_tensor_set(conv2_memory_, conv2_memory.data(), 0, conv2_memory.size() * sizeof(float));
        ggml_backend_tensor_set(gru1_hidden_in_, gru1_hidden.data(), 0, gru1_hidden.size() * sizeof(float));
        ggml_backend_tensor_set(gru2_hidden_in_, gru2_hidden.data(), 0, gru2_hidden.size() * sizeof(float));
        ggml_backend_tensor_set(gru3_hidden_in_, gru3_hidden.data(), 0, gru3_hidden.size() * sizeof(float));
        const auto status = core::compute_backend_graph(weights_.backend.get(), graph_, plan_, "RNNoise");
        ggml_backend_synchronize(weights_.backend.get());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("RNNoise GGML graph compute failed");
        }
        read_output(conv1_output_, conv1_output);
        read_output(conv2_output_, conv2_output);
        read_output(gru1_hidden_out_, next_gru1);
        read_output(gru2_hidden_out_, next_gru2);
        read_output(gru3_hidden_out_, next_gru3);
        read_output(gains_, gains);
        std::array<float, 1> vad_values{};
        ggml_backend_tensor_get(vad_, vad_values.data(), 0, sizeof(float));
        vad = vad_values[0];
    }

public:
    static core::TensorValue sigmoid(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
        return core::wrap_tensor(ggml_sigmoid(ctx.ggml, input.tensor), input.shape, GGML_TYPE_F32);
    }

    static core::TensorValue causal_conv(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const CausalConvGraphWeights & weights) {
        auto projected = modules::LinearModule({weights.linear.in_features, weights.linear.out_features, true})
                             .build(ctx, input, weights.linear.linear);
        return core::wrap_tensor(ggml_tanh(ctx.ggml, projected.tensor), projected.shape, GGML_TYPE_F32);
    }

    static core::TensorValue gru(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & hidden,
        const core::TensorValue & ones,
        const GruGraphWeights & weights) {
        const int64_t hidden_size = weights.hidden_size;
        auto input_gates = modules::LinearModule({weights.input.in_features, weights.input.out_features, true})
                               .build(ctx, input, weights.input.linear);
        auto recurrent_gates = modules::LinearModule({weights.recurrent.in_features, weights.recurrent.out_features, true})
                                   .build(ctx, hidden, weights.recurrent.linear);
        auto input_reset = modules::SliceModule({0, 0, hidden_size}).build(ctx, input_gates);
        auto input_update = modules::SliceModule({0, hidden_size, hidden_size}).build(ctx, input_gates);
        auto input_candidate = modules::SliceModule({0, hidden_size * 2, hidden_size}).build(ctx, input_gates);
        auto recurrent_reset = modules::SliceModule({0, 0, hidden_size}).build(ctx, recurrent_gates);
        auto recurrent_update = modules::SliceModule({0, hidden_size, hidden_size}).build(ctx, recurrent_gates);
        auto recurrent_candidate = modules::SliceModule({0, hidden_size * 2, hidden_size}).build(ctx, recurrent_gates);
        auto reset = sigmoid(ctx, core::wrap_tensor(ggml_add(ctx.ggml, input_reset.tensor, recurrent_reset.tensor), input_reset.shape, GGML_TYPE_F32));
        auto update = sigmoid(ctx, core::wrap_tensor(ggml_add(ctx.ggml, input_update.tensor, recurrent_update.tensor), input_update.shape, GGML_TYPE_F32));
        auto gated_recurrent_candidate = core::wrap_tensor(
            ggml_mul(ctx.ggml, reset.tensor, recurrent_candidate.tensor),
            recurrent_candidate.shape,
            GGML_TYPE_F32);
        auto candidate = core::wrap_tensor(
            ggml_tanh(ctx.ggml, ggml_add(ctx.ggml, input_candidate.tensor, gated_recurrent_candidate.tensor)),
            input_candidate.shape,
            GGML_TYPE_F32);
        auto one_minus_update = core::wrap_tensor(ggml_sub(ctx.ggml, ones.tensor, update.tensor), update.shape, GGML_TYPE_F32);
        auto candidate_part = core::wrap_tensor(ggml_mul(ctx.ggml, one_minus_update.tensor, candidate.tensor), candidate.shape, GGML_TYPE_F32);
        auto hidden_part = core::wrap_tensor(ggml_mul(ctx.ggml, update.tensor, hidden.tensor), hidden.shape, GGML_TYPE_F32);
        return core::wrap_tensor(ggml_add(ctx.ggml, candidate_part.tensor, hidden_part.tensor), hidden.shape, GGML_TYPE_F32);
    }

private:
    static void read_output(ggml_tensor * tensor, std::vector<float> & values) {
        const size_t count = static_cast<size_t>(ggml_nelements(tensor));
        values.resize(count);
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    }

    const RnnoiseWeights & weights_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * features_ = nullptr;
    ggml_tensor * conv1_memory_ = nullptr;
    ggml_tensor * conv2_memory_ = nullptr;
    ggml_tensor * gru1_hidden_in_ = nullptr;
    ggml_tensor * gru2_hidden_in_ = nullptr;
    ggml_tensor * gru3_hidden_in_ = nullptr;
    ggml_tensor * ones_ = nullptr;
    ggml_tensor * conv1_output_ = nullptr;
    ggml_tensor * conv2_output_ = nullptr;
    ggml_tensor * gru1_hidden_out_ = nullptr;
    ggml_tensor * gru2_hidden_out_ = nullptr;
    ggml_tensor * gru3_hidden_out_ = nullptr;
    ggml_tensor * gains_ = nullptr;
    ggml_tensor * vad_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_backend_graph_plan_t plan_ = nullptr;
};

struct RnnoiseSequenceState {
    std::vector<float> conv1_memory;
    std::vector<float> conv2_memory;
    std::vector<float> gru1_hidden;
    std::vector<float> gru2_hidden;
    std::vector<float> gru3_hidden;

    explicit RnnoiseSequenceState(const RnnoiseConfig & config)
        : conv1_memory(static_cast<size_t>(config.feature_size * 2), 0.0f),
          conv2_memory(static_cast<size_t>(config.conv1_channels * 2), 0.0f),
          gru1_hidden(static_cast<size_t>(config.gru_size), 0.0f),
          gru2_hidden(static_cast<size_t>(config.gru_size), 0.0f),
          gru3_hidden(static_cast<size_t>(config.gru_size), 0.0f) {}
};

struct RnnoiseSequenceRunOutput {
    std::vector<float> gains;
    std::vector<float> vad;
};

class RnnoiseSequenceGraph {
public:
    RnnoiseSequenceGraph(const RnnoiseWeights & weights, int64_t capacity_frames)
        : weights_(weights),
          capacity_frames_(capacity_frames) {
        if (capacity_frames_ <= 0) {
            throw std::runtime_error("RNNoise sequence graph capacity must be positive");
        }

        ggml_init_params params{};
        params.mem_size = kRnnoiseSequenceGraphContextBytes;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ctx_.reset(ggml_init(params));
        if (!ctx_) {
            throw std::runtime_error("failed to create RNNoise sequence graph context");
        }

        core::ModuleBuildContext build_ctx{ctx_.get(), "rnnoise.sequence", weights_.backend_type};
        features_ = core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({capacity_frames_, kRnnoiseFeatures})).tensor;
        conv1_memory_in_ = core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({kRnnoiseFeatures * 2})).tensor;
        conv2_memory_in_ = core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({weights_.config.conv1_channels * 2})).tensor;
        gru1_hidden_in_ = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({weights_.config.gru_size})).tensor;
        gru2_hidden_in_ = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({weights_.config.gru_size})).tensor;
        gru3_hidden_in_ = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({weights_.config.gru_size})).tensor;
        ones_ = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({weights_.config.gru_size})).tensor;

        auto features = core::wrap_tensor(features_, core::TensorShape::from_dims({capacity_frames_, kRnnoiseFeatures}));
        auto conv1_memory = core::wrap_tensor(conv1_memory_in_, core::TensorShape::from_dims({kRnnoiseFeatures * 2}));
        auto conv2_memory = core::wrap_tensor(conv2_memory_in_, core::TensorShape::from_dims({weights_.config.conv1_channels * 2}));
        auto gru1_hidden = core::wrap_tensor(gru1_hidden_in_, core::TensorShape::from_dims({weights_.config.gru_size}));
        auto gru2_hidden = core::wrap_tensor(gru2_hidden_in_, core::TensorShape::from_dims({weights_.config.gru_size}));
        auto gru3_hidden = core::wrap_tensor(gru3_hidden_in_, core::TensorShape::from_dims({weights_.config.gru_size}));
        auto ones = core::wrap_tensor(ones_, core::TensorShape::from_dims({weights_.config.gru_size}));
        core::TensorValue gains_flat;
        core::TensorValue vad_flat;
        for (int64_t frame = 0; frame < capacity_frames_; ++frame) {
            auto feature_frame = modules::SliceModule({0, frame, 1}).build(build_ctx, features);
            feature_frame = core::reshape_tensor(
                build_ctx,
                core::ensure_backend_addressable_layout(build_ctx, feature_frame),
                core::TensorShape::from_dims({kRnnoiseFeatures}));
            auto conv1_input = modules::ConcatModule({0}).build(build_ctx, conv1_memory, feature_frame);
            auto conv1_frame = RnnoiseFrameGraph::causal_conv(build_ctx, conv1_input, weights_.conv1.streaming);
            auto conv2_input = modules::ConcatModule({0}).build(build_ctx, conv2_memory, conv1_frame);
            auto conv2_frame = RnnoiseFrameGraph::causal_conv(build_ctx, conv2_input, weights_.conv2.streaming);

            auto conv1_tail = modules::SliceModule({0, kRnnoiseFeatures, kRnnoiseFeatures}).build(build_ctx, conv1_memory);
            conv1_memory = modules::ConcatModule({0}).build(build_ctx, conv1_tail, feature_frame);
            auto conv2_tail = modules::SliceModule({0, weights_.config.conv1_channels, weights_.config.conv1_channels}).build(build_ctx, conv2_memory);
            conv2_memory = modules::ConcatModule({0}).build(build_ctx, conv2_tail, conv1_frame);

            conv2_frame = core::reshape_tensor(
                build_ctx,
                core::ensure_backend_addressable_layout(build_ctx, conv2_frame),
                core::TensorShape::from_dims({weights_.config.conv2_channels}));
            gru1_hidden = RnnoiseFrameGraph::gru(build_ctx, conv2_frame, gru1_hidden, ones, weights_.gru1);
            gru2_hidden = RnnoiseFrameGraph::gru(build_ctx, gru1_hidden, gru2_hidden, ones, weights_.gru2);
            gru3_hidden = RnnoiseFrameGraph::gru(build_ctx, gru2_hidden, gru3_hidden, ones, weights_.gru3);
            auto head = modules::ConcatModule({0}).build(build_ctx, conv2_frame, gru1_hidden);
            head = modules::ConcatModule({0}).build(build_ctx, head, gru2_hidden);
            head = modules::ConcatModule({0}).build(build_ctx, head, gru3_hidden);
            auto gain_logits = modules::LinearModule({weights_.dense_out.in_features, weights_.dense_out.out_features, true})
                                   .build(build_ctx, head, weights_.dense_out.linear);
            auto vad_logits = modules::LinearModule({weights_.vad_dense.in_features, weights_.vad_dense.out_features, true})
                                  .build(build_ctx, head, weights_.vad_dense.linear);
            auto gain = RnnoiseFrameGraph::sigmoid(build_ctx, gain_logits);
            auto vad = RnnoiseFrameGraph::sigmoid(build_ctx, vad_logits);
            gains_flat = gains_flat.valid() ? modules::ConcatModule({0}).build(build_ctx, gains_flat, gain) : gain;
            vad_flat = vad_flat.valid() ? modules::ConcatModule({0}).build(build_ctx, vad_flat, vad) : vad;
        }
        gains_ = core::reshape_tensor(
            build_ctx,
            core::ensure_backend_addressable_layout(build_ctx, gains_flat),
            core::TensorShape::from_dims({capacity_frames_, weights_.config.gain_bands})).tensor;
        vad_ = core::reshape_tensor(
            build_ctx,
            core::ensure_backend_addressable_layout(build_ctx, vad_flat),
            core::TensorShape::from_dims({capacity_frames_, 1})).tensor;
        conv1_memory_out_ = conv1_memory.tensor;
        conv2_memory_out_ = conv2_memory.tensor;
        gru1_hidden_out_ = gru1_hidden.tensor;
        gru2_hidden_out_ = gru2_hidden.tensor;
        gru3_hidden_out_ = gru3_hidden.tensor;
        ggml_set_output(gains_);
        ggml_set_output(vad_);
        ggml_set_output(conv1_memory_out_);
        ggml_set_output(conv2_memory_out_);
        ggml_set_output(gru1_hidden_out_);
        ggml_set_output(gru2_hidden_out_);
        ggml_set_output(gru3_hidden_out_);

        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, gains_);
        ggml_build_forward_expand(graph_, vad_);
        ggml_build_forward_expand(graph_, conv1_memory_out_);
        ggml_build_forward_expand(graph_, conv2_memory_out_);
        ggml_build_forward_expand(graph_, gru1_hidden_out_);
        ggml_build_forward_expand(graph_, gru2_hidden_out_);
        ggml_build_forward_expand(graph_, gru3_hidden_out_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), weights_.backend.get());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate RNNoise sequence graph buffer");
        }
        std::vector<float> one_values(static_cast<size_t>(weights_.config.gru_size), 1.0f);
        ggml_backend_tensor_set(ones_, one_values.data(), 0, one_values.size() * sizeof(float));
        plan_ = core::create_backend_graph_plan_if_host(weights_.backend.get(), graph_);
    }

    ~RnnoiseSequenceGraph() {
        if (plan_ != nullptr) {
            core::free_backend_graph_plan(weights_.backend.get(), plan_);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool supports(int64_t frames) const noexcept {
        return frames > 0 && frames <= capacity_frames_;
    }

    int64_t capacity_frames() const noexcept {
        return capacity_frames_;
    }

    RnnoiseSequenceRunOutput run(
        const float * features,
        int64_t frames,
        RnnoiseSequenceState & state) const {
        if (features == nullptr || frames <= 0 || frames > capacity_frames_) {
            throw std::runtime_error("RNNoise sequence graph input shape mismatch");
        }
        std::vector<float> padded_features(static_cast<size_t>(capacity_frames_ * kRnnoiseFeatures), 0.0f);
        std::copy(features, features + frames * kRnnoiseFeatures, padded_features.begin());
        ggml_backend_tensor_set(features_, padded_features.data(), 0, padded_features.size() * sizeof(float));
        ggml_backend_tensor_set(conv1_memory_in_, state.conv1_memory.data(), 0, state.conv1_memory.size() * sizeof(float));
        ggml_backend_tensor_set(conv2_memory_in_, state.conv2_memory.data(), 0, state.conv2_memory.size() * sizeof(float));
        ggml_backend_tensor_set(gru1_hidden_in_, state.gru1_hidden.data(), 0, state.gru1_hidden.size() * sizeof(float));
        ggml_backend_tensor_set(gru2_hidden_in_, state.gru2_hidden.data(), 0, state.gru2_hidden.size() * sizeof(float));
        ggml_backend_tensor_set(gru3_hidden_in_, state.gru3_hidden.data(), 0, state.gru3_hidden.size() * sizeof(float));
        const auto status = core::compute_backend_graph(weights_.backend.get(), graph_, plan_, "RNNoise sequence");
        ggml_backend_synchronize(weights_.backend.get());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("RNNoise sequence GGML graph compute failed");
        }
        RnnoiseSequenceRunOutput output;
        output.gains.resize(static_cast<size_t>(capacity_frames_ * weights_.config.gain_bands));
        output.vad.resize(static_cast<size_t>(capacity_frames_));
        ggml_backend_tensor_get(gains_, output.gains.data(), 0, output.gains.size() * sizeof(float));
        ggml_backend_tensor_get(vad_, output.vad.data(), 0, output.vad.size() * sizeof(float));
        output.gains.resize(static_cast<size_t>(frames * weights_.config.gain_bands));
        output.vad.resize(static_cast<size_t>(frames));
        read_output(conv1_memory_out_, state.conv1_memory);
        read_output(conv2_memory_out_, state.conv2_memory);
        read_output(gru1_hidden_out_, state.gru1_hidden);
        read_output(gru2_hidden_out_, state.gru2_hidden);
        read_output(gru3_hidden_out_, state.gru3_hidden);
        return output;
    }

private:
    static void read_output(ggml_tensor * tensor, std::vector<float> & values) {
        const size_t count = static_cast<size_t>(ggml_nelements(tensor));
        values.resize(count);
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    }

    const RnnoiseWeights & weights_;
    int64_t capacity_frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * features_ = nullptr;
    ggml_tensor * conv1_memory_in_ = nullptr;
    ggml_tensor * conv2_memory_in_ = nullptr;
    ggml_tensor * gru1_hidden_in_ = nullptr;
    ggml_tensor * gru2_hidden_in_ = nullptr;
    ggml_tensor * gru3_hidden_in_ = nullptr;
    ggml_tensor * ones_ = nullptr;
    ggml_tensor * gains_ = nullptr;
    ggml_tensor * vad_ = nullptr;
    ggml_tensor * conv1_memory_out_ = nullptr;
    ggml_tensor * conv2_memory_out_ = nullptr;
    ggml_tensor * gru1_hidden_out_ = nullptr;
    ggml_tensor * gru2_hidden_out_ = nullptr;
    ggml_tensor * gru3_hidden_out_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_backend_graph_plan_t plan_ = nullptr;
};

RnnoiseWeights::~RnnoiseWeights() = default;

struct RnnoiseStreamingSession::State {
    std::vector<float> conv1_memory;
    std::vector<float> conv2_memory;
    std::vector<float> gru1_hidden;
    std::vector<float> gru2_hidden;
    std::vector<float> gru3_hidden;
    std::vector<float> conv1_output;
    std::vector<float> conv2_output;
    std::unique_ptr<RnnoiseFrameGraph> frame_graph;
    RnnoiseAudioDspState dsp;

    explicit State(const RnnoiseConfig & config)
        : conv1_memory(static_cast<size_t>(config.feature_size * 2), 0.0f),
          conv2_memory(static_cast<size_t>(config.conv1_channels * 2), 0.0f),
          gru1_hidden(static_cast<size_t>(config.gru_size), 0.0f),
          gru2_hidden(static_cast<size_t>(config.gru_size), 0.0f),
          gru3_hidden(static_cast<size_t>(config.gru_size), 0.0f),
          conv1_output(static_cast<size_t>(config.conv1_channels), 0.0f),
          conv2_output(static_cast<size_t>(config.conv2_channels), 0.0f) {}
};

RnnoiseModel::RnnoiseModel() = default;
RnnoiseModel::~RnnoiseModel() = default;
RnnoiseModel::RnnoiseModel(RnnoiseModel &&) noexcept = default;
RnnoiseModel & RnnoiseModel::operator=(RnnoiseModel &&) noexcept = default;

RnnoiseModel::RnnoiseModel(std::shared_ptr<const RnnoiseWeights> weights)
    : weights_(std::move(weights)) {
    if (weights_ == nullptr) {
        throw std::runtime_error("RNNoise model requires weights");
    }
}

RnnoiseModel RnnoiseModel::load_from_safetensors(const std::filesystem::path & checkpoint_path) {
    return load_from_safetensors(checkpoint_path, core::BackendConfig{});
}

RnnoiseModel RnnoiseModel::load_from_safetensors(
    const std::filesystem::path & checkpoint_path,
    const core::BackendConfig & backend_config) {
    auto source = assets::open_tensor_source(checkpoint_path);
    auto weights = std::make_shared<RnnoiseWeights>();
    weights->source_path = checkpoint_path;
    weights->backend.reset(core::init_backend(backend_config));
    weights->backend_type = core::backend_type(weights->backend.get());
    weights->store = std::make_shared<core::BackendWeightStore>(
        weights->backend.get(),
        weights->backend_type,
        "RNNoise",
        16ull * 1024ull * 1024ull);
    weights->conv1 = make_rnnoise_conv_graph(*weights->store, load_conv1d(*source, "conv1", 128, 65, 3));
    weights->conv2 = make_rnnoise_conv_graph(*weights->store, load_conv1d(*source, "conv2", 384, 128, 3));
    weights->gru1 = load_gru_graph(*weights->store, *source, "gru1", 384, 384);
    weights->gru2 = load_gru_graph(*weights->store, *source, "gru2", 384, 384);
    weights->gru3 = load_gru_graph(*weights->store, *source, "gru3", 384, 384);
    weights->dense_out = load_dense_graph(*weights->store, *source, "dense_out", 32, 1536);
    weights->vad_dense = load_dense_graph(*weights->store, *source, "vad_dense", 1, 1536);
    weights->store->upload();
    source->release_storage();
    return RnnoiseModel(std::move(weights));
}

const RnnoiseConfig & RnnoiseModel::config() const noexcept {
    static const RnnoiseConfig empty;
    return weights_ == nullptr ? empty : weights_->config;
}

const std::filesystem::path & RnnoiseModel::source_path() const noexcept {
    static const std::filesystem::path empty;
    return weights_ == nullptr ? empty : weights_->source_path;
}

std::unique_ptr<RnnoiseStreamingSession> RnnoiseModel::create_streaming_session() const {
    if (weights_ == nullptr) {
        throw std::runtime_error("RNNoise model is not initialized");
    }
    return std::make_unique<RnnoiseStreamingSession>(weights_);
}

RnnoiseSequenceOutput RnnoiseModel::infer_features(
    const std::vector<float> & features,
    int64_t frames,
    int64_t feature_size) const {
    if (frames <= 0 || feature_size != config().feature_size ||
        static_cast<int64_t>(features.size()) != frames * feature_size) {
        throw std::runtime_error("RNNoise feature sequence shape mismatch");
    }

    const int64_t required_capacity = std::min<int64_t>(frames, kRnnoiseSequenceChunkFrames);
    if (!weights_->sequence_graph || !weights_->sequence_graph->supports(required_capacity)) {
        weights_->sequence_graph.reset();
        weights_->sequence_graph = std::make_unique<RnnoiseSequenceGraph>(*weights_, required_capacity);
    }

    RnnoiseSequenceState state(config());
    RnnoiseSequenceOutput output;
    output.frames = frames;
    output.gain_bands = config().gain_bands;
    output.gains.reserve(static_cast<size_t>(frames * config().gain_bands));
    output.vad.reserve(static_cast<size_t>(frames));
    int64_t offset = 0;
    while (offset < frames) {
        const int64_t chunk_frames = std::min<int64_t>(frames - offset, weights_->sequence_graph->capacity_frames());
        const auto chunk = weights_->sequence_graph->run(
            features.data() + static_cast<std::ptrdiff_t>(offset * feature_size),
            chunk_frames,
            state);
        output.gains.insert(output.gains.end(), chunk.gains.begin(), chunk.gains.end());
        output.vad.insert(output.vad.end(), chunk.vad.begin(), chunk.vad.end());
        offset += chunk_frames;
    }
    return output;
}

RnnoiseAlignmentPlan rnnoise_alignment_plan(int64_t input_samples, const RnnoiseProcessOptions & options) noexcept {
    static_assert(
        kRnnoiseOutputDelaySamples == 2 * static_cast<int64_t>(kRnnoiseFrameSize),
        "RNNoise group delay must stay two analysis frames");
    RnnoiseAlignmentPlan plan;
    if (input_samples <= 0) {
        return plan;
    }
    plan.crop_offset = options.compensate_output_delay ? kRnnoiseOutputDelaySamples : 0;
    plan.output_samples = input_samples;
    plan.vad_frames = (input_samples + kRnnoiseFrameSize - 1) / kRnnoiseFrameSize;
    plan.frames = (input_samples + plan.crop_offset + kRnnoiseFrameSize - 1) / kRnnoiseFrameSize;
    plan.padded_samples = plan.frames * kRnnoiseFrameSize;
    return plan;
}

RnnoiseWaveformOutput RnnoiseModel::process_mono_48k(
    const std::vector<float> & waveform,
    const RnnoiseProcessOptions & options) const {
    if (waveform.empty()) {
        throw std::runtime_error("RNNoise waveform input is empty");
    }
    const auto plan = rnnoise_alignment_plan(static_cast<int64_t>(waveform.size()), options);
    const int64_t frames = plan.frames;
    std::vector<float> padded(static_cast<size_t>(plan.padded_samples), 0.0f);
    std::copy(waveform.begin(), waveform.end(), padded.begin());
    std::vector<float> output(padded.size(), 0.0f);
    std::vector<float> vad;
    vad.reserve(static_cast<size_t>(frames));

    const int64_t required_capacity = std::min<int64_t>(frames, kRnnoiseSequenceChunkFrames);
    if (!weights_->sequence_graph || !weights_->sequence_graph->supports(required_capacity)) {
        weights_->sequence_graph.reset();
        weights_->sequence_graph = std::make_unique<RnnoiseSequenceGraph>(*weights_, required_capacity);
    }

    RnnoiseAudioDspState dsp;
    RnnoiseSequenceState sequence_state(config());
    std::vector<RnnoiseAnalyzedAudioFrame> analyzed(static_cast<size_t>(weights_->sequence_graph->capacity_frames()));
    std::vector<float> features(static_cast<size_t>(weights_->sequence_graph->capacity_frames() * kRnnoiseFeatures), 0.0f);

    int64_t offset = 0;
    while (offset < frames) {
        const int64_t chunk_frames = std::min<int64_t>(frames - offset, weights_->sequence_graph->capacity_frames());
        std::fill(features.begin(), features.end(), 0.0f);
        for (int64_t frame = 0; frame < chunk_frames; ++frame) {
            const auto frame_offset = static_cast<size_t>((offset + frame) * kRnnoiseFrameSize);
            analyzed[static_cast<size_t>(frame)] = analyze_rnnoise_audio_frame(dsp, padded.data() + frame_offset);
            std::copy(
                analyzed[static_cast<size_t>(frame)].features.begin(),
                analyzed[static_cast<size_t>(frame)].features.end(),
                features.begin() + static_cast<std::ptrdiff_t>(frame * kRnnoiseFeatures));
        }
        const auto chunk = weights_->sequence_graph->run(features.data(), chunk_frames, sequence_state);
        vad.insert(vad.end(), chunk.vad.begin(), chunk.vad.end());
        for (int64_t frame = 0; frame < chunk_frames; ++frame) {
            const auto frame_offset = static_cast<size_t>((offset + frame) * kRnnoiseFrameSize);
            synthesize_rnnoise_audio_frame(
                dsp,
                analyzed[static_cast<size_t>(frame)],
                chunk.gains.data() + static_cast<std::ptrdiff_t>(frame * config().gain_bands),
                output.data() + frame_offset);
        }
        offset += chunk_frames;
    }
    if (static_cast<int64_t>(output.size()) < plan.crop_offset + plan.output_samples) {
        throw std::runtime_error("RNNoise synthesis output is shorter than the delay-compensated crop");
    }
    std::vector<float> aligned(
        output.begin() + static_cast<std::ptrdiff_t>(plan.crop_offset),
        output.begin() + static_cast<std::ptrdiff_t>(plan.crop_offset + plan.output_samples));
    if (static_cast<int64_t>(vad.size()) > plan.vad_frames) {
        vad.resize(static_cast<size_t>(plan.vad_frames));
    }
    return RnnoiseWaveformOutput{kRnnoiseSampleRate, std::move(aligned), std::move(vad)};
}

RnnoiseStreamingSession::RnnoiseStreamingSession(std::shared_ptr<const RnnoiseWeights> weights)
    : weights_(std::move(weights)),
      state_(weights_ == nullptr ? nullptr : std::make_unique<State>(weights_->config)) {
    if (weights_ == nullptr || state_ == nullptr) {
        throw std::runtime_error("RNNoise streaming session requires weights");
    }
    state_->frame_graph = std::make_unique<RnnoiseFrameGraph>(*weights_);
}

RnnoiseStreamingSession::~RnnoiseStreamingSession() = default;
RnnoiseStreamingSession::RnnoiseStreamingSession(RnnoiseStreamingSession &&) noexcept = default;
RnnoiseStreamingSession & RnnoiseStreamingSession::operator=(RnnoiseStreamingSession &&) noexcept = default;

void RnnoiseStreamingSession::reset() {
    if (state_ == nullptr) {
        throw std::runtime_error("RNNoise streaming session is not initialized");
    }
    std::fill(state_->conv1_memory.begin(), state_->conv1_memory.end(), 0.0f);
    std::fill(state_->conv2_memory.begin(), state_->conv2_memory.end(), 0.0f);
    std::fill(state_->gru1_hidden.begin(), state_->gru1_hidden.end(), 0.0f);
    std::fill(state_->gru2_hidden.begin(), state_->gru2_hidden.end(), 0.0f);
    std::fill(state_->gru3_hidden.begin(), state_->gru3_hidden.end(), 0.0f);
    reset_audio_dsp_state(state_->dsp);
}

RnnoiseFrameOutput RnnoiseStreamingSession::process_frame(const float * features, int64_t feature_size) {
    if (features == nullptr || weights_ == nullptr || state_ == nullptr) {
        throw std::runtime_error("RNNoise streaming session is not initialized");
    }
    const auto & config = weights_->config;
    if (feature_size != config.feature_size) {
        throw std::runtime_error("RNNoise frame feature size mismatch");
    }

    RnnoiseFrameOutput output;
    output.gains.resize(static_cast<size_t>(config.gain_bands));
    if (!state_->frame_graph) {
        throw std::runtime_error("RNNoise frame graph is not initialized");
    }
    state_->frame_graph->run(
        features,
        state_->conv1_memory,
        state_->conv2_memory,
        state_->gru1_hidden,
        state_->gru2_hidden,
        state_->gru3_hidden,
        state_->conv1_output,
        state_->conv2_output,
        state_->gru1_hidden,
        state_->gru2_hidden,
        state_->gru3_hidden,
        output.gains,
        output.vad);
    shift_append(state_->conv1_memory, features, config.feature_size, 3);
    shift_append(state_->conv2_memory, state_->conv1_output.data(), config.conv1_channels, 3);
    return output;
}

float RnnoiseStreamingSession::process_audio_frame(const float * input, float * output, int64_t samples) {
    if (input == nullptr || output == nullptr || weights_ == nullptr || state_ == nullptr) {
        throw std::runtime_error("RNNoise audio frame session is not initialized");
    }
    if (samples != kRnnoiseFrameSize) {
        throw std::runtime_error("RNNoise audio frame size mismatch");
    }

    const auto analysis = analyze_rnnoise_audio_frame(state_->dsp, input);
    const auto frame_output = process_frame(analysis.features.data(), kRnnoiseFeatures);
    synthesize_rnnoise_audio_frame(state_->dsp, analysis, frame_output.gains.data(), output);
    return frame_output.vad;
}

}  // namespace engine::audio
