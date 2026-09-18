#pragma once

#include <cmath>
#include <cstring>

#include <ggml.h>

namespace kokoro_ggml::cpu_detail {

// Match ggml's F32 mean (sequential double accumulation, then float division)
// while evaluating independent channels in parallel and reusing the output as
// scratch for centered values. Keep every F32 rounding step explicit.
inline void kokoro_adain_cpu(
    ggml_tensor * dst,
    const ggml_tensor * x,
    const ggml_tensor * gamma,
    const ggml_tensor * beta,
    int ith,
    int nth,
    void * userdata) {
    const float eps = *static_cast<const float *>(userdata);
    const int64_t frames = x->ne[0];
    const int64_t rows = x->ne[1] * x->ne[2];
    for (int64_t row = rows * ith / nth; row < rows * (ith + 1) / nth; ++row) {
        const auto * src = static_cast<const float *>(x->data) + row * frames;
        auto * out = static_cast<float *>(dst->data) + row * frames;
        double sum = 0.0;
        for (int64_t t = 0; t < frames; ++t) sum += static_cast<double>(src[t]);
        const float mean = static_cast<float>(sum) / static_cast<float>(frames);
        double squares = 0.0;
        for (int64_t t = 0; t < frames; ++t) {
            const float centered = src[t] - mean;
            out[t] = centered;
            const float square = centered * centered;
            squares += static_cast<double>(square);
        }
        const float variance = static_cast<float>(squares) / static_cast<float>(frames);
        const float stddev = std::sqrt(variance + eps);
        const float scale = static_cast<const float *>(gamma->data)[row % x->ne[1]];
        const float shift = static_cast<const float *>(beta->data)[row % x->ne[1]];
        for (int64_t t = 0; t < frames; ++t) {
            const float normalized = out[t] / stddev;
            const float scaled = normalized * scale;
            out[t] = scaled + shift;
        }
    }
}

inline void kokoro_snake_cpu(
    ggml_tensor * dst,
    const ggml_tensor * x,
    const ggml_tensor * alpha,
    int ith,
    int nth,
    void *) {
    const int64_t rows = x->ne[1] * x->ne[2];
    for (int64_t row = rows * ith / nth; row < rows * (ith + 1) / nth; ++row) {
        const float a = static_cast<const float *>(alpha->data)[row % x->ne[1]];
        const auto * src = reinterpret_cast<const float *>(static_cast<const char *>(x->data) +
            (row % x->ne[1]) * x->nb[1] + (row / x->ne[1]) * x->nb[2]);
        auto * out = static_cast<float *>(dst->data) + row * x->ne[0];
        for (int64_t t = 0; t < x->ne[0]; ++t) {
            const float ax = src[t] * a;
            const float s = std::sin(ax);
            const float square = s * s;
            const float fraction = square / a;
            out[t] = src[t] + fraction;
        }
    }
}

// Partition complete output rows, avoiding inter-thread cache-line sharing.
// This only copies/pads F32 samples; convolution reduction order is unchanged.
template <typename ConvWeightsT>
void kokoro_im2col_rows(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto & conv = *static_cast<const ConvWeightsT *>(userdata);
    const ggml_tensor * input = dst->src[0];
    const int64_t begin = dst->ne[1] * ith / nth;
    const int64_t end = dst->ne[1] * (ith + 1) / nth;
    auto * out = static_cast<float *>(dst->data);
    for (int64_t frame = begin; frame < end; ++frame) {
        const int64_t base = frame * conv.stride - conv.padding;
        const bool interior = base >= 0 && base + (conv.kernel - 1) * conv.dilation < input->ne[0];
        for (int64_t channel = 0; channel < conv.in_channels; ++channel) {
            const auto * src = reinterpret_cast<const float *>(
                static_cast<const char *>(input->data) + channel * input->nb[1]);
            float * row = out + frame * dst->ne[0] + channel * conv.kernel;
            if (conv.dilation == 1 && interior) {
                std::memcpy(row, src + base, conv.kernel * sizeof(float));
            } else if (interior) {
                for (int64_t k = 0; k < conv.kernel; ++k) {
                    row[k] = src[base + k * conv.dilation];
                }
            } else {
                for (int64_t k = 0; k < conv.kernel; ++k) {
                    const int64_t index = base + k * conv.dilation;
                    row[k] = index >= 0 && index < input->ne[0] ? src[index] : 0.0f;
                }
            }
        }
    }
}
}  // namespace kokoro_ggml::cpu_detail
