#include "../../src/models/kokoro_tts/cpu_kernels.h"

#include <ggml-cpu.h>

#include <cstring>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

struct Conv {
    int64_t in_channels;
    int64_t kernel;
    int64_t stride;
    int64_t padding;
    int64_t dilation;
};

void compare(ggml_context * ctx, ggml_tensor * expected, ggml_tensor * actual, int threads,
             const char * kernel, bool bit_exact) {
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, expected);
    ggml_build_forward_expand(graph, actual);
    for (int repeat = 0; repeat < 2; ++repeat) {
        if (ggml_graph_compute_with_ctx(ctx, graph, threads) != GGML_STATUS_SUCCESS ||
            ggml_nbytes(expected) != ggml_nbytes(actual)) {
            throw std::runtime_error(std::string(kernel) + " CPU kernel execution failed");
        }
        if (bit_exact) {
            if (std::memcmp(expected->data, actual->data, ggml_nbytes(expected)) != 0) {
                throw std::runtime_error(std::string(kernel) + " CPU kernel is not bit-exact with ggml reference");
            }
            continue;
        }
        const auto * want = static_cast<const float *>(expected->data);
        const auto * got = static_cast<const float *>(actual->data);
        for (int64_t i = 0; i < ggml_nelements(expected); ++i) {
            const float abs_error = std::abs(want[i] - got[i]);
            const float tolerance = 2.e-5f + 2.e-5f * std::abs(want[i]);
            if (!std::isfinite(want[i]) || !std::isfinite(got[i]) || abs_error > tolerance) {
                throw std::runtime_error(std::string(kernel) + " CPU kernel differs from ggml reference at element " +
                    std::to_string(i) + ": expected " + std::to_string(want[i]) +
                    ", got " + std::to_string(got[i]));
            }
        }
    }
}

int main() {
    size_t cases = 0;
    for (int threads : {1, 8}) {
        for (int width : {1, 7, 33, 129}) for (int channels : {1, 3, 16})
        for (int kernel : {1, 3, 7}) for (int stride : {1, 2})
        for (int dilation : {1, 3}) for (int padding : {0, 8}) for (int gap : {0, 5}) {
            const int numerator = width + 2 * padding - dilation * (kernel - 1) - 1;
            if (numerator < 0) {
                continue;
            }
            ggml_context * ctx = ggml_init({16 * 1024 * 1024, nullptr, false});
            Conv conv{channels, kernel, stride, padding, dilation};
            ggml_tensor * storage = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width + gap, channels, 1);
            for (int64_t i = 0; i < ggml_nelements(storage); ++i)
                static_cast<float *>(storage->data)[i] = static_cast<float>((i * 17) % 101 - 50) / 13.0f;
            ggml_tensor * input = ggml_view_3d(ctx, storage, width, channels, 1,
                storage->nb[1], storage->nb[2], 0);
            ggml_tensor * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kernel, channels, 2);
            ggml_tensor * expected = ggml_im2col(ctx, weight, input, stride, 0, padding, 0,
                dilation, 0, false, GGML_TYPE_F32);
            ggml_tensor * args[] = {input};
            ggml_tensor * actual = ggml_custom_4d(ctx, GGML_TYPE_F32, kernel * channels,
                numerator / stride + 1, 1, 1, args, 1,
                kokoro_ggml::cpu_detail::kokoro_im2col_rows<Conv>, GGML_N_TASKS_MAX, &conv);
            compare(ctx, expected, actual, threads, "im2col", true);
            ggml_free(ctx);
            ++cases;
        }
        for (int width : {1, 31, 127}) for (int channels : {1, 3, 16}) {
            ggml_context * ctx = ggml_init({16 * 1024 * 1024, nullptr, false});
            ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, channels, 2);
            ggml_tensor * alpha = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, channels, 1);
            for (int64_t i = 0; i < ggml_nelements(input); ++i)
                static_cast<float *>(input->data)[i] = static_cast<float>((i * 17) % 101 - 50) / 13.0f;
            for (int i = 0; i < channels; ++i) {
                static_cast<float *>(alpha->data)[i] = 0.1f + i * 0.37f;
            }
            ggml_tensor * sine = ggml_sin(ctx, ggml_mul(ctx, input, alpha));
            ggml_tensor * expected = ggml_add(ctx, input, ggml_div(ctx, ggml_mul(ctx, sine, sine), alpha));
            ggml_tensor * actual = ggml_map_custom2(ctx, input, alpha,
                kokoro_ggml::cpu_detail::kokoro_snake_cpu, GGML_N_TASKS_MAX, nullptr);
            compare(ctx, expected, actual, threads, "snake", false);
            ggml_free(ctx);
            ++cases;
        }
    }
    for (int threads : {1, 8}) for (int width : {1, 31, 127, 4097})
    for (int channels : {1, 3, 16}) for (int mode : {0, 1, 2}) {
        ggml_context * ctx = ggml_init({16 * 1024 * 1024, nullptr, false});
        ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, channels, 2);
        ggml_tensor * gamma = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, channels, 1);
        ggml_tensor * beta = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, channels, 1);
        for (int64_t i = 0; i < ggml_nelements(input); ++i) {
            static_cast<float *>(input->data)[i] = mode == 0 ? 1.0f :
                static_cast<float>((i * 17) % 101 - 50) * (mode == 1 ? 0.17f : 1.e-8f);
        }
        for (int i = 0; i < channels; ++i) {
            static_cast<float *>(gamma->data)[i] = 0.1f + i * 0.37f;
            static_cast<float *>(beta->data)[i] = -0.2f + i * 0.03f;
        }
        float eps = 1.e-5f;
        ggml_tensor * mean = ggml_mean(ctx, input);
        ggml_tensor * centered = ggml_sub(ctx, input, ggml_repeat(ctx, mean, input));
        ggml_tensor * variance = ggml_mean(ctx, ggml_mul(ctx, centered, centered));
        ggml_tensor * stddev = ggml_sqrt(ctx, ggml_scale_bias(ctx, variance, 1.0f, eps));
        ggml_tensor * normalized = ggml_div(ctx, centered, ggml_repeat(ctx, stddev, input));
        ggml_tensor * expected = ggml_add(ctx, ggml_mul(ctx, normalized,
            ggml_repeat(ctx, gamma, input)), ggml_repeat(ctx, beta, input));
        ggml_tensor * actual = ggml_map_custom3(ctx, input, gamma, beta,
            kokoro_ggml::cpu_detail::kokoro_adain_cpu, GGML_N_TASKS_MAX, &eps);
        compare(ctx, expected, actual, threads, "AdaIN", false);
        ggml_free(ctx);
        ++cases;
    }
    std::cout << "PASS: " << cases
              << " kernel parity cases, each executed twice (1/8 threads); im2col is bit-exact and "
                 "floating-point arithmetic uses cross-platform tolerance.\n";
}
