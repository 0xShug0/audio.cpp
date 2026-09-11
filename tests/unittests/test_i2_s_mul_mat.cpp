// Numeric checks for the ternary GGML_TYPE_I2_S matmul, which is what the
// VibeASR language model runs on.
//
// The op is reachable through plain ggml_mul_mat with an I2_S src0 and an F32
// src1, and it quantizes src1 to int8 itself. Its arithmetic is entirely
// integral up to a single float multiply at the end, so the references below are
// exact rather than approximate: the reference computes sum(w*q) as int32 with
// w in {-1,0,+1} and multiplies by the same combined scale, which is bit for bit
// what the kernel does after it subtracts the row sum out of its {0,1,2} codes.
// A tolerance here would hide a wrong packing that happens to be close.

#include "test_assert.h"

#include <ggml-cpu.h>
#include <ggml.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

extern "C" {
size_t ggml_type_extra_bytes(enum ggml_type type);
void   ggml_i2_s_to_float  (const void  * x, float * y, int64_t n);
size_t ggml_i2_s_from_float(const float * x, void  * y, int64_t n);
void   ggml_i8_s_quantize_act(const float * x, int8_t * q, int64_t n, float * scale, int32_t * sum);
}

using engine::test::require;
using engine::test::require_close;
using engine::test::require_eq;

namespace {

constexpr size_t kCtxBytes = 512u * 1024 * 1024;

const float * inband_scale(const ggml_tensor * t) {
    return reinterpret_cast<const float *>(
        static_cast<const char *>(t->data) + ggml_nbytes(t) - ggml_type_extra_bytes(t->type));
}

// Decode the packing by hand instead of going through ggml_i2_s_to_float, so
// that the layout the kernel reads is pinned independently of the dequantizer
// that was written alongside it: 128 values per 32-byte group, byte gp holding
// the values at group-relative positions gp, 32+gp, 64+gp and 96+gp in bit pairs
// 6, 4, 2, 0.
std::vector<int> unpack_i2_s(const void * data, int64_t n) {
    const uint8_t * q = static_cast<const uint8_t *>(data);

    std::vector<int> out(static_cast<size_t>(n));
    for (int64_t base = 0; base < n; base += 128) {
        const uint8_t * group = q + base / 4;

        for (int gp = 0; gp < 32; ++gp) {
            const uint8_t b = group[gp];

            out[base +  0 + gp] = static_cast<int>((b >> 6) & 3) - 1;
            out[base + 32 + gp] = static_cast<int>((b >> 4) & 3) - 1;
            out[base + 64 + gp] = static_cast<int>((b >> 2) & 3) - 1;
            out[base + 96 + gp] = static_cast<int>((b >> 0) & 3) - 1;
        }
    }
    return out;
}

// Ternary weights with a deliberate mix of all three values and a stride that is
// coprime with 32 and 128, so no code lands in the same bit position of every
// byte and a swapped shift shows up immediately.
std::vector<float> ternary_pattern(size_t n, float scale, int phase) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        const int k = static_cast<int>((i * 7 + phase) % 3);
        v[i] = scale * static_cast<float>(k - 1);
    }
    return v;
}

std::vector<float> activation_pattern(size_t n, float phase, float scale) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        const float x = static_cast<float>(i);
        v[i] = scale * (std::sin(phase + 0.11f * x) + 0.35f * std::cos(0.05f * x - phase));
    }
    return v;
}

void compute(ggml_context * ctx, ggml_tensor * result, int n_threads) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
}

// Reference for one batch of ggml_mul_mat(weight_i2_s, activations_f32).
//
// The activation quantization is the op's own, called directly: the point of
// this test is the packing, the code-to-weight bias, and the scale combination,
// not to re-derive an absmax.
std::vector<float> reference(const std::vector<int> & w,   // [K*N], ternary
                            const std::vector<float> & a,  // [K*M]
                            int64_t K, int64_t N, int64_t M,
                            float w_scale) {
    std::vector<float> out(static_cast<size_t>(N * M));

    std::vector<int8_t> q(static_cast<size_t>(K));
    for (int64_t col = 0; col < M; ++col) {
        float   act_scale = 0.0f;
        int32_t act_sum   = 0;
        ggml_i8_s_quantize_act(a.data() + col * K, q.data(), K, &act_scale, &act_sum);

        const float d = w_scale * act_scale;

        for (int64_t oc = 0; oc < N; ++oc) {
            int32_t dot = 0;
            for (int64_t k = 0; k < K; ++k) {
                dot += w[static_cast<size_t>(oc * K + k)] * static_cast<int32_t>(q[static_cast<size_t>(k)]);
            }
            out[static_cast<size_t>(col * N + oc)] = static_cast<float>(dot) * d;
        }
    }
    return out;
}

std::string shape_label(const char * what, int64_t K, int64_t N, int64_t M, int nth) {
    return std::string(what) + " K=" + std::to_string(K) + " N=" + std::to_string(N) +
           " M=" + std::to_string(M) + " nth=" + std::to_string(nth);
}

// The packer is whole-tensor, so this also checks that a row-major [K, N] weight
// packs into groups that never straddle a row -- true because every I2_S row
// length in the model is a multiple of 128, and false the moment K is not.
void test_pack_layout() {
    ggml_init_params ip = { kCtxBytes, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    const int64_t K = 256;
    const int64_t N = 3;

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_I2_S, K, N);

    const std::vector<float> values = ternary_pattern(static_cast<size_t>(K * N), 0.75f, 1);
    ggml_i2_s_from_float(values.data(), w->data, K * N);

    require_close(*inband_scale(w), 0.75f, 0.0f, "pack layout scale");

    // Row size is ceil(K/128)*32 bytes, and nb[1] has to agree or the kernel
    // walks into the wrong row.
    require_eq(static_cast<int64_t>(w->nb[1]), K / 4, "pack layout row stride");

    const std::vector<int> codes = unpack_i2_s(w->data, K * N);
    for (int64_t i = 0; i < K * N; ++i) {
        const float expected = values[static_cast<size_t>(i)];
        require_close(static_cast<float>(codes[static_cast<size_t>(i)]) * 0.75f, expected, 0.0f,
                      "pack layout value " + std::to_string(i));
    }

    // And the shipped dequantizer agrees with the hand decode.
    std::vector<float> dequantized(static_cast<size_t>(K * N));
    ggml_i2_s_to_float(w->data, dequantized.data(), K * N);
    for (int64_t i = 0; i < K * N; ++i) {
        require_close(dequantized[static_cast<size_t>(i)], values[static_cast<size_t>(i)], 0.0f,
                      "pack layout dequantize " + std::to_string(i));
    }

    ggml_free(ctx);
}

void test_mul_mat(int n_threads, int64_t K, int64_t N, int64_t M) {
    ggml_init_params ip = { kCtxBytes, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_I2_S, K, N);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  K, M);

    const float w_scale = 0.031f;
    const std::vector<float> w_values = ternary_pattern(static_cast<size_t>(K * N), w_scale, static_cast<int>(N));
    ggml_i2_s_from_float(w_values.data(), w->data, K * N);

    const std::vector<float> a_values = activation_pattern(static_cast<size_t>(K * M), 0.4f, 1.7f);
    std::memcpy(a->data, a_values.data(), a_values.size() * sizeof(float));

    ggml_tensor * result = ggml_mul_mat(ctx, w, a);
    require_eq(result->ne[0], N, "mul_mat ne0");
    require_eq(result->ne[1], M, "mul_mat ne1");

    compute(ctx, result, n_threads);

    const std::vector<float> expected =
        reference(unpack_i2_s(w->data, K * N), a_values, K, N, M, *inband_scale(w));

    const float * got = static_cast<const float *>(result->data);
    for (int64_t i = 0; i < N * M; ++i) {
        require_close(got[i], expected[static_cast<size_t>(i)], 0.0f,
                      shape_label("mul_mat", K, N, M, n_threads) + " element " + std::to_string(i));
    }

    ggml_free(ctx);
}

// src1 with a third dimension, so the row indexing has to walk nb12 and the
// output has to walk nb2. The weight is shared across the batch.
void test_mul_mat_batched(int n_threads) {
    ggml_init_params ip = { kCtxBytes, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    const int64_t K = 384;
    const int64_t N = 70;
    const int64_t M = 3;
    const int64_t B = 4;

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_I2_S, K, N);
    ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,  K, M, B);

    const std::vector<float> w_values = ternary_pattern(static_cast<size_t>(K * N), 0.02f, 2);
    ggml_i2_s_from_float(w_values.data(), w->data, K * N);

    const std::vector<float> a_values = activation_pattern(static_cast<size_t>(K * M * B), 1.1f, 0.9f);
    std::memcpy(a->data, a_values.data(), a_values.size() * sizeof(float));

    ggml_tensor * result = ggml_mul_mat(ctx, w, a);
    require_eq(result->ne[2], B, "batched ne2");

    compute(ctx, result, n_threads);

    const std::vector<int> codes = unpack_i2_s(w->data, K * N);
    const float * got = static_cast<const float *>(result->data);

    for (int64_t ib = 0; ib < B; ++ib) {
        const std::vector<float> slice(a_values.begin() + static_cast<size_t>(ib * K * M),
                                       a_values.begin() + static_cast<size_t>((ib + 1) * K * M));
        const std::vector<float> expected = reference(codes, slice, K, N, M, *inband_scale(w));

        for (int64_t i = 0; i < N * M; ++i) {
            require_close(got[ib * N * M + i], expected[static_cast<size_t>(i)], 0.0f,
                          "batched batch " + std::to_string(ib) + " element " + std::to_string(i));
        }
    }

    ggml_free(ctx);
}

// Every code at its maximum (2) against every activation at its maximum (127) is
// the worst case for the int16 accumulation inside the AVX2 body: 32 lanes each
// summing 32 products of 2*127, i.e. 16256, which is why the kernel widens to
// int32 every eight groups rather than at the end of the row. K spans more than
// eight groups so the flush actually has to happen.
void test_accumulator_headroom(int n_threads) {
    ggml_init_params ip = { kCtxBytes, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    const int64_t K = 1536;
    const int64_t N = 5;

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_I2_S, K, N);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  K, 1);

    // All +1: every packed code is 2.
    const std::vector<float> w_values(static_cast<size_t>(K * N), 0.5f);
    ggml_i2_s_from_float(w_values.data(), w->data, K * N);

    // Flat, so the absmax is 1 and every quantized activation is exactly +127.
    const std::vector<float> a_values(static_cast<size_t>(K), 1.0f);
    std::memcpy(a->data, a_values.data(), a_values.size() * sizeof(float));

    ggml_tensor * result = ggml_mul_mat(ctx, w, a);
    compute(ctx, result, n_threads);

    const float expected = static_cast<float>(K * 127) * (0.5f * (1.0f / 127.0f));
    const float * got = static_cast<const float *>(result->data);
    for (int64_t oc = 0; oc < N; ++oc) {
        require_close(got[oc], expected, 0.0f, "headroom output " + std::to_string(oc));
    }

    ggml_free(ctx);
}

// An all-zero weight tensor has scale 0, and an all-zero activation row has
// scale 0. Neither may produce a NaN: the scales are multipliers here, and a
// reciprocal convention would divide by zero in both cases.
void test_degenerate_scales(int n_threads) {
    ggml_init_params ip = { kCtxBytes, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    const int64_t K = 128;
    const int64_t N = 8;
    const int64_t M = 2;

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_I2_S, K, N);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  K, M);

    const std::vector<float> zeros_w(static_cast<size_t>(K * N), 0.0f);
    ggml_i2_s_from_float(zeros_w.data(), w->data, K * N);
    require_close(*inband_scale(w), 0.0f, 0.0f, "degenerate weight scale");

    std::vector<float> a_values = activation_pattern(static_cast<size_t>(K * M), 0.2f, 1.0f);
    std::memcpy(a->data, a_values.data(), a_values.size() * sizeof(float));

    ggml_tensor * zero_w_result = ggml_mul_mat(ctx, w, a);
    compute(ctx, zero_w_result, n_threads);

    const float * got = static_cast<const float *>(zero_w_result->data);
    for (int64_t i = 0; i < N * M; ++i) {
        require(std::isfinite(got[i]), "degenerate weight output finite " + std::to_string(i));
        require_close(got[i], 0.0f, 0.0f, "degenerate weight output " + std::to_string(i));
    }

    // Now a real weight against a zero activation row. Only the second column is
    // zeroed, so the first still has to come out right.
    const std::vector<float> w_values = ternary_pattern(static_cast<size_t>(K * N), 0.25f, 0);
    ggml_i2_s_from_float(w_values.data(), w->data, K * N);

    for (int64_t k = 0; k < K; ++k) {
        a_values[static_cast<size_t>(K + k)] = 0.0f;
    }
    std::memcpy(a->data, a_values.data(), a_values.size() * sizeof(float));

    ggml_tensor * zero_act_result = ggml_mul_mat(ctx, w, a);
    compute(ctx, zero_act_result, n_threads);

    const std::vector<float> expected =
        reference(unpack_i2_s(w->data, K * N), a_values, K, N, M, *inband_scale(w));

    got = static_cast<const float *>(zero_act_result->data);
    for (int64_t i = 0; i < N * M; ++i) {
        require(std::isfinite(got[i]), "degenerate activation output finite " + std::to_string(i));
        require_close(got[i], expected[static_cast<size_t>(i)], 0.0f,
                      "degenerate activation output " + std::to_string(i));
    }
    for (int64_t oc = 0; oc < N; ++oc) {
        require_close(got[N + oc], 0.0f, 0.0f, "degenerate activation zero column " + std::to_string(oc));
    }

    ggml_free(ctx);
}

}  // namespace

int main() {
    try {
        test_pack_layout();

        // Single- and multi-threaded: the activation quantization and the output
        // pass are separated by a barrier, and threads split the output features,
        // so a missing barrier or an overlapping split shows up as a thread-count
        // dependent result.
        for (int nth : {1, 4}) {
            // K=128 is one group. K=1024 is exactly the eight groups the AVX2
            // body accumulates in int16 before widening; K=1152 is that plus one,
            // so the second, shorter flush runs too.
            // N=1 is a single output row (the decode-time lm_head shape), N=64 is
            // exactly one output-channel chunk, N=100 spans two, and N=70 is an
            // unaligned span.
            test_mul_mat(nth, 128,  1,  1);
            test_mul_mat(nth, 128,  64, 1);
            test_mul_mat(nth, 128,  100, 5);
            test_mul_mat(nth, 1024, 70, 1);
            test_mul_mat(nth, 1152, 70, 3);
            // Fewer output features than threads: the tail threads get an empty
            // range and must not write outside it.
            test_mul_mat(nth, 256,  2,  2);
            test_mul_mat_batched(nth);
            test_accumulator_headroom(nth);
            test_degenerate_scales(nth);
        }
    } catch (const std::exception & e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }

    std::cout << "all i2_s mul_mat tests passed\n";
    return 0;
}
