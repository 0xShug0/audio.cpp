// Numeric checks for the VibeASR INT8 pipeline additions to ggml:
// GGML_TYPE_I8_S / GGML_TYPE_I2_S and the five fused ops built on them.
//
// Every op is compared against a plain-loop reference computed from the same
// inputs. The references duplicate the requantization ordering deliberately,
// including the parts that lose range -- ggml_mul_mat_add_relu clamps after
// rounding, so the absmax that sets the output scale still counts the negatives
// that are about to become zero. The point of these tests is to pin the
// arithmetic that the SIMD kernels and the model port have to reproduce, not to
// judge whether that arithmetic is optimal.

#include "test_assert.h"

#include <ggml-cpu.h>
#include <ggml.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// The in-band scale layout is internal to ggml, so the tests reach for the same
// declarations the implementation uses rather than re-deriving the offsets.
extern "C" {
size_t ggml_type_extra_bytes(enum ggml_type type);
void   ggml_i8_s_to_float  (const void  * x, float * y, int64_t n);
size_t ggml_i8_s_from_float(const float * x, void  * y, int64_t n);
void   ggml_i2_s_to_float  (const void  * x, float * y, int64_t n);
size_t ggml_i2_s_from_float(const float * x, void  * y, int64_t n);
}

using engine::test::require;
using engine::test::require_close;
using engine::test::require_eq;

namespace {

constexpr size_t kCtxBytes = 256u * 1024 * 1024;

float * inband_scale(ggml_tensor * t) {
    return reinterpret_cast<float *>(
        static_cast<char *>(t->data) + ggml_nbytes(t) - ggml_type_extra_bytes(t->type));
}

// Quantize into an existing I8_S tensor, returning the scale that was chosen.
float fill_i8_s(ggml_tensor * t, const std::vector<float> & values) {
    require_eq(static_cast<int64_t>(values.size()), ggml_nelements(t), "fill_i8_s size");
    ggml_i8_s_from_float(values.data(), t->data, ggml_nelements(t));
    return *inband_scale(t);
}

std::vector<float> read_i8_s(ggml_tensor * t) {
    std::vector<float> out(ggml_nelements(t));
    ggml_i8_s_to_float(t->data, out.data(), ggml_nelements(t));
    return out;
}

std::vector<float> patterned(size_t n, float phase, float scale) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        const float x = static_cast<float>(i);
        v[i] = scale * (std::sin(phase + 0.13f * x) + 0.4f * std::cos(0.07f * x - phase));
    }
    return v;
}

// The reference requantizer, mirroring ggml_i8_s_requantize: absmax over the
// whole result -- taken before the relu clamp, so a value about to be zeroed
// still widens the scale -- then clamp, then round ties to even.
//
// std::rint, not std::round: the op rounds ties to even so that its AVX2 and
// NEON bodies agree with their scalar tails. std::round would round ties away
// from zero and make the byte-exact comparisons below fail on any tie.
std::vector<int8_t> requantize_ref(const std::vector<float> & values, bool relu, float * scale_out) {
    float amax = 0.0f;
    for (float v : values) {
        amax = std::max(amax, std::fabs(v));
    }

    const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
    const float lo = relu ? 0.0f : -127.0f;

    std::vector<int8_t> q(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        float v = values[i] * id;
        v = std::max(lo, std::min(127.0f, v));
        q[i] = static_cast<int8_t>(std::rint(v));
    }

    *scale_out = amax != 0.0f ? amax / 127.0f : 0.0f;
    return q;
}

void compare_i8(ggml_tensor * got, const std::vector<int8_t> & want, float want_scale,
                const std::string & label) {
    require_eq(static_cast<int64_t>(want.size()), ggml_nelements(got), label + " size");

    const auto * q = static_cast<const int8_t *>(got->data);
    for (size_t i = 0; i < want.size(); ++i) {
        // Exact: both sides do the same rounding on the same floats. A single
        // LSB of drift would mean the scale differs, which is what matters.
        require_eq(static_cast<int>(q[i]), static_cast<int>(want[i]),
                   label + " value at " + std::to_string(i));
    }

    require_close(*inband_scale(got), want_scale, 1e-9f, label + " scale");
}

// Runs a single-node graph on the CPU backend with the given thread count. The
// four requantizing ops reduce an absmax across threads, so the result has to be
// independent of nth -- these tests run every case at 1 and 4.
void compute(ggml_context * ctx, ggml_tensor * result, int n_threads) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
}

// ---------------------------------------------------------------- round trips

void test_i8_s_round_trip() {
    const int64_t n = 4096;
    const auto values = patterned(n, 0.3f, 2.5f);

    std::vector<uint8_t> buf(n + ggml_type_extra_bytes(GGML_TYPE_I8_S));
    const size_t written = ggml_i8_s_from_float(values.data(), buf.data(), n);
    require_eq(written, buf.size(), "i8_s written bytes");

    std::vector<float> back(n);
    ggml_i8_s_to_float(buf.data(), back.data(), n);

    float amax = 0.0f;
    for (float v : values) {
        amax = std::max(amax, std::fabs(v));
    }
    // One step of the quantization grid is amax/127; rounding puts every value
    // within half of that.
    const float tol = amax / 127.0f * 0.5f + 1e-6f;

    for (int64_t i = 0; i < n; ++i) {
        require_close(back[i], values[i], tol, "i8_s round trip at " + std::to_string(i));
    }

    std::cout << "i8_s round trip: n=" << n << " step=" << amax / 127.0f << " OK\n";
}

void test_i2_s_round_trip() {
    // Ternary input: the type represents {-d, 0, +d} exactly, so this has to
    // round trip bit for bit rather than approximately.
    const int64_t n = 128 * 7;
    const float   d = 0.0731f;

    std::vector<float> values(n);
    for (int64_t i = 0; i < n; ++i) {
        const int code = static_cast<int>(i * 7 % 3) - 1;  // -1, 0, +1 cycling
        values[i] = static_cast<float>(code) * d;
    }

    const size_t payload = static_cast<size_t>(n / 128) * 32;
    std::vector<uint8_t> buf(payload + ggml_type_extra_bytes(GGML_TYPE_I2_S));
    const size_t written = ggml_i2_s_from_float(values.data(), buf.data(), n);
    require_eq(written, buf.size(), "i2_s written bytes");

    std::vector<float> back(n);
    ggml_i2_s_to_float(buf.data(), back.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        require_close(back[i], values[i], 1e-7f, "i2_s round trip at " + std::to_string(i));
    }

    // Check the packing itself, not just the round trip: byte gp of a group
    // holds positions gp, 32+gp, 64+gp, 96+gp in bit pairs 6, 4, 2, 0. A
    // self-consistent but differently-ordered packing would pass the round trip
    // above while being unreadable by the SIMD kernels.
    for (int gp = 0; gp < 32; ++gp) {
        const uint8_t b = buf[gp];
        for (int sub = 0; sub < 4; ++sub) {
            const int   code  = (b >> (6 - 2 * sub)) & 3;
            const int   want  = static_cast<int>((sub * 32 + gp) * 7 % 3) - 1;
            const float value = static_cast<float>(code - 1);
            require_close(value, static_cast<float>(want), 1e-7f,
                          "i2_s bit layout at byte " + std::to_string(gp) +
                          " pair " + std::to_string(sub));
        }
    }

    std::cout << "i2_s round trip: n=" << n << " exact, bit layout OK\n";
}

// ---------------------------------------------------------------------- ops

void test_add_scaled(int n_threads) {
    const int64_t C = 96;    // channels, ne[0]
    const int64_t L = 37;    // positions

    ggml_init_params ip = { kCtxBytes, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a     = ggml_new_tensor_2d(ctx, GGML_TYPE_I8_S, C, L);
    ggml_tensor * b     = ggml_new_tensor_2d(ctx, GGML_TYPE_I8_S, C, L);
    ggml_tensor * gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,  C);

    const auto a_f = patterned(C * L, 0.1f, 1.7f);
    const auto b_f = patterned(C * L, 1.9f, 0.6f);

    fill_i8_s(a, a_f);
    fill_i8_s(b, b_f);

    const auto g = patterned(C, 0.5f, 0.25f);
    std::memcpy(gamma->data, g.data(), g.size() * sizeof(float));

    ggml_tensor * result = ggml_add_scaled(ctx, a, b, gamma);
    compute(ctx, result, n_threads);

    // Reference works from the dequantized inputs, so it sees exactly what the
    // op sees -- the input quantization error is shared, not compounded.
    const auto a_q = read_i8_s(a);
    const auto b_q = read_i8_s(b);

    std::vector<float> want(C * L);
    for (int64_t i = 0; i < C * L; ++i) {
        want[i] = a_q[i] * g[i % C] + b_q[i];
    }

    float want_scale = 0.0f;
    const auto want_q = requantize_ref(want, false, &want_scale);
    compare_i8(result, want_q, want_scale, "add_scaled nth=" + std::to_string(n_threads));

    ggml_free(ctx);
    std::cout << "add_scaled: C=" << C << " L=" << L << " nth=" << n_threads << " OK\n";
}

void test_rms_norm_scaled(int n_threads) {
    const int64_t C = 128;
    const int64_t L = 29;
    const float   eps = 1e-5f;

    ggml_init_params ip = { kCtxBytes, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a     = ggml_new_tensor_2d(ctx, GGML_TYPE_I8_S, C, L);
    ggml_tensor * gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,  C);

    const auto a_f = patterned(C * L, 0.7f, 3.1f);
    fill_i8_s(a, a_f);

    const auto g = patterned(C, 1.1f, 0.9f);
    std::memcpy(gamma->data, g.data(), g.size() * sizeof(float));

    ggml_tensor * result = ggml_rms_norm_scaled(ctx, a, gamma, eps);
    compute(ctx, result, n_threads);

    // Reference normalizes in the float domain. The op instead cancels the input
    // scale and keeps the sum of squares in integers, which is the same value
    // computed a different way -- so this also checks that the eps rescaling is
    // right, since eps is the one term that does not cancel.
    const auto a_q = read_i8_s(a);

    std::vector<float> want(C * L);
    for (int64_t row = 0; row < L; ++row) {
        double sum_sq = 0.0;
        for (int64_t i = 0; i < C; ++i) {
            const double v = a_q[row * C + i];
            sum_sq += v * v;
        }
        const float rms_inv = 1.0f / std::sqrt(static_cast<float>(sum_sq / C) + eps);
        for (int64_t i = 0; i < C; ++i) {
            want[row * C + i] = a_q[row * C + i] * rms_inv * g[i];
        }
    }

    float want_scale = 0.0f;
    const auto want_q = requantize_ref(want, false, &want_scale);

    // The two paths reach the same value through different arithmetic, so a
    // borderline element can round to either side of a grid step. Compare the
    // dequantized results with a tolerance of one step instead of demanding
    // identical bytes.
    require_close(*inband_scale(result), want_scale, want_scale * 1e-4f,
                  "rms_norm_scaled scale nth=" + std::to_string(n_threads));

    const auto * q = static_cast<const int8_t *>(result->data);
    for (int64_t i = 0; i < C * L; ++i) {
        require_close(static_cast<float>(q[i]), static_cast<float>(want_q[i]), 1.0f,
                      "rms_norm_scaled value at " + std::to_string(i));
    }

    ggml_free(ctx);
    std::cout << "rms_norm_scaled: C=" << C << " L=" << L << " nth=" << n_threads << " OK\n";
}

// IC and OC are varied by the caller rather than fixed: IC decides how much of
// the contraction the SIMD path covers and how much falls to the scalar tail,
// and OC decides whether the output-channel chunking loop wraps around.
void test_mul_mat_add(bool relu, int n_threads, int64_t IC, int64_t OC) {
    const int64_t N  = 23;

    ggml_init_params ip = { kCtxBytes, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * w    = ggml_new_tensor_2d(ctx, GGML_TYPE_I8_S, IC, OC);
    ggml_tensor * x    = ggml_new_tensor_2d(ctx, GGML_TYPE_I8_S, IC, N);
    ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,  OC);

    fill_i8_s(w, patterned(IC * OC, 0.2f, 0.8f));
    fill_i8_s(x, patterned(IC * N,  1.3f, 2.2f));

    const auto b = patterned(OC, 0.9f, 0.15f);
    std::memcpy(bias->data, b.data(), b.size() * sizeof(float));

    ggml_tensor * result = relu ? ggml_mul_mat_add_relu(ctx, w, x, bias)
                                : ggml_mul_mat_add(ctx, w, x, bias);
    compute(ctx, result, n_threads);

    require_eq(result->ne[0], OC, "mul_mat_add ne0");
    require_eq(result->ne[1], N,  "mul_mat_add ne1");
    require_eq(static_cast<int>(result->type), static_cast<int>(GGML_TYPE_I8_S),
               "mul_mat_add type");

    const auto w_q = read_i8_s(w);
    const auto x_q = read_i8_s(x);

    std::vector<float> want(OC * N);
    for (int64_t col = 0; col < N; ++col) {
        for (int64_t oc = 0; oc < OC; ++oc) {
            double acc = 0.0;
            for (int64_t k = 0; k < IC; ++k) {
                acc += static_cast<double>(w_q[oc * IC + k]) * x_q[col * IC + k];
            }
            want[col * OC + oc] = static_cast<float>(acc) + b[oc];
        }
    }

    float want_scale = 0.0f;
    const auto want_q = requantize_ref(want, relu, &want_scale);

    const std::string label = std::string(relu ? "mul_mat_add_relu" : "mul_mat_add") +
                              " IC=" + std::to_string(IC) + " OC=" + std::to_string(OC) +
                              " nth=" + std::to_string(n_threads);

    require_close(*inband_scale(result), want_scale, want_scale * 1e-4f, label + " scale");

    const auto * q = static_cast<const int8_t *>(result->data);
    bool saw_negative_input = false;
    for (int64_t i = 0; i < OC * N; ++i) {
        require_close(static_cast<float>(q[i]), static_cast<float>(want_q[i]), 1.0f,
                      label + " value at " + std::to_string(i));
        if (want[i] < 0.0f) saw_negative_input = true;
    }

    if (relu) {
        // Without negatives in the pre-activation the clamp would be untested.
        require(saw_negative_input, label + ": inputs never went negative");
        for (int64_t i = 0; i < OC * N; ++i) {
            require(q[i] >= 0, label + " left a negative at " + std::to_string(i));
        }
    }

    ggml_free(ctx);
    std::cout << label << " N=" << N << " OK\n";
}

// K is the contraction length, so it decides how much of the dot product the SIMD
// path covers. N is the number of positions, which is what the depthwise path
// batches -- so it decides whether the chunking loop wraps around.
void test_mul_mat_add_depthwise(int n_threads, int64_t K, int64_t N) {
    const int64_t C = 40;   // channels

    ggml_init_params ip = { kCtxBytes, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    // ne[1] == 1 with ne[2] > 1 is what selects the depthwise path.
    ggml_tensor * w    = ggml_new_tensor_3d(ctx, GGML_TYPE_I8_S, K, 1, C);
    ggml_tensor * x    = ggml_new_tensor_3d(ctx, GGML_TYPE_I8_S, K, N, C);
    ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,  C);

    fill_i8_s(w, patterned(K * C,     0.4f, 1.1f));
    fill_i8_s(x, patterned(K * N * C, 1.7f, 1.9f));

    const auto b = patterned(C, 0.2f, 0.3f);
    std::memcpy(bias->data, b.data(), b.size() * sizeof(float));

    ggml_tensor * result = ggml_mul_mat_add(ctx, w, x, bias);
    compute(ctx, result, n_threads);

    const auto w_q = read_i8_s(w);
    const auto x_q = read_i8_s(x);

    // Output index is ch*N + col: channel-major, one row of N per channel.
    std::vector<float> want(C * N);
    for (int64_t ch = 0; ch < C; ++ch) {
        for (int64_t col = 0; col < N; ++col) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                acc += static_cast<double>(w_q[ch * K + k]) * x_q[ch * N * K + col * K + k];
            }
            want[ch * N + col] = static_cast<float>(acc) + b[ch];
        }
    }

    float want_scale = 0.0f;
    const auto want_q = requantize_ref(want, false, &want_scale);

    const std::string label = "mul_mat_add depthwise K=" + std::to_string(K) +
                              " N=" + std::to_string(N) +
                              " nth=" + std::to_string(n_threads);
    require_close(*inband_scale(result), want_scale, want_scale * 1e-4f, label + " scale");

    const auto * q = static_cast<const int8_t *>(result->data);
    for (int64_t i = 0; i < C * N; ++i) {
        require_close(static_cast<float>(q[i]), static_cast<float>(want_q[i]), 1.0f,
                      label + " value at " + std::to_string(i));
    }

    ggml_free(ctx);
    std::cout << label << ": C=" << C << " OK\n";
}

void test_im2col_asym(int n_threads) {
    const int64_t IW  = 33;
    const int64_t IC  = 5;
    const int64_t KW  = 4;
    const int     s0  = 2;
    const int     lp0 = 3;   // asymmetric on purpose: the whole reason this op
    const int     rp0 = 0;   // exists is that ggml_im2col cannot express it
    const int     d0  = 1;

    ggml_init_params ip = { kCtxBytes, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * kernel = ggml_new_tensor_3d(ctx, GGML_TYPE_I8_S, KW, IC, 1);
    ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_I8_S, IW, IC, 1);

    fill_i8_s(kernel, patterned(KW * IC, 0.1f, 1.0f));
    const float in_scale = fill_i8_s(input, patterned(IW * IC, 0.8f, 1.4f));

    ggml_tensor * result = ggml_im2col_asym(ctx, kernel, input, s0, 0, lp0, rp0, 0,
                                            d0, 0, false, GGML_TYPE_I8_S);

    const int64_t OW = (IW + lp0 + rp0 - d0 * (KW - 1) - 1) / s0 + 1;
    require_eq(result->ne[0], IC * KW, "im2col_asym ne0");
    require_eq(result->ne[1], OW,      "im2col_asym ne1");

    compute(ctx, result, n_threads);

    const auto * in  = static_cast<const int8_t *>(input->data);
    const auto * out = static_cast<const int8_t *>(result->data);

    for (int64_t iow = 0; iow < OW; ++iow) {
        for (int64_t iic = 0; iic < IC; ++iic) {
            for (int64_t ikw = 0; ikw < KW; ++ikw) {
                const int64_t iiw  = iow * s0 + ikw * d0 - lp0;
                const int8_t  want = (iiw < 0 || iiw >= IW) ? 0 : in[iic * IW + iiw];
                require_eq(static_cast<int>(out[iow * (IC * KW) + iic * KW + ikw]),
                           static_cast<int>(want),
                           "im2col_asym at ow=" + std::to_string(iow) +
                           " ic=" + std::to_string(iic) + " kw=" + std::to_string(ikw));
            }
        }
    }

    // Rearrangement only, so the scale must pass through untouched.
    require_close(*inband_scale(result), in_scale, 0.0f, "im2col_asym scale passthrough");

    ggml_free(ctx);
    std::cout << "im2col_asym: IW=" << IW << " KW=" << KW << " lp0=" << lp0
              << " OW=" << OW << " nth=" << n_threads << " OK\n";
}

// The VibeASR encoder flips its activations between channel-major and
// length-major with ggml_cont(ggml_permute(...)), which lands in
// ggml_compute_forward_dup. A byte copy moves the payload but not the in-band
// scale, and the permuted view it is handed has no scale of its own, so the
// scale has to be read through view_src. Getting this wrong leaves the copy
// holding whatever was in the buffer -- values stay right while everything
// downstream is off by an arbitrary factor.
void test_i8_s_cont_permute(int n_threads) {
    const int64_t C = 5;
    const int64_t L = 33;

    ggml_init_params ip = { kCtxBytes, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_I8_S, C, L);
    const float src_scale = fill_i8_s(src, patterned(C * L, 0.6f, 1.9f));

    ggml_tensor * result = ggml_cont(ctx, ggml_permute(ctx, src, 1, 0, 2, 3));
    require_eq(result->ne[0], L, "cont(permute) ne0");
    require_eq(result->ne[1], C, "cont(permute) ne1");

    compute(ctx, result, n_threads);

    const auto * in  = static_cast<const int8_t *>(src->data);
    const auto * out = static_cast<const int8_t *>(result->data);
    for (int64_t ic = 0; ic < C; ++ic) {
        for (int64_t il = 0; il < L; ++il) {
            require_eq(static_cast<int>(out[ic * L + il]), static_cast<int>(in[il * C + ic]),
                       "cont(permute) value at c=" + std::to_string(ic) + " l=" + std::to_string(il));
        }
    }

    // Rearrangement only: the same scale has to come out the other side.
    require_close(*inband_scale(result), src_scale, 0.0f, "cont(permute) scale carried over");

    ggml_free(ctx);
    std::cout << "cont(permute) i8_s: C=" << C << " L=" << L
              << " nth=" << n_threads << " OK\n";
}

}  // namespace

int main() {
    try {
        test_i8_s_round_trip();
        test_i2_s_round_trip();

        // Every requantizing op reduces an absmax across threads, so each is run
        // single-threaded and multi-threaded; a missing barrier shows up as a
        // scale that depends on the thread count.
        for (int nth : {1, 4}) {
            test_add_scaled(nth);
            test_rms_norm_scaled(nth);
            // IC=64: SIMD only, no tail. IC=67: three elements land in the
            // scalar tail, which a truncating block count would silently drop.
            // IC=13: shorter than one 32-byte step, so entirely scalar.
            // OC=48 stays inside one output-channel chunk, OC=100 spans two.
            test_mul_mat_add(false, nth, 64, 48);
            test_mul_mat_add(true,  nth, 64, 48);
            test_mul_mat_add(false, nth, 67, 48);
            test_mul_mat_add(false, nth, 13, 48);
            test_mul_mat_add(false, nth, 67, 100);
            // K=7 is shorter than any vector step, so the dot is all scalar;
            // K=36 runs the 32-byte body plus a 4-element tail. N=19 stays inside
            // one position chunk, N=150 spans three.
            test_mul_mat_add_depthwise(nth, 7,  19);
            test_mul_mat_add_depthwise(nth, 36, 19);
            test_mul_mat_add_depthwise(nth, 7,  150);
            test_im2col_asym(nth);
            test_i8_s_cont_permute(nth);
        }
    } catch (const std::exception & e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }

    std::cout << "all i8_s fused op tests passed\n";
    return 0;
}
