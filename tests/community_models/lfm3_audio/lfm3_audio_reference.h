#pragma once

// Host fp64 reference math for the LFM3-Audio layer stacks: RMSNorm, linear,
// the short conv, gated attention (QK RMSNorm, NEOX RoPE with an optional
// zero tail, windowed masks, per-head sigmoid gate), the LFM2/LFM3 layer with
// optional sandwich norms, and the whole stack with its final norm. Weights
// are read by HF name from a TensorSource, so a graph and its reference see
// the same values. Shared read-only by the blocks, encoder, backbone, depth
// and codec tests, with the tolerances, weight modes, report and backend
// options they have in common.

#include "engine/community_models/lfm3_audio/blocks.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "test_assert.h"

#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace lfm3_audio_test {

namespace lfm3 = engine::community_models::lfm3_audio;
namespace blocks = lfm3::blocks;

using Values = std::vector<double>;  // row-major, outermost dimension first

// F32 mode against the fp64 reference. GPU F32 matmuls of more than one row are not F32: Metal's mul_mm (> 8 rows)
// rounds both operands to half (kernel_mul_mm_f32_f32 uses half tiles) and CUDA's mul_mat_f runs on TF32 tensor cores.
inline constexpr double kReferenceTolerance = 1e-5;
inline constexpr double kGpuReferenceTolerance = 5e-3;

inline double reference_tolerance(const engine::core::ExecutionContext & execution) {
    return execution.backend_type() == engine::core::BackendType::Cpu ? kReferenceTolerance : kGpuReferenceTolerance;
}

inline lfm3::Lfm3WeightOptions reference_mode() {
    return {lfm3::Lfm3WeightType::F32, true};
}

inline lfm3::Lfm3WeightOptions native_mode() {
    return {lfm3::Lfm3WeightType::Native, true};
}

inline const char * mode_name(const lfm3::Lfm3WeightOptions & options) {
    return lfm3::lfm3_reference_precision(options) ? "f32" : "native";
}

inline std::vector<int32_t> positions_from(int64_t start, int64_t steps) {
    std::vector<int32_t> out(static_cast<size_t>(steps));
    for (int64_t i = 0; i < steps; ++i) {
        out[static_cast<size_t>(i)] = static_cast<int32_t>(start + i);
    }
    return out;
}

struct RefLayer {
    blocks::LayerKind kind = blocks::LayerKind::Conv;
    Values operator_norm, ffn_norm;
    Values post_operator_norm, post_ffn_norm;  // empty without sandwich norms
    Values w1, w2, w3;
    Values in_proj, out_proj, conv;  // conv [D][K] (HF [D, 1, K])
    Values q, k, v, o, q_norm, k_norm;
    Values gate;  // [heads][D]; empty without the output gate
};

struct RefStack {
    std::vector<RefLayer> layers;
    Values final_norm;
};

inline Values to_values(const std::vector<float> & values) {
    return Values(values.begin(), values.end());
}

inline RefStack read_ref_stack(const engine::assets::TensorSource & source, const std::string & prefix, const blocks::StackConfig & config) {
    const bool lfm2 = config.names == blocks::Names::Lfm2;
    auto read = [&](const std::string & name) { return to_values(source.require_f32(name)); };

    RefStack out;
    for (size_t layer = 0; layer < config.layers.size(); ++layer) {
        const std::string p = prefix + "layers." + std::to_string(layer) + ".";
        RefLayer w;
        w.kind = config.layers[layer];
        w.operator_norm = read(p + "operator_norm.weight");
        w.ffn_norm = read(p + "ffn_norm.weight");
        if (config.sandwich_norm) {
            w.post_operator_norm = read(p + "post_operator_norm.weight");
            w.post_ffn_norm = read(p + "post_ffn_norm.weight");
        }
        w.w1 = read(p + "feed_forward.w1.weight");
        w.w2 = read(p + "feed_forward.w2.weight");
        w.w3 = read(p + "feed_forward.w3.weight");
        if (w.kind == blocks::LayerKind::Conv) {
            w.in_proj = read(p + "conv.in_proj.weight");
            w.out_proj = read(p + "conv.out_proj.weight");
            w.conv = read(p + "conv.conv.weight");
        } else {
            const std::string a = p + "self_attn.";
            w.q = read(a + "q_proj.weight");
            w.k = read(a + "k_proj.weight");
            w.v = read(a + "v_proj.weight");
            w.o = read(a + (lfm2 ? "out_proj.weight" : "o_proj.weight"));
            w.q_norm = read(a + (lfm2 ? "q_layernorm.weight" : "q_norm.weight"));
            w.k_norm = read(a + (lfm2 ? "k_layernorm.weight" : "k_norm.weight"));
            if (config.output_gate) {
                w.gate = read(a + "gate_proj.weight");
            }
        }
        out.layers.push_back(std::move(w));
    }
    out.final_norm = read(prefix + "embedding_norm.weight");
    return out;
}

inline Values rms_norm(const Values & x, const Values & w, int64_t dim, double eps) {
    Values out(x.size());
    for (size_t row = 0; row < x.size() / static_cast<size_t>(dim); ++row) {
        const double * in = x.data() + row * static_cast<size_t>(dim);
        double sum = 0.0;
        for (int64_t i = 0; i < dim; ++i) {
            sum += in[i] * in[i];
        }
        const double scale = 1.0 / std::sqrt(sum / static_cast<double>(dim) + eps);
        for (int64_t i = 0; i < dim; ++i) {
            out[row * static_cast<size_t>(dim) + static_cast<size_t>(i)] = in[i] * scale * w[static_cast<size_t>(i)];
        }
    }
    return out;
}

// x [rows][in] times w [out][in] transposed.
inline Values linear(const Values & x, const Values & w, int64_t in, int64_t out) {
    const size_t rows = x.size() / static_cast<size_t>(in);
    Values y(rows * static_cast<size_t>(out), 0.0);
    for (size_t r = 0; r < rows; ++r) {
        for (int64_t o = 0; o < out; ++o) {
            double sum = 0.0;
            for (int64_t i = 0; i < in; ++i) {
                sum += x[r * static_cast<size_t>(in) + static_cast<size_t>(i)] * w[static_cast<size_t>(o * in + i)];
            }
            y[r * static_cast<size_t>(out) + static_cast<size_t>(o)] = sum;
        }
    }
    return y;
}

inline Values linear_bias(const Values & x, const Values & w, const Values & b, int64_t in, int64_t out) {
    auto y = linear(x, w, in, out);
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] += b[i % static_cast<size_t>(out)];
    }
    return y;
}

// padded [steps + K - 1][D], kernel_t [K][D].
inline Values causal_conv(const Values & padded, const Values & kernel_t, int64_t d, int64_t k, int64_t steps) {
    Values y(static_cast<size_t>(steps * d), 0.0);
    for (int64_t t = 0; t < steps; ++t) {
        for (int64_t i = 0; i < k; ++i) {
            for (int64_t c = 0; c < d; ++c) {
                y[static_cast<size_t>(t * d + c)] += padded[static_cast<size_t>((t + i) * d + c)] * kernel_t[static_cast<size_t>(i * d + c)];
            }
        }
    }
    return y;
}

// The short conv over a whole sequence from zero history: B, C, x = in_proj(hn); out_proj(C * conv(B * x)).
inline Values short_conv(const blocks::StackConfig & config, const RefLayer & w, const Values & hn) {
    const int64_t d = config.hidden;
    const int64_t k = config.conv_kernel;
    const int64_t steps = static_cast<int64_t>(hn.size()) / d;
    const auto bcx = linear(hn, w.in_proj, d, 3 * d);
    Values padded(static_cast<size_t>((steps + k - 1) * d), 0.0);
    for (int64_t t = 0; t < steps; ++t) {
        for (int64_t c = 0; c < d; ++c) {
            padded[static_cast<size_t>((t + k - 1) * d + c)] =
                bcx[static_cast<size_t>(t * 3 * d + c)] * bcx[static_cast<size_t>(t * 3 * d + 2 * d + c)];
        }
    }
    Values kernel_t(static_cast<size_t>(k * d));
    for (int64_t c = 0; c < d; ++c) {
        for (int64_t i = 0; i < k; ++i) {
            kernel_t[static_cast<size_t>(i * d + c)] = w.conv[static_cast<size_t>(c * k + i)];
        }
    }
    auto y = causal_conv(padded, kernel_t, d, k, steps);
    for (int64_t t = 0; t < steps; ++t) {
        for (int64_t c = 0; c < d; ++c) {
            y[static_cast<size_t>(t * d + c)] *= bcx[static_cast<size_t>(t * 3 * d + d + c)];
        }
    }
    return linear(y, w.out_proj, d, d);
}

// NEOX (rotate_half) RoPE in place on x [steps][heads][hd]: dim i pairs with i + hd/2 at pos * theta^(-2i/hd).
// rotary_dims in (0, hd) zeroes the frequencies of i >= rotary_dims/2 (the LFM3 zero tail).
inline void rope(Values & x, int64_t heads, int64_t hd, const std::vector<int32_t> & positions, double theta, int64_t rotary_dims) {
    const int64_t half = hd / 2;
    for (size_t t = 0; t < positions.size(); ++t) {
        for (int64_t h = 0; h < heads; ++h) {
            double * v = x.data() + (t * static_cast<size_t>(heads) + static_cast<size_t>(h)) * static_cast<size_t>(hd);
            for (int64_t i = 0; i < half; ++i) {
                const bool zero = rotary_dims > 0 && rotary_dims < hd && i >= rotary_dims / 2;
                const double angle = zero ? 0.0 : positions[t] * std::pow(theta, -2.0 * static_cast<double>(i) / static_cast<double>(hd));
                const double a = v[i];
                const double b = v[i + half];
                v[i] = a * std::cos(angle) - b * std::sin(angle);
                v[i + half] = b * std::cos(angle) + a * std::sin(angle);
            }
        }
    }
}

inline bool visible(int64_t q, int64_t k, int64_t window) {
    return k <= q && (window == 0 || q - k < window);
}

inline Values attention(const blocks::StackConfig & config, blocks::LayerKind kind, const RefLayer & w, const Values & hn,
                        const std::vector<int32_t> & positions) {
    const int64_t d = config.hidden;
    const int64_t heads = config.heads;
    const int64_t kv_heads = config.kv_heads;
    const int64_t hd = config.head_dim;
    const int64_t steps = static_cast<int64_t>(positions.size());
    const bool full = kind == blocks::LayerKind::FullAttention;
    const int64_t window = full ? 0 : config.sliding_window;

    auto q = rms_norm(linear(hn, w.q, d, heads * hd), w.q_norm, hd, config.eps);
    auto k = rms_norm(linear(hn, w.k, d, kv_heads * hd), w.k_norm, hd, config.eps);
    const auto v = linear(hn, w.v, d, kv_heads * hd);
    const double theta = full ? config.full_theta : config.sliding_theta;
    rope(q, heads, hd, positions, theta, full ? config.full_rotary_dims : 0);
    rope(k, kv_heads, hd, positions, theta, full ? config.full_rotary_dims : 0);

    Values context(static_cast<size_t>(steps * heads * hd), 0.0);
    const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
    for (int64_t h = 0; h < heads; ++h) {
        const int64_t g = h / (heads / kv_heads);  // HF repeat_kv: query head h reads kv head h / repeats
        for (int64_t i = 0; i < steps; ++i) {
            std::vector<double> scores(static_cast<size_t>(steps), -INFINITY);
            double max_score = -INFINITY;
            for (int64_t j = 0; j < steps; ++j) {
                if (!config.causal || visible(positions[static_cast<size_t>(i)], positions[static_cast<size_t>(j)], window)) {
                    double s = 0.0;
                    for (int64_t c = 0; c < hd; ++c) {
                        s += q[static_cast<size_t>((i * heads + h) * hd + c)] * k[static_cast<size_t>((j * kv_heads + g) * hd + c)];
                    }
                    scores[static_cast<size_t>(j)] = s * scale;
                    max_score = std::max(max_score, s * scale);
                }
            }
            double sum = 0.0;
            for (auto & s : scores) {
                s = std::isinf(s) ? 0.0 : std::exp(s - max_score);
                sum += s;
            }
            for (int64_t j = 0; j < steps; ++j) {
                const double p = scores[static_cast<size_t>(j)] / sum;
                for (int64_t c = 0; c < hd; ++c) {
                    context[static_cast<size_t>((i * heads + h) * hd + c)] += p * v[static_cast<size_t>((j * kv_heads + g) * hd + c)];
                }
            }
        }
    }

    if (!w.gate.empty()) {
        const auto gate = linear(hn, w.gate, d, heads);
        for (int64_t i = 0; i < steps; ++i) {
            for (int64_t h = 0; h < heads; ++h) {
                const double s = 1.0 / (1.0 + std::exp(-gate[static_cast<size_t>(i * heads + h)]));
                for (int64_t c = 0; c < hd; ++c) {
                    context[static_cast<size_t>((i * heads + h) * hd + c)] *= s;
                }
            }
        }
    }
    return linear(context, w.o, heads * hd, d);
}

inline Values feed_forward(const blocks::StackConfig & config, const RefLayer & w, const Values & x) {
    auto a = linear(x, w.w1, config.hidden, config.ffn);
    const auto b = linear(x, w.w3, config.hidden, config.ffn);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = a[i] / (1.0 + std::exp(-a[i])) * b[i];
    }
    return linear(a, w.w2, config.ffn, config.hidden);
}

// x [steps][D], from empty state; returns the final-normed hidden.
inline Values stack(const blocks::StackConfig & config, const RefStack & weights, Values x, const std::vector<int32_t> & positions) {
    const int64_t d = config.hidden;
    for (const auto & w : weights.layers) {
        const auto hn = rms_norm(x, w.operator_norm, d, config.eps);
        auto y = w.kind == blocks::LayerKind::Conv ? short_conv(config, w, hn) : attention(config, w.kind, w, hn, positions);
        if (!w.post_operator_norm.empty()) {
            y = rms_norm(y, w.post_operator_norm, d, config.eps);
        }
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] += y[i];
        }
        auto f = feed_forward(config, w, rms_norm(x, w.ffn_norm, d, config.eps));
        if (!w.post_ffn_norm.empty()) {
            f = rms_norm(f, w.post_ffn_norm, d, config.eps);
        }
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] += f[i];
        }
    }
    return rms_norm(x, weights.final_norm, d, config.eps);
}

inline double rel_error(const std::vector<float> & actual, const Values & expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("rel_error size mismatch: " + std::to_string(actual.size()) + " vs " + std::to_string(expected.size()));
    }
    double diff = 0.0;
    double norm = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        diff += (actual[i] - expected[i]) * (actual[i] - expected[i]);
        norm += expected[i] * expected[i];
    }
    return std::sqrt(diff / norm);
}

inline double rel_error(const std::vector<float> & actual, const std::vector<float> & expected) {
    return rel_error(actual, to_values(expected));
}

inline void report(const std::string & label, double rel, double tolerance) {
    const bool pass = rel <= tolerance;
    std::printf("%s %s: rel %.3e (tol %.0e)\n", pass ? "PASS" : "FAIL", label.c_str(), rel, tolerance);
    std::fflush(stdout);
    engine::test::require(pass, label + " rel " + std::to_string(rel) + " exceeds " + std::to_string(tolerance));
}

// main's [--backend cpu|cuda|metal|best] [--threads N]. Null when the test must exit with `status`: 2 for a bad
// argument, 77 (skipped) for a requested backend that is not available, 1 when the CPU backend fails.
inline std::unique_ptr<engine::core::ExecutionContext> open_backend(int argc, char ** argv, int & status) {
    using engine::core::BackendType;
    engine::core::BackendConfig config{BackendType::Cpu, 0, 4};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const std::string value = i + 1 < argc ? argv[++i] : "";
        if (arg == "--backend" && (value == "cpu" || value == "cuda" || value == "metal" || value == "best")) {
            config.type = value == "cuda" ? BackendType::Cuda : value == "metal" ? BackendType::Metal
                        : value == "best" ? BackendType::BestAvailable : BackendType::Cpu;
        } else if (arg == "--threads" && !value.empty()) {
            config.threads = std::stoi(value);
        } else {
            std::cerr << "usage: " << argv[0] << " [--backend cpu|cuda|metal|best] [--threads N]\n";
            status = 2;
            return nullptr;
        }
    }

    try {
        auto execution = std::make_unique<engine::core::ExecutionContext>(config);
        std::printf("backend: %s\n", ggml_backend_name(execution->backend()));
        return execution;
    } catch (const std::exception & error) {
        if (config.type == BackendType::Cpu) {
            std::cerr << "FAIL: " << error.what() << "\n";
            status = 1;
        } else {
            std::cout << "SKIP: backend unavailable: " << error.what() << "\n";
            status = 77;
        }
        return nullptr;
    }
}

}  // namespace lfm3_audio_test
