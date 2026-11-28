#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/transformers/lfm_decoder_runtime.h"
#include "engine/framework/runtime/errors.h"
#include "test_assert.h"

#include <ggml-cpp.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace core = engine::core;
namespace modules = engine::modules;
using engine::test::require;
using engine::test::require_eq;
using Values = std::vector<double>;
using Kind = modules::LfmDecoderLayerKind;
using core::TensorShape;
using core::TensorValue;

// Selected once from the actual backend in main; never adjusted per fixture or
// after a failure. The oracle always remains independent FP64 arithmetic.
bool use_vector_agreement = false;
constexpr double kPointwiseTolerance = 3e-4;
constexpr double kMinimumCosine = .99999;
constexpr double kMaximumNormGainError = .002;

std::vector<float> pattern(size_t count, double phase, double scale = 0.6) {
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) {
        result[i] = static_cast<float>(scale * (std::sin(.37 * i + phase) + .3 * std::cos(.13 * i - phase)));
    }
    return result;
}

Values doubles(const std::vector<float> & values) { return {values.begin(), values.end()}; }

void close(const std::vector<float> & actual, const Values & expected, const std::string & label) {
    require_eq(actual.size(), expected.size(), label + " size");
    require(!actual.empty(), label + " empty comparison");
    double worst = 0, difference_squared = 0, actual_squared = 0, expected_squared = 0, dot = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]), label + " nonfinite at " + std::to_string(i));
        const double value = actual[i], error = value - expected[i];
        worst = std::max(worst, std::abs(error));
        difference_squared += error * error;
        actual_squared += value * value;
        expected_squared += expected[i] * expected[i];
        dot += value * expected[i];
    }
    require(std::isfinite(difference_squared) && std::isfinite(actual_squared) &&
            std::isfinite(expected_squared) && std::isfinite(dot), label + " nonfinite vector metric accumulation");
    // Define the zero-vector case explicitly, without an epsilon that could hide
    // a nonzero output when the oracle expects exactly zero.
    double relative_l2 = 0, cosine = 1, norm_gain_error = 0;
    if (expected_squared > 0) {
        relative_l2 = std::sqrt(difference_squared) / std::sqrt(expected_squared);
        norm_gain_error = std::abs(std::sqrt(actual_squared) / std::sqrt(expected_squared) - 1.0);
        cosine = actual_squared > 0 ? dot / (std::sqrt(actual_squared) * std::sqrt(expected_squared)) : 0;
    } else if (actual_squared > 0) {
        relative_l2 = std::numeric_limits<double>::infinity();
        norm_gain_error = std::numeric_limits<double>::infinity();
        cosine = 0;
    }
    // Log all metrics before asserting so failures retain direction and gain evidence.
    engine::debug::trace_log_scalar("lfm_decoder_test.comparison.label", std::string_view(label));
    engine::debug::trace_log_scalar("lfm_decoder_test.max_abs_error", worst);
    engine::debug::trace_log_scalar("lfm_decoder_test.comparison.elements", static_cast<uint64_t>(actual.size()));
    engine::debug::trace_log_scalar("lfm_decoder_test.expected_rms", std::sqrt(expected_squared / actual.size()));
    engine::debug::trace_log_scalar("lfm_decoder_test.actual_rms", std::sqrt(actual_squared / actual.size()));
    engine::debug::trace_log_scalar("lfm_decoder_test.error_rms", std::sqrt(difference_squared / actual.size()));
    engine::debug::trace_log_scalar("lfm_decoder_test.relative_l2", relative_l2);
    engine::debug::trace_log_scalar("lfm_decoder_test.cosine", cosine);
    engine::debug::trace_log_scalar("lfm_decoder_test.norm_gain_error", norm_gain_error);
    if (use_vector_agreement) {
        // rel_L2^2 = (norm_ratio - 1)^2 + 2*norm_ratio*(1 - cosine).
        // Gate direction and gain separately; relative L2 remains diagnostic.
        require(cosine >= kMinimumCosine && norm_gain_error <= kMaximumNormGainError,
                label + " vector mismatch: cosine=" + std::to_string(cosine) +
                " norm_gain_error=" + std::to_string(norm_gain_error) + " relative_l2=" + std::to_string(relative_l2));
    } else {
        for (size_t i = 0; i < actual.size(); ++i) {
            const double error = std::abs(actual[i] - expected[i]);
            const double allowed = kPointwiseTolerance + kPointwiseTolerance * std::abs(expected[i]);
            require(error <= allowed, label + " mismatch at " + std::to_string(i) +
                ": actual=" + std::to_string(actual[i]) + " expected=" + std::to_string(expected[i]) +
                " abs_error=" + std::to_string(error) + " allowed=" + std::to_string(allowed));
        }
    }
}

struct Parameter {
    TensorValue value;
    std::vector<float> host;
    void * data = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    void * extra = nullptr;
    ggml_tensor * view_src = nullptr;
    size_t view_offs = 0;
};

// Small deterministic parameters, kept independently on the host for the FP64
// oracle and for detecting mutation of caller-owned tensor bindings/contents.
struct Fixture {
    core::BackendWeightStore store;
    modules::LfmDecoderConfig config;
    modules::LfmDecoderRuntimeWeights weights;
    std::vector<Parameter> parameters;
    bool correlated_cpu_stress = false;

    Fixture(core::ExecutionContext & execution, modules::LfmDecoderConfig c, bool correlated = false)
        : store(execution.backend(), execution.backend_type(), "lfm_decoder_test.weights", 2 * 1024 * 1024),
          config(std::move(c)), correlated_cpu_stress(correlated) {
        const int64_t d = config.hidden_size, ff = config.intermediate_size;
        for (const auto kind : config.layers) {
            modules::LfmDecoderLayerWeights w;
            w.operator_norm = norm(d);
            w.ffn_norm = norm(d);
            if (config.sandwich_norm) {
                w.post_operator_norm = norm(d);
                w.post_ffn_norm = norm(d);
            }
            w.feed_forward = {linear(ff, d), linear(ff, d), linear(d, ff)};
            if (kind == Kind::ShortConv) {
                w.conv.in_proj = linear(3 * d, d, config.conv_bias);
                w.conv.out_proj = linear(d, d, config.conv_bias);
                w.conv.kernel = tensor({d, 1, config.conv_kernel_size});
                if (config.conv_bias) { w.conv.bias = tensor({d}); }
            } else {
                const int64_t q = config.num_attention_heads * config.head_dim;
                const int64_t kv = config.num_key_value_heads * config.head_dim;
                w.attention.q_proj = linear(q, d);
                w.attention.k_proj = linear(kv, d);
                w.attention.v_proj = linear(kv, d);
                w.attention.out_proj = linear(d, q);
                w.attention.q_norm = norm(config.head_dim);
                w.attention.k_norm = norm(config.head_dim);
                if (config.attention_gate) { w.attention.gate_proj = linear(config.num_attention_heads, d); }
            }
            weights.decoder.layers.push_back(w);
        }
        weights.decoder.final_norm = norm(d);
        weights.token_embedding = linear(11, d).weight;
        // Tied embedding/readout storage plus bias exercises shared borrowed leaves.
        weights.output_projection = modules::LinearWeights{*weights.token_embedding, tensor({11})};
    }

    TensorValue tensor(std::initializer_list<int64_t> dims, bool unit = false) {
        const auto shape = TensorShape::from_dims(dims);
        auto values = pattern(static_cast<size_t>(shape.num_elements()), .71 * (parameters.size() + 1), unit ? .08 : .16);
        if (unit) { for (auto & value : values) { value += 1.f; } }
        auto value = store.make_f32(shape, values);
        parameters.push_back({value, std::move(values)});
        return value;
    }
    modules::NormWeights norm(int64_t d) { return {tensor({d}, true), std::nullopt}; }
    modules::LinearWeights linear(int64_t out, int64_t in, bool bias = false) {
        TensorValue weight;
        if (correlated_cpu_stress) {
            weight = tensor({out, in});
        } else {
            // Independent coefficients with variance 1/fan_in, identical on all
            // backends. The sine-pattern matrices are low rank and their dense
            // sums grow with width; retain those only in the CPU stress case.
            std::mt19937 random(static_cast<uint32_t>(0x4c464d00u + parameters.size()));
            const double bound = std::sqrt(3.0 / static_cast<double>(in));
            std::vector<float> values(static_cast<size_t>(out * in));
            for (auto & value : values) {
                value = static_cast<float>((2.0 * (static_cast<double>(random()) / 4294967296.0) - 1.0) * bound);
            }
            weight = store.make_f32(TensorShape::from_dims({out, in}), values);
            parameters.push_back({weight, std::move(values)});
        }
        modules::LinearWeights result{weight, std::nullopt};
        if (bias) { result.bias = tensor({out}); }
        return result;
    }
    const std::vector<float> & host(const TensorValue & value) const {
        for (const auto & p : parameters) { if (p.value.tensor == value.tensor) { return p.host; } }
        throw std::runtime_error("oracle parameter not found");
    }
    void upload() {
        store.upload();
        for (auto & p : parameters) {
            p.data = p.value.tensor->data;
            p.buffer = p.value.tensor->buffer;
            p.extra = p.value.tensor->extra;
            p.view_src = p.value.tensor->view_src;
            p.view_offs = p.value.tensor->view_offs;
        }
    }
    void intact() const {
        for (const auto & p : parameters) {
            auto * t = p.value.tensor;
            require(t->data == p.data && t->buffer == p.buffer && t->extra == p.extra &&
                    t->view_src == p.view_src && t->view_offs == p.view_offs, "borrowed parameter binding changed");
            require(core::read_tensor_f32(t) == p.host, "borrowed parameter contents changed");
        }
    }
};

// Independent row-major FP64 math: no GGML/module operations in the oracle.
Values linear(const Fixture & f, const Values & x, const modules::LinearWeights & w) {
    const int64_t out = w.weight.shape.dims[0], in = w.weight.shape.dims[1];
    const auto & weight = f.host(w.weight);
    Values result(x.size() / in * out);
    for (size_t row = 0; row < x.size() / in; ++row) {
        for (int64_t o = 0; o < out; ++o) {
            double sum = w.bias ? f.host(*w.bias)[o] : 0;
            for (int64_t i = 0; i < in; ++i) { sum += x[row * in + i] * weight[o * in + i]; }
            result[row * out + o] = sum;
        }
    }
    return result;
}

Values norm(const Fixture & f, Values x, const modules::NormWeights & w, int64_t dim) {
    const auto & scale = f.host(*w.weight);
    for (size_t row = 0; row < x.size() / dim; ++row) {
        double squared = 0;
        for (int64_t i = 0; i < dim; ++i) { squared += x[row * dim + i] * x[row * dim + i]; }
        const double divisor = std::sqrt(squared / dim + f.config.norm_eps);
        for (int64_t i = 0; i < dim; ++i) { x[row * dim + i] = x[row * dim + i] / divisor * scale[i]; }
    }
    return x;
}

struct Input {
    int64_t batch = 2, steps = 5;
    std::vector<float> hidden, full, sliding, keep;
    std::vector<int32_t> positions;
};

Input input(int64_t d, int64_t batch = 2, int64_t steps = 5) {
    Input x;
    x.batch = batch;
    x.steps = steps;
    x.hidden = pattern(batch * steps * d, .41);
    for (int64_t t = 0; t < steps; ++t) { x.positions.push_back(static_cast<int32_t>(2 + 3 * t)); }
    for (int64_t q = 0; q < steps; ++q) {
        for (int64_t k = 0; k < steps; ++k) {
            x.full.push_back((q + k) % 3 == 0 ? .125f : 0.f); // bidirectional, with an additive bias
            x.sliding.push_back(k <= q && q - k < 2 ? 0.f : -std::numeric_limits<float>::infinity());
        }
    }
    return x;
}

void rope(Values & x, int64_t batch, int64_t steps, int64_t heads, int64_t dim,
          const std::vector<int32_t> & positions, const modules::LfmDecoderRopeConfig & config,
          const std::vector<float> * factors) {
    const int64_t pairs = factors ? dim / 2 : (config.rotary_dims < 0 ? dim : config.rotary_dims) / 2;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t t = 0; t < steps; ++t) {
            for (int64_t h = 0; h < heads; ++h) {
                const size_t start = ((b * steps + t) * heads + h) * dim;
                for (int64_t i = 0; i < pairs; ++i) {
                    const double divisor = factors ? (*factors)[i] : 1.0;
                    const double angle = positions[t] / (std::pow(static_cast<double>(config.theta), 2.0 * i / dim) * divisor);
                    const double a = x[start + i], z = x[start + dim / 2 + i];
                    x[start + i] = a * std::cos(angle) - z * std::sin(angle);
                    x[start + dim / 2 + i] = z * std::cos(angle) + a * std::sin(angle);
                }
            }
        }
    }
}

Values attention(const Fixture & f, const Input & in, const Values & x,
                 const modules::LfmDecoderAttentionWeights & w, Kind kind) {
    const auto & c = f.config;
    const int64_t heads = c.num_attention_heads, kv = c.num_key_value_heads, dim = c.head_dim, steps = in.steps;
    auto q = norm(f, linear(f, x, w.q_proj), w.q_norm, dim);
    auto k = norm(f, linear(f, x, w.k_proj), w.k_norm, dim);
    auto v = linear(f, x, w.v_proj);
    const auto & mask = kind == Kind::FullAttention ? in.full : in.sliding;
    const auto & rotary = kind == Kind::FullAttention ? c.full_rope : c.sliding_rope;
    const auto & factors = kind == Kind::FullAttention ? f.weights.decoder.full_freq_factors : f.weights.decoder.sliding_freq_factors;
    const auto * divisors = factors ? &f.host(*factors) : nullptr;
    rope(q, in.batch, steps, heads, dim, in.positions, rotary, divisors);
    rope(k, in.batch, steps, kv, dim, in.positions, rotary, divisors);
    const auto gates = w.gate_proj ? linear(f, x, *w.gate_proj) : Values{};
    Values context(in.batch * steps * heads * dim, 0);
    for (int64_t b = 0; b < in.batch; ++b) {
        for (int64_t t = 0; t < steps; ++t) {
            for (int64_t h = 0; h < heads; ++h) {
                const int64_t kh = h / (heads / kv);
                Values scores(steps);
                for (int64_t s = 0; s < steps; ++s) {
                    double dot = 0;
                    for (int64_t i = 0; i < dim; ++i) {
                        dot += q[((b * steps + t) * heads + h) * dim + i] * k[((b * steps + s) * kv + kh) * dim + i];
                    }
                    scores[s] = dot / std::sqrt(static_cast<double>(dim)) + (mask.empty() ? 0 : mask[t * steps + s]);
                }
                const double maximum = *std::max_element(scores.begin(), scores.end());
                double sum = 0;
                for (auto & value : scores) { value = std::exp(value - maximum); sum += value; }
                const double gate = gates.empty() ? 1 : 1 / (1 + std::exp(-gates[(b * steps + t) * heads + h]));
                for (int64_t i = 0; i < dim; ++i) {
                    double value = 0;
                    for (int64_t s = 0; s < steps; ++s) { value += scores[s] / sum * v[((b * steps + s) * kv + kh) * dim + i]; }
                    context[((b * steps + t) * heads + h) * dim + i] = value * gate;
                }
            }
        }
    }
    return linear(f, context, w.out_proj);
}

Values conv(const Fixture & f, const Input & in, Values x, const modules::LfmDecoderConvWeights & w) {
    const int64_t d = f.config.hidden_size, kernel = f.config.conv_kernel_size, rows = in.batch * in.steps;
    if (!in.keep.empty()) {
        for (int64_t r = 0; r < rows; ++r) { for (int64_t i = 0; i < d; ++i) { x[r * d + i] *= in.keep[r]; } }
    }
    const auto packed = linear(f, x, w.in_proj);
    Values bx(rows * d), result(rows * d);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t i = 0; i < d; ++i) {
            bx[r * d + i] = packed[r * 3 * d + i] * packed[r * 3 * d + 2 * d + i] * (in.keep.empty() ? 1 : in.keep[r]);
        }
    }
    const auto & weight = f.host(w.kernel);
    for (int64_t b = 0; b < in.batch; ++b) {
        for (int64_t t = 0; t < in.steps; ++t) {
            for (int64_t i = 0; i < d; ++i) {
                double value = w.bias ? f.host(*w.bias)[i] : 0;
                for (int64_t tap = 0; tap < kernel; ++tap) {
                    const int64_t previous = t - kernel + 1 + tap;
                    if (previous >= 0) { value += weight[i * kernel + tap] * bx[(b * in.steps + previous) * d + i]; }
                }
                result[(b * in.steps + t) * d + i] = packed[(b * in.steps + t) * 3 * d + d + i] * value;
            }
        }
    }
    return linear(f, result, w.out_proj);
}

struct Reference { Values hidden; std::vector<Values> layers; };
Reference reference(const Fixture & f, const Input & in) {
    auto x = doubles(in.hidden);
    Reference result;
    for (size_t layer = 0; layer < f.config.layers.size(); ++layer) {
        const auto & w = f.weights.decoder.layers[layer];
        const auto normalized = norm(f, x, w.operator_norm, f.config.hidden_size);
        auto y = f.config.layers[layer] == Kind::ShortConv ? conv(f, in, normalized, w.conv) :
            attention(f, in, normalized, w.attention, f.config.layers[layer]);
        if (w.post_operator_norm) { y = norm(f, std::move(y), *w.post_operator_norm, f.config.hidden_size); }
        for (size_t i = 0; i < x.size(); ++i) { x[i] += y[i]; }
        auto n = norm(f, x, w.ffn_norm, f.config.hidden_size);
        auto gate = linear(f, n, w.feed_forward.gate_proj);
        const auto up = linear(f, n, w.feed_forward.up_proj);
        for (size_t i = 0; i < gate.size(); ++i) { gate[i] = gate[i] / (1 + std::exp(-gate[i])) * up[i]; }
        auto ff = linear(f, gate, w.feed_forward.down_proj);
        if (w.post_ffn_norm) { ff = norm(f, std::move(ff), *w.post_ffn_norm, f.config.hidden_size); }
        for (size_t i = 0; i < x.size(); ++i) { x[i] += ff[i]; }
        result.layers.push_back(x);
    }
    result.hidden = f.config.final_norm ? norm(f, std::move(x), f.weights.decoder.final_norm, f.config.hidden_size) : std::move(x);
    return result;
}

struct Graph {
    core::ExecutionContext & execution;
    ggml_context_ptr context;
    ggml_gallocr_ptr allocator;
    ggml_cgraph * graph = nullptr;
    explicit Graph(core::ExecutionContext & e) : execution(e), context(ggml_init({16 * 1024 * 1024, nullptr, true})) {
        require(context != nullptr, "graph context allocation");
        graph = ggml_new_graph_custom(context.get(), 8192, false);
    }
    ~Graph() { core::release_backend_graph_resources(execution.backend(), graph, true); }
    TensorValue input(core::ModuleBuildContext & ctx, ggml_type type, const TensorShape & shape) {
        auto result = core::make_tensor(ctx, type, shape);
        ggml_set_input(result.tensor);
        // Every declared input is uploaded, even when this architecture does not
        // consume it (zero RoPE, convolution-only layers, or an unused mask kind).
        ggml_build_forward_expand(graph, result.tensor);
        return result;
    }
};

std::vector<float> graph_result(core::ExecutionContext & execution, const Fixture & f, const Input & in, bool half_mask = false) {
    Graph g(execution);
    core::ModuleBuildContext ctx{g.context.get(), "lfm_decoder_test.graph", execution.backend_type()};
    modules::LfmDecoderInputs x;
    x.hidden = g.input(ctx, GGML_TYPE_F32, TensorShape::from_dims({in.batch, in.steps, f.config.hidden_size}));
    x.positions = g.input(ctx, GGML_TYPE_I32, TensorShape::from_dims({in.steps}));
    const auto mask_type = half_mask ? GGML_TYPE_F16 : GGML_TYPE_F32;
    if (!in.full.empty()) { x.full_mask = g.input(ctx, mask_type, TensorShape::from_dims({in.steps, in.steps})); }
    if (!in.sliding.empty()) { x.sliding_mask = g.input(ctx, mask_type, TensorShape::from_dims({in.steps, in.steps})); }
    if (!in.keep.empty()) { x.conv_padding_mask = g.input(ctx, GGML_TYPE_F32, TensorShape::from_dims({in.batch, in.steps})); }
    x.capture_layers = {static_cast<int64_t>(f.config.layers.size() - 1), 0, static_cast<int64_t>(f.config.layers.size() - 1)};
    auto result = modules::build_lfm_decoder(ctx, x, f.weights.decoder, f.config);
    for (const auto & value : result.captured_hidden) {
        ggml_set_output(value.tensor);
        ggml_build_forward_expand(g.graph, value.tensor);
    }
    ggml_set_output(result.hidden.tensor);
    ggml_build_forward_expand(g.graph, result.hidden.tensor);
    core::validate_backend_graph_with_cpu_extra_buffers(execution.backend(), g.graph, "lfm_decoder_test");
    g.allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
    require(g.allocator && ggml_gallocr_reserve(g.allocator.get(), g.graph) && ggml_gallocr_alloc_graph(g.allocator.get(), g.graph),
            "graph allocation");
    core::write_tensor_f32(x.hidden, in.hidden);
    core::write_tensor_i32(x.positions, in.positions);
    const auto upload_mask = [half_mask](const TensorValue & value, const std::vector<float> & host) {
        if (!half_mask) { core::write_tensor_f32(value, host); return; }
        std::vector<ggml_fp16_t> bits;
        for (const auto item : host) { bits.push_back(ggml_fp32_to_fp16(item)); }
        ggml_backend_tensor_set(value.tensor, bits.data(), 0, bits.size() * sizeof(ggml_fp16_t));
    };
    if (x.full_mask) { upload_mask(*x.full_mask, in.full); }
    if (x.sliding_mask) { upload_mask(*x.sliding_mask, in.sliding); }
    if (x.conv_padding_mask) { core::write_tensor_f32(*x.conv_padding_mask, in.keep); }
    require(core::compute_backend_graph(execution.backend(), g.graph) == GGML_STATUS_SUCCESS, "graph compute");
    ggml_backend_synchronize(execution.backend());
    const auto expected = reference(f, in);
    engine::debug::trace_log_scalar("lfm_decoder_test.fixture.batch", in.batch);
    engine::debug::trace_log_scalar("lfm_decoder_test.fixture.steps", in.steps);
    engine::debug::trace_log_scalar("lfm_decoder_test.fixture.hidden_size", f.config.hidden_size);
    engine::debug::trace_log_scalar("lfm_decoder_test.fixture.head_dim", f.config.head_dim);
    engine::debug::trace_log_scalar("lfm_decoder_test.fixture.full_mask", !in.full.empty());
    engine::debug::trace_log_scalar("lfm_decoder_test.fixture.sliding_mask", !in.sliding.empty());
    engine::debug::trace_log_scalar("lfm_decoder_test.fixture.full_freq_factors", f.weights.decoder.full_freq_factors.has_value());
    engine::debug::trace_log_scalar("lfm_decoder_test.fixture.sliding_freq_factors", f.weights.decoder.sliding_freq_factors.has_value());
    engine::debug::trace_log_scalar("lfm_decoder_test.fixture.attention_gate", f.config.attention_gate);
    engine::debug::trace_log_scalar("lfm_decoder_test.fixture.sandwich_norm", f.config.sandwich_norm);
    engine::debug::trace_log_scalar("lfm_decoder_test.fixture.full_rotary_dims", f.config.full_rope.rotary_dims);
    engine::debug::trace_log_scalar("lfm_decoder_test.fixture.sliding_rotary_dims", f.config.sliding_rope.rotary_dims);
    engine::debug::trace_log_scalar("lfm_decoder_test.fixture.half_mask", half_mask);
    // Report every captured stage before final-hidden validation can throw.
    // This distinguishes errors already present in layer 0 from later amplification.
    for (size_t capture = 0; capture < result.captured_hidden.size(); ++capture) {
        const auto values = core::read_tensor_f32(result.captured_hidden[capture].tensor);
        const auto & oracle = expected.layers[x.capture_layers[capture]];
        require_eq(values.size(), oracle.size(), "capture size");
        double error = 0;
        for (size_t i = 0; i < values.size(); ++i) { error = std::max(error, std::abs(values[i] - oracle[i])); }
        engine::debug::trace_log_scalar("lfm_decoder_test.layer_" + std::to_string(x.capture_layers[capture]) + ".max_abs_error", error);
    }
    close(core::read_tensor_f32(result.hidden.tensor), expected.hidden, "FP64 final hidden");
    require_eq(result.captured_hidden.size(), x.capture_layers.size(), "capture count");
    for (size_t i = 0; i < x.capture_layers.size(); ++i) {
        close(core::read_tensor_f32(result.captured_hidden[i].tensor), expected.layers[x.capture_layers[i]], "FP64 capture");
    }
    return core::read_tensor_f32(result.hidden.tensor);
}

modules::LfmDecoderConfig configuration(std::vector<Kind> layers) {
    modules::LfmDecoderConfig c;
    c.hidden_size = 16;
    c.intermediate_size = 20;
    c.num_attention_heads = 2;
    c.num_key_value_heads = 1;
    c.head_dim = 8;
    c.conv_kernel_size = 7;
    c.layers = std::move(layers);
    c.projection_precision = GGML_PREC_F32;
    c.attention_precision = GGML_PREC_F32;
    c.full_rope.theta = 37.f;
    c.sliding_rope.theta = 113.f;
    return c;
}

void test_math(core::ExecutionContext & execution) {
    for (const bool gate : {false, true}) {
        for (const bool sandwich : {false, true}) {
            for (const int64_t rotated : {int64_t{0}, int64_t{4}, int64_t{-1}}) {
                auto c = configuration({Kind::FullAttention, Kind::SlidingAttention});
                c.attention_gate = gate;
                c.sandwich_norm = sandwich;
                c.full_rope.rotary_dims = rotated;
                c.sliding_rope.rotary_dims = rotated == 4 ? 2 : rotated;
                Fixture f(execution, c);
                f.upload();
                auto x = input(c.hidden_size);
                // F32 first, F16, then F32 again: identical weights and masks,
                // independently allocated graphs, with no tolerance change.
                const auto first_f32 = graph_result(execution, f, x);
                close(graph_result(execution, f, x, true), doubles(first_f32), "F32-F16 mask equivalence");
                close(graph_result(execution, f, x), doubles(first_f32), "F32-F16-F32 graph lifetime");
                // Absent full/sliding masks independently, then both absent.
                x.full.clear();
                graph_result(execution, f, x);
                x.full = input(c.hidden_size).full;
                x.sliding.clear();
                graph_result(execution, f, x);
                x.full.clear();
                graph_result(execution, f, x);
                f.intact();
            }
        }
    }
    for (const int64_t kernel : {int64_t{1}, int64_t{3}, int64_t{7}, int64_t{9}}) {
        for (const bool bias : {false, true}) {
            auto c = configuration({Kind::ShortConv});
            c.conv_kernel_size = kernel == 9 ? 3 : kernel;
            if (kernel == 9) {
                c.head_dim = 64;
                c.attention_lowering = modules::GroupedQueryAttentionLowering::FlashGrouped;
            }
            c.conv_bias = bias;
            c.sandwich_norm = bias;
            c.final_norm = bias;
            Fixture f(execution, c);
            f.upload();
            auto x = input(c.hidden_size);
            x.keep = {1, 1, 0, 1, 1, 0, 1, 1, 0, 1};
            const auto before = graph_result(execution, f, x);
            // Future changes cannot affect earlier conv rows; batch 1 cannot affect batch 0.
            for (int64_t b = 0; b < x.batch; ++b) {
                for (int64_t i = 0; i < c.hidden_size; ++i) { x.hidden[(b * x.steps + x.steps - 1) * c.hidden_size + i] += .8f; }
            }
            const auto after = graph_result(execution, f, x);
            for (int64_t b = 0; b < x.batch; ++b) {
                for (int64_t t = 0; t < x.steps - 1; ++t) {
                    for (int64_t i = 0; i < c.hidden_size; ++i) {
                        const size_t at = (b * x.steps + t) * c.hidden_size + i;
                        require(std::abs(before[at] - after[at]) < 1e-6, "convolution read future input");
                    }
                }
            }
            for (size_t i = x.steps * c.hidden_size; i < x.hidden.size(); ++i) { x.hidden[i] -= 1.3f; }
            const auto other_batch = graph_result(execution, f, x);
            for (int64_t i = 0; i < x.steps * c.hidden_size; ++i) {
                require(std::abs(after[i] - other_batch[i]) < 1e-6, "convolution leaked across batches");
            }
            x.keep.clear();
            graph_result(execution, f, x);
            f.intact();
        }
    }
}

void test_correlated_cpu_stress(core::ExecutionContext & execution) {
    require(execution.backend_type() == core::BackendType::Cpu, "correlated stress fixture is CPU-only");
    auto config = configuration({Kind::FullAttention, Kind::SlidingAttention});
    config.head_dim = 128;
    config.hidden_size = config.head_dim * config.num_attention_heads;
    config.full_rope.rotary_dims = 64;
    config.sliding_rope.rotary_dims = 32;
    Fixture fixture(execution, config, true);
    fixture.upload();
    graph_result(execution, fixture, input(config.hidden_size));
    fixture.intact();
}

void test_frequency_factors(core::ExecutionContext & execution) {
    auto c = configuration({Kind::FullAttention, Kind::SlidingAttention});
    c.head_dim = 128;
    c.hidden_size = c.head_dim * c.num_attention_heads;
    c.full_rope.rotary_dims = 64;
    c.sliding_rope.rotary_dims = 32;
    Fixture f(execution, c);
    // Published full-attention layout: 64 divisors, 32 ones then 32 values of 1e30.
    std::vector<float> full(64, 1e30f), sliding(64, 1e30f);
    std::fill_n(full.begin(), 32, 1.f);
    std::fill_n(sliding.begin(), 16, 1.f);
    const auto full_tensor = f.store.make_f32(TensorShape::from_dims({64}), full);
    f.parameters.push_back({full_tensor, full});
    const auto sliding_tensor = f.store.make_f32(TensorShape::from_dims({64}), sliding);
    f.parameters.push_back({sliding_tensor, sliding});
    f.upload();
    // Batch 1 runs first, matching production shape; batch 2 then exercises
    // attention's head/batch strides with exactly the same parameter storage.
    for (const int64_t batch : {int64_t{1}, int64_t{2}}) {
        f.config = c;
        f.weights.decoder.full_freq_factors.reset();
        f.weights.decoder.sliding_freq_factors.reset();
        const auto in = input(c.hidden_size, batch);
        const auto exact = graph_result(execution, f, in);
        close(graph_result(execution, f, in, true), doubles(exact), "large fixture F32-F16 mask equivalence");
        close(graph_result(execution, f, in), doubles(exact), "large fixture F32-F16-F32 graph lifetime");
        f.weights.decoder.full_freq_factors = full_tensor;
        close(graph_result(execution, f, in), doubles(exact), "full frequency factors vs exact partial rotation");
        f.weights.decoder.full_freq_factors.reset();
        f.weights.decoder.sliding_freq_factors = sliding_tensor;
        close(graph_result(execution, f, in), doubles(exact), "sliding frequency factors independently selected");
        f.weights.decoder.full_freq_factors = full_tensor;
        close(graph_result(execution, f, in), doubles(exact), "both frequency-factor tables");
        // Supplied divisors are authoritative even if metadata says no partial RoPE.
        f.config.full_rope.rotary_dims = -1;
        f.config.sliding_rope.rotary_dims = 0;
        close(graph_result(execution, f, in), doubles(exact), "frequency factors override rotary_dims");
    }
    f.intact();
}

Values runtime_reference(const Fixture & f, const modules::LfmDecoderForward & request, const std::vector<float> & hidden) {
    auto in = input(f.config.hidden_size, 1, request.positions.size());
    in.hidden = hidden;
    in.positions = request.positions;
    in.full.clear();
    in.sliding.clear();
    for (const auto value : request.full_mask) { in.full.push_back(ggml_fp16_to_fp32(value)); }
    for (const auto value : request.sliding_mask) { in.sliding.push_back(ggml_fp16_to_fp32(value)); }
    auto x = reference(f, in).hidden;
    const int64_t d = f.config.hidden_size;
    if (!request.readout_positions.empty()) {
        Values selected;
        for (const auto row : request.readout_positions) { selected.insert(selected.end(), x.begin() + row * d, x.begin() + (row + 1) * d); }
        x = std::move(selected);
    }
    if (request.readout == modules::LfmDecoderReadout::Hidden) { return x; }
    x = linear(f, x, *f.weights.output_projection);
    if (request.output_tokens.empty()) { return x; }
    const int64_t vocab = f.weights.output_projection->weight.shape.dims[0];
    Values selected;
    for (size_t row = 0; row < x.size() / vocab; ++row) {
        for (const auto token : request.output_tokens) { selected.push_back(x[row * vocab + token]); }
    }
    return selected;
}

void test_runtime(core::ExecutionContext & execution) {
    auto c = configuration({Kind::ShortConv, Kind::FullAttention, Kind::SlidingAttention});
    c.full_rope.rotary_dims = 4;
    c.conv_bias = true;
    Fixture f(execution, c);
    const auto device_hidden = f.tensor({1, 5, c.hidden_size});
    const auto device_patches = f.tensor({2, c.hidden_size});
    f.upload();
    modules::LfmDecoderForward a;
    a.token_ids = {1, 7, 2, 9, 3};
    a.positions = {2, 5, 8, 11, 14};
    a.replacement_positions = {3, 1}; // deliberately not sorted
    a.replacement_values = f.host(device_patches);
    a.readout_positions = {4, 0, 4}; // preserve order and duplicates
    a.output_tokens = {9, 2, 9, 0};
    a.readout = modules::LfmDecoderReadout::Logits;
    const auto masks = input(c.hidden_size, 1, 5);
    for (const auto value : masks.full) { a.full_mask.push_back(ggml_fp32_to_fp16(value)); }
    for (const auto value : masks.sliding) { a.sliding_mask.push_back(ggml_fp32_to_fp16(value)); }
    std::vector<float> embedded;
    const auto & table = f.host(*f.weights.token_embedding);
    for (const auto token : a.token_ids) {
        embedded.insert(embedded.end(), table.begin() + token * c.hidden_size, table.begin() + (token + 1) * c.hidden_size);
    }
    for (size_t row = 0; row < a.replacement_positions.size(); ++row) {
        std::copy_n(a.replacement_values.begin() + row * c.hidden_size, c.hidden_size,
                    embedded.begin() + a.replacement_positions[row] * c.hidden_size);
    }
    const auto expected_a = runtime_reference(f, a, embedded);
    for (const size_t slots : {size_t{2}, size_t{1}}) {
        {
            modules::LfmDecoderRuntime runtime(c, f.weights, execution, slots);
            auto got = runtime.forward(a);
            require_eq(got.rows, int64_t{3}, "sparse readout rows");
            require_eq(got.columns, int64_t{4}, "sparse output columns");
            close(got.values, expected_a, "runtime A sparse logits");
            require_eq(runtime.graphs_built(), size_t{1}, "initial topology");
            f.intact();
            auto device_a = a;
            device_a.replacement_values.clear();
            device_a.replacement_tensor = &device_patches;
            close(runtime.forward(device_a).values, expected_a, "device replacement rows");
            require_eq(runtime.graphs_built(), size_t{1}, "host/device replacement shares topology");

            modules::LfmDecoderForward b;
            b.hidden_tensor = &device_hidden;
            b.positions = a.positions;
            b.full_mask = a.full_mask;
            b.sliding_mask = a.sliding_mask;
            const auto expected_b = runtime_reference(f, b, f.host(device_hidden));
            const auto result_b = runtime.forward(b);
            require_eq(result_b.rows, int64_t{5}, "hidden rows");
            require_eq(result_b.columns, c.hidden_size, "hidden columns");
            close(result_b.values, expected_b, "borrowed device hidden");
            require_eq(runtime.graphs_built(), size_t{2}, "second topology");
            f.intact();
            b.hidden_tensor = nullptr;
            b.hidden = f.host(device_hidden);
            close(runtime.forward(b).values, expected_b, "host/device hidden equivalence");
            require_eq(runtime.graphs_built(), size_t{2}, "host/device hidden shares topology");
            close(runtime.forward(a).values, expected_a, "A-B-A");
            const size_t built = slots == 2 ? 2 : 3;
            require_eq(runtime.graphs_built(), built, "A topology reuse or eviction");
            f.intact();
            runtime.release_workspace();
            runtime.release_workspace();
            require_eq(runtime.graphs_built(), built, "release retains topology");
            f.intact();
            close(runtime.forward(a).values, expected_a, "rebind after workspace release");
            require_eq(runtime.graphs_built(), built, "rebind uses cached topology");
            // Same shapes, different sparse indices and values must upload fresh data.
            auto changed = a;
            changed.output_tokens = {1, 8, 0, 5};
            changed.readout_positions = {2, 3, 1};
            close(runtime.forward(changed).values, runtime_reference(f, changed, embedded), "sparse indices refreshed");
            changed.replacement_values = pattern(changed.replacement_values.size(), 1.9);
            auto changed_hidden = embedded;
            for (size_t row = 0; row < changed.replacement_positions.size(); ++row) {
                std::copy_n(changed.replacement_values.begin() + row * c.hidden_size, c.hidden_size,
                            changed_hidden.begin() + changed.replacement_positions[row] * c.hidden_size);
            }
            for (auto & position : changed.positions) { position += 7; }
            changed.full_mask = changed.sliding_mask;
            close(runtime.forward(changed).values, runtime_reference(f, changed, changed_hidden), "replacement/mask/position contents refreshed");
            close(runtime.forward(a).values, expected_a, "restore A after input-content changes");
            require_eq(runtime.graphs_built(), built, "sparse contents do not rebuild topology");
            runtime.release_workspace();
            f.intact();
        }
        f.intact(); // runtime destruction must not unbind or release borrowed parameters
    }
    // A vocabulary-wide readout and a no-replacement token request are separate topologies.
    modules::LfmDecoderRuntime runtime(c, f.weights, execution);
    auto full = a;
    full.output_tokens.clear();
    close(runtime.forward(full).values, runtime_reference(f, full, embedded), "full readout");
    auto tokens = a;
    tokens.replacement_positions.clear();
    tokens.replacement_values.clear();
    tokens.readout = modules::LfmDecoderReadout::Hidden;
    tokens.output_tokens.clear();
    tokens.readout_positions.clear();
    std::vector<float> plain;
    for (const auto token : tokens.token_ids) {
        plain.insert(plain.end(), table.begin() + token * c.hidden_size, table.begin() + (token + 1) * c.hidden_size);
    }
    close(runtime.forward(tokens).values, runtime_reference(f, tokens, plain), "plain device embedding lookup");
    runtime.release_workspace();
    f.intact();

    auto hidden_only_weights = f.weights;
    hidden_only_weights.token_embedding.reset();
    hidden_only_weights.output_projection.reset();
    modules::LfmDecoderRuntime hidden_only(c, hidden_only_weights, execution);
    modules::LfmDecoderForward hidden_request;
    hidden_request.hidden = f.host(device_hidden);
    hidden_request.positions = a.positions;
    close(hidden_only.forward(hidden_request).values, runtime_reference(f, hidden_request, hidden_request.hidden),
          "hidden-only runtime without embedding or readout tables");
    hidden_only.release_workspace();
    f.intact();
}

void test_cached_runtime(core::ExecutionContext & execution) {
    for (const auto & kinds : {std::vector<Kind>{Kind::ShortConv},
                             std::vector<Kind>{Kind::FullAttention, Kind::SlidingAttention},
                             std::vector<Kind>{Kind::ShortConv, Kind::FullAttention, Kind::ShortConv, Kind::SlidingAttention}}) {
        for (const bool flash : {false, true}) {
            for (const int64_t kernel : {int64_t{1}, int64_t{3}, int64_t{7}}) {
                auto c = configuration(kinds);
                c.conv_kernel_size = kernel;
                if (flash) {
                    c.head_dim = 64;
                    c.attention_lowering = modules::GroupedQueryAttentionLowering::FlashGrouped;
                }
                c.conv_bias = kernel == 3;
                c.full_rope.rotary_dims = 4;
                Fixture f(execution, c);
                f.upload();
                modules::LfmDecoderRuntime runtime(c, f.weights, execution, 4);
                auto in = input(c.hidden_size, 1, 9);
                in.full.clear();
                in.sliding.clear();
                for (int64_t q = 0; q < in.steps; ++q) {
                    in.positions[q] = static_cast<int32_t>(q);
                    for (int64_t k = 0; k < in.steps; ++k) {
                        in.full.push_back(k <= q ? 0.f : -INFINITY);
                        in.sliding.push_back(k <= q && q - k < 2 ? 0.f : -INFINITY);
                    }
                }
                const auto expected = reference(f, in).hidden;
                modules::LfmDecoderForward whole;
                whole.hidden = in.hidden;
                whole.positions = in.positions;
                for (auto v : in.full) { whole.full_mask.push_back(ggml_fp32_to_fp16(v)); }
                for (auto v : in.sliding) { whole.sliding_mask.push_back(ggml_fp32_to_fp16(v)); }
                close(runtime.forward(whole).values, expected, "causal whole sequence oracle");
                auto stream = runtime.create_decode_session(5, 2);
                const auto block = [&](int64_t begin, int64_t count) {
                    modules::LfmDecoderForward request;
                    request.hidden.assign(in.hidden.begin() + begin * c.hidden_size,
                                          in.hidden.begin() + (begin + count) * c.hidden_size);
                    return request;
                };
                const auto check = [&](const modules::LfmDecoderForwardResult & got, int64_t begin, int64_t count, const char * label) {
                    close(got.values, Values(expected.begin() + begin * c.hidden_size,
                                             expected.begin() + (begin + count) * c.hidden_size), label);
                    require_eq(stream->valid_steps(), begin + count, "hybrid state advancement");
                };
                check(stream->prefill(block(0, 3)), 0, 3, "cached prefill oracle");
                stream->release_workspace(); // state must survive scratch destruction
                close(runtime.forward(whole).values, expected, "independent forward while stream populated");
                require_eq(stream->valid_steps(), int64_t{3}, "forward leaves hybrid state untouched");
                check(stream->decode_step(block(3, 1)), 3, 1, "cached step after full forward/release");
                stream->reserve(12); // backend-only growth includes convolution tails
                check(stream->prefill(block(4, 2)), 4, 2, "cached block after capacity growth");
                check(stream->decode_step(block(6, 1)), 6, 1, "cached step");
                const auto built = stream->graphs_built();
                check(stream->decode_step(block(7, 1)), 7, 1, "cached step topology reuse");
                require_eq(stream->graphs_built(), built, "step graph reused at next position");
                check(stream->decode_step(block(8, 1)), 8, 1, "cached final step");
                stream->reset();
                require_eq(stream->valid_steps(), int64_t{0}, "reset hybrid state");
                check(stream->prefill(block(0, 3)), 0, 3, "reset reproduces prefix");
                stream->reset();
                for (int64_t i = 0; i < in.steps; ++i) { check(stream->decode_step(block(i, 1)), i, 1, "one-row prefill equivalence"); }
                // Sessions from one runtime have independent KV and short-conv state.
                auto other = runtime.create_decode_session(9, 2);
                close(other->prefill(block(0, 3)).values, Values(expected.begin(), expected.begin() + 3 * c.hidden_size), "independent stream");
                require_eq(stream->valid_steps(), int64_t{9}, "other stream leaves original state intact");
                std::unique_ptr<modules::LfmDecoderRuntime::DecodeSession> detached;
                {
                    modules::LfmDecoderRuntime factory(c, f.weights, execution);
                    detached = factory.create_decode_session(9, 2);
                }
                close(detached->prefill(block(0, 3)).values, Values(expected.begin(), expected.begin() + 3 * c.hidden_size),
                      "decode session outlives factory runtime");
                // Host readback is optional; device results expose the same rows.
                stream->reset();
                const auto device = stream->prefill_device(block(0, 3));
                close(core::read_tensor_f32(device.output.tensor), Values(expected.begin(), expected.begin() + 3 * c.hidden_size),
                      "device-resident cached output");
                auto invalid = block(3, 1);
                invalid.positions = {3};
                bool rejected = false;
                try { stream->prefill(invalid); } catch (const std::invalid_argument &) { rejected = true; }
                require(rejected && stream->valid_steps() == 3, "invalid append leaves stream usable");
                check(stream->decode_step(block(3, 1)), 3, 1, "append after invalid request");
                invalid = block(0, 9);
                rejected = false;
                try { stream->prefill(invalid); } catch (const engine::runtime::CapacityError &) { rejected = true; }
                require(rejected && stream->valid_steps() == 4, "capacity rejection preserves state");
                // Embedding + selected head is fused into the cached graph too.
                stream->reset();
                modules::LfmDecoderForward tokens;
                tokens.token_ids = {2, 7, 1};
                tokens.readout_positions = {2};
                tokens.output_tokens = {8, 0, 8};
                tokens.readout = modules::LfmDecoderReadout::Logits;
                auto token_reference = tokens;
                token_reference.positions = {0, 1, 2};
                std::vector<float> embeddings;
                const auto & table = f.host(*f.weights.token_embedding);
                for (const auto id : tokens.token_ids) {
                    embeddings.insert(embeddings.end(), table.begin() + id * c.hidden_size, table.begin() + (id + 1) * c.hidden_size);
                }
                for (int64_t q = 0; q < 3; ++q) {
                    for (int64_t k = 0; k < 3; ++k) {
                        token_reference.full_mask.push_back(ggml_fp32_to_fp16(k <= q ? 0.f : -INFINITY));
                        token_reference.sliding_mask.push_back(ggml_fp32_to_fp16(k <= q && q - k < 2 ? 0.f : -INFINITY));
                    }
                }
                close(stream->prefill(tokens).values, runtime_reference(f, token_reference, embeddings), "cached embedding and sparse head");
                f.intact();
            }
        }
    }
}


// Compare a growing F16 cache with an independently allocated full-capacity
// control. Input/readout changes exercise the retained decode arenas; releasing
// scratch and growing storage must preserve both KV rows and convolution history.
void test_f16_cache_lifecycle(core::ExecutionContext & execution) {
    for (const auto lowering : {modules::LfmDecoderConvLowering::Depthwise2d, modules::LfmDecoderConvLowering::OrderedTaps}) {
        for (const bool preserve_layout : {false, true}) {
            auto c = configuration({Kind::ShortConv, Kind::FullAttention, Kind::SlidingAttention});
            c.conv_lowering = lowering;
            c.preserve_mul_input_layout = preserve_layout;
            Fixture f(execution, c);
            f.weights.token_embedding = f.tensor({11, c.hidden_size});
            f.weights.output_projection = f.linear(7, c.hidden_size);
            f.upload();
            modules::LfmDecoderRuntime runtime(c, f.weights, execution, 8);
            auto growing = runtime.create_decode_session(4, 2, GGML_TYPE_F16);
            auto control = runtime.create_decode_session(12, 2, GGML_TYPE_F16);
            const auto in = input(c.hidden_size, 1, 11);
            const auto request = [&](int64_t begin, int64_t count) {
                modules::LfmDecoderForward out;
                if (count == 1 && begin % 2) {
                    out.token_ids = {static_cast<int32_t>(begin % 11)};
                } else {
                    out.hidden.assign(in.hidden.begin() + begin * c.hidden_size,
                                      in.hidden.begin() + (begin + count) * c.hidden_size);
                }
                out.readout = begin % 3 ? modules::LfmDecoderReadout::Hidden : modules::LfmDecoderReadout::Logits;
                return out;
            };
            const auto prefix = request(0, 3);
            close(growing->prefill(prefix).values, doubles(control->prefill(prefix).values), "F16 cache prefix");
            for (int64_t row = 3; row < in.steps; ++row) {
                if (row == 4) {
                    growing->release_workspace();
                    growing->reserve(12);
                }
                const auto next = request(row, 1);
                close(growing->decode_step(next).values, doubles(control->decode_step(next).values), "F16 cache growth and graph switching");
                require_eq(growing->valid_steps(), row + 1, "F16 state advances once per step");
            }
            growing->reset();
            control->reset();
            close(growing->prefill(prefix).values, doubles(control->prefill(prefix).values), "F16 reset reproduces prefix");
            f.intact();
        }
    }
}

// The composed builder and DecodeSession must agree on full-cache and explicit
// prefix extents while retaining independently owned KV and convolution history.
void test_composed_cache_extent(core::ExecutionContext & execution) {
    const auto c = configuration({Kind::ShortConv, Kind::FullAttention, Kind::SlidingAttention});
    Fixture f(execution, c);
    f.upload();
    const auto in = input(c.hidden_size, 1, 6);
    constexpr int64_t capacity = 8, steps = 2;
    for (const auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (const int64_t extent : {int64_t{0}, capacity, in.steps}) {
            modules::LfmDecoderRuntime runtime(c, f.weights, execution);
            auto control = runtime.create_decode_session(capacity, 2, type);
            modules::LfmDecoderState state(execution, c, capacity, type);
            const int64_t keys = extent ? extent : capacity;
            for (int64_t begin = 0; begin < in.steps; begin += steps) {
                Graph g(execution);
                core::ModuleBuildContext ctx{g.context.get(), "lfm_decoder_test.composed", execution.backend_type()};
                modules::LfmDecoderInputs x;
                x.hidden = g.input(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, steps, c.hidden_size}));
                x.positions = g.input(ctx, GGML_TYPE_I32, TensorShape::from_dims({steps}));
                x.full_mask = g.input(ctx, GGML_TYPE_F16, TensorShape::from_dims({steps, keys}));
                x.sliding_mask = g.input(ctx, GGML_TYPE_F16, TensorShape::from_dims({steps, keys}));
                const auto slots = g.input(ctx, GGML_TYPE_I64, TensorShape::from_dims({steps}));
                modules::LfmDecoderCacheInputs cache{state.layers(), slots};
                cache.attention_rows = extent;
                for (const int64_t invalid : {int64_t{-1}, capacity + 1}) {
                    cache.attention_rows = invalid;
                    bool rejected = false;
                    try { (void)modules::build_lfm_decoder(ctx, g.graph, x, f.weights.decoder, c, cache); }
                    catch (const std::invalid_argument &) { rejected = true; }
                    require(rejected, "invalid composed attention extent rejected");
                }
                cache.attention_rows = extent;
                if (extent == 0 && begin == 0) {
                    modules::LfmDecoderState wider(execution, c, capacity + 2, type);
                    auto mixed = state.layers();
                    mixed.back() = wider.layers().back();
                    bool rejected = false;
                    try { (void)modules::build_lfm_decoder(ctx, g.graph, x, f.weights.decoder, c, {mixed, slots}); }
                    catch (const std::invalid_argument &) { rejected = true; }
                    require(rejected, "ambiguous full-cache extent rejected");
                }
                const auto out = modules::build_lfm_decoder(ctx, g.graph, x, f.weights.decoder, c, cache).hidden;
                ggml_set_output(out.tensor);
                ggml_build_forward_expand(g.graph, out.tensor);
                core::validate_backend_graph_with_cpu_extra_buffers(execution.backend(), g.graph, "lfm_decoder_test.composed");
                g.allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
                require(g.allocator && ggml_gallocr_reserve(g.allocator.get(), g.graph) &&
                        ggml_gallocr_alloc_graph(g.allocator.get(), g.graph), "composed graph allocation");
                modules::LfmDecoderForward request;
                request.hidden.assign(in.hidden.begin() + begin * c.hidden_size,
                                      in.hidden.begin() + (begin + steps) * c.hidden_size);
                core::write_tensor_f32(x.hidden, request.hidden);
                const std::vector<int32_t> positions{static_cast<int32_t>(begin), static_cast<int32_t>(begin + 1)};
                const std::vector<int64_t> indices{begin, begin + 1};
                core::write_tensor_i32(x.positions, positions);
                ggml_backend_tensor_set(slots.tensor, indices.data(), 0, indices.size() * sizeof(int64_t));
                std::vector<ggml_fp16_t> full(steps * keys), sliding(steps * keys);
                for (int64_t q = 0; q < steps; ++q) {
                    for (int64_t k = 0; k < keys; ++k) {
                        full[q * keys + k] = ggml_fp32_to_fp16(k <= begin + q ? 0.f : -INFINITY);
                        sliding[q * keys + k] = ggml_fp32_to_fp16(k <= begin + q && k > begin + q - 2 ? 0.f : -INFINITY);
                    }
                }
                ggml_backend_tensor_set(x.full_mask->tensor, full.data(), 0, full.size() * sizeof(ggml_fp16_t));
                ggml_backend_tensor_set(x.sliding_mask->tensor, sliding.data(), 0, sliding.size() * sizeof(ggml_fp16_t));
                require(core::compute_backend_graph(execution.backend(), g.graph) == GGML_STATUS_SUCCESS, "composed graph compute");
                ggml_backend_synchronize(execution.backend());
                close(core::read_tensor_f32(out.tensor), doubles(control->prefill(request).values), "composed cache extent");
            }
            f.intact();
        }
    }
}

}  // namespace

int main(int argc, char ** argv) {
    const auto started = std::chrono::steady_clock::now();
    try {
        core::BackendConfig backend;
        backend.threads = 8;
        engine::debug::LoggingConfig logging;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--backend" && i + 1 < argc) {
                const std::string value = argv[++i];
                if (value == "cpu") { backend.type = core::BackendType::Cpu; }
                else if (value == "cuda") { backend.type = core::BackendType::Cuda; }
                else if (value == "vulkan") { backend.type = core::BackendType::Vulkan; }
                else if (value == "metal") { backend.type = core::BackendType::Metal; }
                else { throw std::invalid_argument("unsupported backend: " + value); }
            } else if (arg == "--log" && i + 1 < argc) {
                logging.enabled = true;
                logging.file_path = argv[++i];
            } else { throw std::invalid_argument("expected --backend cpu|cuda|vulkan|metal or --log PATH"); }
        }
        engine::debug::configure_logging(logging);
        core::ExecutionContext execution(backend);
        require(execution.backend() != nullptr, "requested backend unavailable");
        // Source-traced CUDA TF32 / Vulkan FP16 operands can perturb individual
        // near-zero components. Check whole-vector direction AND gain against
        // the unchanged FP64 oracle; never escalate limits after a failure.
        use_vector_agreement = execution.backend_type() == core::BackendType::Cuda ||
                               execution.backend_type() == core::BackendType::Vulkan;
        engine::debug::trace_log_scalar("lfm_decoder_test.comparison.policy", std::string_view(use_vector_agreement
            ? "CUDA TF32 / Vulkan FP16 operands: cosine and norm-gain error against independent FP64 oracle"
            : "Strict pointwise F32 comparison against independent FP64 oracle"));
        if (use_vector_agreement) {
            engine::debug::trace_log_scalar("lfm_decoder_test.tolerance.minimum_cosine", kMinimumCosine);
            engine::debug::trace_log_scalar("lfm_decoder_test.tolerance.maximum_norm_gain_error", kMaximumNormGainError);
        } else {
            engine::debug::trace_log_scalar("lfm_decoder_test.tolerance.atol", kPointwiseTolerance);
            engine::debug::trace_log_scalar("lfm_decoder_test.tolerance.rtol", kPointwiseTolerance);
        }
        test_math(execution);
        test_frequency_factors(execution);
        test_runtime(execution);
        test_cached_runtime(execution);
        test_f16_cache_lifecycle(execution);
        test_composed_cache_extent(execution);
        if (execution.backend_type() == core::BackendType::Cpu) { test_correlated_cpu_stress(execution); }
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        std::cout << "lfm_decoder_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        std::cerr << "lfm_decoder_test: " << error.what() << '\n';
        return 1;
    }
}
