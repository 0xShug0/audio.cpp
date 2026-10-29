#include "engine/community_models/lfm3_audio/blocks.h"

#include "../lfm2_audio/weight_stores.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/grouped_query_attention.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/errors.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::community_models::lfm3_audio::blocks {

using core::TensorShape;
using core::TensorValue;

namespace {

// theta / 1e30 rounds to 0 even under fast-math division, unlike theta / INF.
constexpr float kZeroTailFactor = 1e30f;

// CPU and Metal run ggml_ssm_conv for any kernel width. CUDA's dispatch aborts on K = 7 although supports_op
// accepts it, and other backends are unverified, so they take K shifted row views, or one multiply and row sum
// for a single step.
bool ssm_conv_runs(core::BackendType type) {
    return type == core::BackendType::Cpu || type == core::BackendType::Metal;
}

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("LFM3-Audio blocks: " + message);
}

// [1]*rot/2 + [1e30]*(hd-rot)/2
std::vector<float> zero_tail_freq_factors(int64_t head_dim, int64_t rotary_dims) {
    if (head_dim <= 0 || head_dim % 2 != 0 || rotary_dims < 0 || rotary_dims > head_dim || rotary_dims % 2 != 0) {
        fail("zero_tail_freq_factors needs an even head_dim and even rotary_dims <= head_dim");
    }
    std::vector<float> out(static_cast<size_t>(head_dim / 2), kZeroTailFactor);
    std::fill(out.begin(), out.begin() + rotary_dims / 2, 1.0f);
    return out;
}

void validate_config(const StackConfig & config) {
    if (config.hidden <= 0 || config.ffn <= 0 || config.layers.empty()) {
        fail("stack needs a positive hidden size, FFN size and at least one layer");
    }

    bool conv = false;
    bool attention = false;
    bool sliding = false;
    for (const auto kind : config.layers) {
        conv = conv || kind == LayerKind::Conv;
        attention = attention || kind != LayerKind::Conv;
        sliding = sliding || kind == LayerKind::SlidingAttention;
    }
    if (conv && config.conv_kernel < 2) {
        fail("conv layers need a kernel of at least 2");
    }
    if (attention && (config.heads <= 0 || config.kv_heads <= 0 || config.head_dim <= 0 || config.head_dim % 2 != 0 ||
                      config.heads % config.kv_heads != 0)) {
        fail("attention needs positive heads divisible by kv_heads and an even head_dim");
    }
    if (sliding && config.sliding_window <= 0) {
        fail("sliding attention layers need a positive window");
    }
    if (config.full_rotary_dims < 0 || config.full_rotary_dims > config.head_dim || config.full_rotary_dims % 2 != 0) {
        fail("full_rotary_dims must be even and at most head_dim");
    }
}

std::string layer_prefix(const std::string & prefix, size_t layer) {
    return prefix + "layers." + std::to_string(layer) + ".";
}

modules::NormWeights load_norm(WeightStores & stores, const assets::TensorSource & source, const std::string & name, int64_t size) {
    return {stores.plain().load_f32_tensor(source, name, {size}), std::nullopt};
}

TensorValue rms_norm(core::ModuleBuildContext & ctx, const TensorValue & x, const modules::NormWeights & weights, int64_t size, float eps) {
    return modules::RMSNormModule({size, eps, true, false}).build(ctx, x, weights);
}

TensorValue linear(core::ModuleBuildContext & ctx, const TensorValue & x, const TensorValue & weight, int64_t in, int64_t out) {
    return modules::LinearModule({in, out, false, kMatmulPrecision}).build(ctx, x, {weight, std::nullopt});
}

// Rows [0, rows) of a [1, capacity, kv_heads, head_dim] cache, viewed from the tensor set_rows returned so that the
// read is ordered after the write.
TensorValue cache_rows(core::ModuleBuildContext & ctx, const TensorValue & cache, int64_t rows) {
    auto * t = cache.tensor;
    auto * view = ggml_view_4d(ctx.ggml, t, t->ne[0], t->ne[1], rows, 1, t->nb[1], t->nb[2], t->nb[3], 0);
    return core::wrap_tensor(view, TensorShape::from_dims({1, rows, cache.shape.dims[2], cache.shape.dims[3]}), cache.type);
}

void validate_mask(const TensorValue & mask, int64_t kv, int64_t steps, const char * name) {
    if (mask.tensor == nullptr || mask.tensor->type != GGML_TYPE_F16 || mask.tensor->ne[0] != kv || mask.tensor->ne[1] != steps ||
        mask.tensor->ne[2] != 1 || mask.tensor->ne[3] != 1) {
        fail(std::string(name) + " must be F16 [" + std::to_string(kv) + " keys, " + std::to_string(steps) + " queries]");
    }
}

// in_proj splits into B, C and x; the causal conv runs over B * x, gated by C (Lfm2ShortConv / Lfm3ShortConv).
TensorValue short_conv(core::ModuleBuildContext & ctx, ggml_cgraph * graph, const StackConfig & config, const ConvWeights & w,
                       const TensorValue & normed, LayerState * state) {
    auto * g = ctx.ggml;
    const int64_t d = config.hidden;
    const int64_t k = config.conv_kernel;
    const int64_t steps = normed.shape.dims[1];

    auto bcx = linear(ctx, normed, w.in_proj, d, 3 * d);
    auto chunk = [&](int64_t index) {
        return ggml_view_2d(g, bcx.tensor, d, steps, bcx.tensor->nb[1], static_cast<size_t>(index * d) * ggml_element_size(bcx.tensor));
    };
    auto * bx = ggml_mul(g, chunk(0), chunk(2));

    ggml_tensor * padded = nullptr;
    if (state != nullptr) {
        const auto & tail = state->conv_state;
        if (tail.tensor == nullptr || tail.tensor->type != GGML_TYPE_F32 || tail.tensor->ne[0] != d || tail.tensor->ne[1] != k - 1) {
            fail("conv state must be F32 [kernel - 1, hidden]");
        }
        padded = ggml_concat(g, tail.tensor, bx, 1);
    } else {
        // A stateless sequence starts from zero history.
        padded = ggml_pad_ext(g, bx, 0, 0, static_cast<int>(k - 1), 0, 0, 0, 0, 0);
    }

    auto y = causal_conv(ctx, core::wrap_tensor(padded, TensorShape::from_dims({steps + k - 1, d})), w.kernel_t, steps,
                         w.kernel.tensor != nullptr ? &w.kernel : nullptr);
    auto * gated = ggml_mul(g, chunk(1), ggml_reshape_2d(g, y.tensor, d, steps));

    if (state != nullptr) {
        auto * next_tail = ggml_view_2d(g, padded, d, k - 1, padded->nb[1], static_cast<size_t>(steps) * padded->nb[1]);
        ggml_build_forward_expand(graph, ggml_cpy(g, next_tail, state->conv_state.tensor));
    }

    return linear(ctx, core::wrap_tensor(gated, TensorShape::from_dims({1, steps, d})), w.out_proj, d, d);
}

TensorValue feed_forward(core::ModuleBuildContext & ctx, const TensorValue & x, const LayerWeights & w, const StackConfig & config) {
    modules::GatedFeedForwardConfig ff;
    ff.hidden_size = config.hidden;
    ff.intermediate_size = config.ffn;
    ff.activation = modules::GatedFeedForwardActivation::Silu;
    ff.projection_precision = kMatmulPrecision;
    return modules::GatedFeedForwardModule(ff).build(ctx, x, {{w.w1, std::nullopt}, {w.w3, std::nullopt}, {w.w2, std::nullopt}});
}

}  // namespace

TensorValue load_matmul(WeightStores & stores, ggml_backend_t backend, const assets::TensorSource & source, const std::string & name,
                        const std::vector<int64_t> & shape, const Lfm3WeightOptions & options) {
    const auto meta = source.require_metadata(name);
    const auto storage = lfm3_storage_type(backend, assets::ggml_type_for_tensor_dtype(meta.dtype), options);
    auto stored_shape = meta.shape;
    while (stored_shape.size() > shape.size() && stored_shape.back() == 1) {
        stored_shape.pop_back();
    }
    if (stored_shape.size() != meta.shape.size() && stored_shape == shape) {
        return stores.plain().load_tensor_as_shape(source, name, storage, meta.shape, TensorShape::from_dims({shape.at(0), shape.at(1)}));
    }
    if (storage == assets::TensorStorageType::Native) {
        return stores.load_matmul(source, name, shape);
    }
    return stores.plain().load_tensor(source, name, storage, shape);
}

ggml_context_ptr graph_context(size_t nodes, const std::string & what) {
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize the LFM3-Audio " + what + " graph context");
    }
    return ctx;
}

void compute(core::ExecutionContext & execution, ggml_cgraph * graph, const std::string & what) {
    core::set_backend_threads(execution.backend(), std::max(1, execution.config().threads));
    const ggml_status status = core::compute_backend_graph(execution.backend(), graph);
    ggml_backend_synchronize(execution.backend());
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("LFM3-Audio " + what + " graph compute failed");
    }
}

StackState::StackState(core::ExecutionContext & execution, const StackConfig & config, int64_t capacity, ggml_type cache_type,
                       const std::string & what)
    : capacity_(capacity), ctx_(ggml_init({ggml_tensor_overhead() * 2 * config.layers.size(), nullptr, true})) {
    if (ctx_ == nullptr) {
        throw std::runtime_error("LFM3-Audio " + what + ": failed to initialize the state context");
    }
    core::ModuleBuildContext build{ctx_.get(), "lfm3_audio.state", execution.backend_type()};
    for (const auto kind : config.layers) {
        LayerState state;
        if (kind == LayerKind::Conv) {
            state.conv_state = core::make_tensor(build, GGML_TYPE_F32, TensorShape::from_dims({config.conv_kernel - 1, config.hidden}));
        } else {
            const auto shape = TensorShape::from_dims({1, capacity, config.kv_heads, config.head_dim});
            state.key_cache = core::make_tensor(build, cache_type, shape);
            state.value_cache = core::make_tensor(build, cache_type, shape);
        }
        layers.push_back(state);
    }
    buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(), execution.backend()));
    if (buffer_ == nullptr) {
        throw runtime::CapacityError("LFM3-Audio " + what + " state does not fit in device memory at " + std::to_string(capacity) +
                                     " cache rows");
    }
    // Masked rows are still multiplied before the mask applies: uninitialized NaN there would leak.
    clear();
}

void StackState::clear() {
    ggml_backend_buffer_clear(buffer_.get(), 0);
}

StackWeights load_stack(WeightStores & stores, ggml_backend_t backend, const assets::TensorSource & source, const std::string & prefix,
                        const StackConfig & config, const Lfm3WeightOptions & options) {
    validate_config(config);
    const int64_t d = config.hidden;
    const int64_t hd = config.head_dim;
    const bool lfm2 = config.names == Names::Lfm2;

    StackWeights out;
    for (size_t layer = 0; layer < config.layers.size(); ++layer) {
        const std::string p = layer_prefix(prefix, layer);
        LayerWeights w;
        w.kind = config.layers[layer];
        w.operator_norm = load_norm(stores, source, p + "operator_norm.weight", d);
        w.ffn_norm = load_norm(stores, source, p + "ffn_norm.weight", d);
        if (config.sandwich_norm) {
            w.post_operator_norm = load_norm(stores, source, p + "post_operator_norm.weight", d);
            w.post_ffn_norm = load_norm(stores, source, p + "post_ffn_norm.weight", d);
        }
        w.w1 = load_matmul(stores, backend, source, p + "feed_forward.w1.weight", {config.ffn, d}, options);
        w.w3 = load_matmul(stores, backend, source, p + "feed_forward.w3.weight", {config.ffn, d}, options);
        w.w2 = load_matmul(stores, backend, source, p + "feed_forward.w2.weight", {d, config.ffn}, options);

        if (w.kind == LayerKind::Conv) {
            const int64_t k = config.conv_kernel;
            w.conv.in_proj = load_matmul(stores, backend, source, p + "conv.in_proj.weight", {3 * d, d}, options);
            w.conv.out_proj = load_matmul(stores, backend, source, p + "conv.out_proj.weight", {d, d}, options);
            const auto kernel = source.require_f32(p + "conv.conv.weight", {d, 1, k});
            std::vector<float> kernel_t(kernel.size());
            for (int64_t c = 0; c < d; ++c) {
                for (int64_t i = 0; i < k; ++i) {
                    kernel_t[static_cast<size_t>(i * d + c)] = kernel[static_cast<size_t>(c * k + i)];
                }
            }
            w.conv.kernel_t = stores.plain().make_f32(TensorShape::from_dims({k, d}), std::move(kernel_t));
            w.conv.kernel = stores.plain().make_f32(TensorShape::from_dims({d, k}), kernel);
        } else {
            const std::string a = p + "self_attn.";
            const int64_t q_rows = config.heads * hd;
            const int64_t kv_rows = config.kv_heads * hd;
            w.attention.q = load_matmul(stores, backend, source, a + "q_proj.weight", {q_rows, d}, options);
            w.attention.k = load_matmul(stores, backend, source, a + "k_proj.weight", {kv_rows, d}, options);
            w.attention.v = load_matmul(stores, backend, source, a + "v_proj.weight", {kv_rows, d}, options);
            w.attention.o = load_matmul(stores, backend, source, a + (lfm2 ? "out_proj.weight" : "o_proj.weight"), {d, q_rows}, options);
            w.attention.q_norm = load_norm(stores, source, a + (lfm2 ? "q_layernorm.weight" : "q_norm.weight"), hd);
            w.attention.k_norm = load_norm(stores, source, a + (lfm2 ? "k_layernorm.weight" : "k_norm.weight"), hd);
            if (config.output_gate) {
                w.attention.gate = load_matmul(stores, backend, source, a + "gate_proj.weight", {config.heads, d}, options);
            }
        }
        out.layers.push_back(std::move(w));
    }

    out.final_norm = load_norm(stores, source, prefix + "embedding_norm.weight", d);
    if (config.full_rotary_dims != 0 && config.full_rotary_dims != hd) {
        out.full_freq_factors = stores.plain().make_f32(TensorShape::from_dims({hd / 2}), zero_tail_freq_factors(hd, config.full_rotary_dims));
    }
    return out;
}

TensorValue build_stack(core::ModuleBuildContext & ctx, ggml_cgraph * graph, const StackConfig & config, const StackWeights & weights,
                        const StepInputs & in, std::vector<LayerState> * state, const Lfm3WeightOptions & options) {
    validate_config(config);
    if (weights.layers.size() != config.layers.size() || (state != nullptr && state->size() != config.layers.size())) {
        fail("stack weights and state must have one entry per layer");
    }
    const int64_t d = config.hidden;
    if (in.x.tensor == nullptr || in.x.shape.rank != 3 || in.x.shape.dims[0] != 1 || in.x.shape.dims[2] != d) {
        fail("stack input must be [1, steps, " + std::to_string(d) + "]");
    }

    auto x = in.x;
    for (size_t layer = 0; layer < config.layers.size(); ++layer) {
        const auto & w = weights.layers[layer];
        LayerState * layer_state = state != nullptr ? &(*state)[layer] : nullptr;

        auto hn = rms_norm(ctx, x, w.operator_norm, d, config.eps);
        auto y = w.kind == LayerKind::Conv
                     ? short_conv(ctx, graph, config, w.conv, hn, layer_state)
                     : gated_attention(ctx, config, w.kind, w.attention, hn, in, layer_state, weights, options);
        if (w.post_operator_norm) {
            y = rms_norm(ctx, y, *w.post_operator_norm, d, config.eps);
        }
        x = modules::AddModule{}.build(ctx, x, y);

        auto f = feed_forward(ctx, rms_norm(ctx, x, w.ffn_norm, d, config.eps), w, config);
        if (w.post_ffn_norm) {
            f = rms_norm(ctx, f, *w.post_ffn_norm, d, config.eps);
        }
        x = modules::AddModule{}.build(ctx, x, f);
    }

    return rms_norm(ctx, x, weights.final_norm, d, config.eps);
}

TensorValue causal_conv(core::ModuleBuildContext & ctx, const TensorValue & padded, const TensorValue & kernel_t, int64_t steps,
                        const TensorValue * kernel) {
    auto * g = ctx.ggml;
    auto * p = padded.tensor;
    auto * kt = kernel_t.tensor;
    if (kt == nullptr || kt->type != GGML_TYPE_F32 || !ggml_is_contiguous(kt) || ggml_n_dims(kt) > 2) {
        fail("conv kernel must be a contiguous F32 [kernel, hidden]");
    }
    const int64_t d = kt->ne[0];
    const int64_t k = kt->ne[1];
    if (p == nullptr || p->type != GGML_TYPE_F32 || p->ne[0] != d || p->ne[1] != steps + k - 1 || p->ne[2] != 1 || p->ne[3] != 1 ||
        p->nb[0] != sizeof(float)) {
        fail("conv input must be F32 [steps + kernel - 1, hidden]");
    }

    const bool ssm = ssm_conv_runs(ctx.backend_type);
    if (ssm || steps == 1) {
        // Time innermost: padded as ne [S+K-1, D], the kernel as ne [K, D].
        auto * sx = ggml_cont(g, ggml_transpose(g, p));
        if (kernel != nullptr && (kernel->tensor->type != GGML_TYPE_F32 || kernel->tensor->ne[0] != k || kernel->tensor->ne[1] != d)) {
            fail("conv kernel must be F32 [hidden, kernel]");
        }
        auto * c = kernel != nullptr ? kernel->tensor : ggml_cont(g, ggml_transpose(g, kt));
        // One decode step is one window per channel: sum_k padded[k][d] * w[d][k], 3 kernels instead of 2K - 1.
        auto * y = ssm ? ggml_ssm_conv(g, sx, c) : ggml_sum_rows(g, ggml_mul(g, sx, c));
        return core::wrap_tensor(ggml_reshape_3d(g, y, d, steps, 1), TensorShape::from_dims({1, steps, d}));
    }

    // Tap k reads rows k..k+steps: padded[t + k] = bx[t - (K-1) + k], which kernel column k multiplies.
    ggml_tensor * y = nullptr;
    for (int64_t i = 0; i < k; ++i) {
        auto * rows = ggml_view_2d(g, p, d, steps, p->nb[1], static_cast<size_t>(i) * p->nb[1]);
        auto * tap = ggml_view_1d(g, kt, d, static_cast<size_t>(i) * kt->nb[1]);
        auto * term = ggml_mul(g, rows, tap);
        y = y == nullptr ? term : ggml_add(g, y, term);
    }
    return core::wrap_tensor(y, TensorShape::from_dims({1, steps, d}));
}

TensorValue gated_attention(core::ModuleBuildContext & ctx, const StackConfig & config, LayerKind kind, const AttentionWeights & w,
                            const TensorValue & normed, const StepInputs & in, LayerState * state, const StackWeights & stack,
                            const Lfm3WeightOptions & options) {
    const int64_t d = config.hidden;
    const int64_t heads = config.heads;
    const int64_t kv_heads = config.kv_heads;
    const int64_t hd = config.head_dim;
    const int64_t steps = normed.shape.dims[1];

    auto q = core::reshape_tensor(ctx, linear(ctx, normed, w.q, d, heads * hd), TensorShape::from_dims({1, steps, heads, hd}));
    auto k = core::reshape_tensor(ctx, linear(ctx, normed, w.k, d, kv_heads * hd), TensorShape::from_dims({1, steps, kv_heads, hd}));
    auto v = core::reshape_tensor(ctx, linear(ctx, normed, w.v, d, kv_heads * hd), TensorShape::from_dims({1, steps, kv_heads, hd}));
    q = rms_norm(ctx, q, w.q_norm, hd, config.eps);
    k = rms_norm(ctx, k, w.k_norm, hd, config.eps);

    const bool full = kind == LayerKind::FullAttention;
    const TensorValue * factors = full && stack.full_freq_factors ? &*stack.full_freq_factors : nullptr;
    const modules::RoPEModule rope({hd, GGML_ROPE_TYPE_NEOX, full ? config.full_theta : config.sliding_theta});
    q = rope.build(ctx, q, in.positions, factors);
    k = rope.build(ctx, k, in.positions, factors);

    TensorValue keys = k;
    TensorValue values = v;
    int64_t kv = steps;
    if (state != nullptr) {
        const auto & key_cache = state->key_cache;
        const auto & value_cache = state->value_cache;
        const auto expected = TensorShape::from_dims({1, key_cache.shape.dims[1], kv_heads, hd});
        if (key_cache.tensor == nullptr || value_cache.tensor == nullptr || key_cache.shape.rank != 4 ||
            key_cache.shape.dims != expected.dims || value_cache.shape.dims != expected.dims) {
            fail("attention caches must be [1, capacity, " + std::to_string(kv_heads) + ", " + std::to_string(hd) + "]");
        }
        const int64_t capacity = key_cache.shape.dims[1];
        if (capacity % kKvRowAlignment != 0 || in.kv_rows <= 0 || in.kv_rows % kKvRowAlignment != 0 || in.kv_rows > capacity) {
            fail("cache capacity (" + std::to_string(capacity) + ") and kv_rows (" + std::to_string(in.kv_rows) +
                 ") must be multiples of 256 with kv_rows <= capacity");
        }
        if (!in.slots) {
            fail("a stateful stack needs cache slots");
        }
        const modules::FastKVSetRowsModule set_rows;
        keys = cache_rows(ctx, set_rows.build_block(ctx, key_cache, k, *in.slots), in.kv_rows);
        values = cache_rows(ctx, set_rows.build_block(ctx, value_cache, v, *in.slots), in.kv_rows);
        kv = in.kv_rows;
    }

    std::optional<TensorValue> mask;
    if (config.causal) {
        mask = full ? in.full_mask : in.sliding_mask;
        if (!mask) {
            fail(std::string("a causal stack needs the ") + (full ? "full" : "sliding") + " mask");
        }
        validate_mask(*mask, kv, steps, full ? "full mask" : "sliding mask");
    }

    // Reference precision and head dims outside the set every backend's flash kernel has (CUDA lacks 32) use the
    // explicit softmax; the masks are F16 for both.
    const bool flash = !lfm3_reference_precision(options) && (hd == 64 || hd == 96 || hd == 128);
    const modules::TransposeModule heads_first({{0, 2, 1, 3}, 4});
    auto q_heads = heads_first.build(ctx, q);
    q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    auto context = modules::GroupedQueryAttentionModule({
                                                            hd,
                                                            flash ? modules::GroupedQueryAttentionLowering::FlashGroupedViewKV
                                                                  : modules::GroupedQueryAttentionLowering::ManualRepeat,
                                                            GGML_PREC_F32,
                                                            modules::AttentionCausality::NonCausal,
                                                        })
                       .build(ctx, q_heads, heads_first.build(ctx, keys), heads_first.build(ctx, values), mask);
    context = core::ensure_backend_addressable_layout(ctx, context);  // [1, steps, heads, hd]

    if (w.gate) {
        auto gate = modules::SigmoidModule{}.build(ctx, linear(ctx, normed, *w.gate, d, heads));
        gate = core::reshape_tensor(ctx, gate, TensorShape::from_dims({1, steps, heads, 1}));
        context = core::wrap_tensor(ggml_mul(ctx.ggml, context.tensor, gate.tensor), context.shape);
    }

    context = core::reshape_tensor(ctx, context, TensorShape::from_dims({1, steps, heads * hd}));
    return linear(ctx, context, w.o, heads * hd, d);
}

std::vector<ggml_fp16_t> window_mask(const std::vector<int32_t> & q_pos, int64_t kv, int64_t window) {
    if (kv < 0 || window < 0) {
        fail("window_mask needs non-negative kv and window");
    }
    const ggml_fp16_t visible = ggml_fp32_to_fp16(0.0f);
    std::vector<ggml_fp16_t> out(q_pos.size() * static_cast<size_t>(kv), ggml_fp32_to_fp16(-INFINITY));
    for (size_t i = 0; i < q_pos.size(); ++i) {
        const int64_t q = q_pos[i];
        for (int64_t key = 0; key < kv; ++key) {
            if (key <= q && (window == 0 || q - key < window)) {
                out[i * static_cast<size_t>(kv) + static_cast<size_t>(key)] = visible;
            }
        }
    }
    return out;
}

}  // namespace engine::community_models::lfm3_audio::blocks
