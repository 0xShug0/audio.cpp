#include "engine/community_models/parakeet_tdt/encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/asr_helpers.h"
#include "engine/framework/modules/attention_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/graph_optimizer.h"

#include "../../framework/modules/attention/attention_internal.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::community_models::parakeet_tdt {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kEncoderGraphNodes = 2097152;

// NeMo's ConformerConvolution.depthwise_conv is a CausalConv1D, but the
// full-context (non-streaming) encoder constructs it with an explicit int
// `padding=conv_context_size=(kernel_size-1)//2`, which CausalConv1D treats
// as SYMMETRIC left/right padding, not causal-only left padding. Using
// causal-only padding here shifts every frame's receptive field and corrupts
// the residual stream for the whole encoder stack.
engine::core::TensorValue pad_symmetric_1d(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t kernel) {
    const int64_t pad = (kernel - 1) / 2;
    if (pad <= 0) { return input; }
    return engine::core::wrap_tensor(
        ggml_pad_ext(ctx.ggml, input.tensor, static_cast<int>(pad), static_cast<int>(pad), 0, 0, 0, 0, 0, 0),
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], input.shape.dims[2] + 2 * pad}),
        GGML_TYPE_F32);
}

engine::core::TensorValue build_fastconformer_conv_module(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_btc,
    const ParakeetEncoderLayerWeights & weights,
    const engine::core::TensorValue & keep_mask,
    int64_t conv_kernel) {
    // pointwise_conv1 runs as a Linear over the feature axis, so it wants plain
    // BTC. The BTC->BCT->BTC transpose pair this used to go through cancelled
    // out exactly; only the depthwise conv below actually needs BCT.
    auto x = engine::modules::LinearModule({input_btc.shape.dims[2], 2 * input_btc.shape.dims[2], false}).build(
        ctx,
        input_btc,
        weights.conv_pw1);
    x = engine::modules::GLUModule().build(ctx, x);
    x = engine::modules::MaskingModule().build(ctx, x, keep_mask);
    x = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);

    const int64_t d_model = x.shape.dims[1];
    x = pad_symmetric_1d(ctx, x, conv_kernel);
    // use_bias=true: weights.conv_dw_bias holds the batch-norm bias term folded
    // into the depthwise conv by fold_bn() in weights.cpp (-running_mean*scale +
    // bn_bias) — it is not an optional/zero bias, it is load-bearing.
    x = engine::modules::DepthwiseConv1dModule({d_model, conv_kernel, 1, 0, 1, true})
            .build(ctx, x, {weights.conv_dw_weight, weights.conv_dw_bias});
    x = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = engine::modules::SiluModule().build(ctx, x);
    return engine::modules::LinearModule({d_model, d_model, false}).build(ctx, x, weights.conv_pw2);
}

std::vector<float> make_relative_positional_encoding(int64_t hidden, int64_t frames, int64_t max_frames) {
    if (frames > max_frames) {
        throw std::runtime_error("Parakeet TDT encoder relative position frames exceed maximum");
    }
    const int64_t pos_frames = 2 * frames - 1;
    std::vector<float> values(static_cast<size_t>(pos_frames * hidden), 0.0f);
    constexpr long double kBase = 10000.0L;
    const int64_t half_hidden = hidden / 2;
    std::vector<long double> inv_freq(static_cast<size_t>(half_hidden), 0.0L);
    std::vector<long double> step_sin(static_cast<size_t>(half_hidden), 0.0L);
    std::vector<long double> step_cos(static_cast<size_t>(half_hidden), 0.0L);
    for (int64_t i = 0; i < half_hidden; ++i) {
        const long double exponent = static_cast<long double>(2 * i) / static_cast<long double>(hidden);
        inv_freq[static_cast<size_t>(i)] = 1.0L / std::pow(kBase, exponent);
        step_sin[static_cast<size_t>(i)] = std::sin(inv_freq[static_cast<size_t>(i)]);
        step_cos[static_cast<size_t>(i)] = std::cos(inv_freq[static_cast<size_t>(i)]);
    }
    std::vector<long double> sin_phase(static_cast<size_t>(half_hidden), 0.0L);
    std::vector<long double> cos_phase(static_cast<size_t>(half_hidden), 0.0L);
    for (int64_t i = 0; i < half_hidden; ++i) {
        const long double phase = static_cast<long double>(frames - 1) * inv_freq[static_cast<size_t>(i)];
        sin_phase[static_cast<size_t>(i)] = std::sin(phase);
        cos_phase[static_cast<size_t>(i)] = std::cos(phase);
    }
    for (int64_t p = 0; p < pos_frames; ++p) {
        for (int64_t i = 0; i < half_hidden; ++i) {
            const size_t dst = static_cast<size_t>(p * hidden + 2 * i);
            values[dst] = static_cast<float>(sin_phase[static_cast<size_t>(i)]);
            values[dst + 1] = static_cast<float>(cos_phase[static_cast<size_t>(i)]);
            const long double next_sin = sin_phase[i] * step_cos[i] - cos_phase[i] * step_sin[i];
            const long double next_cos = cos_phase[i] * step_cos[i] + sin_phase[i] * step_sin[i];
            sin_phase[i] = next_sin;
            cos_phase[i] = next_cos;
        }
    }
    return values;
}

// Transformer-XL relative shift, as a pure view.
//
// The generic helper (attention::internal::relative_shift) implements the shift
// the textbook way: append a zero column, reinterpret the buffer one row
// narrower, drop the first row, then slice. Each of those steps has to
// materialize — a concat, a cont for the reshape, a cont for the slice — which
// is ~1.1 MB of copying per layer here, and the slice throws most of it away
// immediately afterwards.
//
// But the whole point of that dance is that the shift is just a change of row
// stride, so it can be expressed directly. Writing out the composition of
// pad/reshape/slice for the columns we actually keep gives
//
//     out[h][i][j] = raw[h][i][(seq_len - 1) - i + j]
//
// and since raw is contiguous with row stride pos_len = 2*seq_len - 1, element
// (i, j) sits at flat offset i*(pos_len - 1) + (seq_len - 1) + j. That is a
// plain strided view: start (seq_len - 1) elements in, step (pos_len - 1)
// between rows, keep seq_len columns. Zero copies, bit-identical values, and it
// folds the trailing SliceModule({3, 0, seq_len}) in for free.
engine::core::TensorValue relative_shift_view(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & raw,
    int64_t seq_len) {
    engine::core::validate_rank_between(raw, 4, 4, "relative_shift_view.input");
    const int64_t heads = raw.shape.dims[1];
    const int64_t pos_len = raw.shape.dims[3];
    if (raw.shape.dims[2] != seq_len || pos_len != 2 * seq_len - 1) {
        throw std::runtime_error("Parakeet TDT relative shift expects a [heads, seq, 2*seq-1] score matrix");
    }
    ggml_tensor * base = raw.tensor;  // ggml ne = (pos_len, seq_len, heads, 1), contiguous
    return engine::core::wrap_tensor(
        ggml_view_4d(
            ctx.ggml,
            base,
            seq_len, seq_len, heads, 1,
            static_cast<size_t>(pos_len - 1) * base->nb[0],
            base->nb[2],
            base->nb[3],
            static_cast<size_t>(seq_len - 1) * base->nb[0]),
        engine::core::TensorShape::from_dims({1, heads, seq_len, seq_len}),
        GGML_TYPE_F32);
}

engine::runtime::GraphOptimizationBackend graph_optimizer_backend_for(engine::core::BackendType type) {
    switch (type) {
        case engine::core::BackendType::Cpu:
            return engine::runtime::GraphOptimizationBackend::Cpu;
        case engine::core::BackendType::Cuda:
        case engine::core::BackendType::Hip:
            return engine::runtime::GraphOptimizationBackend::Gpu;
        case engine::core::BackendType::Vulkan:
        case engine::core::BackendType::Metal:
        case engine::core::BackendType::BestAvailable:
            return engine::runtime::GraphOptimizationBackend::Other;
    }
    return engine::runtime::GraphOptimizationBackend::Other;
}

}  // namespace

// Exported (not anonymous-namespace) so test/parity harnesses can build a
// single encoder layer in isolation against the exact same code path the
// production encoder graph uses, instead of maintaining a separate copy that
// could silently drift out of sync. See
// ParakeetEncoderRuntime::ensure_graph()'s per-layer loop for the only other
// caller, and tests/parakeet_tdt/parity/ for the isolation harness.
engine::core::TensorValue build_encoder_layer(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & attention_mask,
    const engine::core::TensorValue & keep_mask,
    const engine::core::TensorValue & projected_pos_emb,
    const ParakeetEncoderLayerWeights & weights,
    int64_t hidden_size,
    int64_t intermediate_size,
    int64_t heads,
    int64_t conv_kernel,
    bool use_flash_attention) {
    namespace ai = engine::modules::attention::internal;

    auto x_norm = engine::modules::LayerNormModule({hidden_size, 1.0e-5f, true, true}).build(ctx, input, weights.norm_ff1);
    auto ff1 = engine::modules::LinearModule({hidden_size, intermediate_size, false}).build(ctx, x_norm, weights.ff1_linear1);
    ff1 = engine::modules::SiluModule().build(ctx, ff1);
    // The 0.5 residual half-step is folded into ff1_linear2's weights at load
    // time (see scaled_f32 in weights.cpp), so no ggml_scale pass here.
    ff1 = engine::modules::LinearModule({intermediate_size, hidden_size, false}).build(ctx, ff1, weights.ff1_linear2);
    auto x = engine::core::wrap_tensor(ggml_add(ctx.ggml, input.tensor, ff1.tensor), input.shape, GGML_TYPE_F32);

    auto attn_input = engine::modules::LayerNormModule({hidden_size, 1.0e-5f, true, true}).build(ctx, x, weights.norm_attn);

    const int64_t head_dim = hidden_size / heads;
    const int64_t seq_len = input.shape.dims[1];
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    if (!weights.self_attn.qkv_weight.has_value()) {
        throw std::runtime_error("Parakeet TDT encoder layer requires a fused QKV weight");
    }
    // One [hidden, 3*hidden] matmul instead of three separate [hidden, hidden]
    // matmuls — see the load-time fusion comment in weights.cpp's load_encoder_layer.
    auto qkv = engine::modules::LinearModule({hidden_size, 3 * hidden_size, false}).build(ctx, attn_input, {*weights.self_attn.qkv_weight, std::nullopt});

    // Read q/k/v out of the fused [seq, 3*hidden] result as strided views that
    // are already in per-head [heads, seq, head_dim] order, instead of slicing
    // the feature axis and calling ensure_contiguous_layout on each slice.
    // Those conts are not optional once you slice: reshape_heads goes through
    // ggml_reshape, which asserts contiguity. So the slice path copies all
    // three projections in full (3 * seq * hidden floats per layer) purely to
    // satisfy the reshape — and then MatMulModule re-materializes its own
    // operands for k and v anyway, so those two get copied twice. A view costs
    // nothing, and leaves exactly one copy of k and one of v (inside
    // MatMulModule) for the whole attention block. Same values, same order.
    auto qkv_head_view = [&](int64_t feature_offset) {
        ggml_tensor * base = qkv.tensor;  // ggml ne = (3*hidden, seq), contiguous
        return engine::core::wrap_tensor(
            ggml_view_4d(
                ctx.ggml,
                base,
                head_dim, seq_len, heads, 1,
                base->nb[1],                                 // stride to the next time step
                static_cast<size_t>(head_dim) * base->nb[0],  // stride to the next head
                base->nb[2],
                static_cast<size_t>(feature_offset) * base->nb[0]),
            engine::core::TensorShape::from_dims({1, heads, seq_len, head_dim}),
            GGML_TYPE_F32);
    };
    auto q_heads = qkv_head_view(0);
    auto k_heads = qkv_head_view(hidden_size);
    auto v_heads = qkv_head_view(2 * hidden_size);

    auto p = ai::reshape_heads(ctx, projected_pos_emb, heads, head_dim);
    auto p_heads = ai::permute_tensor(ctx, p, {0, 2, 1, 3});

    auto q_u = ai::add_attention_bias(ctx, q_heads, weights.pos_bias_u, heads, head_dim);
    auto q_v = ai::add_attention_bias(ctx, q_heads, weights.pos_bias_v, heads, head_dim);

    auto matrix_bd_raw = engine::modules::MatMulModule().build(ctx, q_v, ai::permute_tensor(ctx, p_heads, {0, 1, 3, 2}));
    auto matrix_bd = relative_shift_view(ctx, matrix_bd_raw, seq_len);

    engine::core::TensorValue context;
    if (use_flash_attention) {
        // matrix_bd is the relative-position score term (Transformer-XL "BD" term);
        // it is exactly the "dense additive attention bias" ggml_flash_attn_ext_with_bias_mask
        // was built for — it fuses QK^T (the "AC" term via q_u), the bias add, the
        // softmax, and the AV product into one op instead of materializing the full
        // attention matrix, same pattern already used (and validated) by the shared
        // relative-attention module's specialized_flash path (see
        // common_relative_attention.cpp's use_specialized_flash_attention branch).
        auto q_flash = ai::ensure_contiguous_layout(ctx, q_u);
        auto k_flash = ai::ensure_contiguous_layout(ctx, k_heads);
        auto v_flash = ai::ensure_contiguous_layout(ctx, v_heads);
        ggml_tensor * flash = ggml_flash_attn_ext_with_bias_mask(
            ctx.ggml,
            q_flash.tensor,
            k_flash.tensor,
            v_flash.tensor,
            ai::ensure_contiguous_layout(ctx, matrix_bd).tensor,
            attention_mask.tensor,
            scale,
            0.0f,
            0.0f);
        ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
        context = engine::core::wrap_tensor(
            flash,
            engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, head_dim}),
            GGML_TYPE_F32);
    } else {
        auto matrix_ac = engine::modules::MatMulModule().build(ctx, q_u, ai::permute_tensor(ctx, k_heads, {0, 1, 3, 2}));
        auto scores = engine::core::wrap_tensor(ggml_add(ctx.ggml, matrix_ac.tensor, matrix_bd.tensor), matrix_ac.shape, GGML_TYPE_F32);
        auto attn = engine::core::wrap_tensor(
            ggml_soft_max_ext(ctx.ggml, ai::ensure_contiguous_layout(ctx, scores).tensor, attention_mask.tensor, scale, 0.0f),
            scores.shape,
            GGML_TYPE_F32);
        context = engine::modules::MatMulModule().build(ctx, attn, v_heads);
        context = ai::permute_tensor(ctx, context, {0, 2, 1, 3});
    }
    context = ai::ensure_contiguous_layout(ctx, context);
    context = engine::core::reshape_tensor(ctx, context, engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], hidden_size}));
    auto attn_output = engine::modules::LinearModule({hidden_size, hidden_size, false}).build(ctx, context, {weights.self_attn.out_weight, std::nullopt});

    x = engine::core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, attn_output.tensor), x.shape, GGML_TYPE_F32);

    auto conv_input = engine::modules::LayerNormModule({hidden_size, 1.0e-5f, true, true}).build(ctx, x, weights.norm_conv);
    auto conv = build_fastconformer_conv_module(ctx, conv_input, weights, keep_mask, conv_kernel);
    x = engine::core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, conv.tensor), x.shape, GGML_TYPE_F32);

    auto ff2_input = engine::modules::LayerNormModule({hidden_size, 1.0e-5f, true, true}).build(ctx, x, weights.norm_ff2);
    auto ff2 = engine::modules::LinearModule({hidden_size, intermediate_size, false}).build(ctx, ff2_input, weights.ff2_linear1);
    ff2 = engine::modules::SiluModule().build(ctx, ff2);
    ff2 = engine::modules::LinearModule({intermediate_size, hidden_size, false}).build(ctx, ff2, weights.ff2_linear2);
    x = engine::core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, ff2.tensor), x.shape, GGML_TYPE_F32);

    return engine::modules::LayerNormModule({hidden_size, 1.0e-5f, true, true}).build(ctx, x, weights.norm_out);
}

struct ParakeetEncoderRuntime::Graph {
    int64_t input_frames = 0;
    int64_t feature_dim = 0;
    int64_t encoded_frames = 0;
    int64_t hidden = 0;
    int64_t decoder_hidden = 0;
    ggml_backend_t backend = nullptr;
    ggml_context * ggml = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_gallocr_t pos_gallocr = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cgraph * pos_graph = nullptr;
    engine::core::TensorValue input;
    engine::core::TensorValue mask1;
    engine::core::TensorValue mask2;
    engine::core::TensorValue mask3;
    engine::core::TensorValue keep_mask;
    engine::core::TensorValue attention_mask;
    engine::core::TensorValue pos_emb;
    engine::core::TensorValue xscale;
    std::vector<engine::core::TensorValue> projected_pos_emb;
    std::vector<engine::core::TensorValue> projected_pos_emb_computed;
    engine::core::TensorValue output;

    ~Graph() {
        if (backend != nullptr) {
            engine::core::release_backend_graph_resources(backend, graph);
            engine::core::release_backend_graph_resources(backend, pos_graph);
        }
        if (pos_gallocr != nullptr) {
            ggml_gallocr_free(pos_gallocr);
        }
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
        if (ggml != nullptr) {
            ggml_free(ggml);
        }
    }
};

ParakeetEncoderRuntime::ParakeetEncoderRuntime(
    std::shared_ptr<const ParakeetTDTAssets> assets,
    std::shared_ptr<const ParakeetWeights> weights,
    engine::core::ExecutionContext & execution_context,
    size_t graph_arena_bytes,
    bool use_flash_attention)
    : assets_(std::move(assets)),
      weights_(std::move(weights)),
      execution_context_(&execution_context),
      graph_arena_bytes_(graph_arena_bytes),
      use_flash_attention_(use_flash_attention) {
    if (assets_ == nullptr || weights_ == nullptr) {
        throw std::runtime_error("Parakeet TDT encoder requires assets and weights");
    }
}

ParakeetEncoderRuntime::~ParakeetEncoderRuntime() = default;

const std::vector<float> & ParakeetEncoderRuntime::relative_positional_encoding(int64_t frames) {
    auto cached = relative_positional_encoding_cache_.find(frames);
    if (cached != relative_positional_encoding_cache_.end()) {
        return cached->second;
    }
    auto inserted = relative_positional_encoding_cache_.emplace(
        frames,
        make_relative_positional_encoding(assets_->config.encoder.hidden_size, frames, assets_->config.encoder.max_position_embeddings));
    return inserted.first->second;
}

void ParakeetEncoderRuntime::ensure_graph(int64_t input_frames, int64_t feature_dim) {
    if (input_frames <= 0 || feature_dim <= 0) {
        throw std::runtime_error("Parakeet TDT encoder graph requires positive input shape");
    }
    // A cached graph is only reused if it is not much bigger than the request.
    //
    // The graph runs at its built capacity no matter how short the real audio
    // is — encode() zero-pads up to it — so an oversized cached graph is paid
    // for in full on every call. Measured on this encoder: a 7.4s clip costs
    // 1018 ms on a matched graph and 10928 ms on a 60s-capacity one, while
    // rebuilding costs ~400 ms once (dominated by the 24 positional
    // projections; the allocation itself is ~0.4 ms). Rebuilding therefore wins
    // outright whenever the mismatch is more than a few percent, and it wins by
    // more with every subsequent call at the new size.
    //
    // The tolerance keeps the common case — a stream of clips whose lengths
    // wobble slightly — from rebuilding on every call, while capping the wasted
    // compute at roughly the same fraction.
    constexpr double kMaxGraphOversizeRatio = 1.10;
    const bool capacity_usable =
        graph_ != nullptr &&
        graph_->input_frames >= input_frames &&
        static_cast<double>(graph_->input_frames) <=
            kMaxGraphOversizeRatio * static_cast<double>(input_frames);
    if (capacity_usable &&
        graph_->backend == execution_context_->backend() &&
        graph_->feature_dim == feature_dim) {
        debug::timing_log_scalar("parakeet_tdt.encoder.graph_rebuild_ms", 0.0);
        debug::trace_log_scalar("parakeet_tdt.encoder.graph_cache_hit", true);
        return;
    }

    const auto build_start = Clock::now();
    const auto & config = assets_->config;
    const auto & enc = config.encoder;
    const auto & enc_weights = weights_->encoder;
    const int64_t k = enc.subsampling_kernel;
    const int64_t s = enc.subsampling_stride;
    // NeMo uses Conv2D with pad=1, kernel=3, stride=2 → ceil(input/2)
    auto conv_out = [k,s](int64_t in) { return (in + 2 - k) / s + 1; };
    const int64_t stage1_frames = conv_out(input_frames);
    const int64_t stage2_frames = conv_out(stage1_frames);
    const int64_t stage3_frames = conv_out(stage2_frames);
    const int64_t stage1_features = conv_out(feature_dim);
    const int64_t stage2_features = conv_out(stage1_features);
    const int64_t stage3_features = conv_out(stage2_features);
    if (stage3_features * enc.subsampling_channels != 4096) {
        throw std::runtime_error("Parakeet TDT encoder subsampling feature shape mismatch");
    }

    auto graph = std::make_unique<Graph>();
    graph->input_frames = input_frames;
    graph->feature_dim = feature_dim;
    graph->encoded_frames = stage3_frames;
    graph->hidden = enc.hidden_size;
    graph->decoder_hidden = enc.hidden_size;  // encoder outputs raw 1024-dim
    graph->backend = execution_context_->backend();
    ggml_init_params params{graph_arena_bytes_, nullptr, true};
    graph->ggml = ggml_init(params);
    if (graph->ggml == nullptr) {
        throw std::runtime_error("Failed to initialize Parakeet TDT encoder graph context");
    }

    engine::core::ModuleBuildContext ctx{graph->ggml, "parakeet_tdt.encoder", execution_context_->backend_type()};
    graph->input = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, input_frames, feature_dim}));
    ggml_set_input(graph->input.tensor);
    graph->mask1 = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({1, stage1_frames}));
    graph->mask2 = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({1, stage2_frames}));
    graph->mask3 = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({1, stage3_frames}));
    graph->keep_mask = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({1, stage3_frames}));
    graph->attention_mask = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({stage3_frames, stage3_frames}));
    graph->pos_emb = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, 2 * stage3_frames - 1, enc.hidden_size}));
    graph->xscale = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, 1, 1}));
    for (auto * tensor : {graph->mask1.tensor, graph->mask2.tensor, graph->mask3.tensor,
                          graph->keep_mask.tensor, graph->attention_mask.tensor,
                          graph->pos_emb.tensor, graph->xscale.tensor}) {
        ggml_set_input(tensor);
    }

    auto x = engine::core::reshape_tensor(ctx, graph->input, engine::core::TensorShape::from_dims({1, 1, input_frames, feature_dim}));
    // NeMo uses Conv2D with pad=1 (not causal). Output matches conv_out formula.
    x = engine::modules::Conv2dModule({1, enc.subsampling_channels, k, k, static_cast<int>(s), static_cast<int>(s), 1, 1, 1, 1, true})
            .build(ctx, x, enc_weights.subsampling.conv_in);
    x = engine::modules::ReluModule().build(ctx, engine::modules::TimeMask4dModule().build(ctx, x, graph->mask1));

    x = engine::modules::DepthwiseConv2dModule({enc.subsampling_channels, k, k, static_cast<int>(s), static_cast<int>(s), 1, 1, 1, 1, true})
            .build(ctx, x, {enc_weights.subsampling.layers[0].depthwise_weight, enc_weights.subsampling.layers[0].depthwise_bias});
    x = engine::modules::TimeMask4dModule().build(ctx, x, graph->mask2);
    x = engine::modules::Conv2dModule({enc.subsampling_channels, enc.subsampling_channels, 1, 1, 1, 1, 0, 0, 1, 1, true})
            .build(ctx, x, enc_weights.subsampling.layers[0].pointwise);
    x = engine::modules::ReluModule().build(ctx, engine::modules::TimeMask4dModule().build(ctx, x, graph->mask2));

    x = engine::modules::DepthwiseConv2dModule({enc.subsampling_channels, k, k, static_cast<int>(s), static_cast<int>(s), 1, 1, 1, 1, true})
            .build(ctx, x, {enc_weights.subsampling.layers[1].depthwise_weight, enc_weights.subsampling.layers[1].depthwise_bias});
    x = engine::modules::TimeMask4dModule().build(ctx, x, graph->mask3);
    x = engine::modules::Conv2dModule({enc.subsampling_channels, enc.subsampling_channels, 1, 1, 1, 1, 0, 0, 1, 1, true})
            .build(ctx, x, enc_weights.subsampling.layers[1].pointwise);
    x = engine::modules::ReluModule().build(ctx, engine::modules::TimeMask4dModule().build(ctx, x, graph->mask3));

    x = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    x = engine::core::wrap_tensor(ggml_cont(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);
    x = engine::core::reshape_tensor(ctx, x, engine::core::TensorShape::from_dims({1, stage3_frames, enc.subsampling_channels * stage3_features}));
    // xscaling (multiply by sqrt(d_model), which Parakeet's RelPositionalEncoding
    // does) is folded into this projection's weight and bias at load time — see
    // load_subsampling in weights.cpp.
    x = engine::modules::LinearModule({enc.subsampling_channels * stage3_features, enc.hidden_size, true}).build(ctx, x, enc_weights.subsampling.linear);

    graph->projected_pos_emb.reserve(static_cast<size_t>(enc.layers));
    graph->projected_pos_emb_computed.reserve(static_cast<size_t>(enc.layers));
    for (int64_t layer = 0; layer < enc.layers; ++layer) {
        graph->projected_pos_emb.push_back(engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({1, 2 * stage3_frames - 1, enc.hidden_size})));
        ggml_set_input(graph->projected_pos_emb.back().tensor);
        ggml_set_output(graph->projected_pos_emb.back().tensor);
        graph->projected_pos_emb_computed.push_back(
            engine::modules::LinearModule({enc.hidden_size, enc.hidden_size, false}).build(
                ctx,
                graph->pos_emb,
                {enc_weights.layers[static_cast<size_t>(layer)].pos_weight, std::nullopt}));
        ggml_set_output(graph->projected_pos_emb_computed.back().tensor);
    }

    for (int64_t layer = 0; layer < enc.layers; ++layer) {
        x = build_encoder_layer(
            ctx,
            x,
            graph->attention_mask,
            graph->keep_mask,
            graph->projected_pos_emb[static_cast<size_t>(layer)],
            enc_weights.layers[static_cast<size_t>(layer)],
            enc.hidden_size,
            enc.intermediate_size,
            enc.heads,
            enc.conv_kernel,
            use_flash_attention_);
    }

    // No projector here — encoder outputs raw 1024-dim. Projection to 640 happens in decoder joint_enc.
    graph->output = x;
    ggml_set_output(graph->output.tensor);

    graph->pos_graph = ggml_new_graph_custom(graph->ggml, 4096, false);
    for (const auto & projected : graph->projected_pos_emb_computed) {
        ggml_build_forward_expand(graph->pos_graph, projected.tensor);
    }
    graph->graph = ggml_new_graph_custom(graph->ggml, kEncoderGraphNodes, false);
    ggml_build_forward_expand(graph->graph, graph->output.tensor);

    // Elides redundant nodes (broadcast repeats folded into the consuming op,
    // reshape/view no-ops, etc.) from the graph once at build time, before
    // allocation — the cached graph is reused across every subsequent encode()
    // call at this input length, so this cost is amortized to ~zero and every
    // reused call benefits from the smaller node count.
    const auto opt_backend = graph_optimizer_backend_for(execution_context_->backend_type());
    const auto pos_opt_report = engine::runtime::optimize_graph(*graph->pos_graph, opt_backend);
    const auto opt_report = engine::runtime::optimize_graph(*graph->graph, opt_backend);
    debug::trace_log_scalar("parakeet_tdt.encoder.graph_optimizer.nodes_before", opt_report.nodes_before);
    debug::trace_log_scalar("parakeet_tdt.encoder.graph_optimizer.nodes_after", opt_report.nodes_after);
    (void)pos_opt_report;

    const auto alloc_start = Clock::now();
    graph->gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(graph->backend));
    if (graph->gallocr == nullptr ||
        !ggml_gallocr_reserve(graph->gallocr, graph->graph) ||
        !ggml_gallocr_alloc_graph(graph->gallocr, graph->graph)) {
        throw std::runtime_error("Failed to allocate Parakeet TDT encoder graph tensors");
    }
    graph->pos_gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(graph->backend));
    if (graph->pos_gallocr == nullptr ||
        !ggml_gallocr_reserve(graph->pos_gallocr, graph->pos_graph) ||
        !ggml_gallocr_alloc_graph(graph->pos_gallocr, graph->pos_graph)) {
        throw std::runtime_error("Failed to allocate Parakeet TDT encoder position graph tensors");
    }
    debug::timing_log_scalar("parakeet_tdt.encoder.graph_alloc_ms", engine::debug::elapsed_ms(alloc_start, Clock::now()));

    const auto pos_upload_start = Clock::now();
    engine::core::write_tensor_f32(
        graph->pos_emb,
        relative_positional_encoding(stage3_frames));
    debug::timing_log_scalar("parakeet_tdt.encoder.pos_upload_ms", engine::debug::elapsed_ms(pos_upload_start, Clock::now()));

    const auto pos_compute_start = Clock::now();
    const auto pos_status = engine::core::compute_backend_graph(execution_context_->backend(), graph->pos_graph, nullptr, "Parakeet TDT encoder pos");
    if (pos_status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Parakeet TDT encoder position graph compute failed");
    }
    debug::timing_log_scalar("parakeet_tdt.encoder.pos_compute_ms", engine::debug::elapsed_ms(pos_compute_start, Clock::now()));

    const auto pos_copy_start = Clock::now();
    for (size_t i = 0; i < graph->projected_pos_emb.size(); ++i) {
        ggml_backend_tensor_copy(graph->projected_pos_emb_computed[i].tensor, graph->projected_pos_emb[i].tensor);
    }
    debug::timing_log_scalar("parakeet_tdt.encoder.pos_copy_ms", engine::debug::elapsed_ms(pos_copy_start, Clock::now()));

    graph_ = std::move(graph);
    const double build_ms = engine::debug::elapsed_ms(build_start, Clock::now());
    debug::timing_log_scalar("parakeet_tdt.encoder.graph_build_ms", build_ms);
    debug::timing_log_scalar("parakeet_tdt.encoder.graph_rebuild_ms", build_ms);
    debug::trace_log_scalar("parakeet_tdt.encoder.graph_cache_hit", false);
    debug::trace_log_scalar("parakeet_tdt.encoder.graph_input_frames", input_frames);
    debug::trace_log_scalar("parakeet_tdt.encoder.graph_encoded_frames", stage3_frames);
}

void ParakeetEncoderRuntime::prepare_capacity(int64_t input_frames, int64_t feature_dim) {
    ensure_graph(input_frames, feature_dim);
}

void ParakeetEncoderRuntime::release_offline_graph() {
    graph_.reset();
}

ParakeetEncodedAudio ParakeetEncoderRuntime::encode(
    const ParakeetFrontendFeatures & features) {
    if (features.frames <= 0 || features.feature_dim <= 0) {
        throw std::runtime_error("Parakeet TDT encoder requires positive frontend shape");
    }
    const auto wall_start = Clock::now();
    ensure_graph(features.frames, features.feature_dim);
    auto & graph = *graph_;

    const auto & enc = assets_->config.encoder;
    std::vector<float> input_scratch(static_cast<size_t>(graph.input_frames * graph.feature_dim), 0.0f);
    for (int64_t t = 0; t < features.frames; ++t) {
        for (int64_t f = 0; f < features.feature_dim; ++f) {
            input_scratch[static_cast<size_t>(t * graph.feature_dim + f)] =
                features.values[static_cast<size_t>(t * features.feature_dim + f)];
        }
    }
    engine::core::write_tensor_f32(graph.input, input_scratch);

    auto conv_out = [&enc](int64_t in) { return (in + 2 - enc.subsampling_kernel) / enc.subsampling_stride + 1; };
    const int64_t valid1 = std::min<int64_t>(graph.mask1.shape.dims[1], conv_out(features.valid_frames));
    const int64_t valid2 = std::min<int64_t>(graph.mask2.shape.dims[1], conv_out(valid1));
    const int64_t valid3 = std::min<int64_t>(graph.encoded_frames, conv_out(valid2));

    engine::modules::fill_asr_keep_mask(mask_scratch_, graph.mask1.shape.dims[1], valid1);
    engine::core::write_tensor_i32(graph.mask1, mask_scratch_);
    engine::modules::fill_asr_keep_mask(mask_scratch_, graph.mask2.shape.dims[1], valid2);
    engine::core::write_tensor_i32(graph.mask2, mask_scratch_);
    engine::modules::fill_asr_keep_mask(mask_scratch_, graph.mask3.shape.dims[1], valid3);
    engine::core::write_tensor_i32(graph.mask3, mask_scratch_);
    engine::modules::fill_asr_keep_mask(mask_scratch_, graph.encoded_frames, valid3);
    engine::core::write_tensor_i32(graph.keep_mask, mask_scratch_);

    // Attention is fully bidirectional across the real frames, but the padded
    // tail must be masked out of it.
    //
    // ensure_graph() reuses a graph whose capacity merely *exceeds* the request,
    // and the input write above zero-pads up to that capacity. Zero input does
    // not stay zero: every subsampling conv has a bias, so the padded tail
    // carries nonzero garbage into the encoder stack. An all-zero attention
    // mask lets the real frames attend to that garbage, and because softmax
    // normalizes across keys, it silently rescales every real frame's attention
    // — feeding a 60s-capacity graph a 7.4s clip drops a whole sentence from the
    // transcription.
    //
    // Mask by key column only, never by query row: that leaves the padded query
    // rows (whose outputs are discarded below anyway) with a full set of valid
    // keys, so no softmax row degenerates to all -inf and produces NaN.
    const size_t frames = static_cast<size_t>(graph.encoded_frames);
    attention_mask_scratch_.assign(frames * frames, 0.0f);
    if (valid3 > 0 && valid3 < graph.encoded_frames) {
        for (size_t query = 0; query < frames; ++query) {
            std::fill(
                attention_mask_scratch_.begin() + static_cast<std::ptrdiff_t>(query * frames + valid3),
                attention_mask_scratch_.begin() + static_cast<std::ptrdiff_t>((query + 1) * frames),
                -std::numeric_limits<float>::infinity());
        }
    }
    engine::core::write_tensor_f32(graph.attention_mask, attention_mask_scratch_);

    engine::core::set_backend_threads(execution_context_->backend(), execution_context_->config().threads);
    const auto compute_start = Clock::now();
    const auto status = engine::core::compute_backend_graph(execution_context_->backend(), graph.graph, nullptr, "Parakeet TDT encoder");
    ggml_backend_synchronize(execution_context_->backend());
    debug::timing_log_scalar("parakeet_tdt.encoder.graph.compute_ms", engine::debug::elapsed_ms(compute_start, Clock::now()));

    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Parakeet TDT encoder graph compute failed");
    }

    output_scratch_.resize(static_cast<size_t>(graph.encoded_frames * graph.decoder_hidden));
    engine::core::read_tensor_f32_into(graph.output.tensor, output_scratch_);

    if (valid3 < graph.encoded_frames) {
        for (int64_t row = valid3; row < graph.encoded_frames; ++row) {
            std::fill_n(
                output_scratch_.begin() + static_cast<std::ptrdiff_t>(row * graph.decoder_hidden),
                static_cast<std::ptrdiff_t>(graph.decoder_hidden),
                0.0f);
        }
    }

    ParakeetEncodedAudio out;
    out.values = output_scratch_;
    out.frames = graph.encoded_frames;
    out.valid_frames = valid3;
    out.hidden_size = graph.decoder_hidden;
    debug::timing_log_scalar("parakeet_tdt.encoder_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    debug::trace_log_scalar("parakeet_tdt.encoder.valid_frames", valid3);
    return out;
}

}  // namespace engine::community_models::parakeet_tdt
