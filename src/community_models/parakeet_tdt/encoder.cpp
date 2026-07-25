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

#include "../../framework/modules/attention/attention_internal.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::community_models::parakeet_tdt {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kEncoderGraphNodes = 2097152;

int64_t causal_conv_output_dim(int64_t input, int64_t kernel, int64_t stride, bool streaming) {
    const int64_t left = streaming ? kernel - stride : kernel - 1;
    const int64_t right = streaming ? 0 : stride - 1;
    return (input + left + right - kernel) / stride + 1;
}

engine::core::TensorValue pad_causal_2d(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t kernel,
    int64_t stride,
    bool streaming) {
    const int64_t freq_left = kernel - 1;
    const int64_t freq_right = stride - 1;
    const int64_t time_left = streaming ? kernel - stride : kernel - 1;
    const int64_t time_right = streaming ? 0 : stride - 1;
    const auto out_shape = engine::core::TensorShape::from_dims({
        input.shape.dims[0],
        input.shape.dims[1],
        input.shape.dims[2] + time_left + time_right,
        input.shape.dims[3] + freq_left + freq_right,
    });
    return engine::core::wrap_tensor(
        ggml_pad_ext(
            ctx.ggml,
            input.tensor,
            static_cast<int>(freq_left),
            static_cast<int>(freq_right),
            static_cast<int>(time_left),
            static_cast<int>(time_right),
            0, 0, 0, 0),
        out_shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue pad_causal_1d(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t left) {
    if (left <= 0) { return input; }
    return engine::core::wrap_tensor(
        ggml_pad_ext(ctx.ggml, input.tensor, static_cast<int>(left), 0, 0, 0, 0, 0, 0, 0),
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], input.shape.dims[2] + left}),
        GGML_TYPE_F32);
}

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
    auto x = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input_btc);
    x = engine::modules::LinearModule({input_btc.shape.dims[2], 2 * input_btc.shape.dims[2], false}).build(
        ctx,
        engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x),
        weights.conv_pw1);
    x = engine::modules::GLUModule().build(ctx, x);
    x = engine::modules::MaskingModule().build(ctx, x, keep_mask);
    x = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);

    const int64_t d_model = x.shape.dims[1];
    x = pad_symmetric_1d(ctx, x, conv_kernel);
    x = engine::modules::DepthwiseConv1dModule({d_model, conv_kernel, 1, 0, 1, false})
            .build(ctx, x, {weights.conv_dw_weight, weights.conv_dw_bias});
    x = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = engine::modules::SiluModule().build(ctx, x);
    return engine::modules::LinearModule({d_model, d_model, false}).build(ctx, x, weights.conv_pw2);
}

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
    int64_t conv_kernel) {
    namespace ai = engine::modules::attention::internal;

    auto x_norm = engine::modules::LayerNormModule({hidden_size, 1.0e-5f, true, true}).build(ctx, input, weights.norm_ff1);
    auto ff1 = engine::modules::LinearModule({hidden_size, intermediate_size, false}).build(ctx, x_norm, weights.ff1_linear1);
    ff1 = engine::modules::SiluModule().build(ctx, ff1);
    ff1 = engine::modules::LinearModule({intermediate_size, hidden_size, false}).build(ctx, ff1, weights.ff1_linear2);
    ff1 = engine::core::wrap_tensor(ggml_scale(ctx.ggml, ff1.tensor, 0.5f), ff1.shape, GGML_TYPE_F32);
    auto x = engine::core::wrap_tensor(ggml_add(ctx.ggml, input.tensor, ff1.tensor), input.shape, GGML_TYPE_F32);

    auto attn_input = engine::modules::LayerNormModule({hidden_size, 1.0e-5f, true, true}).build(ctx, x, weights.norm_attn);

    const int64_t head_dim = hidden_size / heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto q = engine::modules::LinearModule({hidden_size, hidden_size, false}).build(ctx, attn_input, {weights.self_attn.q_weight, std::nullopt});
    auto k = engine::modules::LinearModule({hidden_size, hidden_size, false}).build(ctx, attn_input, {weights.self_attn.k_weight, std::nullopt});
    auto v = engine::modules::LinearModule({hidden_size, hidden_size, false}).build(ctx, attn_input, {weights.self_attn.v_weight, std::nullopt});

    q = ai::reshape_heads(ctx, q, heads, head_dim);
    k = ai::reshape_heads(ctx, k, heads, head_dim);
    v = ai::reshape_heads(ctx, v, heads, head_dim);

    auto q_heads = ai::permute_tensor(ctx, q, {0, 2, 1, 3});
    auto k_heads = ai::permute_tensor(ctx, k, {0, 2, 1, 3});
    auto v_heads = ai::permute_tensor(ctx, v, {0, 2, 1, 3});

    auto p = ai::reshape_heads(ctx, projected_pos_emb, heads, head_dim);
    auto p_heads = ai::permute_tensor(ctx, p, {0, 2, 1, 3});

    auto q_u = ai::add_attention_bias(ctx, q_heads, weights.pos_bias_u, heads, head_dim);
    auto q_v = ai::add_attention_bias(ctx, q_heads, weights.pos_bias_v, heads, head_dim);

    auto matrix_ac = engine::modules::MatMulModule().build(ctx, q_u, ai::permute_tensor(ctx, k_heads, {0, 1, 3, 2}));
    auto matrix_bd = engine::modules::MatMulModule().build(ctx, q_v, ai::permute_tensor(ctx, p_heads, {0, 1, 3, 2}));
    matrix_bd = ai::relative_shift(ctx, matrix_bd);
    matrix_bd = engine::modules::SliceModule({3, 0, k_heads.shape.dims[2]}).build(ctx, matrix_bd);

    auto scores = engine::core::wrap_tensor(ggml_add(ctx.ggml, matrix_ac.tensor, matrix_bd.tensor), matrix_ac.shape, GGML_TYPE_F32);
    auto attn = engine::core::wrap_tensor(
        ggml_soft_max_ext(ctx.ggml, ai::ensure_contiguous_layout(ctx, scores).tensor, attention_mask.tensor, scale, 0.0f),
        scores.shape,
        GGML_TYPE_F32);
    auto context = engine::modules::MatMulModule().build(ctx, attn, v_heads);
    context = ai::permute_tensor(ctx, context, {0, 2, 1, 3});
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
    ff2 = engine::core::wrap_tensor(ggml_scale(ctx.ggml, ff2.tensor, 0.5f), ff2.shape, GGML_TYPE_F32);
    x = engine::core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, ff2.tensor), x.shape, GGML_TYPE_F32);

    return engine::modules::LayerNormModule({hidden_size, 1.0e-5f, true, true}).build(ctx, x, weights.norm_out);
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

}  // namespace

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
    size_t graph_arena_bytes)
    : assets_(std::move(assets)),
      weights_(std::move(weights)),
      execution_context_(&execution_context),
      graph_arena_bytes_(graph_arena_bytes) {
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
    if (graph_ != nullptr &&
        graph_->backend == execution_context_->backend() &&
        graph_->input_frames >= input_frames &&
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
    x = engine::modules::LinearModule({enc.subsampling_channels * stage3_features, enc.hidden_size, true}).build(ctx, x, enc_weights.subsampling.linear);

    // xscaling: multiply by sqrt(d_model) — Parakeet's RelPositionalEncoding does this
    const float xscale_value = std::sqrt(static_cast<float>(enc.hidden_size));
    x = engine::core::wrap_tensor(ggml_scale(ctx.ggml, x.tensor, xscale_value), x.shape, GGML_TYPE_F32);

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
            enc.conv_kernel);
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

ParakeetEncoderStreamState ParakeetEncoderRuntime::make_stream_state() const {
    ParakeetEncoderStreamState state;
    return state;
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

    // Full bidirectional attention: all frames see all other frames (no mask)
    attention_mask_scratch_.resize(static_cast<size_t>(graph.encoded_frames * graph.encoded_frames));
    std::fill(attention_mask_scratch_.begin(), attention_mask_scratch_.end(), 0.0f);
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
