// granite_speech5 Conformer encoder: weight binding + graph execution.
//
// Math reference: transformers 4.57.0 modeling_granite_speech5.py
// (GraniteSpeechCTCEncoder / GraniteSpeechConformerBlock /
// GraniteSpeech5ConformerAttention / GraniteSpeech5ConformerConvModule).
//
// - FFN: LN(1e-5) -> Linear(1024->4096, bias) -> SiLU -> Linear(4096->1024,
//   bias); block applies x + 0.5 * FFN(x).
// - Attention: LN(1e-5) over the ORIGINAL full-length sequence first (the
//   reference applies pre_norm before zero-padding the final block; here no
//   padding is materialized at all -- see below), to_q (no bias) and a packed
//   to_kv (no bias) split into K (first 1024 rows) then V. Scores are
//   materialized per 200-frame block only:
//       S[i,j] = (Q_i . K_j + Q_i . R[i-j]) / sqrt(128),
//     R[k] = rel_pos_emb[clamp(k, -200, 200) + 512] (|i-j| < 200 inside a
//     block, so the clamp is a no-op and every block reuses the same table).
//     The reference pads a short final block with zeros and masks its invalid
//     keys; the valid r x r submatrix is byte-identical without padding, the
//     mask over nonexistent keys is vacuous, and the padded query rows are
//     cropped. This port therefore runs the final block short (length r) with
//     no mask tensor, which is exactly equal on all T real frames.
// - Convolution: LN(1e-5) -> pointwise up Conv1d(1024->4096, k=1, bias) ->
//   GLU (first half * sigmoid(second half), matching torch nn.GLU(dim=1)) ->
//   depthwise Conv1d(2048, k=15, groups=2048, no bias) with symmetric 7+7
//   zero pad -> BatchNorm1d inference affine (eps 1e-5, evaluated as its own
//   op, NOT folded into the convolution weights) -> SiLU -> pointwise down
//   Conv1d(2048->1024, k=1, bias) -> residual. Runs over the original
//   contiguous time sequence across attention block boundaries.
// - Block: x + 0.5*FFN1 -> + attn -> + conv -> + 0.5*FFN2 -> post LN(1e-5).
// - After layer index 7: h += out_mid(softmax(out(h))) per real frame.

#include "engine/community_models/granite_speech5/encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::granite_speech5 {
namespace {

constexpr size_t kEncoderGraphBaseNodes = 8192;
constexpr size_t kEncoderGraphNodesPerBlock = 256;
constexpr size_t kEncoderWeightContextBytes = 4ull * 1024ull * 1024ull;

struct EncoderFfnWeights {
    modules::NormWeights pre_norm;
    modules::LinearWeights up;
    modules::LinearWeights down;
};

struct EncoderAttnWeights {
    modules::NormWeights pre_norm;
    core::TensorValue to_q;
    core::TensorValue to_kv;
    modules::LinearWeights to_out;
    core::TensorValue rel_pos_emb;  // [1025, 128] per-layer Shaw table
};

struct EncoderConvWeights {
    modules::NormWeights norm;
    modules::LinearWeights up;    // from Conv1d k=1 weight, reshaped to [4096, 1024]
    modules::LinearWeights down;  // from Conv1d k=1 weight, reshaped to [1024, 2048]
    core::TensorValue depthwise_weight;  // [2048, 1, 15], no bias
    modules::BatchNorm1dEvalWeights batch_norm;  // eval affine, NOT folded into conv
};

struct EncoderLayerWeights {
    EncoderFfnWeights ff1;
    EncoderAttnWeights attn;
    EncoderConvWeights conv;
    EncoderFfnWeights ff2;
    modules::NormWeights post_norm;
};

struct EncoderWeights {
    modules::LinearWeights input_linear;
    std::vector<EncoderLayerWeights> layers;
    modules::LinearWeights out;
    modules::LinearWeights out_mid;
};

EncoderFfnWeights load_ffn_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type) {
    EncoderFfnWeights weights;
    weights.pre_norm = modules::binding::norm_from_source(store, source, prefix + "pre_norm", 1024);
    weights.up = modules::binding::linear_from_source(store, source, prefix + "up_proj", storage_type, 4096, 1024, true);
    weights.down = modules::binding::linear_from_source(store, source, prefix + "down_proj", storage_type, 1024, 4096, true);
    return weights;
}

modules::BatchNorm1dEvalWeights load_conv_batch_norm(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix) {
    // Inference-time affine from the checkpoint running statistics:
    //   y = gamma * (x - mean) / sqrt(var + eps) + beta
    //     = x * scale + shift,
    // with the HF nn.BatchNorm1d default eps. The convolution weights are
    // used verbatim; nothing is folded.
    const auto gamma = source.require_f32(prefix + "weight", {2048});
    const auto beta = source.require_f32(prefix + "bias", {2048});
    const auto mean = source.require_f32(prefix + "running_mean", {2048});
    const auto var = source.require_f32(prefix + "running_var", {2048});
    std::vector<float> scale(2048);
    std::vector<float> shift(2048);
    for (size_t c = 0; c < 2048; ++c) {
        const float inv = gamma[c] / std::sqrt(var[c] + kGraniteSpeech5EncoderBatchNormEps);
        scale[c] = inv;
        shift[c] = beta[c] - mean[c] * inv;
    }
    modules::BatchNorm1dEvalWeights weights;
    weights.scale = store.make_from_f32(
        core::TensorShape::from_dims({2048}), assets::TensorStorageType::F32, std::move(scale));
    weights.bias = store.make_from_f32(
        core::TensorShape::from_dims({2048}), assets::TensorStorageType::F32, std::move(shift));
    return weights;
}

EncoderLayerWeights load_layer_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    int64_t layer_idx,
    assets::TensorStorageType storage_type) {
    EncoderLayerWeights weights;
    const std::string prefix = "encoder.layers." + std::to_string(layer_idx) + ".";
    weights.ff1 = load_ffn_weights(store, source, prefix + "ff1.", storage_type);

    weights.attn.pre_norm = modules::binding::norm_from_source(store, source, prefix + "attn.pre_norm", 1024);
    weights.attn.to_q = store.load_tensor(source, prefix + "attn.to_q.weight", storage_type, {1024, 1024});
    weights.attn.to_kv = store.load_tensor(source, prefix + "attn.to_kv.weight", storage_type, {2048, 1024});
    weights.attn.to_out = modules::binding::linear_from_source(store, source, prefix + "attn.to_out", storage_type, 1024, 1024, true);
    weights.attn.rel_pos_emb =
        store.load_tensor(source, prefix + "attn.rel_pos_emb.weight", storage_type, {1025, 128});

    weights.conv.norm = modules::binding::norm_from_source(store, source, prefix + "conv.norm", 1024);
    weights.conv.up = {
        store.load_tensor_as_shape(
            source,
            prefix + "conv.up_conv.weight",
            storage_type,
            {4096, 1024, 1},
            core::TensorShape::from_dims({4096, 1024})),
        store.load_f32_tensor(source, prefix + "conv.up_conv.bias", {4096}),
    };
    weights.conv.depthwise_weight =
        store.load_tensor(source, prefix + "conv.depth_conv.conv.weight", storage_type, {2048, 1, 15});
    weights.conv.batch_norm = load_conv_batch_norm(store, source, prefix + "conv.batch_norm.");
    weights.conv.down = {
        store.load_tensor_as_shape(
            source,
            prefix + "conv.down_conv.weight",
            storage_type,
            {1024, 2048, 1},
            core::TensorShape::from_dims({1024, 2048})),
        store.load_f32_tensor(source, prefix + "conv.down_conv.bias", {1024}),
    };

    weights.ff2 = load_ffn_weights(store, source, prefix + "ff2.", storage_type);
    weights.post_norm = modules::binding::norm_from_source(store, source, prefix + "post_norm", 1024);
    return weights;
}

EncoderWeights load_encoder_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    assets::TensorStorageType storage_type) {
    EncoderWeights weights;
    weights.input_linear = modules::binding::linear_from_source(store, source, "encoder.input_linear", storage_type, 1024, 160, true);
    weights.layers.reserve(16);
    for (int64_t layer = 0; layer < kGraniteSpeech5EncoderLayers; ++layer) {
        weights.layers.push_back(load_layer_weights(store, source, layer, storage_type));
    }
    weights.out = modules::binding::linear_from_source(store, source, "encoder.out", storage_type, 348, 1024, true);
    weights.out_mid = modules::binding::linear_from_source(store, source, "encoder.out_mid", storage_type, 1024, 348, true);
    store.upload();
    return weights;
}

core::TensorValue build_ffn(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const EncoderFfnWeights & weights) {
    auto h = modules::LayerNormModule({1024, kGraniteSpeech5EncoderLayerNormEps})
                 .build(ctx, input, weights.pre_norm);
    modules::FeedForwardConfig config;
    config.hidden_size = 1024;
    config.intermediate_size = 4096;
    config.activation = modules::FeedForwardActivation::Silu;
    return modules::FeedForwardModule(config).build(
        ctx, h, {weights.up.weight, weights.up.bias, weights.down.weight, weights.down.bias});
}

// Shaw relative block attention for one block of L rows starting at frame
// offset t0. q_full/k_full/v_full are the full-sequence projections of the
// pre-normed input; rel_idx is the flattened L x L column of (i - j) + 512.
core::TensorValue build_attention_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_full,
    const core::TensorValue & k_full,
    const core::TensorValue & v_full,
    const core::TensorValue & rel_idx,
    const core::TensorValue & rel_emb,
    int64_t t0,
    int64_t block_len) {
    constexpr int64_t kHeads = kGraniteSpeech5EncoderHeads;
    constexpr int64_t kHeadDim = kGraniteSpeech5EncoderHeadDim;
    const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));

    auto q_blk = modules::SliceModule({1, t0, block_len}).build(ctx, q_full);
    auto k_blk = modules::SliceModule({1, t0, block_len}).build(ctx, k_full);
    auto v_blk = modules::SliceModule({1, t0, block_len}).build(ctx, v_full);

    auto q_3d = core::wrap_tensor(
        ggml_reshape_3d(ctx.ggml, q_blk.tensor, kHeadDim, kHeads, block_len),
        core::TensorShape::from_dims({block_len, kHeads, kHeadDim}),
        GGML_TYPE_F32);
    auto k_3d = core::wrap_tensor(
        ggml_reshape_3d(ctx.ggml, k_blk.tensor, kHeadDim, kHeads, block_len),
        core::TensorShape::from_dims({block_len, kHeads, kHeadDim}),
        GGML_TYPE_F32);
    auto v_3d = core::wrap_tensor(
        ggml_reshape_3d(ctx.ggml, v_blk.tensor, kHeadDim, kHeads, block_len),
        core::TensorShape::from_dims({block_len, kHeads, kHeadDim}),
        GGML_TYPE_F32);

    auto q_perm = core::wrap_tensor(
        ggml_permute(ctx.ggml, q_3d.tensor, 0, 2, 1, 3),
        core::TensorShape::from_dims({kHeads, block_len, kHeadDim}),
        GGML_TYPE_F32);
    auto k_perm = core::wrap_tensor(
        ggml_permute(ctx.ggml, k_3d.tensor, 0, 2, 1, 3),
        core::TensorShape::from_dims({kHeads, block_len, kHeadDim}),
        GGML_TYPE_F32);
    auto k_cont = core::wrap_tensor(ggml_cont(ctx.ggml, k_perm.tensor), k_perm.shape, GGML_TYPE_F32);

    auto scores = core::wrap_tensor(
        ggml_mul_mat(ctx.ggml, k_cont.tensor, q_perm.tensor),
        core::TensorShape::from_dims({kHeads, block_len, block_len}),
        GGML_TYPE_F32);
    scores = core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, scale), scores.shape, GGML_TYPE_F32);

    // Shaw relative bias gathered from this layer's own embedding table:
    // rel[i, j, d] = emb[(i - j) + 512, d]. ggml_get_rows only accepts a
    // column index vector, so gather the flattened L x L table then reshape.
    auto rel_flat = core::wrap_tensor(
        ggml_get_rows(ctx.ggml, rel_emb.tensor, rel_idx.tensor),
        core::TensorShape::from_dims({block_len * block_len, kHeadDim}),
        GGML_TYPE_F32);
    auto rel = core::wrap_tensor(
        ggml_reshape_3d(ctx.ggml, rel_flat.tensor, kHeadDim, block_len, block_len),
        core::TensorShape::from_dims({block_len, block_len, kHeadDim}),
        GGML_TYPE_F32);
    const size_t elem_size = ggml_type_size(GGML_TYPE_F32);
    const size_t nb1 = static_cast<size_t>(kHeadDim) * elem_size;
    const size_t nb2 = static_cast<size_t>(block_len * kHeadDim) * elem_size;
    auto rel_slice = core::wrap_tensor(
        ggml_view_3d(ctx.ggml, rel.tensor, kHeadDim, block_len, block_len, nb1, nb2, 0),
        core::TensorShape::from_dims({block_len, block_len, kHeadDim}),
        GGML_TYPE_F32);
    auto rel_cont = core::wrap_tensor(
        ggml_cont(ctx.ggml, rel_slice.tensor), rel_slice.shape, GGML_TYPE_F32);
    auto pos_bias = core::wrap_tensor(
        ggml_mul_mat(ctx.ggml, rel_cont.tensor, q_3d.tensor),
        core::TensorShape::from_dims({block_len, kHeads, block_len}),
        GGML_TYPE_F32);
    pos_bias = core::wrap_tensor(ggml_scale(ctx.ggml, pos_bias.tensor, scale), pos_bias.shape, GGML_TYPE_F32);
    auto pos_perm = core::wrap_tensor(
        ggml_permute(ctx.ggml, pos_bias.tensor, 0, 2, 1, 3),
        core::TensorShape::from_dims({kHeads, block_len, block_len}),
        GGML_TYPE_F32);

    auto total = core::wrap_tensor(
        ggml_add(ctx.ggml, scores.tensor, pos_perm.tensor), scores.shape, GGML_TYPE_F32);
    auto weights = core::wrap_tensor(ggml_soft_max(ctx.ggml, total.tensor), total.shape, GGML_TYPE_F32);

    auto v_perm = core::wrap_tensor(
        ggml_permute(ctx.ggml, v_3d.tensor, 1, 2, 0, 3),
        core::TensorShape::from_dims({kHeads, kHeadDim, block_len}),
        GGML_TYPE_F32);
    auto v_cont = core::wrap_tensor(ggml_cont(ctx.ggml, v_perm.tensor), v_perm.shape, GGML_TYPE_F32);
    auto out_head = core::wrap_tensor(
        ggml_mul_mat(ctx.ggml, v_cont.tensor, weights.tensor),
        core::TensorShape::from_dims({kHeads, block_len, kHeadDim}),
        GGML_TYPE_F32);
    auto out_perm = core::wrap_tensor(
        ggml_permute(ctx.ggml, out_head.tensor, 0, 2, 1, 3),
        core::TensorShape::from_dims({block_len, kHeads, kHeadDim}),
        GGML_TYPE_F32);
    auto out_cont = core::wrap_tensor(
        ggml_cont(ctx.ggml, out_perm.tensor), out_perm.shape, GGML_TYPE_F32);
    return core::wrap_tensor(
        ggml_reshape_3d(ctx.ggml, out_cont.tensor, 1024, block_len, 1),
        core::TensorShape::from_dims({1, block_len, 1024}),
        GGML_TYPE_F32);
}

core::TensorValue build_block_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const EncoderAttnWeights & weights,
    const core::TensorValue & rel_idx_full,
    const core::TensorValue * rel_idx_short,
    int64_t num_frames) {
    auto normed =
        modules::LayerNormModule({1024, kGraniteSpeech5EncoderLayerNormEps}).build(ctx, input, weights.pre_norm);
    auto q_all = modules::LinearModule({1024, 1024, false}).build(ctx, normed, {weights.to_q, std::nullopt});
    auto kv_all = modules::LinearModule({1024, 2048, false}).build(ctx, normed, {weights.to_kv, std::nullopt});
    // Packed KV split: K first, then V (torch chunk(2, dim=-1)).
    auto k_all = core::wrap_tensor(
        ggml_cont(
            ctx.ggml,
            modules::SliceModule({2, 0, 1024}).build(ctx, kv_all).tensor),
        core::TensorShape::from_dims({1, num_frames, 1024}),
        GGML_TYPE_F32);
    auto v_all = core::wrap_tensor(
        ggml_cont(
            ctx.ggml,
            modules::SliceModule({2, 1024, 1024}).build(ctx, kv_all).tensor),
        core::TensorShape::from_dims({1, num_frames, 1024}),
        GGML_TYPE_F32);

    const auto plan = granite_speech5_encoder_blocks_for_steps(num_frames);
    std::vector<core::TensorValue> block_outputs;
    block_outputs.reserve(static_cast<size_t>(plan.num_blocks));
    for (int64_t b = 0; b < plan.num_blocks; ++b) {
        const bool is_last = (b == plan.num_blocks - 1);
        const int64_t len = is_last ? plan.last_block_len : kGraniteSpeech5EncoderContext;
        const core::TensorValue & rel_idx = (len == kGraniteSpeech5EncoderContext) ? rel_idx_full : *rel_idx_short;
        block_outputs.push_back(build_attention_block(
            ctx, q_all, k_all, v_all, rel_idx, weights.rel_pos_emb, b * kGraniteSpeech5EncoderContext, len));
    }
    core::TensorValue attn_out = block_outputs[0];
    for (size_t i = 1; i < block_outputs.size(); ++i) {
        attn_out = modules::ConcatModule({1}).build(ctx, attn_out, block_outputs[i]);
    }
    return modules::LinearModule({1024, 1024, true}).build(ctx, attn_out, weights.to_out);
}

core::TensorValue build_conv_module(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const EncoderConvWeights & weights) {
    auto h = modules::LayerNormModule({1024, kGraniteSpeech5EncoderLayerNormEps}).build(ctx, input, weights.norm);
    h = modules::LinearModule({1024, 4096, true}).build(ctx, h, weights.up);
    h = modules::GLUModule({true}).build(ctx, h);
    h = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, h);
    h = modules::DepthwiseConv1dModule({2048, 15, 1, 7, 1, false})
            .build(ctx, h, {weights.depthwise_weight, std::nullopt});
    h = modules::BatchNorm1dEvalModule({2048}).build(ctx, h, weights.batch_norm);
    h = modules::SiluModule().build(ctx, h);
    h = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, h);
    h = modules::LinearModule({2048, 1024, true}).build(ctx, h, weights.down);
    return core::wrap_tensor(
        ggml_add(ctx.ggml, input.tensor, h.tensor), input.shape, GGML_TYPE_F32);
}

core::TensorValue build_conformer_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const EncoderLayerWeights & weights,
    const core::TensorValue & rel_idx_full,
    const core::TensorValue * rel_idx_short,
    int64_t num_frames) {
    auto ffn1 = build_ffn(ctx, input, weights.ff1);
    auto scaled1 = core::wrap_tensor(
        ggml_scale(ctx.ggml, ffn1.tensor, kGraniteSpeech5MacaronScale), ffn1.shape, GGML_TYPE_F32);
    auto x1 = core::wrap_tensor(
        ggml_add(ctx.ggml, input.tensor, scaled1.tensor), input.shape, GGML_TYPE_F32);

    auto attn = build_block_attention(ctx, x1, weights.attn, rel_idx_full, rel_idx_short, num_frames);
    auto x2 = core::wrap_tensor(ggml_add(ctx.ggml, x1.tensor, attn.tensor), x1.shape, GGML_TYPE_F32);

    auto x3 = build_conv_module(ctx, x2, weights.conv);

    auto ffn2 = build_ffn(ctx, x3, weights.ff2);
    auto scaled2 = core::wrap_tensor(
        ggml_scale(ctx.ggml, ffn2.tensor, kGraniteSpeech5MacaronScale), ffn2.shape, GGML_TYPE_F32);
    auto x4 = core::wrap_tensor(
        ggml_add(ctx.ggml, x3.tensor, scaled2.tensor), x3.shape, GGML_TYPE_F32);

    return modules::LayerNormModule({1024, kGraniteSpeech5EncoderLayerNormEps})
        .build(ctx, x4, weights.post_norm);
}

}  // namespace



struct GraniteSpeech5EncoderRuntime::Impl {
    Impl(
        std::shared_ptr<const GraniteSpeech5Assets> assets,
        core::ExecutionContext & execution,
        assets::TensorStorageType storage_type,
        size_t graph_arena_bytes)
        : assets_(std::move(assets)), execution_(&execution), graph_arena_bytes_(graph_arena_bytes) {
        if (assets_ == nullptr) {
            throw std::runtime_error("granite_speech5 encoder runtime requires assets");
        }
        if (assets_->source == nullptr) {
            throw std::runtime_error("granite_speech5 encoder runtime requires a tensor source");
        }
        check_granite_speech5_encoder_weight_shapes(*assets_->source);
        store_ = std::make_shared<core::BackendWeightStore>(
            execution.backend(),
            execution.backend_type(),
            "granite_speech5.encoder.weights",
            kEncoderWeightContextBytes);
        weights_ = load_encoder_weights(*store_, *assets_->source, storage_type);
    }

    std::shared_ptr<const GraniteSpeech5Assets> assets_;
    core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::shared_ptr<core::BackendWeightStore> store_;
    EncoderWeights weights_;
    std::unique_ptr<core::PreparedGraph> graph_;
    int64_t steps_ = 0;
    bool capture_stages_ = false;
    ggml_tensor * input_ = nullptr;
    std::vector<ggml_tensor *> outputs_;

};

GraniteSpeech5EncoderRuntime::GraniteSpeech5EncoderRuntime(
    std::shared_ptr<const GraniteSpeech5Assets> assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType storage_type,
    size_t graph_arena_bytes)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, storage_type, graph_arena_bytes)) {}

namespace {

core::TensorValue make_i32_leaf(
    ggml_context * ctx, const core::TensorShape & shape, const std::vector<int32_t> & values) {
    if (static_cast<int64_t>(values.size()) != shape.num_elements()) {
        throw std::runtime_error("granite_speech5 encoder index count does not match shape");
    }
    const auto dims = core::to_ggml_dims(shape);
    auto * sized = ggml_new_tensor(ctx, GGML_TYPE_I32, static_cast<int>(shape.rank), dims.data());
    if (sized == nullptr) {
        throw std::runtime_error("granite_speech5 encoder failed to create index tensor");
    }
    return {sized, shape, GGML_TYPE_I32};
}

std::vector<int32_t> shaw_indices(int64_t block_len) {
    // rel_pos_emb index for (i, j): clamp(i - j, -200, 200) + 512.
    std::vector<int32_t> indices(static_cast<size_t>(block_len * block_len));
    for (int64_t i = 0; i < block_len; ++i) {
        for (int64_t j = 0; j < block_len; ++j) {
            const int64_t dist = std::clamp(i - j, -kGraniteSpeech5EncoderContext, kGraniteSpeech5EncoderContext);
            indices[static_cast<size_t>(i * block_len + j)] = static_cast<int32_t>(dist + 512);
        }
    }
    return indices;
}

void set_tensor_bytes(ggml_tensor * tensor, const void * data, size_t bytes) {
    ggml_backend_tensor_set(tensor, data, 0, bytes);
}

std::vector<float> read_f32(ggml_tensor * tensor, int64_t elements) {
    std::vector<float> values(static_cast<size_t>(elements));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}

}  // namespace

GraniteSpeech5EncoderOutputs GraniteSpeech5EncoderRuntime::encode(
    const std::vector<float> & stacked, int64_t steps, bool capture_stages) const {
    if (steps <= 0) {
        throw std::runtime_error("granite_speech5 encoder requires a positive step count");
    }
    if (static_cast<int64_t>(stacked.size()) != steps * kGraniteSpeech5EncoderInputDim) {
        throw std::runtime_error(
            "granite_speech5 encoder stacked size " + std::to_string(stacked.size()) +
            " does not match " + std::to_string(steps) + " steps of width 160");
    }

    if (!impl_->graph_ || impl_->steps_ != steps || impl_->capture_stages_ != capture_stages) {
        impl_->graph_.reset();
        impl_->steps_ = 0;
        const auto plan = granite_speech5_encoder_blocks_for_steps(steps);
        const size_t graph_nodes = kEncoderGraphBaseNodes + kEncoderGraphNodesPerBlock *
            static_cast<size_t>(kGraniteSpeech5EncoderLayers * plan.num_blocks);
        const size_t metadata_bytes = std::min(impl_->graph_arena_bytes_,
            ggml_graph_overhead_custom(graph_nodes, false) + 2 * graph_nodes * ggml_tensor_overhead());
        auto prepared = std::make_unique<core::PreparedGraph>(
            *impl_->execution_, core::PreparedGraphConfig{
                "granite_speech5.encoder", metadata_bytes, graph_nodes});
        auto & ctx = prepared->build_context();
        auto * ggml_ctx = ctx.ggml;
        auto in_tensor = core::wrap_tensor(
            ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, kGraniteSpeech5EncoderInputDim, steps),
            core::TensorShape::from_dims({1, steps, kGraniteSpeech5EncoderInputDim}),
            GGML_TYPE_F32);

        // Shaw index leaves: one full 200x200 table (flattened to a column
        // for ggml_get_rows) shared by every full block of every layer, plus
        // the short-block table when needed.
        auto rel_idx_full_value = make_i32_leaf(
            ggml_ctx, core::TensorShape::from_dims({200 * 200}), shaw_indices(200));
        const bool need_short = plan.last_block_len != kGraniteSpeech5EncoderContext;
        core::TensorValue rel_idx_short_value{nullptr, {}, GGML_TYPE_I32};
        if (need_short) {
            rel_idx_short_value = make_i32_leaf(
                ggml_ctx,
                core::TensorShape::from_dims({plan.last_block_len * plan.last_block_len}),
                shaw_indices(plan.last_block_len));
        }
        // These constants are uploaded only at preparation and must survive reuse.
        ggml_set_output(rel_idx_full_value.tensor);
        if (need_short) ggml_set_output(rel_idx_short_value.tensor);
        const core::TensorValue * rel_idx_short =
            need_short ? &rel_idx_short_value : &rel_idx_full_value;

        auto h = modules::LinearModule({160, 1024, true}).build(ctx, in_tensor, impl_->weights_.input_linear);
        auto input_proj = h;
        core::TensorValue block1{nullptr, {}, GGML_TYPE_F32};
        core::TensorValue layer8_pre{nullptr, {}, GGML_TYPE_F32};
        core::TensorValue layer8_post{nullptr, {}, GGML_TYPE_F32};
        for (int64_t layer = 0; layer < kGraniteSpeech5EncoderLayers; ++layer) {
            h = build_conformer_block(
                ctx, h, impl_->weights_.layers[static_cast<size_t>(layer)], rel_idx_full_value,
                rel_idx_short, steps);
            if (layer == 0) {
                block1 = h;
            }
            if (layer + 1 == kGraniteSpeech5MidFeedbackAfter) {
                // True block-8 output, retained through the feedback below:
                // the frozen reference now holds true pre under the pre
                // stage name (reference.py clones at the hook).
                layer8_pre = h;
                auto mid = modules::LinearModule({1024, 348, true}).build(ctx, h, impl_->weights_.out);
                auto prob = core::wrap_tensor(
                    ggml_soft_max(ctx.ggml, mid.tensor), mid.shape, GGML_TYPE_F32);
                auto injected = modules::LinearModule({348, 1024, true}).build(ctx, prob, impl_->weights_.out_mid);
                h = core::wrap_tensor(
                    ggml_add(ctx.ggml, h.tensor, injected.tensor), h.shape, GGML_TYPE_F32);
                layer8_post = h;
            }
        }

        impl_->input_ = in_tensor.tensor;
        ggml_set_input(in_tensor.tensor);
        impl_->outputs_.clear();
        std::vector<core::TensorValue> stages{h};
        if (capture_stages) {
            stages.insert(stages.end(), {input_proj, block1, layer8_pre, layer8_post});
        }
        for (const auto & stage : stages) {
            // Validation captures must survive intermediate scratch reuse.
            auto * output = ggml_dup(ctx.ggml, stage.tensor);
            ggml_set_output(output);
            ggml_build_forward_expand(prepared->graph(), output);
            impl_->outputs_.push_back(output);
        }
        prepared->allocate();
        // Only used index leaves own a graph buffer; the single-block cases
        // use exactly one of the two tables.
        const bool need_full = (plan.num_blocks > 1) || (plan.last_block_len == kGraniteSpeech5EncoderContext);
        if (need_full) {
            set_tensor_bytes(
                rel_idx_full_value.tensor,
                shaw_indices(200).data(),
                static_cast<size_t>(200 * 200) * sizeof(int32_t));
        }
        if (need_short) {
            const auto short_indices = shaw_indices(plan.last_block_len);
            set_tensor_bytes(
                rel_idx_short_value.tensor,
                short_indices.data(),
                short_indices.size() * sizeof(int32_t));
        }

        impl_->steps_ = steps;
        impl_->capture_stages_ = capture_stages;
        impl_->graph_ = std::move(prepared);
    }
    ggml_backend_tensor_set(impl_->input_, stacked.data(), 0, stacked.size() * sizeof(float));
    impl_->graph_->compute();
    GraniteSpeech5EncoderOutputs outputs;
    outputs.steps = steps;
    const int64_t elements = steps * kGraniteSpeech5EncoderHidden;
    outputs.last = read_f32(impl_->outputs_[0], elements);
    if (capture_stages) {
        outputs.input_proj = read_f32(impl_->outputs_[1], elements);
        outputs.block1 = read_f32(impl_->outputs_[2], elements);
        outputs.layer8_pre_feedback = read_f32(impl_->outputs_[3], elements);
        outputs.layer8_post_feedback = read_f32(impl_->outputs_[4], elements);
    }
    return outputs;
}

GraniteSpeech5EncoderRuntime::~GraniteSpeech5EncoderRuntime() = default;

}  // namespace engine::community_models::granite_speech5
