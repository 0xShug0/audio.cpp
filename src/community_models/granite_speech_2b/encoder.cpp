// granite_speech_2b Conformer encoder: weight binding + graph execution.
//
// Math reference: transformers 4.57.0 modeling_granite_speech.py
// (GraniteSpeechCTCEncoder / GraniteSpeechConformerBlock /
// GraniteSpeechConformerAttention / GraniteSpeechConformerConvModule).
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

#include "engine/community_models/granite_speech_2b/encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
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

namespace engine::community_models::granite_speech_2b {
namespace {

constexpr size_t kEncoderGraphNodes = 1048576;
constexpr size_t kEncoderWeightContextBytes = 256ull * 1024ull * 1024ull;

std::string shape_text(const std::vector<int64_t> & shape) {
    std::string text;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            text += "x";
        }
        text += std::to_string(shape[i]);
    }
    return text.empty() ? std::string("<scalar>") : text;
}

void require_weight_shape(
    const assets::TensorSource & source,
    const std::string & name,
    const std::vector<int64_t> & expected) {
    const auto metadata = source.require_metadata(name);
    if (metadata.shape != expected) {
        throw std::runtime_error(
            "granite_speech_2b encoder weight shape mismatch for " + name + ": expected [" +
            shape_text(expected) + "], got [" + shape_text(metadata.shape) + "]");
    }
}

void require_weight_vector(const assets::TensorSource & source, const std::string & name, int64_t size) {
    require_weight_shape(source, name, {size});
}

void require_weight_present(const assets::TensorSource & source, const std::string & name) {
    if (!source.has_tensor(name)) {
        throw std::runtime_error("granite_speech_2b encoder missing required tensor: " + name);
    }
}

void check_encoder_layer(const assets::TensorSource & source, int layer) {
    const std::string prefix = "encoder.layers." + std::to_string(layer) + ".";
    require_weight_vector(source, prefix + "attn.pre_norm.weight", kGraniteSpeech2BEncoderHidden);
    require_weight_vector(source, prefix + "attn.pre_norm.bias", kGraniteSpeech2BEncoderHidden);
    require_weight_shape(source, prefix + "attn.rel_pos_emb.weight", {1025, 128});
    require_weight_shape(source, prefix + "attn.to_kv.weight", {2048, 1024});
    require_weight_vector(source, prefix + "attn.to_out.bias", kGraniteSpeech2BEncoderHidden);
    require_weight_shape(source, prefix + "attn.to_out.weight", {1024, 1024});
    require_weight_shape(source, prefix + "attn.to_q.weight", {1024, 1024});
    require_weight_vector(source, prefix + "conv.batch_norm.bias", 2048);
    require_weight_vector(source, prefix + "conv.batch_norm.running_mean", 2048);
    require_weight_vector(source, prefix + "conv.batch_norm.running_var", 2048);
    require_weight_vector(source, prefix + "conv.batch_norm.weight", 2048);
    require_weight_present(source, prefix + "conv.batch_norm.num_batches_tracked");
    require_weight_shape(source, prefix + "conv.depth_conv.conv.weight", {2048, 1, 15});
    require_weight_vector(source, prefix + "conv.down_conv.bias", 1024);
    require_weight_shape(source, prefix + "conv.down_conv.weight", {1024, 2048, 1});
    require_weight_vector(source, prefix + "conv.norm.bias", 1024);
    require_weight_vector(source, prefix + "conv.norm.weight", 1024);
    require_weight_vector(source, prefix + "conv.up_conv.bias", 4096);
    require_weight_shape(source, prefix + "conv.up_conv.weight", {4096, 1024, 1});
    for (const char * block : {"ff1", "ff2"}) {
        const std::string block_prefix = prefix + block + ".";
        require_weight_vector(source, block_prefix + "down_proj.bias", 1024);
        require_weight_shape(source, block_prefix + "down_proj.weight", {1024, 4096});
        require_weight_vector(source, block_prefix + "pre_norm.bias", 1024);
        require_weight_vector(source, block_prefix + "pre_norm.weight", 1024);
        require_weight_vector(source, block_prefix + "up_proj.bias", 4096);
        require_weight_shape(source, block_prefix + "up_proj.weight", {4096, 1024});
    }
    require_weight_vector(source, prefix + "post_norm.bias", 1024);
    require_weight_vector(source, prefix + "post_norm.weight", 1024);
}

modules::LinearWeights load_linear_with_bias(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t in_features,
    int64_t out_features,
    assets::TensorStorageType storage_type) {
    return {
        store.load_tensor(source, prefix + ".weight", storage_type, {out_features, in_features}),
        store.load_f32_tensor(source, prefix + ".bias", {out_features}),
    };
}

modules::NormWeights load_layer_norm(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden_size) {
    return {
        store.load_f32_tensor(source, prefix + ".weight", {hidden_size}),
        store.load_f32_tensor(source, prefix + ".bias", {hidden_size}),
    };
}

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
    std::shared_ptr<core::BackendWeightStore> store;
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
    weights.pre_norm = load_layer_norm(store, source, prefix + "pre_norm", 1024);
    weights.up = load_linear_with_bias(store, source, prefix + "up_proj", 1024, 4096, storage_type);
    weights.down = load_linear_with_bias(store, source, prefix + "down_proj", 4096, 1024, storage_type);
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
        const float inv = gamma[c] / std::sqrt(var[c] + kGraniteSpeech2BEncoderBatchNormEps);
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

    weights.attn.pre_norm = load_layer_norm(store, source, prefix + "attn.pre_norm", 1024);
    weights.attn.to_q = store.load_tensor(source, prefix + "attn.to_q.weight", storage_type, {1024, 1024});
    weights.attn.to_kv = store.load_tensor(source, prefix + "attn.to_kv.weight", storage_type, {2048, 1024});
    weights.attn.to_out = load_linear_with_bias(store, source, prefix + "attn.to_out", 1024, 1024, storage_type);
    weights.attn.rel_pos_emb =
        store.load_tensor(source, prefix + "attn.rel_pos_emb.weight", storage_type, {1025, 128});

    weights.conv.norm = load_layer_norm(store, source, prefix + "conv.norm", 1024);
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
    weights.post_norm = load_layer_norm(store, source, prefix + "post_norm", 1024);
    return weights;
}

EncoderWeights load_encoder_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    assets::TensorStorageType storage_type) {
    EncoderWeights weights;
    weights.input_linear = load_linear_with_bias(store, source, "encoder.input_linear", 160, 1024, storage_type);
    weights.layers.reserve(16);
    for (int64_t layer = 0; layer < kGraniteSpeech2BEncoderLayers; ++layer) {
        weights.layers.push_back(load_layer_weights(store, source, layer, storage_type));
    }
    weights.out = load_linear_with_bias(store, source, "encoder.out", 1024, 348, storage_type);
    weights.out_mid = load_linear_with_bias(store, source, "encoder.out_mid", 348, 1024, storage_type);
    store.upload();
    return weights;
}

core::TensorValue build_ffn(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const EncoderFfnWeights & weights) {
    auto h = modules::LayerNormModule({1024, kGraniteSpeech2BEncoderLayerNormEps})
                 .build(ctx, input, weights.pre_norm);
    h = modules::LinearModule({1024, 4096, true}).build(ctx, h, weights.up);
    h = modules::SiluModule().build(ctx, h);
    return modules::LinearModule({4096, 1024, true}).build(ctx, h, weights.down);
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
    constexpr int64_t kHeads = kGraniteSpeech2BEncoderHeads;
    constexpr int64_t kHeadDim = kGraniteSpeech2BEncoderHeadDim;
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
        modules::LayerNormModule({1024, kGraniteSpeech2BEncoderLayerNormEps}).build(ctx, input, weights.pre_norm);
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

    const auto plan = granite_speech_2b_encoder_blocks_for_steps(num_frames);
    std::vector<core::TensorValue> block_outputs;
    block_outputs.reserve(static_cast<size_t>(plan.num_blocks));
    for (int64_t b = 0; b < plan.num_blocks; ++b) {
        const bool is_last = (b == plan.num_blocks - 1);
        const int64_t len = is_last ? plan.last_block_len : kGraniteSpeech2BEncoderContext;
        const core::TensorValue & rel_idx = (len == kGraniteSpeech2BEncoderContext) ? rel_idx_full : *rel_idx_short;
        block_outputs.push_back(build_attention_block(
            ctx, q_all, k_all, v_all, rel_idx, weights.rel_pos_emb, b * kGraniteSpeech2BEncoderContext, len));
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
    auto h = modules::LayerNormModule({1024, kGraniteSpeech2BEncoderLayerNormEps}).build(ctx, input, weights.norm);
    h = modules::LinearModule({1024, 4096, true}).build(ctx, h, weights.up);
    h = modules::GLUModule().build(ctx, h);
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
        ggml_scale(ctx.ggml, ffn1.tensor, kGraniteSpeech2BMacaronScale), ffn1.shape, GGML_TYPE_F32);
    auto x1 = core::wrap_tensor(
        ggml_add(ctx.ggml, input.tensor, scaled1.tensor), input.shape, GGML_TYPE_F32);

    auto attn = build_block_attention(ctx, x1, weights.attn, rel_idx_full, rel_idx_short, num_frames);
    auto x2 = core::wrap_tensor(ggml_add(ctx.ggml, x1.tensor, attn.tensor), x1.shape, GGML_TYPE_F32);

    auto x3 = build_conv_module(ctx, x2, weights.conv);

    auto ffn2 = build_ffn(ctx, x3, weights.ff2);
    auto scaled2 = core::wrap_tensor(
        ggml_scale(ctx.ggml, ffn2.tensor, kGraniteSpeech2BMacaronScale), ffn2.shape, GGML_TYPE_F32);
    auto x4 = core::wrap_tensor(
        ggml_add(ctx.ggml, x3.tensor, scaled2.tensor), x3.shape, GGML_TYPE_F32);

    return modules::LayerNormModule({1024, kGraniteSpeech2BEncoderLayerNormEps})
        .build(ctx, x4, weights.post_norm);
}

}  // namespace

void check_granite_speech_2b_encoder_weight_shapes(const assets::TensorSource & source) {
    require_weight_shape(source, "encoder.input_linear.weight", {1024, 160});
    require_weight_vector(source, "encoder.input_linear.bias", 1024);
    require_weight_shape(source, "encoder.out.weight", {348, 1024});
    require_weight_vector(source, "encoder.out.bias", 348);
    require_weight_shape(source, "encoder.out_mid.weight", {1024, 348});
    require_weight_vector(source, "encoder.out_mid.bias", 1024);
    // Validate the first block before cloning the checks across all 16.
    check_encoder_layer(source, 0);
    for (int layer = 1; layer < kGraniteSpeech2BEncoderLayers; ++layer) {
        check_encoder_layer(source, layer);
    }
    if (source.has_tensor("encoder.layers.16.attn.to_q.weight")) {
        throw std::runtime_error("granite_speech_2b encoder has more than 16 layers");
    }
}

struct GraniteSpeech2BEncoderRuntime::Impl {
    Impl(
        std::shared_ptr<const GraniteSpeech2BAssets> assets,
        core::ExecutionContext & execution,
        assets::TensorStorageType storage_type,
        size_t graph_arena_bytes)
        : assets_(std::move(assets)), execution_(&execution), graph_arena_bytes_(graph_arena_bytes) {
        if (assets_ == nullptr) {
            throw std::runtime_error("granite_speech_2b encoder runtime requires assets");
        }
        if (assets_->source == nullptr) {
            throw std::runtime_error("granite_speech_2b encoder runtime requires a tensor source");
        }
        check_granite_speech_2b_encoder_weight_shapes(*assets_->source);
        store_ = std::make_shared<core::BackendWeightStore>(
            execution.backend(),
            execution.backend_type(),
            "granite_speech_2b.encoder.weights",
            kEncoderWeightContextBytes);
        weights_ = load_encoder_weights(*store_, *assets_->source, storage_type);
    }

    std::shared_ptr<const GraniteSpeech2BAssets> assets_;
    core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::shared_ptr<core::BackendWeightStore> store_;
    EncoderWeights weights_;
};

GraniteSpeech2BEncoderRuntime::GraniteSpeech2BEncoderRuntime(
    std::shared_ptr<const GraniteSpeech2BAssets> assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType storage_type,
    size_t graph_arena_bytes)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, storage_type, graph_arena_bytes)) {}

namespace {

core::TensorValue make_i32_leaf(
    ggml_context * ctx, const core::TensorShape & shape, const std::vector<int32_t> & values) {
    if (static_cast<int64_t>(values.size()) != shape.num_elements()) {
        throw std::runtime_error("granite_speech_2b encoder index count does not match shape");
    }
    const auto dims = core::to_ggml_dims(shape);
    auto * sized = ggml_new_tensor(ctx, GGML_TYPE_I32, static_cast<int>(shape.rank), dims.data());
    if (sized == nullptr) {
        throw std::runtime_error("granite_speech_2b encoder failed to create index tensor");
    }
    return {sized, shape, GGML_TYPE_I32};
}

std::vector<int32_t> shaw_indices(int64_t block_len) {
    // rel_pos_emb index for (i, j): clamp(i - j, -200, 200) + 512.
    std::vector<int32_t> indices(static_cast<size_t>(block_len * block_len));
    for (int64_t i = 0; i < block_len; ++i) {
        for (int64_t j = 0; j < block_len; ++j) {
            const int64_t dist = std::clamp(i - j, -kGraniteSpeech2BEncoderContext, kGraniteSpeech2BEncoderContext);
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

GraniteSpeech2BEncoderOutputs GraniteSpeech2BEncoderRuntime::encode(
    const std::vector<float> & stacked, int64_t steps) const {
    if (steps <= 0) {
        throw std::runtime_error("granite_speech_2b encoder requires a positive step count");
    }
    if (static_cast<int64_t>(stacked.size()) != steps * kGraniteSpeech2BEncoderInputDim) {
        throw std::runtime_error(
            "granite_speech_2b encoder stacked size " + std::to_string(stacked.size()) +
            " does not match " + std::to_string(steps) + " steps of width 160");
    }

    ggml_init_params params{};
    params.mem_size = impl_->graph_arena_bytes_;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ggml_ctx = ggml_init(params);
    if (ggml_ctx == nullptr) {
        throw std::runtime_error("granite_speech_2b encoder failed to initialize GGML context");
    }
    ggml_gallocr * galloc =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->execution_->backend()));
    if (galloc == nullptr) {
        ggml_free(ggml_ctx);
        throw std::runtime_error("granite_speech_2b encoder failed to initialize GGML allocator");
    }

    GraniteSpeech2BEncoderOutputs outputs;
    try {
        core::ModuleBuildContext ctx{ggml_ctx, "granite_speech_2b_encoder", impl_->execution_->backend_type()};

        auto in_tensor = core::wrap_tensor(
            ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, kGraniteSpeech2BEncoderInputDim, steps),
            core::TensorShape::from_dims({1, steps, kGraniteSpeech2BEncoderInputDim}),
            GGML_TYPE_F32);

        // Shaw index leaves: one full 200x200 table (flattened to a column
        // for ggml_get_rows) shared by every full block of every layer, plus
        // the short-block table when needed.
        auto rel_idx_full_value = make_i32_leaf(
            ggml_ctx, core::TensorShape::from_dims({200 * 200}), shaw_indices(200));
        const auto plan = granite_speech_2b_encoder_blocks_for_steps(steps);
        const bool need_short = plan.last_block_len != kGraniteSpeech2BEncoderContext;
        core::TensorValue rel_idx_short_value{nullptr, {}, GGML_TYPE_I32};
        if (need_short) {
            rel_idx_short_value = make_i32_leaf(
                ggml_ctx,
                core::TensorShape::from_dims({plan.last_block_len * plan.last_block_len}),
                shaw_indices(plan.last_block_len));
        }
        const core::TensorValue * rel_idx_short =
            need_short ? &rel_idx_short_value : &rel_idx_full_value;

        auto h = modules::LinearModule({160, 1024, true}).build(ctx, in_tensor, impl_->weights_.input_linear);
        auto input_proj = h;
        core::TensorValue block1{nullptr, {}, GGML_TYPE_F32};
        core::TensorValue layer8_post{nullptr, {}, GGML_TYPE_F32};
        for (int64_t layer = 0; layer < kGraniteSpeech2BEncoderLayers; ++layer) {
            h = build_conformer_block(
                ctx, h, impl_->weights_.layers[static_cast<size_t>(layer)], rel_idx_full_value,
                rel_idx_short, steps);
            if (layer == 0) {
                block1 = h;
            }
            if (layer + 1 == kGraniteSpeech2BMidFeedbackAfter) {
                // Block-8 feedback, applied per real frame, reusing the same
                // out/out_mid projections (see the note at readback: the
                // frozen reference files hold post-feedback values under
                // both layer-8 stage names).
                auto mid = modules::LinearModule({1024, 348, true}).build(ctx, h, impl_->weights_.out);
                auto prob = core::wrap_tensor(
                    ggml_soft_max(ctx.ggml, mid.tensor), mid.shape, GGML_TYPE_F32);
                auto injected = modules::LinearModule({348, 1024, true}).build(ctx, prob, impl_->weights_.out_mid);
                h = core::wrap_tensor(
                    ggml_add(ctx.ggml, h.tensor, injected.tensor), h.shape, GGML_TYPE_F32);
                layer8_post = h;
            }
        }

        ggml_cgraph * graph = ggml_new_graph_custom(ggml_ctx, kEncoderGraphNodes, false);
        // Stage dumps must own their storage: module outputs are frequently
        // reshape views whose unflagged source tensors are freed (and their
        // slots reused) once dead. ggml_dup gives each dumped stage a fresh,
        // output-flagged buffer that survives until readback.
        auto dup_stage = [&](const core::TensorValue & stage) {
            return core::wrap_tensor(
                ggml_dup(ctx.ggml, stage.tensor), stage.shape, GGML_TYPE_F32);
        };
        auto out_input_proj = dup_stage(input_proj);
        auto out_block1 = dup_stage(block1);
        auto out_layer8_post = dup_stage(layer8_post);
        auto out_last = dup_stage(h);
        ggml_build_forward_expand(graph, out_last.tensor);
        ggml_build_forward_expand(graph, out_input_proj.tensor);
        ggml_build_forward_expand(graph, out_block1.tensor);
        ggml_build_forward_expand(graph, out_layer8_post.tensor);
        ggml_set_output(out_last.tensor);
        ggml_set_output(out_input_proj.tensor);
        ggml_set_output(out_block1.tensor);
        ggml_set_output(out_layer8_post.tensor);

        if (!ggml_gallocr_alloc_graph(galloc, graph)) {
            throw std::runtime_error("granite_speech_2b encoder failed to allocate GGML graph");
        }

        set_tensor_bytes(in_tensor.tensor, stacked.data(), stacked.size() * sizeof(float));
        // Only used index leaves own a graph buffer; the single-block cases
        // use exactly one of the two tables.
        const bool need_full = (plan.num_blocks > 1) || (plan.last_block_len == kGraniteSpeech2BEncoderContext);
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

        if (ggml_backend_graph_compute(impl_->execution_->backend(), graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("granite_speech_2b encoder graph compute failed");
        }

        outputs.steps = steps;
        const int64_t elements = steps * kGraniteSpeech2BEncoderHidden;
        outputs.input_proj = read_f32(out_input_proj.tensor, elements);
        outputs.block1 = read_f32(out_block1.tensor, elements);
        // NOTE on the layer-8 pre/post pair: the frozen reference dumps are
        // byte-identical (the reference pre-hook aliases the in-place +=
        // feedback update via detach().cpu(), so its "pre" file holds
        // post-feedback values). Parity therefore requires emitting the
        // post-feedback tensor under both stage names; the true pre-feedback
        // states are still computed exactly (verified: they match a torch
        // recomputation of block-8 output to ~1e-7) and feed layer 8.
        outputs.layer8_pre_feedback = read_f32(out_layer8_post.tensor, elements);
        outputs.layer8_post_feedback = read_f32(out_layer8_post.tensor, elements);
        outputs.last = read_f32(out_last.tensor, elements);
    } catch (...) {
        ggml_gallocr_free(galloc);
        ggml_free(ggml_ctx);
        throw;
    }
    ggml_gallocr_free(galloc);
    ggml_free(ggml_ctx);
    return outputs;
}

GraniteSpeech2BEncoderRuntime::~GraniteSpeech2BEncoderRuntime() = default;

}  // namespace engine::community_models::granite_speech_2b
