#include "engine/community_models/granite5asr/encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
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
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::granite5asr {
namespace {

constexpr size_t kEncoderGraphNodes = 1048576;
constexpr float kLayerNormEpsilon = 1.0e-5f;

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

modules::LinearWeights load_linear_no_bias(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t in_features,
    int64_t out_features,
    assets::TensorStorageType storage_type) {
    return {
        store.load_tensor(source, prefix + ".weight", storage_type, {out_features, in_features}),
        std::nullopt,
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

void load_folded_depthwise_conv(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    int64_t kernel_size,
    assets::TensorStorageType storage_type,
    core::TensorValue & out_weight,
    core::TensorValue & out_bias) {
    const auto raw_w = source.require_f32(prefix + ".depthwise_conv.weight", {channels, 1, kernel_size});
    const auto gamma = source.require_f32(prefix + ".norm.weight", {channels});
    const auto beta = source.require_f32(prefix + ".norm.bias", {channels});
    const auto mean = source.require_f32(prefix + ".norm.running_mean", {channels});
    const auto var = source.require_f32(prefix + ".norm.running_var", {channels});

    std::vector<float> folded_w(static_cast<size_t>(channels * kernel_size), 0.0f);
    std::vector<float> folded_b(static_cast<size_t>(channels), 0.0f);

    for (int64_t c = 0; c < channels; ++c) {
        const float scale = gamma[static_cast<size_t>(c)] / std::sqrt(var[static_cast<size_t>(c)] + 1e-5f);
        folded_b[static_cast<size_t>(c)] = beta[static_cast<size_t>(c)] - mean[static_cast<size_t>(c)] * scale;
        for (int64_t k = 0; k < kernel_size; ++k) {
            folded_w[static_cast<size_t>(c * kernel_size + k)] =
                raw_w[static_cast<size_t>(c * kernel_size + k)] * scale;
        }
    }

    out_weight = store.make_from_f32(
        core::TensorShape::from_dims({channels, 1, kernel_size}),
        storage_type == assets::TensorStorageType::Native ? assets::TensorStorageType::F32 : storage_type,
        std::move(folded_w));
    out_bias = store.make_from_f32(
        core::TensorShape::from_dims({channels}),
        assets::TensorStorageType::F32,
        std::move(folded_b));
}

Granite5LayerWeights load_layer_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    int64_t layer_idx,
    const Granite5EncoderConfig & config,
    assets::TensorStorageType storage_type) {
    Granite5LayerWeights w;
    const std::string pfx = "encoder.layers." + std::to_string(layer_idx) + ".";

    w.is_subsample = (std::find(config.subsample_layers.begin(), config.subsample_layers.end(), layer_idx) != config.subsample_layers.end());

    // FFN1
    w.ffn1_norm = load_layer_norm(store, source, pfx + "norm_feed_forward1", config.hidden_size);
    w.ffn1_fc1 = load_linear_with_bias(store, source, pfx + "feed_forward1.linear1", config.hidden_size, config.intermediate_size, storage_type);
    w.ffn1_fc2 = load_linear_with_bias(store, source, pfx + "feed_forward1.linear2", config.intermediate_size, config.hidden_size, storage_type);

    // Self-Attention
    w.norm_self_att = load_layer_norm(store, source, pfx + "norm_self_att", config.hidden_size);
    w.q_proj = store.load_tensor(source, pfx + "self_attn.q_proj.weight", storage_type, {config.hidden_size, config.hidden_size});
    w.k_proj = store.load_tensor(source, pfx + "self_attn.k_proj.weight", storage_type, {config.hidden_size, config.hidden_size});
    w.v_proj = store.load_tensor(source, pfx + "self_attn.v_proj.weight", storage_type, {config.hidden_size, config.hidden_size});
    w.o_proj = load_linear_with_bias(store, source, pfx + "self_attn.o_proj", config.hidden_size, config.hidden_size, storage_type);

    const int64_t max_pos_rows = 2 * config.max_position_embeddings + 1;
    const auto raw_emb = source.require_f32(pfx + "self_attn.rel_pos_emb.weight", {max_pos_rows, config.head_dim});
    std::vector<float> rel_128(static_cast<size_t>(config.context_size * config.context_size * config.head_dim), 0.0f);
    for (int64_t i = 0; i < config.context_size; ++i) {
        for (int64_t j = 0; j < config.context_size; ++j) {
            const int64_t dist = std::clamp(i - j, static_cast<int64_t>(-512), static_cast<int64_t>(512)) + 512;
            const float * src_d = &raw_emb[static_cast<size_t>(dist * config.head_dim)];
            float * dst_d = &rel_128[static_cast<size_t>((i * config.context_size + j) * config.head_dim)];
            std::memcpy(dst_d, src_d, static_cast<size_t>(config.head_dim) * sizeof(float));
        }
    }
    w.rel_pos_emb = store.make_from_f32(
        core::TensorShape::from_dims({config.head_dim, config.context_size, config.context_size}),
        storage_type == assets::TensorStorageType::Native ? assets::TensorStorageType::F32 : storage_type,
        std::move(rel_128));

    // Convolution
    const int64_t conv_inner_dim = config.hidden_size * config.conv_expansion_factor; // 2048
    w.norm_conv = load_layer_norm(store, source, pfx + "norm_conv", config.hidden_size);
    w.conv_pw1 = load_linear_with_bias(store, source, pfx + "conv.pointwise_lin1", config.hidden_size, conv_inner_dim * 2, storage_type);
    load_folded_depthwise_conv(store, source, pfx + "conv", conv_inner_dim, config.conv_kernel_size, storage_type, w.conv_dw_weight, w.conv_dw_bias);
    w.conv_pw2 = load_linear_with_bias(store, source, pfx + "conv.pointwise_lin2", conv_inner_dim, config.hidden_size, storage_type);

    // FFN2
    w.ffn2_norm = load_layer_norm(store, source, pfx + "norm_feed_forward2", config.hidden_size);
    w.ffn2_fc1 = load_linear_with_bias(store, source, pfx + "feed_forward2.linear1", config.hidden_size, config.intermediate_size, storage_type);
    w.ffn2_fc2 = load_linear_with_bias(store, source, pfx + "feed_forward2.linear2", config.intermediate_size, config.hidden_size, storage_type);

    // Norm Out
    w.norm_out = load_layer_norm(store, source, pfx + "norm_out", config.hidden_size);
    return w;
}

Granite5EncoderWeights load_encoder_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const Granite5ASRConfig & config,
    assets::TensorStorageType storage_type) {
    Granite5EncoderWeights weights;
    weights.input_linear = load_linear_with_bias(
        store, source, "encoder.input_linear",
        config.encoder.input_features, config.encoder.hidden_size, storage_type);

    weights.layers.reserve(static_cast<size_t>(config.encoder.num_layers));
    for (int64_t idx = 0; idx < config.encoder.num_layers; ++idx) {
        weights.layers.push_back(load_layer_weights(store, source, idx, config.encoder, storage_type));
    }

    weights.out = load_linear_with_bias(
        store, source, "encoder.out",
        config.encoder.hidden_size, config.vocab_size, storage_type);
    weights.out_mid = load_linear_with_bias(
        store, source, "encoder.out_mid",
        config.vocab_size, config.encoder.hidden_size, storage_type);

    store.upload();
    return weights;
}

core::TensorValue build_block_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const Granite5LayerWeights & weights,
    const Granite5EncoderConfig & config) {
    const int64_t num_frames = x.shape.dims[1];
    const int64_t d_model = config.hidden_size;
    const int64_t heads = config.num_attention_heads;
    const int64_t d_head = config.head_dim;
    const int64_t c = config.context_size;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_head));

    auto q_all = modules::LinearModule({d_model, d_model, false}).build(ctx, x, {weights.q_proj, std::nullopt});
    auto k_all = modules::LinearModule({d_model, d_model, false}).build(ctx, x, {weights.k_proj, std::nullopt});
    auto v_all = modules::LinearModule({d_model, d_model, false}).build(ctx, x, {weights.v_proj, std::nullopt});

    const int64_t nb_full = num_frames / c;
    const int64_t nr = num_frames % c;
    const int64_t num_blocks = nb_full + (nr > 0 ? 1 : 0);

    std::vector<core::TensorValue> block_outputs;
    block_outputs.reserve(static_cast<size_t>(num_blocks));

    for (int64_t b = 0; b < num_blocks; ++b) {
        const int64_t blk = (b == nb_full) ? nr : c;
        const int64_t t_offset = b * c;

        auto q_blk = modules::SliceModule({1, t_offset, blk}).build(ctx, q_all);
        auto k_blk = modules::SliceModule({1, t_offset, blk}).build(ctx, k_all);
        auto v_blk = modules::SliceModule({1, t_offset, blk}).build(ctx, v_all);

        auto q_3d = core::wrap_tensor(
            ggml_reshape_3d(ctx.ggml, q_blk.tensor, d_head, heads, blk),
            core::TensorShape::from_dims({blk, heads, d_head}),
            GGML_TYPE_F32);
        auto k_3d = core::wrap_tensor(
            ggml_reshape_3d(ctx.ggml, k_blk.tensor, d_head, heads, blk),
            core::TensorShape::from_dims({blk, heads, d_head}),
            GGML_TYPE_F32);
        auto v_3d = core::wrap_tensor(
            ggml_reshape_3d(ctx.ggml, v_blk.tensor, d_head, heads, blk),
            core::TensorShape::from_dims({blk, heads, d_head}),
            GGML_TYPE_F32);

        auto q_perm = core::wrap_tensor(
            ggml_permute(ctx.ggml, q_3d.tensor, 0, 2, 1, 3),
            core::TensorShape::from_dims({heads, blk, d_head}),
            GGML_TYPE_F32);
        auto k_perm = core::wrap_tensor(
            ggml_permute(ctx.ggml, k_3d.tensor, 0, 2, 1, 3),
            core::TensorShape::from_dims({heads, blk, d_head}),
            GGML_TYPE_F32);

        auto k_cont = core::wrap_tensor(
            ggml_cont(ctx.ggml, k_perm.tensor),
            k_perm.shape,
            GGML_TYPE_F32);
        auto scores = core::wrap_tensor(
            ggml_mul_mat(ctx.ggml, k_cont.tensor, q_perm.tensor),
            core::TensorShape::from_dims({heads, blk, blk}),
            GGML_TYPE_F32);

        const size_t elem_size = ggml_type_size(weights.rel_pos_emb.type);
        const size_t nb1 = static_cast<size_t>(d_head) * elem_size;
        const size_t nb2 = static_cast<size_t>(c * d_head) * elem_size;

        auto rel_slice = core::wrap_tensor(
            ggml_view_3d(
                ctx.ggml,
                weights.rel_pos_emb.tensor,
                d_head,
                blk,
                blk,
                nb1,
                nb2,
                0),
            core::TensorShape::from_dims({blk, blk, d_head}),
            weights.rel_pos_emb.type);
        auto rel_slice_cont = core::wrap_tensor(
            ggml_cont(ctx.ggml, rel_slice.tensor),
            rel_slice.shape,
            weights.rel_pos_emb.type);

        auto pos_bias_c = core::wrap_tensor(
            ggml_mul_mat(ctx.ggml, rel_slice_cont.tensor, q_3d.tensor),
            core::TensorShape::from_dims({blk, heads, blk}),
            GGML_TYPE_F32);

        auto pos_bias = core::wrap_tensor(
            ggml_permute(ctx.ggml, pos_bias_c.tensor, 0, 2, 1, 3),
            core::TensorShape::from_dims({heads, blk, blk}),
            GGML_TYPE_F32);

        auto total_scores = core::wrap_tensor(
            ggml_add(ctx.ggml, scores.tensor, pos_bias.tensor),
            scores.shape,
            GGML_TYPE_F32);
        total_scores = core::wrap_tensor(
            ggml_scale(ctx.ggml, total_scores.tensor, scale),
            total_scores.shape,
            GGML_TYPE_F32);

        auto attn_weights = core::wrap_tensor(
            ggml_soft_max(ctx.ggml, total_scores.tensor),
            total_scores.shape,
            GGML_TYPE_F32);

        auto v_perm = core::wrap_tensor(
            ggml_permute(ctx.ggml, v_3d.tensor, 1, 2, 0, 3),
            core::TensorShape::from_dims({heads, d_head, blk}),
            GGML_TYPE_F32);
        auto v_cont = core::wrap_tensor(
            ggml_cont(ctx.ggml, v_perm.tensor),
            v_perm.shape,
            GGML_TYPE_F32);

        auto out_head = core::wrap_tensor(
            ggml_mul_mat(ctx.ggml, v_cont.tensor, attn_weights.tensor),
            core::TensorShape::from_dims({heads, blk, d_head}),
            GGML_TYPE_F32);

        auto out_perm = core::wrap_tensor(
            ggml_permute(ctx.ggml, out_head.tensor, 0, 2, 1, 3),
            core::TensorShape::from_dims({blk, heads, d_head}),
            GGML_TYPE_F32);
        auto out_cont = core::wrap_tensor(
            ggml_cont(ctx.ggml, out_perm.tensor),
            out_perm.shape,
            GGML_TYPE_F32);
        auto out_block = core::wrap_tensor(
            ggml_reshape_3d(ctx.ggml, out_cont.tensor, d_model, blk, 1),
            core::TensorShape::from_dims({1, blk, d_model}),
            GGML_TYPE_F32);
        block_outputs.push_back(out_block);
    }

    core::TensorValue attn_out = block_outputs[0];
    for (size_t i = 1; i < block_outputs.size(); ++i) {
        attn_out = modules::ConcatModule({1}).build(ctx, attn_out, block_outputs[i]);
    }

    return modules::LinearModule({d_model, d_model, true}).build(ctx, attn_out, weights.o_proj);
}

core::TensorValue build_conformer_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_btc,
    const Granite5LayerWeights & weights,
    const Granite5EncoderConfig & config) {
    const int64_t d_model = config.hidden_size;
    const int64_t d_ffn = config.intermediate_size;
    const int64_t d_conv = d_model * config.conv_expansion_factor; // 2048

    // 1. FFN1: x + 0.5 * Linear2(SiLU(Linear1(LN(x))))
    auto h_norm1 = modules::LayerNormModule({d_model, kLayerNormEpsilon}).build(ctx, input_btc, weights.ffn1_norm);
    auto h_ff1 = modules::LinearModule({d_model, d_ffn, true}).build(ctx, h_norm1, weights.ffn1_fc1);
    h_ff1 = modules::SiluModule().build(ctx, h_ff1);
    h_ff1 = modules::LinearModule({d_ffn, d_model, true}).build(ctx, h_ff1, weights.ffn1_fc2);
    auto h_scaled1 = core::wrap_tensor(
        ggml_scale(ctx.ggml, h_ff1.tensor, 0.5f),
        h_ff1.shape,
        GGML_TYPE_F32);
    auto x1 = core::wrap_tensor(
        ggml_add(ctx.ggml, input_btc.tensor, h_scaled1.tensor),
        input_btc.shape,
        GGML_TYPE_F32);

    // 2. Self-Attention: x1 + OutLinear(BlockAttn(LN(x1)))
    auto h_norm_att = modules::LayerNormModule({d_model, kLayerNormEpsilon}).build(ctx, x1, weights.norm_self_att);
    auto h_att = build_block_attention(ctx, h_norm_att, weights, config);
    auto x2 = core::wrap_tensor(
        ggml_add(ctx.ggml, x1.tensor, h_att.tensor),
        x1.shape,
        GGML_TYPE_F32);

    // 3. Conformer Convolution
    auto h_norm_conv = modules::LayerNormModule({d_model, kLayerNormEpsilon}).build(ctx, x2, weights.norm_conv);
    auto h_pw1 = modules::LinearModule({d_model, d_conv * 2, true}).build(ctx, h_norm_conv, weights.conv_pw1);
    auto h_glu = modules::GLUModule().build(ctx, h_pw1); // [1, T, 2048]

    auto h_bct = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, h_glu);

    const int stride = weights.is_subsample ? 2 : 1;
    auto conv_out = modules::DepthwiseConv1dModule({
        d_conv,
        config.conv_kernel_size,
        stride,
        static_cast<int>((config.conv_kernel_size - 1) / 2),
        1,
        true
    }).build(ctx, h_bct, {weights.conv_dw_weight, weights.conv_dw_bias});

    auto conv_btc = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, conv_out);
    conv_btc = modules::SiluModule().build(ctx, conv_btc);
    auto conv_pw2 = modules::LinearModule({d_conv, d_model, true}).build(ctx, conv_btc, weights.conv_pw2);

    core::TensorValue x3;
    if (weights.is_subsample) {
        const int64_t t_half = x2.shape.dims[1] / 2;
        auto x2_cont = core::wrap_tensor(
            ggml_cont(ctx.ggml, x2.tensor),
            x2.shape,
            GGML_TYPE_F32);
        auto x2_pooled = core::wrap_tensor(
            ggml_pool_2d(ctx.ggml, x2_cont.tensor, GGML_OP_POOL_AVG, 1, 2, 1, 2, 0, 0),
            core::TensorShape::from_dims({1, t_half, d_model}),
            GGML_TYPE_F32);
        auto conv_pw2_trimmed = (conv_pw2.shape.dims[1] == t_half)
            ? conv_pw2
            : modules::SliceModule({1, 0, t_half}).build(ctx, conv_pw2);
        x3 = core::wrap_tensor(
            ggml_add(ctx.ggml, conv_pw2_trimmed.tensor, x2_pooled.tensor),
            x2_pooled.shape,
            GGML_TYPE_F32);
    } else {
        x3 = core::wrap_tensor(
            ggml_add(ctx.ggml, x2.tensor, conv_pw2.tensor),
            x2.shape,
            GGML_TYPE_F32);
    }

    // 4. FFN2: x3 + 0.5 * Linear2(SiLU(Linear1(LN(x3))))
    auto h_norm2 = modules::LayerNormModule({d_model, kLayerNormEpsilon}).build(ctx, x3, weights.ffn2_norm);
    auto h_ff2 = modules::LinearModule({d_model, d_ffn, true}).build(ctx, h_norm2, weights.ffn2_fc1);
    h_ff2 = modules::SiluModule().build(ctx, h_ff2);
    h_ff2 = modules::LinearModule({d_ffn, d_model, true}).build(ctx, h_ff2, weights.ffn2_fc2);
    auto h_scaled2 = core::wrap_tensor(
        ggml_scale(ctx.ggml, h_ff2.tensor, 0.5f),
        h_ff2.shape,
        GGML_TYPE_F32);
    auto x4 = core::wrap_tensor(
        ggml_add(ctx.ggml, x3.tensor, h_scaled2.tensor),
        x3.shape,
        GGML_TYPE_F32);

    // 5. Output LayerNorm
    return modules::LayerNormModule({d_model, kLayerNormEpsilon}).build(ctx, x4, weights.norm_out);
}

}  // namespace

Granite5EncoderRuntime::Granite5EncoderRuntime(
    std::shared_ptr<const Granite5ASRAssets> assets,
    engine::core::ExecutionContext & execution_context,
    assets::TensorStorageType storage_type,
    size_t graph_arena_bytes)
    : assets_(std::move(assets)),
      execution_context_(&execution_context),
      weight_store_(
          execution_context.backend(),
          execution_context.backend_type(),
          "Granite 5 ASR encoder weights",
          256ull * 1024ull * 1024ull),
      graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Granite 5 ASR encoder runtime requires assets");
    }
    weights_ = load_encoder_weights(weight_store_, *assets_->source, assets_->config, storage_type);
}

std::vector<int32_t> Granite5EncoderRuntime::transcribe_features(
    const Granite5FrontendFeatures & features) {
    if (features.frames <= 0 || features.values.empty()) {
        return {};
    }

    const auto & config = assets_->config;
    const int64_t num_frames = features.frames;
    const int64_t feat_dim = features.feature_dim;

    ggml_init_params params{};
    params.mem_size = graph_arena_bytes_;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    ggml_context * ggml_ctx = ggml_init(params);
    if (!ggml_ctx) {
        throw std::runtime_error("Failed to initialize GGML context for Granite 5 ASR encoder");
    }

    ggml_gallocr * galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(execution_context_->backend()));
    if (!galloc) {
        ggml_free(ggml_ctx);
        throw std::runtime_error("Failed to initialize GGML allocator for Granite 5 ASR encoder");
    }

    std::vector<int32_t> token_ids;

    try {
        core::ModuleBuildContext ctx{ggml_ctx, "granite5asr_encoder", execution_context_->backend_type()};

        auto in_tensor = core::wrap_tensor(
            ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, feat_dim, num_frames),
            core::TensorShape::from_dims({1, num_frames, feat_dim}),
            GGML_TYPE_F32);

        auto h = modules::LinearModule({feat_dim, config.encoder.hidden_size, true})
                     .build(ctx, in_tensor, weights_.input_linear);

        const int64_t mid_layer_idx = config.encoder.num_layers / 2; // 8
        for (int64_t idx = 0; idx < config.encoder.num_layers; ++idx) {
            h = build_conformer_block(ctx, h, weights_.layers[static_cast<size_t>(idx)], config.encoder);

            if (idx + 1 == mid_layer_idx) {
                auto h_mid = modules::LinearModule({config.encoder.hidden_size, config.vocab_size, true})
                                 .build(ctx, h, weights_.out);
                auto p_mid = core::wrap_tensor(
                    ggml_soft_max(ctx.ggml, h_mid.tensor),
                    h_mid.shape,
                    GGML_TYPE_F32);
                auto h_inj = modules::LinearModule({config.vocab_size, config.encoder.hidden_size, true})
                                 .build(ctx, p_mid, weights_.out_mid);
                h = core::wrap_tensor(
                    ggml_add(ctx.ggml, h.tensor, h_inj.tensor),
                    h.shape,
                    GGML_TYPE_F32);
            }
        }

        auto logits = modules::LinearModule({config.encoder.hidden_size, config.vocab_size, true})
                          .build(ctx, h, weights_.out);

        ggml_cgraph * gf = ggml_new_graph_custom(ggml_ctx, kEncoderGraphNodes, false);
        ggml_build_forward_expand(gf, logits.tensor);

        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            throw std::runtime_error("Failed to allocate GGML graph for Granite 5 ASR encoder");
        }

        ggml_backend_tensor_set(
            in_tensor.tensor,
            features.values.data(),
            0,
            features.values.size() * sizeof(float));

        if (ggml_backend_graph_compute(execution_context_->backend(), gf) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to compute GGML graph for Granite 5 ASR encoder");
        }

        const int64_t out_frames = logits.shape.dims[1];
        const int64_t vocab_size = config.vocab_size;
        std::vector<float> logits_data(static_cast<size_t>(out_frames * vocab_size));
        ggml_backend_tensor_get(
            logits.tensor,
            logits_data.data(),
            0,
            logits_data.size() * sizeof(float));

        token_ids.reserve(static_cast<size_t>(out_frames));
        for (int64_t t = 0; t < out_frames; ++t) {
            const float * frame_logits = &logits_data[static_cast<size_t>(t * vocab_size)];
            int32_t best_id = 0;
            float max_val = frame_logits[0];
            for (int32_t v = 1; v < static_cast<int32_t>(vocab_size); ++v) {
                if (frame_logits[v] > max_val) {
                    max_val = frame_logits[v];
                    best_id = v;
                }
            }
            token_ids.push_back(best_id);
        }
    } catch (...) {
        ggml_gallocr_free(galloc);
        ggml_free(ggml_ctx);
        throw;
    }

    ggml_gallocr_free(galloc);
    ggml_free(ggml_ctx);
    return token_ids;
}

}  // namespace engine::community_models::granite5asr
