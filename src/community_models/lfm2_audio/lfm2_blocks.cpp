#include "lfm2_blocks.h"

#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::community_models::lfm2_audio::lfm2_blocks {

using core::TensorShape;
using core::TensorValue;

bool backend_gathers(ggml_backend_t backend, ggml_type type) {
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize the LFM2-Audio op probe context");
    }

    auto * table = ggml_new_tensor_2d(ctx.get(), type, ggml_blck_size(type), 1);
    auto * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    return ggml_backend_supports_op(backend, ggml_get_rows(ctx.get(), table, ids));
}

std::vector<LayerWeights> load_layers(core::BackendWeightStore & store, const assets::TensorSource & source, const Lfm2BackboneConfig & config) {
    const auto native = assets::TensorStorageType::Native;
    // The projections are only ever src0 of LinearModule's ggml_mul_mat.
    const auto matmul = core::WeightUse::MatMulOnly;
    const int64_t d = config.hidden_size;
    const int64_t ff = config.intermediate_size;
    const int64_t hd = config.head_dim;

    std::vector<LayerWeights> layers;
    for (int64_t layer = 0; layer < config.num_layers(); ++layer) {
        const std::string p = "blk." + std::to_string(layer) + ".";
        LayerWeights w;
        w.attention = config.is_attention_layer(layer);

        w.decoder.input_norm = {store.load_f32_tensor(source, p + "attn_norm.weight", {d}), std::nullopt};
        w.decoder.post_norm = {store.load_f32_tensor(source, p + "ffn_norm.weight", {d}), std::nullopt};
        w.decoder.mlp.gate_proj = {store.load_tensor(source, p + "ffn_gate.weight", native, {ff, d}, matmul), std::nullopt};
        w.decoder.mlp.up_proj = {store.load_tensor(source, p + "ffn_up.weight", native, {ff, d}, matmul), std::nullopt};
        w.decoder.mlp.down_proj = {store.load_tensor(source, p + "ffn_down.weight", native, {d, ff}, matmul), std::nullopt};

        if (w.attention) {
            const int64_t kv = config.kv_heads[static_cast<size_t>(layer)] * hd;
            w.decoder.self_attention.q_weight = store.load_tensor(source, p + "attn_q.weight", native, {config.num_attention_heads * hd, d}, matmul);
            w.decoder.self_attention.k_weight = store.load_tensor(source, p + "attn_k.weight", native, {kv, d}, matmul);
            w.decoder.self_attention.v_weight = store.load_tensor(source, p + "attn_v.weight", native, {kv, d}, matmul);
            w.decoder.self_attention.out_weight = store.load_tensor(source, p + "attn_output.weight", native, {d, config.num_attention_heads * hd}, matmul);
            w.decoder.q_norm = {store.load_f32_tensor(source, p + "attn_q_norm.weight", {hd}), std::nullopt};
            w.decoder.k_norm = {store.load_f32_tensor(source, p + "attn_k_norm.weight", {hd}), std::nullopt};
        } else {
            w.conv.in_proj = store.load_tensor(source, p + "shortconv.in_proj.weight", native, {3 * d, d}, matmul);
            w.conv.out_proj = store.load_tensor(source, p + "shortconv.out_proj.weight", native, {d, d}, matmul);
            w.conv.kernel = store.load_f32_tensor(source, p + "shortconv.conv.weight", {d, config.conv_kernel_size});
        }

        layers.push_back(std::move(w));
    }

    return layers;
}

modules::DecoderLayerConfig attention_layer_config(const Lfm2BackboneConfig & config, int64_t layer) {
    modules::DecoderLayerConfig out;
    out.hidden_size = config.hidden_size;
    out.num_attention_heads = config.num_attention_heads;
    out.num_key_value_heads = config.kv_heads[static_cast<size_t>(layer)];
    out.head_dim = config.head_dim;
    out.intermediate_size = config.intermediate_size;
    out.rms_norm_eps = config.rms_norm_eps;
    out.rope_theta = config.rope_theta;
    out.rope_type = GGML_ROPE_TYPE_NEOX;
    out.use_qk_norm = true;
    out.runtime.static_cache.update_mode = modules::DecoderStaticCacheUpdateMode::DirectSetRows;
    return out;
}

TensorValue rms_norm(core::ModuleBuildContext & ctx, const TensorValue & x, const modules::NormWeights & weights, const Lfm2BackboneConfig & config) {
    return modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false}).build(ctx, x, weights);
}

TensorValue contiguous(core::ModuleBuildContext & ctx, const TensorValue & x) {
    return core::wrap_tensor(ggml_cont(ctx.ggml, x.tensor), x.shape, x.type);
}

ShortConvInput short_conv_input(core::ModuleBuildContext & ctx, const TensorValue & normed, const ShortConvWeights & weights, int64_t d) {
    auto bcx = modules::LinearModule({d, 3 * d, false}).build(ctx, normed, {weights.in_proj, std::nullopt});
    auto b = contiguous(ctx, modules::SliceModule({2, 0, d}).build(ctx, bcx));
    auto c = contiguous(ctx, modules::SliceModule({2, d, d}).build(ctx, bcx));
    auto x = contiguous(ctx, modules::SliceModule({2, 2 * d, d}).build(ctx, bcx));
    auto bx = modules::MulModule{}.build(ctx, b, x);
    bx = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, bx);
    return {c, contiguous(ctx, bx)};
}

TensorValue short_conv_output(
    core::ModuleBuildContext & ctx,
    const TensorValue & residual,
    const TensorValue & conv,
    const TensorValue & gate,
    const ShortConvWeights & weights,
    int64_t d) {
    auto y = modules::MulModule{}.build(ctx, gate, conv);
    y = modules::LinearModule({d, d, false}).build(ctx, y, {weights.out_proj, std::nullopt});
    return modules::AddModule{}.build(ctx, residual, y);
}

TensorValue feed_forward(core::ModuleBuildContext & ctx, const TensorValue & x, const LayerWeights & weights, const Lfm2BackboneConfig & config) {
    auto h = rms_norm(ctx, x, weights.decoder.post_norm, config);

    modules::GatedFeedForwardConfig ff_config;
    ff_config.hidden_size = config.hidden_size;
    ff_config.intermediate_size = config.intermediate_size;
    ff_config.activation = modules::GatedFeedForwardActivation::Silu;
    h = modules::GatedFeedForwardModule(ff_config).build(
        ctx, h, {weights.decoder.mlp.gate_proj, weights.decoder.mlp.up_proj, weights.decoder.mlp.down_proj});
    return modules::AddModule{}.build(ctx, x, h);
}

TensorValue conv_kernel(core::ModuleBuildContext & ctx, const ShortConvWeights & weights, const Lfm2BackboneConfig & config) {
    return core::reshape_tensor(ctx, weights.kernel, TensorShape::from_dims({config.hidden_size, config.conv_kernel_size}));
}

TensorValue build_sequence(
    core::ModuleBuildContext & ctx,
    TensorValue x,
    const TensorValue & positions,
    const std::vector<LayerWeights> & layers,
    const Lfm2BackboneConfig & config,
    const std::optional<TensorValue> & mask,
    SequenceTaps * taps) {
    auto * g = ctx.ggml;
    const int64_t d = config.hidden_size;
    const int64_t k = config.conv_kernel_size;
    const int64_t steps = x.shape.dims[1];

    for (int64_t layer = 0; layer < config.num_layers(); ++layer) {
        const auto & w = layers[static_cast<size_t>(layer)];
        if (w.attention) {
            // Without a mask argument the layer applies ggml_diag_mask_inf,
            // which is the causal mask.
            auto out = modules::DecoderLayerModule(attention_layer_config(config, layer))
                           .build(ctx, x, positions, w.decoder, std::nullopt, std::nullopt, mask);
            x = out.output;
            if (taps != nullptr) {
                taps->keys.push_back(out.key.tensor);
                taps->values.push_back(out.value.tensor);
            }

            continue;
        }

        auto in = short_conv_input(ctx, rms_norm(ctx, x, w.decoder.input_norm, config), w.conv, d);
        // Left-pad with the kernel's history: a sequence starts from zeros.
        auto padded = core::wrap_tensor(
            ggml_pad_ext(g, in.conv_in.tensor, static_cast<int>(k - 1), 0, 0, 0, 0, 0, 0, 0),
            TensorShape::from_dims({1, d, steps + k - 1}),
            GGML_TYPE_F32);
        auto conv = core::wrap_tensor(
            ggml_ssm_conv(g, padded.tensor, conv_kernel(ctx, w.conv, config).tensor),
            TensorShape::from_dims({1, steps, d}),
            GGML_TYPE_F32);

        x = short_conv_output(ctx, x, conv, in.gate, w.conv, d);
        x = feed_forward(ctx, x, w, config);

        if (taps != nullptr) {
            taps->conv_tails.push_back(contiguous(ctx, modules::SliceModule({2, steps, k - 1}).build(ctx, padded)).tensor);
        }
    }

    return x;
}

}  // namespace engine::community_models::lfm2_audio::lfm2_blocks
