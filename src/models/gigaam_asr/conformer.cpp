#include "engine/models/gigaam_asr/conformer.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <cmath>

namespace engine::models::gigaam_asr {

core::TensorValue build_conformer_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & features,
    const core::TensorValue & positions,
    const ConformerConfig & config,
    const ConformerWeights & weights) {
    using core::TensorShape;
    using namespace modules;
    const int64_t hidden = config.hidden_size;
    const int64_t head_dim = hidden / config.heads;
    const LayerNormModule norm({hidden, 1e-5f, true, true});
    const LinearModule project({hidden, hidden, true});
    const LinearModule expand({hidden, config.intermediate_size, true});
    const LinearModule contract({config.intermediate_size, hidden, true});
    const AddModule add;
    const SiluModule silu;

    auto x = Conv1dModule({config.features, hidden, config.subsampling_kernel,
        2, static_cast<int>((config.subsampling_kernel - 1) / 2), 1, true})
        .build(ctx, features, weights.subsampling1);
    x = ReluModule().build(ctx, x);
    x = Conv1dModule({hidden, hidden, config.subsampling_kernel,
        2, static_cast<int>((config.subsampling_kernel - 1) / 2), 1, true})
        .build(ctx, x, weights.subsampling2);
    x = ReluModule().build(ctx, x);
    x = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    const int64_t frames = x.shape.dims[1];
    const auto split_shape = TensorShape::from_dims({1, frames, config.heads, head_dim});
    const auto sequence_shape = TensorShape::from_dims({1, frames, hidden});

    for (const auto & w : weights.layers) {
        auto y = norm.build(ctx, x, w.ffn1_norm);
        y = expand.build(ctx, y, w.ffn1_in);
        y = silu.build(ctx, y);
        y = contract.build(ctx, y, w.ffn1_out);
        y = core::wrap_tensor(ggml_scale(ctx.ggml, y.tensor, 0.5f), y.shape, y.type);
        x = add.build(ctx, x, y);

        y = norm.build(ctx, x, w.attention_norm);
        auto rotated = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, y), split_shape);
        // GigaAM rotates the input before the Q/K projections. V stays unrotated.
        rotated = RoPEModule({head_dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, rotated, positions);
        rotated = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, rotated), sequence_shape);
        auto q = project.build(ctx, rotated, w.query);
        auto k = project.build(ctx, rotated, w.key);
        auto v = project.build(ctx, y, w.value);
        q = core::reshape_tensor(ctx, q, split_shape);
        k = core::reshape_tensor(ctx, k, split_shape);
        v = core::reshape_tensor(ctx, v, split_shape);
        q = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
        k = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
        v = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
        int64_t attention_dim = head_dim;
        if (head_dim == 48 && config.attention == ScaledDotProductAttentionLowering::Flash &&
            ctx.backend_type != core::BackendType::Cpu) {
            // The flash kernels support width 64, not 48. Preserve the original
            // 1/sqrt(48) scale while the additional channels contribute zero.
            attention_dim = 64;
            q = Pad2dModule({0, 16, 0, 0}).build(ctx, q);
            k = Pad2dModule({0, 16, 0, 0}).build(ctx, k);
            v = Pad2dModule({0, 16, 0, 0}).build(ctx, v);
            q = core::wrap_tensor(ggml_scale(ctx.ggml, q.tensor, std::sqrt(64.0f / 48.0f)), q.shape, q.type);
        }
        y = ScaledDotProductAttentionModule({attention_dim, config.attention}).build(ctx, q, k, v);
        if (attention_dim != head_dim) {
            y = SliceModule({3, 0, head_dim}).build(ctx, y);
        }
        y = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, y), sequence_shape);
        y = project.build(ctx, y, w.attention_out);
        x = add.build(ctx, x, y);

        y = norm.build(ctx, x, w.conv_norm);
        y = LinearModule({hidden, 2 * hidden, true}).build(ctx, y, w.pointwise_in);
        y = GLUModule({true}).build(ctx, y);
        y = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, y);
        y = DepthwiseConv1dModule({hidden, config.conv_kernel, 1,
            static_cast<int>((config.conv_kernel - 1) / 2), 1, true}).build(ctx, y, w.depthwise);
        y = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, y);
        y = norm.build(ctx, y, w.depthwise_norm);
        y = silu.build(ctx, y);
        y = project.build(ctx, y, w.pointwise_out);
        x = add.build(ctx, x, y);

        y = norm.build(ctx, x, w.ffn2_norm);
        y = expand.build(ctx, y, w.ffn2_in);
        y = silu.build(ctx, y);
        y = contract.build(ctx, y, w.ffn2_out);
        y = core::wrap_tensor(ggml_scale(ctx.ggml, y.tensor, 0.5f), y.shape, y.type);
        x = add.build(ctx, x, y);
        x = norm.build(ctx, x, w.output_norm);
    }
    return x;
}

}  // namespace engine::models::gigaam_asr
