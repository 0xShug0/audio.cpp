#include "engine/framework/modules/ebranchformer_modules.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/cross_attention.h"
#include "engine/framework/modules/attention/self_attention.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

namespace engine::modules {
using core::TensorValue;

TensorValue EBranchformerBlockModule::build(
    core::ModuleBuildContext & ctx, TensorValue input,
    const EBranchformerBlockWeights & weights, const TensorValue & half_scale,
    const TensorValue * memory, const TensorValue * memory_mask) const {
    const auto hidden_size = config_.hidden_size;
    const auto num_heads = config_.num_heads;
    const auto intermediate_size = config_.intermediate_size;
    const LayerNormModule norm({hidden_size, config_.norm_eps});
    const LayerScaleModule scale;
    const ResidualAddModule residual;

    auto transformed = norm.build(ctx, input, weights.macaron_norm);
    transformed = LinearModule({hidden_size, intermediate_size, true}).build(
        ctx, transformed, {weights.macaron_ffn.fc1_weight, weights.macaron_ffn.fc1_bias});
    transformed = SiluModule().build(ctx, transformed);
    transformed = LinearModule({intermediate_size, hidden_size, true}).build(
        ctx, transformed, {weights.macaron_ffn.fc2_weight, weights.macaron_ffn.fc2_bias});
    transformed = scale.build(ctx, transformed, {half_scale});
    auto x = residual.build(ctx, input, transformed);

    auto attention_input = norm.build(ctx, x, weights.attention_norm);
    AttentionConfig attention_config{hidden_size, num_heads, true};
    attention_config.use_packed_qkv = config_.use_packed_qkv;
    auto attention = SelfAttentionModule(attention_config).build(ctx, attention_input, weights.attention);

    auto cgmlp = norm.build(ctx, x, weights.cgmlp_norm);
    cgmlp = LinearModule({hidden_size, intermediate_size, true})
        .build(ctx, cgmlp, weights.cgmlp.input_projection);
    cgmlp = GeluModule({GeluApproximation::ExactErf}).build(ctx, cgmlp);
    auto value = SliceModule({2, 0, intermediate_size / 2}).build(ctx, cgmlp);
    auto gate = SliceModule({2, intermediate_size / 2, intermediate_size / 2}).build(ctx, cgmlp);
    gate = LayerNormModule({intermediate_size / 2, config_.norm_eps})
        .build(ctx, gate, weights.cgmlp.gate_norm);
    gate = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, gate);
    gate = DepthwiseConv1dModule({intermediate_size / 2, config_.gate_kernel_size, 1, config_.gate_kernel_size / 2, 1, true})
        .build(ctx, gate, weights.cgmlp.gate_conv);
    gate = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, gate);
    cgmlp = MulModule().build(ctx, value, gate);
    cgmlp = LinearModule({intermediate_size / 2, hidden_size, true})
        .build(ctx, cgmlp, weights.cgmlp.output_projection);

    auto merged = ConcatModule({2}).build(ctx, attention, cgmlp);
    auto convolved = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, merged);
    convolved = DepthwiseConv1dModule({hidden_size * 2, config_.merge_kernel_size, 1, config_.merge_kernel_size / 2, 1, true})
        .build(ctx, convolved, weights.merge_conv);
    convolved = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, convolved);
    merged = AddModule().build(ctx, merged, convolved);
    merged = LinearModule({hidden_size * 2, hidden_size, true})
        .build(ctx, merged, weights.merge_projection);
    x = residual.build(ctx, x, merged);

    transformed = norm.build(ctx, x, weights.final_ffn_norm);
    transformed = LinearModule({hidden_size, intermediate_size, true}).build(
        ctx, transformed, {weights.final_ffn.fc1_weight, weights.final_ffn.fc1_bias});
    transformed = SiluModule().build(ctx, transformed);
    transformed = LinearModule({intermediate_size, hidden_size, true}).build(
        ctx, transformed, {weights.final_ffn.fc2_weight, weights.final_ffn.fc2_bias});
    transformed = scale.build(ctx, transformed, {half_scale});
    x = residual.build(ctx, x, transformed);
    if (memory) {
        transformed = norm.build(ctx, x, weights.cross_attention_norm);
        AttentionConfig cross_config{hidden_size, num_heads, true};
        cross_config.use_packed_kv = config_.use_packed_kv;
        transformed = CrossAttentionModule(cross_config)
            .build(ctx, transformed, *memory, weights.cross_attention, *memory_mask);
        x = residual.build(ctx, x, transformed);
    }
    return norm.build(ctx, x, weights.output_norm);
}

}  // namespace engine::modules
