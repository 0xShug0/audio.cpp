#include "engine/framework/modules/attention/scaled_dot_product_attention.h"

#include "attention_internal.h"

#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <cmath>
#include <stdexcept>

namespace engine::modules {
namespace {

inline const core::ModulePortSpec kSdpaInputs[] = {
    {"q_heads", core::PortKind::Activation, false},
    {"k_heads", core::PortKind::Activation, false},
    {"v_heads", core::PortKind::Activation, false},
};

inline const core::ModulePortSpec kSdpaOutputs[] = {
    {"context", core::PortKind::Activation, false},
};

inline const core::ModuleSchema kScaledDotProductAttentionSchema = {
    "ScaledDotProductAttention",
    "nn.attention",
    kSdpaInputs,
    3,
    kSdpaOutputs,
    1,
    "Applies scaled dot-product attention from pre-split [batch, heads, steps, dim] Q/K/V tensors and returns [batch, steps, heads, dim].",
};

void validate_heads(const core::TensorValue & q, const core::TensorValue & k, const core::TensorValue & v, int64_t dim) {
    core::validate_rank_between(q, 4, 4, "q_heads");
    core::validate_rank_between(k, 4, 4, "k_heads");
    core::validate_rank_between(v, 4, 4, "v_heads");
    if (q.shape.dims[0] != k.shape.dims[0] || q.shape.dims[0] != v.shape.dims[0]) {
        throw std::runtime_error("ScaledDotProductAttention batch size mismatch");
    }
    if (q.shape.dims[1] != k.shape.dims[1] || q.shape.dims[1] != v.shape.dims[1]) {
        throw std::runtime_error("ScaledDotProductAttention head count mismatch");
    }
    if (k.shape.dims[2] != v.shape.dims[2]) {
        throw std::runtime_error("ScaledDotProductAttention key/value step mismatch");
    }
    if (q.shape.dims[3] != dim || k.shape.dims[3] != dim || v.shape.dims[3] != dim) {
        throw std::runtime_error("ScaledDotProductAttention head dimension mismatch");
    }
}

core::TensorValue build_explicit(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    const std::optional<core::TensorValue> & attention_mask,
    float scale) {
    const MatMulModule matmul;
    auto scores = matmul.build(ctx, q_heads, TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    scores = core::ensure_backend_addressable_layout(ctx, scores);
    core::TensorValue attn;
    if (attention_mask.has_value()) {
        attn = core::wrap_tensor(
            ggml_soft_max_ext(ctx.ggml, scores.tensor, attention_mask->tensor, scale, 0.0F),
            scores.shape,
            GGML_TYPE_F32);
    } else {
        scores = core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, scale), scores.shape, GGML_TYPE_F32);
        attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor), scores.shape, GGML_TYPE_F32);
    }
    auto context = matmul.build(ctx, attn, v_heads);
    return TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
}

core::TensorValue build_flash(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    const std::optional<core::TensorValue> & attention_mask,
    float scale,
    ggml_prec precision) {
    const auto q = core::ensure_backend_addressable_layout(ctx, q_heads);
    const auto k = core::ensure_backend_addressable_layout(ctx, k_heads);
    const auto v = core::ensure_backend_addressable_layout(ctx, v_heads);
    ggml_tensor * mask = attention_mask.has_value() ? attention_mask->tensor : nullptr;
    auto * flash = ggml_flash_attn_ext(ctx.ggml, q.tensor, k.tensor, v.tensor, mask, scale, 0.0F, 0.0F);
    ggml_flash_attn_ext_set_prec(flash, precision);
    return core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[2], q.shape.dims[1], q.shape.dims[3]}),
        GGML_TYPE_F32);
}

}  // namespace

ScaledDotProductAttentionModule::ScaledDotProductAttentionModule(ScaledDotProductAttentionConfig config)
    : config_(config) {
    if (config_.head_dim <= 0) {
        throw std::runtime_error("ScaledDotProductAttentionConfig.head_dim must be positive");
    }
}

const ScaledDotProductAttentionConfig & ScaledDotProductAttentionModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & ScaledDotProductAttentionModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue ScaledDotProductAttentionModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    const std::optional<core::TensorValue> & attention_mask) const {
    validate_heads(q_heads, k_heads, v_heads, config_.head_dim);
    const float scale = 1.0F / std::sqrt(static_cast<float>(config_.head_dim));
    switch (config_.lowering) {
        case ScaledDotProductAttentionLowering::Explicit:
            return build_explicit(ctx, q_heads, k_heads, v_heads, attention_mask, scale);
        case ScaledDotProductAttentionLowering::Flash:
            return build_flash(ctx, q_heads, k_heads, v_heads, attention_mask, scale, config_.precision);
    }
    throw std::runtime_error("Unsupported scaled dot-product attention lowering");
}

const core::ModuleSchema & ScaledDotProductAttentionModule::static_schema() noexcept {
    return kScaledDotProductAttentionSchema;
}

}  // namespace engine::modules
