#include "engine/framework/modules/optimizations/fast_projection_modules.h"

#include <stdexcept>

namespace engine::modules {

namespace {

const core::ModulePortSpec kPackedProjection4Inputs[] = {
    {"input", core::PortKind::Activation, false},
    {"weight", core::PortKind::Parameter, false},
};

const core::ModulePortSpec kSingleOutput[] = {
    {"output", core::PortKind::Activation, false},
};

const core::ModuleSchema kFastPackedProjection4Schema = {
    "FastPackedProjection4",
    "tensor.optimization",
    kPackedProjection4Inputs,
    2,
    kSingleOutput,
    1,
    "Applies the MiniTTS optimized 4-way packed linear projection.",
};

core::TensorShape flatten_to_matrix_shape(const core::TensorShape & shape) {
    if (shape.rank == 1) {
        return core::TensorShape::from_dims({1, shape.last_dim()});
    }
    return core::TensorShape::from_dims({shape.prefix_elements(), shape.last_dim()});
}

}  // namespace

FastPackedProjection4Module::FastPackedProjection4Module(FastPackedProjection4Config config)
    : config_(config) {
    if (config_.in_features <= 0 || config_.out_features <= 0) {
        throw std::runtime_error("FastPackedProjection4Module features must be positive");
    }
    if (config_.out_features % 4 != 0) {
        throw std::runtime_error("FastPackedProjection4Module out_features must be divisible by 4");
    }
}

const FastPackedProjection4Config & FastPackedProjection4Module::config() const noexcept {
    return config_;
}

const core::ModuleSchema & FastPackedProjection4Module::schema() const noexcept {
    return static_schema();
}

core::TensorValue FastPackedProjection4Module::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const LinearWeights & weights) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    if (!core::uses_ggml_cuda_or_hip_backend(ctx.backend_type)) {
        throw std::runtime_error("FastPackedProjection4Module requires CUDA or HIP backend");
    }

    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    core::validate_last_dim(input, config_.in_features, "input");
    core::validate_shape(
        weights.weight,
        core::TensorShape::from_dims({config_.out_features, config_.in_features}),
        "weight");
    if (weights.bias.has_value()) {
        throw std::runtime_error("FastPackedProjection4Module does not support bias");
    }

    const auto contiguous_input = core::ensure_backend_addressable_layout(ctx, input);
    const auto matrix_input_shape = flatten_to_matrix_shape(contiguous_input.shape);
    const auto matrix_input = core::reshape_tensor(ctx, contiguous_input, matrix_input_shape);

    ggml_tensor * projected_raw =
        ggml_mul_mat_pack4(ctx.ggml, weights.weight.tensor, matrix_input.tensor);
    if (config_.precision != GGML_PREC_DEFAULT) {
        ggml_mul_mat_set_prec(projected_raw, config_.precision);
    }

    auto projected = core::wrap_tensor(
        projected_raw,
        core::TensorShape::from_dims({matrix_input_shape.at(0), config_.out_features}),
        GGML_TYPE_F32);
    return core::reshape_tensor(ctx, projected, input.shape.with_last_dim(config_.out_features));
}

const core::ModuleSchema & FastPackedProjection4Module::static_schema() noexcept {
    return kFastPackedProjection4Schema;
}

}  // namespace engine::modules
