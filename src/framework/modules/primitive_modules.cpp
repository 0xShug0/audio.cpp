#include "engine/framework/modules/primitive_modules.h"
#include "tensor_layout_utils.h"

#include <algorithm>
#include <stdexcept>

namespace engine::modules {

namespace {

const core::ModulePortSpec kBinaryInputs[] = {
    {"lhs", core::PortKind::Activation, false},
    {"rhs", core::PortKind::Activation, false},
};

const core::ModulePortSpec kSingleInput[] = {
    {"input", core::PortKind::Activation, false},
};

const core::ModulePortSpec kResidualInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"residual", core::PortKind::Activation, false},
};

const core::ModulePortSpec kSingleOutput[] = {
    {"output", core::PortKind::Activation, false},
};

const core::ModuleSchema kAddSchema = {
    "Add",
    "tensor.primitive",
    kBinaryInputs,
    2,
    kSingleOutput,
    1,
    "Elementwise addition for tensors with the same logical shape.",
};

const core::ModuleSchema kMatMulSchema = {
    "MatMul",
    "tensor.primitive",
    kBinaryInputs,
    2,
    kSingleOutput,
    1,
    "Performs batched matrix multiplication with logical shapes [..., M, K] x [..., K, N] -> [..., M, N].",
};

const core::ModuleSchema kMulSchema = {
    "Mul",
    "tensor.primitive",
    kBinaryInputs,
    2,
    kSingleOutput,
    1,
    "Elementwise multiplication for tensors with the same logical shape.",
};

const core::ModulePortSpec kTimeMask4dInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"mask", core::PortKind::Activation, false},
};

const core::ModuleSchema kTimeMask4dSchema = {
    "TimeMask4d",
    "tensor.mask",
    kTimeMask4dInputs,
    2,
    kSingleOutput,
    1,
    "Applies a [batch, time] keep mask to a [batch, channels, time, features] tensor.",
};

const core::ModuleSchema kResidualAddSchema = {
    "ResidualAdd",
    "nn.primitive",
    kResidualInputs,
    2,
    kSingleOutput,
    1,
    "Adds a residual tensor to the main input.",
};

const core::ModuleSchema kReduceMeanSchema = {
    "ReduceMean",
    "tensor.primitive",
    kSingleInput,
    1,
    kSingleOutput,
    1,
    "Reduces one logical axis by averaging values.",
};

const core::ModuleSchema kReduceSumSchema = {
    "ReduceSum",
    "tensor.primitive",
    kSingleInput,
    1,
    kSingleOutput,
    1,
    "Reduces one logical axis by summing values.",
};

void validate_same_shape(const core::TensorValue & lhs, const core::TensorValue & rhs, const char * op_name) {
    core::validate_shape(rhs, lhs.shape, op_name);
}

core::TensorShape transpose_last_two(const core::TensorShape & shape) {
    core::TensorShape result = shape;
    const size_t last = result.rank - 1;
    const size_t second_last = result.rank - 2;
    std::swap(result.dims[second_last], result.dims[last]);
    return result;
}

core::TensorValue transpose_last_two(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    std::array<int, core::kMaxTensorRank> axes = {0, 1, 2, 3};
    const size_t last = value.shape.rank - 1;
    const size_t second_last = value.shape.rank - 2;
    axes[second_last] = static_cast<int>(last);
    axes[last] = static_cast<int>(second_last);

    std::array<int, core::kMaxTensorRank> ggml_axes = {0, 1, 2, 3};
    for (size_t out_logical_axis = 0; out_logical_axis < value.shape.rank; ++out_logical_axis) {
        const int in_logical_axis = axes[out_logical_axis];
        const int out_ggml_axis = static_cast<int>(value.shape.rank) - 1 - static_cast<int>(out_logical_axis);
        ggml_axes[out_ggml_axis] = core::logical_axis_to_ggml_axis(value.shape.rank, in_logical_axis);
    }

    return core::wrap_tensor(
        ggml_permute(ctx.ggml, value.tensor, ggml_axes[0], ggml_axes[1], ggml_axes[2], ggml_axes[3]),
        transpose_last_two(value.shape),
        value.type);
}

int normalize_reduce_axis(int axis, size_t rank) {
    if (rank == 0) {
        throw std::runtime_error("Reduce input rank must be positive");
    }
    const int normalized = axis < 0 ? static_cast<int>(rank) + axis : axis;
    if (normalized < 0 || normalized >= static_cast<int>(rank)) {
        throw std::runtime_error("Reduce axis out of range");
    }
    return normalized;
}

core::TensorShape reduced_shape(const core::TensorShape & shape, int axis) {
    core::TensorShape out = shape;
    out.dims[static_cast<size_t>(axis)] = 1;
    return out;
}

template <typename Fn>
core::TensorValue reduce_axis(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ReduceConfig & config,
    Fn fn) {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 1, core::kMaxTensorRank, "input");
    const int axis = normalize_reduce_axis(config.axis, input.shape.rank);
    if (axis == static_cast<int>(input.shape.rank - 1)) {
        const auto contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, input);
        return core::wrap_tensor(fn(ctx.ggml, contiguous.tensor), reduced_shape(input.shape, axis), GGML_TYPE_F32);
    }

    std::array<int, core::kMaxTensorRank> axes = {0, 1, 2, 3};
    std::swap(axes[static_cast<size_t>(axis)], axes[input.shape.rank - 1]);
    const auto transposed = TransposeModule({axes, input.shape.rank}).build(ctx, input);
    const auto contiguous = tensor_layout::ensure_contiguous_layout_if_needed(ctx, transposed);
    auto reduced = core::wrap_tensor(
        fn(ctx.ggml, contiguous.tensor),
        reduced_shape(transposed.shape, static_cast<int>(transposed.shape.rank - 1)),
        GGML_TYPE_F32);
    return TransposeModule({axes, reduced.shape.rank}).build(ctx, reduced);
}

}  // namespace

const core::ModuleSchema & MatMulModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue MatMulModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(lhs, 2, core::kMaxTensorRank, "lhs");
    core::validate_rank_between(rhs, lhs.shape.rank, lhs.shape.rank, "rhs");

    const size_t rank = lhs.shape.rank;
    for (size_t i = 0; i + 2 < rank; ++i) {
        if (lhs.shape.dims[i] != rhs.shape.dims[i]) {
            throw std::runtime_error("MatMul batch dimensions must match");
        }
    }
    if (lhs.shape.dims[rank - 1] != rhs.shape.dims[rank - 2]) {
        throw std::runtime_error("MatMul inner dimensions must match");
    }

    auto rhs_transposed = transpose_last_two(ctx, rhs);
    rhs_transposed = tensor_layout::ensure_contiguous_nontransposed_layout_if_needed(ctx, rhs_transposed);
    core::TensorShape output_shape = lhs.shape;
    output_shape.dims[rank - 1] = rhs.shape.dims[rank - 1];
    ggml_tensor * output = ggml_mul_mat(ctx.ggml, rhs_transposed.tensor, lhs.tensor);
    return core::wrap_tensor(output, output_shape, GGML_TYPE_F32);
}

const core::ModuleSchema & MatMulModule::static_schema() noexcept {
    return kMatMulSchema;
}

const core::ModuleSchema & AddModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue AddModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    validate_same_shape(lhs, rhs, "Add rhs");
    const auto lhs_contiguous = tensor_layout::ensure_contiguous_nontransposed_layout_if_needed(ctx, lhs);
    const auto rhs_contiguous = tensor_layout::ensure_contiguous_nontransposed_layout_if_needed(ctx, rhs);
    return core::wrap_tensor(ggml_add(ctx.ggml, lhs_contiguous.tensor, rhs_contiguous.tensor), lhs.shape, GGML_TYPE_F32);
}

const core::ModuleSchema & AddModule::static_schema() noexcept {
    return kAddSchema;
}

const core::ModuleSchema & MulModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue MulModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    validate_same_shape(lhs, rhs, "Mul rhs");
    const auto lhs_contiguous = tensor_layout::ensure_contiguous_nontransposed_layout_if_needed(ctx, lhs);
    const auto rhs_contiguous = tensor_layout::ensure_contiguous_nontransposed_layout_if_needed(ctx, rhs);
    return core::wrap_tensor(ggml_mul(ctx.ggml, lhs_contiguous.tensor, rhs_contiguous.tensor), lhs.shape, GGML_TYPE_F32);
}

const core::ModuleSchema & MulModule::static_schema() noexcept {
    return kMulSchema;
}

const core::ModuleSchema & TimeMask4dModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue TimeMask4dModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & mask) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 4, 4, "input");
    core::validate_shape(
        mask,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[2]}),
        "time_mask");
    const auto mask_f32 = core::wrap_tensor(ggml_cast(ctx.ggml, mask.tensor, GGML_TYPE_F32), mask.shape, GGML_TYPE_F32);
    const auto mask_4d = core::reshape_tensor(
        ctx,
        mask_f32,
        core::TensorShape::from_dims({input.shape.dims[0], 1, input.shape.dims[2], 1}));
    const auto broadcast = core::wrap_tensor(ggml_repeat(ctx.ggml, mask_4d.tensor, input.tensor), input.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_mul(ctx.ggml, input.tensor, broadcast.tensor), input.shape, GGML_TYPE_F32);
}

const core::ModuleSchema & TimeMask4dModule::static_schema() noexcept {
    return kTimeMask4dSchema;
}

const core::ModuleSchema & ResidualAddModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue ResidualAddModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & residual) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    validate_same_shape(input, residual, "ResidualAdd residual");
    const auto input_contiguous = tensor_layout::ensure_contiguous_nontransposed_layout_if_needed(ctx, input);
    const auto residual_contiguous = tensor_layout::ensure_contiguous_nontransposed_layout_if_needed(ctx, residual);
    return core::wrap_tensor(ggml_add(ctx.ggml, input_contiguous.tensor, residual_contiguous.tensor), input.shape, GGML_TYPE_F32);
}

const core::ModuleSchema & ResidualAddModule::static_schema() noexcept {
    return kResidualAddSchema;
}

ReduceMeanModule::ReduceMeanModule(ReduceConfig config) : config_(config) {
}

const ReduceConfig & ReduceMeanModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & ReduceMeanModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue ReduceMeanModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    return reduce_axis(ctx, input, config_, ggml_mean);
}

const core::ModuleSchema & ReduceMeanModule::static_schema() noexcept {
    return kReduceMeanSchema;
}

ReduceSumModule::ReduceSumModule(ReduceConfig config) : config_(config) {
}

const ReduceConfig & ReduceSumModule::config() const noexcept {
    return config_;
}

const core::ModuleSchema & ReduceSumModule::schema() const noexcept {
    return static_schema();
}

core::TensorValue ReduceSumModule::build(core::ModuleBuildContext & ctx, const core::TensorValue & input) const {
    return reduce_axis(ctx, input, config_, ggml_sum_rows);
}

const core::ModuleSchema & ReduceSumModule::static_schema() noexcept {
    return kReduceSumSchema;
}

}  // namespace engine::modules
