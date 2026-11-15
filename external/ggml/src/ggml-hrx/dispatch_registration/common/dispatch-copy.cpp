#include "dispatch-copy.h"

#include "dispatch-layout-utils.h"
#include "ggml.h"
#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

namespace ggml::hrx {
namespace {

static constexpr uint64_t         kMaximumCopyElements = uint64_t{ 1 } << 30;
static constexpr KernelCatalogRef kCopyF32Kernel       = GGML_HRX_KERNEL_REF("loom_libs", "ggml_copy_f32");
static constexpr KernelCatalogRef kCopyStridedSourceF32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_copy_strided_source_f32");
static constexpr KernelCatalogRef kConcatDim0F32Kernel = GGML_HRX_KERNEL_REF("loom_libs", "ggml_concat_dim0_f32");
static constexpr KernelCatalogRef kConcatDim0StridedSourceF32Kernel =
    GGML_HRX_KERNEL_REF("loom_libs", "ggml_concat_dim0_strided_source_f32");

static const Value * graph_value(const Graph & graph, ValueId id) {
    return graph.values().find(id);
}

static bool packed_f32_layout(const Value & value) {
    size_t expected_stride = sizeof(float);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (value.nb[i] != expected_stride) {
            return false;
        }
        expected_stride *= static_cast<size_t>(value.ne[i]);
    }
    return true;
}

static bool f32_element_strides(const Value & value) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (value.ne[i] <= 0 || value.nb[i] <= 0 || value.nb[i] % sizeof(float) != 0) {
            return false;
        }
    }
    return value.byte_count % sizeof(float) == 0;
}

static void add_concat_strided_source_parameters(KernelSpecialization & kernel, const Value & lhs, const Value & rhs) {
    static constexpr const char * kPrefix = "ggml.concat_dim0_strided_source_f32.";
    kernel.integer_parameters.emplace("lhs_span", lhs.byte_count / sizeof(float));
    kernel.integer_parameters.emplace("rhs_span", rhs.byte_count / sizeof(float));
    kernel.compile_parameters.emplace(std::string(kPrefix) + "lhs_width", std::to_string(lhs.ne[0]));
    kernel.compile_parameters.emplace(std::string(kPrefix) + "rhs_width", std::to_string(rhs.ne[0]));
    kernel.compile_parameters.emplace(std::string(kPrefix) + "row_count",
                                      std::to_string(lhs.ne[1] * lhs.ne[2] * lhs.ne[3]));
    kernel.compile_parameters.emplace(std::string(kPrefix) + "row_ne1", std::to_string(lhs.ne[1]));
    kernel.compile_parameters.emplace(std::string(kPrefix) + "row_ne2", std::to_string(lhs.ne[2]));
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        kernel.compile_parameters.emplace(std::string(kPrefix) + "lhs_stride" + std::to_string(i),
                                          std::to_string(lhs.nb[i] / sizeof(float)));
        kernel.compile_parameters.emplace(std::string(kPrefix) + "rhs_stride" + std::to_string(i),
                                          std::to_string(rhs.nb[i] / sizeof(float)));
    }
}

static bool make_copy_f32_dispatch(const Value & source, const Value & output, Dispatch & dispatch) {
    if (source.type != GGML_TYPE_F32 || output.type != GGML_TYPE_F32 || !output.contiguous ||
        !packed_f32_layout(output) || !f32_element_strides(source) || source.element_count <= 0 ||
        source.element_count != output.element_count ||
        static_cast<uint64_t>(source.element_count) > kMaximumCopyElements || source.storage == output.storage) {
        return false;
    }

    size_t source_span = source.byte_count;
    if (packed_f32_layout(source)) {
        dispatch.kernel = make_kernel_specialization(kCopyF32Kernel);
        dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
    } else {
        // ValueMap imports ggml_nbytes(), including every strided dimension.
        // The row-oriented span helper requires nb[0] == sizeof(float), which
        // is not the contract of a generic transpose/permute copy.
        const size_t source_span_elements = source_span / sizeof(float);
        if (source_span_elements > kMaximumCopyElements) {
            return false;
        }
        dispatch.kernel = make_kernel_specialization(kCopyStridedSourceF32Kernel);
        dispatch.kernel.integer_parameters.emplace("source_span", source_span_elements);
        dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f32.ne0", std::to_string(source.ne[0]));
        dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f32.ne1", std::to_string(source.ne[1]));
        dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f32.ne2", std::to_string(source.ne[2]));
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            dispatch.kernel.compile_parameters.emplace("ggml.copy_strided_source_f32.stride" + std::to_string(i),
                                                       std::to_string(source.nb[i] / sizeof(float)));
        }
    }
    dispatch.kernel.integer_parameters.emplace("element_count", source.element_count);
    dispatch.bindings.push_back({ source.storage_root, source.storage_offset, source_span });
    dispatch.bindings.push_back({ output.id, 0, output.byte_count });
    return true;
}

static bool match_copy_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_CPY || node->inputs.size() != 2) {
        return false;
    }

    const Value * source = graph_value(context.graph, node->inputs[0]);
    const Value * target = graph_value(context.graph, node->inputs[1]);
    const Value * output = graph_value(context.graph, node->output);
    if (source == nullptr || target == nullptr || output == nullptr || target->type != GGML_TYPE_F32 ||
        !target->contiguous || !packed_f32_layout(*target) || target->element_count != output->element_count ||
        target->byte_count != output->byte_count || source->storage == target->storage) {
        return false;
    }

    Dispatch dispatch;
    if (!make_copy_f32_dispatch(*source, *output, dispatch)) {
        return false;
    }

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_cont_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_CONT || node->inputs.size() != 1) {
        return false;
    }

    const Value * source = graph_value(context.graph, node->inputs[0]);
    const Value * output = graph_value(context.graph, node->output);
    if (source == nullptr || output == nullptr) {
        return false;
    }

    Dispatch dispatch;
    if (!make_copy_f32_dispatch(*source, *output, dispatch)) {
        return false;
    }

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_concat_dim0_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const GraphNode * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_CONCAT || node->inputs.size() != 2) {
        return false;
    }

    const Value * lhs    = graph_value(context.graph, node->inputs[0]);
    const Value * rhs    = graph_value(context.graph, node->inputs[1]);
    const Value * output = graph_value(context.graph, node->output);
    if (lhs == nullptr || rhs == nullptr || output == nullptr || lhs->type != GGML_TYPE_F32 ||
        rhs->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 || !output->contiguous ||
        !packed_f32_layout(*output) || !f32_element_strides(*lhs) || !f32_element_strides(*rhs) || lhs->ne[0] <= 0 ||
        lhs->ne[0] > 65536 || rhs->ne[0] <= 0 || rhs->ne[0] > 65536 || output->ne[0] != lhs->ne[0] + rhs->ne[0] ||
        output->element_count <= 0 || static_cast<uint64_t>(output->element_count) > kMaximumCopyElements ||
        lhs->storage == rhs->storage || lhs->storage == output->storage || rhs->storage == output->storage) {
        return false;
    }
    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        if (lhs->ne[dim] != rhs->ne[dim] || lhs->ne[dim] != output->ne[dim]) {
            return false;
        }
    }

    const int64_t row_count = output->element_count / output->ne[0];
    if (row_count < 1 || row_count > 1048576) {
        return false;
    }

    Dispatch dispatch;
    if (packed_f32_layout(*lhs) && packed_f32_layout(*rhs)) {
        dispatch.kernel = make_kernel_specialization(kConcatDim0F32Kernel);
        dispatch.kernel.workload_specialization = WorkloadSpecialization::Dynamic;
        dispatch.kernel.integer_parameters.emplace("row_count", row_count);
        dispatch.kernel.integer_parameters.emplace("lhs_span", lhs->element_count);
        dispatch.kernel.integer_parameters.emplace("rhs_span", rhs->element_count);
        dispatch.kernel.compile_parameters.emplace("ggml.concat_dim0_f32.lhs_width", std::to_string(lhs->ne[0]));
        dispatch.kernel.compile_parameters.emplace("ggml.concat_dim0_f32.rhs_width", std::to_string(rhs->ne[0]));
    } else {
        const size_t lhs_span = lhs->byte_count / sizeof(float);
        const size_t rhs_span = rhs->byte_count / sizeof(float);
        if (lhs_span > kMaximumCopyElements || rhs_span > kMaximumCopyElements) {
            return false;
        }
        dispatch.kernel = make_kernel_specialization(kConcatDim0StridedSourceF32Kernel);
        add_concat_strided_source_parameters(dispatch.kernel, *lhs, *rhs);
    }
    dispatch.bindings.push_back({ lhs->id, 0, lhs->byte_count });
    dispatch.bindings.push_back({ rhs->id, 0, rhs->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });

    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

static bool match_repeat_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const auto * node = context.root_node;
    if (node == nullptr || node->inputs.size() != 1) return false;
    const bool upscale = node->op == GGML_OP_UPSCALE;
    if (upscale) {
        const auto * params = op_params_as<UpscaleParams>(node->params);
        if (params == nullptr || params->mode != GGML_SCALE_MODE_NEAREST) return false;
    }
    const auto * source = context.graph.values().find(node->inputs[0]);
    const auto * output = context.graph.values().find(node->output);
    if (source == nullptr || output == nullptr || source->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32 || !packed_f32_layout(*output) || !f32_element_strides(*source)) {
        return false;
    }
    const size_t source_span = source->byte_count;
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(GGML_HRX_KERNEL_REF("loom_libs", "ggml_repeat_f32"));
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.integer_parameters.emplace("source_span", source_span / sizeof(float));
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (output->ne[i] % source->ne[i] != 0) return false;
        dispatch.kernel.compile_parameters.emplace("ggml.repeat.scale" + std::to_string(i),
            std::to_string(upscale ? output->ne[i] / source->ne[i] : 1));
        dispatch.kernel.compile_parameters.emplace("ggml.repeat.src_ne" + std::to_string(i), std::to_string(source->ne[i]));
        dispatch.kernel.compile_parameters.emplace("ggml.repeat.stride" + std::to_string(i), std::to_string(source->nb[i] / sizeof(float)));
        if (i < 3) dispatch.kernel.compile_parameters.emplace("ggml.repeat.ne" + std::to_string(i), std::to_string(output->ne[i]));
    }
    dispatch.bindings.push_back({source->storage_root, source->storage_offset, source_span});
    dispatch.bindings.push_back({output->id, 0, output->byte_count});
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_bilinear_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const auto * node = context.root_node;
    const auto * params = op_params_as<UpscaleParams>(node->params);
    if (params == nullptr || node->inputs.size() != 1 ||
        (params->mode & 0xff) != GGML_SCALE_MODE_BILINEAR ||
        (params->mode & ~(0xff | GGML_SCALE_FLAG_ALIGN_CORNERS)) != 0) return false;
    const auto * input = graph_value(context.graph, node->inputs[0]);
    const auto * output = graph_value(context.graph, node->output);
    if (input == nullptr || output == nullptr || input->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32 || !f32_element_strides(*input) || !packed_f32_layout(*output) ||
        input->byte_count / sizeof(float) >= kMaximumCopyElements ||
        output->element_count <= 0 || static_cast<uint64_t>(output->element_count) >= kMaximumCopyElements) return false;
    const bool aligned = (params->mode & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(GGML_HRX_KERNEL_REF("loom_libs", "ggml_bilinear_f32"));
    dispatch.kernel.integer_parameters.emplace("elements", output->element_count);
    dispatch.kernel.integer_parameters.emplace("span", input->byte_count / sizeof(float));
    auto & config = dispatch.kernel.compile_parameters;
    config.emplace("ggml.bilinear.pixel_offset", aligned ? "0.0" : "0.5");
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (input->nb[i] / sizeof(float) >= kMaximumCopyElements || input->ne[i] >= static_cast<int64_t>(kMaximumCopyElements)) return false;
        const auto dim = std::to_string(i);
        config.emplace("ggml.bilinear.stride" + dim, std::to_string(input->nb[i] / sizeof(float)));
        if (i < 3) config.emplace("ggml.bilinear.ne" + dim, std::to_string(output->ne[i]));
        if (i < 2) config.emplace("ggml.bilinear.src_ne" + dim, std::to_string(input->ne[i]));
        float factor = static_cast<float>(output->ne[i]) / input->ne[i];
        if (aligned && i < 2 && output->ne[i] > 1 && input->ne[i] > 1)
            factor = static_cast<float>(output->ne[i] - 1) / (input->ne[i] - 1);
        std::ostringstream value;
        value << std::setprecision(std::numeric_limits<float>::max_digits10) << factor;
        config.emplace("ggml.bilinear.sf" + dim, value.str());
    }
    dispatch.bindings.push_back({input->storage_root, input->storage_offset, input->byte_count});
    dispatch.bindings.push_back({output->id, 0, output->byte_count});
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_pad_f32_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const auto * node = context.root_node;
    if (node == nullptr || node->op != GGML_OP_PAD || node->inputs.size() != 1) {
        return false;
    }
    const auto * source = graph_value(context.graph, node->inputs[0]);
    const auto * output = graph_value(context.graph, node->output);
    const auto * params = op_params_as<PadParams>(node->params);
    if (source == nullptr || output == nullptr || params == nullptr || source->type != GGML_TYPE_F32 ||
        output->type != GGML_TYPE_F32 || !packed_f32_layout(*output) || !f32_element_strides(*source) ||
        output->element_count <= 0 || static_cast<uint64_t>(output->element_count) > kMaximumCopyElements ||
        source->byte_count / sizeof(float) > kMaximumCopyElements || source->storage == output->storage) {
        return false;
    }

    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(GGML_HRX_KERNEL_REF("loom_libs", "ggml_pad_f32"));
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.integer_parameters.emplace("source_span", source->byte_count / sizeof(float));
    dispatch.kernel.compile_parameters.emplace("ggml.pad.circular", params->values[8] != 0 ? "1" : "0");
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        const int64_t left = params->values[2 * i];
        const int64_t right = params->values[2 * i + 1];
        if (source->ne[i] > static_cast<int64_t>(kMaximumCopyElements) ||
            source->nb[i] / sizeof(float) > kMaximumCopyElements || output->ne[i] <= 0 ||
            output->ne[i] != source->ne[i] + left + right) {
            return false;
        }
        const std::string dim = std::to_string(i);
        dispatch.kernel.compile_parameters.emplace("ggml.pad.src_ne" + dim, std::to_string(source->ne[i]));
        dispatch.kernel.compile_parameters.emplace("ggml.pad.stride" + dim,
                                                    std::to_string(source->nb[i] / sizeof(float)));
        dispatch.kernel.compile_parameters.emplace("ggml.pad.left" + dim, std::to_string(left > 0 ? left : 0));
        dispatch.kernel.compile_parameters.emplace("ggml.pad.crop" + dim, std::to_string(left < 0 ? -left : 0));
        // Normalize on the host so circular padding also handles padding wider than the source.
        const int64_t shift = ((-left % source->ne[i]) + source->ne[i]) % source->ne[i];
        dispatch.kernel.compile_parameters.emplace("ggml.pad.shift" + dim, std::to_string(shift));
        if (i < 3) {
            dispatch.kernel.compile_parameters.emplace("ggml.pad.ne" + dim, std::to_string(output->ne[i]));
        }
    }
    dispatch.bindings.push_back({ source->storage_root, source->storage_offset, source->byte_count });
    dispatch.bindings.push_back({ output->id, 0, output->byte_count });
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static Dispatch float_copy_dispatch(const Value * source, const Value * output) {
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(GGML_HRX_KERNEL_REF("loom_libs", "ggml_float_copy"));
    dispatch.kernel.integer_parameters.emplace("element_count", source->element_count);
    dispatch.kernel.integer_parameters.emplace("source_span", source->byte_count / ggml_type_size(source->type));
    dispatch.kernel.integer_parameters.emplace("target_span", output->byte_count / ggml_type_size(output->type));
    dispatch.kernel.compile_parameters.emplace("ggml.float_copy.source_type", std::to_string(source->type));
    dispatch.kernel.compile_parameters.emplace("ggml.float_copy.target_type", std::to_string(output->type));
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        const size_t src_size = ggml_type_size(source->type), dst_size = ggml_type_size(output->type);

        dispatch.kernel.compile_parameters.emplace("ggml.float_copy.src_stride" + std::to_string(i), std::to_string(source->nb[i] / src_size));
        dispatch.kernel.compile_parameters.emplace("ggml.float_copy.dst_stride" + std::to_string(i), std::to_string(output->nb[i] / dst_size));
        if (i < 3) {
            dispatch.kernel.compile_parameters.emplace("ggml.float_copy.src_ne" + std::to_string(i), std::to_string(source->ne[i]));
            dispatch.kernel.compile_parameters.emplace("ggml.float_copy.dst_ne" + std::to_string(i), std::to_string(output->ne[i]));
        }
    }
    dispatch.bindings.push_back({source->storage_root, source->storage_offset, source->byte_count});
    dispatch.bindings.push_back({output->storage_root, output->storage_offset, output->byte_count});
    return dispatch;
}

static bool match_float_copy_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const auto * source = context.graph.values().find(context.root_node->inputs[0]);
    const auto * output = context.graph.values().find(context.root_node->output);
    if (source == nullptr || output == nullptr || source->element_count != output->element_count ||
        (source->type != GGML_TYPE_F32 && source->type != GGML_TYPE_F16 && source->type != GGML_TYPE_BF16 && source->type != GGML_TYPE_I32) ||
        (output->type != GGML_TYPE_F32 && output->type != GGML_TYPE_F16 && output->type != GGML_TYPE_BF16 &&
         !(source->type == GGML_TYPE_I32 && output->type == GGML_TYPE_I32))) {
        return false;
    }
    if (source->storage == output->storage && source->storage_offset < output->storage_offset + output->byte_count &&
        output->storage_offset < source->storage_offset + source->byte_count) return false;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (source->nb[i] % ggml_type_size(source->type) || output->nb[i] % ggml_type_size(output->type)) return false;
    }
    Dispatch dispatch = float_copy_dispatch(source, output);
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

static bool match_set_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const auto * node = context.root_node;
    const auto * params = op_params_as<SetParams>(node->params);
    if (params == nullptr || node->inputs.size() != 2) return false;
    const auto * base = context.graph.values().find(node->inputs[0]);
    const auto * source = context.graph.values().find(node->inputs[1]);
    const auto * output = context.graph.values().find(node->output);
    if (base == nullptr || source == nullptr || output == nullptr ||
        base->type != GGML_TYPE_F32 || source->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        !base->contiguous || !output->contiguous || base->element_count != output->element_count) return false;
    const auto & p = params->values;
    for (int i = 0; i < 4; ++i) if (p[i] < 0 || p[i] % sizeof(float)) return false;
    for (size_t stride : source->nb) if (stride % sizeof(float)) return false;
    if (source->storage == output->storage && source->storage_offset < output->storage_offset + output->byte_count &&
        output->storage_offset < source->storage_offset + source->byte_count) return false;
    const bool same_base = base->storage == output->storage && base->storage_offset == output->storage_offset;
    if (!same_base) {
        if (base->storage == output->storage && base->storage_offset < output->storage_offset + output->byte_count &&
            output->storage_offset < base->storage_offset + base->byte_count) return false;
        match.dispatches.push_back(float_copy_dispatch(base, output));
    }
    Value target = *output;
    target.element_count = source->element_count;
    target.ne = source->ne;
    target.nb[0] = sizeof(float);
    for (int i = 1; i < GGML_MAX_DIMS; ++i) target.nb[i] = static_cast<size_t>(p[i - 1]);
    size_t extent = sizeof(float);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) extent += static_cast<size_t>(target.ne[i] - 1) * target.nb[i];
    if (static_cast<size_t>(p[3]) > output->byte_count || extent > output->byte_count - p[3]) return false;
    target.byte_count = extent;
    target.storage_offset += p[3];
    match.dispatches.push_back(float_copy_dispatch(source, &target));
    match.covered_nodes.push_back(context.root_index);
    return true;
}

static bool match_concat_dispatch(const DispatchMatchContext & context, DispatchMatch & match) {
    const auto * node = context.root_node;
    if (node->inputs.size() != 2) return false;
    const auto * left = context.graph.values().find(node->inputs[0]);
    const auto * right = context.graph.values().find(node->inputs[1]);
    const auto * output = context.graph.values().find(node->output);
    if (left == nullptr || right == nullptr || output == nullptr || !output->contiguous ||
        left->type != output->type || right->type != output->type ||
        (output->type != GGML_TYPE_F32 && output->type != GGML_TYPE_F16 && output->type != GGML_TYPE_BF16)) return false;
    int axis = -1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (left->ne[i] == right->ne[i] && left->ne[i] == output->ne[i]) continue;
        if (axis != -1 || output->ne[i] != left->ne[i] + right->ne[i]) return false;
        axis = i;
    }
    if (axis == -1) return false;
    size_t offset = 0;
    for (const Value * source : {left, right}) {
        if (source->storage == output->storage && source->storage_offset < output->storage_offset + output->byte_count &&
            output->storage_offset < source->storage_offset + source->byte_count) return false;
        for (size_t stride : source->nb) if (stride % ggml_type_size(source->type)) return false;
        Value target = *output;
        target.ne = source->ne;
        target.element_count = source->element_count;
        target.storage_offset += offset;
        target.byte_count = ggml_type_size(output->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) target.byte_count += static_cast<size_t>(target.ne[i] - 1) * target.nb[i];
        match.dispatches.push_back(float_copy_dispatch(source, &target));
        offset += static_cast<size_t>(source->ne[axis]) * output->nb[axis];
    }
    match.covered_nodes.push_back(context.root_index);
    return true;
}

void register_copy_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({"common.upscale_bilinear_f32", GGML_OP_UPSCALE, DispatchMatchKind::SingleOp, 0,
                  DispatchSource::Common, match_bilinear_f32_dispatch});
    registry.add({"common.upscale_nearest_f32", GGML_OP_UPSCALE, DispatchMatchKind::SingleOp, 0,
                  DispatchSource::Common, match_repeat_f32_dispatch});
    registry.add({"common.pad_f32", GGML_OP_PAD, DispatchMatchKind::SingleOp, 0,
                  DispatchSource::Common, match_pad_f32_dispatch});
    registry.add({"common.concat_strided", GGML_OP_CONCAT, DispatchMatchKind::SingleOp, -10,
                  DispatchSource::Common, match_concat_dispatch});
    registry.add({"common.set_f32", GGML_OP_SET, DispatchMatchKind::SingleOp, 0,
                  DispatchSource::Common, match_set_dispatch});
    for (const auto op : {GGML_OP_CPY, GGML_OP_CONT}) {
        registry.add({"common.float_copy", op, DispatchMatchKind::SingleOp, -10,
                      DispatchSource::Common, match_float_copy_dispatch});
    }
    registry.add({"common.repeat_f32", GGML_OP_REPEAT, DispatchMatchKind::SingleOp, 0,
                  DispatchSource::Common, match_repeat_f32_dispatch});
    registry.add({
        "common.copy_f32",
        GGML_OP_CPY,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_copy_f32_dispatch,
    });
    registry.add({
        "common.cont_f32",
        GGML_OP_CONT,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_cont_f32_dispatch,
    });
    registry.add({
        "common.concat_dim0_f32",
        GGML_OP_CONCAT,
        DispatchMatchKind::SingleOp,
        0,
        DispatchSource::Common,
        match_concat_dim0_f32_dispatch,
    });
}

}  // namespace ggml::hrx
