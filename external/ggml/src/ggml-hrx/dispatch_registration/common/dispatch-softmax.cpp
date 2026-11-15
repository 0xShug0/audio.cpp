#include "dispatch-softmax.h"

#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace ggml::hrx {
namespace {

bool match_softmax(const DispatchMatchContext & context, DispatchMatch & match) {
    const auto * node = context.root_node;
    const auto * params = op_params_as<SoftMaxParams>(node->params);
    if (params == nullptr || params->max_bias > 0.0f || params->has_sinks ||
        node->inputs.size() != (params->has_mask ? 2u : 1u)) return false;
    const auto * input = context.graph.values().find(node->inputs[0]);
    const auto * output = context.graph.values().find(node->output);
    const auto * mask = params->has_mask ? context.graph.values().find(node->inputs[1]) : nullptr;
    if ((params->has_mask && mask == nullptr) || input == nullptr || output == nullptr || input->type != GGML_TYPE_F32 || output->type != GGML_TYPE_F32 ||
        !input->contiguous || !output->contiguous) return false;
    if (mask != nullptr && ((mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) ||
        !mask->contiguous || mask->ne[0] != input->ne[0] || mask->ne[1] < input->ne[1] ||
        input->ne[2] % mask->ne[2] || input->ne[3] % mask->ne[3])) return false;
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(GGML_HRX_KERNEL_REF("loom_libs", "ggml_softmax_f32"));
    dispatch.kernel.integer_parameters.emplace("rows", input->element_count / input->ne[0]);
    dispatch.kernel.integer_parameters.emplace("input_span", input->byte_count / sizeof(float));
    dispatch.kernel.integer_parameters.emplace("mask_span", mask != nullptr ? mask->byte_count / ggml_type_size(mask->type) : 1);
    auto & config = dispatch.kernel.compile_parameters;
    std::ostringstream scale;
    scale << std::setprecision(std::numeric_limits<float>::max_digits10) << params->scale;
    config.emplace("ggml.softmax.scale", scale.str());
    config.emplace("ggml.softmax.mask_stride0", mask != nullptr ? std::to_string(mask->nb[0] / ggml_type_size(mask->type)) : "1");
    config.emplace("ggml.softmax.zero_all_masked", "0");
    config.emplace("ggml.softmax.width", std::to_string(input->ne[0]));
    config.emplace("ggml.softmax.ne1", std::to_string(input->ne[1]));
    config.emplace("ggml.softmax.ne2", std::to_string(input->ne[2]));
    config.emplace("ggml.softmax.mask_type", mask == nullptr ? "0" : mask->type == GGML_TYPE_F16 ? "2" : "1");
    config.emplace("ggml.softmax.mask_ne2", std::to_string(mask != nullptr ? mask->ne[2] : 1));
    config.emplace("ggml.softmax.mask_ne3", std::to_string(mask != nullptr ? mask->ne[3] : 1));
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        config.emplace("ggml.softmax.stride" + std::to_string(i), std::to_string(input->nb[i] / sizeof(float)));
        config.emplace("ggml.softmax.mask_stride" + std::to_string(i),
                       std::to_string(mask != nullptr ? mask->nb[i] / ggml_type_size(mask->type) : 1));
    }
    dispatch.bindings.push_back({input->storage_root, input->storage_offset, input->byte_count});
    dispatch.bindings.push_back(mask != nullptr ? DispatchBinding{mask->storage_root, mask->storage_offset, mask->byte_count} :
                                                DispatchBinding{input->storage_root, input->storage_offset, input->byte_count});
    dispatch.bindings.push_back({output->id, 0, output->byte_count});
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_softmax_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({"common.softmax_f32", GGML_OP_SOFT_MAX, DispatchMatchKind::SingleOp, -100,
                  DispatchSource::Common, match_softmax});
}

}  // namespace ggml::hrx
