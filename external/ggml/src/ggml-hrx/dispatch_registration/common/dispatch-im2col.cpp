#include "dispatch-im2col.h"

#include "kernel-corpus/kernel-corpus-catalog-verify.h"

#include <utility>

namespace ggml::hrx {
namespace {

bool match_im2col(const DispatchMatchContext & context, DispatchMatch & match) {
    const auto * node = context.root_node;
    const auto * params = op_params_as<Im2ColParams>(node->params);
    if (params == nullptr || node->inputs.size() != 2) return false;
    const auto * kernel = context.graph.values().find(node->inputs[0]);
    const auto * input = context.graph.values().find(node->inputs[1]);
    const auto * output = context.graph.values().find(node->output);
    if (kernel == nullptr || input == nullptr || output == nullptr || !output->contiguous ||
        (input->type != GGML_TYPE_F32 && input->type != GGML_TYPE_F16) ||
        (output->type != GGML_TYPE_F32 && output->type != GGML_TYPE_F16)) return false;
    const auto & p = params->values;
    const bool is_2d = p[6] != 0;
    const size_t scalar_size = ggml_type_size(input->type);
    for (size_t stride : input->nb) if (stride % scalar_size) return false;
    Dispatch dispatch;
    dispatch.kernel = make_kernel_specialization(GGML_HRX_KERNEL_REF("loom_libs", "ggml_im2col"));
    dispatch.kernel.integer_parameters.emplace("element_count", output->element_count);
    dispatch.kernel.integer_parameters.emplace("input_span", input->byte_count / scalar_size);
    auto & config = dispatch.kernel.compile_parameters;
    const std::pair<const char *, int64_t> dimensions[] = {
        {"columns", output->ne[0]}, {"ow", output->ne[1]}, {"oh", is_2d ? output->ne[2] : 1},
        {"iw", input->ne[0]}, {"ih", is_2d ? input->ne[1] : 1},
        {"kw", kernel->ne[0]}, {"kh", is_2d ? kernel->ne[1] : 1},
        {"s0", p[0]}, {"s1", is_2d ? p[1] : 1}, {"d0", p[4]}, {"d1", is_2d ? p[5] : 1},
        {"x_stride", static_cast<int64_t>(input->nb[0] / scalar_size)},
        {"y_stride", is_2d ? static_cast<int64_t>(input->nb[1] / scalar_size) : 0},
        {"channel_stride", static_cast<int64_t>(input->nb[is_2d ? 2 : 1] / scalar_size)},
        {"batch_stride", static_cast<int64_t>(input->nb[is_2d ? 3 : 2] / scalar_size)},
        {"input_f16", input->type == GGML_TYPE_F16}, {"output_f16", output->type == GGML_TYPE_F16},
    };
    for (const auto & item : dimensions) config.emplace(std::string("ggml.im2col.") + item.first, std::to_string(item.second));
    for (int i = 0; i < 2; ++i) {
        const int64_t padding = i == 0 || is_2d ? p[2 + i] : 0;
        config.emplace("ggml.im2col.pad" + std::to_string(i), std::to_string(padding > 0 ? padding : 0));
        config.emplace("ggml.im2col.crop" + std::to_string(i), std::to_string(padding < 0 ? -padding : 0));
    }
    dispatch.bindings.push_back({input->storage_root, input->storage_offset, input->byte_count});
    dispatch.bindings.push_back({output->id, 0, output->byte_count});
    match.covered_nodes.push_back(context.root_index);
    match.dispatches.push_back(std::move(dispatch));
    return true;
}

}  // namespace

void register_im2col_dispatch(DispatchRegistryBuilder & registry) {
    registry.add({"common.im2col", GGML_OP_IM2COL, DispatchMatchKind::SingleOp, 0,
                  DispatchSource::Common, match_im2col});
}

}  // namespace ggml::hrx
