#include "vulkan_graph.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/primitive_modules.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

struct Buffer {
    ggml_backend_buffer_t value;
    ~Buffer() { if (value) ggml_backend_buffer_free(value); }
};
struct GraphLifetime {
    ggml_backend_t backend;
    ggml_cgraph * graph;
    ~GraphLifetime() { engine::core::release_backend_graph_resources(backend, graph); }
};

void exercise(engine::core::BackendConfig config, int64_t length) {
    namespace core = engine::core;
    namespace modules = engine::modules;
    core::ExecutionContext execution(config);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({4 * 1024 * 1024, nullptr, true}), ggml_free);
    if (!ctx) throw std::runtime_error("attention test context allocation failed");
    core::ModuleBuildContext build{ctx.get(), "inflect.attention.row.test", config.type};
    auto probabilities = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({1, 2, length, length}));
    auto values = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({1, 2, length, 4}));
    ggml_set_input(probabilities.tensor);
    ggml_set_input(values.tensor);
    core::TensorValue output;
    for (int64_t row = 0; row < length; ++row) {
        const auto probability_row = engine::models::inflect_v2::detail::attention_probability_row(build, probabilities, row);
        if (config.type == core::BackendType::Vulkan && probability_row.tensor->view_offs != 0)
            throw std::runtime_error("Vulkan attention row still binds an offset view");
        if (config.type == core::BackendType::Cpu && probability_row.tensor->op != GGML_OP_VIEW)
            throw std::runtime_error("CPU attention row was unnecessarily materialized");
        auto value = modules::MatMulModule().build(build, probability_row, values);
        output = output.valid() ? modules::ConcatModule({2}).build(build, output, value) : value;
    }
    ggml_set_output(output.tensor);
    auto * graph = ggml_new_graph_custom(ctx.get(), 2048, false);
    ggml_build_forward_expand(graph, output.tensor);
    engine::models::inflect_v2::detail::configure_vulkan_graph(graph, config.type);
    int matmul_count = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const auto * node = ggml_graph_node(graph, i);
        if (node->op != GGML_OP_MUL_MAT) continue;
        ++matmul_count;
        const auto expected = config.type == core::BackendType::Vulkan ? GGML_PREC_F32 : GGML_PREC_DEFAULT;
        if (node->op_params[0] != expected)
            throw std::runtime_error("Inflect precision policy changed the wrong backend");
    }
    if (matmul_count != length) throw std::runtime_error("attention test did not cover every row");
    Buffer buffer{ggml_backend_alloc_ctx_tensors(ctx.get(), execution.backend())};
    if (!buffer.value) throw std::runtime_error("attention test buffer allocation failed");
    GraphLifetime lifetime{execution.backend(), graph};
    std::vector<float> p(2 * length * length), v(2 * length * 4), actual(2 * length * 4);
    for (int run = 0; run < 8; ++run) {
        for (size_t i = 0; i < p.size(); ++i) p[i] = static_cast<float>(static_cast<int>((i + run) % 5) - 2) * 0.25F;
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(static_cast<int>((i + run) % 7) - 3) * 0.125F;
        ggml_backend_tensor_set(probabilities.tensor, p.data(), 0, p.size() * sizeof(float));
        ggml_backend_tensor_set(values.tensor, v.data(), 0, v.size() * sizeof(float));
        if (core::compute_backend_graph(execution.backend(), graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("attention test compute failed");
        ggml_backend_synchronize(execution.backend());
        ggml_backend_tensor_get(output.tensor, actual.data(), 0, actual.size() * sizeof(float));
        for (int64_t h = 0; h < 2; ++h) for (int64_t r = 0; r < length; ++r) for (int64_t c = 0; c < 4; ++c) {
            float expected = 0.0F;
            for (int64_t k = 0; k < length; ++k) expected += p[(h * length + r) * length + k] * v[(h * length + k) * 4 + c];
            const float a = actual[(h * length + r) * 4 + c];
            if (!std::isfinite(a) || std::abs(a - expected) > 1.0e-5F)
                throw std::runtime_error("aligned attention row differs from host dot products");
        }
    }
}

} // namespace

int main() {
    try {
        for (int64_t length : {3, 5, 7, 17}) exercise({engine::core::BackendType::Cpu, 0, 2}, length);
        const auto devices = engine::core::list_backend_devices();
        auto device = std::find_if(devices.begin(), devices.end(), [](const auto & d) { return d.backend == "Vulkan" && d.type == "GPU"; });
        if (device == devices.end()) device = std::find_if(devices.begin(), devices.end(), [](const auto & d) { return d.backend == "Vulkan"; });
        if (device == devices.end()) return 77;
        for (int64_t length : {3, 5, 7, 17}) exercise({engine::core::BackendType::Vulkan, device->index, 2}, length);
        std::cout << "inflect_vulkan_attention_row_test: ok on " << device->name << '\n';
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
