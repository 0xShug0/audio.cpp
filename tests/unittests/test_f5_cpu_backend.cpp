#include "../../src/community_models/f5_tts/cpu_graph_compute.h"
#include "engine/framework/core/execution_context.h"
#include "ggml.h"

#include <iostream>
#include <stdexcept>

int main() try {
    using namespace engine;
    // The private weak/dlsym symbol formerly used here was unavailable on MSVC.
    core::ExecutionContext execution({core::BackendType::Cpu, 0, 2});
    auto * ctx = ggml_init({1024 * 1024, nullptr, false});
    if (!ctx) throw std::runtime_error("context allocation failed");
    auto * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    auto * output = ggml_scale(ctx, input, 2.0F);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    for (int repeat = 0; repeat < 3; ++repeat) {
        for (int i = 0; i < 4; ++i) static_cast<float *>(input->data)[i] = float(i + repeat);
        if (models::f5_tts::f5_cpu_graph_compute(execution.backend(), graph, 2) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("CPU graph execution failed");
        for (int i = 0; i < 4; ++i)
            if (static_cast<float *>(output->data)[i] != float(2 * (i + repeat)))
                throw std::runtime_error("CPU result mismatch");
    }
    core::release_backend_graph_resources(execution.backend(), graph);
    ggml_free(ctx);
    std::cout << "PASS F5 CPU backend compute and graph reuse\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
}
