#include "engine/framework/core/backend.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/profiler.h"
#include "../../src/models/dettato/gru.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/structural_modules.h"
#include "../unittests/test_assert.h"

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace dettato = engine::models::dettato::detail;

namespace {
using namespace engine;
void check(core::BackendType backend, int64_t frames, int64_t batch, int64_t features, int64_t width, bool reverse, dettato::GRURecurrenceLowering lowering) {
    core::ExecutionContext execution({backend, 0, 8});
    core::PreparedGraph graph(execution, {"dettato_gru_test", 8 * 1024 * 1024, 8192});
    auto &ctx = graph.build_context();
    std::array<core::TensorValue, 10> operands{
        core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({frames, batch, features})),
        core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, width})),
        core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({3 * width, features})),
        core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({3 * width, width})),
        core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({3 * width})),
        core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({3 * width})),
        core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({3 * width, features})),
        core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({3 * width, width})),
        core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({3 * width})),
        core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({3 * width})),
    };
    for (auto &value : operands)
        ggml_set_input(value.tensor);
    const dettato::GRUWeights forward_weights{operands[2], operands[3], operands[4], operands[5]};
    const dettato::GRUWeights reverse_weights{operands[6], operands[7], operands[8], operands[9]};
    const auto both = dettato::BidirectionalGRU({features, width, lowering})
        .build(ctx, operands[0], operands[1], operands[1], forward_weights, reverse_weights);
    const size_t base = reverse ? 6 : 2;
    const auto projection = modules::LinearModule({features, 3 * width, true, GGML_PREC_F32})
                                .build(ctx, operands[0], {operands[base], operands[base + 2]});
    for (const auto &value : {both.sequence, both.forward_hidden, both.reverse_hidden, projection}) {
        ggml_set_output(value.tensor);
        ggml_build_forward_expand(graph.graph(), value.tensor);
    }
    // This fixture checks F32 equations, not Vulkan's default F16 activation
    // conversion. Keep the strict independent reference tolerance unchanged.
    if (backend == core::BackendType::Vulkan)
        for (int i = 0; i < ggml_graph_n_nodes(graph.graph()); ++i) {
            auto *node = ggml_graph_node(graph.graph(), i);
            if (node->op == GGML_OP_MUL_MAT)
                ggml_mul_mat_set_lowering(node, GGML_MUL_MAT_LOWERING_VULKAN_F32_INPUTS);
        }
    graph.allocate();
    for (int repeat = 0; repeat < 3; ++repeat) {
        std::array<std::vector<float>, 10> data;
        for (size_t i = 0; i < data.size(); ++i) {
            data[i].resize(static_cast<size_t>(operands[i].shape.num_elements()));
            for (size_t j = 0; j < data[i].size(); ++j)
                data[i][j] = 0.17f * std::sin(static_cast<float>(j * 3 + i * 19 + repeat * 5));
            core::write_tensor_f32(operands[i], data[i]);
        }
        // Independent scalar equations use double precision, distinct reset and
        // update gates, and the original (1-z)*n + z*h recurrence.
        std::vector<double> hidden(data[1].begin(), data[1].end());
        std::vector<double> expected(static_cast<size_t>(frames * batch * width));
        for (int64_t step = 0; step < frames; ++step) {
            const int64_t t = reverse ? frames - 1 - step : step;
            for (int64_t b = 0; b < batch; ++b) {
                std::vector<double> next(static_cast<size_t>(width));
                for (int64_t h = 0; h < width; ++h) {
                    std::array<double, 3> input{}, recurrent{};
                    for (int64_t g = 0; g < 3; ++g) {
                        const int64_t row = g * width + h;
                        input[g] = data[base + 2][static_cast<size_t>(row)];
                        recurrent[g] = data[base + 3][static_cast<size_t>(row)];
                        for (int64_t f = 0; f < features; ++f)
                            input[g] +=
                                data[base][static_cast<size_t>(row * features + f)] *
                                static_cast<double>(data[0][static_cast<size_t>((t * batch + b) * features + f)]);
                        for (int64_t j = 0; j < width; ++j)
                            recurrent[g] += data[base + 1][static_cast<size_t>(row * width + j)] *
                                            hidden[static_cast<size_t>(b * width + j)];
                    }
                    const double reset = 1 / (1 + std::exp(-input[0] - recurrent[0]));
                    const double update = 1 / (1 + std::exp(-input[1] - recurrent[1]));
                    const double candidate = std::tanh(input[2] + reset * recurrent[2]);
                    next[static_cast<size_t>(h)] =
                        (1 - update) * candidate + update * hidden[static_cast<size_t>(b * width + h)];
                    expected[static_cast<size_t>((t * batch + b) * width + h)] = next[static_cast<size_t>(h)];
                }
                std::copy(next.begin(), next.end(), hidden.begin() + b * width);
            }
        }
        graph.compute();
        const auto projected = core::read_tensor_f32(projection.tensor);
        const auto combined = core::read_tensor_f32(both.sequence.tensor);
        const auto combined_hidden = core::read_tensor_f32((reverse ? both.reverse_hidden : both.forward_hidden).tensor);
        float projection_error = 0;
        for (int64_t row = 0; row < frames * batch; ++row)
            for (int64_t gate = 0; gate < 3 * width; ++gate) {
                double value = data[base + 2][gate];
                for (int64_t f = 0; f < features; ++f)
                    value += static_cast<double>(data[0][row * features + f]) * data[base][gate * features + f];
                projection_error =
                    std::max(projection_error, std::abs(projected[row * 3 * width + gate] - static_cast<float>(value)));
            }
        debug::timing_log_scalar("gru.projection.max_abs", projection_error);
        for (int64_t row = 0; row < frames * batch; ++row)
            for (int64_t h = 0; h < width; ++h)
                test::require_close(combined[static_cast<size_t>(row * 2 * width + (reverse ? width : 0) + h)],
                    static_cast<float>(expected[static_cast<size_t>(row * width + h)]), 2e-5f, "Bidirectional GRU scalar sequence reference");
        for (size_t i = 0; i < combined_hidden.size(); ++i)
            test::require_close(combined_hidden[i], static_cast<float>(hidden[i]), 2e-5f, "Bidirectional GRU scalar final-state reference");
    }
}
} // namespace

int main(int argc, char **argv) {
    try {
        engine::debug::configure_logging({true});
        const auto start = std::chrono::steady_clock::now();
        auto backend = engine::core::BackendType::Cpu;
        if (argc >= 3 && std::string(argv[1]) == "--backend") {
            const std::string name = argv[2];
            if (name == "cuda")
                backend = engine::core::BackendType::Cuda;
            else if (name == "vulkan")
                backend = engine::core::BackendType::Vulkan;
            else if (name != "cpu")
                throw std::runtime_error("Expected cpu, cuda or vulkan");
        }
        auto lowering = dettato::GRURecurrenceLowering::Unrolled;
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == "--host-recurrence") lowering = dettato::GRURecurrenceLowering::HostRecurrence;
        for (const bool reverse : {false, true}) {
            check(backend, 1, 1, 3, 4, reverse, lowering);
            check(backend, 7, 3, 5, 17, reverse, lowering);
            check(backend, 33, 2, 8, 31, reverse, lowering);
        }
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
        std::cout << "GRU: six independent scalar-reference cases, three changed-input reuses each passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
