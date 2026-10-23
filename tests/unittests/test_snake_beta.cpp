#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "test_assert.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {
using namespace engine;

struct Graph {
    core::ExecutionContext & execution;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{
        ggml_init({1024 * 1024, nullptr, true}), ggml_free};
    core::TensorValue input, alpha, beta, output;
    ggml_cgraph * graph;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{
        nullptr, ggml_backend_buffer_free};

    Graph(core::ExecutionContext & execution, core::BackendType lowering,
          core::TensorShape shape, bool logscale, bool shared_beta, ggml_type type)
        : execution(execution) {
        test::require(context != nullptr, "context allocation failed");
        core::ModuleBuildContext ctx{context.get(), "snake_beta_test", lowering};
        const auto channels = shape.dims[shape.rank - 2];
        input = core::make_tensor(ctx, type, shape);
        alpha = core::make_tensor(ctx, type, core::TensorShape::from_dims({channels}));
        beta = shared_beta ? alpha : core::make_tensor(ctx, type, alpha.shape);
        for (auto value : {input, alpha, beta}) ggml_set_input(value.tensor);
        output = modules::SnakeBeta1dModule({channels, logscale}).build(ctx, input, {alpha, beta});
        test::require(output.type == GGML_TYPE_F32, "output must remain F32");
        graph = ggml_new_graph_custom(context.get(), 64, false);
        ggml_build_forward_expand(graph, output.tensor);
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), execution.backend()));
        test::require(buffer != nullptr, "tensor allocation failed");

        const bool fused = lowering == core::BackendType::Cuda &&
            input.tensor->ne[2] == 1 && input.tensor->ne[3] == 1;
        int squares = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
            squares += ggml_graph_node(graph, i)->op == GGML_OP_SQR;
        test::require(squares == (fused ? 1 : 0), "incorrect lowering selection");
        if (fused) {
            const std::array<ggml_op, 5> chain{GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD};
            const int start = ggml_graph_n_nodes(graph) - static_cast<int>(chain.size());
            for (size_t i = 0; i < chain.size(); ++i)
                test::require(ggml_graph_node(graph, start + static_cast<int>(i))->op == chain[i],
                              "CUDA activation chain is not consecutive");
        }
    }

    ~Graph() { core::release_backend_graph_resources(execution.backend(), graph, true); }

    std::vector<float> run(int repeat, bool logscale) {
        std::vector<float> x(input.shape.num_elements()), a(alpha.shape.num_elements()), b(a.size());
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = 2.0F * std::sin(static_cast<float>(i + repeat * 13) * 0.17F);
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = 0.3F + static_cast<float>(i % 7) * 0.2F + repeat * 0.1F;
            b[i] = logscale ? -1.0F + static_cast<float>(i % 5) * 0.4F :
                             (i == 0 ? 0.0F : (i == 1 ? 1.0e-9F : 0.4F + static_cast<float>(i % 5)));
            b[i] += repeat * 0.05F;
        }
        core::write_tensor_float(input, x);
        core::write_tensor_float(alpha, a);
        if (beta.tensor != alpha.tensor) core::write_tensor_float(beta, b);
        test::require(core::compute_backend_graph(execution.backend(), graph) == GGML_STATUS_SUCCESS,
                      "SnakeBeta graph execution failed");
        return core::read_tensor_f32(output.tensor);
    }
};

void check(core::ExecutionContext & execution, core::BackendType lowering,
           core::TensorShape shape, bool logscale, bool shared_beta, ggml_type type,
           double & worst_error, double & lowest_cosine) {
    Graph reference(execution, core::BackendType::Cpu, shape, logscale, shared_beta, type);
    Graph candidate(execution, lowering, shape, logscale, shared_beta, type);
    std::vector<float> first;
    for (const int repeat : {0, 1, 0}) {
        const auto expected = reference.run(repeat, logscale);
        const auto actual = candidate.run(repeat, logscale);
        double dot = 0, aa = 0, bb = 0;
        const bool fused = lowering == core::BackendType::Cuda && shape.num_elements() ==
            shape.dims[shape.rank - 2] * shape.dims[shape.rank - 1];
        for (size_t i = 0; i < actual.size(); ++i) {
            test::require(std::isfinite(actual[i]), "nonfinite activation output");
            const double error = std::abs(static_cast<double>(actual[i]) - expected[i]) /
                std::max(1.0, std::abs(static_cast<double>(expected[i])));
            worst_error = std::max(worst_error, error);
            test::require(error <= 2.0e-6, "activation differs from original formula");
            if (!fused) test::require(actual[i] == expected[i], "fallback changed");
            dot += static_cast<double>(actual[i]) * expected[i];
            aa += static_cast<double>(actual[i]) * actual[i];
            bb += static_cast<double>(expected[i]) * expected[i];
        }
        const double cosine = dot / std::sqrt(aa * bb);
        lowest_cosine = std::min(lowest_cosine, cosine);
        test::require(cosine > 0.99999, "activation cosine parity failed");
        if (first.empty()) first = actual;
        else if (repeat == 0)
            test::require(std::memcmp(first.data(), actual.data(), actual.size() * sizeof(float)) == 0,
                          "reused graph retained stale parameters");
    }
}
}  // namespace

int main(int argc, char ** argv) {
    const auto started = std::chrono::steady_clock::now();
    try {
        core::BackendType backend = core::BackendType::Cpu;
        bool log = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--log") log = true;
            else if (arg == "--backend" && i + 1 < argc) {
                const std::string value = argv[++i];
                if (value == "cpu") backend = core::BackendType::Cpu;
                else if (value == "cuda") backend = core::BackendType::Cuda;
                else throw std::runtime_error("expected cpu or cuda backend");
            } else throw std::runtime_error("unknown argument: " + arg);
        }
        debug::configure_logging({log});
        core::ExecutionContext execution({backend, 0, 2});
        double worst_error = 0, lowest_cosine = 1;
        int cases = 0;
        for (const auto lowering : {core::BackendType::Cpu, core::BackendType::Cuda,
                                   core::BackendType::Vulkan, core::BackendType::Metal, core::BackendType::Hip})
            for (const bool logscale : {false, true})
                for (const auto & shape : {core::TensorShape::from_dims({7, 33}),
                                          core::TensorShape::from_dims({1, 7, 1}),
                                          core::TensorShape::from_dims({1, 16, 257}),
                                          core::TensorShape::from_dims({2, 1, 7, 33}),
                                          core::TensorShape::from_dims({2, 7, 33})}) {
                    check(execution, lowering, shape, logscale, false, GGML_TYPE_F32, worst_error, lowest_cosine);
                    ++cases;
                }
        for (const auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            check(execution, core::BackendType::Cuda, core::TensorShape::from_dims({1, 7, 65}),
                  false, true, type, worst_error, lowest_cosine);
            ++cases;
        }
        std::cout << std::setprecision(17) << "PASS: " << cases << " cases, three executions per graph; max scaled error="
                  << worst_error << ", minimum cosine=" << lowest_cosine << '\n';
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "snake_beta_test failed: " << error.what() << '\n';
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return 1;
    }
}
