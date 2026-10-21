#include "engine/framework/modules/optimizations/neon_conv1d.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/neon/execution.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace engine;

int main() {
    if (!neon::available()) return 77;
    try {
        for (int threads : {1, 4}) for (int frames : {17, 129})
        for (int kernel : {1, 3, 7}) for (int dilation : {1, 3}) for (int stride : {1, 2}) {
            core::ExecutionContext execution({core::BackendType::Cpu, 0, threads});
            auto * ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
            if (!ctx) throw std::runtime_error("context allocation failed");
            core::ModuleBuildContext build{ctx};
            const modules::Conv1dConfig config{3, 5, kernel, stride, dilation * (kernel - 1) / 2, dilation, true};
            auto input = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({1, 3, frames}));
            auto weight = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({5, 3, kernel}));
            auto bias = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({5}));
            auto projection = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({2, 5, 1}));
            auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, execution.backend());
            if (!buffer) throw std::runtime_error("input allocation failed");
            for (auto value : {input, weight, bias, projection}) {
                auto * data = static_cast<float *>(value.tensor->data);
                for (int64_t i = 0; i < ggml_nelements(value.tensor); ++i) data[i] = std::sin(i * 0.37f) * 0.1f;
            }
            modules::NeonConv1dGraph neon(threads);
            auto result = neon.build(build, input, {weight, bias}, config);
            ggml_set_output(result.tensor);
            auto projected = neon.build(build, result, {projection, std::nullopt}, {5, 2, 1, 1, 0, 1, false});
            auto * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, projected.tensor);
            auto * allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
            if (!ggml_gallocr_alloc_graph(allocator, graph)) throw std::runtime_error("graph allocation failed");
            for (int repeat = 0; repeat < 2; ++repeat) {
                static_cast<float *>(input.tensor->data)[0] += 0.01f;
                if (core::compute_backend_graph(execution.backend(), graph) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("graph compute failed");
                }
                const auto * x = static_cast<float *>(input.tensor->data);
                const auto * w = static_cast<float *>(weight.tensor->data);
                const auto * b = static_cast<float *>(bias.tensor->data);
                const auto * y = static_cast<float *>(result.tensor->data);
                const int width = result.tensor->ne[0];
                std::vector<double> expected(5 * width);
                for (int out = 0; out < 5; ++out) for (int t = 0; t < width; ++t) {
                    double reference = b[out];
                    for (int in = 0; in < 3; ++in) for (int k = 0; k < kernel; ++k) {
                        const int at = t * stride - config.padding + k * dilation;
                        if (at >= 0 && at < frames) reference += double(x[in * frames + at]) * w[(out * 3 + in) * kernel + k];
                    }
                    if (!std::isfinite(y[out * width + t]) || std::abs(y[out * width + t] - reference) > 2e-6) {
                        throw std::runtime_error("NEON convolution differs from independent FP64 reference");
                    }
                    expected[out * width + t] = reference;
                }
                const auto * weights = static_cast<float *>(projection.tensor->data);
                const auto * output = static_cast<float *>(projected.tensor->data);
                for (int out = 0; out < 2; ++out) for (int t = 0; t < width; ++t) {
                    double reference = 0;
                    for (int in = 0; in < 5; ++in) reference += expected[in * width + t] * weights[out * 5 + in];
                    if (!std::isfinite(output[out * width + t]) || std::abs(output[out * width + t] - reference) > 2e-6) {
                        throw std::runtime_error("shared convolution plan differs from independent FP64 reference");
                    }
                }
            }
            ggml_gallocr_free(allocator);
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
        std::cout << "NEON convolution: 24 shapes at 1/4 threads, two runs each passed\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
