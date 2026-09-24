#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cuda.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
std::vector<float> run(ggml_backend_t backend, int width, int steps, int lanes,
                       bool residual, bool expose_bias_sum) {
    ggml_context * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    if (!ctx) throw std::runtime_error("context allocation failed");
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, steps * lanes);
    auto * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
    auto * r = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, steps, lanes);
    ggml_set_input(x);
    ggml_set_input(bias);
    ggml_set_input(r);
    auto * sum = ggml_add(ctx, x, bias);
    if (expose_bias_sum) ggml_set_output(sum); // Must prevent skipping a live intermediate.
    auto * shaped = ggml_reshape_3d(ctx, sum, width, steps, lanes);
    auto * output = residual ? ggml_add(ctx, r, shaped) : ggml_gelu_erf(ctx, shaped);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, output);
    auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, graph)) throw std::runtime_error("graph allocation failed");

    const size_t count = static_cast<size_t>(width) * steps * lanes;
    std::vector<float> input(count), biases(width), skip(count), values(count);
    for (size_t i = 0; i < count; ++i) {
        input[i] = std::sin(float(i) * 0.17f) * 12.0f;
        skip[i] = std::cos(float(i) * 0.13f) * 3.0f;
    }
    for (int i = 0; i < width; ++i) biases[i] = std::sin(float(i) * 0.31f);
    if (count >= 8) {
        input[0] = 0.0f;
        input[1] = -0.0f;
        input[2] = std::numeric_limits<float>::infinity();
        input[3] = -std::numeric_limits<float>::infinity();
        input[4] = std::numeric_limits<float>::quiet_NaN();
        input[5] = std::numeric_limits<float>::denorm_min();
        skip[5] = 0.0f;
    }
    for (int repeat = 0; repeat < 3; ++repeat) {
        // The graph allocator may reuse input storage for intermediates.
        ggml_backend_tensor_set(x, input.data(), 0, count * sizeof(float));
        ggml_backend_tensor_set(bias, biases.data(), 0, width * sizeof(float));
        if (residual) ggml_backend_tensor_set(r, skip.data(), 0, count * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("CUDA graph failed");
    }
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(output, values.data(), 0, count * sizeof(float));
    if (expose_bias_sum) {
        std::vector<float> actual(count);
        ggml_backend_tensor_get(sum, actual.data(), 0, count * sizeof(float));
        for (size_t i = 0; i < count; ++i) {
            const float expected = input[i] + biases[i % width];
            // The existing CUDA add can flush subnormals. Its GPU result,
            // rather than host arithmetic, is the reference for that case.
            if (std::fpclassify(expected) == FP_SUBNORMAL && actual[i] == 0.0f) continue;
            if (!(actual[i] == expected || (std::isnan(actual[i]) && std::isnan(expected)))) {
                std::cerr << "live sum width=" << width << " index=" << i << " actual=" << actual[i]
                          << " expected=" << expected << '\n';
                throw std::runtime_error("live intermediate was not preserved");
            }
        }
    }
    // Clear captured graph resources before freeing the allocator/context.
    auto device = ggml_backend_get_device(backend);
    auto reg = ggml_backend_dev_backend_reg(device);
    auto clear = reinterpret_cast<void (*)(ggml_backend_t, ggml_cgraph *)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_clear_graph"));
    if (clear) clear(backend, graph);
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return values;
}
}

int main() {
    if (ggml_backend_cuda_get_device_count() == 0) return 77;
    auto backend = ggml_backend_cuda_init(0);
    if (!backend) return 1;
    try {
        for (int width : {1, 17, 512, 2048}) {
            for (bool residual : {false, true}) {
                const auto reference = run(backend, width, 33, 2, residual, true);
                const auto fused = run(backend, width, 33, 2, residual, false);
                for (size_t i = 0; i < reference.size(); ++i) {
                    if (std::isnan(reference[i]) && std::isnan(fused[i])) continue;
                    if (std::memcmp(&reference[i], &fused[i], sizeof(float)) != 0) {
                        std::cerr << "width=" << width << " residual=" << residual << " index=" << i
                                  << " reference=" << std::hexfloat << reference[i] << " fused=" << fused[i] << '\n';
                        throw std::runtime_error("fused output differs from unfused CUDA output");
                    }
                }
            }
        }
        std::cout << "PASS: bias/GELU and bias/residual, odd widths, broadcast rows, exceptional values, live intermediates\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        ggml_backend_free(backend);
        return 1;
    }
    ggml_backend_free(backend);
}
