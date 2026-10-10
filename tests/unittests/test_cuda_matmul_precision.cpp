#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "ggml-alloc.h"
#include "test_assert.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

namespace {
struct Probe {
    ggml_backend_t backend;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer;
    ggml_cgraph * graph;
    ggml_tensor * output;
    std::vector<float> expected;

    Probe(ggml_backend_t b, int columns, ggml_type weight_type = GGML_TYPE_F32)
        : backend(b), ctx(ggml_init({1024 * 1024, nullptr, true}), ggml_free),
          buffer(nullptr, ggml_backend_buffer_free) {
        engine::test::require(ctx != nullptr, "probe context");
        constexpr int k = 256, rows = 64;
        auto * w = ggml_new_tensor_2d(ctx.get(), weight_type, k, rows);
        auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, columns);
        output = ggml_mul_mat(ctx.get(), w, x);
        ggml_mul_mat_set_prec(output, GGML_PREC_F32);
        graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, output);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        engine::test::require(buffer != nullptr, "probe buffer");
        std::vector<float> weights(k * rows), inputs(k * columns);
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = std::sin(float(i) * .173f) * .71f;
        for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = std::cos(float(i) * .131f) * .63f;
        if (weight_type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(w, weights.data(), 0, weights.size() * sizeof(float));
        } else {
            std::vector<unsigned char> packed(ggml_nbytes(w));
            const auto * traits = ggml_get_type_traits(weight_type);
            engine::test::require(traits->from_float_ref && traits->to_float, "probe conversion");
            traits->from_float_ref(weights.data(), packed.data(), weights.size());
            traits->to_float(packed.data(), weights.data(), weights.size());
            ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
        }
        ggml_backend_tensor_set(x, inputs.data(), 0, inputs.size() * sizeof(float));
        expected.resize(rows * columns);
        for (int col = 0; col < columns; ++col) for (int row = 0; row < rows; ++row) {
            double sum = 0;
            for (int j = 0; j < k; ++j) sum += double(weights[row * k + j]) * inputs[col * k + j];
            expected[col * rows + row] = float(sum);
        }
    }

    std::vector<float> run() {
        engine::test::require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "probe compute");
        ggml_backend_synchronize(backend);
        std::vector<float> result(expected.size());
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        return result;
    }
};
}

int main() {
    bool cuda = false;
    for (const auto & device : engine::core::list_backend_devices()) cuda |= device.backend == "CUDA";
    if (!cuda) return 125;
    try {
        for (int columns : {8, 64}) {
            engine::core::BackendConfig config;
            config.type = engine::core::BackendType::Cuda;
            engine::core::ExecutionContext legacy(config), strict(config);
            Probe old(legacy.backend(), columns);
            const auto before = old.run();
            auto * device = ggml_backend_get_device(strict.backend());
            auto fn = reinterpret_cast<bool (*)(ggml_backend_t, bool)>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(device), "ggml_backend_cuda_set_f32_matmul"));
            engine::test::require(fn && fn(strict.backend(), true), "instance precision control");
            // Keep graph addresses/buffers alive across probes: CUDA graph
            // cache entries must not alias newly allocated test graphs.
            std::vector<std::unique_ptr<Probe>> probes;
            for (const auto weight_type : {GGML_TYPE_F32, GGML_TYPE_BF16, GGML_TYPE_Q8_0}) {
                probes.push_back(std::make_unique<Probe>(strict.backend(), columns, weight_type));
                auto & accurate = *probes.back();
                const auto result = accurate.run();
                engine::test::require(before == old.run(), "strict instance must not change another backend's output");
                double error = 0, scale = 0;
                for (size_t i = 0; i < result.size(); ++i) {
                    const double delta = double(result[i]) - accurate.expected[i];
                    error += delta * delta;
                    scale += double(accurate.expected[i]) * accurate.expected[i];
                }
                const double relative_l2 = std::sqrt(error / scale);
                engine::test::require(relative_l2 < 1e-5, "strict F32 matmul input precision");
                std::cout << "columns=" << columns << " weights=" << ggml_type_name(weight_type)
                          << " relative_l2=" << relative_l2 << " other_instance_unchanged=true\n";
            }
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
