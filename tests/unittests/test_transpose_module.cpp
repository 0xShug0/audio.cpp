#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/structural_modules.h"
#include "test_assert.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<float> make_input_values(int64_t batch, int64_t time, int64_t freq, int64_t channels) {
    std::vector<float> values(static_cast<size_t>(batch * time * freq * channels), 0.0F);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t t = 0; t < time; ++t) {
            for (int64_t f = 0; f < freq; ++f) {
                for (int64_t c = 0; c < channels; ++c) {
                    const auto index = static_cast<size_t>(((b * time + t) * freq + f) * channels + c);
                    values[index] = static_cast<float>(1000 * b + 100 * t + 10 * f + c);
                }
            }
        }
    }
    return values;
}

std::vector<float> expected_transpose_0231(const std::vector<float> & input, int64_t batch, int64_t time, int64_t freq, int64_t channels) {
    std::vector<float> expected(static_cast<size_t>(batch * freq * channels * time), 0.0F);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t f = 0; f < freq; ++f) {
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t t = 0; t < time; ++t) {
                    const auto input_index = static_cast<size_t>(((b * time + t) * freq + f) * channels + c);
                    const auto output_index = static_cast<size_t>(((b * freq + f) * channels + c) * time + t);
                    expected[output_index] = input[input_index];
                }
            }
        }
    }
    return expected;
}

void test_non_self_inverse_transpose_matches_logical_axes() {
    constexpr int64_t batch = 2;
    constexpr int64_t time = 3;
    constexpr int64_t freq = 4;
    constexpr int64_t channels = 5;

    auto * backend = engine::core::init_backend({engine::core::BackendType::Cpu, 0, 1});
    engine::test::require(backend != nullptr, "failed to initialize CPU backend");

    ggml_init_params params{};
    params.mem_size = 4 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ggml_ctx = ggml_init(params);
    if (ggml_ctx == nullptr) {
        ggml_backend_free(backend);
        throw std::runtime_error("failed to initialize ggml context");
    }

    try {
        engine::core::ModuleBuildContext ctx{ggml_ctx, "transpose_module_test", engine::core::BackendType::Cpu};
        auto input = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({batch, time, freq, channels}));

        auto transposed = engine::modules::TransposeModule({{0, 2, 3, 1}, 4}).build(ctx, input);
        engine::test::require_eq(transposed.shape.rank, static_cast<size_t>(4), "transpose rank");
        engine::test::require_eq(transposed.shape.dims[0], batch, "transpose dim 0");
        engine::test::require_eq(transposed.shape.dims[1], freq, "transpose dim 1");
        engine::test::require_eq(transposed.shape.dims[2], channels, "transpose dim 2");
        engine::test::require_eq(transposed.shape.dims[3], time, "transpose dim 3");

        auto output = engine::core::ensure_backend_addressable_layout(ctx, transposed);
        ggml_cgraph * graph = ggml_new_graph_custom(ggml_ctx, 64, false);
        ggml_build_forward_expand(graph, output.tensor);

        auto * buffer = ggml_backend_alloc_ctx_tensors(ggml_ctx, backend);
        engine::test::require(buffer != nullptr, "failed to allocate transpose test tensors");

        const auto input_values = make_input_values(batch, time, freq, channels);
        engine::core::write_tensor_f32(input, input_values);
        const auto status = engine::core::compute_backend_graph(backend, graph, nullptr, "TransposeModule non-self-inverse test");
        engine::test::require(status == GGML_STATUS_SUCCESS, "transpose compute failed");

        const auto actual = engine::core::read_tensor_f32(output.tensor);
        const auto expected = expected_transpose_0231(input_values, batch, time, freq, channels);
        engine::test::require_eq(actual.size(), expected.size(), "transpose output value count");
        for (size_t i = 0; i < expected.size(); ++i) {
            engine::test::require_close(actual[i], expected[i], 0.0F, "transpose value " + std::to_string(i));
        }

        ggml_backend_buffer_free(buffer);
        ggml_free(ggml_ctx);
        ggml_backend_free(backend);
    } catch (...) {
        ggml_free(ggml_ctx);
        ggml_backend_free(backend);
        throw;
    }
}

}  // namespace

int main() {
    try {
        test_non_self_inverse_transpose_matches_logical_axes();
        std::cout << "transpose_module_test passed\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "transpose_module_test failed: " << ex.what() << "\n";
        return 1;
    }
}
