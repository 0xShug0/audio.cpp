#include "test_assert.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

namespace {

void test_matrix(int64_t columns, int64_t rows, int64_t batch) {
    std::vector<float> weights(static_cast<size_t>(columns * rows));
    for (size_t i = 0; i < weights.size(); ++i) {
        const float scale = (i / 128) % 3 == 0 ? 0.0f : ((i / 128) % 2 ? 0.125f : 0.5f);
        weights[i] = (static_cast<int>(i % 3) - 1) * scale;
    }
    std::vector<block_tq2_1> packed(weights.size() / 256);
    quantize_row_tq2_1_ref(weights.data(), packed.data(), static_cast<int64_t>(weights.size()));
    engine::test::require(ggml_validate_row_data(GGML_TYPE_TQ2_1, packed.data(), packed.size() * sizeof(block_tq2_1)),
                          "valid ternary blocks rejected");
    std::vector<float> restored(weights.size());
    dequantize_row_tq2_1(packed.data(), restored.data(), static_cast<int64_t>(weights.size()));
    engine::test::require(restored == weights, "independent 128-weight scales were not preserved exactly");

    auto invalid = packed;
    invalid.front().qs[0] |= 3;
    engine::test::require(!ggml_validate_row_data(GGML_TYPE_TQ2_1, invalid.data(), invalid.size() * sizeof(block_tq2_1)),
                          "reserved ternary code accepted");
    invalid = packed;
    invalid.front().d[1] = ggml_fp32_to_fp16(INFINITY);
    engine::test::require(!ggml_validate_row_data(GGML_TYPE_TQ2_1, invalid.data(), invalid.size() * sizeof(block_tq2_1)),
                          "nonfinite second scale accepted");

    std::vector<float> input(static_cast<size_t>(columns * batch));
    for (size_t i = 0; i < input.size(); ++i) input[i] = (static_cast<int>((i * 17) % 129) - 64) * 0.125f;
    std::vector<block_q8_K> activation(input.size() / 256);
    quantize_row_q8_K_ref(input.data(), activation.data(), static_cast<int64_t>(input.size()));
    std::vector<float> quantized_input(input.size());
    dequantize_row_q8_K(activation.data(), quantized_input.data(), static_cast<int64_t>(input.size()));

    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
        ggml_init({16 * 1024 * 1024, nullptr, false}), ggml_free);
    engine::test::require(context != nullptr, "context allocation failed");
    auto * w = ggml_new_tensor_2d(context.get(), GGML_TYPE_TQ2_1, columns, rows);
    auto * x = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, columns, batch);
    std::memcpy(w->data, packed.data(), packed.size() * sizeof(block_tq2_1));
    std::memcpy(x->data, input.data(), input.size() * sizeof(float));
    auto * output = ggml_mul_mat(context.get(), w, x);
    auto * graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph, output);
    engine::test::require(ggml_graph_compute_with_ctx(context.get(), graph, 2) == GGML_STATUS_SUCCESS,
                          "ternary CPU graph failed");
    const auto * actual = static_cast<const float *>(output->data);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t r = 0; r < rows; ++r) {
            float expected = 0.0f;
            for (int64_t c = 0; c < columns; ++c) {
                expected += weights[r * columns + c] * quantized_input[b * columns + c];
            }
            engine::test::require_close(actual[b * rows + r], expected,
                1e-4f + 1e-5f * std::abs(expected), "ternary matrix multiplication");
        }
    }
}

} // namespace

int main() {
    try {
        engine::test::require(GGML_TYPE_I2_S == 43 && GGML_TYPE_TQ2_1 == 44,
                              "existing GGUF type IDs changed");
        engine::test::require(ggml_row_size(GGML_TYPE_TQ2_1, 256) == 68, "incorrect ternary block size");
        for (const int64_t columns : {256, 512, 1536}) {
            for (const int64_t batch : {1, 3, 32}) test_matrix(columns, 7, batch);
        }
        std::cout << "tq2_1_test passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
