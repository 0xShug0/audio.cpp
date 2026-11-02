#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-metal.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
struct Storage {
    ggml_context * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    ggml_backend_buffer_t buffer = nullptr;
    ~Storage() {
        if (buffer) ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
};
}

int main() {
    auto backend = ggml_backend_metal_init();
    if (!backend) return 77;
    size_t checked = 0;
    try {
        for (int tokens : {1, 3, 31, 32, 33, 241, 299})
        for (int channels : {1, 7, 32, 33, 256})
        for (int taps : {1, 3, 4, 7}) {
            const int sequences = 2;
            const int length = tokens + taps - 1;
            Storage storage;
            auto ctx = storage.ctx;
            auto base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, length * channels * sequences + 2);
            auto x = ggml_view_3d(ctx, base, length, channels, sequences,
                                 length * sizeof(float), length * channels * sizeof(float), sizeof(float));
            auto w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, taps, channels);
            auto legacy = ggml_ssm_conv(ctx, x, w);
            auto tiled = ggml_ssm_conv(ctx, x, w);
            ggml_ssm_conv_set_lowering(tiled, GGML_SSM_CONV_LOWERING_METAL_TILED);
            auto graph = ggml_new_graph(ctx);
            ggml_set_output(legacy);
            ggml_set_output(tiled);
            ggml_build_forward_expand(graph, legacy);
            ggml_build_forward_expand(graph, tiled);
            storage.buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!storage.buffer) throw std::runtime_error("allocation failed");
            std::vector<float> input(ggml_nelements(base)), weights(ggml_nelements(w));
            for (int pattern = 0; pattern < 3; ++pattern) {
                for (size_t i = 0; i < input.size(); ++i) {
                    input[i] = float(int((i * 37 + pattern * 71) % 401) - 200) / 32;
                }
                for (size_t i = 0; i < weights.size(); ++i) {
                    weights[i] = float(int((i * 17 + pattern * 29) % 101) - 50) / 32;
                }
                ggml_backend_tensor_set(base, input.data(), 0, input.size() * sizeof(float));
                ggml_backend_tensor_set(w, weights.data(), 0, weights.size() * sizeof(float));
                if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("compute failed");
                }
                std::vector<float> actual(ggml_nelements(tiled)), reference(actual.size());
                ggml_backend_tensor_get(tiled, actual.data(), 0, actual.size() * sizeof(float));
                ggml_backend_tensor_get(legacy, reference.data(), 0, reference.size() * sizeof(float));
                for (int s = 0; s < sequences; ++s) for (int t = 0; t < tokens; ++t)
                for (int c = 0; c < channels; ++c) {
                    double expected = 0;
                    for (int k = 0; k < taps; ++k) {
                        expected += double(input[1 + (s * channels + c) * length + t + k]) * weights[c * taps + k];
                    }
                    const size_t i = (s * tokens + t) * channels + c;
                    // Binary-rational inputs make every product and sum exact in F32.
                    if (actual[i] != expected || reference[i] != expected) {
                        throw std::runtime_error("independent oracle or legacy GPU mismatch");
                    }
                    ++checked;
                }
            }
        }
        std::cout << "PASS 140 shapes, three input patterns, " << checked
                  << " exact values; legacy Metal and independent double oracle\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        ggml_backend_free(backend);
        return 1;
    }
    ggml_backend_free(backend);
}
