#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-metal.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

struct Shape {
    int64_t k, m, n;
    int64_t a2 = 1, b2 = 1, b3 = 1;
    bool view = false;
};

struct Graph {
    ggml_context * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    ggml_backend_buffer_t buffer = nullptr;
    ~Graph() {
        if (buffer) ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
};

void run_case(ggml_backend_t backend, Shape s, int pattern) {
    Graph g;
    const auto stride = s.k + (s.view ? 4 : 0);
    auto * a_storage = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, stride, s.m, s.a2, 1);
    auto * b_storage = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, stride, s.n, s.b2, s.b3);
    auto * a = s.view ? ggml_view_4d(g.ctx, a_storage, s.k, s.m, s.a2, 1,
        a_storage->nb[1], a_storage->nb[2], a_storage->nb[3], 0) : a_storage;
    auto * b = s.view ? ggml_view_4d(g.ctx, b_storage, s.k, s.n, s.b2, s.b3,
        b_storage->nb[1], b_storage->nb[2], b_storage->nb[3], 0) : b_storage;
    auto * precise = ggml_mul_mat(g.ctx, a, b);
    ggml_mul_mat_set_lowering(precise, GGML_MUL_MAT_LOWERING_METAL_F32_INPUTS);
    auto * graph = ggml_new_graph_custom(g.ctx, 128, false);
    ggml_build_forward_expand(graph, precise);
    ggml_tensor * normal = nullptr;
    if (pattern == 0) {
        normal = ggml_mul_mat(g.ctx, a, b);
        ggml_build_forward_expand(graph, normal);
    }
    require(ggml_backend_supports_op(backend, precise), "full-F32 op unsupported");
    g.buffer = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
    require(g.buffer != nullptr, "allocation failed");
    std::vector<float> av(ggml_nelements(a_storage), 0), bv(ggml_nelements(b_storage), 0);
    for (int64_t batch = 0; batch < s.a2; ++batch)
        for (int64_t row = 0; row < s.m; ++row)
            for (int64_t k = 0; k < s.k; ++k) {
                const auto i = (batch * s.m + row) * stride + k;
                av[i] = static_cast<float>((i % 23) - 11) * (pattern == 1 ? 16384.0F : 0.125F);
                if (pattern == 3) av[i] += static_cast<float>((i % 5) - 2) * 0.0001F;
            }
    for (int64_t batch = 0; batch < s.b2 * s.b3; ++batch)
        for (int64_t row = 0; row < s.n; ++row)
            for (int64_t k = 0; k < s.k; ++k) {
                const auto i = (batch * s.n + row) * stride + k;
                bv[i] = static_cast<float>((i % 17) - 8) * (pattern == 2 ? 32768.0F : 0.0625F);
                if (pattern == 3) bv[i] += static_cast<float>((i % 7) - 3) * 0.00003F;
            }
    ggml_backend_tensor_set(a_storage, av.data(), 0, av.size() * sizeof(float));
    ggml_backend_tensor_set(b_storage, bv.data(), 0, bv.size() * sizeof(float));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute failed");
    std::vector<float> output(ggml_nelements(precise)), old(output.size());
    ggml_backend_tensor_get(precise, output.data(), 0, output.size() * sizeof(float));
    if (normal) ggml_backend_tensor_get(normal, old.data(), 0, old.size() * sizeof(float));
    // Independent scalar double-precision dot products, never a second GGML graph.
    for (int64_t i3 = 0; i3 < s.b3; ++i3)
        for (int64_t i2 = 0; i2 < s.b2; ++i2)
            for (int64_t n = 0; n < s.n; ++n)
                for (int64_t m = 0; m < s.m; ++m) {
                    double expected = 0;
                    const auto ai = ((i2 / (s.b2 / s.a2)) * s.m + m) * stride;
                    const auto bi = ((i3 * s.b2 + i2) * s.n + n) * stride;
                    for (int64_t k = 0; k < s.k; ++k) expected += static_cast<double>(av[ai + k]) * bv[bi + k];
                    const auto o = ((i3 * s.b2 + i2) * s.n + n) * s.m + m;
                    require(std::isfinite(output[o]), "nonfinite full-F32 output");
                    // Dyadic fixtures are exact; the perturbed fixture also tests mantissa precision.
                    const double tolerance = pattern == 3 ? 3e-5 * (1 + std::abs(expected)) : 0;
                    require(std::abs(output[o] - expected) <= tolerance, "full-F32 output differs from scalar oracle");
                    if (normal) require(old[o] == static_cast<float>(expected), "default output changed");
                }
    std::cout << "PASS K=" << s.k << " M=" << s.m << " N=" << s.n << " batch=" << s.b2 << 'x' << s.b3
              << " lhs_batch=" << s.a2 << " view=" << s.view << " pattern=" << pattern << '\n';
}

} // namespace

int main() {
    auto * backend = ggml_backend_metal_init();
    if (!backend) return 77;
    try {
        const std::vector<Shape> cases = {
            {31, 13, 11}, {63, 67, 35}, {64, 64, 32}, {65, 67, 35},
            {127, 67, 35}, {128, 67, 35}, {129, 67, 35},
            {128, 67, 1}, {128, 67, 8}, {1536, 67, 128},
            {65, 67, 35, 1, 3, 2}, {128, 67, 35, 2, 4, 2},
            {64, 67, 35, 1, 1, 1, true}, {65, 67, 35, 1, 3, 2, true},
        };
        for (const auto & shape : cases)
            for (int pattern : {0, 1, 2, 3}) run_case(backend, shape, pattern);
        Graph invalid;
        auto * a = ggml_new_tensor_2d(invalid.ctx, GGML_TYPE_F16, 64, 64);
        auto * b = ggml_new_tensor_2d(invalid.ctx, GGML_TYPE_F32, 64, 32);
        auto * op = ggml_mul_mat(invalid.ctx, a, b);
        require(ggml_backend_supports_op(backend, op), "default F16 path unsupported");
        ggml_mul_mat_set_lowering(op, GGML_MUL_MAT_LOWERING_METAL_F32_INPUTS);
        require(!ggml_backend_supports_op(backend, op), "full-F32 hint must reject F16 operands");
        ggml_backend_free(backend);
        std::cout << "PASS: 56 cases plus opt-in type validation\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        ggml_backend_free(backend);
        return 1;
    }
}
