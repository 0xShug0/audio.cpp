#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-metal.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
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

void check(ggml_backend_t backend, int states, int tokens, bool mamba1, bool strided, ggml_ssm_scan_lowering lowering) {
    constexpr int heads = 4, groups = 2, batch = 2, state_batch = 3;
    const int dim = mamba1 ? 1 : 3;
    Storage storage;
    auto * ctx = storage.ctx;
    auto * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, states, dim, heads, state_batch);
    auto * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, heads, tokens, batch);
    auto * dt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, heads, tokens, batch);
    auto * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mamba1 ? states : 1, heads);
    const int pitch = states * groups + (strided ? 8 : 0);
    const int offset = strided ? 4 : 0;
    auto * bb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pitch, tokens * batch);
    auto * cb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pitch, tokens * batch);
    auto view = [&](ggml_tensor * base) {
        return ggml_view_4d(ctx, base, states, groups, tokens, batch,
            states * sizeof(float), pitch * sizeof(float), pitch * tokens * sizeof(float), offset * sizeof(float));
    };
    auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch);
    auto * out = ggml_ssm_scan(ctx, s, x, dt, a, view(bb), view(cb), ids);
    ggml_ssm_scan_set_lowering(out, lowering);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    if (!ggml_backend_supports_op(backend, out)) throw std::runtime_error("unsupported scan");
    storage.buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!storage.buffer) throw std::runtime_error("allocation failed");
    const std::vector<int32_t> idv = {2, 0};
    ggml_backend_tensor_set(ids, idv.data(), 0, ggml_nbytes(ids));
    std::vector<float> sv(ggml_nelements(s)), xv(ggml_nelements(x)), dv(ggml_nelements(dt));
    std::vector<float> av(ggml_nelements(a)), bv(ggml_nelements(bb)), cv(ggml_nelements(cb));
    std::vector<float> actual(ggml_nelements(out)), previous;
    double worst = 0;
    for (int pattern = 0; pattern < 3; ++pattern) {
        for (size_t i = 0; i < sv.size(); ++i) sv[i] = float(int(i % 13) - 6) / 16;
        for (size_t i = 0; i < xv.size(); ++i) xv[i] = pattern == 0 ? 0 : float(std::sin(i * 0.031 + pattern)) / 4;
        for (size_t i = 0; i < dv.size(); ++i) dv[i] = -2.5f + float(i % 5) / 8;
        for (size_t i = 0; i < av.size(); ++i) av[i] = pattern == 0 ? 0 : -0.2f - float(i % 17) / 32;
        std::fill(bv.begin(), bv.end(), 1000);
        std::fill(cv.begin(), cv.end(), -1000);
        for (int n = 0; n < batch; ++n) for (int t = 0; t < tokens; ++t)
        for (int g = 0; g < groups; ++g) for (int j = 0; j < states; ++j) {
            const int p = (n * tokens + t) * pitch + offset + g * states + j;
            bv[p] = float(int((j + t * 3 + n) % 11) - 5) / 32;
            cv[p] = float(int((j * 3 + t + n + pattern) % 17) - 8) / 32;
        }
        for (auto item : {std::make_pair(s, &sv), {x, &xv}, {dt, &dv}, {a, &av}, {bb, &bv}, {cb, &cv}}) {
            ggml_backend_tensor_set(item.first, item.second->data(), 0, ggml_nbytes(item.first));
        }
        // Independent double-precision recurrence, including remapped initial
        // states and final states. No GGML backend generates the reference.
        std::vector<double> expected(actual.size());
        for (int n = 0; n < batch; ++n) for (int h = 0; h < heads; ++h)
        for (int d = 0; d < dim; ++d) {
            std::vector<double> state(states);
            for (int j = 0; j < states; ++j) state[j] = sv[((idv[n] * heads + h) * dim + d) * states + j];
            for (int t = 0; t < tokens; ++t) {
                const int xi = ((n * tokens + t) * heads + h) * dim + d;
                const double delta = std::log1p(std::exp(double(dv[(n * tokens + t) * heads + h])));
                const int bc = (n * tokens + t) * pitch + offset + (h / (heads / groups)) * states;
                double sum = 0;
                for (int j = 0; j < states; ++j) {
                    const double decay = av[h * a->ne[0] + (mamba1 ? j : 0)];
                    state[j] = state[j] * std::exp(delta * decay) + double(bv[bc + j]) * (double(xv[xi]) * delta);
                    sum += state[j] * cv[bc + j];
                }
                expected[xi] = sum;
            }
            for (int j = 0; j < states; ++j) expected[xv.size() + ((n * heads + h) * dim + d) * states + j] = state[j];
        }
        for (int repeat = 0; repeat < 2; ++repeat) {
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
            ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
            for (size_t i = 0; i < actual.size(); ++i) {
                const double error = std::abs(actual[i] - expected[i]);
                worst = std::max(worst, error);
                // Pattern zero has an exact binary-rational reference.
                const double tolerance = pattern == 0 ? 0 : 2e-5 * (1 + std::abs(expected[i]));
                if (!std::isfinite(actual[i]) || error > tolerance) {
                    std::cerr << "FAIL states=" << states << " tokens=" << tokens << " mamba1=" << mamba1
                              << " strided=" << strided << " pattern=" << pattern << " index=" << i
                              << " actual=" << actual[i] << " expected=" << expected[i] << '\n';
                    throw std::runtime_error("recurrence mismatch");
                }
            }
            if (repeat && actual != previous) throw std::runtime_error("non-repeatable scan");
            previous = actual;
        }
    }
    std::cout << "PASS states=" << states << " tokens=" << tokens << " mamba1=" << mamba1
              << " strided=" << strided << " lowering=" << lowering << " max_abs=" << worst << '\n';
}
}

int main() {
    auto backend = ggml_backend_metal_init();
    if (!backend) return 77;
    int result = 0;
    try {
        for (int states : {16, 32, 64, 128}) for (int tokens : {1, 2, 3, 7, 33, 257})
        for (bool mamba1 : {true, false}) for (bool strided : {false, true})
        for (auto lowering : {GGML_SSM_SCAN_LOWERING_DEFAULT, GGML_SSM_SCAN_LOWERING_METAL_REGISTER16})
            check(backend, states, tokens, mamba1, strided, lowering);
        std::cout << "PASS: 192 shapes/lowerings, three input patterns, two executions per pattern\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        result = 1;
    }
    ggml_backend_free(backend);
    return result;
}
