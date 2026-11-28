#include <ggml.h>
#include <ggml-backend.h>
#include "app/cli/args.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Storage {
    ggml_context * ctx = ggml_init({4*1024*1024, nullptr, true});
    ggml_backend_buffer_t buffer = nullptr;
    ~Storage() {
        if (buffer) ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    void allocate(ggml_backend_t backend) {
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) throw std::runtime_error("allocation failed");
    }
};

void upload(ggml_tensor * tensor, const std::vector<float> & values) {
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size()*sizeof(float));
}

std::vector<float> read(ggml_tensor * tensor) {
    std::vector<float> result(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, ggml_nbytes(tensor));
    return result;
}

template<class T> void compare(const std::vector<float> & actual, const std::vector<T> & expected,
                              double tolerance, const char * reference) {
    if (actual.size() != expected.size()) throw std::runtime_error("size mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - double(expected[i])) > tolerance*(1 + std::abs(expected[i]))) {
            std::cerr << reference << " index=" << i << " actual=" << actual[i] << " expected=" << expected[i] << '\n';
            throw std::runtime_error("numerical mismatch");
        }
    }
}

void compute(ggml_backend_t backend, ggml_cgraph * graph) {
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
}

void convolution(ggml_backend_t backend, int channels, int tokens, bool strided, int kernel = 4) {
    constexpr int batch = 2;
    Storage storage;
    auto * ctx = storage.ctx;
    const int pitch = strided ? 2*channels + 3 : channels;
    auto * backing = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, pitch, tokens, batch);
    auto * channels_first = ggml_view_3d(ctx, backing, channels, tokens, batch,
        pitch*sizeof(float), pitch*tokens*sizeof(float), (strided ? channels : 0)*sizeof(float));
    auto * x = ggml_permute(ctx, channels_first, 1, 0, 2, 3);
    auto * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kernel, channels);
    auto * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
    auto * selected = ggml_ssm_conv_causal_silu(ctx, x, w, bias);
    auto * padded = ggml_pad_ext(ctx, ggml_cont(ctx, x), kernel - 1, 0, 0, 0, 0, 0, 0, 0);
    auto * legacy = ggml_silu(ctx, ggml_add(ctx, ggml_ssm_conv(ctx, padded, w), bias));
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, legacy);
    ggml_build_forward_expand(graph, selected);
    if (!ggml_backend_supports_op(backend, selected)) throw std::runtime_error("unsupported causal convolution");
    storage.allocate(backend);
    std::vector<float> xv(ggml_nelements(backing)), wv(kernel*channels), bv(channels);
    for (int pattern = 0; pattern < 3; ++pattern) {
        for (size_t i = 0; i < xv.size(); ++i) xv[i] = float(int((i + pattern*3)%23) - 11)/16;
        for (size_t i = 0; i < wv.size(); ++i) wv[i] = float(int(i%9) - 4)/8;
        for (int c = 0; c < channels; ++c) bv[c] = float(c%7 - 3)/4;
        upload(backing, xv); upload(w, wv); upload(bias, bv);
        std::vector<double> expected(ggml_nelements(selected));
        for (int b = 0; b < batch; ++b) for (int c = 0; c < channels; ++c) for (int t = 0; t < tokens; ++t) {
            double sum = 0;
            for (int k = 0; k < kernel; ++k) if (t + k >= kernel - 1)
                sum += double(xv[(b*tokens + t + k - kernel + 1)*pitch + c + (strided ? channels : 0)])*wv[c*kernel + k];
            sum += bv[c];
            expected[(b*tokens + t)*channels + c] = sum/(1 + std::exp(-sum));
        }
        std::vector<float> first;
        for (int repeat = 0; repeat < 2; ++repeat) {
            compute(backend, graph);
            const auto result = read(selected);
            compare(result, read(legacy), 2e-6, "legacy convolution GPU");
            compare(result, expected, 2e-6, "double convolution");
            if (repeat && first != result) throw std::runtime_error("non-repeatable convolution");
            first = result;
        }
    }
}

void scan(ggml_backend_t backend, int channels, int tokens, int rank, bool strided, bool scalar_a) {
    constexpr int states = 16, batch = 2, slots = 3;
    const bool vulkan = engine::core::backend_type(backend) == engine::core::BackendType::Vulkan;
    Storage storage;
    auto * ctx = storage.ctx;
    const int xp = channels + (strided ? 3 : 0), pp = rank + 32 + (strided ? 5 : 0);
    const int gp = channels + (strided ? channels : 0), offset = strided ? 2 : 0;
    auto * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, states, 1, channels, slots);
    auto * xb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, xp, tokens, batch);
    auto * x = ggml_view_4d(ctx, xb, 1, channels, tokens, batch,
        sizeof(float), xp*sizeof(float), xp*tokens*sizeof(float), 0);
    auto * pb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, pp, tokens, batch);
    auto * p = ggml_view_3d(ctx, pb, rank + 32, tokens, batch, pp*sizeof(float), pp*tokens*sizeof(float), offset*sizeof(float));
    auto * gb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, gp, tokens, batch);
    auto * gate = ggml_view_3d(ctx, gb, channels, tokens, batch, gp*sizeof(float), gp*tokens*sizeof(float), (strided ? channels : 0)*sizeof(float));
    auto * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, scalar_a ? 1 : states, channels);
    auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch);
    auto * dw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, channels);
    auto * db = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
    auto * D = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, channels);
    auto * selected = ggml_ssm_scan_mamba1(ctx, s, x, p, a, ids, dw, db, D, gate);
    auto * dt_input = ggml_cont(ctx, ggml_view_3d(ctx, p, rank, tokens, batch, p->nb[1], p->nb[2], 0));
    auto * dt = ggml_add(ctx, ggml_mul_mat(ctx, dw, dt_input), db);
    ggml_mul_mat_set_prec(dt->src[0], GGML_PREC_F32);
    auto bc = [&](int start) {
        auto * view = ggml_view_4d(ctx, p, states, 1, tokens, batch,
            states*sizeof(float), p->nb[1], p->nb[2], start*sizeof(float));
        // The legacy Vulkan scan cannot bind arbitrarily offset B/C views.
        // Materialize reference inputs; the new compact op still receives the
        // original strided, offset parameter tensor and is checked against double.
        return vulkan ? ggml_cont(ctx, view) : view;
    };
    auto * old_scan = ggml_ssm_scan(ctx, s, x, dt, a, bc(rank), bc(rank + states), ids);
    ggml_ssm_scan_set_lowering(old_scan, GGML_SSM_SCAN_LOWERING_METAL_REGISTER16);
    auto * old_y = ggml_view_3d(ctx, old_scan, channels, tokens, batch, channels*sizeof(float), channels*tokens*sizeof(float), 0);
    auto * x3 = ggml_view_3d(ctx, x, channels, tokens, batch, x->nb[2], x->nb[3], 0);
    auto * pre_gate = ggml_add(ctx, old_y, ggml_mul(ctx, x3, D));
    // Legacy split-GLU dispatches zero threads for a one-channel row.
    // Retain that edge shape using its unfused legacy GPU SiLU/MUL equivalent.
    auto * legacy_gate = vulkan
        ? ggml_cont(ctx, gate) : gate;
    auto * legacy = channels == 1 ? ggml_mul(ctx, ggml_silu(ctx, legacy_gate), pre_gate)
        : ggml_swiglu_split(ctx, legacy_gate, pre_gate);
    auto * graph = ggml_new_graph(ctx);
    const bool legacy_supported = !vulkan || ggml_backend_supports_op(backend, old_scan);
    if (legacy_supported) ggml_build_forward_expand(graph, legacy);
    ggml_build_forward_expand(graph, selected);
    if (!ggml_backend_supports_op(backend, selected)) throw std::runtime_error("unsupported scan");
    storage.allocate(backend);
    const int32_t idv[] = {2, 0};
    ggml_backend_tensor_set(ids, idv, 0, sizeof(idv));
    std::vector<float> sv(ggml_nelements(s)), xv(ggml_nelements(xb)), pv(ggml_nelements(pb));
    std::vector<float> av(ggml_nelements(a)), wv(ggml_nelements(dw)), bv(channels), dv(channels), gv(ggml_nelements(gb));
    for (int pattern = 0; pattern < 3; ++pattern) {
        for (size_t i = 0; i < sv.size(); ++i) sv[i] = float(int(i%11) - 5)/32;
        for (size_t i = 0; i < xv.size(); ++i) xv[i] = float(std::sin(i*.07 + pattern))*.25f;
        for (size_t i = 0; i < pv.size(); ++i) pv[i] = float(int((i + pattern)%17) - 8)/64;
        for (size_t i = 0; i < av.size(); ++i) av[i] = -.2f - float(i%7)/32;
        for (size_t i = 0; i < wv.size(); ++i) wv[i] = float(int(i%7) - 3)/16;
        for (int c = 0; c < channels; ++c) { bv[c] = pattern == 2 ? 21.0f : -2.5f; dv[c] = float(c%5 - 2)/8; }
        for (size_t i = 0; i < gv.size(); ++i) gv[i] = float(int((i + pattern)%13) - 6)/8;
        upload(s, sv); upload(xb, xv); upload(pb, pv); upload(a, av); upload(dw, wv);
        upload(db, bv); upload(D, dv); upload(gb, gv);
        std::vector<double> expected(ggml_nelements(selected));
        for (int b = 0; b < batch; ++b) for (int c = 0; c < channels; ++c) {
            double state[states];
            for (int j = 0; j < states; ++j) state[j] = sv[(idv[b]*channels + c)*states + j];
            for (int t = 0; t < tokens; ++t) {
                const int pi = (b*tokens + t)*pp + offset, xi = (b*tokens + t)*xp + c;
                double delta = bv[c];
                for (int k = 0; k < rank; ++k) delta += double(wv[c*rank + k])*pv[pi + k];
                delta = delta > 20 ? delta : std::log1p(std::exp(delta));
                double y = 0;
                for (int j = 0; j < states; ++j) {
                    state[j] = state[j]*std::exp(delta*av[c*a->ne[0] + (scalar_a ? 0 : j)]) + pv[pi + rank + j]*xv[xi]*delta;
                    y += state[j]*pv[pi + rank + states + j];
                }
                const double z = gv[(b*tokens + t)*gp + c + (strided ? channels : 0)];
                expected[(b*tokens + t)*channels + c] = z/(1 + std::exp(-z))*(y + xv[xi]*double(dv[c]));
            }
            for (int j = 0; j < states; ++j) expected[channels*tokens*batch + (b*channels + c)*states + j] = state[j];
        }
        std::vector<float> first;
        for (int repeat = 0; repeat < 2; ++repeat) {
            compute(backend, graph);
            auto actual = read(selected);
            std::vector<float> old;
            if (legacy_supported) {
                old = read(old_scan);
                const auto old_output = read(legacy);
                std::copy(old_output.begin(), old_output.end(), old.begin());
            }
            if (repeat == 0) std::cout << "scan channels=" << channels << " tokens=" << tokens << " rank=" << rank
                << " strided=" << strided << " scalar_a=" << scalar_a << " pattern=" << pattern
                << " legacy_gpu=" << legacy_supported << std::endl;
            if (legacy_supported) {
                compare(old, expected, 2e-5, "legacy scan versus double");
                compare(actual, old, 2e-5, "legacy scan GPU and final states");
            }
            compare(actual, expected, 2e-5, "double Mamba recurrence and epilogue");
            if (repeat && actual != first) throw std::runtime_error("non-repeatable scan");
            first = actual;
            if (read(s) != sv) throw std::runtime_error("initial state mutated");
        }
    }
}
}

int main(int argc, char ** argv) {
    const auto backend_name = minitts::cli::find_arg(argc, argv, "--backend").value_or("metal");
    if (!minitts::cli::has_arg(argc, argv, "--log")) throw std::runtime_error("--log is required");
    engine::debug::configure_logging({true, minitts::cli::find_arg(argc, argv, "--log-file")});
    const auto devices = engine::core::list_backend_devices();
    if (std::none_of(devices.begin(), devices.end(), [&](const auto & device) { return device.backend_option == backend_name; })) {
        engine::debug::log_message("SKIP: requested backend has no available device");
        return 77;
    }
    engine::core::ExecutionContext execution({minitts::cli::parse_backend(backend_name), 0, 8});
    auto backend = execution.backend();
    const auto started = std::chrono::steady_clock::now();
    int result = 0;
    try {
        for (int channels : {1, 17, 129}) for (int tokens : {1, 3, 33, 257, 1003})
        for (bool strided : {false, true}) convolution(backend, channels, tokens, strided);
        for (int channels : {1, 17, 129}) for (int tokens : {1, 7, 257}) for (int rank : {1, 3, 4, 8, 16, 33, 64})
        for (bool strided : {false, true}) for (bool scalar : {false, true}) scan(backend, channels, tokens, rank, strided, scalar);
        if (execution.backend_type() == engine::core::BackendType::Vulkan) {
            for (int kernel : {1, 3, 7}) for (int tokens : {1, 3, 33})
                convolution(backend, 17, tokens, true, kernel);
        }
        convolution(backend, 256, 40411, true);
        scan(backend, 128, 257, 4, true, false);
        scan(backend, 256, backend_name == "metal" ? 40411 : 3001, 4, true, false);
        auto cpu = ggml_backend_cpu_init();
        if (!cpu) throw std::runtime_error("CPU correctness backend unavailable");
        try {
            convolution(cpu, 17, 33, true);
            scan(cpu, 17, 7, 4, true, false);
            scan(cpu, 17, 7, 8, true, true);
        } catch (...) { ggml_backend_free(cpu); throw; }
        ggml_backend_free(cpu);
        std::cout << "PASS: legacy GPU where supported and independent double references; strided inputs, remapped states, bias threshold, repeated updates, production shape\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        result = 1;
    }
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
    return result;
}
