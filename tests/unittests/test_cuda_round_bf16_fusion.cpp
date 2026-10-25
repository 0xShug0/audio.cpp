#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include "engine/framework/core/backend.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

// Checks that the CUDA kernels which fold GGML_UNARY_OP_ROUND_BF16 into the
// preceding op (RMS_NORM*w, residual ADD, bf16-reference SwiGLU) are bit-identical
// to the unfused node-by-node evaluation. Marking an intermediate as an output
// disables the fusion, which gives the reference result on the same backend.

namespace {
enum class Pattern { RmsNormMul, Add, SwiGlu };

std::vector<float> values(size_t count, float scale, float phase, bool exceptional) {
    std::vector<float> out(count);
    for (size_t i = 0; i < count; ++i) {
        out[i] = std::sin(float(i) * 0.37f + phase) * scale;
    }
    if (exceptional && count >= 10) {
        out[0] = 0.0f;
        out[1] = -0.0f;
        out[2] = std::numeric_limits<float>::infinity();
        out[3] = -std::numeric_limits<float>::infinity();
        out[4] = std::numeric_limits<float>::quiet_NaN();
        out[5] = std::numeric_limits<float>::denorm_min();
        out[6] = 1.0e-30f;
        out[7] = 3.0e38f;
        out[8] = 1.00390625f;   // exactly halfway between two bf16 values
        out[9] = -65504.5f;
    }
    return out;
}

std::vector<float> run(ggml_backend_t backend, Pattern pattern, int width, int rows, bool reference) {
    ggml_context * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    if (!ctx) throw std::runtime_error("context allocation failed");
    std::vector<ggml_tensor *> inputs;
    std::vector<std::vector<float>> data;
    ggml_tensor * output = nullptr;
    auto input = [&](ggml_tensor * t, float scale, float phase, bool exceptional = true) {
        ggml_set_input(t);
        inputs.push_back(t);
        data.push_back(values(static_cast<size_t>(ggml_nelements(t)), scale, phase, exceptional));
        return t;
    };
    if (pattern == Pattern::RmsNormMul) {
        // Keep the normalized rows finite so the row scale itself is meaningful.
        auto * x = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows), 4.0f, 0.1f, false);
        auto * w = input(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width), 1.5f, 0.7f);
        auto * mul = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), w);
        if (reference) ggml_set_output(mul);
        output = ggml_round_bf16(ctx, mul);
    } else if (pattern == Pattern::Add) {
        auto * a = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows), 9.0f, 0.2f);
        auto * b = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows), 3.0f, 1.3f);
        auto * sum = ggml_add(ctx, a, b);
        if (reference) ggml_set_output(sum);
        output = ggml_round_bf16(ctx, sum);
    } else {
        // Packed [gate | up] rows, as produced by a fused gate_up projection.
        auto * packed = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * width, rows), 6.0f, 0.4f);
        auto * gate = ggml_view_2d(ctx, packed, width, rows, packed->nb[1], 0);
        auto * up = ggml_view_2d(ctx, packed, width, rows, packed->nb[1], width * sizeof(float));
        auto * silu = ggml_round_bf16(ctx, ggml_silu(ctx, ggml_round_bf16(ctx, gate)));
        if (reference) ggml_set_output(silu);
        output = ggml_round_bf16(ctx, ggml_mul(ctx, silu, ggml_round_bf16(ctx, up)));
    }
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output);
    auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, graph)) throw std::runtime_error("graph allocation failed");

    for (int repeat = 0; repeat < 3; ++repeat) {
        // The graph allocator may reuse input storage for intermediates.
        for (size_t i = 0; i < inputs.size(); ++i) {
            ggml_backend_tensor_set(inputs[i], data[i].data(), 0, data[i].size() * sizeof(float));
        }
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("CUDA graph failed");
    }
    ggml_backend_synchronize(backend);
    std::vector<float> result(static_cast<size_t>(ggml_nelements(output)));
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));

    // Clear captured graph resources before freeing the allocator/context.
    auto device = ggml_backend_get_device(backend);
    auto reg = ggml_backend_dev_backend_reg(device);
    auto clear = reinterpret_cast<void (*)(ggml_backend_t, ggml_cgraph *)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_clear_graph"));
    if (clear) clear(backend, graph);
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return result;
}
}

int main() {
    engine::core::ensure_backends_loaded();
    auto reg = ggml_backend_reg_by_name("CUDA");
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) return 77;
    auto backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
    if (!backend) return 1;
    const char * names[] = {"rms_norm*w", "add", "swiglu"};
    try {
        for (Pattern pattern : {Pattern::RmsNormMul, Pattern::Add, Pattern::SwiGlu}) {
            for (int width : {1, 17, 1024, 2048, 8192}) {
                for (int rows : {1, 4}) {
                    const auto reference = run(backend, pattern, width, rows, true);
                    const auto fused = run(backend, pattern, width, rows, false);
                    for (size_t i = 0; i < reference.size(); ++i) {
                        if (std::isnan(reference[i]) && std::isnan(fused[i])) continue;
                        if (std::memcmp(&reference[i], &fused[i], sizeof(float)) != 0) {
                            std::cerr << names[static_cast<int>(pattern)] << " width=" << width << " rows=" << rows
                                      << " index=" << i << " reference=" << std::hexfloat << reference[i]
                                      << " fused=" << fused[i] << '\n';
                            throw std::runtime_error("fused output differs from unfused CUDA output");
                        }
                    }
                }
            }
        }
        std::cout << "PASS: bf16 rounding fused into rms_norm*w, add and swiglu; odd widths, multiple rows, "
                     "strided gate/up views, exceptional values\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        ggml_backend_free(backend);
        return 1;
    }
    ggml_backend_free(backend);
}
