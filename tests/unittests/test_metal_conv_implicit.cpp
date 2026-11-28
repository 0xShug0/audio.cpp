#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-metal.h>
#include <ggml-cpu.h>
#include "engine/framework/modules/conv_modules.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
struct Shape { int width, height, channels, outputs, kw, kh, sw, sh, pw, ph, dw, dh, batch; };

void check_opt_in() {
    using namespace engine;
    auto * ctx = ggml_init({4*1024*1024, nullptr, true});
    for (auto backend : {core::BackendType::Metal, core::BackendType::Cpu,
                         core::BackendType::Cuda, core::BackendType::Vulkan}) {
        for (int channels : {31, 32}) for (bool enabled : {false, true}) {
            core::ModuleBuildContext build{ctx, nullptr, backend};
            auto config = modules::Conv2dConfig{channels, 17, 3, 3, 1, 1, 1, 1, 1, 1, false};
            config.implicit_gemm = enabled;
            auto * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 9, 7, channels, 1);
            auto * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 3, channels, 17);
            auto * output = modules::Conv2dModule(config).build(build,
                core::wrap_tensor(x, core::TensorShape::from_dims({1, channels, 7, 9})),
                {core::wrap_tensor(w, core::TensorShape::from_dims({17, channels, 3, 3})), std::nullopt}).tensor;
            const bool selected = output->op == GGML_OP_CONV_2D && output->op_params[8] == 1;
            if (selected != (enabled && backend == core::BackendType::Metal && channels == 32)) {
                ggml_free(ctx);
                throw std::runtime_error("implicit convolution escaped its opt-in/backend/shape gate");
            }
            if (!selected && output->op != GGML_OP_CONT) {
                ggml_free(ctx);
                throw std::runtime_error("ordinary convolution graph changed");
            }
        }
    }
    ggml_free(ctx);
}

void check(ggml_backend_t backend, const Shape & shape) {
    auto * ctx = ggml_init({4*1024*1024, nullptr, true});
    ggml_backend_buffer_t buffer = nullptr;
    try {
        const auto & s = shape;
        // Both spatial and batch strides are larger than the visible input.
        auto * backing = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.width + 3, s.height + 2, s.channels, s.batch);
        auto * x = ggml_view_4d(ctx, backing, s.width, s.height, s.channels, s.batch,
            backing->nb[1], backing->nb[2], backing->nb[3], backing->nb[1] + sizeof(float));
        auto * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.kw, s.kh, s.channels, s.outputs);
        using namespace engine;
        core::ModuleBuildContext build{ctx, nullptr, core::BackendType::Metal};
        auto config = modules::Conv2dConfig{s.channels, s.outputs, s.kh, s.kw,
            s.sh, s.sw, s.ph, s.pw, s.dh, s.dw, false};
        config.implicit_gemm = true;
        const auto weight = core::wrap_tensor(w, core::TensorShape::from_dims({s.outputs, s.channels, s.kh, s.kw}));
        auto * selected = modules::Conv2dModule(config).build(build,
            core::wrap_tensor(x, core::TensorShape::from_dims({s.batch, s.channels, s.height, s.width})),
            {weight, std::nullopt}).tensor;
        if (selected->op != GGML_OP_CONV_2D || selected->op_params[8] != 1)
            throw std::runtime_error("module did not select implicit convolution");
        auto * channel_input = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
        auto * channel_output = modules::Conv2dModule(config).build(build,
            core::wrap_tensor(channel_input, core::TensorShape::from_dims({s.batch, s.height, s.width, s.channels})),
            {weight, std::nullopt}, modules::Conv2dLayout::ChannelsLast).tensor;
        auto * legacy = ggml_conv_2d(ctx, w, ggml_cont(ctx, x), s.sw, s.sh, s.pw, s.ph, s.dw, s.dh);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, legacy);
        ggml_build_forward_expand(graph, selected);
        ggml_build_forward_expand(graph, channel_output);
        int implicit_nodes = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_CONV_2D && node->op_params[8] == 1) ++implicit_nodes;
            if (node->op == GGML_OP_MUL_MAT) {
                ggml_mul_mat_set_prec(node, GGML_PREC_F32);
                ggml_mul_mat_set_lowering(node, GGML_MUL_MAT_LOWERING_METAL_F32_INPUTS);
            }
        }
        if (implicit_nodes != 2) throw std::runtime_error("both module layouts must select the new lowering");
        if (!ggml_backend_supports_op(backend, selected)) throw std::runtime_error("unsupported implicit convolution");
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) throw std::runtime_error("allocation failed");
        std::vector<float> xv(ggml_nelements(backing)), wv(ggml_nelements(w));
        std::vector<float> actual(ggml_nelements(selected)), gold(actual.size()), first;
        std::vector<double> expected(actual.size());
        const int ow = selected->ne[0], oh = selected->ne[1];
        double worst = 0;
        for (int pattern = 0; pattern < 3; ++pattern) {
            for (size_t i = 0; i < xv.size(); ++i) xv[i] = pattern == 2 ? 65536.0f : float(int((i + pattern)%13) - 6)/16;
            for (size_t i = 0; i < wv.size(); ++i) wv[i] = pattern == 2 ? 1.0f : float(int(i%11) - 5)/32;
            ggml_backend_tensor_set(backing, xv.data(), 0, xv.size()*sizeof(float));
            ggml_backend_tensor_set(w, wv.data(), 0, wv.size()*sizeof(float));
            for (int b = 0; b < s.batch; ++b) for (int oc = 0; oc < s.outputs; ++oc)
            for (int y = 0; y < oh; ++y) for (int col = 0; col < ow; ++col) {
                double value = 0;
                for (int c = 0; c < s.channels; ++c) for (int ky = 0; ky < s.kh; ++ky) for (int kx = 0; kx < s.kw; ++kx) {
                    const int ix = col*s.sw + kx*s.dw - s.pw, iy = y*s.sh + ky*s.dh - s.ph;
                    if (ix < 0 || ix >= s.width || iy < 0 || iy >= s.height) continue;
                    const int xi = ((b*s.channels + c)*(s.height + 2) + iy + 1)*(s.width + 3) + ix + 1;
                    const int wi = ((oc*s.channels + c)*s.kh + ky)*s.kw + kx;
                    value += double(xv[xi])*wv[wi];
                }
                expected[((b*s.outputs + oc)*oh + y)*ow + col] = value;
            }
            for (int repeat = 0; repeat < 2; ++repeat) {
                if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
                ggml_backend_tensor_get(selected, actual.data(), 0, ggml_nbytes(selected));
                ggml_backend_tensor_get(legacy, gold.data(), 0, ggml_nbytes(legacy));
                std::vector<float> channel_values(actual.size());
                ggml_backend_tensor_get(channel_output, channel_values.data(), 0, ggml_nbytes(channel_output));
                for (size_t i = 0; i < actual.size(); ++i) {
                    worst = std::max(worst, std::abs(actual[i] - expected[i]));
                    // Binary-rational products and sums are exactly representable;
                    // both legacy GPU and independent double must match every value.
                    if (!std::isfinite(actual[i]) || actual[i] != gold[i] || actual[i] != expected[i]) {
                        std::cerr << "index=" << i << " selected=" << actual[i] << " legacy=" << gold[i] << " double=" << expected[i] << '\n';
                        throw std::runtime_error("convolution mismatch");
                    }
                }
                for (int b = 0; b < s.batch; ++b) for (int c = 0; c < s.outputs; ++c)
                for (int y = 0; y < oh; ++y) for (int col = 0; col < ow; ++col) {
                    if (channel_values[((b*oh + y)*ow + col)*s.outputs + c] != actual[((b*s.outputs + c)*oh + y)*ow + col])
                        throw std::runtime_error("channels-last module mismatch");
                }
                if (repeat && actual != first) throw std::runtime_error("non-repeatable convolution");
                first = actual;
            }
        }
        std::cout << "PASS " << s.width << 'x' << s.height << " IC=" << s.channels << " OC=" << s.outputs
                  << " K=" << s.kw << 'x' << s.kh << " batch=" << s.batch << " max_abs=" << worst << '\n';
    } catch (...) {
        if (buffer) ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        throw;
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}
}

int main() {
    check_opt_in();
    auto backend = ggml_backend_metal_init();
    if (!backend) return 77;
    int status = 0;
    try {
        for (const auto & shape : {
            Shape{1,1,32,1,1,1,1,1,0,0,1,1,2},
            Shape{17,9,32,17,3,3,1,1,1,1,1,1,2},
            Shape{35,19,64,33,3,1,2,1,1,0,1,1,2},
            Shape{19,17,96,65,1,3,1,2,0,2,1,2,2},
            Shape{31,21,64,64,3,3,2,2,2,2,2,2,1},
            Shape{9,7,128,7,3,3,1,1,3,3,3,3,2},
            Shape{67,35,64,67,3,3,1,1,1,1,1,1,1},
            Shape{33,13,256,64,3,3,1,1,1,4,1,4,1}}) check(backend, shape);
        auto cpu = ggml_backend_cpu_init();
        if (!cpu) throw std::runtime_error("CPU correctness backend unavailable");
        try { check(cpu, {17,9,32,17,3,3,1,1,1,1,1,1,2}); }
        catch (...) { ggml_backend_free(cpu); throw; }
        ggml_backend_free(cpu);
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    ggml_backend_free(backend);
    return status;
}
