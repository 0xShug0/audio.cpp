#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/modules/conv_modules.h"
#include "test_assert.h"

#include <cmath>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

struct Shape {
    int batch, channels, outputs, height, width, kh, kw, sh, sw, ph, pw, dh, dw;
};

void check(const Shape & s, bool depthwise, bool bias_enabled,
    engine::modules::Conv2dLayout layout, engine::modules::Conv2dImpl impl) {
    using namespace engine;
    const bool last = layout == modules::Conv2dLayout::ChannelsLast;
    const int outputs = depthwise ? s.channels : s.outputs;
    const int inputs_per_group = depthwise ? 1 : s.channels;
    const int oh = (s.height + 2 * s.ph - s.dh * (s.kh - 1) - 1) / s.sh + 1;
    const int ow = (s.width + 2 * s.pw - s.dw * (s.kw - 1) - 1) / s.sw + 1;
    const auto input_shape = last
        ? core::TensorShape::from_dims({s.batch, s.height, s.width, s.channels})
        : core::TensorShape::from_dims({s.batch, s.channels, s.height, s.width});
    const auto expected_shape = last
        ? core::TensorShape::from_dims({s.batch, oh, ow, outputs})
        : core::TensorShape::from_dims({s.batch, outputs, oh, ow});
    const auto index = [last](int n, int c, int h, int w, int channels, int height, int width) {
        return static_cast<size_t>(last ? ((n * height + h) * width + w) * channels + c
                                      : ((n * channels + c) * height + h) * width + w);
    };
    core::ExecutionContext execution({core::BackendType::Cpu, 0, 2});
    core::PreparedGraph graph(execution, {"conv2d scalar reference", 4 * 1024 * 1024, 1024});
    auto & ctx = graph.build_context();
    auto input = core::make_tensor(ctx, GGML_TYPE_F32, input_shape);
    auto weight = core::make_tensor(ctx, GGML_TYPE_F32,
        core::TensorShape::from_dims({outputs, inputs_per_group, s.kh, s.kw}));
    std::optional<core::TensorValue> bias;
    if (bias_enabled) bias = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({outputs}));
    ggml_set_input(input.tensor);
    ggml_set_input(weight.tensor);
    if (bias) ggml_set_input(bias->tensor);
    core::TensorValue output;
    if (depthwise) {
        modules::DepthwiseConv2dConfig config{s.channels, s.kh, s.kw, s.sh, s.sw, s.ph, s.pw, s.dh, s.dw, bias_enabled};
        if (impl != modules::Conv2dImpl::Default) config.impl = impl;
        const modules::DepthwiseConv2dModule module(config);
        output = last ? module.build(ctx, input, {weight, bias}, layout) : module.build(ctx, input, {weight, bias});
    } else {
        modules::Conv2dConfig config{s.channels, outputs, s.kh, s.kw, s.sh, s.sw, s.ph, s.pw, s.dh, s.dw, bias_enabled};
        if (impl != modules::Conv2dImpl::Default) config.impl = impl;
        const modules::Conv2dModule module(config);
        output = last ? module.build(ctx, input, {weight, bias}, layout) : module.build(ctx, input, {weight, bias});
    }
    for (size_t axis = 0; axis < expected_shape.rank; ++axis)
        test::require_eq(output.shape.dims[axis], expected_shape.dims[axis], "convolution output dimension");
    output = core::ensure_backend_addressable_layout(ctx, output);
    ggml_set_output(output.tensor);
    ggml_build_forward_expand(graph.graph(), output.tensor);
    graph.allocate();
    std::vector<float> x(static_cast<size_t>(input_shape.num_elements()));
    std::vector<float> w(static_cast<size_t>(outputs * inputs_per_group * s.kh * s.kw));
    std::vector<float> b(static_cast<size_t>(outputs));
    // Reuse the same graph with different inputs and weights to catch stale data.
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i < x.size(); ++i) x[i] = 0.25F * std::sin(0.31F * static_cast<float>(i) + pass);
        for (size_t i = 0; i < w.size(); ++i) w[i] = 0.2F * std::cos(0.17F * static_cast<float>(i) - pass);
        for (size_t i = 0; i < b.size(); ++i) b[i] = 0.03F * static_cast<float>(static_cast<int>(i) - 2 + pass);
        core::write_tensor_f32(input, x);
        core::write_tensor_f32(weight, w);
        if (bias) core::write_tensor_f32(*bias, b);
        graph.compute();
        const auto actual = core::read_tensor_f32(output.tensor);
        test::require_eq(actual.size(), static_cast<size_t>(expected_shape.num_elements()), "convolution output size");
        for (int n = 0; n < s.batch; ++n) for (int oc = 0; oc < outputs; ++oc)
            for (int h = 0; h < oh; ++h) for (int v = 0; v < ow; ++v) {
                double expected = bias_enabled ? b[static_cast<size_t>(oc)] : 0.0;
                for (int ic = 0; ic < inputs_per_group; ++ic)
                    for (int kh = 0; kh < s.kh; ++kh) for (int kw = 0; kw < s.kw; ++kw) {
                        const int ih = h * s.sh - s.ph + kh * s.dh;
                        const int iw = v * s.sw - s.pw + kw * s.dw;
                        if (ih < 0 || ih >= s.height || iw < 0 || iw >= s.width) continue;
                        const auto xi = index(n, depthwise ? oc : ic, ih, iw, s.channels, s.height, s.width);
                        const auto wi = static_cast<size_t>(((oc * inputs_per_group + ic) * s.kh + kh) * s.kw + kw);
                        expected += static_cast<double>(x[xi]) * w[wi];
                    }
                const auto oi = index(n, oc, h, v, outputs, oh, ow);
                test::require(std::isfinite(actual[oi]), "convolution produced non-finite output");
                test::require_close(actual[oi], static_cast<float>(expected), 2e-5F,
                    std::string(depthwise ? "depthwise" : "conv2d") + (last ? " ChannelsLast" : " ChannelsFirst") +
                    (impl == modules::Conv2dImpl::Default ? " Default" : " ChannelFast"));
            }
    }
}

}  // namespace

int main() try {
    using namespace engine::modules;
    const Shape cases[] = {
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1},
        {2, 3, 5, 7, 9, 3, 3, 1, 1, 1, 1, 1, 1},
        {1, 5, 3, 8, 11, 3, 2, 2, 1, 1, 0, 1, 1},
        {2, 3, 7, 9, 10, 2, 3, 1, 2, 0, 2, 2, 2},
        {1, 4, 3, 5, 7, 1, 1, 2, 2, 0, 0, 1, 1},
    };
    int count = 0;
    for (const auto & shape : cases) for (bool depthwise : {false, true}) for (bool bias : {false, true})
        for (auto layout : {Conv2dLayout::ChannelsFirst, Conv2dLayout::ChannelsLast})
            for (auto impl : {Conv2dImpl::Default, Conv2dImpl::ChannelFast}) {
                check(shape, depthwise, bias, layout, impl);
                ++count;
            }
    std::cout << count << " convolution cases passed against scalar reference (two executions each)\n";
    return 0;
} catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    return 1;
}
