#include "engine/framework/neon/execution.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace n = engine::neon;

void expect(const std::vector<float> &actual, const std::vector<float> &expected, float tolerance = 0) {
    if (actual.size() != expected.size()) throw std::runtime_error("NEON sequence shape mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > tolerance) {
            throw std::runtime_error("NEON sequence scalar parity failed at " + std::to_string(i));
        }
    }
}

int main() try {
    for (int threads : {1, 4}) {
        n::ExecutionPlan plan;
        const auto a = plan.input(), b = plan.input();
        const auto joined = plan.add(n::Concat(1), {a, b});
        const auto sliced = plan.add(n::Slice(1, 1, 3), {joined});
        const auto transposed = plan.add(n::Transpose2d(), {sliced});
        plan.set_output(plan.add(n::Reshape({6, 1}), {transposed}));
        expect(plan.run({n::TensorView(std::vector<float>{1,2,3,4,5,6}, {2,3}),
                         n::TensorView(std::vector<float>{7,8,9,10}, {2,2})}, threads).values,
               {2,5,3,6,7,9});

        for (float scale : {0.25f, 0.07f, 1.0f / 256}) for (int zero_point : {-128, 0, 128}) {
            const int minimum = zero_point == 128 ? 0 : -128, maximum = zero_point == 128 ? 255 : 127;
            std::vector<float> input{-100, -32.125f, -1.375f, -0.375f, -0.125f, 0, 0.125f, 0.375f, 1.375f, 31.875f, 100};
            input.push_back(-0.5f * scale);
            input.push_back(0.5f * scale);
            std::vector<float> expected;
            for (float value : input) expected.push_back(std::clamp(std::round(value * (1.0f / scale)), float(minimum - zero_point), float(maximum - zero_point)) * scale);
            n::ExecutionPlan quantization;
            const auto in = quantization.input();
            quantization.set_output(quantization.add(n::QuantizeDequantize(scale, zero_point, minimum, maximum), {in}));
            expect(quantization.run({n::TensorView(input, {1,int(input.size())})}, threads).values, expected);
        }

        n::ExecutionPlan activation;
        const auto in = activation.input();
        const auto value = activation.add(n::Clamp(-2, 2), {in});
        activation.set_output(activation.add(n::Activation(n::ActivationKind::Sigmoid), {value}));
        expect(activation.run({n::TensorView(std::vector<float>{-8,-1,0,1,8}, {1,5})}, threads).values,
               {1/(1+std::exp(2.0f)), 1/(1+std::exp(1.0f)), 0.5f, 1/(1+std::exp(-1.0f)), 1/(1+std::exp(-2.0f))});

        for (int kernel : {1, 3, 5}) for (int stride : {1, 2}) for (int dilation : {1, 2}) {
            const int channels = 3, frames = 15;
            std::vector<float> input(channels * frames), weight(channels * kernel), bias{0.1f, -0.2f, 0.3f};
            for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i));
            for (size_t i = 0; i < weight.size(); ++i) weight[i] = std::cos(float(i));
            n::Conv1dConfig config{channels, channels, kernel, stride, dilation, 1, 2};
            const int count = (frames + 3 - dilation * (kernel - 1) - 1) / stride + 1;
            std::vector<float> expected(channels * count);
            for (int channel = 0; channel < channels; ++channel) for (int frame = 0; frame < count; ++frame) {
                float sum = bias[channel];
                for (int tap = 0; tap < kernel; ++tap) {
                    const int at = frame * stride - 1 + tap * dilation;
                    if (at >= 0 && at < frames) sum += input[channel * frames + at] * weight[channel * kernel + tap];
                }
                expected[channel * count + frame] = sum;
            }
            n::ExecutionPlan convolution;
            const auto signal = convolution.input();
            convolution.set_output(convolution.add(n::DepthwiseConv1d(config, weight, bias), {signal}));
            expect(convolution.run({n::TensorView(input, {channels, frames})}, threads).values, expected, 2e-6f);
        }
    }
    std::cout << "NEON tensor operators passed\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
