#include "engine/framework/neon/blocks.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace engine::neon;

namespace {

std::vector<float> values(size_t count, float scale = 0.07F) {
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) result[i] = scale * std::sin(float(i) * 1.13F + 0.3F);
    return result;
}

void close(const std::vector<float> &actual, const std::vector<float> &expected, float tolerance = 3e-5F) {
    if (actual.size() != expected.size()) throw std::runtime_error("output length mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > tolerance) {
            throw std::runtime_error("numerical mismatch at " + std::to_string(i) + ": " +
                std::to_string(actual[i]) + " vs " + std::to_string(expected[i]));
        }
    }
}

std::vector<float> conv(const std::vector<float> &x, int in, int out, int kernel, int dilation,
    const Conv1dWeights &w) {
    const int frames = int(x.size()) / in, padding = (kernel - 1) * dilation / 2;
    std::vector<float> result(out * frames);
    for (int o = 0; o < out; ++o) for (int t = 0; t < frames; ++t) {
        double sum = w.bias.empty() ? 0.0 : w.bias[o];
        for (int c = 0; c < in; ++c) for (int k = 0; k < kernel; ++k) {
            const int source = t + k * dilation - padding;
            if (source >= 0 && source < frames) sum += double(x[c * frames + source]) * w.weight[(o * in + c) * kernel + k];
        }
        result[o * frames + t] = float(sum);
    }
    return result;
}

std::vector<float> transpose(const std::vector<float> &x, ConvTranspose1dConfig c, const Conv1dWeights &w) {
    const int frames = int(x.size()) / c.input_channels;
    const int length = (frames - 1) * c.stride - 2 * c.padding + c.kernel_size + c.output_padding;
    std::vector<double> sums(c.output_channels * length);
    for (int o = 0; o < c.output_channels; ++o) {
        for (int t = 0; t < length; ++t) sums[o * length + t] = w.bias.empty() ? 0 : w.bias[o];
        for (int in = 0; in < c.input_channels; ++in) for (int t = 0; t < frames; ++t)
            for (int k = 0; k < c.kernel_size; ++k) {
                const int target = t * c.stride + k - c.padding;
                if (target >= 0 && target < length)
                    sums[o * length + target] += double(x[in * frames + t]) * w.weight[(in * c.output_channels + o) * c.kernel_size + k];
            }
    }
    return {sums.begin(), sums.end()};
}

void test_transpose() {
    for (int stride : {1, 2, 3, 6, 10}) for (int kernel : {1, 3, 6, 12, 20}) {
        for (int padding : {0, (kernel - 1) / 2}) for (int output_padding : {0, stride - 1}) {
            ConvTranspose1dConfig c{3, 5, kernel, stride, padding, output_padding};
            Conv1dWeights w{values(3 * 5 * kernel), values(5)};
            ExecutionPlan plan;
            auto input = plan.input();
            plan.set_output(build_conv_transpose1d(plan, input, c, w.weight, w.bias));
            for (int threads : {1, 4}) for (int frames : {1, 3, 17, 3}) {
                auto x = values(3 * frames, 0.2F);
                close(plan.run({TensorView(x.data(), x.size(), {3, frames})}, threads).values, transpose(x, c, w));
            }
        }
    }
}

void leaky(std::vector<float> &x, float slope) { for (float &v : x) if (v < 0) v *= slope; }

void test_hifigan(bool paired, bool conditioned) {
    HifiGanConfig c;
    c.input_channels = 3; c.initial_channels = 8;
    c.conditioning_channels = conditioned ? 2 : 0;
    c.upsample_rates = {2, 3}; c.upsample_kernels = {4, 6};
    c.residual_kernels = {3, 5}; c.residual_dilations = {{1, 3}, {1, 2}};
    c.paired_convolutions = paired;
    c.output_negative_slope = 0.01F;
    HifiGanWeights w;
    auto weights = [](int in, int out, int kernel) -> Conv1dWeights { return {values(in * out * kernel), values(out)}; };
    w.pre = weights(3, 8, 7); w.post = weights(2, 1, 7);
    if (conditioned) w.conditioning = weights(2, 8, 1);
    for (int stage = 0; stage < 2; ++stage) {
        const int in = 8 >> stage, out = in / 2;
        w.upsample.push_back({values(in * out * c.upsample_kernels[stage]), values(out)});
        for (int kernel : c.residual_kernels) {
            HifiGanWeights::Residual block;
            for (int layer = 0; layer < 2; ++layer) {
                block.first.push_back(weights(out, out, kernel));
                if (paired) block.second.push_back(weights(out, out, kernel));
            }
            w.residual.push_back(std::move(block));
        }
    }
    ExecutionPlan plan;
    auto input = plan.input();
    std::optional<ExecutionPlan::Value> conditioning;
    if (conditioned) conditioning = plan.input();
    plan.set_output(build_hifigan(plan, input, conditioning, c, w));
    for (int threads : {1, 4}) for (int frames : {3, 9, 3}) for (int gframes : {1, frames}) {
        auto x = values(3 * frames, 0.3F), g = values(2 * gframes, 0.5F);
        auto reference = conv(x, 3, 8, 7, 1, w.pre);
        if (conditioned) {
            auto projected = conv(g, 2, 8, 1, 1, w.conditioning);
            for (int ch = 0; ch < 8; ++ch) for (int t = 0; t < frames; ++t)
                reference[ch * frames + t] += projected[ch * gframes + (gframes == 1 ? 0 : t)];
        }
        int channels = 8;
        for (int stage = 0; stage < 2; ++stage) {
            leaky(reference, c.negative_slope);
            reference = transpose(reference, {channels, channels / 2, c.upsample_kernels[stage],
                c.upsample_rates[stage], (c.upsample_kernels[stage] - c.upsample_rates[stage]) / 2}, w.upsample[stage]);
            channels /= 2;
            std::vector<float> sum(reference.size(), 0);
            for (int k = 0; k < 2; ++k) {
                auto residual = reference;
                const auto &block = w.residual[stage * 2 + k];
                for (int layer = 0; layer < 2; ++layer) {
                    auto h = residual;
                    leaky(h, c.negative_slope);
                    h = conv(h, channels, channels, c.residual_kernels[k], c.residual_dilations[k][layer], block.first[layer]);
                    if (paired) {
                        leaky(h, c.negative_slope);
                        h = conv(h, channels, channels, c.residual_kernels[k], 1, block.second[layer]);
                    }
                    for (size_t i = 0; i < h.size(); ++i) residual[i] += h[i];
                }
                for (size_t i = 0; i < sum.size(); ++i) sum[i] += residual[i];
            }
            for (float &v : sum) v *= 0.5F;
            reference = std::move(sum);
        }
        leaky(reference, c.output_negative_slope);
        reference = conv(reference, channels, 1, 7, 1, w.post);
        for (float &v : reference) v = std::tanh(v);
        std::vector<TensorView> inputs{TensorView(x.data(), x.size(), {3, frames})};
        if (conditioned) inputs.emplace_back(g.data(), g.size(), Shape{2, gframes});
        close(plan.run(inputs, threads).values, reference);
    }
}

void test_style_ops() {
    ExecutionPlan plan;
    auto input = plan.input(), affine = plan.input();
    auto x = plan.add(AdaptiveInstanceNorm(1e-5F), {input, affine});
    x = plan.add(Snake({0.5F, -1.5F}), {x});
    plan.set_output(plan.add(ReflectPad1d(1, 2), {x}));
    for (int threads : {1, 4}) for (int frames : {3, 17, 3}) {
        auto data = values(2 * frames, 0.5F), parameters = values(4);
        std::vector<float> reference(2 * (frames + 3));
        for (int c = 0; c < 2; ++c) {
            double mean = 0, variance = 0;
            for (int t = 0; t < frames; ++t) mean += data[c * frames + t];
            mean /= frames;
            for (int t = 0; t < frames; ++t) variance += std::pow(double(data[c * frames + t]) - mean, 2);
            std::vector<float> channel(frames);
            for (int t = 0; t < frames; ++t) {
                float v = float((data[c * frames + t] - mean) / std::sqrt(variance / frames + 1e-5));
                v = v * (1 + parameters[c]) + parameters[c + 2];
                const float alpha = c == 0 ? 0.5F : -1.5F, s = std::sin(alpha * v);
                channel[t] = v + s * s / alpha;
            }
            for (int t = 0; t < frames + 3; ++t) {
                int source = t - 1;
                if (source < 0) source = -source;
                if (source >= frames) source = 2 * frames - 2 - source;
                reference[c * (frames + 3) + t] = channel[source];
            }
        }
        close(plan.run({TensorView(data.data(), data.size(), {2, frames}),
            TensorView(parameters.data(), parameters.size(), {4, 1})}, threads).values, reference);
    }
}

void test_validation() {
    auto rejects = [](auto operation) {
        try { operation(); } catch (const std::invalid_argument &) { return; }
        throw std::runtime_error("invalid configuration accepted");
    };
    rejects([] { Snake({0.0F}); });
    rejects([] { AdaptiveInstanceNorm(0.0F); });
    rejects([] { ReflectPad1d(-1, 0); });
    rejects([] {
        ExecutionPlan plan;
        build_conv_transpose1d(plan, plan.input(), {2, 3, 4, 0}, {});
    });
    rejects([] {
        ExecutionPlan plan;
        HifiGanConfig c;
        build_hifigan(plan, plan.input(), std::nullopt, c, {});
    });
    rejects([] {
        ExecutionPlan plan;
        auto x = plan.input(), g = plan.input();
        plan.set_output(plan.add(BroadcastAdd(), {x, g}));
        std::vector<float> input(6), conditioning(4);
        plan.run({TensorView(input, {2, 3}), TensorView(conditioning, {2, 2})}, 1);
    });
}

} // namespace

int main() {
    if (!available()) return 77;
    try {
        test_transpose();
        test_style_ops();
        test_validation();
        for (bool paired : {false, true}) for (bool conditioned : {false, true}) test_hifigan(paired, conditioned);
        std::cout << "NEON vocoder operators and complete HiFi-GAN match independent references\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
