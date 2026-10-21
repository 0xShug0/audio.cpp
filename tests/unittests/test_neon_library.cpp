#include "engine/framework/neon/blocks.h"
#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <iostream>
#include <stdexcept>

namespace arm = engine::neon;

namespace {

std::vector<float> values(size_t size) {
    std::vector<float> result(size);
    for (size_t i = 0; i < size; ++i) {
        result[i] = 0.15F * std::sin(float(i) * 0.21F);
    }
    return result;
}

void close(float actual, double expected) {
    engine::test::require(std::isfinite(actual) && std::abs(actual - expected) < 4e-5 * (1 + std::abs(expected)),
                          "native library scalar reference");
}

void linear() {
    for (int features : {3, 16, 31, 32, 320}) {
        constexpr int outputs = 7;
        const auto weight = values(features * outputs), bias = values(outputs);
        const arm::Linear operation(features, outputs, weight, bias);
        for (int frames : {1, 3, 17}) {
            const auto input = values(features * frames);
            for (auto order : {arm::Layout::ChannelFirst, arm::Layout::FrameMajor}) {
                arm::ExecutionPlan plan;
                const auto source = plan.input();
                plan.set_output(plan.add(operation, {source}), order);
                for (int threads : {1, 4, 7}) {
                    const auto &actual =
                        plan.run({arm::TensorView(input, arm::Shape({features, frames}, order))}, threads);
                    for (int row = 0; row < outputs; ++row) {
                        for (int frame = 0; frame < frames; ++frame) {
                            double expected = bias[row];
                            for (int feature = 0; feature < features; ++feature) {
                                const size_t index = order == arm::Layout::FrameMajor ? frame * features + feature
                                                                                      : feature * frames + frame;
                                expected += double(input[index]) * weight[row * features + feature];
                            }
                            close(actual.values[order == arm::Layout::FrameMajor ? frame * outputs + row
                                                                                 : row * frames + frame],
                                  expected);
                        }
                    }
                }
            }
        }
    }
}

void quantized_linear() {
    for (int features : {32, 64, 320}) {
        constexpr int outputs = 7;
        arm::BlockInt8Weights weights;
        weights.values.resize(features * outputs);
        weights.scales.resize(features * outputs / 32);
        for (size_t i = 0; i < weights.values.size(); ++i) {
            weights.values[i] = static_cast<int8_t>(int(i % 255) - 127);
        }
        for (size_t i = 0; i < weights.scales.size(); ++i) {
            weights.scales[i] = 0.001F * float(i % 5 + 1);
        }
        const arm::QuantizedLinear operation(features, outputs, weights);
        for (int frames : {1, 3, 17}) {
            const auto input = values(features * frames);
            for (auto order : {arm::Layout::ChannelFirst, arm::Layout::FrameMajor}) {
                arm::ExecutionPlan plan;
                const auto source = plan.input();
                plan.set_output(plan.add(operation, {source}), order);
                for (int threads : {1, 4, 7}) {
                    const auto &actual =
                        plan.run({arm::TensorView(input, arm::Shape({features, frames}, order))}, threads);
                    for (int row = 0; row < outputs; ++row) {
                        for (int frame = 0; frame < frames; ++frame) {
                            double expected = 0;
                            for (int block = 0; block < features / 32; ++block) {
                                float maximum = 0;
                                for (int i = 0; i < 32; ++i) {
                                    const int feature = block * 32 + i;
                                    const size_t index = order == arm::Layout::FrameMajor ? frame * features + feature
                                                                                          : feature * frames + frame;
                                    maximum = std::max(maximum, std::abs(input[index]));
                                }
                                const float scale = maximum / 127;
                                int exponent = 0;
                                std::frexp(scale, &exponent);
                                const double step = std::ldexp(1.0, std::max(exponent - 11, -24));
                                const double rounded = std::nearbyint(scale / step) * step;
                                int dot = 0;
                                for (int i = 0; i < 32; ++i) {
                                    const int feature = block * 32 + i;
                                    const size_t index = order == arm::Layout::FrameMajor ? frame * features + feature
                                                                                          : feature * frames + frame;
                                    dot += weights.values[row * features + feature] *
                                           static_cast<int>(std::nearbyint(input[index] * (scale ? 1.0F / scale : 0)));
                                }
                                expected += dot * rounded * weights.scales[row * (features / 32) + block];
                            }
                            close(actual.values[order == arm::Layout::FrameMajor ? frame * outputs + row
                                                                                 : row * frames + frame],
                                  expected);
                        }
                    }
                }
            }
        }
    }
}

void embedding_indices() {
    arm::BlockInt8Weights weight;
    weight.values.resize(3 * 64);
    weight.scales.assign(6, 0.025F);
    for (size_t i = 0; i < weight.values.size(); ++i) {
        weight.values[i] = static_cast<int8_t>(int(i % 255) - 127);
    }
    arm::ExecutionPlan plan;
    const auto input = plan.input();
    plan.set_output(plan.add(arm::Embedding(64, 3, weight), {input}));
    for (int32_t row : {2, 0, 1, 2}) {
        for (int threads : {1, 4, 7}) {
            const auto &output = plan.run({arm::TensorView(&row, 1, {1})}, threads);
            for (int i = 0; i < 64; ++i) {
                close(output.values[i], weight.values[row * 64 + i] * double(weight.scales[row * 2 + i / 32]));
            }
        }
    }
    int32_t invalid = 3;
    bool rejected = false;
    try {
        plan.run({arm::TensorView(&invalid, 1, {1})}, 4);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    engine::test::require(rejected, "embedding rejects out-of-range index before dispatch");
    invalid = 0;
    plan.run({arm::TensorView(&invalid, 1, {1})}, 4);
}

void decoder_operators() {
    for (auto layout : {arm::Layout::ChannelFirst, arm::Layout::FrameMajor}) {
        arm::ExecutionPlan rope;
        const auto features = rope.input(), position = rope.input();
        rope.set_output(rope.add(arm::RotaryEmbedding(5, 4, 10000), {features, position}), layout);
        const auto input = values(10 * 3);
        int32_t start = 7;
        const auto &actual =
            rope.run({arm::TensorView(input, arm::Shape({10, 3}, layout)), arm::TensorView(&start, 1, {1})}, 4);
        for (int head = 0; head < 2; ++head) {
            for (int frame = 0; frame < 3; ++frame) {
                auto offset = [&](int d) {
                    return layout == arm::Layout::FrameMajor ? frame * 10 + head * 5 + d : (head * 5 + d) * 3 + frame;
                };
                for (int pair = 0; pair < 2; ++pair) {
                    const double angle = (start + frame) * std::pow(10000.0, -2.0 * pair / 4);
                    close(actual.values[offset(2 * pair)],
                          input[offset(2 * pair)] * std::cos(angle) - input[offset(2 * pair + 1)] * std::sin(angle));
                    close(actual.values[offset(2 * pair + 1)],
                          input[offset(2 * pair)] * std::sin(angle) + input[offset(2 * pair + 1)] * std::cos(angle));
                }
                close(actual.values[offset(4)], input[offset(4)]);
            }
        }
    }

    constexpr int heads = 4, kv_heads = 2, dimension = 5, capacity = 7;
    const auto q = values(heads * dimension), keys = values(capacity * kv_heads * dimension);
    auto vals = values(keys.size()), current_key = values(kv_heads * dimension), current_value = current_key;
    for (float &v : vals) {
        v += 0.3F;
    }
    for (float &v : current_value) {
        v -= 0.2F;
    }
    arm::ExecutionPlan attention;
    const auto query = attention.input(), key = attention.input(), value = attention.input();
    const auto new_key = attention.input(), new_value = attention.input(), position = attention.input();
    attention.set_output(
        attention.add(arm::Attention(heads, kv_heads, dimension), {query, key, value, new_key, new_value, position}));
    for (int32_t index : {0, 1, 5, 6, 0}) {
        for (int threads : {1, 4, 7}) {
            const auto &result = attention.run(
                {arm::TensorView(q, {heads * dimension, 1}),
                 arm::TensorView(keys, arm::Shape({kv_heads * dimension, capacity}, arm::Layout::FrameMajor)),
                 arm::TensorView(vals, arm::Shape({kv_heads * dimension, capacity}, arm::Layout::FrameMajor)),
                 arm::TensorView(current_key, {kv_heads * dimension, 1}),
                 arm::TensorView(current_value, {kv_heads * dimension, 1}), arm::TensorView(&index, 1, {1})},
                threads);
            for (int h = 0; h < heads; ++h) {
                const int kh = h / (heads / kv_heads);
                std::vector<double> probabilities(index + 1);
                double total = 0;
                for (int frame = 0; frame <= index; ++frame) {
                    double score = 0;
                    for (int d = 0; d < dimension; ++d) {
                        const float k = frame == index ? current_key[kh * dimension + d]
                                                       : keys[(frame * kv_heads + kh) * dimension + d];
                        score += double(q[h * dimension + d]) * k;
                    }
                    probabilities[frame] = std::exp(score / std::sqrt(double(dimension)));
                    total += probabilities[frame];
                }
                for (int d = 0; d < dimension; ++d) {
                    double expected = 0;
                    for (int frame = 0; frame <= index; ++frame) {
                        const float v = frame == index ? current_value[kh * dimension + d]
                                                       : vals[(frame * kv_heads + kh) * dimension + d];
                        expected += probabilities[frame] / total * v;
                    }
                    close(result.values[h * dimension + d], expected);
                }
            }
        }
    }
    arm::ExecutionPlan retained;
    const auto input = retained.input();
    const auto intermediate = retained.add(arm::Activation(arm::ActivationKind::SiLU), {input});
    retained.retain(intermediate);
    retained.set_output(retained.add(arm::Activation(arm::ActivationKind::SiLU), {intermediate}));
    const auto source = values(17);
    retained.run({arm::TensorView(source, {17, 1})}, 4);
    const auto view = retained.view(intermediate);
    for (size_t i = 0; i < source.size(); ++i) {
        close(view.data[i], source[i] / (1 + std::exp(-double(source[i]))));
    }
    arm::ExecutionPlan glu;
    const auto gates = glu.input();
    glu.set_output(glu.add(arm::SwiGLU(), {gates}));
    const auto packed = values(14 * 3);
    const auto &output = glu.run({arm::TensorView(packed, {14, 3})}, 7);
    for (size_t i = 0; i < output.values.size(); ++i) {
        const double gate = packed[output.values.size() + i];
        close(output.values[i], packed[i] * gate / (1 + std::exp(-gate)));
    }
}

void convolution() {
    for (int kernel : {1, 2, 3, 7}) {
        for (int stride : {1, 2}) {
            for (int dilation : {1, 2}) {
                constexpr int channels = 5, output_channels = 7;
                const int padding = dilation * kernel / 2;
                arm::Conv1dConfig config{channels, output_channels, kernel, stride, dilation, padding, padding};
                const auto weight = values(channels * output_channels * kernel), bias = values(output_channels);
                const arm::Conv1d operation(config, weight, bias);
                arm::ExecutionPlan plan;
                const auto input = plan.input();
                plan.set_output(plan.add(operation, {input}));
                for (int frames : {17, 3, 63, 1, 17}) {
                    for (int threads : {1, 4, 2, 7}) {
                        const auto source = values(channels * frames);
                        const auto &result = plan.run({arm::TensorView(source, {channels, frames})}, threads);
                        const int output_frames = result.shape.dimensions[1];
                        for (int out = 0; out < output_channels; ++out) {
                            for (int frame = 0; frame < output_frames; ++frame) {
                                double expected = bias[out];
                                for (int channel = 0; channel < channels; ++channel) {
                                    for (int tap = 0; tap < kernel; ++tap) {
                                        const int position = frame * stride + tap * dilation - padding;
                                        if (position >= 0 && position < frames) {
                                            expected += double(weight[(out * channels + channel) * kernel + tap]) *
                                                        source[channel * frames + position];
                                        }
                                    }
                                }
                                close(result.values[out * output_frames + frame], expected);
                            }
                        }
                    }
                }
            }
        }
    }
}

void branching_and_layout() {
    arm::ExecutionPlan plan;
    const auto input = plan.input();
    const auto first = plan.add(arm::Activation(arm::ActivationKind::SiLU), {input});
    const auto second = plan.add(arm::Activation(arm::ActivationKind::GELU), {input});
    const auto sum = plan.add(arm::ScaleAdd({0.3F, -0.2F}), {first, second});
    const auto constant = plan.add(arm::ConstantChannels(1, 2.0F), {sum});
    const auto joined = plan.add(arm::ConcatChannels(), {sum, constant});
    plan.set_output(joined, arm::Layout::FrameMajor);
    for (int threads : {1, 4, 7, 2, 1}) {
        for (int frames : {1, 17, 3, 63, 17}) {
            const auto source = values(2 * frames);
            const auto &result = plan.run({arm::TensorView(source, {2, frames})}, threads);
            for (int frame = 0; frame < frames; ++frame) {
                for (int channel = 0; channel < 2; ++channel) {
                    const double x = source[channel * frames + frame];
                    const double silu = x / (1 + std::exp(-x));
                    const double gelu = 0.5 * x * (1 + std::erf(x / std::sqrt(2.0)));
                    close(result.values[frame * 3 + channel], silu + (channel == 0 ? 0.3 : -0.2) * gelu);
                }
                close(result.values[frame * 3 + 2], 2.0);
            }
        }
    }
    plan.set_output(input);
    const std::vector<float> source{1, 2, 3, 4};
    engine::test::require(plan.run({arm::TensorView(source, {2, 2})}, 4).values == source,
                          "borrowed input output preserved");
}

void convolution_2d() {
    constexpr int channels = 2, height = 7, width = 9, out_channels = 3;
    arm::Conv2dConfig config{channels, out_channels, 3, 3, 1, 1, 1, 1, 1, 1};
    const auto weight = values(channels * out_channels * 9), bias = values(out_channels),
               source = values(channels * height * width);
    arm::ExecutionPlan plan;
    const auto input = plan.input();
    auto value = plan.add(arm::Conv2d(config, weight, bias), {input});
    value = plan.add(arm::Activation(arm::ActivationKind::LeakyReLU, 0.2F), {value});
    value = plan.add(arm::ClampMin(-0.02F), {value});
    value = plan.add(arm::MaxPool2d(2, 2, 2, 2), {value});
    plan.set_output(value);
    std::vector<double> reference(out_channels * height * width);
    for (int out = 0; out < out_channels; ++out) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                double value = bias[out];
                for (int channel = 0; channel < channels; ++channel) {
                    for (int kh = 0; kh < 3; ++kh) {
                        for (int kw = 0; kw < 3; ++kw) {
                            const int ih = h + kh - 1, iw = w + kw - 1;
                            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                value += double(weight[((out * channels + channel) * 3 + kh) * 3 + kw]) *
                                         source[(channel * height + ih) * width + iw];
                            }
                        }
                    }
                }
                reference[(out * height + h) * width + w] = std::max(value < 0 ? value * 0.2 : value, -0.02);
            }
        }
    }
    for (int threads : {1, 4, 2, 7}) {
        const auto &result = plan.run({arm::TensorView(source, {channels, height, width})}, threads);
        for (int out = 0; out < out_channels; ++out) {
            for (int h = 0; h < 3; ++h) {
                for (int w = 0; w < 4; ++w) {
                    double expected = -1e30;
                    for (int kh = 0; kh < 2; ++kh) {
                        for (int kw = 0; kw < 2; ++kw) {
                            expected = std::max(expected, reference[(out * height + h * 2 + kh) * width + w * 2 + kw]);
                        }
                    }
                    close(result.values[(out * 3 + h) * 4 + w], expected);
                }
            }
        }
    }
}

void validation() {
    auto rejected = [](const auto &operation) {
        bool failed = false;
        try {
            operation();
        } catch (const std::invalid_argument &) {
            failed = true;
        }
        engine::test::require(failed, "invalid library contract rejected");
    };
    rejected([] { arm::Conv2d({1, 2, 3, 3}, {}, {}); });
    rejected([] { arm::Conv1d({1, 2, 3}, {}, {}); });
    rejected([] { arm::Linear(3, 2, {}); });
    rejected([] { arm::QuantizedLinear(31, 2, {}); });
    rejected([] { arm::QuantizedLinear(32, 2, {std::vector<int8_t>(64), {1, -1}}); });
    rejected([] { arm::TensorView(std::vector<float>{1}, {2, 2}); });
    rejected([] {
        arm::ExecutionPlan a, b;
        const auto input = a.input();
        b.add(arm::Activation(arm::ActivationKind::GELU), {input});
    });
    arm::ExecutionPlan plan;
    const auto input = plan.input();
    plan.set_output(plan.add(arm::LayerNorm({1, 1}, {0, 0}, 1e-6F), {input}));
    rejected([&] { plan.run({arm::TensorView(std::vector<float>{1, 2, 3}, {3, 1})}, 4); });
    rejected([&] { plan.run({arm::TensorView(std::vector<float>{1, 2, 3}, {3, 1})}, 4); });
    const auto &result = plan.run({arm::TensorView(std::vector<float>{1, 3}, {2, 1})}, 4);
    close(result.values[0], -1 / std::sqrt(1 + 1e-6));
    close(result.values[1], 1 / std::sqrt(1 + 1e-6));
    rejected([&] { plan.run({arm::TensorView(std::vector<float>{1, 2, 3}, {3, 1})}, 4); });
    rejected([&] { plan.run({arm::TensorView(std::vector<float>{1, 2, 3}, {3, 1})}, 4); });
    close(plan.run({arm::TensorView(std::vector<float>{1, 3}, {2, 1})}, 4).values[0], -1 / std::sqrt(1 + 1e-6));
}

void output_layout_conversion() {
    const std::vector<float> source{1, 2, 3, 4, 5, 6};
    arm::ExecutionPlan plan;
    const auto input = plan.input();
    for (int threads : {1, 4, 7}) {
        plan.set_output(input, arm::Layout::ChannelFirst);
        const auto &result =
            plan.run({arm::TensorView(source, arm::Shape({2, 3}, arm::Layout::FrameMajor))}, threads);
        engine::test::require(result.values == std::vector<float>({1, 3, 5, 2, 4, 6}),
                              "executor converts frame-major output to channel-first");
        plan.set_output(input, arm::Layout::FrameMajor);
        engine::test::require(
            plan.run({arm::TensorView(source, arm::Shape({2, 3}, arm::Layout::FrameMajor))}, threads).values ==
                source,
            "executor preserves frame-major output");
    }
}

void blocks_and_shared_weights() {
    constexpr int channels = 3, expanded = 5, kernel = 3;
    const arm::ConvNeXtWeights weight{values(channels * kernel),
                                      values(channels),
                                      std::vector<float>(channels, 1),
                                      values(channels),
                                      values(expanded * channels),
                                      values(expanded),
                                      values(channels * expanded),
                                      values(channels),
                                      values(channels)};
    arm::ExecutionPlan plan;
    const auto input = plan.input();
    plan.set_output(arm::build_convnext(plan, input, channels, expanded, kernel, 1e-5F, weight));
    for (int frames : {1, 17, 63, 2}) {
        for (int threads : {1, 4, 7}) {
            const auto source = values(channels * frames);
            std::vector<double> normalized(channels * frames), hidden(expanded * frames);
            for (int frame = 0; frame < frames; ++frame) {
                std::vector<double> convolution(channels);
                double mean = 0, variance = 0;
                for (int channel = 0; channel < channels; ++channel) {
                    convolution[channel] = weight.depthwise_bias[channel];
                    for (int tap = 0; tap < kernel; ++tap) {
                        const int position = frame + tap - kernel / 2;
                        if (position >= 0 && position < frames) {
                            convolution[channel] += double(weight.depthwise_weight[channel * kernel + tap]) *
                                                    source[channel * frames + position];
                        }
                    }
                    mean += convolution[channel];
                }
                mean /= channels;
                for (double value : convolution) {
                    variance += (value - mean) * (value - mean);
                }
                const double inverse = 1 / std::sqrt(variance / channels + 1e-5F);
                for (int channel = 0; channel < channels; ++channel) {
                    normalized[channel * frames + frame] =
                        (convolution[channel] - mean) * inverse * weight.norm_weight[channel] +
                        weight.norm_bias[channel];
                }
                for (int out = 0; out < expanded; ++out) {
                    double value = weight.expand_bias[out];
                    for (int channel = 0; channel < channels; ++channel) {
                        value += double(weight.expand_weight[out * channels + channel]) *
                                 normalized[channel * frames + frame];
                    }
                    hidden[out * frames + frame] = 0.5 * value * (1 + std::erf(value / std::sqrt(2.0)));
                }
            }
            const auto &result = plan.run({arm::TensorView(source, {channels, frames})}, threads);
            for (int channel = 0; channel < channels; ++channel) {
                for (int frame = 0; frame < frames; ++frame) {
                    double projected = weight.project_bias[channel];
                    for (int hidden_channel = 0; hidden_channel < expanded; ++hidden_channel) {
                        projected += double(weight.project_weight[channel * expanded + hidden_channel]) *
                                     hidden[hidden_channel * frames + frame];
                    }
                    close(result.values[channel * frames + frame],
                          source[channel * frames + frame] + weight.scale[channel] * projected);
                }
            }
        }
    }

    const arm::Conv1d shared({2, 3, 1}, values(6), values(3));
    auto run = [&] {
        arm::ExecutionPlan local;
        const auto input = local.input();
        local.set_output(local.add(shared, {input}));
        return local.run({arm::TensorView(values(34), {2, 17})}, 4).values;
    };
    auto first = std::async(std::launch::async, run);
    const auto second = run();
    engine::test::require(first.get() == second, "shared immutable operator with independent execution workspaces");
}

void replace_operator() {
    const arm::Conv1d first({2, 3, 1}, values(6), values(3));
    const arm::Conv1d second({2, 7, 3, 1, 1, 1, 1}, values(42), values(7));
    arm::ExecutionPlan plan;
    const auto input = plan.input();
    const auto node = plan.add(first, {input});
    plan.retain(node);
    const arm::Activation activation(arm::ActivationKind::SiLU);
    plan.set_output(plan.add(activation, {node}));
    for (int threads : {1, 4, 1}) for (int frames : {3, 17, 3}) {
        for (const auto * operation : {&first, &second, &first}) {
            plan.replace(node, *operation);
            bool invalidated = false;
            try { (void)plan.view(node); }
            catch (const std::invalid_argument &) { invalidated = true; }
            engine::test::require(invalidated, "replacement invalidates retained results");
            arm::ExecutionPlan reference;
            const auto fresh_input = reference.input();
            reference.set_output(reference.add(activation, {reference.add(*operation, {fresh_input})}));
            auto data = values(2 * frames);
            data[0] += threads * 0.01f;
            const auto & expected = reference.run({arm::TensorView(data, {2, frames})}, threads);
            const auto & actual = plan.run({arm::TensorView(data, {2, frames})}, threads);
            engine::test::require(actual.shape == expected.shape && actual.values == expected.values,
                                  "replacement matches a fresh graph, including downstream shapes");
            engine::test::require(plan.view(node).shape.dimensions[1] == frames, "retained replacement result");
        }
    }
    arm::ExecutionPlan other;
    const auto foreign = other.add(first, {other.input()});
    for (auto invalid : {input, foreign}) {
        bool rejected = false;
        try { plan.replace(invalid, first); }
        catch (const std::invalid_argument &) { rejected = true; }
        engine::test::require(rejected, "replacement rejects input nodes and foreign values");
    }
    plan.replace(node, arm::ScaleAdd({1}));
    bool rejected = false;
    try { (void)plan.run({arm::TensorView(values(6), {2, 3})}, 1); }
    catch (const std::invalid_argument &) { rejected = true; }
    engine::test::require(rejected, "replacement validates operator input arity");
    plan.replace(node, first);
    (void)plan.run({arm::TensorView(values(6), {2, 3})}, 1);
}

} // namespace

int main() {
    try {
        linear();
        quantized_linear();
        decoder_operators();
        embedding_indices();
        convolution();
        branching_and_layout();
        convolution_2d();
        validation();
        output_layout_conversion();
        blocks_and_shared_weights();
        replace_operator();
        std::cout << "NEON library contracts and scalar parity passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
