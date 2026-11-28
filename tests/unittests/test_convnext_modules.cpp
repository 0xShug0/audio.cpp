#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/convnext_modules.h"
#include "test_assert.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace engine;

void check(bool grn) {
    constexpr int batches = 2, frames = 5, channels = 2, hidden = 3;
    const std::vector<float> dw{.1f, .4f, -.1f, -.2f, .5f, .3f}, db{.05f, -.02f};
    const std::vector<float> nw{1.1f, .8f}, nb{.01f, -.03f};
    const std::vector<float> up{.2f, -.1f, .3f, .4f, -.2f, .25f}, ub{.05f, .1f, -.08f};
    const std::vector<float> down{.2f, .3f, -.4f, -.3f, .1f, .2f}, ob{.02f, -.01f};
    const std::vector<float> scale{.7f, .4f}, gamma{.2f, -.1f, .3f}, beta{.01f, -.02f, .05f};
    core::ExecutionContext execution({core::BackendType::Cpu, 0, 8});
    core::PreparedGraph graph(execution, {"test.convnext", 2 * 1024 * 1024, 512});
    auto &ctx = graph.build_context();
    std::vector<std::pair<core::TensorValue, std::vector<float>>> constants;
    const auto tensor = [&](core::TensorShape shape, const std::vector<float> &data) {
        auto value = core::make_tensor(ctx, GGML_TYPE_F32, shape);
        ggml_set_input(value.tensor);
        ggml_set_output(value.tensor);
        constants.emplace_back(value, data);
        return value;
    };
    auto input = core::make_tensor(ctx, GGML_TYPE_F32,
        core::TensorShape::from_dims({batches, frames, channels}));
    ggml_set_input(input.tensor);
    ggml_set_output(input.tensor);
    modules::ConvNeXt1dWeights weights;
    weights.depthwise = {tensor(core::TensorShape::from_dims({channels, 1, 3}), dw),
                         tensor(core::TensorShape::from_dims({channels}), db)};
    weights.norm = {tensor(core::TensorShape::from_dims({channels}), nw),
                    tensor(core::TensorShape::from_dims({channels}), nb)};
    weights.expansion = {tensor(core::TensorShape::from_dims({hidden, channels}), up),
                         tensor(core::TensorShape::from_dims({hidden}), ub)};
    weights.projection = {tensor(core::TensorShape::from_dims({channels, hidden}), down),
                          tensor(core::TensorShape::from_dims({channels}), ob)};
    if (grn) {
        weights.response_gamma = tensor(core::TensorShape::from_dims({hidden}), gamma);
        weights.response_beta = tensor(core::TensorShape::from_dims({hidden}), beta);
    } else {
        weights.gamma = tensor(core::TensorShape::from_dims({channels}), scale);
    }
    const auto output = modules::build_convnext1d(ctx, input, weights);
    ggml_set_output(output.tensor);
    ggml_build_forward_expand(graph.graph(), output.tensor);
    graph.allocate();
    for (const auto &entry : constants) core::write_tensor_f32(entry.first, entry.second);
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<float> x(batches * frames * channels);
        for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i) * .37f + pass);
        core::write_tensor_f32(input, x);
        graph.compute();
        const auto actual = core::read_tensor_f32(output.tensor);
        for (int b = 0; b < batches; ++b) {
            double expanded[frames][hidden]{};
            for (int t = 0; t < frames; ++t) {
                double row[channels]{};
                for (int c = 0; c < channels; ++c) {
                    row[c] = db[c];
                    for (int k = 0; k < 3; ++k) {
                        const int position = t + k - 1;
                        if (position >= 0 && position < frames)
                            row[c] += x[(b * frames + position) * channels + c] * double(dw[c * 3 + k]);
                    }
                }
                const double mean = (row[0] + row[1]) / channels;
                const double variance = ((row[0] - mean) * (row[0] - mean) +
                    (row[1] - mean) * (row[1] - mean)) / channels;
                for (int c = 0; c < channels; ++c)
                    row[c] = (row[c] - mean) / std::sqrt(variance + 1e-6) * nw[c] + nb[c];
                for (int h = 0; h < hidden; ++h) {
                    double value = ub[h];
                    for (int c = 0; c < channels; ++c) value += row[c] * up[h * channels + c];
                    expanded[t][h] = .5 * value * (1 + std::erf(value / std::sqrt(2.)));
                }
            }
            double response[hidden]{}, mean_response = 0;
            if (grn) {
                for (int h = 0; h < hidden; ++h) {
                    for (int t = 0; t < frames; ++t) response[h] += expanded[t][h] * expanded[t][h];
                    response[h] = std::sqrt(response[h]);
                    mean_response += response[h] / hidden;
                }
            }
            for (int t = 0; t < frames; ++t) for (int c = 0; c < channels; ++c) {
                double value = ob[c];
                for (int h = 0; h < hidden; ++h) {
                    double feature = expanded[t][h];
                    if (grn) feature += gamma[h] * feature * response[h] / (mean_response + 1e-6) + beta[h];
                    value += feature * down[c * hidden + h];
                }
                const auto index = (b * frames + t) * channels + c;
                const double expected = x[index] + value * (grn ? 1 : scale[c]);
                test::require_close(actual[index], float(expected), 3e-6f,
                    grn ? "ConvNeXtV2 independent scalar reference" : "ConvNeXt default independent scalar reference");
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 3 || std::string(argv[1]) != "--log") return 2;
    debug::configure_logging({true, argv[2]});
    const auto started = std::chrono::steady_clock::now();
    try {
        check(false);
        check(true);
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        std::cout << "convnext_modules: PASS\n";
        return 0;
    } catch (const std::exception &error) {
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        std::cerr << error.what() << '\n';
        return 1;
    }
}
