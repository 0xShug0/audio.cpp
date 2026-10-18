#include "engine/framework/neon/execution.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/core/backend.h"
#include "kernels.h"
#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace cpu = engine::neon::detail;

namespace {
void close(float actual, double expected, const char *label) {
    engine::test::require(std::isfinite(actual) && std::isfinite(expected), label);
    engine::test::require(std::abs(actual - expected) <= 3e-5 * (1 + std::abs(expected)), label);
}

void check_conv(int C, int O, int K, int T, int threads, bool rows, bool bias) {
    std::mt19937 rng(C * 71 + O * 13 + K);
    std::uniform_real_distribution<float> uniform(-0.4F, 0.4F);
    std::vector<float> x(C * T), w(O * C * K), b(O), y(O * T + 7, 123.0F);
    for (auto *v : {&x, &w, &b}) {
        for (auto &value : *v) {
            value = uniform(rng);
        }
    }
    const auto packed = cpu::pack_conv_weights(w.data(), bias ? b.data() : nullptr, C, O, K);
    cpu::Workspace workspace;
    workspace.prepare(T + 17, C, K);
    const auto *allocated = workspace.packed_input.data();
    workspace.prepare(T, C, K);
    engine::test::require(allocated == workspace.packed_input.data(), "workspace reuse");
    cpu::WorkerTeam(threads).run([&](cpu::WorkerContext &worker) {
        cpu::convolve_1d(packed, x.data(), y.data(), rows ? O : 0, T, workspace, worker);
    });
    for (int o = 0; o < O; ++o) {
        for (int t = 0; t < T; ++t) {
            double value = bias ? b[o] : 0.0;
            for (int c = 0; c < C; ++c) {
                for (int k = 0; k < K; ++k) {
                    const int pos = t + k - K / 2;
                    if (pos >= 0 && pos < T) {
                        value += double(w[(o * C + c) * K + k]) * x[c * T + pos];
                    }
                }
            }
            close(y[rows ? t * O + o : o * T + t], value, "conv1d double reference");
        }
    }
    for (size_t i = O * T; i < y.size(); ++i) {
        engine::test::require(y[i] == 123.0F, "conv output boundary");
    }
}

void check_norm_depthwise(int C, int T, int K, int threads) {
    std::vector<float> x(C * T), w(C * K), b(C), g(C), y(C * T), norm(C * T);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(float(i) * 0.37F);
    }
    for (size_t i = 0; i < w.size(); ++i) {
        w[i] = std::cos(float(i) * 0.13F) * 0.1F;
    }
    for (int c = 0; c < C; ++c) {
        b[c] = c * 0.01F;
        g[c] = 1.0F + c * 0.02F;
    }
    cpu::Workspace workspace;
    workspace.prepare(T, C, K);
    cpu::WorkerTeam(threads).run([&](cpu::WorkerContext &worker) {
        cpu::convolve_depthwise_1d(x.data(), w.data(), b.data(), y.data(), C, T, K, worker);
        worker.synchronize();
        cpu::normalize_channels(y.data(), g.data(), b.data(), norm.data(), C, T, 1e-5F, workspace, worker);
    });
    for (int t = 0; t < T; ++t) {
        std::vector<double> reference(C);
        double mean = 0, variance = 0;
        for (int c = 0; c < C; ++c) {
            reference[c] = b[c];
            for (int k = 0; k < K; ++k) {
                const int pos = t + k - K / 2;
                if (pos >= 0 && pos < T) {
                    reference[c] += double(w[c * K + k]) * x[c * T + pos];
                }
            }
            close(y[c * T + t], reference[c], "depthwise double reference");
            mean += reference[c];
        }
        mean /= C;
        for (const auto value : reference) {
            variance += (value - mean) * (value - mean);
        }
        const double inv = 1 / std::sqrt(variance / C + 1e-5F);
        for (int c = 0; c < C; ++c) {
            close(norm[c * T + t], (reference[c] - mean) * inv * g[c] + b[c], "norm double reference");
        }
    }
}

void check_selection() {
    using engine::core::BackendType;
    const auto parse = [](const std::unordered_map<std::string, std::string> & options) {
        return engine::runtime::parse_exec_plan(options, "model.exec_plan",
            {{"decoder", "ggml",
              {{"ggml", {BackendType::Cpu, BackendType::Metal}, {}, {}},
               {"neon", {BackendType::Cpu}, {}, {}}}}});
    };
    engine::test::require(parse({}).at("decoder").impl == "ggml", "default graph unchanged");
    for (const auto * value : {"{}", R"({"decoder":{}})", R"({"decoder":{"impl":"default"}})"}) {
        engine::test::require(parse({{"model.exec_plan", value}}).at("decoder").impl == "ggml",
                              "omitted implementation retains graph");
    }
    const auto neon = parse({{"model.exec_plan", R"({"decoder":{"impl":"neon","backend":"cpu"}})"}});
    engine::test::require(neon.at("decoder").impl == "neon" &&
        neon.at("decoder").backend == BackendType::Cpu, "native execution plan");
    for (const auto * value : {"", "[]", R"({"encoder":{}})",
             R"({"decoder":{"impl":"auto"}})", R"({"decoder":{"impl":"neon","backend":"metal"}})",
             R"({"decoder":{"impl":"neon","artifacts":{"weights":"unused"}}})",
             R"({"decoder":{"impl":"neon","params":{"threads":1}}})",
             R"({"decoder":{"unknown":true}})"}) {
        bool rejected = false;
        try {
            (void)parse({{"model.exec_plan", value}});
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        engine::test::require(rejected, "invalid execution plan rejected");
    }
}
void check_conv2d(int H, int W, int KH, int KW, int stride, int dilation, int padding, int threads) {
    constexpr int C = 5, O = 7;
    std::vector<float> x(C * H * W), weights(O * C * KH * KW), bias(O);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(float(i) * 0.31F);
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = std::cos(float(i) * 0.23F) * 0.05F;
    }
    for (int i = 0; i < O; ++i) {
        bias[i] = i * 0.02F;
    }
    engine::neon::Conv2dConfig config;
    config.input_channels = C;
    config.output_channels = O;
    config.kernel_height = KH;
    config.kernel_width = KW;
    config.stride_height = config.stride_width = stride;
    config.dilation_height = config.dilation_width = dilation;
    config.padding_height = config.padding_width = padding;
    engine::neon::ExecutionPlan plan;
    const auto input = plan.input();
    plan.set_output(plan.add(engine::neon::Conv2d(config, weights, bias), {input}));
    const auto &result = plan.run({engine::neon::TensorView(x, {C, H, W})}, threads);
    const int OH = result.shape.dimensions[1], OW = result.shape.dimensions[2];
    const auto &output = result.values;
    for (int o = 0; o < O; ++o) {
        for (int h = 0; h < OH; ++h) {
            for (int w = 0; w < OW; ++w) {
                double value = bias[o];
                for (int c = 0; c < C; ++c) {
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            const int ih = h * stride - padding + kh * dilation,
                                      iw = w * stride - padding + kw * dilation;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                value += double(x[(c * H + ih) * W + iw]) * weights[((o * C + c) * KH + kh) * KW + kw];
                            }
                        }
                    }
                }
                close(output[(o * OH + h) * OW + w], value, "conv2d double reference");
            }
        }
    }
}

void check_pool_activation(int threads) {
    constexpr int C = 5, H = 7, W = 9, OH = 3, OW = 4;
    std::vector<float> x(C * H * W), out(C * OH * OW);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(float(i) * 0.31F);
    }
    const auto original = x;
    cpu::WorkerTeam(threads).run(
        [&](cpu::WorkerContext &worker) { cpu::pool_max_2d(x.data(), out.data(), C, H, W, 2, 2, 2, 2, worker); });
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < OH; ++h) {
            for (int w = 0; w < OW; ++w) {
                float value = -std::numeric_limits<float>::infinity();
                for (int dh = 0; dh < 2; ++dh) {
                    for (int dw = 0; dw < 2; ++dw) {
                        value = std::max(value, x[(c * H + h * 2 + dh) * W + w * 2 + dw]);
                    }
                }
                engine::test::require(out[(c * OH + h) * OW + w] == value, "pool reference");
            }
        }
    }
    cpu::apply_leaky_relu_clamp(x.data(), 0, x.size(), 0.2F, -0.4F);
    for (size_t i = 0; i < x.size(); ++i) {
        close(x[i], std::max(original[i] < 0 ? original[i] * 0.2F : original[i], -0.4F), "activation reference");
    }
}
} // namespace

int main() {
    try {
        check_selection();
        for (const int threads : {1, 2, 4, 7}) {
            for (const int stride : {1, 2}) {
                check_conv2d(7, 9, 3, 3, stride, 1, 1, threads);
                check_conv2d(7, 9, 3, 1, stride, 2, 1, threads);
                check_conv2d(7, 9, 1, 3, stride, 1, 0, threads);
                check_conv2d(7, 9, 2, 2, stride, 1, 0, threads);
            }
            check_conv2d(3, 1, 3, 1, 1, 1, 0, threads);
            check_pool_activation(threads);
        }
        for (const int threads : {1, 2, 4, 7}) {
            for (const int T : {1, 2, 15, 16, 17, 63}) {
                for (const int K : {1, 3, 7, 33}) {
                    for (const bool rows : {false, true}) {
                        check_conv(5, 7, K, T, threads, rows, true);
                        check_conv(8, 12, K, T, threads, rows, false);
                    }
                }
                check_norm_depthwise(7, T, 7, threads);
            }
        }
        std::vector<float> values(259);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = (float(i) - 129.0F) * 0.08F;
        }
        const auto reference = values;
        cpu::apply_gelu(values.data(), 0, values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            close(values[i], 0.5 * reference[i] * (1 + std::erf(reference[i] / std::sqrt(2.0))), "GELU reference");
        }
        values = reference;
        cpu::apply_silu(values.data(), 0, values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            close(values[i], reference[i] / (1 + std::exp(-double(reference[i]))), "SiLU reference");
        }
        bool rejected = false;
        try {
            cpu::pack_conv_weights(values.data(), nullptr, 5, 7, 0);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        engine::test::require(rejected, "invalid kernels rejected");
        std::cout << "NEON blocks parity passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
