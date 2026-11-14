#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/gated_delta_net_module.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
namespace core = engine::core;
namespace modules = engine::modules;

void compare(const std::vector<float> & actual, const std::vector<float> & expected, const char * label) {
    if (actual.size() != expected.size()) throw std::runtime_error(std::string(label) + " size mismatch");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 2e-5F * (1 + std::abs(expected[i]))) {
            throw std::runtime_error(std::string(label) + " mismatch at " + std::to_string(i) +
                ": " + std::to_string(actual[i]) + " vs " + std::to_string(expected[i]));
        }
    }
}

void test_block(core::ExecutionContext & execution, bool packed) {
    constexpr int hidden = 8, steps = 3, kernel = 4;
    const int heads = packed ? 2 : 4, dim = packed ? 64 : 32;
    const int batch = packed ? 2 : 1, key_heads = 2;
    const int kd = key_heads * dim, vd = heads * dim, channels = 2 * kd + vd;
    const int mixed_width = channels + vd + 2 * heads;
    const int rows = batch * steps, history = kernel - 1;
    core::BackendWeightStore store(execution.backend(), execution.backend_type(), "test.gdn.weights", 1024 * 1024);
    std::vector<float> input(rows * hidden), projection(mixed_width * hidden), conv_weight(channels * kernel),
        norm(dim), a_log(heads), dt(heads), output_weight(hidden * vd), initial_state(batch * vd * dim),
        tail(batch * channels * history);
    for (size_t i = 0; i < input.size(); ++i) input[i] = .3F * std::cos(float(i) * .27F);
    for (size_t i = 0; i < projection.size(); ++i) projection[i] = .15F * std::sin(float(i) * .17F);
    for (size_t i = 0; i < conv_weight.size(); ++i) conv_weight[i] = .4F * std::cos(float(i) * .07F);
    for (size_t i = 0; i < norm.size(); ++i) norm[i] = .8F + .01F * float(i);
    for (size_t i = 0; i < output_weight.size(); ++i) output_weight[i] = .2F * std::sin(float(i) * .31F);
    for (size_t i = 0; i < initial_state.size(); ++i) initial_state[i] = .01F * std::sin(float(i) * .21F);
    for (size_t i = 0; i < tail.size(); ++i) tail[i] = packed ? .1F * std::cos(float(i) * .11F) : 0.0F;
    for (int h = 0; h < heads; ++h) { a_log[h] = -.7F + float(h) * .1F; dt[h] = .2F - float(h) * .3F; }
    const auto x = store.make_f32(core::TensorShape::from_dims({batch, steps, hidden}), input);
    const auto state = store.make_f32(core::TensorShape::from_dims({batch, 1, vd * dim}), initial_state);
    const auto conv_tail = store.make_f32(core::TensorShape::from_dims({batch, channels, history}), tail);
    modules::GatedDeltaNetWeights weights;
    const auto project_weights = [&](int first, int width) {
        return modules::LinearWeights{store.make_f32(core::TensorShape::from_dims({width, hidden}),
            std::vector<float>(projection.begin() + first * hidden, projection.begin() + (first + width) * hidden)), std::nullopt};
    };
    if (packed) weights.in_proj = project_weights(0, mixed_width);
    else {
        weights.in_proj_qkv = project_weights(0, channels);
        weights.in_proj_z = project_weights(channels, vd);
        weights.in_proj_a = project_weights(channels + vd, heads);
        weights.in_proj_b = project_weights(channels + vd + heads, heads);
    }
    weights.conv_weight = store.make_f32(core::TensorShape::from_dims({channels, kernel}), conv_weight);
    weights.norm.weight = store.make_f32(core::TensorShape::from_dims({dim}), norm);
    weights.a_log = store.make_f32(core::TensorShape::from_dims({heads}), a_log);
    weights.dt_bias = store.make_f32(core::TensorShape::from_dims({heads}), dt);
    weights.out_proj = {store.make_f32(core::TensorShape::from_dims({hidden, vd}), output_weight), std::nullopt};
    store.upload();
    modules::GatedDeltaNetConfig config;
    config.hidden_size = hidden;
    config.key_heads = key_heads;
    config.value_heads = heads;
    config.key_head_dim = config.value_head_dim = dim;
    config.conv_kernel = kernel;
    config.projection_precision = GGML_PREC_F32;
    config.qk_norm = packed ? modules::GatedDeltaNetQkNorm::SumSquares : modules::GatedDeltaNetQkNorm::L2;
    config.norm_epsilon = 1e-5F;
    // Amplify the distinction between clamp and additive-epsilon normalization.
    config.qk_epsilon = .04F;
    const modules::GatedDeltaNetModule module(config);
    core::PreparedGraph graph(execution, {"test.gdn", 8 * 1024 * 1024, 2048});
    auto & ctx = graph.build_context();
    const auto history_input = packed ? conv_tail : core::TensorValue{};
    const auto output = module.build(ctx, x, state, weights, history_input);
    const auto first = module.build(ctx, modules::SliceModule({1, 0, 1}).build(ctx, x), state, weights, history_input);
    const auto rest = module.build(ctx, modules::SliceModule({1, 1, steps - 1}).build(ctx, x), first.next_recurrent_state,
        weights, first.next_conv_tail);
    const auto joined = modules::ConcatModule({1}).build(ctx, first.output, rest.output);
    for (auto * tensor : {output.output.tensor, output.next_conv_tail.tensor, output.next_recurrent_state.tensor,
            joined.tensor, rest.next_conv_tail.tensor, rest.next_recurrent_state.tensor}) {
        ggml_set_output(tensor);
        ggml_build_forward_expand(graph.graph(), tensor);
    }
    graph.allocate();
    graph.compute();
    // Independent scalar convolution, delta update and gated RMS normalization.
    std::vector<float> projected(rows * mixed_width), expected(rows * hidden), expected_tail(tail), recurrent(initial_state);
    for (int r = 0; r < rows; ++r) for (int j = 0; j < mixed_width; ++j)
        for (int i = 0; i < hidden; ++i) projected[r * mixed_width + j] += input[r * hidden + i] * projection[j * hidden + i];
    const auto silu = [](float value) { return value / (1.0F + std::exp(-value)); };
    for (int b = 0; b < batch; ++b) for (int t = 0; t < steps; ++t) {
        std::vector<float> convolved(channels), gated(vd);
        for (int c = 0; c < channels; ++c) {
            float sum = 0;
            for (int k = 0; k < kernel; ++k) {
                const int index = t + k - history;
                const float sample = index < 0 ? tail[(b * channels + c) * history + index + history]
                    : projected[(b * steps + index) * mixed_width + c];
                sum += sample * conv_weight[c * kernel + k];
            }
            convolved[c] = silu(sum);
        }
        for (int h = 0; h < heads; ++h) {
            const int kh = h / (heads / key_heads);
            std::vector<float> q(dim), k(dim), values(dim), attended(dim);
            float qsum = 0, ksum = 0;
            for (int i = 0; i < dim; ++i) {
                q[i] = convolved[kh * dim + i]; k[i] = convolved[kd + kh * dim + i];
                qsum += q[i] * q[i]; ksum += k[i] * k[i];
                values[i] = convolved[2 * kd + h * dim + i];
            }
            const float qscale = packed ? 1 / std::sqrt(qsum + config.qk_epsilon) : 1 / (std::max)(std::sqrt(qsum), config.qk_epsilon);
            const float kscale = packed ? 1 / std::sqrt(ksum + config.qk_epsilon) : 1 / (std::max)(std::sqrt(ksum), config.qk_epsilon);
            for (int i = 0; i < dim; ++i) { q[i] *= qscale; k[i] *= kscale; }
            const int base = (b * steps + t) * mixed_width;
            const float decay = std::exp(-std::log1p(std::exp(projected[base + channels + vd + h] + dt[h])) * std::exp(a_log[h]));
            const float beta = 1 / (1 + std::exp(-projected[base + channels + vd + heads + h]));
            auto * state_head = recurrent.data() + (b * heads + h) * dim * dim;
            for (int i = 0; i < dim * dim; ++i) state_head[i] *= decay;
            float square_sum = 0;
            for (int j = 0; j < dim; ++j) {
                float prediction = 0;
                for (int i = 0; i < dim; ++i) prediction += state_head[j * dim + i] * k[i];
                const float delta = (values[j] - prediction) * beta;
                for (int i = 0; i < dim; ++i) {
                    state_head[j * dim + i] += k[i] * delta;
                    attended[j] += state_head[j * dim + i] * q[i];
                }
                attended[j] /= std::sqrt(float(dim));
                square_sum += attended[j] * attended[j];
            }
            const float norm_scale = 1 / std::sqrt(square_sum / dim + config.norm_epsilon);
            for (int j = 0; j < dim; ++j)
                gated[h * dim + j] = attended[j] * norm_scale * norm[j] * silu(projected[base + channels + h * dim + j]);
        }
        for (int j = 0; j < hidden; ++j) for (int i = 0; i < vd; ++i)
            expected[(b * steps + t) * hidden + j] += output_weight[j * vd + i] * gated[i];
    }
    for (int b = 0; b < batch; ++b) for (int c = 0; c < channels; ++c) for (int t = 0; t < history; ++t)
        expected_tail[(b * channels + c) * history + t] = projected[(b * steps + t) * mixed_width + c];
    compare(core::read_tensor_f32(output.output.tensor), expected, "scalar output");
    compare(core::read_tensor_f32(output.next_recurrent_state.tensor), recurrent, "scalar state");
    compare(core::read_tensor_f32(output.next_conv_tail.tensor), expected_tail, "scalar tail");
    compare(core::read_tensor_f32(joined.tensor), expected, "prefill/decode output");
    compare(core::read_tensor_f32(rest.next_recurrent_state.tensor), recurrent, "prefill/decode state");
    compare(core::read_tensor_f32(rest.next_conv_tail.tensor), expected_tail, "prefill/decode tail");
}
}  // namespace

int main(int argc, char ** argv) {
    try {
        engine::debug::configure_logging({true, std::nullopt});
        const auto start = std::chrono::steady_clock::now();
        core::BackendConfig config; config.threads = 8;
        if (argc > 1 && std::string(argv[1]) == "cuda") config.type = core::BackendType::Cuda;
        if (argc > 1 && std::string(argv[1]) == "vulkan") config.type = core::BackendType::Vulkan;
        core::ExecutionContext execution(config);
        test_block(execution, false);
        test_block(execution, true);
        engine::debug::timing_log_scalar("session.wall_ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        std::cout << "Gated DeltaNet module scalar and state-continuity tests passed\n";
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
