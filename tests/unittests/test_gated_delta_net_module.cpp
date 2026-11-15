#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/gated_delta_net_module.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <array>
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

void test_block(core::ExecutionContext & execution, bool packed, bool zero_qk = false, bool rounded = false) {
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
    if (zero_qk) std::fill(projection.begin(), projection.begin() + 2 * kd * hidden, 0.0F);
    for (size_t i = 0; i < conv_weight.size(); ++i) conv_weight[i] = .4F * std::cos(float(i) * .07F);
    for (size_t i = 0; i < norm.size(); ++i) norm[i] = .8F + .01F * float(i);
    for (size_t i = 0; i < output_weight.size(); ++i) output_weight[i] = .2F * std::sin(float(i) * .31F);
    for (size_t i = 0; i < initial_state.size(); ++i) initial_state[i] = .01F * std::sin(float(i) * .21F);
    for (size_t i = 0; i < tail.size(); ++i) tail[i] = packed && !zero_qk ? .1F * std::cos(float(i) * .11F) : 0.0F;
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
    config.round_bf16_activations = rounded;
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
    // Zonos3's pre-refactor packed projection: a flattened F32 linear operation.
    // Keep this outside the new module so layout/precision regressions are visible.
    ggml_tensor * legacy_projection = nullptr;
    ggml_tensor * legacy_first = nullptr, * legacy_rest = nullptr;
    if (packed) {
        legacy_projection = ggml_mul_mat(ctx.ggml, weights.in_proj.weight.tensor,
            ggml_reshape_2d(ctx.ggml, x.tensor, hidden, rows));
        ggml_mul_mat_set_prec(legacy_projection, GGML_PREC_F32);
        ggml_set_output(legacy_projection);
        ggml_build_forward_expand(graph.graph(), legacy_projection);
        const auto old_project = [&](int begin, int count) {
            auto segment = modules::SliceModule({1, begin, count}).build(ctx, x);
            auto * result = ggml_mul_mat(ctx.ggml, weights.in_proj.weight.tensor,
                ggml_reshape_2d(ctx.ggml, ggml_cont(ctx.ggml, segment.tensor), hidden, batch * count));
            ggml_mul_mat_set_prec(result, GGML_PREC_F32);
            ggml_set_output(result);
            ggml_build_forward_expand(graph.graph(), result);
            return result;
        };
        legacy_first = old_project(0, 1);
        legacy_rest = old_project(1, steps - 1);
    }

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
    std::vector<float> projected(rows * mixed_width);
    for (int r = 0; r < rows; ++r) for (int j = 0; j < mixed_width; ++j)
        for (int i = 0; i < hidden; ++i) projected[r * mixed_width + j] += input[r * hidden + i] * projection[j * hidden + i];
    if (legacy_projection) {
        const auto legacy = core::read_tensor_f32(legacy_projection);
        float max_difference = 0;
        for (size_t i = 0; i < projected.size(); ++i)
            max_difference = (std::max)(max_difference, std::abs(projected[i] - legacy[i]));
        std::cout << "Legacy packed projection vs scalar max_abs=" << max_difference << '\n';
        // CUDA's existing projection can use TF32; this is not a new GDN operation.
        // Compare the refactored tail and independent scalar recurrence against
        // the old projection's actual output, keeping the existing tolerance.
        if (execution.backend_type() != core::BackendType::Cuda)
            compare(legacy, projected, "legacy projection scalar");
        projected = legacy;
    }
    const auto reference = [&](std::vector<float> projected) {
        std::vector<float> expected(rows * hidden), expected_tail(tail), recurrent(initial_state);
        const auto round = [rounded](float value) { return rounded ? ggml_bf16_to_fp32(ggml_fp32_to_bf16(value)) : value; };
        for (auto & value : projected) value = round(value);
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
                convolved[c] = round(silu(round(sum)));
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
                for (int i = 0; i < dim; ++i) { q[i] = round(q[i] * qscale); k[i] = round(k[i] * kscale); }
                const int base = (b * steps + t) * mixed_width;
                const float decay = std::exp(-std::log1p(std::exp(projected[base + channels + vd + h] + dt[h])) * std::exp(a_log[h]));
                const float beta = round(1 / (1 + std::exp(-projected[base + channels + vd + heads + h])));
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
                    attended[j] = round(attended[j] / std::sqrt(float(dim)));
                    square_sum += attended[j] * attended[j];
                }
                const float norm_scale = 1 / std::sqrt(square_sum / dim + config.norm_epsilon);
                for (int j = 0; j < dim; ++j)
                    gated[h * dim + j] = round(round(attended[j] * norm_scale * norm[j]) * round(silu(projected[base + channels + h * dim + j])));
            }
            for (int j = 0; j < hidden; ++j) for (int i = 0; i < vd; ++i)
                expected[(b * steps + t) * hidden + j] += output_weight[j * vd + i] * gated[i];
        }
        for (int b = 0; b < batch; ++b) for (int c = 0; c < channels; ++c) for (int t = 0; t < history; ++t)
            expected_tail[(b * channels + c) * history + t] = projected[(b * steps + t) * mixed_width + c];
        for (auto & value : expected) value = round(value);
        for (auto & value : recurrent) value = round(value);
        return std::array<std::vector<float>, 3>{expected, recurrent, expected_tail};
    };
    const auto full = reference(projected);
    compare(core::read_tensor_f32(output.output.tensor), full[0], "scalar output");
    compare(core::read_tensor_f32(output.next_recurrent_state.tensor), full[1], "scalar state");
    compare(core::read_tensor_f32(output.next_conv_tail.tensor), full[2], "scalar tail");
    // Rounded state is quantized at each segment boundary, so a split has a different
    // numerical contract. Full-block scalar checks above still cover BF16 boundaries.
    if (!rounded) {
        auto split_projection = projected;
        if (legacy_first) {
            const auto first_rows = core::read_tensor_f32(legacy_first);
            const auto rest_rows = core::read_tensor_f32(legacy_rest);
            // The old CUDA projection can select different TF32 kernels by shape.
            // Reconstruct the split sequence from those old projections, then use
            // the same independent scalar convolution/recurrence with no relaxed tolerance.
            for (int b = 0; b < batch; ++b) {
                std::copy_n(first_rows.data() + b * mixed_width, mixed_width,
                    split_projection.data() + b * steps * mixed_width);
                std::copy_n(rest_rows.data() + b * (steps - 1) * mixed_width,
                    (steps - 1) * mixed_width, split_projection.data() + (b * steps + 1) * mixed_width);
            }
        }
        const auto split = reference(split_projection);
        compare(core::read_tensor_f32(joined.tensor), split[0], "prefill/decode output");
        compare(core::read_tensor_f32(rest.next_recurrent_state.tensor), split[1], "prefill/decode state");
        compare(core::read_tensor_f32(rest.next_conv_tail.tensor), split[2], "prefill/decode tail");
    }
    // Same prepared graph, changed inputs, then restored inputs/state: no graph rebuild.
    const auto original_output = core::read_tensor_f32(output.output.tensor);
    const auto original_state = core::read_tensor_f32(output.next_recurrent_state.tensor);
    const auto original_tail = core::read_tensor_f32(output.next_conv_tail.tensor);
    std::vector<float> changed(input.size(), 0.0F);
    ggml_backend_tensor_set(x.tensor, changed.data(), 0, changed.size() * sizeof(float));
    graph.compute();
    ggml_backend_tensor_set(x.tensor, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(state.tensor, initial_state.data(), 0, initial_state.size() * sizeof(float));
    graph.compute();
    compare(core::read_tensor_f32(output.output.tensor), original_output, "reused graph output");
    compare(core::read_tensor_f32(output.next_recurrent_state.tensor), original_state, "reused graph state");
    compare(core::read_tensor_f32(output.next_conv_tail.tensor), original_tail, "reused graph tail");

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
        test_block(execution, false, true);
        test_block(execution, true, true);
        test_block(execution, false, false, true);
        engine::debug::timing_log_scalar("session.wall_ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        std::cout << "Gated DeltaNet module scalar and state-continuity tests passed\n";
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
