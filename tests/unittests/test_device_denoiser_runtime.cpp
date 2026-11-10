#include "engine/framework/sampling/device_denoiser_runtime.h"
#include "engine/framework/core/graph_workspace.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

using namespace engine;

namespace {

enum class DenoiserEpilogue { BiasGelu, BiasSilu, BiasGeluQuick, BiasGeluErf, LayerAffine, RmsAffine, GatedResidual, ReshapedAffine, FanoutResidual, ReshapedGatedResidual };

void check_private_view_preparation(core::ExecutionContext & execution) {
    const auto device = ggml_backend_get_device(execution.backend());
    const auto prepare = device ? reinterpret_cast<ggml_backend_set_graph_fusions_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device),
            "ggml_backend_set_graph_fusions")) : nullptr;
    if (!prepare) return;
    core::PreparedGraph graph(execution, {"test.private_view_preparation", 64 * 1024, 32});
    auto * ctx = graph.build_context().ggml;
    auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 17, 3, 2);
    auto * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 17);
    auto * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 17);
    for (auto * tensor : {input, bias, scale}) ggml_set_input(tensor);
    auto * biased = ggml_add(ctx, input, bias);
    auto * view = ggml_reshape_2d(ctx, biased, 17, 6);
    auto * gate = ggml_scale(ctx, scale, 0.5F);
    auto * gated = ggml_mul(ctx, view, gate);
    auto * output = ggml_add(ctx, gated, ggml_reshape_2d(ctx, input, 17, 6));
    ggml_set_output(output);
    ggml_build_forward_expand(graph.graph(), output);
    const auto index = [&](const ggml_tensor * tensor) {
        for (int i = 0; i < ggml_graph_n_nodes(graph.graph()); ++i)
            if (ggml_graph_node(graph.graph(), i) == tensor) return i;
        throw std::runtime_error("Preparation removed a graph node");
    };
    if (!(index(biased) < index(view) && index(view) < index(gate) && index(gate) < index(gated)))
        throw std::runtime_error("Private-view fixture must separate the epilogue with parameter work");
    const int nodes = ggml_graph_n_nodes(graph.graph());
    const auto supported = prepare(execution.backend(), graph.graph(), GGML_BACKEND_GRAPH_FUSION_GATED_RESIDUAL);
    const bool adjacent = index(view) == index(biased) + 1 && index(gated) == index(view) + 1 &&
        index(output) == index(gated) + 1 && index(gate) < index(biased);
    prepare(execution.backend(), graph.graph(), 0);
    if (supported && (!adjacent || ggml_graph_n_nodes(graph.graph()) != nodes))
        throw std::runtime_error("Preparation must move private metadata without removing nodes");
}

void check_denoiser_fusions(core::ExecutionContext & execution, int64_t width,
    DenoiserEpilogue epilogue, bool per_batch, bool half = false, bool rounded = false, bool bf16_output = false, bool promote_gate_input = true) {
    const bool reshaped = epilogue == DenoiserEpilogue::ReshapedAffine;
    const bool affine = reshaped || epilogue == DenoiserEpilogue::LayerAffine || epilogue == DenoiserEpilogue::RmsAffine;
    const bool fanout = epilogue == DenoiserEpilogue::FanoutResidual;
    const bool reshaped_gate = epilogue == DenoiserEpilogue::ReshapedGatedResidual;
    const bool gated = fanout || reshaped_gate || epilogue == DenoiserEpilogue::GatedResidual;
    struct Inputs { core::TensorValue latent, scale, bias; };
    Inputs inputs[2];
    std::unique_ptr<modules::DeviceDenoiserRuntime> runtimes[2];
    for (int policy = 0; policy < 2; ++policy) {
        runtimes[policy] = std::make_unique<modules::DeviceDenoiserRuntime>(execution,
            core::PreparedGraphConfig{"test.denoiser_fusions", 1024 * 1024, 128},
            [&, policy](core::ModuleBuildContext & ctx) -> modules::DeviceDenoiserGraph {
                auto & in = inputs[policy];
                in.latent = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, reshaped ? 6 : 3, width}));
                if (affine || gated) {
                    in.scale = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({reshaped || per_batch ? 2 : 1, reshaped ? 3 : 1, width}));
                    ggml_set_input(in.scale.tensor);
                    ggml_set_output(in.scale.tensor);
                }
                in.bias = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({per_batch ? 2 : 1, 1, width}));
                // These constants are written once and reused across submissions.
                // INPUT does not extend their allocator lifetime; OUTPUT does.
                ggml_set_output(in.bias.tensor);
                for (auto value : {in.latent, in.bias}) ggml_set_input(value.tensor);
                auto * prediction = in.latent.tensor;
                if (affine) {
                    prediction = epilogue == DenoiserEpilogue::RmsAffine
                        ? ggml_rms_norm(ctx.ggml, prediction, 0.003F)
                        : ggml_norm(ctx.ggml, prediction, 0.003F);
                    if (reshaped) prediction = ggml_reshape_3d(ctx.ggml, prediction, width, 3, 4);
                    const float factor = !half && !rounded && !per_batch ? 0.5F : 1.0F;
                    prediction = ggml_mul(ctx.ggml, prediction, ggml_scale_bias(ctx.ggml, in.scale.tensor, factor, 1.0F));
                    if (reshaped) prediction = ggml_reshape_3d(ctx.ggml, prediction, width, 6, 2);
                }
                auto * bias = ggml_scale(ctx.ggml, in.bias.tensor, 0.75F);
                if (half && !affine) {
                    prediction = ggml_cast(ctx.ggml, prediction, GGML_TYPE_F16);
                    bias = ggml_cast(ctx.ggml, bias, GGML_TYPE_F16);
                }
                prediction = ggml_add(ctx.ggml, prediction, bias);
                if (reshaped_gate) prediction = ggml_reshape_2d(ctx.ggml, prediction, width, 6);
                if (gated) {
                    if (half && promote_gate_input) prediction = ggml_cast(ctx.ggml, prediction, GGML_TYPE_F32);
                    if (reshaped_gate) prediction = ggml_reshape_3d(ctx.ggml, prediction, width, 3, 2);
                    auto * residual = fanout ? prediction : in.latent.tensor;
                    auto * gate = fanout ? in.scale.tensor : ggml_scale(ctx.ggml, in.scale.tensor, 0.5F);
                    prediction = ggml_mul(ctx.ggml, prediction, gate);
                    prediction = promote_gate_input ? ggml_add(ctx.ggml, prediction, residual)
                        : ggml_add_cast(ctx.ggml, prediction, residual, GGML_TYPE_F32);
                } else if (!affine) {
                    switch (epilogue) {
                        case DenoiserEpilogue::BiasSilu: prediction = ggml_silu(ctx.ggml, prediction); break;
                        case DenoiserEpilogue::BiasGeluQuick: prediction = ggml_gelu_quick(ctx.ggml, prediction); break;
                        case DenoiserEpilogue::BiasGeluErf: prediction = ggml_gelu_erf(ctx.ggml, prediction); break;
                        default: prediction = ggml_gelu(ctx.ggml, prediction); break;
                    }
                    if (!half && !bf16_output) prediction = ggml_round_bf16(ctx.ggml, prediction);
                }
                if (affine && half) {
                    prediction = ggml_reshape_2d(ctx.ggml, prediction, width, 6);
                    prediction = ggml_cast(ctx.ggml, prediction, GGML_TYPE_F16);
                }
                if (rounded) {
                    prediction = ggml_reshape_2d(ctx.ggml, prediction, width, 6);
                    prediction = ggml_round_bf16(ctx.ggml, prediction);
                }
                if (bf16_output) {
                    // A keeps the existing fused rounded-F32 result plus cast;
                    // B may write BF16 directly. Both must produce identical bits.
                    if (policy == 0) ggml_set_output(prediction);
                    prediction = ggml_reshape_2d(ctx.ggml, prediction, width, 6);
                    prediction = ggml_cast(ctx.ggml, prediction, GGML_TYPE_BF16);
                }
                const auto output_shape = (affine && half) || rounded || bf16_output
                    ? core::TensorShape::from_dims({6, width}) : in.latent.shape;
                const auto output = core::wrap_tensor(prediction, output_shape, prediction->type);
                const auto updated = core::wrap_tensor(ggml_scale(ctx.ggml, in.latent.tensor, 1.0F),
                    in.latent.shape, GGML_TYPE_F32);
                return {in.latent, output, updated};
            }, modules::DeviceDenoiserOptimizations{policy != 0 || bf16_output, policy != 0 || bf16_output,
                policy != 0 || bf16_output});
        std::vector<float> values(static_cast<size_t>(inputs[policy].latent.shape.num_elements()));
        std::vector<float> scales(affine || gated ? static_cast<size_t>(inputs[policy].scale.shape.num_elements()) : 0);
        std::vector<float> biases(static_cast<size_t>(inputs[policy].bias.shape.num_elements()));
        for (size_t i = 0; i < values.size(); ++i) values[i] = rounded && !bf16_output ? (i % 2 ? -1.0F : 1.0F) : std::sin(float(i) * 0.17F);
        for (size_t i = 0; i < scales.size(); ++i) scales[i] = rounded && !bf16_output ? 0.0F : std::cos(float(i) * 0.13F);
        for (size_t i = 0; i < biases.size(); ++i) biases[i] = rounded && !bf16_output ? 0.125F : std::sin(float(i) * 0.07F) * 0.3F;
        core::write_tensor_f32(inputs[policy].latent, values);
        if (affine || gated) core::write_tensor_f32(inputs[policy].scale, scales);
        core::write_tensor_f32(inputs[policy].bias, biases);
    }
    // Both policies coexist on one backend; executing the optimized graph must
    // not alter the baseline graph's dispatch or cached state.
    const auto baseline = runtimes[0]->predict();
    const auto optimized = runtimes[1]->predict();
    const auto baseline_again = runtimes[0]->predict();
    const auto optimized_again = runtimes[1]->predict();
    if (baseline_again != baseline || optimized_again != optimized)
        throw std::runtime_error("Denoiser fusion graph isolation mismatch width=" + std::to_string(width) +
            " epilogue=" + std::to_string(static_cast<int>(epilogue)) + " batch=" + std::to_string(per_batch) +
            " half=" + std::to_string(half) + " rounded=" + std::to_string(rounded) + " baseline_changed=" + std::to_string(baseline_again != baseline));
    if (bf16_output && baseline != optimized) {
        for (size_t i = 0; i < baseline.size(); ++i) {
            if (baseline[i] != optimized[i])
                throw std::runtime_error("Direct BF16 epilogue mismatch width=" + std::to_string(width) +
                    " epilogue=" + std::to_string(static_cast<int>(epilogue)) + " batch=" + std::to_string(per_batch) +
                    " index=" + std::to_string(i) + " before=" + std::to_string(baseline[i]) +
                    " after=" + std::to_string(optimized[i]));
        }
    }
    for (size_t i = 0; i < baseline.size(); ++i)
        if (!std::isfinite(optimized[i]) || std::abs(baseline[i] - optimized[i]) > (affine ? 0.00001F : 0.000001F))
            throw std::runtime_error("Denoiser fusion output mismatch");
}

void check_device_denoiser(core::ExecutionContext & execution) {
    core::TensorValue latent, delta;
    int builds = 0;
    modules::DeviceDenoiserRuntime runtime(execution,
        {"test.device_denoiser", 1024 * 1024, 128},
        [&](core::ModuleBuildContext & ctx) -> modules::DeviceDenoiserGraph {
            ++builds;
            latent = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({3, 7}));
            delta = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
            ggml_set_input(latent.tensor);
            ggml_set_input(delta.tensor);
            const auto prediction = modules::MulModule(true).build(ctx, latent, delta);
            const auto updated = modules::AddModule().build(ctx, latent, prediction);
            return {latent, prediction, updated};
        });
    const std::vector<float> initial(21, 2.0F);
    const std::vector<float> deltas{0.5F, -0.5F, 1.0F};
    modules::FlowSamplerGraphSequence sequence;
    sequence.steps = deltas.size();
    sequence.step_bindings = {{delta, 0, sizeof(float), deltas.data(), deltas.size() * sizeof(float)}};
    for (int request = 0; request < 3; ++request) {
        const auto result = runtime.sample(initial, sequence);
        if (builds != 1 || result != std::vector<float>(21, 3.0F))
            throw std::runtime_error("Device denoiser graph reuse mismatch request=" + std::to_string(request) +
                " builds=" + std::to_string(builds) + " first=" + std::to_string(result.empty() ? -999.0F : result.front()));
    }
    core::write_tensor_f32(latent, initial);
    core::write_tensor_f32(delta, {0.5F});
    if (runtime.predict() != std::vector<float>(21, 1.0F))
        throw std::runtime_error("Device denoiser prediction mismatch");
}

void check_conditioning(core::ExecutionContext & execution,
    modules::AdaptiveLayerNormMode mode, bool table_backed, bool strided) {
    core::PreparedGraph graph(execution, {"test.flow_sampler_conditioning", 1024 * 1024, 256});
    auto & ctx = graph.build_context();
    const float eps = 0.003F;
    const std::vector<float> values{
        1.0F, 2.0F, 4.0F, 7.0F, -3.0F, -1.0F, 2.0F, 6.0F,
        0.5F, 0.5F, 0.5F, 0.5F, 0.01F, 0.02F, 0.03F, 0.04F,
        8.0F, 5.0F, 2.0F, -1.0F, -0.2F, -0.1F, 0.1F, 0.2F};
    const std::vector<float> shifts{0.25F, -0.5F, 1.25F, -1.5F};
    const std::vector<float> scales{-1.25F, -0.25F, 0.5F, 1.5F};
    auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, 3, 4}));
    ggml_set_input(input.tensor);
    const modules::AdaptiveLayerNormModule module(table_backed
        ? modules::AdaptiveLayerNormConfig{4, eps, mode, 0, 1}
        : modules::AdaptiveLayerNormConfig{4, eps, mode, 0, 0});
    core::TensorValue output, table, modulation, shift, scale;
    std::vector<float> table_values, modulation_values;
    if (table_backed) {
        table = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, 4}));
        modulation = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, 8}));
        for (float value : shifts) table_values.push_back(value * 0.5F);
        for (float value : scales) table_values.push_back(value * 0.5F);
        modulation_values = table_values;
        ggml_set_input(table.tensor);
        ggml_set_input(modulation.tensor);
        if (mode == modules::AdaptiveLayerNormMode::LayerAffine) {
            for (const auto config : {modules::AdaptiveLayerNormConfig{4, eps, mode},
                    modules::AdaptiveLayerNormConfig{4, eps, mode, 0, -1}}) {
                bool rejected = false;
                try {
                    const modules::AdaptiveLayerNormModule invalid_table(config);
                    invalid_table.build(ctx, input, modulation, {table});
                }
                catch (const std::runtime_error &) { rejected = true; }
                if (!rejected) throw std::runtime_error("Table binding accepted a negative modulation index");
            }
        }
        output = module.build(ctx, input, modulation, {table});
    } else if (strided) {
        // Interleaved table columns become genuinely strided [1,1,4] planes.
        table = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({4, 2}));
        ggml_set_input(table.tensor);
        for (size_t i = 0; i < 4; ++i) {
            table_values.push_back(shifts[i]);
            table_values.push_back(scales[i]);
        }
        auto * shift_column = ggml_view_3d(ctx.ggml, table.tensor, 1, 4, 1,
            2 * sizeof(float), 8 * sizeof(float), 0);
        auto * scale_column = ggml_view_3d(ctx.ggml, table.tensor, 1, 4, 1,
            2 * sizeof(float), 8 * sizeof(float), sizeof(float));
        shift = core::wrap_tensor(ggml_permute(ctx.ggml, shift_column, 1, 0, 2, 3),
            core::TensorShape::from_dims({1, 1, 4}), GGML_TYPE_F32);
        scale = core::wrap_tensor(ggml_permute(ctx.ggml, scale_column, 1, 0, 2, 3),
            core::TensorShape::from_dims({1, 1, 4}), GGML_TYPE_F32);
        if (ggml_is_contiguous(shift.tensor) || ggml_is_contiguous(scale.tensor))
            throw std::runtime_error("Conditioning fixture must contain strided planes");
        output = module.build(ctx, input, modules::AdaptiveLayerNormParameters{shift, scale});
    } else {
        shift = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, 4}));
        scale = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, 4}));
        ggml_set_input(shift.tensor);
        ggml_set_input(scale.tensor);
        output = module.build(ctx, input, modules::AdaptiveLayerNormParameters{shift, scale});
    }
    ggml_set_output(output.tensor);
    ggml_build_forward_expand(graph.graph(), output.tensor);
    graph.allocate();
    core::write_tensor_f32(input, values);
    if (table_backed || strided) core::write_tensor_f32(table, table_values);
    if (table_backed) core::write_tensor_f32(modulation, modulation_values);
    if (!table_backed && !strided) {
        if (mode != modules::AdaptiveLayerNormMode::Scale)
            core::write_tensor_f32(shift, shifts);
        core::write_tensor_f32(scale, scales);
    }
    graph.compute();
    const auto actual = core::read_tensor_f32(output.tensor);
    if (actual.size() != values.size()) throw std::runtime_error("Conditioning output shape mismatch");
    // Independent double oracle: population variance, epsilon inside sqrt,
    // and featurewise broadcast over both batch and token dimensions.
    for (size_t row = 0; row < 6; ++row) {
        double mean = 0.0, square_mean = 0.0, variance = 0.0;
        for (size_t col = 0; col < 4; ++col) {
            const double x = values[row * 4 + col];
            mean += x / 4.0;
            square_mean += x * x / 4.0;
        }
        for (size_t col = 0; col < 4; ++col) {
            const double centered = static_cast<double>(values[row * 4 + col]) - mean;
            variance += centered * centered / 4.0;
        }
        for (size_t col = 0; col < 4; ++col) {
            const size_t index = row * 4 + col;
            double normalized = values[index];
            if (mode == modules::AdaptiveLayerNormMode::LayerAffine)
                normalized = (normalized - mean) / std::sqrt(variance + static_cast<double>(eps));
            else if (mode == modules::AdaptiveLayerNormMode::RmsAffine)
                normalized /= std::sqrt(square_mean + static_cast<double>(eps));
            const double expected = mode == modules::AdaptiveLayerNormMode::Scale
                ? normalized * static_cast<double>(scales[col])
                : normalized * (1.0 + static_cast<double>(scales[col])) + static_cast<double>(shifts[col]);
            if (!std::isfinite(actual[index]) || std::abs(static_cast<double>(actual[index]) - expected) > 3e-5)
                throw std::runtime_error("AdaptiveLayerNorm host-double oracle mismatch: mode=" +
                    std::to_string(static_cast<int>(mode)) + " index=" + std::to_string(index) +
                    " table=" + std::to_string(table_backed) + " strided=" + std::to_string(strided));
        }
    }
}

void check(core::ExecutionContext & execution, modules::FlowSamplerEulerGraphConfig rounding) {
    // Sequence constants have caller-owned storage outside graph scratch, so
    // the allocator cannot reuse them for intermediate results between steps.
    std::unique_ptr<ggml_context, decltype(&ggml_free)> inputs(
        ggml_init({16 * 1024, nullptr, true}), ggml_free);
    if (!inputs) throw std::runtime_error("Cannot allocate test input metadata");
    core::ModuleBuildContext input_ctx{inputs.get(), "test.predictions", execution.backend_type()};
    auto positive = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2, 2}));
    auto negative = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({2}));
    auto guidance = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> input_buffer(
        ggml_backend_alloc_ctx_tensors(inputs.get(), execution.backend()), ggml_backend_buffer_free);
    if (!input_buffer) throw std::runtime_error("Cannot allocate test input buffer");
    core::PreparedGraph graph(execution, {"test.flow_sampler_graph", 1024 * 1024, 128});
    auto & ctx = graph.build_context();
    ctx.elide_identity_reshapes = true;
    const auto shape = core::TensorShape::from_dims({2});
    auto latent = core::make_tensor(ctx, GGML_TYPE_F32, shape);
    auto dt = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
    auto branch = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
    for (const auto value : {latent, positive, negative, guidance, dt, branch}) ggml_set_input(value.tensor);
    auto selected = core::wrap_tensor(ggml_get_rows(ctx.ggml, positive.tensor, branch.tensor),
        core::TensorShape::from_dims({1, 2}), GGML_TYPE_F32);
    selected = core::reshape_tensor(ctx, selected, shape);
    auto velocity = modules::FlowSamplerCfgGraphModule().build(ctx, selected, negative, guidance);
    auto updated = modules::FlowSamplerEulerGraphModule(rounding).build(ctx, latent, velocity, dt);
    ggml_build_forward_expand(graph.graph(), ggml_cpy(ctx.ggml, updated.tensor, latent.tensor));
    graph.allocate();
    modules::DeviceFlowSamplerRuntime runtime(graph, latent);

    const std::vector<float> pos{0.731F, -0.217F, -0.317F, 0.811F}, neg{-0.119F, 0.413F};
    const float scale = 2.25F;
    const std::vector<float> initial{0.371F, -0.683F}, deltas{-0.13F, -0.27F, -0.6F};
    modules::FlowSamplerGraphSequence sequence;
    sequence.steps = deltas.size();
    core::write_tensor_f32(positive, pos);
    core::write_tensor_f32(negative, neg);
    core::write_tensor_f32(guidance, std::vector<float>{scale});
    modules::FlowSamplerGraphStepBinding binding;
    binding.target = dt;
    binding.bytes_per_step = sizeof(float);
    binding.data = deltas.data();
    binding.data_bytes = deltas.size() * sizeof(float);
    sequence.step_bindings.push_back(std::move(binding));
    const std::vector<int32_t> branches{0, 1, 0};
    modules::FlowSamplerGraphStepBinding branch_binding;
    branch_binding.target = branch;
    branch_binding.bytes_per_step = sizeof(int32_t);
    branch_binding.data = branches.data();
    branch_binding.data_bytes = branches.size() * sizeof(int32_t);
    sequence.step_bindings.push_back(std::move(branch_binding));

    auto expected = initial;
    for (size_t step = 0; step < deltas.size(); ++step) {
        for (size_t i = 0; i < expected.size(); ++i) {
            float update = (neg[i] + scale * (pos[static_cast<size_t>(branches[step]) * 2 + i] - neg[i])) * deltas[step];
            if (rounding.delta_rounding == modules::FlowSamplerGraphRounding::Bf16)
                core::round_f32_to_bf16_in_place(&update, 1);
            expected[i] += update;
            if (rounding.output_rounding == modules::FlowSamplerGraphRounding::Bf16)
                core::round_f32_to_bf16_in_place(&expected[i], 1);
        }
    }
    auto * original_graph = graph.graph();
    auto * original_storage = latent.tensor->data;
    for (int repeat = 0; repeat < 2; ++repeat) {
        const auto actual = runtime.run(initial, sequence);
        for (size_t i = 0; i < expected.size(); ++i) {
            if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 1e-6F) {
                debug::trace_log_f32("test.flow_sampler_graph.actual", {2}, actual);
                debug::trace_log_f32("test.flow_sampler_graph.expected", {2}, expected);
                debug::trace_log_scalar("test.flow_sampler_graph.delta_rounding",
                    std::string_view(rounding.delta_rounding == modules::FlowSamplerGraphRounding::Bf16 ? "Bf16" : "FullPrecision"));
                debug::trace_log_scalar("test.flow_sampler_graph.output_rounding",
                    std::string_view(rounding.output_rounding == modules::FlowSamplerGraphRounding::Bf16 ? "Bf16" : "FullPrecision"));
                debug::trace_log_scalar("test.flow_sampler_graph.repeat", repeat);
                debug::trace_log_scalar("test.flow_sampler_graph.mismatch_index", i);
                debug::trace_log_scalar("test.flow_sampler_graph.absolute_error",
                    std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i])));
                debug::trace_log_scalar("test.flow_sampler_graph.negative_aliases_velocity",
                    negative.tensor->data == velocity.tensor->data);
                debug::trace_log_scalar("test.flow_sampler_graph.latent_aliases_updated",
                    latent.tensor->data == updated.tensor->data);
                debug::trace_log_f32("test.flow_sampler_graph.negative_after", {2},
                    core::read_tensor_f32(negative.tensor));
                debug::trace_log_f32("test.flow_sampler_graph.negative_initial", {2}, neg);
                throw std::runtime_error("Repeated CFG/Euler sequence changed arithmetic");
            }
        }
        if (graph.graph() != original_graph || latent.tensor->data != original_storage)
            throw std::runtime_error("Sequence replaced graph or latent storage");
        if (core::read_tensor_f32(positive.tensor) != pos || core::read_tensor_f32(negative.tensor) != neg ||
            core::read_tensor_f32(guidance.tensor) != std::vector<float>{scale})
            throw std::runtime_error("Sequence modified caller-owned constants");
    }
    auto invalid = sequence;
    --invalid.step_bindings.front().data_bytes;
    const auto before_invalid = core::read_tensor_f32(latent.tensor);
    bool rejected = false;
    try { runtime.run(initial, invalid); }
    catch (const std::runtime_error &) { rejected = true; }
    if (!rejected || core::read_tensor_f32(latent.tensor) != before_invalid)
        throw std::runtime_error("Invalid bindings must fail before mutating latent");
    invalid = sequence;
    invalid.step_bindings.front().data = nullptr;
    rejected = false;
    try { runtime.run(initial, invalid); }
    catch (const std::runtime_error &) { rejected = true; }
    if (!rejected || core::read_tensor_f32(latent.tensor) != before_invalid)
        throw std::runtime_error("Null sources must fail before mutating latent");
    invalid = sequence;
    invalid.step_bindings.front().target = latent;
    rejected = false;
    try { runtime.run(initial, invalid); }
    catch (const std::runtime_error &) { rejected = true; }
    if (!rejected || core::read_tensor_f32(latent.tensor) != before_invalid)
        throw std::runtime_error("Overlapping bindings must fail before mutating latent");
    invalid = sequence;
    invalid.step_bindings.front().target = core::wrap_tensor(
        ggml_view_1d(inputs.get(), latent.tensor, 1, sizeof(float)),
        core::TensorShape::from_dims({1}), GGML_TYPE_F32);
    rejected = false;
    try { runtime.run(initial, invalid); }
    catch (const std::runtime_error &) { rejected = true; }
    if (!rejected || core::read_tensor_f32(latent.tensor) != before_invalid)
        throw std::runtime_error("Aliased view bindings must fail before mutating latent");
}

void check_denoiser_phases(core::ExecutionContext & execution) {
    core::PreparedGraph state(execution, {"test.phase_state", 64 * 1024, 32});
    auto & ctx = state.build_context();
    auto latent = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({17}));
    auto dt = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
    for (const auto & value : {latent, dt}) {
        ggml_set_input(value.tensor);
        ggml_set_output(value.tensor);
        ggml_build_forward_expand(state.graph(), value.tensor);
    }
    state.allocate();
    core::GraphWorkspace workspace(execution.backend());
    core::PreparedGraph shared(execution, {"test.shared_allocation", 64 * 1024, 32});
    auto shared_input = core::make_tensor(shared.build_context(), GGML_TYPE_F32,
        core::TensorShape::from_dims({17}));
    ggml_set_output(shared_input.tensor);
    ggml_build_forward_expand(shared.graph(), shared_input.tensor);
    workspace.add(shared);
    workspace.prepare();
    bool rejected = false;
    try { shared.allocate(); }
    catch (const std::runtime_error &) { rejected = true; }
    if (!rejected) throw std::runtime_error("Shared graph accepted a second allocation");
    std::vector<modules::DeviceDenoiserRuntime::Builder> phases;
    for (int phase = 0; phase < 2; ++phase) {
        phases.push_back([&, phase](core::ModuleBuildContext & build) -> modules::DeviceDenoiserGraph {
            const auto velocity = core::wrap_tensor(ggml_scale(build.ggml, latent.tensor,
                phase == 0 ? 0.5F : -0.25F), latent.shape, GGML_TYPE_F32);
            return {latent, velocity, modules::FlowSamplerEulerGraphModule().build(build, latent, velocity, dt)};
        });
    }
    {
        core::TensorValue unallocated;
        const auto unsafe = [&](core::ModuleBuildContext & build) -> modules::DeviceDenoiserGraph {
            if (!unallocated.tensor) unallocated = core::make_tensor(build, GGML_TYPE_F32,
                core::TensorShape::from_dims({17}));
            return {unallocated, unallocated, unallocated};
        };
        const std::vector<modules::DeviceDenoiserRuntime::Builder> unsafe_phases{
            unsafe, unsafe};
        rejected = false;
        try { modules::DeviceDenoiserRuntime invalid(execution, {"test.unsafe_phases", 128 * 1024, 32}, unsafe_phases); }
        catch (const std::runtime_error &) { rejected = true; }
        if (!rejected) throw std::runtime_error("Multiphase state was allocated in shared scratch");
    }
    modules::DeviceDenoiserRuntime runtime(execution, {"test.denoiser_phases", 128 * 1024, 32}, phases);
    const std::vector<float> initial(17, 0.5F);
    const std::vector<float> steps{0.25F, 0.5F, -0.25F, 0.125F, -0.5F};
    modules::FlowSamplerGraphSequence sequence;
    sequence.steps = steps.size();
    sequence.graph_indices = {0, 1, 0, 1, 1};
    sequence.step_bindings = {{dt, 0, sizeof(float), steps.data(), steps.size() * sizeof(float)}};
    float expected = initial.front();
    for (size_t i = 0; i < steps.size(); ++i)
        expected += expected * (sequence.graph_indices[i] == 0 ? 0.5F : -0.25F) * steps[i];
    const auto first = runtime.sample(initial, sequence);
    const auto repeated = runtime.sample(initial, sequence);
    if (first != repeated)
        throw std::runtime_error("Device denoiser phases retained state between sequences");
    for (float value : first)
        if (value != expected) throw std::runtime_error("Device denoiser phase schedule differs from host Euler");
    sequence.graph_indices.back() = phases.size();
    rejected = false;
    try { runtime.sample(initial, sequence); } catch (const std::runtime_error &) { rejected = true; }
    if (!rejected || core::read_tensor_f32(latent.tensor) != repeated)
        throw std::runtime_error("Invalid denoiser phase must fail before mutating state");
}

}  // namespace

int main(int argc, char ** argv) {
    const auto started = std::chrono::steady_clock::now();
    try {
        if ((argc != 3 && argc != 5) || std::string(argv[1]) != "--log" ||
            (argc == 5 && (std::string(argv[3]) != "--backend" ||
                (std::string(argv[4]) != "cuda" && std::string(argv[4]) != "vulkan"))))
            throw std::runtime_error("Usage: test_device_denoiser_runtime --log PATH");
        debug::LoggingConfig logging;
        logging.enabled = true;
        logging.file_path = argv[2];
        debug::configure_logging(logging);
        const bool cuda = argc == 5 && std::string(argv[4]) == "cuda";
        const bool vulkan = argc == 5 && std::string(argv[4]) == "vulkan";
        core::ExecutionContext execution({cuda ? core::BackendType::Cuda :
            vulkan ? core::BackendType::Vulkan : core::BackendType::Cpu, 0, 8});
        check_device_denoiser(execution);
        check_denoiser_phases(execution);
        check_private_view_preparation(execution);
        for (int64_t width : {17, 64, 1024, 1025, 1536}) {
            for (auto epilogue : {DenoiserEpilogue::BiasGelu, DenoiserEpilogue::BiasSilu,
                    DenoiserEpilogue::BiasGeluQuick, DenoiserEpilogue::BiasGeluErf, DenoiserEpilogue::LayerAffine,
                    DenoiserEpilogue::RmsAffine, DenoiserEpilogue::GatedResidual, DenoiserEpilogue::ReshapedAffine, DenoiserEpilogue::FanoutResidual, DenoiserEpilogue::ReshapedGatedResidual})
                for (bool per_batch : {false, true}) check_denoiser_fusions(execution, width, epilogue, per_batch);
            check_denoiser_fusions(execution, width, DenoiserEpilogue::LayerAffine, false, false, true);
            if (vulkan) {
                for (auto epilogue : {DenoiserEpilogue::BiasGelu, DenoiserEpilogue::BiasSilu,
                        DenoiserEpilogue::BiasGeluQuick, DenoiserEpilogue::BiasGeluErf})
                    for (bool per_batch : {false, true})
                        check_denoiser_fusions(execution, width, epilogue, per_batch, false, false, true);
                for (auto epilogue : {DenoiserEpilogue::LayerAffine, DenoiserEpilogue::RmsAffine})
                    for (bool per_batch : {false, true})
                        check_denoiser_fusions(execution, width, epilogue, per_batch, false, true, true);
                check_denoiser_fusions(execution, width, DenoiserEpilogue::GatedResidual, true, false, true, true);
            }
            if (cuda) {
                check_denoiser_fusions(execution, width, DenoiserEpilogue::BiasGelu, false, true);
                check_denoiser_fusions(execution, width, DenoiserEpilogue::LayerAffine, false, true);
                check_denoiser_fusions(execution, width, DenoiserEpilogue::GatedResidual, false, true);
                check_denoiser_fusions(execution, width, DenoiserEpilogue::ReshapedGatedResidual, false, true);
                if (cuda) check_denoiser_fusions(execution, width, DenoiserEpilogue::GatedResidual,
                    false, true, false, false, false);
            }
        }
        for (auto mode : {modules::AdaptiveLayerNormMode::LayerAffine, modules::AdaptiveLayerNormMode::RmsAffine,
                modules::AdaptiveLayerNormMode::Affine, modules::AdaptiveLayerNormMode::Scale}) {
            check_conditioning(execution, mode, true, false);
            check_conditioning(execution, mode, false, false);
            check_conditioning(execution, mode, false, true);
        }
        for (auto delta : {modules::FlowSamplerGraphRounding::FullPrecision, modules::FlowSamplerGraphRounding::Bf16})
            for (auto output : {modules::FlowSamplerGraphRounding::FullPrecision, modules::FlowSamplerGraphRounding::Bf16})
                check(execution, {delta, output});
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        std::cout << "device_denoiser_runtime: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        std::cerr << error.what() << '\n';
        return 1;
    }
}
