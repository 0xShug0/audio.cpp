#include "engine/framework/modules/mamba_modules.h"

#include "engine/framework/modules/linear_module.h"

#include <ggml.h>

#include <stdexcept>

namespace engine::modules {
namespace {

void validate_config(const Mamba1Config & config) {
    if (config.model_size <= 0 || config.inner_size <= 0 || config.state_size <= 0 ||
        config.conv_kernel <= 0 || config.dt_rank <= 0) {
        throw std::runtime_error("Mamba1Module dimensions must be positive");
    }
}

}  // namespace

Mamba1Module::Mamba1Module(Mamba1Config config) : config_(config) {
    validate_config(config_);
}

const Mamba1Config & Mamba1Module::config() const noexcept {
    return config_;
}

Mamba1Outputs Mamba1Module::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Mamba1Weights & weights,
    const core::TensorValue & sequence_ids,
    const std::optional<Mamba1State> & state) const {
    if (ctx.ggml == nullptr) {
        throw std::runtime_error("ModuleBuildContext.ggml is null");
    }
    core::validate_rank_between(input, 3, 3, "mamba input");
    core::validate_last_dim(input, config_.model_size, "mamba input");
    const int64_t batch = input.shape.dims[0];
    const int64_t length = input.shape.dims[1];
    const int64_t inner = config_.inner_size;
    const int64_t states = config_.state_size;
    const int64_t history = config_.conv_kernel - 1;
    core::validate_shape(sequence_ids, core::TensorShape::from_dims({batch}), "mamba sequence ids");
    if (state.has_value()) {
        core::validate_shape(state->conv, core::TensorShape::from_dims({batch, inner, history}), "mamba conv state");
        core::validate_shape(state->ssm, core::TensorShape::from_dims({batch, inner, 1, states}), "mamba ssm state");
    }
    // CUDA/HIP and WebGPU use the decomposed Mamba-1 graph with the legacy scan.
    const bool legacy = core::uses_ggml_cuda_or_hip_backend(ctx.backend_type) ||
                        ctx.backend_type == core::BackendType::WebGPU;
    // Vulkan keeps the projections' activations in F32, as realtime_reuse does.
    const bool precise = ctx.backend_type == core::BackendType::Vulkan;
    auto * g = ctx.ggml;

    const auto sequence_shape = core::TensorShape::from_dims({batch, length, inner});
    auto * xz = LinearModule({config_.model_size, 2 * inner, false, GGML_PREC_DEFAULT, false, precise}).build(ctx, input, {weights.in_proj, std::nullopt}).tensor;
    auto * u = ggml_view_3d(g, xz, inner, length, batch, xz->nb[1], xz->nb[2], 0);
    auto * z = ggml_view_3d(g, xz, inner, length, batch, xz->nb[1], xz->nb[2], inner * ggml_element_size(xz));
    if (ctx.backend_type == core::BackendType::Vulkan) {
        z = ggml_cont(g, z);
    }
    // The convolution runs along time: [length, inner, batch].
    auto * u_time = ggml_permute(g, u, 1, 0, 2, 3);

    Mamba1Outputs out;
    ggml_tensor * conv = nullptr;
    if (state.has_value()) {
        auto * joined = ggml_concat(g, state->conv.tensor, ggml_cont(g, u_time), 0);
        conv = ggml_ssm_conv(g, joined, weights.conv_weight.tensor);
        out.next_state = Mamba1State{
            core::wrap_tensor(ggml_view_3d(g, joined, history, inner, batch, joined->nb[1], joined->nb[2],
                length * ggml_element_size(joined)), state->conv.shape),
            {},
        };
    } else if (!legacy) {
        conv = ggml_ssm_conv_causal_silu(g, u_time, weights.conv_weight.tensor, weights.conv_bias.tensor);
    } else {
        auto * padded = ggml_pad_ext(g, ggml_cont(g, u_time), static_cast<int>(history), 0, 0, 0, 0, 0, 0, 0);
        conv = ggml_ssm_conv(g, padded, weights.conv_weight.tensor);
        ggml_ssm_conv_set_fusion(conv, GGML_SSM_CONV_FUSION_CAUSAL_PAD);
    }
    if (state.has_value() || legacy) {
        conv = ggml_silu(g, ggml_add(g, conv, ggml_reshape_3d(g, weights.conv_bias.tensor, inner, 1, 1)));
    }

    auto * projected = LinearModule({inner, config_.dt_rank + 2 * states, false, GGML_PREC_DEFAULT, false, precise})
        .build(ctx, core::wrap_tensor(conv, sequence_shape), {weights.x_proj, std::nullopt}).tensor;
    auto * initial = state.has_value()
        ? state->ssm.tensor
        : ggml_fill(g, ggml_new_tensor_4d(g, GGML_TYPE_F32, states, 1, inner, batch), 0.0F);
    auto * x = ggml_reshape_4d(g, conv, 1, inner, length, batch);
    const size_t y_bytes = static_cast<size_t>(inner * length * batch) * sizeof(float);

    ggml_tensor * scan = nullptr;
    ggml_tensor * y = nullptr;
    if (!legacy) {
        scan = ggml_ssm_scan_mamba1(g, initial, x, projected, weights.a.tensor, sequence_ids.tensor,
            weights.dt_weight.tensor, weights.dt_bias.tensor, weights.d.tensor, z);
        y = ggml_view_3d(g, scan, inner, length, batch, inner * sizeof(float), inner * length * sizeof(float), 0);
    } else {
        auto * dt = ggml_cont(g, ggml_view_3d(g, projected, config_.dt_rank, length, batch,
            projected->nb[1], projected->nb[2], 0));
        auto * delta = LinearModule({config_.dt_rank, inner, true})
            .build(ctx, core::wrap_tensor(dt, core::TensorShape::from_dims({batch, length, config_.dt_rank})),
                {weights.dt_weight, weights.dt_bias}).tensor;
        auto * b = ggml_view_4d(g, projected, states, 1, length, batch, states * sizeof(float),
            projected->nb[1], projected->nb[2], config_.dt_rank * sizeof(float));
        auto * c = ggml_view_4d(g, projected, states, 1, length, batch, states * sizeof(float),
            projected->nb[1], projected->nb[2], (config_.dt_rank + states) * sizeof(float));
        scan = ggml_ssm_scan(g, initial, x, delta, weights.a.tensor, b, c, sequence_ids.tensor);
        const bool fused_gate = !state.has_value();
        if (fused_gate) {
            ggml_ssm_scan_set_fusion(scan, GGML_SSM_SCAN_FUSION_GATE);
        }
        auto * raw = ggml_view_3d(g, scan, inner, length, batch, inner * sizeof(float), inner * length * sizeof(float), 0);
        auto * skipped = ggml_add(g, raw, ggml_mul(g, conv, ggml_reshape_3d(g, weights.d.tensor, inner, 1, 1)));
        y = ggml_swiglu_split(g, z, skipped);
        if (fused_gate) {
            y = ggml_cpy(g, y, raw);
        }
    }
    if (state.has_value()) {
        out.next_state->ssm = core::wrap_tensor(
            ggml_view_4d(g, scan, states, 1, inner, batch, initial->nb[1], initial->nb[2], initial->nb[3], y_bytes),
            state->ssm.shape);
    }
    out.output = LinearModule({inner, config_.model_size, false, GGML_PREC_DEFAULT, false, precise})
        .build(ctx, core::wrap_tensor(y, sequence_shape), {weights.out_proj, std::nullopt});
    return out;
}

}  // namespace engine::modules
