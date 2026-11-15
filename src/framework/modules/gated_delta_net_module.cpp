#include "engine/framework/modules/gated_delta_net_module.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <cmath>
#include <stdexcept>

namespace engine::modules {
namespace {

const core::ModulePortSpec kInputs[] = {
    {"input", core::PortKind::Activation, false},
    {"recurrent_state", core::PortKind::Activation, false},
    {"conv_tail", core::PortKind::Activation, true},
};
const core::ModulePortSpec kOutputs[] = {
    {"output", core::PortKind::Activation, false},
    {"next_conv_tail", core::PortKind::Activation, false},
    {"next_recurrent_state", core::PortKind::Activation, false},
};
const core::ModuleSchema kSchema = {
    "GatedDeltaNet", "nn.block", kInputs, 3, kOutputs, 3,
    "Causal convolution and gated delta recurrence with borrowed state.",
};

}  // namespace

GatedDeltaNetModule::GatedDeltaNetModule(GatedDeltaNetConfig config) : config_(config) {
    if (config.hidden_size <= 0 || config.key_head_dim <= 0 || config.key_heads <= 0 ||
        config.value_head_dim != config.key_head_dim || config.value_heads <= 0 || config.conv_kernel <= 1 ||
        config.value_heads % config.key_heads != 0 || !std::isfinite(config.norm_epsilon) ||
        config.norm_epsilon <= 0 || !std::isfinite(config.qk_epsilon) || config.qk_epsilon <= 0) {
        throw std::invalid_argument("Invalid GatedDeltaNet configuration");
    }
}

const GatedDeltaNetConfig & GatedDeltaNetModule::config() const noexcept { return config_; }
const core::ModuleSchema & GatedDeltaNetModule::schema() const noexcept { return static_schema(); }
const core::ModuleSchema & GatedDeltaNetModule::static_schema() noexcept { return kSchema; }

GatedDeltaNetOutput GatedDeltaNetModule::build(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const core::TensorValue & state,
    const GatedDeltaNetWeights & weights, const core::TensorValue & conv_tail) const {
    core::validate_rank_between(input, 3, 3, "input");
    core::validate_last_dim(input, config_.hidden_size, "input");
    const auto batch = input.shape.dims[0], steps = input.shape.dims[1];
    const auto kd = config_.key_head_dim * config_.key_heads;
    const auto vd = config_.value_head_dim * config_.value_heads;
    const auto channels = 2 * kd + vd;
    const auto rows = batch * steps;
    const auto history = config_.conv_kernel - 1;
    if (!state.valid() || state.type != GGML_TYPE_F32 ||
        ggml_nelements(state.tensor) != batch * vd * config_.key_head_dim) {
        throw std::invalid_argument("GatedDeltaNet recurrent state shape/type mismatch");
    }
    if (conv_tail.valid()) {
        core::validate_shape(conv_tail, core::TensorShape::from_dims({batch, channels, history}), "conv_tail");
    }
    auto * g = ctx.ggml;
    const auto round = [&](const core::TensorValue & x) {
        return config_.round_bf16_activations ? Bf16RoundModule{}.build(ctx, x) : x;
    };
    const auto project = [&](const LinearWeights & w, int64_t width) {
        return round(LinearModule({config_.hidden_size, width, false, config_.projection_precision}).build(ctx, input, w));
    };
    core::TensorValue qkv, z, a, beta;
    const bool packed = weights.in_proj.weight.valid();
    if (packed) {
        const auto mixed = project(weights.in_proj, channels + vd + 2 * config_.value_heads);
        qkv = SliceModule({2, 0, channels}).build(ctx, mixed);
        z = SliceModule({2, channels, vd}).build(ctx, mixed);
        a = SliceModule({2, channels + vd, config_.value_heads}).build(ctx, mixed);
        beta = SliceModule({2, channels + vd + config_.value_heads, config_.value_heads}).build(ctx, mixed);
    } else {
        qkv = project(weights.in_proj_qkv, channels);
    }
    qkv = TransposeModule({{0, 2, 1}, 3}).build(ctx, qkv);
    // Concat reads strided projections directly; padding needs addressable rows.
    if (!conv_tail.valid()) qkv = core::ensure_backend_addressable_layout(ctx, qkv);
    auto window = conv_tail.valid() ? ConcatModule({2}).build(ctx, conv_tail, qkv)
        : core::wrap_tensor(ggml_pad_ext(g, qkv.tensor, static_cast<int>(history), 0, 0, 0, 0, 0, 0, 0),
            core::TensorShape::from_dims({batch, channels, steps + history}));
    auto kernel = core::reshape_tensor(ctx, weights.conv_weight,
        core::TensorShape::from_dims({channels, config_.conv_kernel}));
    auto conv = core::wrap_tensor(ggml_ssm_conv(g, window.tensor, kernel.tensor),
        core::TensorShape::from_dims({batch, steps, channels}));
    conv = round(SiluModule{}.build(ctx, round(conv)));
    const auto heads = [&](int64_t offset, int64_t count, int64_t width) {
        return core::wrap_tensor(ggml_view_4d(g, conv.tensor, width, count, steps, batch,
            width * sizeof(float), conv.tensor->nb[1], conv.tensor->nb[2], offset * sizeof(float)),
            core::TensorShape::from_dims({batch, steps, count, width}));
    };
    const auto normalize = [&](core::TensorValue x) {
        if (config_.qk_norm == GatedDeltaNetQkNorm::L2) {
            x = core::wrap_tensor(ggml_l2_norm(g, x.tensor, config_.qk_epsilon), x.shape);
        } else {
            x = core::wrap_tensor(ggml_scale(g,
                ggml_rms_norm(g, x.tensor, config_.qk_epsilon / static_cast<float>(config_.key_head_dim)),
                1.0F / std::sqrt(static_cast<float>(config_.key_head_dim))), x.shape);
        }
        x = round(x);
        const auto repeats = config_.value_heads / config_.key_heads;
        if (repeats != 1) {
            auto * shaped = ggml_reshape_4d(g, ggml_cont(g, x.tensor), config_.key_head_dim, 1, config_.key_heads, rows);
            auto * repeated = ggml_repeat_4d(g, shaped, config_.key_head_dim, repeats, config_.key_heads, rows);
            x = core::wrap_tensor(ggml_reshape_4d(g, repeated, config_.key_head_dim, config_.value_heads, steps, batch),
                core::TensorShape::from_dims({batch, steps, config_.value_heads, config_.key_head_dim}));
        }
        return x;
    };
    auto query = normalize(heads(0, config_.key_heads, config_.key_head_dim));
    auto key = normalize(heads(kd, config_.key_heads, config_.key_head_dim));
    auto value = heads(2 * kd, config_.value_heads, config_.value_head_dim);
    if (!packed) beta = project(weights.in_proj_b, config_.value_heads);
    beta = round(SigmoidModule{}.build(ctx, beta));
    if (!packed) a = project(weights.in_proj_a, config_.value_heads);
    // Per-head constants broadcast directly over time and batch. Materializing
    // them (or the strided packed a projection) adds copies at every decode step.
    a = core::wrap_tensor(ggml_softplus(g, ggml_add(g, a.tensor, weights.dt_bias.tensor)), a.shape);
    auto decay = core::wrap_tensor(ggml_scale(g,
        ggml_mul(g, a.tensor, ggml_exp(g, weights.a_log.tensor)), -1.0F), a.shape);
    const auto gate_shape = core::TensorShape::from_dims({batch, steps, config_.value_heads, 1});
    beta = core::reshape_tensor(ctx, beta, gate_shape);
    decay = core::reshape_tensor(ctx, decay, gate_shape);
    auto * delta = ggml_gated_delta_net(g, query.tensor, key.tensor, value.tensor, decay.tensor, beta.tensor, state.tensor);
    auto output = core::wrap_tensor(ggml_view_2d(g, delta, vd, rows, vd * sizeof(float), 0),
        core::TensorShape::from_dims({rows, vd}));
    auto next_state = core::wrap_tensor(ggml_view_1d(g, delta, ggml_nelements(state.tensor), rows * vd * sizeof(float)),
        core::TensorShape::from_dims({ggml_nelements(state.tensor)}));
    output = round(output);
    next_state = core::reshape_tensor(ctx, round(next_state), state.shape);
    if (!packed) z = project(weights.in_proj_z, vd);
    z = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, z),
        core::TensorShape::from_dims({rows * config_.value_heads, config_.value_head_dim}));
    output = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, output), z.shape);
    output = round(RMSNormModule({config_.value_head_dim, config_.norm_epsilon, true, false}, config_.norm_lowering)
        .build(ctx, output, weights.norm));
    z = round(SiluModule{}.build(ctx, z));
    output = round(MulModule{}.build(ctx, output, z));
    output = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, output),
        core::TensorShape::from_dims({batch, steps, vd}));
    output = round(LinearModule({vd, config_.hidden_size, false, config_.projection_precision}).build(ctx, output, weights.out_proj));
    auto next_tail = SliceModule({2, steps, history}).build(ctx, window);
    return {output, next_tail, next_state};
}

}  // namespace engine::modules
