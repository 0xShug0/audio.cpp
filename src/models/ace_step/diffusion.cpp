#include "engine/models/ace_step/diffusion.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/io/binary.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/sampling/diffusion_math.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/framework/modules/structural_modules.h"
#include "helper_utils.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::ace_step {
namespace {

namespace modules = engine::modules;

using Clock = std::chrono::steady_clock;

struct PrecomputedCrossAttentionKV {
    core::TensorValue key;
    core::TensorValue value;
};

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

core::TensorValue ensure_contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    return core::ensure_backend_addressable_layout(ctx, input);
}

core::TensorValue ensure_f32(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    if (input.type == GGML_TYPE_F32) {
        return input;
    }
    return core::wrap_tensor(ggml_cast(ctx.ggml, input.tensor, GGML_TYPE_F32), input.shape, GGML_TYPE_F32);
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    auto contiguous = ensure_contiguous(ctx, input);
    return core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue repeat_kv_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t repeats) {
    if (repeats == 1) {
        return input;
    }
    std::vector<core::TensorValue> heads;
    heads.reserve(static_cast<size_t>(input.shape.dims[1] * repeats));
    for (int64_t head = 0; head < input.shape.dims[1]; ++head) {
        auto one = modules::SliceModule({1, head, 1}).build(ctx, input);
        for (int64_t rep = 0; rep < repeats; ++rep) {
            heads.push_back(one);
        }
    }
    auto output = heads.front();
    for (size_t i = 1; i < heads.size(); ++i) {
        output = modules::ConcatModule({1}).build(ctx, output, heads[i]);
    }
    return output;
}

core::TensorValue attention_from_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const std::optional<core::TensorValue> & attention_mask,
    core::BackendType backend_type) {
    (void)backend_type;
    return modules::ScaledDotProductAttentionModule({
        dim,
        modules::ScaledDotProductAttentionLowering::Explicit,
        GGML_PREC_F32,
    }).build(ctx, q_heads, k_heads, v_heads, attention_mask);
}

core::TensorValue build_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden_states,
    const core::TensorValue & positions,
    const AceStepDiTAttentionWeights & weights,
    const AceStepDiffusionConfig & config,
    const std::optional<core::TensorValue> & attention_mask,
    core::BackendType backend_type,
    const std::optional<PrecomputedCrossAttentionKV> & cross_attention_kv = std::nullopt,
    const std::optional<core::TensorValue> & encoder_hidden_states = std::nullopt) {
    const int64_t dim = ace_step_diffusion_attention_head_dim(config, "ACE-Step diffusion");
    const int64_t kv_repeats = config.num_attention_heads / config.num_key_value_heads;
    const bool is_cross = cross_attention_kv.has_value() || encoder_hidden_states.has_value();
    auto q = modules::LinearModule({config.hidden_size, config.num_attention_heads * dim, false, GGML_PREC_F32})
                 .build(ctx, hidden_states, {weights.q_weight, std::nullopt});
    core::TensorValue k;
    core::TensorValue v;
    if (cross_attention_kv.has_value()) {
        k = cross_attention_kv->key;
        v = cross_attention_kv->value;
    } else {
        const auto & kv_source = is_cross ? *encoder_hidden_states : hidden_states;
        k = modules::LinearModule({config.hidden_size, config.num_key_value_heads * dim, false, GGML_PREC_F32})
                .build(ctx, kv_source, {weights.k_weight, std::nullopt});
        v = modules::LinearModule({config.hidden_size, config.num_key_value_heads * dim, false, GGML_PREC_F32})
                .build(ctx, kv_source, {weights.v_weight, std::nullopt});
    }
    q = modules::RMSNormModule({dim, config.rms_norm_eps, true, false})
            .build(ctx, reshape_heads(ctx, q, config.num_attention_heads, dim), {weights.q_norm, std::nullopt});
    if (!cross_attention_kv.has_value()) {
        k = modules::RMSNormModule({dim, config.rms_norm_eps, true, false})
                .build(ctx, reshape_heads(ctx, k, config.num_key_value_heads, dim), {weights.k_norm, std::nullopt});
        v = reshape_heads(ctx, v, config.num_key_value_heads, dim);
    }
    if (!is_cross) {
        q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, q, positions);
        k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, k, positions);
    }
    auto q_heads = ensure_contiguous(
        ctx,
        modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q));
    auto k_heads = repeat_kv_heads(
        ctx,
        modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k),
        kv_repeats);
    auto v_heads = repeat_kv_heads(
        ctx,
        modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v),
        kv_repeats);
    k_heads = ensure_contiguous(ctx, k_heads);
    v_heads = ensure_contiguous(ctx, v_heads);
    auto context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask, backend_type);
    context = ensure_contiguous(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({hidden_states.shape.dims[0], hidden_states.shape.dims[1], config.hidden_size}));
    return modules::LinearModule({config.hidden_size, config.hidden_size, false, GGML_PREC_F32})
        .build(ctx, context, {weights.out_weight, std::nullopt});
}

core::TensorValue apply_modulated_rms_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & norm_weight,
    const core::TensorValue & one,
    const core::TensorValue & shift,
    const core::TensorValue & scale,
    float rms_eps) {
    auto hidden = modules::RMSNormModule({input.shape.dims[input.shape.rank - 1], rms_eps, true, false})
                      .build(ctx, input, {norm_weight, std::nullopt});
    auto scaled = core::wrap_tensor(ggml_add(ctx.ggml, scale.tensor, one.tensor), scale.shape, GGML_TYPE_F32);
    hidden = modules::MulModule{}.build(ctx, hidden, scaled);
    return modules::AddModule{}.build(ctx, hidden, shift);
}

core::TensorValue expand_conditioning(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & conditioning,
    int64_t seq_len) {
    auto reshaped = core::reshape_tensor(
        ctx,
        conditioning,
        core::TensorShape::from_dims({conditioning.shape.dims[0], 1, conditioning.shape.dims[1]}));
    return modules::RepeatModule({core::TensorShape::from_dims({conditioning.shape.dims[0], seq_len, conditioning.shape.dims[1]})})
        .build(ctx, reshaped);
}

struct TimeEmbeddingOutputs {
    core::TensorValue temb;
    core::TensorValue timestep_proj;
};

core::TensorValue build_time_frequency_tensor(
    core::ModuleBuildContext & ctx,
    ggml_tensor * freqs_tensor,
    const core::TensorValue & timestep) {
    const int64_t batch_size = timestep.shape.dims[0];
    auto freqs = core::wrap_tensor(freqs_tensor, core::TensorShape::from_dims({128}), GGML_TYPE_F32);
    auto expanded_freqs = core::reshape_tensor(ctx, freqs, core::TensorShape::from_dims({1, 128}));
    expanded_freqs = modules::RepeatModule({core::TensorShape::from_dims({batch_size, 128})}).build(ctx, expanded_freqs);
    auto expanded_time = modules::RepeatModule({core::TensorShape::from_dims({batch_size, 128})}).build(ctx, timestep);
    return modules::MulModule{}.build(ctx, expanded_time, expanded_freqs);
}

TimeEmbeddingOutputs build_time_embedding(
    core::ModuleBuildContext & ctx,
    ggml_tensor * freqs_tensor,
    const core::TensorValue & timestep,
    const AceStepTimeEmbeddingWeights & weights,
    int64_t hidden_size) {
    auto scaled_t = core::wrap_tensor(ggml_scale(ctx.ggml, timestep.tensor, 1000.0F), timestep.shape, GGML_TYPE_F32);
    auto args = build_time_frequency_tensor(ctx, freqs_tensor, scaled_t);
    auto cos_part = core::wrap_tensor(ggml_cos(ctx.ggml, args.tensor), args.shape, GGML_TYPE_F32);
    auto sin_part = core::wrap_tensor(ggml_sin(ctx.ggml, args.tensor), args.shape, GGML_TYPE_F32);
    auto embedding = modules::ConcatModule({1}).build(ctx, cos_part, sin_part);
    auto temb = modules::LinearModule({256, hidden_size, true, GGML_PREC_F32}).build(ctx, embedding, weights.fc1);
    temb = modules::SiluModule{}.build(ctx, temb);
    temb = modules::LinearModule({hidden_size, hidden_size, true, GGML_PREC_F32}).build(ctx, temb, weights.fc2);
    auto proj_in = modules::SiluModule{}.build(ctx, temb);
    auto timestep_proj =
        modules::LinearModule({hidden_size, hidden_size * 6, true, GGML_PREC_F32}).build(ctx, proj_in, weights.time_proj);
    timestep_proj = core::reshape_tensor(
        ctx,
        timestep_proj,
        core::TensorShape::from_dims({timestep_proj.shape.dims[0], int64_t{6}, hidden_size}));
    return TimeEmbeddingOutputs{temb, timestep_proj};
}

core::TensorValue build_mlp(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::LinearWeights & gate_proj,
    const modules::LinearWeights & up_proj,
    const modules::LinearWeights & down_proj,
    const AceStepDiffusionConfig & config) {
    auto gate = modules::LinearModule({config.hidden_size, config.intermediate_size, false, GGML_PREC_F32})
                    .build(ctx, input, gate_proj);
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule({config.hidden_size, config.intermediate_size, false, GGML_PREC_F32})
                  .build(ctx, input, up_proj);
    auto ff = modules::MulModule{}.build(ctx, gate, up);
    return modules::LinearModule({config.intermediate_size, config.hidden_size, false, GGML_PREC_F32})
        .build(ctx, ff, down_proj);
}

core::TensorValue slice_conditioning_plane(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & conditioning,
    int64_t plane,
    int64_t hidden_size,
    int64_t seq_len) {
    auto slice = modules::SliceModule({1, plane, 1}).build(ctx, conditioning);
    slice = ensure_contiguous(ctx, slice);
    slice = core::reshape_tensor(
        ctx,
        slice,
        core::TensorShape::from_dims({conditioning.shape.dims[0], hidden_size}));
    return expand_conditioning(ctx, slice, seq_len);
}

core::TensorValue dit_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden_states,
    const core::TensorValue & positions,
    const core::TensorValue & timestep_proj,
    const core::TensorValue & one,
    const std::optional<core::TensorValue> & self_attention_mask,
    const core::TensorValue & encoder_attention_mask,
    const AceStepDiTLayerWeights & weights,
    const AceStepDiffusionConfig & config,
    core::BackendType backend_type,
    const std::optional<PrecomputedCrossAttentionKV> & cross_attention_kv,
    const std::optional<core::TensorValue> & encoder_hidden_states = std::nullopt) {
    const int64_t hidden_size = config.hidden_size;
    const int64_t seq_len = hidden_states.shape.dims[1];
    auto scale_shift_table = modules::RepeatModule({timestep_proj.shape}).build(ctx, ensure_f32(ctx, weights.scale_shift_table));
    auto modulation = modules::AddModule{}.build(ctx, scale_shift_table, timestep_proj);
    auto shift_msa = slice_conditioning_plane(ctx, modulation, 0, hidden_size, seq_len);
    auto scale_msa = slice_conditioning_plane(ctx, modulation, 1, hidden_size, seq_len);
    auto gate_msa = slice_conditioning_plane(ctx, modulation, 2, hidden_size, seq_len);
    auto c_shift_msa = slice_conditioning_plane(ctx, modulation, 3, hidden_size, seq_len);
    auto c_scale_msa = slice_conditioning_plane(ctx, modulation, 4, hidden_size, seq_len);
    auto c_gate_msa = slice_conditioning_plane(ctx, modulation, 5, hidden_size, seq_len);

    auto norm_hidden = apply_modulated_rms_norm(
        ctx,
        hidden_states,
        weights.self_attn_norm,
        one,
        shift_msa,
        scale_msa,
        config.rms_norm_eps);
    auto attn_out = build_attention(
        ctx,
        norm_hidden,
        positions,
        weights.self_attn,
        config,
        self_attention_mask,
        backend_type);
    attn_out = modules::MulModule{}.build(ctx, attn_out, gate_msa);
    auto x = modules::AddModule{}.build(ctx, hidden_states, attn_out);

    auto cross_norm = modules::RMSNormModule({hidden_size, config.rms_norm_eps, true, false})
                          .build(ctx, x, {weights.cross_attn_norm, std::nullopt});
    auto cross_out = build_attention(
        ctx,
        cross_norm,
        positions,
        weights.cross_attn,
        config,
        encoder_attention_mask,
        backend_type,
        cross_attention_kv,
        encoder_hidden_states);
    x = modules::AddModule{}.build(ctx, x, cross_out);

    auto mlp_norm = apply_modulated_rms_norm(
        ctx,
        x,
        weights.mlp_norm,
        one,
        c_shift_msa,
        c_scale_msa,
        config.rms_norm_eps);
    auto ff = build_mlp(ctx, mlp_norm, weights.mlp_gate, weights.mlp_up, weights.mlp_down, config);
    ff = modules::MulModule{}.build(ctx, ff, c_gate_msa);
    return modules::AddModule{}.build(ctx, x, ff);
}

std::vector<float> build_cross_attention_mask_values(
    int64_t seq_len,
    const std::vector<int32_t> & encoder_attention_mask,
    int64_t valid_seq_len) {
    const int64_t encoder_tokens = static_cast<int64_t>(encoder_attention_mask.size());
    if (seq_len <= 0 || encoder_tokens <= 0 || valid_seq_len <= 0 || valid_seq_len > seq_len) {
        throw std::runtime_error("ACE-Step diffusion cross attention mask shape is invalid");
    }
    std::vector<float> values(static_cast<size_t>(seq_len * encoder_tokens), 0.0F);
    const float masked = std::numeric_limits<float>::lowest();
    for (int64_t q = 0; q < seq_len; ++q) {
        for (int64_t k = 0; k < encoder_tokens; ++k) {
            if (q >= valid_seq_len) {
                values[static_cast<size_t>(q * encoder_tokens + k)] = masked;
            }
        }
    }
    return values;
}

std::vector<float> build_sliding_mask_values(int64_t tokens, int64_t sliding_window) {
    std::vector<float> values(static_cast<size_t>(tokens * tokens), 0.0F);
    const float masked = std::numeric_limits<float>::lowest();
    for (int64_t q = 0; q < tokens; ++q) {
        for (int64_t k = 0; k < tokens; ++k) {
            if (std::llabs(q - k) > sliding_window) {
                values[static_cast<size_t>(q * tokens + k)] = masked;
            }
        }
    }
    return values;
}

std::vector<float> build_self_attention_padding_mask_values(int64_t seq_len, int64_t valid_seq_len) {
    if (seq_len <= 0 || valid_seq_len <= 0 || valid_seq_len > seq_len) {
        throw std::runtime_error("ACE-Step diffusion self attention padding mask is invalid");
    }
    std::vector<float> values(static_cast<size_t>(seq_len * seq_len), 0.0F);
    const float masked = std::numeric_limits<float>::lowest();
    for (int64_t q = 0; q < valid_seq_len; ++q) {
        for (int64_t k = valid_seq_len; k < seq_len; ++k) {
            values[static_cast<size_t>(q * seq_len + k)] = masked;
        }
    }
    return values;
}

std::vector<float> make_timestep_freqs() {
    std::vector<float> freqs(128, 0.0F);
    for (int64_t i = 0; i < 128; ++i) {
        freqs[static_cast<size_t>(i)] =
            std::exp(-std::log(10000.0F) * static_cast<float>(i) / 128.0F);
    }
    return freqs;
}

std::vector<float> valid_timesteps() {
    return {
        1.0F, 0.95454544F, 0.93333334F, 0.9F, 0.875F,
        0.85714287F, 0.83333331F, 0.76923078F, 0.75F,
        0.66666669F, 0.64285713F, 0.625F, 0.54545456F,
        0.5F, 0.4F, 0.375F, 0.3F, 0.25F, 0.22222222F, 0.125F
    };
}

std::vector<float> shift_schedule(float shift) {
    if (shift == 1.0F) {
        return {1.0F, 0.875F, 0.75F, 0.625F, 0.5F, 0.375F, 0.25F, 0.125F};
    }
    if (shift == 2.0F) {
        return {1.0F, 0.93333334F, 0.85714287F, 0.76923078F, 0.66666669F, 0.54545456F, 0.4F, 0.22222222F};
    }
    return {1.0F, 0.95454544F, 0.9F, 0.83333331F, 0.75F, 0.64285713F, 0.5F, 0.3F};
}

float nearest_valid_shift(float shift) {
    constexpr float kValid[] = {1.0F, 2.0F, 3.0F};
    float best = kValid[0];
    float best_distance = std::abs(shift - best);
    for (float candidate : kValid) {
        const float distance = std::abs(shift - candidate);
        if (distance < best_distance) {
            best = candidate;
            best_distance = distance;
        }
    }
    return best;
}

float nearest_valid_timestep(float t, const std::vector<float> & valid) {
    float best = valid.front();
    float best_distance = std::abs(t - best);
    for (float candidate : valid) {
        const float distance = std::abs(t - candidate);
        if (distance < best_distance) {
            best = candidate;
            best_distance = distance;
        }
    }
    return best;
}

std::vector<float> make_python_schedule(const AceStepGenerationOptions & options, bool is_turbo) {
    std::vector<float> schedule;
    if (!options.timesteps.empty()) {
        schedule = options.timesteps;
        while (!schedule.empty() && schedule.back() == 0.0F) {
            schedule.pop_back();
        }
        if (is_turbo && schedule.size() > 20) {
            schedule.resize(20);
        }
        if (!schedule.empty() && is_turbo) {
            const auto valid = valid_timesteps();
            for (float & t : schedule) {
                t = nearest_valid_timestep(t, valid);
            }
            return schedule;
        }
        if (!schedule.empty()) {
            return schedule;
        }
    }

    if (options.num_inference_steps > 0) {
        const int64_t n = is_turbo ? std::min<int64_t>(options.num_inference_steps, 20) : options.num_inference_steps;
        schedule.reserve(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            float t = 1.0F - static_cast<float>(i) / static_cast<float>(n);
            if (options.shift != 1.0F) {
                t = options.shift * t / (1.0F + (options.shift - 1.0F) * t);
            }
            schedule.push_back(t);
        }
        return schedule;
    }

    return shift_schedule(nearest_valid_shift(options.shift));
}

std::vector<float> gaussian_noise(size_t count, uint32_t seed, uint64_t start_index = 0) {
    return engine::sampling::generate_torch_cuda_randn(
        count,
        seed,
        engine::sampling::TorchRandnPrecision::Float32,
        start_index);
}

std::vector<float> renoise(
    const std::vector<float> & x,
    float t,
    uint32_t seed,
    uint64_t & noise_offset,
    const std::vector<float> * noise_override = nullptr) {
    std::vector<float> sampled_noise;
    const std::vector<float> * noise = noise_override;
    if (noise == nullptr) {
        sampled_noise = gaussian_noise(x.size(), seed, noise_offset);
        noise_offset += static_cast<uint64_t>(x.size());
        noise = &sampled_noise;
    }
    if (noise->size() != x.size()) {
        throw std::runtime_error("ACE-Step diffusion renoise input/noise size mismatch");
    }
    return engine::sampling::renoise(x, *noise, t);
}

void apply_velocity_norm_clamp(
    std::vector<float> & velocity,
    const std::vector<float> & hidden,
    int64_t frames,
    int64_t channels,
    float threshold) {
    const size_t count = static_cast<size_t>(frames * channels);
    if (velocity.size() != count || hidden.size() != count) {
        throw std::runtime_error("ACE-Step diffusion velocity clamp shape mismatch");
    }
    engine::sampling::clamp_velocity_norm(velocity, hidden, threshold);
}

void apply_velocity_ema(
    std::vector<float> & velocity,
    const std::vector<float> & previous_velocity,
    float factor) {
    if (factor <= 0.0F || previous_velocity.empty()) {
        return;
    }
    for (size_t i = 0; i < velocity.size(); ++i) {
        velocity[i] = (1.0F - factor) * velocity[i] + factor * previous_velocity[i];
    }
}

bool cfg_interval_active(float timestep, const AceStepGenerationOptions & options) {
    return engine::sampling::timestep_in_interval(timestep, options.cfg_interval_start, options.cfg_interval_end);
}

int64_t python_round_nonnegative(float value) {
    if (value < 0.0F) {
        throw std::runtime_error("ACE-Step diffusion expected a nonnegative value for Python-style rounding");
    }
    const double x = static_cast<double>(value);
    const double floor_value = std::floor(x);
    const double fraction = x - floor_value;
    int64_t rounded = static_cast<int64_t>(floor_value);
    if (fraction > 0.5) {
        ++rounded;
    } else if (fraction == 0.5 && (rounded % 2) != 0) {
        ++rounded;
    }
    return rounded;
}

std::vector<float> apply_cfg_guidance(
    const std::vector<float> & pred_cond,
    const std::vector<float> & pred_uncond,
    float guidance_scale) {
    return engine::sampling::cfg_guidance(pred_cond, pred_uncond, guidance_scale);
}

std::vector<float> apply_apg_guidance(
    const std::vector<float> & pred_cond,
    const std::vector<float> & pred_uncond,
    float guidance_scale,
    int64_t frames,
    int64_t channels,
    std::vector<float> & momentum) {
    return engine::sampling::apg_guidance(pred_cond, pred_uncond, guidance_scale, frames, channels, momentum);
}

std::vector<float> apply_adg_guidance(
    const std::vector<float> & latents,
    const std::vector<float> & pred_cond,
    const std::vector<float> & pred_uncond,
    float sigma,
    float guidance_scale,
    int64_t frames,
    int64_t channels) {
    return engine::sampling::adg_guidance(latents, pred_cond, pred_uncond, sigma, guidance_scale, frames, channels);
}

void subtract_velocity_step(std::vector<float> & hidden, const std::vector<float> & velocity, float step) {
    engine::sampling::euler_step_in_place(hidden, velocity, step);
}

void write_x0_prefix(std::vector<float> & hidden, const std::vector<float> & velocity, float t) {
    engine::sampling::euler_step_in_place(hidden, velocity, t);
}

std::vector<float> pad_latent_values(const AceStepLatents & latents, int64_t padded_frames) {
    if (latents.frames <= 0 || latents.channels <= 0 || padded_frames < latents.frames) {
        throw std::runtime_error("ACE-Step diffusion latent padding request is invalid");
    }
    std::vector<float> padded(static_cast<size_t>(padded_frames * latents.channels), 0.0F);
    std::copy(latents.values.begin(), latents.values.end(), padded.begin());
    return padded;
}

std::vector<float> pad_context_values(const AceStepLatents & latents, int64_t padded_frames, int64_t expected_channels) {
    if (latents.channels != expected_channels) {
        throw std::runtime_error("ACE-Step diffusion context latent channel count mismatch");
    }
    return pad_latent_values(latents, padded_frames);
}

std::vector<float> pad_encoder_hidden_values(
    const AceStepEncoderConditioning & conditioning,
    int64_t padded_tokens,
    int64_t hidden_size) {
    if (conditioning.hidden_size != hidden_size || padded_tokens < conditioning.tokens) {
        throw std::runtime_error("ACE-Step diffusion encoder hidden padding request is invalid");
    }
    std::vector<float> padded(static_cast<size_t>(padded_tokens * hidden_size), 0.0F);
    std::copy(conditioning.values.begin(), conditioning.values.end(), padded.begin());
    return padded;
}

std::vector<float> null_encoder_hidden_values(
    const std::vector<float> & null_condition_emb,
    int64_t padded_tokens,
    int64_t hidden_size) {
    if (padded_tokens <= 0 || hidden_size <= 0 ||
        static_cast<int64_t>(null_condition_emb.size()) != hidden_size) {
        throw std::runtime_error("ACE-Step diffusion null condition embedding shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(padded_tokens * hidden_size), 0.0F);
    for (int64_t token = 0; token < padded_tokens; ++token) {
        std::copy(
            null_condition_emb.begin(),
            null_condition_emb.end(),
            out.begin() + static_cast<std::ptrdiff_t>(token * hidden_size));
    }
    return out;
}

std::vector<float> duplicate_batch_values(const std::vector<float> & values, int64_t batch_size) {
    if (batch_size <= 0) {
        throw std::runtime_error("ACE-Step diffusion batch size is invalid");
    }
    if (batch_size == 1) {
        return values;
    }
    std::vector<float> out;
    out.reserve(values.size() * static_cast<size_t>(batch_size));
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        out.insert(out.end(), values.begin(), values.end());
    }
    return out;
}

std::vector<int32_t> pad_encoder_attention_mask(
    const AceStepEncoderConditioning & conditioning,
    int64_t padded_tokens) {
    if (static_cast<int64_t>(conditioning.attention_mask.size()) != conditioning.tokens || padded_tokens < conditioning.tokens) {
        throw std::runtime_error("ACE-Step diffusion encoder attention mask padding request is invalid");
    }
    std::vector<int32_t> padded(static_cast<size_t>(padded_tokens), 0);
    std::copy(conditioning.attention_mask.begin(), conditioning.attention_mask.end(), padded.begin());
    return padded;
}

void apply_repaint_step_injection_in_place(
    std::vector<float> & hidden,
    const std::vector<float> & clean_src_latents,
    const std::vector<int32_t> & repaint_mask,
    float next_timestep,
    const std::vector<float> & noise,
    int64_t channels) {
    if (hidden.size() != clean_src_latents.size() || hidden.size() != noise.size()) {
        throw std::runtime_error("ACE-Step repaint injection latent shape mismatch");
    }
    if (channels <= 0 ||
        hidden.size() % static_cast<size_t>(channels) != 0 ||
        repaint_mask.size() != hidden.size() / static_cast<size_t>(channels)) {
        throw std::runtime_error("ACE-Step repaint injection mask shape mismatch");
    }
    for (size_t frame = 0; frame < repaint_mask.size(); ++frame) {
        if (repaint_mask[frame] != 0) {
            continue;
        }
        const size_t offset = frame * static_cast<size_t>(channels);
        for (int64_t channel = 0; channel < channels; ++channel) {
            const size_t index = offset + static_cast<size_t>(channel);
            hidden[index] = next_timestep * noise[index] + (1.0F - next_timestep) * clean_src_latents[index];
        }
    }
}

std::vector<float> build_soft_repaint_mask(const std::vector<int32_t> & repaint_mask, int64_t crossfade_frames) {
    return engine::sampling::build_soft_mask(repaint_mask, crossfade_frames);
}

void apply_repaint_boundary_blend(
    std::vector<float> & generated,
    const std::vector<float> & clean_src_latents,
    const std::vector<int32_t> & repaint_mask,
    int64_t crossfade_frames,
    int64_t channels) {
    const auto soft_mask = build_soft_repaint_mask(repaint_mask, crossfade_frames);
    engine::sampling::blend_by_mask_in_place(generated, clean_src_latents, soft_mask, channels);
}

bool dcw_mode_is_valid(const std::string & mode) {
    return mode == "low" || mode == "high" || mode == "double" || mode == "pix";
}

bool dcw_is_active(const AceStepGenerationOptions & options, bool reference_skips_wavelet_dcw) {
    if (!dcw_mode_is_valid(options.dcw_mode)) {
        throw std::runtime_error("ACE-Step diffusion dcw_mode is invalid");
    }
    if (!options.dcw_enabled) {
        return false;
    }
    if (reference_skips_wavelet_dcw && options.dcw_mode != "pix") {
        // In the validated index-tts reference environment, the 2B Base path
        // logs wavelet DCW as unavailable and leaves the latent unchanged.
        return false;
    }
    if (options.dcw_mode == "double") {
        return options.dcw_scaler != 0.0F || options.dcw_high_scaler != 0.0F;
    }
    return options.dcw_scaler != 0.0F;
}

struct WaveletSpec {
    const char * name = nullptr;
    std::vector<float> dec_lo;
    std::vector<float> dec_hi;
};

const WaveletSpec & lookup_wavelet_spec(const std::string & name) {
    static const std::vector<WaveletSpec> kSpecs = {
        {
            "haar",
            {0.7071067811865476F, 0.7071067811865476F},
            {-0.7071067811865476F, 0.7071067811865476F},
        },
        {
            "db2",
            {-0.12940952255126037F, 0.2241438680420134F, 0.8365163037378079F, 0.48296291314453416F},
            {-0.48296291314453416F, 0.8365163037378079F, -0.2241438680420134F, -0.12940952255126037F},
        },
        {
            "db4",
            {-0.010597401785069032F, 0.0328830116668852F, 0.030841381835560764F, -0.18703481171909309F,
             -0.027983769416859854F, 0.6308807679298589F, 0.7148465705529157F, 0.2303778133088965F},
            {-0.2303778133088965F, 0.7148465705529157F, -0.6308807679298589F, -0.027983769416859854F,
             0.18703481171909309F, 0.030841381835560764F, -0.0328830116668852F, -0.010597401785069032F},
        },
        {
            "sym4",
            {-0.07576571478927333F, -0.02963552764599851F, 0.49761866763201545F, 0.8037387518059161F,
             0.29785779560527736F, -0.09921954357684722F, -0.012603967262037833F, 0.0322231006040427F},
            {-0.0322231006040427F, -0.012603967262037833F, 0.09921954357684722F, 0.29785779560527736F,
             -0.8037387518059161F, 0.49761866763201545F, 0.02963552764599851F, -0.07576571478927333F},
        },
        {
            "sym8",
            {-0.0033824159510061256F, -0.0005421323317911481F, 0.03169508781149298F, 0.007607487324917605F,
             -0.1432942383508097F, -0.061273359067658524F, 0.4813596512583722F, 0.7771857517005235F,
             0.3644418948353314F, -0.05194583810770904F, -0.027219029917056003F, 0.049137179673607506F,
             0.003808752013890615F, -0.01495225833704823F, -0.0003029205147213668F, 0.0018899503327594609F},
            {-0.0018899503327594609F, -0.0003029205147213668F, 0.01495225833704823F, 0.003808752013890615F,
             -0.049137179673607506F, -0.027219029917056003F, 0.05194583810770904F, 0.3644418948353314F,
             -0.7771857517005235F, 0.4813596512583722F, 0.061273359067658524F, -0.1432942383508097F,
             -0.007607487324917605F, 0.03169508781149298F, 0.0005421323317911481F, -0.0033824159510061256F},
        },
        {
            "coif2",
            {-0.000720549445520347F, -0.0018232088709110323F, 0.005611434819368834F, 0.02368017194684777F,
             -0.05943441864643109F, -0.07648859907828076F, 0.4170051844232391F, 0.8127236354494135F,
             0.3861100668227629F, -0.0673725547237256F, -0.04146493678687178F, 0.01638733646320364F},
            {-0.01638733646320364F, -0.04146493678687178F, 0.0673725547237256F, 0.3861100668227629F,
             -0.8127236354494135F, 0.4170051844232391F, 0.07648859907828076F, -0.05943441864643109F,
             -0.02368017194684777F, 0.005611434819368834F, 0.0018232088709110323F, -0.000720549445520347F},
        },
    };
    for (const auto & spec : kSpecs) {
        if (name == spec.name) {
            return spec;
        }
    }
    throw std::runtime_error("ACE-Step diffusion dcw_wavelet is unsupported");
}

struct WaveletBands {
    std::vector<float> low;
    std::vector<float> high;
    const WaveletSpec * spec = nullptr;
    int64_t frames = 0;
    int64_t channels = 0;
};

WaveletBands wavelet_dwt_prefix(
    const std::vector<float> & values,
    int64_t frames,
    int64_t channels,
    const WaveletSpec & spec) {
    const int64_t filter_size = static_cast<int64_t>(spec.dec_lo.size());
    const int64_t band_frames = (frames + filter_size - 1) / 2;
    WaveletBands bands;
    bands.low.assign(static_cast<size_t>(band_frames * channels), 0.0F);
    bands.high.assign(static_cast<size_t>(band_frames * channels), 0.0F);
    bands.spec = &spec;
    bands.frames = frames;
    bands.channels = channels;
    for (int64_t pair = 0; pair < band_frames; ++pair) {
        for (int64_t channel = 0; channel < channels; ++channel) {
            const size_t index = static_cast<size_t>(pair * channels + channel);
            float low = 0.0F;
            float high = 0.0F;
            for (int64_t k = 0; k < filter_size; ++k) {
                const int64_t src_index = 2 * pair + 1 - k;
                if (src_index < 0 || src_index >= frames) {
                    continue;
                }
                const float value = values[static_cast<size_t>(src_index * channels + channel)];
                low += spec.dec_lo[static_cast<size_t>(k)] * value;
                high += spec.dec_hi[static_cast<size_t>(k)] * value;
            }
            bands.low[index] = low;
            bands.high[index] = high;
        }
    }
    return bands;
}

std::vector<float> wavelet_idwt_prefix(const WaveletBands & bands) {
    if (bands.spec == nullptr) {
        throw std::runtime_error("ACE-Step diffusion wavelet IDWT requires valid filter spec");
    }
    const int64_t filter_size = static_cast<int64_t>(bands.spec->dec_lo.size());
    const int64_t band_frames = static_cast<int64_t>(bands.low.size() / bands.channels);
    std::vector<float> out(static_cast<size_t>(bands.frames * bands.channels), 0.0F);
    for (int64_t pair = 0; pair < band_frames; ++pair) {
        for (int64_t channel = 0; channel < bands.channels; ++channel) {
            const size_t index = static_cast<size_t>(pair * bands.channels + channel);
            for (int64_t k = 0; k < filter_size; ++k) {
                const int64_t dst_index = 2 * pair + 1 - k;
                if (dst_index < 0 || dst_index >= bands.frames) {
                    continue;
                }
                out[static_cast<size_t>(dst_index * bands.channels + channel)] +=
                    bands.low[index] * bands.spec->dec_lo[static_cast<size_t>(k)] +
                    bands.high[index] * bands.spec->dec_hi[static_cast<size_t>(k)];
            }
        }
    }
    return out;
}

void apply_dcw_correction(
    std::vector<float> & hidden,
    const std::vector<float> & denoised,
    int64_t frames,
    int64_t channels,
    float t_curr,
    bool reference_skips_wavelet_dcw,
    const AceStepGenerationOptions & options) {
    if (!dcw_is_active(options, reference_skips_wavelet_dcw)) {
        return;
    }
    if (hidden.size() < denoised.size()) {
        throw std::runtime_error("ACE-Step diffusion DCW hidden/denoised size mismatch");
    }
    if (options.dcw_mode == "pix") {
        if (options.dcw_scaler == 0.0F) {
            return;
        }
        for (size_t i = 0; i < denoised.size(); ++i) {
            hidden[i] = hidden[i] + options.dcw_scaler * (hidden[i] - denoised[i]);
        }
        return;
    }
    const auto & spec = lookup_wavelet_spec(options.dcw_wavelet);
    auto x_bands = wavelet_dwt_prefix(hidden, frames, channels, spec);
    const auto y_bands = wavelet_dwt_prefix(denoised, frames, channels, spec);
    const float low_scaler = t_curr * options.dcw_scaler;
    const float high_scaler = (1.0F - t_curr) *
        (options.dcw_mode == "double" ? options.dcw_high_scaler : options.dcw_scaler);
    if (options.dcw_mode == "low" || options.dcw_mode == "double") {
        for (size_t i = 0; i < x_bands.low.size(); ++i) {
            x_bands.low[i] = x_bands.low[i] + low_scaler * (x_bands.low[i] - y_bands.low[i]);
        }
    }
    if (options.dcw_mode == "high" || options.dcw_mode == "double") {
        for (size_t i = 0; i < x_bands.high.size(); ++i) {
            x_bands.high[i] = x_bands.high[i] + high_scaler * (x_bands.high[i] - y_bands.high[i]);
        }
    }
    const auto corrected = wavelet_idwt_prefix(x_bands);
    std::copy(corrected.begin(), corrected.end(), hidden.begin());
}

std::vector<float> load_noise_or_sample(const std::string & noise_file, size_t count, uint32_t seed) {
    if (noise_file.empty()) {
        return gaussian_noise(count, seed);
    }
    auto values = io::read_f32_file(noise_file);
    if (values.size() != count) {
        throw std::runtime_error(
            "ACE-Step diffusion noise file size mismatch: expected " +
            std::to_string(count) + " floats, got " + std::to_string(values.size()));
    }
    return values;
}

struct NoiseSchedule {
    std::vector<float> initial_noise;
    std::vector<std::vector<float>> renoise_noises;
};

NoiseSchedule load_noise_schedule_or_sample(const std::string & noise_file, size_t count, uint32_t seed) {
    NoiseSchedule schedule;
    if (noise_file.empty()) {
        schedule.initial_noise = gaussian_noise(count, seed);
        return schedule;
    }
    auto values = io::read_f32_file(noise_file);
    if (values.empty() || values.size() % count != 0) {
        throw std::runtime_error(
            "ACE-Step diffusion noise file size mismatch: expected a positive multiple of " +
            std::to_string(count) + " floats, got " + std::to_string(values.size()));
    }
    const size_t chunks = values.size() / count;
    schedule.initial_noise.assign(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(count));
    schedule.renoise_noises.reserve(chunks > 0 ? chunks - 1 : 0);
    for (size_t chunk = 1; chunk < chunks; ++chunk) {
        const auto begin = values.begin() + static_cast<std::ptrdiff_t>(chunk * count);
        schedule.renoise_noises.emplace_back(begin, begin + static_cast<std::ptrdiff_t>(count));
    }
    return schedule;
}

}  // namespace

class AceStepDiffusionRuntime::Impl {
public:
    class CrossAttentionCacheGraph {
    public:
        CrossAttentionCacheGraph(
            std::shared_ptr<const AceStepAssets> assets,
            ggml_backend_t backend,
            core::BackendType backend_type,
            int threads,
            std::shared_ptr<const AceStepDiffusionWeights> weights,
            int64_t encoder_tokens_capacity,
            size_t graph_arena_bytes)
            : assets_(std::move(assets)),
              backend_(backend),
              backend_type_(backend_type),
              threads_(threads),
              weights_(std::move(weights)),
              encoder_tokens_(encoder_tokens_capacity) {
            build(graph_arena_bytes);
        }

        ~CrossAttentionCacheGraph() {
            if (backend_ != nullptr && graph_ != nullptr) {
                engine::core::release_backend_graph_resources(backend_, graph_);
            }
            if (buffer_ != nullptr) {
                ggml_backend_buffer_free(buffer_);
            }
        }

        bool can_run(int64_t encoder_tokens) const noexcept {
            return encoder_tokens <= encoder_tokens_;
        }

        int64_t encoder_token_capacity() const noexcept { return encoder_tokens_; }

        struct Output {
            std::vector<std::vector<float>> keys;
            std::vector<std::vector<float>> values;
        };

        Output run(const std::vector<float> & encoder_hidden_states) const {
            const auto & config = assets_->config.diffusion;
            if (static_cast<int64_t>(encoder_hidden_states.size()) != encoder_tokens_ * config.hidden_size) {
                throw std::runtime_error("ACE-Step diffusion cross-attention cache encoder hidden state shape mismatch");
            }
            core::write_tensor_f32(encoder_value_, encoder_hidden_states);
            core::set_backend_threads(backend_, threads_);
            const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
            ggml_backend_synchronize(backend_);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("ACE-Step diffusion cross-attention cache graph compute failed");
            }

            Output out;
            out.keys.resize(key_outputs_.size());
            out.values.resize(value_outputs_.size());
            const size_t values_per_layer =
                static_cast<size_t>(encoder_tokens_ * config.num_key_value_heads * config.head_dim);
            for (size_t layer = 0; layer < key_outputs_.size(); ++layer) {
                out.keys[layer].resize(values_per_layer);
                out.values[layer].resize(values_per_layer);
                core::read_tensor_f32_into(key_outputs_[layer], out.keys[layer]);
                core::read_tensor_f32_into(value_outputs_[layer], out.values[layer]);
            }
            return out;
        }

    private:
        void build(size_t graph_arena_bytes) {
            const auto & config = assets_->config.diffusion;
            ggml_init_params params{graph_arena_bytes, nullptr, true};
            ctx_.reset(ggml_init(params));
            if (ctx_ == nullptr) {
                throw std::runtime_error("ACE-Step diffusion cross-attention cache ggml context initialization failed");
            }
            core::ModuleBuildContext ctx{ctx_.get(), "ace_step.diffusion.cross_cache", backend_type_};

            encoder_value_ = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, encoder_tokens_, config.hidden_size}));
            auto encoder_hidden_states = modules::LinearModule({config.hidden_size, config.hidden_size, true, GGML_PREC_F32})
                                             .build(ctx, encoder_value_, weights_->condition_embedder);
            key_outputs_.reserve(weights_->layers.size());
            value_outputs_.reserve(weights_->layers.size());
            for (const auto & layer : weights_->layers) {
                auto k = modules::LinearModule(
                             {config.hidden_size, config.num_key_value_heads * config.head_dim, false, GGML_PREC_F32})
                             .build(ctx, encoder_hidden_states, {layer.cross_attn.k_weight, std::nullopt});
                auto v = modules::LinearModule(
                             {config.hidden_size, config.num_key_value_heads * config.head_dim, false, GGML_PREC_F32})
                             .build(ctx, encoder_hidden_states, {layer.cross_attn.v_weight, std::nullopt});
                k = modules::RMSNormModule({config.head_dim, config.rms_norm_eps, true, false})
                        .build(ctx, reshape_heads(ctx, k, config.num_key_value_heads, config.head_dim), {layer.cross_attn.k_norm, std::nullopt});
                v = reshape_heads(ctx, v, config.num_key_value_heads, config.head_dim);
                key_outputs_.push_back(k.tensor);
                value_outputs_.push_back(v.tensor);
                ggml_set_output(key_outputs_.back());
                ggml_set_output(value_outputs_.back());
            }

            graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
            for (size_t layer = 0; layer < key_outputs_.size(); ++layer) {
                ggml_build_forward_expand(graph_, key_outputs_[layer]);
                ggml_build_forward_expand(graph_, value_outputs_[layer]);
            }
            buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_);
            if (buffer_ == nullptr) {
                throw std::runtime_error("ACE-Step diffusion cross-attention cache backend buffer allocation failed");
            }
        }

        std::shared_ptr<const AceStepAssets> assets_;
        ggml_backend_t backend_ = nullptr;
        core::BackendType backend_type_ = core::BackendType::Cpu;
        int threads_ = 1;
        std::shared_ptr<const AceStepDiffusionWeights> weights_;
        int64_t encoder_tokens_ = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
        core::TensorValue encoder_value_;
        std::vector<ggml_tensor *> key_outputs_;
        std::vector<ggml_tensor *> value_outputs_;
        ggml_cgraph * graph_ = nullptr;
        ggml_backend_buffer_t buffer_ = nullptr;
    };

    class StepGraph {
    public:
        StepGraph(
            std::shared_ptr<const AceStepAssets> assets,
            ggml_backend_t backend,
            core::BackendType backend_type,
            int threads,
            std::shared_ptr<const AceStepDiffusionWeights> weights,
            int64_t batch_size,
            int64_t original_frames_capacity,
            int64_t encoder_tokens_capacity,
            size_t graph_arena_bytes)
            : assets_(std::move(assets)),
              backend_(backend),
              backend_type_(backend_type),
              threads_(threads),
              weights_(std::move(weights)),
              batch_size_(batch_size),
              original_frames_capacity_(original_frames_capacity),
              padded_frames_(round_up_frames(original_frames_capacity, assets_->config.diffusion.patch_size)),
              patch_frames_(padded_frames_ / assets_->config.diffusion.patch_size),
              encoder_tokens_(encoder_tokens_capacity) {
            if (batch_size_ <= 0) {
                throw std::runtime_error("ACE-Step diffusion graph batch size is invalid");
            }
            build(graph_arena_bytes);
        }

        ~StepGraph() {
            if (backend_ != nullptr && graph_ != nullptr) {
                engine::core::release_backend_graph_resources(backend_, graph_);
            }
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
            }
        }

        bool can_run(int64_t original_frames, int64_t encoder_tokens) const noexcept {
            return original_frames <= original_frames_capacity_ && encoder_tokens <= encoder_tokens_;
        }

        void set_static_inputs(
            int64_t original_frames,
            const std::vector<float> & context_latents,
            const CrossAttentionCacheGraph::Output & cross_attention_cache,
            const std::vector<int32_t> & encoder_attention_mask) const {
            const auto & config = assets_->config.diffusion;
            const int64_t request_padded_frames = round_up_frames(original_frames, config.patch_size);
            const int64_t request_patch_frames = request_padded_frames / config.patch_size;
            if (original_frames <= 0 || original_frames > original_frames_capacity_) {
                throw std::runtime_error("ACE-Step diffusion request frame count exceeds graph capacity");
            }
            if (static_cast<int64_t>(context_latents.size()) !=
                batch_size_ * padded_frames_ * config.latent_channels * 2) {
                throw std::runtime_error("ACE-Step diffusion context latent shape mismatch");
            }
            if (static_cast<int64_t>(encoder_attention_mask.size()) != encoder_tokens_) {
                throw std::runtime_error("ACE-Step diffusion encoder attention mask shape mismatch");
            }
            if (cross_attention_cache.keys.size() != weights_->layers.size() ||
                cross_attention_cache.values.size() != weights_->layers.size()) {
                throw std::runtime_error("ACE-Step diffusion cross-attention cache layer count mismatch");
            }
            const size_t cross_values_per_layer = static_cast<size_t>(
                batch_size_ * encoder_tokens_ * config.num_key_value_heads * config.head_dim);
            core::write_tensor_f32(context_value_, context_latents);
            core::write_tensor_f32(
                encoder_attention_mask_value_,
                duplicate_batch_values(
                    build_cross_attention_mask_values(
                        patch_frames_,
                        encoder_attention_mask,
                        request_patch_frames),
                    batch_size_));
            core::write_tensor_f32(
                self_attention_padding_mask_value_,
                duplicate_batch_values(
                    build_self_attention_padding_mask_values(patch_frames_, request_patch_frames),
                    batch_size_));
            core::write_tensor_f32(
                sliding_attention_mask_value_,
                duplicate_batch_values(
                    build_sliding_mask_values(patch_frames_, config.sliding_window),
                    batch_size_));
            for (size_t layer = 0; layer < cross_key_value_inputs_.size(); ++layer) {
                if (cross_attention_cache.keys[layer].size() != cross_values_per_layer ||
                    cross_attention_cache.values[layer].size() != cross_values_per_layer) {
                    throw std::runtime_error("ACE-Step diffusion cross-attention cache tensor shape mismatch");
                }
                core::write_tensor_f32(cross_key_value_inputs_[layer], cross_attention_cache.keys[layer]);
                core::write_tensor_f32(cross_value_value_inputs_[layer], cross_attention_cache.values[layer]);
            }
        }

        std::vector<float> run_step(
            int64_t original_frames,
            const std::vector<float> & hidden_states,
            float timestep,
            float timestep_r) const {
            const auto & config = assets_->config.diffusion;
            if (original_frames <= 0 || original_frames > original_frames_capacity_) {
                throw std::runtime_error("ACE-Step diffusion request frame count exceeds graph capacity");
            }
            if (static_cast<int64_t>(hidden_states.size()) != batch_size_ * original_frames * config.latent_channels) {
                throw std::runtime_error("ACE-Step diffusion hidden state shape mismatch");
            }
            std::vector<float> padded_hidden(static_cast<size_t>(batch_size_ * padded_frames_ * config.latent_channels), 0.0F);
            const size_t input_batch_values = static_cast<size_t>(original_frames * config.latent_channels);
            const size_t padded_batch_values = static_cast<size_t>(padded_frames_ * config.latent_channels);
            for (int64_t batch = 0; batch < batch_size_; ++batch) {
                std::copy(
                    hidden_states.begin() + static_cast<std::ptrdiff_t>(batch * input_batch_values),
                    hidden_states.begin() + static_cast<std::ptrdiff_t>((batch + 1) * input_batch_values),
                    padded_hidden.begin() + static_cast<std::ptrdiff_t>(batch * padded_batch_values));
            }
            core::write_tensor_f32(hidden_value_, padded_hidden);
            const std::vector<float> timestep_values(static_cast<size_t>(batch_size_), timestep);
            const std::vector<float> timestep_r_values(static_cast<size_t>(batch_size_), timestep_r);
            core::write_tensor_f32(timestep_value_, timestep_values);
            core::write_tensor_f32(timestep_r_value_, timestep_r_values);
            ggml_backend_tensor_set(time_freqs_, time_freq_values_.data(), 0, time_freq_values_.size() * sizeof(float));
            ggml_backend_tensor_set(time_freqs_r_, time_freq_values_.data(), 0, time_freq_values_.size() * sizeof(float));
            ggml_backend_tensor_set(
                positions_,
                position_values_.data(),
                0,
                position_values_.size() * sizeof(int32_t));
            core::set_backend_threads(backend_, threads_);
            const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
            ggml_backend_synchronize(backend_);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("ACE-Step diffusion graph compute failed");
            }
            std::vector<float> full;
            core::read_tensor_f32_into(output_, full);
            std::vector<float> out(static_cast<size_t>(batch_size_ * original_frames * config.latent_channels), 0.0F);
            for (int64_t batch = 0; batch < batch_size_; ++batch) {
                std::copy(
                    full.begin() + static_cast<std::ptrdiff_t>(batch * padded_batch_values),
                    full.begin() + static_cast<std::ptrdiff_t>(batch * padded_batch_values + input_batch_values),
                    out.begin() + static_cast<std::ptrdiff_t>(batch * input_batch_values));
            }
            return out;
        }

        int64_t padded_frames() const noexcept { return padded_frames_; }
        int64_t encoder_token_capacity() const noexcept { return encoder_tokens_; }
        int64_t batch_size() const noexcept { return batch_size_; }
        static int64_t padded_frames_for_request(int64_t frames, int64_t patch_size) {
            return round_up_frames(frames, patch_size);
        }

    private:
        static int64_t round_up_frames(int64_t frames, int64_t patch_size) {
            const int64_t remainder = frames % patch_size;
            return remainder == 0 ? frames : (frames + patch_size - remainder);
        }

        void build(size_t graph_arena_bytes) {
            const auto & config = assets_->config.diffusion;
            ggml_init_params params{graph_arena_bytes, nullptr, true};
            ctx_.reset(ggml_init(params));
            if (ctx_ == nullptr) {
                throw std::runtime_error("ACE-Step diffusion ggml context initialization failed");
            }
            core::ModuleBuildContext ctx{ctx_.get(), "ace_step.diffusion", backend_type_};

            hidden_value_ = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({batch_size_, padded_frames_, config.latent_channels}));
            context_value_ = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({batch_size_, padded_frames_, config.latent_channels * 2}));
            encoder_attention_mask_value_ = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({batch_size_, 1, patch_frames_, encoder_tokens_}));
            timestep_value_ = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({batch_size_, 1}));
            timestep_r_value_ = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({batch_size_, 1}));
            time_freqs_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_F32, 128);
            time_freqs_r_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_F32, 128);
            positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, patch_frames_);
            sliding_attention_mask_value_ = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({batch_size_, 1, patch_frames_, patch_frames_}));
            self_attention_padding_mask_value_ = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({batch_size_, 1, patch_frames_, patch_frames_}));
            ggml_set_input(hidden_value_.tensor);
            ggml_set_input(context_value_.tensor);
            ggml_set_input(encoder_attention_mask_value_.tensor);
            ggml_set_input(timestep_value_.tensor);
            ggml_set_input(timestep_r_value_.tensor);
            ggml_set_input(time_freqs_);
            ggml_set_input(time_freqs_r_);
            ggml_set_input(positions_);
            ggml_set_input(sliding_attention_mask_value_.tensor);
            ggml_set_input(self_attention_padding_mask_value_.tensor);

            auto hidden = modules::ConcatModule({2}).build(ctx, context_value_, hidden_value_);
            hidden = modules::TransposeModule({{0, 2, 1}, hidden.shape.rank}).build(ctx, hidden);
            hidden = modules::Conv1dModule({
                config.in_channels,
                config.hidden_size,
                config.patch_size,
                static_cast<int>(config.patch_size),
                0,
                1,
                true,
            }).build(ctx, hidden, weights_->proj_in);
            hidden = modules::TransposeModule({{0, 2, 1}, hidden.shape.rank}).build(ctx, hidden);

            auto timestep = build_time_embedding(ctx, time_freqs_, timestep_value_, weights_->time_embed, config.hidden_size);
            auto timestep_delta = core::wrap_tensor(
                ggml_sub(ctx.ggml, timestep_value_.tensor, timestep_r_value_.tensor),
                timestep_value_.shape,
                GGML_TYPE_F32);
            auto timestep_r =
                build_time_embedding(ctx, time_freqs_r_, timestep_delta, weights_->time_embed_r, config.hidden_size);
            auto temb = modules::AddModule{}.build(ctx, timestep.temb, timestep_r.temb);
            auto timestep_proj = modules::AddModule{}.build(ctx, timestep.timestep_proj, timestep_r.timestep_proj);
            auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({patch_frames_}), GGML_TYPE_I32);
            const auto sliding_mask = core::wrap_tensor(
                sliding_attention_mask_value_.tensor,
                core::TensorShape::from_dims({batch_size_, 1, patch_frames_, patch_frames_}),
                GGML_TYPE_F32);
            const auto self_padding_mask = core::wrap_tensor(
                self_attention_padding_mask_value_.tensor,
                core::TensorShape::from_dims({batch_size_, 1, patch_frames_, patch_frames_}),
                GGML_TYPE_F32);
            const core::TensorShape cross_cache_shape =
                core::TensorShape::from_dims({batch_size_, encoder_tokens_, config.num_key_value_heads, config.head_dim});
            cross_key_value_inputs_.reserve(weights_->layers.size());
            cross_value_value_inputs_.reserve(weights_->layers.size());

            for (size_t i = 0; i < weights_->layers.size(); ++i) {
                cross_key_value_inputs_.push_back(core::make_tensor(ctx, GGML_TYPE_F32, cross_cache_shape));
                cross_value_value_inputs_.push_back(core::make_tensor(ctx, GGML_TYPE_F32, cross_cache_shape));
                ggml_set_input(cross_key_value_inputs_.back().tensor);
                ggml_set_input(cross_value_value_inputs_.back().tensor);
                const std::optional<core::TensorValue> self_mask =
                    (config.layer_types[i] == "sliding_attention")
                    ? std::optional<core::TensorValue>(
                          modules::AddModule{}.build(ctx, sliding_mask, self_padding_mask))
                    : std::optional<core::TensorValue>(self_padding_mask);
                const PrecomputedCrossAttentionKV cross_attention_kv{
                    cross_key_value_inputs_.back(),
                    cross_value_value_inputs_.back(),
                };
                hidden = dit_layer(
                    ctx,
                    hidden,
                    positions,
                    timestep_proj,
                    weights_->one,
                    self_mask,
                    encoder_attention_mask_value_,
                    weights_->layers[i],
                    config,
                    backend_type_,
                    cross_attention_kv,
                    std::nullopt);
            }

            const auto final_modulation_shape = core::TensorShape::from_dims({batch_size_, 2, config.hidden_size});
            auto final_scale_shift_table = modules::RepeatModule({final_modulation_shape}).build(
                ctx,
                ensure_f32(ctx, weights_->final_scale_shift_table));
            auto final_temb = modules::RepeatModule({final_modulation_shape}).build(
                ctx,
                core::reshape_tensor(ctx, temb, core::TensorShape::from_dims({batch_size_, 1, config.hidden_size})));
            auto final_modulation = modules::AddModule{}.build(ctx, final_scale_shift_table, final_temb);
            auto shift = modules::SliceModule({1, 0, 1}).build(ctx, final_modulation);
            auto scale = modules::SliceModule({1, 1, 1}).build(ctx, final_modulation);
            shift = ensure_contiguous(ctx, shift);
            scale = ensure_contiguous(ctx, scale);
            shift = core::reshape_tensor(ctx, shift, core::TensorShape::from_dims({batch_size_, config.hidden_size}));
            scale = core::reshape_tensor(ctx, scale, core::TensorShape::from_dims({batch_size_, config.hidden_size}));
            hidden = apply_modulated_rms_norm(
                ctx,
                hidden,
                weights_->norm_out,
                weights_->one,
                expand_conditioning(ctx, shift, patch_frames_),
                expand_conditioning(ctx, scale, patch_frames_),
                config.rms_norm_eps);
            hidden = modules::TransposeModule({{0, 2, 1}, hidden.shape.rank}).build(ctx, hidden);
            hidden = modules::ConvTranspose1dModule({
                config.hidden_size,
                config.latent_channels,
                config.patch_size,
                static_cast<int>(config.patch_size),
                0,
                1,
                true,
            }).build(ctx, hidden, weights_->proj_out);
            hidden = modules::TransposeModule({{0, 2, 1}, hidden.shape.rank}).build(ctx, hidden);

            output_ = hidden.tensor;
            ggml_set_output(output_);
            graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
            ggml_build_forward_expand(graph_, output_);
            gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
            if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
                throw std::runtime_error("ACE-Step diffusion backend buffer allocation failed");
            }

            time_freq_values_ = make_timestep_freqs();
            ggml_backend_tensor_set(time_freqs_, time_freq_values_.data(), 0, time_freq_values_.size() * sizeof(float));
            ggml_backend_tensor_set(time_freqs_r_, time_freq_values_.data(), 0, time_freq_values_.size() * sizeof(float));
            position_values_.assign(static_cast<size_t>(patch_frames_), 0);
            for (int64_t i = 0; i < patch_frames_; ++i) {
                position_values_[static_cast<size_t>(i)] = static_cast<int32_t>(i);
            }
            ggml_backend_tensor_set(positions_, position_values_.data(), 0, position_values_.size() * sizeof(int32_t));
        }

        std::shared_ptr<const AceStepAssets> assets_;
        ggml_backend_t backend_ = nullptr;
        core::BackendType backend_type_ = core::BackendType::Cpu;
        int threads_ = 1;
        std::shared_ptr<const AceStepDiffusionWeights> weights_;
        int64_t batch_size_ = 1;
        int64_t original_frames_capacity_ = 0;
        int64_t padded_frames_ = 0;
        int64_t patch_frames_ = 0;
        int64_t encoder_tokens_ = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
        core::TensorValue hidden_value_;
        core::TensorValue context_value_;
        core::TensorValue encoder_attention_mask_value_;
        core::TensorValue timestep_value_;
        core::TensorValue timestep_r_value_;
        ggml_tensor * time_freqs_ = nullptr;
        ggml_tensor * time_freqs_r_ = nullptr;
        ggml_tensor * positions_ = nullptr;
        std::vector<float> time_freq_values_;
        std::vector<int32_t> position_values_;
        core::TensorValue sliding_attention_mask_value_;
        core::TensorValue self_attention_padding_mask_value_;
        std::vector<core::TensorValue> cross_key_value_inputs_;
        std::vector<core::TensorValue> cross_value_value_inputs_;
        ggml_tensor * output_ = nullptr;
        ggml_cgraph * graph_ = nullptr;
        ggml_gallocr_t gallocr_ = nullptr;
    };

    Impl(
        std::shared_ptr<const AceStepAssets> assets,
        core::ExecutionContext & execution,
        std::shared_ptr<const AceStepDitWeightsRuntime> dit_weights_runtime,
        size_t graph_arena_bytes)
        : assets_(std::move(assets)),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          graph_arena_bytes_(graph_arena_bytes),
          weights_(dit_weights_runtime->diffusion_weights()) {
        if (assets_ == nullptr) {
            throw std::runtime_error("ACE-Step diffusion runtime requires assets");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("ACE-Step diffusion backend is not initialized");
        }
    }

    static CrossAttentionCacheGraph::Output combine_cross_attention_cache(
        const CrossAttentionCacheGraph::Output & cond,
        const CrossAttentionCacheGraph::Output & uncond) {
        if (cond.keys.size() != uncond.keys.size() || cond.values.size() != uncond.values.size()) {
            throw std::runtime_error("ACE-Step diffusion CFG cross-attention cache layer count mismatch");
        }
        CrossAttentionCacheGraph::Output out;
        out.keys.resize(cond.keys.size());
        out.values.resize(cond.values.size());
        for (size_t layer = 0; layer < cond.keys.size(); ++layer) {
            if (cond.keys[layer].size() != uncond.keys[layer].size() ||
                cond.values[layer].size() != uncond.values[layer].size()) {
                throw std::runtime_error("ACE-Step diffusion CFG cross-attention cache shape mismatch");
            }
            out.keys[layer].reserve(cond.keys[layer].size() + uncond.keys[layer].size());
            out.keys[layer].insert(out.keys[layer].end(), cond.keys[layer].begin(), cond.keys[layer].end());
            out.keys[layer].insert(out.keys[layer].end(), uncond.keys[layer].begin(), uncond.keys[layer].end());
            out.values[layer].reserve(cond.values[layer].size() + uncond.values[layer].size());
            out.values[layer].insert(out.values[layer].end(), cond.values[layer].begin(), cond.values[layer].end());
            out.values[layer].insert(out.values[layer].end(), uncond.values[layer].begin(), uncond.values[layer].end());
        }
        return out;
    }

    AceStepLatents generate_latents(
        const AceStepDiffusionConditioning & conditioning,
        const AceStepGenerationOptions & options) {
        const auto total_start = Clock::now();
        const auto & pre = conditioning.pre_dit;
        const auto & config = assets_->config.diffusion;
        if (pre.context_latents.frames <= 0 || pre.context_latents.channels != config.latent_channels * 2) {
            throw std::runtime_error("ACE-Step diffusion requires valid context latents");
        }
        if (pre.encoder_hidden_states.tokens <= 0 || pre.encoder_hidden_states.hidden_size != config.hidden_size) {
            throw std::runtime_error("ACE-Step diffusion requires valid encoder hidden states");
        }
        const int64_t encoder_token_capacity = std::max<int64_t>(
            pre.encoder_hidden_states.tokens,
            pre.encoder_hidden_states_non_cover.tokens);
        const bool use_diffusion_cfg = !config.is_turbo && options.guidance_scale > 1.0F;
        const bool reference_skips_wavelet_dcw = assets_->selection.dit_model_path == "acestep-v15-base";
        const int64_t diffusion_batch_size = use_diffusion_cfg ? int64_t{2} : int64_t{1};
        if (use_diffusion_cfg && weights_->null_condition_emb_host.empty()) {
            throw std::runtime_error("ACE-Step Base CFG requires null_condition_emb");
        }
        const auto graph_prepare_start = Clock::now();
        if (!graph_ ||
            graph_->batch_size() != diffusion_batch_size ||
            !graph_->can_run(pre.context_latents.frames, encoder_token_capacity)) {
            graph_.reset();
            graph_ = std::make_unique<StepGraph>(
                assets_,
                backend_,
                backend_type_,
                threads_,
                weights_,
                diffusion_batch_size,
                pre.context_latents.frames,
                encoder_token_capacity,
                graph_arena_bytes_);
        }
        engine::debug::timing_log_scalar(
            "ace_step.diffusion.graph.prepare_ms",
            engine::debug::elapsed_ms(graph_prepare_start, Clock::now()));
        engine::debug::trace_log_scalar("ace_step.diffusion.context_frames", pre.context_latents.frames);
        engine::debug::trace_log_scalar("ace_step.diffusion.encoder_token_capacity", encoder_token_capacity);
        engine::debug::trace_log_scalar("ace_step.diffusion.batch_size", diffusion_batch_size);
        const int64_t padded_frames = graph_->padded_frames();
        const auto padding_start = Clock::now();
        const auto context_padded = pad_context_values(pre.context_latents, padded_frames, config.latent_channels * 2);
        std::vector<float> cfg_context_padded;
        if (use_diffusion_cfg) {
            cfg_context_padded = duplicate_batch_values(context_padded, diffusion_batch_size);
        }
        const auto encoder_hidden_padded =
            pad_encoder_hidden_values(pre.encoder_hidden_states, graph_->encoder_token_capacity(), config.hidden_size);
        const auto encoder_attention_mask_padded =
            pad_encoder_attention_mask(pre.encoder_hidden_states, graph_->encoder_token_capacity());
        engine::debug::timing_log_scalar("ace_step.diffusion.padding_ms", engine::debug::elapsed_ms(padding_start, Clock::now()));
        const auto cross_cache_prepare_start = Clock::now();
        if (!cross_cache_graph_ || !cross_cache_graph_->can_run(graph_->encoder_token_capacity())) {
            cross_cache_graph_ = std::make_unique<CrossAttentionCacheGraph>(
                assets_,
                backend_,
                backend_type_,
                threads_,
                weights_,
                graph_->encoder_token_capacity(),
                graph_arena_bytes_);
        }
        engine::debug::timing_log_scalar(
            "ace_step.diffusion.cross_cache.graph.prepare_ms",
            engine::debug::elapsed_ms(cross_cache_prepare_start, Clock::now()));
        const auto cross_cache_start = Clock::now();
        auto cross_attention_cache = cross_cache_graph_->run(encoder_hidden_padded);
        std::optional<CrossAttentionCacheGraph::Output> null_cross_attention_cache;
        if (use_diffusion_cfg) {
            const auto null_encoder_hidden_padded = null_encoder_hidden_values(
                weights_->null_condition_emb_host,
                graph_->encoder_token_capacity(),
                config.hidden_size);
            null_cross_attention_cache = cross_cache_graph_->run(null_encoder_hidden_padded);
            cross_attention_cache = combine_cross_attention_cache(cross_attention_cache, *null_cross_attention_cache);
        }
        engine::debug::timing_log_scalar(
            "ace_step.diffusion.cross_cache_ms",
            engine::debug::elapsed_ms(cross_cache_start, Clock::now()));
        std::vector<float> non_cover_context_padded;
        std::vector<float> non_cover_cfg_context_padded;
        std::vector<float> non_cover_encoder_hidden_padded;
        std::vector<int32_t> non_cover_encoder_attention_mask_padded;
        std::optional<CrossAttentionCacheGraph::Output> non_cover_cross_attention_cache;
        if (options.audio_cover_strength < 1.0F) {
            const auto non_cover_start = Clock::now();
            if (pre.context_latents_non_cover.values.empty() ||
                pre.encoder_hidden_states_non_cover.values.empty()) {
                throw std::runtime_error("ACE-Step diffusion requires non-cover conditioning state");
            }
            non_cover_context_padded =
                pad_context_values(pre.context_latents_non_cover, padded_frames, config.latent_channels * 2);
            if (use_diffusion_cfg) {
                non_cover_cfg_context_padded = duplicate_batch_values(non_cover_context_padded, diffusion_batch_size);
            }
            non_cover_encoder_hidden_padded = pad_encoder_hidden_values(
                pre.encoder_hidden_states_non_cover,
                graph_->encoder_token_capacity(),
                config.hidden_size);
            non_cover_encoder_attention_mask_padded = pad_encoder_attention_mask(
                pre.encoder_hidden_states_non_cover,
                graph_->encoder_token_capacity());
            const auto non_cover_cross_cache_start = Clock::now();
            non_cover_cross_attention_cache = cross_cache_graph_->run(non_cover_encoder_hidden_padded);
            if (use_diffusion_cfg) {
                non_cover_cross_attention_cache = combine_cross_attention_cache(
                    *non_cover_cross_attention_cache,
                    *null_cross_attention_cache);
            }
            engine::debug::timing_log_scalar(
                "ace_step.diffusion.non_cover_cross_cache_ms",
                engine::debug::elapsed_ms(non_cover_cross_cache_start, Clock::now()));
            engine::debug::timing_log_scalar(
                "ace_step.diffusion.non_cover_prepare_ms",
                engine::debug::elapsed_ms(non_cover_start, Clock::now()));
        }
        const auto repaint_mask_start = Clock::now();
        const auto & src_latents = pre.src_latents.values;
        const auto & clean_src_latents = pre.target_latents.values;
        std::vector<int32_t> repaint_mask = pre.repaint_mask;
        if (repaint_mask.empty()) {
            repaint_mask.assign(static_cast<size_t>(pre.context_latents.frames), 1);
        }
        engine::debug::timing_log_scalar(
            "ace_step.diffusion.repaint_mask_ms",
            engine::debug::elapsed_ms(repaint_mask_start, Clock::now()));
        const std::vector<float> * active_context_padded =
            use_diffusion_cfg ? &cfg_context_padded : &context_padded;
        const CrossAttentionCacheGraph::Output * active_cross_attention_cache = &cross_attention_cache;
        const std::vector<int32_t> * active_encoder_attention_mask_padded = &encoder_attention_mask_padded;
        const auto static_inputs_start = Clock::now();
        graph_->set_static_inputs(
            pre.context_latents.frames,
            *active_context_padded,
            *active_cross_attention_cache,
            *active_encoder_attention_mask_padded);
        engine::debug::timing_log_scalar(
            "ace_step.diffusion.static_inputs_ms",
            engine::debug::elapsed_ms(static_inputs_start, Clock::now()));

        const auto noise_start = Clock::now();
        std::vector<float> hidden(static_cast<size_t>(pre.context_latents.frames * config.latent_channels), 0.0F);
        auto noise_schedule = load_noise_schedule_or_sample(
            options.noise_file,
            static_cast<size_t>(pre.context_latents.frames * config.latent_channels),
            options.seed);
        auto noise = noise_schedule.initial_noise;
        uint64_t diffusion_noise_offset = static_cast<uint64_t>(noise.size());
        std::copy(noise.begin(), noise.end(), hidden.begin());
        if (options.retake_variance > 0.0F) {
            const uint32_t retake_seed = options.retake_seed.value_or(options.seed);
            auto retake_noise = load_noise_or_sample(
                std::string(),
                static_cast<size_t>(pre.context_latents.frames * config.latent_channels),
                retake_seed);
            constexpr float kHalfPi = 1.5707963267948966F;
            const float variance_radians = options.retake_variance * kHalfPi;
            const float keep = std::cos(variance_radians);
            const float inject = std::sin(variance_radians);
            for (size_t i = 0; i < noise.size(); ++i) {
                hidden[i] = keep * hidden[i] + inject * retake_noise[i];
            }
        }
        engine::debug::timing_log_scalar("ace_step.diffusion.noise_init_ms", engine::debug::elapsed_ms(noise_start, Clock::now()));

        const auto schedule_start = Clock::now();
        std::vector<float> schedule = make_python_schedule(options, config.is_turbo);
        if (options.cover_noise_strength > 0.0F) {
            const float effective_noise_level = 1.0F - options.cover_noise_strength;
            const auto nearest = std::min_element(
                schedule.begin(),
                schedule.end(),
                [effective_noise_level](float lhs, float rhs) {
                    return std::abs(lhs - effective_noise_level) < std::abs(rhs - effective_noise_level);
                });
            if (nearest != schedule.end()) {
                const float nearest_t = *nearest;
                auto cover_init = renoise(src_latents, nearest_t, options.seed, diffusion_noise_offset, &noise);
                std::copy(cover_init.begin(), cover_init.end(), hidden.begin());
                schedule.erase(schedule.begin(), nearest);
            }
        }
        engine::debug::timing_log_scalar("ace_step.diffusion.schedule_ms", engine::debug::elapsed_ms(schedule_start, Clock::now()));
        engine::debug::trace_log_scalar("ace_step.diffusion.schedule_steps", schedule.size());
        const int64_t injection_cutoff = python_round_nonnegative(
            options.repaint_injection_ratio * static_cast<float>(schedule.size()));
        size_t renoise_noise_index = 0;
        double velocity_graph_ms = 0.0;
        int64_t velocity_calls = 0;
        std::vector<float> apg_momentum;
        auto run_velocity = [&](float timestep, bool primary_eval) {
            graph_->set_static_inputs(
                pre.context_latents.frames,
                *active_context_padded,
                *active_cross_attention_cache,
                *active_encoder_attention_mask_padded);
            const auto velocity_start = Clock::now();
            std::vector<float> cfg_hidden;
            const std::vector<float> * graph_hidden = &hidden;
            if (use_diffusion_cfg) {
                cfg_hidden = duplicate_batch_values(hidden, diffusion_batch_size);
                graph_hidden = &cfg_hidden;
            }
            auto graph_out = graph_->run_step(pre.context_latents.frames, *graph_hidden, timestep, timestep);
            velocity_graph_ms += engine::debug::elapsed_ms(velocity_start, Clock::now());
            const size_t branch_values = static_cast<size_t>(pre.context_latents.frames * config.latent_channels);
            std::vector<float> out;
            if (!use_diffusion_cfg) {
                out = std::move(graph_out);
            } else {
                if (graph_out.size() != branch_values * 2) {
                    throw std::runtime_error("ACE-Step diffusion CFG graph output shape mismatch");
                }
                std::vector<float> pred_cond(graph_out.begin(), graph_out.begin() + static_cast<std::ptrdiff_t>(branch_values));
                if (!cfg_interval_active(timestep, options)) {
                    out = std::move(pred_cond);
                } else {
                    std::vector<float> pred_uncond(
                        graph_out.begin() + static_cast<std::ptrdiff_t>(branch_values),
                        graph_out.end());
                    if (options.use_adg) {
                        out = (primary_eval || timestep > 0.0F)
                            ? apply_adg_guidance(
                                  hidden,
                                  pred_cond,
                                  pred_uncond,
                                  timestep,
                                  options.guidance_scale,
                                  pre.context_latents.frames,
                                  config.latent_channels)
                            : apply_cfg_guidance(pred_cond, pred_uncond, options.guidance_scale);
                    } else if (primary_eval) {
                        out = apply_apg_guidance(
                            pred_cond,
                            pred_uncond,
                            options.guidance_scale,
                            pre.context_latents.frames,
                            config.latent_channels,
                            apg_momentum);
                    } else {
                        out = apply_cfg_guidance(pred_cond, pred_uncond, options.guidance_scale);
                    }
                }
            }
            ++velocity_calls;
            return out;
        };
        std::vector<float> previous_velocity;
        const int64_t cover_steps = static_cast<int64_t>(schedule.size() * options.audio_cover_strength);
        bool switched_to_non_cover = false;
        const auto sampling_loop_start = Clock::now();
        for (size_t i = 0; i < schedule.size(); ++i) {
            if (!switched_to_non_cover &&
                options.audio_cover_strength < 1.0F &&
                static_cast<int64_t>(i) >= cover_steps) {
                switched_to_non_cover = true;
                active_context_padded =
                    use_diffusion_cfg ? &non_cover_cfg_context_padded : &non_cover_context_padded;
                active_cross_attention_cache = &*non_cover_cross_attention_cache;
                active_encoder_attention_mask_padded = &non_cover_encoder_attention_mask_padded;
                graph_->set_static_inputs(
                    pre.context_latents.frames,
                    *active_context_padded,
                    *active_cross_attention_cache,
                    *active_encoder_attention_mask_padded);
            }
            const float t = schedule[i];
            const std::vector<float> hidden_before = hidden;
            std::vector<float> velocity = run_velocity(t, true);
            apply_velocity_norm_clamp(
                velocity,
                hidden,
                pre.context_latents.frames,
                config.latent_channels,
                options.velocity_norm_threshold);
            apply_velocity_ema(velocity, previous_velocity, options.velocity_ema_factor);
            const std::vector<float> velocity_for_denoise = velocity;
            std::vector<float> denoised = engine::sampling::denoise_from_velocity(hidden_before, velocity_for_denoise, t);
            if (i + 1 == schedule.size()) {
                write_x0_prefix(hidden, velocity, t);
                apply_dcw_correction(
                    hidden,
                    denoised,
                    pre.context_latents.frames,
                    config.latent_channels,
                    t,
                    reference_skips_wavelet_dcw,
                    options);
                if (!pre.repaint_mask.empty() && i < static_cast<size_t>(injection_cutoff)) {
                    apply_repaint_step_injection_in_place(
                        hidden,
                        clean_src_latents,
                        repaint_mask,
                        0.0F,
                        noise,
                        config.latent_channels);
                }
                previous_velocity = velocity;
                break;
            }
            const float next_t = schedule[i + 1];
            if (options.infer_method == "sde") {
                const std::vector<float> * renoise_noise = nullptr;
                if (!options.noise_file.empty()) {
                    if (renoise_noise_index >= noise_schedule.renoise_noises.size()) {
                        throw std::runtime_error("ACE-Step diffusion SDE renoise noise schedule exhausted");
                    }
                    renoise_noise = &noise_schedule.renoise_noises[renoise_noise_index++];
                }
                const float sde_renoise_timestep = config.is_turbo
                    ? next_t
                    : 1.0F - (static_cast<float>(i + 1) / static_cast<float>(schedule.size()));
                auto renoised = renoise(
                    denoised,
                    sde_renoise_timestep,
                    options.seed,
                    diffusion_noise_offset,
                    renoise_noise);
                std::copy(renoised.begin(), renoised.end(), hidden.begin());
                apply_dcw_correction(
                    hidden,
                    denoised,
                    pre.context_latents.frames,
                    config.latent_channels,
                    t,
                    reference_skips_wavelet_dcw,
                    options);
                if (!pre.repaint_mask.empty() && i < static_cast<size_t>(injection_cutoff)) {
                    apply_repaint_step_injection_in_place(
                        hidden,
                        clean_src_latents,
                        repaint_mask,
                        sde_renoise_timestep,
                        noise,
                        config.latent_channels);
                }
                previous_velocity = velocity;
                continue;
            }
            const float dt = t - next_t;
            if (options.sampler_mode == "heun") {
                std::vector<float> predicted = engine::sampling::euler_step(hidden, velocity, dt);
                const std::vector<float> hidden_before = hidden;
                hidden = predicted;
                std::vector<float> velocity_2 = run_velocity(next_t, false);
                hidden = hidden_before;
                apply_velocity_norm_clamp(
                    velocity_2,
                    predicted,
                    pre.context_latents.frames,
                    config.latent_channels,
                    options.velocity_norm_threshold);
                if (options.velocity_ema_factor > 0.0F) {
                    for (size_t j = 0; j < velocity_2.size(); ++j) {
                        velocity_2[j] =
                            (1.0F - options.velocity_ema_factor) * velocity_2[j] +
                            options.velocity_ema_factor * velocity[j];
                    }
                }
                hidden = engine::sampling::heun_step(hidden_before, velocity, velocity_2, dt);
                apply_dcw_correction(
                    hidden,
                    denoised,
                    pre.context_latents.frames,
                    config.latent_channels,
                    t,
                    reference_skips_wavelet_dcw,
                    options);
                if (!pre.repaint_mask.empty() && i < static_cast<size_t>(injection_cutoff)) {
                    apply_repaint_step_injection_in_place(
                        hidden,
                        clean_src_latents,
                        repaint_mask,
                        next_t,
                        noise,
                        config.latent_channels);
                }
                previous_velocity = engine::sampling::heun_combine_velocity(velocity, velocity_2);
                continue;
            }
            subtract_velocity_step(hidden, velocity, dt);
            apply_dcw_correction(
                hidden,
                denoised,
                pre.context_latents.frames,
                config.latent_channels,
                t,
                reference_skips_wavelet_dcw,
                options);
            if (!pre.repaint_mask.empty() && i < static_cast<size_t>(injection_cutoff)) {
                apply_repaint_step_injection_in_place(
                    hidden,
                    clean_src_latents,
                    repaint_mask,
                    next_t,
                    noise,
                    config.latent_channels);
            }
            previous_velocity = velocity;
        }
        engine::debug::timing_log_scalar(
            "ace_step.diffusion.sampling_loop_ms",
            engine::debug::elapsed_ms(sampling_loop_start, Clock::now()));
        engine::debug::timing_log_scalar("ace_step.diffusion.velocity.graph.total_ms", velocity_graph_ms);
        engine::debug::trace_log_scalar("ace_step.diffusion.velocity_calls", velocity_calls);

        const auto repaint_blend_start = Clock::now();
        if (!pre.repaint_mask.empty() && options.repaint_crossfade_frames > 0) {
            apply_repaint_boundary_blend(
                hidden,
                clean_src_latents,
                repaint_mask,
                options.repaint_crossfade_frames,
                config.latent_channels);
        }
        engine::debug::timing_log_scalar(
            "ace_step.diffusion.repaint_blend_ms",
            engine::debug::elapsed_ms(repaint_blend_start, Clock::now()));

        const auto output_start = Clock::now();
        AceStepLatents out;
        out.frames = pre.context_latents.frames;
        out.channels = config.latent_channels;
        out.values.assign(
            hidden.begin(),
            hidden.begin() + static_cast<std::ptrdiff_t>(out.frames * out.channels));
        engine::debug::timing_log_scalar("ace_step.diffusion.output_ms", engine::debug::elapsed_ms(output_start, Clock::now()));
        engine::debug::timing_log_scalar("ace_step.diffusion.total_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
        return out;
    }

    void release_graph_workspace() {
        graph_.reset();
        cross_cache_graph_.reset();
    }

    std::shared_ptr<const AceStepAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
        int threads_ = 1;
        size_t graph_arena_bytes_ = 0;
        std::shared_ptr<const AceStepDiffusionWeights> weights_;
        std::unique_ptr<CrossAttentionCacheGraph> cross_cache_graph_;
        std::unique_ptr<StepGraph> graph_;
};

AceStepDiffusionRuntime::AceStepDiffusionRuntime(
    std::shared_ptr<const AceStepAssets> assets,
    core::ExecutionContext & execution,
    std::shared_ptr<const AceStepDitWeightsRuntime> dit_weights_runtime,
    size_t graph_arena_bytes)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          std::move(dit_weights_runtime),
          graph_arena_bytes)) {}

AceStepDiffusionRuntime::~AceStepDiffusionRuntime() = default;

AceStepLatents AceStepDiffusionRuntime::generate_latents(
    const AceStepDiffusionConditioning & conditioning,
    const AceStepGenerationOptions & options) const {
    return impl_->generate_latents(conditioning, options);
}

void AceStepDiffusionRuntime::release_graph_workspace() const {
    impl_->release_graph_workspace();
}

}  // namespace engine::models::ace_step
