#include "flow_impl.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::models::dots_tts::detail {

void GgmlContextDeleter::operator()(ggml_context * ctx) const noexcept {
    if (ctx != nullptr) {
        ggml_free(ctx);
    }
}

void GgmlGallocrDeleter::operator()(ggml_gallocr_t alloc) const noexcept {
    if (alloc != nullptr) {
        ggml_gallocr_free(alloc);
    }
}

int64_t head_dim(const DotsTransformerConfig & config) {
    if (config.num_heads <= 0 || config.hidden_size <= 0 || config.hidden_size % config.num_heads != 0) {
        throw std::runtime_error("DotTTS DiT attention config is invalid");
    }
    return config.hidden_size / config.num_heads;
}

ggml_type cached_dit_kv_type(core::BackendType backend_type) {
    return backend_type == core::BackendType::Cuda ? GGML_TYPE_F16 : GGML_TYPE_F32;
}

core::TensorValue cast_tensor(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    ggml_type type) {
    if (value.type == type) {
        return value;
    }
    return core::wrap_tensor(
        ggml_cast(ctx.ggml, core::ensure_backend_addressable_layout(ctx, value).tensor, type),
        value.shape,
        type);
}

core::TensorValue cast_cached_dit_kv(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    bool enabled) {
    if (!enabled) {
        return value;
    }
    return cast_tensor(ctx, value, cached_dit_kv_type(ctx.backend_type));
}

core::TensorValue expand_conditioning_like(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    const core::TensorValue & like) {
    if (value.shape.rank == like.shape.rank) {
        return value;
    }
    if (value.shape.rank + 1 != like.shape.rank) {
        throw std::runtime_error("DotTTS DiT conditioning tensor is not broadcast-compatible");
    }
    core::TensorShape shape = {};
    shape.rank = like.shape.rank;
    shape.dims[0] = value.shape.dims[0];
    shape.dims[1] = 1;
    for (size_t i = 1; i < value.shape.rank; ++i) {
        shape.dims[i + 1] = value.shape.dims[i];
    }
    auto reshaped = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, value), shape);
    return core::wrap_tensor(ggml_repeat(ctx.ggml, reshaped.tensor, like.tensor), like.shape, GGML_TYPE_F32);
}

core::TensorValue modulate(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & normalized,
    const core::TensorValue & shift,
    const core::TensorValue & scale) {
    auto scale_like = expand_conditioning_like(ctx, scale, normalized);
    auto shift_like = expand_conditioning_like(ctx, shift, normalized);
    auto scaled = modules::MulModule().build(ctx, normalized, scale_like);
    auto shifted = modules::AddModule().build(ctx, normalized, scaled);
    return modules::AddModule().build(ctx, shifted, shift_like);
}

core::TensorValue reshape_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t heads, int64_t dim) {
    return core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, input),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

std::vector<float> timestep_freqs() {
    std::vector<float> values(static_cast<size_t>(kTimestepFrequencyDim / 2), 0.0F);
    const float half = static_cast<float>(kTimestepFrequencyDim / 2);
    for (int64_t i = 0; i < kTimestepFrequencyDim / 2; ++i) {
        values[static_cast<size_t>(i)] = std::exp(-std::log(10000.0F) * static_cast<float>(i) / half);
    }
    return values;
}

core::TensorValue timestep_embedding(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & timesteps,
    const core::TensorValue & freqs_tensor,
    const modules::LinearWeights & fc1,
    const modules::LinearWeights & fc2,
    int64_t hidden_size) {
    core::validate_shape(timesteps, core::TensorShape::from_dims({timesteps.shape.dims[0]}), "DotTTS DiT timesteps");
    const int64_t batch = timesteps.shape.dims[0];
    auto t = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, timesteps), core::TensorShape::from_dims({batch, 1}));
    auto expanded = modules::RepeatModule({core::TensorShape::from_dims({batch, kTimestepFrequencyDim / 2})}).build(ctx, t);
    auto freqs = core::reshape_tensor(ctx, freqs_tensor, core::TensorShape::from_dims({1, kTimestepFrequencyDim / 2}));
    freqs = modules::RepeatModule({core::TensorShape::from_dims({batch, kTimestepFrequencyDim / 2})}).build(ctx, freqs);
    auto args = modules::MulModule().build(ctx, expanded, freqs);
    auto cos_part = core::wrap_tensor(ggml_cos(ctx.ggml, args.tensor), args.shape, GGML_TYPE_F32);
    auto sin_part = core::wrap_tensor(ggml_sin(ctx.ggml, args.tensor), args.shape, GGML_TYPE_F32);
    auto embedding = modules::ConcatModule({1}).build(ctx, cos_part, sin_part);
    auto hidden = modules::LinearModule({kTimestepFrequencyDim, hidden_size, true})
                      .build(ctx, embedding, fc1);
    hidden = modules::SiluModule().build(ctx, hidden);
    return modules::LinearModule({hidden_size, hidden_size, true})
        .build(ctx, hidden, fc2);
}

core::TensorValue timestep_embedding(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & timesteps,
    const DotFlowWeights & weights) {
    return timestep_embedding(
        ctx,
        timesteps,
        weights.timestep_freqs,
        weights.timestep_fc1,
        weights.timestep_fc2,
        weights.config.dit.hidden_size);
}

DotAttentionOutput dit_attention_with_cache(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const DotDitBlockWeights & weights,
    const DotsTransformerConfig & config,
    const std::optional<core::TensorValue> & attention_mask,
    const std::optional<core::TensorValue> & prefix_key = std::nullopt,
    const std::optional<core::TensorValue> & prefix_value = std::nullopt,
    bool use_16bit_kv = false) {
    const int64_t dim = head_dim(config);
    auto qkv = modules::LinearModule({config.hidden_size, 3 * config.hidden_size, false}).build(ctx, input, {weights.qkv_proj, std::nullopt});
    auto q = modules::SliceModule({static_cast<int>(qkv.shape.rank - 1), 0, config.hidden_size}).build(ctx, qkv);
    auto k = modules::SliceModule({static_cast<int>(qkv.shape.rank - 1), config.hidden_size, config.hidden_size}).build(ctx, qkv);
    auto v = modules::SliceModule({static_cast<int>(qkv.shape.rank - 1), 2 * config.hidden_size, config.hidden_size}).build(ctx, qkv);
    q = modules::RMSNormModule({dim, kTorchBFloat16Eps, true, false}).build(ctx, reshape_heads(ctx, q, config.num_heads, dim), {weights.q_norm, std::nullopt});
    k = modules::RMSNormModule({dim, kTorchBFloat16Eps, true, false}).build(ctx, reshape_heads(ctx, k, config.num_heads, dim), {weights.k_norm, std::nullopt});
    v = reshape_heads(ctx, v, config.num_heads, dim);
    if (config.rotary_bias) {
        q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rotary_theta}).build(ctx, q, positions);
        k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rotary_theta}).build(ctx, k, positions);
    }
    k = cast_cached_dit_kv(ctx, k, use_16bit_kv);
    v = cast_cached_dit_kv(ctx, v, use_16bit_kv);
    auto attention_key = k;
    auto attention_value = v;
    if (prefix_key.has_value() || prefix_value.has_value()) {
        if (!prefix_key.has_value() || !prefix_value.has_value()) {
            throw std::runtime_error("DotTTS cached DiT attention requires both prefix key and prefix value");
        }
        auto prefix_key_value = prefix_key->type == k.type ? *prefix_key : cast_tensor(ctx, *prefix_key, k.type);
        auto prefix_value_value = prefix_value->type == v.type ? *prefix_value : cast_tensor(ctx, *prefix_value, v.type);
        attention_key = modules::ConcatModule({1}).build(ctx, prefix_key_value, k);
        attention_value = modules::ConcatModule({1}).build(ctx, prefix_value_value, v);
    }
    q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, attention_key);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, attention_value);
    const auto lowering = ctx.backend_type == core::BackendType::Cuda
        ? modules::ScaledDotProductAttentionLowering::FlashPreserveViews
        : modules::ScaledDotProductAttentionLowering::Explicit;
    auto context = modules::ScaledDotProductAttentionModule({
        dim,
        lowering,
        GGML_PREC_DEFAULT,
        modules::AttentionCausality::NonCausal,
    }).build(ctx, q, k_heads, v_heads, attention_mask);
    context = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, context), core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.hidden_size}));
    return {
        modules::LinearModule({config.hidden_size, config.hidden_size, true}).build(ctx, context, {weights.o_proj, weights.o_bias}),
        k,
        v,
    };
}

core::TensorValue dit_block_with_mods(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & mods,
    const core::TensorValue & positions,
    const DotDitBlockWeights & weights,
    const DotsTransformerConfig & config,
    const std::optional<core::TensorValue> & attention_mask);

core::TensorValue dit_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & condition,
    const core::TensorValue & positions,
    const DotDitBlockWeights & weights,
    const DotsTransformerConfig & config,
    const std::optional<core::TensorValue> & attention_mask = std::nullopt) {
    auto mods = modules::LinearModule({config.hidden_size, config.hidden_size * 6, true})
                    .build(ctx, modules::SiluModule().build(ctx, condition), weights.adaln);
    return dit_block_with_mods(ctx, input, mods, positions, weights, config, attention_mask);
}

DotBlockOutput dit_block_with_mods_and_cache(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & mods,
    const core::TensorValue & positions,
    const DotDitBlockWeights & weights,
    const DotsTransformerConfig & config,
    const std::optional<core::TensorValue> & attention_mask,
    const std::optional<core::TensorValue> & prefix_key,
    const std::optional<core::TensorValue> & prefix_value,
    bool use_16bit_kv) {
    auto shift_attn = modules::SliceModule({static_cast<int>(mods.shape.rank - 1), 0, config.hidden_size}).build(ctx, mods);
    auto scale_attn = modules::SliceModule({static_cast<int>(mods.shape.rank - 1), config.hidden_size, config.hidden_size}).build(ctx, mods);
    auto gate_attn = expand_conditioning_like(
        ctx,
        modules::SliceModule({static_cast<int>(mods.shape.rank - 1), 2 * config.hidden_size, config.hidden_size}).build(ctx, mods),
        input);
    auto shift_ffn = modules::SliceModule({static_cast<int>(mods.shape.rank - 1), 3 * config.hidden_size, config.hidden_size}).build(ctx, mods);
    auto scale_ffn = modules::SliceModule({static_cast<int>(mods.shape.rank - 1), 4 * config.hidden_size, config.hidden_size}).build(ctx, mods);
    auto gate_ffn = expand_conditioning_like(
        ctx,
        modules::SliceModule({static_cast<int>(mods.shape.rank - 1), 5 * config.hidden_size, config.hidden_size}).build(ctx, mods),
        input);

    auto attn_in = modules::LayerNormModule({config.hidden_size, kLayerNormEps, false, false}).build(ctx, input, {});
    attn_in = modulate(ctx, attn_in, shift_attn, scale_attn);
    auto attention = dit_attention_with_cache(ctx, attn_in, positions, weights, config, attention_mask, prefix_key, prefix_value, use_16bit_kv);
    auto gated_attention = modules::MulModule().build(ctx, gate_attn, attention.output);
    auto x = modules::AddModule().build(ctx, input, gated_attention);
    auto ffn = modules::LayerNormModule({config.hidden_size, kLayerNormEps, false, false}).build(ctx, x, {});
    ffn = modulate(ctx, ffn, shift_ffn, scale_ffn);
    ffn = modules::LinearModule({config.hidden_size, config.ffn_hidden_size, true}).build(ctx, ffn, weights.fc1);
    ffn = modules::GeluModule({modules::GeluApproximation::Tanh}).build(ctx, ffn);
    ffn = modules::LinearModule({config.ffn_hidden_size, config.hidden_size, true}).build(ctx, ffn, weights.fc2);
    auto gated_ffn = modules::MulModule().build(ctx, gate_ffn, ffn);
    return {modules::AddModule().build(ctx, x, gated_ffn), attention.key, attention.value};
}

core::TensorValue dit_block_with_mods(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & mods,
    const core::TensorValue & positions,
    const DotDitBlockWeights & weights,
    const DotsTransformerConfig & config,
    const std::optional<core::TensorValue> & attention_mask = std::nullopt) {
    return dit_block_with_mods_and_cache(ctx, input, mods, positions, weights, config, attention_mask).output;
}

core::TensorValue final_projection_with_mods(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & mods,
    const DotFlowWeights & weights);

core::TensorValue final_projection(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & condition,
    const DotFlowWeights & weights) {
    const auto & config = weights.config.dit;
    auto mods = modules::LinearModule({config.hidden_size, 2 * config.hidden_size, true})
                    .build(ctx, modules::SiluModule().build(ctx, condition), weights.final_adaln);
    return final_projection_with_mods(ctx, input, mods, weights);
}

core::TensorValue final_projection_with_mods(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & mods,
    const DotFlowWeights & weights) {
    const auto & config = weights.config.dit;
    auto shift = modules::SliceModule({static_cast<int>(mods.shape.rank - 1), 0, config.hidden_size}).build(ctx, mods);
    auto scale = modules::SliceModule({static_cast<int>(mods.shape.rank - 1), config.hidden_size, config.hidden_size}).build(ctx, mods);
    auto normalized = modules::LayerNormModule({config.hidden_size, kLayerNormEps, false, false}).build(ctx, input, {});
    normalized = modulate(ctx, normalized, shift, scale);
    return modules::LinearModule({config.hidden_size, weights.config.latent_dim, true})
        .build(ctx, normalized, weights.final_linear);
}

core::TensorValue build_flow_condition(
    core::ModuleBuildContext & ctx,
    const DotFlowWeights & weights,
    const core::TensorValue & timesteps,
    const std::optional<core::TensorValue> & durations,
    const core::TensorValue & speaker_condition) {
    auto condition = modules::AddModule().build(ctx, timestep_embedding(ctx, timesteps, weights), speaker_condition);
    if (durations.has_value()) {
        if (!weights.duration_fc1.has_value() || !weights.duration_fc2.has_value()) {
            throw std::runtime_error("DotTTS MeanFlow duration embedding weights are missing");
        }
        auto duration_condition = timestep_embedding(
            ctx,
            *durations,
            weights.timestep_freqs,
            *weights.duration_fc1,
            *weights.duration_fc2,
            weights.config.dit.hidden_size);
        condition = modules::AddModule().build(ctx, condition, duration_condition);
    }
    return condition;
}

core::TensorValue build_modulation_graph(
    core::ModuleBuildContext & ctx,
    const DotFlowWeights & weights,
    const core::TensorValue & timesteps,
    const std::optional<core::TensorValue> & durations,
    const core::TensorValue & speaker_condition) {
    const auto & config = weights.config.dit;
    auto condition = build_flow_condition(ctx, weights, timesteps, durations, speaker_condition);
    auto activated = modules::SiluModule().build(ctx, condition);
    std::vector<core::TensorValue> parts;
    parts.reserve(weights.blocks.size() + 1);
    for (const auto & block : weights.blocks) {
        parts.push_back(modules::LinearModule({config.hidden_size, config.hidden_size * 6, true})
                            .build(ctx, activated, block.adaln));
    }
    parts.push_back(modules::LinearModule({config.hidden_size, 2 * config.hidden_size, true})
                        .build(ctx, activated, weights.final_adaln));
    if (parts.empty()) {
        throw std::runtime_error("DotTTS modulation graph has no parts");
    }
    auto out = parts.front();
    for (size_t i = 1; i < parts.size(); ++i) {
        out = modules::ConcatModule({1}).build(ctx, out, parts[i]);
    }
    return out;
}

core::TensorValue build_velocity_graph(
    core::ModuleBuildContext & ctx,
    const DotFlowWeights & weights,
    const core::TensorValue & sequence,
    const core::TensorValue & timesteps,
    const std::optional<core::TensorValue> & durations,
    const core::TensorValue & speaker_condition,
    const core::TensorValue & positions,
    const std::optional<core::TensorValue> & attention_mask,
    const std::optional<core::TensorValue> & modulations,
    int64_t output_start,
    int64_t output_length) {
    const auto & config = weights.config.dit;
    std::optional<core::TensorValue> condition;
    if (!modulations.has_value()) {
        condition = build_flow_condition(ctx, weights, timesteps, durations, speaker_condition);
    }
    auto x = modules::LinearModule({weights.config.dit.hidden_size, weights.config.dit.hidden_size, true})
                 .build(ctx, sequence, weights.input_layer);
    for (size_t layer = 0; layer < weights.blocks.size(); ++layer) {
        if (modulations.has_value()) {
            auto mods = modules::SliceModule({
                static_cast<int>(modulations->shape.rank - 1),
                static_cast<int64_t>(layer) * 6 * config.hidden_size,
                6 * config.hidden_size,
            }).build(ctx, *modulations);
            x = dit_block_with_mods(ctx, x, mods, positions, weights.blocks[layer], config, attention_mask);
        } else {
            x = dit_block(ctx, x, *condition, positions, weights.blocks[layer], config, attention_mask);
        }
    }
    if (output_length > 0) {
        x = modules::SliceModule({1, output_start, output_length}).build(ctx, x);
    }
    if (modulations.has_value()) {
        auto final_mods = modules::SliceModule({
            static_cast<int>(modulations->shape.rank - 1),
            static_cast<int64_t>(weights.blocks.size()) * 6 * config.hidden_size,
            2 * config.hidden_size,
        }).build(ctx, *modulations);
        return final_projection_with_mods(ctx, x, final_mods, weights);
    }
    return final_projection(ctx, x, *condition, weights);
}

core::TensorValue load_fused_qkv_projection(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden_size,
    assets::TensorStorageType storage_type,
    core::BackendType backend_type);

DotDitBlockWeights load_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const DotsTransformerConfig & config,
    int64_t layer,
    assets::TensorStorageType storage_type,
    core::BackendType backend_type) {
    const int64_t dim = head_dim(config);
    const std::string prefix = "velocity_field_predictor.blocks." + std::to_string(layer);
    DotDitBlockWeights out;
    out.adaln = {
        store.load_tensor(source, prefix + ".adaLN_modulation.1.weight", storage_type, {6 * config.hidden_size, config.hidden_size}),
        store.load_f32_tensor(source, prefix + ".adaLN_modulation.1.bias", {6 * config.hidden_size}),
    };
    out.qkv_proj = load_fused_qkv_projection(store, source, prefix, config.hidden_size, storage_type, backend_type);
    out.q_norm = store.load_f32_tensor(source, prefix + ".attn.q_norm.weight", {dim});
    out.k_norm = store.load_f32_tensor(source, prefix + ".attn.k_norm.weight", {dim});
    out.o_proj = store.load_tensor(source, prefix + ".attn.o_proj.weight", storage_type, {config.hidden_size, config.hidden_size});
    out.o_bias = store.load_f32_tensor(source, prefix + ".attn.o_proj.bias", {config.hidden_size});
    out.fc1 = {
        store.load_tensor(source, prefix + ".ffn.fc1.weight", storage_type, {config.ffn_hidden_size, config.hidden_size}),
        store.load_f32_tensor(source, prefix + ".ffn.fc1.bias", {config.ffn_hidden_size}),
    };
    out.fc2 = {
        store.load_tensor(source, prefix + ".ffn.fc2.weight", storage_type, {config.hidden_size, config.ffn_hidden_size}),
        store.load_f32_tensor(source, prefix + ".ffn.fc2.bias", {config.hidden_size}),
    };
    return out;
}

void validate_config(const DotsConfig & config) {
    if (config.dit.hidden_size <= 0 || config.dit.ffn_hidden_size <= 0 || config.dit.num_layers <= 0) {
        throw std::runtime_error("DotTTS DiT config contains non-positive dimensions");
    }
    if (!config.dit.modulation || config.dit.norm_layer != "RMSNorm" || config.dit.qkv_bias || !config.dit.qk_norm || !config.dit.rotary_bias) {
        throw std::runtime_error("DotTTS DiT checkpoint layout is not supported by the native component");
    }
    if (config.latent_dim <= 0 || config.llm.hidden_size <= 0 || config.campplus_embedding_size <= 0) {
        throw std::runtime_error("DotTTS flow projection config contains invalid dimensions");
    }
}

std::shared_ptr<DotFlowWeights> load_weights(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    DotsConfig config,
    assets::TensorStorageType storage_type) {
    if (source == nullptr) {
        throw std::runtime_error("DotTTS flow component requires tensor source");
    }
    validate_config(config);
    auto weights = std::make_shared<DotFlowWeights>();
    weights->config = config;
    weights->execution_context = std::make_shared<core::ExecutionContext>(backend);
    weights->store = std::make_shared<core::BackendWeightStore>(
        weights->execution_context->backend(),
        weights->execution_context->backend_type(),
        "dots_tts.flow.weights",
        kWeightContextBytes);
    weights->hidden_proj = {
        weights->store->load_tensor(*source, "hidden_proj.weight", storage_type, {config.dit.hidden_size, config.llm.hidden_size}),
        weights->store->load_f32_tensor(*source, "hidden_proj.bias", {config.dit.hidden_size}),
    };
    weights->latent_proj = {
        weights->store->load_tensor(*source, "latent_proj.weight", storage_type, {config.dit.hidden_size, config.latent_dim}),
        weights->store->load_f32_tensor(*source, "latent_proj.bias", {config.dit.hidden_size}),
    };
    weights->coordinate_proj = {
        weights->store->load_tensor(*source, "coordinate_proj.weight", storage_type, {config.dit.hidden_size, config.latent_dim}),
        weights->store->load_f32_tensor(*source, "coordinate_proj.bias", {config.dit.hidden_size}),
    };
    weights->xvec_proj = {
        weights->store->load_tensor(*source, "xvec_proj.0.weight", storage_type, {config.dit.hidden_size, config.campplus_embedding_size}),
        weights->store->load_f32_tensor(*source, "xvec_proj.0.bias", {config.dit.hidden_size}),
    };
    weights->xvec_norm = {
        weights->store->load_f32_tensor(*source, "xvec_proj.1.weight", {config.dit.hidden_size}),
        weights->store->load_f32_tensor(*source, "xvec_proj.1.bias", {config.dit.hidden_size}),
    };
    weights->input_layer = {
        weights->store->load_tensor(*source, "velocity_field_predictor.input_layer.weight", storage_type, {config.dit.hidden_size, config.dit.hidden_size}),
        weights->store->load_f32_tensor(*source, "velocity_field_predictor.input_layer.bias", {config.dit.hidden_size}),
    };
    weights->timestep_freqs = weights->store->make_f32(core::TensorShape::from_dims({kTimestepFrequencyDim / 2}), timestep_freqs());
    weights->timestep_fc1 = {
        weights->store->load_tensor(*source, "velocity_field_predictor.time_embedder.mlp.0.weight", storage_type, {config.dit.hidden_size, kTimestepFrequencyDim}),
        weights->store->load_f32_tensor(*source, "velocity_field_predictor.time_embedder.mlp.0.bias", {config.dit.hidden_size}),
    };
    weights->timestep_fc2 = {
        weights->store->load_tensor(*source, "velocity_field_predictor.time_embedder.mlp.2.weight", storage_type, {config.dit.hidden_size, config.dit.hidden_size}),
        weights->store->load_f32_tensor(*source, "velocity_field_predictor.time_embedder.mlp.2.bias", {config.dit.hidden_size}),
    };
    if (config.meanflow.has_value() && config.meanflow->enabled && config.meanflow->use_duration_embedding) {
        weights->duration_fc1 = modules::LinearWeights{
            weights->store->load_tensor(*source, "velocity_field_predictor.duration_embedder.mlp.0.weight", storage_type, {config.dit.hidden_size, kTimestepFrequencyDim}),
            weights->store->load_f32_tensor(*source, "velocity_field_predictor.duration_embedder.mlp.0.bias", {config.dit.hidden_size}),
        };
        weights->duration_fc2 = modules::LinearWeights{
            weights->store->load_tensor(*source, "velocity_field_predictor.duration_embedder.mlp.2.weight", storage_type, {config.dit.hidden_size, config.dit.hidden_size}),
            weights->store->load_f32_tensor(*source, "velocity_field_predictor.duration_embedder.mlp.2.bias", {config.dit.hidden_size}),
        };
    }
    weights->blocks.reserve(static_cast<size_t>(config.dit.num_layers));
    for (int64_t layer = 0; layer < config.dit.num_layers; ++layer) {
        weights->blocks.push_back(load_block(
            *weights->store,
            *source,
            config.dit,
            layer,
            storage_type,
            weights->execution_context->backend_type()));
    }
    weights->final_adaln = {
        weights->store->load_tensor(*source, "velocity_field_predictor.output_layer.adaLN_modulation.1.weight", storage_type, {2 * config.dit.hidden_size, config.dit.hidden_size}),
        weights->store->load_f32_tensor(*source, "velocity_field_predictor.output_layer.adaLN_modulation.1.bias", {2 * config.dit.hidden_size}),
    };
    weights->final_linear = {
        weights->store->load_tensor(*source, "velocity_field_predictor.output_layer.linear.weight", storage_type, {config.latent_dim, config.dit.hidden_size}),
        weights->store->load_f32_tensor(*source, "velocity_field_predictor.output_layer.linear.bias", {config.latent_dim}),
    };
    weights->store->upload();
    return weights;
}

std::vector<uint8_t> build_decode_mask(int64_t batch_size, int64_t total_len, int64_t fm_seq_len, int64_t latent_patch_size, int64_t hidden_patch_size) {
    std::vector<uint8_t> mask(static_cast<size_t>(batch_size * total_len * total_len), 0);
    const int64_t latent_start = total_len - latent_patch_size;
    const int64_t block_start = fm_seq_len - hidden_patch_size;
    if (block_start < 0) {
        throw std::runtime_error("DotTTS flow decode mask requires a current hidden patch");
    }
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const int64_t base = batch * total_len * total_len;
        if (block_start > 0) {
            for (int64_t row = 0; row < block_start; ++row) {
                for (int64_t col = 0; col <= row; ++col) {
                    mask[static_cast<size_t>(base + row * total_len + col)] = 1;
                }
            }
        }
        for (int64_t row = block_start; row < fm_seq_len; ++row) {
            for (int64_t col = 0; col < fm_seq_len; ++col) {
                mask[static_cast<size_t>(base + row * total_len + col)] = 1;
            }
            for (int64_t col = latent_start; col < total_len; ++col) {
                mask[static_cast<size_t>(base + row * total_len + col)] = 1;
            }
        }
        for (int64_t row = latent_start; row < total_len; ++row) {
            for (int64_t col = 0; col < fm_seq_len; ++col) {
                mask[static_cast<size_t>(base + row * total_len + col)] = 1;
            }
            for (int64_t col = latent_start; col < total_len; ++col) {
                mask[static_cast<size_t>(base + row * total_len + col)] = 1;
            }
        }
    }
    return mask;
}

std::vector<int32_t> build_decode_positions(int64_t total_len, int64_t fm_seq_len, int64_t latent_patch_size) {
    std::vector<int32_t> positions(static_cast<size_t>(total_len), 0);
    for (int64_t i = 0; i < fm_seq_len; ++i) {
        positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    const int64_t latent_start = total_len - latent_patch_size;
    for (int64_t i = 0; i < latent_patch_size; ++i) {
        positions[static_cast<size_t>(latent_start + i)] = static_cast<int32_t>(fm_seq_len + i);
    }
    return positions;
}

std::vector<int32_t> build_position_range(int64_t start, int64_t steps) {
    if (start < 0 || steps <= 0) {
        throw std::runtime_error("DotTTS DiT position range is invalid");
    }
    std::vector<int32_t> positions(static_cast<size_t>(steps), 0);
    for (int64_t i = 0; i < steps; ++i) {
        positions[static_cast<size_t>(i)] = static_cast<int32_t>(start + i);
    }
    return positions;
}

int64_t resolve_generate_length_bucket(int64_t requested_patches) {
    if (requested_patches <= 0) {
        throw std::runtime_error("DotTTS DiT cache bucket requires a positive patch count");
    }
    constexpr int64_t kBuckets[] = {64, 128, 256, 512};
    for (const int64_t bucket : kBuckets) {
        if (requested_patches <= bucket) {
            return bucket;
        }
    }
    throw std::runtime_error("DotTTS DiT cache request exceeds the largest supported generation bucket");
}

int64_t resolve_dit_cache_capacity_tokens(int64_t fm_seq_len, int64_t unit_len) {
    if (fm_seq_len <= 0 || unit_len <= 0) {
        throw std::runtime_error("DotTTS DiT cache capacity request is invalid");
    }
    const int64_t requested_patches = std::max<int64_t>(1, (fm_seq_len + unit_len - 1) / unit_len);
    return resolve_generate_length_bucket(requested_patches) * unit_len;
}

std::vector<ggml_fp16_t> build_prefill_mask_values(int64_t steps) {
    if (steps <= 0) {
        throw std::runtime_error("DotTTS DiT prefill mask requires positive steps");
    }
    const auto masked = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    const auto visible = ggml_fp32_to_fp16(0.0F);
    std::vector<ggml_fp16_t> values(static_cast<size_t>(steps * steps), masked);
    for (int64_t row = 0; row < steps; ++row) {
        for (int64_t col = 0; col <= row; ++col) {
            values[static_cast<size_t>(row * steps + col)] = visible;
        }
    }
    return values;
}

std::vector<ggml_fp16_t> build_cached_update_mask_values(
    int64_t capacity,
    int64_t persistent_len,
    int64_t unit_len) {
    if (capacity <= 0 || persistent_len < 0 || unit_len <= 0 || persistent_len + unit_len > capacity) {
        throw std::runtime_error("DotTTS DiT cached update mask shape is invalid");
    }
    const int64_t tail_len = 2 * unit_len;
    const int64_t cols = capacity + tail_len;
    const auto masked = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    const auto visible = ggml_fp32_to_fp16(0.0F);
    std::vector<ggml_fp16_t> values(static_cast<size_t>(tail_len * cols), masked);
    for (int64_t row = 0; row < tail_len; ++row) {
        for (int64_t col = 0; col < persistent_len; ++col) {
            values[static_cast<size_t>(row * cols + col)] = visible;
        }
        if (row < unit_len) {
            for (int64_t tail_col = 0; tail_col <= row; ++tail_col) {
                values[static_cast<size_t>(row * cols + capacity + tail_col)] = visible;
            }
        } else {
            for (int64_t tail_col = 0; tail_col < tail_len; ++tail_col) {
                values[static_cast<size_t>(row * cols + capacity + tail_col)] = visible;
            }
        }
    }
    return values;
}

core::TensorValue set_dit_cache_flat_rows(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    const core::TensorValue & rows,
    const core::TensorValue & row_index) {
    core::validate_rank_between(cache, 4, 4, "DotTTS DiT cache");
    core::validate_rank_between(rows, 4, 4, "DotTTS DiT cache rows");
    if (rows.shape.dims[2] != cache.shape.dims[2] ||
        rows.shape.dims[3] != cache.shape.dims[3]) {
        throw std::runtime_error("DotTTS DiT cache row shape mismatch");
    }
    const int64_t row_count = rows.shape.dims[0] * rows.shape.dims[1];
    if (row_index.shape.rank != 1 || row_index.shape.dims[0] != row_count) {
        throw std::runtime_error("DotTTS DiT cache row indices shape mismatch");
    }
    if (row_index.type != GGML_TYPE_I32) {
        throw std::runtime_error("DotTTS DiT cache row indices must be i32");
    }
    const int64_t steps = cache.shape.dims[1];
    const int64_t row_elems = cache.shape.dims[2] * cache.shape.dims[3];
    auto flat_cache = core::reshape_tensor(ctx, cache, core::TensorShape::from_dims({cache.shape.dims[0] * steps, row_elems}));
    auto contiguous_rows = core::ensure_backend_addressable_layout(ctx, cast_tensor(ctx, rows, GGML_TYPE_F32));
    auto flat_rows = core::reshape_tensor(ctx, contiguous_rows, core::TensorShape::from_dims({row_count, row_elems}));
    auto * updated = ggml_set_rows(ctx.ggml, flat_cache.tensor, flat_rows.tensor, row_index.tensor);
    updated->src[2] = cache.tensor;
    return core::reshape_tensor(ctx, core::wrap_tensor(updated, flat_cache.shape, cache.type), cache.shape);
}

core::TensorValue view_dit_cache_banks(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    int64_t bank_start,
    int64_t capacity,
    int64_t heads,
    int64_t dim) {
    core::validate_rank_between(cache, 4, 4, "DotTTS DiT cache");
    if (bank_start < 0 || bank_start + 1 >= cache.shape.dims[0] ||
        cache.shape.dims[1] != capacity || cache.shape.dims[2] != heads || cache.shape.dims[3] != dim) {
        throw std::runtime_error("DotTTS DiT cache bank view shape mismatch");
    }
    return core::wrap_tensor(
        ggml_view_4d(
            ctx.ggml,
            cache.tensor,
            dim,
            heads,
            capacity,
            2,
            cache.tensor->nb[1],
            cache.tensor->nb[2],
            cache.tensor->nb[3],
            static_cast<size_t>(bank_start) * cache.tensor->nb[3]),
        core::TensorShape::from_dims({2, capacity, heads, dim}),
        cache.type);
}

assets::TensorStorageType derived_qkv_storage_type(
    const assets::TensorSource & source,
    std::string_view q_name,
    assets::TensorStorageType requested,
    core::BackendType backend_type) {
    if (requested != assets::TensorStorageType::Native) {
        return requested;
    }
    auto storage = assets::tensor_storage_type_for_dtype(source.require_metadata(q_name).dtype);
    if ((backend_type == core::BackendType::Vulkan || backend_type == core::BackendType::Metal) &&
        storage == assets::TensorStorageType::BF16) {
        return assets::TensorStorageType::F16;
    }
    return storage;
}

core::TensorValue load_fused_qkv_projection(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden_size,
    assets::TensorStorageType storage_type,
    core::BackendType backend_type) {
    const std::string q_name = prefix + ".attn.q_proj.weight";
    const std::string k_name = prefix + ".attn.k_proj.weight";
    const std::string v_name = prefix + ".attn.v_proj.weight";
    const auto packed_storage = derived_qkv_storage_type(source, q_name, storage_type, backend_type);
    const auto q = source.require_tensor(q_name, packed_storage, {hidden_size, hidden_size});
    const auto k = source.require_tensor(k_name, packed_storage, {hidden_size, hidden_size});
    const auto v = source.require_tensor(v_name, packed_storage, {hidden_size, hidden_size});
    if (q.type != k.type || q.type != v.type) {
        throw std::runtime_error("DotTTS DiT QKV projection tensors must use matching storage types");
    }
    std::vector<std::byte> qkv;
    qkv.reserve(q.bytes.size() + k.bytes.size() + v.bytes.size());
    qkv.insert(qkv.end(), q.bytes.begin(), q.bytes.end());
    qkv.insert(qkv.end(), k.bytes.begin(), k.bytes.end());
    qkv.insert(qkv.end(), v.bytes.begin(), v.bytes.end());
    return store.make_tensor(
        core::TensorShape::from_dims({3 * hidden_size, hidden_size}),
        q.type,
        qkv.data(),
        qkv.size());
}

void add_scaled(std::vector<float> & lhs, const std::vector<float> & rhs, float scale) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("DotTTS flow vector size mismatch");
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        lhs[i] += rhs[i] * scale;
    }
}

std::vector<float> sum_scaled(
    const std::vector<float> & base,
    const std::vector<float> & delta,
    float scale) {
    auto out = base;
    add_scaled(out, delta, scale);
    return out;
}

void append_branch_sequence(
    std::vector<float> & out,
    const std::vector<float> & prefix,
    const std::vector<float> & noise_projection,
    int64_t fm_seq_len,
    int64_t hidden_size,
    int64_t latent_patch_size) {
    const int64_t total_len = fm_seq_len + latent_patch_size;
    if (static_cast<int64_t>(prefix.size()) < fm_seq_len * hidden_size ||
        static_cast<int64_t>(noise_projection.size()) != latent_patch_size * hidden_size) {
        throw std::runtime_error("DotTTS flow branch sequence shape mismatch");
    }
    out.insert(out.end(), prefix.begin(), prefix.begin() + static_cast<std::ptrdiff_t>(fm_seq_len * hidden_size));
    out.insert(out.end(), noise_projection.begin(), noise_projection.end());
    if (static_cast<int64_t>(out.size()) % (total_len * hidden_size) != 0) {
        throw std::runtime_error("DotTTS flow branch sequence assembly failed");
    }
}

std::vector<float> cfg_combine_velocity(
    const DotsVelocityOutput & velocity,
    int64_t latent_patch_size,
    float guidance_scale) {
    const int64_t latent_dim = velocity.latent_dim;
    if (velocity.frames != latent_patch_size ||
        static_cast<int64_t>(velocity.values.size()) != 2 * latent_patch_size * latent_dim) {
        throw std::runtime_error("DotTTS flow CFG velocity shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(latent_patch_size * latent_dim), 0.0F);
    const int64_t branch_stride = latent_patch_size * latent_dim;
    for (int64_t frame = 0; frame < latent_patch_size; ++frame) {
        for (int64_t dim = 0; dim < latent_dim; ++dim) {
            const size_t index = static_cast<size_t>(frame * latent_dim + dim);
            const float cond = velocity.values[index];
            const float uncond = velocity.values[static_cast<size_t>(branch_stride) + index];
            out[static_cast<size_t>(frame * latent_dim + dim)] = cond + guidance_scale * (cond - uncond);
        }
    }
    return out;
}

class LinearProjectionRunner {
public:
    LinearProjectionRunner(
        std::shared_ptr<const DotFlowWeights> weights,
        const modules::LinearWeights DotFlowWeights::* projection,
        int64_t in_features,
        int64_t out_features,
        const char * label)
        : weights_(std::move(weights)),
          projection_(projection),
          in_features_(in_features),
          out_features_(out_features),
          label_(label) {}

    ~LinearProjectionRunner() { release_graph(); }

    DotsProjectedSequence run(const std::vector<float> & input, int64_t steps) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (steps <= 0 || static_cast<int64_t>(input.size()) != steps * in_features_) {
            throw std::runtime_error(std::string(label_) + " input shape mismatch");
        }
        ensure_graph(steps);
        ggml_backend_tensor_set(input_, input.data(), 0, input.size() * sizeof(float));
        if (core::compute_graph(*weights_->execution_context, graph_, plan_, label_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string(label_) + " graph compute failed");
        }
        DotsProjectedSequence out;
        out.steps = steps;
        out.hidden_size = out_features_;
        out.values = core::read_tensor_f32(output_);
        return out;
    }

    void release_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(weights_->execution_context->backend(), graph_);
        }
        gallocr_.reset();
        ggml_.reset();
        plan_.reset();
        graph_ = nullptr;
        input_ = nullptr;
        output_ = nullptr;
        steps_ = 0;
    }

private:
    void ensure_graph(int64_t steps) {
        if (ggml_ != nullptr && steps_ == steps) {
            return;
        }
        release_graph();
        ggml_init_params params{kSmallGraphContextBytes, nullptr, true};
        ggml_.reset(ggml_init(params));
        if (ggml_ == nullptr) {
            throw std::runtime_error(std::string("failed to initialize ") + label_ + " graph context");
        }
        core::ModuleBuildContext build_ctx{ggml_.get(), label_, weights_->execution_context->backend_type()};
        auto input = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, steps, in_features_}));
        input_ = input.tensor;
        auto output = modules::LinearModule({in_features_, out_features_, true}).build(build_ctx, input, weights_.get()->*projection_);
        output_ = output.tensor;
        graph_ = ggml_new_graph_custom(ggml_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        core::validate_backend_graph_supported(weights_->execution_context->backend(), graph_, label_);
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_->execution_context->backend())));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_.get(), graph_) || !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            release_graph();
            throw std::runtime_error(std::string("failed to allocate ") + label_ + " graph memory");
        }
        core::prepare_host_graph_plan(*weights_->execution_context, graph_, plan_);
        steps_ = steps;
    }

    std::shared_ptr<const DotFlowWeights> weights_;
    const modules::LinearWeights DotFlowWeights::* projection_ = nullptr;
    int64_t in_features_ = 0;
    int64_t out_features_ = 0;
    const char * label_ = nullptr;
    std::mutex mutex_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ggml_;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
    ggml_cgraph * graph_ = nullptr;
    core::HostGraphPlan plan_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    int64_t steps_ = 0;
};

class SpeakerProjectionRunner {
public:
    explicit SpeakerProjectionRunner(std::shared_ptr<const DotFlowWeights> weights)
        : weights_(std::move(weights)) {}

    ~SpeakerProjectionRunner() { release_graph(); }

    DotsProjectedSequence run(const std::vector<float> & speaker) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto & config = weights_->config;
        if (static_cast<int64_t>(speaker.size()) != config.campplus_embedding_size) {
            throw std::runtime_error("DotTTS speaker projection input shape mismatch");
        }
        ensure_graph();
        ggml_backend_tensor_set(input_, speaker.data(), 0, speaker.size() * sizeof(float));
        if (core::compute_graph(*weights_->execution_context, graph_, plan_, "dots_tts.flow.speaker_proj") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DotTTS speaker projection graph compute failed");
        }
        DotsProjectedSequence out;
        out.steps = 1;
        out.hidden_size = config.dit.hidden_size;
        out.values = core::read_tensor_f32(output_);
        return out;
    }

    void release_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(weights_->execution_context->backend(), graph_);
        }
        gallocr_.reset();
        ggml_.reset();
        plan_.reset();
        graph_ = nullptr;
        input_ = nullptr;
        output_ = nullptr;
    }

private:
    void ensure_graph() {
        if (ggml_ != nullptr) {
            return;
        }
        ggml_init_params params{kSmallGraphContextBytes, nullptr, true};
        ggml_.reset(ggml_init(params));
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize DotTTS speaker projection graph context");
        }
        core::ModuleBuildContext build_ctx{ggml_.get(), "dots_tts.flow.speaker_proj", weights_->execution_context->backend_type()};
        auto input = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, weights_->config.campplus_embedding_size}));
        input_ = input.tensor;
        auto output = modules::LinearModule({weights_->config.campplus_embedding_size, weights_->config.dit.hidden_size, true})
                          .build(build_ctx, input, weights_->xvec_proj);
        output = modules::LayerNormModule({weights_->config.dit.hidden_size, kLayerNormEps, true, true})
                     .build(build_ctx, output, weights_->xvec_norm);
        output_ = output.tensor;
        graph_ = ggml_new_graph_custom(ggml_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        core::validate_backend_graph_supported(weights_->execution_context->backend(), graph_, "dots_tts.flow.speaker_proj");
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_->execution_context->backend())));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_.get(), graph_) || !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate DotTTS speaker projection graph memory");
        }
        core::prepare_host_graph_plan(*weights_->execution_context, graph_, plan_);
    }

    std::shared_ptr<const DotFlowWeights> weights_;
    std::mutex mutex_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ggml_;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
    ggml_cgraph * graph_ = nullptr;
    core::HostGraphPlan plan_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

class ModulationRunner {
public:
    explicit ModulationRunner(std::shared_ptr<const DotFlowWeights> weights)
        : weights_(std::move(weights)) {}

    ~ModulationRunner() { release_graph(); }

    DotModulationOutput run(
        const std::vector<float> & timesteps,
        const std::vector<float> & durations,
        const std::vector<float> & speaker_condition,
        DotsFlowRuntimeStats * runtime_stats = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto & config = weights_->config;
        const int64_t rows = static_cast<int64_t>(timesteps.size());
        const int64_t width = (6 * config.dit.num_layers + 2) * config.dit.hidden_size;
        if (rows <= 0 ||
            (!durations.empty() && static_cast<int64_t>(durations.size()) != rows) ||
            static_cast<int64_t>(speaker_condition.size()) != rows * config.dit.hidden_size) {
            throw std::runtime_error("DotTTS modulation graph input shape mismatch");
        }
        const auto graph_start = Clock::now();
        ensure_graph(rows, !durations.empty());
        const double graph_ms = engine::debug::elapsed_ms(graph_start);

        const auto upload_start = Clock::now();
        ggml_backend_tensor_set(timesteps_, timesteps.data(), 0, timesteps.size() * sizeof(float));
        if (!durations.empty()) {
            ggml_backend_tensor_set(durations_, durations.data(), 0, durations.size() * sizeof(float));
        }
        ggml_backend_tensor_set(speaker_condition_, speaker_condition.data(), 0, speaker_condition.size() * sizeof(float));
        const double upload_ms = engine::debug::elapsed_ms(upload_start);

        const auto compute_start = Clock::now();
        if (core::compute_graph(*weights_->execution_context, graph_, plan_, "dots_tts.flow.modulation") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DotTTS modulation graph compute failed");
        }
        const double compute_ms = engine::debug::elapsed_ms(compute_start);
        auto values = core::read_tensor_f32(output_);
        engine::core::round_f32_to_bf16_in_place(values);
        core::write_tensor_f32(cache_, values);

        DotModulationOutput out;
        out.rows = rows;
        out.width = width;
        out.backend_value = cache_;
        if (runtime_stats != nullptr) {
            runtime_stats->modulation_graph_ms += graph_ms;
            runtime_stats->modulation_input_upload_ms += upload_ms;
            runtime_stats->modulation_compute_ms += compute_ms;
        }
        return out;
    }

    void release_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(weights_->execution_context->backend(), graph_);
        }
        gallocr_.reset();
        ggml_.reset();
        plan_.reset();
        graph_ = nullptr;
        timesteps_ = nullptr;
        durations_ = nullptr;
        speaker_condition_ = nullptr;
        output_ = nullptr;
        cache_ = {};
        if (cache_buffer_ != nullptr) {
            ggml_backend_buffer_free(cache_buffer_);
            cache_buffer_ = nullptr;
        }
        cache_ctx_.reset();
        rows_ = 0;
        has_durations_ = false;
    }

private:
    void ensure_graph(int64_t rows, bool has_durations) {
        if (ggml_ != nullptr && rows_ == rows && has_durations_ == has_durations) {
            return;
        }
        release_graph();
        ggml_init_params params{kSmallGraphContextBytes, nullptr, true};
        ggml_.reset(ggml_init(params));
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize DotTTS modulation graph context");
        }
        core::ModuleBuildContext build_ctx{ggml_.get(), "dots_tts.flow.modulation", weights_->execution_context->backend_type()};
        auto timesteps = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({rows}));
        timesteps_ = timesteps.tensor;
        std::optional<core::TensorValue> durations;
        if (has_durations) {
            durations = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({rows}));
            durations_ = durations->tensor;
        }
        auto speaker = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({rows, weights_->config.dit.hidden_size}));
        speaker_condition_ = speaker.tensor;
        auto output = build_modulation_graph(build_ctx, *weights_, timesteps, durations, speaker);
        output_ = output.tensor;
        ggml_init_params cache_params{kSmallGraphContextBytes, nullptr, true};
        cache_ctx_.reset(ggml_init(cache_params));
        if (cache_ctx_ == nullptr) {
            release_graph();
            throw std::runtime_error("failed to initialize DotTTS modulation cache context");
        }
        core::ModuleBuildContext cache_build_ctx{cache_ctx_.get(), "dots_tts.flow.modulation.cache", weights_->execution_context->backend_type()};
        cache_ = core::make_tensor(cache_build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({rows, (6 * weights_->config.dit.num_layers + 2) * weights_->config.dit.hidden_size}));
        cache_buffer_ = ggml_backend_alloc_ctx_tensors(cache_ctx_.get(), weights_->execution_context->backend());
        if (cache_buffer_ == nullptr) {
            release_graph();
            throw std::runtime_error("failed to allocate DotTTS modulation cache memory");
        }
        graph_ = ggml_new_graph_custom(ggml_.get(), 262144, false);
        ggml_build_forward_expand(graph_, output_);
        core::validate_backend_graph_supported(weights_->execution_context->backend(), graph_, "dots_tts.flow.modulation");
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_->execution_context->backend())));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_.get(), graph_) || !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate DotTTS modulation graph memory");
        }
        core::prepare_host_graph_plan(*weights_->execution_context, graph_, plan_);
        rows_ = rows;
        has_durations_ = has_durations;
    }

    std::shared_ptr<const DotFlowWeights> weights_;
    std::mutex mutex_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ggml_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> cache_ctx_;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
    ggml_cgraph * graph_ = nullptr;
    core::HostGraphPlan plan_;
    ggml_tensor * timesteps_ = nullptr;
    ggml_tensor * durations_ = nullptr;
    ggml_tensor * speaker_condition_ = nullptr;
    ggml_tensor * output_ = nullptr;
    core::TensorValue cache_;
    ggml_backend_buffer_t cache_buffer_ = nullptr;
    int64_t rows_ = 0;
    bool has_durations_ = false;
};

class VelocityRunner {
public:
    explicit VelocityRunner(std::shared_ptr<const DotFlowWeights> weights)
        : weights_(std::move(weights)) {}

    ~VelocityRunner() { release_graph(); }

    DotsVelocityOutput run(
        const std::vector<float> & sequence,
        int64_t steps,
        const std::vector<float> & timesteps,
        const std::vector<float> & durations,
        const std::vector<float> & speaker_condition,
        int64_t batch_size,
        const std::vector<int32_t> & positions,
        const std::vector<uint8_t> & attention_mask,
        const DotModulationOutput * modulations = nullptr,
        int64_t modulation_row_start = 0,
        int64_t output_start = 0,
        int64_t output_length = 0,
        DotsFlowRuntimeStats * runtime_stats = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto & config = weights_->config;
        const bool use_precomputed_modulations = modulations != nullptr;
        if (batch_size <= 0 || steps <= 0) {
            throw std::runtime_error("DotTTS velocity graph requires positive batch and step counts");
        }
        if (static_cast<int64_t>(sequence.size()) != batch_size * steps * config.dit.hidden_size) {
            throw std::runtime_error("DotTTS velocity graph input shape mismatch");
        }
        if (use_precomputed_modulations) {
            if (!modulations->backend_value.valid() ||
                modulations->width != (6 * config.dit.num_layers + 2) * config.dit.hidden_size ||
                modulation_row_start < 0 ||
                modulation_row_start + batch_size > modulations->rows) {
                throw std::runtime_error("DotTTS velocity modulation input shape mismatch");
            }
        } else if (static_cast<int64_t>(timesteps.size()) != batch_size ||
                   (!durations.empty() && static_cast<int64_t>(durations.size()) != batch_size) ||
                   static_cast<int64_t>(speaker_condition.size()) != batch_size * config.dit.hidden_size) {
            throw std::runtime_error("DotTTS velocity conditioning input shape mismatch");
        }
        if (!positions.empty() && static_cast<int64_t>(positions.size()) != steps) {
            throw std::runtime_error("DotTTS velocity position id size mismatch");
        }
        if (!attention_mask.empty() && static_cast<int64_t>(attention_mask.size()) != batch_size * steps * steps) {
            throw std::runtime_error("DotTTS velocity attention mask size mismatch");
        }
        if (output_start < 0 || output_length < 0 || output_start + output_length > steps) {
            throw std::runtime_error("DotTTS velocity output slice is out of range");
        }
        const auto graph_start = Clock::now();
        ensure_graph(
            batch_size,
            steps,
            !attention_mask.empty(),
            !durations.empty(),
            use_precomputed_modulations,
            use_precomputed_modulations ? modulations->backend_value : core::TensorValue{},
            output_start,
            output_length);
        const double graph_ms = engine::debug::elapsed_ms(graph_start);

        const auto upload_start = Clock::now();
        ggml_backend_tensor_set(sequence_, sequence.data(), 0, sequence.size() * sizeof(float));
        if (use_precomputed_modulations) {
            modulation_indices_values_.resize(static_cast<size_t>(batch_size));
            for (int64_t i = 0; i < batch_size; ++i) {
                modulation_indices_values_[static_cast<size_t>(i)] = static_cast<int32_t>(modulation_row_start + i);
            }
            ggml_backend_tensor_set(modulation_indices_, modulation_indices_values_.data(), 0, modulation_indices_values_.size() * sizeof(int32_t));
        } else {
            ggml_backend_tensor_set(timesteps_, timesteps.data(), 0, timesteps.size() * sizeof(float));
            if (!durations.empty()) {
                ggml_backend_tensor_set(durations_, durations.data(), 0, durations.size() * sizeof(float));
            }
            ggml_backend_tensor_set(speaker_condition_, speaker_condition.data(), 0, speaker_condition.size() * sizeof(float));
        }
        std::vector<int32_t> position_values = positions;
        if (position_values.empty()) {
            position_values.assign(static_cast<size_t>(steps), 0);
            for (int64_t i = 0; i < steps; ++i) {
                position_values[static_cast<size_t>(i)] = static_cast<int32_t>(i);
            }
        }
        ggml_backend_tensor_set(positions_, position_values.data(), 0, position_values.size() * sizeof(int32_t));
        if (!attention_mask.empty()) {
            attention_mask_values_.assign(static_cast<size_t>(batch_size * steps * steps), ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
            for (size_t i = 0; i < attention_mask.size(); ++i) {
                if (attention_mask[i] != 0) {
                    attention_mask_values_[i] = ggml_fp32_to_fp16(0.0F);
                }
            }
            ggml_backend_tensor_set(attention_mask_, attention_mask_values_.data(), 0, attention_mask_values_.size() * sizeof(ggml_fp16_t));
        }
        const double upload_ms = engine::debug::elapsed_ms(upload_start);
        const auto compute_start = Clock::now();
        if (core::compute_graph(*weights_->execution_context, graph_, plan_, "dots_tts.flow.velocity") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DotTTS velocity graph compute failed");
        }
        const double compute_ms = engine::debug::elapsed_ms(compute_start);
        DotsVelocityOutput out;
        out.frames = output_length > 0 ? output_length : steps;
        out.latent_dim = config.latent_dim;
        const auto read_start = Clock::now();
        out.values = core::read_tensor_f32(output_);
        const double read_ms = engine::debug::elapsed_ms(read_start);
        if (runtime_stats != nullptr) {
            ++runtime_stats->velocity_calls;
            runtime_stats->velocity_graph_ms += graph_ms;
            runtime_stats->velocity_input_upload_ms += upload_ms;
            runtime_stats->velocity_compute_ms += compute_ms;
            runtime_stats->velocity_output_read_ms += read_ms;
        }
        return out;
    }

    void release_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(weights_->execution_context->backend(), graph_);
        }
        gallocr_.reset();
        ggml_.reset();
        plan_.reset();
        graph_ = nullptr;
        sequence_ = nullptr;
        timesteps_ = nullptr;
        durations_ = nullptr;
        speaker_condition_ = nullptr;
        modulation_indices_ = nullptr;
        modulation_indices_values_.clear();
        positions_ = nullptr;
        attention_mask_ = nullptr;
        attention_mask_values_.clear();
        output_ = nullptr;
        batch_size_ = 0;
        steps_ = 0;
        output_start_ = 0;
        output_length_ = 0;
        has_attention_mask_ = false;
        has_durations_ = false;
        use_precomputed_modulations_ = false;
        modulation_source_tensor_ = nullptr;
    }

private:
    void ensure_graph(
        int64_t batch_size,
        int64_t steps,
        bool has_attention_mask,
        bool has_durations,
        bool use_precomputed_modulations,
        const core::TensorValue & all_modulations,
        int64_t output_start,
        int64_t output_length) {
        if (ggml_ != nullptr && batch_size_ == batch_size && steps_ == steps &&
            has_attention_mask_ == has_attention_mask && has_durations_ == has_durations &&
            use_precomputed_modulations_ == use_precomputed_modulations &&
            modulation_source_tensor_ == all_modulations.tensor &&
            output_start_ == output_start && output_length_ == output_length) {
            return;
        }
        release_graph();
        ggml_init_params params{kLargeGraphContextBytes, nullptr, true};
        ggml_.reset(ggml_init(params));
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize DotTTS velocity graph context");
        }
        core::ModuleBuildContext build_ctx{ggml_.get(), "dots_tts.flow.velocity", weights_->execution_context->backend_type()};
        auto sequence = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_size, steps, weights_->config.dit.hidden_size}));
        std::optional<core::TensorValue> timesteps;
        std::optional<core::TensorValue> durations;
        std::optional<core::TensorValue> speaker;
        std::optional<core::TensorValue> modulations;
        if (use_precomputed_modulations) {
            auto modulation_indices = core::make_tensor(build_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({batch_size}));
            modulation_indices_ = modulation_indices.tensor;
            modulations = core::wrap_tensor(
                ggml_get_rows(
                    build_ctx.ggml,
                    core::ensure_backend_addressable_layout(build_ctx, all_modulations).tensor,
                    modulation_indices.tensor),
                core::TensorShape::from_dims({batch_size, all_modulations.shape.dims[1]}),
                GGML_TYPE_F32);
        } else {
            timesteps = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_size}));
            timesteps_ = timesteps->tensor;
            speaker = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_size, weights_->config.dit.hidden_size}));
            speaker_condition_ = speaker->tensor;
        }
        if (!use_precomputed_modulations && has_durations) {
            durations = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_size}));
            durations_ = durations->tensor;
        }
        auto positions = core::make_tensor(build_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
        std::optional<core::TensorValue> attention_mask;
        if (has_attention_mask) {
            attention_mask = core::make_tensor(build_ctx, GGML_TYPE_F16, core::TensorShape::from_dims({batch_size, 1, steps, steps}));
            attention_mask_ = attention_mask->tensor;
        }
        sequence_ = sequence.tensor;
        positions_ = positions.tensor;
        auto output = build_velocity_graph(
            build_ctx,
            *weights_,
            sequence,
            timesteps.value_or(core::TensorValue{}),
            durations,
            speaker.value_or(core::TensorValue{}),
            positions,
            attention_mask,
            modulations,
            output_start,
            output_length);
        output_ = output.tensor;
        graph_ = ggml_new_graph_custom(ggml_.get(), 262144, false);
        ggml_build_forward_expand(graph_, output_);
        core::validate_backend_graph_supported(weights_->execution_context->backend(), graph_, "dots_tts.flow.velocity");
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_->execution_context->backend())));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_.get(), graph_) || !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate DotTTS velocity graph memory");
        }
        core::prepare_host_graph_plan(*weights_->execution_context, graph_, plan_);
        batch_size_ = batch_size;
        steps_ = steps;
        has_attention_mask_ = has_attention_mask;
        has_durations_ = has_durations;
        use_precomputed_modulations_ = use_precomputed_modulations;
        modulation_source_tensor_ = all_modulations.tensor;
        output_start_ = output_start;
        output_length_ = output_length;
    }

    std::shared_ptr<const DotFlowWeights> weights_;
    std::mutex mutex_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ggml_;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
    ggml_cgraph * graph_ = nullptr;
    core::HostGraphPlan plan_;
    ggml_tensor * sequence_ = nullptr;
    ggml_tensor * timesteps_ = nullptr;
    ggml_tensor * durations_ = nullptr;
    ggml_tensor * speaker_condition_ = nullptr;
    ggml_tensor * modulation_indices_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    std::vector<ggml_fp16_t> attention_mask_values_;
    std::vector<int32_t> modulation_indices_values_;
    int64_t batch_size_ = 0;
    int64_t steps_ = 0;
    int64_t output_start_ = 0;
    int64_t output_length_ = 0;
    bool has_attention_mask_ = false;
    bool has_durations_ = false;
    bool use_precomputed_modulations_ = false;
    ggml_tensor * modulation_source_tensor_ = nullptr;
};

DotFlowSharedRuntime::DotFlowSharedRuntime(std::shared_ptr<const DotFlowWeights> weights)
    : weights(std::move(weights)),
      hidden_projection(std::make_unique<LinearProjectionRunner>(
          this->weights,
          &DotFlowWeights::hidden_proj,
          this->weights->config.llm.hidden_size,
          this->weights->config.dit.hidden_size,
          "dots_tts.flow.hidden_proj")),
      latent_projection(std::make_unique<LinearProjectionRunner>(
          this->weights,
          &DotFlowWeights::latent_proj,
          this->weights->config.latent_dim,
          this->weights->config.dit.hidden_size,
          "dots_tts.flow.latent_proj")),
      coordinate_projection(std::make_unique<LinearProjectionRunner>(
          this->weights,
          &DotFlowWeights::coordinate_proj,
          this->weights->config.latent_dim,
          this->weights->config.dit.hidden_size,
          "dots_tts.flow.coordinate_proj")),
      speaker_projection(std::make_unique<SpeakerProjectionRunner>(this->weights)),
      modulation(std::make_unique<ModulationRunner>(this->weights)),
      velocity(std::make_unique<VelocityRunner>(this->weights)) {}

DotFlowSharedRuntime::~DotFlowSharedRuntime() = default;

std::unique_ptr<DotFlowSharedRuntime> load_flow_shared_runtime(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    DotsConfig config,
    assets::TensorStorageType storage_type) {
    return std::make_unique<DotFlowSharedRuntime>(load_weights(
        std::move(source),
        backend,
        std::move(config),
        storage_type));
}

DotsProjectedSequence run_hidden_projection(DotFlowSharedRuntime & runtime, const std::vector<float> & hidden, int64_t steps) {
    return runtime.hidden_projection->run(hidden, steps);
}

DotsProjectedSequence run_latent_projection(DotFlowSharedRuntime & runtime, const std::vector<float> & latents, int64_t frames) {
    return runtime.latent_projection->run(latents, frames);
}

DotsProjectedSequence run_coordinate_projection(DotFlowSharedRuntime & runtime, const std::vector<float> & latents, int64_t frames) {
    return runtime.coordinate_projection->run(latents, frames);
}

DotsProjectedSequence run_speaker_projection(DotFlowSharedRuntime & runtime, const std::vector<float> & speaker) {
    return runtime.speaker_projection->run(speaker);
}

DotModulationOutput run_modulation(
    DotFlowSharedRuntime & runtime,
    const std::vector<float> & timesteps,
    const std::vector<float> & durations,
    const std::vector<float> & speaker_condition,
    DotsFlowRuntimeStats * runtime_stats) {
    return runtime.modulation->run(timesteps, durations, speaker_condition, runtime_stats);
}

DotsVelocityOutput run_velocity(
    DotFlowSharedRuntime & runtime,
    const std::vector<float> & sequence,
    int64_t steps,
    const std::vector<float> & timesteps,
    const std::vector<float> & durations,
    const std::vector<float> & speaker_condition,
    int64_t batch_size,
    const std::vector<int32_t> & positions,
    const std::vector<uint8_t> & attention_mask,
    const DotModulationOutput * modulations,
    int64_t modulation_row_start,
    int64_t output_start,
    int64_t output_length,
    DotsFlowRuntimeStats * runtime_stats) {
    return runtime.velocity->run(
        sequence,
        steps,
        timesteps,
        durations,
        speaker_condition,
        batch_size,
        positions,
        attention_mask,
        modulations,
        modulation_row_start,
        output_start,
        output_length,
        runtime_stats);
}

void release_flow_graphs(DotFlowSharedRuntime & runtime) {
    runtime.hidden_projection->release_graph();
    runtime.latent_projection->release_graph();
    runtime.coordinate_projection->release_graph();
    runtime.speaker_projection->release_graph();
    runtime.modulation->release_graph();
    runtime.velocity->release_graph();
}

}  // namespace engine::models::dots_tts::detail

namespace engine::models::dots_tts {

struct DotsFlowDecodeState::Impl {
    detail::FlowDecodeCacheState state;
};

DotsFlowDecodeState::DotsFlowDecodeState()
    : impl_(std::make_unique<Impl>()) {}

DotsFlowDecodeState::~DotsFlowDecodeState() = default;
DotsFlowDecodeState::DotsFlowDecodeState(DotsFlowDecodeState &&) noexcept = default;
DotsFlowDecodeState & DotsFlowDecodeState::operator=(DotsFlowDecodeState &&) noexcept = default;

struct DotsFlowComponent::Impl {
    explicit Impl(std::unique_ptr<detail::DotFlowSharedRuntime> runtime)
        : runtime(std::move(runtime)) {}

    std::unique_ptr<detail::DotFlowSharedRuntime> runtime;
};

DotsFlowComponent DotsFlowComponent::load_from_tensor_source(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    DotsConfig config,
    assets::TensorStorageType weight_storage_type) {
    DotsFlowComponent component;
    component.impl_ = std::make_unique<Impl>(detail::load_flow_shared_runtime(
        std::move(source),
        backend,
        std::move(config),
        weight_storage_type));
    return component;
}

DotsFlowComponent::DotsFlowComponent() = default;
DotsFlowComponent::~DotsFlowComponent() = default;
DotsFlowComponent::DotsFlowComponent(DotsFlowComponent &&) noexcept = default;
DotsFlowComponent & DotsFlowComponent::operator=(DotsFlowComponent &&) noexcept = default;

bool DotsFlowComponent::is_loaded() const noexcept {
    return impl_ != nullptr && impl_->runtime != nullptr;
}

DotsProjectedSequence DotsFlowComponent::project_llm_hidden(const std::vector<float> & hidden, int64_t steps) const {
    if (impl_ == nullptr || impl_->runtime == nullptr) {
        throw std::runtime_error("DotTTS flow component is not initialized");
    }
    return detail::run_hidden_projection(*impl_->runtime, hidden, steps);
}

DotsProjectedSequence DotsFlowComponent::project_latents(const std::vector<float> & latents, int64_t frames) const {
    if (impl_ == nullptr || impl_->runtime == nullptr) {
        throw std::runtime_error("DotTTS flow component is not initialized");
    }
    return detail::run_latent_projection(*impl_->runtime, latents, frames);
}

DotsProjectedSequence DotsFlowComponent::project_speaker(const std::vector<float> & speaker) const {
    if (impl_ == nullptr || impl_->runtime == nullptr) {
        throw std::runtime_error("DotTTS flow component is not initialized");
    }
    return detail::run_speaker_projection(*impl_->runtime, speaker);
}

DotsVelocityOutput DotsFlowComponent::predict_velocity(
    const std::vector<float> & sequence,
    int64_t steps,
    const std::vector<float> & timesteps,
    const std::vector<float> & durations,
    const std::vector<float> & speaker_condition,
    int64_t batch_size,
    const std::vector<int32_t> & positions,
    const std::vector<uint8_t> & attention_mask) const {
    if (impl_ == nullptr || impl_->runtime == nullptr) {
        throw std::runtime_error("DotTTS flow component is not initialized");
    }
    return detail::run_velocity(
        *impl_->runtime,
        sequence,
        steps,
        timesteps,
        durations,
        speaker_condition,
        batch_size,
        positions,
        attention_mask);
}

DotsLatentMatrix DotsFlowComponent::decode_next_soar(const DotsFlowDecodeRequest & request) const {
    if (impl_ == nullptr || impl_->runtime == nullptr) {
        throw std::runtime_error("DotTTS flow component is not initialized");
    }
    auto * state = request.decode_state != nullptr ? &request.decode_state->impl_->state : nullptr;
    return detail::decode_next_soar(*impl_->runtime, request, state);
}

DotsLatentMatrix DotsFlowComponent::decode_next_meanflow(const DotsFlowDecodeRequest & request) const {
    if (impl_ == nullptr || impl_->runtime == nullptr) {
        throw std::runtime_error("DotTTS flow component is not initialized");
    }
    auto * state = request.decode_state != nullptr ? &request.decode_state->impl_->state : nullptr;
    return detail::decode_next_meanflow(*impl_->runtime, request, state);
}

void DotsFlowComponent::release_runtime_graphs() {
    if (impl_ == nullptr || impl_->runtime == nullptr) {
        return;
    }
    detail::release_flow_graphs(*impl_->runtime);
}

}  // namespace engine::models::dots_tts
