#include "engine/models/stable_audio/foundation/rf_dit.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/sampling/torch_random.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace engine::models::stable_audio::foundation {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kRfWeightContextBytes = 3600ull * 1024ull * 1024ull;
constexpr int64_t kTimestepFeaturesDim = 256;
constexpr float kPi = 3.14159265358979323846F;
constexpr float kTransformerNormEps = 1.0e-5F;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

modules::LinearWeights load_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_features,
    int64_t in_features,
    bool use_bias) {
    modules::LinearWeights weights;
    weights.weight = store.load_tensor(source, prefix + ".weight", storage_type, {out_features, in_features});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return weights;
}

modules::LinearWeights load_linear_as_shape(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & tensor_name,
    assets::TensorStorageType storage_type,
    int64_t out_features,
    int64_t in_features,
    bool use_bias) {
    modules::LinearWeights weights;
    weights.weight = store.load_tensor_as_shape(
        source,
        tensor_name + ".weight",
        storage_type,
        {out_features, in_features, 1},
        core::TensorShape::from_dims({out_features, in_features}));
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, tensor_name + ".bias", {out_features});
    }
    return weights;
}

int64_t head_dim(const StableAudioConfig & config) {
    if (config.num_heads <= 0 || config.embed_dim % config.num_heads != 0) {
        throw std::runtime_error("Stable Audio Foundation RF DiT head dimensions are invalid");
    }
    return config.embed_dim / config.num_heads;
}

int64_t context_heads(const StableAudioConfig & config) {
    const int64_t dim = head_dim(config);
    if (config.cond_token_dim <= 0 || config.cond_token_dim % dim != 0) {
        throw std::runtime_error("Stable Audio Foundation RF DiT context head dimensions are invalid");
    }
    return config.cond_token_dim / dim;
}

int64_t ff_inner_dim(const StableAudioConfig & config) {
    return config.embed_dim * 4;
}

core::TensorValue ensure_contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    return core::ensure_backend_addressable_layout(ctx, input);
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    return core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, input),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue repeat_kv_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t repeats) {
    if (repeats == 1) {
        return input;
    }
    const auto contiguous = ensure_contiguous(ctx, input);
    const int64_t batch = contiguous.shape.dims[0];
    const int64_t kv_heads = contiguous.shape.dims[1];
    const int64_t steps = contiguous.shape.dims[2];
    const int64_t dim = contiguous.shape.dims[3];
    auto expanded = core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({batch, kv_heads, 1, steps * dim}));
    expanded = modules::RepeatModule({core::TensorShape::from_dims({batch, kv_heads, repeats, steps * dim})})
                   .build(ctx, expanded);
    expanded = ensure_contiguous(ctx, expanded);
    return core::reshape_tensor(ctx, expanded, core::TensorShape::from_dims({batch, kv_heads * repeats, steps, dim}));
}

core::TensorValue layer_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & gamma,
    const core::TensorValue & beta) {
    return modules::LayerNormModule({input.shape.last_dim(), kTransformerNormEps, true, true})
        .build(ctx, input, {gamma, beta});
}

core::TensorValue sub_tensor(core::ModuleBuildContext & ctx, const core::TensorValue & lhs, const core::TensorValue & rhs) {
    return core::wrap_tensor(
        ggml_sub(ctx.ggml, ensure_contiguous(ctx, lhs).tensor, ensure_contiguous(ctx, rhs).tensor),
        lhs.shape,
        GGML_TYPE_F32);
}

core::TensorValue div_tensor(core::ModuleBuildContext & ctx, const core::TensorValue & lhs, const core::TensorValue & rhs) {
    return core::wrap_tensor(
        ggml_div(ctx.ggml, ensure_contiguous(ctx, lhs).tensor, ensure_contiguous(ctx, rhs).tensor),
        lhs.shape,
        GGML_TYPE_F32);
}

core::TensorValue repeat_like(core::ModuleBuildContext & ctx, const core::TensorValue & input, const core::TensorValue & like) {
    return core::wrap_tensor(ggml_repeat(ctx.ggml, input.tensor, like.tensor), like.shape, GGML_TYPE_F32);
}

core::TensorValue scale_bias_tensor(core::ModuleBuildContext & ctx, const core::TensorValue & input, float scale, float bias) {
    return core::wrap_tensor(ggml_scale_bias(ctx.ggml, ensure_contiguous(ctx, input).tensor, scale, bias), input.shape, GGML_TYPE_F32);
}

core::TensorValue build_timestep_embedding(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & features,
    const FoundationRFDiTWeights & weights,
    const StableAudioConfig & config) {
    auto hidden = modules::LinearModule({kTimestepFeaturesDim, config.embed_dim, true, GGML_PREC_F32})
                      .build(ctx, features, weights.timestep_embed_0);
    hidden = modules::SiluModule{}.build(ctx, hidden);
    return modules::LinearModule({config.embed_dim, config.embed_dim, true, GGML_PREC_F32})
        .build(ctx, hidden, weights.timestep_embed_2);
}

core::TensorValue rf_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim) {
    const auto q = ensure_contiguous(ctx, q_heads);
    const auto k = ensure_contiguous(ctx, k_heads);
    const auto v = ensure_contiguous(ctx, v_heads);
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q.tensor,
        k.tensor,
        v.tensor,
        nullptr,
        1.0F / std::sqrt(static_cast<float>(dim)),
        0.0F,
        0.0F);
    ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
    return core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[2], q.shape.dims[1], dim}),
        GGML_TYPE_F32);
}

core::TensorValue apply_padding_to_v(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & v_heads,
    const core::TensorValue & padding_mask) {
    const auto mask = modules::RepeatModule({v_heads.shape}).build(ctx, padding_mask);
    return modules::MulModule{}.build(ctx, v_heads, mask);
}

core::TensorValue self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue & padding_mask,
    const FoundationRfAttentionWeights & weights,
    const StableAudioConfig & config) {
    const int64_t dim = head_dim(config);
    auto qkv = modules::LinearModule({config.embed_dim, config.embed_dim * 3, false, GGML_PREC_F32})
                   .build(ctx, input, weights.to_qkv);
    auto q = modules::SliceModule({2, 0, config.embed_dim}).build(ctx, qkv);
    auto k = modules::SliceModule({2, config.embed_dim, config.embed_dim}).build(ctx, qkv);
    auto v = modules::SliceModule({2, 2 * config.embed_dim, config.embed_dim}).build(ctx, qkv);
    q = reshape_heads(ctx, q, config.num_heads, dim);
    k = reshape_heads(ctx, k, config.num_heads, dim);
    v = reshape_heads(ctx, v, config.num_heads, dim);
    q = modules::RoPEModule({std::max<int64_t>(dim / 2, 32), GGML_ROPE_TYPE_NEOX, 10000.0F}).build(ctx, q, positions);
    k = modules::RoPEModule({std::max<int64_t>(dim / 2, 32), GGML_ROPE_TYPE_NEOX, 10000.0F}).build(ctx, k, positions);
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    v_heads = apply_padding_to_v(ctx, v_heads, padding_mask);
    auto context = rf_attention(ctx, q_heads, k_heads, v_heads, dim);
    context = core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, context),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.embed_dim}));
    return modules::LinearModule({config.embed_dim, config.embed_dim, false, GGML_PREC_F32})
        .build(ctx, context, weights.to_out);
}

core::TensorValue cross_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & context_input,
    const FoundationRfAttentionWeights & weights,
    const StableAudioConfig & config) {
    const int64_t dim = head_dim(config);
    const int64_t kv_heads = context_heads(config);
    if (config.num_heads % kv_heads != 0) {
        throw std::runtime_error("Stable Audio Foundation cross-attention head repeat mismatch");
    }
    auto q = modules::LinearModule({config.embed_dim, config.embed_dim, false, GGML_PREC_F32})
                 .build(ctx, input, weights.to_q);
    auto kv = modules::LinearModule({config.cond_token_dim, config.cond_token_dim * 2, false, GGML_PREC_F32})
                  .build(ctx, context_input, weights.to_kv);
    auto k = modules::SliceModule({2, 0, config.cond_token_dim}).build(ctx, kv);
    auto v = modules::SliceModule({2, config.cond_token_dim, config.cond_token_dim}).build(ctx, kv);
    q = reshape_heads(ctx, q, config.num_heads, dim);
    k = reshape_heads(ctx, k, kv_heads, dim);
    v = reshape_heads(ctx, v, kv_heads, dim);
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    k_heads = repeat_kv_heads(ctx, k_heads, config.num_heads / kv_heads);
    v_heads = repeat_kv_heads(ctx, v_heads, config.num_heads / kv_heads);
    auto context = rf_attention(ctx, q_heads, k_heads, v_heads, dim);
    context = core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, context),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.embed_dim}));
    return modules::LinearModule({config.embed_dim, config.embed_dim, false, GGML_PREC_F32})
        .build(ctx, context, weights.to_out);
}

core::TensorValue swiglu_ff(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const FoundationRfFeedForwardWeights & weights,
    const StableAudioConfig & config) {
    auto projected = modules::LinearModule({config.embed_dim, 2 * ff_inner_dim(config), true, GGML_PREC_F32})
                         .build(ctx, input, weights.in_proj);
    auto value = modules::SliceModule({2, 0, ff_inner_dim(config)}).build(ctx, projected);
    auto gate = modules::SliceModule({2, ff_inner_dim(config), ff_inner_dim(config)}).build(ctx, projected);
    gate = modules::SiluModule{}.build(ctx, gate);
    auto hidden = modules::MulModule{}.build(ctx, value, gate);
    return modules::LinearModule({ff_inner_dim(config), config.embed_dim, true, GGML_PREC_F32})
        .build(ctx, hidden, weights.out_proj);
}

core::TensorValue build_rf_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & cross_context,
    const core::TensorValue & positions,
    const core::TensorValue & padding_mask,
    const FoundationRfLayerWeights & weights,
    const StableAudioConfig & config) {
    auto hidden = layer_norm(ctx, input, weights.pre_norm_gamma, weights.pre_norm_beta);
    auto attn = self_attention(ctx, hidden, positions, padding_mask, weights.self_attn, config);
    hidden = modules::AddModule{}.build(ctx, input, attn);
    auto cross_norm = layer_norm(ctx, hidden, weights.cross_attend_norm_gamma, weights.cross_attend_norm_beta);
    auto cross = cross_attention(ctx, cross_norm, cross_context, weights.cross_attn, config);
    hidden = modules::AddModule{}.build(ctx, hidden, cross);
    auto ff_in = layer_norm(ctx, hidden, weights.ff_norm_gamma, weights.ff_norm_beta);
    auto ff = swiglu_ff(ctx, ff_in, weights.ff, config);
    return modules::AddModule{}.build(ctx, hidden, ff);
}

FoundationRfAttentionWeights load_self_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const StableAudioConfig & config,
    assets::TensorStorageType storage_type) {
    FoundationRfAttentionWeights weights;
    weights.to_qkv = load_linear(store, source, prefix + ".to_qkv", storage_type, config.embed_dim * 3, config.embed_dim, false);
    weights.to_out = load_linear(store, source, prefix + ".to_out", storage_type, config.embed_dim, config.embed_dim, false);
    return weights;
}

FoundationRfAttentionWeights load_cross_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const StableAudioConfig & config,
    assets::TensorStorageType storage_type) {
    FoundationRfAttentionWeights weights;
    weights.to_q = load_linear(store, source, prefix + ".to_q", storage_type, config.embed_dim, config.embed_dim, false);
    weights.to_kv = load_linear(store, source, prefix + ".to_kv", storage_type, config.cond_token_dim * 2, config.cond_token_dim, false);
    weights.to_out = load_linear(store, source, prefix + ".to_out", storage_type, config.embed_dim, config.embed_dim, false);
    return weights;
}

FoundationRfFeedForwardWeights load_feed_forward(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const StableAudioConfig & config,
    assets::TensorStorageType storage_type) {
    FoundationRfFeedForwardWeights weights;
    weights.in_proj = load_linear(
        store,
        source,
        prefix + ".ff.0.proj",
        storage_type,
        ff_inner_dim(config) * 2,
        config.embed_dim,
        true);
    weights.out_proj = load_linear(
        store,
        source,
        prefix + ".ff.2",
        storage_type,
        config.embed_dim,
        ff_inner_dim(config),
        true);
    return weights;
}

FoundationRfLayerWeights load_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const StableAudioConfig & config,
    assets::TensorStorageType storage_type) {
    FoundationRfLayerWeights weights;
    weights.pre_norm_gamma = store.load_f32_tensor(source, prefix + ".pre_norm.gamma", {config.embed_dim});
    weights.pre_norm_beta = store.load_f32_tensor(source, prefix + ".pre_norm.beta", {config.embed_dim});
    weights.cross_attend_norm_gamma = store.load_f32_tensor(source, prefix + ".cross_attend_norm.gamma", {config.embed_dim});
    weights.cross_attend_norm_beta = store.load_f32_tensor(source, prefix + ".cross_attend_norm.beta", {config.embed_dim});
    weights.ff_norm_gamma = store.load_f32_tensor(source, prefix + ".ff_norm.gamma", {config.embed_dim});
    weights.ff_norm_beta = store.load_f32_tensor(source, prefix + ".ff_norm.beta", {config.embed_dim});
    weights.self_attn = load_self_attention(store, source, prefix + ".self_attn", config, storage_type);
    weights.cross_attn = load_cross_attention(store, source, prefix + ".cross_attn", config, storage_type);
    weights.ff = load_feed_forward(store, source, prefix + ".ff", config, storage_type);
    return weights;
}

}  // namespace

FoundationRFDiTWeights load_foundation_rf_dit_weights(
    const StableAudioAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type) {
    const auto & config = assets.config;
    const auto & source = *assets.model_weights;
    FoundationRFDiTWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "stable_audio.foundation.rf_dit.weights",
        weight_context_bytes == 0 ? kRfWeightContextBytes : weight_context_bytes);
    weights.timestep_frequencies =
        source.require_f32("model.model.timestep_features.weight", {kTimestepFeaturesDim / 2, 1});
    weights.to_cond_embed_0 = load_linear(*weights.store, source, "model.model.to_cond_embed.0", weight_storage_type, config.cond_token_dim, config.cond_token_dim, false);
    weights.to_cond_embed_2 = load_linear(*weights.store, source, "model.model.to_cond_embed.2", weight_storage_type, config.cond_token_dim, config.cond_token_dim, false);
    weights.to_global_embed_0 = load_linear(*weights.store, source, "model.model.to_global_embed.0", weight_storage_type, config.embed_dim, config.global_cond_dim, false);
    weights.to_global_embed_2 = load_linear(*weights.store, source, "model.model.to_global_embed.2", weight_storage_type, config.embed_dim, config.embed_dim, false);
    weights.timestep_embed_0 = load_linear(*weights.store, source, "model.model.to_timestep_embed.0", weight_storage_type, config.embed_dim, kTimestepFeaturesDim, true);
    weights.timestep_embed_2 = load_linear(*weights.store, source, "model.model.to_timestep_embed.2", weight_storage_type, config.embed_dim, config.embed_dim, true);
    weights.preprocess_conv = load_linear_as_shape(*weights.store, source, "model.model.preprocess_conv", weight_storage_type, config.io_channels, config.io_channels, false);
    weights.postprocess_conv = load_linear_as_shape(*weights.store, source, "model.model.postprocess_conv", weight_storage_type, config.io_channels, config.io_channels, false);
    weights.project_in = load_linear(*weights.store, source, "model.model.transformer.project_in", weight_storage_type, config.embed_dim, config.io_channels, false);
    weights.project_out = load_linear(*weights.store, source, "model.model.transformer.project_out", weight_storage_type, config.io_channels, config.embed_dim, false);
    weights.layers.reserve(static_cast<size_t>(config.depth));
    for (int64_t layer = 0; layer < config.depth; ++layer) {
        weights.layers.push_back(load_layer(
            *weights.store,
            source,
            "model.model.transformer.layers." + std::to_string(layer),
            config,
            weight_storage_type));
    }
    weights.store->upload();
    return weights;
}

class FoundationRFDiTRuntime::Graph {
public:
    Graph(
        core::ExecutionContext & execution,
        std::shared_ptr<const StableAudioAssets> assets,
        const FoundationRFDiTWeights & weights,
        const StableAudioSamplingState & sampling,
        bool use_cfg)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          assets_(std::move(assets)),
          batch_(sampling.batch),
          latent_tokens_(sampling.latent_sample_size),
          use_cfg_(use_cfg),
          timestep_freqs_(weights.timestep_frequencies),
          weights_(weights) {
        if (backend_ == nullptr) {
            throw std::runtime_error("Stable Audio Foundation RF DiT backend initialization failed");
        }
        if (batch_ <= 0 || latent_tokens_ <= 0) {
            throw std::runtime_error("Stable Audio Foundation RF DiT graph requires positive batch and latent length");
        }
        build();
    }

    ~Graph() {
        if (backend_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(backend_type_, backend_, graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(const StableAudioSamplingState & sampling) const noexcept {
        return sampling.batch == batch_ && sampling.latent_sample_size == latent_tokens_;
    }

    bool matches_cfg(bool use_cfg) const noexcept {
        return use_cfg_ == use_cfg;
    }

    std::vector<float> sample(
        const StableAudioSamplingState & sampling,
        const StableAudioConditioningInputs & conditioning,
        uint64_t seed,
        uint64_t & rng_offset_blocks,
        const engine::sampling::TorchCudaSamplingPolicy & rng_policy,
        std::string_view sampler_type,
        float sigma_min,
        float sigma_max,
        float rho,
        float cfg_scale,
        float apg_scale) const {
        validate_inputs(sampling, conditioning);
        if (!sampler_type.empty() &&
            sampler_type != "euler" &&
            sampler_type != "pingpong" &&
            sampler_type != "dpmpp-2m" &&
            sampler_type != "dpmpp-3m-sde") {
            throw std::runtime_error("Stable Audio Foundation sampler_type is not supported for the v sampler");
        }
        if (sampler_type == "dpmpp-2m") {
            static_cast<void>(seed);
            static_cast<void>(rng_offset_blocks);
            static_cast<void>(rng_policy);
            return sample_dpmpp_2m(
                sampling,
                conditioning,
                sigma_min,
                sigma_max,
                rho,
                cfg_scale,
                apg_scale);
        }
        if (sampler_type == "dpmpp-3m-sde") {
            return sample_dpmpp_3m_sde(
                sampling,
                conditioning,
                seed,
                rng_offset_blocks,
                rng_policy,
                sigma_min,
                sigma_max,
                rho,
                cfg_scale,
                apg_scale);
        }
        static_cast<void>(seed);
        static_cast<void>(rng_offset_blocks);
        static_cast<void>(rng_policy);
        return sample_v_ddim(sampling, conditioning, cfg_scale, apg_scale);
    }

private:
    std::vector<float> sample_v_ddim(
        const StableAudioSamplingState & sampling,
        const StableAudioConditioningInputs & conditioning,
        float cfg_scale,
        float apg_scale) const {
        const auto total_start = Clock::now();
        std::vector<float> latent = sampling.noise;
        std::vector<float> graph_x(static_cast<size_t>(graph_batch() * latent_tokens_ * assets_->config.io_channels));
        const auto cross = make_cross_conditioning(conditioning);
        const auto global = make_global_conditioning(conditioning);
        const auto padding = make_padding_mask(sampling);
        core::write_tensor_f32(cross_, cross);
        core::write_tensor_f32(global_, global);
        core::write_tensor_f32(padding_mask_, padding);
        if (use_cfg_) {
            core::write_tensor_f32(cfg_scale_, std::vector<float>{cfg_scale});
            core::write_tensor_f32(apg_scale_, std::vector<float>{apg_scale});
            core::write_tensor_f32(latent_padding_mask_, sampling.padding_mask);
        }
        core::set_backend_threads(backend_, threads_);
        std::vector<float> model_output;
        model_output.reserve(static_cast<size_t>(batch_ * assets_->config.io_channels * latent_tokens_));
        if (sampling.schedule_points <= 1 ||
            static_cast<int64_t>(sampling.schedule.size()) != batch_ * sampling.schedule_points) {
            throw std::runtime_error("Stable Audio Foundation RF DiT schedule shape mismatch");
        }
        std::vector<float> timestep_values(static_cast<size_t>(batch_));
        std::vector<float> alpha_values(static_cast<size_t>(batch_));
        std::vector<float> sigma_values(static_cast<size_t>(batch_));
        std::vector<float> next_alpha_values(static_cast<size_t>(batch_));
        std::vector<float> next_sigma_values(static_cast<size_t>(batch_));
        std::vector<float> timestep_features(static_cast<size_t>(graph_batch() * kTimestepFeaturesDim));
        for (int step = 0; step < static_cast<int>(sampling.schedule_points) - 1; ++step) {
            write_latent_input(latent, graph_x);
            for (int64_t b = 0; b < batch_; ++b) {
                const size_t base = static_cast<size_t>(b * sampling.schedule_points);
                const float timestep = sampling.schedule[base + static_cast<size_t>(step)];
                const float next_timestep = sampling.schedule[base + static_cast<size_t>(step + 1)];
                timestep_values[static_cast<size_t>(b)] = timestep;
                alpha_values[static_cast<size_t>(b)] = std::cos(timestep * kPi * 0.5F);
                sigma_values[static_cast<size_t>(b)] = std::sin(timestep * kPi * 0.5F);
                next_alpha_values[static_cast<size_t>(b)] = std::cos(next_timestep * kPi * 0.5F);
                next_sigma_values[static_cast<size_t>(b)] = std::sin(next_timestep * kPi * 0.5F);
            }
            write_timestep_features(timestep_values, timestep_features);
            core::write_tensor_f32(x_, graph_x);
            core::write_tensor_f32(timestep_features_, timestep_features);
            if (use_cfg_) {
                core::write_tensor_f32(sigma_, sigma_values);
                core::write_tensor_f32(alpha_, alpha_values);
            }
            const auto compute_start = Clock::now();
            const ggml_status status = engine::core::compute_backend_graph(backend_, graph_, nullptr, "stable_audio.foundation.rf_dit");
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Stable Audio Foundation RF DiT graph compute failed");
            }
            if (step == 0) {
                engine::debug::timing_log_scalar(
                    "stable_audio.foundation.rf_dit.step0.graph.compute_ms",
                    engine::debug::elapsed_ms(compute_start, Clock::now()));
            }
            model_output = core::read_tensor_f32(output_);
            for (const float value : model_output) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error(
                        "Stable Audio Foundation RF DiT produced non-finite model output at step " + std::to_string(step));
                }
            }
            for (size_t i = 0; i < latent.size(); ++i) {
                const int64_t b = static_cast<int64_t>(i) / (assets_->config.io_channels * latent_tokens_);
                const float alpha = alpha_values[static_cast<size_t>(b)];
                const float sigma = sigma_values[static_cast<size_t>(b)];
                const float pred = latent[i] * alpha - model_output[i] * sigma;
                if (step + 1 < static_cast<int>(sampling.schedule_points) - 1) {
                    const float eps = latent[i] * sigma + model_output[i] * alpha;
                    latent[i] = pred * next_alpha_values[static_cast<size_t>(b)] +
                                eps * next_sigma_values[static_cast<size_t>(b)];
                } else {
                    latent[i] = pred;
                }
                if (!std::isfinite(latent[i])) {
                    throw std::runtime_error(
                        "Stable Audio Foundation RF DiT produced non-finite latent at step " + std::to_string(step));
                }
            }
        }
        engine::debug::timing_log_scalar(
            "stable_audio.foundation.rf_dit.total_ms",
            engine::debug::elapsed_ms(total_start, Clock::now()));
        return latent;
    }

    std::vector<float> build_polyexponential_sigmas(
        int64_t steps,
        float sigma_min,
        float sigma_max,
        float rho) const {
        if (steps <= 0) {
            throw std::runtime_error("Stable Audio Foundation DPM++ sampler requires positive steps");
        }
        if (!(sigma_min > 0.0F) || !(sigma_max > sigma_min) || !(rho > 0.0F)) {
            throw std::runtime_error("Stable Audio Foundation DPM++ sampler sigma schedule is invalid");
        }
        std::vector<float> sigmas(static_cast<size_t>(steps + 1), 0.0F);
        for (int64_t i = 0; i < steps; ++i) {
            const float ramp = steps == 1
                ? 1.0F
                : std::pow(1.0F - static_cast<float>(i) / static_cast<float>(steps - 1), rho);
            sigmas[static_cast<size_t>(i)] =
                std::exp(ramp * (std::log(sigma_max) - std::log(sigma_min)) + std::log(sigma_min));
        }
        return sigmas;
    }

    void run_dit_model(
        const StableAudioConditioningInputs & conditioning,
        const StableAudioSamplingState & sampling,
        const std::vector<float> & latent_bct,
        float timestep,
        float cfg_scale,
        float apg_scale,
        std::vector<float> & graph_x,
        std::vector<float> & timestep_features,
        std::vector<float> & model_output,
        int step) const {
        const auto cross = make_cross_conditioning(conditioning);
        const auto global = make_global_conditioning(conditioning);
        const auto padding = make_padding_mask(sampling);
        core::write_tensor_f32(cross_, cross);
        core::write_tensor_f32(global_, global);
        core::write_tensor_f32(padding_mask_, padding);
        if (use_cfg_) {
            core::write_tensor_f32(cfg_scale_, std::vector<float>{cfg_scale});
            core::write_tensor_f32(apg_scale_, std::vector<float>{apg_scale});
            core::write_tensor_f32(latent_padding_mask_, sampling.padding_mask);
        }
        write_latent_input(latent_bct, graph_x);
        std::vector<float> timestep_values(static_cast<size_t>(batch_), timestep);
        write_timestep_features(timestep_values, timestep_features);
        core::write_tensor_f32(x_, graph_x);
        core::write_tensor_f32(timestep_features_, timestep_features);
        if (use_cfg_) {
            const float sigma = std::sin(timestep * kPi * 0.5F);
            const float alpha = std::cos(timestep * kPi * 0.5F);
            core::write_tensor_f32(sigma_, std::vector<float>(static_cast<size_t>(batch_), sigma));
            core::write_tensor_f32(alpha_, std::vector<float>(static_cast<size_t>(batch_), alpha));
        }
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_, nullptr, "stable_audio.foundation.rf_dit");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Stable Audio Foundation RF DiT graph compute failed");
        }
        if (step == 0) {
            engine::debug::timing_log_scalar(
                "stable_audio.foundation.rf_dit.step0.graph.compute_ms",
                engine::debug::elapsed_ms(compute_start, Clock::now()));
        }
        model_output = core::read_tensor_f32(output_);
        for (const float value : model_output) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Stable Audio Foundation RF DiT produced non-finite model output at step " + std::to_string(step));
            }
        }
    }

    std::vector<float> sample_dpmpp_3m_sde(
        const StableAudioSamplingState & sampling,
        const StableAudioConditioningInputs & conditioning,
        uint64_t seed,
        uint64_t & rng_offset_blocks,
        const engine::sampling::TorchCudaSamplingPolicy & rng_policy,
        float sigma_min,
        float sigma_max,
        float rho,
        float cfg_scale,
        float apg_scale) const {
        const auto total_start = Clock::now();
        const int64_t steps = sampling.schedule_points - 1;
        const auto sigmas = build_polyexponential_sigmas(steps, sigma_min, sigma_max, rho);
        std::vector<float> x = sampling.noise;
        for (float & value : x) {
            value *= sigmas.front();
        }
        std::vector<float> denoiser_input(x.size(), 0.0F);
        std::vector<float> graph_x(static_cast<size_t>(graph_batch() * latent_tokens_ * assets_->config.io_channels));
        std::vector<float> timestep_features(static_cast<size_t>(graph_batch() * kTimestepFeaturesDim));
        std::vector<float> model_output;
        std::vector<float> denoised_1;
        std::vector<float> denoised_2;
        float h_1 = 0.0F;
        float h_2 = 0.0F;
        bool has_h_1 = false;
        bool has_h_2 = false;
        core::set_backend_threads(backend_, threads_);
        for (int step = 0; step < static_cast<int>(steps); ++step) {
            const float sigma = sigmas[static_cast<size_t>(step)];
            const float sigma_next = sigmas[static_cast<size_t>(step + 1)];
            const float c_in = 1.0F / std::sqrt(sigma * sigma + 1.0F);
            const float c_skip = 1.0F / (sigma * sigma + 1.0F);
            const float c_out = -sigma / std::sqrt(sigma * sigma + 1.0F);
            const float timestep = std::atan(sigma) * 2.0F / kPi;
            for (size_t i = 0; i < x.size(); ++i) {
                denoiser_input[i] = x[i] * c_in;
            }
            run_dit_model(
                conditioning,
                sampling,
                denoiser_input,
                timestep,
                cfg_scale,
                apg_scale,
                graph_x,
                timestep_features,
                model_output,
                step);
            std::vector<float> denoised(x.size(), 0.0F);
            for (size_t i = 0; i < x.size(); ++i) {
                denoised[i] = model_output[i] * c_out + x[i] * c_skip;
                if (!std::isfinite(denoised[i])) {
                    throw std::runtime_error(
                        "Stable Audio Foundation RF DiT produced non-finite denoised output at step " + std::to_string(step));
                }
            }
            if (sigma_next == 0.0F) {
                x = std::move(denoised);
                break;
            }
            const float h = -std::log(sigma_next) + std::log(sigma);
            const float h_eta = h * 2.0F;
            const float base_scale = std::exp(-h_eta);
            const float denoised_scale = -std::expm1(-h_eta);
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] = base_scale * x[i] + denoised_scale * denoised[i];
            }
            if (has_h_2) {
                const float r0 = h_1 / h;
                const float r1 = h_2 / h;
                const float phi_2 = std::expm1(-h_eta) / h_eta + 1.0F;
                const float phi_3 = phi_2 / h_eta - 0.5F;
                for (size_t i = 0; i < x.size(); ++i) {
                    const float d1_0 = (denoised[i] - denoised_1[i]) / r0;
                    const float d1_1 = (denoised_1[i] - denoised_2[i]) / r1;
                    const float d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1);
                    const float d2 = (d1_0 - d1_1) / (r0 + r1);
                    x[i] += phi_2 * d1 - phi_3 * d2;
                }
            } else if (has_h_1) {
                const float r = h_1 / h;
                const float phi_2 = std::expm1(-h_eta) / h_eta + 1.0F;
                for (size_t i = 0; i < x.size(); ++i) {
                    x[i] += phi_2 * ((denoised[i] - denoised_1[i]) / r);
                }
            }
            auto noise = engine::sampling::generate_torch_cuda_tensor_iterator_randn(
                x.size(),
                seed,
                rng_offset_blocks,
                rng_policy,
                engine::sampling::TorchRandnPrecision::Float32);
            rng_offset_blocks += engine::sampling::torch_cuda_tensor_iterator_offset_blocks(
                static_cast<uint64_t>(x.size()),
                rng_policy);
            const float noise_scale = sigma_next * std::sqrt(-std::expm1(-2.0F * h));
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] += noise[i] * noise_scale;
                if (!std::isfinite(x[i])) {
                    throw std::runtime_error(
                        "Stable Audio Foundation RF DiT produced non-finite latent at step " + std::to_string(step));
                }
            }
            denoised_2 = std::move(denoised_1);
            denoised_1 = std::move(denoised);
            h_2 = h_1;
            h_1 = h;
            has_h_2 = has_h_1;
            has_h_1 = true;
        }
        engine::debug::timing_log_scalar(
            "stable_audio.foundation.rf_dit.total_ms",
            engine::debug::elapsed_ms(total_start, Clock::now()));
        return x;
    }

    std::vector<float> sample_dpmpp_2m(
        const StableAudioSamplingState & sampling,
        const StableAudioConditioningInputs & conditioning,
        float sigma_min,
        float sigma_max,
        float rho,
        float cfg_scale,
        float apg_scale) const {
        const auto total_start = Clock::now();
        const int64_t steps = sampling.schedule_points - 1;
        const auto sigmas = build_polyexponential_sigmas(steps, sigma_min, sigma_max, rho);
        std::vector<float> x = sampling.noise;
        for (float & value : x) {
            value *= sigmas.front();
        }
        std::vector<float> denoiser_input(x.size(), 0.0F);
        std::vector<float> graph_x(static_cast<size_t>(graph_batch() * latent_tokens_ * assets_->config.io_channels));
        std::vector<float> timestep_features(static_cast<size_t>(graph_batch() * kTimestepFeaturesDim));
        std::vector<float> model_output;
        std::vector<float> old_denoised;
        bool has_old_denoised = false;
        core::set_backend_threads(backend_, threads_);
        for (int step = 0; step < static_cast<int>(steps); ++step) {
            const float sigma = sigmas[static_cast<size_t>(step)];
            const float sigma_next = sigmas[static_cast<size_t>(step + 1)];
            const float c_in = 1.0F / std::sqrt(sigma * sigma + 1.0F);
            const float c_skip = 1.0F / (sigma * sigma + 1.0F);
            const float c_out = -sigma / std::sqrt(sigma * sigma + 1.0F);
            const float timestep = std::atan(sigma) * 2.0F / kPi;
            for (size_t i = 0; i < x.size(); ++i) {
                denoiser_input[i] = x[i] * c_in;
            }
            run_dit_model(
                conditioning,
                sampling,
                denoiser_input,
                timestep,
                cfg_scale,
                apg_scale,
                graph_x,
                timestep_features,
                model_output,
                step);
            std::vector<float> denoised(x.size(), 0.0F);
            for (size_t i = 0; i < x.size(); ++i) {
                denoised[i] = model_output[i] * c_out + x[i] * c_skip;
                if (!std::isfinite(denoised[i])) {
                    throw std::runtime_error(
                        "Stable Audio Foundation RF DiT produced non-finite denoised output at step " + std::to_string(step));
                }
            }
            if (sigma_next == 0.0F) {
                x = std::move(denoised);
                break;
            }
            const float h = -std::log(sigma_next) + std::log(sigma);
            const float x_scale = sigma_next / sigma;
            const float denoised_scale = -std::expm1(-h);
            if (!has_old_denoised) {
                for (size_t i = 0; i < x.size(); ++i) {
                    x[i] = x_scale * x[i] + denoised_scale * denoised[i];
                }
            } else {
                const float h_last = std::log(sigmas[static_cast<size_t>(step - 1)]) - std::log(sigma);
                const float r = h_last / h;
                const float denoised_coeff = 1.0F + 1.0F / (2.0F * r);
                const float old_coeff = -1.0F / (2.0F * r);
                for (size_t i = 0; i < x.size(); ++i) {
                    const float denoised_d = denoised_coeff * denoised[i] + old_coeff * old_denoised[i];
                    x[i] = x_scale * x[i] + denoised_scale * denoised_d;
                }
            }
            for (const float value : x) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error(
                        "Stable Audio Foundation RF DiT produced non-finite latent at step " + std::to_string(step));
                }
            }
            old_denoised = std::move(denoised);
            has_old_denoised = true;
        }
        engine::debug::timing_log_scalar(
            "stable_audio.foundation.rf_dit.total_ms",
            engine::debug::elapsed_ms(total_start, Clock::now()));
        return x;
    }

    void build() {
        const auto & config = assets_->config;
        const int64_t total_tokens = 1 + latent_tokens_;
        ggml_init_params params{512ull * 1024ull * 1024ull, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("Stable Audio Foundation RF DiT ggml context initialization failed");
        }
        auto input_ctx = ctx_for_inputs();
        x_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({graph_batch(), latent_tokens_, config.io_channels}));
        timestep_features_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({graph_batch(), kTimestepFeaturesDim}));
        cross_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({graph_batch(), config.prompt_max_length + 2, config.cond_token_dim}));
        global_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({graph_batch(), config.global_cond_dim}));
        padding_mask_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({graph_batch(), 1, total_tokens, 1}));
        latent_padding_mask_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, latent_tokens_}));
        sigma_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, 1, 1}));
        alpha_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, 1, 1}));
        cfg_scale_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        apg_scale_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1}));
        positions_tensor_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({total_tokens}));
        ggml_set_input(x_.tensor);
        ggml_set_input(timestep_features_.tensor);
        ggml_set_input(cross_.tensor);
        ggml_set_input(global_.tensor);
        ggml_set_input(padding_mask_.tensor);
        ggml_set_input(latent_padding_mask_.tensor);
        ggml_set_input(sigma_.tensor);
        ggml_set_input(alpha_.tensor);
        ggml_set_input(cfg_scale_.tensor);
        ggml_set_input(apg_scale_.tensor);
        ggml_set_input(positions_tensor_.tensor);
        core::ModuleBuildContext build_ctx{ctx_.get(), "stable_audio.foundation.rf_dit", backend_type_};
        auto output = build_graph_output(build_ctx);
        output_ = output.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("Stable Audio Foundation RF DiT backend buffer allocation failed");
        }
        positions_.assign(static_cast<size_t>(total_tokens), 0);
        for (int64_t i = 0; i < total_tokens; ++i) {
            positions_[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        core::write_tensor_i32(positions_tensor_, positions_);
    }

    core::ModuleBuildContext ctx_for_inputs() {
        return core::ModuleBuildContext{ctx_.get(), "stable_audio.foundation.rf_dit.inputs", backend_type_};
    }

    void write_timestep_features(const std::vector<float> & sigmas, std::vector<float> & out) const {
        const int64_t half = kTimestepFeaturesDim / 2;
        for (int64_t b = 0; b < graph_batch(); ++b) {
            const float sigma = sigmas[static_cast<size_t>(b % batch_)];
            float * row = out.data() + static_cast<std::ptrdiff_t>(b * kTimestepFeaturesDim);
            for (int64_t i = 0; i < half; ++i) {
                const float arg = sigma * timestep_freqs_[static_cast<size_t>(i)] * 2.0F * kPi;
                row[i] = std::cos(arg);
                row[half + i] = std::sin(arg);
            }
        }
    }

    core::TensorValue build_graph_output(core::ModuleBuildContext & ctx) {
        const auto & config = assets_->config;
        auto hidden = modules::LinearModule({config.io_channels, config.io_channels, false, GGML_PREC_F32})
                          .build(ctx, x_, weights_.preprocess_conv);
        hidden = modules::AddModule{}.build(ctx, hidden, x_);
        hidden = modules::LinearModule({config.io_channels, config.embed_dim, false, GGML_PREC_F32})
                     .build(ctx, hidden, weights_.project_in);
        auto cross = modules::LinearModule({config.cond_token_dim, config.cond_token_dim, false, GGML_PREC_F32})
                         .build(ctx, cross_, weights_.to_cond_embed_0);
        cross = modules::SiluModule{}.build(ctx, cross);
        cross = modules::LinearModule({config.cond_token_dim, config.cond_token_dim, false, GGML_PREC_F32})
                    .build(ctx, cross, weights_.to_cond_embed_2);
        auto global = modules::LinearModule({config.global_cond_dim, config.embed_dim, false, GGML_PREC_F32})
                          .build(ctx, global_, weights_.to_global_embed_0);
        global = modules::SiluModule{}.build(ctx, global);
        global = modules::LinearModule({config.embed_dim, config.embed_dim, false, GGML_PREC_F32})
                     .build(ctx, global, weights_.to_global_embed_2);
        global = modules::AddModule{}.build(ctx, global, build_timestep_embedding(ctx, timestep_features_, weights_, config));
        global = core::reshape_tensor(ctx, ensure_contiguous(ctx, global), core::TensorShape::from_dims({graph_batch(), 1, config.embed_dim}));
        hidden = modules::ConcatModule({1}).build(ctx, global, hidden);
        for (const auto & layer : weights_.layers) {
            hidden = build_rf_layer(ctx, hidden, cross, positions_tensor_, padding_mask_, layer, config);
        }
        hidden = modules::LinearModule({config.embed_dim, config.io_channels, false, GGML_PREC_F32})
                     .build(ctx, hidden, weights_.project_out);
        hidden = modules::SliceModule({1, 1, latent_tokens_}).build(ctx, hidden);
        auto post = modules::LinearModule({config.io_channels, config.io_channels, false, GGML_PREC_F32})
                        .build(ctx, hidden, weights_.postprocess_conv);
        auto raw_output = modules::AddModule{}.build(ctx, hidden, post);
        if (!use_cfg_) {
            return modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, raw_output);
        }
        auto cfg_output = build_cfg_model_output(ctx, raw_output);
        return modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, cfg_output);
    }

    core::TensorValue build_cfg_model_output(core::ModuleBuildContext & ctx, const core::TensorValue & raw_output) {
        const auto & config = assets_->config;
        auto x_cond = modules::SliceModule({0, 0, batch_}).build(ctx, x_);
        auto cond = modules::SliceModule({0, 0, batch_}).build(ctx, raw_output);
        auto uncond = modules::SliceModule({0, batch_, batch_}).build(ctx, raw_output);
        auto sigma = core::reshape_tensor(ctx, sigma_, core::TensorShape::from_dims({batch_, 1, 1}));
        sigma = repeat_like(ctx, sigma, cond);
        auto alpha = core::reshape_tensor(ctx, alpha_, core::TensorShape::from_dims({batch_, 1, 1}));
        alpha = repeat_like(ctx, alpha, cond);
        auto x_alpha = modules::MulModule{}.build(ctx, x_cond, alpha);
        auto cond_x0 = sub_tensor(ctx, x_alpha, modules::MulModule{}.build(ctx, cond, sigma));
        auto uncond_x0 = sub_tensor(ctx, x_alpha, modules::MulModule{}.build(ctx, uncond, sigma));
        auto diff = sub_tensor(ctx, cond_x0, uncond_x0);
        auto mask = core::reshape_tensor(ctx, latent_padding_mask_, core::TensorShape::from_dims({batch_, latent_tokens_, 1}));
        mask = modules::RepeatModule({core::TensorShape::from_dims({batch_, latent_tokens_, config.io_channels})}).build(ctx, mask);
        auto cond_masked = modules::MulModule{}.build(ctx, cond_x0, mask);
        auto norm_sq = modules::MulModule{}.build(ctx, cond_masked, cond_masked);
        norm_sq = modules::ReduceSumModule({2}).build(ctx, norm_sq);
        norm_sq = modules::ReduceSumModule({1}).build(ctx, norm_sq);
        auto norm = core::wrap_tensor(
            ggml_sqrt(ctx.ggml, ggml_scale_bias(ctx.ggml, norm_sq.tensor, 1.0F, 1.0e-16F)),
            norm_sq.shape,
            GGML_TYPE_F32);
        auto v1_normalized = div_tensor(ctx, cond_masked, repeat_like(ctx, norm, cond_masked));
        auto dot = modules::MulModule{}.build(ctx, diff, v1_normalized);
        dot = modules::ReduceSumModule({2}).build(ctx, dot);
        dot = modules::ReduceSumModule({1}).build(ctx, dot);
        auto parallel = modules::MulModule{}.build(ctx, repeat_like(ctx, dot, diff), v1_normalized);
        auto orthogonal = modules::MulModule{}.build(ctx, sub_tensor(ctx, diff, parallel), mask);
        auto apg_value = core::reshape_tensor(ctx, apg_scale_, core::TensorShape::from_dims({1, 1, 1}));
        auto apg = repeat_like(ctx, apg_value, diff);
        auto one_minus_apg = scale_bias_tensor(ctx, apg, -1.0F, 1.0F);
        auto cfg_diff = modules::AddModule{}.build(
            ctx,
            modules::MulModule{}.build(ctx, apg, orthogonal),
            modules::MulModule{}.build(ctx, one_minus_apg, diff));
        auto cfg_value = core::reshape_tensor(ctx, cfg_scale_, core::TensorShape::from_dims({1, 1, 1}));
        auto cfg = scale_bias_tensor(ctx, repeat_like(ctx, cfg_value, diff), 1.0F, -1.0F);
        auto cfg_denoised = modules::AddModule{}.build(ctx, cond_x0, modules::MulModule{}.build(ctx, cfg, cfg_diff));
        return div_tensor(ctx, sub_tensor(ctx, x_alpha, cfg_denoised), sigma);
    }

    void validate_inputs(const StableAudioSamplingState & sampling, const StableAudioConditioningInputs & conditioning) const {
        const auto & config = assets_->config;
        if (!matches(sampling)) {
            throw std::runtime_error("Stable Audio Foundation RF DiT graph shape mismatch");
        }
        if (conditioning.positive.batch != batch_ ||
            conditioning.positive.cross_tokens != config.prompt_max_length + 2 ||
            conditioning.positive.cond_dim != config.cond_token_dim ||
            static_cast<int64_t>(conditioning.positive.global.size()) != batch_ * config.global_cond_dim) {
            throw std::runtime_error("Stable Audio Foundation RF DiT positive conditioning shape mismatch");
        }
        if (conditioning.has_negative &&
            (conditioning.negative.batch != batch_ ||
             conditioning.negative.cross_tokens != config.prompt_max_length + 2 ||
             conditioning.negative.cond_dim != config.cond_token_dim)) {
            throw std::runtime_error("Stable Audio Foundation RF DiT negative conditioning shape mismatch");
        }
    }

    std::vector<float> make_cross_conditioning(const StableAudioConditioningInputs & conditioning) const {
        const auto & config = assets_->config;
        const int64_t tokens = config.prompt_max_length + 2;
        std::vector<float> out(static_cast<size_t>(graph_batch() * tokens * config.cond_token_dim), 0.0F);
        const int64_t row = tokens * config.cond_token_dim;
        for (int64_t b = 0; b < batch_; ++b) {
            std::copy_n(
                conditioning.positive.cross_attention.data() + static_cast<std::ptrdiff_t>(b * row),
                static_cast<size_t>(row),
                out.data() + static_cast<std::ptrdiff_t>(b * row));
            if (use_cfg_ && conditioning.has_negative) {
                std::copy_n(
                    conditioning.negative.cross_attention.data() + static_cast<std::ptrdiff_t>(b * row),
                    static_cast<size_t>(row),
                    out.data() + static_cast<std::ptrdiff_t>((batch_ + b) * row));
            }
        }
        return out;
    }

    std::vector<float> make_global_conditioning(const StableAudioConditioningInputs & conditioning) const {
        const auto & config = assets_->config;
        std::vector<float> out(static_cast<size_t>(graph_batch() * config.global_cond_dim), 0.0F);
        for (int64_t b = 0; b < batch_; ++b) {
            const float * src = conditioning.positive.global.data() + static_cast<std::ptrdiff_t>(b * config.global_cond_dim);
            std::copy_n(src, static_cast<size_t>(config.global_cond_dim), out.data() + static_cast<std::ptrdiff_t>(b * config.global_cond_dim));
            if (use_cfg_) {
                std::copy_n(src, static_cast<size_t>(config.global_cond_dim), out.data() + static_cast<std::ptrdiff_t>((batch_ + b) * config.global_cond_dim));
            }
        }
        return out;
    }

    std::vector<float> make_padding_mask(const StableAudioSamplingState & sampling) const {
        const int64_t total = 1 + latent_tokens_;
        std::vector<float> out(static_cast<size_t>(graph_batch() * total), 0.0F);
        for (int64_t bb = 0; bb < graph_batch(); ++bb) {
            const int64_t b = bb % batch_;
            out[static_cast<size_t>(bb * total)] = 1.0F;
            for (int64_t t = 0; t < latent_tokens_; ++t) {
                out[static_cast<size_t>(bb * total + 1 + t)] =
                    sampling.padding_mask[static_cast<size_t>(b * latent_tokens_ + t)];
            }
        }
        return out;
    }

    void write_latent_input(const std::vector<float> & latent, std::vector<float> & graph_x) const {
        const int64_t channels = assets_->config.io_channels;
        for (int64_t bb = 0; bb < graph_batch(); ++bb) {
            const int64_t b = bb % batch_;
            for (int64_t t = 0; t < latent_tokens_; ++t) {
                for (int64_t c = 0; c < channels; ++c) {
                    graph_x[static_cast<size_t>((bb * latent_tokens_ + t) * channels + c)] =
                        latent[static_cast<size_t>((b * channels + c) * latent_tokens_ + t)];
                }
            }
        }
    }

    int64_t graph_batch() const noexcept {
        return use_cfg_ ? 2 * batch_ : batch_;
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const StableAudioAssets> assets_;
    int64_t batch_ = 0;
    int64_t latent_tokens_ = 0;
    bool use_cfg_ = false;
    std::vector<float> timestep_freqs_;
    const FoundationRFDiTWeights & weights_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue x_;
    core::TensorValue timestep_features_;
    core::TensorValue cross_;
    core::TensorValue global_;
    core::TensorValue padding_mask_;
    core::TensorValue latent_padding_mask_;
    core::TensorValue sigma_;
    core::TensorValue alpha_;
    core::TensorValue cfg_scale_;
    core::TensorValue apg_scale_;
    core::TensorValue positions_tensor_;
    ggml_tensor * output_ = nullptr;
    std::vector<int32_t> positions_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

FoundationRFDiTRuntime::FoundationRFDiTRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const StableAudioAssets> assets,
    assets::TensorStorageType weight_storage_type)
    : execution_(&execution),
      assets_(std::move(assets)),
      weight_storage_type_(weight_storage_type) {
    if (execution_ == nullptr) {
        throw std::runtime_error("Stable Audio Foundation RF DiT runtime requires execution context");
    }
    if (assets_ == nullptr) {
        throw std::runtime_error("Stable Audio Foundation RF DiT runtime requires assets");
    }
}

FoundationRFDiTRuntime::~FoundationRFDiTRuntime() = default;

const FoundationRFDiTWeights & require_rf_weights(
    std::unique_ptr<FoundationRFDiTWeights> & weights,
    core::ExecutionContext & execution,
    const std::shared_ptr<const StableAudioAssets> & assets,
    assets::TensorStorageType weight_storage_type) {
    if (!weights) {
        weights = std::make_unique<FoundationRFDiTWeights>(load_foundation_rf_dit_weights(
            *assets,
            execution.backend(),
            execution.backend_type(),
            0,
            weight_storage_type));
    }
    return *weights;
}

void FoundationRFDiTRuntime::prepare(const StableAudioSamplingState & sampling, float cfg_scale) const {
    const bool use_cfg = cfg_scale != 1.0F;
    const auto & weights = require_rf_weights(weights_, *execution_, assets_, weight_storage_type_);
    if (!graph_ || !graph_->matches(sampling) || !graph_->matches_cfg(use_cfg)) {
        graph_ = std::make_unique<Graph>(*execution_, assets_, weights, sampling, use_cfg);
    }
}

std::vector<float> FoundationRFDiTRuntime::sample(
    const StableAudioSamplingState & sampling,
    const StableAudioConditioningInputs & conditioning,
    uint64_t seed,
    uint64_t & rng_offset_blocks,
    const engine::sampling::TorchCudaSamplingPolicy & rng_policy,
    std::string_view sampler_type,
    float sigma_min,
    float sigma_max,
    float rho,
    float cfg_scale,
    float apg_scale) const {
    prepare(sampling, cfg_scale);
    return graph_->sample(
        sampling,
        conditioning,
        seed,
        rng_offset_blocks,
        rng_policy,
        sampler_type,
        sigma_min,
        sigma_max,
        rho,
        cfg_scale,
        apg_scale);
}

}  // namespace engine::models::stable_audio::foundation
