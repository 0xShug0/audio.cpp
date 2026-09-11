#include "engine/models/stable_audio/same_autoencoder.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/sampling/torch_random.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::stable_audio {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kSameWeightContextBytesSmall = 900ull * 1024ull * 1024ull;
constexpr size_t kSameWeightContextBytesMedium = 2200ull * 1024ull * 1024ull;
constexpr int64_t kSameChunkLatents = 128;
constexpr int64_t kSameChunkOverlap = 32;
constexpr float kSameBottleneckEvalNoiseScale = 1.0e-3F;
constexpr float kSameNormEps = 1.0e-3F;
constexpr float kPi = 3.14159265358979323846F;

int64_t round_up_to_multiple(int64_t value, int64_t multiple) {
    if (multiple <= 0) {
        throw std::runtime_error("Stable Audio SAME alignment multiple must be positive");
    }
    return ((value + multiple - 1) / multiple) * multiple;
}

int64_t decoder_graph_latents(const StableAudioConfig & config, int64_t latent_tokens, bool chunked) {
    if (chunked && latent_tokens > kSameChunkLatents) {
        return kSameChunkLatents;
    }
    if (!config.medium_architecture && config.same_chunk_midpoint_shift) {
        return round_up_to_multiple(latent_tokens, config.pretransform_encoder_chunk_size / config.same_strides.back());
    }
    return latent_tokens;
}

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

std::vector<float> squeeze_weight_g(std::vector<float> values, int64_t out_channels, const std::string & prefix) {
    if (static_cast<int64_t>(values.size()) != out_channels) {
        throw std::runtime_error("Stable Audio SAME weight_g shape mismatch: " + prefix);
    }
    return values;
}

std::vector<float> fold_weight_norm_dim0(
    const std::vector<float> & weight_v,
    const std::vector<float> & weight_g,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    const std::string & prefix) {
    const int64_t inner = in_channels * kernel;
    if (static_cast<int64_t>(weight_v.size()) != out_channels * inner ||
        static_cast<int64_t>(weight_g.size()) != out_channels) {
        throw std::runtime_error("Stable Audio SAME weight_norm tensor shape mismatch: " + prefix);
    }
    std::vector<float> out(weight_v.size(), 0.0F);
    for (int64_t row = 0; row < out_channels; ++row) {
        const size_t base = static_cast<size_t>(row * inner);
        double sum = 0.0;
        for (int64_t i = 0; i < inner; ++i) {
            const float value = weight_v[base + static_cast<size_t>(i)];
            sum += static_cast<double>(value) * static_cast<double>(value);
        }
        const float scale = static_cast<float>(static_cast<double>(weight_g[static_cast<size_t>(row)]) / std::sqrt(sum));
        for (int64_t i = 0; i < inner; ++i) {
            out[base + static_cast<size_t>(i)] = weight_v[base + static_cast<size_t>(i)] * scale;
        }
    }
    return out;
}

modules::Conv1dWeights load_weight_norm_conv1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    bool use_bias) {
    auto conv_storage_type = assets::resolve_tensor_storage_type(source, prefix + ".weight_v", storage_type);
    const ggml_type conv_ggml_type = assets::ggml_type_for_tensor_storage(conv_storage_type);
    if (ggml_is_quantized(conv_ggml_type) && kernel % ggml_blck_size(conv_ggml_type) != 0) {
        conv_storage_type = assets::TensorStorageType::F32;
    }
    const auto weight_v = source.require_f32(prefix + ".weight_v", {out_channels, in_channels, kernel});
    const auto weight_g = squeeze_weight_g(source.require_f32(prefix + ".weight_g", {out_channels, 1, 1}), out_channels, prefix);
    modules::Conv1dWeights weights;
    weights.weight = store.make_from_f32(
        core::TensorShape::from_dims({out_channels, in_channels, kernel}),
        conv_storage_type,
        fold_weight_norm_dim0(weight_v, weight_g, out_channels, in_channels, kernel, prefix));
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    }
    return weights;
}

core::TensorValue ensure_contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    return core::ensure_backend_addressable_layout(ctx, input);
}

core::TensorValue scale_tensor(core::ModuleBuildContext & ctx, const core::TensorValue & input, float scale) {
    return core::wrap_tensor(ggml_scale(ctx.ggml, ensure_contiguous(ctx, input).tensor, scale), input.shape, GGML_TYPE_F32);
}

core::TensorValue broadcast_last_dim(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    const core::TensorValue & like,
    int64_t dim) {
    core::validate_shape(value, core::TensorShape::from_dims({dim}), "Stable Audio SAME broadcast value");
    core::TensorShape shape = {};
    shape.rank = like.shape.rank;
    for (size_t i = 0; i < shape.rank; ++i) {
        shape.dims[i] = 1;
    }
    shape.dims[shape.rank - 1] = dim;
    auto reshaped = core::reshape_tensor(ctx, value, shape);
    return modules::RepeatModule({like.shape}).build(ctx, reshaped);
}

core::TensorValue broadcast_scalar(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    const core::TensorValue & like) {
    core::validate_shape(value, core::TensorShape::from_dims({1}), "Stable Audio SAME scalar");
    core::TensorShape shape = {};
    shape.rank = like.shape.rank;
    for (size_t i = 0; i < shape.rank; ++i) {
        shape.dims[i] = 1;
    }
    auto reshaped = core::reshape_tensor(ctx, value, shape);
    return modules::RepeatModule({like.shape}).build(ctx, reshaped);
}

core::TensorValue dynamic_tanh_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const StableAudioSameNormWeights & weights) {
    const int64_t dim = input.shape.last_dim();
    auto alpha = broadcast_scalar(ctx, weights.alpha, input);
    auto hidden = modules::MulModule{}.build(ctx, input, alpha);
    hidden = modules::TanhModule{}.build(ctx, hidden);
    auto gamma = broadcast_last_dim(ctx, weights.gamma, hidden, dim);
    auto beta = broadcast_last_dim(ctx, weights.beta, hidden, dim);
    hidden = modules::MulModule{}.build(ctx, hidden, gamma);
    return modules::AddModule{}.build(ctx, hidden, beta);
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

core::TensorValue same_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    const core::TensorValue * attention_mask,
    int64_t dim) {
    const auto q = ensure_contiguous(ctx, q_heads);
    const auto k = ensure_contiguous(ctx, k_heads);
    const auto v = ensure_contiguous(ctx, v_heads);
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q.tensor,
        k.tensor,
        v.tensor,
        attention_mask != nullptr ? attention_mask->tensor : nullptr,
        1.0F / std::sqrt(static_cast<float>(dim)),
        0.0F,
        0.0F);
    ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
    return core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[2], q.shape.dims[1], dim}),
        GGML_TYPE_F32);
}

core::TensorValue same_self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue * attention_mask,
    const StableAudioSameAttentionWeights & weights,
    const StableAudioConfig & config,
    int64_t dim) {
    const int64_t heads = dim / config.same_dim_heads;
    auto qkv = modules::LinearModule({dim, dim * (config.same_differential ? 5 : 3), false, GGML_PREC_F32})
        .build(ctx, input, weights.to_qkv);
    auto q = modules::SliceModule({2, 0, dim}).build(ctx, qkv);
    auto k = modules::SliceModule({2, dim, dim}).build(ctx, qkv);
    auto v = modules::SliceModule({2, 2 * dim, dim}).build(ctx, qkv);
    core::TensorValue q_diff;
    core::TensorValue k_diff;
    if (config.same_differential) {
        q_diff = modules::SliceModule({2, 3 * dim, dim}).build(ctx, qkv);
        k_diff = modules::SliceModule({2, 4 * dim, dim}).build(ctx, qkv);
    }
    q = dynamic_tanh_norm(ctx, reshape_heads(ctx, q, heads, config.same_dim_heads), weights.q_norm);
    k = dynamic_tanh_norm(ctx, reshape_heads(ctx, k, heads, config.same_dim_heads), weights.k_norm);
    v = reshape_heads(ctx, v, heads, config.same_dim_heads);
    q = modules::RoPEModule({config.same_dim_heads / 2, GGML_ROPE_TYPE_NEOX, 10000.0F}).build(ctx, q, positions);
    k = modules::RoPEModule({config.same_dim_heads / 2, GGML_ROPE_TYPE_NEOX, 10000.0F}).build(ctx, k, positions);
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = same_attention(ctx, q_heads, k_heads, v_heads, attention_mask, config.same_dim_heads);
    if (config.same_differential) {
        q_diff = dynamic_tanh_norm(ctx, reshape_heads(ctx, q_diff, heads, config.same_dim_heads), weights.q_norm);
        k_diff = dynamic_tanh_norm(ctx, reshape_heads(ctx, k_diff, heads, config.same_dim_heads), weights.k_norm);
        q_diff = modules::RoPEModule({config.same_dim_heads / 2, GGML_ROPE_TYPE_NEOX, 10000.0F}).build(ctx, q_diff, positions);
        k_diff = modules::RoPEModule({config.same_dim_heads / 2, GGML_ROPE_TYPE_NEOX, 10000.0F}).build(ctx, k_diff, positions);
        auto qd_heads = modules::TransposeModule({{0, 2, 1, 3}, q_diff.shape.rank}).build(ctx, q_diff);
        auto kd_heads = modules::TransposeModule({{0, 2, 1, 3}, k_diff.shape.rank}).build(ctx, k_diff);
        auto diff_context = same_attention(ctx, qd_heads, kd_heads, v_heads, attention_mask, config.same_dim_heads);
        context = modules::AddModule{}.build(ctx, context, scale_tensor(ctx, diff_context, -1.0F));
    }
    context = core::reshape_tensor(
        ctx,
        ensure_contiguous(ctx, context),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], dim}));
    return modules::LinearModule({dim, dim, false, GGML_PREC_F32}).build(ctx, context, weights.to_out);
}

core::TensorValue same_feed_forward(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const StableAudioSameFeedForwardWeights & weights,
    bool sinusoidal) {
    const int64_t dim = input.shape.last_dim();
    const int64_t inner = dim * 3;
    auto projected = modules::LinearModule({dim, 2 * inner, true, GGML_PREC_F32}).build(ctx, input, weights.in_proj);
    auto value = modules::SliceModule({2, 0, inner}).build(ctx, projected);
    auto gate = modules::SliceModule({2, inner, inner}).build(ctx, projected);
    if (sinusoidal) {
        gate = core::wrap_tensor(ggml_sin(ctx.ggml, scale_tensor(ctx, gate, kPi).tensor), gate.shape, GGML_TYPE_F32);
    } else {
        gate = modules::SiluModule{}.build(ctx, gate);
    }
    auto hidden = modules::MulModule{}.build(ctx, value, gate);
    return modules::LinearModule({inner, dim, true, GGML_PREC_F32}).build(ctx, hidden, weights.out_proj);
}

core::TensorValue same_transformer_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue * attention_mask,
    const StableAudioSameTransformerWeights & weights,
    const StableAudioConfig & config,
    bool sinusoidal) {
    auto attn_in = dynamic_tanh_norm(ctx, input, weights.pre_norm);
    auto attn = same_self_attention(ctx, attn_in, positions, attention_mask, weights.self_attn, config, input.shape.last_dim());
    auto hidden = modules::AddModule{}.build(ctx, input, attn);
    auto ff_in = dynamic_tanh_norm(ctx, hidden, weights.ff_norm);
    auto ff = same_feed_forward(ctx, ff_in, weights.ff, sinusoidal);
    return modules::AddModule{}.build(ctx, hidden, ff);
}

std::vector<float> adapt_interleaved_audio(
    const runtime::AudioBuffer & audio,
    int target_sample_rate,
    int target_channels,
    int64_t target_frames) {
    if (audio.sample_rate <= 0 || audio.channels <= 0 || target_sample_rate <= 0 || target_channels <= 0 || target_frames <= 0) {
        throw std::runtime_error("Stable Audio SAME encode received invalid audio shape");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("Stable Audio SAME encode audio samples are not divisible by channels");
    }
    std::vector<float> samples = audio.samples;
    int channels = audio.channels;
    if (audio.sample_rate != target_sample_rate) {
        const int64_t frames = static_cast<int64_t>(samples.size() / static_cast<size_t>(channels));
        std::vector<std::vector<float>> resampled_channels;
        resampled_channels.reserve(static_cast<size_t>(channels));
        int64_t resampled_frames = -1;
        for (int ch = 0; ch < channels; ++ch) {
            std::vector<float> mono(static_cast<size_t>(frames), 0.0F);
            for (int64_t frame = 0; frame < frames; ++frame) {
                mono[static_cast<size_t>(frame)] = samples[static_cast<size_t>(frame * channels + ch)];
            }
            engine::audio::SoxrResampleOptions options;
            options.profile = engine::audio::SoxrResampleProfile::QualityOnly;
            options.output_length_policy = engine::audio::SoxrOutputLengthPolicy::ExactExpected;
            options.warning_context = "stable_audio.same_encode";
            options.fallback_description = "linear resampling";
            auto resampled = engine::audio::resample_mono_soxr_or_linear(
                mono,
                audio.sample_rate,
                target_sample_rate,
                options);
            if (resampled_frames < 0) {
                resampled_frames = static_cast<int64_t>(resampled.size());
            } else if (resampled_frames != static_cast<int64_t>(resampled.size())) {
                throw std::runtime_error("Stable Audio SAME encode resampled channels have inconsistent lengths");
            }
            resampled_channels.push_back(std::move(resampled));
        }
        samples.assign(static_cast<size_t>(resampled_frames * channels), 0.0F);
        for (int64_t frame = 0; frame < resampled_frames; ++frame) {
            for (int ch = 0; ch < channels; ++ch) {
                samples[static_cast<size_t>(frame * channels + ch)] =
                    resampled_channels[static_cast<size_t>(ch)][static_cast<size_t>(frame)];
            }
        }
    }
    if (channels != target_channels) {
        if (target_channels == 1) {
            samples = engine::audio::mixdown_interleaved_to_mono_average(samples, channels);
        } else if (channels == 1) {
            samples = engine::audio::duplicate_mono_to_interleaved_channels(samples, target_channels);
        } else {
            auto mono = engine::audio::mixdown_interleaved_to_mono_average(samples, channels);
            samples = engine::audio::duplicate_mono_to_interleaved_channels(mono, target_channels);
        }
        channels = target_channels;
    }
    const size_t target_samples = static_cast<size_t>(target_frames * target_channels);
    if (samples.size() < target_samples) {
        samples.resize(target_samples, 0.0F);
    } else if (samples.size() > target_samples) {
        samples.resize(target_samples);
    }
    return samples;
}

StableAudioSameNormWeights load_dyt_norm(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t dim) {
    StableAudioSameNormWeights weights;
    weights.alpha = store.load_f32_tensor(source, prefix + ".alpha", {1});
    weights.beta = store.load_f32_tensor(source, prefix + ".beta", {dim});
    weights.gamma = store.load_f32_tensor(source, prefix + ".gamma", {dim});
    return weights;
}

StableAudioSameAttentionWeights load_same_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const StableAudioConfig & config,
    int64_t dim,
    assets::TensorStorageType storage_type) {
    StableAudioSameAttentionWeights weights;
    weights.to_qkv = load_linear(
        store,
        source,
        prefix + ".to_qkv",
        storage_type,
        dim * (config.same_differential ? 5 : 3),
        dim,
        false);
    weights.to_out = load_linear(store, source, prefix + ".to_out", storage_type, dim, dim, false);
    weights.q_norm = load_dyt_norm(store, source, prefix + ".q_norm", config.same_dim_heads);
    weights.k_norm = load_dyt_norm(store, source, prefix + ".k_norm", config.same_dim_heads);
    return weights;
}

StableAudioSameFeedForwardWeights load_same_ff(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t dim,
    assets::TensorStorageType storage_type) {
    StableAudioSameFeedForwardWeights weights;
    const int64_t inner = dim * 3;
    weights.in_proj = load_linear(store, source, prefix + ".ff.0.proj", storage_type, 2 * inner, dim, true);
    weights.out_proj = load_linear(store, source, prefix + ".ff.2", storage_type, dim, inner, true);
    return weights;
}

StableAudioSameTransformerWeights load_same_transformer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const StableAudioConfig & config,
    int64_t dim,
    assets::TensorStorageType storage_type) {
    StableAudioSameTransformerWeights weights;
    weights.pre_norm = load_dyt_norm(store, source, prefix + ".pre_norm", dim);
    weights.ff_norm = load_dyt_norm(store, source, prefix + ".ff_norm", dim);
    weights.self_attn = load_same_attention(store, source, prefix + ".self_attn", config, dim, storage_type);
    weights.ff = load_same_ff(store, source, prefix + ".ff", dim, storage_type);
    return weights;
}

StableAudioSameResamplingBlockWeights load_decoder_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const StableAudioConfig & config,
    assets::TensorStorageType storage_type) {
    const int64_t block_index = 3;
    const int64_t in_channels = config.same_channels * config.same_c_mults.back();
    const int64_t out_channels = config.same_decoder_out_channels;
    const int64_t kernel = config.same_decoder_conv_mapping ? 3 : 1;
    const std::string prefix = "pretransform.model.decoder.layers." + std::to_string(block_index);
    StableAudioSameResamplingBlockWeights weights;
    weights.mapping = load_weight_norm_conv1d(
        store,
        source,
        prefix + ".mapping",
        storage_type,
        out_channels,
        in_channels,
        kernel,
        true);
    weights.new_tokens = store.load_f32_tensor(source, prefix + ".new_tokens", {1, 1, in_channels});
    const int64_t depth = config.same_decoder_transformer_depths.back();
    weights.transformers.reserve(static_cast<size_t>(depth));
    for (int64_t layer = 0; layer < depth; ++layer) {
        weights.transformers.push_back(load_same_transformer(
            store,
            source,
            prefix + ".transformers." + std::to_string(layer),
            config,
            in_channels,
            storage_type));
    }
    return weights;
}

StableAudioSameResamplingBlockWeights load_encoder_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const StableAudioConfig & config,
    assets::TensorStorageType storage_type) {
    const int64_t block_index = 0;
    const int64_t in_channels = config.same_encoder_in_channels;
    const int64_t out_channels = config.same_channels * config.same_c_mults.front();
    const int64_t kernel = config.same_encoder_conv_mapping ? 3 : 1;
    const std::string prefix = "pretransform.model.encoder.layers." + std::to_string(block_index);
    StableAudioSameResamplingBlockWeights weights;
    weights.mapping = load_weight_norm_conv1d(
        store,
        source,
        prefix + ".mapping",
        storage_type,
        out_channels,
        in_channels,
        kernel,
        true);
    weights.new_tokens = store.load_f32_tensor(source, prefix + ".new_tokens", {1, 1, out_channels});
    const int64_t depth = config.same_encoder_transformer_depths.front();
    weights.transformers.reserve(static_cast<size_t>(depth));
    for (int64_t layer = 0; layer < depth; ++layer) {
        weights.transformers.push_back(load_same_transformer(
            store,
            source,
            prefix + ".transformers." + std::to_string(layer),
            config,
            out_channels,
            storage_type));
    }
    return weights;
}

}  // namespace

StableAudioSameWeights load_stable_audio_same_weights(
    const StableAudioAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type) {
    const auto & config = assets.config;
    const auto & source = *assets.model_weights;
    StableAudioSameWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "stable_audio.same.weights",
        weight_context_bytes != 0 ? weight_context_bytes :
            (config.medium_architecture ? kSameWeightContextBytesMedium : kSameWeightContextBytesSmall));
    weights.bottleneck_bias = weights.store->load_f32_tensor(source, "pretransform.model.bottleneck.bias", {1, config.latent_dim, 1});
    weights.bottleneck_scaling_factor = weights.store->load_f32_tensor(source, "pretransform.model.bottleneck.scaling_factor", {1, config.latent_dim, 1});
    weights.bottleneck_running_std = weights.store->load_f32_tensor(source, "pretransform.model.bottleneck.running_std", {1});
    weights.decoder_in = load_linear(
        *weights.store,
        source,
        "pretransform.model.decoder.layers.1",
        weight_storage_type,
        config.same_channels * config.same_c_mults.back(),
        config.latent_dim,
        true);
    weights.decoder_block = load_decoder_block(*weights.store, source, config, weight_storage_type);
    weights.encoder_block = load_encoder_block(*weights.store, source, config, weight_storage_type);
    weights.encoder_out = load_linear(
        *weights.store,
        source,
        "pretransform.model.encoder.layers.2",
        weight_storage_type,
        config.latent_dim,
        config.same_channels * config.same_c_mults.front(),
        true);
    weights.store->upload();
    return weights;
}

class StableAudioSameRuntime::Graph {
public:
    Graph(
        core::ExecutionContext & execution,
        std::shared_ptr<const StableAudioAssets> assets,
        const StableAudioSameWeights & weights,
        int64_t batch,
        int64_t latent_tokens,
        assets::TensorStorageType weight_storage_type)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          assets_(std::move(assets)),
          batch_(batch),
          latent_tokens_(latent_tokens),
          dim_(assets_->config.same_channels * assets_->config.same_c_mults.back()),
          block_tokens_(assets_->config.medium_architecture ? latent_tokens_ * (assets_->config.same_strides.back() + 1) :
                                                      assets_->config.pretransform_encoder_chunk_size +
                                                          assets_->config.pretransform_encoder_chunk_size /
                                                              assets_->config.same_strides.back()),
          weights_(weights) {
        (void) weight_storage_type;
        if (backend_ == nullptr) {
            throw std::runtime_error("Stable Audio SAME backend initialization failed");
        }
        if (batch_ <= 0 || latent_tokens_ <= 0) {
            throw std::runtime_error("Stable Audio SAME graph requires positive batch");
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

    bool matches(int64_t batch, int64_t latent_tokens) const noexcept {
        return batch == batch_ && latent_tokens == latent_tokens_;
    }

    std::vector<float> decode_chunk(
        const std::vector<float> & latents,
        const std::vector<float> & bottleneck_noise,
        const std::vector<float> & token_noise,
        double & input_upload_ms,
        double & graph_compute_ms,
        double & output_read_ms) const {
        const auto & config = assets_->config;
        const size_t latent_values = static_cast<size_t>(batch_ * config.latent_dim * latent_tokens_);
        const size_t token_noise_values = static_cast<size_t>(batch_ * latent_tokens_ * config.same_strides.back() * dim_);
        if (latents.size() != latent_values ||
            bottleneck_noise.size() != latent_values ||
            token_noise.size() != token_noise_values) {
            throw std::runtime_error("Stable Audio SAME decode chunk input shape mismatch");
        }
        const auto input_start = Clock::now();
        core::write_tensor_f32(input_, latents);
        core::write_tensor_f32(bottleneck_noise_, bottleneck_noise);
        core::write_tensor_f32(new_token_noise_, token_noise);
        core::write_tensor_i32(positions_, positions_values_);
        if (config.medium_architecture) {
            core::write_tensor_f16(attention_mask_, attention_mask_values_);
        }
        input_upload_ms += engine::debug::elapsed_ms(input_start, Clock::now());
        core::set_backend_threads(backend_, threads_);
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_, nullptr, "stable_audio.same_decode");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Stable Audio SAME decode graph compute failed");
        }
        const double compute_ms = engine::debug::elapsed_ms(compute_start, Clock::now());
        graph_compute_ms += compute_ms;
        const auto output_start = Clock::now();
        const size_t output_values = static_cast<size_t>(
            batch_ * config.audio_channels * latent_tokens_ * config.downsampling_ratio);
        std::vector<float> output(output_values, 0.0F);
        ggml_backend_tensor_get(output_, output.data(), 0, output.size() * sizeof(float));
        output_read_ms += engine::debug::elapsed_ms(output_start, Clock::now());
        for (const float value : output) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("Stable Audio SAME decode produced non-finite output");
            }
        }
        return output;
    }

private:
    void build() {
        const auto & config = assets_->config;
        ggml_init_params params{config.medium_architecture ? 1024ull * 1024ull * 1024ull : 512ull * 1024ull * 1024ull, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("Stable Audio SAME ggml context initialization failed");
        }
        auto input_ctx = ctx_for_inputs();
        input_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({batch_, config.latent_dim, latent_tokens_}));
        bottleneck_noise_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({batch_, config.latent_dim, latent_tokens_}));
        new_token_noise_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({batch_ * latent_tokens_, config.same_strides.back(), dim_}));
        positions_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({block_tokens_}));
        ggml_set_input(input_.tensor);
        ggml_set_input(bottleneck_noise_.tensor);
        ggml_set_input(new_token_noise_.tensor);
        ggml_set_input(positions_.tensor);
        if (config.medium_architecture) {
            attention_mask_ = core::make_tensor(
                input_ctx,
                GGML_TYPE_F16,
                core::TensorShape::from_dims({batch_, 1, block_tokens_, block_tokens_}));
            ggml_set_input(attention_mask_.tensor);
        }
        core::ModuleBuildContext build_ctx{ctx_.get(), "stable_audio.same_decode", backend_type_};
        auto output = build_graph_output(build_ctx);
        output_ = output.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("Stable Audio SAME decode backend buffer allocation failed");
        }
        positions_values_.assign(static_cast<size_t>(block_tokens_), 0);
        for (int64_t i = 0; i < block_tokens_; ++i) {
            positions_values_[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        core::write_tensor_i32(positions_, positions_values_);
        if (config.medium_architecture) {
            attention_mask_values_ = make_sliding_attention_mask();
            core::write_tensor_f16(attention_mask_, attention_mask_values_);
        }
    }

    core::ModuleBuildContext ctx_for_inputs() {
        return core::ModuleBuildContext{ctx_.get(), "stable_audio.same_decode.inputs", backend_type_};
    }

    std::vector<float> make_sliding_attention_mask() const {
        std::vector<float> mask(static_cast<size_t>(batch_ * block_tokens_ * block_tokens_), -std::numeric_limits<float>::infinity());
        const int64_t left = config_window_left();
        const int64_t right = config_window_right();
        for (int64_t b = 0; b < batch_; ++b) {
            for (int64_t q = 0; q < block_tokens_; ++q) {
                for (int64_t k = std::max<int64_t>(0, q - left); k <= std::min<int64_t>(block_tokens_ - 1, q + right); ++k) {
                    mask[static_cast<size_t>((b * block_tokens_ + q) * block_tokens_ + k)] = 0.0F;
                }
            }
        }
        return mask;
    }

    int64_t config_window_left() const {
        return assets_->config.same_sliding_window.empty() ? 0 :
            assets_->config.same_sliding_window.front() * (assets_->config.same_strides.back() + 1);
    }

    int64_t config_window_right() const {
        return assets_->config.same_sliding_window.size() < 2 ? config_window_left() :
            assets_->config.same_sliding_window[1] * (assets_->config.same_strides.back() + 1);
    }

    core::TensorValue build_graph_output(core::ModuleBuildContext & ctx) {
        const auto & config = assets_->config;
        auto running_std = core::reshape_tensor(ctx, weights_.bottleneck_running_std, core::TensorShape::from_dims({1, 1, 1}));
        running_std = modules::RepeatModule({input_.shape}).build(ctx, running_std);
        auto hidden = modules::MulModule{}.build(ctx, input_, running_std);
        auto noise = modules::MulModule{}.build(ctx, bottleneck_noise_, running_std);
        noise = scale_tensor(ctx, noise, kSameBottleneckEvalNoiseScale);
        hidden = modules::AddModule{}.build(ctx, hidden, noise);

        hidden = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, hidden);
        hidden = modules::LinearModule({config.latent_dim, dim_, true, GGML_PREC_F32}).build(ctx, hidden, weights_.decoder_in);
        hidden = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, hidden);
        hidden = build_decoder_block(ctx, hidden);
        return hidden;
    }

    core::TensorValue build_decoder_block(core::ModuleBuildContext & ctx, const core::TensorValue & input_bct) {
        const auto & config = assets_->config;
        const int64_t stride = config.same_strides.back();
        auto hidden = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, input_bct);
        hidden = core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, hidden),
            core::TensorShape::from_dims({batch_ * latent_tokens_, 1, dim_}));
        auto new_tokens = core::reshape_tensor(ctx, weights_.decoder_block.new_tokens, core::TensorShape::from_dims({1, 1, dim_}));
        new_tokens = modules::RepeatModule({core::TensorShape::from_dims({batch_ * latent_tokens_, stride, dim_})}).build(ctx, new_tokens);
        if (config.same_decoder_mask_noise > 0.0F) {
            new_tokens = modules::AddModule{}.build(ctx, new_tokens, scale_tensor(ctx, new_token_noise_, config.same_decoder_mask_noise));
        }
        hidden = modules::ConcatModule({1}).build(ctx, hidden, new_tokens);
        hidden = core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, hidden),
            core::TensorShape::from_dims({batch_, latent_tokens_ * (stride + 1), dim_}));

        if (config.medium_architecture) {
            const core::TensorValue * mask = attention_mask_.valid() ? &attention_mask_ : nullptr;
            const int64_t sinusoidal = config.same_decoder_sinusoidal_blocks.empty() ? 0 : config.same_decoder_sinusoidal_blocks.back();
            const int64_t depth = static_cast<int64_t>(weights_.decoder_block.transformers.size());
            for (int64_t i = 0; i < depth; ++i) {
                const bool use_sin = (depth - i) < sinusoidal;
                hidden = same_transformer_layer(
                    ctx,
                    hidden,
                    positions_,
                    mask,
                    weights_.decoder_block.transformers[static_cast<size_t>(i)],
                    config,
                    use_sin);
            }
        } else if (config.same_chunk_midpoint_shift) {
            hidden = build_chunk_midpoint_decoder(ctx, hidden);
        } else {
            hidden = core::reshape_tensor(
                ctx,
                ensure_contiguous(ctx, hidden),
                core::TensorShape::from_dims({batch_ * (hidden.shape.dims[1] / block_tokens_), block_tokens_, dim_}));
            for (const auto & layer : weights_.decoder_block.transformers) {
                hidden = same_transformer_layer(ctx, hidden, positions_, nullptr, layer, config, false);
            }
            hidden = core::reshape_tensor(
                ctx,
                ensure_contiguous(ctx, hidden),
                core::TensorShape::from_dims({batch_, latent_tokens_ * (stride + 1), dim_}));
        }

        hidden = core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, hidden),
            core::TensorShape::from_dims({batch_ * latent_tokens_, stride + 1, dim_}));
        hidden = modules::SliceModule({1, 1, stride}).build(ctx, hidden);
        hidden = core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, hidden),
            core::TensorShape::from_dims({batch_, latent_tokens_ * stride, dim_}));
        hidden = core::wrap_tensor(ggml_cont(ctx.ggml, hidden.tensor), hidden.shape, hidden.type);
        hidden = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, hidden);
        const int64_t kernel = config.same_decoder_conv_mapping ? 3 : 1;
        hidden = modules::Conv1dModule({
            dim_,
            config.same_decoder_out_channels,
            kernel,
            1,
            static_cast<int>(kernel / 2),
            1,
            true,
        }).build(ctx, hidden, weights_.decoder_block.mapping);
        hidden = core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, hidden),
            core::TensorShape::from_dims({
                batch_,
                config.audio_channels,
                config.pretransform_patch_size,
                latent_tokens_ * stride,
            }));
        hidden = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, hidden);
        hidden = core::wrap_tensor(ggml_cont(ctx.ggml, hidden.tensor), hidden.shape, hidden.type);
        return core::reshape_tensor(
            ctx,
            hidden,
            core::TensorShape::from_dims({batch_, config.audio_channels, latent_tokens_ * config.downsampling_ratio}));
    }

    core::TensorValue build_chunk_midpoint_decoder(core::ModuleBuildContext & ctx, core::TensorValue hidden) {
        const auto & config = assets_->config;
        const int64_t depth = static_cast<int64_t>(weights_.decoder_block.transformers.size());
        const int64_t split = depth / 2;
        const int64_t shift = block_tokens_ / 2;
        const int64_t chunks = hidden.shape.dims[1] / block_tokens_;
        hidden = core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, hidden),
            core::TensorShape::from_dims({batch_ * chunks, block_tokens_, dim_}));
        for (int64_t i = 0; i < split; ++i) {
            hidden = same_transformer_layer(
                ctx,
                hidden,
                positions_,
                nullptr,
                weights_.decoder_block.transformers[static_cast<size_t>(i)],
                config,
                false);
        }
        hidden = core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, hidden),
            core::TensorShape::from_dims({batch_, chunks * block_tokens_, dim_}));
        const auto prefix = modules::SliceModule({1, 0, shift}).build(ctx, hidden);
        const auto suffix = modules::SliceModule({1, hidden.shape.dims[1] - shift, shift}).build(ctx, hidden);
        hidden = modules::ConcatModule({1}).build(ctx, modules::ConcatModule({1}).build(ctx, prefix, hidden), suffix);
        const int64_t shifted_chunks = hidden.shape.dims[1] / block_tokens_;
        hidden = core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, hidden),
            core::TensorShape::from_dims({batch_ * shifted_chunks, block_tokens_, dim_}));
        for (int64_t i = split; i < depth; ++i) {
            hidden = same_transformer_layer(
                ctx,
                hidden,
                positions_,
                nullptr,
                weights_.decoder_block.transformers[static_cast<size_t>(i)],
                config,
                false);
        }
        hidden = core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, hidden),
            core::TensorShape::from_dims({batch_, shifted_chunks * block_tokens_, dim_}));
        return modules::SliceModule({1, shift, hidden.shape.dims[1] - 2 * shift}).build(ctx, hidden);
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const StableAudioAssets> assets_;
    int64_t batch_ = 0;
    int64_t latent_tokens_ = 0;
    int64_t dim_ = 0;
    int64_t block_tokens_ = 0;
    const StableAudioSameWeights & weights_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue input_;
    core::TensorValue bottleneck_noise_;
    core::TensorValue new_token_noise_;
    core::TensorValue positions_;
    core::TensorValue attention_mask_;
    ggml_tensor * output_ = nullptr;
    std::vector<int32_t> positions_values_;
    std::vector<float> attention_mask_values_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

class StableAudioSameRuntime::EncoderGraph {
public:
    EncoderGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const StableAudioAssets> assets,
        const StableAudioSameWeights & weights,
        assets::TensorStorageType weight_storage_type)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          assets_(std::move(assets)),
          dim_(assets_->config.same_channels * assets_->config.same_c_mults.front()),
          block_tokens_(assets_->config.medium_architecture ? kSameChunkLatents * (assets_->config.same_strides.front() + 1) :
                                                      assets_->config.pretransform_encoder_chunk_size +
                                                          assets_->config.pretransform_encoder_chunk_size /
                                                              assets_->config.same_strides.front()),
          weights_(weights) {
        (void) weight_storage_type;
        if (backend_ == nullptr) {
            throw std::runtime_error("Stable Audio SAME encoder backend initialization failed");
        }
        build();
    }

    ~EncoderGraph() {
        if (backend_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(backend_type_, backend_, graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    std::vector<float> encode_chunk(
        const std::vector<float> & patched_audio,
        const std::vector<float> & token_noise) const {
        const auto & config = assets_->config;
        const size_t input_values = static_cast<size_t>(
            config.same_encoder_in_channels * kSameChunkLatents * config.same_strides.front());
        const size_t token_values = static_cast<size_t>(kSameChunkLatents * dim_);
        if (patched_audio.size() != input_values ||
            (config.same_encoder_mask_noise > 0.0F && token_noise.size() != token_values)) {
            throw std::runtime_error("Stable Audio SAME encoder chunk input shape mismatch");
        }
        core::write_tensor_f32(input_, patched_audio);
        if (config.same_encoder_mask_noise > 0.0F) {
            core::write_tensor_f32(new_token_noise_, token_noise);
        }
        core::write_tensor_i32(positions_, positions_values_);
        if (config.medium_architecture) {
            core::write_tensor_f16(attention_mask_, attention_mask_values_);
        }
        core::set_backend_threads(backend_, threads_);
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_, nullptr, "stable_audio.same_encode");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Stable Audio SAME encoder graph compute failed");
        }
        engine::debug::timing_log_scalar(
            "stable_audio.same_encode.chunk.graph.compute_ms",
            engine::debug::elapsed_ms(compute_start, Clock::now()));
        auto output = core::read_tensor_f32(output_);
        for (const float value : output) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("Stable Audio SAME encode produced non-finite latent");
            }
        }
        return output;
    }

private:
    void build() {
        const auto & config = assets_->config;
        ggml_init_params params{config.medium_architecture ? 1024ull * 1024ull * 1024ull : 512ull * 1024ull * 1024ull, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("Stable Audio SAME encoder ggml context initialization failed");
        }
        auto input_ctx = ctx_for_inputs();
        input_ = core::make_tensor(
            input_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, config.same_encoder_in_channels, kSameChunkLatents * config.same_strides.front()}));
        positions_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({block_tokens_}));
        ggml_set_input(input_.tensor);
        ggml_set_input(positions_.tensor);
        if (config.same_encoder_mask_noise > 0.0F) {
            new_token_noise_ = core::make_tensor(
                input_ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({kSameChunkLatents, 1, dim_}));
            ggml_set_input(new_token_noise_.tensor);
        }
        if (config.medium_architecture) {
            attention_mask_ = core::make_tensor(
                input_ctx,
                GGML_TYPE_F16,
                core::TensorShape::from_dims({1, 1, block_tokens_, block_tokens_}));
            ggml_set_input(attention_mask_.tensor);
        }
        core::ModuleBuildContext build_ctx{ctx_.get(), "stable_audio.same_encode", backend_type_};
        auto output = build_graph_output(build_ctx);
        output_ = output.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("Stable Audio SAME encoder backend buffer allocation failed");
        }
        positions_values_.assign(static_cast<size_t>(block_tokens_), 0);
        for (int64_t i = 0; i < block_tokens_; ++i) {
            positions_values_[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        core::write_tensor_i32(positions_, positions_values_);
        if (config.medium_architecture) {
            attention_mask_values_ = make_sliding_attention_mask();
            core::write_tensor_f16(attention_mask_, attention_mask_values_);
        }
    }

    core::ModuleBuildContext ctx_for_inputs() {
        return core::ModuleBuildContext{ctx_.get(), "stable_audio.same_encode.inputs", backend_type_};
    }

    std::vector<float> make_sliding_attention_mask() const {
        std::vector<float> mask(static_cast<size_t>(block_tokens_ * block_tokens_), -std::numeric_limits<float>::infinity());
        const int64_t window = assets_->config.same_sliding_window.empty() ? 0 :
            assets_->config.same_sliding_window.front() * (assets_->config.same_strides.front() + 1);
        for (int64_t q = 0; q < block_tokens_; ++q) {
            for (int64_t k = std::max<int64_t>(0, q - window); k <= std::min<int64_t>(block_tokens_ - 1, q + window); ++k) {
                mask[static_cast<size_t>(q * block_tokens_ + k)] = 0.0F;
            }
        }
        return mask;
    }

    core::TensorValue build_graph_output(core::ModuleBuildContext & ctx) const {
        const auto & config = assets_->config;
        const int64_t stride = config.same_strides.front();
        const int64_t kernel = config.same_encoder_conv_mapping ? 3 : 1;
        auto hidden = modules::Conv1dModule({
            config.same_encoder_in_channels,
            dim_,
            kernel,
            1,
            static_cast<int>(kernel / 2),
            1,
            true,
        }).build(ctx, input_, weights_.encoder_block.mapping);
        hidden = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, hidden);
        hidden = core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, hidden),
            core::TensorShape::from_dims({kSameChunkLatents, stride, dim_}));
        auto new_tokens = core::reshape_tensor(ctx, weights_.encoder_block.new_tokens, core::TensorShape::from_dims({1, 1, dim_}));
        new_tokens = modules::RepeatModule({core::TensorShape::from_dims({kSameChunkLatents, 1, dim_})}).build(ctx, new_tokens);
        if (config.same_encoder_mask_noise > 0.0F) {
            new_tokens = modules::AddModule{}.build(ctx, new_tokens, scale_tensor(ctx, new_token_noise_, config.same_encoder_mask_noise));
        }
        hidden = modules::ConcatModule({1}).build(ctx, hidden, new_tokens);
        hidden = core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, hidden),
            core::TensorShape::from_dims({1, kSameChunkLatents * (stride + 1), dim_}));
        if (config.medium_architecture) {
            const core::TensorValue * mask = attention_mask_.valid() ? &attention_mask_ : nullptr;
            for (const auto & layer : weights_.encoder_block.transformers) {
                hidden = same_transformer_layer(ctx, hidden, positions_, mask, layer, config, false);
            }
        } else if (config.same_chunk_midpoint_shift) {
            hidden = build_chunk_midpoint_encoder(ctx, hidden);
        } else {
            hidden = core::reshape_tensor(
                ctx,
                ensure_contiguous(ctx, hidden),
                core::TensorShape::from_dims({hidden.shape.dims[1] / block_tokens_, block_tokens_, dim_}));
            for (const auto & layer : weights_.encoder_block.transformers) {
                hidden = same_transformer_layer(ctx, hidden, positions_, nullptr, layer, config, false);
            }
            hidden = core::reshape_tensor(
                ctx,
                ensure_contiguous(ctx, hidden),
                core::TensorShape::from_dims({1, kSameChunkLatents * (stride + 1), dim_}));
        }
        hidden = core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, hidden),
            core::TensorShape::from_dims({kSameChunkLatents, stride + 1, dim_}));
        hidden = modules::SliceModule({1, stride, 1}).build(ctx, hidden);
        hidden = core::reshape_tensor(ctx, ensure_contiguous(ctx, hidden), core::TensorShape::from_dims({1, kSameChunkLatents, dim_}));
        hidden = modules::LinearModule({dim_, config.latent_dim, true, GGML_PREC_F32}).build(ctx, hidden, weights_.encoder_out);
        hidden = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, hidden);
        auto scale = modules::RepeatModule({hidden.shape}).build(ctx, weights_.bottleneck_scaling_factor);
        auto bias = modules::RepeatModule({hidden.shape}).build(ctx, weights_.bottleneck_bias);
        hidden = modules::AddModule{}.build(ctx, modules::MulModule{}.build(ctx, hidden, scale), bias);
        auto running_std = core::reshape_tensor(ctx, weights_.bottleneck_running_std, core::TensorShape::from_dims({1, 1, 1}));
        running_std = modules::RepeatModule({hidden.shape}).build(ctx, running_std);
        return core::wrap_tensor(ggml_div(ctx.ggml, ensure_contiguous(ctx, hidden).tensor, ensure_contiguous(ctx, running_std).tensor), hidden.shape, GGML_TYPE_F32);
    }

    core::TensorValue build_chunk_midpoint_encoder(core::ModuleBuildContext & ctx, core::TensorValue hidden) const {
        const auto & config = assets_->config;
        const int64_t depth = static_cast<int64_t>(weights_.encoder_block.transformers.size());
        const int64_t split = depth / 2;
        const int64_t shift = block_tokens_ / 2;
        const int64_t chunks = hidden.shape.dims[1] / block_tokens_;
        hidden = core::reshape_tensor(
            ctx,
            ensure_contiguous(ctx, hidden),
            core::TensorShape::from_dims({chunks, block_tokens_, dim_}));
        for (int64_t i = 0; i < split; ++i) {
            hidden = same_transformer_layer(ctx, hidden, positions_, nullptr, weights_.encoder_block.transformers[static_cast<size_t>(i)], config, false);
        }
        hidden = core::reshape_tensor(ctx, ensure_contiguous(ctx, hidden), core::TensorShape::from_dims({1, chunks * block_tokens_, dim_}));
        const auto prefix = modules::SliceModule({1, 0, shift}).build(ctx, hidden);
        const auto suffix = modules::SliceModule({1, hidden.shape.dims[1] - shift, shift}).build(ctx, hidden);
        hidden = modules::ConcatModule({1}).build(ctx, modules::ConcatModule({1}).build(ctx, prefix, hidden), suffix);
        const int64_t shifted_chunks = hidden.shape.dims[1] / block_tokens_;
        hidden = core::reshape_tensor(ctx, ensure_contiguous(ctx, hidden), core::TensorShape::from_dims({shifted_chunks, block_tokens_, dim_}));
        for (int64_t i = split; i < depth; ++i) {
            hidden = same_transformer_layer(ctx, hidden, positions_, nullptr, weights_.encoder_block.transformers[static_cast<size_t>(i)], config, false);
        }
        hidden = core::reshape_tensor(ctx, ensure_contiguous(ctx, hidden), core::TensorShape::from_dims({1, shifted_chunks * block_tokens_, dim_}));
        return modules::SliceModule({1, shift, hidden.shape.dims[1] - 2 * shift}).build(ctx, hidden);
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const StableAudioAssets> assets_;
    int64_t dim_ = 0;
    int64_t block_tokens_ = 0;
    const StableAudioSameWeights & weights_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue input_;
    core::TensorValue new_token_noise_;
    core::TensorValue positions_;
    core::TensorValue attention_mask_;
    ggml_tensor * output_ = nullptr;
    std::vector<int32_t> positions_values_;
    std::vector<float> attention_mask_values_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

StableAudioSameRuntime::StableAudioSameRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const StableAudioAssets> assets,
    assets::TensorStorageType weight_storage_type)
    : execution_(&execution),
      assets_(std::move(assets)),
      weight_storage_type_(weight_storage_type),
      rng_policy_(engine::sampling::resolve_torch_cuda_sampling_policy(
          execution.backend_type(),
          execution.config().device,
          "stable_audio.same.rng",
          "Stable Audio SAME")) {
    if (execution_ == nullptr) {
        throw std::runtime_error("Stable Audio SAME runtime requires execution context");
    }
    if (assets_ == nullptr) {
        throw std::runtime_error("Stable Audio SAME runtime requires assets");
    }
}

StableAudioSameRuntime::~StableAudioSameRuntime() = default;

const StableAudioSameWeights & require_same_weights(
    std::unique_ptr<StableAudioSameWeights> & weights,
    core::ExecutionContext & execution,
    const std::shared_ptr<const StableAudioAssets> & assets,
    assets::TensorStorageType weight_storage_type) {
    if (!weights) {
        weights = std::make_unique<StableAudioSameWeights>(load_stable_audio_same_weights(
            *assets,
            execution.backend(),
            execution.backend_type(),
            0,
            weight_storage_type));
    }
    return *weights;
}

void StableAudioSameRuntime::prepare_decode(int64_t batch, int64_t latent_tokens, bool chunked) const {
    const auto & weights = require_same_weights(weights_, *execution_, assets_, weight_storage_type_);
    const int64_t graph_latents = decoder_graph_latents(assets_->config, latent_tokens, chunked);
    if (!graph_ || !graph_->matches(batch, graph_latents)) {
        graph_ = std::make_unique<Graph>(*execution_, assets_, weights, batch, graph_latents, weight_storage_type_);
    }
}

std::vector<runtime::AudioBuffer> StableAudioSameRuntime::decode(
    const std::vector<float> & latents,
    int64_t batch,
    int64_t latent_tokens,
    uint64_t seed,
    uint64_t & rng_offset_blocks,
    bool chunked) const {
    const auto & config = assets_->config;
    if (static_cast<int64_t>(latents.size()) != batch * config.latent_dim * latent_tokens) {
        throw std::runtime_error("Stable Audio SAME decode latent shape mismatch");
    }
    const int64_t graph_latents = decoder_graph_latents(config, latent_tokens, chunked);
    prepare_decode(batch, latent_tokens, chunked);
    std::vector<int64_t> starts;
    if (!chunked || latent_tokens <= kSameChunkLatents) {
        starts.push_back(0);
    } else {
        const int64_t hop = kSameChunkLatents - kSameChunkOverlap;
        for (int64_t start = 0; start <= latent_tokens - kSameChunkLatents; start += hop) {
            starts.push_back(start);
        }
        const int64_t final_start = latent_tokens - kSameChunkLatents;
        if (starts.empty() || starts.back() != final_start) {
            starts.push_back(final_start);
        }
    }
    const int64_t frames_per_chunk = graph_latents * config.downsampling_ratio;
    const int64_t total_frames = latent_tokens * config.downsampling_ratio;
    std::vector<float> channel_major(static_cast<size_t>(batch * config.audio_channels * total_frames), 0.0F);
    std::vector<float> chunk_latent(static_cast<size_t>(batch * config.latent_dim * graph_latents), 0.0F);
    std::vector<float> bottleneck_noise(static_cast<size_t>(batch * config.latent_dim * graph_latents), 0.0F);
    std::vector<float> bottleneck_noise_source;
    std::vector<float> token_noise;
    double chunk_latent_ms = 0.0;
    double bottleneck_noise_ms = 0.0;
    double token_noise_ms = 0.0;
    double input_upload_ms = 0.0;
    double graph_compute_ms = 0.0;
    double output_read_ms = 0.0;
    double output_pack_ms = 0.0;
    for (size_t chunk_index = 0; chunk_index < starts.size(); ++chunk_index) {
        const int64_t start = starts[chunk_index];
        const auto latent_start = Clock::now();
        std::fill(chunk_latent.begin(), chunk_latent.end(), 0.0F);
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < config.latent_dim; ++c) {
                for (int64_t t = 0; t < graph_latents; ++t) {
                    const int64_t src_t = start + t;
                    if (src_t < latent_tokens) {
                        chunk_latent[static_cast<size_t>((b * config.latent_dim + c) * graph_latents + t)] =
                            latents[static_cast<size_t>((b * config.latent_dim + c) * latent_tokens + src_t)];
                    }
                }
            }
        }
        chunk_latent_ms += engine::debug::elapsed_ms(latent_start, Clock::now());
        const int64_t source_latents = std::min<int64_t>(graph_latents, latent_tokens - start);
        const size_t bottleneck_source_values = static_cast<size_t>(batch * config.latent_dim * source_latents);
        const auto bottleneck_start = Clock::now();
        bottleneck_noise_source.resize(bottleneck_source_values);
        engine::sampling::fill_torch_cuda_tensor_iterator_randn(
            bottleneck_noise_source.data(),
            bottleneck_source_values,
            seed,
            rng_offset_blocks,
            rng_policy_,
            engine::sampling::TorchRandnPrecision::Float32);
        rng_offset_blocks += engine::sampling::torch_cuda_tensor_iterator_offset_blocks(
            static_cast<uint64_t>(bottleneck_source_values),
            rng_policy_);
        std::fill(bottleneck_noise.begin(), bottleneck_noise.end(), 0.0F);
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < config.latent_dim; ++c) {
                for (int64_t t = 0; t < source_latents; ++t) {
                    bottleneck_noise[static_cast<size_t>((b * config.latent_dim + c) * graph_latents + t)] =
                        bottleneck_noise_source[static_cast<size_t>((b * config.latent_dim + c) * source_latents + t)];
                }
            }
        }
        bottleneck_noise_ms += engine::debug::elapsed_ms(bottleneck_start, Clock::now());
        const size_t token_noise_values = static_cast<size_t>(
            batch * graph_latents * config.same_strides.back() *
            (config.same_channels * config.same_c_mults.back()));
        const auto token_noise_start = Clock::now();
        token_noise.resize(token_noise_values);
        engine::sampling::fill_torch_cuda_tensor_iterator_randn(
            token_noise.data(),
            token_noise_values,
            seed,
            rng_offset_blocks,
            rng_policy_,
            engine::sampling::TorchRandnPrecision::Float32);
        rng_offset_blocks += engine::sampling::torch_cuda_tensor_iterator_offset_blocks(
            static_cast<uint64_t>(token_noise_values),
            rng_policy_);
        token_noise_ms += engine::debug::elapsed_ms(token_noise_start, Clock::now());
        const auto decoded = graph_->decode_chunk(
            chunk_latent,
            bottleneck_noise,
            token_noise,
            input_upload_ms,
            graph_compute_ms,
            output_read_ms);
        if (static_cast<int64_t>(decoded.size()) != batch * config.audio_channels * frames_per_chunk) {
            throw std::runtime_error("Stable Audio SAME decoder output shape mismatch");
        }
        const bool first = chunk_index == 0;
        const bool last = chunk_index + 1 == starts.size();
        const int64_t out_start =
            (chunked && latent_tokens > kSameChunkLatents && last) ?
                total_frames - frames_per_chunk :
                start * config.downsampling_ratio;
        const int64_t left = first ? 0 : (kSameChunkOverlap / 2) * config.downsampling_ratio;
        const int64_t right = last ? frames_per_chunk : frames_per_chunk - (kSameChunkOverlap / 2) * config.downsampling_ratio;
        const auto pack_start = Clock::now();
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < config.audio_channels; ++c) {
                for (int64_t frame = left; frame < right; ++frame) {
                    const int64_t dst_frame = out_start + frame;
                    if (dst_frame < 0 || dst_frame >= total_frames) {
                        continue;
                    }
                    channel_major[static_cast<size_t>((b * config.audio_channels + c) * total_frames + dst_frame)] =
                        decoded[static_cast<size_t>((b * config.audio_channels + c) * frames_per_chunk + frame)];
                }
            }
        }
        output_pack_ms += engine::debug::elapsed_ms(pack_start, Clock::now());
    }
    engine::debug::timing_log_scalar("stable_audio.same_decode.chunk_count", static_cast<double>(starts.size()));
    engine::debug::timing_log_scalar("stable_audio.same_decode.host.chunk_latent_ms", chunk_latent_ms);
    engine::debug::timing_log_scalar("stable_audio.same_decode.host.bottleneck_noise_ms", bottleneck_noise_ms);
    engine::debug::timing_log_scalar("stable_audio.same_decode.host.token_noise_ms", token_noise_ms);
    engine::debug::timing_log_scalar("stable_audio.same_decode.host.input_upload_ms", input_upload_ms);
    engine::debug::timing_log_scalar("stable_audio.same_decode.graph.compute_total_ms", graph_compute_ms);
    engine::debug::timing_log_scalar("stable_audio.same_decode.host.output_read_ms", output_read_ms);
    engine::debug::timing_log_scalar("stable_audio.same_decode.host.output_pack_ms", output_pack_ms);
    std::vector<runtime::AudioBuffer> out;
    out.reserve(static_cast<size_t>(batch));
    for (int64_t b = 0; b < batch; ++b) {
        std::vector<float> interleaved(static_cast<size_t>(config.audio_channels * total_frames), 0.0F);
        for (int64_t frame = 0; frame < total_frames; ++frame) {
            for (int64_t c = 0; c < config.audio_channels; ++c) {
                interleaved[static_cast<size_t>(frame * config.audio_channels + c)] =
                    channel_major[static_cast<size_t>((b * config.audio_channels + c) * total_frames + frame)];
            }
        }
        out.push_back(runtime::AudioBuffer{config.sample_rate, static_cast<int>(config.audio_channels), std::move(interleaved)});
    }
    return out;
}

std::vector<float> StableAudioSameRuntime::encode(
    const runtime::AudioBuffer & audio,
    int64_t audio_sample_size,
    uint64_t seed,
    uint64_t & rng_offset_blocks) const {
    const auto & config = assets_->config;
    if (audio_sample_size <= 0 || audio_sample_size % config.downsampling_ratio != 0) {
        throw std::runtime_error("Stable Audio SAME encode audio_sample_size is invalid");
    }
    const auto & weights = require_same_weights(weights_, *execution_, assets_, weight_storage_type_);
    if (!encoder_graph_) {
        encoder_graph_ = std::make_unique<EncoderGraph>(*execution_, assets_, weights, weight_storage_type_);
    }
    const int64_t latent_tokens = audio_sample_size / config.downsampling_ratio;
    const int64_t patch_frames = audio_sample_size / config.pretransform_patch_size;
    auto prepared = adapt_interleaved_audio(
        audio,
        config.sample_rate,
        static_cast<int>(config.audio_channels),
        audio_sample_size);
    std::vector<int64_t> starts;
    if (latent_tokens <= kSameChunkLatents) {
        starts.push_back(0);
    } else {
        const int64_t hop = kSameChunkLatents - kSameChunkOverlap;
        for (int64_t start = 0; start <= latent_tokens - kSameChunkLatents; start += hop) {
            starts.push_back(start);
        }
        const int64_t final_start = latent_tokens - kSameChunkLatents;
        if (starts.empty() || starts.back() != final_start) {
            starts.push_back(final_start);
        }
    }
    std::vector<float> out(static_cast<size_t>(config.latent_dim * latent_tokens), 0.0F);
    const int64_t chunk_patch_frames = kSameChunkLatents * config.same_strides.front();
    std::vector<float> token_noise;
    for (size_t chunk_index = 0; chunk_index < starts.size(); ++chunk_index) {
        const int64_t latent_start = starts[chunk_index];
        const int64_t patch_start = latent_start * config.same_strides.front();
        std::vector<float> patched(static_cast<size_t>(config.same_encoder_in_channels * chunk_patch_frames), 0.0F);
        for (int64_t l = 0; l < chunk_patch_frames; ++l) {
            const int64_t src_patch = patch_start + l;
            if (src_patch >= patch_frames) {
                continue;
            }
            for (int64_t c = 0; c < config.audio_channels; ++c) {
                for (int64_t p = 0; p < config.pretransform_patch_size; ++p) {
                    const int64_t channel = c * config.pretransform_patch_size + p;
                    const int64_t frame = src_patch * config.pretransform_patch_size + p;
                    patched[static_cast<size_t>(channel * chunk_patch_frames + l)] =
                        prepared[static_cast<size_t>(frame * config.audio_channels + c)];
                }
            }
        }
        const size_t token_noise_values = static_cast<size_t>(kSameChunkLatents * (config.same_channels * config.same_c_mults.front()));
        if (config.same_encoder_mask_noise > 0.0F) {
            token_noise.resize(token_noise_values);
            engine::sampling::fill_torch_cuda_tensor_iterator_randn(
                token_noise.data(),
                token_noise_values,
                seed,
                rng_offset_blocks,
                rng_policy_,
                engine::sampling::TorchRandnPrecision::Float32);
            rng_offset_blocks += engine::sampling::torch_cuda_tensor_iterator_offset_blocks(
                static_cast<uint64_t>(token_noise_values),
                rng_policy_);
        } else {
            token_noise.clear();
        }
        const auto encoded = encoder_graph_->encode_chunk(patched, token_noise);
        if (static_cast<int64_t>(encoded.size()) != config.latent_dim * kSameChunkLatents) {
            throw std::runtime_error("Stable Audio SAME encoder output shape mismatch");
        }
        const bool first = chunk_index == 0;
        const bool last = chunk_index + 1 == starts.size();
        const int64_t out_start = last ? latent_tokens - kSameChunkLatents : latent_start;
        const int64_t left = first ? 0 : kSameChunkOverlap / 2;
        const int64_t right = last ? kSameChunkLatents : kSameChunkLatents - kSameChunkOverlap / 2;
        for (int64_t c = 0; c < config.latent_dim; ++c) {
            for (int64_t t = left; t < right; ++t) {
                out[static_cast<size_t>(c * latent_tokens + out_start + t)] =
                    encoded[static_cast<size_t>(c * kSameChunkLatents + t)];
            }
        }
    }
    return out;
}

void StableAudioSameRuntime::release_runtime_graphs() const {
    graph_.reset();
    encoder_graph_.reset();
}

}  // namespace engine::models::stable_audio
