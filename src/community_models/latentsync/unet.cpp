#include "engine/community_models/latentsync/unet.h"

#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/diffusion_math.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <ggml-alloc.h>
#include <stdexcept>
#include <tuple>

namespace engine::models::latentsync {

namespace {

LatentSyncAttentionWeights load_attention(core::BackendWeightStore & store,
    const assets::TensorSource & source, const std::string & prefix,
    int64_t channels, int64_t context_channels, assets::TensorStorageType type) {
    using modules::binding::linear_from_source;
    return {
        linear_from_source(store, source, prefix + ".to_q", type, channels, channels, false),
        linear_from_source(store, source, prefix + ".to_k", type, channels, context_channels, false),
        linear_from_source(store, source, prefix + ".to_v", type, channels, context_channels, false),
        linear_from_source(store, source, prefix + ".to_out.0", type, channels, channels, true),
    };
}

}  // namespace

LatentSyncSpatialTransformerWeights load_spatial_transformer_weights(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    const std::string & prefix, int64_t channels, int64_t audio_channels,
    assets::TensorStorageType type) {
    namespace binding = modules::binding;
    LatentSyncSpatialTransformerWeights out;
    out.input_norm = binding::norm_from_source(store, source, prefix + ".norm", channels);
    out.input = binding::conv2d_from_source(store, source, prefix + ".proj_in", type, channels, channels, 1, 1, true);
    out.output = binding::conv2d_from_source(store, source, prefix + ".proj_out", type, channels, channels, 1, 1, true);
    const auto block = prefix + ".transformer_blocks.0";
    for (size_t i = 0; i < out.norms.size(); ++i) {
        out.norms[i] = binding::norm_from_source(store, source, block + ".norm" + std::to_string(i + 1), channels);
    }
    out.self_attention = load_attention(store, source, block + ".attn1", channels, channels, type);
    out.audio_attention = load_attention(store, source, block + ".attn2", channels, audio_channels, type);
    out.geglu = binding::linear_from_source(store, source, block + ".ff.net.0.proj", type, channels * 8, channels, true);
    out.feed_forward_out = binding::linear_from_source(store, source, block + ".ff.net.2", type, channels, channels * 4, true);
    return out;
}

LatentSyncTemporalTransformerWeights load_temporal_transformer_weights(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    const std::string & prefix, int64_t channels, int64_t max_frames,
    assets::TensorStorageType type) {
    namespace binding = modules::binding;
    LatentSyncTemporalTransformerWeights out;
    out.input_norm = binding::norm_from_source(store, source, prefix + ".norm", channels);
    out.input = binding::linear_from_source(store, source, prefix + ".proj_in", type, channels, channels, true);
    out.output = binding::linear_from_source(store, source, prefix + ".proj_out", type, channels, channels, true);
    const auto block = prefix + ".transformer_blocks.0";
    for (size_t i = 0; i < out.norms.size(); ++i) {
        const auto attention = block + ".attention_blocks." + std::to_string(i);
        out.norms[i] = binding::norm_from_source(store, source, block + ".norms." + std::to_string(i), channels);
        out.attention[i] = load_attention(store, source, attention, channels, channels, type);
        out.position[i] = store.load_tensor(source, attention + ".pos_encoder.pe",
            assets::TensorStorageType::F32, {1, max_frames, channels});
    }
    out.feed_forward_norm = binding::norm_from_source(store, source, block + ".ff_norm", channels);
    out.geglu = binding::linear_from_source(store, source, block + ".ff.net.0.proj", type, channels * 8, channels, true);
    out.feed_forward_out = binding::linear_from_source(store, source, block + ".ff.net.2", type, channels, channels * 4, true);
    return out;
}

core::TensorValue LatentSyncUnetAttention::build_attention(core::ModuleBuildContext & ctx,
    const core::TensorValue & query, const core::TensorValue & conditioning,
    const LatentSyncAttentionWeights & weights) const {
    using namespace modules;
    const int64_t channels = query.shape.dims[2];
    if (heads_ <= 0 || channels % heads_ != 0) {
        throw std::runtime_error("LatentSync attention channels must be divisible by heads");
    }
    const int64_t dim = channels / heads_;
    const auto project = [&](const core::TensorValue & x, const LinearWeights & projection) {
        auto value = LinearModule({x.shape.dims[2], channels, false,
            GGML_PREC_DEFAULT, false, false, cuda_f16_}).build(ctx, x, projection);
        value = core::reshape_tensor(ctx, value,
            core::TensorShape::from_dims({x.shape.dims[0], x.shape.dims[1], heads_, dim}));
        return TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, value);
    };
    // Short temporal windows have thousands of independent spatial batches.
    // Metal flash reserves multi-GiB scratch for them; explicit scores stay small.
    const auto lowering = ctx.backend_type == core::BackendType::Metal && query.shape.dims[1] <= 24
        ? ScaledDotProductAttentionLowering::Explicit
        : cuda_f16_ && ctx.backend_type == core::BackendType::Cuda && dim == 40
        ? ScaledDotProductAttentionLowering::CudaFlashF16Head40
        : dim == 40 || dim == 80
            ? ScaledDotProductAttentionLowering::FlashPreserveViews
            : ScaledDotProductAttentionLowering::Explicit;
    auto q = project(query, weights.query);
    const auto k = project(conditioning, weights.key);
    const auto v = project(conditioning, weights.value);
    if (cuda_f16_ && lowering == ScaledDotProductAttentionLowering::FlashPreserveViews) {
        q = core::wrap_tensor(ggml_cast(ctx.ggml, q.tensor, GGML_TYPE_F32), q.shape, GGML_TYPE_F32);
    }
    const ScaledDotProductAttentionModule attention({dim, lowering, GGML_PREC_F32});
    core::TensorValue x;
    if (ctx.backend_type == core::BackendType::Cuda &&
        lowering == ScaledDotProductAttentionLowering::FlashPreserveViews && query.shape.dims[0] * heads_ > 65535) {
        const int64_t chunk = std::max<int64_t>(1, 32768 / heads_);
        for (int64_t offset = 0; offset < query.shape.dims[0]; offset += chunk) {
            const int64_t count = std::min(chunk, query.shape.dims[0] - offset);
            const SliceModule slice({0, offset, count});
            const auto part = attention.build(ctx, slice.build(ctx, q), slice.build(ctx, k), slice.build(ctx, v));
            x = offset == 0 ? part : ConcatModule({0}).build(ctx, x, part);
        }
    } else {
        x = attention.build(ctx, q, k, v);
    }
    if (cuda_f16_ && x.type != GGML_TYPE_F16) {
        x = core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F16), x.shape, GGML_TYPE_F16);
    }
    x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), query.shape);
    return LinearModule({channels, channels, true,
        GGML_PREC_DEFAULT, false, false, cuda_f16_}).build(ctx, x, weights.output);
}

core::TensorValue LatentSyncUnetAttention::build_feed_forward(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const modules::LinearWeights & geglu,
    const modules::LinearWeights & output) const {
    using namespace modules;
    const int64_t channels = input.shape.dims[2], inner = channels * 4;
    const auto packed = LinearModule({channels, inner * 2, true,
        GGML_PREC_DEFAULT, false, false, cuda_f16_}).build(ctx, input, geglu);
    if (cuda_f16_ || ctx.backend_type != core::BackendType::Cpu) {
        return LinearModule({inner, channels, true,
            GGML_PREC_DEFAULT, false, false, cuda_f16_}).build(ctx, GegluModule().build(ctx, packed), output);
    }
    const auto hidden = SliceModule({2, 0, inner}).build(ctx, packed);
    const auto gate = GeluModule({GeluApproximation::ExactErf}).build(ctx,
        SliceModule({2, inner, inner}).build(ctx, packed));
    return LinearModule({inner, channels, true,
        GGML_PREC_DEFAULT, false, false, cuda_f16_}).build(ctx, MulModule().build(ctx, hidden, gate), output);
}

core::TensorValue LatentSyncUnetAttention::build_spatial(core::ModuleBuildContext & ctx,
    const core::TensorValue & image, const core::TensorValue & audio,
    const LatentSyncSpatialTransformerWeights & weights) const {
    using namespace modules;
    const int64_t batch = image.shape.dims[0], channels = image.shape.dims[1];
    const int64_t steps = image.shape.dims[2] * image.shape.dims[3];
    auto x = modules::GroupNormModule({channels, 32, 1e-6F, true, true, cuda_f16_, ctx.backend_type == core::BackendType::Metal}).build(ctx, image, weights.input_norm);
    x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({batch, channels, steps}));
    x = TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
    // A 1x1 convolution is a per-token linear projection. Stay in the transformer's
    // token layout instead of materializing im2col and transposing its result.
    const auto projection_shape = core::TensorShape::from_dims({channels, channels});
    x = LinearModule({channels, channels, true, GGML_PREC_DEFAULT, false, false, cuda_f16_}).build(ctx, x,
        {core::reshape_tensor(ctx, weights.input.weight, projection_shape), weights.input.bias});
    const NormConfig norm_config{channels, 1e-5F, true, true, false, cuda_f16_};
    auto normalized = LayerNormModule(norm_config).build(ctx, x, weights.norms[0]);
    x = AddModule(cuda_f16_).build(ctx, x, build_attention(ctx, normalized, normalized, weights.self_attention));
    normalized = LayerNormModule(norm_config).build(ctx, x, weights.norms[1]);
    x = AddModule(cuda_f16_).build(ctx, x, build_attention(ctx, normalized, audio, weights.audio_attention));
    normalized = LayerNormModule(norm_config).build(ctx, x, weights.norms[2]);
    x = AddModule(cuda_f16_).build(ctx, x, build_feed_forward(ctx, normalized, weights.geglu, weights.feed_forward_out));
    x = LinearModule({channels, channels, true, GGML_PREC_DEFAULT, false, false, cuda_f16_}).build(ctx, x,
        {core::reshape_tensor(ctx, weights.output.weight, projection_shape), weights.output.bias});
    x = TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
    x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), image.shape);
    if (image.type == GGML_TYPE_F16 && x.type != GGML_TYPE_F16) {
        x = core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F16), x.shape, GGML_TYPE_F16);
    }
    return AddModule(image.type == GGML_TYPE_F16).build(ctx, image, x);
}

core::TensorValue LatentSyncUnetAttention::build_temporal(core::ModuleBuildContext & ctx,
    const core::TensorValue & image, int64_t frames,
    const LatentSyncTemporalTransformerWeights & weights) const {
    using namespace modules;
    const int64_t batch_frames = image.shape.dims[0], channels = image.shape.dims[1];
    const int64_t steps = image.shape.dims[2] * image.shape.dims[3];
    if (frames <= 0 || batch_frames % frames != 0 || frames > weights.position[0].shape.dims[1]) {
        throw std::runtime_error("Invalid LatentSync temporal attention frame count");
    }
    const int64_t batch = batch_frames / frames;
    auto x = modules::GroupNormModule({channels, 32, 1e-6F, true, true, cuda_f16_, ctx.backend_type == core::BackendType::Metal}).build(ctx, image, weights.input_norm);
    x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({batch_frames, channels, steps}));
    x = TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
    x = LinearModule({channels, channels, true, GGML_PREC_DEFAULT, false, false, cuda_f16_}).build(ctx, x, weights.input);
    const NormConfig norm_config{channels, 1e-5F, true, true, false, cuda_f16_};
    if (cuda_f16_) {
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({batch, frames, steps, channels}));
        x = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
        x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x),
            core::TensorShape::from_dims({batch * steps, frames, channels}));
    }
    for (size_t i = 0; i < weights.attention.size(); ++i) {
        auto normalized = LayerNormModule(norm_config).build(ctx, x, weights.norms[i]);
        if (!cuda_f16_) {
            normalized = core::reshape_tensor(ctx, normalized, core::TensorShape::from_dims({batch, frames, steps, channels}));
            normalized = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, normalized);
            normalized = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, normalized),
                core::TensorShape::from_dims({batch * steps, frames, channels}));
        }
        auto position = SliceModule({1, 0, frames}).build(ctx, weights.position[i]);
        if (cuda_f16_) {
            position = core::wrap_tensor(ggml_cast(ctx.ggml, position.tensor, GGML_TYPE_F16),
                position.shape, GGML_TYPE_F16);
        }
        normalized = cuda_f16_
            ? AddModule(true, true).build(ctx, normalized, position)
            : AddModule().build(ctx, normalized, RepeatModule({normalized.shape}).build(ctx, position));
        auto attended = build_attention(ctx, normalized, normalized, weights.attention[i]);
        if (!cuda_f16_) {
            attended = core::reshape_tensor(ctx, attended, core::TensorShape::from_dims({batch, steps, frames, channels}));
            attended = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, attended);
            attended = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, attended), x.shape);
        }
        x = AddModule(cuda_f16_).build(ctx, x, attended);
    }
    const auto normalized = LayerNormModule(norm_config).build(ctx, x, weights.feed_forward_norm);
    x = AddModule(cuda_f16_).build(ctx, x,
        build_feed_forward(ctx, normalized, weights.geglu, weights.feed_forward_out));
    x = LinearModule({channels, channels, true, GGML_PREC_DEFAULT, false, false, cuda_f16_}).build(ctx, x, weights.output);
    if (cuda_f16_) {
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({batch, steps, frames, channels}));
        x = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
        x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x),
            core::TensorShape::from_dims({batch_frames, steps, channels}));
    }
    x = TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
    x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), image.shape);
    if (image.type == GGML_TYPE_F16 && x.type != GGML_TYPE_F16) {
        x = core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F16), x.shape, GGML_TYPE_F16);
    }
    return AddModule(image.type == GGML_TYPE_F16).build(ctx, image, x);
}

LatentSyncUnetWeights load_unet_weights(core::BackendWeightStore & store,
    const assets::TensorSource & source, const LatentSyncUnetConfig & config,
    assets::TensorStorageType type) {
    namespace binding = modules::binding;
    LatentSyncUnetWeights out;
    const int64_t time_channels = config.channels.front() * 4;
    const auto resnet = [&](const std::string & prefix, int64_t in_channels, int64_t channels) {
        LatentSyncUnetResnetWeights weights;
        weights.norm1 = binding::norm_from_source(store, source, prefix + ".norm1", in_channels);
        weights.norm2 = binding::norm_from_source(store, source, prefix + ".norm2", channels);
        weights.conv1 = binding::conv2d_from_source(store, source, prefix + ".conv1", type, channels, in_channels, 3, 3, true);
        weights.conv2 = binding::conv2d_from_source(store, source, prefix + ".conv2", type, channels, channels, 3, 3, true);
        weights.timestep = binding::linear_from_source(store, source, prefix + ".time_emb_proj", type, channels, time_channels, true);
        if (in_channels != channels) {
            weights.shortcut = binding::conv2d_from_source(store, source, prefix + ".conv_shortcut", type, channels, in_channels, 1, 1, true);
        }
        return weights;
    };
    out.input = binding::conv2d_from_source(store, source, "unet.conv_in", type, config.channels.front(), config.input_channels, 3, 3, true);
    out.output = binding::conv2d_from_source(store, source, "unet.conv_out", type, config.output_channels, config.channels.front(), 3, 3, true);
    out.output_norm = binding::norm_from_source(store, source, "unet.conv_norm_out", config.channels.front());
    out.time_in = binding::linear_from_source(store, source, "unet.time_embedding.linear_1", type, time_channels, config.channels.front(), true);
    out.time_out = binding::linear_from_source(store, source, "unet.time_embedding.linear_2", type, time_channels, time_channels, true);
    int64_t current = config.channels.front();
    for (size_t i = 0; i < out.down.size(); ++i) {
        auto & stage = out.down[i];
        const int64_t channels = config.channels[i];
        const auto prefix = "unet.down_blocks." + std::to_string(i);
        for (int64_t j = 0; j < config.layers_per_block; ++j) {
            const auto index = std::to_string(j);
            stage.resnets.push_back(resnet(prefix + ".resnets." + index, current, channels));
            current = channels;
            if (i + 1 < out.down.size()) {
                stage.spatial.push_back(load_spatial_transformer_weights(store, source,
                    prefix + ".attentions." + index, channels, config.audio_channels, type));
            }
            stage.temporal.push_back(load_temporal_transformer_weights(store, source,
                prefix + ".motion_modules." + index + ".temporal_transformer", channels, config.max_frames, type));
        }
        if (i + 1 < out.down.size()) {
            stage.resample = binding::conv2d_from_source(store, source,
                prefix + ".downsamplers.0.conv", type, channels, channels, 3, 3, true);
        }
    }
    for (size_t i = 0; i < out.mid_resnets.size(); ++i) {
        out.mid_resnets[i] = resnet("unet.mid_block.resnets." + std::to_string(i), current, current);
    }
    out.mid_attention = load_spatial_transformer_weights(store, source,
        "unet.mid_block.attentions.0", current, config.audio_channels, type);
    for (size_t i = 0; i < out.up.size(); ++i) {
        auto & stage = out.up[i];
        const size_t level = out.up.size() - i - 1;
        const int64_t channels = config.channels[level];
        const int64_t last_skip = config.channels[level == 0 ? 0 : level - 1];
        const auto prefix = "unet.up_blocks." + std::to_string(i);
        for (int64_t j = 0; j <= config.layers_per_block; ++j) {
            const int64_t skip = j == config.layers_per_block ? last_skip : channels;
            const auto index = std::to_string(j);
            stage.resnets.push_back(resnet(prefix + ".resnets." + index, current + skip, channels));
            current = channels;
            if (i > 0) {
                stage.spatial.push_back(load_spatial_transformer_weights(store, source,
                    prefix + ".attentions." + index, channels, config.audio_channels, type));
            }
            stage.temporal.push_back(load_temporal_transformer_weights(store, source,
                prefix + ".motion_modules." + index + ".temporal_transformer", channels, config.max_frames, type));
        }
        if (i + 1 < out.up.size()) {
            stage.resample = binding::conv2d_from_source(store, source,
                prefix + ".upsamplers.0.conv", type, channels, channels, 3, 3, true);
        }
    }
    return out;
}

core::TensorValue LatentSyncUnet::build_video_norm(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, int64_t frames, const modules::NormWeights & weights) const {
    using namespace modules;
    const int64_t batch = input.shape.dims[0] / frames, channels = input.shape.dims[1];
    const int64_t spatial = input.shape.dims[2] * input.shape.dims[3];
    // Unlike attention normalization, ResnetBlock3D normalizes over frames and spatial dimensions together.
    auto x = core::reshape_tensor(ctx, input, core::TensorShape::from_dims({batch, frames, channels, spatial}));
    x = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    x = modules::GroupNormModule({channels, 32, 1e-5F, true, true, config_.cuda_f16_activations, ctx.backend_type == core::BackendType::Metal}).build(ctx, x, weights);
    x = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    return core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), input.shape);
}

core::TensorValue LatentSyncUnet::build_resnet(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const core::TensorValue & timestep, int64_t frames,
    const LatentSyncUnetResnetWeights & weights) const {
    using namespace modules;
    const int64_t in_channels = input.shape.dims[1], channels = weights.conv1.weight.shape.dims[0];
    auto x = SiluModule(config_.cuda_f16_activations).build(ctx, build_video_norm(ctx, input, frames, weights.norm1));
    x = Conv2dModule({in_channels, channels, 3, 3, 1, 1, 1, 1}, config_.cuda_f16_activations,
        config_.cuda_f16_activations).build(ctx, x, weights.conv1);
    auto time = LinearModule({config_.channels.front() * 4, channels, true})
        .build(ctx, SiluModule().build(ctx, timestep), weights.timestep);
    time = core::reshape_tensor(ctx, time, core::TensorShape::from_dims({timestep.shape.dims[0], 1, channels}));
    time = RepeatModule({core::TensorShape::from_dims({timestep.shape.dims[0], frames, channels})}).build(ctx, time);
    time = core::reshape_tensor(ctx, time, core::TensorShape::from_dims({input.shape.dims[0], channels, 1, 1}));
    if (config_.cuda_f16_activations) {
        time = core::wrap_tensor(ggml_cast(ctx.ggml, time.tensor, GGML_TYPE_F16), time.shape, GGML_TYPE_F16);
    }
    x = config_.cuda_f16_activations
        ? AddModule(true, true).build(ctx, x, time)
        : AddModule().build(ctx, x, RepeatModule({x.shape}).build(ctx, time));
    x = SiluModule(config_.cuda_f16_activations).build(ctx, build_video_norm(ctx, x, frames, weights.norm2));
    x = Conv2dModule({channels, channels, 3, 3, 1, 1, 1, 1}, config_.cuda_f16_activations,
        config_.cuda_f16_activations).build(ctx, x, weights.conv2);
    const auto residual = weights.shortcut
        ? Conv2dModule({in_channels, channels, 1, 1}, config_.cuda_f16_activations,
            config_.cuda_f16_activations).build(ctx, input, *weights.shortcut) : input;
    return AddModule(config_.cuda_f16_activations).build(ctx, residual, x);
}

core::TensorValue LatentSyncUnet::build(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const core::TensorValue & audio,
    const core::TensorValue & timestep_features, int64_t frames,
    const LatentSyncUnetWeights & weights, const core::TensorValue * cached_feature,
    core::TensorValue * cache_output) const {
    using namespace modules;
    core::validate_rank_between(input, 4, 4, "LatentSync U-Net input");
    if (frames <= 0 || frames > config_.max_frames || input.shape.dims[0] % frames != 0 ||
        input.shape.dims[1] != config_.input_channels || input.shape.dims[2] % 8 != 0 || input.shape.dims[3] % 8 != 0) {
        throw std::runtime_error("Invalid LatentSync U-Net video dimensions");
    }
    core::validate_shape(timestep_features,
        core::TensorShape::from_dims({input.shape.dims[0] / frames, config_.channels.front()}), "LatentSync timestep features");
    const int64_t time_channels = config_.channels.front() * 4;
    auto time = LinearModule({config_.channels.front(), time_channels, true}).build(ctx, timestep_features, weights.time_in);
    time = SiluModule().build(ctx, time);
    time = LinearModule({time_channels, time_channels, true}).build(ctx, time, weights.time_out);
    auto x = Conv2dModule({config_.input_channels, config_.channels.front(), 3, 3, 1, 1, 1, 1},
        config_.cuda_f16_activations, config_.cuda_f16_activations).build(ctx, input, weights.input);
    const LatentSyncUnetAttention attention(config_.heads, config_.cuda_f16_activations);
    const auto transformer_audio = config_.cuda_f16_activations
        ? core::wrap_tensor(ggml_cast(ctx.ggml, audio.tensor, GGML_TYPE_F16), audio.shape, GGML_TYPE_F16)
        : audio;
    std::vector<core::TensorValue> skips{x};
    if (!cached_feature) {
        for (const auto & stage : weights.down) {
            for (size_t i = 0; i < stage.resnets.size(); ++i) {
                x = build_resnet(ctx, x, time, frames, stage.resnets[i]);
                if (!stage.spatial.empty()) x = attention.build_spatial(ctx, x, transformer_audio, stage.spatial[i]);
                x = attention.build_temporal(ctx, x, frames, stage.temporal[i]);
                skips.push_back(x);
            }
            if (stage.resample) {
                x = Conv2dModule({x.shape.dims[1], x.shape.dims[1], 3, 3, 2, 2, 1, 1},
                    config_.cuda_f16_activations, config_.cuda_f16_activations).build(ctx, x, *stage.resample);
                skips.push_back(x);
            }
        }
        x = build_resnet(ctx, x, time, frames, weights.mid_resnets[0]);
        x = attention.build_spatial(ctx, x, transformer_audio, weights.mid_attention);
        x = build_resnet(ctx, x, time, frames, weights.mid_resnets[1]);
    } else {
        // DeepCache branch 0: the penultimate motion block depends only on the
        // cached spatial feature, so its deterministic output can be retained too.
        x = *cached_feature;
    }
    for (const auto & stage : weights.up) {
        if (cached_feature && &stage != &weights.up.back()) continue;
        for (size_t i = cached_feature ? stage.resnets.size() - 1 : 0; i < stage.resnets.size(); ++i) {
            x = ConcatModule({1}).build(ctx, x, skips.back());
            skips.pop_back();
            x = build_resnet(ctx, x, time, frames, stage.resnets[i]);
            if (!stage.spatial.empty()) x = attention.build_spatial(ctx, x, transformer_audio, stage.spatial[i]);
            x = attention.build_temporal(ctx, x, frames, stage.temporal[i]);
            if (cache_output && &stage == &weights.up.back() && i + 2 == stage.resnets.size()) {
                *cache_output = x;
            }
        }
        if (stage.resample) {
            const auto upsample_input = config_.cuda_f16_activations
                ? core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F32), x.shape, GGML_TYPE_F32) : x;
            x = NearestUpsample2dModule({x.shape.dims[2] * 2, x.shape.dims[3] * 2}).build(ctx, upsample_input);
            x = Conv2dModule({x.shape.dims[1], x.shape.dims[1], 3, 3, 1, 1, 1, 1},
                config_.cuda_f16_activations, config_.cuda_f16_activations).build(ctx, x, *stage.resample);
        }
    }
    x = SiluModule(config_.cuda_f16_activations).build(ctx, build_video_norm(ctx, x, frames, weights.output_norm));
    x = Conv2dModule({config_.channels.front(), config_.output_channels, 3, 3, 1, 1, 1, 1},
        config_.cuda_f16_activations, config_.cuda_f16_activations).build(ctx, x, weights.output);
    return config_.cuda_f16_activations
        ? core::wrap_tensor(ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_F32), x.shape, GGML_TYPE_F32) : x;
}

namespace {

class LatentSyncUnetGraph {
public:
    LatentSyncUnetGraph(core::ExecutionContext & execution, const LatentSyncUnetConfig & config,
        const LatentSyncUnetWeights & weights, int64_t batch, int64_t frames,
        int64_t height, int64_t width, int64_t audio_tokens, bool cache,
        LatentSyncUnetGraph * full = nullptr) : backend_(execution.backend()) {
        const bool replay = full != nullptr;
        constexpr size_t nodes = 16384;
        context_.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!context_) throw std::runtime_error("LatentSync U-Net graph context allocation failed");
        core::ModuleBuildContext ctx{context_.get(), "latentsync.unet", execution.backend_type()};
        if (full) {
            input_ = full->input_;
            audio_ = full->audio_;
            timestep_ = full->timestep_;
        } else {
            input_ = core::make_tensor(ctx, GGML_TYPE_F32,
                core::TensorShape::from_dims({batch * frames, config.input_channels, height, width}));
            audio_ = core::make_tensor(ctx, GGML_TYPE_F32,
                core::TensorShape::from_dims({batch * frames, audio_tokens, config.audio_channels}));
            timestep_ = core::make_tensor(ctx, GGML_TYPE_F32,
                core::TensorShape::from_dims({batch, config.channels.front()}));
        }
        ggml_set_input(input_.tensor);
        ggml_set_input(audio_.tensor);
        ggml_set_input(timestep_.tensor);
        if (replay) {
            cache_context_.reset(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
            if (!cache_context_) throw std::runtime_error("LatentSync U-Net cache context allocation failed");
            core::ModuleBuildContext cache_ctx{cache_context_.get(), "latentsync.unet.cache", execution.backend_type()};
            cached_feature_ = core::make_tensor(cache_ctx,
                config.cuda_f16_activations ? GGML_TYPE_F16 : GGML_TYPE_F32,
                core::TensorShape::from_dims({batch * frames, config.channels.front(), height, width}));
            ggml_set_input(cached_feature_.tensor);
            cache_buffer_.reset(ggml_backend_alloc_ctx_tensors(cache_context_.get(), backend_));
            if (!cache_buffer_) throw std::runtime_error("LatentSync U-Net cache buffer allocation failed");
        }
        output_ = LatentSyncUnet(config).build(ctx, input_, audio_, timestep_, frames, weights,
            replay ? &cached_feature_ : nullptr, cache && !replay ? &cached_feature_ : nullptr);
        if (cache && !replay) ggml_set_output(cached_feature_.tensor);
        ggml_set_output(output_.tensor);
        graph_ = ggml_new_graph_custom(context_.get(), nodes, false);
        ggml_build_forward_expand(graph_, output_.tensor);
    }

    ~LatentSyncUnetGraph() { core::release_backend_graph_resources(backend_, graph_, true); }

    void allocate(ggml_gallocr_t allocator, LatentSyncUnetGraph * replay) {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> plan_context{nullptr, ggml_free};
        auto * plan = graph_;
        if (replay) {
            // Bind both cached graphs once; reallocating their shared arena invalidates tensor buffers.
            const size_t nodes = ggml_graph_size(graph_) + ggml_graph_size(replay->graph_);
            plan_context.reset(ggml_init({ggml_graph_overhead_custom(nodes, false), nullptr, true}));
            if (!plan_context) throw std::runtime_error("LatentSync U-Net allocation plan context failed");
            plan = ggml_new_graph_custom(plan_context.get(), nodes, false);
            ggml_build_forward_expand(plan, output_.tensor);
            ggml_build_forward_expand(plan, replay->output_.tensor);
        }
        if (!allocator || !ggml_gallocr_alloc_graph(allocator, plan)) {
            throw std::runtime_error("LatentSync U-Net graph allocation failed");
        }
        debug::timing_log_scalar("latentsync.unet.graph_buffer_mb",
            ggml_gallocr_get_buffer_size(allocator, 0) / 1048576.0);
    }

    void copy_cache_to(LatentSyncUnetGraph & replay) const {
        ggml_backend_tensor_copy(cached_feature_.tensor, replay.cached_feature_.tensor);
    }

    std::vector<std::byte> read_cache() const {
        return core::read_tensor_bytes(cached_feature_.tensor);
    }

    void write_cache(const std::vector<std::byte> & bytes) {
        core::write_tensor_bytes(cached_feature_, bytes);
    }

    std::vector<float> predict(const std::vector<float> & input, const std::vector<float> & audio,
        const std::vector<float> & timestep_features) {
        core::write_tensor_f32(input_, input);
        core::write_tensor_f32(audio_, audio);
        core::write_tensor_f32(timestep_, timestep_features);
        if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LatentSync U-Net graph execution failed");
        }
        return core::read_tensor_f32(output_.tensor);
    }

private:
    ggml_backend_t backend_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> cache_context_{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> cache_buffer_{nullptr, ggml_backend_buffer_free};
    ggml_cgraph * graph_ = nullptr;
    core::TensorValue input_, audio_, timestep_, output_, cached_feature_;
};

}  // namespace

struct LatentSyncUnetRuntime::Impl {
    core::ExecutionContext & execution;
    LatentSyncUnetConfig config;
    core::BackendWeightStore store;
    LatentSyncUnetWeights weights;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator;
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> shape;
    std::unique_ptr<LatentSyncUnetGraph> graph;
    std::unique_ptr<LatentSyncUnetGraph> replay_graph;
    std::array<std::vector<std::byte>, 2> cfg_cache;

    Impl(core::ExecutionContext & execution_in, const assets::TensorSource & source,
        LatentSyncUnetConfig config_in, assets::TensorStorageType type)
        : execution(execution_in), config(config_in),
          store(execution.backend(), execution.backend_type(), "latentsync.unet.weights", 4 * 1024 * 1024),
          weights(load_unet_weights(store, source, config, type)),
          allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())),
              &ggml_gallocr_free) {
        if (config.cuda_f16_activations && execution.backend_type() != core::BackendType::Cuda) {
            throw std::runtime_error("LatentSync U-Net FP16 cuDNN execution requires CUDA");
        }
        if (config.cuda_f16_activations && type != assets::TensorStorageType::F16) {
            throw std::runtime_error("LatentSync U-Net FP16 execution requires FP16 weights");
        }
#ifndef GGML_CUDA_CUDNN_CONV2D
        if (config.cuda_f16_activations) {
            throw std::runtime_error("LatentSync U-Net FP16 cuDNN execution is not enabled in this build");
        }
#endif
        store.upload();
    }
};

LatentSyncUnetRuntime::LatentSyncUnetRuntime(core::ExecutionContext & execution,
    const assets::TensorSource & source, LatentSyncUnetConfig config, assets::TensorStorageType weight_type)
    : impl_(std::make_unique<Impl>(execution, source, config, weight_type)) {}

LatentSyncUnetRuntime::~LatentSyncUnetRuntime() = default;

std::vector<float> LatentSyncUnetRuntime::predict(const std::vector<float> & input,
    const std::vector<float> & audio, const std::vector<float> & timestep_features,
    int64_t batch, int64_t frames, int64_t height, int64_t width, int64_t audio_tokens,
    int64_t cache_step, int64_t cache_slot) {
    const auto start = std::chrono::steady_clock::now();
    if (batch <= 0 || frames <= 0 || height <= 0 || width <= 0 || audio_tokens <= 0) {
        throw std::runtime_error("LatentSync U-Net dimensions must be positive");
    }
    const auto shape = std::make_tuple(batch, frames, height, width, audio_tokens);
    const bool cache = cache_step >= 0;
    if (!impl_->graph || impl_->shape != shape || cache != bool(impl_->replay_graph)) {
        impl_->replay_graph.reset();
        impl_->graph.reset();
        impl_->graph = std::make_unique<LatentSyncUnetGraph>(impl_->execution, impl_->config,
            impl_->weights, batch, frames, height, width, audio_tokens, cache);
        if (cache) {
            impl_->replay_graph = std::make_unique<LatentSyncUnetGraph>(impl_->execution, impl_->config,
                impl_->weights, batch, frames, height, width, audio_tokens, true, impl_->graph.get());
        }
        impl_->graph->allocate(impl_->allocator.get(), impl_->replay_graph.get());
        impl_->shape = shape;
    }
    if (cache_step > 0 && impl_->config.sequential_cfg) {
        impl_->replay_graph->write_cache(impl_->cfg_cache[static_cast<size_t>(cache_slot)]);
    }
    auto & graph = cache_step > 0 ? impl_->replay_graph : impl_->graph;
    auto output = graph->predict(input, audio, timestep_features);
    if (cache_step == 0) {
        if (impl_->config.sequential_cfg) {
            impl_->cfg_cache[static_cast<size_t>(cache_slot)] = impl_->graph->read_cache();
        } else {
            impl_->graph->copy_cache_to(*impl_->replay_graph);
        }
    }
    debug::timing_log_scalar("latentsync.unet.wall_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    return output;
}

std::vector<float> LatentSyncUnetRuntime::denoise(const LatentSyncLatentClip & clip,
    int64_t steps, float guidance_scale, LatentSyncDdimConfig schedule_config, int64_t cache_interval) {
    const auto start = std::chrono::steady_clock::now();
    if (clip.frames <= 0 || clip.height <= 0 || clip.width <= 0 || clip.audio_tokens <= 0 ||
        !std::isfinite(guidance_scale) || guidance_scale < 0 || cache_interval < 1) {
        throw std::runtime_error("Invalid LatentSync denoising dimensions or guidance scale");
    }
    const size_t pixels = static_cast<size_t>(clip.height * clip.width);
    const size_t latent_frame = pixels * 4;
    const size_t latent_size = latent_frame * static_cast<size_t>(clip.frames);
    if (clip.noise.size() != latent_size || clip.masked_image.size() != latent_size ||
        clip.reference_image.size() != latent_size || clip.mask.size() != pixels * static_cast<size_t>(clip.frames) ||
        clip.audio.size() != static_cast<size_t>(clip.frames * clip.audio_tokens * impl_->config.audio_channels)) {
        throw std::runtime_error("LatentSync latent clip tensor size mismatch");
    }
    const bool cfg = guidance_scale > 1.0F;
    const bool sequential_cfg = cfg && impl_->config.sequential_cfg;
    const int64_t lanes = cfg && !sequential_cfg ? 2 : 1;
    const size_t input_frame = pixels * 13;
    std::vector<float> input(input_frame * static_cast<size_t>(clip.frames * lanes));
    std::vector<float> audio(clip.audio.size() * static_cast<size_t>(lanes), 0.0F);
    const std::vector<float> unconditioned_audio(sequential_cfg ? clip.audio.size() : 0, 0.0F);
    std::copy(clip.audio.begin(), clip.audio.end(), audio.end() - clip.audio.size());
    for (int64_t lane = 0; lane < lanes; ++lane) {
        for (int64_t frame = 0; frame < clip.frames; ++frame) {
            const size_t base = static_cast<size_t>(lane * clip.frames + frame) * input_frame;
            std::copy_n(clip.mask.begin() + frame * pixels, pixels, input.begin() + base + latent_frame);
            std::copy_n(clip.masked_image.begin() + frame * latent_frame, latent_frame, input.begin() + base + pixels * 5);
            std::copy_n(clip.reference_image.begin() + frame * latent_frame, latent_frame, input.begin() + base + pixels * 9);
        }
    }
    const LatentSyncDdimSchedule schedule(schedule_config, steps);
    auto latents = clip.noise;
    for (size_t i = 0; i < schedule.timesteps().size(); ++i) {
        for (int64_t lane = 0; lane < lanes; ++lane) {
            for (int64_t frame = 0; frame < clip.frames; ++frame) {
                const size_t base = static_cast<size_t>(lane * clip.frames + frame) * input_frame;
                std::copy_n(latents.begin() + frame * latent_frame, latent_frame, input.begin() + base);
            }
        }
        const auto time = latentsync_timestep_features(schedule.timesteps()[i], impl_->config.channels.front(), lanes);
        const int64_t cache_step = cache_interval == 1 ? -1 : static_cast<int64_t>(i) % cache_interval;
        std::vector<float> noise;
        if (sequential_cfg) {
            const auto unconditioned = predict(input, unconditioned_audio, time, 1,
                clip.frames, clip.height, clip.width, clip.audio_tokens, cache_step, 0);
            const auto conditioned = predict(input, audio, time, 1,
                clip.frames, clip.height, clip.width, clip.audio_tokens, cache_step, 1);
            noise = sampling::cfg_guidance(conditioned, unconditioned, guidance_scale);
        } else {
            noise = predict(input, audio, time, lanes, clip.frames, clip.height, clip.width,
                clip.audio_tokens, cache_step);
        }
        if (cfg && !sequential_cfg) {
            const std::vector<float> unconditioned(noise.begin(), noise.begin() + latent_size);
            const std::vector<float> conditioned(noise.begin() + latent_size, noise.end());
            noise = sampling::cfg_guidance(conditioned, unconditioned, guidance_scale);
        }
        schedule.step(latents, noise, i);
    }
    debug::timing_log_scalar("latentsync.denoise.wall_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    return latents;
}

}  // namespace engine::models::latentsync
