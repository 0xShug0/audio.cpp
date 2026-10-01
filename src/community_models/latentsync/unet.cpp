#include "engine/community_models/latentsync/unet.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <stdexcept>

namespace engine::models::latentsync {

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
    x = GroupNormModule({channels, 32, 1e-5F, true, true, config_.cuda_f16_activations}).build(ctx, x, weights);
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

}  // namespace engine::models::latentsync
