#pragma once

#include "engine/community_models/latentsync/sampling.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <array>
#include <memory>
#include <vector>

namespace engine::models::latentsync {

struct LatentSyncAttentionWeights {
    modules::LinearWeights query, key, value, output;
};

struct LatentSyncSpatialTransformerWeights {
    modules::NormWeights input_norm;
    modules::Conv2dWeights input, output;
    std::array<modules::NormWeights, 3> norms;
    LatentSyncAttentionWeights self_attention, audio_attention;
    modules::LinearWeights geglu, feed_forward_out;
};

struct LatentSyncTemporalTransformerWeights {
    modules::NormWeights input_norm, feed_forward_norm;
    modules::LinearWeights input, output, geglu, feed_forward_out;
    std::array<modules::NormWeights, 2> norms;
    std::array<LatentSyncAttentionWeights, 2> attention;
    std::array<core::TensorValue, 2> position;
};

LatentSyncSpatialTransformerWeights load_spatial_transformer_weights(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    const std::string & prefix, int64_t channels, int64_t audio_channels,
    assets::TensorStorageType type);

LatentSyncTemporalTransformerWeights load_temporal_transformer_weights(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    const std::string & prefix, int64_t channels, int64_t max_frames,
    assets::TensorStorageType type);

class LatentSyncUnetAttention {
public:
    explicit LatentSyncUnetAttention(int64_t heads = 8, bool cuda_f16 = false)
        : heads_(heads), cuda_f16_(cuda_f16) {}

    // Images use [batch * frames, channels, height, width]; audio uses [batch * frames, tokens, channels].
    core::TensorValue build_spatial(core::ModuleBuildContext & ctx,
        const core::TensorValue & image, const core::TensorValue & audio,
        const LatentSyncSpatialTransformerWeights & weights) const;
    core::TensorValue build_temporal(core::ModuleBuildContext & ctx,
        const core::TensorValue & image, int64_t frames,
        const LatentSyncTemporalTransformerWeights & weights) const;

private:
    core::TensorValue build_attention(core::ModuleBuildContext & ctx,
        const core::TensorValue & query, const core::TensorValue & conditioning,
        const LatentSyncAttentionWeights & weights) const;
    core::TensorValue build_feed_forward(core::ModuleBuildContext & ctx,
        const core::TensorValue & input, const modules::LinearWeights & geglu,
        const modules::LinearWeights & output) const;
    int64_t heads_;
    bool cuda_f16_;
};

struct LatentSyncUnetConfig {
    LatentSyncUnetConfig() = default;
    explicit LatentSyncUnetConfig(bool cuda_f16, bool sequential_cfg = false)
        : cuda_f16_activations(cuda_f16), sequential_cfg(sequential_cfg) {}
    std::array<int64_t, 4> channels{320, 640, 1280, 1280};
    int64_t input_channels = 13;
    int64_t output_channels = 4;
    int64_t audio_channels = 384;
    int64_t layers_per_block = 2;
    int64_t heads = 8;
    int64_t max_frames = 24;
    bool cuda_f16_activations = false;
    bool sequential_cfg = false;
};

struct LatentSyncUnetResnetWeights {
    modules::NormWeights norm1, norm2;
    modules::Conv2dWeights conv1, conv2;
    modules::LinearWeights timestep;
    std::optional<modules::Conv2dWeights> shortcut;
};

struct LatentSyncUnetStageWeights {
    std::vector<LatentSyncUnetResnetWeights> resnets;
    std::vector<LatentSyncSpatialTransformerWeights> spatial;
    std::vector<LatentSyncTemporalTransformerWeights> temporal;
    std::optional<modules::Conv2dWeights> resample;
};

struct LatentSyncUnetWeights {
    modules::Conv2dWeights input, output;
    modules::NormWeights output_norm;
    modules::LinearWeights time_in, time_out;
    std::array<LatentSyncUnetStageWeights, 4> down, up;
    std::array<LatentSyncUnetResnetWeights, 2> mid_resnets;
    LatentSyncSpatialTransformerWeights mid_attention;
};

LatentSyncUnetWeights load_unet_weights(core::BackendWeightStore & store,
    const assets::TensorSource & source, const LatentSyncUnetConfig & config,
    assets::TensorStorageType type);

class LatentSyncUnet {
public:
    explicit LatentSyncUnet(LatentSyncUnetConfig config) : config_(config) {}
    // Folded video [batch * frames, channels, height, width], audio [batch * frames, tokens, channels].
    // timestep_features is the cosine/sine projection [batch, channels[0]].
    core::TensorValue build(core::ModuleBuildContext & ctx,
        const core::TensorValue & input, const core::TensorValue & audio,
        const core::TensorValue & timestep_features, int64_t frames,
        const LatentSyncUnetWeights & weights,
        const core::TensorValue * cached_feature = nullptr,
        core::TensorValue * cache_output = nullptr) const;

private:
    core::TensorValue build_video_norm(core::ModuleBuildContext & ctx,
        const core::TensorValue & input, int64_t frames, const modules::NormWeights & weights) const;
    core::TensorValue build_resnet(core::ModuleBuildContext & ctx,
        const core::TensorValue & input, const core::TensorValue & timestep, int64_t frames,
        const LatentSyncUnetResnetWeights & weights) const;
    LatentSyncUnetConfig config_;
};

struct LatentSyncLatentClip {
    int64_t frames = 0;
    int64_t height = 0;
    int64_t width = 0;
    int64_t audio_tokens = 50;
    // All image tensors are frame-major NCHW. Noise and image latents have four channels; mask has one.
    std::vector<float> noise, mask, masked_image, reference_image, audio;
};

class LatentSyncUnetRuntime {
public:
    LatentSyncUnetRuntime(core::ExecutionContext & execution, const assets::TensorSource & source,
        LatentSyncUnetConfig config, assets::TensorStorageType weight_type);
    ~LatentSyncUnetRuntime();

    std::vector<float> predict(const std::vector<float> & input, const std::vector<float> & audio,
        const std::vector<float> & timestep_features, int64_t batch, int64_t frames,
        int64_t height, int64_t width, int64_t audio_tokens, int64_t cache_step = -1,
        int64_t cache_slot = 0);

    std::vector<float> denoise(const LatentSyncLatentClip & clip, int64_t steps,
        float guidance_scale, LatentSyncDdimConfig schedule_config = {}, int64_t cache_interval = 1);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::latentsync
