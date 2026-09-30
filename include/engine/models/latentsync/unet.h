#pragma once

#include "engine/models/latentsync/unet_attention.h"

#include <vector>

namespace engine::models::latentsync {

struct LatentSyncUnetConfig {
    std::array<int64_t, 4> channels{320, 640, 1280, 1280};
    int64_t input_channels = 13;
    int64_t output_channels = 4;
    int64_t audio_channels = 384;
    int64_t layers_per_block = 2;
    int64_t heads = 8;
    int64_t max_frames = 24;
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

}  // namespace engine::models::latentsync
