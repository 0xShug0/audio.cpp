#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <array>

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

}  // namespace engine::models::latentsync
