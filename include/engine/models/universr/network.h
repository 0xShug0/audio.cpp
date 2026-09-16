#pragma once

#include "engine/models/universr/assets.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

namespace engine::models::universr {

struct ConvNeXtWeights {
    modules::Conv2dWeights depthwise;
    modules::NormWeights norm;
    modules::LinearWeights expansion;
    core::TensorValue grn_gamma;
    modules::LinearWeights projection;
};

ConvNeXtWeights load_convnext_weights(core::BackendWeightStore & store,
    const assets::TensorSource & source, const std::string & prefix,
    int64_t channels, assets::TensorStorageType storage);

// Matches upstream Block: NCHW input/output, reflection padding, and spatial GRN.
core::TensorValue build_convnext_block(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const ConvNeXtWeights & weights);
struct ConditioningWeights {
    core::TensorValue frequency_pe;
    core::TensorValue sample_rate_embedding;
    core::TensorValue unconditional_embedding;
    core::TensorValue time_frequencies;
    modules::LinearWeights sample_rate_projector;
    modules::LinearWeights low_film;
    modules::LinearWeights high_film;
    modules::LinearWeights head;
    modules::LinearWeights sample_rate_hidden;
    modules::LinearWeights sample_rate_film;
    std::vector<ConvNeXtWeights> blocks;
};

ConditioningWeights load_conditioning_weights(core::BackendWeightStore & store,
    const assets::TensorSource & source, const UniverSRConfig & config,
    assets::TensorStorageType storage);

core::TensorValue build_conditioning_encoder(core::ModuleBuildContext & ctx,
    const core::TensorValue & low_spectrum, const core::TensorValue & sample_rate_embedding,
    const ConditioningWeights & weights);

core::TensorValue build_spatial_conditioning(core::ModuleBuildContext & ctx,
    const core::TensorValue & condition, const UniverSRConfig & config,
    const ConditioningWeights & weights);

core::TensorValue build_time_embedding(core::ModuleBuildContext & ctx,
    const core::TensorValue & time, const core::TensorValue & sample_rate_embedding,
    const ConditioningWeights & weights);

core::TensorValue build_projected_conditioning(core::ModuleBuildContext & ctx,
    const core::TensorValue & condition, const UniverSRConfig & config,
    const ConditioningWeights & weights, const core::TensorValue & projection);
struct TimeBlockWeights {
    modules::LinearWeights time_hidden;
    modules::LinearWeights time_output;
    ConvNeXtWeights block;
};

struct EncoderWeights {
    std::vector<TimeBlockWeights> blocks;
    modules::NormWeights norm;
    modules::Conv2dWeights downsample;
};

struct DecoderWeights {
    // [4 * output_channels, input_channels], with each output channel's 2x2 patch in row-major order.
    core::TensorValue upsample_weight;
    core::TensorValue upsample_bias;
    std::vector<TimeBlockWeights> blocks;
};

struct UNetBackboneWeights {
    modules::LinearWeights input;
    modules::NormWeights input_norm;
    std::vector<EncoderWeights> encoders;
    std::vector<TimeBlockWeights> middle;
    std::vector<DecoderWeights> decoders;
    modules::LinearWeights output;
};

UNetBackboneWeights load_unet_backbone_weights(core::BackendWeightStore & store,
    const assets::TensorSource & source, const UniverSRConfig & config,
    assets::TensorStorageType storage);

core::TensorValue build_time_block(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const core::TensorValue & time,
    const TimeBlockWeights & weights);

// Input is the NCHW concatenation of noisy spectrum and spatial conditioning.
core::TensorValue build_unet_backbone(core::ModuleBuildContext & ctx,
    const core::TensorValue & input, const core::TensorValue & time,
    const UNetBackboneWeights & weights, bool input_projected = false);

}  // namespace engine::models::universr
