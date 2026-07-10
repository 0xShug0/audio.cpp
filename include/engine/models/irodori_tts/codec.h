#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/models/irodori_tts/assets.h"

#include <ggml-backend.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::core {
class BackendWeightStore;
}

namespace engine::models::irodori_tts {

struct IrodoriCodecResidualUnitWeights {
    modules::Snake1dWeights snake0;
    modules::Conv1dWeights conv0;
    modules::Snake1dWeights snake1;
    modules::Conv1dWeights conv1;
};

struct IrodoriCodecDecoderBlockWeights {
    modules::Snake1dWeights up_snake;
    modules::ConvTranspose1dWeights up_conv;
    IrodoriCodecResidualUnitWeights residual_0;
    IrodoriCodecResidualUnitWeights residual_1;
    IrodoriCodecResidualUnitWeights residual_2;
};

struct IrodoriCodecEncoderBlockWeights {
    IrodoriCodecResidualUnitWeights residual_0;
    IrodoriCodecResidualUnitWeights residual_1;
    IrodoriCodecResidualUnitWeights residual_2;
    modules::Snake1dWeights down_snake;
    modules::Conv1dWeights down_conv;
};

struct IrodoriCodecWatermarkPassthroughWeights {
    modules::Snake1dWeights snake;
    modules::Conv1dWeights conv;
};

struct IrodoriCodecWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::Conv1dWeights encoder_input;
    std::vector<IrodoriCodecEncoderBlockWeights> encoder_blocks;
    modules::Snake1dWeights encoder_output_snake;
    modules::Conv1dWeights encoder_output;
    modules::Conv1dWeights quantizer_in_proj;
    modules::Conv1dWeights quantizer_out_proj;
    modules::Conv1dWeights decoder_input;
    std::vector<IrodoriCodecDecoderBlockWeights> decoder_blocks;
    IrodoriCodecWatermarkPassthroughWeights watermark_passthrough;
};

IrodoriCodecWeights load_irodori_codec_weights(
    const IrodoriAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType conv_storage_type);

core::TensorValue build_irodori_codec_decode(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & latent_btd,
    const IrodoriCodecWeights & weights,
    const IrodoriCodecConfig & config);

core::TensorValue build_irodori_codec_encode(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & waveform_bct,
    const IrodoriCodecWeights & weights,
    const IrodoriCodecConfig & config);

}  // namespace engine::models::irodori_tts
