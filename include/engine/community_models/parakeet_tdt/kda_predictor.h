#pragma once

#include "engine/community_models/parakeet_tdt/predictor.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"

#include <array>
#include <vector>

namespace engine::community_models::parakeet_tdt {

struct KdaLayerWeights {
    engine::modules::NormWeights norm;
    std::array<engine::modules::LinearWeights, 3> qkv;
    std::array<engine::modules::DepthwiseConv1dWeights, 3> conv;
    engine::modules::LinearWeights forget_in, forget_out, beta, gate_in, gate_out, output;
    engine::modules::NormWeights output_norm;
    engine::core::TensorValue a_log, dt_bias;
};

struct KdaPredictorWeights {
    std::vector<KdaLayerWeights> layers;
    engine::modules::NormWeights final_norm;
};

struct KdaPredictorConfig {
    int64_t decoder_hidden_size;
    int64_t kda_heads;
    int64_t kda_head_dim;
    int64_t kda_conv_size;
    bool kda_final_norm;
};

std::shared_ptr<const Predictor> load_kda_predictor(
    core::BackendWeightStore &, const ParakeetTDTAssets &, assets::TensorStorageType);

PredictorOutput build_kda_predictor(
    engine::core::ModuleBuildContext & ctx,
    const KdaPredictorConfig & config,
    const KdaPredictorWeights & weights,
    const engine::core::TensorValue & input,
    const std::vector<engine::core::TensorValue> & recurrent,
    const std::vector<engine::core::TensorValue> & convolution);

}  // namespace engine::community_models::parakeet_tdt
