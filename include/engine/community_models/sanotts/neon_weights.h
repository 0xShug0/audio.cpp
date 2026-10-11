#pragma once

#include "engine/community_models/sanotts/config.h"
#include "engine/framework/neon/blocks.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::core {
struct TensorValue;
}

namespace engine::models::sanotts {

struct SanoTtsAffineWeights {
    std::vector<float> weight;
    std::vector<float> bias;
};

struct SanoTtsNeonWeights {
    SanoTtsAffineWeights projection;
    std::vector<neon::ResidualConvWeights> acoustic_blocks;
    SanoTtsAffineWeights embedding;
    SanoTtsAffineWeights input_norm;
    std::vector<neon::ConvNeXtWeights> decoder_blocks;
    SanoTtsAffineWeights final_norm;
    SanoTtsAffineWeights head;
};

// Loading, validation and the model-specific acoustic/embedding fold happen here.
// The decoder receives typed, bound data and never looks up weight names.
using SanoTtsWeightReader = std::function<std::vector<float>(const std::string &)>;
SanoTtsNeonWeights load_sanotts_neon_weights(const SanoTtsConfig &config, const SanoTtsWeightReader &reader,
                                             int threads);
SanoTtsNeonWeights load_sanotts_neon_weights(const SanoTtsConfig &config,
                                             const std::unordered_map<std::string, core::TensorValue> &tensors,
                                             int threads);

} // namespace engine::models::sanotts
