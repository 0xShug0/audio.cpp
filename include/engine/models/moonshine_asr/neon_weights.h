#pragma once

#include "engine/framework/neon/operators.h"
#include <variant>
#include <vector>

namespace engine::assets {
enum class TensorStorageType;
}

namespace engine::models::moonshine_asr {
struct MoonshineAssets;

struct MoonshineNeonLinear {
    int input_features, output_features;
    std::variant<std::vector<float>, neon::BlockInt8Weights> weight;
    std::vector<float> bias;
};
struct MoonshineNeonDecoderLayer {
    std::vector<float> input_norm, cross_norm, mlp_norm;
    MoonshineNeonLinear self_query, self_key, self_value, self_output;
    MoonshineNeonLinear cross_query, cross_key, cross_value, cross_output;
    MoonshineNeonLinear mlp_input, mlp_output;
};
struct MoonshineNeonDecoderWeights {
    MoonshineNeonLinear token_embedding;
    std::vector<float> norm;
    std::vector<MoonshineNeonDecoderLayer> layers;
    MoonshineNeonLinear head;
};
MoonshineNeonDecoderWeights load_moonshine_neon_decoder_weights(const MoonshineAssets &assets,
                                                                assets::TensorStorageType storage);
// Model names and dtype conversion belong to binding.

} // namespace engine::models::moonshine_asr
