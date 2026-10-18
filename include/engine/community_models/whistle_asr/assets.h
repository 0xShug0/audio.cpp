#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/assets/resource_bundle.h"
#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::whistle_asr {

struct WhistleToken {
    std::string text;
    float score = 0.0f;
    uint8_t type = 0;
};

struct WhistleAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> weights;
    std::vector<float> mel_filterbank;
    std::array<std::array<uint16_t, 512>, 2> hadamard_permutations;
    std::vector<WhistleToken> tokenizer_pieces;
};

[[nodiscard]] std::string decode_whistle_tokens(
    const std::vector<WhistleToken> & pieces, const std::vector<int32_t> & ids);

[[nodiscard]] std::shared_ptr<const WhistleAssets> load_whistle_assets(
    const std::filesystem::path & model_path);

}  // namespace engine::community_models::whistle_asr
