#pragma once

#include "engine/community_models/fish_drama3/backbone.h"
#include "engine/community_models/fish_drama3/tower.h"
#include "engine/framework/codecs/fish_dac_codec_runtime.h"
#include "engine/models/fish_audio/assets.h"

namespace engine::models::fish_drama3 {

struct Drama3Assets {
    assets::ResourceBundle resources;
    Drama3BackboneDecoderConfig backbone;
    Drama3TowerConfig tower;
    codecs::FishDacCodecConfig codec;
    int32_t eos = 248046;
    std::shared_ptr<const fish_audio::FishAudioAssets> tokenizer_assets;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> codec_weights;
};

std::shared_ptr<const Drama3Assets> load_drama3_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::fish_drama3
