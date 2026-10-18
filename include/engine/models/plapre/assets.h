#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/modules/vocoders/hift_vocoder.h"
#include "engine/models/plapre/generator.h"
#include "engine/models/plapre/kanade.h"

#include <map>

namespace engine::models::plapre {

struct PlapreAssets {
    assets::ResourceBundle resources;
    PlapreLlamaConfig ar_config;
    KanadeMelConfig mel_config;
    KanadeEncoderConfig encoder_config;
    modules::HiftVocoderConfig hift_config;
    std::map<std::string, std::vector<float>> speakers;
};

std::shared_ptr<const PlapreAssets> load_plapre_assets(const std::filesystem::path & path);

}  // namespace engine::models::plapre
