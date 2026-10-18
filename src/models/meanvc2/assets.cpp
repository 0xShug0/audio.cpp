#include "engine/models/meanvc2/assets.h"

#include "engine/framework/model_spec/package.h"

#include <utility>

namespace engine::models::meanvc2 {

std::shared_ptr<const MeanVC2Assets> load_meanvc2_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<MeanVC2Assets>();
    assets->resources = engine::model_spec::load_resource_bundle_for_family(model_path, "meanvc2");
    assets->flow_weights = assets->resources.open_tensor_source("vc");
    assets->vocos_weights = assets->resources.open_tensor_source("vocos");
    assets->asr_weights = assets->resources.open_tensor_source("asr");
    assets->speaker_wavlm_weights = assets->resources.open_tensor_source("speaker_wavlm");
    assets->speaker_ecapa_weights = assets->resources.open_tensor_source("speaker_ecapa");
    return assets;
}

}  // namespace engine::models::meanvc2
