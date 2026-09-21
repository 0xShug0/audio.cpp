#include "engine/community_models/moss_tts_v15/assets.h"

#include "engine/framework/model_spec/package.h"

#include <utility>

namespace engine::models::moss_tts_v15 {

std::shared_ptr<const Assets> load_moss_tts_v15_assets(const std::filesystem::path & model_path) {
    Assets assets;
    assets.resources = engine::model_spec::load_resource_bundle(
        model_path, engine::model_spec::default_spec_path("moss_tts_v15"));
    assets.config = decoders::parse_moss_tts_delay_config(
        assets.resources.parse_json("config"), "MOSS-TTS-v1.5");
    assets.model_weights = assets.resources.open_tensor_source("model_weights");
    assets.audio_tokenizer_weights = assets.resources.open_tensor_source("audio_tokenizer_weights");
    return std::make_shared<Assets>(std::move(assets));
}

}  // namespace engine::models::moss_tts_v15
