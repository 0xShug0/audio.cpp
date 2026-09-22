#include "engine/community_models/moss_voicegen/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::moss_voicegen {
namespace json = engine::io::json;
MossVoiceGenConfig parse_model_config(const json::Value & root) {
    // The whole config is the moss_tts_delay family's; the parser lives beside the
    // runtimes that consume it.
    return decoders::parse_moss_tts_delay_config(root, "MOSS-VoiceGenerator");
}

MossVoiceGenConfig parse_config(const assets::ResourceBundle & resources) {
    return parse_model_config(resources.parse_json("config"));
}

std::shared_ptr<const MossVoiceGenAssets> load_moss_voicegen_assets(const std::filesystem::path & model_path) {
    MossVoiceGenAssets assets;
    assets.resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("moss_voicegen"));
    assets.config = parse_config(assets.resources);
    assets.model_weights = assets.resources.open_tensor_source("model_weights");
    assets.audio_tokenizer_weights = assets.resources.open_tensor_source("audio_tokenizer_weights");
    return std::make_shared<MossVoiceGenAssets>(std::move(assets));
}

}  // namespace engine::models::moss_voicegen
