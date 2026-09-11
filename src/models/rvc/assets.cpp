#include "engine/models/rvc/assets.h"

#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <utility>

namespace engine::models::rvc {
namespace {

RvcVoiceModel make_voice(
    const engine::assets::ResourceBundle & resources,
    std::string id,
    std::string version,
    int sample_rate,
    std::string checkpoint_id,
    std::string index_id,
    std::string index_vectors_id) {
    RvcVoiceModel out;
    out.id = std::move(id);
    out.version = std::move(version);
    out.sample_rate = sample_rate;
    out.checkpoint = resources.open_tensor_source(checkpoint_id);
    out.synthesizer_layout = infer_rvc_synthesizer_layout(*out.checkpoint, out.sample_rate, out.id);
    out.index_path = resources.require_file(index_id);
    out.index_vectors = resources.open_tensor_source(index_vectors_id);
    out.has_f0 = false;
    for (const auto & tensor : out.checkpoint->tensors()) {
        if (tensor.name == "enc_p.emb_pitch.weight") {
            out.has_f0 = true;
        }
        if (tensor.name == "emb_g.weight") {
            if (tensor.shape.size() != 2 || tensor.shape[0] <= 0) {
                throw std::runtime_error("RVC emb_g.weight shape is invalid for voice id: " + out.id);
            }
            out.speaker_count = static_cast<int>(tensor.shape[0]);
        }
    }
    if (out.speaker_count <= 0) {
        throw std::runtime_error("RVC voice missing emb_g.weight speaker table: " + out.id);
    }
    return out;
}

void add_voice(RvcAssets & assets, RvcVoiceModel voice) {
    const auto id = voice.id;
    auto inserted = assets.voices.emplace(id, std::move(voice));
    if (!inserted.second) {
        throw std::runtime_error("duplicate RVC voice id: " + id);
    }
}

}  // namespace

std::shared_ptr<const RvcAssets> load_rvc_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<RvcAssets>();
    assets->resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("rvc"));
    assets->hubert = assets->resources.open_tensor_source("support_hubert_base");
    assets->rmvpe = assets->resources.open_tensor_source("support_rmvpe");
    add_voice(*assets, make_voice(
        assets->resources,
        "chocola",
        "v1",
        40000,
        "voice_v1_chocola_checkpoint",
        "voice_v1_chocola_index",
        "voice_v1_chocola_index_vectors"));
    add_voice(*assets, make_voice(
        assets->resources,
        "fraise",
        "v1",
        40000,
        "voice_v1_fraise_checkpoint",
        "voice_v1_fraise_index",
        "voice_v1_fraise_index_vectors"));
    add_voice(*assets, make_voice(
        assets->resources,
        "default",
        "v2",
        40000,
        "voice_v2_default_checkpoint",
        "voice_v2_default_index",
        "voice_v2_default_index_vectors"));
    add_voice(*assets, make_voice(
        assets->resources,
        "manthos",
        "v2",
        40000,
        "voice_v2_manthos_checkpoint",
        "voice_v2_manthos_index",
        "voice_v2_manthos_index_vectors"));
    return assets;
}

}  // namespace engine::models::rvc
