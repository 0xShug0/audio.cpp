#include "engine/community_models/piper_tts/assets.h"

#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <utility>

namespace engine::models::piper_tts {
namespace {

namespace json = engine::io::json;

PiperTtsConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    if (json::require_string(root, "format") != "piper_tts_inference_config_v1") {
        throw std::runtime_error("Piper TTS requires piper_tts_inference_config_v1 config");
    }
    PiperTtsConfig out;
    out.vocab_size = json::require_i64(root, "num_symbols");
    out.sample_rate = json::require_i64(root.require("audio"), "sample_rate");
    out.espeak_voice = json::require_string(root.require("espeak"), "voice");
    for (const auto & [symbol, ids] : root.require("phoneme_id_map").as_object()) {
        const auto values = json::number_array_as<int32_t>(ids);
        if (values.size() != 1) {
            throw std::runtime_error("Piper TTS requires one ID per phoneme symbol");
        }
        out.phoneme_id_map.emplace(symbol, values.front());
    }
    engine::io::require_positive(out.vocab_size, "Piper TTS symbol count");
    engine::io::require_positive(out.sample_rate, "Piper TTS sample rate");
    if (out.vocab_size != 256 || out.sample_rate != 22050 ||
        out.espeak_voice.empty() || out.phoneme_id_map.empty()) {
        throw std::runtime_error("Piper TTS config does not match the supported medium voice architecture");
    }
    return out;
}

}  // namespace

std::shared_ptr<const PiperTtsAssets> load_piper_tts_assets(
    const std::filesystem::path & model_path) {
    auto resources = engine::model_spec::load_resource_bundle_for_family(
        model_path,
        "piper_tts");
    PiperTtsAssets out;
    out.config = parse_config(resources);
    out.weights = resources.open_tensor_source("weights");
    out.resources = std::move(resources);
    if (out.weights->tensors().empty()) {
        throw std::runtime_error("Piper TTS weights are empty");
    }
    return std::make_shared<PiperTtsAssets>(std::move(out));
}

}  // namespace engine::models::piper_tts
