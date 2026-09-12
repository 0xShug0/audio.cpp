#include "engine/community_models/coqui_speedy_speech/assets.h"

#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <sstream>
#include <utility>

namespace engine::community_models::coqui_speedy_speech {
namespace {
namespace json = engine::io::json;

Config parse_config(const engine::assets::ResourceBundle &resources) {
  const auto root = resources.parse_json("config");
  if (json::require_string(root, "format") != "coqui_speedy_speech_v1") {
    throw std::runtime_error(
        "Coqui SpeedySpeech requires coqui_speedy_speech_v1 config");
  }
  Config out;
  out.sample_rate = json::require_i64(root, "sample_rate");
  out.hop_length = json::require_i64(root, "hop_length");
  out.num_mels = json::require_i64(root, "num_mels");
  out.vocab_size = json::require_i64(root, "vocab_size");
  out.hidden_channels = json::require_i64(root, "hidden_channels");
  out.encoder_dilations =
      json::number_array_as<int64_t>(root.require("encoder_dilations"));
  out.decoder_dilations =
      json::number_array_as<int64_t>(root.require("decoder_dilations"));
  out.phonemes = json::require_string(root, "phonemes");
  out.punctuations = json::require_string(root, "punctuations");
  if (out.sample_rate != 22050 || out.hop_length != 256 || out.num_mels != 80 ||
      out.vocab_size != 130 || out.hidden_channels != 128 ||
      out.encoder_dilations.size() != 13 ||
      out.decoder_dilations.size() != 17) {
    throw std::runtime_error("unsupported Coqui SpeedySpeech architecture");
  }
  return out;
}
} // namespace

std::shared_ptr<const Assets>
load_assets(const std::filesystem::path &model_path) {
  auto resources = engine::model_spec::load_resource_bundle_for_family(
      model_path, "coqui_speedy_speech");
  Assets out;
  out.config = parse_config(resources);
  out.weights = resources.open_tensor_source("weights");
  std::istringstream lexicon(resources.read_text("lexicon"));
  for (std::string line; std::getline(lexicon, line);) {
    const auto tab = line.find('\t');
    if (tab != std::string::npos && tab != 0 && tab + 1 < line.size())
      out.lexicon.emplace(line.substr(0, tab), line.substr(tab + 1));
  }
  if (out.lexicon.size() < 100000)
    throw std::runtime_error("Coqui SpeedySpeech Gruut lexicon is incomplete");
  out.resources = std::move(resources);
  if (!out.weights->has_tensor("acoustic.emb.weight") ||
      !out.weights->has_tensor("vocoder.conv_pre.weight")) {
    throw std::runtime_error(
        "Coqui SpeedySpeech package is missing inference tensors");
  }
  return std::make_shared<Assets>(std::move(out));
}

} // namespace engine::community_models::coqui_speedy_speech
