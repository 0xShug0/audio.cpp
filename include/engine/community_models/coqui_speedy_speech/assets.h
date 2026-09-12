#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::community_models::coqui_speedy_speech {

struct Config {
  int64_t sample_rate = 22050;
  int64_t hop_length = 256;
  int64_t num_mels = 80;
  int64_t vocab_size = 130;
  int64_t hidden_channels = 128;
  std::vector<int64_t> encoder_dilations;
  std::vector<int64_t> decoder_dilations;
  std::string phonemes;
  std::string punctuations;
};

struct Assets {
  engine::assets::ResourceBundle resources;
  Config config;
  std::shared_ptr<const engine::assets::TensorSource> weights;
  std::unordered_map<std::string, std::string> lexicon;
};

std::shared_ptr<const Assets>
load_assets(const std::filesystem::path &model_path);

} // namespace engine::community_models::coqui_speedy_speech
