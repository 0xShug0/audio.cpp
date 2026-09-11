#pragma once

#include "engine/framework/assets/resource_bundle.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::assets {
class TensorSource;
}

namespace engine::models::fun_asr_nano {

struct FunAsrNanoFrontendConfig {
  int sample_rate = 16000;
  int64_t feature_size = 80;
  int64_t frame_length_ms = 25;
  int64_t frame_shift_ms = 10;
  int64_t lfr_m = 7;
  int64_t lfr_n = 6;
  float preemphasis = 0.97F;
};

struct FunAsrNanoEncoderConfig {
  int64_t num_mel_bins = 80;
  int64_t num_stacked_frames = 7;
  int64_t input_size = 560;
  int64_t d_model = 512;
  int64_t attention_heads = 4;
  int64_t ffn_dim = 2048;
  int64_t layers = 50;
  int64_t timestamp_prediction_layers = 20;
  int64_t kernel_size = 11;
  int64_t max_position_embeddings = 2049;
  std::string activation = "relu";
};

struct FunAsrNanoAdaptorConfig {
  int64_t d_model = 1024;
  int64_t attention_heads = 8;
  int64_t ffn_dim = 256;
  int64_t layers = 2;
  std::string activation = "relu";
};

struct FunAsrNanoTextConfig {
  int64_t vocab_size = 151936;
  int64_t hidden_size = 1024;
  int64_t intermediate_size = 3072;
  int64_t layers = 28;
  int64_t attention_heads = 16;
  int64_t key_value_heads = 8;
  int64_t head_dim = 128;
  int64_t max_position_embeddings = 40960;
  int64_t bos_token_id = 151643;
  int64_t eos_token_id = 151645;
  float rms_norm_eps = 1.0e-6F;
  float rope_theta = 1000000.0F;
  bool tie_word_embeddings = true;
};

struct FunAsrNanoConfig {
  std::string model_type;
  int64_t audio_token_id = 151646;
  int64_t projector_hidden_size = 2048;
  bool tie_word_embeddings = true;
  FunAsrNanoFrontendConfig frontend;
  FunAsrNanoEncoderConfig encoder;
  FunAsrNanoAdaptorConfig adaptor;
  FunAsrNanoTextConfig text;
  std::vector<std::string> supported_languages = {"zh", "en", "ja"};
};

struct FunAsrNanoAssets {
  assets::ResourceBundle resources;
  FunAsrNanoConfig config;
  std::shared_ptr<const assets::TensorSource> model_weights;
};

std::shared_ptr<const FunAsrNanoAssets>
load_fun_asr_nano_assets(const std::filesystem::path &model_path);
std::shared_ptr<const FunAsrNanoAssets>
load_fun_asr_nano_assets(assets::ResourceBundle resources);

} // namespace engine::models::fun_asr_nano
