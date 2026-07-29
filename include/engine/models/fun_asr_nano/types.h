#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::fun_asr_nano {

struct FunAsrNanoAudioFeatures {
  std::vector<float> values;
  int64_t frames = 0;
  int64_t feature_dim = 0;
  int64_t valid_frames = 0;
};

struct FunAsrNanoEncoderStage {
  std::string name;
  std::vector<float> values;
};

struct FunAsrNanoEncoderEmbeddings {
  std::vector<float> values;
  int64_t frames = 0;
  int64_t valid_frames = 0;
  int64_t hidden_size = 0;
  std::vector<FunAsrNanoEncoderStage> stages;
};

} // namespace engine::models::fun_asr_nano
