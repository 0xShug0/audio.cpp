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

struct FunAsrNanoAdaptorStage {
  std::string name;
  std::vector<float> values;
};

struct FunAsrNanoAdaptorEmbeddings {
  std::vector<float> values;
  int64_t tokens = 0;
  int64_t hidden_size = 0;
  std::vector<FunAsrNanoAdaptorStage> stages;
};

struct FunAsrNanoPromptRequest {
  std::string prompt;
  std::string language = "auto";
  bool enable_itn = true;
};

struct FunAsrNanoPrompt {
  std::vector<int32_t> input_ids;
  std::vector<int32_t> audio_token_positions;
  std::vector<int32_t> attention_mask;
};

struct FunAsrNanoGenerationOptions {
  int64_t max_new_tokens = 512;
  bool capture_logits = false;
};

struct FunAsrNanoGeneratedTokens {
  std::vector<int32_t> token_ids;
  std::vector<std::vector<float>> step_logits;
};

} // namespace engine::models::fun_asr_nano
