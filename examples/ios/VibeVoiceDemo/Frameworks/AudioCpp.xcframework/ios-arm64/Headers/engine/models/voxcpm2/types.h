#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::voxcpm2 {

struct VoxCPM2GenerationOptions {
  int64_t min_tokens = 2;
  int64_t max_tokens = 4096;
  int64_t num_inference_steps = 10;
  float guidance_scale = 2.0F;
  bool retry_badcase = true;
  int64_t retry_badcase_max_times = 3;
  float retry_badcase_ratio_threshold = 6.0F;
  uint32_t seed = 1234;
  std::string cfm_noise_file;
};

struct VoxCPM2PromptAudio {
  runtime::AudioBuffer audio;
  std::string text;
};

struct VoxCPM2EncodedPrompt {
  std::string prompt_text;
  std::vector<float> prompt_features;
  int64_t prompt_patches = 0;
  std::vector<float> reference_features;
  int64_t reference_patches = 0;
};

struct VoxCPM2Request {
  std::string text;
  std::optional<VoxCPM2PromptAudio> prompt = std::nullopt;
  std::optional<runtime::AudioBuffer> reference_audio = std::nullopt;
  VoxCPM2GenerationOptions generation;
};

struct VoxCPM2TextPrompt {
  std::string text;
  std::vector<int32_t> input_ids;
};

struct VoxCPM2Result {
  runtime::AudioBuffer audio;
  std::vector<float> generated_features;
  int64_t generated_patches = 0;
  std::vector<float> decode_features;
  int64_t decode_patches = 0;
  int64_t decode_trim_patches = 0;
};

struct VoxCPM2StreamingChunk {
  std::vector<float> decode_features;
  int64_t decode_patches = 0;
  int64_t generated_patches = 0;
};

struct VoxCPM2StreamingResult {
  std::vector<VoxCPM2StreamingChunk> chunks;
  int64_t generated_patches = 0;
};

} // namespace engine::models::voxcpm2
