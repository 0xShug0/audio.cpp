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
  // Streaming: how many already-generated patches are decoded together with
  // each emitted patch as left context, and how many later patches are waited
  // for as right context. The AudioVAE decoder is not causal; a patch decoded
  // alone deviates at both edges from the full-sequence decode, and the hard
  // concatenation is heard as a click at every patch boundary. Measured on
  // VoxCPM2 Q8 (48 kHz, plain text, fixed seed) against the offline decode:
  // alone, 40 % of all samples deviate by more than 0.01 and seams jump by up
  // to 0.55 of full scale; with 3 patches of left context the largest
  // deviation is 0.0015 (about 50 steps of 16-bit), with 4 it is 0.0005, with
  // 5 or more 0.0004. Right context changes nothing audible and each patch of
  // it delays the stream by one patch of audio. The decoder graph is built in
  // capacity tiers, so a window of 4 patches (3 left) costs the same as 3 and
  // a window of 5 the same as 8: 3 / 0 is the last cheap point.
  int64_t stream_left_context = 3;
  int64_t stream_right_context = 0;
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
  // Context patches at the front and back of decode_features: decoded with
  // the chunk for continuity, then dropped from its audio.
  int64_t trim_front_patches = 0;
  int64_t trim_back_patches = 0;
};

struct VoxCPM2StreamingResult {
  std::vector<VoxCPM2StreamingChunk> chunks;
  int64_t generated_patches = 0;
};

} // namespace engine::models::voxcpm2
