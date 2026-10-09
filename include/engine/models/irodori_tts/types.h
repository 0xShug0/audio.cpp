#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace engine::models::irodori_tts {

struct IrodoriGenerationOptions {
  int64_t num_inference_steps = 40;
  float text_guidance_scale = 3.0F;
  float caption_guidance_scale = 3.0F;
  float speaker_guidance_scale = 5.0F;
  std::string guidance_mode = "independent";
  float guidance_min_t = 0.5F;
  float guidance_max_t = 1.0F;
  uint32_t seed = 0;
  bool seed_specified = false;
  float duration_scale = 1.0F;
  float min_seconds = 0.5F;
  float max_seconds = 30.0F;
  float duration_seconds = 0.0F;
  bool duration_seconds_specified = false;
  bool context_kv_cache = true;
  bool trim_tail = true;
  int64_t tail_window_size = 20;
  float tail_std_threshold = 0.05F;
  float tail_mean_threshold = 0.1F;
};

struct IrodoriRequest {
  std::string text;
  std::string caption;
  bool no_ref = true;
  runtime::AudioBuffer reference_audio;
  bool has_reference_audio = false;
  // Speaker Inversion embedding used as the speaker condition instead of
  // reference audio.
  std::optional<std::filesystem::path> speaker_embedding_path;
  // Reference audio length cap in seconds (Python's max_ref_seconds); 0 keeps
  // the whole reference.
  float max_ref_sec = 0.0F;
  IrodoriGenerationOptions generation;
};

} // namespace engine::models::irodori_tts
