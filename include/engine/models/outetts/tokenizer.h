#pragma once

#include "engine/models/outetts/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine::models::outetts {

struct OuteTTSVoiceFeatures {
  int energy = 0;
  int spectral_centroid = 0;
  int pitch = 0;
};

struct OuteTTSVoiceWord {
  std::string text;
  double duration = 0.0;
  OuteTTSVoiceFeatures features;
  std::vector<int32_t> codebook1;
  std::vector<int32_t> codebook2;
};

struct OuteTTSVoiceProfile {
  std::string text;
  OuteTTSVoiceFeatures global_features;
  std::vector<OuteTTSVoiceWord> words;
};

struct OuteTTSTextGenerationBudget {
  int64_t words = 0;
  int64_t non_whitespace_codepoints = 0;
  int64_t recommended_max_new_tokens = 0;
};

// Mirrors the sizing heuristic used by the upstream OuteTTS 1.0 runner. Audio
// code generation is much denser than text tokenization, so a character-sized
// chunk cannot safely use the same numeric value as its generated-token cap.
OuteTTSTextGenerationBudget
estimate_text_generation_budget(std::string_view text);

class OuteTTSTokenizer {
public:
  explicit OuteTTSTokenizer(std::shared_ptr<const OuteTTSAssets> assets);

  std::vector<int32_t> build_prompt(const std::string &text) const;
  std::vector<int32_t>
  build_clone_prompt(const std::string &text,
                     const OuteTTSVoiceProfile &profile) const;
  bool is_stop_token(int32_t token) const noexcept;
  int32_t eos_id() const noexcept;
  int32_t audio_end_id() const noexcept;
  bool append_audio_code(int32_t token, std::vector<int32_t> &codebook1,
                         std::vector<int32_t> &codebook2) const;

private:
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};

} // namespace engine::models::outetts
