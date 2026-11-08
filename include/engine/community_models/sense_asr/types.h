#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine::community_models::sense_asr {

struct SenseAsrAudioFeatures {
  std::vector<float> values;
  int64_t frames = 0;
  int64_t feature_dim = 0;
};

struct SenseAsrEncoderOutput {
  std::vector<float> logits;
  int64_t frames = 0;
  int64_t vocab_size = 0;
};

// A word, or a single character in scripts written without spaces, located
// by the CTC emission frames of its tokens. Frames count encoder output
// frames after the query-token prefix (one LFR frame each); end_frame is
// exclusive.
struct SenseAsrWordFrames {
  std::string text;
  int64_t start_frame = 0;
  int64_t end_frame = 0;
};

struct SenseAsrDecodedTokens {
  std::vector<int32_t> ids;
  std::string text;
  std::vector<std::string> tags;
  std::string language;
  std::vector<SenseAsrWordFrames> words;
};

struct SenseAsrTranscriptionOptions {
  std::string language = "auto";
  bool enable_itn = true;
  bool keep_tags = false;
  bool return_timestamps = false;
};

} // namespace engine::community_models::sense_asr
