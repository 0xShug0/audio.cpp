#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::echo_tts {

struct EchoTokenizedText {
    std::vector<int32_t> input_ids;
    std::vector<float> mask;  // 1.0 for real tokens, 0.0 for padding
    std::string normalized_text;
    bool truncated = false;
};

// Applies the WhisperD-style normalisation from inference.py::tokenizer_encode
// and returns the normalised string. Exposed separately because the session
// reports the normalised text back to the caller.
std::string normalize_echo_text(const std::string & text);

// Byte-level tokenizer: a BOS 0 followed by the raw UTF-8 bytes of the
// normalised text. `max_length` is the hard cap (768 upstream) and counts the
// BOS. When `pad_to_max` is false the returned vectors are exactly as long as
// the encoded text.
EchoTokenizedText tokenize_echo_text(
    const std::string & text,
    int64_t max_length,
    bool normalize = true,
    bool pad_to_max = false);

}  // namespace engine::models::echo_tts
