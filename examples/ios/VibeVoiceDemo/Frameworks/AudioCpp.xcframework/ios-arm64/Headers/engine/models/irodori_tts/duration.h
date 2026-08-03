#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::irodori_tts {

std::vector<float> build_irodori_duration_features(
    const std::string & text,
    int64_t token_count,
    int64_t max_text_len,
    bool has_speaker);

}  // namespace engine::models::irodori_tts
