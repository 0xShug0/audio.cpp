#pragma once

#include <cstdint>
#include <vector>

namespace engine::models::higgs_audio_tts {

constexpr int32_t kHiggsBocId = 1024;
constexpr int32_t kHiggsEocId = 1025;
constexpr int32_t kHiggsStopCode = -1;

int64_t higgs_delayed_frame_count(int64_t raw_frames, int64_t codebooks);

std::vector<int32_t> apply_higgs_delay_pattern(
    const std::vector<int32_t> & raw_codes,
    int64_t raw_frames,
    int64_t codebooks);

std::vector<int32_t> reverse_higgs_delay_pattern(
    const std::vector<int32_t> & delayed_codes,
    int64_t delayed_frames,
    int64_t codebooks);

}  // namespace engine::models::higgs_audio_tts
