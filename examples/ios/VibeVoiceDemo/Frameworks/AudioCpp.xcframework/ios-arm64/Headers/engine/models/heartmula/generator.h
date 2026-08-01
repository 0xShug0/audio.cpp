#pragma once

#include "engine/models/heartmula/mula.h"
#include "engine/models/heartmula/tokenizer_text.h"

#include <cstdint>
#include <vector>

namespace engine::models::heartmula {

struct HeartMuLaGeneratedFrames {
    std::vector<int32_t> codes;
    int64_t codebooks = 0;
    int64_t frames = 0;
    uint64_t codec_randn_philox_offset = 0;
    uint64_t codec_randn_call_offset_blocks = 1;
};

HeartMuLaGeneratedFrames generate_heartmula_frames(
    const HeartMuLaPromptRequest & request,
    const HeartMuLaTextTokenizer & tokenizer,
    const HeartMuLaWeightsRuntime & mula,
    uint64_t seed);

}  // namespace engine::models::heartmula
