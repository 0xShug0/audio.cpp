#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <vector>

namespace engine::models::glm_tts {

struct GlmTTSMelFeatures {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t dims = 80;
};

struct GlmTTSFbankFeatures {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t dims = 80;
};

GlmTTSMelFeatures compute_glm_tts_prompt_mel(
    const runtime::AudioBuffer & audio);

GlmTTSFbankFeatures compute_glm_tts_campplus_fbank(
    const runtime::AudioBuffer & audio);

}  // namespace engine::models::glm_tts
