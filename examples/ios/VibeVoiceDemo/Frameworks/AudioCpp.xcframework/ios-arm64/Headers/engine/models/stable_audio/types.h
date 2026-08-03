#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::stable_audio {

struct StableAudioInpaintRegion {
    float start_seconds = 0.0F;
    float end_seconds = 0.0F;
};

struct StableAudioRequest {
    std::vector<std::string> prompts;
    std::vector<std::string> negative_prompts;
    std::vector<float> durations_seconds;
    int batch_size = 1;
    int num_inference_steps = 8;
    std::string sampler;
    float guidance_scale = 1.0F;
    float apg_scale = 1.0F;
    uint64_t seed = 0;
    bool seed_specified = false;
    bool truncate_output_to_duration = true;
    bool chunked_decode = true;
    float duration_padding_seconds = 6.0F;
    float init_noise_level = 1.0F;
    std::optional<runtime::AudioBuffer> init_audio = std::nullopt;
    std::optional<runtime::AudioBuffer> inpaint_audio = std::nullopt;
    std::vector<StableAudioInpaintRegion> inpaint_regions;
};

}  // namespace engine::models::stable_audio
