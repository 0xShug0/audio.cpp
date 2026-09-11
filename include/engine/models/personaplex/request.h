#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/models/personaplex/assets.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::personaplex {

struct PersonaPlexGenerationOptions {
    std::string voice_id = "NATF2";
    std::optional<runtime::AudioBuffer> voice_prompt_audio = std::nullopt;
    std::string system_prompt;
    float temperature = 0.8F;
    float text_temperature = 0.8F;
    int64_t top_k = 250;
    int64_t text_top_k = 250;
    bool do_sample = true;
    std::optional<uint32_t> seed;
};

struct PersonaPlexRequest {
    runtime::AudioBuffer audio;
    PersonaPlexGenerationOptions generation;
};

struct PersonaPlexVoicePromptState {
    std::vector<float> embeddings;
    std::vector<int64_t> cache;
    int64_t frames = 0;
};

PersonaPlexRequest make_personaplex_request(
    const runtime::TaskRequest & request,
    const PersonaPlexAssets & assets);

PersonaPlexGenerationOptions make_personaplex_generation_options(
    const runtime::TaskRequest & request,
    const PersonaPlexAssets & assets);

PersonaPlexVoicePromptState load_personaplex_voice_prompt(
    const PersonaPlexAssets & assets,
    const std::string & voice_id);

}  // namespace engine::models::personaplex
