#pragma once

#include "engine/models/vibevoice/types.h"

#include <filesystem>

namespace engine::models::vibevoice {

VibeVoiceReferenceVoiceState load_vibevoice_reference_voice_state(
    const std::filesystem::path & path);

void save_vibevoice_reference_voice_state(
    const std::filesystem::path & path,
    const VibeVoiceReferenceVoiceState & state);

}  // namespace engine::models::vibevoice
