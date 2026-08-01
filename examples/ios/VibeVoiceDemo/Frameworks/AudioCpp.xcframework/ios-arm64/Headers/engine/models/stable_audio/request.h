#pragma once

#include "engine/models/stable_audio/types.h"

namespace engine::models::stable_audio {

StableAudioRequest parse_stable_audio_request(const runtime::TaskRequest & request);

}  // namespace engine::models::stable_audio
