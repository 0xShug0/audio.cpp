#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::audio_flamingo_next {
std::shared_ptr<runtime::IVoiceModelLoader> make_audio_flamingo_next_loader();
}  // namespace engine::models::audio_flamingo_next
