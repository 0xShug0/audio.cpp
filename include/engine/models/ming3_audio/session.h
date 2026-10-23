#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::ming3_audio {

std::shared_ptr<runtime::IVoiceModelLoader> make_ming3_audio_loader();

}  // namespace engine::models::ming3_audio
