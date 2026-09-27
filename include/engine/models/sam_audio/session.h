#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::sam_audio {

std::shared_ptr<runtime::IVoiceModelLoader> make_sam_audio_loader();

}  // namespace engine::models::sam_audio
