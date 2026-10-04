#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::any_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_any_tts_loader();

}  // namespace engine::models::any_tts
