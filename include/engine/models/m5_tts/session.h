#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::m5_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_m5_tts_loader();

}  // namespace engine::models::m5_tts
