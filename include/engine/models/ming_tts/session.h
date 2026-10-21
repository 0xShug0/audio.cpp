#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::ming_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_ming_tts_loader();

}  // namespace engine::models::ming_tts
