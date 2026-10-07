#pragma once

#include "engine/framework/runtime/registry.h"

namespace engine::models::voxtral_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_voxtral_tts_loader();

}  // namespace engine::models::voxtral_tts
