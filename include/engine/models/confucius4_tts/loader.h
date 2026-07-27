#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::confucius4_tts {

std::shared_ptr<runtime::IVoiceModelLoader> make_confucius4_tts_loader();

}  // namespace engine::models::confucius4_tts
