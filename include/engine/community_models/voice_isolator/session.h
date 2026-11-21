#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::voice_isolator {
std::shared_ptr<runtime::IVoiceModelLoader> make_voice_isolator_loader();
}
