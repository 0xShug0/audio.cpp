#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::voxmel_tts {
std::shared_ptr<runtime::IVoiceModelLoader> make_voxmel_tts_loader();
}
