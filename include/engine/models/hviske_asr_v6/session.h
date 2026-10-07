#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::hviske_asr_v6 {
std::shared_ptr<runtime::IVoiceModelLoader> make_hviske_asr_v6_loader();
}
