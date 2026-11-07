#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::moss_soundeffect {
std::shared_ptr<runtime::IVoiceModelLoader> make_moss_soundeffect_loader();
}
