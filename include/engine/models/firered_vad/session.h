#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::firered_vad {

std::shared_ptr<runtime::IVoiceModelLoader> make_firered_vad_loader();

}  // namespace engine::models::firered_vad
