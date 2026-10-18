#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::pulsevad {

std::shared_ptr<runtime::IVoiceModelLoader> make_pulsevad_loader();

}  // namespace engine::models::pulsevad
