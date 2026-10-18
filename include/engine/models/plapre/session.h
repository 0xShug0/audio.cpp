#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::plapre {

std::shared_ptr<runtime::IVoiceModelLoader> make_plapre_loader();

}  // namespace engine::models::plapre
