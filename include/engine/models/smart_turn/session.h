#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::smart_turn {

std::shared_ptr<runtime::IVoiceModelLoader> make_smart_turn_loader();

}  // namespace engine::models::smart_turn
