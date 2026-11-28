#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::zonos2 {

std::shared_ptr<runtime::IVoiceModelLoader> make_zonos2_loader();

}  // namespace engine::models::zonos2
