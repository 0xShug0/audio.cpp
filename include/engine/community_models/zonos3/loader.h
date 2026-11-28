#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::zonos3 {

std::shared_ptr<runtime::IVoiceModelLoader> make_zonos3_loader();

}  // namespace engine::models::zonos3
