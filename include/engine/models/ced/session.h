#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::ced {

std::shared_ptr<runtime::IVoiceModelLoader> make_ced_loader();

}  // namespace engine::models::ced
