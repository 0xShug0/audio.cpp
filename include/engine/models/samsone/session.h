#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::samsone {

std::shared_ptr<runtime::IVoiceModelLoader> make_samsone_loader();

}  // namespace engine::models::samsone
