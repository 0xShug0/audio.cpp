#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::mulacover {

std::shared_ptr<runtime::IVoiceModelLoader> make_mulacover_loader();

}  // namespace engine::models::mulacover
