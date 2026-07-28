#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::rvc {

std::shared_ptr<runtime::IVoiceModelLoader> make_rvc_loader();

}  // namespace engine::models::rvc
