#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::seed_vc {

std::shared_ptr<runtime::IVoiceModelLoader> make_seed_vc_loader();

}  // namespace engine::models::seed_vc
