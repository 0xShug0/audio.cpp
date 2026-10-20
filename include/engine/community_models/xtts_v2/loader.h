#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::xtts_v2 {
std::shared_ptr<runtime::IVoiceModelLoader> make_xtts_v2_loader();
}  // namespace engine::models::xtts_v2
