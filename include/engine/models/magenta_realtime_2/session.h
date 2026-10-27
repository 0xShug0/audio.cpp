#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::magenta_realtime_2 {

std::shared_ptr<runtime::IVoiceModelLoader> make_magenta_realtime_2_loader();

}  // namespace engine::models::magenta_realtime_2
