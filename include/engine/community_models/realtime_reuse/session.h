#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::realtime_reuse {
std::shared_ptr<runtime::IVoiceModelLoader> make_realtime_reuse_loader();
}
