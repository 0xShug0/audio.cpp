#pragma once
#include "engine/framework/runtime/model.h"

namespace engine::models::latentsync {
std::shared_ptr<runtime::IVoiceModelLoader> make_latentsync_loader();
}
