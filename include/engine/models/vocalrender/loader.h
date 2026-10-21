#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::vocalrender {

std::shared_ptr<runtime::IVoiceModelLoader> make_vocalrender_loader();

} // namespace engine::models::vocalrender
