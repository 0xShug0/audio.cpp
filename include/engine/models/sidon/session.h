#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::sidon {

std::shared_ptr<runtime::IVoiceModelLoader> make_sidon_loader();

}  // namespace engine::models::sidon
