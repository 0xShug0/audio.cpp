#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::community_models::reverb {

std::shared_ptr<runtime::IVoiceModelLoader> make_reverb_loader();

}  // namespace engine::community_models::reverb
