#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::lfm_d1_omni {

std::shared_ptr<runtime::IVoiceModelLoader> make_lfm_d1_omni_loader();

} // namespace engine::models::lfm_d1_omni
