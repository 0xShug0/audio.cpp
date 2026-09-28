#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::tone_color_vc {

std::shared_ptr<runtime::IVoiceModelLoader> make_tone_color_vc_loader();

}  // namespace engine::models::tone_color_vc
