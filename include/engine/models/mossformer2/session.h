#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::mossformer2 {

std::shared_ptr<runtime::IVoiceModelLoader> make_mossformer2_loader();

}  // namespace engine::models::mossformer2
