#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::moss_tts_local {

std::shared_ptr<runtime::IVoiceModelLoader> make_moss_tts_local_loader();

}  // namespace engine::models::moss_tts_local
