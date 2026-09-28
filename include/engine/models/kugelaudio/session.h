#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::kugelaudio {

std::shared_ptr<runtime::IVoiceModelLoader> make_kugelaudio_loader();

}  // namespace engine::models::kugelaudio
