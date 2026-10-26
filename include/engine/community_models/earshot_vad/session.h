#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/community_models/earshot_vad/runtime.h"

#include <memory>

namespace engine::community_models::earshot_vad {

std::unique_ptr<runtime::ILoadedVoiceModel> load_earshot_vad_model(const runtime::ModelLoadRequest & request);
std::shared_ptr<runtime::IVoiceModelLoader> make_earshot_vad_loader();

}  // namespace engine::community_models::earshot_vad
