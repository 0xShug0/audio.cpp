#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::vevo2 {

std::unique_ptr<runtime::ILoadedVoiceModel> load_vevo2_model(const runtime::ModelLoadRequest & request);
std::shared_ptr<runtime::IVoiceModelLoader> make_vevo2_loader();

}  // namespace engine::models::vevo2
