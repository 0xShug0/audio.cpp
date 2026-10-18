#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>
#include <string>

namespace engine::models::zipvoice {

std::shared_ptr<runtime::IVoiceModelLoader> make_zipvoice_loader();

}  // namespace engine::models::zipvoice
