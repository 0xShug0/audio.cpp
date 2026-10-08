#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::sherpa_kws {

std::shared_ptr<runtime::IVoiceModelLoader> make_sherpa_kws_loader();

}  // namespace engine::models::sherpa_kws
