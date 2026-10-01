#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::index_echo {

std::shared_ptr<runtime::IVoiceModelLoader> make_index_echo_loader();

}  // namespace engine::models::index_echo
