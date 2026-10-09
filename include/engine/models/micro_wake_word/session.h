#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::micro_wake_word {

std::shared_ptr<runtime::IVoiceModelLoader> make_micro_wake_word_loader();

}  // namespace engine::models::micro_wake_word
