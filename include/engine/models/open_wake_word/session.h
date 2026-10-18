#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::open_wake_word {
std::shared_ptr<runtime::IVoiceModelLoader> make_open_wake_word_loader();
}
