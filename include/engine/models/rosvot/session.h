#pragma once
#include "engine/framework/runtime/model.h"
namespace engine::models::rosvot {
std::shared_ptr<runtime::IVoiceModelLoader> make_rosvot_loader();
}
