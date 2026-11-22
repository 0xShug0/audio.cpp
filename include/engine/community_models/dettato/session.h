#pragma once
#include "engine/framework/runtime/model.h"
namespace engine::community_models::dettato {
std::shared_ptr<runtime::IVoiceModelLoader> make_dettato_loader();
}
