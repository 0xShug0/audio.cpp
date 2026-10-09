#pragma once

#include "engine/framework/runtime/model.h"

#include <memory>

namespace engine::models::ast_audioset {

std::shared_ptr<runtime::IVoiceModelLoader> make_ast_audioset_loader();

}  // namespace engine::models::ast_audioset
