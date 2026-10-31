#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::nemotron_transcribe {
std::shared_ptr<runtime::IVoiceModelLoader> make_nemotron_transcribe_loader();
}  // namespace engine::models::nemotron_transcribe
