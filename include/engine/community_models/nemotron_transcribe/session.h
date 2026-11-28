#pragma once

#include "engine/framework/runtime/model.h"

#include <string>

namespace engine::models::nemotron_transcribe {
std::string language_name(const std::string & value);
std::shared_ptr<runtime::IVoiceModelLoader> make_nemotron_transcribe_loader();
}  // namespace engine::models::nemotron_transcribe
