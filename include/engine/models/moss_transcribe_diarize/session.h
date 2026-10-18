#pragma once

#include "engine/framework/runtime/model.h"

namespace engine::models::moss_transcribe_diarize {

std::shared_ptr<runtime::IVoiceModelLoader> make_moss_transcribe_diarize_loader();

}  // namespace engine::models::moss_transcribe_diarize
