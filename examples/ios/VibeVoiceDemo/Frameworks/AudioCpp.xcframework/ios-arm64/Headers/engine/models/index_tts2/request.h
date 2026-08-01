#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/models/index_tts2/types.h"

namespace engine::models::index_tts2 {

IndexTTS2Request parse_index_tts2_request(const runtime::TaskRequest & request);

}  // namespace engine::models::index_tts2
