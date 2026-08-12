#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/models/index_tts2/types.h"

namespace engine::models::index_tts2 {

// Normalizes the "language" request option (v2.5): trims, lowercases, and maps
// "auto" to an empty string (tokenizer-side language inference).
std::string normalize_index_tts2_language(const std::string & value);

IndexTTS2Request parse_index_tts2_request(const runtime::TaskRequest & request);

}  // namespace engine::models::index_tts2
