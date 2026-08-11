#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/models/index_tts2_5/types.h"

namespace engine::models::index_tts2_5 {

// Normalizes the "lang" request option: trims, lowercases, and maps "auto" to
// an empty string (tokenizer-side language inference).
std::string normalize_index_tts2_5_lang(const std::string & value);

IndexTTS25Request parse_index_tts2_5_request(const runtime::TaskRequest & request);

}  // namespace engine::models::index_tts2_5
