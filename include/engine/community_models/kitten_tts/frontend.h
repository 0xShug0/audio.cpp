#pragma once

#include "engine/community_models/kitten_tts/assets.h"
#include "engine/framework/audio/espeak_phonemizer.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::kitten_tts {

struct KittenSynthesisInput {
    std::string voice_id;
    std::string language_code;
    std::string phonemes;
    size_t normalized_text_length = 0;
    std::vector<int32_t> input_ids;
    std::vector<float> style;
    float speaking_rate = 1.0f;
};

struct KittenFrontendSessionState {
    std::string voice_id;
    std::string language_code;
    const KittenVoicePack *voice_pack = nullptr;
    float speaking_rate = 1.0f;
};

KittenFrontendSessionState resolve_kitten_frontend_session_state(const std::optional<runtime::Transcript> &text,
                                                                 const std::optional<runtime::VoiceCondition> &voice,
                                                                 const KittenAssets &assets);

void validate_kitten_frontend_session_state(const runtime::Transcript &text,
                                            const std::optional<runtime::VoiceCondition> &voice,
                                            const KittenFrontendSessionState &state, const KittenAssets &assets);

KittenSynthesisInput build_kitten_synthesis_input(const runtime::Transcript &text,
                                                  const KittenFrontendSessionState &state, const KittenAssets &assets,
                                                  const engine::audio::EspeakPhonemizer &phonemizer);

int64_t estimate_kitten_request_tokens(const runtime::SessionPreparationRequest &request,
                                       const KittenFrontendSessionState &state, const KittenAssets &assets,
                                       const engine::audio::EspeakPhonemizer &phonemizer);

} // namespace engine::models::kitten_tts
