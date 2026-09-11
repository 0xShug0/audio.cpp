#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/models/kokoro_tts/assets.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::kokoro_tts {

struct KokoroSynthesisInput {
    std::string voice_id;
    std::string language_code;
    std::string phonemes;
    std::vector<int32_t> input_ids;
    std::vector<float> style;
    float speaking_rate = 1.0f;
};

struct KokoroFrontendSessionState {
    std::string voice_id;
    std::string language_code;
    const KokoroVoicePack * voice_pack = nullptr;
    float speaking_rate = 1.0f;
};

KokoroFrontendSessionState resolve_kokoro_frontend_session_state(
    const std::optional<runtime::Transcript> & text,
    const std::optional<runtime::VoiceCondition> & voice,
    const KokoroAssets & assets);

void validate_kokoro_frontend_session_state(
    const runtime::Transcript & text,
    const std::optional<runtime::VoiceCondition> & voice,
    const KokoroFrontendSessionState & state,
    const KokoroAssets & assets);

KokoroSynthesisInput build_kokoro_synthesis_input(
    const runtime::Transcript & text,
    const KokoroFrontendSessionState & state,
    const KokoroAssets & assets);

int64_t estimate_kokoro_request_tokens(
    const runtime::SessionPreparationRequest & request,
    const KokoroFrontendSessionState & state,
    const KokoroAssets & assets);

}  // namespace engine::models::kokoro_tts
