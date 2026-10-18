#pragma once

#include "engine/framework/runtime/session.h"

namespace minitts::server {

// On /v1/audio/speech/live, `input` is the text a text-to-speech model speaks,
// so those models need it. A model that answers or converts the live audio
// (s2s, vc, svc) takes it as an optional prompt and has its own default.
inline bool live_speech_requires_input(engine::runtime::VoiceTaskKind task) {
    switch (task) {
        case engine::runtime::VoiceTaskKind::SpeechToSpeech:
        case engine::runtime::VoiceTaskKind::VoiceConversion:
        case engine::runtime::VoiceTaskKind::Svc:
            return false;
        default:
            return true;
    }
}

} // namespace minitts::server
