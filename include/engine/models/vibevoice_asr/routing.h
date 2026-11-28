#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <string>

namespace engine::models::vibevoice_asr {

class VibeVoiceASRSession;

struct VibeVoiceASRImplementation {
    const std::string family;
    static const VibeVoiceASRImplementation offline;
    static const VibeVoiceASRImplementation streaming_model;

    bool streaming;
    int64_t default_max_tokens;
    runtime::TaskResult (VibeVoiceASRSession::*run)(const runtime::TaskRequest &);
};

const VibeVoiceASRImplementation & require_vibevoice_asr_implementation(const std::string & family);

}  // namespace engine::models::vibevoice_asr
