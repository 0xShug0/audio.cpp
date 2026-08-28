#pragma once

#include "engine/community_models/audio8_tts/assets.h"
#include "engine/community_models/audio8_tts/types.h"
#include "engine/framework/runtime/session.h"

#include <filesystem>
#include <optional>
#include <vector>

namespace engine::models::audio8_tts {

// Torch fallback for Falcon-H1 (0.1B) slow backbone.
// See src/community_models/audio8_tts/falcon_bridge.py and
// /workspace/models/Audio8-TTS-Preview-0.1b/modeling_arktts.py
// This is a temporary bridge until native ggml mamba kernels (ggml_ssm_conv / ggml_ssm_scan)
// are fully ported from llama.cpp/src/models/mamba-base.cpp:149.
runtime::AudioBuffer
generate_audio_via_torch_falcon(const Audio8TtsAssets & assets,
                                 const Audio8TtsRequest & request,
                                 const std::vector<Audio8TtsCodes> & reference_codes,
                                 const std::optional<Audio8TtsConversationTurn> & previous_turn);

bool is_falcon_backbone(const Audio8TtsAssets & assets) noexcept;

}  // namespace engine::models::audio8_tts
