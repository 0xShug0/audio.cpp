#pragma once

// MOSS-TTS-v1.5: the 8B, 32-codebook member of the moss_tts_delay family.
//
// The geometry, token ids, backbone, heads and delay decoder are all the
// family's, shared with MOSS-VoiceGenerator; what belongs to this checkpoint is
// the prompt (reference audio and a duration budget, where voice design has
// neither) and the session that drives them.

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/decoders/moss_tts_delay/config.h"

#include <filesystem>
#include <memory>

namespace engine::models::moss_tts_v15 {

using Config = decoders::MossTtsDelayConfig;

struct Assets {
    assets::ResourceBundle resources;
    Config config;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> audio_tokenizer_weights;
};

std::shared_ptr<const Assets> load_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::moss_tts_v15
