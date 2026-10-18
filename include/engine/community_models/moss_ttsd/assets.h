#pragma once

// MOSS-TTSD: the dialogue member of the moss_tts_delay family (8B, 16 codebooks).
//
// The geometry, token ids, backbone, heads, delay decoder and prompt assembly
// are all the family's, shared with MOSS-TTS-v1.5 and MOSS-VoiceGenerator. What
// belongs to this checkpoint is the rendering of its <user_inst> block and a
// session that clones by continuation rather than by reference alone.

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/decoders/moss_tts_delay/config.h"

#include <filesystem>
#include <memory>

namespace engine::models::moss_ttsd {

using Config = decoders::MossTtsDelayConfig;

struct Assets {
    assets::ResourceBundle resources;
    Config config;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> audio_tokenizer_weights;
};

std::shared_ptr<const Assets> load_moss_ttsd_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::moss_ttsd
