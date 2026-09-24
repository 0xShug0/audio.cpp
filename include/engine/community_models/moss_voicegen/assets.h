#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/decoders/moss_tts_delay/config.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::moss_voicegen {

// The geometry, token ids and runtimes are the moss_tts_delay family's, shared with
// the other checkpoints that use this architecture; see
// engine/framework/decoders/moss_tts_delay/config.h. The aliases keep this model's
// own names readable at its call sites.
using MossVoiceGenBackboneConfig = decoders::MossTtsDelayBackboneConfig;
using MossVoiceGenConfig = decoders::MossTtsDelayConfig;

struct MossVoiceGenAssets {
    assets::ResourceBundle resources;
    MossVoiceGenConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> audio_tokenizer_weights;
};

std::shared_ptr<const MossVoiceGenAssets> load_moss_voicegen_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::moss_voicegen
