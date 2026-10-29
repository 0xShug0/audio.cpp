#pragma once

#include "engine/community_models/nemotron_transcribe/ctc.h"
#include "engine/community_models/nemotron_transcribe/decoder.h"
#include "engine/community_models/nemotron_transcribe/encoder.h"
#include "engine/framework/assets/resource_bundle.h"

namespace engine::models::nemotron_transcribe {

struct TranscribeAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> decoder_weights;
    std::shared_ptr<const assets::TensorSource> encoder_weights;
    FrontendConfig frontend;
    EncoderConfig encoder;
    DecoderConfig decoder;
    CtcConfig ctc;
    std::vector<float> mel_filterbank;
    std::vector<float> frontend_window;
    std::string alignment_tokenizer;
    int64_t maximum_context = 262144;
    // Zero selects the native duration-based recognition budget.
    int64_t max_new_tokens = 0;
};

std::shared_ptr<const TranscribeAssets> load_transcribe_assets(const std::filesystem::path & path);

}  // namespace engine::models::nemotron_transcribe
