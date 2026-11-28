#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/audio/mel_spectrogram_frontend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/speech_encoders/lfm3_audio_encoder_runtime.h"
#include "engine/framework/modules/transformers/lfm_decoder_runtime.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace engine::models::lfm_d1_omni {

struct LfmD1OmniBackboneSpec {
    modules::LfmDecoderConfig decoder;
    int64_t vocab_size = 0;
    int64_t context_length = 0;
    int64_t sliding_window = 0;
};

struct LfmD1OmniAudioSpec : modules::Lfm3AudioEncoderConfig {
    audio::MelSpectrogramFrontendConfig frontend;
    // Applied after the frontend: its shift/divisor fields currently require dynamic clipping.
    float normalization_offset = 4.0f;
    float normalization_divisor = 4.0f;
};

struct LfmD1OmniDecisionSpec {
    double noul_temperature = 0;
    double choice_temperature = 0;
    double score_temperature = 0;
    // Official d1-Omni-2B contract, selected by lfm3.decision.type.
    int64_t codes = 2, prompt = 1, set_size = 16;
    bool numbers = true;
};

struct LfmD1OmniAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> model, mmproj;
    std::shared_ptr<const tokenizers::LlamaBpeTokenizer> tokenizer;
    LfmD1OmniBackboneSpec backbone;
    LfmD1OmniAudioSpec audio;
    LfmD1OmniDecisionSpec decision;
    int32_t audio_token_id = -1;
    // Always populated from GGUF temperatures and pinned official estimate/multiple tables.
    io::json::Value calibration;
};

// A main GGUF uses the exact sibling mmproj-<main filename>. A directory selects
// d1-Omni-2B-Q8_0.gguf and mmproj-d1-Omni-2B-Q8_0.gguf, without scanning variants.
std::shared_ptr<const LfmD1OmniAssets> load_lfm_d1_omni_assets(const std::filesystem::path & model_path);
// Existing spec-package entry: model/mmproj are the matching pair.
std::shared_ptr<const LfmD1OmniAssets> load_lfm_d1_omni_assets(assets::ResourceBundle resources);

struct LfmD1OmniModelWeights {
    modules::LfmDecoderWeights backbone;
    core::TensorValue token_embedding;
    modules::LinearWeights output_projection;
};

// Append to the session's existing store; caller uploads once after binding all components.
// Assets and the store must outlive upload and every graph using the returned tensors.
LfmD1OmniModelWeights load_model_weights(const LfmD1OmniAssets & assets, core::BackendWeightStore & store);
modules::Lfm3AudioEncoderWeights load_audio_weights(const LfmD1OmniAssets & assets, core::BackendWeightStore & store);

}  // namespace engine::models::lfm_d1_omni
