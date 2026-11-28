#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/codecs/oobleck_audio_vae_runtime.h"
#include "engine/framework/modules/text_encoders/t5_encoder_runtime.h"
#include "engine/framework/modules/transformers/dit.h"

namespace engine::models::foundation_1 {

struct FoundationAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> weights, text_weights;
    modules::T5BaseEncoderConfig text_config;
    modules::DitConfig dit_config;
    modules::DitWeightNames dit_names;
    codecs::OobleckAudioVaeConfig codec_config;
    int64_t max_text_tokens = 128, sample_rate = 44100, sample_size = 882000;
};

std::shared_ptr<const FoundationAssets> load_foundation_assets(const std::filesystem::path & model_path);

modules::T5EncoderRuntimeWeights load_foundation_text_weights(
    std::shared_ptr<const FoundationAssets> assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType storage_type);

struct NumberConditioningWeights {
    std::shared_ptr<core::BackendWeightStore> storage;
    modules::LinearWeights start, duration;
    std::vector<float> duration_frequencies;
};
NumberConditioningWeights load_foundation_number_weights(
    const FoundationAssets & assets, core::ExecutionContext & execution);

} // namespace engine::models::foundation_1
