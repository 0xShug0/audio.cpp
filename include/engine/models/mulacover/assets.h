#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/codecs/heartcodec_runtime.h"
#include "engine/framework/modules/transformers/decoder.h"

#include <array>

namespace engine::models::mulacover {

struct MuLaCoverConfig {
    modules::DecoderStackConfig backbone, decoder;
    int64_t max_seq_len = 8192;
    int64_t text_vocab_size = 0, audio_vocab_size = 0, audio_num_codebooks = 0, muq_dim = 0;
    int64_t qwen_dim = 0;
    int64_t peft_layer_num = 0;
    bool train_tag = false;
};

struct MuLaCoverGenerationConfig {
    int64_t text_bos_id = 0, text_eos_id = 0, audio_eos_id = 0, empty_id = 0;
    int64_t tag_start_id = 0;
    int64_t tag_end_id = 0;
    int64_t muq_start_id = 0;
    int64_t muq_end_id = 0;
    int64_t qwen_start_id = 0;
    int64_t qwen_end_id = 0;
};

struct MuLaCoverAssets {
    assets::ResourceBundle resources;
    MuLaCoverConfig mula_config;
    MuLaCoverGenerationConfig generation_config;
    // Qwen configuration and tokenizer files live in resources under qwen_* IDs.
    std::shared_ptr<const assets::TensorSource> mula_weights;
    std::shared_ptr<const assets::TensorSource> qwen_weights;
    // Required sources; neural runtimes are created on first audio input.
    std::shared_ptr<const assets::TensorSource> transcriber_weights;
    std::array<std::shared_ptr<const assets::TensorSource>, 5> chord_weights;
    codecs::HeartCodecConfig codec_config;
    std::shared_ptr<const assets::TensorSource> codec_weights;
};

std::shared_ptr<const MuLaCoverAssets> load_mulacover_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::mulacover
