#pragma once

// LFM3-Audio assets. The model is one checkpoint: the backbone with its ASR
// encoder and MaskGIT depth decoder ("model" tensors, HF names such as
// lfm.model.layers.0...), and the LiquidCodec decoder ("codec" tensors). It
// loads from the upstream safetensors directory, with config.json,
// generation_config.json, tokenizer.json and audio_tokenizer/config.json, or
// from Liquid's published GGUFs (published_gguf.h), which give the same.

#include "engine/community_models/lfm3_audio/blocks.h"
#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace engine::community_models::lfm3_audio {

struct Lfm3TokenSpecials {         // from generation_config.json
    std::vector<int32_t> eos_ids;  // [124900, 125014]
};

struct Lfm3BackboneConfig {  // lfm.* ; vocab 128000, tied head
    blocks::StackConfig stack;
    int64_t vocab_size = 0;
};

struct Lfm3EncoderConfig {
    int64_t n_mels = 128, conv_channels = 256, mel_after_downsample = 16;  // 128/8
    blocks::StackConfig stack;  // LFM2 names, 17 layers, conv k3, all attention = sliding(32), theta 1e6
    int64_t output_size = 2048;
};

struct Lfm3DepthConfig {
    blocks::StackConfig stack;  // 3 layers, bidirectional, gate, no sandwich, theta 1e6, hd 32
    int64_t input_size = 2048, codebooks = 12, audio_vocab_size = 2017;
    std::vector<int64_t> maskgit_schedule;  // [4,4,4] (config maskgit_steps=3)
    [[nodiscard]] int32_t end_of_audio() const noexcept { return static_cast<int32_t>(audio_vocab_size - 1); }
};

struct Lfm3CodecConfig {
    blocks::StackConfig stack;  // dims from tensor shapes: the upstream python hard-codes them
    std::vector<int64_t> fsq_levels;  // [8,7,6,6]
    int64_t codebooks = 12, quantizer_dim = 256, upsample = 6, n_fft = 1280, hop = 320;
    int sample_rate = 24000;
};

// generation_config.json audio_* (the published GGUF's clip.gen.audio.sampling.*): the speech sampling of
// TTS and S2S when a request sets none; the model card's settings when the file has none.
struct Lfm3SpeechSampling {
    float temperature = 1.0f;
    int64_t top_k = 128;
    float repetition_penalty = 1.1f;
    int64_t repetition_window = 4096;  // frames of history per codebook
};

struct Lfm3GenerationConfig {
    int64_t interleaved_text = 6, interleaved_audio = 12;
    Lfm3TokenSpecials specials;
    Lfm3SpeechSampling speech;
};

struct Lfm3AudioAssets {
    assets::ResourceBundle resources;  // required by SpecBackedVoiceModelLoader::inspect
    Lfm3BackboneConfig backbone;
    Lfm3EncoderConfig encoder;
    Lfm3DepthConfig depth;
    Lfm3CodecConfig codec;
    Lfm3GenerationConfig generation;
    std::shared_ptr<const assets::TensorSource> model;
    std::shared_ptr<const assets::TensorSource> codec_weights;  // resource id "codec" (decoder.*, quantizer.*, from_quantizer.*)
    std::filesystem::path tokenizer_json;
    // Published GGUFs: no resources or tokenizer.json, but the backbone's vocabulary and its path.
    tokenizers::LlamaBpeGgufVocabulary vocabulary;
    std::filesystem::path published_backbone;
};

std::shared_ptr<const Lfm3AudioAssets> load_lfm3_audio_assets(const std::filesystem::path & model_path);

}  // namespace engine::community_models::lfm3_audio
