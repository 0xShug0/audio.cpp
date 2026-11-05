#pragma once

// HF and GGUF checkpoints bind to the same component configurations and weights.

#include "engine/community_models/lfm3_audio/blocks.h"
#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/tokenizers/llama_bpe.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"

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
    std::shared_ptr<const assets::TensorSource> audio_weights;  // GGUF mmproj; HF components use model
    std::shared_ptr<const assets::TensorSource> codec_weights;  // HF codec checkpoint or GGUF mmproj
    std::filesystem::path tokenizer_json;
    // Published GGUFs: no resources or tokenizer.json, but the backbone's vocabulary and its path.
    tokenizers::LlamaBpeGgufVocabulary vocabulary;
    std::filesystem::path published_backbone;
};

struct Lfm3BackboneWeights {
    Lfm3BackboneWeights();
    ~Lfm3BackboneWeights();
    Lfm3BackboneWeights(Lfm3BackboneWeights &&) noexcept;

    std::unique_ptr<WeightStores> stores;
    blocks::StackWeights stack;
    core::TensorValue token_lookup;  // embed_tokens, or an F16 copy where the backend cannot gather its type
    core::TensorValue text_head;     // embed_tokens (tied), or a CPU-repacked copy of it
    core::TensorValue frame_lookup;  // codebook_embedding [codebooks * audio_vocab, hidden]
};

struct Lfm3EncoderWeights {
    Lfm3EncoderWeights();
    ~Lfm3EncoderWeights();
    Lfm3EncoderWeights(Lfm3EncoderWeights &&) noexcept;

    std::unique_ptr<WeightStores> stores;
    modules::Conv2dWeights conv_in, conv_mid, lin_mid, conv_end, lin_end;
    modules::LinearWeights lin_out;
    blocks::StackWeights stack;
    modules::NormWeights projector_norm;
    modules::LinearWeights projector_1, projector_2;
};

struct Lfm3DepthWeights {
    Lfm3DepthWeights();
    ~Lfm3DepthWeights();
    Lfm3DepthWeights(Lfm3DepthWeights &&) noexcept;

    std::unique_ptr<WeightStores> stores;
    core::TensorValue input_weight;  // depth_linear [codebooks * hidden, input_size]
    core::TensorValue input_bias;    // [codebooks * hidden]
    core::TensorValue mask_rows;     // mask_embedding repeated per codebook, F32 [codebooks, hidden]
    core::TensorValue embeddings;    // depth_embeddings.{i}.embedding packed, [codebooks * vocab, hidden]
    core::TensorValue norms;         // depth_embeddings.{i}.embedding_norm packed, F32 [codebooks, hidden]
    blocks::StackWeights stack;
};

struct Lfm3CodecWeights {
    Lfm3CodecWeights();
    ~Lfm3CodecWeights();
    Lfm3CodecWeights(Lfm3CodecWeights &&) noexcept;

    std::unique_ptr<WeightStores> stores;
    // project_out stays F32: its 48 inputs are not a multiple of the 32-wide blocks of BF16/F16 GPU kernels.
    core::TensorValue project_out, project_out_bias, from_quantizer, from_quantizer_bias, lin, lin_bias;
    blocks::StackWeights stack;
    std::vector<float> window;
};

Lfm3BackboneWeights load_lfm3_backbone_weights(const Lfm3AudioAssets & assets, core::ExecutionContext & execution,
    const Lfm3WeightOptions & options);
Lfm3EncoderWeights load_lfm3_encoder_weights(const Lfm3AudioAssets & assets, core::ExecutionContext & execution,
    const Lfm3WeightOptions & options);
Lfm3DepthWeights load_lfm3_depth_weights(const Lfm3AudioAssets & assets, core::ExecutionContext & execution,
    const Lfm3WeightOptions & options);
Lfm3CodecWeights load_lfm3_codec_weights(const Lfm3AudioAssets & assets, core::ExecutionContext & execution,
    const Lfm3WeightOptions & options);

// Liquid's backbone GGUF paired with its audio mmproj.
[[nodiscard]] bool is_lfm3_audio_gguf(const std::filesystem::path & model_path) noexcept;
[[nodiscard]] std::shared_ptr<const Lfm3AudioAssets> load_lfm3_audio_gguf_assets(const std::filesystem::path & model_path);

std::shared_ptr<const Lfm3AudioAssets> load_lfm3_audio_assets(const std::filesystem::path & model_path);

}  // namespace engine::community_models::lfm3_audio
