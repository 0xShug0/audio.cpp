#pragma once

// Liquid's published LFM3-Audio GGUFs (LiquidAI/LFM3-Audio-2B-GGUF), in
// llama.cpp's layout: a backbone <name>.gguf (general.architecture lfm3) and
// mmproj-<name>.gguf (clip.audio.projector_type lfm3a) with the encoder, the
// MaskGIT depth decoder (a.gen.code.*) and the LiquidCodec decoder
// (a.gen.wav.*). They load as the checkpoint does: the tensors under its HF
// names and shapes (encoder.downsample.lin_out.weight without its unit axis),
// and the config.json, generation_config.json and codec config.json that
// llama.cpp's converter turned into metadata or constants.

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/json.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <filesystem>
#include <memory>

namespace engine::community_models::lfm3_audio {

struct Lfm3PublishedGguf {
    std::filesystem::path backbone_path;
    std::filesystem::path mmproj_path;
    std::shared_ptr<const assets::TensorSource> model;  // HF names of model.safetensors
    std::shared_ptr<const assets::TensorSource> codec;  // HF names of audio_tokenizer/model.safetensors
    io::json::Value config;
    io::json::Value generation_config;
    io::json::Value codec_config;
    tokenizers::LlamaBpeGgufVocabulary vocabulary;
};

// A backbone GGUF, or a directory holding one, whose directory also holds an
// lfm3a mmproj GGUF; an unreadable file is not one.
[[nodiscard]] bool is_lfm3_audio_published_gguf(const std::filesystem::path & model_path) noexcept;

// A directory must hold one backbone; its mmproj is mmproj-<backbone file>,
// else the Q8_0 one Liquid pairs with the 4-bit backbones, else the only one.
[[nodiscard]] Lfm3PublishedGguf open_lfm3_audio_published_gguf(const std::filesystem::path & model_path);

}  // namespace engine::community_models::lfm3_audio
