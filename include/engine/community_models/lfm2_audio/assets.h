#pragma once

// LFM2.5-Audio assets. The model ships as llama.cpp-format GGUF components in
// one directory: the LFM2 backbone with its text tokenizer, and an mmproj file
// with the FastConformer encoder and the audio adapter. The session picks each
// component through a session option, so quantizations can be mixed without
// renaming the published files.
//
// Reference implementation: https://github.com/Liquid4All/liquid-audio

#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::lfm2_audio {

// LFM2 hybrid backbone. A layer with zero KV heads is a gated short-conv
// block; the others are GQA attention blocks.
struct Lfm2BackboneConfig {
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_attention_heads = 0;
    int64_t head_dim = 0;
    int64_t conv_kernel_size = 0;
    int64_t context_length = 0;
    std::vector<int64_t> kv_heads;  // one entry per layer
    float rms_norm_eps = 1e-5f;
    float rope_theta = 1e6f;

    [[nodiscard]] int64_t num_layers() const noexcept { return static_cast<int64_t>(kv_heads.size()); }
    [[nodiscard]] bool is_attention_layer(int64_t layer) const { return kv_heads.at(static_cast<size_t>(layer)) > 0; }
};

// NeMo FastConformer encoder (dw_striding x8) followed by the audio adapter
// MLP that maps encoder frames to backbone embeddings.
struct Lfm2FastConformerEncoderConfig {
    int64_t n_mels = 0;
    int64_t hidden_size = 0;
    int64_t num_layers = 0;
    int64_t num_heads = 0;
    int64_t intermediate_size = 0;
    int64_t conv_kernel_size = 0;
    int64_t subsampling_channels = 0;
    int64_t adapter_hidden_size = 0;
    int64_t output_size = 0;
    float layer_norm_eps = 1e-5f;
};

struct Lfm2TextVocabulary {
    std::vector<std::string> tokens;
    std::vector<std::string> merges;
    std::vector<int32_t> token_types;
    std::string pre_tokenizer;
};

// The model root and whatever the loader can check without choosing a
// component: every file the session could select lives under model_root.
struct Lfm2AudioAssets {
    std::filesystem::path model_root;
    // Set when --model named a backbone GGUF file instead of a directory.
    std::string default_model_gguf;
};

// The components chosen for one session.
struct Lfm2AudioComponents {
    std::filesystem::path model_path;
    std::filesystem::path mmproj_path;
    std::shared_ptr<const assets::TensorSource> model;
    std::shared_ptr<const assets::TensorSource> mmproj;
    Lfm2BackboneConfig backbone;
    Lfm2FastConformerEncoderConfig encoder;
    Lfm2TextVocabulary vocabulary;
    std::vector<std::string> languages;
};

std::shared_ptr<const Lfm2AudioAssets> load_lfm2_audio_assets(const std::filesystem::path & model_path);

// Resolves the backbone and mmproj GGUFs. An empty name means the default:
// the only backbone GGUF in the model root, and "mmproj-<backbone file>".
std::shared_ptr<const Lfm2AudioComponents> load_lfm2_audio_components(
    const Lfm2AudioAssets & assets,
    const std::string & model_gguf,
    const std::string & mmproj_gguf);

}  // namespace engine::community_models::lfm2_audio
