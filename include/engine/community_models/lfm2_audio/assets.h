#pragma once

// LFM2.5-Audio assets. The model ships as llama.cpp-format GGUF components in
// one directory: the LFM2 backbone with its text tokenizer, an mmproj file
// with the FastConformer encoder and the audio adapter, and for speech output
// a vocoder file (depthformer) and a tokenizer file (audio detokenizer). The
// session picks each component through a session option, so quantizations can
// be mixed without renaming the published files.
//
// Reference implementation: liquid-audio v1.3.0,
// https://github.com/Liquid4All/liquid-audio/tree/v1.3.0. File paths in the
// lfm2_audio comments are relative to its src/liquid_audio/.

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

// The depthformer in the vocoder GGUF: from the backbone's hidden state it
// predicts the codes of one audio frame, one codebook after another
// (LFM2AudioModel._sample_audio_frame, model/lfm2_audio.py). Its blocks are
// pre-norm GQA attention with QK-norm and interleaved RoPE, then SwiGLU, with
// no final norm (RawLMBackbone(has_embedding=False), model/transformer.py).
struct Lfm2DepthformerConfig {
    int64_t input_size = 0;  // the backbone hidden size
    int64_t hidden_size = 0;
    int64_t num_layers = 0;
    int64_t num_heads = 0;
    int64_t num_kv_heads = 0;
    int64_t head_dim = 0;
    int64_t intermediate_size = 0;
    int64_t codebooks = 0;
    int64_t audio_vocab_size = 0;  // the codes plus end-of-audio, which is the last
    float rms_norm_eps = 1e-5f;
    float rope_theta = 1e6f;

    [[nodiscard]] int32_t end_of_audio() const noexcept { return static_cast<int32_t>(audio_vocab_size - 1); }
};

// The audio detokenizer (detokenizer.py): the mean of the frame's code
// embeddings, repeated `upsample` times, an LFM2 hybrid with causal
// sliding-window attention, and a linear head to log-magnitude and phase for
// an ISTFT.
struct Lfm2DetokenizerConfig {
    Lfm2BackboneConfig lfm;  // vocab_size unused
    int64_t sliding_window = 0;
    int64_t output_size = 0;
    int64_t codebooks = 0;
    int64_t codebook_size = 0;
    int64_t upsample = 0;
    int64_t n_fft = 0;
    int64_t hop_length = 0;
    int sample_rate = 0;
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

// Whether `model_path` (a directory or a GGUF file) holds any LFM2-Audio
// component: an LFM2 backbone or an lfm2a mmproj GGUF, complete or not. The
// loader claims such paths so that load_lfm2_audio_assets can say what is
// missing. Never throws.
bool has_lfm2_audio_component(const std::filesystem::path & model_path);

// Resolves the backbone and mmproj GGUFs. An empty name means the default:
// the only backbone GGUF in the model root, and "mmproj-<backbone file>" or
// else the only mmproj GGUF.
std::shared_ptr<const Lfm2AudioComponents> load_lfm2_audio_components(
    const Lfm2AudioAssets & assets,
    const std::string & model_gguf,
    const std::string & mmproj_gguf);

// The components speech output adds: the vocoder GGUF (depthformer and the
// detokenizer's code embedding) and the detokenizer GGUF. The backbone's
// audio-frame input embedding stays in the mmproj file.
struct Lfm2AudioOutputComponents {
    std::filesystem::path vocoder_path;
    std::filesystem::path detokenizer_path;
    std::shared_ptr<const assets::TensorSource> vocoder;
    std::shared_ptr<const assets::TensorSource> detokenizer;
    Lfm2DepthformerConfig depthformer;
    Lfm2DetokenizerConfig detokenizer_config;
};

// An empty name means "vocoder-<backbone file>" / "tokenizer-<backbone file>",
// or else the only vocoder- / tokenizer- GGUF in the model root.
std::shared_ptr<const Lfm2AudioOutputComponents> load_lfm2_audio_output_components(
    const Lfm2AudioAssets & assets,
    const Lfm2AudioComponents & components,
    const std::string & vocoder_gguf,
    const std::string & detokenizer_gguf);

}  // namespace engine::community_models::lfm2_audio
