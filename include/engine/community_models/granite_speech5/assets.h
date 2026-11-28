#pragma once

// Checkpoint configuration, resource loading and tensor-shape validation.
// Both native checkpoints and GGUF packages use the framework TensorSource.
// Only tokenizer.json is needed; it already contains the vocabulary and merges.

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::granite_speech5 {

namespace assets = engine::assets;

inline constexpr const char * kGraniteSpeech5Family = "granite_speech5";
inline constexpr const char * kGraniteSpeech5ModelType = "granite_speech";
inline constexpr const char * kGraniteSpeech5Revision = "ec49ca6b1a5e7e9176127e09cb303ab78cbacdf7";

struct GraniteSpeech5FrontendConfig {
    int64_t sample_rate = 16000;
    int64_t n_fft = 512;
    int64_t win_length = 400;
    int64_t hop_length = 160;
    int64_t n_mels = 80;
};

struct GraniteSpeech5EncoderConfig {
    int64_t input_dim = 160;
    int64_t hidden_dim = 1024;
    int64_t num_layers = 16;
    int64_t num_heads = 8;
    int64_t dim_head = 128;
    int64_t context_size = 200;
    int64_t output_dim = 348;
    int64_t conv_expansion_factor = 2;
    int64_t conv_kernel_size = 15;
    int64_t max_pos_emb = 512;
    int64_t feedforward_mult = 4;
};

struct GraniteSpeech5ProjectorConfig {
    int64_t window_size = 15;
    int64_t downsample_rate = 5;
    int64_t hidden_size = 1024;
    int64_t num_hidden_layers = 2;
    int64_t num_attention_heads = 16;
    int64_t intermediate_size = 4096;
};

struct GraniteSpeech5TextConfig {
    int64_t num_hidden_layers = 40;
    int64_t hidden_size = 2048;
    int64_t num_attention_heads = 16;
    int64_t num_key_value_heads = 4;
    int64_t intermediate_size = 4096;
    int64_t vocab_size = 100353;
    int64_t bos_token_id = 100257;
    int64_t eos_token_id = 100257;
    int64_t pad_token_id = 100256;
    int64_t audio_token_id = 100352;
    int64_t embedding_multiplier = 12;
    float attention_multiplier = 0.0078125F;
    int64_t logits_scaling = 8;
    float residual_multiplier = 0.22F;
    int64_t rope_theta = 10000;
    float rope_factor = 4.0F;
    int64_t rope_original_max_position_embeddings = 4096;
    std::string rope_type = "yarn";
    int64_t max_position_embeddings = 16384;
};

struct GraniteSpeech5Config {
    std::string model_type = kGraniteSpeech5ModelType;
    bool has_lora_adapter = false;
    GraniteSpeech5FrontendConfig frontend;
    GraniteSpeech5EncoderConfig encoder;
    GraniteSpeech5ProjectorConfig projector;
    GraniteSpeech5TextConfig text;
};

struct GraniteSpeech5Assets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> source;
    GraniteSpeech5Config config;
};

// Metadata-only probe: inspects config.json sidecar text, never tensor data.
bool granite_speech5_can_load(const std::filesystem::path & model_path);

// True when a directory holds native weight files (top-level *.safetensors /
// *.bin). GGUF package directories never do: their weights live inside the
// single package GGUF. Used to keep the installed-package fallback from
// claiming a foreign native checkpoint directory that merely sits next to a
// stray GGUF.
bool granite_speech5_directory_has_native_weights(const std::filesystem::path & dir);

// Full typed load: native checkpoint directories build the manual resource
// bundle (native `model:.` roots, tensor entry `weights` ->
// model:model.safetensors),. GGUF packages (a standalone
// .gguf file, or a directory holding the single package GGUF instead of a
// native checkpoint) resolve through the spec-backed bundle, which validates
// the embedded model spec family and materializes the 8 sidecars; both paths
// then parse the model-local configs and run the tensor inventory check.
// Tensor storage stays caller-selected (native keeps the GGUF mixed types).
std::shared_ptr<const GraniteSpeech5Assets> load_granite_speech5_assets(
    const std::filesystem::path & model_path);

// Inventory gate used before weight binding before binding: asserts exactly 40 text, 16
// encoder, and 2 Q-Former layers plus both CTC feedback projections, and
// validates the brief-pinned shapes of the encoder/projector/text bindings.
void check_granite_speech5_encoder_weight_shapes(const assets::TensorSource & source);
void check_granite_speech5_projector_weight_shapes(const assets::TensorSource & source);
void check_granite_speech5_tensor_inventory(const assets::TensorSource & source);

}  // namespace engine::community_models::granite_speech5
