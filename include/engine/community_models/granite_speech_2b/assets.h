#pragma once

// granite_speech_2b (IBM Speech 5.0 2B autoregressive) assets: typed checkpoint
// config, resource bundle, tensor inventory, and early validation.
//
// B1 scope: parse + validate only. No graphs are built here; the encoder,
// projector, and parity probe bind in B2, the shared decoder in C, and the
// loader/session/spec/GGUF wiring in D/E.
//
// B1 -> B2 interface (stable surface B2 builds on):
//   - granite_speech_2b_can_load(path): metadata-only probe. Reads
//     config.json from the model directory; never opens tensor data. Returns
//     true only for `model_type == "granite_speech"` checkpoints carrying the
//     conditional-generation triple (granite_speech_encoder + blip_2_qformer +
//     granite text). TurboCTC (`granite_speech5_ctc`/`granite5asr`) and
//     `granite_speech_nar` snapshots return false. A family hint may filter
//     probes but never substitutes for this checkpoint identity check.
//   - load_granite_speech_2b_assets(path): full typed load. Keeps the logical
//     tensor entry name `weights` (model:model.safetensors native root) so the
//     Task D spec binding stays identical. Retains the reopenable TensorSource
//     in GraniteSpeech2BAssets::source; B2 calls source->release_storage()
//     only after its graph uploads finish.
//   - check_granite_speech_2b_tensor_inventory(source): asserts 40 text / 16
//     encoder / 2 Q-Former layers plus both CTC feedback projections
//     (encoder.out, encoder.out_mid). B2/C call this before binding.
//   - Error type: std::runtime_error with actual-vs-expected detail. All
//     feature tensors below are time-major rows: mel is [frames, 80],
//     stacked is [steps, 160] (see frontend.h).
//
// Sidecar decision (recorded): the checkpoint ships vocab.json, merges.txt,
// and added_tokens.json, but all three are content-identical to what
// tokenizer.json already embeds (verified: 100352 vocab entries, 100000
// merges, <|audio|> id 100352 in both). Only tokenizer.json is required.

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::community_models::granite_speech_2b {

namespace assets = engine::assets;

inline constexpr const char * kGraniteSpeech2BFamily = "granite_speech_2b";
inline constexpr const char * kGraniteSpeech2BModelType = "granite_speech";
inline constexpr const char * kGraniteSpeech2BRevision = "ec49ca6b1a5e7e9176127e09cb303ab78cbacdf7";

struct GraniteSpeech2BFrontendConfig {
    int64_t sample_rate = 16000;
    int64_t n_fft = 512;
    int64_t win_length = 400;
    int64_t hop_length = 160;
    int64_t n_mels = 80;
};

struct GraniteSpeech2BEncoderConfig {
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

struct GraniteSpeech2BProjectorConfig {
    int64_t window_size = 15;
    int64_t downsample_rate = 5;
    int64_t hidden_size = 1024;
    int64_t num_hidden_layers = 2;
    int64_t num_attention_heads = 16;
    int64_t intermediate_size = 4096;
};

struct GraniteSpeech2BTextConfig {
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

struct GraniteSpeech2BConfig {
    std::string model_type = kGraniteSpeech2BModelType;
    bool has_lora_adapter = false;
    GraniteSpeech2BFrontendConfig frontend;
    GraniteSpeech2BEncoderConfig encoder;
    GraniteSpeech2BProjectorConfig projector;
    GraniteSpeech2BTextConfig text;
};

struct GraniteSpeech2BAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> source;
    GraniteSpeech2BConfig config;
};

// Metadata-only probe: inspects config.json sidecar text, never tensor data.
bool granite_speech_2b_can_load(const std::filesystem::path & model_path);

// Full typed load: builds the manual resource bundle (native `model:.`
// roots, tensor entry `weights` -> model:model.safetensors), parses the
// model-local configs, and runs the tensor inventory check.
std::shared_ptr<const GraniteSpeech2BAssets> load_granite_speech_2b_assets(
    const std::filesystem::path & model_path);

// Inventory gate used by B2/C before binding: asserts exactly 40 text, 16
// encoder, and 2 Q-Former layers plus both CTC feedback projections, and
// validates the brief-pinned shapes of the encoder/projector/text bindings.
void check_granite_speech_2b_tensor_inventory(const assets::TensorSource & source);

}  // namespace engine::community_models::granite_speech_2b
