#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/community_models/vieneu_v3_turbo/types.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace engine::models::vieneu_v3_turbo {

struct VieNeuTTSTalkerConfig {
    int64_t max_position_embeddings = 32768;
    int64_t hidden_size = 0;
    int64_t text_hidden_size = 0;
    int64_t text_vocab_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t num_code_groups = 0;
    int64_t vocab_size = 0;
    int64_t codec_bos_id = 0;
    int64_t codec_eos_token_id = 0;
    int64_t codec_think_id = 0;
    int64_t codec_nothink_id = 0;
    int64_t codec_pad_id = 0;
    int64_t codec_think_bos_id = 0;
    int64_t codec_think_eos_id = 0;
    std::unordered_map<std::string, int64_t> codec_language_id;
    std::unordered_map<std::string, int64_t> speaker_id;
    std::unordered_map<std::string, std::optional<std::string>> speaker_dialect;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
};

struct VieNeuTTSCodePredictorConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t vocab_size = 0;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
};

struct VieNeuTTSSpeechTokenizerConfig {
    std::string model_type;
    int input_sample_rate = 0;
    int output_sample_rate = 0;
    int64_t num_quantizers = 0;
    int64_t codebook_size = 0;
    int64_t semantic_codebook_size = 0;
};

struct VieNeuTTSSpeakerEncoderConfig {
    int64_t embedding_dim = 0;
    int sample_rate = 0;
};

struct VieNeuTTSConfig {
    VieNeuTTSVariant variant = VieNeuTTSVariant::Base;
    std::string tts_model_type;
    std::string tts_model_size;
    std::string tokenizer_type;
    int64_t max_new_tokens = 300;
    VieNeuTTSTalkerConfig talker;
    VieNeuTTSCodePredictorConfig code_predictor;
    VieNeuTTSSpeechTokenizerConfig speech_tokenizer;
    VieNeuTTSSpeakerEncoderConfig speaker_encoder;
    int64_t tts_bos_token_id = 0;
    int64_t tts_eos_token_id = 0;
    int64_t tts_pad_token_id = 0;
    int64_t text_prompt_start_token_id = 3;
    int64_t text_prompt_end_token_id = 4;
    int64_t audio_ref_slot_token_id = 7;
    int64_t default_style_token_id = 16;
    int64_t audio_pad_token_id = 1024;
    int64_t speech_generation_start_token_id = 5;
    bool has_speaker_encoder = false;

    // VieNeu-TTS specific extensions
    bool is_vieneu = true;
    int64_t local_num_hidden_layers = 1;
    int64_t local_num_attention_heads = 8;
    int64_t local_intermediate_size = 2048;
    bool use_speaker_embedding = false;
    int64_t speaker_embedding_dim = 192;
};

struct VieNeuTTSAssets {
    assets::ResourceBundle resources;
    VieNeuTTSConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> speech_tokenizer_weights;
};

// Family name and the legacy alias it was first published under. Packages whose
// embedded model spec still says `vietneu_tts` (the original community GGUF) keep
// loading through the alias.
inline constexpr const char * kFamily = "vieneu_v3_turbo";
inline constexpr const char * kLegacyFamily = "vietneu_tts";

// Resolves the package spec for this family, falling back to the legacy alias when the
// package (a GGUF with an embedded spec) was built under the old name.
std::filesystem::path resolve_package_spec_path();

std::shared_ptr<const VieNeuTTSAssets> load_vieneu_v3_turbo_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::vieneu_v3_turbo
