#pragma once

#include "engine/framework/assets/resource_bundle.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::moss_voicegen {

// Qwen3 backbone geometry, read from the checkpoint's "language_config" block.
struct MossVoiceGenBackboneConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t max_position_embeddings = 0;
    int64_t vocab_size = 0;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
    bool tie_word_embeddings = true;
};

struct MossVoiceGenConfig {
    MossVoiceGenBackboneConfig backbone;
    // MOSS-VoiceGenerator emits 1 + n_vq ids per step: one text id and one code per
    // codebook. n_vq is 16 here, half of what the moss_tts_delay family allows.
    int64_t num_codebooks = 0;
    int64_t audio_vocab_size = 0;
    int64_t audio_pad_code = 0;
    // The checkpoint's config.json omits these; MossTTSDelayConfig's defaults apply and
    // match what the tokenizer resolves for <|im_start|>/<|im_end|>.
    int64_t pad_token_id = 151643;
    int64_t im_start_token_id = 151644;
    int64_t im_end_token_id = 151645;
    int64_t audio_start_token_id = 0;
    int64_t audio_end_token_id = 0;
    int64_t audio_user_slot_token_id = 0;
    int64_t audio_assistant_gen_slot_token_id = 0;
    int64_t audio_assistant_delay_slot_token_id = 0;
    int64_t sampling_rate = 0;
};

struct MossVoiceGenAssets {
    assets::ResourceBundle resources;
    MossVoiceGenConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> audio_tokenizer_weights;
};

std::shared_ptr<const MossVoiceGenAssets> load_moss_voicegen_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::moss_voicegen
