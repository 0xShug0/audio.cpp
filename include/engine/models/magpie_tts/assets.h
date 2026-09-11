#pragma once

#include "engine/framework/assets/resource_bundle.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::assets {
class TensorSource;
}

namespace engine::models::magpie_tts {

struct MagpieTTSConfig {
    int64_t text_vocab_size = 0;
    int64_t embedding_dim = 0;
    int64_t hidden_dim = 0;
    int64_t decoder_layers = 0;
    int64_t decoder_heads = 0;
    int64_t decoder_ffn_dim = 0;
    int64_t decoder_max_length = 0;
    int64_t decoder_cross_heads = 0;
    int64_t decoder_cross_head_dim = 0;
    int64_t context_length = 0;
    int64_t context_dim = 0;
    int64_t speakers = 0;
    int64_t audio_codebooks = 0;
    int64_t frame_stacking_factor = 0;
    int64_t codebook_size = 0;
    int64_t all_tokens_per_codebook = 0;
    int64_t local_layers = 0;
    int64_t local_heads = 0;
    int64_t local_hidden_dim = 0;
    int64_t local_ffn_dim = 0;
    int64_t local_context = 0;
    int64_t max_decoder_steps = 500;
    int64_t min_generated_frames = 4;
    int64_t short_sentence_threshold = 35;
    int64_t near_end_threshold = 3;
    int64_t finished_limit_first_chunk = 1;
    int64_t finished_limit_with_eot = 1;
    int64_t finished_limit_without_eot = 1;
    int64_t forceful_chunk_end_threshold = 1;
    int64_t history_len_heuristic = 1;
    int64_t attention_sink_threshold = 4;
    int64_t chunked_attention_sink_threshold = 3;
    int64_t attention_prior_lookahead_window = 6;
    int64_t start_prior_after_n_audio_steps = 0;
    float temperature = 0.6F;
    float argmax_temperature = 0.01F;
    int64_t top_k = 80;
    float guidance_scale = 2.5F;
    float attention_prior_epsilon = 0.1F;
    bool apply_attention_prior = true;
    bool ignore_finished_sentence_tracking = true;
    std::vector<int64_t> apply_prior_to_layers;
    std::vector<int64_t> estimate_alignment_from_layers;
    std::vector<float> prior_weights;
    std::vector<float> prior_weights_init;
    int64_t sample_rate = 22050;
    int64_t samples_per_frame = 1024;
    int64_t codec_input_dim = 32;
    int64_t codec_base_channels = 864;
    std::vector<int64_t> codec_upsample_rates = {8, 8, 4, 2, 2};
    std::vector<int64_t> codec_resblock_kernel_sizes = {3, 7, 11};
    std::vector<int64_t> codec_resblock_dilation_sizes = {1, 3, 5};
    std::vector<int32_t> codec_fsq_num_levels;
    std::vector<int32_t> codec_fsq_dim_base_index;
    std::vector<std::string> speaker_names;
};

struct MagpieTTSAssets {
    assets::ResourceBundle resources;
    MagpieTTSConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> codec_weights;
};

std::shared_ptr<const MagpieTTSAssets> load_magpie_tts_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::magpie_tts
