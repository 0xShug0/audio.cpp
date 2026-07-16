#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/modules/whisper_embedding.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::vevo2 {

struct Vevo2ARConfig {
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t max_position_embeddings = 0;
    int64_t max_window_layers = 0;
    int64_t bos_token_id = 151643;
    int64_t eos_token_id = 151645;
    int64_t pad_token_id = 151643;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 1000000.0F;
    bool tie_word_embeddings = true;
    std::string model_type;
    std::vector<int64_t> generation_eos_token_ids;
    int64_t generation_top_k = 25;
    float generation_top_p = 0.8F;
    float generation_temperature = 1.0F;
    float generation_repetition_penalty = 1.1F;
    bool generation_do_sample = true;
};

struct Vevo2CocoTokenizerConfig {
    std::string coco_type;
    int64_t downsample_rate = 0;
    int64_t codebook_size = 0;
    int64_t hidden_size = 0;
    int64_t codebook_dim = 0;
    int64_t vocos_dim = 0;
    int64_t vocos_intermediate_dim = 0;
    int64_t vocos_num_layers = 0;
    bool use_normed_whisper = true;
    int64_t whisper_dim = 0;
    int64_t chromagram_dim = 0;
};

struct Vevo2AcousticPreprocessConfig {
    int64_t sample_rate = 24000;
    int64_t hop_size = 480;
    int64_t n_fft = 1920;
    int64_t num_mels = 128;
    int64_t win_size = 1920;
    float fmin = 0.0F;
    float fmax = 12000.0F;
    float mel_var = 8.14F;
    float mel_mean = -4.92F;
};

struct Vevo2FMConfig {
    Vevo2AcousticPreprocessConfig preprocess;
    int64_t mel_dim = 128;
    int64_t hidden_size = 1024;
    int64_t num_layers = 16;
    int64_t num_heads = 16;
    float cfg_scale = 0.2F;
    bool use_cond_code = true;
    int64_t cond_codebook_size = 16384;
    int64_t cond_scale_factor = 4;
    float sigma = 1.0e-5F;
    std::string time_scheduler = "cos";
    int64_t cond_sample_rate = 16000;
    Vevo2CocoTokenizerConfig coco_content_style;
    bool use_text_as_condition = false;
    int64_t text_cond_codebook_size = 0;
};

struct Vevo2VocoderConfig {
    Vevo2AcousticPreprocessConfig preprocess;
    int64_t input_channels = 128;
    int64_t dim = 1024;
    int64_t intermediate_dim = 4096;
    int64_t num_layers = 30;
    int64_t n_fft = 1920;
    int64_t hop_size = 480;
    std::string padding = "same";
};

struct Vevo2Config {
    Vevo2ARConfig ar;
    Vevo2CocoTokenizerConfig prosody_tokenizer;
    Vevo2CocoTokenizerConfig content_style_tokenizer;
    Vevo2FMConfig fm;
    Vevo2FMConfig fm_text;
    Vevo2VocoderConfig vocoder;
    engine::modules::WhisperEmbeddingConfig whisper;
};

struct Vevo2Assets {
    assets::ResourceBundle resources;
    Vevo2Config config;
    std::shared_ptr<const assets::TensorSource> content_style_tokenizer_weights;
    std::shared_ptr<const assets::TensorSource> prosody_tokenizer_weights;
    std::shared_ptr<const assets::TensorSource> ar_weights;
    std::shared_ptr<const assets::TensorSource> fm_weights;
    std::shared_ptr<const assets::TensorSource> fm_whisper_stats;
    std::shared_ptr<const assets::TensorSource> vocoder_weights_0;
    std::shared_ptr<const assets::TensorSource> whisper_weights;
};

std::shared_ptr<const Vevo2Assets> load_vevo2_assets(
    const std::filesystem::path & model_path,
    const std::optional<std::filesystem::path> & whisper_model_path = std::nullopt);

}  // namespace engine::models::vevo2
