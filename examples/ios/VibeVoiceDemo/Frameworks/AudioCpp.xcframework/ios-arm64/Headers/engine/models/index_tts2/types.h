#pragma once

#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::index_tts2 {

struct IndexTTS2GptConfig {
    int64_t model_dim = 1280;
    int64_t max_mel_tokens = 1815;
    int64_t max_text_tokens = 600;
    int64_t heads = 20;
    bool use_mel_codes_as_input = true;
    int64_t mel_length_compression = 1024;
    int64_t layers = 24;
    int64_t number_text_tokens = 12000;
    int64_t number_mel_codes = 8194;
    int64_t start_mel_token = 8192;
    int64_t stop_mel_token = 8193;
    int64_t start_text_token = 0;
    int64_t stop_text_token = 1;
    bool train_solo_embeddings = false;
    std::string condition_type = "conformer_perceiver";
    int64_t condition_output_size = 512;
    int64_t condition_linear_units = 2048;
    int64_t condition_attention_heads = 8;
    int64_t condition_num_blocks = 6;
    std::string condition_input_layer = "conv2d2";
    int64_t condition_perceiver_mult = 2;
    int64_t emo_condition_output_size = 512;
    int64_t emo_condition_linear_units = 1024;
    int64_t emo_condition_attention_heads = 4;
    int64_t emo_condition_num_blocks = 4;
    std::string emo_condition_input_layer = "conv2d2";
    int64_t emo_condition_perceiver_mult = 2;
};

struct IndexTTS2SemanticCodecConfig {
    int64_t codebook_size = 8192;
    int64_t hidden_size = 1024;
    int64_t codebook_dim = 8;
    int64_t vocos_dim = 384;
    int64_t vocos_intermediate_dim = 2048;
    int64_t vocos_num_layers = 12;
};

struct IndexTTS2S2MelConfig {
    int sample_rate = 22050;
    int64_t n_fft = 1024;
    int64_t win_length = 1024;
    int64_t hop_length = 256;
    int64_t n_mels = 80;
    float fmin = 0.0F;
    std::optional<float> fmax = std::nullopt;
    std::string dit_type = "DiT";
    std::string reg_loss_type = "l1";
    int64_t style_dim = 192;
    int64_t length_regulator_channels = 512;
    bool length_regulator_is_discrete = false;
    int64_t length_regulator_in_channels = 1024;
    int64_t length_regulator_content_codebook_size = 2048;
    std::vector<int64_t> length_regulator_sampling_ratios;
    bool length_regulator_vector_quantize = false;
    int64_t length_regulator_n_codebooks = 1;
    float length_regulator_quantizer_dropout = 0.0F;
    bool length_regulator_f0_condition = false;
    int64_t length_regulator_n_f0_bins = 512;
    int64_t dit_hidden_dim = 512;
    int64_t dit_num_heads = 8;
    int64_t dit_depth = 13;
    float dit_class_dropout_prob = 0.1F;
    int64_t dit_block_size = 8192;
    int64_t dit_in_channels = 80;
    bool dit_style_condition = true;
    std::string dit_final_layer_type = "wavenet";
    std::string dit_target = "mel";
    int64_t dit_content_dim = 512;
    int64_t dit_content_codebook_size = 1024;
    std::string dit_content_type = "discrete";
    bool dit_f0_condition = false;
    int64_t dit_n_f0_bins = 512;
    int64_t dit_content_codebooks = 1;
    bool dit_is_causal = false;
    bool dit_long_skip_connection = true;
    bool dit_zero_prompt_speech_token = false;
    bool dit_time_as_token = false;
    bool dit_style_as_token = false;
    bool dit_uvit_skip_connection = true;
    bool dit_add_resblock_in_transformer = false;
    int64_t wavenet_hidden_dim = 512;
    int64_t wavenet_num_layers = 8;
    int64_t wavenet_kernel_size = 5;
    int64_t wavenet_dilation_rate = 1;
    float wavenet_dropout = 0.2F;
    bool wavenet_style_condition = true;
};

struct IndexTTS2Config {
    std::string version = "2.0";
    int dataset_sample_rate = 24000;
    bool dataset_squeeze = false;
    int dataset_mel_sample_rate = 24000;
    int64_t dataset_mel_n_fft = 1024;
    int64_t dataset_mel_hop_length = 256;
    int64_t dataset_mel_win_length = 1024;
    int64_t dataset_mel_n_mels = 100;
    float dataset_mel_fmin = 0.0F;
    bool dataset_mel_normalize = false;
    IndexTTS2GptConfig gpt;
    IndexTTS2SemanticCodecConfig semantic_codec;
    IndexTTS2S2MelConfig s2mel;
    std::vector<int64_t> emo_num;
};

struct IndexTTS2GenerationOptions {
    bool do_sample = true;
    float top_p = 0.8F;
    int top_k = 30;
    float temperature = 0.8F;
    float length_penalty = 0.0F;
    int num_beams = 3;
    float repetition_penalty = 10.0F;
    int max_mel_tokens = 1500;
    uint32_t seed = 0;
};

struct IndexTTS2Request {
    std::string text;
    std::optional<runtime::AudioBuffer> speaker_audio = std::nullopt;
    std::optional<runtime::AudioBuffer> emotion_audio = std::nullopt;
    float emotion_alpha = 1.0F;
    std::optional<std::vector<float>> emotion_vector = std::nullopt;
    bool use_emotion_text = false;
    std::optional<std::string> emotion_text = std::nullopt;
    bool use_random_emotion = false;
    int interval_silence_ms = 200;
    int max_text_tokens_per_segment = 120;
    IndexTTS2GenerationOptions generation;
};

}  // namespace engine::models::index_tts2
