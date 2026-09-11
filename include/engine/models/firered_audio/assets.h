#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace engine::models::firered_audio {

struct FireRedAudioBackboneConfig {
    int64_t vocab_size = 248096;
    int64_t hidden_size = 4096;
    int64_t intermediate_size = 12288;
    int64_t layers = 32;
    int64_t heads = 16;
    int64_t kv_heads = 4;
    int64_t head_dim = 256;
    int64_t full_attention_interval = 4;
    int64_t linear_conv_kernel_dim = 4;
    int64_t linear_key_head_dim = 128;
    int64_t linear_num_key_heads = 16;
    int64_t linear_num_value_heads = 32;
    int64_t linear_value_head_dim = 128;
    float rms_norm_eps = 1.0e-6F;
    float rope_theta = 10000000.0F;
    float partial_rotary_factor = 0.25F;
};

struct FireRedAudioPatchEncoderConfig {
    int64_t vae_dim = 64;
    int64_t out_dim = 4096;
    int64_t patch_size = 4;
    int64_t hidden_size = 1024;
    int64_t intermediate_size = 4096;
    int64_t layers = 8;
    int64_t heads = 16;
};

struct FireRedAudioFlowConfig {
    int64_t vae_channels = 64;
    int64_t backbone_hidden_size = 4096;
    int64_t hidden_size = 1024;
    int64_t intermediate_size = 4096;
    int64_t layers = 11;
    int64_t heads = 16;
    int64_t patch_size = 4;
    int64_t history_patches = 2;
};

struct FireRedAudioAudioEncoderConfig {
    int64_t sample_rate = 16000;
    int64_t num_mel_bins = 128;
    int64_t n_fft = 400;
    int64_t hop_length = 160;
    int64_t d_model = 1280;
    int64_t encoder_layers = 32;
    int64_t encoder_attention_heads = 20;
    int64_t encoder_ffn_dim = 5120;
    int64_t max_source_positions = 1500;
    int64_t n_window = 1500;
    int64_t output_dim = 4096;
    float layer_norm_eps = 1.0e-5F;
};

struct FireRedAudioRedAeConfig {
    int64_t sample_rate = 24000;
    int64_t audio_patch_size = 480;
    int64_t bottleneck_dim = 64;
    int64_t enc_hidden_size = 896;
    int64_t enc_intermediate_size = 3584;
    int64_t enc_layers = 18;
    int64_t enc_heads = 14;
    int64_t enc_kv_heads = 2;
    int64_t enc_head_dim = 128;
    int64_t enc_sliding_window = 0;
    int64_t enc_extra_downsample_rate = 2;
    int64_t enc_downsample_layers = 4;
    int64_t dec_hidden_size = 896;
    int64_t dec_intermediate_size = 3584;
    int64_t dec_layers = 18;
    int64_t dec_heads = 14;
    int64_t dec_kv_heads = 2;
    int64_t dec_head_dim = 128;
    int64_t dec_sliding_window = 0;
};

struct FireRedAudioSpecialTokens {
    int32_t sosp = 248077;
    int32_t eosp = 248078;
    int32_t audio = 248091;
    int32_t audio_no_latent = 248092;
    int32_t im_end = 248046;
    int32_t endoftext = 248000;
    int32_t sot = 248083;
    int32_t eot = 248084;
};

struct FireRedAudioAssets {
    std::filesystem::path model_root;
    std::filesystem::path gguf_path;
    engine::assets::ResourceBundle resources;
    FireRedAudioBackboneConfig backbone;
    FireRedAudioAudioEncoderConfig audio_encoder;
    FireRedAudioPatchEncoderConfig patch_encoder;
    FireRedAudioFlowConfig flow;
    FireRedAudioRedAeConfig redae;
    FireRedAudioSpecialTokens special_tokens;
    std::shared_ptr<const engine::assets::TensorSource> model_weights;
    std::shared_ptr<const engine::assets::TensorSource> redae_decoder_weights;
};

std::shared_ptr<const FireRedAudioAssets> load_firered_audio_assets(
    const std::filesystem::path & model_path);

}  // namespace engine::models::firered_audio
