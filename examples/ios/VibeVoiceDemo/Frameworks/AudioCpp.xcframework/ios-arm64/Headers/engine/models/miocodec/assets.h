#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::miocodec {

struct MioCodecConfig {
    std::vector<int> local_ssl_layers;
    std::vector<int> global_ssl_layers;
    bool normalize_ssl_features = true;
    int sample_rate = 44100;
    int n_fft = 392;
    int hop_length = 98;
    int downsample_factor = 2;
    bool use_conv_downsample = true;
    bool use_wave_decoder = true;
    int wave_upsample_factor = 2;
    int wave_decoder_dim = 512;
    int wave_resnet_num_blocks = 2;
    int wave_resnet_num_groups = 32;
    int local_encoder_dim = 768;
    int local_encoder_layers = 6;
    int local_encoder_heads = 12;
    int local_encoder_window_size = 125;
    int local_encoder_max_seq_len = 512;
    std::vector<int> quantizer_levels;
    int quantizer_input_dim = 768;
    int quantizer_output_dim = 768;
    int codebook_size = 0;
    int global_encoder_input_channels = 768;
    int global_encoder_output_channels = 128;
    int global_encoder_layers = 4;
    int global_encoder_dim = 384;
    int global_encoder_intermediate_dim = 1152;
    int wave_prenet_dim = 768;
    int wave_prenet_output_dim = 512;
    int wave_prenet_layers = 6;
    int wave_prenet_heads = 12;
    int wave_prenet_window_size = 65;
    int wave_prenet_max_seq_len = 512;
    int wave_decoder_layers = 8;
    int wave_decoder_heads = 8;
    int wave_decoder_window_size = 65;
    int wave_decoder_max_seq_len = 512;
    int wave_decoder_condition_dim = 128;
    std::vector<int> wave_upsampler_factors;
    std::vector<int> wave_upsampler_kernel_sizes;
};

struct MioCodecAssets {
    assets::ResourceBundle resources;
    MioCodecConfig config;
    std::shared_ptr<const assets::TensorSource> model_weights;
    std::shared_ptr<const assets::TensorSource> wavlm_weights;
};

std::shared_ptr<const MioCodecAssets> load_miocodec_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::miocodec
