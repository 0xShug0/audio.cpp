#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::audiosr {

struct AudioSRConfig {
    int sample_rate = 48000;
    int n_fft = 2048;
    int hop_length = 480;
    int win_length = 2048;
    int n_mels = 256;
    float mel_fmin = 20.0F;
    float mel_fmax = 24000.0F;
    int segment_seconds = 512;
    int segment_seconds_divisor = 100;

    int latent_channels = 16;
    int encoder_z_channels = 16;
    int vae_base_channels = 128;
    int unet_model_channels = 128;
    int unet_in_channels = 32;
    int unet_out_channels = 16;
    int unet_time_embed_dim = 512;
    int diffusion_timesteps = 1000;
    float beta_linear_start = 0.0015F;
    float beta_linear_end = 0.0195F;
    float vae_scale_factor = 0.9227914F;
    float unconditional_lowpass_value = -11.4981F;

    int vocoder_upsample_initial_channel = 1536;
    int vocoder_num_mels = 256;
    int vocoder_resblock_kernels[4] = {3, 7, 11, 15};
    int vocoder_upsample_rates[5] = {6, 5, 4, 2, 2};
    int vocoder_upsample_kernels[5] = {12, 10, 8, 4, 4};
};

struct AudioSRAssets {
    std::filesystem::path model_root;
    std::filesystem::path gguf_path;
    std::string variant = "basic";
    AudioSRConfig config;
    engine::assets::ResourceBundle resources;
    std::vector<float> frontend_lowpass_sos;
    std::vector<float> frontend_lowpass_sos_valid;
    std::shared_ptr<const engine::assets::TensorSource> weights;
};

std::shared_ptr<const AudioSRAssets> load_audiosr_assets(const std::filesystem::path & model_path);

}  // namespace engine::models::audiosr
