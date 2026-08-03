#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/model.h"

#include <memory>
#include <string>
#include <vector>

namespace engine::models::demucs {

struct HTDemucsManifest {
    std::string model_type;
    std::string name;
};

struct HTDemucsConfig {
    std::string class_name;
    std::string signature;
    std::string checkpoint_file;
    std::vector<std::string> sources;
    int audio_channels = 0;
    int sample_rate = 0;
    float segment_seconds = 0.0f;
    int64_t segment_samples = 0;
    int channels = 0;
    int growth = 0;
    int n_fft = 0;
    int hop_length = 0;
    int wiener_iters = 0;
    bool wiener_residual = false;
    bool cac = true;
    int depth = 0;
    bool rewrite = true;
    int multi_freqs_depth = 0;
    float freq_emb_scale = 0.0f;
    float embedding_scale = 1.0f;
    bool embedding_smooth = false;
    int kernel_size = 0;
    int stride = 0;
    int time_stride = 0;
    int context = 0;
    int context_enc = 0;
    int norm_starts = 0;
    int norm_groups = 0;
    int dconv_mode = 0;
    int dconv_depth = 0;
    int dconv_comp = 0;
    float dconv_init = 0.0f;
    int bottom_channels = 0;
    int transformer_layers = 0;
    float transformer_hidden_scale = 0.0f;
    int transformer_heads = 0;
    float transformer_dropout = 0.0f;
    bool transformer_layer_scale = false;
    bool transformer_gelu = true;
    bool transformer_norm_in = true;
    bool transformer_norm_in_group = false;
    bool transformer_group_norm = false;
    bool transformer_norm_first = true;
    bool transformer_norm_out = true;
    bool transformer_cross_first = false;
    float transformer_max_period = 10000.0f;
    float transformer_weight_pos_embed = 1.0f;

    int input_freq_channels = 0;
    int output_freq_channels = 0;
    int output_time_channels = 0;
    int stft_freq_bins = 0;
    int stft_frames = 0;
};

struct HTDemucsSubmodelAssets {
    std::shared_ptr<const assets::TensorSource> tensor_source;
    HTDemucsConfig config;
};

struct HTDemucsAssets {
    assets::ResourceBundle resources;
    HTDemucsManifest manifest;
    std::vector<std::shared_ptr<const HTDemucsSubmodelAssets>> submodels;
};

void validate_demucs_weight_storage_type(assets::TensorStorageType storage_type);
std::shared_ptr<const HTDemucsAssets> load_htdemucs_assets(const runtime::ModelLoadRequest & request);

}  // namespace engine::models::demucs
