#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace engine::models::seed_vc {

struct SeedVcMelConfig {
    int64_t sample_rate = 0;
    int64_t n_fft = 0;
    int64_t win_size = 0;
    int64_t hop_size = 0;
    int64_t num_mels = 0;
    float fmin = 0.0F;
    float fmax = 0.0F;
};

struct SeedVcV2DitConfig {
    int64_t block_size = 0;
    int64_t depth = 0;
    int64_t num_heads = 0;
    int64_t hidden_dim = 0;
    int64_t in_channels = 0;
    int64_t content_dim = 0;
    int64_t style_encoder_dim = 0;
    bool time_as_token = false;
    bool style_as_token = false;
    bool uvit_skip_connection = false;
};

struct SeedVcLengthRegulatorConfig {
    int64_t channels = 0;
    int64_t codebook_size = 0;
    bool is_discrete = false;
    bool f0_condition = false;
    std::vector<int64_t> sampling_ratios;
};

struct SeedVcV2ArConfig {
    int64_t dim = 0;
    int64_t head_dim = 0;
    int64_t n_local_heads = 0;
    int64_t intermediate_size = 0;
    int64_t n_head = 0;
    int64_t n_layer = 0;
    int64_t vocab_size = 0;
    float rope_base = 10000.0F;
};

struct SeedVcAstralConfig {
    int64_t ssl_output_layer = 0;
    int64_t encoder_dim = 0;
    int64_t encoder_blocks = 0;
    int64_t encoder_intermediate_dim = 0;
    int64_t encoder_dilation = 0;
    int64_t encoder_input_dim = 0;
    int64_t quantizer_codebook_size = 0;
    int64_t quantizer_dim = 0;
    bool skip_ssl = false;
};

struct SeedVcV1DitConfig {
    int64_t hidden_dim = 0;
    int64_t num_heads = 0;
    int64_t depth = 0;
    int64_t block_size = 0;
    int64_t in_channels = 0;
    int64_t content_dim = 0;
    int64_t content_codebook_size = 0;
    int64_t n_f0_bins = 0;
    bool style_condition = false;
    bool f0_condition = false;
    bool time_as_token = false;
    bool style_as_token = false;
    bool uvit_skip_connection = false;
    bool long_skip_connection = false;
    std::string final_layer_type;
};

struct SeedVcV1WavenetConfig {
    int64_t hidden_dim = 0;
    int64_t num_layers = 0;
    int64_t kernel_size = 0;
    int64_t dilation_rate = 0;
    bool style_condition = false;
};

struct SeedVcBigVganConfig {
    int64_t sampling_rate = 0;
    int64_t num_mels = 0;
    int64_t n_fft = 0;
    int64_t hop_size = 0;
    int64_t win_size = 0;
    int64_t upsample_initial_channel = 0;
    bool snake_logscale = false;
    std::vector<int64_t> upsample_rates;
    std::vector<int64_t> upsample_kernel_sizes;
    std::vector<int64_t> resblock_kernel_sizes;
};

struct SeedVcHiftConfig {
    int64_t in_channels = 0;
    int64_t base_channels = 0;
    int64_t nb_harmonics = 0;
    int64_t sampling_rate = 0;
    float nsf_alpha = 0.0F;
    float nsf_sigma = 0.0F;
    float nsf_voiced_threshold = 0.0F;
    std::vector<int64_t> upsample_rates;
    std::vector<int64_t> upsample_kernel_sizes;
    int64_t istft_n_fft = 0;
    int64_t istft_hop = 0;
    std::vector<int64_t> resblock_kernel_sizes;
    std::vector<std::vector<int64_t>> resblock_dilation_sizes;
    std::vector<int64_t> source_resblock_kernel_sizes;
    std::vector<std::vector<int64_t>> source_resblock_dilation_sizes;
    float lrelu_slope = 0.0F;
    float audio_limit = 0.0F;
    int64_t f0_num_class = 0;
    int64_t f0_in_channels = 0;
    int64_t f0_cond_channels = 0;
};

struct SeedVcConfig {
    SeedVcMelConfig v2_mel;
    SeedVcV2DitConfig v2_cfm;
    SeedVcLengthRegulatorConfig v2_cfm_length_regulator;
    SeedVcV2ArConfig v2_ar;
    SeedVcLengthRegulatorConfig v2_ar_length_regulator;
    SeedVcAstralConfig v2_astral_narrow;
    SeedVcAstralConfig v2_astral_wide;
    int64_t v2_style_embedding_size = 0;
    int64_t v2_style_feat_dim = 0;

    SeedVcMelConfig v1_mel;
    SeedVcV1DitConfig v1_dit;
    SeedVcLengthRegulatorConfig v1_length_regulator;
    int64_t v1_style_dim = 0;
    SeedVcV1WavenetConfig v1_wavenet;

    SeedVcMelConfig v1_whisper_bigvgan_mel;
    SeedVcV1DitConfig v1_whisper_bigvgan_dit;
    SeedVcLengthRegulatorConfig v1_whisper_bigvgan_length_regulator;
    SeedVcV1WavenetConfig v1_whisper_bigvgan_wavenet;
    int64_t v1_whisper_bigvgan_style_dim = 0;

    SeedVcMelConfig v1_xlsr_hift_mel;
    SeedVcV1DitConfig v1_xlsr_hift_dit;
    SeedVcLengthRegulatorConfig v1_xlsr_hift_length_regulator;
    SeedVcV1WavenetConfig v1_xlsr_hift_wavenet;
    int64_t v1_xlsr_hift_style_dim = 0;

    SeedVcBigVganConfig bigvgan_22k;
    SeedVcBigVganConfig bigvgan_44k;
    SeedVcHiftConfig hift;
};

struct SeedVcAssets {
    assets::ResourceBundle resources;
    SeedVcConfig config;
    std::shared_ptr<const assets::TensorSource> v2_ar_weights;
    std::shared_ptr<const assets::TensorSource> v2_cfm_weights;
    std::shared_ptr<const assets::TensorSource> v1_svc_weights;
    std::shared_ptr<const assets::TensorSource> v1_whisper_bigvgan_weights;
    std::shared_ptr<const assets::TensorSource> v1_xlsr_hift_weights;
    std::shared_ptr<const assets::TensorSource> astral_bsq32_weights;
    std::shared_ptr<const assets::TensorSource> astral_bsq2048_weights;
    std::shared_ptr<const assets::TensorSource> campplus_weights;
    std::shared_ptr<const assets::TensorSource> rmvpe_weights;
    std::shared_ptr<const assets::TensorSource> hift_weights;
    std::shared_ptr<const assets::TensorSource> bigvgan_22k_weights;
    std::shared_ptr<const assets::TensorSource> bigvgan_44k_weights;
    std::shared_ptr<const assets::TensorSource> whisper_small_weights;
    std::shared_ptr<const assets::TensorSource> hubert_large_weights;
    std::shared_ptr<const assets::TensorSource> wav2vec2_xlsr_weights;
};

std::shared_ptr<const SeedVcAssets> load_seed_vc_assets(const std::filesystem::path & model_path);
assets::TensorStorageType seed_vc_component_storage_type(
    const assets::TensorSource & source,
    std::string_view tensor_name,
    assets::TensorStorageType requested_type);

}  // namespace engine::models::seed_vc
