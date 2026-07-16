#pragma once

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/model.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace engine::models::roformer {

inline constexpr std::string_view kMelBandRoformerFamily = "mel_band_roformer";

struct RoformerArchitectureConfig {
    int sample_rate = 0;
    int channels = 0;
    int chunk_size = 0;
    int inference_batch_size = 0;
    int inference_num_overlap = 0;
    bool inference_normalize = false;
    int dim = 0;
    int depth = 0;
    int num_bands = 0;
    bool stereo = false;
    int num_stems = 0;
    int time_transformer_depth = 1;
    int freq_transformer_depth = 1;
    int linear_transformer_depth = 0;
    int dim_head = 0;
    int heads = 0;
    int n_fft = 0;
    int hop_length = 0;
    int win_length = 0;
    bool stft_normalized = false;
    int mask_estimator_depth = 0;
    int mlp_expansion_factor = 4;
    bool skip_connection = false;
    bool has_final_norm = false;
    int stft_freq_bins = 0;
    int chunk_frames = 0;
    int total_band_input_dim = 0;
    float rope_theta = 10000.0f;
    std::vector<int64_t> band_input_dims;
    std::vector<int64_t> merged_freq_indices;
    std::vector<int64_t> merged_band_counts;
    std::vector<std::string> instruments;
    std::optional<std::string> target_instrument = std::nullopt;
};

struct RoformerAssets {
    assets::ResourceBundle resources;
    std::shared_ptr<const assets::TensorSource> tensor_source;
    RoformerArchitectureConfig config;
};

void validate_roformer_weight_storage_type(assets::TensorStorageType storage_type);
std::shared_ptr<const RoformerAssets> load_mel_band_roformer_assets(
    const runtime::ModelLoadRequest & request);

}  // namespace engine::models::roformer
