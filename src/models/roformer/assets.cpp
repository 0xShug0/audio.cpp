#include "engine/models/roformer/assets.h"

#include "engine/framework/assets/model_package.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/io/json.h"

#include <algorithm>
#include <stdexcept>

namespace engine::models::roformer {
namespace json = engine::io::json;

void validate_roformer_weight_storage_type(assets::TensorStorageType storage_type) {
    switch (storage_type) {
    case assets::TensorStorageType::Native:
    case assets::TensorStorageType::F32:
    case assets::TensorStorageType::F16:
    case assets::TensorStorageType::BF16:
    case assets::TensorStorageType::Q8_0:
        return;
    default:
        throw std::runtime_error(
            "mel_band_roformer weight_type currently supports only native, f32, f16, bf16, and q8_0");
    }
}

namespace {

void validate_config(const json::Value & value) {
    const auto model_type = json::require_string(value, "model_type");
    if (model_type != std::string(kMelBandRoformerFamily)) {
        throw std::runtime_error(
            "mel_band_roformer config model_type mismatch: expected mel_band_roformer, got " + model_type);
    }
}

assets::ResourceBundle load_resources(const runtime::ModelLoadRequest & request) {
    return assets::load_resource_bundle_from_package_spec(
        request.model_path,
        assets::default_model_package_spec_path(std::string(kMelBandRoformerFamily)));
}

void fill_mel_band_layout(
    RoformerArchitectureConfig & config,
    int num_bands) {
    engine::audio::MelFilterbankConfig mel_config;
    mel_config.sample_rate = config.sample_rate;
    mel_config.n_fft = config.n_fft;
    mel_config.n_mels = num_bands;
    mel_config.lowfreq = 0.0f;
    mel_config.highfreq = 0.0f;
    mel_config.slaney_norm = true;
    auto filterbank = engine::audio::MelFilterbank().build(mel_config);
    if (filterbank.shape.size() != 2) {
        throw std::runtime_error("RoFormer mel filterbank must be rank-2");
    }
    const int64_t bands = filterbank.shape[0];
    const int64_t freqs = filterbank.shape[1];
    if (bands != num_bands) {
        throw std::runtime_error("RoFormer mel filterbank band count mismatch");
    }
    auto & values = filterbank.values;
    values[0] = 1.0f;
    values[static_cast<size_t>((bands - 1) * freqs + (freqs - 1))] = 1.0f;

    std::vector<int64_t> num_freqs_per_band(static_cast<size_t>(bands), 0);
    std::vector<int64_t> num_bands_per_freq(static_cast<size_t>(freqs), 0);
    std::vector<int64_t> freq_indices;
    for (int64_t band = 0; band < bands; ++band) {
        for (int64_t freq = 0; freq < freqs; ++freq) {
            if (values[static_cast<size_t>(band * freqs + freq)] <= 0.0f) {
                continue;
            }
            ++num_freqs_per_band[static_cast<size_t>(band)];
            ++num_bands_per_freq[static_cast<size_t>(freq)];
            if (config.stereo) {
                freq_indices.push_back(freq * 2);
                freq_indices.push_back(freq * 2 + 1);
            } else {
                freq_indices.push_back(freq);
            }
        }
    }

    config.band_input_dims.reserve(static_cast<size_t>(bands));
    for (const int64_t count : num_freqs_per_band) {
        config.band_input_dims.push_back(2 * count * config.channels);
    }
    config.num_bands = num_bands;
    config.total_band_input_dim = 0;
    for (const int64_t dim : config.band_input_dims) {
        config.total_band_input_dim += static_cast<int>(dim);
    }
    config.merged_freq_indices = std::move(freq_indices);
    config.merged_band_counts.reserve(static_cast<size_t>(freqs * config.channels));
    for (int64_t freq = 0; freq < freqs; ++freq) {
        const int64_t count = std::max<int64_t>(1, num_bands_per_freq[static_cast<size_t>(freq)]);
        for (int channel = 0; channel < config.channels; ++channel) {
            config.merged_band_counts.push_back(count);
        }
    }
}

RoformerArchitectureConfig parse_config(const json::Value & parsed) {
    if (!parsed.is_object()) {
        throw std::runtime_error("mel_band_roformer config root must be an object");
    }
    validate_config(parsed);
    RoformerArchitectureConfig config;
    config.sample_rate = json::require_i32(parsed, "sample_rate");
    config.channels = 2;
    config.stereo = true;
    config.chunk_size = json::require_i32(parsed, "chunk_size");
    config.inference_batch_size = 1;
    config.inference_num_overlap = json::require_i32(parsed, "num_overlap");
    config.inference_normalize = false;
    config.dim = json::require_i32(parsed, "dim");
    config.depth = json::require_i32(parsed, "depth");
    config.num_bands = json::require_i32(parsed, "num_bands");
    config.num_stems = json::require_i32(parsed, "num_stems");
    config.time_transformer_depth = json::optional_i32(parsed, "time_transformer_depth", 1);
    config.freq_transformer_depth = json::optional_i32(parsed, "freq_transformer_depth", 1);
    config.dim_head = json::require_i32(parsed, "dim_head");
    config.heads = json::require_i32(parsed, "heads");
    config.n_fft = json::require_i32(parsed, "n_fft");
    config.hop_length = json::require_i32(parsed, "hop_length");
    config.win_length = json::require_i32(parsed, "win_length");
    config.stft_normalized = json::optional_bool(parsed, "stft_normalized", false);
    config.mask_estimator_depth = json::require_i32(parsed, "mask_estimator_depth");
    config.mlp_expansion_factor = json::optional_i32(parsed, "mlp_expansion_factor", 4);
    config.skip_connection = false;
    config.stft_freq_bins = config.n_fft / 2 + 1;
    config.chunk_frames = 1 + config.chunk_size / config.hop_length;
    config.instruments = {"vocals"};
    config.target_instrument = std::string("vocals");
    config.has_final_norm = json::optional_bool(parsed, "has_final_norm", false);
    fill_mel_band_layout(config, config.num_bands);
    return config;
}

}  // namespace

std::shared_ptr<const RoformerAssets> load_mel_band_roformer_assets(
    const runtime::ModelLoadRequest & request) {
    auto assets = std::make_shared<RoformerAssets>();
    assets->resources = load_resources(request);
    assets->tensor_source = assets->resources.open_tensor_source("weights");
    const auto parsed = assets->resources.parse_json("config");
    assets->config = parse_config(parsed);
    return assets;
}

}  // namespace engine::models::roformer
