#include "engine/models/roformer/assets.h"

#include "engine/framework/model_spec/package.h"
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
            "RoFormer weight_type currently supports only native, f32, f16, bf16, and q8_0");
    }
}

namespace {

void validate_config(const json::Value & value, std::string_view family) {
    const auto model_type = json::require_string(value, "model_type");
    if (model_type != family) {
        throw std::runtime_error(
            std::string(family) + " config model_type mismatch: expected " +
            std::string(family) + ", got " + model_type);
    }
}

assets::ResourceBundle load_resources(
    const runtime::ModelLoadRequest & request,
    std::string_view family) {
    return engine::model_spec::load_resource_bundle(
        request.model_path,
        engine::model_spec::default_spec_path(std::string(family)));
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

void fill_bs_band_layout(
    RoformerArchitectureConfig & config,
    const std::vector<int64_t> & freqs_per_band) {
    if (freqs_per_band.size() < 2) {
        throw std::runtime_error("bs_roformer requires at least two frequency bands");
    }
    int64_t total_freqs = 0;
    for (const int64_t count : freqs_per_band) {
        if (count <= 0) {
            throw std::runtime_error("bs_roformer freqs_per_bands values must be positive");
        }
        total_freqs += count;
    }
    if (total_freqs != config.stft_freq_bins) {
        throw std::runtime_error(
            "bs_roformer freqs_per_bands sum mismatch: expected " +
            std::to_string(config.stft_freq_bins) + ", got " +
            std::to_string(total_freqs));
    }

    config.num_bands = static_cast<int>(freqs_per_band.size());
    config.total_band_input_dim = 0;
    config.band_input_dims.reserve(freqs_per_band.size());
    config.merged_freq_indices.reserve(
        static_cast<size_t>(config.stft_freq_bins * config.channels));
    int64_t freq = 0;
    for (const int64_t count : freqs_per_band) {
        const int64_t dim = 2 * count * config.channels;
        config.band_input_dims.push_back(dim);
        config.total_band_input_dim += static_cast<int>(dim);
        for (int64_t local = 0; local < count; ++local, ++freq) {
            for (int channel = 0; channel < config.channels; ++channel) {
                config.merged_freq_indices.push_back(freq * config.channels + channel);
            }
        }
    }
    config.merged_band_counts.assign(
        static_cast<size_t>(config.stft_freq_bins * config.channels), 1);
}

RoformerArchitectureConfig parse_config(
    const json::Value & parsed,
    std::string_view family) {
    if (!parsed.is_object()) {
        throw std::runtime_error(std::string(family) + " config root must be an object");
    }
    validate_config(parsed, family);
    RoformerArchitectureConfig config;
    config.family = std::string(family);
    config.sample_rate = json::require_i32(parsed, "sample_rate");
    config.stereo = json::optional_bool(parsed, "stereo", true);
    config.channels = config.stereo ? 2 : 1;
    config.chunk_size = json::require_i32(parsed, "chunk_size");
    config.inference_batch_size = json::optional_i32(parsed, "batch_size", 1);
    config.inference_num_overlap = json::require_i32(parsed, "num_overlap");
    config.inference_normalize = json::optional_bool(parsed, "normalize", false);
    config.dim = json::require_i32(parsed, "dim");
    config.depth = json::require_i32(parsed, "depth");
    config.num_stems = json::require_i32(parsed, "num_stems");
    config.time_transformer_depth = json::optional_i32(parsed, "time_transformer_depth", 1);
    config.freq_transformer_depth = json::optional_i32(parsed, "freq_transformer_depth", 1);
    config.linear_transformer_depth = json::optional_i32(parsed, "linear_transformer_depth", 0);
    if (config.linear_transformer_depth != 0) {
        throw std::runtime_error(
            std::string(family) + " native runtime does not yet support linear_transformer_depth");
    }
    config.dim_head = json::require_i32(parsed, "dim_head");
    config.heads = json::require_i32(parsed, "heads");
    config.n_fft = json::require_i32(parsed, "n_fft");
    config.hop_length = json::require_i32(parsed, "hop_length");
    config.win_length = json::require_i32(parsed, "win_length");
    config.stft_normalized = json::optional_bool(parsed, "stft_normalized", false);
    config.mask_estimator_depth = json::require_i32(parsed, "mask_estimator_depth");
    if (config.mask_estimator_depth <= 0) {
        throw std::runtime_error(
            std::string(family) + " mask_estimator_depth must be positive");
    }
    // The two upstream implementations use the same config field with
    // different semantics. BS-RoFormer counts every linear layer, including
    // its output projection. Mel-Band RoFormer counts only the hidden
    // Linear+Tanh blocks and appends a separate output projection.
    config.mask_estimator_linear_layers =
        config.mask_estimator_depth + (family == kMelBandRoformerFamily ? 1 : 0);
    config.mlp_expansion_factor = json::optional_i32(parsed, "mlp_expansion_factor", 4);
    config.skip_connection = json::optional_bool(parsed, "skip_connection", false);
    if (config.skip_connection) {
        throw std::runtime_error(
            std::string(family) + " native runtime does not yet support skip_connection");
    }
    config.stft_freq_bins = config.n_fft / 2 + 1;
    config.chunk_frames = 1 + config.chunk_size / config.hop_length;
    config.instruments = {"vocals"};
    config.target_instrument = std::string("vocals");
    if (family == kBsRoformerFamily) {
        config.fused_qkv = true;
        config.transformer_output_norm = false;
        config.has_final_norm = true;
        fill_bs_band_layout(config, json::require_i64_array(parsed, "freqs_per_bands"));
    } else {
        config.num_bands = json::require_i32(parsed, "num_bands");
        config.fused_qkv = false;
        config.transformer_output_norm = true;
        config.has_final_norm = json::optional_bool(parsed, "has_final_norm", false);
        fill_mel_band_layout(config, config.num_bands);
    }
    return config;
}

}  // namespace

std::shared_ptr<const RoformerAssets> load_mel_band_roformer_assets(
    const runtime::ModelLoadRequest & request) {
    return load_roformer_assets(request, kMelBandRoformerFamily);
}

std::shared_ptr<const RoformerAssets> load_bs_roformer_assets(
    const runtime::ModelLoadRequest & request) {
    return load_roformer_assets(request, kBsRoformerFamily);
}

std::shared_ptr<const RoformerAssets> load_roformer_assets(
    const runtime::ModelLoadRequest & request,
    std::string_view family) {
    auto assets = std::make_shared<RoformerAssets>();
    assets->resources = load_resources(request, family);
    assets->tensor_source = assets->resources.open_tensor_source("weights");
    const auto parsed = assets->resources.parse_json("config");
    assets->config = parse_config(parsed, family);
    return assets;
}

}  // namespace engine::models::roformer
