#include "engine/models/demucs/assets.h"

#include "engine/framework/assets/model_package.h"
#include "engine/framework/io/json.h"

#include <cmath>
#include <stdexcept>

namespace engine::models::demucs {
namespace json = engine::io::json;
namespace {

HTDemucsManifest parse_package_manifest(const assets::ResourceBundle & resources) {
    const auto manifest = resources.parse_json("manifest");
    HTDemucsManifest out;
    out.model_type = json::require_string(manifest, "model_type");
    out.name = json::require_string(manifest, "name");

    if (out.model_type == "demucs_single") {
        (void) manifest.require("model");
    } else if (out.model_type == "demucs_single_alias") {
        (void) manifest.require("model");
    } else if (out.model_type == "demucs_bag") {
        throw std::runtime_error("HTDemucs package-spec loader currently supports only single-model manifests");
    } else {
        throw std::runtime_error("Unsupported HTDemucs manifest type: " + out.model_type);
    }
    return out;
}

HTDemucsConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("submodel_config");
    HTDemucsConfig config;
    config.class_name = json::require_string(root, "class_name");
    if (config.class_name != "HTDemucs") {
        throw std::runtime_error("Only HTDemucs class_name is supported, got: " + config.class_name);
    }
    config.signature = json::require_string(root, "signature");
    config.checkpoint_file = json::require_string(root, "checkpoint_file");
    config.sources = json::require_string_array(root, "sources");
    config.audio_channels = json::require_i64(root, "audio_channels");
    config.sample_rate = json::require_i64(root, "samplerate");
    config.segment_seconds = json::require_f32(root, "segment");

    const auto & kwargs = root.require("kwargs");
    config.channels = json::require_i64(kwargs, "channels");
    config.growth = json::require_i64(kwargs, "growth");
    config.n_fft = json::require_i64(kwargs, "nfft");
    config.hop_length = config.n_fft / 4;
    config.wiener_iters = json::require_i64(kwargs, "wiener_iters");
    config.wiener_residual = json::require_bool(kwargs, "wiener_residual");
    config.cac = json::require_bool(kwargs, "cac");
    config.depth = json::require_i64(kwargs, "depth");
    config.rewrite = json::require_bool(kwargs, "rewrite");
    config.multi_freqs_depth = json::require_i64(kwargs, "multi_freqs_depth");
    config.freq_emb_scale = json::require_f32(kwargs, "freq_emb");
    config.embedding_scale = json::require_f32(kwargs, "emb_scale");
    config.embedding_smooth = json::require_bool(kwargs, "emb_smooth");
    config.kernel_size = json::require_i64(kwargs, "kernel_size");
    config.stride = json::require_i64(kwargs, "stride");
    config.time_stride = json::require_i64(kwargs, "time_stride");
    config.context = json::require_i64(kwargs, "context");
    config.context_enc = json::require_i64(kwargs, "context_enc");
    config.norm_starts = json::require_i64(kwargs, "norm_starts");
    config.norm_groups = json::require_i64(kwargs, "norm_groups");
    config.dconv_mode = json::require_i64(kwargs, "dconv_mode");
    config.dconv_depth = json::require_i64(kwargs, "dconv_depth");
    config.dconv_comp = json::require_i64(kwargs, "dconv_comp");
    config.dconv_init = json::require_f32(kwargs, "dconv_init");
    config.bottom_channels = json::require_i64(kwargs, "bottom_channels");
    config.transformer_layers = json::require_i64(kwargs, "t_layers");
    config.transformer_hidden_scale = json::require_f32(kwargs, "t_hidden_scale");
    config.transformer_heads = json::require_i64(kwargs, "t_heads");
    config.transformer_dropout = json::require_f32(kwargs, "t_dropout");
    config.transformer_layer_scale = json::require_bool(kwargs, "t_layer_scale");
    config.transformer_gelu = json::require_bool(kwargs, "t_gelu");
    config.transformer_norm_in_group = json::require_bool(kwargs, "t_norm_in_group");
    config.transformer_group_norm = json::require_bool(kwargs, "t_group_norm");
    config.transformer_norm_in = json::require_bool(kwargs, "t_norm_in");
    config.transformer_norm_first = json::require_bool(kwargs, "t_norm_first");
    config.transformer_norm_out = json::require_bool(kwargs, "t_norm_out");
    config.transformer_cross_first = json::require_bool(kwargs, "t_cross_first");
    config.transformer_max_period = json::require_f32(kwargs, "t_max_period");
    config.transformer_weight_pos_embed = json::require_f32(kwargs, "t_weight_pos_embed");

    if (kwargs.require("channels_time").is_null() == false) {
        throw std::runtime_error("HTDemucs channels_time != null is not supported yet");
    }
    if (!kwargs.require("multi_freqs").as_array().empty()) {
        throw std::runtime_error("HTDemucs multi_freqs is not supported yet");
    }
    if (json::require_string(kwargs, "t_emb") != "sin") {
        throw std::runtime_error("HTDemucs only supports t_emb=sin");
    }
    if (config.transformer_norm_in_group || config.transformer_group_norm) {
        throw std::runtime_error("HTDemucs native runtime currently supports only layer-norm transformer checkpoints");
    }
    if (json::require_i64(kwargs, "t_sin_random_shift") != 0) {
        throw std::runtime_error("HTDemucs only supports t_sin_random_shift=0");
    }
    if (json::require_bool(kwargs, "t_sparse_self_attn") || json::require_bool(kwargs, "t_sparse_cross_attn")) {
        throw std::runtime_error("HTDemucs sparse attention is not supported");
    }
    if (!config.cac) {
        throw std::runtime_error("HTDemucs native runtime currently supports only cac=true");
    }
    if (config.wiener_iters != 0 || config.wiener_residual) {
        throw std::runtime_error("HTDemucs native runtime currently supports only cac path with wiener_iters=0");
    }

    config.segment_samples = static_cast<int64_t>(std::llround(static_cast<double>(config.sample_rate) * config.segment_seconds));
    config.stft_freq_bins = config.n_fft / 2;
    config.stft_frames = static_cast<int>(std::ceil(static_cast<double>(config.segment_samples) / static_cast<double>(config.hop_length)));
    config.input_freq_channels = config.audio_channels * 2;
    config.output_freq_channels = static_cast<int>(config.sources.size()) * config.audio_channels * 2;
    config.output_time_channels = static_cast<int>(config.sources.size()) * config.audio_channels;
    return config;
}

std::shared_ptr<const HTDemucsSubmodelAssets> load_submodel(const assets::ResourceBundle & resources) {
    auto sub = std::make_shared<HTDemucsSubmodelAssets>();
    sub->tensor_source = resources.open_tensor_source("submodel_weights");
    sub->config = parse_config(resources);
    return sub;
}

}  // namespace

void validate_demucs_weight_storage_type(assets::TensorStorageType storage_type) {
    switch (storage_type) {
    case assets::TensorStorageType::Native:
    case assets::TensorStorageType::F32:
    case assets::TensorStorageType::F16:
    case assets::TensorStorageType::BF16:
    case assets::TensorStorageType::Q8_0:
        return;
    default:
        throw std::runtime_error(
            "htdemucs weight_type currently supports only native, f32, f16, bf16, and q8_0");
    }
}

std::shared_ptr<const HTDemucsAssets> load_htdemucs_assets(const runtime::ModelLoadRequest & request) {
    auto out = std::make_shared<HTDemucsAssets>();
    out->resources = assets::load_resource_bundle_from_package_spec(
        request.model_path,
        assets::default_model_package_spec_path("htdemucs"));
    out->manifest = parse_package_manifest(out->resources);
    out->submodels.push_back(load_submodel(out->resources));
    return out;
}

}  // namespace engine::models::demucs
