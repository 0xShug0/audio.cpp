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
    out.model_type = manifest.require("model_type").as_string();
    out.name = manifest.require("name").as_string();

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
    config.class_name = root.require("class_name").as_string();
    if (config.class_name != "HTDemucs") {
        throw std::runtime_error("Only HTDemucs class_name is supported, got: " + config.class_name);
    }
    config.signature = root.require("signature").as_string();
    config.checkpoint_file = root.require("checkpoint_file").as_string();
    for (const auto & item : root.require("sources").as_array()) {
        config.sources.push_back(item.as_string());
    }
    config.audio_channels = root.require("audio_channels").as_i64();
    config.sample_rate = root.require("samplerate").as_i64();
    config.segment_seconds = root.require("segment").as_f32();

    const auto & kwargs = root.require("kwargs");
    config.channels = kwargs.require("channels").as_i64();
    config.growth = kwargs.require("growth").as_i64();
    config.n_fft = kwargs.require("nfft").as_i64();
    config.hop_length = config.n_fft / 4;
    config.wiener_iters = kwargs.require("wiener_iters").as_i64();
    config.wiener_residual = kwargs.require("wiener_residual").as_bool();
    config.cac = kwargs.require("cac").as_bool();
    config.depth = kwargs.require("depth").as_i64();
    config.rewrite = kwargs.require("rewrite").as_bool();
    config.multi_freqs_depth = kwargs.require("multi_freqs_depth").as_i64();
    config.freq_emb_scale = kwargs.require("freq_emb").as_f32();
    config.embedding_scale = kwargs.require("emb_scale").as_f32();
    config.embedding_smooth = kwargs.require("emb_smooth").as_bool();
    config.kernel_size = kwargs.require("kernel_size").as_i64();
    config.stride = kwargs.require("stride").as_i64();
    config.time_stride = kwargs.require("time_stride").as_i64();
    config.context = kwargs.require("context").as_i64();
    config.context_enc = kwargs.require("context_enc").as_i64();
    config.norm_starts = kwargs.require("norm_starts").as_i64();
    config.norm_groups = kwargs.require("norm_groups").as_i64();
    config.dconv_mode = kwargs.require("dconv_mode").as_i64();
    config.dconv_depth = kwargs.require("dconv_depth").as_i64();
    config.dconv_comp = kwargs.require("dconv_comp").as_i64();
    config.dconv_init = kwargs.require("dconv_init").as_f32();
    config.bottom_channels = kwargs.require("bottom_channels").as_i64();
    config.transformer_layers = kwargs.require("t_layers").as_i64();
    config.transformer_hidden_scale = kwargs.require("t_hidden_scale").as_f32();
    config.transformer_heads = kwargs.require("t_heads").as_i64();
    config.transformer_dropout = kwargs.require("t_dropout").as_f32();
    config.transformer_layer_scale = kwargs.require("t_layer_scale").as_bool();
    config.transformer_gelu = kwargs.require("t_gelu").as_bool();
    config.transformer_norm_in_group = kwargs.require("t_norm_in_group").as_bool();
    config.transformer_group_norm = kwargs.require("t_group_norm").as_bool();
    config.transformer_norm_in = kwargs.require("t_norm_in").as_bool();
    config.transformer_norm_first = kwargs.require("t_norm_first").as_bool();
    config.transformer_norm_out = kwargs.require("t_norm_out").as_bool();
    config.transformer_cross_first = kwargs.require("t_cross_first").as_bool();
    config.transformer_max_period = kwargs.require("t_max_period").as_f32();
    config.transformer_weight_pos_embed = kwargs.require("t_weight_pos_embed").as_f32();

    if (kwargs.require("channels_time").is_null() == false) {
        throw std::runtime_error("HTDemucs channels_time != null is not supported yet");
    }
    if (!kwargs.require("multi_freqs").as_array().empty()) {
        throw std::runtime_error("HTDemucs multi_freqs is not supported yet");
    }
    if (kwargs.require("t_emb").as_string() != "sin") {
        throw std::runtime_error("HTDemucs only supports t_emb=sin");
    }
    if (config.transformer_norm_in_group || config.transformer_group_norm) {
        throw std::runtime_error("HTDemucs native runtime currently supports only layer-norm transformer checkpoints");
    }
    if (kwargs.require("t_sin_random_shift").as_i64() != 0) {
        throw std::runtime_error("HTDemucs only supports t_sin_random_shift=0");
    }
    if (kwargs.require("t_sparse_self_attn").as_bool() || kwargs.require("t_sparse_cross_attn").as_bool()) {
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
