#include "engine/models/seed_vc/assets.h"

#include "engine/framework/assets/model_package.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace engine::models::seed_vc {
namespace json = engine::io::json;
namespace {

bool has_suffix(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

SeedVcLengthRegulatorConfig parse_length_regulator(
    const json::Value & value,
    const char * codebook_size_key) {
    SeedVcLengthRegulatorConfig config;
    config.channels = json::require_i64(value, "channels");
    config.codebook_size = json::require_i64(value, codebook_size_key);
    config.is_discrete = json::require_bool(value, "is_discrete");
    config.f0_condition = json::require_bool(value, "f0_condition");
    config.sampling_ratios = json::optional_i64_array(value, "sampling_ratios");
    return config;
}

SeedVcAstralConfig parse_astral_config(
    const json::Value & root,
    bool skip_ssl_default) {
    const auto & encoder = root.require("encoder");
    const auto & quantizer = root.require("quantizer");
    SeedVcAstralConfig config;
    config.ssl_output_layer = json::require_i64(root, "ssl_output_layer");
    config.encoder_dim = json::require_i64(encoder, "dim");
    config.encoder_blocks = json::require_i64(encoder, "num_blocks");
    config.encoder_intermediate_dim = json::require_i64(encoder, "intermediate_dim");
    config.encoder_dilation = json::require_i64(encoder, "dilation");
    config.encoder_input_dim = json::require_i64(encoder, "input_dim");
    config.quantizer_codebook_size = json::require_i64(quantizer, "codebook_size");
    config.quantizer_dim = json::require_i64(quantizer, "dim");
    config.skip_ssl = json::optional_bool(root, "skip_ssl", skip_ssl_default);
    return config;
}

SeedVcBigVganConfig parse_bigvgan_config(const json::Value & root) {
    SeedVcBigVganConfig config;
    config.sampling_rate = json::require_i64(root, "sampling_rate");
    config.num_mels = json::require_i64(root, "num_mels");
    config.n_fft = json::require_i64(root, "n_fft");
    config.hop_size = json::require_i64(root, "hop_size");
    config.win_size = json::require_i64(root, "win_size");
    config.upsample_initial_channel = json::require_i64(root, "upsample_initial_channel");
    config.snake_logscale = json::require_bool(root, "snake_logscale");
    config.upsample_rates = json::require_i64_array(root, "upsample_rates");
    config.upsample_kernel_sizes = json::require_i64_array(root, "upsample_kernel_sizes");
    config.resblock_kernel_sizes = json::require_i64_array(root, "resblock_kernel_sizes");
    return config;
}

std::vector<std::vector<int64_t>> parse_int_matrix(const json::Value & value) {
    std::vector<std::vector<int64_t>> out;
    const auto & rows = value.as_array();
    out.reserve(rows.size());
    for (const auto & row : rows) {
        out.push_back(json::number_array_as<int64_t>(row));
    }
    return out;
}

SeedVcHiftConfig parse_hift_config(const json::Value & root) {
    SeedVcHiftConfig config;
    const auto & hift = root.require("hift");
    config.in_channels = json::require_i64(hift, "in_channels");
    config.base_channels = json::require_i64(hift, "base_channels");
    config.nb_harmonics = json::require_i64(hift, "nb_harmonics");
    config.sampling_rate = json::require_i64(hift, "sampling_rate");
    config.nsf_alpha = json::require_f32(hift, "nsf_alpha");
    config.nsf_sigma = json::require_f32(hift, "nsf_sigma");
    config.nsf_voiced_threshold = json::require_f32(hift, "nsf_voiced_threshold");
    config.upsample_rates = json::require_i64_array(hift, "upsample_rates");
    config.upsample_kernel_sizes = json::require_i64_array(hift, "upsample_kernel_sizes");
    const auto & istft = hift.require("istft_params");
    config.istft_n_fft = json::require_i64(istft, "n_fft");
    config.istft_hop = json::require_i64(istft, "hop_len");
    config.resblock_kernel_sizes = json::require_i64_array(hift, "resblock_kernel_sizes");
    config.resblock_dilation_sizes = parse_int_matrix(hift.require("resblock_dilation_sizes"));
    config.source_resblock_kernel_sizes = json::require_i64_array(hift, "source_resblock_kernel_sizes");
    config.source_resblock_dilation_sizes = parse_int_matrix(hift.require("source_resblock_dilation_sizes"));
    config.lrelu_slope = json::require_f32(hift, "lrelu_slope");
    config.audio_limit = json::require_f32(hift, "audio_limit");

    const auto & f0_predictor = root.require("f0_predictor");
    config.f0_num_class = json::require_i64(f0_predictor, "num_class");
    config.f0_in_channels = json::require_i64(f0_predictor, "in_channels");
    config.f0_cond_channels = json::require_i64(f0_predictor, "cond_channels");
    return config;
}

void parse_v1_variant(
    const json::Value & v1,
    SeedVcMelConfig & mel,
    SeedVcV1DitConfig & dit,
    SeedVcLengthRegulatorConfig & length_regulator,
    SeedVcV1WavenetConfig & wavenet,
    int64_t & style_dim) {
    const auto & preprocess = v1.require("preprocess_params");
    const auto & spect = preprocess.require("spect_params");
    mel.sample_rate = json::require_i64(preprocess, "sr");
    mel.n_fft = json::require_i64(spect, "n_fft");
    mel.win_size = json::require_i64(spect, "win_length");
    mel.hop_size = json::require_i64(spect, "hop_length");
    mel.num_mels = json::require_i64(spect, "n_mels");
    mel.fmin = json::require_f32(spect, "fmin");
    if (const auto * fmax = spect.find("fmax"); fmax != nullptr && !fmax->is_null()) {
        if (fmax->is_number()) {
            mel.fmax = fmax->as_f32();
        } else if (fmax->as_string() != "None") {
            throw std::runtime_error("Seed-VC V1 fmax config must be numeric or None");
        }
    }

    const auto & model_params = v1.require("model_params");
    style_dim = json::require_i64(model_params.require("style_encoder"), "dim");
    length_regulator =
        parse_length_regulator(model_params.require("length_regulator"), "content_codebook_size");
    const auto & v1_dit = model_params.require("DiT");
    dit.hidden_dim = json::require_i64(v1_dit, "hidden_dim");
    dit.num_heads = json::require_i64(v1_dit, "num_heads");
    dit.depth = json::require_i64(v1_dit, "depth");
    dit.block_size = json::require_i64(v1_dit, "block_size");
    dit.in_channels = json::require_i64(v1_dit, "in_channels");
    dit.content_dim = json::require_i64(v1_dit, "content_dim");
    dit.content_codebook_size = json::require_i64(v1_dit, "content_codebook_size");
    dit.n_f0_bins = json::require_i64(v1_dit, "n_f0_bins");
    dit.style_condition = json::require_bool(v1_dit, "style_condition");
    dit.f0_condition = json::require_bool(v1_dit, "f0_condition");
    dit.time_as_token = json::require_bool(v1_dit, "time_as_token");
    dit.style_as_token = json::require_bool(v1_dit, "style_as_token");
    dit.uvit_skip_connection = json::require_bool(v1_dit, "uvit_skip_connection");
    dit.long_skip_connection = json::require_bool(v1_dit, "long_skip_connection");
    dit.final_layer_type = json::require_string(v1_dit, "final_layer_type");
    if (const auto * wavenet_json = model_params.find("wavenet")) {
        wavenet.hidden_dim = json::require_i64(*wavenet_json, "hidden_dim");
        wavenet.num_layers = json::require_i64(*wavenet_json, "num_layers");
        wavenet.kernel_size = json::require_i64(*wavenet_json, "kernel_size");
        wavenet.dilation_rate = json::require_i64(*wavenet_json, "dilation_rate");
        wavenet.style_condition = json::require_bool(*wavenet_json, "style_condition");
    }
}

SeedVcConfig parse_config(const assets::ResourceBundle & resources) {
    SeedVcConfig config;
    const auto v2 = resources.parse_json("v2_wrapper_config");
    const auto & v2_mel = v2.require("mel_fn");
    config.v2_mel.sample_rate = json::require_i64(v2_mel, "sampling_rate");
    config.v2_mel.n_fft = json::require_i64(v2_mel, "n_fft");
    config.v2_mel.win_size = json::require_i64(v2_mel, "win_size");
    config.v2_mel.hop_size = json::require_i64(v2_mel, "hop_size");
    config.v2_mel.num_mels = json::require_i64(v2_mel, "num_mels");
    config.v2_mel.fmin = json::require_f32(v2_mel, "fmin");
    if (const auto * fmax = v2_mel.find("fmax"); fmax != nullptr && !fmax->is_null()) {
        config.v2_mel.fmax = fmax->as_f32();
    }

    const auto & v2_cfm = v2.require("cfm").require("estimator");
    config.v2_cfm.block_size = json::require_i64(v2_cfm, "block_size");
    config.v2_cfm.depth = json::require_i64(v2_cfm, "depth");
    config.v2_cfm.num_heads = json::require_i64(v2_cfm, "num_heads");
    config.v2_cfm.hidden_dim = json::require_i64(v2_cfm, "hidden_dim");
    config.v2_cfm.in_channels = json::require_i64(v2_cfm, "in_channels");
    config.v2_cfm.content_dim = json::require_i64(v2_cfm, "content_dim");
    config.v2_cfm.style_encoder_dim = json::require_i64(v2_cfm, "style_encoder_dim");
    config.v2_cfm.time_as_token = json::require_bool(v2_cfm, "time_as_token");
    config.v2_cfm.style_as_token = json::require_bool(v2_cfm, "style_as_token");
    config.v2_cfm.uvit_skip_connection = json::require_bool(v2_cfm, "uvit_skip_connection");
    config.v2_cfm_length_regulator = parse_length_regulator(v2.require("cfm_length_regulator"), "codebook_size");

    const auto & v2_ar = v2.require("ar").require("model").require("config");
    config.v2_ar.dim = json::require_i64(v2_ar, "dim");
    config.v2_ar.head_dim = json::require_i64(v2_ar, "head_dim");
    config.v2_ar.n_local_heads = json::require_i64(v2_ar, "n_local_heads");
    config.v2_ar.intermediate_size = json::require_i64(v2_ar, "intermediate_size");
    config.v2_ar.n_head = json::require_i64(v2_ar, "n_head");
    config.v2_ar.n_layer = json::require_i64(v2_ar, "n_layer");
    config.v2_ar.vocab_size = json::require_i64(v2_ar, "vocab_size");
    config.v2_ar.rope_base = json::require_f32(v2_ar, "rope_base");
    config.v2_ar_length_regulator = parse_length_regulator(v2.require("ar_length_regulator"), "codebook_size");

    const auto & style_encoder = v2.require("style_encoder");
    config.v2_style_feat_dim = json::require_i64(style_encoder, "feat_dim");
    config.v2_style_embedding_size = json::require_i64(style_encoder, "embedding_size");
    config.v2_astral_narrow = parse_astral_config(resources.parse_json("astral_bsq32_config"), true);
    config.v2_astral_wide = parse_astral_config(resources.parse_json("astral_bsq2048_config"), false);

    parse_v1_variant(
        resources.parse_json("v1_svc_config"),
        config.v1_mel,
        config.v1_dit,
        config.v1_length_regulator,
        config.v1_wavenet,
        config.v1_style_dim);
    parse_v1_variant(
        resources.parse_json("v1_whisper_bigvgan_config"),
        config.v1_whisper_bigvgan_mel,
        config.v1_whisper_bigvgan_dit,
        config.v1_whisper_bigvgan_length_regulator,
        config.v1_whisper_bigvgan_wavenet,
        config.v1_whisper_bigvgan_style_dim);
    parse_v1_variant(
        resources.parse_json("v1_xlsr_hift_config"),
        config.v1_xlsr_hift_mel,
        config.v1_xlsr_hift_dit,
        config.v1_xlsr_hift_length_regulator,
        config.v1_xlsr_hift_wavenet,
        config.v1_xlsr_hift_style_dim);

    config.bigvgan_22k = parse_bigvgan_config(resources.parse_json("bigvgan_22k_config"));
    config.bigvgan_44k = parse_bigvgan_config(resources.parse_json("bigvgan_44k_config"));
    config.hift = parse_hift_config(resources.parse_json("hift_config"));
    return config;
}

}  // namespace

std::shared_ptr<const SeedVcAssets> load_seed_vc_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<SeedVcAssets>();
    assets->resources = assets::load_resource_bundle_from_package_spec(
        model_path,
        assets::default_model_package_spec_path("seed_vc"));
    assets->config = parse_config(assets->resources);
    assets->v2_ar_weights = assets->resources.open_tensor_source("v2_ar_weights");
    assets->v2_cfm_weights = assets->resources.open_tensor_source("v2_cfm_weights");
    assets->v1_svc_weights = assets->resources.open_tensor_source("v1_svc_weights");
    assets->v1_whisper_bigvgan_weights = assets->resources.open_tensor_source("v1_whisper_bigvgan_weights");
    assets->v1_xlsr_hift_weights = assets->resources.open_tensor_source("v1_xlsr_hift_weights");
    assets->astral_bsq32_weights = assets->resources.open_tensor_source("astral_bsq32_weights");
    assets->astral_bsq2048_weights = assets->resources.open_tensor_source("astral_bsq2048_weights");
    assets->campplus_weights = assets->resources.open_tensor_source("campplus_weights");
    assets->rmvpe_weights = assets->resources.open_tensor_source("rmvpe_weights");
    assets->hift_weights = assets->resources.open_tensor_source("hift_weights");
    assets->bigvgan_22k_weights = assets->resources.open_tensor_source("bigvgan_22k_weights");
    assets->bigvgan_44k_weights = assets->resources.open_tensor_source("bigvgan_44k_weights");
    assets->whisper_small_weights = assets->resources.open_tensor_source("whisper_small_weights");
    assets->hubert_large_weights = assets->resources.open_tensor_source("hubert_large_weights");
    assets->wav2vec2_xlsr_weights = assets->resources.open_tensor_source("wav2vec2_xlsr_weights");
    return assets;
}

assets::TensorStorageType seed_vc_component_storage_type(
    const assets::TensorSource & source,
    std::string_view tensor_name,
    assets::TensorStorageType requested_type) {
    const auto metadata = source.require_metadata(tensor_name);
    if (metadata.shape.size() <= 1 ||
        tensor_name == "model.encoder.embed_positions.weight" ||
        has_suffix(tensor_name, ".bias") ||
        has_suffix(tensor_name, ".gamma") ||
        has_suffix(tensor_name, ".beta") ||
        has_suffix(tensor_name, ".running_mean") ||
        has_suffix(tensor_name, ".running_var")) {
        return assets::TensorStorageType::F32;
    }
    return requested_type;
}

}  // namespace engine::models::seed_vc
