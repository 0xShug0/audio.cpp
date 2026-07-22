#include "engine/models/miocodec/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/yaml.h"

#include <numeric>
#include <stdexcept>

namespace engine::models::miocodec {
namespace yaml = engine::io::yaml;
namespace {

int compute_codebook_size(const std::vector<int> & values) {
    if (values.empty()) {
        throw std::runtime_error("MioCodec quantizer levels must not be empty");
    }
    return std::accumulate(values.begin(), values.end(), 1, [](int lhs, int rhs) {
        if (rhs <= 0) {
            throw std::runtime_error("MioCodec quantizer levels must be positive");
        }
        return lhs * rhs;
    });
}

MioCodecConfig parse_config(const assets::ResourceBundle & resources) {
    const auto parsed = resources.parse_flattened_yaml("config");
    MioCodecConfig config;
    config.local_ssl_layers = yaml::require_list_int(parsed, "model.init_args.config.local_ssl_layers");
    config.global_ssl_layers = yaml::require_list_int(parsed, "model.init_args.config.global_ssl_layers");
    config.normalize_ssl_features = yaml::require_bool(parsed, "model.init_args.config.normalize_ssl_features");
    config.downsample_factor = yaml::require_int(parsed, "model.init_args.config.downsample_factor");
    config.use_conv_downsample = yaml::require_bool(parsed, "model.init_args.config.use_conv_downsample");
    config.sample_rate = yaml::require_int(parsed, "model.init_args.config.sample_rate");
    config.n_fft = yaml::require_int(parsed, "model.init_args.config.n_fft");
    config.hop_length = yaml::require_int(parsed, "model.init_args.config.hop_length");
    config.use_wave_decoder = yaml::require_bool(parsed, "model.init_args.config.use_wave_decoder");
    config.wave_upsample_factor = yaml::require_int(parsed, "model.init_args.config.wave_upsample_factor");
    config.wave_decoder_dim = yaml::require_int(parsed, "model.init_args.config.wave_decoder_dim");
    config.wave_resnet_num_blocks = yaml::require_int(parsed, "model.init_args.config.wave_resnet_num_blocks");
    config.wave_resnet_num_groups = yaml::require_int(parsed, "model.init_args.config.wave_resnet_num_groups");
    config.wave_upsampler_factors = yaml::require_list_int(parsed, "model.init_args.config.wave_upsampler_factors");
    config.wave_upsampler_kernel_sizes = yaml::require_list_int(parsed, "model.init_args.config.wave_upsampler_kernel_sizes");

    config.local_encoder_dim = yaml::require_int(parsed, "model.init_args.local_encoder.init_args.dim");
    config.local_encoder_layers = yaml::require_int(parsed, "model.init_args.local_encoder.init_args.n_layers");
    config.local_encoder_heads = yaml::require_int(parsed, "model.init_args.local_encoder.init_args.n_heads");
    config.local_encoder_window_size = yaml::require_int(parsed, "model.init_args.local_encoder.init_args.window_size");
    config.local_encoder_max_seq_len = yaml::require_int(parsed, "model.init_args.local_encoder.init_args.max_seq_len");
    config.quantizer_input_dim = yaml::require_int(parsed, "model.init_args.local_quantizer.init_args.input_dim");
    config.quantizer_output_dim = yaml::require_int(parsed, "model.init_args.local_quantizer.init_args.output_dim");
    config.quantizer_levels = yaml::require_list_int(parsed, "model.init_args.local_quantizer.init_args.levels");
    config.codebook_size = compute_codebook_size(config.quantizer_levels);

    config.global_encoder_input_channels = yaml::require_int(parsed, "model.init_args.global_encoder.init_args.input_channels");
    config.global_encoder_output_channels = yaml::require_int(parsed, "model.init_args.global_encoder.init_args.output_channels");
    config.global_encoder_layers = yaml::require_int(parsed, "model.init_args.global_encoder.init_args.num_layers");
    config.global_encoder_dim = yaml::require_int(parsed, "model.init_args.global_encoder.init_args.dim");
    config.global_encoder_intermediate_dim = yaml::require_int(parsed, "model.init_args.global_encoder.init_args.intermediate_dim");

    config.wave_prenet_dim = yaml::require_int(parsed, "model.init_args.wave_prenet.init_args.dim");
    config.wave_prenet_output_dim = yaml::require_int(parsed, "model.init_args.wave_prenet.init_args.output_dim");
    config.wave_prenet_layers = yaml::require_int(parsed, "model.init_args.wave_prenet.init_args.n_layers");
    config.wave_prenet_heads = yaml::require_int(parsed, "model.init_args.wave_prenet.init_args.n_heads");
    config.wave_prenet_window_size = yaml::require_int(parsed, "model.init_args.wave_prenet.init_args.window_size");
    config.wave_prenet_max_seq_len = yaml::require_int(parsed, "model.init_args.wave_prenet.init_args.max_seq_len");

    config.wave_decoder_layers = yaml::require_int(parsed, "model.init_args.wave_decoder.init_args.n_layers");
    config.wave_decoder_heads = yaml::require_int(parsed, "model.init_args.wave_decoder.init_args.n_heads");
    config.wave_decoder_window_size = yaml::require_int(parsed, "model.init_args.wave_decoder.init_args.window_size");
    config.wave_decoder_max_seq_len = yaml::require_int(parsed, "model.init_args.wave_decoder.init_args.max_seq_len");
    config.wave_decoder_condition_dim = yaml::require_int(parsed, "model.init_args.wave_decoder.init_args.adanorm_condition_dim");
    return config;
}

}  // namespace

std::shared_ptr<const MioCodecAssets> load_miocodec_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<MioCodecAssets>();
    assets->resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("miocodec"));
    assets->config = parse_config(assets->resources);
    assets->model_weights = assets->resources.open_tensor_source("weights");
    assets->wavlm_weights = assets->resources.open_tensor_source("wavlm_weights");
    return assets;
}

}  // namespace engine::models::miocodec
