#include "engine/models/plapre/assets.h"

#include "engine/framework/model_spec/package.h"

#include <cmath>
#include <stdexcept>

namespace engine::models::plapre {

std::shared_ptr<const PlapreAssets> load_plapre_assets(const std::filesystem::path & path) {
    auto assets = std::make_shared<PlapreAssets>();
    assets->resources = model_spec::load_resource_bundle_for_family(path, "plapre");
    assets->ar_config = PlapreLlamaConfig::from_json(assets->resources.parse_json("config"));
    const auto codec = assets->resources.parse_json("kanade_config");
    const auto & config = codec.require("config");
    const auto & prenet = codec.require("mel_prenet").require("init_args");
    const auto & decoder = codec.require("mel_decoder").require("init_args");
    const auto & postnet = codec.require("mel_postnet").require("init_args");
    auto & mel = assets->mel_config;
    mel.hidden = prenet.require("dim").as_i64();
    mel.heads = prenet.require("n_heads").as_i64();
    mel.layers = prenet.require("n_layers").as_i64();
    mel.window_size = prenet.require("window_size").as_i64();
    mel.rope_theta = prenet.require("rope_theta").as_f32();
    mel.speaker_dim = decoder.require("adanorm_condition_dim").as_i64();
    mel.mel_bins = config.require("n_mels").as_i64();
    mel.upsample_factor = config.require("mel_upsample_factor").as_i64();
    mel.postnet_channels = postnet.require("channels").as_i64();
    mel.postnet_layers = postnet.require("num_layers").as_i64();
    mel.postnet_kernel = postnet.require("kernel_size").as_i64();
    const auto weights = assets->resources.open_tensor_source("kanade");
    mel.intermediate = weights->require_metadata("mel_prenet.layers.0.feed_forward.w1.weight").shape.at(0);
    mel.levels.clear();
    for (const auto & value : codec.require("local_quantizer").require("init_args").require("levels").as_array()) {
        mel.levels.push_back(value.as_i64());
    }
    auto & encoder = assets->encoder_config;
    const auto & local = codec.require("local_encoder").require("init_args");
    const auto & global = codec.require("global_encoder").require("init_args");
    encoder.local.hidden = local.require("dim").as_i64();
    encoder.local.heads = local.require("n_heads").as_i64();
    encoder.local.layers = local.require("n_layers").as_i64();
    encoder.local.window_size = local.require("window_size").as_i64();
    encoder.local.rope_theta = local.require("rope_theta").as_f32();
    encoder.local.intermediate = weights->require_metadata("local_encoder.layers.0.feed_forward.w1.weight").shape.at(0);
    encoder.local.speaker_dim = global.require("output_channels").as_i64();
    encoder.global_channels = global.require("dim").as_i64();
    encoder.global_intermediate = global.require("intermediate_dim").as_i64();
    encoder.global_layers = global.require("num_layers").as_i64();
    encoder.attention_channels = weights->require_metadata("global_encoder.pooling.attn.0.weight").shape.at(0);
    encoder.downsample_factor = config.require("downsample_factor").as_i64();
    encoder.normalize_ssl_features = config.require("normalize_ssl_features").as_bool();
    encoder.local_ssl_layers.clear();
    encoder.global_ssl_layers.clear();
    for (const auto & value : config.require("local_ssl_layers").as_array()) encoder.local_ssl_layers.push_back(value.as_i64());
    for (const auto & value : config.require("global_ssl_layers").as_array()) encoder.global_ssl_layers.push_back(value.as_i64());
    encoder.wavlm.output_hidden_layer = codec.require("ssl_feature_extractor").require("init_args").require("output_layer").as_i64();
    encoder.wavlm.num_hidden_layers = encoder.wavlm.output_hidden_layer;
    encoder.levels = mel.levels;
    if (config.require("sample_rate").as_i64() != 24000 || config.require("hop_length").as_i64() != 480 ||
        config.require("downsample_factor").as_i64() != 2 || mel.upsample_factor != 2 ||
        config.require("padding").as_string() != "same" || config.require("vocoder_name").as_string() != "hift" ||
        config.require("mel_interpolation_mode").as_string() != "linear" ||
        decoder.require("dim").as_i64() != mel.hidden || decoder.require("n_heads").as_i64() != mel.heads ||
        decoder.require("n_layers").as_i64() != mel.layers || decoder.require("window_size").as_i64() != mel.window_size ||
        decoder.require("rope_theta").as_f32() != mel.rope_theta ||
        !decoder.require("use_adaln_zero").as_bool() || !postnet.require("use_layer_norm").as_bool() ||
        mel.levels != std::vector<int64_t>({8, 8, 8, 5, 5})) {
        throw std::runtime_error("unsupported Plapre Kanade codec configuration");
    }
    const auto speakers = assets->resources.parse_json("speakers");
    for (const auto & [name, values] : speakers.as_object()) {
        auto & speaker = assets->speakers[name];
        for (const auto & value : values.as_array()) {
            const float x = value.as_f32();
            if (!std::isfinite(x)) throw std::runtime_error("Plapre speaker embedding is nonfinite");
            speaker.push_back(x);
        }
        if (speaker.size() != static_cast<size_t>(mel.speaker_dim)) {
            throw std::runtime_error("Plapre speaker embedding has the wrong dimension");
        }
    }
    auto & hift = assets->hift_config;
    hift.in_channels = mel.mel_bins;
    hift.base_channels = 512;
    hift.nb_harmonics = 8;
    hift.sampling_rate = 24000;
    hift.upsample_rates = {8, 5, 3};
    hift.upsample_kernel_sizes = {16, 11, 7};
    hift.istft_n_fft = 16;
    hift.istft_hop = 4;
    hift.resblock_kernel_sizes = {3, 7, 11};
    hift.resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    hift.source_resblock_kernel_sizes = {7, 7, 11};
    hift.source_resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    hift.f0_num_class = 1;
    hift.f0_in_channels = mel.mel_bins;
    hift.f0_cond_channels = 512;
    hift.f0_condnet_kernel_sizes = {3, 3, 3, 3, 3};
    hift.weight_layout = modules::HiftVocoderWeightLayout::TorchParametrizedWeightNorm;
    hift.source_mode = modules::HiftVocoderSourceMode::SineGen2;
    return assets;
}

}  // namespace engine::models::plapre
