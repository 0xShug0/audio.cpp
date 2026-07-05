#include "engine/models/parakeet_tdt/assets.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/json.h"
#include "engine/framework/tokenizers/hf_tokenizer_json.h"

#include <utility>

namespace engine::models::parakeet_tdt {

std::shared_ptr<const ParakeetAssets> load_parakeet_assets(const std::filesystem::path & model_root) {
    assets::ResourceBundle resources(model_root);
    resources.add_model_files({
        {"config", "config.json"},
        {"processor", "processor_config.json"},
        {"tokenizer", "tokenizer.json"},
        {"weights", "model.safetensors"},
    });

    auto assets = std::make_shared<ParakeetAssets>();
    assets->model_config = load_parakeet_model_config(resources.require_file("config"));
    assets->feature_config = load_parakeet_feature_config(resources.require_file("processor"));
    assets->tokenizer = load_parakeet_tokenizer(resources.require_file("tokenizer"));
    assets->model_weights = resources.open_tensor_source("weights");
    return assets;
}

ParakeetModelConfig load_parakeet_model_config(const std::filesystem::path & path) {
    const engine::io::json::Value root = engine::io::json::parse_file(path);
    ParakeetModelConfig config;
    config.model_type = root.require("model_type").as_string();
    config.variant = root.find("architectures") != nullptr && !root.require("architectures").as_array().empty()
        ? root.require("architectures").as_array().front().as_string()
        : "ParakeetForTDT";
    config.blank_token_id = root.require("blank_token_id").as_i64();
    config.pad_token_id = root.require("pad_token_id").as_i64();
    config.vocab_size = root.require("vocab_size").as_i64();
    config.decoder_hidden_size = root.require("decoder_hidden_size").as_i64();
    config.num_decoder_layers = root.require("num_decoder_layers").as_i64();
    config.max_symbols_per_step = root.require("max_symbols_per_step").as_i64();
    config.durations = engine::io::json::number_array_as<int32_t>(root.require("durations"));

    const engine::io::json::Value & encoder = root.require("encoder_config");
    config.encoder.hidden_size = encoder.require("hidden_size").as_i64();
    config.encoder.intermediate_size = encoder.require("intermediate_size").as_i64();
    config.encoder.num_attention_heads = encoder.require("num_attention_heads").as_i64();
    config.encoder.num_key_value_heads = encoder.require("num_key_value_heads").as_i64();
    config.encoder.num_hidden_layers = encoder.require("num_hidden_layers").as_i64();
    config.encoder.num_mel_bins = encoder.require("num_mel_bins").as_i64();
    config.encoder.max_position_embeddings = encoder.require("max_position_embeddings").as_i64();
    config.encoder.conv_kernel_size = encoder.require("conv_kernel_size").as_i64();
    config.encoder.subsampling_factor = encoder.require("subsampling_factor").as_i64();
    config.encoder.subsampling_conv_channels = encoder.require("subsampling_conv_channels").as_i64();
    config.encoder.subsampling_conv_kernel_size = encoder.require("subsampling_conv_kernel_size").as_i64();
    config.encoder.subsampling_conv_stride = encoder.require("subsampling_conv_stride").as_i64();
    config.encoder.attention_bias = encoder.require("attention_bias").as_bool();
    config.encoder.convolution_bias = encoder.require("convolution_bias").as_bool();
    config.encoder.scale_input = encoder.require("scale_input").as_bool();
    config.encoder.hidden_act = encoder.require("hidden_act").as_string();
    return config;
}

ParakeetFeatureExtractorConfig load_parakeet_feature_config(const std::filesystem::path & path) {
    const engine::io::json::Value root = engine::io::json::parse_file(path);
    const engine::io::json::Value & feature = root.require("feature_extractor");
    ParakeetFeatureExtractorConfig config;
    config.feature_size = feature.require("feature_size").as_i64();
    config.sample_rate = feature.require("sampling_rate").as_i64();
    config.n_fft = feature.require("n_fft").as_i64();
    config.win_length = feature.require("win_length").as_i64();
    config.hop_length = feature.require("hop_length").as_i64();
    config.preemphasis = feature.require("preemphasis").as_f32();
    config.return_attention_mask = feature.require("return_attention_mask").as_bool();
    return config;
}

std::shared_ptr<engine::tokenizers::HuggingFaceTokenizerJson> load_parakeet_tokenizer(const std::filesystem::path & path) {
    return engine::tokenizers::load_huggingface_tokenizer_json(path);
}

}  // namespace engine::models::parakeet_tdt
