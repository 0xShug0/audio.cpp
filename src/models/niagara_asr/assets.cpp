#include "engine/models/niagara_asr/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <utility>

namespace engine::models::niagara_asr {
namespace json = engine::io::json;
namespace {

std::string lmuformer_prefix(int64_t layer) {
    if (layer == 0) {
        return "p__torch_params_encoder_lmuformer_";
    }
    return "p__torch_params_encoder_lmuformer_" + std::to_string(layer) + "_";
}

std::string indexed_layer_name(
    const std::string & prefix,
    const std::string & base,
    int64_t layer,
    bool omit_zero) {
    if (layer == 0 && omit_zero) {
        return prefix + base;
    }
    return prefix + base + "_" + std::to_string(layer);
}

int64_t count_lmuformer_layers(const assets::TensorSource & source) {
    int64_t layers = source.has_tensor(lmuformer_prefix(0) + "ffn_dense_1_kernel") ? 1 : 0;
    for (int64_t layer = 1; source.has_tensor(lmuformer_prefix(layer) + "ffn_dense_1_kernel"); ++layer) {
        ++layers;
    }
    return layers;
}

NiagaraEncoderConfig infer_encoder_config(
    const assets::TensorSource & source,
    int64_t vocab_size,
    int64_t feature_size) {
    NiagaraEncoderConfig config;
    const auto dense = source.require_metadata("p__torch_params_encoder_dense_kernel");
    if (dense.shape.size() != 2) {
        throw std::runtime_error("Niagara ASR encoder dense kernel must be rank 2");
    }
    config.hidden_size = dense.shape[1];
    if (config.hidden_size <= 0 || dense.shape[0] % config.hidden_size != 0) {
        throw std::runtime_error("Niagara ASR encoder dense kernel has invalid hidden size");
    }
    config.dense_input_features = dense.shape[0] / config.hidden_size;
    if (feature_size <= 0 || config.dense_input_features <= 0) {
        throw std::runtime_error("Niagara ASR encoder dense input shape is invalid");
    }

    config.num_layers = count_lmuformer_layers(source);
    if (config.num_layers <= 0) {
        throw std::runtime_error("Niagara ASR is missing LMUFormer layers");
    }

    const auto ffn = source.require_metadata(lmuformer_prefix(0) + "ffn_dense_1_kernel");
    if (ffn.shape.size() != 2 || ffn.shape[0] != config.hidden_size) {
        throw std::runtime_error("Niagara ASR FFN kernel shape does not match hidden size");
    }
    config.intermediate_size = ffn.shape[1];

    const auto state = source.require_metadata(lmuformer_prefix(0) + "state_space_state_space_c");
    if (state.shape.size() != 3 || state.shape[2] != 1) {
        throw std::runtime_error("Niagara ASR state-space C tensor has unsupported shape");
    }
    if (state.shape[0] != config.hidden_size && state.shape[0] != config.hidden_size * 2) {
        throw std::runtime_error("Niagara ASR state-space channel count must be hidden or 2 * hidden");
    }
    config.state_channels = state.shape[0];
    config.state_size = state.shape[1];

    assets::require_tensor_shape(source, "p__torch_params_encoder_dense_bias", {config.hidden_size});
    assets::require_tensor_shape(source, "p__torch_params_encoder_subsampling_mask_aware_conv2d_kernel", {1, 1, 1, 4});
    assets::require_tensor_shape(source, "p__torch_params_encoder_subsampling_mask_aware_conv2d_1_kernel",
        {4, 3, 4, config.hidden_size});
    assets::require_tensor_shape(source, "p__torch_params_encoder_subsampling_mask_aware_conv2d_2_kernel",
        {4, 3, config.hidden_size, config.hidden_size});
    assets::require_tensor_shape(source, "p__torch_params_encoder_subsampling_mask_aware_conv2d_1_bias", {config.hidden_size});
    assets::require_tensor_shape(source, "p__torch_params_encoder_subsampling_mask_aware_conv2d_2_bias", {config.hidden_size});
    assets::require_tensor_shape(source, "p__torch_params_shared_dec_bias", {vocab_size + 1});
    assets::require_tensor_shape(source, "p__torch_params_shared_dec_kernel", {config.hidden_size, vocab_size + 1});

    for (int64_t layer = 0; layer < config.num_layers; ++layer) {
        const auto prefix = lmuformer_prefix(layer);
        const auto self_attention_dense =
            indexed_layer_name(prefix, "self_attention_dense", layer + 1, false);
        const auto global_attention =
            indexed_layer_name(prefix, "self_attention_global_attention", layer, true);
        assets::require_tensor_shape(source, prefix + "ffn_dense_1_kernel", {config.hidden_size, config.intermediate_size});
        assets::require_tensor_shape(source, prefix + "ffn_dense_1_bias", {config.intermediate_size});
        assets::require_tensor_shape(source, prefix + "ffn_dense_2_kernel", {config.intermediate_size, config.hidden_size});
        assets::require_tensor_shape(source, prefix + "ffn_dense_2_bias", {config.hidden_size});
        assets::require_tensor_shape(source, prefix + "ffn_1_dense_1_kernel", {config.hidden_size, config.intermediate_size});
        assets::require_tensor_shape(source, prefix + "ffn_1_dense_1_bias", {config.intermediate_size});
        assets::require_tensor_shape(source, prefix + "ffn_1_dense_2_kernel", {config.intermediate_size, config.hidden_size});
        assets::require_tensor_shape(source, prefix + "ffn_1_dense_2_bias", {config.hidden_size});
        assets::require_tensor_shape(source, prefix + "self_attention_query_kernel", {config.hidden_size, config.hidden_size});
        assets::require_tensor_shape(source, prefix + "self_attention_key_kernel", {config.hidden_size, config.hidden_size});
        assets::require_tensor_shape(source, prefix + "self_attention_value_kernel", {config.hidden_size, config.hidden_size});
        assets::require_tensor_shape(source, self_attention_dense + "_kernel", {config.hidden_size, config.hidden_size});
        assets::require_tensor_shape(source, global_attention + "_kernel", {config.hidden_size, config.hidden_size});
        assets::require_tensor_shape(source, prefix + "state_space_dense_1_kernel", {config.hidden_size, config.state_channels});
        assets::require_tensor_shape(source, prefix + "state_space_dense_1_bias", {config.state_channels});
        assets::require_tensor_shape(source, prefix + "state_space_dense_2_kernel", {config.hidden_size, config.hidden_size});
        assets::require_tensor_shape(source, prefix + "state_space_dense_2_bias", {config.hidden_size});
        assets::require_tensor_shape(source, prefix + "state_space_state_space_c", {config.state_channels, config.state_size, 1});
        assets::require_tensor_shape(source, prefix + "state_space_state_space_d", {config.state_channels, 1});
    }

    return config;
}

NiagaraAsrConfig parse_config(
    const assets::ResourceBundle & resources,
    const assets::TensorSource & source) {
    const auto root = resources.parse_json("config");
    NiagaraAsrConfig config;
    config.model_type = json::optional_string(root, "model_type", "niagara_asr");
    config.vocab_size = json::optional_i64(root, "vocab_size", 256);
    config.blank_id = config.vocab_size;

    if (resources.has_file("preprocessor_config")) {
        const auto preproc = resources.parse_json("preprocessor_config");
        config.frontend.sample_rate = json::optional_i64(preproc, "sampling_rate",
            json::optional_i64(preproc, "sample_rate", 16000));
        config.frontend.n_fft = json::optional_i64(preproc, "n_fft", 400);
        config.frontend.win_length = json::optional_i64(preproc, "win_length", 400);
        config.frontend.hop_length = json::optional_i64(preproc, "hop_length", 160);
        config.frontend.n_mels = json::optional_i64(preproc, "feature_size",
            json::optional_i64(preproc, "num_mel_bins", 80));
    }

    if (config.frontend.sample_rate != 16000 || config.frontend.n_fft <= 0 ||
        config.frontend.win_length <= 0 || config.frontend.hop_length <= 0 ||
        config.frontend.n_mels != 80 || config.vocab_size <= 0) {
        throw std::runtime_error("Niagara ASR config contains unsupported frontend/model values");
    }
    config.encoder = infer_encoder_config(source, config.vocab_size, config.frontend.n_mels);
    return config;
}

}  // namespace

std::shared_ptr<const NiagaraAsrAssets> load_niagara_asr_assets(
    const std::filesystem::path & model_path) {
    auto resources = engine::model_spec::load_resource_bundle_for_family(model_path, "niagara_asr");
    auto assets_out = std::make_shared<NiagaraAsrAssets>();
    assets_out->resources = std::move(resources);
    assets_out->source = assets_out->resources.open_tensor_source("weights");
    if (assets_out->source == nullptr) {
        throw std::runtime_error("Niagara ASR is missing GGUF weights");
    }
    assets_out->config = parse_config(assets_out->resources, *assets_out->source);
    assets_out->tokenizer_pieces =
        tokenizers::load_sentencepiece_model(assets_out->resources.require_file("tokenizer_spm"));
    if (static_cast<int64_t>(assets_out->tokenizer_pieces.size()) != assets_out->config.vocab_size) {
        throw std::runtime_error("Niagara ASR tokenizer size does not match config vocab_size");
    }
    return assets_out;
}

}  // namespace engine::models::niagara_asr
