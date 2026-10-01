#include "engine/models/index_echo/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <utility>

namespace engine::models::index_echo {
namespace {

namespace json = engine::io::json;

qwen3_asr::Qwen3ASRAudioEncoderConfig audio_config(const json::Value & value) {
    qwen3_asr::Qwen3ASRAudioEncoderConfig config;
    config.num_mel_bins = json::require_i64(value, "num_mel_bins");
    config.encoder_layers = json::require_i64(value, "encoder_layers");
    config.encoder_attention_heads = json::require_i64(value, "encoder_attention_heads");
    config.encoder_ffn_dim = json::require_i64(value, "encoder_ffn_dim");
    config.d_model = json::require_i64(value, "d_model");
    config.max_source_positions = json::require_i64(value, "max_source_positions");
    config.n_window = json::require_i64(value, "n_window");
    config.n_window_infer = json::require_i64(value, "n_window_infer");
    config.conv_chunksize = json::require_i64(value, "conv_chunksize");
    config.downsample_hidden_size = json::require_i64(value, "downsample_hidden_size");
    config.output_dim = json::require_i64(value, "output_dim");
    config.activation_function = json::require_string(value, "activation_function");
    if (config.activation_function != "gelu" || config.num_mel_bins != 128) {
        throw std::runtime_error("Index-Echo requires a 128-mel GELU Qwen3-Omni audio tower");
    }
    return config;
}

modules::Qwen35DecoderConfig text_config(const json::Value & llm) {
    const auto * nested = llm.find("text_config");
    const auto & value = nested != nullptr ? *nested : llm;
    modules::Qwen35DecoderConfig config;
    config.vocab_size = json::require_i64(value, "vocab_size");
    config.hidden_size = json::require_i64(value, "hidden_size");
    config.intermediate_size = json::require_i64(value, "intermediate_size");
    config.layers = json::require_i64(value, "num_hidden_layers");
    config.heads = json::require_i64(value, "num_attention_heads");
    config.kv_heads = json::require_i64(value, "num_key_value_heads");
    config.head_dim = json::require_i64(value, "head_dim");
    config.full_attention_interval = json::require_i64(value, "full_attention_interval");
    config.linear_conv_kernel_dim = json::require_i64(value, "linear_conv_kernel_dim");
    config.linear_key_head_dim = json::require_i64(value, "linear_key_head_dim");
    config.linear_num_key_heads = json::require_i64(value, "linear_num_key_heads");
    config.linear_num_value_heads = json::require_i64(value, "linear_num_value_heads");
    config.linear_value_head_dim = json::require_i64(value, "linear_value_head_dim");
    config.rms_norm_eps = json::optional_f32(value, "rms_norm_eps", config.rms_norm_eps);
    const auto & rope = value.require("rope_parameters");
    config.rope_theta = json::require_f32(rope, "rope_theta");
    config.partial_rotary_factor = json::require_f32(rope, "partial_rotary_factor");
    config.weight_prefix = nested != nullptr ? "model.language_model" : "model";
    config.lm_head_prefix = "lm_head";
    config.tie_word_embeddings = json::optional_bool(value, "tie_word_embeddings", false);
    config.decode_cache_steps = 4096;
    if (config.hidden_size != 2048 && config.hidden_size != 4096) {
        throw std::runtime_error("Index-Echo supports 2B and 9B Qwen3.5 text decoders");
    }
    return config;
}

}  // namespace

std::shared_ptr<const IndexEchoAssets> load_index_echo_assets(const std::filesystem::path & model_path) {
    auto out = std::make_shared<IndexEchoAssets>();
    out->resources = engine::model_spec::load_resource_bundle_for_family(model_path, "index_echo");
    const auto audio = audio_config(out->resources.parse_json("audio_config"));
    const auto llm = out->resources.parse_json("llm_config");
    const auto text = text_config(llm);

    auto qwen3_audio = std::make_shared<qwen3_asr::Qwen3ASRAssets>();
    qwen3_audio->config.audio_encoder = audio;
    qwen3_audio->config.frontend.sample_rate = 16000;
    qwen3_audio->config.frontend.feature_size = audio.num_mel_bins;
    qwen3_audio->config.frontend.hop_length = 160;
    qwen3_audio->config.frontend.n_fft = 400;
    qwen3_audio->model_weights = out->resources.open_tensor_source("audio_weights");
    out->qwen3_omni_audio = std::move(qwen3_audio);

    out->qwen35_config = text;
    out->qwen35_weights = out->resources.open_tensor_source("stlm_weights");
    out->connector_weights = out->resources.open_tensor_source("connector_weights");
    out->audio_hidden_size = audio.output_dim;
    out->text_hidden_size = text.hidden_size;
    return out;
}

}  // namespace engine::models::index_echo
