#include "engine/models/fun_asr_nano/assets.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::fun_asr_nano {
namespace json = engine::io::json;
namespace {

FunAsrNanoEncoderConfig parse_encoder(const json::Value &value) {
  FunAsrNanoEncoderConfig config;
  config.num_mel_bins = json::require_i64(value, "num_mel_bins");
  config.num_stacked_frames = json::optional_i64(value, "num_stacked_frames",
                                                 config.num_stacked_frames);
  config.input_size = config.num_mel_bins * config.num_stacked_frames;
  config.d_model = json::require_i64(value, "d_model");
  config.attention_heads = json::require_i64(value, "encoder_attention_heads");
  config.ffn_dim = json::require_i64(value, "encoder_ffn_dim");
  config.layers = json::require_i64(value, "encoder_layers");
  config.timestamp_prediction_layers =
      json::require_i64(value, "num_timestamp_prediction_blocks");
  config.kernel_size = json::require_i64(value, "kernel_size");
  config.max_position_embeddings = json::optional_i64(
      value, "max_position_embeddings", config.max_position_embeddings);
  config.activation =
      json::optional_string(value, "activation_function", config.activation);
  if (config.num_mel_bins <= 0 || config.num_stacked_frames <= 0 ||
      config.d_model <= 0 || config.attention_heads <= 0 ||
      config.ffn_dim <= 0 || config.layers <= 0 ||
      config.timestamp_prediction_layers < 0 || config.kernel_size <= 0) {
    throw std::runtime_error(
        "Fun-ASR-Nano encoder dimensions must be positive");
  }
  if (config.input_size != 560 || config.d_model != 512 ||
      config.attention_heads != 4 || config.ffn_dim != 2048 ||
      config.layers != 50 || config.timestamp_prediction_layers != 20 ||
      config.kernel_size != 11 || config.max_position_embeddings != 2049 ||
      config.activation != "relu") {
    throw std::runtime_error(
        "Fun-ASR-Nano config does not match the published encoder "
        "architecture: 560-wide ReLU encoder required");
  }
  if (config.d_model % config.attention_heads != 0) {
    throw std::runtime_error(
        "Fun-ASR-Nano encoder width must be divisible by attention heads");
  }
  return config;
}

FunAsrNanoTextConfig parse_text(const json::Value &value) {
  FunAsrNanoTextConfig config;
  config.vocab_size = json::require_i64(value, "vocab_size");
  config.hidden_size = json::require_i64(value, "hidden_size");
  config.intermediate_size = json::require_i64(value, "intermediate_size");
  config.layers = json::require_i64(value, "num_hidden_layers");
  config.attention_heads = json::require_i64(value, "num_attention_heads");
  config.key_value_heads = json::require_i64(value, "num_key_value_heads");
  if (config.attention_heads <= 0) {
    throw std::runtime_error("Fun-ASR-Nano text dimensions must be positive");
  }
  config.head_dim = json::optional_i64(
      value, "head_dim", config.hidden_size / config.attention_heads);
  config.max_position_embeddings =
      json::require_i64(value, "max_position_embeddings");
  config.bos_token_id =
      json::optional_i64(value, "bos_token_id", config.bos_token_id);
  config.eos_token_id =
      json::optional_i64(value, "eos_token_id", config.eos_token_id);
  config.rms_norm_eps =
      json::optional_f32(value, "rms_norm_eps", config.rms_norm_eps);
  config.tie_word_embeddings = json::optional_bool(value, "tie_word_embeddings",
                                                   config.tie_word_embeddings);
  const auto *rope = value.find("rope_parameters");
  if (rope != nullptr && rope->is_object()) {
    config.rope_theta =
        json::optional_f32(*rope, "rope_theta", config.rope_theta);
  } else {
    config.rope_theta =
        json::optional_f32(value, "rope_theta", config.rope_theta);
  }
  if (config.hidden_size <= 0 || config.intermediate_size <= 0 ||
      config.layers <= 0 || config.attention_heads <= 0 ||
      config.key_value_heads <= 0 || config.head_dim <= 0 ||
      config.vocab_size <= 0 || config.max_position_embeddings <= 0) {
    throw std::runtime_error("Fun-ASR-Nano text dimensions must be positive");
  }
  if (config.vocab_size != 151936 || config.hidden_size != 1024 ||
      config.intermediate_size != 3072 || config.layers != 28 ||
      config.attention_heads != 16 || config.key_value_heads != 8 ||
      config.head_dim != 128 || config.max_position_embeddings != 40960 ||
      config.rope_theta != 1000000.0F) {
    throw std::runtime_error(
        "Fun-ASR-Nano config does not match the published Qwen architecture");
  }
  if (config.attention_heads % config.key_value_heads != 0) {
    throw std::runtime_error(
        "Fun-ASR-Nano query heads must be divisible by key/value heads");
  }
  return config;
}

FunAsrNanoAdaptorConfig parse_adaptor(const json::Value &root,
                                      const FunAsrNanoTextConfig &text) {
  FunAsrNanoAdaptorConfig config;
  const auto *nested = root.find("adaptor_config");
  if (nested != nullptr && nested->is_object()) {
    config.d_model = json::optional_i64(*nested, "d_model", text.hidden_size);
    config.attention_heads = json::optional_i64(
        *nested, "encoder_attention_heads", config.attention_heads);
    config.ffn_dim =
        json::optional_i64(*nested, "encoder_ffn_dim", text.hidden_size / 4);
    config.layers =
        json::optional_i64(*nested, "encoder_layers", config.layers);
    config.activation = json::optional_string(*nested, "activation_function",
                                              config.activation);
  } else {
    config.d_model = text.hidden_size;
    config.attention_heads = json::optional_i64(
        root, "adaptor_num_attention_heads", config.attention_heads);
    config.ffn_dim = text.hidden_size / 4;
    config.layers =
        json::optional_i64(root, "adaptor_num_hidden_layers", config.layers);
    config.activation =
        json::optional_string(root, "activation_function", config.activation);
  }
  if (config.d_model != text.hidden_size) {
    throw std::runtime_error(
        "Fun-ASR-Nano adaptor width must match the text hidden size");
  }
  if (config.ffn_dim != text.hidden_size / 4) {
    throw std::runtime_error("Fun-ASR-Nano adaptor FFN must equal one quarter "
                             "of the text hidden size");
  }
  if (config.attention_heads != 8 || config.layers != 2 ||
      config.activation != "relu") {
    throw std::runtime_error("Fun-ASR-Nano config does not match the published "
                             "adaptor architecture");
  }
  return config;
}

FunAsrNanoFrontendConfig
parse_frontend(const assets::ResourceBundle &resources,
               const FunAsrNanoEncoderConfig &encoder) {
  const auto processor = resources.parse_json("processor_config");
  const auto *nested = processor.find("feature_extractor");
  const auto &value =
      nested != nullptr && nested->is_object() ? *nested : processor;
  FunAsrNanoFrontendConfig config;
  config.sample_rate =
      json::optional_i32(value, "sampling_rate", config.sample_rate);
  config.feature_size =
      json::optional_i64(value, "feature_size", config.feature_size);
  config.frame_length_ms =
      json::optional_i64(value, "frame_length", config.frame_length_ms);
  config.frame_shift_ms =
      json::optional_i64(value, "frame_shift", config.frame_shift_ms);
  config.lfr_m = json::optional_i64(value, "lfr_m", config.lfr_m);
  config.lfr_n = json::optional_i64(value, "lfr_n", config.lfr_n);
  config.preemphasis =
      json::optional_f32(value, "preemphasis", config.preemphasis);
  if (config.sample_rate != 16000 ||
      config.feature_size != encoder.num_mel_bins ||
      config.lfr_m != encoder.num_stacked_frames ||
      config.frame_length_ms != 25 || config.frame_shift_ms != 10 ||
      config.lfr_n != 6 || config.preemphasis != 0.97F) {
    throw std::runtime_error(
        "Fun-ASR-Nano config does not match the published frontend");
  }
  return config;
}

FunAsrNanoConfig parse_config(const assets::ResourceBundle &resources) {
  const auto root = resources.parse_json("config");
  FunAsrNanoConfig config;
  config.model_type = json::require_string(root, "model_type");
  if (config.model_type != "fun_asr_nano") {
    throw std::runtime_error(
        "Fun-ASR-Nano config has an unexpected model_type");
  }
  config.audio_token_id =
      json::optional_i64(root, "audio_token_id", config.audio_token_id);
  config.encoder = parse_encoder(root.require("encoder_config"));
  config.text = parse_text(root.require("text_config"));
  config.adaptor = parse_adaptor(root, config.text);
  config.projector_hidden_size =
      json::optional_i64(root, "projector_hidden_size",
                         json::optional_i64(root, "adaptor_intermediate_size",
                                            config.projector_hidden_size));
  config.tie_word_embeddings = json::optional_bool(
      root, "tie_word_embeddings", config.text.tie_word_embeddings);
  if (!config.tie_word_embeddings || !config.text.tie_word_embeddings) {
    throw std::runtime_error(
        "Fun-ASR-Nano currently requires tied text embeddings");
  }
  if (config.projector_hidden_size != 2048) {
    throw std::runtime_error("Fun-ASR-Nano projector hidden size must be 2048");
  }
  if (config.audio_token_id < 0) {
    throw std::runtime_error(
        "Fun-ASR-Nano audio token configuration is invalid");
  }
  config.frontend = parse_frontend(resources, config.encoder);
  const auto generation = resources.parse_json("generation_config");
  config.text.bos_token_id =
      json::optional_i64(generation, "bos_token_id", config.text.bos_token_id);
  const auto eos_ids =
      json::require_i64_array_or_scalar(generation, "eos_token_id");
  if (eos_ids.size() != 1) {
    throw std::runtime_error(
        "Fun-ASR-Nano currently requires one EOS token id");
  }
  config.text.eos_token_id = eos_ids.front();
  return config;
}

std::shared_ptr<const FunAsrNanoAssets>
make_assets(assets::ResourceBundle resources) {
  for (const char *id :
       {"config", "generation_config", "processor_config", "tokenizer_json"}) {
    if (!resources.has_file(id)) {
      throw std::runtime_error(
          std::string("Fun-ASR-Nano is missing required resource: ") + id);
    }
  }
  FunAsrNanoAssets assets;
  assets.resources = std::move(resources);
  assets.config = parse_config(assets.resources);
  assets.model_weights = assets.resources.open_tensor_source("weights");
  for (const char *tensor : {
           "model.audio_tower.stem.self_attn.q_proj.weight",
           "model.language_model.embed_tokens.weight",
       }) {
    if (!assets.model_weights->has_tensor(tensor)) {
      throw std::runtime_error(
          std::string("Fun-ASR-Nano is missing required tensor: ") + tensor);
    }
  }
  return std::make_shared<FunAsrNanoAssets>(std::move(assets));
}

} // namespace

std::shared_ptr<const FunAsrNanoAssets>
load_fun_asr_nano_assets(const std::filesystem::path &model_path) {
  return make_assets(engine::model_spec::load_resource_bundle_for_family(
      model_path, "fun_asr_nano"));
}

std::shared_ptr<const FunAsrNanoAssets>
load_fun_asr_nano_assets(assets::ResourceBundle resources) {
  return make_assets(std::move(resources));
}

} // namespace engine::models::fun_asr_nano
