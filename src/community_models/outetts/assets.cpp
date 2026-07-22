#include "engine/community_models/outetts/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"
#include "engine/models/qwen3_asr/assets.h"

#include <stdexcept>

namespace engine::models::outetts {
namespace json = engine::io::json;
namespace {

OuteTTSConfig parse_config(const assets::ResourceBundle &resources) {
  const auto root = resources.parse_json("config");
  if (json::require_string(root, "model_type") != "llama") {
    throw std::runtime_error("OuteTTS expects a LlamaForCausalLM checkpoint");
  }
  OuteTTSConfig out;
  out.bos_token_id = json::require_i64(root, "bos_token_id");
  out.eos_token_id = json::require_i64(root, "eos_token_id");
  out.hidden_size = json::require_i64(root, "hidden_size");
  out.intermediate_size = json::require_i64(root, "intermediate_size");
  out.max_position_embeddings =
      json::require_i64(root, "max_position_embeddings");
  out.num_attention_heads = json::require_i64(root, "num_attention_heads");
  out.num_hidden_layers = json::require_i64(root, "num_hidden_layers");
  out.num_key_value_heads = json::require_i64(root, "num_key_value_heads");
  out.head_dim = json::optional_i64(root, "head_dim",
                                    out.hidden_size / out.num_attention_heads);
  out.vocab_size = json::require_i64(root, "vocab_size");
  out.rms_norm_eps = json::optional_f32(root, "rms_norm_eps", out.rms_norm_eps);
  out.rope_theta = json::optional_f32(root, "rope_theta", out.rope_theta);
  const auto &rope = root.require("rope_scaling");
  if (json::require_string(rope, "rope_type") != "llama3") {
    throw std::runtime_error("OuteTTS expects llama3 rope scaling");
  }
  out.rope_scaling.factor = json::require_f32(rope, "factor");
  out.rope_scaling.low_freq_factor = json::require_f32(rope, "low_freq_factor");
  out.rope_scaling.high_freq_factor =
      json::require_f32(rope, "high_freq_factor");
  out.rope_scaling.original_max_position_embeddings =
      json::require_i64(rope, "original_max_position_embeddings");

  const auto generation = resources.parse_json("generation_config");
  out.pad_token_id = json::require_i64(generation, "pad_token_id");
  engine::io::require_positive(out.hidden_size, "OuteTTS hidden_size");
  engine::io::require_positive(out.intermediate_size,
                               "OuteTTS intermediate_size");
  engine::io::require_positive(out.num_hidden_layers, "OuteTTS layer count");
  engine::io::require_positive(out.num_attention_heads,
                               "OuteTTS attention heads");
  engine::io::require_positive(out.num_key_value_heads, "OuteTTS KV heads");
  engine::io::require_divisible(out.num_attention_heads,
                                out.num_key_value_heads,
                                "OuteTTS attention heads");
  if (out.num_attention_heads * out.head_dim != out.hidden_size) {
    throw std::runtime_error(
        "OuteTTS hidden size does not match attention heads times head_dim");
  }
  return out;
}

OuteTTSGenerationConfig
parse_generation(const assets::ResourceBundle &resources) {
  const auto root = resources.parse_json("generation_config");
  OuteTTSGenerationConfig out;
  out.temperature = json::optional_f32(root, "temperature", out.temperature);
  out.repetition_penalty =
      json::optional_f32(root, "repetition_penalty", out.repetition_penalty);
  out.top_k = json::optional_i64(root, "top_k", out.top_k);
  out.top_p = json::optional_f32(root, "top_p", out.top_p);
  out.min_p = json::optional_f32(root, "min_p", out.min_p);
  return out;
}

void validate_anchors(const OuteTTSAssets &assets) {
  const auto &c = assets.config;
  const auto &model = *assets.model_weights;
  assets::require_tensor_shape(model, "model.embed_tokens.weight",
                               {c.vocab_size, c.hidden_size});
  assets::require_tensor_shape(model, "model.norm.weight", {c.hidden_size});
  assets::require_tensor_shape(
      model, "model.layers.0.self_attn.q_proj.weight",
      {c.num_attention_heads * c.head_dim, c.hidden_size});
  assets::require_tensor_shape(
      model, "model.layers.0.self_attn.k_proj.weight",
      {c.num_key_value_heads * c.head_dim, c.hidden_size});
  assets::require_tensor_shape(model, "model.layers.0.mlp.gate_proj.weight",
                               {c.intermediate_size, c.hidden_size});

  const auto &dac = *assets.dac_weights;
  assets::require_tensor_shape(dac, "quantizer.quantizers.0.codebook.weight",
                               {c.codebook_size, 8});
  assets::require_tensor_shape(dac, "quantizer.quantizers.1.codebook.weight",
                               {c.codebook_size, 8});
  assets::require_tensor_shape(dac, "quantizer.quantizers.0.in_proj.weight_v",
                               {8, c.dac_latent_dim, 1});
  assets::require_tensor_shape(dac, "quantizer.quantizers.1.in_proj.weight_v",
                               {8, c.dac_latent_dim, 1});
  assets::require_tensor_shape(dac, "encoder.block.0.weight_v", {64, 1, 7});
  assets::require_tensor_shape(dac, "encoder.block.6.weight_v",
                               {c.dac_latent_dim, c.dac_latent_dim, 3});
  assets::require_tensor_shape(dac, "decoder.model.0.weight_v",
                               {c.dac_decoder_dim, c.dac_latent_dim, 7});
  assets::require_tensor_shape(dac, "decoder.model.6.weight_v", {1, 96, 7});
}

std::shared_ptr<const engine::models::qwen3_asr::Qwen3ASRAssets>
load_embedded_aligner(const assets::ResourceBundle &resources) {
  if (!resources.has_file("aligner_config"))
    return nullptr;
  for (const char *id :
       {"aligner_generation_config", "aligner_tokenizer_config"}) {
    if (!resources.has_file(id)) {
      throw std::runtime_error(
          std::string("OuteTTS embedded aligner is missing resource: ") + id);
    }
  }
  if (!resources.has_file("aligner_preprocessor_config") &&
      !resources.has_file("aligner_processor_config")) {
    throw std::runtime_error(
        "OuteTTS embedded aligner is missing its processor configuration");
  }
  if (!(resources.has_file("aligner_tokenizer_json") ||
        (resources.has_file("aligner_vocab") &&
         resources.has_file("aligner_merges")))) {
    throw std::runtime_error(
        "OuteTTS embedded aligner is missing tokenizer resources");
  }

  const auto source = resources.open_tensor_source("aligner_weights");
  if (source->tensors().empty()) {
    throw std::runtime_error(
        "OuteTTS embedded aligner has no aligner_weights tensors");
  }
  assets::ResourceBundle aligner(resources.model_root());
  const auto add_required = [&](const char *target, const char *source_id) {
    aligner.add_file(target, resources.require_file(source_id));
  };
  const auto add_optional = [&](const char *target, const char *source_id) {
    if (resources.has_file(source_id))
      aligner.add_file(target, resources.require_file(source_id));
  };
  add_required("config", "aligner_config");
  add_required("generation_config", "aligner_generation_config");
  add_required("tokenizer_config", "aligner_tokenizer_config");
  add_optional("preprocessor_config", "aligner_preprocessor_config");
  add_optional("processor_config", "aligner_processor_config");
  add_optional("chat_template", "aligner_chat_template");
  add_optional("chat_template_jinja", "aligner_chat_template_jinja");
  add_optional("vocab", "aligner_vocab");
  add_optional("merges", "aligner_merges");
  add_optional("tokenizer_json", "aligner_tokenizer_json");
  aligner.add_tensor_source(
      "weights", resources.require_file("aligner_weights"), "aligner_weights");
  return engine::models::qwen3_asr::load_qwen3_asr_assets(std::move(aligner));
}

} // namespace

std::shared_ptr<const OuteTTSAssets>
load_outetts_assets(const std::filesystem::path &model_path) {
  auto resources = engine::model_spec::load_resource_bundle(
      model_path, engine::model_spec::default_spec_path("outetts"));
  OuteTTSAssets model_assets;
  model_assets.config = parse_config(resources);
  model_assets.generation = parse_generation(resources);
  model_assets.model_weights = resources.open_tensor_source("model_weights");
  model_assets.dac_weights = resources.open_tensor_source("dac_weights");
  model_assets.embedded_aligner = load_embedded_aligner(resources);
  model_assets.resources = std::move(resources);
  validate_anchors(model_assets);
  return std::make_shared<OuteTTSAssets>(std::move(model_assets));
}

} // namespace engine::models::outetts
