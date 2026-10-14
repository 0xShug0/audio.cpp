#include "engine/community_models/fish_drama3/assets.h"

#include "engine/framework/io/config.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <utility>

namespace engine::models::fish_drama3 {
namespace json = engine::io::json;

std::shared_ptr<const Drama3Assets> load_drama3_assets(const std::filesystem::path & model_path) {
    auto out = std::make_shared<Drama3Assets>();
    out->resources = engine::model_spec::load_resource_bundle_for_family(model_path, "fish_drama3");
    const auto root = out->resources.parse_json("config");
    if (json::require_string(root, "model_type") != "fish_omni" ||
        json::require_string(root, "speech_template_version") != "reference-bracket-v1") {
        throw std::runtime_error("Fish Drama 3 requires a fish_omni reference-bracket-v1 checkpoint");
    }
    const auto & text = root.require("text_config");
    auto & b = out->backbone;
    b.vocab_size = json::require_i64(text, "vocab_size");
    b.hidden_size = json::require_i64(text, "hidden_size");
    b.intermediate_size = json::require_i64(text, "moe_intermediate_size");
    if (json::require_i64(text, "shared_expert_intermediate_size") != b.intermediate_size) {
        throw std::runtime_error("Drama 3 shared and routed expert widths must match");
    }
    b.experts = json::require_i64(text, "num_experts");
    b.experts_per_token = json::require_i64(text, "num_experts_per_tok");
    b.layers = json::require_i64(text, "num_hidden_layers");
    b.heads = json::require_i64(text, "num_attention_heads");
    b.kv_heads = json::require_i64(text, "num_key_value_heads");
    b.head_dim = json::require_i64(text, "head_dim");
    b.linear_conv_kernel_dim = json::require_i64(text, "linear_conv_kernel_dim");
    b.linear_key_head_dim = json::require_i64(text, "linear_key_head_dim");
    b.linear_value_head_dim = json::require_i64(text, "linear_value_head_dim");
    b.linear_num_key_heads = json::require_i64(text, "linear_num_key_heads");
    b.linear_num_value_heads = json::require_i64(text, "linear_num_value_heads");
    b.rms_norm_eps = json::require_f32(text, "rms_norm_eps");
    b.rope_theta = json::require_f32(text.require("rope_parameters"), "rope_theta");
    b.partial_rotary_factor = json::require_f32(text.require("rope_parameters"), "partial_rotary_factor");
    b.tie_word_embeddings = json::optional_bool(text, "tie_word_embeddings", false);
    const auto & layer_types = text.require("layer_types").as_array();
    if (layer_types.size() != static_cast<size_t>(b.layers)) throw std::runtime_error("Drama 3 layer type count mismatch");
    for (int64_t layer = 0; layer < b.layers; ++layer) {
        const auto expected = (layer + 1) % 4 == 0 ? "full_attention" : "linear_attention";
        if (layer_types[layer].as_string() != expected) throw std::runtime_error("unsupported Drama 3 hybrid layer layout");
    }
    if (b.experts <= 0 || b.experts_per_token <= 0 || b.experts_per_token > b.experts ||
        !json::optional_bool(text, "norm_topk_prob", true) || !json::require_bool(text, "layernorm_zero_centered_gamma")) {
        throw std::runtime_error("unsupported Drama 3 expert or normalization configuration");
    }
    const auto & audio = root.require("audio_decoder_config");
    auto & t = out->tower;
    t.hidden_size = json::require_i64(audio, "dim");
    t.intermediate_size = json::require_i64(audio, "intermediate_size");
    t.heads = json::require_i64(audio, "n_head");
    t.kv_heads = json::require_i64(audio, "n_local_heads");
    t.head_dim = json::require_i64(audio, "head_dim");
    t.codebook_size = json::require_i64(audio, "vocab_size");
    t.shared_layers = json::require_i64(audio, "shared_enrichment_layers");
    t.residual_layers = json::require_i64(audio, "num_codebooks") - 1;
    t.backbone_codebooks = json::require_i64(audio, "vq_backbone_codebooks");
    t.norm_eps = json::require_f32(audio, "norm_eps");
    t.rope_theta = json::require_f32(audio, "rope_base");
    t.rotary_fraction = json::require_f32(audio, "rotary_percent");
    t.semantic_begin = json::require_i32(root, "semantic_start_token_id");
    out->eos = json::require_i32(root, "eos_token_id");
    if (t.hidden_size != b.hidden_size || t.residual_layers != 9 ||
        json::require_i64(root, "semantic_end_token_id") + 1 - t.semantic_begin != t.codebook_size ||
        !json::require_bool(audio, "rotary_interleaved") ||
        !json::require_bool(audio, "layernorm_zero_centered_gamma") ||
        !json::require_bool(audio, "attention_output_gate") ||
        json::require_string(audio, "decoder_type") != "codebook_mtp_shared") {
        throw std::runtime_error("unsupported Drama 3 tower or semantic range");
    }
    out->model_weights = out->resources.open_tensor_source("model_weights");
    out->codec_weights = out->resources.open_tensor_source("codec_weights");
    (void) out->model_weights->require_metadata("model.language_model.layers.0.mlp.experts.gate_proj.weight");
    (void) out->model_weights->require_metadata("audio_decoder_mtp_shared.layers.0.eh_proj.weight");
    const auto & residual_codebook = out->codec_weights->require_metadata("quantizer.quantizer.quantizers.0.codebook.weight");
    out->codec.residual_codebook_size = residual_codebook.shape[0];
    auto tokenizer = std::make_shared<fish_audio::FishAudioAssets>();
    tokenizer->resources = out->resources;
    tokenizer->config.semantic_start_token_id = t.semantic_begin;
    tokenizer->config.semantic_end_token_id = t.semantic_begin + t.codebook_size - 1;
    tokenizer->config.im_end_token_id = out->eos;
    out->tokenizer_assets = std::move(tokenizer);
    return out;
}

}  // namespace engine::models::fish_drama3
