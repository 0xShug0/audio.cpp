#include "engine/community_models/audio8_tts/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <string>

namespace engine::models::audio8_tts {
namespace json = engine::io::json;
namespace {

// configuration_arktts.py — arktts ships one flat config (no nested text_config /
// audio_decoder_config sub-objects like fish_qwen3_omni), so all slow-AR keys sit at
// the root of config.json (AGENTS.md §4.1).
Audio8TtsTextConfig parse_text_config(const json::Value & value) {
    Audio8TtsTextConfig config;
    config.vocab_size = json::require_i64(value, "vocab_size");
    config.n_layer = json::require_i64(value, "n_layer");
    config.dim = json::require_i64(value, "dim");
    config.intermediate_size = json::optional_i64(value, "intermediate_size", config.dim);
    if (config.intermediate_size == config.dim) {
        config.intermediate_size = json::optional_i64(value, "hidden_size", config.dim);
    }
    config.n_head = json::require_i64(value, "n_head");
    config.n_local_heads = json::optional_i64(value, "n_local_heads", config.n_head);
    if (config.n_local_heads == config.n_head) {
        config.n_local_heads = json::optional_i64(value, "num_key_value_heads", config.n_head);
        config.n_local_heads = json::optional_i64(value, "n_local_heads", config.n_local_heads);
    }
    config.head_dim = json::optional_i64(value, "head_dim", 64);
    config.max_seq_len = json::require_i64(value, "max_seq_len");
    config.rope_base = json::optional_f32(value, "rope_base", config.rope_base);
    config.rope_base = json::optional_f32(value, "rope_theta", config.rope_base);
    config.norm_eps = json::optional_f32(value, "norm_eps", config.norm_eps);
    config.tie_word_embeddings = json::optional_bool(value, "tie_word_embeddings", config.tie_word_embeddings);
    config.attention_qk_norm = json::optional_bool(value, "attention_qk_norm", config.attention_qk_norm);
    config.slow_backbone = json::optional_string(value, "slow_backbone", "qwen");
    config.mamba_d_state = json::optional_i64(value, "mamba_d_state", config.mamba_d_state);
    config.mamba_d_conv = json::optional_i64(value, "mamba_d_conv", config.mamba_d_conv);
    config.mamba_expand = json::optional_i64(value, "mamba_expand", config.mamba_expand);
    config.mamba_n_heads = json::optional_i64(value, "mamba_n_heads", config.mamba_n_heads);
    config.mamba_n_groups = json::optional_i64(value, "mamba_n_groups", config.mamba_n_groups);
    config.mamba_d_head = json::optional_i64(value, "mamba_d_head", config.mamba_d_head);
    config.mamba_d_ssm = json::optional_i64(value, "mamba_d_ssm", config.mamba_d_ssm);
    config.mamba_chunk_size = json::optional_i64(value, "mamba_chunk_size", config.mamba_chunk_size);
    config.embedding_multiplier = json::optional_f32(value, "embedding_multiplier", config.embedding_multiplier);
    config.lm_head_multiplier = json::optional_f32(value, "lm_head_multiplier", config.lm_head_multiplier);
    config.attention_in_multiplier = json::optional_f32(value, "attention_in_multiplier", config.attention_in_multiplier);
    config.attention_out_multiplier = json::optional_f32(value, "attention_out_multiplier", config.attention_out_multiplier);
    config.ssm_in_multiplier = json::optional_f32(value, "ssm_in_multiplier", config.ssm_in_multiplier);
    config.ssm_out_multiplier = json::optional_f32(value, "ssm_out_multiplier", config.ssm_out_multiplier);
    config.key_multiplier = json::optional_f32(value, "key_multiplier", config.key_multiplier);
    {
        auto v = json::optional_f32_array(value, "ssm_multipliers");
        if (!v.empty()) config.ssm_multipliers = v;
    }
    {
        auto v = json::optional_f32_array(value, "mlp_multipliers");
        if (!v.empty()) config.mlp_multipliers = v;
    }
    engine::io::require_positive(config.vocab_size, "text vocab_size");
    engine::io::require_positive(config.n_layer, "text n_layer");
    engine::io::require_positive(config.dim, "text dim");
    engine::io::require_positive(config.n_head, "text n_head");
    engine::io::require_positive(config.n_local_heads, "text n_local_heads");
    engine::io::require_positive(config.head_dim, "text head_dim");
    engine::io::require_positive(config.max_seq_len, "text max_seq_len");
    engine::io::require_divisible(config.n_head, config.n_local_heads, "text n_head / n_local_heads");
    return config;
}

// configuration_arktts.py — fast-AR keys carry a fast_ prefix at the flat config root;
// the fast vocabulary spans one codebook (codebook_size), not the text vocab, and the
// checkpoint ships an untied fast_output.weight regardless of the slow LM head tying.
Audio8TtsFastConfig parse_fast_config(const json::Value & value) {
    Audio8TtsFastConfig config;
    config.vocab_size = json::require_i64(value, "codebook_size");
    config.num_codebooks = json::require_i64(value, "num_codebooks");
    config.n_layer = json::require_i64(value, "n_fast_layer");
    config.dim = json::require_i64(value, "fast_dim");
    config.intermediate_size = json::require_i64(value, "fast_intermediate_size");
    config.n_head = json::require_i64(value, "fast_n_head");
    config.n_local_heads = json::optional_i64(value, "fast_n_local_heads", config.n_head);
    config.head_dim = json::require_i64(value, "fast_head_dim");
    config.max_seq_len = json::optional_i64(value, "fast_max_seq_len", config.num_codebooks + 1);
    config.rope_base = json::optional_f32(value, "rope_base", config.rope_base);
    config.norm_eps = json::optional_f32(value, "norm_eps", config.norm_eps);
    config.tie_word_embeddings = json::optional_bool(value, "fast_tie_word_embeddings", false);
    config.attention_qk_norm = json::optional_bool(value, "fast_attention_qk_norm", config.attention_qk_norm);
    engine::io::require_positive(config.vocab_size, "fast vocab_size");
    engine::io::require_positive(config.num_codebooks, "fast num_codebooks");
    engine::io::require_positive(config.n_layer, "fast n_layer");
    engine::io::require_positive(config.dim, "fast dim");
    engine::io::require_positive(config.intermediate_size, "fast intermediate_size");
    engine::io::require_positive(config.n_head, "fast n_head");
    engine::io::require_positive(config.n_local_heads, "fast n_local_heads");
    engine::io::require_positive(config.head_dim, "fast head_dim");
    engine::io::require_divisible(config.n_head, config.n_local_heads, "fast n_head / n_local_heads");
    return config;
}

// configuration_arktts.py — arktts ships one flat config.json; semantic token ids are
// named semantic_begin_id / semantic_end_id there (AGENTS.md §4.1).
Audio8TtsConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    Audio8TtsConfig config;
    config.model_type = json::optional_string(root, "model_type", "");
    if (config.model_type != "arktts") {
        throw std::runtime_error("Audio8 TTS model_type mismatch");
    }
    config.torch_dtype = json::optional_string(root, "torch_dtype", config.torch_dtype);
    config.semantic_start_token_id = json::require_i64(root, "semantic_begin_id");
    config.semantic_end_token_id = json::require_i64(root, "semantic_end_id");
    config.im_end_token_id = json::require_i64(root, "eos_token_id");
    config.norm_fastlayer_input = json::optional_bool(root, "norm_fastlayer_input", false);
    config.text = parse_text_config(root);
    config.fast = parse_fast_config(root);
    config.codec.total_codebooks = config.fast.num_codebooks;
    if (config.fast.dim != config.text.dim) {
        throw std::runtime_error("Audio8 TTS fast dim must match text dim");
    }
    if (!config.text.tie_word_embeddings) {
        throw std::runtime_error("Audio8 TTS expects tied text embeddings");
    }
    if (config.semantic_start_token_id <= 0 || config.semantic_end_token_id < config.semantic_start_token_id) {
        throw std::runtime_error("Audio8 TTS semantic token range is invalid");
    }
    return config;
}

// model.safetensors stores QKV pre-packed per layer as wqkv (+ a bias row on slow
// layers only) — AGENTS.md §4.2; anchors pin that layout before graph building.
// Supports both Qwen (0.6B/1.0B: embeddings.weight, layers.*) and Falcon-H1/Mamba
// (0.1B: slow.embed_tokens.weight, slow.layers.*.mamba/self_attn).
void validate_weight_anchors(const Audio8TtsAssets & assets) {
    const bool is_qwen = assets.model_weights->has_tensor("embeddings.weight");
    const bool is_mamba = assets.model_weights->has_tensor("slow.embed_tokens.weight");
    if (!is_qwen && !is_mamba) {
        throw std::runtime_error("Audio8 TTS model_weights missing embeddings (expected embeddings.weight or slow.embed_tokens.weight)");
    }
    assets.model_weights->require_metadata("codebook_embeddings.weight");
    if (is_qwen) {
        assets.model_weights->require_metadata("layers.0.attention.wqkv.weight");
        assets.model_weights->require_metadata("layers.0.attention.wqkv.bias");
        assets.model_weights->require_metadata("layers.0.attention.wo.weight");
    } else {
        assets.model_weights->require_metadata("slow.layers.0.mamba.in_proj.weight");
        assets.model_weights->require_metadata("slow.layers.0.self_attn.q_proj.weight");
        assets.model_weights->require_metadata("slow.final_layernorm.weight");
    }
    assets.model_weights->require_metadata("fast_layers.0.attention.wqkv.weight");
    assets.model_weights->require_metadata("fast_embeddings.weight");
    assets.model_weights->require_metadata("fast_output.weight");
    assets.codec_weights->require_metadata("quantizer.semantic_quantizer.quantizers.0.codebook.weight");
    assets.codec_weights->require_metadata("decoder.model.0.conv.weight");
}

}  // namespace

std::shared_ptr<const Audio8TtsAssets> load_audio8_tts_assets(const std::filesystem::path & model_path) {
    Audio8TtsAssets assets;
    assets.resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("audio8_tts"));
    assets.config = parse_config(assets.resources);
    assets.model_weights = assets.resources.open_tensor_source("model_weights");
    assets.codec_weights = assets.resources.open_tensor_source("codec_weights");
    validate_weight_anchors(assets);
    return std::make_shared<Audio8TtsAssets>(std::move(assets));
}

}  // namespace engine::models::audio8_tts
