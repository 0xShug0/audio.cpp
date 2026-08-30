#include "engine/community_models/sopro_tts/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::community_models::sopro_tts {
namespace {

namespace json = engine::io::json;

constexpr const char * kFamily = "sopro_tts";

SoproModelConfig parse_model(const json::Value & root) {
    SoproModelConfig out;
    const auto * node = root.find("model");
    if (node == nullptr) {
        throw std::runtime_error("Sopro config.json is missing the \"model\" section");
    }
    const auto & model = *node;
    out.latent_dim = json::optional_i64(model, "latent_dim", out.latent_dim);
    out.semantic_vocab_size = json::optional_i64(model, "semantic_vocab_size", out.semantic_vocab_size);
    out.text_vocab_size = json::optional_i64(model, "text_vocab_size", out.text_vocab_size);
    out.max_text_len = json::optional_i64(model, "max_text_len", out.max_text_len);
    out.cond_in_dim = json::optional_i64(model, "cond_in_dim", out.cond_in_dim);
    out.cond_hidden_dim = json::optional_i64(model, "cond_hidden_dim", out.cond_hidden_dim);
    out.ar_model_dim = json::optional_i64(model, "ar_model_dim", out.ar_model_dim);
    out.ar_blocks = json::optional_i64(model, "ar_blocks", out.ar_blocks);
    out.ar_heads = json::optional_i64(model, "ar_heads", out.ar_heads);
    // ModelConfig.__post_init__: ar_kv_heads defaults to ar_heads.
    out.ar_kv_heads = json::optional_nullable_i64(model, "ar_kv_heads", out.ar_heads);
    out.ar_ffn_mult = json::optional_f32(model, "ar_ffn_mult", out.ar_ffn_mult);
    out.ar_qk_rms_norm = json::optional_bool(model, "ar_qk_rms_norm", out.ar_qk_rms_norm);
    out.style_prefix_tokens = json::optional_i64(model, "style_prefix_tokens", out.style_prefix_tokens);
    out.acoustic_time_embed_dim = json::optional_i64(model, "acoustic_time_embed_dim", out.acoustic_time_embed_dim);
    out.acoustic_sway_sampling_coef = json::optional_f32(model, "acoustic_sway_sampling_coef", out.acoustic_sway_sampling_coef);
    out.acoustic_upsampler_kernel_size = json::optional_i64(model, "acoustic_upsampler_kernel_size", out.acoustic_upsampler_kernel_size);
    out.acoustic_dit_dim = json::optional_i64(model, "acoustic_dit_dim", out.acoustic_dit_dim);
    out.acoustic_dit_depth = json::optional_i64(model, "acoustic_dit_depth", out.acoustic_dit_depth);
    out.acoustic_dit_heads = json::optional_i64(model, "acoustic_dit_heads", out.acoustic_dit_heads);
    out.acoustic_dit_dim_head = json::optional_i64(model, "acoustic_dit_dim_head", out.acoustic_dit_dim_head);
    out.acoustic_dit_ff_mult = json::optional_f32(model, "acoustic_dit_ff_mult", out.acoustic_dit_ff_mult);
    out.acoustic_spk_dim = json::optional_i64(model, "acoustic_spk_dim", out.acoustic_spk_dim);
    out.acoustic_pre_lookahead_frames = json::optional_i64(model, "acoustic_pre_lookahead_frames", out.acoustic_pre_lookahead_frames);
    out.acoustic_pos_kernel_size = json::optional_i64(model, "acoustic_pos_kernel_size", out.acoustic_pos_kernel_size);
    out.acoustic_sigma_min = json::optional_f32(model, "acoustic_sigma_min", out.acoustic_sigma_min);
    out.acoustic_num_left_chunks = json::optional_i64(model, "acoustic_num_left_chunks", out.acoustic_num_left_chunks);
    out.acoustic_mel_n_mels = json::optional_i64(model, "acoustic_mel_n_mels", out.acoustic_mel_n_mels);
    out.acoustic_mel_hop_length = json::optional_i64(model, "acoustic_mel_hop_length", out.acoustic_mel_hop_length);
    // ModelConfig.__post_init__: acoustic_mu_dim defaults to acoustic_mel_n_mels.
    out.acoustic_mu_dim = json::optional_nullable_i64(model, "acoustic_mu_dim", out.acoustic_mel_n_mels);
    out.acoustic_mel_mean = json::optional_f32_array(model, "acoustic_mel_mean");
    out.acoustic_mel_std = json::optional_f32_array(model, "acoustic_mel_std");
    if (static_cast<int64_t>(out.acoustic_mel_mean.size()) != out.acoustic_mel_n_mels ||
        static_cast<int64_t>(out.acoustic_mel_std.size()) != out.acoustic_mel_n_mels) {
        throw std::runtime_error(
            "Sopro config.json must provide acoustic_mel_mean/acoustic_mel_std with "
            "acoustic_mel_n_mels entries");
    }
    if (out.ar_model_dim % out.ar_heads != 0) {
        throw std::runtime_error("Sopro ar_model_dim must be divisible by ar_heads");
    }
    return out;
}

SoproSemanticEncoderConfig parse_semantic_encoder(const json::Value & root) {
    SoproSemanticEncoderConfig out;
    const auto * node = root.find("semantic_encoder");
    if (node == nullptr) {
        return out;
    }
    const auto & cfg = *node;
    out.n_mels = json::optional_i64(cfg, "n_mels", out.n_mels);
    out.d_model = json::optional_i64(cfg, "d_model", out.d_model);
    out.layers = json::optional_i64(cfg, "layers", out.layers);
    out.heads = json::optional_i64(cfg, "heads", out.heads);
    out.ffn_dim = json::optional_i64(cfg, "ffn_dim", out.ffn_dim);
    out.max_positions = json::optional_i64(cfg, "max_positions", out.max_positions);
    out.fsq_levels = json::optional_i64_array(cfg, "fsq_levels", out.fsq_levels);
    out.sample_rate = json::optional_i64(cfg, "sample_rate", out.sample_rate);
    out.n_fft = json::optional_i64(cfg, "n_fft", out.n_fft);
    out.hop_length = json::optional_i64(cfg, "hop_length", out.hop_length);
    out.token_samples_24k = json::optional_i64(cfg, "token_samples_24k", out.token_samples_24k);
    if (out.fsq_levels.empty()) {
        throw std::runtime_error("Sopro semantic_encoder.fsq_levels must not be empty");
    }
    return out;
}

SoproSpeakerEncoderConfig parse_speaker_encoder(const json::Value & root) {
    SoproSpeakerEncoderConfig out;
    const auto * node = root.find("speaker_encoder");
    if (node == nullptr) {
        return out;
    }
    const auto & cfg = *node;
    out.sample_rate = json::optional_i64(cfg, "sample_rate", out.sample_rate);
    out.n_mels = json::optional_i64(cfg, "n_mels", out.n_mels);
    out.n_fft = json::optional_i64(cfg, "n_fft", out.n_fft);
    out.win_length = json::optional_i64(cfg, "win_length", out.win_length);
    out.hop_length = json::optional_i64(cfg, "hop_length", out.hop_length);
    out.f_min = json::optional_f32(cfg, "f_min", out.f_min);
    out.f_max = json::optional_f32(cfg, "f_max", out.f_max);
    out.mel_log_floor = json::optional_f32(cfg, "mel_log_floor", out.mel_log_floor);
    out.stem_channels = json::optional_i64(cfg, "stem_channels", out.stem_channels);
    out.stage_channels = json::optional_i64_array(cfg, "stage_channels", out.stage_channels);
    out.blocks_per_stage = json::optional_i64_array(cfg, "blocks_per_stage", out.blocks_per_stage);
    out.dilation_cycle = json::optional_i64_array(cfg, "dilation_cycle", out.dilation_cycle);
    out.depthwise_kernel_size = json::optional_i64(cfg, "depthwise_kernel_size", out.depthwise_kernel_size);
    out.se_reduction = json::optional_i64(cfg, "se_reduction", out.se_reduction);
    out.id_emb_dim = json::optional_i64(cfg, "id_emb_dim", out.id_emb_dim);
    out.style_emb_dim = json::optional_i64(cfg, "style_emb_dim", out.style_emb_dim);
    out.style_ctrl_dim = json::optional_i64(cfg, "style_ctrl_dim", out.style_ctrl_dim);
    out.id_head_hidden = json::optional_i64(cfg, "id_head_hidden", out.id_head_hidden);
    out.style_head_hidden = json::optional_i64(cfg, "style_head_hidden", out.style_head_hidden);
    out.attn_hidden = json::optional_i64(cfg, "attn_hidden", out.attn_hidden);
    if (out.stage_channels.size() != out.blocks_per_stage.size() || out.stage_channels.empty()) {
        throw std::runtime_error(
            "Sopro speaker_encoder.stage_channels and blocks_per_stage must be "
            "non-empty and the same length");
    }
    if (out.dilation_cycle.empty()) {
        throw std::runtime_error("Sopro speaker_encoder.dilation_cycle must not be empty");
    }
    return out;
}

SoproVocoderConfig parse_vocoder(const json::Value & root, const std::string & key) {
    SoproVocoderConfig out;
    const auto * node = root.find(key);
    if (node == nullptr) {
        return out;
    }
    const auto & cfg = *node;
    out.sample_rate = json::optional_i64(cfg, "sample_rate", out.sample_rate);
    out.n_fft = json::optional_i64(cfg, "n_fft", out.n_fft);
    out.hop_length = json::optional_i64(cfg, "hop_length", out.hop_length);
    out.n_mels = json::optional_i64(cfg, "n_mels", out.n_mels);
    out.dim = json::optional_i64(cfg, "dim", out.dim);
    out.intermediate_dim = json::optional_i64(cfg, "intermediate_dim", out.intermediate_dim);
    out.num_layers = json::optional_i64(cfg, "num_layers", out.num_layers);
    out.max_magnitude = json::optional_f32(cfg, "max_magnitude", out.max_magnitude);
    out.causal = json::optional_bool(cfg, "causal", out.causal);
    out.lookahead_frames = json::optional_nullable_i64(cfg, "lookahead_frames", out.lookahead_frames);
    out.block_lookaheads = json::optional_i64_array(cfg, "block_lookaheads");
    return out;
}

SoproGenerationConfig parse_generation(const json::Value & root) {
    SoproGenerationConfig out;
    const auto * node = root.find("generation");
    if (node == nullptr) {
        return out;
    }
    const auto & cfg = *node;
    out.temperature = json::optional_f32(cfg, "temperature", out.temperature);
    out.top_p = json::optional_f32(cfg, "top_p", out.top_p);
    out.top_k = json::optional_i64(cfg, "top_k", out.top_k);
    out.steps = json::optional_i64(cfg, "steps", out.steps);
    out.max_seconds = json::optional_f32(cfg, "max_seconds", out.max_seconds);
    out.min_seconds = json::optional_f32(cfg, "min_seconds", out.min_seconds);
    out.max_segment_chars = json::optional_i64(cfg, "max_segment_chars", out.max_segment_chars);
    out.ref_seconds = json::optional_f32(cfg, "ref_seconds", out.ref_seconds);
    out.style_tokens = json::optional_i64(cfg, "style_tokens", out.style_tokens);
    out.prompt_tokens = json::optional_i64(cfg, "prompt_tokens", out.prompt_tokens);
    out.stream_chunk_frames = json::optional_i64(cfg, "stream_chunk_frames", out.stream_chunk_frames);
    return out;
}

SoproTTSConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    SoproTTSConfig out;
    out.sample_rate = json::optional_i64(root, "sample_rate", out.sample_rate);
    out.model = parse_model(root);
    out.semantic_encoder = parse_semantic_encoder(root);
    out.speaker_encoder = parse_speaker_encoder(root);
    out.vocoder = parse_vocoder(root, "vocoder");
    out.vocoder_streaming = parse_vocoder(root, "vocoder_streaming");
    out.generation = parse_generation(root);
    const int64_t codebook = out.semantic_encoder.codebook_size();
    if (codebook != out.model.semantic_vocab_size) {
        throw std::runtime_error(
            "Sopro config mismatch: prod(semantic_encoder.fsq_levels) = " +
            std::to_string(codebook) + " but model.semantic_vocab_size = " +
            std::to_string(out.model.semantic_vocab_size));
    }
    if (out.semantic_encoder.token_samples_24k % out.model.acoustic_mel_hop_length != 0) {
        throw std::runtime_error(
            "Sopro config mismatch: semantic_encoder.token_samples_24k must be a "
            "multiple of model.acoustic_mel_hop_length");
    }
    return out;
}

}  // namespace

int64_t SoproModelConfig::ar_ffn_dim() const noexcept {
    // SwiGLUFeedForward: hidden = max(1, round(mult * dim)).
    const auto hidden = static_cast<int64_t>(
        std::llround(static_cast<double>(ar_ffn_mult) * static_cast<double>(ar_model_dim)));
    return hidden < 1 ? 1 : hidden;
}

int64_t SoproModelConfig::acoustic_dit_ff_dim() const noexcept {
    const auto hidden = static_cast<int64_t>(
        std::llround(static_cast<double>(acoustic_dit_ff_mult) * static_cast<double>(acoustic_dit_dim)));
    return hidden < 1 ? 1 : hidden;
}

int64_t SoproSemanticEncoderConfig::digit_dim() const noexcept {
    int64_t sum = 0;
    for (const int64_t level : fsq_levels) {
        sum += level;
    }
    return sum;
}

int64_t SoproSemanticEncoderConfig::codebook_size() const noexcept {
    int64_t product = 1;
    for (const int64_t level : fsq_levels) {
        product *= level;
    }
    return product;
}

int64_t SoproTTSConfig::hop_ratio() const noexcept {
    return semantic_encoder.token_samples_24k / model.acoustic_mel_hop_length;
}

void require_frontend_buffers(
    const assets::TensorSource & source,
    const char * stage,
    std::initializer_list<const char *> tensor_names) {
    for (const char * name : tensor_names) {
        if (!source.has_tensor(name)) {
            throw std::runtime_error(
                std::string("Sopro ") + stage + " checkpoint is missing '" + name +
                "'. audio.cpp reuses torchaudio's stored analysis window and mel "
                "filterbank rather than rebuilding them; re-export the checkpoint "
                "with persistent buffers.");
        }
    }
}

std::shared_ptr<const SoproTTSAssets> load_sopro_tts_assets(
    const std::filesystem::path & model_path) {
    auto assets = std::make_shared<SoproTTSAssets>();
    assets->resources = engine::model_spec::load_resource_bundle(
        model_path, engine::model_spec::default_spec_path(kFamily));
    assets->config = parse_config(assets->resources);
    assets->model_weights = assets->resources.open_tensor_source("model");
    assets->semantic_encoder_weights = assets->resources.open_tensor_source("semantic_encoder");
    assets->speaker_encoder_weights = assets->resources.open_tensor_source("speaker_encoder");
    assets->vocoder_weights = assets->resources.open_tensor_source("vocoder");
    assets->tokenizer_path = assets->resources.require_file("tokenizer");
    return assets;
}

}  // namespace engine::community_models::sopro_tts
