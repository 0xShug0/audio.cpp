#include "engine/models/firered_audio/assets.h"

#include "engine/framework/io/config.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <string>

namespace engine::models::firered_audio {
namespace {

namespace json = engine::io::json;

constexpr const char * kFamily = "firered_audio";

std::filesystem::path find_gguf_path(const engine::assets::ResourceBundle & resources) {
    for (const auto & file : resources.files()) {
        if (file.path.extension() == ".gguf") {
            return file.path;
        }
    }
    return {};
}

FireRedAudioBackboneConfig parse_backbone_config(const json::Value & root) {
    const auto & cfg = root.require("backbone_config");
    FireRedAudioBackboneConfig out;
    out.vocab_size = json::optional_i64(cfg, "vocab_size", out.vocab_size);
    out.hidden_size = json::optional_i64(cfg, "hidden_size", out.hidden_size);
    out.intermediate_size = json::optional_i64(cfg, "intermediate_size", out.intermediate_size);
    out.layers = json::optional_i64(cfg, "num_hidden_layers", out.layers);
    out.heads = json::optional_i64(cfg, "num_attention_heads", out.heads);
    out.kv_heads = json::optional_i64(cfg, "num_key_value_heads", out.kv_heads);
    out.head_dim = json::optional_i64(cfg, "head_dim", out.head_dim);
    out.full_attention_interval = json::optional_i64(cfg, "full_attention_interval", out.full_attention_interval);
    out.linear_conv_kernel_dim = json::optional_i64(cfg, "linear_conv_kernel_dim", out.linear_conv_kernel_dim);
    out.linear_key_head_dim = json::optional_i64(cfg, "linear_key_head_dim", out.linear_key_head_dim);
    out.linear_num_key_heads = json::optional_i64(cfg, "linear_num_key_heads", out.linear_num_key_heads);
    out.linear_num_value_heads = json::optional_i64(cfg, "linear_num_value_heads", out.linear_num_value_heads);
    out.linear_value_head_dim = json::optional_i64(cfg, "linear_value_head_dim", out.linear_value_head_dim);
    out.rms_norm_eps = json::optional_f32(cfg, "rms_norm_eps", out.rms_norm_eps);
    out.rope_theta = json::optional_f32(cfg, "rope_theta", out.rope_theta);
    if (const auto * rope = cfg.find("rope_parameters"); rope != nullptr && rope->is_object()) {
        out.rope_theta = json::optional_f32(*rope, "rope_theta", out.rope_theta);
        out.partial_rotary_factor = json::optional_f32(*rope, "partial_rotary_factor", out.partial_rotary_factor);
    }
    engine::io::require_positive(out.hidden_size, "FireRedAudio backbone hidden size");
    engine::io::require_positive(out.layers, "FireRedAudio backbone layers");
    if (!(out.partial_rotary_factor > 0.0F && out.partial_rotary_factor <= 1.0F)) {
        throw std::runtime_error("FireRedAudio backbone partial_rotary_factor must be in (0, 1]");
    }
    return out;
}

FireRedAudioPatchEncoderConfig parse_patch_encoder_config(const json::Value & root) {
    const auto & cfg = root.require("patch_encoder_config");
    FireRedAudioPatchEncoderConfig out;
    out.vae_dim = json::optional_i64(cfg, "vae_dim", out.vae_dim);
    out.out_dim = json::optional_i64(cfg, "out_dim", out.out_dim);
    out.patch_size = json::optional_i64(cfg, "patch_size", out.patch_size);
    out.hidden_size = json::optional_i64(cfg, "hidden_size", out.hidden_size);
    out.layers = json::optional_i64(cfg, "depth", out.layers);
    out.heads = json::optional_i64(cfg, "num_heads", out.heads);
    const int64_t mlp_ratio = json::optional_i64(cfg, "mlp_ratio", 4);
    out.intermediate_size = out.hidden_size * mlp_ratio;
    engine::io::require_positive(out.patch_size, "FireRedAudio patch size");
    return out;
}

FireRedAudioFlowConfig parse_flow_config(const json::Value & root) {
    const auto & cfg = root.require("dit_config");
    FireRedAudioFlowConfig out;
    out.vae_channels = json::optional_i64(cfg, "vae_channels", out.vae_channels);
    out.backbone_hidden_size = json::optional_i64(cfg, "backbone_hidden_size", out.backbone_hidden_size);
    out.hidden_size = json::optional_i64(cfg, "hidden_size", out.hidden_size);
    out.layers = json::optional_i64(cfg, "depth", out.layers);
    out.heads = json::optional_i64(cfg, "num_heads", out.heads);
    out.patch_size = json::optional_i64(cfg, "patch_size", out.patch_size);
    out.history_patches = json::optional_i64(cfg, "history_patches", out.history_patches);
    const float mlp_ratio = json::optional_f32(cfg, "mlp_ratio", 4.0F);
    out.intermediate_size = static_cast<int64_t>(static_cast<float>(out.hidden_size) * mlp_ratio);
    engine::io::require_positive(out.hidden_size, "FireRedAudio flow hidden size");
    return out;
}

FireRedAudioAudioEncoderConfig parse_audio_encoder_config(const json::Value & root) {
    const auto & cfg = root.require("audio_encoder_config");
    FireRedAudioAudioEncoderConfig out;
    out.num_mel_bins = json::optional_i64(cfg, "num_mel_bins", out.num_mel_bins);
    out.d_model = json::optional_i64(cfg, "d_model", out.d_model);
    out.encoder_layers = json::optional_i64(cfg, "encoder_layers", out.encoder_layers);
    out.encoder_attention_heads = json::optional_i64(cfg, "encoder_attention_heads", out.encoder_attention_heads);
    out.encoder_ffn_dim = json::optional_i64(cfg, "encoder_ffn_dim", out.encoder_ffn_dim);
    out.max_source_positions = json::optional_i64(cfg, "max_source_positions", out.max_source_positions);
    out.n_window = json::optional_i64(cfg, "n_window", out.n_window);
    out.output_dim = json::optional_i64(cfg, "output_dim", out.output_dim);
    if (const auto * processor = root.find("processor_config"); processor != nullptr && processor->is_object()) {
        if (const auto * feature = processor->find("feature_extractor"); feature != nullptr && feature->is_object()) {
            out.sample_rate = json::optional_i64(*feature, "sampling_rate", out.sample_rate);
            out.n_fft = json::optional_i64(*feature, "n_fft", out.n_fft);
            out.hop_length = json::optional_i64(*feature, "hop_length", out.hop_length);
            out.num_mel_bins = json::optional_i64(*feature, "feature_size", out.num_mel_bins);
        }
    }
    engine::io::require_positive(out.sample_rate, "FireRedAudio audio encoder sample rate");
    engine::io::require_positive(out.num_mel_bins, "FireRedAudio audio encoder mel bins");
    engine::io::require_positive(out.d_model, "FireRedAudio audio encoder hidden size");
    engine::io::require_positive(out.encoder_layers, "FireRedAudio audio encoder layers");
    engine::io::require_positive(out.encoder_attention_heads, "FireRedAudio audio encoder heads");
    if (out.d_model % out.encoder_attention_heads != 0) {
        throw std::runtime_error("FireRedAudio audio encoder hidden size must divide by heads");
    }
    return out;
}

FireRedAudioRedAeConfig parse_redae_config(const json::Value & root) {
    const auto & cfg = root.require("red_vae_config");
    FireRedAudioRedAeConfig out;
    out.sample_rate = json::optional_i64(cfg, "audio_sample_rate", out.sample_rate);
    out.audio_patch_size = json::optional_i64(cfg, "audio_patch_size", out.audio_patch_size);
    out.bottleneck_dim = json::optional_i64(cfg, "out_dim", out.bottleneck_dim);
    out.enc_hidden_size = json::optional_i64(cfg, "hidden_size", out.enc_hidden_size);
    out.enc_intermediate_size = json::optional_i64(cfg, "intermediate_size", out.enc_intermediate_size);
    out.enc_layers = json::optional_i64(cfg, "num_hidden_layers", out.enc_layers);
    out.enc_heads = json::optional_i64(cfg, "num_attention_heads", out.enc_heads);
    out.enc_kv_heads = json::optional_i64(cfg, "num_key_value_heads", out.enc_kv_heads);
    out.enc_extra_downsample_rate = json::optional_i64(cfg, "extra_downsample_rate", out.enc_extra_downsample_rate);
    out.enc_downsample_layers = json::optional_i64(cfg, "downsample_num_hidden_layers", out.enc_downsample_layers);
    const bool use_sliding_window = json::optional_bool(cfg, "use_sliding_window", false);
    const int64_t max_window_layers = json::optional_i64(cfg, "max_window_layers", out.enc_layers);
    if (use_sliding_window) {
        if (max_window_layers != 0) {
            throw std::runtime_error("FireRedAudio RedAE supports only all-layer sliding attention");
        }
        out.enc_sliding_window = json::optional_i64(cfg, "sliding_window", 0);
        out.dec_sliding_window = out.enc_sliding_window;
    }
    out.enc_head_dim = json::optional_i64(cfg, "head_dim", out.enc_head_dim);
    out.dec_hidden_size = out.enc_hidden_size;
    out.dec_intermediate_size = out.enc_intermediate_size;
    out.dec_layers = out.enc_layers;
    out.dec_heads = out.enc_heads;
    out.dec_kv_heads = out.enc_kv_heads;
    out.dec_head_dim = out.enc_head_dim;
    engine::io::require_positive(out.audio_patch_size, "FireRedAudio RedAE audio patch size");
    return out;
}

FireRedAudioSpecialTokens parse_special_tokens(const json::Value & root) {
    FireRedAudioSpecialTokens out;
    out.sosp = static_cast<int32_t>(json::optional_i64(root, "sosp_idx", out.sosp));
    out.eosp = static_cast<int32_t>(json::optional_i64(root, "eosp_idx", out.eosp));
    out.audio = static_cast<int32_t>(json::optional_i64(root, "audio_special_token_id", out.audio));
    out.audio_no_latent = static_cast<int32_t>(json::optional_i64(root, "audio_special_no_latent_id", out.audio_no_latent));
    return out;
}

void validate_tensor_shapes(const FireRedAudioAssets & assets) {
    const auto & model = *assets.model_weights;
    const auto & decoder = *assets.redae_decoder_weights;
    engine::assets::require_tensor_shape(
        model,
        "audio_encoder.conv1.weight",
        {assets.audio_encoder.d_model, assets.audio_encoder.num_mel_bins, 3});
    engine::assets::require_tensor_shape(
        model,
        "audio_encoder.adapter.linear2.weight",
        {assets.audio_encoder.output_dim, assets.audio_encoder.output_dim});
    engine::assets::require_tensor_shape(
        model,
        "backbone_llm.model.language_model.embed_tokens.weight",
        {assets.backbone.vocab_size, assets.backbone.hidden_size});
    engine::assets::require_tensor_shape(
        model,
        "patch_encoder.in_proj.0.weight",
        {assets.patch_encoder.hidden_size, assets.patch_encoder.vae_dim});
    engine::assets::require_tensor_shape(
        model,
        "dit.backbone_input_proj.weight",
        {assets.flow.hidden_size, assets.flow.backbone_hidden_size});
    engine::assets::require_tensor_shape(
        model,
        "red_vae.in_proj.0.weight",
        {assets.redae.enc_hidden_size, assets.redae.audio_patch_size});
    engine::assets::require_tensor_shape(
        decoder,
        "decoder.in_proj.weight",
        {assets.redae.enc_extra_downsample_rate * assets.redae.dec_hidden_size, assets.redae.bottleneck_dim});
}

}  // namespace

std::shared_ptr<const FireRedAudioAssets> load_firered_audio_assets(
    const std::filesystem::path & model_path) {
    auto assets = std::make_shared<FireRedAudioAssets>();
    assets->resources = engine::model_spec::load_resource_bundle_for_family(model_path, kFamily);
    assets->model_root = assets->resources.model_root();
    assets->gguf_path = find_gguf_path(assets->resources);
    const auto root = assets->resources.parse_json("config");
    assets->backbone = parse_backbone_config(root);
    assets->audio_encoder = parse_audio_encoder_config(root);
    assets->patch_encoder = parse_patch_encoder_config(root);
    assets->flow = parse_flow_config(root);
    assets->redae = parse_redae_config(root);
    assets->special_tokens = parse_special_tokens(root);
    assets->model_weights = assets->resources.open_tensor_source("model_weights");
    assets->redae_decoder_weights = assets->resources.open_tensor_source("redae_decoder_weights");
    validate_tensor_shapes(*assets);
    return assets;
}

}  // namespace engine::models::firered_audio
