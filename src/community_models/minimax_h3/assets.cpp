#include "engine/community_models/minimax_h3/assets.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/config.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/model_spec/package.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace engine::models::minimax_h3 {
namespace {

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.rfind(prefix, 0) == 0;
}

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool is_gguf_file(const std::filesystem::path & path) {
    return engine::io::is_existing_file(path) && lower_ascii(path.extension().string()) == ".gguf";
}

std::optional<std::filesystem::path> direct_dit_entry_path(const std::filesystem::path & model_path) {
    if (!is_gguf_file(model_path)) {
        return std::nullopt;
    }
    const auto filename = lower_ascii(model_path.filename().string());
    if (filename == "dit.gguf" || filename == "dit_int8.gguf") {
        return std::filesystem::weakly_canonical(model_path);
    }
    throw std::runtime_error(
        "MiniMax-H3 direct GGUF model path must point to dit.gguf or dit_int8.gguf, got: " +
        model_path.filename().string());
}

int64_t count_indexed_layers(
    const assets::TensorSource & source,
    std::string_view prefix,
    std::string_view suffix) {
    std::unordered_set<std::string> indices;
    for (const auto & tensor : source.tensors()) {
        if (!starts_with(tensor.name, prefix) || !ends_with(tensor.name, suffix)) {
            continue;
        }
        const auto start = prefix.size();
        const auto stop = tensor.name.find('.', start);
        if (stop != std::string::npos && stop > start) {
            indices.insert(tensor.name.substr(start, stop - start));
        }
    }
    return static_cast<int64_t>(indices.size());
}

std::vector<int64_t> ordered_indexed_kernels(
    const assets::TensorSource & source,
    std::string_view prefix,
    std::string_view suffix) {
    std::vector<int64_t> indices;
    for (const auto & tensor : source.tensors()) {
        if (!starts_with(tensor.name, prefix) || !ends_with(tensor.name, suffix)) {
            continue;
        }
        const auto start = prefix.size();
        const auto stop = tensor.name.find('.', start);
        if (stop == std::string::npos || stop == start) {
            continue;
        }
        char * end = nullptr;
        const long index = std::strtol(tensor.name.c_str() + start, &end, 10);
        if (end != tensor.name.c_str() + stop || index < 0) {
            continue;
        }
        indices.push_back(static_cast<int64_t>(index));
    }
    std::sort(indices.begin(), indices.end());
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    std::vector<int64_t> kernels;
    kernels.reserve(indices.size());
    for (const auto index : indices) {
        const auto metadata = source.require_metadata(
            std::string(prefix) + std::to_string(index) + std::string(suffix));
        if (metadata.shape.size() != 3 || metadata.shape.at(2) <= 0) {
            throw std::runtime_error("MiniMax-H3 audio VAE upsample weight shape is invalid");
        }
        kernels.push_back(metadata.shape.at(2));
    }
    if (kernels.empty()) {
        throw std::runtime_error("MiniMax-H3 audio VAE contains no decoder upsample weights");
    }
    return kernels;
}

std::vector<int64_t> infer_bigvgan_rates_from_kernels(const std::vector<int64_t> & kernels) {
    std::vector<int64_t> rates;
    rates.reserve(kernels.size());
    for (const auto kernel : kernels) {
        if (kernel <= 1) {
            throw std::runtime_error("MiniMax-H3 audio VAE upsample kernel must be greater than one");
        }
        rates.push_back((kernel + 1) / 2);
    }
    return rates;
}

void resolve_real_shape_config(
    MiniMaxH3Config & config,
    const assets::TensorSource & text_encoder,
    const assets::TensorSource & dit,
    const assets::TensorSource & audio_vae,
    const assets::TensorSource & video_vae) {
    const auto token_embedding = text_encoder.require_metadata("model.language_model.embed_tokens.weight");
    const auto prompt_q = text_encoder.require_metadata("model.language_model.layers.0.self_attn.q_proj.weight");
    const auto prompt_k = text_encoder.require_metadata("model.language_model.layers.0.self_attn.k_proj.weight");
    const auto prompt_gate = text_encoder.require_metadata("model.language_model.layers.0.mlp.gate_proj.weight");
    config.vocab_size = token_embedding.shape.at(0);
    config.prompt_hidden = token_embedding.shape.at(1);
    config.prompt_layers = count_indexed_layers(text_encoder, "model.language_model.layers.", ".input_layernorm.weight");
    config.prompt_head_dim = text_encoder.require_metadata("model.language_model.layers.0.self_attn.q_norm.weight").shape.at(0);
    config.prompt_heads = prompt_q.shape.at(0) / config.prompt_head_dim;
    config.prompt_kv_heads = prompt_k.shape.at(0) / config.prompt_head_dim;
    config.prompt_intermediate = prompt_gate.shape.at(0);
    config.prompt_eps = 1.0e-6F;
    config.prompt_rope_theta = 5000000.0F;

    const auto audio_patch = dit.require_metadata("audio_patch_proj.weight");
    const auto video_patch = dit.require_metadata("video_patch_proj.weight");
    const auto qkv = dit.require_metadata("blocks.0.attn.qkv_proj.weight");
    const auto fc1 = dit.require_metadata("blocks.0.mlp.fc1.weight");
    const auto rope = dit.require_metadata("rope.inv_freq");
    const auto curve_table = dit.has_tensor("adaln_t_table")
        ? std::optional<assets::TensorMetadata>(dit.require_metadata("adaln_t_table"))
        : std::nullopt;
    config.dit_layers = count_indexed_layers(dit, "blocks.", ".norm1.weight");
    config.token_refiner_layers = count_indexed_layers(dit, "token_refiner.blocks.", ".norm1.weight");
    config.hidden = audio_patch.shape.at(0);
    config.audio_latents_dim = audio_patch.shape.at(1);
    config.video_latents_dim = video_patch.shape.at(1) / 4;
    config.head_dim = dit.require_metadata("blocks.0.attn.q_norm.weight").shape.at(0);
    config.heads = qkv.shape.at(0) / (3 * config.head_dim);
    config.ffn = fc1.shape.at(0) / 2;
    config.text_dim = config.prompt_hidden;
    config.time_embed_hidden = config.hidden;
    if (curve_table.has_value()) {
        if (curve_table->shape.size() != 2 || curve_table->shape.at(0) < 2 || curve_table->shape.at(1) <= 0) {
            throw std::runtime_error("MiniMax-H3 adaln_t_table must have shape [grid, rank]");
        }
        config.adaln_curve_grid = curve_table->shape.at(0);
        config.time_embed_dim = curve_table->shape.at(1);
        config.timestep_input_dim = 0;
    } else {
        const auto timestep_proj_in = dit.require_metadata("time_embedder.proj_in.weight");
        config.timestep_input_dim = timestep_proj_in.shape.at(1);
        config.time_embed_dim = dit.require_metadata("time_embedder.proj_out.weight").shape.at(0);
    }
    config.rope_inv_freq_len = rope.shape.at(0);
    config.norm_eps = 1.0e-5F;
    config.qk_norm_eps = 1.0e-5F;
    config.final_norm_eps = 1.0e-5F;

    config.audio_vae_latent_channels = audio_vae.require_metadata("latent_mean").shape.at(0);
    config.audio_vae_latent_dim = audio_vae.require_metadata("dec_in_proj.weight").shape.at(0);
    config.audio_vae_bigvgan_weight_norm = !audio_vae.has_tensor("decoder.conv_pre.weight");
    if (config.audio_vae_bigvgan_weight_norm) {
        config.audio_vae_decoder_dim = audio_vae.require_metadata("decoder.conv_pre.weight_g").shape.at(0);
        config.audio_vae_decoder_kernels = ordered_indexed_kernels(audio_vae, "decoder.ups.", ".0.weight_v");
    } else {
        config.audio_vae_decoder_dim = audio_vae.require_metadata("decoder.conv_pre.weight").shape.at(0);
        config.audio_vae_decoder_kernels = ordered_indexed_kernels(audio_vae, "decoder.ups.", ".0.weight");
    }
    config.audio_vae_decoder_rates = infer_bigvgan_rates_from_kernels(config.audio_vae_decoder_kernels);
    config.sample_rate = 32000;

    const auto post_quant = video_vae.require_metadata("post_quant_conv.weight");
    const auto x_embedder = video_vae.require_metadata("decoder.x_embedder.weight");
    const auto video_qkv = video_vae.require_metadata("decoder.transformer_blocks.0.attn.to_qkv.weight");
    const auto w1 = video_vae.require_metadata("decoder.transformer_blocks.0.ff.w1.weight");
    const auto proj_out = video_vae.require_metadata("decoder.proj_out.weight");
    config.video_vae_latent_channels = post_quant.shape.at(0);
    config.video_vae_hidden = x_embedder.shape.at(0);
    config.video_vae_head_dim = 64;
    config.video_vae_heads = video_qkv.shape.at(0) / (3 * config.video_vae_head_dim);
    config.video_vae_layers = count_indexed_layers(video_vae, "decoder.transformer_blocks.", ".norm1.weight");
    config.video_vae_register_tokens = video_vae.require_metadata("decoder.register_tokens").shape.at(1);
    config.video_vae_patch_size = 16;
    config.video_vae_patch_size_t = 4;
    config.video_vae_token_drop = 3;
    config.video_vae_clip_length = 17;
    if (w1.shape.at(0) != config.video_vae_hidden * 8 || proj_out.shape.at(0) != 3 * 4 * 16 * 16) {
        throw std::runtime_error("MiniMax-H3 video VAE decoder shape is invalid");
    }

    config.height = 768;
    config.width = 1344;
    config.num_frames = 124;
    config.video_latent_t = ((config.num_frames - 5) / 17) * 5 + 2;
    config.video_latent_h = config.height / 16;
    config.video_latent_w = config.width / 16;
    config.audio_channels = 2;
    config.audio_steps = static_cast<int64_t>(std::llround(static_cast<double>(config.num_frames) / 24.0 * 40.0));
    config.video_patches = config.video_latent_t * (config.video_latent_h / 2) * (config.video_latent_w / 2);
    config.denoise_steps = 50;
}

void validate_real_weight_anchors(const MiniMaxH3Assets & assets) {
    assets.text_encoder_weights->require_metadata("model.language_model.embed_tokens.weight");
    assets.text_encoder_weights->require_metadata("model.language_model.layers.0.self_attn.q_proj.weight");
    assets.dit_weights->require_metadata("audio_patch_proj.weight");
    assets.dit_weights->require_metadata("video_patch_proj.weight");
    assets.dit_weights->require_metadata("blocks.0.attn.qkv_proj.weight");
    assets.dit_weights->require_metadata("final_layer.audio_out.weight");
    assets.audio_vae_weights->require_metadata("latent_mean");
    assets.audio_vae_weights->require_metadata("latent_std");
    assets.audio_vae_weights->require_metadata("dec_in_proj.weight");
    if (assets.audio_vae_weights->has_tensor("decoder.conv_pre.weight")) {
        assets.audio_vae_weights->require_metadata("decoder.conv_post.weight");
    } else {
        assets.audio_vae_weights->require_metadata("decoder.conv_pre.weight_g");
        assets.audio_vae_weights->require_metadata("decoder.conv_post.weight_g");
    }
    assets.video_vae_weights->require_metadata("post_quant_conv.weight");
    assets.video_vae_weights->require_metadata("decoder.x_embedder.weight");
    assets.video_vae_weights->require_metadata("decoder.transformer_blocks.0.attn.to_qkv.weight");
    assets.video_vae_weights->require_metadata("decoder.proj_out.weight");
}

}  // namespace

std::shared_ptr<const MiniMaxH3Assets> load_minimax_h3_assets(const std::filesystem::path & model_path) {
    const auto dit_entry_path = direct_dit_entry_path(model_path);
    MiniMaxH3Assets assets;
    assets.resources = engine::model_spec::load_resource_bundle_for_family(model_path, "minimax_h3");
    if (!assets.resources.has_file("configuration")) {
        throw std::runtime_error("MiniMax-H3 requires a real Q4 per-component GGUF package");
    }
    assets.text_encoder_weights = assets.resources.open_tensor_source("text_encoder_weights");
    assets.dit_weights = dit_entry_path.has_value()
        ? engine::assets::open_tensor_source(*dit_entry_path)
        : assets.resources.open_tensor_source("dit_weights");
    assets.audio_vae_weights = assets.resources.open_tensor_source("audio_vae_weights");
    assets.video_vae_weights = assets.resources.open_tensor_source("video_vae_weights");
    validate_real_weight_anchors(assets);
    resolve_real_shape_config(
        assets.config,
        *assets.text_encoder_weights,
        *assets.dit_weights,
        *assets.audio_vae_weights,
        *assets.video_vae_weights);
    return std::make_shared<MiniMaxH3Assets>(std::move(assets));
}

}  // namespace engine::models::minimax_h3
