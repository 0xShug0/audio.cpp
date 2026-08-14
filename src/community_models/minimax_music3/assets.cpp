#include "engine/community_models/minimax_music3/assets.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/model_spec/package.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

namespace engine::models::minimax_music3 {
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

std::optional<std::filesystem::path> direct_lm_entry_path(const std::filesystem::path & model_path) {
    if (!engine::io::is_existing_file(model_path) ||
        lower_ascii(model_path.extension().string()) != ".gguf") {
        return std::nullopt;
    }
    const auto filename = lower_ascii(model_path.filename().string());
    if (starts_with(filename, "lm_")) {
        return std::filesystem::weakly_canonical(model_path);
    }
    throw std::runtime_error(
        "MiniMax-Music3 direct GGUF model path must point to the lm_*.gguf entry file, got: " +
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

void resolve_shape_config(
    MiniMaxMusic3Config & config,
    const assets::TensorSource & lm,
    const assets::TensorSource & depth,
    const assets::TensorSource & dit,
    const assets::TensorSource & cond,
    const assets::TensorSource & vocoder) {
    const auto embed = lm.require_metadata("model.embed_tokens.weight");
    const auto q = lm.require_metadata("model.layers.0.self_attn.q_proj.weight");
    const auto k = lm.require_metadata("model.layers.0.self_attn.k_proj.weight");
    const auto gate = lm.require_metadata("model.layers.0.mlp.gate_proj.weight");
    const auto head = lm.require_metadata("lm_head_sliced.weight");
    config.lm_vocab_size = embed.shape.at(0);
    config.lm_hidden = embed.shape.at(1);
    config.lm_layers = count_indexed_layers(lm, "model.layers.", ".input_layernorm.weight");
    config.lm_head_dim = lm.require_metadata("model.layers.0.self_attn.q_norm.weight").shape.at(0);
    config.lm_heads = q.shape.at(0) / config.lm_head_dim;
    config.lm_kv_heads = k.shape.at(0) / config.lm_head_dim;
    config.lm_intermediate = gate.shape.at(0);
    config.lm_logits = head.shape.at(0);

    const auto depth_embed = depth.require_metadata("audio_embeddings.weight");
    const auto depth_head = depth.require_metadata("audio_heads.0.weight");
    config.depth_hidden = depth_embed.shape.at(1);
    config.depth_layers = count_indexed_layers(depth, "layers.", ".input_layernorm.weight");
    config.depth_audio_vocab = depth_head.shape.at(0);
    config.depth_codebooks = depth_embed.shape.at(0) / config.depth_audio_vocab + 1;
    config.depth_intermediate = depth.require_metadata("layers.0.gate_proj.weight").shape.at(0);
    config.depth_max_positions = depth.require_metadata("pos_embedding.weight").shape.at(0);

    const auto cond_proj = cond.require_metadata("proj.weight");
    config.cond_out_dim = cond_proj.shape.at(0);
    config.cond_hidden = cond_proj.shape.at(1);
    config.cond_layers = cond.require_metadata("layer_weight_logits").shape.at(0);

    const auto dit_proj_in = dit.require_metadata("proj_in.weight");
    const auto dit_proj_out = dit.require_metadata("proj_out.weight");
    const auto dit_ff_in = dit.require_metadata("transformer_blocks.0.ff_in.weight");
    const auto dit_time = dit.require_metadata("time_proj.weight");
    config.dit_in_channels = dit_proj_out.shape.at(0);
    config.dit_condition_dim = dit_proj_in.shape.at(1) - 2 * config.dit_in_channels;
    config.dit_layers = count_indexed_layers(dit, "transformer_blocks.", ".norm1.weight");
    const int64_t dit_inner = dit_proj_in.shape.at(0);
    config.dit_head_dim = 64;
    config.dit_heads = dit_inner / config.dit_head_dim;
    config.dit_ff_inner = dit_ff_in.shape.at(0) / 2;
    config.dit_fourier_dim = dit_time.shape.at(0) * 2;

    const auto voc_in = vocoder.require_metadata("dec_in_proj.weight");
    const auto voc_conv_in = vocoder.require_metadata("conv_in.weight");
    config.vocoder_latent_channels = voc_in.shape.at(1) * 2;
    config.vocoder_input_dim = voc_in.shape.at(0);
    config.vocoder_hidden_dim = voc_conv_in.shape.at(0);
    const int64_t blocks = count_indexed_layers(vocoder, "blocks.", ".snake1.alpha");
    config.vocoder_strides.clear();
    for (int64_t block = 0; block < blocks; ++block) {
        const auto conv_t = vocoder.require_metadata(
            "blocks." + std::to_string(block) + ".conv_t1.weight");
        if (conv_t.shape.size() != 3 || conv_t.shape.at(2) % 2 != 0) {
            throw std::runtime_error("MiniMax-Music3 vocoder upsample kernel shape is invalid");
        }
        config.vocoder_strides.push_back(conv_t.shape.at(2) / 2);
    }
    if (config.vocoder_strides.empty()) {
        throw std::runtime_error("MiniMax-Music3 vocoder contains no upsample blocks");
    }
}

void validate_weight_anchors(const MiniMaxMusic3Assets & assets) {
    assets.lm_weights->require_metadata("model.embed_tokens.weight");
    assets.lm_weights->require_metadata("model.layers.0.self_attn.q_proj.weight");
    assets.lm_weights->require_metadata("model.norm.weight");
    assets.lm_weights->require_metadata("lm_head_sliced.weight");
    assets.depth_decoder_weights->require_metadata("audio_embeddings.weight");
    assets.depth_decoder_weights->require_metadata("projection.weight");
    assets.depth_decoder_weights->require_metadata("audio_heads.0.weight");
    assets.dit_weights->require_metadata("preprocess_conv.weight");
    assets.dit_weights->require_metadata("proj_in.weight");
    assets.dit_weights->require_metadata("time_proj.weight");
    assets.dit_weights->require_metadata("transformer_blocks.0.attn.to_q.weight");
    assets.condition_encoder_weights->require_metadata("layer_weight_logits");
    assets.condition_encoder_weights->require_metadata("proj.weight");
    assets.vocoder_weights->require_metadata("dec_in_proj.weight");
    assets.vocoder_weights->require_metadata("conv_in.weight");
    assets.vocoder_weights->require_metadata("conv_out.weight");
}

}  // namespace

std::shared_ptr<const MiniMaxMusic3Assets> load_minimax_music3_assets(const std::filesystem::path & model_path) {
    const auto lm_entry_path = direct_lm_entry_path(model_path);
    MiniMaxMusic3Assets assets;
    assets.resources = engine::model_spec::load_resource_bundle_for_family(model_path, "minimax_music3");
    assets.lm_weights = lm_entry_path.has_value()
        ? engine::assets::open_tensor_source(*lm_entry_path)
        : assets.resources.open_tensor_source("lm_weights");
    assets.depth_decoder_weights = assets.resources.open_tensor_source("depth_decoder_weights");
    assets.dit_weights = assets.resources.open_tensor_source("dit_weights");
    assets.condition_encoder_weights = assets.resources.open_tensor_source("condition_encoder_weights");
    assets.vocoder_weights = assets.resources.open_tensor_source("vocoder_weights");
    validate_weight_anchors(assets);
    resolve_shape_config(
        assets.config,
        *assets.lm_weights,
        *assets.depth_decoder_weights,
        *assets.dit_weights,
        *assets.condition_encoder_weights,
        *assets.vocoder_weights);
    return std::make_shared<MiniMaxMusic3Assets>(std::move(assets));
}

}  // namespace engine::models::minimax_music3
