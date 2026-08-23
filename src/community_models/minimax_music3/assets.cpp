#include "engine/community_models/minimax_music3/assets.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/filesystem.h"

#include <filesystem>
#include <stdexcept>
#include <utility>

namespace engine::models::minimax_music3 {
namespace {

std::vector<int64_t> i64_array(const engine::io::json::Value & value) {
    std::vector<int64_t> out;
    for (const auto & item : value.as_array()) {
        out.push_back(item.as_i64());
    }
    return out;
}

engine::io::json::Value parse_json_file(const std::filesystem::path & path) {
    if (!engine::io::is_existing_file(path)) {
        throw std::runtime_error("missing MiniMax Music 3 config file: " + path.string());
    }
    return engine::io::json::parse_file(path);
}

MiniMaxMusic3QwenConfig parse_qwen_config(const std::filesystem::path & model_root) {
    const auto root = parse_json_file(model_root / "config" / "language_model.json");
    MiniMaxMusic3QwenConfig out;
    out.vocab_size = root.require("vocab_size").as_i64();
    out.hidden_size = root.require("hidden_size").as_i64();
    out.intermediate_size = root.require("intermediate_size").as_i64();
    out.layers = root.require("num_hidden_layers").as_i64();
    out.attention_heads = root.require("num_attention_heads").as_i64();
    out.kv_heads = root.require("num_key_value_heads").as_i64();
    out.head_dim = root.require("head_dim").as_i64();
    out.max_position_embeddings = root.require("max_position_embeddings").as_i64();
    out.rms_norm_eps = root.require("rms_norm_eps").as_f32();
    if (const auto * rope = root.find("rope_parameters")) {
        out.rope_theta = rope->require("rope_theta").as_f32();
    }
    return out;
}

MiniMaxMusic3DepthConfig parse_depth_config(const std::filesystem::path & model_root) {
    const auto root = parse_json_file(model_root / "config" / "rvq_depth_decoder.json");
    MiniMaxMusic3DepthConfig out;
    out.audio_vocab_size = root.require("audio_vocab_size").as_i64();
    out.hidden_size = root.require("hidden_size").as_i64();
    out.intermediate_size = root.require("intermediate_size").as_i64();
    out.max_position_embeddings = root.require("max_position_embeddings").as_i64();
    out.attention_heads = root.require("num_attention_heads").as_i64();
    out.codebooks = root.require("num_codebooks").as_i64();
    out.layers = root.require("num_layers").as_i64();
    return out;
}

MiniMaxMusic3ConditionConfig parse_condition_config(const std::filesystem::path & model_root) {
    const auto root = parse_json_file(model_root / "config" / "condition_encoder.json");
    MiniMaxMusic3ConditionConfig out;
    out.condition_hidden_dim = root.require("condition_hidden_dim").as_i64();
    out.condition_layers = root.require("num_condition_layers").as_i64();
    out.out_dim = root.require("out_dim").as_i64();
    out.input_sample_rate = root.require("input_sampling_rate").as_i64();
    out.input_hop_length = root.require("input_hop_length").as_i64();
    out.output_sample_rate = root.require("output_sampling_rate").as_i64();
    out.output_hop_length = root.require("output_hop_length").as_i64();
    return out;
}

MiniMaxMusic3FlowConfig parse_flow_config(const std::filesystem::path & model_root) {
    const auto root = parse_json_file(model_root / "config" / "transformer.json");
    MiniMaxMusic3FlowConfig out;
    out.in_channels = root.require("in_channels").as_i64();
    out.condition_dim = root.require("condition_dim").as_i64();
    out.layers = root.require("num_layers").as_i64();
    out.attention_heads = root.require("num_attention_heads").as_i64();
    out.head_dim = root.require("attention_head_dim").as_i64();
    out.ff_inner_dim = root.require("ff_inner_dim").as_i64();
    out.rotary_dim = root.require("rotary_dim").as_i64();
    out.fourier_embedding_dim = root.require("fourier_embedding_dim").as_i64();
    return out;
}

MiniMaxMusic3VocoderConfig parse_vocoder_config(const std::filesystem::path & model_root) {
    const auto root = parse_json_file(model_root / "config" / "vocoder.json");
    MiniMaxMusic3VocoderConfig out;
    out.latent_channels = root.require("latent_channels").as_i64();
    out.decoder_input_dim = root.require("decoder_input_dim").as_i64();
    out.decoder_hidden_dim = root.require("decoder_hidden_dim").as_i64();
    out.sample_rate = static_cast<int>(root.require("sampling_rate").as_i64());
    out.upsample_ratios = i64_array(root.require("upsampling_ratios"));
    return out;
}

std::shared_ptr<const assets::TensorSource> open_model_root_gguf(
    const std::filesystem::path & model_root,
    const std::filesystem::path & relative_path) {
    const auto path = model_root / relative_path;
    if (!engine::io::is_existing_file(path)) {
        throw std::runtime_error("missing MiniMax Music 3 component GGUF: " + path.string());
    }
    return assets::open_tensor_source(path);
}

}  // namespace

void validate_minimax_music3_anchors(const MiniMaxMusic3Assets & assets) {
    assets.language_model_weights->require_metadata("model.embed_tokens.weight");
    assets.language_model_weights->require_metadata("model.layers.0.self_attn.q_proj.weight");
    assets.language_model_weights->require_metadata("model.norm.weight");
    assets.depth_decoder_weights->require_metadata("audio_embeddings.weight");
    assets.depth_decoder_weights->require_metadata("layers.0.attn.to_q.weight");
    assets.condition_encoder_weights->require_metadata("proj.weight");
    assets.transformer_weights->require_metadata("preprocess_conv.weight");
    assets.transformer_weights->require_metadata("transformer_blocks.0.attn.to_q.weight");
    assets.vocoder_weights->require_metadata("dec_in_proj.weight");
    assets.vocoder_weights->require_metadata("conv_in.weight_v");
}

std::shared_ptr<const MiniMaxMusic3Assets> load_minimax_music3_assets(
    const std::filesystem::path & model_path) {
    auto assets = std::make_shared<MiniMaxMusic3Assets>();
    assets->model_root = assets::prepare_model_directory(model_path).model_root;
    assets->tokenizer_json_path = assets->model_root / "tokenizer" / "tokenizer.json";
    assets->tokenizer_config_path = assets->model_root / "tokenizer" / "tokenizer_config.json";
    if (!engine::io::is_existing_file(assets->tokenizer_json_path)) {
        throw std::runtime_error("missing MiniMax Music 3 tokenizer file: " + assets->tokenizer_json_path.string());
    }
    if (!engine::io::is_existing_file(assets->tokenizer_config_path)) {
        throw std::runtime_error(
            "missing MiniMax Music 3 tokenizer config file: " + assets->tokenizer_config_path.string());
    }
    assets->config.qwen = parse_qwen_config(assets->model_root);
    assets->config.depth = parse_depth_config(assets->model_root);
    assets->config.condition = parse_condition_config(assets->model_root);
    assets->config.flow = parse_flow_config(assets->model_root);
    assets->config.vocoder = parse_vocoder_config(assets->model_root);
    assets->language_model_weights = open_model_root_gguf(assets->model_root, "language_model_q4_0.gguf");
    assets->depth_decoder_weights = open_model_root_gguf(assets->model_root, "rvq_depth_decoder_q8_0.gguf");
    assets->condition_encoder_weights = open_model_root_gguf(assets->model_root, "condition_encoder.gguf");
    assets->transformer_weights = open_model_root_gguf(assets->model_root, "transformer_q4_0.gguf");
    assets->vocoder_weights = open_model_root_gguf(assets->model_root, "vocoder.gguf");
    validate_minimax_music3_anchors(*assets);
    return assets;
}

assets::TensorStorageType conv_safe_storage_type(
    const assets::TensorSource & source,
    const std::string & tensor_prefix,
    assets::TensorStorageType requested) {
    if (requested == assets::TensorStorageType::BF16) {
        return assets::TensorStorageType::F32;
    }
    if (requested == assets::TensorStorageType::Native &&
        assets::ggml_type_for_tensor_dtype(source.require_metadata(tensor_prefix + ".weight").dtype) ==
            GGML_TYPE_BF16) {
        return assets::TensorStorageType::F32;
    }
    return requested;
}

}  // namespace engine::models::minimax_music3
