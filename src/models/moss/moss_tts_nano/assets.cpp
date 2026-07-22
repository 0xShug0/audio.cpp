#include "engine/models/moss/moss_tts_nano/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <utility>

namespace engine::models::moss_tts_nano {
namespace json = engine::io::json;
namespace {

MossTTSNanoGlobalTransformerConfig parse_global_config(const json::Value & root) {
    const auto & gpt2 = root.require("gpt2_config");
    MossTTSNanoGlobalTransformerConfig config;
    config.vocab_size = json::require_i64(gpt2, "vocab_size");
    config.hidden_size = json::require_i64(gpt2, "n_embd");
    config.intermediate_size = json::optional_i64(gpt2, "n_inner", 4 * config.hidden_size);
    config.num_hidden_layers = json::require_i64(gpt2, "n_layer");
    config.num_attention_heads = json::require_i64(gpt2, "n_head");
    config.max_position_embeddings = json::require_i64(gpt2, "n_positions");
    config.text_vocab_size = config.vocab_size;
    config.layer_norm_epsilon = json::optional_f32(gpt2, "layer_norm_epsilon", config.layer_norm_epsilon);
    config.rope_base = json::optional_f32(gpt2, "rope_base", config.rope_base);
    return config;
}

MossTTSNanoLocalTransformerConfig parse_local_config(
    const json::Value & root,
    const MossTTSNanoGlobalTransformerConfig & global,
    int64_t n_vq) {
    MossTTSNanoLocalTransformerConfig config;
    config.hidden_size = global.hidden_size;
    config.intermediate_size = global.intermediate_size;
    config.num_hidden_layers = json::optional_i64(root, "local_transformer_layers", 1);
    config.num_attention_heads = global.num_attention_heads;
    config.max_position_embeddings = n_vq + 1;
    config.layer_norm_epsilon = global.layer_norm_epsilon;
    config.rope_base = global.rope_base;
    return config;
}

MossTTSNanoAudioTokenizerConfig parse_audio_tokenizer_config(const json::Value & root) {
    MossTTSNanoAudioTokenizerConfig config;
    config.model_type = json::optional_string(root, "model_type", config.model_type);
    config.sample_rate = static_cast<int>(json::optional_i64(root, "sampling_rate", json::optional_i64(root, "sample_rate", config.sample_rate)));
    config.channels = static_cast<int>(json::optional_i64(root, "number_channels", config.channels));
    config.downsample_rate = json::optional_i64(root, "downsample_rate", config.downsample_rate);
    config.code_dim = json::optional_i64(root, "code_dim", config.code_dim);
    const auto & quantizer = root.require("quantizer_kwargs");
    config.rvq_dim = json::optional_i64(quantizer, "rvq_dim", config.rvq_dim);
    config.codebook_dim = json::optional_i64(quantizer, "codebook_dim", config.codebook_dim);
    config.codebook_size = json::optional_i64(quantizer, "codebook_size", config.codebook_size);
    config.num_quantizers = json::optional_i64(quantizer, "num_quantizers", config.num_quantizers);
    return config;
}

MossTTSNanoConfig parse_model_config(
    const json::Value & tts_root,
    const json::Value & tokenizer_root) {
    MossTTSNanoConfig config;
    config.model_type = json::require_string(tts_root, "model_type");
    config.architecture = json::require_string(tts_root, "model_architecture");
    config.n_vq = json::optional_i64(tts_root, "n_vq", config.n_vq);
    config.audio_vocab_size = json::optional_i64(tts_root, "audio_vocab_size", config.audio_vocab_size);
    config.audio_pad_token_id = json::optional_i64(tts_root, "audio_pad_token_id", config.audio_pad_token_id);
    config.pad_token_id = json::optional_i64(tts_root, "pad_token_id", config.pad_token_id);
    config.im_start_token_id = json::require_i64(tts_root, "im_start_token_id");
    config.im_end_token_id = json::require_i64(tts_root, "im_end_token_id");
    config.audio_start_token_id = json::require_i64(tts_root, "audio_start_token_id");
    config.audio_end_token_id = json::require_i64(tts_root, "audio_end_token_id");
    config.audio_user_slot_token_id = json::require_i64(tts_root, "audio_user_slot_token_id");
    config.audio_assistant_slot_token_id = json::require_i64(tts_root, "audio_assistant_slot_token_id");
    config.audio_codebook_sizes = json::optional_i64_array(
        tts_root,
        "audio_codebook_sizes",
        std::vector<int64_t>(static_cast<size_t>(config.n_vq), config.audio_vocab_size));
    config.global_transformer = parse_global_config(tts_root);
    config.local_transformer = parse_local_config(tts_root, config.global_transformer, config.n_vq);
    config.audio_tokenizer = parse_audio_tokenizer_config(tokenizer_root);
    if (config.model_type != "moss_tts_nano") {
        throw std::runtime_error("MOSS-TTS-Nano currently supports model_type=moss_tts_nano");
    }
    if (config.audio_tokenizer.model_type != "moss-audio-tokenizer") {
        throw std::runtime_error("MOSS-TTS-Nano requires the MOSS audio tokenizer");
    }
    if (config.audio_tokenizer.num_quantizers != config.n_vq) {
        throw std::runtime_error("MOSS-TTS-Nano audio tokenizer quantizer count does not match n_vq");
    }
    return config;
}

MossTTSNanoConfig parse_config(const assets::ResourceBundle & resources) {
    return parse_model_config(
        resources.parse_json("config"),
        resources.parse_json("audio_tokenizer_config"));
}

}  // namespace

std::shared_ptr<const MossTTSNanoAssets> load_moss_tts_nano_assets(const std::filesystem::path & model_path) {
    MossTTSNanoAssets assets;
    assets.resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("moss_tts_nano"));
    assets.config = parse_config(assets.resources);
    assets.model_weights = assets.resources.open_tensor_source("model_weights");
    assets.audio_tokenizer_weights = assets.resources.open_tensor_source("audio_tokenizer_weights");
    return std::make_shared<MossTTSNanoAssets>(std::move(assets));
}

}  // namespace engine::models::moss_tts_nano
