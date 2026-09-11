#include "engine/models/magpie_tts/assets.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace engine::models::magpie_tts {
namespace {

namespace json = engine::io::json;

int64_t optional_i64(const json::Value & root, const char * key, int64_t fallback) {
    const auto * value = root.find(key);
    return value == nullptr || value->is_null() ? fallback : value->as_i64();
}

float optional_f32(const json::Value & root, const char * key, float fallback) {
    const auto * value = root.find(key);
    return value == nullptr || value->is_null() ? fallback : static_cast<float>(value->as_number());
}

bool optional_bool(const json::Value & root, const char * key, bool fallback) {
    const auto * value = root.find(key);
    return value == nullptr || value->is_null() ? fallback : value->as_bool();
}

std::vector<int64_t> optional_i64_array(const json::Value & root, const char * key, std::vector<int64_t> fallback) {
    const auto * value = root.find(key);
    return value == nullptr || value->is_null() ? std::move(fallback) : json::number_array_as<int64_t>(*value);
}

std::vector<float> optional_f32_array(const json::Value & root, const char * key, std::vector<float> fallback) {
    const auto * value = root.find(key);
    if (value == nullptr || value->is_null()) {
        return fallback;
    }
    std::vector<float> out;
    for (const auto & item : value->as_array()) {
        out.push_back(static_cast<float>(item.as_number()));
    }
    return out;
}

std::vector<int32_t> require_i32_tensor(
    const assets::TensorSource & source,
    const std::string & name,
    const std::vector<int64_t> & expected_shape) {
    const auto raw = source.require_tensor_data(name);
    if (raw.metadata.shape != expected_shape || (raw.metadata.dtype != "I32" && raw.metadata.dtype != "i32")) {
        throw std::runtime_error("MagpieTTS codec tensor mismatch for " + name);
    }
    if (raw.bytes.size() % sizeof(int32_t) != 0) {
        throw std::runtime_error("MagpieTTS codec tensor byte size mismatch for " + name);
    }
    std::vector<int32_t> values(raw.bytes.size() / sizeof(int32_t));
    std::memcpy(values.data(), raw.bytes.data(), raw.bytes.size());
    return values;
}

std::vector<std::string> parse_speaker_names(const assets::ResourceBundle & resources) {
    std::vector<std::string> names;
    if (!resources.has_file("speakers")) {
        return names;
    }
    const auto root = resources.parse_json("speakers");
    if (root.is_array()) {
        for (const auto & value : root.as_array()) {
            names.push_back(value.as_string());
        }
        return names;
    }
    if (root.is_object()) {
        int64_t max_index = -1;
        for (const auto & [_, value] : root.as_object()) {
            max_index = std::max(max_index, value.as_i64());
        }
        if (max_index < 0) {
            return names;
        }
        names.assign(static_cast<size_t>(max_index + 1), "");
        for (const auto & [name, value] : root.as_object()) {
            const int64_t index = value.as_i64();
            if (index < 0 || index > max_index) {
                throw std::runtime_error("MagpieTTS speaker index is outside the speaker map range");
            }
            names[static_cast<size_t>(index)] = name;
        }
    }
    return names;
}

MagpieTTSConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    MagpieTTSConfig config;
    config.text_vocab_size = json::require_i64(root, "text_vocab_size");
    config.embedding_dim = json::require_i64(root, "embedding_dim");
    config.hidden_dim = optional_i64(root, "hidden_dim", config.embedding_dim);
    const auto & decoder = root.require("decoder");
    config.decoder_layers = json::require_i64(decoder, "n_layers");
    config.decoder_heads = json::require_i64(decoder, "sa_n_heads");
    config.decoder_ffn_dim = json::optional_i64(decoder, "d_ffn", config.embedding_dim * 4);
    config.decoder_max_length = json::optional_i64(decoder, "max_length_causal_mask", 2048);
    config.decoder_cross_heads = json::optional_i64(decoder, "xa_n_heads", 1);
    config.decoder_cross_head_dim = json::optional_i64(decoder, "xa_d_head", 128);
    config.context_length = json::require_i64(root, "baked_context_length");
    config.context_dim = json::require_i64(root, "baked_context_dim");
    config.speakers = json::require_i64(root, "baked_speakers");
    config.audio_codebooks = json::require_i64(root, "audio_codebooks");
    config.frame_stacking_factor = optional_i64(root, "frame_stacking_factor", 1);
    config.codebook_size = optional_i64(root, "codebook_size", json::require_i64(root, "audio_vocab_size") - 8);
    config.all_tokens_per_codebook = json::require_i64(root, "audio_vocab_size");
    config.local_layers = json::require_i64(root, "local_transformer_n_layers");
    config.local_heads = json::require_i64(root, "local_transformer_n_heads");
    config.local_hidden_dim = json::require_i64(root, "local_transformer_hidden_dim");
    config.local_ffn_dim = optional_i64(root, "local_transformer_d_ffn", config.local_hidden_dim * 4);
    config.local_context = optional_i64(root, "local_transformer_max_length_causal_mask", 18);
    if (const auto * inference = root.find("inference_parameters")) {
        config.max_decoder_steps = optional_i64(*inference, "max_decoder_steps", config.max_decoder_steps);
        config.min_generated_frames = optional_i64(*inference, "min_generated_frames", config.min_generated_frames);
        config.short_sentence_threshold =
            optional_i64(*inference, "short_sentence_threshold", config.short_sentence_threshold);
        config.near_end_threshold = optional_i64(*inference, "near_end_threshold", config.near_end_threshold);
        config.finished_limit_first_chunk =
            optional_i64(*inference, "finished_limit_first_chunk", config.finished_limit_first_chunk);
        config.finished_limit_with_eot =
            optional_i64(*inference, "finished_limit_with_eot", config.finished_limit_with_eot);
        config.finished_limit_without_eot =
            optional_i64(*inference, "finished_limit_without_eot", config.finished_limit_without_eot);
        config.forceful_chunk_end_threshold =
            optional_i64(*inference, "forceful_chunk_end_threshold", config.forceful_chunk_end_threshold);
        config.history_len_heuristic = optional_i64(*inference, "history_len_heuristic", config.history_len_heuristic);
        config.attention_sink_threshold =
            optional_i64(*inference, "attention_sink_threshold", config.attention_sink_threshold);
        config.chunked_attention_sink_threshold =
            optional_i64(*inference, "chunked_attention_sink_threshold", config.chunked_attention_sink_threshold);
        config.attention_prior_lookahead_window =
            optional_i64(*inference, "attention_prior_lookahead_window", config.attention_prior_lookahead_window);
        config.start_prior_after_n_audio_steps =
            optional_i64(*inference, "start_prior_after_n_audio_steps", config.start_prior_after_n_audio_steps);
        config.temperature = optional_f32(*inference, "temperature", config.temperature);
        config.argmax_temperature = optional_f32(*inference, "argmax_temperature", config.argmax_temperature);
        config.top_k = optional_i64(*inference, "topk", config.top_k);
        config.guidance_scale = optional_f32(*inference, "cfg_scale", config.guidance_scale);
        config.attention_prior_epsilon =
            optional_f32(*inference, "attention_prior_epsilon", config.attention_prior_epsilon);
        config.apply_attention_prior =
            optional_bool(*inference, "apply_attention_prior", config.apply_attention_prior);
        config.ignore_finished_sentence_tracking =
            optional_bool(*inference, "ignore_finished_sentence_tracking", config.ignore_finished_sentence_tracking);
        config.apply_prior_to_layers =
            optional_i64_array(*inference, "apply_prior_to_layers", std::move(config.apply_prior_to_layers));
        config.estimate_alignment_from_layers =
            optional_i64_array(*inference, "estimate_alignment_from_layers", std::move(config.estimate_alignment_from_layers));
        config.prior_weights = optional_f32_array(*inference, "prior_weights", std::move(config.prior_weights));
        config.prior_weights_init =
            optional_f32_array(*inference, "prior_weights_init", std::move(config.prior_weights_init));
    }
    if (resources.has_file("codec_config")) {
        const auto codec_root = resources.parse_json("codec_config");
        config.sample_rate = optional_i64(codec_root, "sample_rate", config.sample_rate);
        config.samples_per_frame = optional_i64(codec_root, "samples_per_frame", config.samples_per_frame);
        config.codec_input_dim = optional_i64(codec_root, "input_dim", config.codec_input_dim);
        config.codec_base_channels = optional_i64(codec_root, "base_channels", config.codec_base_channels);
        config.codec_upsample_rates =
            optional_i64_array(codec_root, "up_sample_rates", std::move(config.codec_upsample_rates));
        config.codec_resblock_kernel_sizes =
            optional_i64_array(codec_root, "resblock_kernel_sizes", std::move(config.codec_resblock_kernel_sizes));
        config.codec_resblock_dilation_sizes =
            optional_i64_array(codec_root, "resblock_dilation_sizes", std::move(config.codec_resblock_dilation_sizes));
    }
    config.speaker_names = parse_speaker_names(resources);
    return config;
}

void validate_config(const MagpieTTSConfig & config) {
    engine::io::require_positive(config.text_vocab_size, "MagpieTTS text_vocab_size");
    engine::io::require_positive(config.embedding_dim, "MagpieTTS embedding_dim");
    engine::io::require_positive(config.decoder_layers, "MagpieTTS decoder_n_layers");
    engine::io::require_positive(config.decoder_heads, "MagpieTTS decoder_n_heads");
    engine::io::require_divisible(config.embedding_dim, config.decoder_heads, "MagpieTTS decoder head size");
    engine::io::require_positive(config.context_length, "MagpieTTS baked_context_length");
    engine::io::require_positive(config.context_dim, "MagpieTTS baked_context_dim");
    engine::io::require_positive(config.speakers, "MagpieTTS baked_speakers");
    engine::io::require_positive(config.audio_codebooks, "MagpieTTS audio_codebooks");
    engine::io::require_positive(config.frame_stacking_factor, "MagpieTTS frame_stacking_factor");
    engine::io::require_positive(config.codebook_size, "MagpieTTS codebook_size");
    engine::io::require_positive(config.all_tokens_per_codebook, "MagpieTTS audio_vocab_size");
    engine::io::require_positive(config.local_layers, "MagpieTTS local_transformer_n_layers");
    engine::io::require_positive(config.local_heads, "MagpieTTS local_transformer_n_heads");
    engine::io::require_divisible(config.local_hidden_dim, config.local_heads, "MagpieTTS local head size");
    engine::io::require_positive(config.max_decoder_steps, "MagpieTTS max_decoder_steps");
    engine::io::require_positive(config.min_generated_frames, "MagpieTTS min_generated_frames");
    engine::io::require_positive(config.top_k, "MagpieTTS topk");
    engine::io::require_positive(config.codec_input_dim, "MagpieTTS codec input_dim");
    engine::io::require_positive(config.codec_base_channels, "MagpieTTS codec base_channels");
    if (config.codec_upsample_rates.empty()) {
        throw std::runtime_error("MagpieTTS codec up_sample_rates must not be empty");
    }
    if (config.codec_resblock_kernel_sizes.empty() || config.codec_resblock_dilation_sizes.empty()) {
        throw std::runtime_error("MagpieTTS codec residual kernel/dilation lists must not be empty");
    }
    if (config.all_tokens_per_codebook != config.codebook_size + 8) {
        throw std::runtime_error("MagpieTTS audio_vocab_size must equal codebook_size plus 8 special tokens");
    }
    if (config.context_dim != config.embedding_dim) {
        throw std::runtime_error("MagpieTTS baked context dimension must match embedding_dim");
    }
}

void validate_model_weights(const MagpieTTSConfig & config, const assets::TensorSource & source) {
    assets::require_tensor_shape(source, "text_embedding.weight", {config.text_vocab_size, config.embedding_dim});
    assets::require_tensor_shape(
        source,
        "baked_context_embedding.weight",
        {config.speakers, config.context_length * config.context_dim});
    assets::require_tensor_shape(
        source,
        "decoder.layers.0.self_attention.qkv_net.weight",
        {config.embedding_dim * 3, config.embedding_dim});
    assets::require_tensor_shape(
        source,
        "decoder.layers.0.cross_attention.q_net.weight",
        {config.decoder_cross_heads * config.decoder_cross_head_dim, config.embedding_dim});
    assets::require_tensor_shape(
        source,
        "decoder.layers.0.cross_attention.kv_net.weight",
        {config.decoder_cross_heads * config.decoder_cross_head_dim * 2, config.context_dim});
    assets::require_tensor_shape(
        source,
        "local_transformer.layers.0.self_attention.qkv_net.weight",
        {config.local_hidden_dim * 3, config.local_hidden_dim});
    assets::require_tensor_shape(
        source,
        "audio_embeddings.0.weight",
        {config.all_tokens_per_codebook, config.embedding_dim});
    assets::require_tensor_shape(
        source,
        "local_transformer_out_projections.0.weight",
        {config.all_tokens_per_codebook, config.local_hidden_dim});
    assets::require_tensor_shape(
        source,
        "final_proj.weight",
        {config.audio_codebooks * config.frame_stacking_factor * config.all_tokens_per_codebook, config.embedding_dim});
}

void validate_codec_weights(const assets::TensorSource & source) {
    assets::require_tensor_shape(source, "audio_decoder.pre_conv.conv.parametrizations.weight.original0", {864, 1, 1});
    assets::require_tensor_shape(source, "audio_decoder.pre_conv.conv.parametrizations.weight.original1", {864, 32, 7});
}

}  // namespace

std::shared_ptr<const MagpieTTSAssets> load_magpie_tts_assets(const std::filesystem::path & model_path) {
    MagpieTTSAssets assets;
    assets.resources = model_spec::load_resource_bundle_for_family(model_path, "magpie_tts");
    assets.config = parse_config(assets.resources);
    validate_config(assets.config);
    assets.model_weights = assets.resources.open_tensor_source("model");
    assets.codec_weights = assets.resources.open_tensor_source("codec");
    assets.config.codec_fsq_num_levels =
        require_i32_tensor(*assets.codec_weights, "vector_quantizer.fsqs.0.num_levels", {1, 4, 1});
    assets.config.codec_fsq_dim_base_index =
        require_i32_tensor(*assets.codec_weights, "vector_quantizer.fsqs.0.dim_base_index", {1, 4, 1});
    validate_model_weights(assets.config, *assets.model_weights);
    validate_codec_weights(*assets.codec_weights);
    return std::make_shared<MagpieTTSAssets>(std::move(assets));
}

}  // namespace engine::models::magpie_tts
