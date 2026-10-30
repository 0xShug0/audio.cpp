#include "engine/community_models/edda_asr/model.h"
#include "engine/framework/model_spec/package.h"

#include <cmath>
#include <stdexcept>

namespace engine::community_models::edda_asr {
namespace {

int32_t require_token(const tokenizers::LlamaBpeTokenizer & tokenizer, const std::string & piece) {
    const auto id = tokenizer.find_token_id(piece);
    if (!id) {
        throw std::runtime_error("Edda tokenizer does not contain " + piece);
    }
    return *id;
}

// Mirrors hurtig.build_target_layer_ids: draft context taps spread over the target decoder.
std::vector<int64_t> target_layer_ids(int64_t target_layers, int64_t draft_layers) {
    if (draft_layers == 1) {
        return {target_layers / 2};
    }
    const int64_t start = 1, span = target_layers - 3 - start;
    std::vector<int64_t> ids;
    for (int64_t i = 0; i < draft_layers; ++i) {
        ids.push_back(start + std::llround(static_cast<double>(i * span) / static_cast<double>(draft_layers - 1)));
    }
    return ids;
}

std::optional<DrafterConfig> load_drafter_config(const EddaAssets & assets) {
    if (!assets.resources.has_file("drafter_config")) {
        return std::nullopt;
    }
    std::shared_ptr<const assets::TensorSource> source;
    try {
        source = assets.resources.open_tensor_source("drafter");
    } catch (const std::exception &) {
        return std::nullopt;
    }
    if (!source || !source->has_tensor("fc.weight")) {
        return std::nullopt;
    }
    const auto json = assets.resources.parse_json("drafter_config");
    DrafterConfig config;
    config.layers = io::json::optional_i64(json, "n_layers", config.layers);
    config.target_layers = io::json::optional_i64(json, "n_target_layers", config.target_layers);
    config.block = io::json::optional_i64(json, "block", config.block);
    config.conv_kernel = io::json::optional_i64(json, "conv_kernel", config.conv_kernel);
    config.conv_group = io::json::optional_i64(json, "conv_group", config.conv_group);
    config.selector_rank = io::json::optional_i64(json, "sel_rank", config.selector_rank);
    config.selector_top_k = io::json::optional_i64(json, "sel_top_k", config.selector_top_k);
    if (config.layers < 1 || config.target_layers != assets.decoder_layers || config.block < 2 ||
        config.conv_kernel != 2 || config.conv_group < 1 || assets.encoder.n_audio_state % config.conv_group != 0 ||
        config.selector_rank < 1 || config.selector_top_k < 1) {
        throw std::runtime_error("Unsupported Edda drafter configuration");
    }
    config.target_layer_ids = target_layer_ids(config.target_layers, config.layers);
    return config;
}

}  // namespace

std::shared_ptr<const EddaAssets> load_assets(const std::filesystem::path & path) {
    auto out = std::make_shared<EddaAssets>();
    out->resources = model_spec::load_resource_bundle(path, model_spec::default_spec_path("edda_asr"));
    out->source = out->resources.open_tensor_source("weights");
    const auto config = out->resources.parse_json("config");
    if (io::json::require_string(config, "model_type") != "whisper") {
        throw std::runtime_error("Edda requires a Whisper checkpoint");
    }
    auto & e = out->encoder;
    e.n_mels = io::json::require_i64(config, "num_mel_bins");
    e.n_audio_ctx = io::json::require_i64(config, "max_source_positions");
    e.n_audio_state = io::json::require_i64(config, "d_model");
    e.n_audio_head = io::json::require_i64(config, "encoder_attention_heads");
    e.n_audio_layer = io::json::require_i64(config, "encoder_layers");
    out->decoder_layers = io::json::require_i64(config, "decoder_layers");
    out->decoder_heads = io::json::require_i64(config, "decoder_attention_heads");
    out->decoder_ffn = io::json::require_i64(config, "decoder_ffn_dim");
    out->vocabulary_size = io::json::require_i64(config, "vocab_size");
    out->max_positions = io::json::require_i64(config, "max_target_positions");
    out->eos = io::json::require_i32(config, "eos_token_id");
    if (io::json::require_string(config, "activation_function") != "gelu" ||
        io::json::require_i64(config, "encoder_ffn_dim") != 4 * e.n_audio_state ||
        e.n_audio_state % out->decoder_heads != 0) {
        throw std::runtime_error("Unsupported Edda Whisper transformer configuration");
    }
    out->tokenizer = tokenizers::load_llama_bpe_tokenizer({
        out->resources.require_file("vocab"), out->resources.require_file("merges"),
        out->resources.require_file("tokenizer_config"), out->resources.require_file("tokenizer"),
        tokenizers::LlamaBpePreTokenizer::Gpt2});
    out->start_of_transcript = require_token(*out->tokenizer, "<|startoftranscript|>");
    out->transcribe = require_token(*out->tokenizer, "<|transcribe|>");
    out->no_timestamps = require_token(*out->tokenizer, "<|notimestamps|>");
    // Whisper places every control, language and timestamp token after <|endoftext|>.
    out->special_tokens.resize(static_cast<size_t>(out->vocabulary_size));
    for (int32_t id = out->eos; id < out->vocabulary_size; ++id) {
        out->special_tokens[static_cast<size_t>(id)] = true;
    }
    const auto generation = out->resources.parse_json("generation_config");
    const auto token_list = [&](const char * key) {
        std::vector<int32_t> ids;
        for (auto id : io::json::optional_i64_array(generation, key)) {
            if (id >= 0 && id < out->vocabulary_size) {
                ids.push_back(static_cast<int32_t>(id));
            }
        }
        return ids;
    };
    out->suppress = token_list("suppress_tokens");
    out->begin_suppress = token_list("begin_suppress_tokens");
    out->drafter = load_drafter_config(*out);
    if (out->drafter) {
        out->drafter_source = out->resources.open_tensor_source("drafter");
    }
    return out;
}

std::vector<int32_t> EddaAssets::prompt(const std::string & language) const {
    return {start_of_transcript, require_token(*tokenizer, "<|" + language + "|>"), transcribe, no_timestamps};
}

std::string EddaAssets::decode_text(const std::vector<int32_t> & tokens) const {
    std::vector<int32_t> visible;
    visible.reserve(tokens.size());
    for (const auto id : tokens) {
        if (id >= 0 && id < vocabulary_size && !special_tokens[static_cast<size_t>(id)]) {
            visible.push_back(id);
        }
    }
    return tokenizer->decode(visible, false);
}

}  // namespace engine::community_models::edda_asr
