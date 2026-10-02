#include "engine/models/crisperwhisper/model.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::crisperwhisper {

std::shared_ptr<const CrisperWhisperAssets> load_assets(const std::filesystem::path & path) {
    auto out = std::make_shared<CrisperWhisperAssets>();
    out->resources = model_spec::load_resource_bundle(path, model_spec::default_spec_path("crisperwhisper"));
    out->source = out->resources.open_tensor_source("weights");
    const auto config = out->resources.parse_json("config");
    if (io::json::require_string(config, "model_type") != "whisper") {
        throw std::runtime_error("CrisperWhisper requires a Whisper checkpoint");
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
        throw std::runtime_error("Unsupported CrisperWhisper transformer configuration");
    }
    out->tokenizer = tokenizers::load_llama_bpe_tokenizer({
        out->resources.require_file("vocab"), out->resources.require_file("merges"),
        out->resources.require_file("tokenizer_config"), out->resources.require_file("tokenizer"),
        tokenizers::LlamaBpePreTokenizer::Gpt2});
    // Added vocal events are output tokens, not prompt/control tokens.
    out->control_tokens.resize(static_cast<size_t>(out->vocabulary_size));
    for (int32_t id = 0; id < out->vocabulary_size; ++id) {
        const auto piece = out->tokenizer->decode({id}, false);
        out->control_tokens[id] = (piece.size() >= 4 && piece.compare(0, 2, "<|") == 0 &&
            piece.compare(piece.size() - 2, 2, "|>") == 0) ||
            piece.compare(0, 10, "[verbatim_") == 0 || piece.compare(0, 10, "[intended_") == 0 ||
            piece == "<ctx>" || piece == "<ectx>" || piece == "<htx>" || piece == "<ehtx>" ||
            piece == "<vtx>" || piece == "<evtx>" || piece == "<sot>" || piece == "<eot>";
    }
    const auto generation = out->resources.parse_json("generation_config");
    for (auto id : io::json::optional_i64_array(generation, "suppress_tokens")) {
        if (id >= 0 && id < out->vocabulary_size) {
            out->suppress.push_back(static_cast<int32_t>(id));
        }
    }
    for (const auto & pair : generation.require("alignment_heads").as_array()) {
        const auto & p = pair.as_array();
        if (p.size() != 2 || p[0].as_i64() < 0 || p[0].as_i64() >= out->decoder_layers ||
            p[1].as_i64() < 0 || p[1].as_i64() >= out->decoder_heads) {
            throw std::runtime_error("Invalid CrisperWhisper alignment head");
        }
        out->alignment_heads.emplace_back(p[0].as_i64(), p[1].as_i64());
    }
    if (out->alignment_heads.empty()) {
        throw std::runtime_error("CrisperWhisper alignment heads are missing");
    }
    return out;
}

std::string CrisperWhisperAssets::decode_text(const std::vector<int32_t> & tokens) const {
    std::vector<int32_t> visible;
    visible.reserve(tokens.size());
    for (const auto id : tokens) {
        if (!control_tokens.at(static_cast<size_t>(id))) {
            visible.push_back(id);
        }
    }
    return tokenizer->decode(visible, false);
}

std::vector<int32_t> CrisperWhisperAssets::prompt(
    const std::string & language, const std::string & mode, const std::string & context,
    const std::string & transcript) const {
    if (mode != "verbatim" && mode != "intended" && mode != "verbatimize") {
        throw std::runtime_error("CrisperWhisper transcription_mode must be verbatim, intended or verbatimize");
    }
    std::string text;
    for (int i = 1; i <= 5; ++i) {
        text += "[" + (mode == "verbatimize" ? std::string("verbatim") : mode) + "_" + std::to_string(i) + "]";
    }
    if (mode == "verbatimize") {
        text += " <vtx> " + transcript + " <evtx>";
    } else if (!context.empty()) {
        text += " <ctx> " + context + " <ectx>";
    }
    auto tokens = tokenizer->encode(text, true);
    for (const auto & token : {std::string("<|startoftranscript|>"), "<|" + language + "|>",
                              std::string("<|transcribe|>"), std::string("<|notimestamps|>")}) {
        const auto id = tokenizer->find_token_id(token);
        if (!id) {
            throw std::runtime_error("CrisperWhisper tokenizer does not contain " + token);
        }
        tokens.push_back(*id);
    }
    return tokens;
}

}  // namespace engine::models::crisperwhisper
