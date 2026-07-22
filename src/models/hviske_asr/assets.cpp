#include "engine/models/hviske_asr/assets.h"

#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/json.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::hviske_asr {
namespace json = engine::io::json;
namespace {

int64_t seconds_to_samples(const json::Value & root, const char * key, int64_t sample_rate) {
    return static_cast<int64_t>(std::llround(root.require(key).as_number() * static_cast<double>(sample_rate)));
}

HviskeConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    const auto generation = resources.parse_json("generation_config");

    HviskeConfig config;
    config.model_type = json::require_string(root, "model_type");
    if (const auto * architectures = root.find("architectures");
        architectures != nullptr && architectures->is_array() && !architectures->as_array().empty()) {
        config.variant = architectures->as_array().front().as_string();
    } else {
        config.variant = "CohereAsrForConditionalGeneration";
    }
    config.max_audio_clip_seconds = json::require_i64(root, "max_audio_clip_s");
    config.overlap_chunk_seconds = json::require_i64(root, "overlap_chunk_second");
    config.min_energy_window_samples = json::require_i64(root, "min_energy_window_samples");
    config.supported_languages = json::require_string_array(root, "supported_languages");

    const auto & pre = root.require("preprocessor");
    config.frontend.sample_rate = json::require_i64(pre, "sample_rate");
    config.frontend.features = json::require_i64(pre, "features");
    config.frontend.n_fft = json::require_i64(pre, "n_fft");
    config.frontend.win_length = seconds_to_samples(pre, "window_size", config.frontend.sample_rate);
    config.frontend.hop_length = seconds_to_samples(pre, "window_stride", config.frontend.sample_rate);
    config.frontend.pad_to = json::require_i64(pre, "pad_to");
    config.frontend.dither = json::require_f32(pre, "dither");

    const auto & enc = root.require("encoder");
    config.encoder.feat_in = json::require_i64(enc, "feat_in");
    config.encoder.hidden_size = json::require_i64(enc, "d_model");
    config.encoder.intermediate_size =
        config.encoder.hidden_size * json::require_i64(enc, "ff_expansion_factor");
    config.encoder.layers = json::require_i64(enc, "n_layers");
    config.encoder.heads = json::require_i64(enc, "n_heads");
    config.encoder.conv_kernel = json::require_i64(enc, "conv_kernel_size");
    config.encoder.subsampling_conv_channels = json::require_i64(enc, "subsampling_conv_channels");
    config.encoder.subsampling_factor = json::require_i64(enc, "subsampling_factor");
    config.encoder.pos_emb_max_len = json::require_i64(enc, "pos_emb_max_len");

    const auto & dec = root.require("transf_decoder").require("config_dict");
    config.decoder.vocab_size = json::require_i64(root, "vocab_size");
    config.decoder.hidden_size = json::require_i64(dec, "hidden_size");
    config.decoder.intermediate_size = json::require_i64(dec, "inner_size");
    config.decoder.layers = json::require_i64(dec, "num_layers");
    config.decoder.heads = json::require_i64(dec, "num_attention_heads");
    config.decoder.max_sequence_length = json::require_i64(dec, "max_sequence_length");
    config.decoder.output_log_probs = json::require_bool(root.require("head"), "log_softmax");
    config.decoder.pad_token_id = json::require_i64(generation, "pad_token_id");
    config.decoder.eos_token_id = json::require_i64(generation, "eos_token_id");
    config.decoder.bos_token_id = json::require_i64(generation, "bos_token_id");
    config.decoder.decoder_start_token_id = json::require_i64(generation, "decoder_start_token_id");
    return config;
}

void validate_config(const HviskeConfig & config) {
    if (config.model_type != "cohere_asr") {
        throw std::runtime_error("Hviske ASR expects Cohere ASR config, got: " + config.model_type);
    }
}

}  // namespace

std::shared_ptr<const HviskeASRAssets> load_hviske_asr_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<HviskeASRAssets>();
    assets->resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("hviske_asr"));
    assets->config = parse_config(assets->resources);
    validate_config(assets->config);
    assets->tokenizer_pieces = engine::tokenizers::load_sentencepiece_model(
        assets->resources.require_file("tokenizer"));
    assets->model_weights = assets->resources.open_tensor_source("weights");
    return assets;
}

std::vector<int32_t> tokenize_hviske_prompt(
    const HviskeASRAssets & assets,
    const std::string & language,
    bool punctuation) {
    const std::string pnc = punctuation ? "<|pnc|>" : "<|nopnc|>";
    const std::string prompt =
        "<|startofcontext|><|startoftranscript|><|emo:undefined|><|" + language + "|><|" +
        language + "|>" + pnc + "<|noitn|><|notimestamp|><|nodiarize|>";
    return engine::tokenizers::tokenize_sentencepiece(assets.tokenizer_pieces, prompt);
}

std::string decode_hviske_tokens(
    const HviskeASRAssets & assets,
    const std::vector<int32_t> & token_ids) {
    std::vector<int32_t> visible;
    visible.reserve(token_ids.size());
    for (const int32_t token_id : token_ids) {
        if (token_id < 0 || token_id >= static_cast<int32_t>(assets.tokenizer_pieces.size())) {
            continue;
        }
        const auto type = assets.tokenizer_pieces[static_cast<size_t>(token_id)].type;
        if (type == engine::tokenizers::SentencePieceType::Normal ||
            type == engine::tokenizers::SentencePieceType::Byte) {
            visible.push_back(token_id);
        }
    }
    return engine::tokenizers::decode_sentencepiece(assets.tokenizer_pieces, visible);
}

}  // namespace engine::models::hviske_asr
