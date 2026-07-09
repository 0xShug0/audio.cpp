#include "engine/models/hviske_asr/assets.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/json.h"

#include <stdexcept>
#include <utility>

namespace engine::models::hviske_asr {
namespace {

int64_t seconds_to_samples(const engine::io::json::Value & root, const char * key, int64_t sample_rate) {
    return static_cast<int64_t>(root.require(key).as_number() * static_cast<double>(sample_rate));
}

std::vector<std::string> string_array(const engine::io::json::Value & value) {
    std::vector<std::string> out;
    for (const auto & item : value.as_array()) {
        out.push_back(item.as_string());
    }
    return out;
}

HviskeConfig load_config(const std::filesystem::path & path) {
    const auto root = engine::io::json::parse_file(path);
    HviskeConfig config;
    config.model_type = root.require("model_type").as_string();
    if (const auto * architectures = root.find("architectures");
        architectures != nullptr && architectures->is_array() && !architectures->as_array().empty()) {
        config.variant = architectures->as_array().front().as_string();
    } else {
        config.variant = "CohereAsrForConditionalGeneration";
    }
    config.max_audio_clip_seconds = root.require("max_audio_clip_s").as_i64();
    config.overlap_chunk_seconds = root.require("overlap_chunk_second").as_i64();
    config.min_energy_window_samples = root.require("min_energy_window_samples").as_i64();
    config.supported_languages = string_array(root.require("supported_languages"));

    const auto & pre = root.require("preprocessor");
    config.frontend.sample_rate = pre.require("sample_rate").as_i64();
    config.frontend.features = pre.require("features").as_i64();
    config.frontend.n_fft = pre.require("n_fft").as_i64();
    config.frontend.win_length = seconds_to_samples(pre, "window_size", config.frontend.sample_rate);
    config.frontend.hop_length = seconds_to_samples(pre, "window_stride", config.frontend.sample_rate);
    config.frontend.pad_to = pre.require("pad_to").as_i64();
    config.frontend.dither = pre.require("dither").as_f32();

    const auto & enc = root.require("encoder");
    config.encoder.feat_in = enc.require("feat_in").as_i64();
    config.encoder.hidden_size = enc.require("d_model").as_i64();
    config.encoder.intermediate_size =
        config.encoder.hidden_size * enc.require("ff_expansion_factor").as_i64();
    config.encoder.layers = enc.require("n_layers").as_i64();
    config.encoder.heads = enc.require("n_heads").as_i64();
    config.encoder.conv_kernel = enc.require("conv_kernel_size").as_i64();
    config.encoder.subsampling_conv_channels = enc.require("subsampling_conv_channels").as_i64();
    config.encoder.subsampling_factor = enc.require("subsampling_factor").as_i64();
    config.encoder.pos_emb_max_len = enc.require("pos_emb_max_len").as_i64();

    const auto & dec = root.require("transf_decoder").require("config_dict");
    config.decoder.vocab_size = root.require("vocab_size").as_i64();
    config.decoder.hidden_size = dec.require("hidden_size").as_i64();
    config.decoder.intermediate_size = dec.require("inner_size").as_i64();
    config.decoder.layers = dec.require("num_layers").as_i64();
    config.decoder.heads = dec.require("num_attention_heads").as_i64();
    config.decoder.max_sequence_length = dec.require("max_sequence_length").as_i64();
    config.decoder.output_log_probs = root.require("head").require("log_softmax").as_bool();
    return config;
}

void apply_generation_config(HviskeConfig & config, const std::filesystem::path & path) {
    const auto root = engine::io::json::parse_file(path);
    config.decoder.pad_token_id = root.require("pad_token_id").as_i64();
    config.decoder.eos_token_id = root.require("eos_token_id").as_i64();
    config.decoder.bos_token_id = root.require("bos_token_id").as_i64();
    config.decoder.decoder_start_token_id = root.require("decoder_start_token_id").as_i64();
}

}  // namespace

std::shared_ptr<const HviskeAssets> load_hviske_assets(const std::filesystem::path & model_root) {
    engine::assets::ResourceBundle resources(model_root);
    resources.add_model_files({
        {"config", "config.json"},
        {"generation_config", "generation_config.json"},
        {"tokenizer", "tokenizer.model"},
        {"weights", "model.safetensors"},
    });

    auto assets = std::make_shared<HviskeAssets>();
    assets->model_root = resources.model_root();
    assets->config = load_config(resources.require_file("config"));
    apply_generation_config(assets->config, resources.require_file("generation_config"));
    if (assets->config.model_type != "cohere_asr") {
        throw std::runtime_error("Hviske ASR expects Cohere ASR config, got: " + assets->config.model_type);
    }
    assets->tokenizer_pieces = engine::tokenizers::load_sentencepiece_model(resources.require_file("tokenizer"));
    assets->model_weights = resources.open_tensor_source("weights");
    return assets;
}

std::vector<int32_t> tokenize_hviske_prompt(
    const HviskeAssets & assets,
    const std::string & language,
    bool punctuation) {
    const std::string pnc = punctuation ? "<|pnc|>" : "<|nopnc|>";
    const std::string prompt =
        "<|startofcontext|><|startoftranscript|><|emo:undefined|><|" + language + "|><|" +
        language + "|>" + pnc + "<|noitn|><|notimestamp|><|nodiarize|>";
    return engine::tokenizers::tokenize_sentencepiece(assets.tokenizer_pieces, prompt);
}

std::string decode_hviske_tokens(
    const HviskeAssets & assets,
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
