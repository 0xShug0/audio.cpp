#include "engine/models/gigaam_asr/model.h"

#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::gigaam_asr {

std::string GigaAMAssets::decode(const std::vector<int32_t> & ids) const {
    if (!pieces.empty()) {
        return tokenizers::decode_sentencepiece(pieces, ids);
    }
    std::string text;
    for (const auto id : ids) {
        text += characters.at(static_cast<size_t>(id));
    }
    return text;
}

std::shared_ptr<const GigaAMAssets> load_gigaam_assets(const std::filesystem::path & path) {
    namespace json = io::json;
    auto out = std::make_shared<GigaAMAssets>();
    out->resources = model_spec::load_resource_bundle_for_family(path, "gigaam_asr");
    out->source = out->resources.open_tensor_source("weights");
    const auto config = out->resources.parse_json("config");
    const auto & encoder = config.require("encoder");
    auto & c = out->encoder;
    c.features = json::require_i64(encoder, "feat_in");
    c.hidden_size = json::require_i64(encoder, "d_model");
    c.intermediate_size = c.hidden_size * json::require_i64(encoder, "ff_expansion_factor");
    c.heads = json::require_i64(encoder, "n_heads");
    c.conv_kernel = json::require_i64(encoder, "conv_kernel_size");
    c.subsampling_kernel = json::require_i64(encoder, "subs_kernel_size");
    c.rope_theta = static_cast<float>(json::require_i64(encoder, "pos_emb_max_len"));
    out->layers = json::require_i64(encoder, "n_layers");
    if (json::require_string(encoder, "self_attention_model") != "rotary" ||
        json::require_string(encoder, "conv_norm_type") != "layer_norm" ||
        json::require_string(encoder, "subsampling") != "conv1d" ||
        json::require_i64(encoder, "subsampling_factor") != 4) {
        throw std::runtime_error("GigaAM requires the v3/multilingual rotary Conformer architecture");
    }
    const auto & head = config.require("head");
    const auto & decoding = config.require("decoding");
    const auto decoder_type = json::require_string(decoding, "_target_");
    out->rnnt = decoder_type == "modeling_gigaam.RNNTGreedyDecoding";
    if (!out->rnnt && decoder_type != "modeling_gigaam.CTCGreedyDecoding") {
        throw std::runtime_error("Unsupported GigaAM decoding type: " + decoder_type);
    }
    if (out->rnnt) {
        const auto & predictor = head.require("decoder");
        const auto & joint = head.require("joint");
        if (json::require_i64(predictor, "pred_rnn_layers") != 1) {
            throw std::runtime_error("GigaAM RNN-T requires a single-layer LSTM predictor");
        }
        out->predictor_hidden = json::require_i64(predictor, "pred_hidden");
        out->joint_hidden = json::require_i64(joint, "joint_hidden");
        out->classes = json::require_i64(joint, "num_classes");
        out->max_symbols_per_step = json::optional_i64(decoding, "max_symbols_per_step", 10);
    } else {
        out->classes = json::require_i64(head, "num_classes");
    }
    const auto tokenizer = json::optional_nullable_string(decoding, "model_path", "");
    if (!tokenizer.empty()) {
        out->resources.add_model_file("tokenizer", tokenizer);
        out->pieces = tokenizers::load_sentencepiece_model(out->resources.require_file("tokenizer"));
    } else {
        out->characters = json::require_string_array(decoding, "vocabulary");
    }
    if (static_cast<int64_t>(out->pieces.size() + out->characters.size()) + 1 != out->classes) {
        throw std::runtime_error("GigaAM tokenizer size does not match the classifier");
    }

    const auto & pre = config.require("preprocessor");
    if (json::require_i64(pre, "sample_rate") != 16000 ||
        json::require_i64(pre, "features") != c.features) {
        throw std::runtime_error("GigaAM frontend configuration does not match the encoder");
    }
    audio::STFTConfig stft;
    stft.n_fft = json::require_i64(pre, "n_fft");
    stft.win_length = json::require_i64(pre, "win_length");
    stft.hop_length = json::require_i64(pre, "hop_length");
    stft.center = json::require_bool(pre, "center");
    const auto window = out->source->require_f32("preprocessor.featurizer.0.spectrogram.window", {stft.win_length});
    const int64_t bins = stft.n_fft / 2 + 1;
    const auto bank = out->source->require_f32("preprocessor.featurizer.0.mel_scale.fb", {bins, c.features});
    audio::AudioTensor filterbank;
    filterbank.shape = {c.features, bins};
    filterbank.values.resize(bank.size());
    for (int64_t m = 0; m < c.features; ++m) {
        for (int64_t f = 0; f < bins; ++f) {
            filterbank.values[static_cast<size_t>(m * bins + f)] = bank[static_cast<size_t>(f * c.features + m)];
        }
    }
    out->frontend = std::make_unique<GigaAMFrontend>(stft, window, std::move(filterbank));
    return out;
}

}  // namespace engine::models::gigaam_asr
