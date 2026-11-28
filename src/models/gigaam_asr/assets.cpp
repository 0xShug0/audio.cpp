#include "engine/models/gigaam_asr/model.h"

#include "engine/framework/model_spec/package.h"

#include <algorithm>
#include <stdexcept>

namespace engine::models::gigaam_asr {

void load_zapformer_config(GigaAMAssets & out, const io::json::Value & config);
void load_frontend(GigaAMAssets & assets, const io::json::Value & config, int64_t features);

size_t GigaAMAssets::head_index(const std::string & name) const {
    if (head_names.empty()) {
        return 0;
    }
    for (size_t i = 0; i < head_names.size(); ++i) {
        if (head_names[i] == name) {
            return i;
        }
    }
    throw std::runtime_error("GigaAM checkpoint has no head named " + name);
}

const std::vector<tokenizers::SentencePiecePiece> & GigaAMAssets::pieces_for(size_t head) const {
    return head_pieces.empty() ? pieces : head_pieces.at(head);
}

std::string GigaAMAssets::decode(const std::vector<int32_t> & ids, size_t head) const {
    const auto & head_tokens = pieces_for(head);
    if (!head_tokens.empty()) {
        return tokenizers::decode_sentencepiece(head_tokens, ids);
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
    if (json::optional_string(encoder, "_target_", "").find("ZapformerEncoder") != std::string::npos) {
        load_zapformer_config(*out, config);
        load_frontend(*out, config, out->zapformer_encoder.features);
        return out;
    }
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

    load_frontend(*out, config, c.features);
    return out;
}

namespace {

std::vector<int64_t> zap_tuple(const io::json::Value & object, const std::string & key, size_t count) {
    const auto & value = object.require(key);
    std::vector<int64_t> out;
    if (value.is_string()) {
        const auto & text = value.as_string();
        size_t start = 0;
        while (start <= text.size()) {
            const auto end = std::min(text.find(',', start), text.size());
            out.push_back(std::stoll(text.substr(start, end - start)));
            start = end + 1;
        }
    } else if (value.is_array()) {
        for (const auto & item : value.as_array()) out.push_back(item.as_i64());
    } else {
        out.push_back(value.as_i64());
    }
    if (out.size() == 1 && count > 1) {
        out.assign(count, out[0]);
    }
    if (count != 0 && out.size() != count) {
        throw std::runtime_error("GigaAM-v4 encoder option " + key + " has the wrong number of values");
    }
    return out;
}

}  // namespace

void load_zapformer_config(GigaAMAssets & out, const io::json::Value & config) {
    namespace json = io::json;
    const auto & encoder = config.require("encoder");
    if (json::optional_string(encoder, "activation", "swish") != "SiLU") {
        throw std::runtime_error("GigaAM-v4 port supports the SiLU Zapformer activation only");
    }
    if (json::optional_bool(encoder, "causal", false)) {
        throw std::runtime_error("GigaAM-v4 port supports the full-context (non-causal) encoder only");
    }
    if (!json::optional_bool(encoder, "causal_frontend", false)) {
        throw std::runtime_error("GigaAM-v4 port supports the causal frontend only");
    }
    if (encoder.find("embed_dim") != nullptr && !encoder.require("embed_dim").is_null()) {
        throw std::runtime_error("GigaAM-v4 port requires embed_dim = encoder_dim[0]");
    }
    auto & z = out.zapformer_encoder;
    out.zapformer = true;
    z.features = json::optional_i64(encoder, "in_channels", 64);
    const auto frontend = zap_tuple(encoder, "frontend_channels", 3);
    std::copy(frontend.begin(), frontend.end(), z.frontend_channels);
    const auto ds = zap_tuple(encoder, "downsampling_factor", 0);
    const auto n = ds.size();
    const auto dims = zap_tuple(encoder, "encoder_dim", n);
    const auto layers = zap_tuple(encoder, "num_encoder_layers", n);
    const auto heads = zap_tuple(encoder, "num_heads", n);
    const auto vdims = zap_tuple(encoder, "value_head_dim", n);
    const auto multiples = zap_tuple(encoder, "feedforward_multiple", n);
    const auto kernels = zap_tuple(encoder, "conv_params", n);
    std::vector<int64_t> qdims;
    if (encoder.find("query_head_dim") != nullptr && !encoder.require("query_head_dim").is_null()) {
        qdims = zap_tuple(encoder, "query_head_dim", n);
    } else {
        for (size_t i = 0; i < n; ++i) qdims.push_back(dims[i] / heads[i]);
    }
    z.embed_dim = dims[0];
    for (size_t i = 0; i < n; ++i) {
        if (qdims[i] % 2 != 0 || (kernels[i] != 0 && kernels[i] % 2 == 0)) {
            throw std::runtime_error("GigaAM-v4 stack has an unsupported head or kernel width");
        }
        z.stacks.push_back({dims[i], layers[i], heads[i], qdims[i], vdims[i], multiples[i], kernels[i], ds[i]});
    }

    const auto & heads_config = config.require("heads");
    const auto & decodings = config.require("decodings");
    for (const auto * name : {"e2e", "raw"}) {
        const auto * head = heads_config.find(name);
        if (head == nullptr) {
            continue;
        }
        if (json::require_string(*head, "_target_").find("RNNTHead") == std::string::npos) {
            throw std::runtime_error("GigaAM-v4 port supports RNN-T heads only");
        }
        const auto & predictor = head->require("decoder");
        const auto & joint = head->require("joint");
        if (json::require_i64(predictor, "pred_rnn_layers") != 1) {
            throw std::runtime_error("GigaAM RNN-T requires a single-layer LSTM predictor");
        }
        const auto hidden = json::require_i64(predictor, "pred_hidden");
        const auto joint_hidden = json::require_i64(joint, "joint_hidden");
        const auto classes = json::require_i64(joint, "num_classes");
        if (json::require_i64(joint, "enc_hidden") != z.embed_dim ||
            (!out.head_names.empty() && (hidden != out.predictor_hidden || joint_hidden != out.joint_hidden ||
                                         classes != out.classes))) {
            throw std::runtime_error("GigaAM-v4 heads must share their sizes and match the encoder width");
        }
        out.predictor_hidden = hidden;
        out.joint_hidden = joint_hidden;
        out.classes = classes;
        const auto & decoding = decodings.require(name);
        out.max_symbols_per_step = json::optional_i64(decoding, "max_symbols_per_step", 10);
        const auto file = json::require_string(decoding, "model_path");
        const std::string id = std::string("tokenizer_") + name;
        out.resources.add_model_file(id, file);
        out.head_pieces.push_back(tokenizers::load_sentencepiece_model(out.resources.require_file(id)));
        if (static_cast<int64_t>(out.head_pieces.back().size()) + 1 != classes) {
            throw std::runtime_error("GigaAM tokenizer size does not match the classifier");
        }
        out.head_names.push_back(name);
    }
    if (out.head_names.empty()) {
        throw std::runtime_error("GigaAM-v4 checkpoint has no e2e or raw head");
    }
    out.rnnt = true;
}

void load_frontend(GigaAMAssets & assets, const io::json::Value & config, int64_t features) {
    namespace json = io::json;
    auto * out = &assets;
    const auto & pre = config.require("preprocessor");
    if (json::require_i64(pre, "sample_rate") != 16000 ||
        json::require_i64(pre, "features") != features) {
        throw std::runtime_error("GigaAM frontend configuration does not match the encoder");
    }
    audio::STFTConfig stft;
    stft.n_fft = json::require_i64(pre, "n_fft");
    stft.win_length = json::require_i64(pre, "win_length");
    stft.hop_length = json::require_i64(pre, "hop_length");
    stft.center = json::require_bool(pre, "center");
    const auto window = out->source->require_f32("preprocessor.featurizer.0.spectrogram.window", {stft.win_length});
    const int64_t bins = stft.n_fft / 2 + 1;
    const auto bank = out->source->require_f32("preprocessor.featurizer.0.mel_scale.fb", {bins, features});
    audio::AudioTensor filterbank;
    filterbank.shape = {features, bins};
    filterbank.values.resize(bank.size());
    for (int64_t m = 0; m < features; ++m) {
        for (int64_t f = 0; f < bins; ++f) {
            filterbank.values[static_cast<size_t>(m * bins + f)] = bank[static_cast<size_t>(f * features + m)];
        }
    }
    out->frontend = std::make_unique<GigaAMFrontend>(stft, window, std::move(filterbank));
}

}  // namespace engine::models::gigaam_asr
