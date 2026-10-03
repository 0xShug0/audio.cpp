#include "engine/models/owsm/model.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::owsm {
namespace {

int32_t model_id_from_sentencepiece(int32_t id) {
    if (id == 0) {
        return 1;
    }
    if (id >= 3 && id < 50000) {
        return id - 1;
    }
    throw std::runtime_error("OWSM v4 SentencePiece produced a reserved BOS or EOS token");
}

int32_t sentencepiece_id_from_model(int32_t id) {
    if (id == 1) {
        return 0;
    }
    if (id >= 2 && id <= 49998) {
        return id + 1;
    }
    return -1;
}

}  // namespace

int32_t OWSMV4Assets::token_id(const std::string & token) const {
    if (token == "<blank>") {
        return config.blank_id;
    }
    if (token == "<sos>") {
        return config.sos_id;
    }
    if (token == "<eos>") {
        return config.eos_id;
    }
    if (token == "<sop>") {
        return config.sop_id;
    }
    for (const auto & piece : sentencepiece) {
        if (piece.text == token) {
            return model_id_from_sentencepiece(piece.id);
        }
    }
    throw std::runtime_error("OWSM v4 tokenizer is missing " + token);
}

std::vector<int32_t> OWSMV4Assets::tokenize_text(const std::string & text) const {
    auto ids = tokenizers::tokenize_sentencepiece(sentencepiece, text);
    for (auto & id : ids) {
        id = model_id_from_sentencepiece(id);
    }
    return ids;
}

std::string OWSMV4Assets::decode_visible(const std::vector<int32_t> & ids) const {
    std::vector<int32_t> visible;
    visible.reserve(ids.size());
    for (const auto id : ids) {
        const auto sentencepiece_id = sentencepiece_id_from_model(id);
        if (sentencepiece_id < 0) {
            continue;
        }
        const auto & text = sentencepiece[static_cast<size_t>(sentencepiece_id)].text;
        if (text.size() >= 2 && text.front() == '<' && text.back() == '>') {
            continue;
        }
        visible.push_back(sentencepiece_id);
    }
    return tokenizers::decode_sentencepiece(sentencepiece, visible);
}

std::shared_ptr<const OWSMV4Assets> load_owsm_assets(const std::filesystem::path & path) {
    auto out = std::make_shared<OWSMV4Assets>();
    out->resources = model_spec::load_resource_bundle(path, model_spec::default_spec_path("owsm"));
    out->source = out->resources.open_tensor_source("weights");
    const auto config = out->resources.parse_json("config");
    auto & model = out->config;
    model.variant = io::json::require_string(config, "variant");
    model.hidden_size = io::json::require_i64(config, "hidden_size");
    model.num_heads = io::json::require_i64(config, "num_heads");
    model.encoder_layers = io::json::require_i64(config, "encoder_layers");
    model.decoder_layers = io::json::require_i64(config, "decoder_layers");
    model.intermediate_size = io::json::require_i64(config, "intermediate_size");
    model.vocabulary_size = io::json::require_i64(config, "vocabulary_size");
    model.max_audio_samples = io::json::require_i64(config, "max_audio_samples");
    model.frontend_frames = io::json::require_i64(config, "frontend_frames");
    model.encoder_frames = io::json::require_i64(config, "encoder_frames");
    model.max_decode_tokens = io::json::require_i64(config, "max_decode_tokens");
    if ((model.variant != "base" && model.variant != "small" && model.variant != "medium") ||
        model.hidden_size <= 0 || model.hidden_size % model.num_heads != 0 ||
        model.encoder_layers <= 0 || model.decoder_layers <= 0 ||
        model.intermediate_size != model.hidden_size * 4 || model.vocabulary_size != 50002 ||
        model.max_audio_samples != 480000 || model.frontend_frames != 3001 ||
        model.encoder_frames != 374 || model.max_decode_tokens != 374) {
        throw std::runtime_error("OWSM v4 config does not describe a supported base, small, or medium checkpoint");
    }

    out->sentencepiece = tokenizers::load_sentencepiece_model(out->resources.require_file("tokenizer"));
    if (out->sentencepiece.size() != 50000 || out->token_id("<notimestamps>") != model.notimestamps_id ||
        out->token_id("<0.00>") != model.first_timestamp_id ||
        out->token_id("<30.00>") != model.last_timestamp_id) {
        throw std::runtime_error("OWSM v4 tokenizer does not match the published 50,000-piece vocabulary");
    }

    audio::MelSpectrogramFrontendConfig frontend;
    frontend.sample_rate = 16000;
    frontend.n_fft = 512;
    frontend.hop_length = 160;
    frontend.win_length = 400;
    frontend.n_mels = 128;
    frontend.mel_fmax = 8000.0F;
    frontend.filterbank_normalization = audio::MelFilterbankNormalization::Slaney;
    frontend.stft_pad_mode = audio::STFTPadMode::Reflect;
    frontend.window = audio::MelHannWindow::Periodic;
    frontend.stft_center = true;
    frontend.waveform_padding = audio::MelWaveformPadding::None;
    frontend.filterbank_projection = audio::MelFilterbankProjection::DenseF32;
    frontend.spectrum_mode = audio::MelSpectrumMode::PowerBeforeProjection;
    frontend.log_floor = 1.0e-10;
    frontend.log_precision = audio::MelLogPrecision::F32;
    frontend.value_transform = audio::MelValueTransform::Ln;
    frontend.layout = audio::MelOutputLayout::TimeMajor;
    frontend.require_mono = true;
    frontend.minimum_samples = 400;
    out->frontend = audio::get_cached_mel_spectrogram_frontend(frontend);
    out->feature_mean = out->source->require_f32("normalize.mean", {128});
    out->feature_std = out->source->require_f32("normalize.std", {128});
    return out;
}

std::unique_ptr<OWSMV4Weights> load_owsm_weights(
    const OWSMV4Assets & assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType type) {
    const auto & config = assets.config;
    auto out = std::make_unique<OWSMV4Weights>();
    out->store = std::make_unique<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "owsm.weights", 4 * 1024 * 1024);
    auto & store = *out->store;
    const auto & source = *assets.source;
    const auto d = config.hidden_size;
    const auto ff = config.intermediate_size;

    const auto linear = [&](const std::string & name, int64_t output, int64_t input) {
        return modules::binding::linear_from_source(store, source, name, type, output, input, true);
    };
    const auto norm = [&](const std::string & name, int64_t size) {
        return modules::binding::norm_from_source(store, source, name, size);
    };
    const auto feed_forward = [&](const std::string & name) {
        const auto first = linear(name + ".w_1", ff, d);
        const auto second = linear(name + ".w_2", d, ff);
        return modules::FeedForwardWeights{first.weight, first.bias, second.weight, second.bias};
    };
    const auto attention = [&](const std::string & name, bool packed_qkv, bool packed_kv) {
        modules::AttentionWeights weights;
        if (packed_qkv) {
            const auto packed = modules::PackedLinearWeightsBuilder({d, {
                {name + ".linear_q.weight", name + ".linear_q.bias", d},
                {name + ".linear_k.weight", name + ".linear_k.bias", d},
                {name + ".linear_v.weight", name + ".linear_v.bias", d}}, true}).build(store, source, type);
            weights.qkv_weight = packed.weight;
            weights.qkv_bias = packed.bias;
        } else {
            const auto query = linear(name + ".linear_q", d, d);
            weights.q_weight = query.weight;
            weights.q_bias = query.bias;
            if (packed_kv) {
                const auto packed = modules::PackedLinearWeightsBuilder({d, {
                    {name + ".linear_k.weight", name + ".linear_k.bias", d},
                    {name + ".linear_v.weight", name + ".linear_v.bias", d}}, true}).build(store, source, type);
                weights.qkv_weight = packed.weight;
                weights.qkv_bias = packed.bias;
            }
        }
        const auto output = linear(name + ".linear_out", d, d);
        weights.out_weight = output.weight;
        weights.out_bias = output.bias;
        return weights;
    };

    out->subsampling.conv0 = modules::binding::conv2d_from_source(
        store, source, "encoder.embed.conv.0", assets::TensorStorageType::F32, d, 1, 3, 3, true);
    out->subsampling.conv1 = modules::binding::conv2d_from_source(
        store, source, "encoder.embed.conv.2", assets::TensorStorageType::F32, d, d, 3, 3, true);
    out->subsampling.conv2 = modules::binding::conv2d_from_source(
        store, source, "encoder.embed.conv.4", assets::TensorStorageType::F32, d, d, 3, 3, true);
    out->subsampling.projection = linear("encoder.embed.out", d, d * 15);

    out->encoder.reserve(static_cast<size_t>(config.encoder_layers));
    for (int64_t index = 0; index < config.encoder_layers; ++index) {
        const auto prefix = "encoder.encoders." + std::to_string(index);
        modules::EBranchformerBlockWeights layer;
        layer.macaron_ffn = feed_forward(prefix + ".feed_forward_macaron");
        layer.macaron_norm = norm(prefix + ".norm_ff_macaron", d);
        layer.attention_norm = norm(prefix + ".norm_mha", d);
        layer.attention = attention(prefix + ".attn", true, false);
        layer.cgmlp_norm = norm(prefix + ".norm_mlp", d);
        layer.cgmlp.input_projection = linear(prefix + ".cgmlp.channel_proj1.0", ff, d);
        layer.cgmlp.gate_norm = norm(prefix + ".cgmlp.csgu.norm", ff / 2);
        layer.cgmlp.gate_conv = modules::binding::depthwise_conv1d_from_source(
            store, source, prefix + ".cgmlp.csgu.conv", assets::TensorStorageType::F32, ff / 2, 31, true);
        layer.cgmlp.output_projection = linear(prefix + ".cgmlp.channel_proj2", d, ff / 2);
        layer.merge_conv = modules::binding::depthwise_conv1d_from_source(
            store, source, prefix + ".depthwise_conv_fusion", assets::TensorStorageType::F32, d * 2, 31, true);
        layer.merge_projection = linear(prefix + ".merge_proj", d, d * 2);
        layer.final_ffn = feed_forward(prefix + ".feed_forward");
        layer.final_ffn_norm = norm(prefix + ".norm_ff", d);
        layer.output_norm = norm(prefix + ".norm_final", d);
        out->encoder.push_back(std::move(layer));
    }
    out->encoder_norm = norm("encoder.after_norm", d);

    out->embedding = store.load_tensor(source, "decoder.embed.0.weight", type, {config.vocabulary_size, d});
    out->decoder.reserve(static_cast<size_t>(config.decoder_layers));
    for (int64_t index = 0; index < config.decoder_layers; ++index) {
        const auto prefix = "decoder.decoders." + std::to_string(index);
        modules::TransformerDecoderBlockWeights layer;
        layer.norm1 = norm(prefix + ".norm1", d);
        layer.self_attention = attention(prefix + ".self_attn", true, false);
        layer.norm2 = norm(prefix + ".norm2", d);
        layer.cross_attention = attention(prefix + ".src_attn", false, true);
        layer.norm3 = norm(prefix + ".norm3", d);
        layer.feed_forward = feed_forward(prefix + ".feed_forward");
        out->decoder.push_back(std::move(layer));
    }
    out->decoder_norm = norm("decoder.after_norm", d);
    out->output = linear("decoder.output_layer", config.vocabulary_size, d);
    out->half_scale = store.make_f32(core::TensorShape::from_dims({d}), std::vector<float>(static_cast<size_t>(d), 0.5F));
    out->embedding_scale = store.make_f32(
        core::TensorShape::from_dims({d}), std::vector<float>(static_cast<size_t>(d), std::sqrt(static_cast<float>(d))));
    store.upload();
    return out;
}

}  // namespace engine::models::owsm
