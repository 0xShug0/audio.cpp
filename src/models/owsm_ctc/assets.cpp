#include "engine/models/owsm_ctc/model.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::owsm_ctc {
namespace {

int32_t model_id_from_sentencepiece(int32_t id) {
    if (id == 0) {
        return 1;
    }
    if (id >= 3 && id < 50000) {
        return id - 1;
    }
    throw std::runtime_error("OWSM-CTC v4 SentencePiece produced a reserved BOS or EOS token");
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

int32_t OWSMCTCV4Assets::token_id(const std::string & token) const {
    if (token == "<blank>") {
        return config.blank_id;
    }
    for (const auto & piece : sentencepiece) {
        if (piece.text == token) {
            return model_id_from_sentencepiece(piece.id);
        }
    }
    throw std::runtime_error("OWSM-CTC v4 tokenizer is missing " + token);
}

std::string OWSMCTCV4Assets::decode_visible(const std::vector<int32_t> & ids) const {
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

std::shared_ptr<const OWSMCTCV4Assets> load_owsm_ctc_assets(const std::filesystem::path & path) {
    auto out = std::make_shared<OWSMCTCV4Assets>();
    out->resources = model_spec::load_resource_bundle(path, model_spec::default_spec_path("owsm_ctc"));
    out->source = out->resources.open_tensor_source("weights");
    const auto config = out->resources.parse_json("config");
    auto & model = out->config;
    model.variant = io::json::require_string(config, "variant");
    model.prompt_hidden_size = io::json::require_i64(config, "prompt_hidden_size");
    model.prompt_num_heads = io::json::require_i64(config, "prompt_num_heads");
    model.prompt_layers = io::json::require_i64(config, "prompt_layers");
    model.prompt_intermediate_size = io::json::require_i64(config, "prompt_intermediate_size");
    model.interctc_layers = io::json::require_i64_array(config, "interctc_layers");
    model.cross_attention_layers = io::json::require_i64_array(config, "cross_attention_layers");
    model.hidden_size = io::json::require_i64(config, "hidden_size");
    model.num_heads = io::json::require_i64(config, "num_heads");
    model.encoder_layers = io::json::require_i64(config, "encoder_layers");
    model.intermediate_size = io::json::require_i64(config, "intermediate_size");
    model.vocabulary_size = io::json::require_i64(config, "vocabulary_size");
    model.max_audio_samples = io::json::require_i64(config, "max_audio_samples");
    model.frontend_frames = io::json::require_i64(config, "frontend_frames");
    model.encoder_frames = io::json::require_i64(config, "encoder_frames");
    if (model.variant != "medium" || model.hidden_size != 1024 || model.num_heads != 16 ||
        model.encoder_layers != 27 || model.intermediate_size != 4096 ||
        model.vocabulary_size != 50002 || model.max_audio_samples != 480000 ||
        model.frontend_frames != 3001 || model.encoder_frames != 374 ||
        model.prompt_hidden_size != 512 || model.prompt_num_heads != 8 || model.prompt_layers != 4 ||
        model.prompt_intermediate_size != 2048 ||
        model.interctc_layers != std::vector<int64_t>{6, 12, 15, 21} ||
        model.cross_attention_layers != std::vector<int64_t>{2, 5, 8, 11, 14, 17, 20, 23, 26}) {
        throw std::runtime_error("OWSM-CTC v4 config does not describe the supported 1B checkpoint");
    }

    out->sentencepiece = tokenizers::load_sentencepiece_model(out->resources.require_file("tokenizer"));
    if (out->sentencepiece.size() != 50000) {
        throw std::runtime_error("OWSM-CTC v4 tokenizer does not match the published 50,000-piece vocabulary");
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

std::unique_ptr<OWSMCTCV4Weights> load_owsm_ctc_weights(
    const OWSMCTCV4Assets & assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType type) {
    const auto & config = assets.config;
    auto out = std::make_unique<OWSMCTCV4Weights>();
    out->store = std::make_unique<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "owsm_ctc.weights", 4 * 1024 * 1024);
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
    const auto attention = [&](const std::string & name, bool packed_qkv, bool packed_kv, int64_t width) {
        modules::AttentionWeights weights;
        if (packed_qkv) {
            const auto packed = modules::PackedLinearWeightsBuilder({width, {
                {name + ".linear_q.weight", name + ".linear_q.bias", width},
                {name + ".linear_k.weight", name + ".linear_k.bias", width},
                {name + ".linear_v.weight", name + ".linear_v.bias", width}}, true}).build(store, source, type);
            weights.qkv_weight = packed.weight;
            weights.qkv_bias = packed.bias;
        } else {
            const auto query = linear(name + ".linear_q", width, width);
            weights.q_weight = query.weight;
            weights.q_bias = query.bias;
            if (packed_kv) {
                const auto packed = modules::PackedLinearWeightsBuilder({width, {
                    {name + ".linear_k.weight", name + ".linear_k.bias", width},
                    {name + ".linear_v.weight", name + ".linear_v.bias", width}}, true}).build(store, source, type);
                weights.qkv_weight = packed.weight;
                weights.qkv_bias = packed.bias;
            }
        }
        const auto output = linear(name + ".linear_out", width, width);
        weights.out_weight = output.weight;
        weights.out_bias = output.bias;
        return weights;
    };

    if (execution.backend_type() == core::BackendType::Cpu) {
        // Zero-extend 3x3 to 4x4 so the single-channel convolution can use CPU GEMM.
        const auto original = source.require_f32("encoder.embed.conv.0.weight", {d, 1, 3, 3});
        std::vector<float> padded(static_cast<size_t>(d * 16), 0.0F);
        for (int64_t channel = 0; channel < d; ++channel) {
            for (int64_t row = 0; row < 3; ++row) {
                std::copy_n(original.data() + channel * 9 + row * 3, 3,
                    padded.data() + channel * 16 + row * 4);
            }
        }
        out->subsampling.conv0.weight = store.make_f32(
            core::TensorShape::from_dims({d, 1, 4, 4}), std::move(padded));
        out->subsampling.conv0.bias = store.load_f32_tensor(source, "encoder.embed.conv.0.bias", {d});
    } else {
        out->subsampling.conv0 = modules::binding::conv2d_from_source(
            store, source, "encoder.embed.conv.0", assets::TensorStorageType::F32, d, 1, 3, 3, true);
    }
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
        layer.attention = attention(prefix + ".attn", true, false, d);
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
        if (std::find(config.cross_attention_layers.begin(), config.cross_attention_layers.end(), index) !=
            config.cross_attention_layers.end()) {
            layer.cross_attention_norm = norm(prefix + ".norm_cross_attn", d);
            layer.cross_attention = attention(prefix + ".cross_attn", false, true, d);
        }
        out->encoder.push_back(std::move(layer));
    }
    out->encoder_norm = norm("encoder.after_norm", d);

    const auto p = config.prompt_hidden_size;
    const auto pf = config.prompt_intermediate_size;
    out->embedding = store.load_tensor(source, "embed.weight", type, {config.vocabulary_size, p});
    out->prefix_projection = linear("embed_proj", d, p);
    out->prompt_projection = linear("prompt_proj", d, p);
    if (execution.backend_type() == core::BackendType::Cpu &&
        assets::resolve_tensor_storage_type(source, "encoder.conditioning_layer.weight", type) ==
            assets::TensorStorageType::F32) {
        // The CPU GEMM kernel requires a SIMD-aligned reduction width.
        const auto width = (config.vocabulary_size + 15) / 16 * 16;
        const auto original = source.require_f32("encoder.conditioning_layer.weight",
            {d, config.vocabulary_size});
        std::vector<float> padded(static_cast<size_t>(d * width), 0.0F);
        for (int64_t row = 0; row < d; ++row) {
            std::copy_n(original.data() + row * config.vocabulary_size,
                config.vocabulary_size, padded.data() + row * width);
        }
        out->ctc_conditioning.weight = store.make_f32(
            core::TensorShape::from_dims({d, width}), std::move(padded));
        out->ctc_conditioning.bias = store.load_f32_tensor(source,
            "encoder.conditioning_layer.bias", {d});
    } else {
        out->ctc_conditioning = linear("encoder.conditioning_layer", d, config.vocabulary_size);
    }
    if (execution.backend_type() == core::BackendType::Cpu) {
        const auto rows = (config.vocabulary_size + 15) / 16 * 16;
        auto head = source.require_tensor("ctc.ctc_lo.weight", type, {config.vocabulary_size, d});
        const auto row_bytes = head.bytes.size() / static_cast<size_t>(config.vocabulary_size);
        head.bytes.resize(static_cast<size_t>(rows) * row_bytes, std::byte{0});
        out->output.weight = store.make_tensor(core::TensorShape::from_dims({rows, d}),
            head.type, head.bytes.data(), head.bytes.size());
        auto bias = source.require_f32("ctc.ctc_lo.bias", {config.vocabulary_size});
        bias.resize(static_cast<size_t>(rows), 0.0F);
        out->output.bias = store.make_f32(core::TensorShape::from_dims({rows}), std::move(bias));
    } else {
        out->output = linear("ctc.ctc_lo", config.vocabulary_size, d);
    }
    out->prompt_norm = norm("prompt_encoder.after_norm", p);
    for (int64_t index = 0; index < config.prompt_layers; ++index) {
        const auto prefix = "prompt_encoder.encoders." + std::to_string(index);
        modules::TransformerEncoderBlockWeights layer;
        layer.norm1 = norm(prefix + ".norm1", p);
        layer.self_attention = attention(prefix + ".self_attn", true, false, p);
        layer.norm2 = norm(prefix + ".norm2", p);
        const auto first = linear(prefix + ".feed_forward.w_1", pf, p);
        const auto second = linear(prefix + ".feed_forward.w_2", p, pf);
        layer.feed_forward = {first.weight, first.bias, second.weight, second.bias};
        out->prompt_encoder.push_back(std::move(layer));
    }
    out->prompt_scale = store.make_f32(core::TensorShape::from_dims({p}),
        std::vector<float>(static_cast<size_t>(p), std::sqrt(static_cast<float>(p))));

    out->half_scale = store.make_f32(core::TensorShape::from_dims({d}), std::vector<float>(static_cast<size_t>(d), 0.5F));
    out->embedding_scale = store.make_f32(
        core::TensorShape::from_dims({d}), std::vector<float>(static_cast<size_t>(d), std::sqrt(static_cast<float>(d))));
    store.upload();
    return out;
}

}  // namespace engine::models::owsm_ctc
