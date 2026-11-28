#include "engine/models/lfm_d1_omni/assets.h"

#include "engine/framework/assets/gguf_metadata.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::lfm_d1_omni {
namespace {
namespace fs = std::filesystem;
using assets::TensorStorageType;
using core::TensorShape;
using modules::LfmDecoderLayerKind;

// LiquidAI/d1-Omni-2B calibration.json, upstream revision
// 81222dae8bf8bdd7f903643b5b2a8803bf2e3176. These are checkpoint contract data,
// not defaults for another model. The loader identifies the d1-Omni decision contract.
constexpr const char * kContractTables = R"json({
  "estimate": [
    [0.030000000000000002, 0.30952380952380953],
    [0.09, 0.40575396825396826],
    [0.15000000000000002, 0.4513888888888889],
    [0.21000000000000002, 0.49007936507936506],
    [0.27, 0.5168650793650794],
    [0.32999999999999996, 0.5525793650793651],
    [0.39, 0.5952380952380952],
    [0.44999999999999996, 0.6398809523809523],
    [0.51, 0.6646825396825397],
    [0.57, 0.6805555555555556],
    [0.63, 0.7093253968253969],
    [0.69, 0.7281746031746031],
    [0.75, 0.7668650793650794],
    [0.81, 0.7996031746031746],
    [0.87, 0.8382936507936508],
    [0.93, 0.8938492063492064],
    [0.9799999999999999, 0.9508928571428571]
  ],
  "multiple": [
    [0.0005885864014364985, 0.0],
    [0.007570615044117842, 0.0024875621890547263],
    [0.020294553199804555, 0.020100502512562814],
    [0.026795323031471443, 0.025],
    [0.05788014834070439, 0.04791154791154791],
    [0.12111951862189904, 0.11707317073170732],
    [0.17282815321178932, 0.1485148514851485],
    [0.2522862852730819, 0.208955223880597],
    [0.3889613887792048, 0.29],
    [0.6457394381624046, 0.6080402010050251],
    [0.9463102341663232, 0.8918918918918919]
  ]
})json";

LfmD1OmniBackboneSpec backbone_spec(const assets::GgufMetadata & meta, const assets::TensorSource & source) {
    if (meta.require_string("general.architecture") != "lfm3" || meta.require_string("general.type") != "model" ||
        meta.require_string("lfm3.decision.type") != "lfm3-d1-omni") {
        throw std::runtime_error("d1: expected an official lfm3-d1-omni backbone GGUF");
    }
    LfmD1OmniBackboneSpec out;
    auto & d = out.decoder;
    const int64_t layers = meta.require_int("lfm3.block_count");
    d.hidden_size = meta.require_int("lfm3.embedding_length");
    d.intermediate_size = meta.require_int("lfm3.feed_forward_length");
    d.num_attention_heads = meta.require_int("lfm3.attention.head_count");
    d.head_dim = meta.require_int("lfm3.attention.key_length");
    d.conv_kernel_size = meta.require_int("lfm3.shortconv.l_cache");
    d.norm_eps = static_cast<float>(meta.require_number("lfm3.attention.layer_norm_rms_epsilon"));
    d.full_rope.theta = static_cast<float>(meta.require_number("lfm3.rope.freq_base"));
    d.sliding_rope.theta = static_cast<float>(meta.require_number("lfm3.rope.freq_base_swa"));
    const auto kv = meta.require_int_array("lfm3.attention.head_count_kv");
    const auto sliding = meta.require_int_array("lfm3.attention.sliding_window_pattern", true);
    if (layers <= 0 || static_cast<int64_t>(kv.size()) != layers || kv.size() != sliding.size()) {
        throw std::runtime_error("d1: backbone layer metadata lengths disagree");
    }
    for (size_t i = 0; i < kv.size(); ++i) {
        if (kv[i] < 0 || (kv[i] == 0 && sliding[i])) { throw std::runtime_error("d1: invalid backbone layer kind"); }
        if (kv[i] > 0) {
            if (d.num_key_value_heads != 0 && d.num_key_value_heads != kv[i]) {
                throw std::runtime_error("d1: decoder requires a common KV head count");
            }
            d.num_key_value_heads = kv[i];
        }
        d.layers.push_back(kv[i] == 0 ? LfmDecoderLayerKind::ShortConv
            : sliding[i] ? LfmDecoderLayerKind::SlidingAttention : LfmDecoderLayerKind::FullAttention);
    }
    if (d.hidden_size <= 0 || d.intermediate_size <= 0 || d.head_dim <= 0 || d.head_dim % 2 != 0 ||
        d.num_attention_heads <= 0 || d.num_key_value_heads <= 0 || d.num_attention_heads % d.num_key_value_heads != 0 ||
        d.hidden_size != d.num_attention_heads * d.head_dim || d.conv_kernel_size <= 0 || d.norm_eps <= 0 ||
        meta.require_int("lfm3.attention.value_length") != d.head_dim || meta.require_int("lfm3.rope.dimension_count") != d.head_dim) {
        throw std::runtime_error("d1: unsupported backbone dimensions");
    }
    out.context_length = meta.require_int("lfm3.context_length");
    out.sliding_window = meta.require_int("lfm3.attention.sliding_window");
    if (out.context_length <= 0 || out.sliding_window <= 0 || meta.require_bool("lfm3.attention.causal")) {
        throw std::runtime_error("d1: expected noncausal full attention and a positive context/window");
    }
    const auto embedding_metadata = source.require_metadata("token_embd.weight");
    const auto & embedding = embedding_metadata.shape;
    if (embedding.size() != 2 || embedding[0] <= 0 || embedding[1] != d.hidden_size || source.has_tensor("output.weight")) {
        throw std::runtime_error("d1: expected the tied token embedding/output head");
    }
    out.vocab_size = embedding[0];
    // GGUF encodes partial rotation as 1 / 1e30 frequency factors. Convert only this known recipe.
    const auto factors = source.require_f32("rope_freqs.weight", {d.head_dim / 2});
    int64_t active = 0;
    bool tail = false;
    for (const float factor : factors) {
        if (!tail && factor == 1.0f) { ++active; }
        else if (factor == 1e30f) { tail = true; }
        else { throw std::runtime_error("d1: unsupported RoPE frequency-factor schedule"); }
    }
    d.full_rope.rotary_dims = 2 * active;
    return out;
}

LfmD1OmniAudioSpec audio_spec(const assets::GgufMetadata & meta, const assets::TensorSource & source, int64_t output_size) {
    if (meta.require_string("general.architecture") != "clip" || meta.require_string("general.type") != "mmproj" ||
        !meta.require_bool("clip.has_audio_encoder") || meta.require_string("clip.audio.projector_type") != "lfm3a") {
        throw std::runtime_error("d1: expected an lfm3a audio projector GGUF");
    }
    LfmD1OmniAudioSpec out;
    auto & d = out.decoder;
    d.hidden_size = meta.require_int("clip.audio.embedding_length");
    d.intermediate_size = meta.require_int("clip.audio.feed_forward_length");
    d.num_attention_heads = meta.require_int("clip.audio.attention.head_count");
    d.num_key_value_heads = meta.require_int("clip.audio.attention.head_count_kv");
    d.norm_eps = static_cast<float>(meta.require_number("clip.audio.attention.layer_norm_epsilon"));
    d.attention_gate = false;
    d.sandwich_norm = false;
    const int64_t layers = meta.require_int("clip.audio.block_count");
    const int64_t mels = meta.require_int("clip.audio.num_mel_bins");
    if (layers <= 0 || d.hidden_size <= 0 || d.intermediate_size <= 0 || d.num_attention_heads <= 0 ||
        d.num_key_value_heads <= 0 || d.hidden_size % d.num_attention_heads != 0 ||
        d.num_attention_heads % d.num_key_value_heads != 0 || d.norm_eps <= 0 || mels <= 0 || mels % 8 != 0) {
        throw std::runtime_error("d1: unsupported audio encoder dimensions");
    }
    d.head_dim = d.hidden_size / d.num_attention_heads;
    for (int64_t i = 0; i < layers; ++i) {
        const std::string p = "a.blk." + std::to_string(i) + ".";
        const bool conv = source.has_tensor(p + "conv_dw.weight");
        if (conv == source.has_tensor(p + "attn_q.weight")) { throw std::runtime_error("d1: ambiguous audio encoder layer kind"); }
        d.layers.push_back(conv ? LfmDecoderLayerKind::ShortConv : LfmDecoderLayerKind::FullAttention);
        if (conv) {
            const auto shape = source.require_metadata(p + "conv_dw.weight").shape;
            if (shape.size() != 2 || shape[0] != d.hidden_size || shape[1] <= 0 ||
                (d.conv_kernel_size != 0 && d.conv_kernel_size != shape[1])) {
                throw std::runtime_error("d1: inconsistent audio convolution kernel shape");
            }
            d.conv_kernel_size = shape[1];
        }
    }
    out.output_size = meta.require_int("clip.audio.projection_dim");
    out.sliding_window = meta.require_int("clip.audio.window_size");
    const auto conv = source.require_metadata("a.conv1d.0.weight").shape;
    if (conv.size() != 4 || conv[0] <= 0 || conv[1] != 1 || conv[2] != 3 || conv[3] != 3 ||
        out.output_size != output_size || out.sliding_window <= 0) {
        throw std::runtime_error("d1: audio downsampler/projector dimensions disagree");
    }
    out.conv_channels = conv[0];
    auto & f = out.frontend;
    f.sample_rate = 16000;
    f.n_fft = f.win_length = 400;
    f.hop_length = 160;
    f.n_mels = mels;
    out.mel_bins = mels;
    f.mel_fmax = 8000;
    f.window = audio::MelHannWindow::Periodic;
    f.filterbank_normalization = audio::MelFilterbankNormalization::Slaney;
    f.filterbank_projection = audio::MelFilterbankProjection::DenseF32;
    f.log_precision = audio::MelLogPrecision::F32;
    f.layout = audio::MelOutputLayout::FeatureMajor;
    f.waveform_padding = audio::MelWaveformPadding::None;
    f.stft_center = false;
    f.spectrum_mode = audio::MelSpectrumMode::PowerBeforeProjection;
    f.value_transform = audio::MelValueTransform::Log10;
    f.log_floor = 1e-12;
    f.require_mono = true;
    return out;
}

modules::LfmDecoderWeights bind_decoder(core::BackendWeightStore & store, const assets::TensorSource & source,
                                        const modules::LfmDecoderConfig & config, bool audio_encoder) {
    modules::LfmDecoderWeights out;
    const int64_t d = config.hidden_size, ff = config.intermediate_size, hd = config.head_dim;
    for (size_t i = 0; i < config.layers.size(); ++i) {
        const std::string p = std::string(audio_encoder ? "a.blk." : "blk.") + std::to_string(i) + ".";
        modules::LfmDecoderLayerWeights w;
        w.operator_norm = {store.load_f32_tensor(source, p + (audio_encoder ? "ln1.weight" : "attn_norm.weight"), {d}), std::nullopt};
        w.ffn_norm = {store.load_f32_tensor(source, p + (audio_encoder ? "ln2.weight" : "ffn_norm.weight"), {d}), std::nullopt};
        if (config.sandwich_norm) {
            w.post_operator_norm = modules::NormWeights{store.load_f32_tensor(source, p + "post_attention_norm.weight", {d}), std::nullopt};
            w.post_ffn_norm = modules::NormWeights{store.load_f32_tensor(source, p + "post_ffw_norm.weight", {d}), std::nullopt};
        }
        const auto linear = [&](const std::string & name, int64_t rows, int64_t cols) {
            return modules::LinearWeights{store.load_tensor(source, p + name + ".weight", TensorStorageType::Native, {rows, cols}), std::nullopt};
        };
        w.feed_forward = {linear("ffn_gate", ff, d), linear("ffn_up", ff, d), linear("ffn_down", d, ff)};
        if (config.layers[i] == LfmDecoderLayerKind::ShortConv) {
            w.conv.in_proj = linear(audio_encoder ? "conv_pw1" : "shortconv.in_proj", 3 * d, d);
            w.conv.out_proj = linear(audio_encoder ? "conv_pw2" : "shortconv.out_proj", d, d);
            const int64_t k = config.conv_kernel_size;
            w.conv.kernel = store.load_tensor_as_shape(source,
                p + (audio_encoder ? "conv_dw.weight" : "shortconv.conv.weight"),
                TensorStorageType::F32, {d, k}, TensorShape::from_dims({d, 1, k}));
        } else {
            auto & a = w.attention;
            a.q_proj = linear("attn_q", config.num_attention_heads * hd, d);
            a.k_proj = linear("attn_k", config.num_key_value_heads * hd, d);
            a.v_proj = linear("attn_v", config.num_key_value_heads * hd, d);
            a.out_proj = linear(audio_encoder ? "attn_out" : "attn_output", d, config.num_attention_heads * hd);
            a.q_norm = {store.load_f32_tensor(source, p + "attn_q_norm.weight", {hd}), std::nullopt};
            a.k_norm = {store.load_f32_tensor(source, p + "attn_k_norm.weight", {hd}), std::nullopt};
            if (config.attention_gate) { a.gate_proj = linear("attn_gate", config.num_attention_heads, d); }
        }
        out.layers.push_back(std::move(w));
    }
    out.final_norm = {store.load_f32_tensor(source, audio_encoder ? "a.post_ln.weight" : "output_norm.weight", {d}), std::nullopt};
    if (!audio_encoder && source.has_tensor("rope_freqs.weight")) {
        out.full_freq_factors = store.load_f32_tensor(source, "rope_freqs.weight", {hd / 2});
    }
    return out;
}

}  // namespace

std::shared_ptr<const LfmD1OmniAssets> load_lfm_d1_omni_assets(const fs::path & model_path) {
    const auto main = fs::is_directory(model_path) ? model_path / "d1-Omni-2B-Q8_0.gguf" : model_path;
    if (main.extension() != ".gguf" || !fs::is_regular_file(main)) {
        throw std::runtime_error("d1: expected the main GGUF file: " + main.string());
    }
    const auto path = fs::weakly_canonical(main);
    assets::ResourceBundle resources(path.parent_path(), {true});
    resources.add_tensor_source("model", path);
    resources.add_tensor_source("mmproj", path.parent_path() / ("mmproj-" + path.filename().string()));
    return load_lfm_d1_omni_assets(std::move(resources));
}

std::shared_ptr<const LfmD1OmniAssets> load_lfm_d1_omni_assets(assets::ResourceBundle resources) {
    const auto main_path = resources.require_file("model");
    const auto projector_path = resources.require_file("mmproj");
    const auto filename = main_path.filename().string();
    if (main_path.extension() != ".gguf" || projector_path.filename() != "mmproj-" + filename) {
        throw std::runtime_error("d1: main/mmproj resources must be the matching published GGUF pair");
    }
    auto out = std::make_shared<LfmD1OmniAssets>();
    out->resources = std::move(resources);
    out->model = out->resources.open_tensor_source("model", {true});
    const assets::GgufMetadata meta(main_path);
    out->backbone = backbone_spec(meta, *out->model);
    auto & decision = out->decision;
    decision.noul_temperature = meta.require_number("lfm3.decision.temperature.noul");
    decision.choice_temperature = meta.require_number("lfm3.decision.temperature.choice");
    decision.score_temperature = meta.require_number("lfm3.decision.temperature.score");
    if (decision.noul_temperature <= 0 || decision.choice_temperature <= 0 || decision.score_temperature <= 0) {
        throw std::runtime_error("d1: decision temperatures must be positive");
    }
    const auto contract_tables = io::json::parse(kContractTables);
    io::json::Value::Object temperatures;
    temperatures.emplace("noul", io::json::Value::make_number(decision.noul_temperature));
    for (const char * bucket : {"choice:small", "choice:mid", "choice:large", "choice:inventory"}) {
        temperatures.emplace(bucket, io::json::Value::make_number(decision.choice_temperature));
    }
    for (int levels = 2; levels <= 10; ++levels) {
        temperatures.emplace("score:" + std::to_string(levels), io::json::Value::make_number(decision.score_temperature));
    }
    out->calibration = io::json::Value::make_object({
        {"default", io::json::Value::make_number(decision.noul_temperature)},
        {"temperatures", io::json::Value::make_object(std::move(temperatures))},
        {"tables", contract_tables},
        {"checkpoint", io::json::Value::make_string("LiquidAI/d1-Omni-2B")},
        {"revision", io::json::Value::make_string("81222dae8bf8bdd7f903643b5b2a8803bf2e3176")},
    });
    if (meta.require_string("tokenizer.ggml.model") != "gpt2" || meta.require_string("tokenizer.ggml.pre") != "lfm2") {
        throw std::runtime_error("d1: unsupported tokenizer model/pre-tokenizer");
    }
    tokenizers::LlamaBpeGgufVocabulary vocabulary;
    vocabulary.tokens = meta.require_string_array("tokenizer.ggml.tokens");
    vocabulary.merges = meta.require_string_array("tokenizer.ggml.merges");
    const auto types = meta.require_int_array("tokenizer.ggml.token_type");
    vocabulary.token_types.assign(types.begin(), types.end());
    if (static_cast<int64_t>(vocabulary.tokens.size()) != out->backbone.vocab_size) {
        throw std::runtime_error("d1: tokenizer and embedding vocabulary sizes disagree");
    }
    // GGUF's lfm2 preset implies ignore_merges=true unless explicitly stored.
    vocabulary.ignore_merges = !meta.has("tokenizer.ggml.ignore_merges") || meta.require_bool("tokenizer.ggml.ignore_merges");
    tokenizers::LlamaBpeTokenizerSpec tokenizer;
    tokenizer.pre_type = tokenizers::LlamaBpePreTokenizer::Llama3;
    tokenizer.gguf_vocabulary = std::move(vocabulary);
    out->tokenizer = std::make_shared<tokenizers::LlamaBpeTokenizer>(std::move(tokenizer));
    const auto audio_id = out->tokenizer->find_token_id("<|audio|>");
    if (!audio_id) { throw std::runtime_error("d1: tokenizer has no audio placeholder"); }
    out->audio_token_id = *audio_id;
    out->mmproj = out->resources.open_tensor_source("mmproj", {true});
    out->audio = audio_spec(assets::GgufMetadata(projector_path), *out->mmproj, out->backbone.decoder.hidden_size);
    return out;
}

LfmD1OmniModelWeights load_model_weights(const LfmD1OmniAssets & assets, core::BackendWeightStore & store) {
    LfmD1OmniModelWeights out;
    out.backbone = bind_decoder(store, *assets.model, assets.backbone.decoder, false);
    out.token_embedding = store.load_tensor(*assets.model, "token_embd.weight", TensorStorageType::Native,
                                           {assets.backbone.vocab_size, assets.backbone.decoder.hidden_size});
    out.output_projection = {out.token_embedding, std::nullopt};
    return out;
}

modules::Lfm3AudioEncoderWeights load_audio_weights(const LfmD1OmniAssets & assets, core::BackendWeightStore & store) {
    const auto & config = assets.audio;
    const auto & source = *assets.mmproj;
    const int64_t c = config.conv_channels, d = config.decoder.hidden_size, output = config.output_size;
    const auto conv = [&](int index, int64_t in, int64_t kernel) {
        const std::string p = "a.conv1d." + std::to_string(index);
        return modules::Conv2dWeights{
            store.load_f32_tensor(source, p + ".weight", {c, in, kernel, kernel}),
            store.make_f32(TensorShape::from_dims({c}), source.require_f32(p + ".bias", {c, 1, 1}))};
    };
    modules::Lfm3AudioEncoderWeights out;
    out.conv_in = conv(0, 1, 3);
    out.conv_mid = conv(2, 1, 3);
    out.lin_mid = conv(3, c, 1);
    out.conv_end = conv(5, 1, 3);
    out.lin_end = conv(6, c, 1);
    out.lin_out = {store.load_tensor(source, "a.pre_encode.out.weight", TensorStorageType::Native, {d, c * (config.mel_bins / 8)}),
                   store.load_f32_tensor(source, "a.pre_encode.out.bias", {d})};
    out.decoder = bind_decoder(store, source, config.decoder, true);
    out.projector_norm = {store.load_f32_tensor(source, "mm.a.mlp.0.weight", {d}), store.load_f32_tensor(source, "mm.a.mlp.0.bias", {d})};
    out.projector_1 = {store.load_tensor(source, "mm.a.mlp.1.weight", TensorStorageType::Native, {output, d}),
                       store.load_f32_tensor(source, "mm.a.mlp.1.bias", {output})};
    out.projector_2 = {store.load_tensor(source, "mm.a.mlp.3.weight", TensorStorageType::Native, {output, output}),
                       store.load_f32_tensor(source, "mm.a.mlp.3.bias", {output})};
    return out;
}

}  // namespace engine::models::lfm_d1_omni
