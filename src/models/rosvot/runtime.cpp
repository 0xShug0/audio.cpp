#include "engine/models/rosvot/runtime.h"

#include "engine/framework/audio/mel_spectrogram_frontend.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/conformer_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/pitch_extractors/rmvpe_pitch_extractor.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace engine::models::rosvot {
namespace detail {
using namespace modules;
using core::TensorValue;
using core::TensorShape;
using core::ModuleBuildContext;
using Clock = std::chrono::steady_clock;
constexpr int kRate = 24000, kHop = 128;

// The convolutional parts use B,C,T; layer normalization and attention use B,T,C.
TensorValue transpose_sequence(ModuleBuildContext & ctx, const TensorValue & x) {
    return TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
}

TensorValue nonpadding(ModuleBuildContext & ctx, const TensorValue & btc) {
    auto * mask = ggml_step(ctx.ggml, ggml_sum_rows(ctx.ggml, ggml_abs(ctx.ggml, btc.tensor)));
    return core::wrap_tensor(mask, TensorShape::from_dims({1, btc.shape.dims[1], 1}));
}

struct Network {
    EncoderConfig config;
    core::BackendWeightStore store;
    std::unordered_map<std::string, TensorValue> weights;
    std::unordered_map<std::string, ChannelAffineWeights> batch_norms;

    Network(const assets::TensorSource & source, EncoderConfig c, const core::ExecutionContext & execution,
        const std::string & label)
        : config(c), store(execution.backend(), execution.config().type, label, 4ull << 20) {
        for (const auto & tensor : source.tensors()) {
            // These inference-only parameters are represented by folded affine weights.
            if (tensor.name.find(".conv_module.norm.") != std::string::npos) continue;
            weights.emplace(tensor.name, binding::tensor_from_named_source(store, source, tensor.name,
                assets::TensorStorageType::Native));
        }
        for (int64_t i = 0; i < config.conformer_layers; ++i) {
            const auto prefix = "net.net.mid.net.encoder_layers." + std::to_string(i) + ".conv_module.norm";
            batch_norms.emplace(prefix, binding::batch_norm_eval_from_source(store, source, prefix,
                config.hidden_size, 1e-5f));
        }
        store.upload();
    }

    TensorValue norm(ModuleBuildContext & ctx, const TensorValue & x, const std::string & prefix,
        float eps = 1e-5f) const {
        return LayerNormModule({x.shape.last_dim(), eps}).build(ctx, x,
            {weights.at(prefix + ".weight"), weights.at(prefix + ".bias")});
    }

    TensorValue conv(ModuleBuildContext & ctx, const TensorValue & x, const std::string & prefix,
        bool transposed = false) const {
        const auto & w = weights.at(prefix + ".weight");
        // A singleton time axis can be logically contiguous after transpose
        // while retaining a channel stride; im2col requires packed samples.
        auto input = x;
        if (x.tensor->nb[0] != ggml_type_size(x.tensor->type)) {
            input = core::wrap_tensor(ggml_cont(ctx.ggml, x.tensor), x.shape);
        }
        const int64_t kernel = w.shape.dims[2];
        if (transposed) {
            // PyTorch output_padding=1: compute padding=0 then crop one sample
            // on the left and none on the right, giving exactly 2*T frames.
            auto y = ConvTranspose1dModule({w.shape.dims[0], w.shape.dims[1], kernel, 2, 0}).build(
                ctx, input, {w, weights.at(prefix + ".bias")});
            return SliceModule({2, 1, x.shape.dims[2] * 2}).build(ctx, y);
        }
        return Conv1dModule({w.shape.dims[1], w.shape.dims[0], kernel, 1, static_cast<int>(kernel / 2)}).build(
            ctx, input, {w, weights.at(prefix + ".bias")});
    }

    TensorValue linear(ModuleBuildContext & ctx, const TensorValue & x, const std::string & prefix) const {
        auto w = weights.at(prefix + ".weight");
        if (w.shape.rank == 3) w = core::reshape_tensor(ctx, w, TensorShape::from_dims({w.shape.dims[0], w.shape.dims[1]}));
        return LinearModule({x.shape.last_dim(), w.shape.dims[0], true}).build(ctx, x,
            {w, weights.at(prefix + ".bias")});
    }

    TensorValue residual(ModuleBuildContext & ctx, TensorValue x, const std::string & prefix,
        int count, float eps = 1e-5f, const std::optional<TensorValue> & keep_mask = std::nullopt) const {
        const auto mask_bct = keep_mask ? *keep_mask : transpose_sequence(ctx,
            nonpadding(ctx, core::ensure_backend_addressable_layout(ctx, transpose_sequence(ctx, x))));
        for (int i = 0; i < count; ++i) {
            const auto p = prefix + ".blocks." + std::to_string(i);
            auto y = transpose_sequence(ctx, norm(ctx, transpose_sequence(ctx, x), p + ".0", eps));
            y = conv(ctx, y, p + ".1");
            y = ScaleBiasModule({1.0f / std::sqrt(3.0f), 0}).build(ctx, y);
            y = LeakyReluModule().build(ctx, y);
            y = conv(ctx, y, p + ".4");
            x = MulModule(true).build(ctx, AddModule().build(ctx, x, y), mask_bct);
        }
        return x;
    }

    TensorValue conv_blocks(ModuleBuildContext & ctx, TensorValue x, const std::string & prefix, int count) const {
        const auto mask = transpose_sequence(ctx, nonpadding(ctx,
            core::ensure_backend_addressable_layout(ctx, transpose_sequence(ctx, x))));
        x = residual(ctx, x, prefix + ".res_blocks.0", count, 1e-5f, mask);
        x = MulModule(true).build(ctx, x, mask);
        x = transpose_sequence(ctx, norm(ctx, transpose_sequence(ctx, x), prefix + ".last_norm"));
        x = MulModule(true).build(ctx, x, mask);
        return MulModule(true).build(ctx, conv(ctx, x, prefix + ".post_net1"), mask);
    }

    TensorValue conformer(ModuleBuildContext & ctx, TensorValue x, const TensorValue & position) const {
        const auto keep = nonpadding(ctx, core::ensure_backend_addressable_layout(ctx, x));
        auto key_keep = core::reshape_tensor(ctx, keep, TensorShape::from_dims({1, x.shape.dims[1]}));
        auto attention_mask = RepeatModule({TensorShape::from_dims({x.shape.dims[1], x.shape.dims[1]})}).build(
            ctx, ScaleBiasModule({1e9f, -1e9f}).build(ctx, key_keep));
        // ggml_scale requires packed outer strides even for singleton axes.
        if (x.tensor->nb[0] != ggml_type_size(x.tensor->type) ||
            x.tensor->nb[2] != x.tensor->nb[1] * x.tensor->ne[1] ||
            x.tensor->nb[3] != x.tensor->nb[2] * x.tensor->ne[2]) {
            x = core::wrap_tensor(ggml_cont(ctx.ggml, x.tensor), x.shape);
        }
        x = ScaleBiasModule({std::sqrt(static_cast<float>(config.hidden_size)), 0}).build(ctx, x);
        for (int64_t i = 0; i < config.conformer_layers; ++i) {
            const auto p = "net.net.mid.net.encoder_layers." + std::to_string(i);
            RelativeConformerBlockWeights w;
            w.ffn1_norm = {weights.at(p + ".norm_ff_macaron.weight"), weights.at(p + ".norm_ff_macaron.bias")};
            w.ffn1_fc1 = {weights.at(p + ".feed_forward_macaron.w_1.weight"), weights.at(p + ".feed_forward_macaron.w_1.bias")};
            w.ffn1_fc2 = {weights.at(p + ".feed_forward_macaron.w_2.weight"), weights.at(p + ".feed_forward_macaron.w_2.bias")};
            w.norm1 = {weights.at(p + ".norm_mha.weight"), weights.at(p + ".norm_mha.bias")};
            const auto ap = p + ".self_attn";
            w.self_attention.attention = {weights.at(ap + ".linear_q.weight"), weights.at(ap + ".linear_q.bias"),
                weights.at(ap + ".linear_k.weight"), weights.at(ap + ".linear_k.bias"),
                weights.at(ap + ".linear_v.weight"), weights.at(ap + ".linear_v.bias"), {}, {},
                weights.at(ap + ".linear_out.weight"), weights.at(ap + ".linear_out.bias")};
            w.self_attention.pos_weight = weights.at(ap + ".linear_pos.weight");
            w.self_attention.pos_bias_u = weights.at(ap + ".pos_bias_u");
            w.self_attention.pos_bias_v = weights.at(ap + ".pos_bias_v");
            w.conv.norm = {weights.at(p + ".norm_conv.weight"), weights.at(p + ".norm_conv.bias")};
            w.conv.pointwise_in = {core::reshape_tensor(ctx, weights.at(p + ".conv_module.pointwise_conv1.weight"),
                TensorShape::from_dims({2 * config.hidden_size, config.hidden_size})),
                weights.at(p + ".conv_module.pointwise_conv1.bias")};
            w.conv.depthwise = {weights.at(p + ".conv_module.depthwise_conv.weight"), weights.at(p + ".conv_module.depthwise_conv.bias")};
            w.conv.depthwise_norm = batch_norms.at(p + ".conv_module.norm");
            w.conv.pointwise_out = {core::reshape_tensor(ctx, weights.at(p + ".conv_module.pointwise_conv2.weight"),
                TensorShape::from_dims({config.hidden_size, config.hidden_size})),
                weights.at(p + ".conv_module.pointwise_conv2.bias")};
            w.norm2 = {weights.at(p + ".norm_ff.weight"), weights.at(p + ".norm_ff.bias")};
            w.ffn2_fc1 = {weights.at(p + ".feed_forward.w_1.weight"), weights.at(p + ".feed_forward.w_1.bias")};
            w.ffn2_fc2 = {weights.at(p + ".feed_forward.w_2.weight"), weights.at(p + ".feed_forward.w_2.bias")};
            w.final_norm = {weights.at(p + ".norm_final.weight"), weights.at(p + ".norm_final.bias")};
            // The checkpoint stores FFN projections as width-one convolutions.
            for (auto * projection : {&w.ffn1_fc1, &w.ffn1_fc2, &w.ffn2_fc1, &w.ffn2_fc2}) {
                projection->weight = core::reshape_tensor(ctx, projection->weight,
                    TensorShape::from_dims({projection->weight.shape.dims[0], projection->weight.shape.dims[1]}));
            }
            ConformerBlockConfig block{config.hidden_size, 4, w.ffn1_fc1.weight.shape.dims[0], config.conformer_kernel};
            block.contiguous_glu_gate = true;
            block.ffn_activation = FeedForwardActivation::Relu;
            block.legacy_position_layout = true;
            x = RelativeConformerBlockModule(block).build(ctx, x, position, w, attention_mask);
        }
        return MulModule(true).build(ctx, norm(ctx, x, "net.net.mid.net.layer_norm"), keep);
    }

    TensorValue encode(ModuleBuildContext & ctx, TensorValue mel, const TensorValue & pitch,
        const TensorValue & uv, const std::optional<TensorValue> & word_boundary,
        const TensorValue & position) const {
        auto x = conv_blocks(ctx, conv(ctx, transpose_sequence(ctx, mel), "mel_proj"), "mel_encoder", 2);
        x = transpose_sequence(ctx, x);
        const auto indices_shape = TensorShape::from_dims({1, mel.shape.dims[1]});
        const auto & pitch_weight = weights.at("pitch_embed.weight");
        const auto & uv_weight = weights.at("uv_embed.weight");
        auto pe = PitchEmbedModule({pitch_weight.shape.dims[0], config.hidden_size}).build(
            ctx, core::reshape_tensor(ctx, pitch, indices_shape), pitch_weight);
        auto ue = EmbeddingModule({uv_weight.shape.dims[0], config.hidden_size}).build(
            ctx, core::reshape_tensor(ctx, uv, indices_shape), uv_weight);
        x = AddModule().build(ctx, AddModule().build(ctx, x, pe), ue);
        if (word_boundary) {
            const auto & word_weight = weights.at("word_bd_embed.weight");
            auto we = EmbeddingModule({word_weight.shape.dims[0], config.hidden_size}).build(
                ctx, core::reshape_tensor(ctx, *word_boundary, indices_shape), word_weight);
            x = AddModule().build(ctx, x, we);
        }
        x = conv_blocks(ctx, transpose_sequence(ctx, x), "cond_encoder", 1);
        std::vector<TensorValue> skips;
        for (int64_t i = 0; i < config.down_layers; ++i) {
            const auto p = "net.net.down.layers." + std::to_string(i);
            x = residual(ctx, x, p + ".0", 1);
            x = conv(ctx, x, p + ".1");
            x = residual(ctx, x, p + ".2", 1);
            skips.push_back(x);
            auto * pooled = ggml_pool_2d(ctx.ggml, core::ensure_backend_addressable_layout(ctx, x).tensor,
                GGML_OP_POOL_AVG, 2, 1, 2, 1, 0, 0);
            x = core::wrap_tensor(pooled, TensorShape::from_dims({1, config.hidden_size, x.shape.dims[2] / 2}));
        }
        x = conv(ctx, transpose_sequence(ctx, norm(ctx, transpose_sequence(ctx, x), "net.net.down.last_norm", 1e-6f)), "net.net.down.post_net");
        x = conv(ctx, x, "net.net.mid.pre");
        x = transpose_sequence(ctx, conformer(ctx, transpose_sequence(ctx, x), position));
        x = conv(ctx, x, "net.net.mid.post");
        for (int64_t i = 0; i < config.down_layers; ++i) {
            const std::string p = "net.net.up.";
            x = conv(ctx, x, p + "ups." + std::to_string(i) + ".0", true);
            x = transpose_sequence(ctx, norm(ctx, transpose_sequence(ctx, x), p + "ups." + std::to_string(i) + ".1", 1e-6f));
            x = LeakyReluModule().build(ctx, x);
            x = ConcatModule({1}).build(ctx, x, skips[static_cast<size_t>(config.down_layers - i - 1)]);
            x = conv(ctx, x, p + "layers." + std::to_string(i) + ".0");
            x = residual(ctx, x, p + "layers." + std::to_string(i) + ".1", 1);
        }
        x = conv(ctx, transpose_sequence(ctx, norm(ctx, transpose_sequence(ctx, x), "net.net.up.last_norm", 1e-6f)), "net.net.up.post_net");
        return transpose_sequence(ctx, x);
    }
};

struct FrameGraph {
    core::PreparedGraph owner;
    TensorValue mel, pitch, uv, word, position, feature, boundary, attention;
    int64_t frames;
    FrameGraph(const core::ExecutionContext & execution, const Network & net, int64_t t, bool words)
        : owner(execution, {words ? "rosvot.words" : "rosvot.notes", 32ull << 20, 24000, 1ull << 20}), frames(t) {
        auto & ctx = owner.build_context();
        auto input = [&](ggml_type type, TensorShape shape) {
            auto v = core::make_tensor(ctx, type, shape); ggml_set_input(v.tensor); return v;
        };
        mel = input(GGML_TYPE_F32, TensorShape::from_dims({1, t, net.config.mel_bins}));
        pitch = input(GGML_TYPE_I32, TensorShape::from_dims({t}));
        uv = input(GGML_TYPE_I32, TensorShape::from_dims({t}));
        if (!words) word = input(GGML_TYPE_I32, TensorShape::from_dims({t}));
        const int64_t down = t >> net.config.down_layers;
        position = core::make_tensor(owner.persistent_build_context(), GGML_TYPE_F32, TensorShape::from_dims({1, down, net.config.hidden_size}));
        auto encoded = net.encode(ctx, mel, pitch, uv, words ? std::nullopt : std::optional<TensorValue>(word), position);
        boundary = net.linear(ctx, encoded, words ? "word_bd_out" : "note_bd_out");
        boundary = ScaleBiasModule({1.0f / net.config.boundary_temperature, 0}).build(ctx, boundary);
        boundary = core::wrap_tensor(ggml_clamp(ctx.ggml, boundary.tensor, -16, 16), boundary.shape);
        ggml_build_forward_expand(owner.graph(), boundary.tensor);
        if (!words) {
            auto & pc = owner.persistent_build_context();
            feature = core::make_tensor(pc, GGML_TYPE_F32, encoded.shape);
            auto * copied = ggml_cpy(ctx.ggml, core::ensure_backend_addressable_layout(ctx, encoded).tensor, feature.tensor);
            ggml_build_forward_expand(owner.graph(), copied);
            attention = net.linear(ctx, encoded, "pitch_decoder.multihead_dot_attn");
            attention = SigmoidModule().build(ctx, attention);
            attention = ReduceMeanModule({2}).build(ctx, attention);
            ggml_build_forward_expand(owner.graph(), attention.tensor);
        }
        owner.allocate_persistent();
        owner.allocate();
        // Upstream initializes a reversed 5000-position table, then takes its
        // leading T rows; it does not regenerate a T-position table per request.
        std::vector<float> pe(static_cast<size_t>(down * net.config.hidden_size));
        for (int64_t r = 0; r < down; ++r) for (int64_t c = 0; c < net.config.hidden_size; c += 2) {
            const float inv = std::exp(static_cast<float>(c) * (-std::log(10000.0f) / static_cast<float>(net.config.hidden_size)));
            const float angle = static_cast<float>(std::max<int64_t>(5000, down) - 1 - r) * inv;
            pe[static_cast<size_t>(r * net.config.hidden_size + c)] = std::sin(angle);
            pe[static_cast<size_t>(r * net.config.hidden_size + c + 1)] = std::cos(angle);
        }
        core::write_tensor_f32(position, pe);

    }
    std::vector<float> run(const std::vector<float> & m, const std::vector<int32_t> & p,
        const std::vector<int32_t> & u, const std::vector<int32_t> & w = {}) {
        core::write_tensor_f32(mel, m);
        core::write_tensor_i32(pitch, p);
        core::write_tensor_i32(uv, u);
        if (word.valid()) core::write_tensor_i32(word, w);
        owner.compute();
        std::vector<float> out(static_cast<size_t>(frames));
        core::read_tensor_f32_into(boundary.tensor, out);
        return out;
    }
};

// Identical boundary regulation to upstream: choose each positive run's peak,
// merge close peaks, then anchor note boundaries to supplied/predicted words.
std::vector<int32_t> regulate_boundary(const std::vector<float> & logits, float threshold,
    int gap, int64_t real_frames, const std::vector<int32_t> & reference = {}, int reference_gap = 0) {
    std::vector<int32_t> out(logits.size(), 0);
    int64_t last = -1, start = -1;
    for (int64_t i = 0; i < static_cast<int64_t>(logits.size()); ++i) {
        if (1.0f / (1.0f + std::exp(-logits[static_cast<size_t>(i)])) > threshold) {
            if (start < 0) start = i;
        } else if (start >= 0 && start < i) {
            auto best = std::max_element(logits.begin() + start, logits.begin() + i) - logits.begin();
            if (best - last < gap && last > 0) {
                best = static_cast<int64_t>(std::nearbyint(static_cast<double>(best + last) / 2));
                out[static_cast<size_t>(last)] = 0;
            }
            out[static_cast<size_t>(best)] = 1; last = best; start = -1;
        }
    }
    if (!reference.empty() && reference_gap > 0) {
        for (int64_t j = 0; j < static_cast<int64_t>(out.size()); ++j) if (reference[static_cast<size_t>(j)]) {
            const int64_t lo = std::max<int64_t>(0, j - reference_gap);
            const int64_t hi = std::min<int64_t>(out.size(), j + reference_gap);
            const auto sum = std::accumulate(out.begin() + lo, out.begin() + hi, 0);
            if (sum == 0) out[static_cast<size_t>(j)] = 1;
            else if (sum == 1 && !out[static_cast<size_t>(j)]) {
                std::copy(reference.begin() + lo, reference.begin() + hi, out.begin() + lo);
            } else if (sum > 1) {
                for (int k = 1; k <= reference_gap; ++k) {
                    const auto a = static_cast<size_t>(std::max<int64_t>(0, j - k));
                    const auto b = static_cast<size_t>(std::min<int64_t>(out.size() - 1, j + k));
                    if (out[a] && !reference[a]) { out[a] = 0; break; }
                    if (out[b] && !reference[b]) { out[b] = 0; break; }
                }
                out[static_cast<size_t>(j)] = 1;
            }
        }
    }
    out.front() = 0;
    std::fill(out.begin() + std::max<int64_t>(0, real_frames - 1), out.end(), 0);
    return out;
}

struct PitchGraph {
    core::PreparedGraph owner;
    TensorValue pooling, denominator, logits;
    int64_t notes;
    PitchGraph(const core::ExecutionContext & execution, const Network & net, const TensorValue & features, int64_t n)
        : owner(execution, {"rosvot.pitch", 4ull << 20, 4000}), notes(n) {
        auto & ctx = owner.build_context();
        pooling = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, n, features.shape.dims[1]}));
        ggml_set_input(pooling.tensor);
        denominator = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, n, 1}));
        ggml_set_input(denominator.tensor);
        auto x = MatMulModule().build(ctx, pooling, features);
        x = core::wrap_tensor(ggml_div(ctx.ggml, x.tensor, denominator.tensor), x.shape);
        x = net.conv_blocks(ctx, transpose_sequence(ctx, x), "pitch_decoder.post", 1);
        logits = net.linear(ctx, transpose_sequence(ctx, x), "pitch_decoder.pitch_out");
        logits = ScaleBiasModule({1.0f / net.config.pitch_temperature, 0}).build(ctx, logits);
        ggml_set_output(logits.tensor);
        ggml_build_forward_expand(owner.graph(), logits.tensor);
        owner.allocate();
    }
};

} // namespace detail
using namespace detail;

std::shared_ptr<const RosvotAssets> load_rosvot_assets(const std::filesystem::path & path) {
    auto out = std::make_shared<RosvotAssets>();
    out->resources = model_spec::load_resource_bundle_for_family(path, "rosvot");
    const auto cfg = out->resources.parse_json("config");
    auto read = [](const io::json::Value & v, bool words) {
        EncoderConfig c;
        c.hidden_size = io::json::require_i64(v, "hidden_size");
        c.mel_bins = io::json::require_i64(v, "use_mel_bins");
        c.down_layers = io::json::require_i64(v, "down_layers");
        c.conformer_layers = io::json::require_i64(v, "bkb_layers");
        c.conformer_kernel = io::json::require_i64(v, "conformer_kernel");
        c.pitch_heads = io::json::require_i64(v, "pitch_attn_num_head");
        c.note_num = io::json::require_i64(v, "note_num");
        c.note_start = io::json::require_i64(v, "note_start");
        c.pitch_temperature = io::json::require_f32(v, "note_pitch_temperature");
        c.boundary_temperature = io::json::require_f32(v, words ? "word_bd_temperature" : "note_bd_temperature");
        c.boundary_gap = static_cast<int>(std::nearbyint(io::json::require_f32(v, words ? "word_bd_min_gap" : "note_bd_min_gap") * kRate / 1000.0 / kHop));
        c.reference_gap = static_cast<int>(std::nearbyint(io::json::require_f32(v, "note_bd_ref_min_gap") * kRate / 1000.0 / kHop));
        if (c.hidden_size <= 0 || c.hidden_size % 4 || c.mel_bins <= 0 || c.mel_bins > 80 ||
            c.down_layers < 1 || c.down_layers > 8 || c.conformer_layers < 1 || c.conformer_kernel % 2 != 1 ||
            c.boundary_temperature <= 0 || c.pitch_temperature <= 0) throw std::runtime_error("Invalid ROSVOT architecture configuration");
        return c;
    };
    out->notes = read(cfg.require("rosvot"), false); out->words = read(cfg.require("rwbd"), true);
    out->note_weights = out->resources.open_tensor_source("note_weights");
    out->word_weights = out->resources.open_tensor_source("word_weights");
    out->pitch_weights = out->resources.open_tensor_source("pitch_weights");
    return out;
}

struct RosvotRuntime::Impl {
    std::shared_ptr<const RosvotAssets> assets;
    const core::ExecutionContext & execution;
    Network notes, words;
    modules::RmvpePitchExtractorComponent pitch;
    std::shared_ptr<const audio::MelSpectrogramFrontend> frontend;
    std::unique_ptr<FrameGraph> word_graph, note_graph;
    std::unique_ptr<PitchGraph> pitch_graph;
    Impl(std::shared_ptr<const RosvotAssets> a, const core::ExecutionContext & e)
        : assets(std::move(a)), execution(e),
          notes(*assets->note_weights, assets->notes, e, "rosvot.notes.weights"),
          words(*assets->word_weights, assets->words, e, "rosvot.words.weights") {
        debug::trace_log_scalar("rosvot.threads", static_cast<int64_t>(e.config().threads));
        modules::RmvpePitchExtractorConfig pitch_config;
        pitch_config.trace_prefix = "rosvot.rmvpe";
#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        pitch_config.cuda_native = e.config().type == core::BackendType::Cuda;
#endif
        pitch = modules::RmvpePitchExtractorComponent(assets->pitch_weights, e.config(),
            assets::TensorStorageType::F32, pitch_config);
        audio::MelSpectrogramFrontendConfig c;
        c.sample_rate = kRate; c.n_fft = c.win_length = 512; c.hop_length = kHop;
        c.n_mels = 80; c.mel_fmin = 30; c.mel_fmax = 12000;
        c.magnitude_epsilon = 1e-9f; c.value_transform = audio::MelValueTransform::Log10;
        c.layout = audio::MelOutputLayout::TimeMajor;
        frontend = audio::get_cached_mel_spectrogram_frontend(c);
    }
};

RosvotRuntime::RosvotRuntime(std::shared_ptr<const RosvotAssets> assets, const core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(std::move(assets), execution)) {}
RosvotRuntime::~RosvotRuntime() = default;

Transcription RosvotRuntime::transcribe(const std::vector<float> & waveform, float note_threshold,
    float word_threshold, const std::vector<double> & word_durations) {
    auto & s = *impl_;
    const auto start = Clock::now();
    std::vector<double> durations;
    auto supplied = word_durations;
    for (size_t i = 0; i < supplied.size(); ++i) {
        if (supplied[i] < 0.02) {
            if (i == 0 && supplied.size() > 1) supplied[i + 1] += supplied[i];
            else if (!durations.empty()) durations.back() += supplied[i];
        } else durations.push_back(supplied[i]);
    }
    int64_t real = static_cast<int64_t>((waveform.size() + kHop - 1) / kHop);
    if (!word_durations.empty()) {
        if (durations.empty()) throw std::runtime_error("ROSVOT word durations contain no usable word");
        const double total = std::accumulate(durations.begin(), durations.end(), 0.0);
        real = std::min(real, static_cast<int64_t>(total * kRate / kHop + 0.5));
        if (real < 2) throw std::runtime_error("ROSVOT word duration timeline is too short");
    }
    const int64_t multiple = 1ll << std::max(s.assets->notes.down_layers, s.assets->words.down_layers);
    const int64_t frames = (real + multiple - 1) / multiple * multiple;
    if (waveform.size() <= 192 || frames > 30000) throw std::runtime_error("ROSVOT expects more than 8 ms and at most 160 seconds of audio per request");
    std::vector<float> padded = waveform;
    if (padded.size() < static_cast<size_t>(frames * kHop)) padded.resize(static_cast<size_t>(frames * kHop), 0.0f);
    for (auto & v : padded) v = std::clamp(v, -1.0f, 1.0f);
    const auto m = s.frontend->extract_mono(padded, s.execution.config().threads);
    std::vector<float> mel(static_cast<size_t>(frames * s.assets->notes.mel_bins), 0);
    for (int64_t t = 0; t < real; ++t) std::copy_n(m.values.begin() + t * 80,
        s.assets->notes.mel_bins, mel.begin() + t * s.assets->notes.mel_bins);
    debug::timing_log_scalar("rosvot.frontend_ms", debug::elapsed_ms(start));
    const auto pitch_start = Clock::now();
    audio::TorchaudioSincHannResampleOptions resampling;
    resampling.lowpass_filter_width = 128;
    const auto pitch_audio = audio::resample_mono_torchaudio_sinc_hann(padded, kRate, 16000, resampling);
    debug::timing_log_scalar("rosvot.pitch_resample_ms", debug::elapsed_ms(pitch_start));
    auto f0 = s.pitch.infer_16k_mono(pitch_audio, 0.03f, s.execution.config().threads);
    for (auto & hz : f0) if (hz < 50 || hz > 900) hz = 0;
    for (size_t i = 0; i + 7 < f0.size(); ++i) if (f0[i] == 0 && f0[i + 7] == 0) {
        std::fill(f0.begin() + static_cast<ptrdiff_t>(i), f0.begin() + static_cast<ptrdiff_t>(i + 8), 0.0f);
    }
    std::vector<int32_t> coarse(static_cast<size_t>(frames), 1), uv(static_cast<size_t>(frames), 0);
    const float low = 1127 * std::log(1 + 50.0f / 700), high = 1127 * std::log(1 + 900.0f / 700);
    for (int64_t t = 0; t < real; ++t) {
        const double index = static_cast<double>(t) * kHop / kRate / 0.01;
        const auto a = std::min(static_cast<size_t>(index), f0.size() - 1);
        const auto b = std::min(a + 1, f0.size() - 1);
        const float frac = static_cast<float>(index - static_cast<double>(a));
        const bool unvoiced = (1 - frac) * (f0[a] == 0 ? 1.0f : 0.0f) + frac * (f0[b] == 0 ? 1.0f : 0.0f) > 0.5f;
        float hz = unvoiced ? 0 : f0[a] * (1 - frac) + f0[b] * frac;
        if (hz > 0) hz = std::clamp(hz, 50.0f, 900.0f);
        const float value = hz > 0 ? (1127 * std::log(1 + hz / 700) - low) * 254 / (high - low) + 1 : 1;
        coarse[static_cast<size_t>(t)] = static_cast<int32_t>(std::clamp(value, 1.0f, 255.0f) + 0.5f);
        uv[static_cast<size_t>(t)] = unvoiced ? 1 : 0;
    }
    debug::timing_log_scalar("rosvot.pitch_extraction_ms", debug::elapsed_ms(pitch_start));
    Transcription out;
    const auto words_start = Clock::now();
    std::vector<int32_t> word(static_cast<size_t>(frames), 0);
    if (durations.empty()) {
        if (!s.word_graph || s.word_graph->frames != frames) s.word_graph = std::make_unique<FrameGraph>(s.execution, s.words, frames, true);
        word = regulate_boundary(s.word_graph->run(mel, coarse, uv), word_threshold,
            s.assets->words.boundary_gap, real);
    } else {
        double offset = 0;
        for (size_t i = 0; i + 1 < durations.size(); ++i) {
            offset += durations[i];
            const auto frame = static_cast<int64_t>(std::floor(offset * kRate / kHop + 0.5));
            if (frame > 0 && frame < real) word[static_cast<size_t>(frame)] = 1;
        }
        out.word_durations = durations;
    }
    debug::timing_log_scalar("rosvot.word_boundary_ms", debug::elapsed_ms(words_start));
    const auto encoder_start = Clock::now();
    if (!s.note_graph || s.note_graph->frames != frames) {
        s.pitch_graph.reset();
        s.note_graph = std::make_unique<FrameGraph>(s.execution, s.notes, frames, false);
    }
    auto boundaries = regulate_boundary(s.note_graph->run(mel, coarse, uv, word), note_threshold,
        s.assets->notes.boundary_gap, real, word, s.assets->notes.reference_gap);
    std::vector<float> attention(static_cast<size_t>(frames));
    core::read_tensor_f32_into(s.note_graph->attention.tensor, attention);
    debug::timing_log_scalar("rosvot.encoder_ms", debug::elapsed_ms(encoder_start));
    std::vector<int64_t> starts{0};
    for (int64_t t = 1; t < frames; ++t) if (boundaries[static_cast<size_t>(t)]) starts.push_back(t);
    const int64_t count = static_cast<int64_t>(starts.size());
    starts.push_back(frames);
    std::vector<float> pooling(static_cast<size_t>(count * frames), 0.0f);
    std::vector<float> denominators(static_cast<size_t>(count), 1e-5f);
    for (int64_t n = 0; n < count; ++n) {
        float & denominator = denominators[static_cast<size_t>(n)];
        for (auto t = starts[static_cast<size_t>(n)]; t < starts[static_cast<size_t>(n + 1)]; ++t) denominator += attention[static_cast<size_t>(t)];
        for (auto t = starts[static_cast<size_t>(n)]; t < starts[static_cast<size_t>(n + 1)]; ++t)
            pooling[static_cast<size_t>(n * frames + t)] = attention[static_cast<size_t>(t)];
    }
    const auto decoder_start = Clock::now();
    if (!s.pitch_graph || s.pitch_graph->notes != count) s.pitch_graph = std::make_unique<PitchGraph>(s.execution, s.notes, s.note_graph->feature, count);
    core::write_tensor_f32(s.pitch_graph->pooling, pooling);
    core::write_tensor_f32(s.pitch_graph->denominator, denominators);
    s.pitch_graph->owner.compute();
    const int64_t classes = s.assets->notes.note_num + 4;
    std::vector<float> logits(static_cast<size_t>(count * classes));
    core::read_tensor_f32_into(s.pitch_graph->logits.tensor, logits);
    int64_t word_index = 1;
    for (int64_t n = 0; n < count; ++n) {
        const auto offset = logits.begin() + n * classes;
        const int pitch = static_cast<int>(std::max_element(offset, offset + classes) - offset);
        const int valid_pitch = pitch >= s.assets->notes.note_start && pitch <= s.assets->notes.note_num ? pitch : 0;
        const auto first = starts[static_cast<size_t>(n)];
        const auto last = n + 1 == count ? real - 1 : starts[static_cast<size_t>(n + 1)];
        if (n && word[static_cast<size_t>(first)]) ++word_index;
        out.notes.push_back({false, 0, first * static_cast<double>(kHop) / kRate,
            last * static_cast<double>(kHop) / kRate, valid_pitch, "Singing"});
        out.note_to_word.push_back(word_index);
    }
    if (!durations.empty()) {
        // Upstream's supplied-word interval regulation and same-word slur merge.
        std::vector<int64_t> word_ids(static_cast<size_t>(real), 0);
        int64_t id = 0;
        for (int64_t t = 0; t < real; ++t) if (word[static_cast<size_t>(t)]) word_ids[static_cast<size_t>(t)] = ++id;
        if (id + 1 != static_cast<int64_t>(durations.size()))
            throw std::runtime_error("ROSVOT supplied word boundaries do not fit the input timeline");
        std::vector<double> offsets{0};
        for (const auto duration : durations) offsets.push_back(offsets.back() + duration);
        for (size_t n = 0; n < out.notes.size(); ++n) {
            const auto first = starts[n];
            const auto last = n + 1 == out.notes.size() ? real - 1 : starts[n + 1];
            const double midpoint = 0.5 * (out.notes[n].onset_seconds + out.notes[n].offset_seconds);
            int64_t assigned = std::upper_bound(offsets.begin(), offsets.end(), midpoint) - offsets.begin() - 1;
            if (word_ids[static_cast<size_t>(first)]) {
                assigned = word_ids[static_cast<size_t>(first)];
                out.notes[n].onset_seconds = offsets[static_cast<size_t>(assigned)];
            }
            if (word_ids[static_cast<size_t>(last)]) {
                assigned = word_ids[static_cast<size_t>(last)] - 1;
                out.notes[n].offset_seconds = offsets[static_cast<size_t>(assigned + 1)];
            }
            out.note_to_word[n] = assigned + 1;
        }
        std::vector<midi::MidiNote> merged;
        std::vector<int64_t> mapping;
        for (size_t i = 0; i < out.notes.size(); ++i) {
            if (!merged.empty() && mapping.back() == out.note_to_word[i] && merged.back().pitch == out.notes[i].pitch)
                merged.back().offset_seconds = out.notes[i].offset_seconds;
            else { merged.push_back(out.notes[i]); mapping.push_back(out.note_to_word[i]); }
        }
        out.notes = std::move(merged); out.note_to_word = std::move(mapping);
    }
    debug::timing_log_scalar("rosvot.note_decode_ms", debug::elapsed_ms(decoder_start));
    return out;
}

} // namespace engine::models::rosvot
