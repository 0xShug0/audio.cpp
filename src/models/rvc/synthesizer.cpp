#include "engine/models/rvc/synthesizer.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::rvc {
namespace {

using engine::core::TensorShape;
using engine::core::TensorValue;

struct RvcSynthesizerWeights {
    std::shared_ptr<engine::core::ExecutionContext> execution_context;
    std::shared_ptr<engine::core::BackendWeightStore> store;
    std::unordered_map<std::string, TensorValue> tensors;
    std::unordered_map<std::string, std::vector<float>> relative_embeddings;
    int sample_rate = 0;
    RvcSynthesizerLayout layout;
    bool v1 = false;
    bool has_f0 = true;
};

struct RvcGraphInput {
    TensorValue tensor;
    std::vector<float> values;
};

struct RvcTraceTensor {
    std::string name;
    TensorValue tensor;
    std::vector<int64_t> dims;
};

constexpr int64_t kInterChannels = 192;
constexpr int64_t kHiddenChannels = 192;
constexpr int64_t kFilterChannels = 768;
constexpr int64_t kHeads = 2;
constexpr int64_t kHeadDim = 96;
constexpr int64_t kTextLayers = 6;
constexpr int64_t kRelativeWindowSize = 10;
constexpr float kLayerNormEps = 1.0e-5F;
constexpr float kLReluSlope = 0.1F;

TensorShape shape_from_vector(const std::vector<int64_t> & dims) {
    switch (dims.size()) {
        case 1:
            return TensorShape::from_dims({dims[0]});
        case 2:
            return TensorShape::from_dims({dims[0], dims[1]});
        case 3:
            return TensorShape::from_dims({dims[0], dims[1], dims[2]});
        case 4:
            return TensorShape::from_dims({dims[0], dims[1], dims[2], dims[3]});
        default:
            throw std::runtime_error("RVC synthesizer tensor rank must be 1..4");
    }
}

bool has_suffix(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string without_suffix(const std::string & value, const std::string & suffix) {
    if (!has_suffix(value, suffix)) {
        throw std::runtime_error("RVC synthesizer internal suffix error");
    }
    return value.substr(0, value.size() - suffix.size());
}

std::vector<float> fold_weight_norm(
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    const std::vector<int64_t> & v_shape) {
    if (v_shape.empty()) {
        throw std::runtime_error("RVC weight_norm tensor shape is empty: " + prefix);
    }
    const auto g = source.require_f32(prefix + ".weight_g");
    const auto v = source.require_f32(prefix + ".weight_v", v_shape);
    const int64_t dim0 = v_shape[0];
    const int64_t rest = static_cast<int64_t>(v.size()) / dim0;
    if (static_cast<int64_t>(g.size()) != dim0) {
        throw std::runtime_error("RVC weight_norm g shape mismatch: " + prefix);
    }
    std::vector<float> out(v.size(), 0.0F);
    for (int64_t row = 0; row < dim0; ++row) {
        double sum = 0.0;
        const size_t offset = static_cast<size_t>(row * rest);
        for (int64_t i = 0; i < rest; ++i) {
            const float value = v[offset + static_cast<size_t>(i)];
            sum += static_cast<double>(value) * static_cast<double>(value);
        }
        const double norm = std::sqrt(sum);
        if (norm == 0.0) {
            throw std::runtime_error("RVC weight_norm norm is zero: " + prefix);
        }
        const float scale = static_cast<float>(static_cast<double>(g[static_cast<size_t>(row)]) / norm);
        for (int64_t i = 0; i < rest; ++i) {
            out[offset + static_cast<size_t>(i)] = v[offset + static_cast<size_t>(i)] * scale;
        }
    }
    return out;
}

std::shared_ptr<RvcSynthesizerWeights> load_weights(
    std::shared_ptr<const engine::assets::TensorSource> source,
    engine::core::BackendConfig backend,
    engine::assets::TensorStorageType storage_type,
    int sample_rate,
    RvcSynthesizerLayout layout,
    bool v1,
    bool has_f0) {
    if (source == nullptr) {
        throw std::runtime_error("RVC synthesizer requires a tensor source");
    }
    auto weights = std::make_shared<RvcSynthesizerWeights>();
    weights->sample_rate = sample_rate;
    weights->layout = layout;
    weights->v1 = v1;
    weights->has_f0 = has_f0;
    weights->execution_context = std::make_shared<engine::core::ExecutionContext>(backend);
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        weights->execution_context->backend(),
        weights->execution_context->backend_type(),
        "rvc.synthesizer.weights",
        1024ull * 1024ull * 1024ull);
    const auto tensors = source->tensors();
    weights->tensors.reserve(tensors.size());
    for (const auto & tensor : tensors) {
        if (has_suffix(tensor.name, ".weight_g") || has_suffix(tensor.name, ".weight_v")) {
            continue;
        }
        const bool force_f32 =
            has_suffix(tensor.name, ".bias") ||
            has_suffix(tensor.name, ".gamma") ||
            has_suffix(tensor.name, ".beta") ||
            tensor.name == "emb_g.weight" ||
            tensor.name == "enc_p.emb_pitch.weight";
        weights->tensors.emplace(
            tensor.name,
            force_f32
                ? engine::modules::binding::f32_tensor_from_named_source(*weights->store, *source, tensor.name)
                : engine::modules::binding::tensor_from_named_source(*weights->store, *source, tensor.name, storage_type));
        if (has_suffix(tensor.name, ".emb_rel_k") || has_suffix(tensor.name, ".emb_rel_v")) {
            weights->relative_embeddings.emplace(tensor.name, source->require_f32(tensor.name, tensor.shape));
        }
    }
    for (const auto & tensor : tensors) {
        if (!has_suffix(tensor.name, ".weight_v")) {
            continue;
        }
        const auto prefix = without_suffix(tensor.name, ".weight_v");
        const auto folded = fold_weight_norm(*source, prefix, tensor.shape);
        weights->tensors[prefix + ".weight"] = weights->store->make_from_f32(
            shape_from_vector(tensor.shape),
            storage_type,
            folded);
    }
    weights->store->upload();
    source->release_storage();
    return weights;
}

TensorValue require_tensor(const RvcSynthesizerWeights & weights, const std::string & name) {
    const auto it = weights.tensors.find(name);
    if (it == weights.tensors.end()) {
        throw std::runtime_error("RVC synthesizer missing tensor: " + name);
    }
    return it->second;
}

engine::modules::LinearWeights linear_weights(const RvcSynthesizerWeights & weights, const std::string & prefix) {
    return {
        require_tensor(weights, prefix + ".weight"),
        require_tensor(weights, prefix + ".bias")};
}

engine::modules::Conv1dWeights conv1d_weights(
    const RvcSynthesizerWeights & weights,
    const std::string & prefix,
    bool bias = true) {
    engine::modules::Conv1dWeights out;
    out.weight = require_tensor(weights, prefix + ".weight");
    if (bias) {
        out.bias = require_tensor(weights, prefix + ".bias");
    }
    return out;
}

engine::modules::ConvTranspose1dWeights conv_transpose1d_weights(
    const RvcSynthesizerWeights & weights,
    const std::string & prefix) {
    return {
        require_tensor(weights, prefix + ".weight"),
        require_tensor(weights, prefix + ".bias")};
}

engine::modules::NormWeights gamma_beta_weights(const RvcSynthesizerWeights & weights, const std::string & prefix) {
    return {
        require_tensor(weights, prefix + ".gamma"),
        require_tensor(weights, prefix + ".beta")};
}

TensorValue contiguous(engine::core::ModuleBuildContext & ctx, const TensorValue & value) {
    return engine::core::ensure_backend_addressable_layout(ctx, value);
}

TensorValue sub_same(engine::core::ModuleBuildContext & ctx, const TensorValue & lhs, const TensorValue & rhs) {
    return engine::core::wrap_tensor(
        ggml_sub(ctx.ggml, contiguous(ctx, lhs).tensor, contiguous(ctx, rhs).tensor),
        lhs.shape,
        GGML_TYPE_F32);
}

TensorValue mul_scalar(engine::core::ModuleBuildContext & ctx, const TensorValue & value, float factor) {
    return engine::core::wrap_tensor(
        ggml_scale(ctx.ggml, contiguous(ctx, value).tensor, factor),
        value.shape,
        GGML_TYPE_F32);
}

TensorValue exp_value(engine::core::ModuleBuildContext & ctx, const TensorValue & value) {
    return engine::core::wrap_tensor(
        ggml_exp(ctx.ggml, contiguous(ctx, value).tensor),
        value.shape,
        GGML_TYPE_F32);
}

TensorValue repeat_like(engine::core::ModuleBuildContext & ctx, const TensorValue & value, const TensorValue & like) {
    return engine::core::wrap_tensor(
        ggml_repeat(ctx.ggml, contiguous(ctx, value).tensor, contiguous(ctx, like).tensor),
        like.shape,
        GGML_TYPE_F32);
}

TensorValue pad_last_right(engine::core::ModuleBuildContext & ctx, const TensorValue & value, int64_t right) {
    if (right <= 0) {
        return value;
    }
    auto shape = value.shape;
    shape.dims[shape.rank - 1] += right;
    return engine::core::wrap_tensor(
        ggml_pad(ctx.ggml, contiguous(ctx, value).tensor, static_cast<int>(right), 0, 0, 0),
        shape,
        GGML_TYPE_F32);
}

TensorValue leaky_relu(engine::core::ModuleBuildContext & ctx, const TensorValue & value) {
    return engine::core::wrap_tensor(
        ggml_leaky_relu(ctx.ggml, contiguous(ctx, value).tensor, kLReluSlope, false),
        value.shape,
        GGML_TYPE_F32);
}

TensorValue transpose_btc_bct(engine::core::ModuleBuildContext & ctx, const TensorValue & value) {
    return engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, value);
}

TensorValue rvc_layer_norm_bct(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input_bct,
    const engine::modules::NormWeights & weights) {
    auto btc = transpose_btc_bct(ctx, input_bct);
    btc = engine::modules::LayerNormModule({input_bct.shape.dims[1], kLayerNormEps, true, true}).build(ctx, btc, weights);
    return transpose_btc_bct(ctx, btc);
}

TensorValue relative_position_to_absolute_position(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & x) {
    const int64_t frames = x.shape.dims[2];
    auto y = pad_last_right(ctx, x, 1);
    y = engine::core::reshape_tensor(ctx, contiguous(ctx, y), TensorShape::from_dims({1, kHeads, frames * 2 * frames}));
    y = pad_last_right(ctx, y, frames - 1);
    y = engine::core::reshape_tensor(ctx, contiguous(ctx, y), TensorShape::from_dims({1, kHeads, frames + 1, 2 * frames - 1}));
    y = engine::modules::SliceModule({2, 0, frames}).build(ctx, y);
    return engine::modules::SliceModule({3, frames - 1, frames}).build(ctx, y);
}

TensorValue absolute_position_to_relative_position(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & x) {
    const int64_t frames = x.shape.dims[2];
    auto y = pad_last_right(ctx, x, frames - 1);
    y = engine::core::reshape_tensor(ctx, contiguous(ctx, y), TensorShape::from_dims({1, kHeads, frames * (2 * frames - 1)}));
    auto zeros = engine::modules::SliceModule({2, 0, frames}).build(ctx, y);
    zeros = mul_scalar(ctx, zeros, 0.0F);
    y = engine::modules::ConcatModule({2}).build(ctx, zeros, y);
    y = engine::core::reshape_tensor(ctx, contiguous(ctx, y), TensorShape::from_dims({1, kHeads, frames, 2 * frames}));
    return engine::modules::SliceModule({3, 1, 2 * frames - 1}).build(ctx, y);
}

std::vector<float> expanded_relative_embedding(
    const RvcSynthesizerWeights & weights,
    const std::string & name,
    int64_t frames) {
    const auto it = weights.relative_embeddings.find(name);
    if (it == weights.relative_embeddings.end()) {
        throw std::runtime_error("RVC synthesizer missing CPU relative embedding: " + name);
    }
    const auto & source = it->second;
    constexpr int64_t source_positions = kRelativeWindowSize * 2 + 1;
    if (static_cast<int64_t>(source.size()) != source_positions * kHeadDim) {
        throw std::runtime_error("RVC relative embedding shape mismatch: " + name);
    }
    const int64_t relative_positions = 2 * frames - 1;
    const int64_t pad_length = std::max<int64_t>(frames - (kRelativeWindowSize + 1), 0);
    const int64_t slice_start = std::max<int64_t>((kRelativeWindowSize + 1) - frames, 0);
    std::vector<float> out(static_cast<size_t>(kHeads * relative_positions * kHeadDim), 0.0F);
    for (int64_t head = 0; head < kHeads; ++head) {
        for (int64_t pos = 0; pos < relative_positions; ++pos) {
            const int64_t padded_pos = slice_start + pos;
            const int64_t source_pos = padded_pos - pad_length;
            if (source_pos < 0 || source_pos >= source_positions) {
                continue;
            }
            const auto * src = source.data() + static_cast<size_t>(source_pos * kHeadDim);
            auto * dst = out.data() + static_cast<size_t>((head * relative_positions + pos) * kHeadDim);
            std::copy(src, src + kHeadDim, dst);
        }
    }
    return out;
}

TensorValue relative_embedding_input(
    engine::core::ModuleBuildContext & ctx,
    const RvcSynthesizerWeights & weights,
    const std::string & name,
    int64_t frames,
    std::vector<RvcGraphInput> & graph_inputs) {
    auto tensor = engine::core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        TensorShape::from_dims({1, kHeads, 2 * frames - 1, kHeadDim}));
    graph_inputs.push_back({tensor, expanded_relative_embedding(weights, name, frames)});
    return tensor;
}

TensorValue rvc_self_attention(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & x_bct,
    const RvcSynthesizerWeights & weights,
    int64_t layer,
    std::vector<RvcGraphInput> & graph_inputs) {
    const int64_t frames = x_bct.shape.dims[2];
    const std::string prefix = "enc_p.encoder.attn_layers." + std::to_string(layer);
    auto q = engine::modules::Conv1dModule({kHiddenChannels, kHiddenChannels, 1, 1, 0, 1, true})
                 .build(ctx, x_bct, conv1d_weights(weights, prefix + ".conv_q"));
    auto k = engine::modules::Conv1dModule({kHiddenChannels, kHiddenChannels, 1, 1, 0, 1, true})
                 .build(ctx, x_bct, conv1d_weights(weights, prefix + ".conv_k"));
    auto v = engine::modules::Conv1dModule({kHiddenChannels, kHiddenChannels, 1, 1, 0, 1, true})
                 .build(ctx, x_bct, conv1d_weights(weights, prefix + ".conv_v"));

    q = engine::core::reshape_tensor(ctx, contiguous(ctx, q), TensorShape::from_dims({1, kHeads, kHeadDim, frames}));
    k = engine::core::reshape_tensor(ctx, contiguous(ctx, k), TensorShape::from_dims({1, kHeads, kHeadDim, frames}));
    v = engine::core::reshape_tensor(ctx, contiguous(ctx, v), TensorShape::from_dims({1, kHeads, kHeadDim, frames}));
    q = engine::modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, q);
    k = engine::modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k);
    v = engine::modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, v);

    auto q_scaled = mul_scalar(ctx, q, static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHeadDim))));
    auto scores = engine::modules::MatMulModule().build(
        ctx,
        q_scaled,
        engine::modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k));
    auto rel_k = relative_embedding_input(ctx, weights, prefix + ".emb_rel_k", frames, graph_inputs);
    auto rel_logits = engine::modules::MatMulModule().build(
        ctx,
        q_scaled,
        engine::modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, rel_k));
    scores = engine::modules::AddModule{}.build(ctx, scores, relative_position_to_absolute_position(ctx, rel_logits));
    auto attn = engine::core::wrap_tensor(ggml_soft_max(ctx.ggml, contiguous(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
    auto out = engine::modules::MatMulModule().build(ctx, attn, v);
    auto rel_v = relative_embedding_input(ctx, weights, prefix + ".emb_rel_v", frames, graph_inputs);
    auto relative_weights = absolute_position_to_relative_position(ctx, attn);
    out = engine::modules::AddModule{}.build(ctx, out, engine::modules::MatMulModule().build(ctx, relative_weights, rel_v));
    out = engine::modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, out);
    out = engine::core::reshape_tensor(ctx, contiguous(ctx, out), TensorShape::from_dims({1, kHiddenChannels, frames}));
    return engine::modules::Conv1dModule({kHiddenChannels, kHiddenChannels, 1, 1, 0, 1, true})
        .build(ctx, out, conv1d_weights(weights, prefix + ".conv_o"));
}

TensorValue rvc_ffn(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & x_bct,
    const RvcSynthesizerWeights & weights,
    int64_t layer) {
    const std::string prefix = "enc_p.encoder.ffn_layers." + std::to_string(layer);
    auto y = engine::modules::Conv1dModule({kHiddenChannels, kFilterChannels, 3, 1, 1, 1, true})
                 .build(ctx, x_bct, conv1d_weights(weights, prefix + ".conv_1"));
    y = engine::modules::ReluModule().build(ctx, y);
    return engine::modules::Conv1dModule({kFilterChannels, kHiddenChannels, 3, 1, 1, 1, true})
        .build(ctx, y, conv1d_weights(weights, prefix + ".conv_2"));
}

TensorValue build_text_encoder_stats(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & features_btd,
    const TensorValue & pitch_i32,
    const RvcSynthesizerWeights & weights,
    std::vector<RvcGraphInput> & graph_inputs,
    std::vector<RvcTraceTensor> & trace_tensors) {
    auto x = engine::modules::LinearModule({features_btd.shape.dims[2], kHiddenChannels, true, GGML_PREC_F32})
                 .build(ctx, features_btd, linear_weights(weights, "enc_p.emb_phone"));
    const bool trace_enabled = engine::debug::trace_log_enabled();
    if (trace_enabled) {
        trace_tensors.push_back({
            "rvc.synth.phone_pre_bct",
            contiguous(ctx, transpose_btc_bct(ctx, x)),
            {1, kHiddenChannels, features_btd.shape.dims[1]}});
    }
    if (weights.has_f0) {
        auto pitch = engine::modules::EmbeddingModule({256, kHiddenChannels})
                         .build(ctx, pitch_i32, require_tensor(weights, "enc_p.emb_pitch.weight"));
        if (trace_enabled) {
            trace_tensors.push_back({
                "rvc.synth.pitch_pre_bct",
                contiguous(ctx, transpose_btc_bct(ctx, pitch)),
                {1, kHiddenChannels, features_btd.shape.dims[1]}});
        }
        x = engine::modules::AddModule{}.build(ctx, x, pitch);
        if (trace_enabled) {
            trace_tensors.push_back({
                "rvc.synth.sum_pre_bct",
                contiguous(ctx, transpose_btc_bct(ctx, x)),
                {1, kHiddenChannels, features_btd.shape.dims[1]}});
        }
    }
    x = mul_scalar(ctx, x, static_cast<float>(std::sqrt(static_cast<double>(kHiddenChannels))));
    if (trace_enabled) {
        trace_tensors.push_back({
            "rvc.synth.scaled_pre_bct",
            contiguous(ctx, transpose_btc_bct(ctx, x)),
            {1, kHiddenChannels, features_btd.shape.dims[1]}});
    }
    x = leaky_relu(ctx, x);
    x = transpose_btc_bct(ctx, x);
    if (trace_enabled) {
        trace_tensors.push_back({"rvc.synth.text_embedding_bct", contiguous(ctx, x), {1, kHiddenChannels, x.shape.dims[2]}});
    }
    for (int64_t layer = 0; layer < kTextLayers; ++layer) {
        auto y = rvc_self_attention(ctx, x, weights, layer, graph_inputs);
        x = rvc_layer_norm_bct(
            ctx,
            engine::modules::AddModule{}.build(ctx, x, y),
            gamma_beta_weights(weights, "enc_p.encoder.norm_layers_1." + std::to_string(layer)));
        y = rvc_ffn(ctx, x, weights, layer);
        x = rvc_layer_norm_bct(
            ctx,
            engine::modules::AddModule{}.build(ctx, x, y),
            gamma_beta_weights(weights, "enc_p.encoder.norm_layers_2." + std::to_string(layer)));
        if (trace_enabled && (layer == 0 || layer == kTextLayers - 1)) {
            trace_tensors.push_back({
                "rvc.synth.text_layer_" + std::to_string(layer),
                contiguous(ctx, x),
                {1, kHiddenChannels, x.shape.dims[2]}});
        }
    }
    auto x_btc = transpose_btc_bct(ctx, x);
    auto proj_weight = engine::core::reshape_tensor(
        ctx,
        require_tensor(weights, "enc_p.proj.weight"),
        TensorShape::from_dims({kInterChannels * 2, kHiddenChannels}));
    auto stats_btc = engine::modules::LinearModule({kHiddenChannels, kInterChannels * 2, true, GGML_PREC_F32})
                         .build(ctx, x_btc, {proj_weight, require_tensor(weights, "enc_p.proj.bias")});
    return transpose_btc_bct(ctx, stats_btc);
}

TensorValue fused_tanh_sigmoid(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & x) {
    auto left = engine::modules::SliceModule({1, 0, kHiddenChannels}).build(ctx, x);
    auto right = engine::modules::SliceModule({1, kHiddenChannels, kHiddenChannels}).build(ctx, x);
    left = engine::modules::TanhModule().build(ctx, left);
    right = engine::modules::SigmoidModule().build(ctx, right);
    return engine::modules::MulModule().build(ctx, left, right);
}

TensorValue build_wn(
    engine::core::ModuleBuildContext & ctx,
    TensorValue x,
    const TensorValue & g,
    const RvcSynthesizerWeights & weights,
    const std::string & prefix) {
    auto cond = engine::modules::Conv1dModule({256, kHiddenChannels * 2 * 3, 1, 1, 0, 1, true})
                    .build(ctx, g, conv1d_weights(weights, prefix + ".cond_layer"));
    TensorValue output = mul_scalar(ctx, x, 0.0F);
    for (int64_t layer = 0; layer < 3; ++layer) {
        auto x_in = engine::modules::Conv1dModule({kHiddenChannels, kHiddenChannels * 2, 5, 1, 2, 1, true})
                        .build(ctx, x, conv1d_weights(weights, prefix + ".in_layers." + std::to_string(layer)));
        auto g_l = engine::modules::SliceModule({1, layer * kHiddenChannels * 2, kHiddenChannels * 2}).build(ctx, cond);
        g_l = repeat_like(ctx, g_l, x_in);
        auto acts = fused_tanh_sigmoid(ctx, engine::modules::AddModule{}.build(ctx, x_in, g_l));
        const int64_t res_skip_channels = layer < 2 ? kHiddenChannels * 2 : kHiddenChannels;
        auto res_skip = engine::modules::Conv1dModule({kHiddenChannels, res_skip_channels, 1, 1, 0, 1, true})
                            .build(ctx, acts, conv1d_weights(weights, prefix + ".res_skip_layers." + std::to_string(layer)));
        if (layer < 2) {
            auto res = engine::modules::SliceModule({1, 0, kHiddenChannels}).build(ctx, res_skip);
            auto skip = engine::modules::SliceModule({1, kHiddenChannels, kHiddenChannels}).build(ctx, res_skip);
            x = engine::modules::AddModule{}.build(ctx, x, res);
            output = engine::modules::AddModule{}.build(ctx, output, skip);
        } else {
            output = engine::modules::AddModule{}.build(ctx, output, res_skip);
        }
    }
    return output;
}

TensorValue flip_channels(engine::core::ModuleBuildContext & ctx, const TensorValue & x) {
    TensorValue reversed;
    for (int64_t channel = kInterChannels - 1; channel >= 0; --channel) {
        auto slice = engine::modules::SliceModule({1, channel, 1}).build(ctx, x);
        reversed = reversed.valid() ? engine::modules::ConcatModule({1}).build(ctx, reversed, slice) : slice;
    }
    return reversed;
}

TensorValue residual_coupling_reverse(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & x,
    const TensorValue & g,
    const RvcSynthesizerWeights & weights,
    int64_t flow_index) {
    auto x0 = engine::modules::SliceModule({1, 0, kInterChannels / 2}).build(ctx, x);
    auto x1 = engine::modules::SliceModule({1, kInterChannels / 2, kInterChannels / 2}).build(ctx, x);
    const std::string prefix = "flow.flows." + std::to_string(flow_index);
    auto h = engine::modules::Conv1dModule({kInterChannels / 2, kHiddenChannels, 1, 1, 0, 1, true})
                 .build(ctx, x0, conv1d_weights(weights, prefix + ".pre"));
    h = build_wn(ctx, h, g, weights, prefix + ".enc");
    auto mean = engine::modules::Conv1dModule({kHiddenChannels, kInterChannels / 2, 1, 1, 0, 1, true})
                    .build(ctx, h, conv1d_weights(weights, prefix + ".post"));
    x1 = sub_same(ctx, x1, mean);
    return engine::modules::ConcatModule({1}).build(ctx, x0, x1);
}

TensorValue speaker_embedding_g(
    engine::core::ModuleBuildContext & ctx,
    const RvcSynthesizerWeights & weights,
    int speaker_id) {
    auto emb = engine::modules::SliceModule({0, speaker_id, 1}).build(ctx, require_tensor(weights, "emb_g.weight"));
    emb = engine::core::reshape_tensor(ctx, contiguous(ctx, emb), TensorShape::from_dims({1, 256, 1}));
    return emb;
}

TensorValue build_flow_reverse(
    engine::core::ModuleBuildContext & ctx,
    TensorValue z,
    const TensorValue & g,
    const RvcSynthesizerWeights & weights) {
    z = flip_channels(ctx, z);
    z = residual_coupling_reverse(ctx, z, g, weights, 6);
    z = flip_channels(ctx, z);
    z = residual_coupling_reverse(ctx, z, g, weights, 4);
    z = flip_channels(ctx, z);
    z = residual_coupling_reverse(ctx, z, g, weights, 2);
    z = flip_channels(ctx, z);
    z = residual_coupling_reverse(ctx, z, g, weights, 0);
    return z;
}

TensorValue source_module(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & sine_btc,
    const RvcSynthesizerWeights & weights) {
    auto source = engine::modules::LinearModule({1, 1, true, GGML_PREC_F32})
                      .build(ctx, sine_btc, linear_weights(weights, "dec.m_source.l_linear"));
    source = engine::modules::TanhModule().build(ctx, source);
    return transpose_btc_bct(ctx, source);
}

TensorValue resblock1(
    engine::core::ModuleBuildContext & ctx,
    TensorValue x,
    const RvcSynthesizerWeights & weights,
    int64_t index,
    int64_t channels,
    int64_t kernel) {
    const int64_t dilations[3] = {1, 3, 5};
    const std::string prefix = "dec.resblocks." + std::to_string(index);
    for (int64_t layer = 0; layer < 3; ++layer) {
        auto xt = leaky_relu(ctx, x);
        const int64_t dilation = dilations[layer];
        const int64_t padding = (kernel * dilation - dilation) / 2;
        xt = engine::modules::Conv1dModule({channels, channels, kernel, 1, static_cast<int>(padding), static_cast<int>(dilation), true})
                 .build(ctx, xt, conv1d_weights(weights, prefix + ".convs1." + std::to_string(layer)));
        xt = leaky_relu(ctx, xt);
        xt = engine::modules::Conv1dModule({channels, channels, kernel, 1, static_cast<int>((kernel - 1) / 2), 1, true})
                 .build(ctx, xt, conv1d_weights(weights, prefix + ".convs2." + std::to_string(layer)));
        x = engine::modules::AddModule{}.build(ctx, x, xt);
    }
    return x;
}

TensorValue conv_transpose1d_pytorch_padding(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & x,
    const RvcSynthesizerWeights & weights,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int64_t stride,
    int64_t padding) {
    auto raw = engine::modules::ConvTranspose1dModule({
        in_channels,
        out_channels,
        kernel,
        static_cast<int>(stride),
        0,
        1,
        true}).build(ctx, x, conv_transpose1d_weights(weights, prefix));
    const int64_t trimmed_frames = raw.shape.dims[2] - 2 * padding;
    if (trimmed_frames <= 0) {
        throw std::runtime_error("RVC conv-transpose padding trim removed all frames");
    }
    return engine::modules::SliceModule({2, padding, trimmed_frames}).build(ctx, raw);
}

TensorValue build_generator(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & z,
    const TensorValue & g,
    const TensorValue & sine_btc,
    const RvcSynthesizerWeights & weights) {
    TensorValue source;
    if (weights.has_f0) {
        source = source_module(ctx, sine_btc, weights);
    }
    auto x = engine::modules::Conv1dModule({kInterChannels, 512, 7, 1, 3, 1, true})
                 .build(ctx, z, conv1d_weights(weights, "dec.conv_pre"));
    auto cond = engine::modules::Conv1dModule({256, 512, 1, 1, 0, 1, true})
                    .build(ctx, g, conv1d_weights(weights, "dec.cond"));
    x = engine::modules::AddModule{}.build(ctx, x, repeat_like(ctx, cond, x));

    const int64_t channels[4] = {256, 128, 64, 32};
    const int64_t in_channels[4] = {512, 256, 128, 64};
    const int64_t kernels[3] = {3, 7, 11};
    for (int64_t up = 0; up < 4; ++up) {
        const int64_t up_rate = weights.layout.upsample_rates[static_cast<size_t>(up)];
        const int64_t up_kernel = weights.layout.upsample_kernel_sizes[static_cast<size_t>(up)];
        x = leaky_relu(ctx, x);
        x = conv_transpose1d_pytorch_padding(
            ctx,
            x,
            weights,
            "dec.ups." + std::to_string(up),
            in_channels[up],
            channels[up],
            up_kernel,
            up_rate,
            (up_kernel - up_rate) / 2);
        if (weights.has_f0) {
            int64_t noise_stride = 1;
            for (int64_t next = up + 1; next < 4; ++next) {
                noise_stride *= weights.layout.upsample_rates[static_cast<size_t>(next)];
            }
            const int64_t noise_kernel = noise_stride == 1 ? 1 : noise_stride * 2;
            const int64_t noise_padding = noise_stride / 2;
            auto x_source = engine::modules::Conv1dModule({
                1,
                channels[up],
                noise_kernel,
                static_cast<int>(noise_stride),
                static_cast<int>(noise_padding),
                1,
                true}).build(ctx, source, conv1d_weights(weights, "dec.noise_convs." + std::to_string(up)));
            x = engine::modules::AddModule{}.build(ctx, x, x_source);
        }
        TensorValue sum;
        for (int64_t kernel_index = 0; kernel_index < 3; ++kernel_index) {
            auto rb = resblock1(ctx, x, weights, up * 3 + kernel_index, channels[up], kernels[kernel_index]);
            sum = sum.valid() ? engine::modules::AddModule{}.build(ctx, sum, rb) : rb;
        }
        x = mul_scalar(ctx, sum, 1.0F / 3.0F);
    }
    x = leaky_relu(ctx, x);
    x = engine::modules::Conv1dModule({32, 1, 7, 1, 3, 1, false})
            .build(ctx, x, conv1d_weights(weights, "dec.conv_post", false));
    return engine::modules::TanhModule().build(ctx, x);
}

}  // namespace

struct RvcSynthesizer::State {
    std::shared_ptr<RvcSynthesizerWeights> weights;
    mutable std::mutex mutex;
    mutable ggml_context * graph_ctx = nullptr;
    mutable ggml_gallocr_t gallocr = nullptr;
    mutable ggml_cgraph * graph = nullptr;
    mutable std::vector<RvcGraphInput> graph_inputs;
    mutable std::vector<RvcTraceTensor> trace_tensors;
    mutable TensorValue features;
    mutable TensorValue pitch;
    mutable TensorValue noise;
    mutable TensorValue sine;
    mutable TensorValue m;
    mutable TensorValue logs;
    mutable TensorValue z_p;
    mutable TensorValue z;
    mutable TensorValue output;
    mutable int64_t frames = 0;
    mutable int64_t feature_dim = 0;
    mutable int speaker_id = -1;
};

RvcSynthesizer::RvcSynthesizer(
    std::shared_ptr<const engine::assets::TensorSource> source,
    engine::core::BackendConfig backend,
    engine::assets::TensorStorageType storage_type,
    int sample_rate,
    RvcSynthesizerLayout layout,
    bool v1,
    bool has_f0)
    : state_(std::make_shared<State>()) {
    state_->weights = load_weights(
        std::move(source),
        std::move(backend),
        storage_type,
        sample_rate,
        layout,
        v1,
        has_f0);
}

RvcSynthesizer::~RvcSynthesizer() {
    if (state_ != nullptr) {
        if (state_->gallocr != nullptr) {
            ggml_gallocr_free(state_->gallocr);
            state_->gallocr = nullptr;
        }
        if (state_->graph_ctx != nullptr) {
            ggml_free(state_->graph_ctx);
            state_->graph_ctx = nullptr;
        }
    }
}
RvcSynthesizer::RvcSynthesizer(RvcSynthesizer &&) noexcept = default;
RvcSynthesizer & RvcSynthesizer::operator=(RvcSynthesizer &&) noexcept = default;

RvcSynthesizerOutput RvcSynthesizer::infer(const RvcSynthesizerInput & input) const {
    if (state_ == nullptr || state_->weights == nullptr) {
        throw std::runtime_error("RVC synthesizer is not initialized");
    }
    if (input.frames <= 0 || input.feature_dim <= 0) {
        throw std::runtime_error("RVC synthesizer input shape is invalid");
    }
    const bool has_f0 = state_->weights->has_f0;
    const int64_t hop_samples = state_->weights->layout.hop_samples;
    if (static_cast<int64_t>(input.features.size()) != input.frames * input.feature_dim ||
        (has_f0 && static_cast<int64_t>(input.pitch.size()) != input.frames) ||
        (has_f0 && static_cast<int64_t>(input.sine_source.size()) != input.frames * hop_samples) ||
        (!has_f0 && !input.pitch.empty()) ||
        (!has_f0 && !input.sine_source.empty())) {
        throw std::runtime_error("RVC synthesizer input values do not match shape");
    }
    std::lock_guard<std::mutex> lock(state_->mutex);
    const bool rebuild_graph =
        state_->graph_ctx == nullptr ||
        state_->frames != input.frames ||
        state_->feature_dim != input.feature_dim ||
        state_->speaker_id != input.speaker_id;
    if (rebuild_graph && state_->graph_ctx != nullptr) {
        ggml_gallocr_free(state_->gallocr);
        state_->gallocr = nullptr;
        ggml_free(state_->graph_ctx);
        state_->graph_ctx = nullptr;
    }
    if (rebuild_graph) {
        state_->graph_inputs.clear();
        state_->trace_tensors.clear();
        state_->frames = input.frames;
        state_->feature_dim = input.feature_dim;
        state_->speaker_id = input.speaker_id;
        state_->graph_ctx = ggml_init({4096ull * 1024ull * 1024ull, nullptr, true});
        if (state_->graph_ctx == nullptr) {
            throw std::runtime_error("failed to initialize RVC synthesizer graph context");
        }
        engine::core::ModuleBuildContext ctx{
            state_->graph_ctx,
            "rvc.synthesizer.text_encoder",
            state_->weights->execution_context->config().type};
        state_->features = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            TensorShape::from_dims({1, input.frames, input.feature_dim}));
        if (has_f0) {
            state_->pitch = engine::core::make_tensor(
                ctx,
                GGML_TYPE_I32,
                TensorShape::from_dims({1, input.frames}));
        }
        state_->noise = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            TensorShape::from_dims({1, kInterChannels, input.frames}));
        if (has_f0) {
            state_->sine = engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                TensorShape::from_dims({1, input.frames * hop_samples, 1}));
        }
        auto stats = build_text_encoder_stats(
            ctx,
            state_->features,
            state_->pitch,
            *state_->weights,
            state_->graph_inputs,
            state_->trace_tensors);
        stats = contiguous(ctx, stats);
        state_->m = contiguous(ctx, engine::modules::SliceModule({1, 0, kInterChannels}).build(ctx, stats));
        state_->logs = contiguous(ctx, engine::modules::SliceModule({1, kInterChannels, kInterChannels}).build(ctx, stats));
        state_->z_p = engine::modules::AddModule{}.build(
            ctx,
            state_->m,
            mul_scalar(ctx, engine::modules::MulModule().build(ctx, exp_value(ctx, state_->logs), state_->noise), 0.66666F));
        auto g = speaker_embedding_g(ctx, *state_->weights, input.speaker_id);
        state_->z = build_flow_reverse(ctx, state_->z_p, g, *state_->weights);
        state_->output = build_generator(ctx, state_->z, g, state_->sine, *state_->weights);
        state_->graph = ggml_new_graph_custom(state_->graph_ctx, 524288, false);
        if (engine::debug::trace_log_enabled()) {
            for (const auto & item : state_->trace_tensors) {
                ggml_set_output(item.tensor.tensor);
                ggml_build_forward_expand(state_->graph, item.tensor.tensor);
            }
            ggml_set_output(state_->m.tensor);
            ggml_set_output(state_->logs.tensor);
            ggml_set_output(state_->z_p.tensor);
            ggml_set_output(state_->z.tensor);
            ggml_build_forward_expand(state_->graph, state_->m.tensor);
            ggml_build_forward_expand(state_->graph, state_->logs.tensor);
            ggml_build_forward_expand(state_->graph, state_->z_p.tensor);
            ggml_build_forward_expand(state_->graph, state_->z.tensor);
        }
        ggml_set_output(state_->output.tensor);
        ggml_build_forward_expand(state_->graph, state_->output.tensor);
        state_->gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(state_->weights->execution_context->backend()));
        if (state_->gallocr == nullptr ||
            !ggml_gallocr_reserve(state_->gallocr, state_->graph) ||
            !ggml_gallocr_alloc_graph(state_->gallocr, state_->graph)) {
            throw std::runtime_error("failed to allocate RVC synthesizer graph tensors");
        }
    }
    engine::core::write_tensor_f32(state_->features, input.features);
    if (has_f0) {
        engine::core::write_tensor_i32(state_->pitch, input.pitch);
        engine::core::write_tensor_f32(state_->sine, input.sine_source);
    }
    auto noise = engine::sampling::generate_torch_cuda_randn(
        static_cast<size_t>(kInterChannels * input.frames),
        1234,
        engine::sampling::TorchRandnPrecision::Float32);
    engine::core::write_tensor_f32(state_->noise, noise);
    for (const auto & graph_input : state_->graph_inputs) {
        engine::core::write_tensor_f32(graph_input.tensor, graph_input.values);
    }
    if (engine::core::compute_backend_graph(state_->weights->execution_context->backend(), state_->graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml_backend_graph_compute failed for RVC synthesizer");
    }
    if (engine::debug::trace_log_enabled()) {
        for (const auto & item : state_->trace_tensors) {
            engine::debug::trace_log_f32(
                item.name,
                item.dims,
                engine::core::read_tensor_f32(item.tensor.tensor));
        }
        engine::debug::trace_log_f32(
            "rvc.synth.m",
            {1, kInterChannels, input.frames},
            engine::core::read_tensor_f32(state_->m.tensor));
        engine::debug::trace_log_f32(
            "rvc.synth.logs",
            {1, kInterChannels, input.frames},
            engine::core::read_tensor_f32(state_->logs.tensor));
        engine::debug::trace_log_f32(
            "rvc.synth.z_p",
            {1, kInterChannels, input.frames},
            engine::core::read_tensor_f32(state_->z_p.tensor));
        engine::debug::trace_log_f32(
            "rvc.synth.z",
            {1, kInterChannels, input.frames},
            engine::core::read_tensor_f32(state_->z.tensor));
    }
    auto audio = engine::core::read_tensor_f32(state_->output.tensor);
    RvcSynthesizerOutput out;
    out.sample_rate = state_->weights->sample_rate;
    out.audio = std::move(audio);
    return out;
}

}  // namespace engine::models::rvc
