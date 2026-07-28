#include "engine/models/rvc/hubert.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::rvc {
namespace {

using engine::core::TensorShape;
using engine::core::TensorValue;

constexpr int64_t kHidden = 768;
constexpr int64_t kIntermediate = 3072;
constexpr int64_t kLayers = 12;
constexpr int64_t kHeads = 12;
constexpr int64_t kHeadDim = 64;
constexpr int64_t kConvPos = 128;
constexpr int64_t kConvPosGroups = 16;
constexpr float kLayerNormEps = 1.0e-5F;
const std::vector<int64_t> kConvDim{512, 512, 512, 512, 512, 512, 512};
const std::vector<int64_t> kConvKernel{10, 3, 3, 3, 3, 2, 2};
const std::vector<int64_t> kConvStride{5, 2, 2, 2, 2, 2, 2};

struct RvcHubertWeights {
    std::shared_ptr<engine::core::ExecutionContext> execution_context;
    std::shared_ptr<engine::core::BackendWeightStore> store;
    std::unordered_map<std::string, TensorValue> tensors;
};

struct RvcHubertGraphInput {
    TensorValue tensor;
    std::vector<float> values;
};

int64_t tensor_elements(const std::vector<int64_t> & shape) {
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, [](int64_t lhs, int64_t rhs) {
        if (rhs <= 0) {
            throw std::runtime_error("RVC HuBERT tensor shape contains non-positive dimension");
        }
        return lhs * rhs;
    });
}

TensorValue require_tensor(const RvcHubertWeights & weights, const std::string & name) {
    const auto it = weights.tensors.find(name);
    if (it == weights.tensors.end()) {
        throw std::runtime_error("RVC HuBERT missing tensor: " + name);
    }
    return it->second;
}

engine::modules::NormWeights norm_weights(const RvcHubertWeights & weights, const std::string & prefix) {
    return {
        require_tensor(weights, prefix + ".weight"),
        require_tensor(weights, prefix + ".bias")};
}

engine::modules::LinearWeights linear_weights(const RvcHubertWeights & weights, const std::string & prefix) {
    return {
        require_tensor(weights, prefix + ".weight"),
        require_tensor(weights, prefix + ".bias")};
}

engine::modules::Conv1dWeights conv1d_weights(
    const RvcHubertWeights & weights,
    const std::string & prefix,
    bool bias) {
    engine::modules::Conv1dWeights out;
    out.weight = require_tensor(weights, prefix + ".weight");
    if (bias) {
        out.bias = require_tensor(weights, prefix + ".bias");
    }
    return out;
}

TensorValue contiguous(engine::core::ModuleBuildContext & ctx, const TensorValue & value) {
    return engine::core::ensure_backend_addressable_layout(ctx, value);
}

TensorValue transpose_bct_btc(engine::core::ModuleBuildContext & ctx, const TensorValue & value) {
    return engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, value);
}

TensorValue add_same(engine::core::ModuleBuildContext & ctx, const TensorValue & lhs, const TensorValue & rhs) {
    return engine::modules::AddModule().build(ctx, lhs, rhs);
}

TensorValue mul_scalar(engine::core::ModuleBuildContext & ctx, const TensorValue & value, float factor) {
    return engine::core::wrap_tensor(
        ggml_scale(ctx.ggml, contiguous(ctx, value).tensor, factor),
        value.shape,
        GGML_TYPE_F32);
}

std::vector<float> effective_pos_conv_weight(const engine::assets::TensorSource & source) {
    const int64_t in_channels = kHidden / kConvPosGroups;
    const auto g = source.require_f32("encoder.pos_conv.0.weight_g", {1, 1, kConvPos});
    const auto v = source.require_f32("encoder.pos_conv.0.weight_v", {kHidden, in_channels, kConvPos});
    std::vector<float> weight(v.size());
    for (int64_t k = 0; k < kConvPos; ++k) {
        double sum = 0.0;
        for (int64_t out = 0; out < kHidden; ++out) {
            for (int64_t in = 0; in < in_channels; ++in) {
                const size_t index = static_cast<size_t>((out * in_channels + in) * kConvPos + k);
                sum += static_cast<double>(v[index]) * static_cast<double>(v[index]);
            }
        }
        const double norm = std::sqrt(sum);
        if (norm == 0.0) {
            throw std::runtime_error("RVC HuBERT positional conv weight norm is zero");
        }
        const float scale = static_cast<float>(static_cast<double>(g[static_cast<size_t>(k)]) / norm);
        for (int64_t out = 0; out < kHidden; ++out) {
            for (int64_t in = 0; in < in_channels; ++in) {
                const size_t index = static_cast<size_t>((out * in_channels + in) * kConvPos + k);
                weight[index] = v[index] * scale;
            }
        }
    }
    return weight;
}

TensorValue group_norm_affine(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const engine::modules::NormWeights & weights) {
    auto input_f32 = input.type == GGML_TYPE_F32
        ? input
        : engine::core::wrap_tensor(ggml_cast(ctx.ggml, input.tensor, GGML_TYPE_F32), input.shape, GGML_TYPE_F32);
    auto mean = engine::modules::ReduceMeanModule({2}).build(ctx, input_f32);
    auto mean_rep = engine::core::wrap_tensor(ggml_repeat(ctx.ggml, mean.tensor, input_f32.tensor), input_f32.shape, GGML_TYPE_F32);
    auto centered = engine::core::wrap_tensor(ggml_sub(ctx.ggml, input_f32.tensor, mean_rep.tensor), input_f32.shape, GGML_TYPE_F32);
    auto variance = engine::modules::ReduceMeanModule({2}).build(ctx, engine::modules::MulModule().build(ctx, centered, centered));
    auto stddev = engine::core::wrap_tensor(
        ggml_sqrt(ctx.ggml, ggml_scale_bias(ctx.ggml, variance.tensor, 1.0F, kLayerNormEps)),
        variance.shape,
        GGML_TYPE_F32);
    auto stddev_rep = engine::core::wrap_tensor(ggml_repeat(ctx.ggml, stddev.tensor, input_f32.tensor), input_f32.shape, GGML_TYPE_F32);
    auto out = engine::core::wrap_tensor(ggml_div(ctx.ggml, centered.tensor, stddev_rep.tensor), input_f32.shape, GGML_TYPE_F32);
    auto gamma = engine::core::reshape_tensor(ctx, *weights.weight, TensorShape::from_dims({1, input.shape.dims[1], 1}));
    auto beta = engine::core::reshape_tensor(ctx, *weights.bias, TensorShape::from_dims({1, input.shape.dims[1], 1}));
    out = engine::core::wrap_tensor(
        ggml_mul(ctx.ggml, out.tensor, ggml_repeat(ctx.ggml, gamma.tensor, out.tensor)),
        out.shape,
        GGML_TYPE_F32);
    return engine::core::wrap_tensor(
        ggml_add(ctx.ggml, out.tensor, ggml_repeat(ctx.ggml, beta.tensor, out.tensor)),
        out.shape,
        GGML_TYPE_F32);
}

TensorValue grouped_pos_conv(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input_bct,
    const engine::modules::Conv1dWeights & weights) {
    const int64_t channels_per_group = kHidden / kConvPosGroups;
    const auto input_contiguous = contiguous(ctx, input_bct);
    TensorValue out;
    for (int64_t group = 0; group < kConvPosGroups; ++group) {
        auto input_group = engine::modules::SliceModule({1, group * channels_per_group, channels_per_group}).build(ctx, input_contiguous);
        auto weight_group = engine::modules::SliceModule({0, group * channels_per_group, channels_per_group}).build(ctx, weights.weight);
        engine::modules::Conv1dWeights group_weights{weight_group, std::nullopt};
        if (weights.bias.has_value()) {
            group_weights.bias = engine::modules::SliceModule({0, group * channels_per_group, channels_per_group}).build(ctx, *weights.bias);
        }
        auto group_out = engine::modules::Conv1dModule({
            channels_per_group,
            channels_per_group,
            kConvPos,
            1,
            static_cast<int>(kConvPos / 2),
            1,
            weights.bias.has_value()}).build(ctx, input_group, group_weights);
        out = out.valid() ? engine::modules::ConcatModule({1}).build(ctx, out, group_out) : group_out;
    }
    out = engine::modules::SliceModule({2, 0, input_bct.shape.dims[2]}).build(ctx, out);
    return engine::modules::GeluModule({engine::modules::GeluApproximation::ExactErf}).build(ctx, out);
}

TensorValue self_attention(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & hidden_btc,
    const RvcHubertWeights & weights,
    int64_t layer,
    const TensorValue * attention_mask) {
    const int64_t batch = hidden_btc.shape.dims[0];
    const int64_t tokens = hidden_btc.shape.dims[1];
    const std::string prefix = "encoder.layers." + std::to_string(layer) + ".self_attn";
    auto q = engine::modules::LinearModule({kHidden, kHidden, true, GGML_PREC_F32})
                 .build(ctx, hidden_btc, linear_weights(weights, prefix + ".q_proj"));
    auto k = engine::modules::LinearModule({kHidden, kHidden, true, GGML_PREC_F32})
                 .build(ctx, hidden_btc, linear_weights(weights, prefix + ".k_proj"));
    auto v = engine::modules::LinearModule({kHidden, kHidden, true, GGML_PREC_F32})
                 .build(ctx, hidden_btc, linear_weights(weights, prefix + ".v_proj"));
    q = engine::core::reshape_tensor(ctx, contiguous(ctx, q), TensorShape::from_dims({batch, tokens, kHeads, kHeadDim}));
    k = engine::core::reshape_tensor(ctx, contiguous(ctx, k), TensorShape::from_dims({batch, tokens, kHeads, kHeadDim}));
    v = engine::core::reshape_tensor(ctx, contiguous(ctx, v), TensorShape::from_dims({batch, tokens, kHeads, kHeadDim}));
    q = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    k = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
    v = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
    auto scores = engine::modules::MatMulModule().build(ctx, q, engine::modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k));
    scores = mul_scalar(ctx, scores, static_cast<float>(1.0 / std::sqrt(static_cast<double>(kHeadDim))));
    if (attention_mask != nullptr) {
        auto repeated_mask = engine::core::wrap_tensor(
            ggml_repeat(ctx.ggml, attention_mask->tensor, scores.tensor),
            scores.shape,
            GGML_TYPE_F32);
        scores = add_same(ctx, scores, repeated_mask);
    }
    auto attn = engine::core::wrap_tensor(ggml_soft_max(ctx.ggml, contiguous(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
    auto context = engine::modules::MatMulModule().build(ctx, attn, v);
    context = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, context);
    context = engine::core::reshape_tensor(ctx, contiguous(ctx, context), TensorShape::from_dims({batch, tokens, kHidden}));
    return engine::modules::LinearModule({kHidden, kHidden, true, GGML_PREC_F32})
        .build(ctx, context, linear_weights(weights, prefix + ".out_proj"));
}

TensorValue feed_forward(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & hidden_btc,
    const RvcHubertWeights & weights,
    int64_t layer) {
    const std::string prefix = "encoder.layers." + std::to_string(layer);
    auto x = engine::modules::LinearModule({kHidden, kIntermediate, true, GGML_PREC_F32})
                 .build(ctx, hidden_btc, linear_weights(weights, prefix + ".fc1"));
    x = engine::modules::GeluModule({engine::modules::GeluApproximation::ExactErf}).build(ctx, x);
    return engine::modules::LinearModule({kIntermediate, kHidden, true, GGML_PREC_F32})
        .build(ctx, x, linear_weights(weights, prefix + ".fc2"));
}

TensorValue build_graph(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const RvcHubertWeights & weights,
    bool v1_features,
    std::vector<RvcHubertGraphInput> & graph_inputs) {
    auto hidden = engine::core::reshape_tensor(ctx, input, TensorShape::from_dims({1, 1, input.shape.dims[1]}));
    int64_t in_channels = 1;
    for (size_t i = 0; i < kConvDim.size(); ++i) {
        const std::string prefix = "feature_extractor.conv_layers." + std::to_string(i);
        hidden = engine::modules::Conv1dModule({
            in_channels,
            kConvDim[i],
            kConvKernel[i],
            static_cast<int>(kConvStride[i]),
            0,
            1,
            false}).build(ctx, hidden, conv1d_weights(weights, prefix + ".conv", false));
        if (i == 0) {
            hidden = group_norm_affine(ctx, hidden, norm_weights(weights, prefix + ".layer_norm"));
        }
        hidden = engine::modules::GeluModule({engine::modules::GeluApproximation::ExactErf}).build(ctx, hidden);
        in_channels = kConvDim[i];
    }
    hidden = transpose_bct_btc(ctx, hidden);
    hidden = engine::modules::LayerNormModule({kConvDim.back(), kLayerNormEps, true, true})
                 .build(ctx, hidden, norm_weights(weights, "feature_projection.layer_norm"));
    hidden = engine::modules::LinearModule({kConvDim.back(), kHidden, true, GGML_PREC_F32})
                 .build(ctx, hidden, linear_weights(weights, "feature_projection.projection"));
    auto pos = grouped_pos_conv(ctx, transpose_bct_btc(ctx, hidden), conv1d_weights(weights, "encoder.pos_conv_embed.conv", true));
    hidden = add_same(ctx, hidden, transpose_bct_btc(ctx, pos));
    hidden = engine::modules::LayerNormModule({kHidden, kLayerNormEps, true, true})
                 .build(ctx, hidden, norm_weights(weights, "encoder.layer_norm"));

    const int64_t raw_tokens = hidden.shape.dims[1];
    TensorValue attention_mask;
    if (raw_tokens % 2 != 0) {
        hidden = engine::modules::PaddingModule({raw_tokens + 1}).build(ctx, hidden);
        attention_mask = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            TensorShape::from_dims({1, 1, 1, hidden.shape.dims[1]}));
        std::vector<float> mask(static_cast<size_t>(hidden.shape.dims[1]), 0.0F);
        mask.back() = -1.0e4F;
        graph_inputs.push_back({attention_mask, std::move(mask)});
    }

    const int64_t output_layer = v1_features ? 10 : 12;
    for (int64_t layer = 0; layer < output_layer; ++layer) {
        auto attn = self_attention(ctx, hidden, weights, layer, attention_mask.valid() ? &attention_mask : nullptr);
        hidden = add_same(ctx, hidden, attn);
        hidden = engine::modules::LayerNormModule({kHidden, kLayerNormEps, true, true})
                     .build(ctx, hidden, norm_weights(weights, "encoder.layers." + std::to_string(layer) + ".self_attn_layer_norm"));
        auto ff = feed_forward(ctx, hidden, weights, layer);
        hidden = add_same(ctx, hidden, ff);
        hidden = engine::modules::LayerNormModule({kHidden, kLayerNormEps, true, true})
                     .build(ctx, hidden, norm_weights(weights, "encoder.layers." + std::to_string(layer) + ".final_layer_norm"));
    }
    if (hidden.shape.dims[1] != raw_tokens) {
        hidden = engine::modules::SliceModule({1, 0, raw_tokens}).build(ctx, hidden);
    }
    if (v1_features) {
        hidden = engine::modules::LinearModule({kHidden, 256, true, GGML_PREC_F32})
                     .build(ctx, hidden, linear_weights(weights, "final_proj"));
    }
    return hidden;
}

void load_tensor(
    RvcHubertWeights & weights,
    const engine::assets::TensorSource & source,
    const std::string & internal,
    const std::string & source_name,
    const std::vector<int64_t> & shape,
    engine::assets::TensorStorageType storage_type) {
    (void)tensor_elements(shape);
    const auto effective_storage_type = shape.size() == 1 ? engine::assets::TensorStorageType::F32 : storage_type;
    weights.tensors.emplace(internal, weights.store->load_tensor(source, source_name, effective_storage_type, shape));
}

std::shared_ptr<RvcHubertWeights> load_weights(
    std::shared_ptr<const engine::assets::TensorSource> source,
    engine::core::BackendConfig backend,
    engine::assets::TensorStorageType storage_type) {
    if (source == nullptr) {
        throw std::runtime_error("RVC HuBERT requires a tensor source");
    }
    auto weights = std::make_shared<RvcHubertWeights>();
    weights->execution_context = std::make_shared<engine::core::ExecutionContext>(backend);
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        weights->execution_context->backend(),
        weights->execution_context->backend_type(),
        "rvc.hubert.weights",
        1024ull * 1024ull * 1024ull);
    load_tensor(*weights, *source, "feature_extractor.conv_layers.0.conv.weight", "feature_extractor.conv_layers.0.0.weight", {512, 1, 10}, storage_type);
    load_tensor(*weights, *source, "feature_extractor.conv_layers.0.layer_norm.weight", "feature_extractor.conv_layers.0.2.weight", {512}, storage_type);
    load_tensor(*weights, *source, "feature_extractor.conv_layers.0.layer_norm.bias", "feature_extractor.conv_layers.0.2.bias", {512}, storage_type);
    for (int64_t i = 1; i < 7; ++i) {
        load_tensor(
            *weights,
            *source,
            "feature_extractor.conv_layers." + std::to_string(i) + ".conv.weight",
            "feature_extractor.conv_layers." + std::to_string(i) + ".0.weight",
            {512, 512, kConvKernel[static_cast<size_t>(i)]},
            storage_type);
    }
    load_tensor(*weights, *source, "feature_projection.layer_norm.weight", "layer_norm.weight", {512}, storage_type);
    load_tensor(*weights, *source, "feature_projection.layer_norm.bias", "layer_norm.bias", {512}, storage_type);
    load_tensor(*weights, *source, "feature_projection.projection.weight", "post_extract_proj.weight", {768, 512}, storage_type);
    load_tensor(*weights, *source, "feature_projection.projection.bias", "post_extract_proj.bias", {768}, storage_type);
    weights->tensors.emplace(
        "encoder.pos_conv_embed.conv.weight",
        weights->store->make_from_f32(TensorShape::from_dims({768, 48, 128}), storage_type, effective_pos_conv_weight(*source)));
    load_tensor(*weights, *source, "encoder.pos_conv_embed.conv.bias", "encoder.pos_conv.0.bias", {768}, storage_type);
    load_tensor(*weights, *source, "encoder.layer_norm.weight", "encoder.layer_norm.weight", {768}, storage_type);
    load_tensor(*weights, *source, "encoder.layer_norm.bias", "encoder.layer_norm.bias", {768}, storage_type);
    for (int64_t layer = 0; layer < kLayers; ++layer) {
        const std::string prefix = "encoder.layers." + std::to_string(layer);
        for (const std::string proj : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
            load_tensor(*weights, *source, prefix + ".self_attn." + proj + ".weight", prefix + ".self_attn." + proj + ".weight", {768, 768}, storage_type);
            load_tensor(*weights, *source, prefix + ".self_attn." + proj + ".bias", prefix + ".self_attn." + proj + ".bias", {768}, storage_type);
        }
        load_tensor(*weights, *source, prefix + ".self_attn_layer_norm.weight", prefix + ".self_attn_layer_norm.weight", {768}, storage_type);
        load_tensor(*weights, *source, prefix + ".self_attn_layer_norm.bias", prefix + ".self_attn_layer_norm.bias", {768}, storage_type);
        load_tensor(*weights, *source, prefix + ".fc1.weight", prefix + ".fc1.weight", {3072, 768}, storage_type);
        load_tensor(*weights, *source, prefix + ".fc1.bias", prefix + ".fc1.bias", {3072}, storage_type);
        load_tensor(*weights, *source, prefix + ".fc2.weight", prefix + ".fc2.weight", {768, 3072}, storage_type);
        load_tensor(*weights, *source, prefix + ".fc2.bias", prefix + ".fc2.bias", {768}, storage_type);
        load_tensor(*weights, *source, prefix + ".final_layer_norm.weight", prefix + ".final_layer_norm.weight", {768}, storage_type);
        load_tensor(*weights, *source, prefix + ".final_layer_norm.bias", prefix + ".final_layer_norm.bias", {768}, storage_type);
    }
    load_tensor(*weights, *source, "final_proj.weight", "final_proj.weight", {256, 768}, storage_type);
    load_tensor(*weights, *source, "final_proj.bias", "final_proj.bias", {256}, storage_type);
    weights->store->upload();
    source->release_storage();
    return weights;
}

class RvcHubertRunner {
public:
    explicit RvcHubertRunner(std::shared_ptr<RvcHubertWeights> weights)
        : weights_(std::move(weights)) {}

    ~RvcHubertRunner() {
        release_graph();
    }

    RvcHubertFeatures encode(const std::vector<float> & waveform, bool v1_features) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (waveform.empty()) {
            throw std::runtime_error("RVC HuBERT input waveform is empty");
        }
        ensure_graph(static_cast<int64_t>(waveform.size()), v1_features);
        engine::core::write_tensor_f32(input_, waveform);
        if (engine::core::compute_backend_graph(weights_->execution_context->backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for RVC HuBERT");
        }
        RvcHubertFeatures out;
        out.values = engine::core::read_tensor_f32(output_.tensor);
        out.frames = output_.shape.dims[1];
        out.dim = output_.shape.dims[2];
        return out;
    }

private:
    void release_graph() {
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (ctx_ != nullptr) {
            ggml_free(ctx_);
            ctx_ = nullptr;
        }
        graph_ = nullptr;
        input_ = {};
        output_ = {};
        graph_inputs_.clear();
        samples_ = 0;
    }

    void ensure_graph(int64_t samples, bool v1_features) {
        if (ctx_ != nullptr && samples_ == samples && v1_features_ == v1_features) {
            return;
        }
        release_graph();
        ctx_ = ggml_init({1536ull * 1024ull * 1024ull, nullptr, true});
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize RVC HuBERT graph context");
        }
        engine::core::ModuleBuildContext build_ctx{ctx_, "rvc.hubert", weights_->execution_context->config().type};
        input_ = engine::core::make_tensor(build_ctx, GGML_TYPE_F32, TensorShape::from_dims({1, samples}));
        output_ = build_graph(build_ctx, input_, *weights_, v1_features, graph_inputs_);
        graph_ = ggml_new_graph_custom(ctx_, 262144, false);
        ggml_set_output(output_.tensor);
        ggml_build_forward_expand(graph_, output_.tensor);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_->execution_context->backend()));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate RVC HuBERT graph tensors");
        }
        for (const auto & graph_input : graph_inputs_) {
            engine::core::write_tensor_f32(graph_input.tensor, graph_input.values);
        }
        samples_ = samples;
        v1_features_ = v1_features;
    }

    std::shared_ptr<RvcHubertWeights> weights_;
    std::mutex mutex_;
    ggml_context * ctx_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    TensorValue input_;
    TensorValue output_;
    std::vector<RvcHubertGraphInput> graph_inputs_;
    int64_t samples_ = 0;
    bool v1_features_ = false;
};

}  // namespace

struct RvcHubertEncoder::State {
    std::shared_ptr<RvcHubertWeights> weights;
    std::unique_ptr<RvcHubertRunner> runner;
};

RvcHubertEncoder::RvcHubertEncoder(
    std::shared_ptr<const engine::assets::TensorSource> source,
    engine::core::BackendConfig backend,
    engine::assets::TensorStorageType storage_type)
    : state_(std::make_shared<State>()) {
    state_->weights = load_weights(std::move(source), std::move(backend), storage_type);
    state_->runner = std::make_unique<RvcHubertRunner>(state_->weights);
}

RvcHubertEncoder::~RvcHubertEncoder() = default;
RvcHubertEncoder::RvcHubertEncoder(RvcHubertEncoder &&) noexcept = default;
RvcHubertEncoder & RvcHubertEncoder::operator=(RvcHubertEncoder &&) noexcept = default;

RvcHubertFeatures RvcHubertEncoder::encode_16k_mono(
    const std::vector<float> & waveform_16k,
    bool v1_features) const {
    if (state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("RVC HuBERT encoder is not initialized");
    }
    return state_->runner->encode(waveform_16k, v1_features);
}

}  // namespace engine::models::rvc
