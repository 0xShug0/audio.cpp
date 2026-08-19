#include "engine/framework/modules/speech_encoders/hubert_encoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>

#include <cmath>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::modules {
namespace {

int64_t tensor_elements(const std::vector<int64_t> & shape) {
    if (shape.empty()) {
        throw std::runtime_error("HuBERT tensor shape is empty");
    }
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, [](int64_t lhs, int64_t rhs) {
        if (rhs <= 0) {
            throw std::runtime_error("HuBERT tensor shape contains non-positive dimension");
        }
        return lhs * rhs;
    });
}

void validate_config(const HubertEncoderConfig & config) {
    if (config.hidden_size <= 0 || config.intermediate_size <= 0 || config.num_hidden_layers <= 0 ||
        config.output_hidden_layer < 0 || config.num_attention_heads <= 0 || config.conv_in_channels <= 0 ||
        config.num_conv_pos_embeddings <= 0 || config.num_conv_pos_embedding_groups <= 0) {
        throw std::runtime_error("HuBERT config contains non-positive dimensions");
    }
    if (config.output_hidden_layer > config.num_hidden_layers) {
        throw std::runtime_error("HuBERT output layer cannot exceed hidden layer count");
    }
    if (config.final_projection_size < 0) {
        throw std::runtime_error("HuBERT final projection size cannot be negative");
    }
    if (config.hidden_size % config.num_attention_heads != 0 ||
        config.hidden_size % config.num_conv_pos_embedding_groups != 0) {
        throw std::runtime_error("HuBERT hidden size must be divisible by head and positional-conv group counts");
    }
    if (config.conv_dim.empty() || config.conv_dim.size() != config.conv_kernel.size() ||
        config.conv_dim.size() != config.conv_stride.size()) {
        throw std::runtime_error("HuBERT convolution config is inconsistent");
    }
    for (const int64_t value : config.conv_dim) {
        if (value <= 0) {
            throw std::runtime_error("HuBERT convolution dimensions must be positive");
        }
    }
    for (const int64_t value : config.conv_kernel) {
        if (value <= 0) {
            throw std::runtime_error("HuBERT convolution kernels must be positive");
        }
    }
    for (const int64_t value : config.conv_stride) {
        if (value <= 0) {
            throw std::runtime_error("HuBERT convolution strides must be positive");
        }
    }
}

core::TensorValue require_tensor(const HubertEncoderWeights & weights, const std::string & name) {
    const auto it = weights.tensors.find(name);
    if (it == weights.tensors.end()) {
        throw std::runtime_error("HuBERT missing tensor: " + name);
    }
    return it->second;
}

NormWeights norm_weights(const HubertEncoderWeights & weights, const std::string & prefix) {
    return NormWeights{
        require_tensor(weights, prefix + ".weight"),
        require_tensor(weights, prefix + ".bias")};
}

LinearWeights linear_weights(const HubertEncoderWeights & weights, const std::string & prefix) {
    return LinearWeights{
        require_tensor(weights, prefix + ".weight"),
        require_tensor(weights, prefix + ".bias")};
}

Conv1dWeights conv1d_weights(
    const HubertEncoderWeights & weights,
    const std::string & prefix,
    bool use_bias = true) {
    Conv1dWeights out;
    out.weight = require_tensor(weights, prefix + ".weight");
    if (use_bias) {
        out.bias = require_tensor(weights, prefix + ".bias");
    }
    return out;
}

core::TensorValue contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(ctx, value);
}

core::TensorValue transpose_bct_btc(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, value);
}

core::TensorValue add_same(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    return AddModule().build(ctx, lhs, rhs);
}

core::TensorValue scale(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    float factor) {
    return core::wrap_tensor(ggml_scale(ctx.ggml, contiguous(ctx, value).tensor, factor), value.shape, GGML_TYPE_F32);
}

core::TensorValue group_norm_affine(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NormWeights & weights,
    float eps) {
    auto input_f32 = input.type == GGML_TYPE_F32
        ? input
        : core::wrap_tensor(ggml_cast(ctx.ggml, contiguous(ctx, input).tensor, GGML_TYPE_F32), input.shape, GGML_TYPE_F32);
    auto mean = ReduceMeanModule({2}).build(ctx, input_f32);
    auto mean_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, mean.tensor, input_f32.tensor), input_f32.shape, GGML_TYPE_F32);
    auto centered = core::wrap_tensor(ggml_sub(ctx.ggml, input_f32.tensor, mean_rep.tensor), input_f32.shape, GGML_TYPE_F32);
    auto variance = ReduceMeanModule({2}).build(ctx, MulModule().build(ctx, centered, centered));
    auto stddev = core::wrap_tensor(
        ggml_sqrt(ctx.ggml, ggml_scale_bias(ctx.ggml, variance.tensor, 1.0F, eps)),
        variance.shape,
        GGML_TYPE_F32);
    auto stddev_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, stddev.tensor, input_f32.tensor), input_f32.shape, GGML_TYPE_F32);
    auto out = core::wrap_tensor(ggml_div(ctx.ggml, centered.tensor, stddev_rep.tensor), input_f32.shape, GGML_TYPE_F32);
    auto gamma = core::reshape_tensor(ctx, *weights.weight, core::TensorShape::from_dims({1, input.shape.dims[1], 1}));
    auto beta = core::reshape_tensor(ctx, *weights.bias, core::TensorShape::from_dims({1, input.shape.dims[1], 1}));
    out = core::wrap_tensor(
        ggml_mul(ctx.ggml, out.tensor, ggml_repeat(ctx.ggml, gamma.tensor, out.tensor)),
        out.shape,
        GGML_TYPE_F32);
    return core::wrap_tensor(
        ggml_add(ctx.ggml, out.tensor, ggml_repeat(ctx.ggml, beta.tensor, out.tensor)),
        out.shape,
        GGML_TYPE_F32);
}

int64_t conv1d_output_frames(int64_t input_frames, int64_t kernel, int64_t stride, int64_t padding) {
    return (input_frames + 2 * padding - kernel) / stride + 1;
}

std::vector<float> fold_weight_norm_conv1d(
    const engine::assets::TensorSource & source,
    const std::string & weight_g_name,
    const std::string & weight_v_name,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_size) {
    const auto g = source.require_f32(weight_g_name, {1, 1, kernel_size});
    const auto v = source.require_f32(weight_v_name, {out_channels, in_channels, kernel_size});
    std::vector<float> weight(v.size());
    for (int64_t k = 0; k < kernel_size; ++k) {
        double sum = 0.0;
        for (int64_t out = 0; out < out_channels; ++out) {
            for (int64_t in = 0; in < in_channels; ++in) {
                const size_t index = static_cast<size_t>((out * in_channels + in) * kernel_size + k);
                sum += static_cast<double>(v[index]) * static_cast<double>(v[index]);
            }
        }
        const double norm = std::sqrt(sum);
        if (norm == 0.0) {
            throw std::runtime_error("HuBERT positional-conv weight norm is zero");
        }
        const float scale_value = static_cast<float>(static_cast<double>(g[static_cast<size_t>(k)]) / norm);
        for (int64_t out = 0; out < out_channels; ++out) {
            for (int64_t in = 0; in < in_channels; ++in) {
                const size_t index = static_cast<size_t>((out * in_channels + in) * kernel_size + k);
                weight[index] = v[index] * scale_value;
            }
        }
    }
    return weight;
}

std::vector<float> effective_weight_norm_conv1d(
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_size) {
    // Prefer a pre-folded kernel, then fairseq legacy weight_g/weight_v, then
    // PyTorch parametrizations (original0/original1). The fold is identical
    // for the latter two layouts: per-kernel-column normalization over the
    // flattened (out, in) dims with a {1, 1, kernel} scale vector.
    if (source.has_tensor(prefix + ".weight")) {
        return source.require_f32(prefix + ".weight", {out_channels, in_channels, kernel_size});
    }
    if (source.has_tensor(prefix + ".weight_g") && source.has_tensor(prefix + ".weight_v")) {
        return fold_weight_norm_conv1d(
            source,
            prefix + ".weight_g",
            prefix + ".weight_v",
            out_channels,
            in_channels,
            kernel_size);
    }
    if (source.has_tensor(prefix + ".parametrizations.weight.original0") &&
        source.has_tensor(prefix + ".parametrizations.weight.original1")) {
        return fold_weight_norm_conv1d(
            source,
            prefix + ".parametrizations.weight.original0",
            prefix + ".parametrizations.weight.original1",
            out_channels,
            in_channels,
            kernel_size);
    }
    throw std::runtime_error(
        "HuBERT positional-conv weights missing: expected " + prefix +
        ".weight, .weight_g/.weight_v, or .parametrizations.weight.original0/.original1");
}

std::string join_name(const std::string & lhs, const std::string & rhs) {
    if (lhs.empty()) {
        return rhs;
    }
    if (rhs.empty()) {
        return lhs;
    }
    return lhs + "." + rhs;
}

std::string indexed_prefix(const std::string & prefix, int64_t index) {
    return join_name(prefix, std::to_string(index));
}

core::TensorValue load_tensor_alias(
    HubertEncoderWeights & weights,
    const engine::assets::TensorSource & source,
    const std::string & source_name,
    engine::assets::TensorStorageType storage_type,
    const std::vector<int64_t> & shape) {
    weights.parameter_count += tensor_elements(shape);
    ++weights.loaded_tensor_count;
    return storage_type == engine::assets::TensorStorageType::F32
        ? weights.store->load_f32_tensor(source, source_name, shape)
        : weights.store->load_tensor(source, source_name, storage_type, shape);
}

void put_tensor_alias(
    HubertEncoderWeights & weights,
    const engine::assets::TensorSource & source,
    const std::string & logical_name,
    const std::string & source_name,
    engine::assets::TensorStorageType storage_type,
    const std::vector<int64_t> & shape) {
    weights.tensors.emplace(
        logical_name,
        load_tensor_alias(weights, source, source_name, storage_type, shape));
}

void put_f32_tensor_alias(
    HubertEncoderWeights & weights,
    const engine::assets::TensorSource & source,
    const std::string & logical_name,
    const std::string & source_name,
    const std::vector<int64_t> & shape) {
    put_tensor_alias(weights, source, logical_name, source_name, engine::assets::TensorStorageType::F32, shape);
}

core::TensorValue grouped_pos_conv(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const Conv1dWeights & weights,
    const HubertEncoderConfig & config) {
    const int64_t groups = config.num_conv_pos_embedding_groups;
    const int64_t channels_per_group = config.hidden_size / groups;
    const auto input_contiguous = contiguous(ctx, input_bct);
    core::TensorValue out;
    for (int64_t group = 0; group < groups; ++group) {
        auto input_group = SliceModule({1, group * channels_per_group, channels_per_group}).build(ctx, input_contiguous);
        auto weight_group = SliceModule({0, group * channels_per_group, channels_per_group}).build(ctx, weights.weight);
        Conv1dWeights group_weights{weight_group, std::nullopt};
        if (weights.bias.has_value()) {
            group_weights.bias = SliceModule({0, group * channels_per_group, channels_per_group}).build(ctx, *weights.bias);
        }
        auto group_out = Conv1dModule({
            channels_per_group,
            channels_per_group,
            config.num_conv_pos_embeddings,
            1,
            static_cast<int>(config.num_conv_pos_embeddings / 2),
            1,
            weights.bias.has_value()}).build(ctx, input_group, group_weights);
        out = out.valid() ? ConcatModule({1}).build(ctx, out, group_out) : group_out;
    }
    if (config.num_conv_pos_embeddings % 2 == 0) {
        out = SliceModule({2, 0, input_bct.shape.dims[2]}).build(ctx, out);
    }
    return GeluModule({GeluApproximation::ExactErf}).build(ctx, out);
}

core::TensorValue build_self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden_btc,
    const HubertEncoderWeights & weights,
    int64_t layer_index,
    const core::TensorValue * attention_mask) {
    const auto & config = weights.config;
    const int64_t head_dim = config.hidden_size / config.num_attention_heads;
    const std::string prefix = "encoder.layers." + std::to_string(layer_index) + ".attention";
    auto q = LinearModule({config.hidden_size, config.hidden_size, true, GGML_PREC_F32})
                 .build(ctx, hidden_btc, linear_weights(weights, prefix + ".q_proj"));
    auto k = LinearModule({config.hidden_size, config.hidden_size, true, GGML_PREC_F32})
                 .build(ctx, hidden_btc, linear_weights(weights, prefix + ".k_proj"));
    auto v = LinearModule({config.hidden_size, config.hidden_size, true, GGML_PREC_F32})
                 .build(ctx, hidden_btc, linear_weights(weights, prefix + ".v_proj"));
    q = core::reshape_tensor(
        ctx,
        contiguous(ctx, q),
        core::TensorShape::from_dims({hidden_btc.shape.dims[0], hidden_btc.shape.dims[1], config.num_attention_heads, head_dim}));
    k = core::reshape_tensor(
        ctx,
        contiguous(ctx, k),
        core::TensorShape::from_dims({hidden_btc.shape.dims[0], hidden_btc.shape.dims[1], config.num_attention_heads, head_dim}));
    v = core::reshape_tensor(
        ctx,
        contiguous(ctx, v),
        core::TensorShape::from_dims({hidden_btc.shape.dims[0], hidden_btc.shape.dims[1], config.num_attention_heads, head_dim}));
    q = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    k = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
    v = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
    const auto k_t = TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k);
    auto scores = MatMulModule().build(ctx, q, k_t);
    scores = scale(ctx, scores, static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim))));
    if (attention_mask != nullptr) {
        auto repeated_mask = core::wrap_tensor(
            ggml_repeat(ctx.ggml, attention_mask->tensor, scores.tensor),
            scores.shape,
            GGML_TYPE_F32);
        scores = AddModule{}.build(ctx, scores, repeated_mask);
    }
    auto attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, contiguous(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
    auto context = MatMulModule().build(ctx, attn, v);
    context = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, context);
    context = core::reshape_tensor(
        ctx,
        contiguous(ctx, context),
        core::TensorShape::from_dims({hidden_btc.shape.dims[0], hidden_btc.shape.dims[1], config.hidden_size}));
    return LinearModule({config.hidden_size, config.hidden_size, true, GGML_PREC_F32})
        .build(ctx, context, linear_weights(weights, prefix + ".out_proj"));
}

core::TensorValue build_feed_forward(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden_btc,
    const HubertEncoderWeights & weights,
    int64_t layer_index) {
    const auto & config = weights.config;
    const std::string prefix = "encoder.layers." + std::to_string(layer_index) + ".feed_forward";
    auto x = LinearModule({config.hidden_size, config.intermediate_size, true, GGML_PREC_F32})
                 .build(ctx, hidden_btc, linear_weights(weights, prefix + ".intermediate_dense"));
    x = GeluModule({GeluApproximation::ExactErf}).build(ctx, x);
    return LinearModule({config.intermediate_size, config.hidden_size, true, GGML_PREC_F32})
        .build(ctx, x, linear_weights(weights, prefix + ".output_dense"));
}

core::TensorValue build_hubert_graph(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_values,
    const HubertEncoderWeights & weights,
    const HubertEncoderRunConfig & run_config,
    std::vector<std::pair<core::TensorValue, std::vector<float>>> & graph_inputs) {
    const auto & config = weights.config;
    const int64_t output_hidden_layer = run_config.output_hidden_layer >= 0
        ? run_config.output_hidden_layer
        : config.output_hidden_layer;
    if (output_hidden_layer < 0) {
        throw std::runtime_error("HuBERT run output layer cannot be negative");
    }
    if (output_hidden_layer > config.num_hidden_layers) {
        throw std::runtime_error("HuBERT run output layer cannot exceed hidden layer count");
    }
    if (run_config.apply_final_projection && config.final_projection_size <= 0) {
        throw std::runtime_error("HuBERT run requested final projection but no projection is configured");
    }
    auto hidden = core::reshape_tensor(
        ctx,
        input_values,
        core::TensorShape::from_dims({input_values.shape.dims[0], 1, input_values.shape.dims[1]}));
    int64_t in_channels = 1;
    int64_t frames = input_values.shape.dims[1];
    for (size_t index = 0; index < config.conv_dim.size(); ++index) {
        const std::string prefix = "feature_extractor.conv_layers." + std::to_string(index);
        const bool use_conv_bias = config.feature_extractor_norm == HubertFeatureExtractorNorm::LayerNormEveryLayer;
        hidden = Conv1dModule({
            in_channels,
            config.conv_dim[index],
            config.conv_kernel[index],
            static_cast<int>(config.conv_stride[index]),
            0,
            1,
            use_conv_bias}).build(ctx, hidden, conv1d_weights(weights, prefix + ".conv", use_conv_bias));
        if (config.feature_extractor_norm == HubertFeatureExtractorNorm::LayerNormEveryLayer) {
            hidden = transpose_bct_btc(ctx, hidden);
            hidden = LayerNormModule({config.conv_dim[index], config.layer_norm_eps, true, true})
                         .build(ctx, hidden, norm_weights(weights, prefix + ".layer_norm"));
            hidden = transpose_bct_btc(ctx, hidden);
        } else if (index == 0) {
            hidden = group_norm_affine(ctx, hidden, norm_weights(weights, prefix + ".layer_norm"), config.layer_norm_eps);
        }
        hidden = GeluModule({GeluApproximation::ExactErf}).build(ctx, hidden);
        frames = conv1d_output_frames(frames, config.conv_kernel[index], config.conv_stride[index], 0);
        in_channels = config.conv_dim[index];
    }
    hidden = transpose_bct_btc(ctx, hidden);
    hidden = LayerNormModule({config.conv_dim.back(), config.layer_norm_eps, true, true})
                 .build(ctx, hidden, norm_weights(weights, "feature_projection.layer_norm"));
    hidden = LinearModule({config.conv_dim.back(), config.hidden_size, true, GGML_PREC_F32})
                 .build(ctx, hidden, linear_weights(weights, "feature_projection.projection"));

    if (config.apply_positional_embedding) {
        auto pos = grouped_pos_conv(
            ctx,
            transpose_bct_btc(ctx, hidden),
            conv1d_weights(weights, "encoder.pos_conv_embed.conv"),
            config);
        pos = transpose_bct_btc(ctx, pos);
        hidden = add_same(ctx, hidden, pos);
    }

    if (config.apply_encoder_input_layer_norm) {
        hidden = LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                     .build(ctx, hidden, norm_weights(weights, "encoder.layer_norm"));
    }

    const int64_t raw_tokens = hidden.shape.dims[1];
    core::TensorValue attention_mask;
    if (config.pad_odd_tokens_with_attention_mask && raw_tokens % 2 != 0) {
        hidden = PaddingModule({raw_tokens + 1}).build(ctx, hidden);
        attention_mask = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, 1, 1, hidden.shape.dims[1]}));
        std::vector<float> mask(static_cast<size_t>(hidden.shape.dims[1]), 0.0F);
        mask.back() = -1.0e4F;
        graph_inputs.push_back({attention_mask, std::move(mask)});
    }

    for (int64_t layer = 0; layer < output_hidden_layer; ++layer) {
        const std::string prefix = "encoder.layers." + std::to_string(layer);
        if (config.encoder_layer_norm_order == HubertEncoderLayerNormOrder::PreNorm) {
            const auto attn_residual = hidden;
            hidden = LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                         .build(ctx, hidden, norm_weights(weights, prefix + ".layer_norm"));
            hidden = build_self_attention(ctx, hidden, weights, layer, attention_mask.valid() ? &attention_mask : nullptr);
            hidden = add_same(ctx, attn_residual, hidden);
            const auto ff_in = LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                                   .build(ctx, hidden, norm_weights(weights, prefix + ".final_layer_norm"));
            hidden = add_same(ctx, hidden, build_feed_forward(ctx, ff_in, weights, layer));
        } else {
            auto attn = build_self_attention(ctx, hidden, weights, layer, attention_mask.valid() ? &attention_mask : nullptr);
            hidden = add_same(ctx, hidden, attn);
            hidden = LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                         .build(ctx, hidden, norm_weights(weights, prefix + ".self_attn_layer_norm"));
            auto ff = build_feed_forward(ctx, hidden, weights, layer);
            hidden = add_same(ctx, hidden, ff);
            hidden = LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                         .build(ctx, hidden, norm_weights(weights, prefix + ".final_layer_norm"));
        }
    }
    if (hidden.shape.dims[1] != raw_tokens) {
        hidden = SliceModule({1, 0, raw_tokens}).build(ctx, hidden);
    }
    if (config.apply_final_layer_norm) {
        hidden = LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                     .build(ctx, hidden, norm_weights(weights, "encoder.layer_norm"));
    }
    if (run_config.apply_final_projection) {
        hidden = LinearModule({config.hidden_size, config.final_projection_size, true, GGML_PREC_F32})
                     .build(ctx, hidden, linear_weights(weights, "final_proj"));
    }
    return hidden;
}

class HubertRunner {
public:
    explicit HubertRunner(std::shared_ptr<const HubertEncoderWeights> weights)
        : weights_(std::move(weights)) {
        if (weights_ == nullptr || weights_->execution_context == nullptr) {
            throw std::runtime_error("HuBERT runner requires weights and execution context");
        }
    }

    ~HubertRunner() {
        release_graph();
    }

    HubertEncoderOutput encode(
        const std::vector<float> & input_values,
        int64_t batch,
        int64_t samples,
        HubertEncoderRunConfig run_config) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (batch != 1) {
            throw std::runtime_error("HuBERT encoder currently requires batch size 1");
        }
        if (samples <= 0 || static_cast<int64_t>(input_values.size()) != batch * samples) {
            throw std::runtime_error("HuBERT encoder input size mismatch");
        }
        ensure_graph(batch, samples, run_config);
        core::write_tensor_f32(input_, input_values);
        for (const auto & graph_input : graph_inputs_) {
            core::write_tensor_f32(graph_input.first, graph_input.second);
        }
        if (engine::core::compute_backend_graph(weights_->execution_context->backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for HuBERT encoder");
        }
        HubertEncoderOutput out;
        out.hidden_states = core::read_tensor_f32(output_.tensor);
        out.batch = batch;
        out.tokens = output_.shape.dims[1];
        out.hidden_size = output_.shape.dims[2];
        return out;
    }

    void release_runtime_graph() {
        std::lock_guard<std::mutex> lock(mutex_);
        release_graph();
    }

private:
    void release_graph() {
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
            ggml_ = nullptr;
        }
        graph_ = nullptr;
        input_ = {};
        output_ = {};
        graph_inputs_.clear();
        batch_ = 0;
        samples_ = 0;
        output_hidden_layer_ = -1;
        apply_final_projection_ = false;
    }

    void ensure_graph(int64_t batch, int64_t samples, const HubertEncoderRunConfig & run_config) {
        const int64_t output_hidden_layer = run_config.output_hidden_layer >= 0
            ? run_config.output_hidden_layer
            : weights_->config.output_hidden_layer;
        if (ggml_ != nullptr &&
            batch_ == batch &&
            samples_ == samples &&
            output_hidden_layer_ == output_hidden_layer &&
            apply_final_projection_ == run_config.apply_final_projection) {
            return;
        }
        release_graph();
        ggml_init_params params{
            1024ull * 1024ull * 1024ull,
            nullptr,
            true,
        };
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize HuBERT graph context");
        }
        core::ModuleBuildContext ctx{
            ggml_,
            "framework.hubert.encode",
            weights_->execution_context->config().type};
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, samples}));
        output_ = build_hubert_graph(ctx, input_, *weights_, run_config, graph_inputs_);
        graph_ = ggml_new_graph_custom(ggml_, 131072, false);
        ggml_set_output(output_.tensor);
        ggml_build_forward_expand(graph_, output_.tensor);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_->execution_context->backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate HuBERT graph tensors");
        }
        batch_ = batch;
        samples_ = samples;
        output_hidden_layer_ = output_hidden_layer;
        apply_final_projection_ = run_config.apply_final_projection;
    }

    std::shared_ptr<const HubertEncoderWeights> weights_;
    std::mutex mutex_;
    ggml_context * ggml_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    core::TensorValue input_;
    core::TensorValue output_;
    std::vector<std::pair<core::TensorValue, std::vector<float>>> graph_inputs_;
    int64_t batch_ = 0;
    int64_t samples_ = 0;
    int64_t output_hidden_layer_ = -1;
    bool apply_final_projection_ = false;
};

}  // namespace

HubertEncoderComponent HubertEncoderComponent::load_from_safetensors(
    const std::filesystem::path & checkpoint_path,
    core::BackendConfig backend,
    HubertEncoderConfig config) {
    const auto source = engine::assets::open_tensor_source(checkpoint_path);
    return load_from_tensor_source(std::move(source), std::move(backend), std::move(config));
}

HubertEncoderComponent HubertEncoderComponent::load_from_tensor_source(
    std::shared_ptr<const engine::assets::TensorSource> source,
    core::BackendConfig backend,
    HubertEncoderConfig config) {
    return load_from_tensor_source(
        std::move(source),
        std::move(backend),
        std::move(config),
        HubertEncoderWeightBinding{});
}

HubertEncoderComponent HubertEncoderComponent::load_from_tensor_source(
    std::shared_ptr<const engine::assets::TensorSource> source,
    core::BackendConfig backend,
    HubertEncoderConfig config,
    HubertEncoderWeightBinding binding) {
    if (source == nullptr) {
        throw std::runtime_error("HuBERT tensor source is missing");
    }
    validate_config(config);
    auto weights = std::make_shared<HubertEncoderWeights>();
    weights->config = std::move(config);
    weights->source_path = source->source_path();
    weights->execution_context = std::make_shared<core::ExecutionContext>(backend);
    weights->store = std::make_shared<core::BackendWeightStore>(
        weights->execution_context->backend(),
        weights->execution_context->backend_type(),
        "framework.hubert.weights",
        1024ull * 1024ull * 1024ull);

    const auto & config_ref = weights->config;
    weights->tensors.reserve(static_cast<size_t>(config_ref.num_hidden_layers * 12 + config_ref.conv_dim.size() * 3 + 8));
    for (int64_t index = 0; index < static_cast<int64_t>(config_ref.conv_dim.size()); ++index) {
        const auto logical_layer = indexed_prefix("feature_extractor.conv_layers", index);
        const auto source_layer = indexed_prefix(binding.feature_extractor_layers, index);
        const int64_t in_channels = index == 0 ? config_ref.conv_in_channels : config_ref.conv_dim[static_cast<size_t>(index - 1)];
        put_tensor_alias(
            *weights,
            *source,
            join_name(logical_layer, "conv.weight"),
            join_name(join_name(source_layer, binding.feature_extractor_conv), "weight"),
            binding.conv_storage_type,
            {config_ref.conv_dim[static_cast<size_t>(index)], in_channels, config_ref.conv_kernel[static_cast<size_t>(index)]});
        if (config_ref.feature_extractor_norm == HubertFeatureExtractorNorm::LayerNormEveryLayer) {
            put_f32_tensor_alias(
                *weights,
                *source,
                join_name(logical_layer, "conv.bias"),
                join_name(join_name(source_layer, binding.feature_extractor_conv), "bias"),
                {config_ref.conv_dim[static_cast<size_t>(index)]});
        }
        if (config_ref.feature_extractor_norm == HubertFeatureExtractorNorm::LayerNormEveryLayer ||
            index == 0) {
            put_f32_tensor_alias(
                *weights,
                *source,
                join_name(logical_layer, "layer_norm.weight"),
                join_name(join_name(source_layer, binding.feature_extractor_layer_norm), "weight"),
                {config_ref.conv_dim[static_cast<size_t>(index)]});
            put_f32_tensor_alias(
                *weights,
                *source,
                join_name(logical_layer, "layer_norm.bias"),
                join_name(join_name(source_layer, binding.feature_extractor_layer_norm), "bias"),
                {config_ref.conv_dim[static_cast<size_t>(index)]});
        }
    }
    put_f32_tensor_alias(
        *weights,
        *source,
        "feature_projection.layer_norm.weight",
        join_name(binding.feature_projection_layer_norm, "weight"),
        {config_ref.conv_dim.back()});
    put_f32_tensor_alias(
        *weights,
        *source,
        "feature_projection.layer_norm.bias",
        join_name(binding.feature_projection_layer_norm, "bias"),
        {config_ref.conv_dim.back()});
    put_tensor_alias(
        *weights,
        *source,
        "feature_projection.projection.weight",
        join_name(binding.feature_projection_projection, "weight"),
        binding.projection_storage_type,
        {config_ref.hidden_size, config_ref.conv_dim.back()});
    put_f32_tensor_alias(
        *weights,
        *source,
        "feature_projection.projection.bias",
        join_name(binding.feature_projection_projection, "bias"),
        {config_ref.hidden_size});
    weights->tensors.emplace(
        "encoder.pos_conv_embed.conv.weight",
        weights->store->make_from_f32(
            core::TensorShape::from_dims({
                config_ref.hidden_size,
                config_ref.hidden_size / config_ref.num_conv_pos_embedding_groups,
                config_ref.num_conv_pos_embeddings}),
            binding.positional_conv_storage_type,
            effective_weight_norm_conv1d(
                *source,
                binding.positional_conv,
                config_ref.hidden_size,
                config_ref.hidden_size / config_ref.num_conv_pos_embedding_groups,
                config_ref.num_conv_pos_embeddings)));
    weights->parameter_count += config_ref.hidden_size *
        (config_ref.hidden_size / config_ref.num_conv_pos_embedding_groups) *
        config_ref.num_conv_pos_embeddings;
    ++weights->loaded_tensor_count;
    put_f32_tensor_alias(
        *weights,
        *source,
        "encoder.pos_conv_embed.conv.bias",
        join_name(binding.positional_conv, "bias"),
        {config_ref.hidden_size});
    if (config_ref.apply_encoder_input_layer_norm || config_ref.apply_final_layer_norm) {
        put_f32_tensor_alias(
            *weights,
            *source,
            "encoder.layer_norm.weight",
            join_name(binding.encoder_layer_norm, "weight"),
            {config_ref.hidden_size});
        put_f32_tensor_alias(
            *weights,
            *source,
            "encoder.layer_norm.bias",
            join_name(binding.encoder_layer_norm, "bias"),
            {config_ref.hidden_size});
    }
    for (int64_t layer = 0; layer < config_ref.num_hidden_layers; ++layer) {
        const auto logical_layer = indexed_prefix("encoder.layers", layer);
        const auto source_layer = indexed_prefix(binding.encoder_layers, layer);
        const auto logical_attention = join_name(logical_layer, "attention");
        const auto source_attention = join_name(source_layer, binding.layer.attention);
        for (const std::string proj : {"q_proj", "k_proj", "v_proj", "out_proj"}) {
            put_tensor_alias(
                *weights,
                *source,
                join_name(join_name(logical_attention, proj), "weight"),
                join_name(join_name(source_attention, proj), "weight"),
                binding.attention_storage_type,
                {config_ref.hidden_size, config_ref.hidden_size});
            put_f32_tensor_alias(
                *weights,
                *source,
                join_name(join_name(logical_attention, proj), "bias"),
                join_name(join_name(source_attention, proj), "bias"),
                {config_ref.hidden_size});
        }
        if (config_ref.encoder_layer_norm_order == HubertEncoderLayerNormOrder::PreNorm) {
            put_f32_tensor_alias(
                *weights,
                *source,
                join_name(logical_layer, "layer_norm.weight"),
                join_name(join_name(source_layer, binding.layer.pre_attention_layer_norm), "weight"),
                {config_ref.hidden_size});
            put_f32_tensor_alias(
                *weights,
                *source,
                join_name(logical_layer, "layer_norm.bias"),
                join_name(join_name(source_layer, binding.layer.pre_attention_layer_norm), "bias"),
                {config_ref.hidden_size});
        } else {
            put_f32_tensor_alias(
                *weights,
                *source,
                join_name(logical_layer, "self_attn_layer_norm.weight"),
                join_name(join_name(source_layer, binding.layer.post_attention_layer_norm), "weight"),
                {config_ref.hidden_size});
            put_f32_tensor_alias(
                *weights,
                *source,
                join_name(logical_layer, "self_attn_layer_norm.bias"),
                join_name(join_name(source_layer, binding.layer.post_attention_layer_norm), "bias"),
                {config_ref.hidden_size});
        }
        put_tensor_alias(
            *weights,
            *source,
            join_name(logical_layer, "feed_forward.intermediate_dense.weight"),
            join_name(join_name(source_layer, binding.layer.feed_forward_intermediate), "weight"),
            binding.feed_forward_storage_type,
            {config_ref.intermediate_size, config_ref.hidden_size});
        put_f32_tensor_alias(
            *weights,
            *source,
            join_name(logical_layer, "feed_forward.intermediate_dense.bias"),
            join_name(join_name(source_layer, binding.layer.feed_forward_intermediate), "bias"),
            {config_ref.intermediate_size});
        put_tensor_alias(
            *weights,
            *source,
            join_name(logical_layer, "feed_forward.output_dense.weight"),
            join_name(join_name(source_layer, binding.layer.feed_forward_output), "weight"),
            binding.feed_forward_storage_type,
            {config_ref.hidden_size, config_ref.intermediate_size});
        put_f32_tensor_alias(
            *weights,
            *source,
            join_name(logical_layer, "feed_forward.output_dense.bias"),
            join_name(join_name(source_layer, binding.layer.feed_forward_output), "bias"),
            {config_ref.hidden_size});
        put_f32_tensor_alias(
            *weights,
            *source,
            join_name(logical_layer, "final_layer_norm.weight"),
            join_name(join_name(source_layer, binding.layer.final_layer_norm), "weight"),
            {config_ref.hidden_size});
        put_f32_tensor_alias(
            *weights,
            *source,
            join_name(logical_layer, "final_layer_norm.bias"),
            join_name(join_name(source_layer, binding.layer.final_layer_norm), "bias"),
            {config_ref.hidden_size});
    }
    if (config_ref.final_projection_size > 0) {
        put_tensor_alias(
            *weights,
            *source,
            "final_proj.weight",
            join_name(binding.final_projection, "weight"),
            binding.final_projection_storage_type,
            {config_ref.final_projection_size, config_ref.hidden_size});
        put_f32_tensor_alias(
            *weights,
            *source,
            "final_proj.bias",
            join_name(binding.final_projection, "bias"),
            {config_ref.final_projection_size});
    }
    if (weights->loaded_tensor_count <= 0) {
        throw std::runtime_error("HuBERT tensor source contains no loaded tensors");
    }
    weights->store->upload();
    source->release_storage();
    return HubertEncoderComponent(std::move(weights), backend);
}

struct HubertEncoderComponent::State {
    std::unique_ptr<HubertRunner> runner;
};

HubertEncoderComponent::HubertEncoderComponent(
    std::shared_ptr<const HubertEncoderWeights> weights,
    core::BackendConfig backend)
    : weights_(std::move(weights)),
      backend_(backend),
      state_(std::make_shared<State>()) {
    if (weights_ == nullptr) {
        throw std::runtime_error("HuBERT component requires weights");
    }
    state_->runner = std::make_unique<HubertRunner>(weights_);
}

const core::BackendConfig & HubertEncoderComponent::backend() const noexcept {
    return backend_;
}

const std::shared_ptr<const HubertEncoderWeights> & HubertEncoderComponent::weights() const noexcept {
    return weights_;
}

int64_t HubertEncoderComponent::hidden_size() const noexcept {
    return weights_ == nullptr ? 0 : weights_->config.hidden_size;
}

int64_t HubertEncoderComponent::loaded_tensor_count() const noexcept {
    return weights_ == nullptr ? 0 : weights_->loaded_tensor_count;
}

int64_t HubertEncoderComponent::parameter_count() const noexcept {
    return weights_ == nullptr ? 0 : weights_->parameter_count;
}

HubertEncoderOutput HubertEncoderComponent::encode(
    const std::vector<float> & input_values,
    int64_t batch,
    int64_t samples) const {
    return encode(input_values, batch, samples, {});
}

HubertEncoderOutput HubertEncoderComponent::encode(
    const std::vector<float> & input_values,
    int64_t batch,
    int64_t samples,
    HubertEncoderRunConfig run_config) const {
    if (state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("HuBERT component is not initialized");
    }
    return state_->runner->encode(input_values, batch, samples, run_config);
}

void HubertEncoderComponent::release_runtime_graph() {
    if (state_ != nullptr && state_->runner != nullptr) {
        state_->runner->release_runtime_graph();
    }
}

}  // namespace engine::modules
