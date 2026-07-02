#include "engine/models/chatterbox/s3gen_flow.h"
#include "components/s3gen_weights.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/debug/profiler.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/deferred_tensor_writer.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>

namespace engine::models::chatterbox {
namespace {

S3FlowEncoderWeights::LayerNormWeights load_flow_layer_norm(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    S3FlowEncoderWeights::LayerNormWeights weights;
    weights.weight_tensor = store.load_f32_tensor(source, prefix + ".weight", {channels});
    weights.bias_tensor = store.load_f32_tensor(source, prefix + ".bias", {channels});
    return weights;
}

S3FlowEncoderWeights::LinearWeights load_flow_linear(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_features,
    int64_t in_features,
    bool use_bias,
    engine::assets::TensorStorageType weight_storage_type) {
    S3FlowEncoderWeights::LinearWeights linear;
    linear.out_features = out_features;
    linear.in_features = in_features;
    linear.use_bias = use_bias;
    linear.weight_tensor = store.load_tensor(source, prefix + ".weight", weight_storage_type, {out_features, in_features});
    if (use_bias) {
        linear.bias_tensor = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return linear;
}

S3FlowEncoderWeights::Conv1dWeights load_flow_conv1d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    engine::assets::TensorStorageType weight_storage_type) {
    S3FlowEncoderWeights::Conv1dWeights conv;
    conv.out_channels = out_channels;
    conv.in_channels = in_channels;
    conv.kernel = kernel;
    conv.stride = stride;
    conv.padding = padding;
    conv.weight_tensor = store.load_tensor(source, prefix + ".weight", weight_storage_type, {out_channels, in_channels, kernel});
    conv.bias_tensor = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    return conv;
}

S3FlowDecoderWeights::LayerNormWeights load_decoder_layer_norm(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    S3FlowDecoderWeights::LayerNormWeights weights;
    weights.weight_tensor = store.load_f32_tensor(source, prefix + ".weight", {channels});
    weights.bias_tensor = store.load_f32_tensor(source, prefix + ".bias", {channels});
    return weights;
}

S3FlowDecoderWeights::LinearWeights load_decoder_linear(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_features,
    int64_t in_features,
    bool use_bias,
    engine::assets::TensorStorageType weight_storage_type) {
    S3FlowDecoderWeights::LinearWeights linear;
    linear.out_features = out_features;
    linear.in_features = in_features;
    linear.use_bias = use_bias;
    linear.weight_tensor = store.load_tensor(source, prefix + ".weight", weight_storage_type, {out_features, in_features});
    if (use_bias) {
        linear.bias_tensor = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return linear;
}

S3FlowDecoderWeights::Conv1dWeights load_decoder_conv1d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t stride,
    bool use_bias,
    engine::assets::TensorStorageType weight_storage_type) {
    S3FlowDecoderWeights::Conv1dWeights conv;
    conv.out_channels = out_channels;
    conv.in_channels = in_channels;
    conv.kernel = kernel;
    conv.stride = stride;
    conv.use_bias = use_bias;
    conv.weight_tensor = store.load_tensor(source, prefix + ".weight", weight_storage_type, {out_channels, in_channels, kernel});
    if (use_bias) {
        conv.bias_tensor = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    }
    return conv;
}

std::vector<float> decoder_sinusoidal_pos_emb(
    const std::vector<float> & timesteps,
    int64_t batch,
    int64_t dim,
    float scale = 1000.0f) {
    if ((dim % 2) != 0) {
        throw std::runtime_error("decoder sinusoidal embedding requires even dim");
    }
    const int64_t half_dim = dim / 2;
    std::vector<float> out(static_cast<size_t>(batch * dim), 0.0f);
    const double emb_scale = std::log(10000.0) / static_cast<double>(half_dim - 1);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t i = 0; i < half_dim; ++i) {
            const double freq = std::exp(-static_cast<double>(i) * emb_scale);
            const double angle = static_cast<double>(scale) * static_cast<double>(timesteps[static_cast<size_t>(b)]) * freq;
            out[static_cast<size_t>(b * dim + i)] = static_cast<float>(std::sin(angle));
            out[static_cast<size_t>(b * dim + half_dim + i)] = static_cast<float>(std::cos(angle));
        }
    }
    return out;
}

std::vector<float> flow_relative_positional_encoding(int64_t frames, int64_t channels) {
    const int64_t rel_frames = (frames * 2) - 1;
    std::vector<float> positive(static_cast<size_t>(frames * channels), 0.0f);
    std::vector<float> negative(static_cast<size_t>(frames * channels), 0.0f);
    for (int64_t pos = 0; pos < frames; ++pos) {
        for (int64_t i = 0; i < channels; i += 2) {
            const double div = std::exp(-std::log(10000.0) * static_cast<double>(i) / static_cast<double>(channels));
            positive[static_cast<size_t>(pos * channels + i)] = static_cast<float>(std::sin(static_cast<double>(pos) * div));
            negative[static_cast<size_t>(pos * channels + i)] = static_cast<float>(std::sin(-static_cast<double>(pos) * div));
            if (i + 1 < channels) {
                positive[static_cast<size_t>(pos * channels + i + 1)] = static_cast<float>(std::cos(static_cast<double>(pos) * div));
                negative[static_cast<size_t>(pos * channels + i + 1)] = static_cast<float>(std::cos(-static_cast<double>(pos) * div));
            }
        }
    }
    std::vector<float> out(static_cast<size_t>(rel_frames * channels), 0.0f);
    for (int64_t pos = 0; pos < frames; ++pos) {
        std::copy_n(
            positive.data() + static_cast<std::ptrdiff_t>((frames - 1 - pos) * channels),
            channels,
            out.data() + static_cast<std::ptrdiff_t>(pos * channels));
    }
    for (int64_t pos = 1; pos < frames; ++pos) {
        std::copy_n(
            negative.data() + static_cast<std::ptrdiff_t>(pos * channels),
            channels,
            out.data() + static_cast<std::ptrdiff_t>((frames - 1 + pos) * channels));
    }
    return out;
}

bool same_backend(const engine::core::BackendConfig & lhs, const engine::core::BackendConfig & rhs) {
    return lhs.type == rhs.type && lhs.device == rhs.device && lhs.threads == rhs.threads;
}

using Clock = std::chrono::steady_clock;

void copy_backend_tensor(
    const engine::core::TensorValue & src,
    const engine::core::TensorValue & dst) {
    ggml_backend_tensor_copy(src.tensor, dst.tensor);
}

void release_graph_resources(
    const engine::core::ExecutionContext * execution_context,
    ggml_cgraph *& graph,
    ggml_context *& ggml,
    ggml_backend_buffer_t & buffer) {
    if (execution_context != nullptr && graph != nullptr) {
        engine::core::release_backend_graph_resources(execution_context->backend(), graph);
        graph = nullptr;
    }
    if (buffer != nullptr) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (ggml != nullptr) {
        ggml_free(ggml);
        ggml = nullptr;
    }
}

int64_t valid_frames_from_mask(const std::vector<float> & mask, int64_t batch, int64_t frames) {
    if (batch <= 0 || frames <= 0) {
        throw std::runtime_error("S3 valid_frames_from_mask requires positive batch and frames");
    }
    if (static_cast<int64_t>(mask.size()) != batch * frames) {
        throw std::runtime_error("S3 mask size mismatch");
    }
    int64_t valid = 0;
    for (int64_t frame = 0; frame < frames; ++frame) {
        const float value = mask[static_cast<size_t>(frame)];
        if (value >= 0.5f) {
            ++valid;
            continue;
        }
        break;
    }
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (int64_t frame = 0; frame < frames; ++frame) {
            const float value = mask[static_cast<size_t>(batch_index * frames + frame)];
            const float expected = frame < valid ? 1.0f : 0.0f;
            if (std::fabs(value - expected) > 1.0e-6f) {
                throw std::runtime_error("S3 mask must be a shared prefix-ones suffix-zeros layout");
            }
        }
    }
    return valid;
}

std::vector<float> make_attention_key_mask(
    int64_t batch,
    int64_t query_frames,
    int64_t key_frames,
    int64_t valid_key_frames,
    int64_t heads = 1) {
    std::vector<float> values(static_cast<size_t>(batch * heads * query_frames * key_frames), 0.0f);
    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
        for (int64_t head = 0; head < heads; ++head) {
            for (int64_t query_frame = 0; query_frame < query_frames; ++query_frame) {
                for (int64_t key_frame = valid_key_frames; key_frame < key_frames; ++key_frame) {
                    values[static_cast<size_t>(((batch_index * heads + head) * query_frames + query_frame) * key_frames + key_frame)] = -10000.0f;
                }
            }
        }
    }
    return values;
}

engine::core::TensorValue contiguous(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    return engine::core::ensure_backend_addressable_layout(ctx, input);
}

engine::core::TensorValue make_f32_graph_constant(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorShape & shape,
    const std::vector<float> & values,
    engine::core::DeferredTensorWriter & writer) {
    auto tensor = writer.make_f32_tensor(ctx, shape, values);
    ggml_set_input(tensor.tensor);
    ggml_set_output(tensor.tensor);
    return tensor;
}

engine::core::TensorValue add_tensor_values(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & lhs,
    const engine::core::TensorValue & rhs) {
    auto lhs_contiguous = contiguous(ctx, lhs);
    auto rhs_contiguous = contiguous(ctx, rhs);
    return engine::core::wrap_tensor(
        ggml_add(ctx.ggml, lhs_contiguous.tensor, rhs_contiguous.tensor),
        lhs.shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue mul_tensor_values(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & lhs,
    const engine::core::TensorValue & rhs) {
    auto lhs_contiguous = contiguous(ctx, lhs);
    auto rhs_contiguous = contiguous(ctx, rhs);
    return engine::core::wrap_tensor(
        ggml_mul(ctx.ggml, lhs_contiguous.tensor, rhs_contiguous.tensor),
        lhs.shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue permute_tensor(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    std::array<int, engine::core::kMaxTensorRank> axes) {
    std::array<int, engine::core::kMaxTensorRank> ggml_axes = {0, 1, 2, 3};
    engine::core::TensorShape output_shape = {};
    output_shape.rank = input.shape.rank;
    for (size_t out_logical_axis = 0; out_logical_axis < input.shape.rank; ++out_logical_axis) {
        const int in_logical_axis = axes[out_logical_axis];
        output_shape.dims[out_logical_axis] = input.shape.dims[in_logical_axis];
        const int out_ggml_axis = static_cast<int>(input.shape.rank) - 1 - static_cast<int>(out_logical_axis);
        ggml_axes[out_ggml_axis] = engine::core::logical_axis_to_ggml_axis(input.shape.rank, in_logical_axis);
    }
    return engine::core::wrap_tensor(
        ggml_permute(ctx.ggml, input.tensor, ggml_axes[0], ggml_axes[1], ggml_axes[2], ggml_axes[3]),
        output_shape,
        input.type);
}

engine::core::TensorValue transpose_last_two(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    std::array<int, engine::core::kMaxTensorRank> axes = {0, 1, 2, 3};
    const size_t last = input.shape.rank - 1;
    const size_t second_last = input.shape.rank - 2;
    axes[second_last] = static_cast<int>(last);
    axes[last] = static_cast<int>(second_last);
    return permute_tensor(ctx, input, axes);
}

engine::core::TensorValue batched_matmul(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & lhs,
    const engine::core::TensorValue & rhs) {
    auto rhs_transposed = transpose_last_two(ctx, rhs);
    rhs_transposed = contiguous(ctx, rhs_transposed);
    auto output_shape = lhs.shape;
    output_shape.dims[lhs.shape.rank - 1] = rhs.shape.dims[rhs.shape.rank - 1];
    return engine::core::wrap_tensor(ggml_mul_mat(ctx.ggml, rhs_transposed.tensor, lhs.tensor), output_shape, GGML_TYPE_F32);
}

template <typename LinearWeights>
engine::core::TensorValue linear_lastdim(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const LinearWeights & weights,
    engine::core::DeferredTensorWriter & writer) {
    (void) writer;
    const int64_t rows = input.shape.prefix_elements();
    auto flat = engine::core::reshape_tensor(
        ctx,
        contiguous(ctx, input),
        engine::core::TensorShape::from_dims({rows, input.shape.last_dim()}));
    auto projected = engine::core::wrap_tensor(
        ggml_mul_mat(ctx.ggml, weights.weight_tensor.tensor, flat.tensor),
        engine::core::TensorShape::from_dims({rows, weights.out_features}),
        GGML_TYPE_F32);
    if (weights.use_bias) {
        const auto bias_view = engine::core::reshape_tensor(
            ctx,
            weights.bias_tensor,
            engine::core::TensorShape::from_dims({1, weights.out_features}));
        projected = engine::core::wrap_tensor(
            ggml_add(ctx.ggml, projected.tensor, bias_view.tensor),
            projected.shape,
            GGML_TYPE_F32);
    }
    return engine::core::reshape_tensor(ctx, projected, input.shape.with_last_dim(weights.out_features));
}

template <typename LayerNormWeights>
engine::core::TensorValue layer_norm_lastdim(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const LayerNormWeights & weights,
    engine::core::DeferredTensorWriter & writer,
    float eps = 1.0e-5f) {
    (void) writer;
    auto normalized = engine::core::wrap_tensor(ggml_norm(ctx.ggml, contiguous(ctx, input).tensor, eps), input.shape, GGML_TYPE_F32);
    auto scaled = engine::core::wrap_tensor(
        ggml_mul(ctx.ggml, normalized.tensor, weights.weight_tensor.tensor),
        input.shape,
        GGML_TYPE_F32);
    return engine::core::wrap_tensor(
        ggml_add(ctx.ggml, scaled.tensor, weights.bias_tensor.tensor),
        input.shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue transpose_bct_btc(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    return permute_tensor(ctx, input, {0, 2, 1});
}

engine::core::TensorValue zero_prefix_bct(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t prefix_frames);

engine::core::TensorValue bct_add_bias(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & bias_tensor) {
    const int64_t channels = input.shape.dims[1];
    const auto bias = engine::core::reshape_tensor(
        ctx,
        bias_tensor,
        engine::core::TensorShape::from_dims({1, channels, 1}));
    return engine::core::wrap_tensor(
        ggml_add(ctx.ggml, input.tensor, bias.tensor),
        input.shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue add_attention_bias(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & bias_tensor,
    int64_t heads,
    int64_t head_dim) {
    const auto bias_view = engine::core::reshape_tensor(
        ctx,
        bias_tensor,
        engine::core::TensorShape::from_dims({1, heads, 1, head_dim}));
    return engine::core::wrap_tensor(
        ggml_add(ctx.ggml, input.tensor, bias_view.tensor),
        input.shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue zero_like_last_column(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    auto * view = ggml_view_4d(
        ctx.ggml,
        input.tensor,
        1,
        input.tensor->ne[1],
        input.tensor->ne[2],
        input.tensor->ne[3],
        input.tensor->nb[1],
        input.tensor->nb[2],
        input.tensor->nb[3],
        0);
    auto first = engine::core::wrap_tensor(
        view,
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], input.shape.dims[2], 1}),
        GGML_TYPE_F32);
    auto first_contiguous = contiguous(ctx, first);
    return engine::core::wrap_tensor(ggml_scale(ctx.ggml, first_contiguous.tensor, 0.0f), first.shape, GGML_TYPE_F32);
}

engine::core::TensorValue slice_last_dim_4d(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t start,
    int64_t count);

engine::core::TensorValue relative_shift(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    const int64_t query = input.shape.dims[2];
    const int64_t pos = input.shape.dims[3];
    auto zero_col = zero_like_last_column(ctx, input);
    auto padded = engine::core::wrap_tensor(
        ggml_concat(ctx.ggml, input.tensor, zero_col.tensor, engine::core::logical_axis_to_ggml_axis(4, 3)),
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], query, pos + 1}),
        GGML_TYPE_F32);
    auto padded_contiguous = contiguous(ctx, padded);
    auto flattened = engine::core::reshape_tensor(
        ctx,
        padded_contiguous,
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], query * (pos + 1)}));
    auto * shifted_view = ggml_view_3d(
        ctx.ggml,
        flattened.tensor,
        flattened.tensor->ne[0] - query,
        flattened.tensor->ne[1],
        flattened.tensor->ne[2],
        flattened.tensor->nb[1],
        flattened.tensor->nb[2],
        static_cast<size_t>(query) * sizeof(float));
    auto shifted = engine::core::wrap_tensor(
        shifted_view,
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], (query * (pos + 1)) - query}),
        GGML_TYPE_F32);
    shifted = contiguous(ctx, shifted);
    auto reshaped = engine::core::reshape_tensor(
        ctx,
        shifted,
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], query, pos}));
    return slice_last_dim_4d(ctx, reshaped, 0, query);
}

engine::core::TensorValue slice_last_dim_4d(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t start,
    int64_t count) {
    auto * view = ggml_view_4d(
        ctx.ggml,
        input.tensor,
        count,
        input.tensor->ne[1],
        input.tensor->ne[2],
        input.tensor->ne[3],
        input.tensor->nb[1],
        input.tensor->nb[2],
        input.tensor->nb[3],
        static_cast<size_t>(start) * sizeof(float));
    return engine::core::wrap_tensor(
        view,
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], input.shape.dims[2], count}),
        GGML_TYPE_F32);
}

engine::core::TensorValue mish_activation(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    engine::core::DeferredTensorWriter & writer) {
    static const std::vector<float> kOne = {1.0f};
    const auto one = make_f32_graph_constant(ctx, engine::core::TensorShape::from_dims({1}), kOne, writer);
    const auto input_contiguous = contiguous(ctx, input);
    auto ex = engine::core::wrap_tensor(ggml_exp(ctx.ggml, input_contiguous.tensor), input.shape, GGML_TYPE_F32);
    auto softplus = engine::core::wrap_tensor(ggml_log(ctx.ggml, ggml_add(ctx.ggml, ex.tensor, one.tensor)), input.shape, GGML_TYPE_F32);
    auto squashed = engine::core::wrap_tensor(ggml_tanh(ctx.ggml, softplus.tensor), input.shape, GGML_TYPE_F32);
    return mul_tensor_values(ctx, input_contiguous, squashed);
}

engine::core::TensorValue causal_conv1d_bct(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const S3FlowDecoderWeights::Conv1dWeights & weights,
    engine::core::DeferredTensorWriter & writer) {
    (void) writer;
    const int64_t batch = input.shape.dims[0];
    const int64_t frames = input.shape.dims[2];
    if (weights.kernel == 1 && weights.stride == 1) {
        auto input_btc = transpose_bct_btc(ctx, input);
        const int64_t rows = batch * frames;
        auto flat = engine::core::reshape_tensor(
            ctx,
            contiguous(ctx, input_btc),
            engine::core::TensorShape::from_dims({rows, weights.in_channels}));
        const auto weight = engine::core::reshape_tensor(
            ctx,
            weights.weight_tensor,
            engine::core::TensorShape::from_dims({weights.out_channels, weights.in_channels}));
        auto projected = engine::core::wrap_tensor(
            ggml_mul_mat(ctx.ggml, weight.tensor, flat.tensor),
            engine::core::TensorShape::from_dims({rows, weights.out_channels}),
            GGML_TYPE_F32);
        if (weights.use_bias) {
            const auto bias_view = engine::core::reshape_tensor(
                ctx,
                weights.bias_tensor,
                engine::core::TensorShape::from_dims({1, weights.out_channels}));
            projected = engine::core::wrap_tensor(
                ggml_add(ctx.ggml, projected.tensor, bias_view.tensor),
                projected.shape,
                GGML_TYPE_F32);
        }
        auto output_btc = engine::core::reshape_tensor(
            ctx,
            projected,
            engine::core::TensorShape::from_dims({batch, frames, weights.out_channels}));
        return transpose_bct_btc(ctx, output_btc);
    }
    const int64_t left_pad = weights.kernel - 1;
    const auto weight = weights.weight_tensor;
    engine::core::TensorValue bias;
    engine::core::TensorValue bias_view;
    if (weights.use_bias) {
        bias = weights.bias_tensor;
        bias_view = engine::core::reshape_tensor(
            ctx,
            bias,
            engine::core::TensorShape::from_dims({1, weights.out_channels, 1}));
    }
    if (batch > 1) {
        engine::core::TensorValue output;
        for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
            auto input_single = engine::modules::SliceModule({0, batch_index, 1}).build(ctx, input);
            auto padded_single = engine::modules::ConcatModule({2}).build(
                ctx,
                zero_prefix_bct(ctx, input_single, left_pad),
                input_single);
            auto output_single = engine::core::wrap_tensor(
                ggml_conv_1d(ctx.ggml, contiguous(ctx, weight).tensor, contiguous(ctx, padded_single).tensor, 1, 0, 1),
                engine::core::TensorShape::from_dims({1, weights.out_channels, frames}),
                GGML_TYPE_F32);
            if (weights.use_bias) {
                output_single = engine::core::wrap_tensor(
                    ggml_add(ctx.ggml, output_single.tensor, bias_view.tensor),
                    output_single.shape,
                    GGML_TYPE_F32);
            }
            output = output.valid()
                ? engine::modules::ConcatModule({0}).build(ctx, output, output_single)
                : output_single;
        }
        return contiguous(ctx, output);
    }
    auto padded = engine::modules::ConcatModule({2}).build(
        ctx,
        zero_prefix_bct(ctx, input, left_pad),
        input);
    auto output = engine::core::wrap_tensor(
        ggml_conv_1d(ctx.ggml, contiguous(ctx, weight).tensor, contiguous(ctx, padded).tensor, 1, 0, 1),
        engine::core::TensorShape::from_dims({batch, weights.out_channels, frames}),
        GGML_TYPE_F32);
    if (weights.use_bias) {
        output = engine::core::wrap_tensor(
            ggml_add(ctx.ggml, output.tensor, bias_view.tensor),
            output.shape,
            GGML_TYPE_F32);
    }
    return output;
}

engine::core::TensorValue causal_block_forward(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_bct,
    const S3FlowDecoderWeights::CausalBlockWeights & weights,
    engine::core::DeferredTensorWriter & writer) {
    auto x = causal_conv1d_bct(ctx, input_bct, weights.conv, writer);
    x = transpose_bct_btc(ctx, x);
    x = layer_norm_lastdim(ctx, x, weights.norm, writer);
    x = transpose_bct_btc(ctx, x);
    return mish_activation(ctx, x, writer);
}

engine::core::TensorValue repeat_spks_bct(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & spks_bt,
    int64_t frames) {
    auto spks_bct = engine::core::reshape_tensor(
        ctx,
        spks_bt,
        engine::core::TensorShape::from_dims({spks_bt.shape.dims[0], spks_bt.shape.dims[1], 1}));
    auto target = engine::core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        engine::core::TensorShape::from_dims({spks_bt.shape.dims[0], spks_bt.shape.dims[1], frames}));
    return engine::core::wrap_tensor(ggml_repeat(ctx.ggml, spks_bct.tensor, target.tensor), target.shape, GGML_TYPE_F32);
}

engine::core::TensorValue cat_channels_bct(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & a,
    const engine::core::TensorValue & b) {
    return engine::core::wrap_tensor(
        ggml_concat(ctx.ggml, a.tensor, b.tensor, engine::core::logical_axis_to_ggml_axis(3, 1)),
        engine::core::TensorShape::from_dims({a.shape.dims[0], a.shape.dims[1] + b.shape.dims[1], a.shape.dims[2]}),
        GGML_TYPE_F32);
}

engine::core::TensorValue add_time_bias_bct(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_bct,
    const engine::core::TensorValue & time_bt) {
    auto time_bct = engine::core::reshape_tensor(
        ctx,
        time_bt,
        engine::core::TensorShape::from_dims({time_bt.shape.dims[0], time_bt.shape.dims[1], 1}));
    return engine::core::wrap_tensor(
        ggml_add(ctx.ggml, input_bct.tensor, time_bct.tensor),
        input_bct.shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue zero_prefix_bct(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    int64_t prefix_frames) {
    auto first = engine::modules::SliceModule({2, 0, 1}).build(ctx, input);
    auto prefix = engine::modules::RepeatModule({
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], prefix_frames})})
                      .build(ctx, first);
    auto prefix_contiguous = contiguous(ctx, prefix);
    return engine::core::wrap_tensor(ggml_scale(ctx.ggml, prefix_contiguous.tensor, 0.0f), prefix.shape, GGML_TYPE_F32);
}

engine::core::TensorValue self_attention_no_mask(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_btc,
    const S3FlowDecoderWeights::TransformerBlockWeights & weights,
    engine::core::DeferredTensorWriter & writer,
    const std::optional<engine::core::TensorValue> & attention_mask = std::nullopt) {
    constexpr int64_t heads = 8;
    constexpr int64_t head_dim = 64;
    constexpr int64_t inner_dim = heads * head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto q = linear_lastdim(ctx, input_btc, weights.attn_q, writer);
    auto k = linear_lastdim(ctx, input_btc, weights.attn_k, writer);
    auto v = linear_lastdim(ctx, input_btc, weights.attn_v, writer);
    const int64_t batch = input_btc.shape.dims[0];
    const int64_t frames = input_btc.shape.dims[1];
    q = engine::core::reshape_tensor(ctx, q, engine::core::TensorShape::from_dims({batch, frames, heads, head_dim}));
    k = engine::core::reshape_tensor(ctx, k, engine::core::TensorShape::from_dims({batch, frames, heads, head_dim}));
    v = engine::core::reshape_tensor(ctx, v, engine::core::TensorShape::from_dims({batch, frames, heads, head_dim}));
    q = permute_tensor(ctx, q, {0, 2, 1, 3});
    k = permute_tensor(ctx, k, {0, 2, 1, 3});
    v = permute_tensor(ctx, v, {0, 2, 1, 3});
    engine::core::TensorValue context;
    if (attention_mask.has_value()) {
        auto q_contiguous = contiguous(ctx, q);
        auto k_contiguous = contiguous(ctx, k);
        auto v_contiguous = contiguous(ctx, v);
        auto * flash = ggml_flash_attn_ext(
            ctx.ggml,
            q_contiguous.tensor,
            k_contiguous.tensor,
            v_contiguous.tensor,
            attention_mask->tensor,
            scale,
            0.0f,
            0.0f);
        ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
        context = engine::core::wrap_tensor(
            flash,
            engine::core::TensorShape::from_dims({batch, frames, heads, head_dim}),
            GGML_TYPE_F32);
    } else {
        auto k_t = permute_tensor(ctx, k, {0, 1, 3, 2});
        auto scores = batched_matmul(ctx, q, k_t);
        scores = engine::core::wrap_tensor(ggml_scale(ctx.ggml, scores.tensor, scale), scores.shape, GGML_TYPE_F32);
        auto attn = engine::core::wrap_tensor(ggml_soft_max(ctx.ggml, contiguous(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
        context = batched_matmul(ctx, attn, v);
        context = permute_tensor(ctx, context, {0, 2, 1, 3});
    }
    context = contiguous(ctx, context);
    context = engine::core::reshape_tensor(ctx, context, engine::core::TensorShape::from_dims({batch, frames, inner_dim}));
    return linear_lastdim(ctx, context, weights.attn_out, writer);
}

engine::core::TensorValue transformer_block_forward(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_btc,
    const S3FlowDecoderWeights::TransformerBlockWeights & weights,
    engine::core::DeferredTensorWriter & writer,
    const std::optional<engine::core::TensorValue> & attention_mask = std::nullopt) {
    auto x = layer_norm_lastdim(ctx, input_btc, weights.norm1, writer);
    auto attn = self_attention_no_mask(ctx, x, weights, writer, attention_mask);
    auto y = add_tensor_values(ctx, input_btc, attn);
    x = layer_norm_lastdim(ctx, y, weights.norm3, writer);
    auto ff = linear_lastdim(ctx, x, weights.ff_proj_in, writer);
    ff = engine::core::wrap_tensor(ggml_gelu_erf(ctx.ggml, ff.tensor), ff.shape, GGML_TYPE_F32);
    ff = linear_lastdim(ctx, ff, weights.ff_proj_out, writer);
    return add_tensor_values(ctx, y, ff);
}

engine::core::TensorValue resnet_block_forward(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_bct,
    const engine::core::TensorValue & time_emb,
    const S3FlowDecoderWeights::ResnetBlockWeights & weights,
    engine::core::DeferredTensorWriter & writer) {
    auto h = causal_block_forward(ctx, input_bct, weights.block1, writer);
    auto time_proj = mish_activation(ctx, time_emb, writer);
    time_proj = linear_lastdim(ctx, time_proj, weights.time_mlp, writer);
    h = add_time_bias_bct(ctx, h, time_proj);
    h = causal_block_forward(ctx, h, weights.block2, writer);
    auto residual = causal_conv1d_bct(ctx, input_bct, weights.res_conv, writer);
    return add_tensor_values(ctx, h, residual);
}

engine::core::TensorValue flow_conv1d_bct_backend(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const S3FlowEncoderWeights::Conv1dWeights & weights,
    engine::core::DeferredTensorWriter & writer) {
    (void) writer;
    const int64_t out_frames =
        (input.shape.dims[2] + 2 * weights.padding - weights.kernel) / weights.stride + 1;
    auto output = engine::core::wrap_tensor(
        ggml_conv_1d(
            ctx.ggml,
            contiguous(ctx, weights.weight_tensor).tensor,
            contiguous(ctx, input).tensor,
            static_cast<int>(weights.stride),
            static_cast<int>(weights.padding),
            1),
        engine::core::TensorShape::from_dims({input.shape.dims[0], weights.out_channels, out_frames}),
        GGML_TYPE_F32);
    output = bct_add_bias(ctx, output, weights.bias_tensor);
    return output;
}

engine::core::TensorValue flow_prelookahead_backend(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_btc,
    int64_t frames,
    int64_t hidden,
    const S3FlowEncoderWeights::Conv1dWeights & conv1,
    const S3FlowEncoderWeights::Conv1dWeights & conv2,
    engine::core::DeferredTensorWriter & writer) {
    auto input_bct = transpose_bct_btc(ctx, input_btc);
    auto padded1 = engine::core::wrap_tensor(
        ggml_pad_ext(ctx.ggml, contiguous(ctx, input_bct).tensor, 0, 3, 0, 0, 0, 0, 0, 0),
        engine::core::TensorShape::from_dims({1, hidden, frames + 3}),
        GGML_TYPE_F32);
    auto h = flow_conv1d_bct_backend(ctx, padded1, conv1, writer);
    h = engine::core::wrap_tensor(ggml_leaky_relu(ctx.ggml, h.tensor, 0.01f, false), h.shape, GGML_TYPE_F32);
    auto padded2 = engine::core::wrap_tensor(
        ggml_pad_ext(ctx.ggml, contiguous(ctx, h).tensor, 2, 0, 0, 0, 0, 0, 0, 0),
        engine::core::TensorShape::from_dims({1, hidden, frames + 2}),
        GGML_TYPE_F32);
    auto y_bct = flow_conv1d_bct_backend(ctx, padded2, conv2, writer);
    auto y_btc = transpose_bct_btc(ctx, y_bct);
    return add_tensor_values(ctx, input_btc, y_btc);
}

engine::core::TensorValue flow_relative_attention_backend(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_btc,
    const engine::core::TensorValue & pos_emb_btc,
    int64_t frames,
    int64_t hidden,
    const S3FlowEncoderWeights::RelativeAttentionWeights & weights,
    engine::core::DeferredTensorWriter & writer,
    const std::optional<engine::core::TensorValue> & attention_mask = std::nullopt,
    int64_t heads = 8,
    int64_t head_dim = 64) {
    auto q = linear_lastdim(ctx, input_btc, weights.q, writer);
    auto k = linear_lastdim(ctx, input_btc, weights.k, writer);
    auto v = linear_lastdim(ctx, input_btc, weights.v, writer);
    const int64_t pos_frames = (frames * 2) - 1;
    auto p = linear_lastdim(ctx, pos_emb_btc, weights.pos, writer);

    q = engine::core::reshape_tensor(ctx, q, engine::core::TensorShape::from_dims({1, frames, heads, head_dim}));
    k = engine::core::reshape_tensor(ctx, k, engine::core::TensorShape::from_dims({1, frames, heads, head_dim}));
    v = engine::core::reshape_tensor(ctx, v, engine::core::TensorShape::from_dims({1, frames, heads, head_dim}));
    p = engine::core::reshape_tensor(ctx, p, engine::core::TensorShape::from_dims({1, pos_frames, heads, head_dim}));

    auto q_heads = permute_tensor(ctx, q, {0, 2, 1, 3});
    auto k_heads = permute_tensor(ctx, k, {0, 2, 1, 3});
    auto v_heads = permute_tensor(ctx, v, {0, 2, 1, 3});
    auto p_heads = permute_tensor(ctx, p, {0, 2, 1, 3});

    auto q_u = add_attention_bias(ctx, q_heads, weights.pos_bias_u_tensor, heads, head_dim);
    auto q_v = add_attention_bias(ctx, q_heads, weights.pos_bias_v_tensor, heads, head_dim);

    auto matrix_ac = batched_matmul(ctx, q_u, permute_tensor(ctx, k_heads, {0, 1, 3, 2}));
    auto matrix_bd = batched_matmul(ctx, q_v, permute_tensor(ctx, p_heads, {0, 1, 3, 2}));
    matrix_bd = relative_shift(ctx, matrix_bd);
    matrix_bd = slice_last_dim_4d(ctx, matrix_bd, 0, frames);

    auto scores = add_tensor_values(ctx, matrix_ac, matrix_bd);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    scores = engine::core::wrap_tensor(
        ggml_scale(ctx.ggml, scores.tensor, scale),
        scores.shape,
        GGML_TYPE_F32);
    if (attention_mask.has_value()) {
        scores = add_tensor_values(ctx, scores, *attention_mask);
    }
    auto attn = engine::core::wrap_tensor(ggml_soft_max(ctx.ggml, contiguous(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
    auto context = batched_matmul(ctx, attn, v_heads);
    context = permute_tensor(ctx, context, {0, 2, 1, 3});
    context = contiguous(ctx, context);
    context = engine::core::reshape_tensor(ctx, context, engine::core::TensorShape::from_dims({1, frames, hidden}));
    return linear_lastdim(ctx, context, weights.out, writer);
}

engine::core::TensorValue flow_upsample_repeat_conv_backend(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_btc,
    int64_t frames,
    int64_t hidden,
    const S3FlowEncoderWeights::Conv1dWeights & weights,
    engine::core::DeferredTensorWriter & writer) {
    std::vector<float> repeat_weight(static_cast<size_t>(hidden * hidden * 2), 0.0f);
    for (int64_t c = 0; c < hidden; ++c) {
        const size_t base = static_cast<size_t>((c * hidden + c) * 2);
        repeat_weight[base] = 1.0f;
        repeat_weight[base + 1] = 1.0f;
    }
    auto input_bct = transpose_bct_btc(ctx, input_btc);
    const auto repeat_kernel = make_f32_graph_constant(
        ctx,
        engine::core::TensorShape::from_dims({hidden, hidden, 2}),
        repeat_weight,
        writer);
    auto repeated = engine::modules::ConvTranspose1dModule({
        hidden,
        hidden,
        2,
        2,
        0,
        1,
        false,
    }).build(ctx, input_bct, engine::modules::ConvTranspose1dWeights{repeat_kernel, std::nullopt});
    auto padded = engine::core::wrap_tensor(
        ggml_pad_ext(ctx.ggml, repeated.tensor, 4, 0, 0, 0, 0, 0, 0, 0),
        engine::core::TensorShape::from_dims({1, hidden, frames * 2 + 4}),
        GGML_TYPE_F32);
    auto y_bct = flow_conv1d_bct_backend(ctx, padded, weights, writer);
    return transpose_bct_btc(ctx, y_bct);
}

class FlowEncoderBackendRunner {
public:
    FlowEncoderBackendRunner(
        const S3FlowEncoderWeights & weights,
        int64_t frames,
        int64_t hidden_size,
        const engine::core::BackendConfig & backend_config)
        : input_frames_(frames),
          hidden_size_(hidden_size),
          output_frames_(frames * 2),
          embed_prelook_runner_(weights, frames, hidden_size, backend_config),
          upsample_runner_(weights, frames, hidden_size, backend_config),
          after_norm_runner_(weights.after_norm, *weights.execution_context, output_frames_, hidden_size, backend_config) {
        encoder_runners_.reserve(weights.encoders.size());
        for (const auto & layer : weights.encoders) {
            encoder_runners_.push_back(
                std::make_unique<FlowEncoderLayerBackendRunner>(
                    layer, *weights.execution_context, input_frames_, hidden_size_, backend_config));
        }
        up_encoder_runners_.reserve(weights.up_encoders.size());
        for (const auto & layer : weights.up_encoders) {
            up_encoder_runners_.push_back(
                std::make_unique<FlowEncoderLayerBackendRunner>(
                    layer, *weights.execution_context, output_frames_, hidden_size_, backend_config));
        }
    }

    S3FlowEncoderOutputs run(const std::vector<float> & input, int64_t valid_frames) {
        const auto attention_mask_values = make_attention_key_mask(1, input_frames_, input_frames_, valid_frames);
        embed_prelook_runner_.write_input(input);
        embed_prelook_runner_.compute();

        auto * previous_output = &embed_prelook_runner_.output_tensor();
        for (auto & runner : encoder_runners_) {
            runner->write_attention_mask(attention_mask_values);
            copy_backend_tensor(*previous_output, runner->input_tensor());
            runner->compute();
            previous_output = &runner->output_tensor();
        }

        copy_backend_tensor(*previous_output, upsample_runner_.input_tensor());
        upsample_runner_.compute();

        const int64_t valid_output_frames = valid_frames * 2;
        const auto up_attention_mask_values = make_attention_key_mask(1, output_frames_, output_frames_, valid_output_frames);
        previous_output = &upsample_runner_.output_tensor();
        for (auto & runner : up_encoder_runners_) {
            runner->write_attention_mask(up_attention_mask_values);
            copy_backend_tensor(*previous_output, runner->input_tensor());
            runner->compute();
            previous_output = &runner->output_tensor();
        }

        copy_backend_tensor(*previous_output, after_norm_runner_.input_tensor());
        after_norm_runner_.compute();

        S3FlowEncoderOutputs outputs;
        outputs.hidden = after_norm_runner_.read_output();
        outputs.frames = valid_output_frames;
        outputs.storage_frames = output_frames_;
        outputs.hidden_size = hidden_size_;
        return outputs;
    }

private:
    class FlowEncoderEmbedPrelookBackendRunner {
    public:
        FlowEncoderEmbedPrelookBackendRunner(
            const S3FlowEncoderWeights & weights,
            int64_t frames,
            int64_t hidden_size,
            const engine::core::BackendConfig &)
            : execution_context_(weights.execution_context) {
            ggml_init_params params = {};
            params.mem_size = 192ull * 1024ull * 1024ull;
            params.mem_buffer = nullptr;
            params.no_alloc = true;
            ggml_ = ggml_init(params);
            if (ggml_ == nullptr) {
                throw std::runtime_error("failed to initialize ggml context for S3 flow encoder embed/prelook");
            }

            engine::core::ModuleBuildContext ctx = {};
            ctx.ggml = ggml_;
            ctx.module_instance_name = "s3_flow_encoder_embed_prelook";
            input_tensor_ = engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, frames, hidden_size}));

            auto x = linear_lastdim(ctx, input_tensor_, weights.embed_linear, writer_);
            x = layer_norm_lastdim(ctx, x, weights.embed_norm, writer_);
            x = engine::core::wrap_tensor(
                ggml_scale(ctx.ggml, x.tensor, std::sqrt(static_cast<float>(hidden_size))),
                x.shape,
                GGML_TYPE_F32);
            output_ = flow_prelookahead_backend(
                ctx,
                x,
                frames,
                hidden_size,
                weights.prelook_conv1,
                weights.prelook_conv2,
                writer_);

            graph_ = ggml_new_graph_custom(ggml_, 16384, false);
            ggml_build_forward_expand(graph_, output_.tensor);
            buffer_ = ggml_backend_alloc_ctx_tensors(ggml_, execution_context_->backend());
            if (buffer_ == nullptr) {
                throw std::runtime_error("failed to allocate backend tensors for S3 flow encoder embed/prelook");
            }
            writer_.flush();
            engine::core::prepare_host_graph_plan(*execution_context_, graph_, cpu_plan_);
        }

        ~FlowEncoderEmbedPrelookBackendRunner() {
            release_graph_resources(execution_context_, graph_, ggml_, buffer_);
        }

        void write_input(const std::vector<float> & input) {
            engine::core::write_tensor_f32(input_tensor_, input);
        }

        void compute() {
            if (engine::core::compute_graph(*execution_context_, graph_, cpu_plan_) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("ggml_backend_graph_compute failed for S3 flow encoder embed/prelook");
            }
        }

        const engine::core::TensorValue & output_tensor() const {
            return output_;
        }

    private:
        const engine::core::ExecutionContext * execution_context_ = nullptr;
        ggml_context * ggml_ = nullptr;
        ggml_backend_buffer_t buffer_ = nullptr;
        ggml_cgraph * graph_ = nullptr;
        engine::core::TensorValue input_tensor_;
        engine::core::TensorValue output_;
        engine::core::HostGraphPlan cpu_plan_;
        engine::core::DeferredTensorWriter writer_;
    };

    class FlowEncoderLayerBackendRunner {
    public:
        FlowEncoderLayerBackendRunner(
            const S3FlowEncoderWeights::EncoderLayerWeights & weights,
            const engine::core::ExecutionContext & execution_context,
            int64_t frames,
            int64_t hidden_size,
            const engine::core::BackendConfig & backend_config)
            : attention_runner_(weights, execution_context, frames, hidden_size, backend_config),
              ff_runner_(weights, execution_context, frames, hidden_size, backend_config) {}

        engine::core::TensorValue & input_tensor() {
            return attention_runner_.input_tensor();
        }

        void write_attention_mask(const std::vector<float> & values) {
            attention_runner_.write_attention_mask(values);
        }

        const engine::core::TensorValue & output_tensor() const {
            return ff_runner_.output_tensor();
        }

        void compute() {
            attention_runner_.compute();
            copy_backend_tensor(attention_runner_.output_tensor(), ff_runner_.input_tensor());
            ff_runner_.compute();
        }

    private:
        class FlowEncoderAttentionBackendRunner {
        public:
            FlowEncoderAttentionBackendRunner(
                const S3FlowEncoderWeights::EncoderLayerWeights & weights,
                const engine::core::ExecutionContext & execution_context,
                int64_t frames,
                int64_t hidden_size,
                const engine::core::BackendConfig &)
                : execution_context_(&execution_context) {
                ggml_init_params params = {};
                params.mem_size = 192ull * 1024ull * 1024ull;
                params.mem_buffer = nullptr;
                params.no_alloc = true;
                ggml_ = ggml_init(params);
                if (ggml_ == nullptr) {
                    throw std::runtime_error("failed to initialize ggml context for S3 flow encoder attention");
                }

                engine::core::ModuleBuildContext ctx = {};
                ctx.ggml = ggml_;
                ctx.module_instance_name = "s3_flow_encoder_attention";
                input_tensor_ = engine::core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    engine::core::TensorShape::from_dims({1, frames, hidden_size}));
                attention_mask_ = engine::core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    engine::core::TensorShape::from_dims({1, 1, frames, frames}));
                const auto pos_tensor = make_f32_graph_constant(
                    ctx,
                    engine::core::TensorShape::from_dims({1, (frames * 2) - 1, hidden_size}),
                    flow_relative_positional_encoding(frames, hidden_size),
                    writer_);
                auto x = layer_norm_lastdim(ctx, input_tensor_, weights.norm_mha, writer_);
                auto attn = flow_relative_attention_backend(
                    ctx,
                    x,
                    pos_tensor,
                    frames,
                    hidden_size,
                    weights.attn,
                    writer_,
                    attention_mask_);
                output_ = add_tensor_values(ctx, input_tensor_, attn);

                graph_ = ggml_new_graph_custom(ggml_, 24576, false);
                ggml_build_forward_expand(graph_, output_.tensor);
                buffer_ = ggml_backend_alloc_ctx_tensors(ggml_, execution_context_->backend());
                if (buffer_ == nullptr) {
                    throw std::runtime_error("failed to allocate backend tensors for S3 flow encoder attention");
                }
                writer_.flush();
                engine::core::prepare_host_graph_plan(*execution_context_, graph_, cpu_plan_);
            }

            ~FlowEncoderAttentionBackendRunner() {
                release_graph_resources(execution_context_, graph_, ggml_, buffer_);
            }

            engine::core::TensorValue & input_tensor() {
                return input_tensor_;
            }

            void write_attention_mask(const std::vector<float> & values) {
                engine::core::write_tensor_f32(attention_mask_, values);
            }

            const engine::core::TensorValue & output_tensor() const {
                return output_;
            }

            void compute() {
                if (engine::core::compute_graph(*execution_context_, graph_, cpu_plan_) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("ggml_backend_graph_compute failed for S3 flow encoder attention");
                }
            }

        private:
            const engine::core::ExecutionContext * execution_context_ = nullptr;
            ggml_context * ggml_ = nullptr;
            ggml_backend_buffer_t buffer_ = nullptr;
            ggml_cgraph * graph_ = nullptr;
            engine::core::TensorValue input_tensor_;
            engine::core::TensorValue attention_mask_;
            engine::core::TensorValue output_;
            engine::core::HostGraphPlan cpu_plan_;
            engine::core::DeferredTensorWriter writer_;
        };

        class FlowEncoderFeedForwardBackendRunner {
        public:
            FlowEncoderFeedForwardBackendRunner(
                const S3FlowEncoderWeights::EncoderLayerWeights & weights,
                const engine::core::ExecutionContext & execution_context,
                int64_t frames,
                int64_t hidden_size,
                const engine::core::BackendConfig &)
                : execution_context_(&execution_context) {
                ggml_init_params params = {};
                params.mem_size = 128ull * 1024ull * 1024ull;
                params.mem_buffer = nullptr;
                params.no_alloc = true;
                ggml_ = ggml_init(params);
                if (ggml_ == nullptr) {
                    throw std::runtime_error("failed to initialize ggml context for S3 flow encoder feed-forward");
                }

                engine::core::ModuleBuildContext ctx = {};
                ctx.ggml = ggml_;
                ctx.module_instance_name = "s3_flow_encoder_ff";
                input_tensor_ = engine::core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    engine::core::TensorShape::from_dims({1, frames, hidden_size}));
                auto x = layer_norm_lastdim(ctx, input_tensor_, weights.norm_ff, writer_);
                auto ff = linear_lastdim(ctx, x, weights.ff.w1, writer_);
                ff = engine::core::wrap_tensor(ggml_silu(ctx.ggml, ff.tensor), ff.shape, GGML_TYPE_F32);
                ff = linear_lastdim(ctx, ff, weights.ff.w2, writer_);
                output_ = add_tensor_values(ctx, input_tensor_, ff);

                graph_ = ggml_new_graph_custom(ggml_, 16384, false);
                ggml_build_forward_expand(graph_, output_.tensor);
                buffer_ = ggml_backend_alloc_ctx_tensors(ggml_, execution_context_->backend());
                if (buffer_ == nullptr) {
                    throw std::runtime_error("failed to allocate backend tensors for S3 flow encoder feed-forward");
                }
                writer_.flush();
                engine::core::prepare_host_graph_plan(*execution_context_, graph_, cpu_plan_);
            }

            ~FlowEncoderFeedForwardBackendRunner() {
                release_graph_resources(execution_context_, graph_, ggml_, buffer_);
            }

            engine::core::TensorValue & input_tensor() {
                return input_tensor_;
            }

            const engine::core::TensorValue & output_tensor() const {
                return output_;
            }

            void compute() {
                if (engine::core::compute_graph(*execution_context_, graph_, cpu_plan_) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("ggml_backend_graph_compute failed for S3 flow encoder feed-forward");
                }
            }

        private:
            const engine::core::ExecutionContext * execution_context_ = nullptr;
            ggml_context * ggml_ = nullptr;
            ggml_backend_buffer_t buffer_ = nullptr;
            ggml_cgraph * graph_ = nullptr;
            engine::core::TensorValue input_tensor_;
            engine::core::TensorValue output_;
            engine::core::HostGraphPlan cpu_plan_;
            engine::core::DeferredTensorWriter writer_;
        };

        FlowEncoderAttentionBackendRunner attention_runner_;
        FlowEncoderFeedForwardBackendRunner ff_runner_;
    };

    class FlowEncoderUpsampleBackendRunner {
    public:
        FlowEncoderUpsampleBackendRunner(
            const S3FlowEncoderWeights & weights,
            int64_t frames,
            int64_t hidden_size,
            const engine::core::BackendConfig &)
            : execution_context_(weights.execution_context) {
            ggml_init_params params = {};
            params.mem_size = 192ull * 1024ull * 1024ull;
            params.mem_buffer = nullptr;
            params.no_alloc = true;
            ggml_ = ggml_init(params);
            if (ggml_ == nullptr) {
                throw std::runtime_error("failed to initialize ggml context for S3 flow encoder upsample");
            }

            engine::core::ModuleBuildContext ctx = {};
            ctx.ggml = ggml_;
            ctx.module_instance_name = "s3_flow_encoder_upsample";
            input_tensor_ = engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, frames, hidden_size}));

            auto x = flow_upsample_repeat_conv_backend(
                ctx,
                input_tensor_,
                frames,
                hidden_size,
                weights.up_layer_conv,
                writer_);
            x = linear_lastdim(ctx, x, weights.up_embed_linear, writer_);
            x = layer_norm_lastdim(ctx, x, weights.up_embed_norm, writer_);
            output_ = engine::core::wrap_tensor(
                ggml_scale(ctx.ggml, x.tensor, std::sqrt(static_cast<float>(hidden_size))),
                x.shape,
                GGML_TYPE_F32);

            graph_ = ggml_new_graph_custom(ggml_, 16384, false);
            ggml_build_forward_expand(graph_, output_.tensor);
            buffer_ = ggml_backend_alloc_ctx_tensors(ggml_, execution_context_->backend());
            if (buffer_ == nullptr) {
                throw std::runtime_error("failed to allocate backend tensors for S3 flow encoder upsample");
            }
            writer_.flush();
            engine::core::prepare_host_graph_plan(*execution_context_, graph_, cpu_plan_);
        }

        ~FlowEncoderUpsampleBackendRunner() {
            release_graph_resources(execution_context_, graph_, ggml_, buffer_);
        }

        engine::core::TensorValue & input_tensor() {
            return input_tensor_;
        }

        const engine::core::TensorValue & output_tensor() const {
            return output_;
        }

        void compute() {
            if (engine::core::compute_graph(*execution_context_, graph_, cpu_plan_) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("ggml_backend_graph_compute failed for S3 flow encoder upsample");
            }
        }

    private:
        const engine::core::ExecutionContext * execution_context_ = nullptr;
        ggml_context * ggml_ = nullptr;
        ggml_backend_buffer_t buffer_ = nullptr;
        ggml_cgraph * graph_ = nullptr;
        engine::core::TensorValue input_tensor_;
        engine::core::TensorValue output_;
        engine::core::HostGraphPlan cpu_plan_;
        engine::core::DeferredTensorWriter writer_;
    };

    class FlowEncoderAfterNormBackendRunner {
    public:
        FlowEncoderAfterNormBackendRunner(
            const S3FlowEncoderWeights::LayerNormWeights & weights,
            const engine::core::ExecutionContext & execution_context,
            int64_t frames,
            int64_t hidden_size,
            const engine::core::BackendConfig &)
            : execution_context_(&execution_context) {
            ggml_init_params params = {};
            params.mem_size = 64ull * 1024ull * 1024ull;
            params.mem_buffer = nullptr;
            params.no_alloc = true;
            ggml_ = ggml_init(params);
            if (ggml_ == nullptr) {
                throw std::runtime_error("failed to initialize ggml context for S3 flow encoder after norm");
            }

            engine::core::ModuleBuildContext ctx = {};
            ctx.ggml = ggml_;
            ctx.module_instance_name = "s3_flow_encoder_after_norm";
            input_tensor_ = engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, frames, hidden_size}));
            output_ = layer_norm_lastdim(ctx, input_tensor_, weights, writer_);

            graph_ = ggml_new_graph_custom(ggml_, 4096, false);
            ggml_build_forward_expand(graph_, output_.tensor);
            buffer_ = ggml_backend_alloc_ctx_tensors(ggml_, execution_context_->backend());
            if (buffer_ == nullptr) {
                throw std::runtime_error("failed to allocate backend tensors for S3 flow encoder after norm");
            }
            writer_.flush();
            engine::core::prepare_host_graph_plan(*execution_context_, graph_, cpu_plan_);
        }

        ~FlowEncoderAfterNormBackendRunner() {
            release_graph_resources(execution_context_, graph_, ggml_, buffer_);
        }

        engine::core::TensorValue & input_tensor() {
            return input_tensor_;
        }

        void compute() {
            if (engine::core::compute_graph(*execution_context_, graph_, cpu_plan_) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("ggml_backend_graph_compute failed for S3 flow encoder after norm");
            }
        }

        std::vector<float> read_output() const {
            return engine::core::read_tensor_f32(output_.tensor);
        }

    private:
        const engine::core::ExecutionContext * execution_context_ = nullptr;
        ggml_context * ggml_ = nullptr;
        ggml_backend_buffer_t buffer_ = nullptr;
        ggml_cgraph * graph_ = nullptr;
        engine::core::TensorValue input_tensor_;
        engine::core::TensorValue output_;
        engine::core::HostGraphPlan cpu_plan_;
        engine::core::DeferredTensorWriter writer_;
    };

    int64_t input_frames_ = 0;
    int64_t hidden_size_ = 0;
    int64_t output_frames_ = 0;
    FlowEncoderEmbedPrelookBackendRunner embed_prelook_runner_;
    std::vector<std::unique_ptr<FlowEncoderLayerBackendRunner>> encoder_runners_;
    FlowEncoderUpsampleBackendRunner upsample_runner_;
    std::vector<std::unique_ptr<FlowEncoderLayerBackendRunner>> up_encoder_runners_;
    FlowEncoderAfterNormBackendRunner after_norm_runner_;
};

class FlowDecoderBackendRunner {
public:
    FlowDecoderBackendRunner(
        const S3FlowDecoderWeights & weights,
        int64_t batch,
        int64_t capacity_frames,
        const engine::core::BackendConfig &)
        : batch_(batch),
          capacity_frames_(capacity_frames),
          execution_context_(weights.execution_context) {
        if (weights.down_blocks.size() != 1 || weights.up_blocks.size() != 1) {
            throw std::runtime_error("S3 flow decoder runner expects exactly one down block and one up block");
        }
    }

    ~FlowDecoderBackendRunner() {
        if (graph_ != nullptr) {
            engine::core::release_backend_graph_resources(backend_, graph_);
            graph_ = nullptr;
        }
        if (graph_ggml_ != nullptr) {
            ggml_free(graph_ggml_);
            graph_ggml_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
    }

    void ensure_active_graph(
        const S3FlowDecoderWeights & weights,
        int64_t frames,
        const engine::core::BackendConfig &) {
        constexpr int64_t mel_channels = 80;
        constexpr int64_t speaker_channels = 80;
        if (frames <= 0 || frames > capacity_frames_) {
            throw std::runtime_error("S3 flow decoder active frames exceed capacity");
        }
        if (graph_ != nullptr && active_frames_ == frames) {
            return;
        }
        if (graph_ != nullptr) {
            engine::core::release_backend_graph_resources(backend_, graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (graph_ggml_ != nullptr) {
            ggml_free(graph_ggml_);
            graph_ggml_ = nullptr;
        }
        output_tensor_ = {};
        attention_mask_ = {};
        skip_tensor_ = {};
        cpu_plan_.reset();
        active_frames_ = frames;
        cached_attention_mask_valid_frames_ = -1;
        cached_attention_mask_values_.clear();

        ggml_init_params params = {};
        params.mem_size = 512ull * 1024ull * 1024ull;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        graph_ggml_ = ggml_init(params);
        if (graph_ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize ggml graph context for S3 flow decoder");
        }
        engine::core::ModuleBuildContext ctx = {};
        ctx.ggml = graph_ggml_;
        ctx.module_instance_name = "s3_flow_decoder";
        engine::core::DeferredTensorWriter writer;
        backend_ = execution_context_->backend();
        if (!backend_) {
            throw std::runtime_error("S3 flow decoder backend is not initialized");
        }

        x_in_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({batch_, mel_channels, frames}));
        mu_in_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({batch_, mel_channels, frames}));
        time_in_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({batch_, 320}));
        spks_in_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({batch_, speaker_channels}));
        cond_in_ = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({batch_, mel_channels, frames}));
        ggml_set_input(x_in_.tensor);
        ggml_set_input(mu_in_.tensor);
        ggml_set_input(time_in_.tensor);
        ggml_set_input(spks_in_.tensor);
        ggml_set_input(cond_in_.tensor);
        attention_mask_ =
            engine::core::make_tensor(ctx, GGML_TYPE_F16, engine::core::TensorShape::from_dims({batch_, 8, frames, frames}));
        ggml_set_input(attention_mask_.tensor);
        ggml_set_output(x_in_.tensor);
        ggml_set_output(mu_in_.tensor);
        ggml_set_output(time_in_.tensor);
        ggml_set_output(spks_in_.tensor);
        ggml_set_output(cond_in_.tensor);
        ggml_set_output(attention_mask_.tensor);
        auto time_hidden = linear_lastdim(ctx, time_in_, weights.time_mlp_1, writer);
        time_hidden = engine::core::wrap_tensor(ggml_silu(ctx.ggml, time_hidden.tensor), time_hidden.shape, GGML_TYPE_F32);
        time_hidden = linear_lastdim(ctx, time_hidden, weights.time_mlp_2, writer);

        auto hidden_bct = cat_channels_bct(ctx, x_in_, mu_in_);
        auto repeat_spks = repeat_spks_bct(ctx, spks_in_, frames);
        hidden_bct = cat_channels_bct(ctx, hidden_bct, repeat_spks);
        hidden_bct = cat_channels_bct(ctx, hidden_bct, cond_in_);
        hidden_bct = resnet_block_forward(ctx, hidden_bct, time_hidden, weights.down_blocks[0].resnet, writer);
        auto hidden_btc = transpose_bct_btc(ctx, hidden_bct);
        for (const auto & transformer : weights.down_blocks[0].transformers) {
            hidden_btc = transformer_block_forward(ctx, hidden_btc, transformer, writer, attention_mask_);
        }
        skip_tensor_ = contiguous(ctx, transpose_bct_btc(ctx, hidden_btc));
        hidden_bct = causal_conv1d_bct(ctx, skip_tensor_, weights.down_blocks[0].downsample, writer);

        for (const auto & mid : weights.mid_blocks) {
            hidden_bct = resnet_block_forward(ctx, hidden_bct, time_hidden, mid.resnet, writer);
            hidden_btc = transpose_bct_btc(ctx, hidden_bct);
            for (const auto & transformer : mid.transformers) {
                hidden_btc = transformer_block_forward(ctx, hidden_btc, transformer, writer, attention_mask_);
            }
            hidden_bct = contiguous(ctx, transpose_bct_btc(ctx, hidden_btc));
        }

        hidden_bct = cat_channels_bct(ctx, hidden_bct, skip_tensor_);
        hidden_bct = resnet_block_forward(ctx, hidden_bct, time_hidden, weights.up_blocks[0].resnet, writer);
        hidden_btc = transpose_bct_btc(ctx, hidden_bct);
        for (const auto & transformer : weights.up_blocks[0].transformers) {
            hidden_btc = transformer_block_forward(ctx, hidden_btc, transformer, writer, attention_mask_);
        }
        hidden_bct = contiguous(ctx, transpose_bct_btc(ctx, hidden_btc));
        hidden_bct = causal_conv1d_bct(ctx, hidden_bct, weights.up_blocks[0].upsample, writer);
        hidden_bct = causal_block_forward(ctx, hidden_bct, weights.final_block, writer);
        output_tensor_ = causal_conv1d_bct(ctx, hidden_bct, weights.final_proj, writer);
        ggml_set_output(output_tensor_.tensor);

        graph_ = ggml_new_graph_custom(graph_ggml_, 262144, false);
        ggml_build_forward_expand(graph_, output_tensor_.tensor);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate backend graph tensors for S3 flow decoder");
        }
        if (execution_context_->uses_host_graph_plan()) {
            engine::core::prepare_host_graph_plan(*execution_context_, graph_, cpu_plan_);
        }
        writer.flush();
    }

    void set_conditioning(
        const S3FlowDecoderWeights & weights,
        const std::vector<float> & mu,
        const std::vector<float> & spks,
        const std::vector<float> & cond,
        int64_t frames,
        const engine::core::BackendConfig & backend_config,
        S3FlowDecoderRunTiming * timing = nullptr) {
        ensure_active_graph(weights, frames, backend_config);
        const auto started = Clock::now();
        engine::core::write_tensor_f32(mu_in_, mu);
        engine::core::write_tensor_f32(spks_in_, spks);
        engine::core::write_tensor_f32(cond_in_, cond);
        if (timing != nullptr) {
            timing->conditioning_write_ms += engine::debug::elapsed_ms(started);
        }
    }

    S3FlowDecoderOutputs run(
        const std::vector<float> & x,
        const std::vector<float> & mask,
        const std::vector<float> & t,
        S3FlowDecoderRunTiming * timing = nullptr) {
        constexpr int64_t mel_channels = 80;
        constexpr int64_t time_dim = 320;
        if (timing != nullptr) {
            ++timing->calls;
        }
        auto started = Clock::now();
        auto time_emb_values = decoder_sinusoidal_pos_emb(t, batch_, time_dim);
        if (timing != nullptr) {
            timing->time_embedding_ms += engine::debug::elapsed_ms(started);
        }
        const int64_t valid_frames = valid_frames_from_mask(mask, batch_, active_frames_);
        started = Clock::now();
        engine::core::write_tensor_f32(x_in_, x);
        ensure_attention_mask(valid_frames);
        if (timing != nullptr) {
            timing->time_write_ms += engine::debug::elapsed_ms(started);
        }
        started = Clock::now();
        engine::core::write_tensor_f32(time_in_, time_emb_values);
        if (timing != nullptr) {
            timing->input_write_ms += engine::debug::elapsed_ms(started);
        }
        started = Clock::now();
        const ggml_status status = engine::core::compute_graph(*execution_context_, graph_, cpu_plan_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for S3 flow decoder");
        }
        if (timing != nullptr) {
            timing->graph_compute_ms += engine::debug::elapsed_ms(started);
        }
        S3FlowDecoderOutputs outputs;
        started = Clock::now();
        outputs.mel = engine::core::read_tensor_f32(output_tensor_.tensor);
        if (timing != nullptr) {
            timing->output_read_ms += engine::debug::elapsed_ms(started);
        }
        outputs.channels = mel_channels;
        outputs.frames = active_frames_;
        outputs.storage_frames = active_frames_;
        return outputs;
    }

private:
    void ensure_attention_mask(int64_t valid_frames) {
        if (cached_attention_mask_valid_frames_ == valid_frames) {
            return;
        }
        cached_attention_mask_values_ = make_attention_key_mask(batch_, active_frames_, active_frames_, valid_frames, 8);
        engine::core::write_tensor_f16(attention_mask_, cached_attention_mask_values_);
        cached_attention_mask_valid_frames_ = valid_frames;
    }

    int64_t batch_ = 0;
    int64_t capacity_frames_ = 0;
    int64_t active_frames_ = 0;
    const engine::core::ExecutionContext * execution_context_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    ggml_context * graph_ggml_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    engine::core::TensorValue x_in_;
    engine::core::TensorValue mu_in_;
    engine::core::TensorValue time_in_;
    engine::core::TensorValue spks_in_;
    engine::core::TensorValue cond_in_;
    engine::core::TensorValue attention_mask_;
    engine::core::TensorValue skip_tensor_;
    engine::core::TensorValue output_tensor_;
    engine::core::HostGraphPlan cpu_plan_;
    int64_t cached_attention_mask_valid_frames_ = -1;
    std::vector<float> cached_attention_mask_values_;
};

}  // namespace

struct S3FlowSessionCache::State {
    explicit State(engine::core::BackendConfig) {}

    void prepare_encoder_capacity(
        const S3FlowEncoderWeights & weights,
        int64_t capacity,
        int64_t hidden_size,
        const engine::core::BackendConfig & backend_config) {
        if (!encoder_runner ||
            encoder_capacity_frames < capacity ||
            encoder_hidden_size != hidden_size ||
            !same_backend(encoder_backend, backend_config)) {
            encoder_runner.reset();
            encoder_runner = std::make_unique<FlowEncoderBackendRunner>(weights, capacity, hidden_size, backend_config);
            encoder_capacity_frames = capacity;
            encoder_hidden_size = hidden_size;
            encoder_backend = backend_config;
        }
    }

    void prepare_decoder_capacity(
        const S3FlowDecoderWeights & weights,
        int64_t batch,
        int64_t capacity,
        const engine::core::BackendConfig & backend_config) {
        if (!decoder_runner ||
            decoder_batch != batch ||
            decoder_capacity_frames < capacity ||
            !same_backend(decoder_backend, backend_config)) {
            decoder_runner.reset();
            decoder_runner = std::make_unique<FlowDecoderBackendRunner>(weights, batch, capacity, backend_config);
            decoder_batch = batch;
            decoder_capacity_frames = capacity;
            decoder_backend = backend_config;
        }
    }

    FlowEncoderBackendRunner & encoder_runner_for_capacity(
        const S3FlowEncoderWeights & weights,
        int64_t capacity_frames,
        int64_t hidden_size,
        const engine::core::BackendConfig & backend_config) {
        prepare_encoder_capacity(weights, capacity_frames, hidden_size, backend_config);
        return *encoder_runner;
    }

    FlowDecoderBackendRunner & decoder_runner_for_capacity(
        const S3FlowDecoderWeights & weights,
        int64_t batch,
        int64_t capacity_frames,
        const engine::core::BackendConfig & backend_config) {
        prepare_decoder_capacity(weights, batch, capacity_frames, backend_config);
        return *decoder_runner;
    }

    std::unique_ptr<FlowEncoderBackendRunner> encoder_runner;
    int64_t encoder_capacity_frames = 0;
    int64_t encoder_hidden_size = 0;
    engine::core::BackendConfig encoder_backend;
    std::unique_ptr<FlowDecoderBackendRunner> decoder_runner;
    int64_t decoder_batch = 0;
    int64_t decoder_capacity_frames = 0;
    engine::core::BackendConfig decoder_backend;
};

S3FlowSessionCache::S3FlowSessionCache(engine::core::BackendConfig backend)
    : state_(std::make_unique<State>(backend)) {}

S3FlowSessionCache::~S3FlowSessionCache() = default;
S3FlowSessionCache::S3FlowSessionCache(S3FlowSessionCache &&) noexcept = default;
S3FlowSessionCache & S3FlowSessionCache::operator=(S3FlowSessionCache &&) noexcept = default;

void S3FlowSessionCache::release_encoder_graphs() {
    state_->encoder_runner.reset();
    state_->encoder_capacity_frames = 0;
    state_->encoder_hidden_size = 0;
    state_->encoder_backend = {};
}

void S3FlowSessionCache::release_decoder_graphs() {
    state_->decoder_runner.reset();
    state_->decoder_batch = 0;
    state_->decoder_capacity_frames = 0;
    state_->decoder_backend = {};
}

std::shared_ptr<const S3FlowEncoderWeights> load_s3_flow_encoder_weights(
    const std::filesystem::path & checkpoint_path,
    const engine::core::ExecutionContext & execution_context,
    engine::assets::TensorStorageType weight_storage_type) {
    const auto source = engine::assets::open_tensor_source(checkpoint_path);
    auto weights = std::make_shared<S3FlowEncoderWeights>();
    weights->execution_context = &execution_context;
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        execution_context.backend(),
        execution_context.backend_type(),
        "chatterbox.s3_flow_encoder.weights",
        1024ull * 1024ull * 1024ull);
    weights->input_embedding_tensor = weights->store->load_tensor(
        *source,
        "flow.input_embedding.weight",
        weight_storage_type,
        {6561, 512});
    weights->speaker_affine = load_flow_linear(*weights->store, *source, "flow.spk_embed_affine_layer", 80, 192, true, weight_storage_type);
    weights->encoder_proj = load_flow_linear(*weights->store, *source, "flow.encoder_proj", 80, 512, true, weight_storage_type);
    weights->embed_linear = load_flow_linear(*weights->store, *source, "flow.encoder.embed.out.0", 512, 512, true, weight_storage_type);
    weights->embed_norm = load_flow_layer_norm(*weights->store, *source, "flow.encoder.embed.out.1", 512);
    weights->prelook_conv1 =
        load_flow_conv1d(*weights->store, *source, "flow.encoder.pre_lookahead_layer.conv1", 512, 512, 4, 1, 0, weight_storage_type);
    weights->prelook_conv2 =
        load_flow_conv1d(*weights->store, *source, "flow.encoder.pre_lookahead_layer.conv2", 512, 512, 3, 1, 0, weight_storage_type);
    auto load_layer = [&](const std::string & prefix) {
        S3FlowEncoderWeights::EncoderLayerWeights layer;
        layer.norm_mha = load_flow_layer_norm(*weights->store, *source, prefix + ".norm_mha", 512);
        layer.attn.q = load_flow_linear(*weights->store, *source, prefix + ".self_attn.linear_q", 512, 512, true, weight_storage_type);
        layer.attn.k = load_flow_linear(*weights->store, *source, prefix + ".self_attn.linear_k", 512, 512, true, weight_storage_type);
        layer.attn.v = load_flow_linear(*weights->store, *source, prefix + ".self_attn.linear_v", 512, 512, true, weight_storage_type);
        layer.attn.out = load_flow_linear(*weights->store, *source, prefix + ".self_attn.linear_out", 512, 512, true, weight_storage_type);
        layer.attn.pos = load_flow_linear(*weights->store, *source, prefix + ".self_attn.linear_pos", 512, 512, false, weight_storage_type);
        layer.attn.pos_bias_u_tensor = weights->store->load_f32_tensor(*source, prefix + ".self_attn.pos_bias_u", {8, 64});
        layer.attn.pos_bias_v_tensor = weights->store->load_f32_tensor(*source, prefix + ".self_attn.pos_bias_v", {8, 64});
        layer.norm_ff = load_flow_layer_norm(*weights->store, *source, prefix + ".norm_ff", 512);
        layer.ff.w1 = load_flow_linear(*weights->store, *source, prefix + ".feed_forward.w_1", 2048, 512, true, weight_storage_type);
        layer.ff.w2 = load_flow_linear(*weights->store, *source, prefix + ".feed_forward.w_2", 512, 2048, true, weight_storage_type);
        return layer;
    };
    for (int i = 0; i < 6; ++i) {
        weights->encoders.push_back(load_layer("flow.encoder.encoders." + std::to_string(i)));
    }
    weights->up_layer_conv = load_flow_conv1d(*weights->store, *source, "flow.encoder.up_layer.conv", 512, 512, 5, 1, 0, weight_storage_type);
    weights->up_embed_linear = load_flow_linear(*weights->store, *source, "flow.encoder.up_embed.out.0", 512, 512, true, weight_storage_type);
    weights->up_embed_norm = load_flow_layer_norm(*weights->store, *source, "flow.encoder.up_embed.out.1", 512);
    for (int i = 0; i < 4; ++i) {
        weights->up_encoders.push_back(load_layer("flow.encoder.up_encoders." + std::to_string(i)));
    }
    weights->after_norm = load_flow_layer_norm(*weights->store, *source, "flow.encoder.after_norm", 512);
    weights->store->upload();
    return weights;
}

S3FlowEncoderOutputs compute_s3_flow_encoder_forward(
    S3FlowSessionCache & cache,
    const S3FlowEncoderWeights & weights,
    const std::vector<float> & input,
    int64_t frames,
    int64_t capacity_frames,
    int64_t hidden_size,
    engine::core::BackendConfig backend) {
    return cache.state_->encoder_runner_for_capacity(weights, capacity_frames, hidden_size, backend).run(input, frames);
}

std::shared_ptr<const S3FlowDecoderWeights> load_s3_flow_decoder_weights(
    const std::filesystem::path & checkpoint_path,
    const engine::core::ExecutionContext & execution_context,
    engine::assets::TensorStorageType weight_storage_type) {
    const auto source = engine::assets::open_tensor_source(checkpoint_path);
    auto weights = std::make_shared<S3FlowDecoderWeights>();
    weights->execution_context = &execution_context;
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        execution_context.backend(),
        execution_context.backend_type(),
        "chatterbox.s3_flow_decoder.weights",
        2048ull * 1024ull * 1024ull);

    auto load_causal_block = [&](const std::string & prefix, int64_t in_channels, int64_t out_channels) {
        S3FlowDecoderWeights::CausalBlockWeights block;
        block.conv = load_decoder_conv1d(*weights->store, *source, prefix + ".block.0", out_channels, in_channels, 3, 1, true, weight_storage_type);
        block.norm = load_decoder_layer_norm(*weights->store, *source, prefix + ".block.2", out_channels);
        return block;
    };

    auto load_resnet = [&](const std::string & prefix, int64_t in_channels, int64_t out_channels) {
        S3FlowDecoderWeights::ResnetBlockWeights block;
        block.time_mlp = load_decoder_linear(*weights->store, *source, prefix + ".mlp.1", out_channels, 1024, true, weight_storage_type);
        block.block1 = load_causal_block(prefix + ".block1", in_channels, out_channels);
        block.block2 = load_causal_block(prefix + ".block2", out_channels, out_channels);
        block.res_conv = load_decoder_conv1d(*weights->store, *source, prefix + ".res_conv", out_channels, in_channels, 1, 1, true, weight_storage_type);
        return block;
    };

    auto load_transformer = [&](const std::string & prefix) {
        S3FlowDecoderWeights::TransformerBlockWeights block;
        block.norm1 = load_decoder_layer_norm(*weights->store, *source, prefix + ".norm1", 256);
        block.attn_q = load_decoder_linear(*weights->store, *source, prefix + ".attn1.to_q", 512, 256, false, weight_storage_type);
        block.attn_k = load_decoder_linear(*weights->store, *source, prefix + ".attn1.to_k", 512, 256, false, weight_storage_type);
        block.attn_v = load_decoder_linear(*weights->store, *source, prefix + ".attn1.to_v", 512, 256, false, weight_storage_type);
        block.attn_out = load_decoder_linear(*weights->store, *source, prefix + ".attn1.to_out.0", 256, 512, true, weight_storage_type);
        block.norm3 = load_decoder_layer_norm(*weights->store, *source, prefix + ".norm3", 256);
        block.ff_proj_in = load_decoder_linear(*weights->store, *source, prefix + ".ff.net.0.proj", 1024, 256, true, weight_storage_type);
        block.ff_proj_out = load_decoder_linear(*weights->store, *source, prefix + ".ff.net.2", 256, 1024, true, weight_storage_type);
        return block;
    };

    weights->time_mlp_1 =
        load_decoder_linear(*weights->store, *source, "flow.decoder.estimator.time_mlp.linear_1", 1024, 320, true, weight_storage_type);
    weights->time_mlp_2 =
        load_decoder_linear(*weights->store, *source, "flow.decoder.estimator.time_mlp.linear_2", 1024, 1024, true, weight_storage_type);

    weights->down_blocks.resize(1);
    weights->down_blocks[0].resnet = load_resnet("flow.decoder.estimator.down_blocks.0.0", 320, 256);
    for (int i = 0; i < 4; ++i) {
        weights->down_blocks[0].transformers.push_back(load_transformer("flow.decoder.estimator.down_blocks.0.1." + std::to_string(i)));
    }
    weights->down_blocks[0].downsample =
        load_decoder_conv1d(*weights->store, *source, "flow.decoder.estimator.down_blocks.0.2", 256, 256, 3, 1, true, weight_storage_type);

    weights->mid_blocks.resize(12);
    for (int block_index = 0; block_index < 12; ++block_index) {
        weights->mid_blocks[static_cast<size_t>(block_index)].resnet =
            load_resnet("flow.decoder.estimator.mid_blocks." + std::to_string(block_index) + ".0", 256, 256);
        for (int i = 0; i < 4; ++i) {
            weights->mid_blocks[static_cast<size_t>(block_index)].transformers.push_back(
                load_transformer("flow.decoder.estimator.mid_blocks." + std::to_string(block_index) + ".1." + std::to_string(i)));
        }
    }

    weights->up_blocks.resize(1);
    weights->up_blocks[0].resnet = load_resnet("flow.decoder.estimator.up_blocks.0.0", 512, 256);
    for (int i = 0; i < 4; ++i) {
        weights->up_blocks[0].transformers.push_back(load_transformer("flow.decoder.estimator.up_blocks.0.1." + std::to_string(i)));
    }
    weights->up_blocks[0].upsample =
        load_decoder_conv1d(*weights->store, *source, "flow.decoder.estimator.up_blocks.0.2", 256, 256, 3, 1, true, weight_storage_type);

    weights->final_block = load_causal_block("flow.decoder.estimator.final_block", 256, 256);
    weights->final_proj = load_decoder_conv1d(*weights->store, *source, "flow.decoder.estimator.final_proj", 80, 256, 1, 1, true, weight_storage_type);
    weights->store->upload();
    return weights;
}

S3FlowDecoderOutputs compute_s3_flow_decoder_forward(
    S3FlowSessionCache & cache,
    const S3FlowDecoderWeights & weights,
    const std::vector<float> & x,
    const std::vector<float> & mask,
    const std::vector<float> & mu,
    const std::vector<float> & t,
    const std::vector<float> & spks,
    const std::vector<float> & cond,
    int64_t batch,
    int64_t frames,
    int64_t capacity_frames,
    engine::core::BackendConfig backend) {
    const int64_t valid_frames = valid_frames_from_mask(mask, batch, frames);
    auto & runner = cache.state_->decoder_runner_for_capacity(weights, batch, capacity_frames, backend);
    runner.set_conditioning(weights, mu, spks, cond, frames, backend);
    auto outputs = runner.run(x, mask, t);
    outputs.frames = valid_frames;
    return outputs;
}

S3FlowCFMOutputs compute_s3_flow_cfm_euler(
    S3FlowSessionCache & cache,
    const S3FlowDecoderWeights & weights,
    const std::vector<float> & noise,
    const std::vector<float> & mask,
    const std::vector<float> & mu,
    const std::vector<float> & spks,
    const std::vector<float> & cond,
    int64_t batch,
    int64_t frames,
    int64_t capacity_frames,
    int64_t num_steps,
    float cfg_rate,
    bool cosine_schedule,
    engine::core::BackendConfig backend,
    S3FlowCFMTimingBreakdown * timing) {
    constexpr int64_t mel_channels = 80;
    if (timing != nullptr) {
        *timing = {};
        timing->steps = num_steps;
    }
    auto started = Clock::now();
    std::vector<float> x = noise;
    if (timing != nullptr) {
        timing->initial_state_ms += engine::debug::elapsed_ms(started);
    }

    started = Clock::now();
    std::vector<float> t_span(static_cast<size_t>(num_steps + 1), 0.0f);
    constexpr float kPi = 3.14159265358979323846f;
    for (int64_t i = 0; i <= num_steps; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(num_steps);
        if (cosine_schedule) {
            t = 1.0f - std::cos(t * 0.5f * kPi);
        }
        t_span[static_cast<size_t>(i)] = t;
    }
    if (timing != nullptr) {
        timing->schedule_ms += engine::debug::elapsed_ms(started);
    }

    const size_t batch_stride = static_cast<size_t>(mel_channels * frames);
    const int64_t valid_frames = valid_frames_from_mask(mask, batch, frames);
    if (timing != nullptr) {
        timing->decoder_calls = num_steps;
    }

    const int64_t cfg_batch = batch * 2;
    started = Clock::now();
    auto & runner = cache.state_->decoder_runner_for_capacity(weights, cfg_batch, capacity_frames, backend);
    if (timing != nullptr) {
        timing->runner_setup_ms += engine::debug::elapsed_ms(started);
    }
    started = Clock::now();
    std::vector<float> cfg_mu(static_cast<size_t>(cfg_batch) * batch_stride, 0.0f);
    std::vector<float> cfg_cond(static_cast<size_t>(cfg_batch) * batch_stride, 0.0f);
    std::vector<float> cfg_mask(static_cast<size_t>(cfg_batch * frames), 0.0f);
    std::vector<float> cfg_spks(static_cast<size_t>(cfg_batch * mel_channels), 0.0f);
    if (timing != nullptr) {
        timing->zero_conditioning_ms += engine::debug::elapsed_ms(started);
    }
    std::copy(mu.begin(), mu.end(), cfg_mu.begin());
    std::copy(cond.begin(), cond.end(), cfg_cond.begin());
    std::copy(mask.begin(), mask.end(), cfg_mask.begin());
    std::copy(mask.begin(), mask.end(), cfg_mask.begin() + static_cast<ptrdiff_t>(mask.size()));
    std::copy(spks.begin(), spks.end(), cfg_spks.begin());
    runner.set_conditioning(
        weights,
        cfg_mu,
        cfg_spks,
        cfg_cond,
        frames,
        backend,
        timing == nullptr ? nullptr : &timing->conditioned);
    std::vector<float> cfg_x(static_cast<size_t>(cfg_batch) * batch_stride, 0.0f);
    std::vector<float> cfg_t(static_cast<size_t>(cfg_batch), 0.0f);
    std::vector<float> cfg_dxdt(batch_stride * static_cast<size_t>(batch), 0.0f);
    for (int64_t step = 0; step < num_steps; ++step) {
        const float t0 = t_span[static_cast<size_t>(step)];
        const float t1 = t_span[static_cast<size_t>(step + 1)];
        const float dt = t1 - t0;
        std::copy(x.begin(), x.end(), cfg_x.begin());
        std::copy(x.begin(), x.end(), cfg_x.begin() + static_cast<ptrdiff_t>(x.size()));
        std::fill(cfg_t.begin(), cfg_t.end(), t0);
        const auto cfg_outputs = runner.run(
            cfg_x,
            cfg_mask,
            cfg_t,
            timing == nullptr ? nullptr : &timing->conditioned);
        const float * conditioned_ptr = cfg_outputs.mel.data();
        const float * unconditioned_ptr = conditioned_ptr + static_cast<ptrdiff_t>(x.size());
        started = Clock::now();
        for (size_t i = 0; i < cfg_dxdt.size(); ++i) {
            cfg_dxdt[i] =
                (1.0f + cfg_rate) * conditioned_ptr[i] -
                cfg_rate * unconditioned_ptr[i];
            x[i] += dt * cfg_dxdt[i];
        }
        if (timing != nullptr) {
            timing->host_update_ms += engine::debug::elapsed_ms(started);
        }
    }

    S3FlowCFMOutputs outputs;
    outputs.mel = std::move(x);
    outputs.channels = mel_channels;
    outputs.frames = valid_frames;
    outputs.storage_frames = frames;
    return outputs;
}

}  // namespace engine::models::chatterbox
