#include "engine/framework/modules/wavlm_encoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::modules {
namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t kMaxPositionBiasCacheEntries = 8;

int64_t tensor_elements(const std::vector<int64_t> & shape) {
    if (shape.empty()) {
        throw std::runtime_error("WavLM tensor shape is empty");
    }
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, [](int64_t lhs, int64_t rhs) {
        if (rhs <= 0) {
            throw std::runtime_error("WavLM tensor shape contains non-positive dimension");
        }
        return lhs * rhs;
    });
}

bool has_suffix(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void validate_config(const WavlmEncoderConfig & config) {
    if (config.hidden_size <= 0 || config.intermediate_size <= 0 || config.num_hidden_layers <= 0 ||
        config.output_hidden_layer <= 0 || config.num_attention_heads <= 0 || config.num_buckets <= 0 ||
        config.max_distance <= 0 || config.num_conv_pos_embeddings <= 0 ||
        config.num_conv_pos_embedding_groups <= 0) {
        throw std::runtime_error("WavLM config contains non-positive dimensions");
    }
    if (config.output_hidden_layer > config.num_hidden_layers) {
        throw std::runtime_error("WavLM output layer cannot exceed hidden layer count");
    }
    if (config.hidden_size % config.num_attention_heads != 0 ||
        config.hidden_size % config.num_conv_pos_embedding_groups != 0) {
        throw std::runtime_error("WavLM hidden size must be divisible by head and positional-conv group counts");
    }
    if (config.conv_dim.empty() || config.conv_dim.size() != config.conv_kernel.size() ||
        config.conv_dim.size() != config.conv_stride.size()) {
        throw std::runtime_error("WavLM convolution config is inconsistent");
    }
    for (const int64_t value : config.conv_dim) {
        if (value <= 0) {
            throw std::runtime_error("WavLM convolution dimensions must be positive");
        }
    }
    for (const int64_t value : config.conv_kernel) {
        if (value <= 0) {
            throw std::runtime_error("WavLM convolution kernels must be positive");
        }
    }
    for (const int64_t value : config.conv_stride) {
        if (value <= 0) {
            throw std::runtime_error("WavLM convolution strides must be positive");
        }
    }
}

core::TensorValue require_tensor(const WavlmEncoderWeights & weights, const std::string & name) {
    const auto it = weights.tensors.find(name);
    if (it == weights.tensors.end()) {
        throw std::runtime_error("WavLM missing tensor: " + name);
    }
    return it->second;
}

NormWeights norm_weights(const WavlmEncoderWeights & weights, const std::string & prefix) {
    return NormWeights{
        require_tensor(weights, prefix + ".weight"),
        require_tensor(weights, prefix + ".bias")};
}

LinearWeights linear_weights(const WavlmEncoderWeights & weights, const std::string & prefix) {
    return LinearWeights{
        require_tensor(weights, prefix + ".weight"),
        require_tensor(weights, prefix + ".bias")};
}

Conv1dWeights conv1d_weights(const WavlmEncoderWeights & weights, const std::string & prefix, bool use_bias) {
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

core::TensorValue ensure_f32(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    if (value.type == GGML_TYPE_F32) {
        return value;
    }
    return core::wrap_tensor(ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F32), value.shape, GGML_TYPE_F32);
}

core::TensorValue transpose_bct_btc(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return TransposeModule({{0, 2, 1, 3}, value.shape.rank}).build(ctx, value);
}

core::TensorValue add_same(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    return AddModule().build(ctx, lhs, rhs);
}

core::TensorValue add_scalar(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    float scalar) {
    return core::wrap_tensor(ggml_scale_bias(ctx.ggml, contiguous(ctx, value).tensor, 1.0F, scalar), value.shape, GGML_TYPE_F32);
}

core::TensorValue mul_scalar(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    float scalar) {
    return core::wrap_tensor(ggml_scale(ctx.ggml, contiguous(ctx, value).tensor, scalar), value.shape, GGML_TYPE_F32);
}

int64_t conv1d_output_frames(int64_t input_frames, int64_t kernel, int64_t stride, int64_t padding) {
    return (input_frames + 2 * padding - kernel) / stride + 1;
}

core::TensorValue group_norm_affine(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t groups,
    float eps,
    const NormWeights & weights) {
    core::TensorValue output;
    if (input.shape.rank == 3 && groups == input.shape.dims[1]) {
        auto input_f32 = input.type == GGML_TYPE_F32
            ? input
            : core::wrap_tensor(ggml_cast(ctx.ggml, input.tensor, GGML_TYPE_F32), input.shape, GGML_TYPE_F32);
        auto mean = ReduceMeanModule({2}).build(ctx, input_f32);
        auto mean_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, mean.tensor, input_f32.tensor), input_f32.shape, GGML_TYPE_F32);
        auto centered = core::wrap_tensor(ggml_sub(ctx.ggml, input_f32.tensor, mean_rep.tensor), input_f32.shape, GGML_TYPE_F32);
        auto variance = ReduceMeanModule({2}).build(ctx, MulModule().build(ctx, centered, centered));
        auto stddev = core::wrap_tensor(
            ggml_sqrt(ctx.ggml, ggml_scale_bias(ctx.ggml, variance.tensor, 1.0F, eps)),
            variance.shape,
            GGML_TYPE_F32);
        auto stddev_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, stddev.tensor, input_f32.tensor), input_f32.shape, GGML_TYPE_F32);
        output = core::wrap_tensor(ggml_div(ctx.ggml, centered.tensor, stddev_rep.tensor), input_f32.shape, GGML_TYPE_F32);
    } else {
        output = core::wrap_tensor(ggml_group_norm(ctx.ggml, input.tensor, groups, eps), input.shape, GGML_TYPE_F32);
    }
    if (weights.weight.has_value()) {
        auto weight = core::reshape_tensor(ctx, ensure_f32(ctx, *weights.weight), core::TensorShape::from_dims({1, input.shape.dims[1], 1}));
        auto repeated = core::wrap_tensor(ggml_repeat(ctx.ggml, weight.tensor, output.tensor), output.shape, GGML_TYPE_F32);
        output = core::wrap_tensor(ggml_mul(ctx.ggml, output.tensor, repeated.tensor), output.shape, GGML_TYPE_F32);
    }
    if (weights.bias.has_value()) {
        auto bias = core::reshape_tensor(ctx, ensure_f32(ctx, *weights.bias), core::TensorShape::from_dims({1, input.shape.dims[1], 1}));
        auto repeated = core::wrap_tensor(ggml_repeat(ctx.ggml, bias.tensor, output.tensor), output.shape, GGML_TYPE_F32);
        output = core::wrap_tensor(ggml_add(ctx.ggml, output.tensor, repeated.tensor), output.shape, GGML_TYPE_F32);
    }
    return output;
}

core::TensorValue masked_group_norm_affine(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t groups,
    float eps,
    const NormWeights & weights,
    const core::TensorValue & time_mask,
    const core::TensorValue & inv_valid_time_count) {
    if (input.shape.rank != 3 || groups != input.shape.dims[1]) {
        return group_norm_affine(ctx, input, groups, eps, weights);
    }
    core::validate_shape(
        time_mask,
        core::TensorShape::from_dims({input.shape.dims[0], 1, input.shape.dims[2]}),
        "WavLM group-norm time mask");
    core::validate_shape(
        inv_valid_time_count,
        core::TensorShape::from_dims({input.shape.dims[0], 1, 1}),
        "WavLM group-norm inverse count");
    if (time_mask.type != GGML_TYPE_F32 || inv_valid_time_count.type != GGML_TYPE_F32) {
        throw std::runtime_error("WavLM group-norm mask tensors must be F32");
    }

    auto input_f32 = input.type == GGML_TYPE_F32
        ? input
        : core::wrap_tensor(ggml_cast(ctx.ggml, input.tensor, GGML_TYPE_F32), input.shape, GGML_TYPE_F32);
    auto mask_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, time_mask.tensor, input_f32.tensor), input_f32.shape, GGML_TYPE_F32);
    auto masked_input = core::wrap_tensor(ggml_mul(ctx.ggml, input_f32.tensor, mask_rep.tensor), input_f32.shape, GGML_TYPE_F32);
    auto sum = ReduceSumModule({2}).build(ctx, masked_input);
    auto inv_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, inv_valid_time_count.tensor, sum.tensor), sum.shape, GGML_TYPE_F32);
    auto mean = core::wrap_tensor(ggml_mul(ctx.ggml, sum.tensor, inv_rep.tensor), sum.shape, GGML_TYPE_F32);
    auto mean_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, mean.tensor, input_f32.tensor), input_f32.shape, GGML_TYPE_F32);
    auto centered = core::wrap_tensor(ggml_sub(ctx.ggml, input_f32.tensor, mean_rep.tensor), input_f32.shape, GGML_TYPE_F32);
    auto squared = MulModule().build(ctx, centered, centered);
    auto masked_squared = core::wrap_tensor(ggml_mul(ctx.ggml, squared.tensor, mask_rep.tensor), squared.shape, GGML_TYPE_F32);
    auto variance_sum = ReduceSumModule({2}).build(ctx, masked_squared);
    auto variance = core::wrap_tensor(ggml_mul(ctx.ggml, variance_sum.tensor, inv_rep.tensor), variance_sum.shape, GGML_TYPE_F32);
    auto stddev = core::wrap_tensor(
        ggml_sqrt(ctx.ggml, ggml_scale_bias(ctx.ggml, variance.tensor, 1.0F, eps)),
        variance.shape,
        GGML_TYPE_F32);
    auto stddev_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, stddev.tensor, input_f32.tensor), input_f32.shape, GGML_TYPE_F32);
    auto output = core::wrap_tensor(ggml_div(ctx.ggml, centered.tensor, stddev_rep.tensor), input_f32.shape, GGML_TYPE_F32);
    output = core::wrap_tensor(ggml_mul(ctx.ggml, output.tensor, mask_rep.tensor), output.shape, GGML_TYPE_F32);
    if (weights.weight.has_value()) {
        auto weight = core::reshape_tensor(ctx, ensure_f32(ctx, *weights.weight), core::TensorShape::from_dims({1, input.shape.dims[1], 1}));
        auto repeated = core::wrap_tensor(ggml_repeat(ctx.ggml, weight.tensor, output.tensor), output.shape, GGML_TYPE_F32);
        output = core::wrap_tensor(ggml_mul(ctx.ggml, output.tensor, repeated.tensor), output.shape, GGML_TYPE_F32);
    }
    if (weights.bias.has_value()) {
        auto bias = core::reshape_tensor(ctx, ensure_f32(ctx, *weights.bias), core::TensorShape::from_dims({1, input.shape.dims[1], 1}));
        auto repeated = core::wrap_tensor(ggml_repeat(ctx.ggml, bias.tensor, output.tensor), output.shape, GGML_TYPE_F32);
        output = core::wrap_tensor(ggml_add(ctx.ggml, output.tensor, repeated.tensor), output.shape, GGML_TYPE_F32);
        output = core::wrap_tensor(ggml_mul(ctx.ggml, output.tensor, mask_rep.tensor), output.shape, GGML_TYPE_F32);
    }
    return output;
}

std::vector<float> effective_weight_norm_conv1d(
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_size) {
    const auto g = source.require_f32(prefix + ".weight_g", {1, 1, kernel_size});
    const auto v = source.require_f32(prefix + ".weight_v", {out_channels, in_channels, kernel_size});
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
            throw std::runtime_error("WavLM positional-conv weight norm is zero");
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

core::TensorValue grouped_pos_conv(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const Conv1dWeights & weights,
    const WavlmEncoderConfig & config) {
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

int64_t relative_position_bucket(int64_t relative_position, const WavlmEncoderConfig & config) {
    int64_t num_buckets = config.num_buckets / 2;
    int64_t bucket = 0;
    if (relative_position > 0) {
        bucket += num_buckets;
    }
    int64_t distance = std::llabs(relative_position);
    const int64_t max_exact = num_buckets / 2;
    if (distance < max_exact) {
        return bucket + distance;
    }
    const double log_ratio = std::log(static_cast<double>(distance) / static_cast<double>(max_exact)) /
        std::log(static_cast<double>(config.max_distance) / static_cast<double>(max_exact));
    int64_t large_bucket = max_exact + static_cast<int64_t>(log_ratio * static_cast<double>(num_buckets - max_exact));
    large_bucket = std::min(large_bucket, num_buckets - 1);
    return bucket + large_bucket;
}

std::vector<float> compute_position_bias_from_values(
    const std::vector<float> & rel_embedding,
    const WavlmEncoderConfig & config,
    int64_t batch,
    int64_t tokens) {
    std::vector<float> out(static_cast<size_t>(batch * config.num_attention_heads * tokens * tokens), 0.0F);
    std::vector<int64_t> bucket_by_offset(static_cast<size_t>(2 * tokens - 1));
    for (int64_t offset = 1 - tokens; offset < tokens; ++offset) {
        bucket_by_offset[static_cast<size_t>(offset + tokens - 1)] = relative_position_bucket(offset, config);
    }
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < config.num_attention_heads; ++h) {
            auto * plane = out.data() + static_cast<size_t>((b * config.num_attention_heads + h) * tokens * tokens);
            for (int64_t q = 0; q < tokens; ++q) {
                auto * row = plane + static_cast<size_t>(q * tokens);
                for (int64_t k = 0; k < tokens; ++k) {
                    const int64_t bucket = bucket_by_offset[static_cast<size_t>(k - q + tokens - 1)];
                    row[k] = rel_embedding[static_cast<size_t>(bucket * config.num_attention_heads + h)];
                }
            }
        }
    }
    return out;
}

} // namespace

class WavlmPositionBiasCache final {
public:
    std::shared_ptr<const std::vector<float>> values_for(
        const std::vector<float> & rel_embedding,
        const WavlmEncoderConfig & config,
        int64_t batch,
        int64_t tokens,
        bool & cache_hit) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->batch == batch && it->tokens == tokens) {
                cache_hit = true;
                auto values = it->values;
                if (it != entries_.begin()) {
                    auto entry = std::move(*it);
                    entries_.erase(it);
                    entries_.insert(entries_.begin(), std::move(entry));
                }
                return values;
            }
        }

        cache_hit = false;
        auto values = std::make_shared<std::vector<float>>(
            compute_position_bias_from_values(rel_embedding, config, batch, tokens));
        Entry entry;
        entry.batch = batch;
        entry.tokens = tokens;
        entry.values = values;
        entries_.insert(entries_.begin(), std::move(entry));
        if (entries_.size() > kMaxPositionBiasCacheEntries) {
            entries_.pop_back();
        }
        return values;
    }

private:
    struct Entry {
        int64_t batch = 0;
        int64_t tokens = 0;
        std::shared_ptr<const std::vector<float>> values;
    };

    std::mutex mutex_;
    std::vector<Entry> entries_;
};

namespace {

core::TensorValue build_wavlm_self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden_btc,
    const core::TensorValue & position_bias,
    const core::TensorValue & attention_mask,
    const WavlmEncoderWeights & weights,
    int64_t layer_index) {
    const auto & config = weights.config;
    const int64_t batch = hidden_btc.shape.dims[0];
    const int64_t tokens = hidden_btc.shape.dims[1];
    const int64_t head_dim = config.hidden_size / config.num_attention_heads;
    const std::string prefix = "encoder.layers." + std::to_string(layer_index) + ".attention";

    auto q = LinearModule({config.hidden_size, config.hidden_size, true, GGML_PREC_F32})
                 .build(ctx, hidden_btc, linear_weights(weights, prefix + ".q_proj"));
    auto k = LinearModule({config.hidden_size, config.hidden_size, true, GGML_PREC_F32})
                 .build(ctx, hidden_btc, linear_weights(weights, prefix + ".k_proj"));
    auto v = LinearModule({config.hidden_size, config.hidden_size, true, GGML_PREC_F32})
                 .build(ctx, hidden_btc, linear_weights(weights, prefix + ".v_proj"));

    q = core::reshape_tensor(ctx, contiguous(ctx, q), core::TensorShape::from_dims({batch, tokens, config.num_attention_heads, head_dim}));
    k = core::reshape_tensor(ctx, contiguous(ctx, k), core::TensorShape::from_dims({batch, tokens, config.num_attention_heads, head_dim}));
    v = core::reshape_tensor(ctx, contiguous(ctx, v), core::TensorShape::from_dims({batch, tokens, config.num_attention_heads, head_dim}));
    q = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    k = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
    v = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);

    auto query_layer = core::reshape_tensor(
        ctx,
        contiguous(ctx, hidden_btc),
        core::TensorShape::from_dims({batch, tokens, config.num_attention_heads, head_dim}));
    query_layer = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, query_layer);
    auto gates = LinearModule({head_dim, 8, true, GGML_PREC_F32})
                     .build(ctx, query_layer, linear_weights(weights, prefix + ".gru_rel_pos_linear"));
    auto gate_a = SliceModule({3, 0, 4}).build(ctx, gates);
    auto gate_b = SliceModule({3, 4, 4}).build(ctx, gates);
    gate_a = SigmoidModule().build(ctx, ReduceSumModule({3}).build(ctx, gate_a));
    gate_b = SigmoidModule().build(ctx, ReduceSumModule({3}).build(ctx, gate_b));
    gate_a = core::reshape_tensor(ctx, contiguous(ctx, gate_a), core::TensorShape::from_dims({batch, config.num_attention_heads, tokens, 1}));
    gate_b = core::reshape_tensor(ctx, contiguous(ctx, gate_b), core::TensorShape::from_dims({batch, config.num_attention_heads, tokens, 1}));
    auto rel_const = core::reshape_tensor(
        ctx,
        require_tensor(weights, prefix + ".gru_rel_pos_const"),
        core::TensorShape::from_dims({1, config.num_attention_heads, 1, 1}));
    rel_const = RepeatModule({gate_b.shape}).build(ctx, rel_const);
    auto gated = MulModule().build(ctx, gate_b, rel_const);
    gated = add_scalar(ctx, gated, -1.0F);
    gated = MulModule().build(ctx, gate_a, gated);
    gated = add_scalar(ctx, gated, 2.0F);
    gated = RepeatModule({position_bias.shape}).build(ctx, gated);
    auto rel_bias = MulModule().build(ctx, gated, position_bias);

    const auto k_t = TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k);
    auto scores = MatMulModule().build(ctx, q, k_t);
    scores = mul_scalar(ctx, scores, static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim))));
    scores = add_same(ctx, scores, rel_bias);
    auto attn = core::wrap_tensor(
        ggml_soft_max_ext(ctx.ggml, contiguous(ctx, scores).tensor, attention_mask.tensor, 1.0F, 0.0F),
        scores.shape,
        GGML_TYPE_F32);
    auto context = MatMulModule().build(ctx, attn, v);
    context = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, context);
    context = core::reshape_tensor(ctx, contiguous(ctx, context), core::TensorShape::from_dims({batch, tokens, config.hidden_size}));
    return LinearModule({config.hidden_size, config.hidden_size, true, GGML_PREC_F32})
        .build(ctx, context, linear_weights(weights, prefix + ".out_proj"));
}

core::TensorValue build_feed_forward(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden_btc,
    const WavlmEncoderWeights & weights,
    int64_t layer_index) {
    const auto & config = weights.config;
    const std::string prefix = "encoder.layers." + std::to_string(layer_index);
    auto x = LinearModule({config.hidden_size, config.intermediate_size, true, GGML_PREC_F32})
                 .build(ctx, hidden_btc, linear_weights(weights, prefix + ".fc1"));
    x = GeluModule({GeluApproximation::ExactErf}).build(ctx, x);
    return LinearModule({config.intermediate_size, config.hidden_size, true, GGML_PREC_F32})
        .build(ctx, x, linear_weights(weights, prefix + ".fc2"));
}

std::vector<core::TensorValue> build_wavlm_graph_layers(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_values,
    const core::TensorValue & first_conv_time_mask,
    const core::TensorValue & first_conv_inv_valid_time_count,
    const core::TensorValue & token_mask,
    const core::TensorValue & position_bias,
    const core::TensorValue & attention_mask,
    const WavlmEncoderWeights & weights,
    const std::vector<int64_t> & output_layers) {
    const auto & config = weights.config;
    if (output_layers.empty()) {
        throw std::runtime_error("WavLM graph requires at least one output layer");
    }
    int64_t max_output_layer = 0;
    for (const int64_t layer : output_layers) {
        if (layer <= 0 || layer > config.num_hidden_layers) {
            throw std::runtime_error("WavLM output layer is out of range");
        }
        max_output_layer = std::max(max_output_layer, layer);
    }
    auto hidden = core::reshape_tensor(ctx, input_values, core::TensorShape::from_dims({input_values.shape.dims[0], 1, input_values.shape.dims[1]}));
    int64_t in_channels = 1;
    for (size_t index = 0; index < config.conv_dim.size(); ++index) {
        const std::string prefix = "feature_extractor.conv_layers." + std::to_string(index);
        hidden = Conv1dModule({
            in_channels,
            config.conv_dim[index],
            config.conv_kernel[index],
            static_cast<int>(config.conv_stride[index]),
            0,
            1,
            false}).build(ctx, hidden, conv1d_weights(weights, prefix + ".conv", false));
        if (index == 0) {
            hidden = masked_group_norm_affine(
                ctx,
                hidden,
                config.conv_dim[index],
                config.layer_norm_eps,
                norm_weights(weights, prefix + ".layer_norm"),
                first_conv_time_mask,
                first_conv_inv_valid_time_count);
        }
        hidden = GeluModule({GeluApproximation::ExactErf}).build(ctx, hidden);
        in_channels = config.conv_dim[index];
    }

    hidden = transpose_bct_btc(ctx, hidden);
    hidden = LayerNormModule({config.conv_dim.back(), config.layer_norm_eps, true, true})
                 .build(ctx, hidden, norm_weights(weights, "feature_projection.layer_norm"));
    hidden = LinearModule({config.conv_dim.back(), config.hidden_size, true, GGML_PREC_F32})
                 .build(ctx, hidden, linear_weights(weights, "feature_projection.projection"));
    hidden = MaskingModule().build(ctx, hidden, token_mask);

    auto pos = grouped_pos_conv(
        ctx,
        transpose_bct_btc(ctx, hidden),
        conv1d_weights(weights, "encoder.pos_conv_embed.conv", true),
        config);
    hidden = add_same(ctx, hidden, transpose_bct_btc(ctx, pos));
    hidden = LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                 .build(ctx, hidden, norm_weights(weights, "encoder.layer_norm"));
    hidden = MaskingModule().build(ctx, hidden, token_mask);

    std::vector<core::TensorValue> outputs;
    outputs.reserve(output_layers.size());
    for (int64_t layer = 0; layer < max_output_layer; ++layer) {
        const std::string prefix = "encoder.layers." + std::to_string(layer);
        auto attn = build_wavlm_self_attention(ctx, hidden, position_bias, attention_mask, weights, layer);
        hidden = add_same(ctx, hidden, attn);
        hidden = LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                     .build(ctx, hidden, norm_weights(weights, prefix + ".self_attn_layer_norm"));
        hidden = MaskingModule().build(ctx, hidden, token_mask);
        const auto ff_in = hidden;
        auto ff = build_feed_forward(ctx, hidden, weights, layer);
        hidden = add_same(ctx, ff_in, ff);
        hidden = LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                     .build(ctx, hidden, norm_weights(weights, prefix + ".final_layer_norm"));
        hidden = MaskingModule().build(ctx, hidden, token_mask);
        const int64_t layer_index = layer + 1;
        if (std::find(output_layers.begin(), output_layers.end(), layer_index) != output_layers.end()) {
            outputs.push_back(hidden);
        }
    }
    return outputs;
}

void load_tensor_alias(
    WavlmEncoderWeights & weights,
    const assets::TensorSource & source,
    const std::string & internal_name,
    const std::string & source_name,
    const std::vector<int64_t> & shape) {
    const bool broadcast_affine =
        has_suffix(internal_name, ".bias") ||
        has_suffix(internal_name, "layer_norm.weight") ||
        has_suffix(internal_name, "gru_rel_pos_const");
    const auto storage_type = broadcast_affine
        ? assets::TensorStorageType::F32
        : weights.config.weight_storage_type;
    weights.tensors.emplace(
        internal_name,
        weights.store->load_tensor(source, source_name, storage_type, shape));
    weights.parameter_count += tensor_elements(shape);
    ++weights.loaded_tensor_count;
}

class WavlmRunner {
public:
    explicit WavlmRunner(std::shared_ptr<const WavlmEncoderWeights> weights, bool reuse_graph)
        : weights_(std::move(weights)),
          reuse_graph_(reuse_graph) {
        if (weights_ == nullptr || weights_->execution_context == nullptr) {
            throw std::runtime_error("WavLM runner requires weights and execution context");
        }
        const int64_t expected_rel_values = weights_->config.num_buckets * weights_->config.num_attention_heads;
        if (static_cast<int64_t>(weights_->relative_attention_embedding.size()) != expected_rel_values) {
            throw std::runtime_error("WavLM runner relative attention embedding size mismatch");
        }
    }

    ~WavlmRunner() {
        release_graph();
    }

    WavlmEncoderOutput encode(const std::vector<float> & input_values, int64_t batch, int64_t samples) {
        auto layers = encode_layers(input_values, batch, samples, {weights_->config.output_hidden_layer});
        WavlmEncoderOutput out;
        out.hidden_states = std::move(layers.hidden_states.front());
        out.batch = layers.batch;
        out.tokens = layers.tokens;
        out.hidden_size = layers.hidden_size;
        return out;
    }

    WavlmEncoderLayerOutput encode_layers(
        const std::vector<float> & input_values,
        int64_t batch,
        int64_t samples,
        std::vector<int64_t> output_layers) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (batch != 1) {
            throw std::runtime_error("WavLM encoder currently requires batch size 1");
        }
        if (samples <= 0 || static_cast<int64_t>(input_values.size()) != batch * samples) {
            throw std::runtime_error("WavLM encoder input size mismatch");
        }
        if (output_layers.empty()) {
            throw std::runtime_error("WavLM encoder requires output layers");
        }
        std::sort(output_layers.begin(), output_layers.end());
        output_layers.erase(std::unique(output_layers.begin(), output_layers.end()), output_layers.end());
        if (!reuse_graph_) {
            release_graph();
        }
        ensure_graph(batch, samples, output_layers);
        const int64_t actual_tokens = wavlm_output_frames(samples);
        if (actual_tokens > token_capacity_) {
            throw std::runtime_error("WavLM request token count exceeds graph capacity");
        }
        auto timing_start = Clock::now();
        std::vector<float> padded_input(static_cast<size_t>(sample_capacity_), 0.0F);
        std::copy(input_values.begin(), input_values.end(), padded_input.begin());
        core::write_tensor_f32(input_, padded_input);
        engine::debug::timing_log_scalar("framework.wavlm.input_upload_ms", engine::debug::elapsed_ms(timing_start));
        timing_start = Clock::now();
        upload_runtime_masks(samples, actual_tokens);
        engine::debug::timing_log_scalar("framework.wavlm.mask_upload_ms", engine::debug::elapsed_ms(timing_start));
        timing_start = Clock::now();
        if (engine::core::compute_backend_graph(weights_->execution_context->backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for WavLM encoder");
        }
        engine::debug::timing_log_scalar("framework.wavlm.graph.compute_ms", engine::debug::elapsed_ms(timing_start));
        WavlmEncoderLayerOutput out;
        out.layer_indices = output_layers_;
        out.hidden_states.reserve(outputs_.size());
        timing_start = Clock::now();
        for (const auto & output : outputs_) {
            auto values = core::read_tensor_f32(output.tensor);
            values.resize(static_cast<size_t>(actual_tokens * output.shape.dims[2]));
            out.hidden_states.push_back(std::move(values));
        }
        engine::debug::timing_log_scalar("framework.wavlm.output_read_ms", engine::debug::elapsed_ms(timing_start));
        out.batch = batch;
        out.tokens = actual_tokens;
        out.hidden_size = outputs_.front().shape.dims[2];
        return out;
    }

    void release_runtime_graph() {
        std::lock_guard<std::mutex> lock(mutex_);
        release_graph();
    }

private:
    void release_graph() {
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
            buffer_ = nullptr;
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
            ggml_ = nullptr;
        }
        graph_ = nullptr;
        input_ = {};
        first_conv_time_mask_ = {};
        first_conv_inv_valid_time_count_ = {};
        token_mask_ = {};
        attention_mask_ = {};
        position_bias_ = {};
        outputs_.clear();
        output_layers_.clear();
        batch_ = 0;
        sample_capacity_ = 0;
        first_conv_frame_capacity_ = 0;
        token_capacity_ = 0;
    }

    std::shared_ptr<const std::vector<float>> position_bias_values_for_capacity(
        int64_t batch,
        int64_t tokens,
        bool & cache_hit) const {
        if (weights_->position_bias_cache == nullptr) {
            throw std::runtime_error("WavLM position-bias cache is not initialized");
        }
        return weights_->position_bias_cache->values_for(
            weights_->relative_attention_embedding,
            weights_->config,
            batch,
            tokens,
            cache_hit);
    }

    void ensure_graph(int64_t batch, int64_t samples, const std::vector<int64_t> & output_layers) {
        if (ggml_ != nullptr && batch_ == batch && sample_capacity_ >= samples && output_layers_ == output_layers) {
            engine::debug::timing_log_scalar("framework.wavlm.graph.rebuilt", false);
            engine::debug::timing_log_scalar("framework.wavlm.graph.reused", true);
            engine::debug::timing_log_scalar("framework.wavlm.graph.build_ms", 0.0);
            engine::debug::timing_log_scalar("framework.wavlm.position_bias.cache_hit", true);
            engine::debug::timing_log_scalar("framework.wavlm.position_bias_ms", 0.0);
            engine::debug::timing_log_scalar("framework.wavlm.sample_capacity", sample_capacity_);
            engine::debug::timing_log_scalar("framework.wavlm.token_capacity", token_capacity_);
            return;
        }
        const auto build_start = Clock::now();
        release_graph();
        ggml_init_params params{
            1536ull * 1024ull * 1024ull,
            nullptr,
            true,
        };
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize WavLM graph context");
        }
        core::ModuleBuildContext ctx{
            ggml_,
            "framework.wavlm.encode",
            weights_->execution_context->config().type};
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, samples}));
        ggml_set_input(input_.tensor);
        const int64_t first_conv_frames = wavlm_first_conv_output_frames(samples);
        first_conv_time_mask_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({batch, 1, first_conv_frames}));
        ggml_set_input(first_conv_time_mask_.tensor);
        first_conv_inv_valid_time_count_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({batch, 1, 1}));
        ggml_set_input(first_conv_inv_valid_time_count_.tensor);
        const int64_t tokens = wavlm_output_frames(samples);
        token_mask_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({batch, tokens}));
        ggml_set_input(token_mask_.tensor);
        attention_mask_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({batch, 1, tokens, tokens}));
        ggml_set_input(attention_mask_.tensor);
        position_bias_ = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({batch, weights_->config.num_attention_heads, tokens, tokens}));
        outputs_ = build_wavlm_graph_layers(
            ctx,
            input_,
            first_conv_time_mask_,
            first_conv_inv_valid_time_count_,
            token_mask_,
            position_bias_,
            attention_mask_,
            *weights_,
            output_layers);
        graph_ = ggml_new_graph_custom(ggml_, 262144, false);
        for (const auto & output : outputs_) {
            ggml_set_output(output.tensor);
            ggml_build_forward_expand(graph_, output.tensor);
        }
        buffer_ = ggml_backend_alloc_ctx_tensors(ggml_, weights_->execution_context->backend());
        if (buffer_ == nullptr) {
            release_graph();
            throw std::runtime_error("failed to allocate WavLM graph tensors");
        }
        const auto position_bias_start = Clock::now();
        bool position_bias_cache_hit = false;
        const auto position_bias_values = position_bias_values_for_capacity(batch, tokens, position_bias_cache_hit);
        core::write_tensor_f32(position_bias_, *position_bias_values);
        engine::debug::timing_log_scalar("framework.wavlm.position_bias.cache_hit", position_bias_cache_hit);
        engine::debug::timing_log_scalar("framework.wavlm.position_bias_ms", engine::debug::elapsed_ms(position_bias_start));
        output_layers_ = output_layers;
        batch_ = batch;
        sample_capacity_ = samples;
        first_conv_frame_capacity_ = first_conv_frames;
        token_capacity_ = tokens;
        engine::debug::timing_log_scalar("framework.wavlm.graph.rebuilt", true);
        engine::debug::timing_log_scalar("framework.wavlm.graph.reused", false);
        engine::debug::timing_log_scalar("framework.wavlm.graph.build_ms", engine::debug::elapsed_ms(build_start));
        engine::debug::timing_log_scalar("framework.wavlm.sample_capacity", sample_capacity_);
        engine::debug::timing_log_scalar("framework.wavlm.token_capacity", token_capacity_);
    }

    void upload_runtime_masks(int64_t actual_samples, int64_t actual_tokens) {
        const int64_t actual_first_conv_frames = wavlm_first_conv_output_frames(actual_samples);
        std::vector<float> first_conv_time_mask(static_cast<size_t>(first_conv_frame_capacity_), 0.0F);
        std::fill(first_conv_time_mask.begin(), first_conv_time_mask.begin() + actual_first_conv_frames, 1.0F);
        core::write_tensor_f32(first_conv_time_mask_, first_conv_time_mask);
        const std::vector<float> first_conv_inv_count{
            1.0F / static_cast<float>(actual_first_conv_frames),
        };
        core::write_tensor_f32(first_conv_inv_valid_time_count_, first_conv_inv_count);

        std::vector<int32_t> token_mask(static_cast<size_t>(token_capacity_), 0);
        std::fill(token_mask.begin(), token_mask.begin() + actual_tokens, 1);
        core::write_tensor_i32(token_mask_, token_mask);

        std::vector<float> attention_mask(
            static_cast<size_t>(batch_ * token_capacity_ * token_capacity_),
            0.0F);
        const float neg_inf = -std::numeric_limits<float>::infinity();
        for (int64_t b = 0; b < batch_; ++b) {
            for (int64_t q = 0; q < token_capacity_; ++q) {
                auto * row = attention_mask.data() + static_cast<size_t>((b * token_capacity_ + q) * token_capacity_);
                for (int64_t k = actual_tokens; k < token_capacity_; ++k) {
                    row[k] = neg_inf;
                }
            }
        }
        core::write_tensor_f32(attention_mask_, attention_mask);
    }

    int64_t wavlm_first_conv_output_frames(int64_t samples) const {
        const auto & config = weights_->config;
        const int64_t frames = conv1d_output_frames(samples, config.conv_kernel.front(), config.conv_stride.front(), 0);
        if (frames <= 0) {
            throw std::runtime_error("WavLM input is too short for first convolution");
        }
        return frames;
    }

    int64_t wavlm_output_frames(int64_t samples) const {
        int64_t frames = samples;
        for (size_t i = 0; i < weights_->config.conv_dim.size(); ++i) {
            frames = conv1d_output_frames(frames, weights_->config.conv_kernel[i], weights_->config.conv_stride[i], 0);
        }
        if (frames <= 0) {
            throw std::runtime_error("WavLM input is too short for convolutional feature extractor");
        }
        return frames;
    }

    std::shared_ptr<const WavlmEncoderWeights> weights_;
    bool reuse_graph_ = true;
    std::mutex mutex_;
    ggml_context * ggml_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    core::TensorValue input_;
    core::TensorValue first_conv_time_mask_;
    core::TensorValue first_conv_inv_valid_time_count_;
    core::TensorValue token_mask_;
    core::TensorValue attention_mask_;
    core::TensorValue position_bias_;
    std::vector<core::TensorValue> outputs_;
    std::vector<int64_t> output_layers_;
    int64_t batch_ = 0;
    int64_t sample_capacity_ = 0;
    int64_t first_conv_frame_capacity_ = 0;
    int64_t token_capacity_ = 0;
};

}  // namespace

WavlmEncoderComponent WavlmEncoderComponent::load_from_safetensors(
    const std::filesystem::path & checkpoint_path,
    core::BackendConfig backend,
    WavlmEncoderConfig config) {
    validate_config(config);
    const auto source = engine::assets::open_tensor_source(checkpoint_path);
    auto weights = std::make_shared<WavlmEncoderWeights>();
    weights->config = std::move(config);
    weights->source_path = source->source_path();
    weights->execution_context = std::make_shared<core::ExecutionContext>(backend);
    weights->store = std::make_shared<core::BackendWeightStore>(
        weights->execution_context->backend(),
        weights->execution_context->backend_type(),
        "framework.wavlm.weights",
        512ull * 1024ull * 1024ull);

    const auto & config_ref = weights->config;
    load_tensor_alias(*weights, *source, "feature_extractor.conv_layers.0.conv.weight", "feature_extractor.conv_layers.0.0.weight", {config_ref.conv_dim[0], 1, config_ref.conv_kernel[0]});
    load_tensor_alias(*weights, *source, "feature_extractor.conv_layers.0.layer_norm.weight", "feature_extractor.conv_layers.0.2.weight", {config_ref.conv_dim[0]});
    load_tensor_alias(*weights, *source, "feature_extractor.conv_layers.0.layer_norm.bias", "feature_extractor.conv_layers.0.2.bias", {config_ref.conv_dim[0]});
    for (int64_t i = 1; i < static_cast<int64_t>(config_ref.conv_dim.size()); ++i) {
        load_tensor_alias(
            *weights,
            *source,
            "feature_extractor.conv_layers." + std::to_string(i) + ".conv.weight",
            "feature_extractor.conv_layers." + std::to_string(i) + ".0.weight",
            {config_ref.conv_dim[static_cast<size_t>(i)], config_ref.conv_dim[static_cast<size_t>(i - 1)], config_ref.conv_kernel[static_cast<size_t>(i)]});
    }
    load_tensor_alias(*weights, *source, "feature_projection.layer_norm.weight", "layer_norm.weight", {config_ref.conv_dim.back()});
    load_tensor_alias(*weights, *source, "feature_projection.layer_norm.bias", "layer_norm.bias", {config_ref.conv_dim.back()});
    load_tensor_alias(*weights, *source, "feature_projection.projection.weight", "post_extract_proj.weight", {config_ref.hidden_size, config_ref.conv_dim.back()});
    load_tensor_alias(*weights, *source, "feature_projection.projection.bias", "post_extract_proj.bias", {config_ref.hidden_size});
    load_tensor_alias(*weights, *source, "encoder.pos_conv_embed.conv.bias", "encoder.pos_conv.0.bias", {config_ref.hidden_size});
    weights->tensors.emplace(
        "encoder.pos_conv_embed.conv.weight",
        weights->store->make_f32(
            core::TensorShape::from_dims({
                config_ref.hidden_size,
                config_ref.hidden_size / config_ref.num_conv_pos_embedding_groups,
                config_ref.num_conv_pos_embeddings}),
            effective_weight_norm_conv1d(
                *source,
                "encoder.pos_conv.0",
                config_ref.hidden_size,
                config_ref.hidden_size / config_ref.num_conv_pos_embedding_groups,
                config_ref.num_conv_pos_embeddings)));
    weights->parameter_count += config_ref.hidden_size *
        (config_ref.hidden_size / config_ref.num_conv_pos_embedding_groups) *
        config_ref.num_conv_pos_embeddings;
    ++weights->loaded_tensor_count;
    load_tensor_alias(*weights, *source, "encoder.layer_norm.weight", "encoder.layer_norm.weight", {config_ref.hidden_size});
    load_tensor_alias(*weights, *source, "encoder.layer_norm.bias", "encoder.layer_norm.bias", {config_ref.hidden_size});
    weights->relative_attention_embedding =
        source->require_f32("encoder.layers.0.self_attn.relative_attention_bias.weight", {config_ref.num_buckets, config_ref.num_attention_heads});
    weights->position_bias_cache = std::make_shared<WavlmPositionBiasCache>();
    weights->parameter_count += config_ref.num_buckets * config_ref.num_attention_heads;
    const int64_t head_dim = config_ref.hidden_size / config_ref.num_attention_heads;
    for (int64_t i = 0; i < config_ref.num_hidden_layers; ++i) {
        const std::string internal = "encoder.layers." + std::to_string(i);
        const std::string src = "encoder.layers." + std::to_string(i);
        load_tensor_alias(*weights, *source, internal + ".attention.q_proj.weight", src + ".self_attn.q_proj.weight", {config_ref.hidden_size, config_ref.hidden_size});
        load_tensor_alias(*weights, *source, internal + ".attention.q_proj.bias", src + ".self_attn.q_proj.bias", {config_ref.hidden_size});
        load_tensor_alias(*weights, *source, internal + ".attention.k_proj.weight", src + ".self_attn.k_proj.weight", {config_ref.hidden_size, config_ref.hidden_size});
        load_tensor_alias(*weights, *source, internal + ".attention.k_proj.bias", src + ".self_attn.k_proj.bias", {config_ref.hidden_size});
        load_tensor_alias(*weights, *source, internal + ".attention.v_proj.weight", src + ".self_attn.v_proj.weight", {config_ref.hidden_size, config_ref.hidden_size});
        load_tensor_alias(*weights, *source, internal + ".attention.v_proj.bias", src + ".self_attn.v_proj.bias", {config_ref.hidden_size});
        load_tensor_alias(*weights, *source, internal + ".attention.out_proj.weight", src + ".self_attn.out_proj.weight", {config_ref.hidden_size, config_ref.hidden_size});
        load_tensor_alias(*weights, *source, internal + ".attention.out_proj.bias", src + ".self_attn.out_proj.bias", {config_ref.hidden_size});
        load_tensor_alias(*weights, *source, internal + ".attention.gru_rel_pos_linear.weight", src + ".self_attn.grep_linear.weight", {8, head_dim});
        load_tensor_alias(*weights, *source, internal + ".attention.gru_rel_pos_linear.bias", src + ".self_attn.grep_linear.bias", {8});
        load_tensor_alias(*weights, *source, internal + ".attention.gru_rel_pos_const", src + ".self_attn.grep_a", {1, config_ref.num_attention_heads, 1, 1});
        load_tensor_alias(*weights, *source, internal + ".self_attn_layer_norm.weight", src + ".self_attn_layer_norm.weight", {config_ref.hidden_size});
        load_tensor_alias(*weights, *source, internal + ".self_attn_layer_norm.bias", src + ".self_attn_layer_norm.bias", {config_ref.hidden_size});
        load_tensor_alias(*weights, *source, internal + ".fc1.weight", src + ".fc1.weight", {config_ref.intermediate_size, config_ref.hidden_size});
        load_tensor_alias(*weights, *source, internal + ".fc1.bias", src + ".fc1.bias", {config_ref.intermediate_size});
        load_tensor_alias(*weights, *source, internal + ".fc2.weight", src + ".fc2.weight", {config_ref.hidden_size, config_ref.intermediate_size});
        load_tensor_alias(*weights, *source, internal + ".fc2.bias", src + ".fc2.bias", {config_ref.hidden_size});
        load_tensor_alias(*weights, *source, internal + ".final_layer_norm.weight", src + ".final_layer_norm.weight", {config_ref.hidden_size});
        load_tensor_alias(*weights, *source, internal + ".final_layer_norm.bias", src + ".final_layer_norm.bias", {config_ref.hidden_size});
    }
    weights->store->upload();
    source->release_storage();

    return WavlmEncoderComponent(std::move(weights), backend);
}

struct WavlmEncoderComponent::State {
    std::unique_ptr<WavlmRunner> runner;
};

WavlmEncoderComponent::WavlmEncoderComponent(
    std::shared_ptr<const WavlmEncoderWeights> weights,
    core::BackendConfig backend,
    bool reuse_graph)
    : weights_(std::move(weights)),
      backend_(backend),
      state_(std::make_shared<State>()) {
    if (weights_ == nullptr) {
        throw std::runtime_error("WavLM component requires weights");
    }
    state_->runner = std::make_unique<WavlmRunner>(weights_, reuse_graph);
}

const core::BackendConfig & WavlmEncoderComponent::backend() const noexcept {
    return backend_;
}

const std::shared_ptr<const WavlmEncoderWeights> & WavlmEncoderComponent::weights() const noexcept {
    return weights_;
}

int64_t WavlmEncoderComponent::hidden_size() const noexcept {
    return weights_ == nullptr ? 0 : weights_->config.hidden_size;
}

int64_t WavlmEncoderComponent::loaded_tensor_count() const noexcept {
    return weights_ == nullptr ? 0 : weights_->loaded_tensor_count;
}

int64_t WavlmEncoderComponent::parameter_count() const noexcept {
    return weights_ == nullptr ? 0 : weights_->parameter_count;
}

WavlmEncoderOutput WavlmEncoderComponent::encode(
    const std::vector<float> & input_values,
    int64_t batch,
    int64_t samples) const {
    if (state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("WavLM component is not initialized");
    }
    return state_->runner->encode(input_values, batch, samples);
}

WavlmEncoderLayerOutput WavlmEncoderComponent::encode_layers(
    const std::vector<float> & input_values,
    int64_t batch,
    int64_t samples,
    const std::vector<int64_t> & output_layers) const {
    if (state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("WavLM component is not initialized");
    }
    return state_->runner->encode_layers(input_values, batch, samples, output_layers);
}

void WavlmEncoderComponent::release_runtime_graph() {
    if (state_ != nullptr && state_->runner != nullptr) {
        state_->runner->release_runtime_graph();
    }
}

}  // namespace engine::modules
