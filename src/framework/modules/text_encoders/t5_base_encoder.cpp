#include "engine/framework/modules/text_encoders/t5_base_encoder.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace engine::modules {
namespace {

void validate_config(const T5BaseEncoderConfig & config) {
    if (config.hidden_size <= 0 || config.layers <= 0 || config.attention_heads <= 0 ||
        config.head_dim <= 0 || config.intermediate_size <= 0 || config.vocab_size <= 0) {
        throw std::runtime_error("T5BaseEncoderConfig dimensions must be positive");
    }
    if (config.attention_heads * config.head_dim != config.hidden_size) {
        throw std::runtime_error("T5BaseEncoderConfig attention_heads * head_dim must equal hidden_size");
    }
    if (config.relative_attention_num_buckets <= 0 || config.relative_attention_max_distance <= 0) {
        throw std::runtime_error("T5BaseEncoderConfig relative attention values must be positive");
    }
    if (!(config.rms_norm_eps > 0.0F)) {
        throw std::runtime_error("T5BaseEncoderConfig rms_norm_eps must be positive");
    }
}

core::TensorValue contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    return core::ensure_backend_addressable_layout(ctx, input);
}

std::array<int, core::kMaxTensorRank> transpose_last_two_axes(size_t rank) {
    std::array<int, core::kMaxTensorRank> axes = {0, 1, 2, 3};
    if (rank < 2) {
        throw std::runtime_error("T5BaseEncoder transpose requires rank >= 2");
    }
    std::swap(axes[rank - 2], axes[rank - 1]);
    return axes;
}

core::TensorValue matmul_f32(core::ModuleBuildContext & ctx, const core::TensorValue & lhs, const core::TensorValue & rhs) {
    core::validate_rank_between(lhs, 2, core::kMaxTensorRank, "T5BaseEncoder matmul lhs");
    core::validate_rank_between(rhs, lhs.shape.rank, lhs.shape.rank, "T5BaseEncoder matmul rhs");
    const size_t rank = lhs.shape.rank;
    for (size_t axis = 0; axis + 2 < rank; ++axis) {
        if (lhs.shape.dims[axis] != rhs.shape.dims[axis]) {
            throw std::runtime_error("T5BaseEncoder matmul batch dimensions must match");
        }
    }
    if (lhs.shape.dims[rank - 1] != rhs.shape.dims[rank - 2]) {
        throw std::runtime_error("T5BaseEncoder matmul inner dimensions must match");
    }
    auto rhs_transposed = TransposeModule({transpose_last_two_axes(rank), rank}).build(ctx, rhs);
    rhs_transposed = contiguous(ctx, rhs_transposed);
    core::TensorShape output_shape = lhs.shape;
    output_shape.dims[rank - 1] = rhs.shape.dims[rank - 1];
    ggml_tensor * output = ggml_mul_mat(ctx.ggml, rhs_transposed.tensor, lhs.tensor);
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    return core::wrap_tensor(output, output_shape, GGML_TYPE_F32);
}

core::TensorValue t5_layer_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & weight,
    const T5BaseEncoderConfig & config) {
    return RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
        .build(ctx, input, NormWeights{weight, std::nullopt});
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t head_dim) {
    const auto input_contiguous = contiguous(ctx, input);
    return core::reshape_tensor(
        ctx,
        input_contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, head_dim}));
}

core::TensorValue relative_position_bias(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & relative_position_buckets,
    const core::TensorValue & relative_attention_bias,
    const T5BaseEncoderConfig & config,
    int64_t batch) {
    auto bias = EmbeddingModule({config.relative_attention_num_buckets, config.attention_heads})
                    .build(ctx, relative_position_buckets, relative_attention_bias);
    bias = core::reshape_tensor(
        ctx,
        contiguous(ctx, bias),
        core::TensorShape::from_dims({1, relative_position_buckets.shape.dims[0], relative_position_buckets.shape.dims[1], config.attention_heads}));
    bias = core::wrap_tensor(
        ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, bias.tensor, 2, 0, 1, 3)),
        core::TensorShape::from_dims({1, config.attention_heads, relative_position_buckets.shape.dims[0], relative_position_buckets.shape.dims[1]}),
        GGML_TYPE_F32);
    return RepeatModule({core::TensorShape::from_dims(
        {batch, config.attention_heads, relative_position_buckets.shape.dims[0], relative_position_buckets.shape.dims[1]})})
        .build(ctx, bias);
}

core::TensorValue self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & position_bias,
    const core::TensorValue & additive_attention_mask,
    const T5BaseEncoderLayerWeights & weights,
    const T5BaseEncoderConfig & config) {
    const LinearModule q_proj({config.hidden_size, config.hidden_size, false, GGML_PREC_F32});
    const LinearModule k_proj({config.hidden_size, config.hidden_size, false, GGML_PREC_F32});
    const LinearModule v_proj({config.hidden_size, config.hidden_size, false, GGML_PREC_F32});
    const LinearModule o_proj({config.hidden_size, config.hidden_size, false, GGML_PREC_F32});
    auto q = q_proj.build(ctx, input, weights.q_proj);
    auto k = k_proj.build(ctx, input, weights.k_proj);
    auto v = v_proj.build(ctx, input, weights.v_proj);
    q = reshape_heads(ctx, q, config.attention_heads, config.head_dim);
    k = reshape_heads(ctx, k, config.attention_heads, config.head_dim);
    v = reshape_heads(ctx, v, config.attention_heads, config.head_dim);
    q = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    k = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
    v = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
    auto scores = matmul_f32(ctx, q, TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k));
    scores = AddModule{}.build(ctx, scores, position_bias);
    scores = AddModule{}.build(ctx, scores, additive_attention_mask);
    auto attn = core::wrap_tensor(
        ggml_soft_max(ctx.ggml, contiguous(ctx, scores).tensor),
        scores.shape,
        GGML_TYPE_F32);
    auto context = matmul_f32(ctx, attn, v);
    context = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, context);
    context = core::reshape_tensor(ctx, contiguous(ctx, context), core::TensorShape::from_dims(
        {input.shape.dims[0], input.shape.dims[1], config.hidden_size}));
    return o_proj.build(ctx, context, weights.o_proj);
}

core::TensorValue feed_forward(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const T5BaseEncoderLayerWeights & weights,
    const T5BaseEncoderConfig & config) {
    auto hidden = LinearModule({config.hidden_size, config.intermediate_size, false, GGML_PREC_F32})
                      .build(ctx, input, weights.wi_proj);
    if (config.feed_forward_kind == T5BaseFeedForwardKind::Relu) {
        hidden = ReluModule{}.build(ctx, hidden);
    } else if (config.feed_forward_kind == T5BaseFeedForwardKind::GatedGeluTanh) {
        if (!weights.gate_proj.weight.valid()) {
            throw std::runtime_error("T5BaseEncoder gated GELU feed-forward requires gate_proj.weight");
        }
        auto gate = LinearModule({config.hidden_size, config.intermediate_size, false, GGML_PREC_F32})
                        .build(ctx, input, weights.gate_proj);
        gate = GeluModule({GeluApproximation::Tanh}).build(ctx, gate);
        hidden = MulModule{}.build(ctx, hidden, gate);
    } else {
        throw std::runtime_error("Unsupported T5BaseEncoder feed-forward kind");
    }
    return LinearModule({config.intermediate_size, config.hidden_size, false, GGML_PREC_F32})
        .build(ctx, hidden, weights.wo_proj);
}

core::TensorValue encoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & position_bias,
    const core::TensorValue & additive_attention_mask,
    const T5BaseEncoderLayerWeights & weights,
    const T5BaseEncoderConfig & config) {
    auto hidden = t5_layer_norm(ctx, input, weights.self_attention_layer_norm, config);
    hidden = self_attention(ctx, hidden, position_bias, additive_attention_mask, weights, config);
    auto output = AddModule{}.build(ctx, input, hidden);
    hidden = t5_layer_norm(ctx, output, weights.ffn_layer_norm, config);
    hidden = feed_forward(ctx, hidden, weights, config);
    return AddModule{}.build(ctx, output, hidden);
}

}  // namespace

std::vector<int32_t> t5_base_relative_position_buckets(
    int64_t query_length,
    int64_t key_length,
    int64_t num_buckets,
    int64_t max_distance,
    bool bidirectional) {
    if (query_length <= 0 || key_length <= 0 || num_buckets <= 0 || max_distance <= 0) {
        throw std::runtime_error("T5 relative position bucket dimensions must be positive");
    }
    std::vector<int32_t> out(static_cast<size_t>(query_length * key_length), 0);
    for (int64_t q = 0; q < query_length; ++q) {
        for (int64_t k = 0; k < key_length; ++k) {
            int64_t relative_buckets = 0;
            int64_t relative_position = k - q;
            int64_t buckets = num_buckets;
            if (bidirectional) {
                buckets /= 2;
                if (relative_position > 0) {
                    relative_buckets += buckets;
                }
                relative_position = std::llabs(relative_position);
            } else {
                relative_position = -std::min<int64_t>(relative_position, 0);
            }
            const int64_t max_exact = buckets / 2;
            int64_t bucket = relative_position;
            if (relative_position >= max_exact) {
                const double log_ratio = std::log(static_cast<double>(relative_position) / static_cast<double>(max_exact)) /
                    std::log(static_cast<double>(max_distance) / static_cast<double>(max_exact));
                bucket = max_exact + static_cast<int64_t>(log_ratio * static_cast<double>(buckets - max_exact));
                bucket = std::min(bucket, buckets - 1);
            }
            relative_buckets += bucket;
            out[static_cast<size_t>(q * key_length + k)] = static_cast<int32_t>(relative_buckets);
        }
    }
    return out;
}

T5BaseEncoderModule::T5BaseEncoderModule(T5BaseEncoderConfig config) : config_(config) {
    validate_config(config_);
}

const T5BaseEncoderConfig & T5BaseEncoderModule::config() const noexcept {
    return config_;
}

core::TensorValue T5BaseEncoderModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_ids,
    const core::TensorValue & relative_position_buckets,
    const core::TensorValue & additive_attention_mask,
    const T5BaseEncoderWeights & weights) const {
    core::validate_rank_between(input_ids, 2, 2, "T5BaseEncoder input_ids");
    const int64_t batch = input_ids.shape.dims[0];
    const int64_t tokens = input_ids.shape.dims[1];
    core::validate_shape(relative_position_buckets, core::TensorShape::from_dims({tokens, tokens}), "T5BaseEncoder relative_position_buckets");
    core::validate_shape(
        additive_attention_mask,
        core::TensorShape::from_dims({batch, config_.attention_heads, tokens, tokens}),
        "T5BaseEncoder additive_attention_mask");
    if (static_cast<int64_t>(weights.layers.size()) != config_.layers) {
        throw std::runtime_error("T5BaseEncoder layer count mismatch");
    }
    auto hidden = EmbeddingModule({config_.vocab_size, config_.hidden_size}).build(ctx, input_ids, weights.embed_tokens);
    core::TensorValue shared_position_bias;
    if (config_.shared_relative_position_bias) {
        core::validate_shape(weights.relative_attention_bias, core::TensorShape::from_dims(
            {config_.relative_attention_num_buckets, config_.attention_heads}), "T5BaseEncoder relative_attention_bias");
        shared_position_bias = relative_position_bias(
            ctx, relative_position_buckets, weights.relative_attention_bias, config_, batch);
    }
    for (int64_t i = 0; i < config_.layers; ++i) {
        auto position_bias = shared_position_bias;
        if (!config_.shared_relative_position_bias) {
            const auto & layer = weights.layers[static_cast<size_t>(i)];
            if (!layer.relative_attention_bias.has_value()) {
                throw std::runtime_error("T5BaseEncoder layer relative position bias is missing");
            }
            core::validate_shape(*layer.relative_attention_bias, core::TensorShape::from_dims(
                {config_.relative_attention_num_buckets, config_.attention_heads}),
                "T5BaseEncoder layer relative_attention_bias");
            position_bias = relative_position_bias(
                ctx, relative_position_buckets, *layer.relative_attention_bias, config_, batch);
        }
        hidden = encoder_layer(ctx, hidden, position_bias, additive_attention_mask, weights.layers[static_cast<size_t>(i)], config_);
    }
    return t5_layer_norm(ctx, hidden, weights.final_layer_norm, config_);
}

}  // namespace engine::modules
