#include "engine/framework/modules/zipformer_modules.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace engine::modules {
namespace {

core::TensorValue contiguous(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(ctx, value);
}

core::TensorValue linear(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ZipformerLinearWeights & weights,
    const char * name) {
    core::validate_rank_between(weights.weight, 2, 2, name);
    if (weights.weight.shape.dims[1] != input.shape.last_dim()) {
        throw std::runtime_error(
            std::string(name) + " input dimension mismatch");
    }
    return LinearModule({
        weights.weight.shape.dims[1],
        weights.weight.shape.dims[0],
        true,
    }).build(ctx, input, {weights.weight, weights.bias});
}

core::TensorValue transpose(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::array<int, core::kMaxTensorRank> & axes) {
    const size_t rank = input.shape.rank;
    core::TensorShape output = {};
    output.rank = rank;
    std::array<bool, core::kMaxTensorRank> seen = {false, false, false, false};
    std::array<int, core::kMaxTensorRank> ggml_axes = {0, 1, 2, 3};
    for (size_t out_axis = 0; out_axis < rank; ++out_axis) {
        const int in_axis = axes[out_axis];
        if (in_axis < 0 ||
            in_axis >= static_cast<int>(rank) ||
            seen[static_cast<size_t>(in_axis)]) {
            throw std::runtime_error("invalid Zipformer transpose");
        }
        seen[static_cast<size_t>(in_axis)] = true;
        output.dims[out_axis] = input.shape.dims[in_axis];
        const int out_ggml =
            core::logical_axis_to_ggml_axis(rank, out_axis);
        const int in_ggml =
            core::logical_axis_to_ggml_axis(rank, in_axis);
        ggml_axes[static_cast<size_t>(in_ggml)] = out_ggml;
    }
    const auto source = contiguous(ctx, input);
    return core::wrap_tensor(
        ggml_permute(
            ctx.ggml,
            source.tensor,
            ggml_axes[0],
            ggml_axes[1],
            ggml_axes[2],
            ggml_axes[3]),
        output,
        input.type);
}

std::vector<float> compact_relative_position(
    int64_t seq,
    int64_t left_context,
    int64_t position_dim) {
    constexpr float pi = 3.14159265358979323846F;
    const int64_t length = left_context + 2 * seq - 1;
    std::vector<float> result(
        static_cast<size_t>(length * position_dim), 0.0F);
    const float compression =
        std::sqrt(static_cast<float>(position_dim));
    const float length_scale =
        static_cast<float>(position_dim) / (2.0F * pi);
    for (int64_t row = 0; row < length; ++row) {
        const float x =
            static_cast<float>(row - (left_context + seq - 1));
        const float sign =
            x < 0.0F ? -1.0F : (x > 0.0F ? 1.0F : 0.0F);
        const float compressed =
            compression * sign *
            (std::log(std::fabs(x) + compression) -
             std::log(compression));
        const float angle = std::atan(compressed / length_scale);
        for (int64_t dim = 0; dim < position_dim / 2; ++dim) {
            const float frequency = static_cast<float>(dim + 1);
            result[static_cast<size_t>(row * position_dim + 2 * dim)] =
                std::cos(angle * frequency);
            result[static_cast<size_t>(row * position_dim + 2 * dim + 1)] =
                std::sin(angle * frequency);
        }
        result[static_cast<size_t>(row * position_dim + position_dim - 1)] =
            1.0F;
    }
    return result;
}

}  // namespace

ZipformerRelativeAttentionModule::ZipformerRelativeAttentionModule(
    ZipformerRelativeAttentionConfig config)
    : config_(config) {
    if (config_.num_heads <= 0 ||
        config_.query_head_dim <= 0 ||
        config_.position_dim <= 0 ||
        config_.position_head_dim <= 0) {
        throw std::runtime_error(
            "ZipformerRelativeAttentionModule config dimensions must be positive");
    }
}

ZipformerRelativeAttentionOutputs ZipformerRelativeAttentionModule::build(
    core::ModuleBuildContext & ctx,
    ZipformerConstantFactory & constant_factory,
    const core::TensorValue & input,
    const core::TensorValue & cached_key,
    const core::TensorValue & padding_mask,
    const ZipformerRelativeAttentionWeights & weights) const {
    const int64_t seq = input.shape.dims[0];
    const int64_t left_context = cached_key.shape.dims[0];
    const int64_t source_frames = left_context + seq;
    const int64_t query_channels =
        config_.num_heads * config_.query_head_dim;
    const int64_t position_channels =
        config_.num_heads * config_.position_head_dim;
    const size_t expected_position_values =
        static_cast<size_t>(
            position_channels * config_.position_dim);
    if (weights.linear_pos_weight.size() != expected_position_values) {
        throw std::runtime_error(
            "Zipformer relative position weight size mismatch");
    }

    auto projected =
        linear(ctx, input, weights.in_proj, "Zipformer relative attention in_proj");
    auto current_key =
        SliceModule({2, query_channels, query_channels})
            .build(ctx, projected);
    auto all_keys = ConcatModule({0}).build(ctx, cached_key, current_key);

    ZipformerRelativeAttentionOutputs result;
    result.key_state =
        SliceModule({0, seq, left_context}).build(ctx, all_keys);

    const auto position =
        compact_relative_position(
            seq,
            left_context,
            config_.position_dim);
    result.weights.reserve(static_cast<size_t>(config_.num_heads));
    for (int64_t head = 0; head < config_.num_heads; ++head) {
        auto query =
            SliceModule({
                2,
                head * config_.query_head_dim,
                config_.query_head_dim,
            }).build(ctx, projected);
        auto key =
            SliceModule({
                2,
                head * config_.query_head_dim,
                config_.query_head_dim,
            }).build(ctx, all_keys);
        auto position_query =
            SliceModule({
                2,
                2 * query_channels + head * config_.position_head_dim,
                config_.position_head_dim,
            }).build(ctx, projected);
        query = transpose(ctx, query, {1, 0, 2, 3});
        key = transpose(ctx, key, {1, 2, 0, 3});
        auto scores = MatMulModule().build(ctx, query, key);
        position_query = transpose(ctx, position_query, {1, 0, 2, 3});

        core::TensorValue position_scores;
        for (int64_t dim = 0; dim < config_.position_head_dim; ++dim) {
            std::vector<float> table(
                static_cast<size_t>(seq * source_frames), 0.0F);
            const int64_t output_row =
                head * config_.position_head_dim + dim;
            for (int64_t target = 0; target < seq; ++target) {
                for (int64_t source = 0;
                     source < source_frames;
                     ++source) {
                    const int64_t relative =
                        (seq - 1 - target) + source;
                    double value = 0.0;
                    for (int64_t p = 0; p < config_.position_dim; ++p) {
                        value +=
                            static_cast<double>(
                                position[static_cast<size_t>(
                                    relative * config_.position_dim + p)]) *
                            static_cast<double>(
                                weights.linear_pos_weight[
                                    static_cast<size_t>(
                                        output_row * config_.position_dim + p)]);
                    }
                    table[static_cast<size_t>(
                        target * source_frames + source)] =
                        static_cast<float>(value);
                }
            }
            auto relative = constant_factory(
                core::TensorShape::from_dims({1, seq, source_frames}),
                std::move(table));
            auto query_dim_value =
                SliceModule({2, dim, 1}).build(ctx, position_query);
            auto repeated =
                RepeatModule({
                    core::TensorShape::from_dims(
                        {1, seq, source_frames}),
                }).build(ctx, query_dim_value);
            auto term = MulModule().build(ctx, repeated, relative);
            position_scores = position_scores.valid()
                ? AddModule().build(ctx, position_scores, term)
                : term;
        }

        result.weights.push_back(
            SoftmaxModule().build(
                ctx,
                AddModule().build(
                    ctx,
                    AddModule().build(ctx, scores, position_scores),
                    padding_mask)));
    }
    return result;
}

ZipformerStatefulOutputs ZipformerNonlinearAttentionModule::build(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & attention,
    const core::TensorValue & cached_value,
    const ZipformerNonlinearAttentionWeights & weights) const {
    auto projected =
        linear(ctx, input, weights.in_proj, "Zipformer nonlinear attention in_proj");
    const int64_t hidden = projected.shape.last_dim() / 3;
    auto gate_input =
        SliceModule({2, 0, hidden}).build(ctx, projected);
    auto value =
        SliceModule({2, hidden, hidden}).build(ctx, projected);
    auto output_gate =
        SliceModule({2, 2 * hidden, hidden}).build(ctx, projected);
    value = MulModule().build(
        ctx,
        value,
        TanhModule().build(ctx, gate_input));
    const int64_t seq = value.shape.dims[0];
    const int64_t left_context = cached_value.shape.dims[0];
    value = ConcatModule({0}).build(ctx, cached_value, value);
    auto new_state =
        SliceModule({0, seq, left_context}).build(ctx, value);
    auto joined = transpose(ctx, value, {1, 0, 2, 3});
    joined = MatMulModule().build(ctx, attention, joined);
    joined = transpose(ctx, joined, {1, 0, 2, 3});
    return {
        linear(
            ctx,
            MulModule().build(ctx, joined, output_gate),
            weights.out_proj,
            "Zipformer nonlinear attention out_proj"),
        new_state,
    };
}

ZipformerConvolutionModule::ZipformerConvolutionModule(
    ZipformerConvolutionConfig config)
    : config_(config) {
    if (config_.kernel_size <= 0) {
        throw std::runtime_error(
            "ZipformerConvolutionModule kernel size must be positive");
    }
}

ZipformerStatefulOutputs ZipformerConvolutionModule::build(
    core::ModuleBuildContext & ctx,
    ZipformerConstantFactory & constant_factory,
    const core::TensorValue & input,
    const core::TensorValue & cached_value,
    const ZipformerConvolutionWeights & weights) const {
    const int64_t channels = input.shape.last_dim();
    auto projected =
        linear(ctx, input, weights.in_proj, "Zipformer convolution in_proj");
    auto value =
        SliceModule({2, 0, channels}).build(ctx, projected);
    auto gate =
        SliceModule({2, channels, channels}).build(ctx, projected);
    value = MulModule().build(
        ctx,
        value,
        SigmoidModule().build(ctx, gate));
    auto channel_first = transpose(ctx, value, {1, 2, 0, 3});
    const int64_t left_pad = config_.kernel_size / 2;
    auto causal_input =
        ConcatModule({2}).build(ctx, cached_value, channel_first);
    auto new_state =
        contiguous(
            ctx,
            SliceModule({2, input.shape.dims[0], left_pad})
                .build(ctx, causal_input));
    auto causal =
        DepthwiseConv1dModule(
            {channels, left_pad + 1, 1, 0, 1, true})
            .build(ctx, causal_input, weights.causal_conv);
    auto chunk =
        DepthwiseConv1dModule({
            channels,
            config_.kernel_size,
            1,
            static_cast<int>(left_pad),
            1,
            true,
        }).build(ctx, channel_first, weights.chunkwise_conv);

    if (weights.chunk_scale_left.size() !=
            static_cast<size_t>(channels * config_.kernel_size) ||
        weights.chunk_scale_right.size() !=
            static_cast<size_t>(channels * config_.kernel_size)) {
        throw std::runtime_error(
            "Zipformer convolution chunk scale size mismatch");
    }
    const int64_t seq = input.shape.dims[0];
    std::vector<float> chunk_scale(
        static_cast<size_t>(channels * seq), 1.0F);
    for (int64_t channel = 0; channel < channels; ++channel) {
        for (int64_t frame = 0; frame < seq; ++frame) {
            float edge = 0.0F;
            if (seq < config_.kernel_size) {
                edge += weights.chunk_scale_left[static_cast<size_t>(
                    channel * config_.kernel_size + frame)];
                edge += weights.chunk_scale_right[static_cast<size_t>(
                    channel * config_.kernel_size +
                    config_.kernel_size - seq + frame)];
            } else {
                if (frame < config_.kernel_size) {
                    edge += weights.chunk_scale_left[static_cast<size_t>(
                        channel * config_.kernel_size + frame)];
                }
                if (frame >= seq - config_.kernel_size) {
                    edge += weights.chunk_scale_right[static_cast<size_t>(
                        channel * config_.kernel_size +
                        frame - (seq - config_.kernel_size))];
                }
            }
            chunk_scale[static_cast<size_t>(channel * seq + frame)] += edge;
        }
    }
    auto scale_tensor = constant_factory(
        core::TensorShape::from_dims({1, channels, seq}),
        std::move(chunk_scale));
    auto scaled_chunk = MulModule().build(ctx, chunk, scale_tensor);
    auto combined = AddModule().build(ctx, causal, scaled_chunk);
    combined = transpose(ctx, combined, {2, 0, 1, 3});
    combined = SwooshRModule().build(ctx, combined);
    return {
        linear(
            ctx,
            combined,
            weights.out_proj,
            "Zipformer convolution out_proj"),
        new_state,
    };
}

}  // namespace engine::modules
