#include "engine/community_models/kroko_asr/zipformer.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace engine::models::kroko_asr {
namespace {

constexpr int64_t kInputDim = 192;
constexpr int64_t kPosDim = 48;
constexpr int64_t kPosHeadDim = 4;
constexpr size_t kWeightContextBytes = 64ull * 1024ull * 1024ull;
constexpr size_t kGraphContextBytes = 192ull * 1024ull * 1024ull;
constexpr size_t kGraphNodes = 131072;

struct ContextDeleter {
    void operator()(ggml_context * value) const noexcept {
        if (value != nullptr) {
            ggml_free(value);
        }
    }
};

struct Param {
    std::vector<int64_t> shape;
    std::vector<float> host;
    core::TensorValue tensor;
};

struct GraphConstant {
    ggml_tensor * tensor = nullptr;
    std::vector<float> values;
};

struct StateTensor {
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    core::TensorValue value;
    std::vector<float> host;
};

struct LayerState {
    StateTensor key;
    StateTensor nonlin;
    StateTensor val1;
    StateTensor val2;
    StateTensor conv1;
    StateTensor conv2;
};

struct LayerStateValue {
    core::TensorValue key;
    core::TensorValue nonlin;
    core::TensorValue val1;
    core::TensorValue val2;
    core::TensorValue conv1;
    core::TensorValue conv2;
};

struct AttentionResult {
    std::vector<core::TensorValue> weights;
    core::TensorValue key;
};

struct ValueResult {
    core::TensorValue output;
    core::TensorValue state;
};

struct LayerResult {
    core::TensorValue output;
    LayerStateValue state;
};

struct StackMask {
    ggml_tensor * input = nullptr;
    core::TensorValue value;
    int64_t frames = 0;
    int64_t left_context = 0;
    int64_t downsampling = 1;
    std::vector<float> host;
};

const Param & param(
    const std::unordered_map<std::string, Param> & params,
    const std::string & name) {
    const auto it = params.find(name);
    if (it == params.end()) {
        throw std::runtime_error(
            "missing Kroko Zipformer tensor: " + name);
    }
    return it->second;
}

core::TensorShape shape_of(const std::vector<int64_t> & dims) {
    switch (dims.size()) {
        case 1:
            return core::TensorShape::from_dims({dims[0]});
        case 2:
            return core::TensorShape::from_dims({dims[0], dims[1]});
        case 3:
            return core::TensorShape::from_dims(
                {dims[0], dims[1], dims[2]});
        case 4:
            return core::TensorShape::from_dims(
                {dims[0], dims[1], dims[2], dims[3]});
        default:
            throw std::runtime_error(
                "Kroko Zipformer tensor rank is unsupported");
    }
}

core::TensorValue contiguous(
    core::ModuleBuildContext & context,
    const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(context, value);
}

core::TensorValue transpose(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    const std::array<int, 4> & axes) {
    const size_t rank = input.shape.rank;
    core::TensorShape output = {};
    output.rank = rank;
    std::array<bool, 4> seen = {false, false, false, false};
    std::array<int, 4> ggml_axes = {0, 1, 2, 3};
    for (size_t out_axis = 0; out_axis < rank; ++out_axis) {
        const int in_axis = axes[out_axis];
        if (in_axis < 0 ||
            in_axis >= static_cast<int>(rank) ||
            seen[static_cast<size_t>(in_axis)]) {
            throw std::runtime_error(
                "invalid Kroko Zipformer transpose");
        }
        seen[static_cast<size_t>(in_axis)] = true;
        output.dims[out_axis] = input.shape.dims[in_axis];
        const int out_ggml =
            core::logical_axis_to_ggml_axis(rank, out_axis);
        const int in_ggml =
            core::logical_axis_to_ggml_axis(rank, in_axis);
        ggml_axes[in_ggml] = out_ggml;
    }
    const auto source = contiguous(context, input);
    return core::wrap_tensor(
        ggml_permute(
            context.ggml,
            source.tensor,
            ggml_axes[0],
            ggml_axes[1],
            ggml_axes[2],
            ggml_axes[3]),
        output,
        input.type);
}

core::TensorValue add(
    core::ModuleBuildContext & context,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    return modules::AddModule().build(context, lhs, rhs);
}

core::TensorValue sub(
    core::ModuleBuildContext & context,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    const auto a = contiguous(context, lhs);
    const auto b = contiguous(context, rhs);
    return core::wrap_tensor(
        ggml_sub(context.ggml, a.tensor, b.tensor),
        lhs.shape,
        GGML_TYPE_F32);
}

core::TensorValue mul(
    core::ModuleBuildContext & context,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    return modules::MulModule().build(context, lhs, rhs);
}

core::TensorValue scale(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    float value) {
    const auto source = contiguous(context, input);
    return core::wrap_tensor(
        ggml_scale(context.ggml, source.tensor, value),
        input.shape,
        GGML_TYPE_F32);
}

core::TensorValue scale_bias(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    float factor,
    float bias) {
    const auto source = contiguous(context, input);
    return core::wrap_tensor(
        ggml_scale_bias(
            context.ggml, source.tensor, factor, bias),
        input.shape,
        GGML_TYPE_F32);
}

core::TensorValue swoosh(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    float offset,
    float constant) {
    auto shifted = scale_bias(context, input, 1.0F, -offset);
    auto result = core::wrap_tensor(
        ggml_softplus(context.ggml, shifted.tensor),
        input.shape,
        GGML_TYPE_F32);
    result = add(context, result, scale(context, input, -0.08F));
    return scale_bias(context, result, 1.0F, constant);
}

core::TensorValue swoosh_l(
    core::ModuleBuildContext & context,
    const core::TensorValue & input) {
    return swoosh(context, input, 4.0F, -0.035F);
}

core::TensorValue swoosh_r(
    core::ModuleBuildContext & context,
    const core::TensorValue & input) {
    return swoosh(
        context, input, 1.0F, -0.313261687F);
}

core::TensorValue linear(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix) {
    const auto & weight = param(params, prefix + ".weight");
    const auto & bias = param(params, prefix + ".bias");
    if (weight.shape.size() != 2 ||
        weight.shape[1] != input.shape.last_dim()) {
        throw std::runtime_error(
            "Kroko Zipformer linear shape mismatch: " + prefix);
    }
    return modules::LinearModule(
        {weight.shape[1], weight.shape[0], true})
        .build(
            context,
            input,
            {weight.tensor, bias.tensor});
}

core::TensorValue repeat_last_scale(
    core::ModuleBuildContext & context,
    const core::TensorValue & channel_scale,
    const core::TensorValue & target) {
    std::vector<int64_t> view_dims(target.shape.rank, 1);
    view_dims.back() = target.shape.last_dim();
    auto view = core::reshape_tensor(
        context, channel_scale, shape_of(view_dims));
    return modules::RepeatModule({target.shape}).build(
        context, view);
}

core::TensorValue bypass(
    core::ModuleBuildContext & context,
    const core::TensorValue & original,
    const core::TensorValue & transformed,
    const Param & bypass_scale) {
    return add(
        context,
        original,
        mul(
            context,
            sub(context, transformed, original),
            repeat_last_scale(
                context, bypass_scale.tensor, original)));
}

core::TensorValue bias_norm(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix) {
    const auto & bias = param(params, prefix + ".bias");
    const auto & log_scale = param(params, prefix + ".log_scale");
    if (log_scale.host.size() != 1) {
        throw std::runtime_error(
            "Kroko BiasNorm scale is unavailable: " + prefix);
    }
    std::vector<int64_t> bias_dims(input.shape.rank, 1);
    bias_dims.back() = input.shape.last_dim();
    auto bias_view = core::reshape_tensor(
        context, bias.tensor, shape_of(bias_dims));
    auto bias_repeated =
        modules::RepeatModule({input.shape}).build(
            context, bias_view);
    auto centered = sub(context, input, bias_repeated);
    auto squared = core::wrap_tensor(
        ggml_sqr(context.ggml, centered.tensor),
        centered.shape,
        GGML_TYPE_F32);
    auto mean = modules::ReduceMeanModule(
        {static_cast<int>(input.shape.rank - 1)})
                    .build(context, squared);
    auto denominator = core::wrap_tensor(
        ggml_sqrt(context.ggml, mean.tensor),
        mean.shape,
        GGML_TYPE_F32);
    auto denominator_repeated =
        modules::RepeatModule({input.shape}).build(
            context, denominator);
    auto normalized = core::wrap_tensor(
        ggml_div(
            context.ggml,
            input.tensor,
            denominator_repeated.tensor),
        input.shape,
        GGML_TYPE_F32);
    return scale(
        context,
        normalized,
        std::exp(log_scale.host.front()));
}

std::vector<float> compact_relative_position(
    int64_t seq,
    int64_t left_context) {
    constexpr float pi = 3.14159265358979323846F;
    const int64_t length = left_context + 2 * seq - 1;
    std::vector<float> result(
        static_cast<size_t>(length * kPosDim), 0.0F);
    const float compression = std::sqrt(
        static_cast<float>(kPosDim));
    const float length_scale =
        static_cast<float>(kPosDim) / (2.0F * pi);
    for (int64_t row = 0; row < length; ++row) {
        const float x =
            static_cast<float>(
                row - (left_context + seq - 1));
        const float sign =
            x < 0.0F ? -1.0F : (x > 0.0F ? 1.0F : 0.0F);
        const float compressed =
            compression * sign *
            (std::log(std::fabs(x) + compression) -
             std::log(compression));
        const float angle =
            std::atan(compressed / length_scale);
        for (int64_t dim = 0; dim < kPosDim / 2; ++dim) {
            const float frequency = static_cast<float>(dim + 1);
            result[static_cast<size_t>(
                row * kPosDim + 2 * dim)] =
                std::cos(angle * frequency);
            result[static_cast<size_t>(
                row * kPosDim + 2 * dim + 1)] =
                std::sin(angle * frequency);
        }
        result[static_cast<size_t>(
            row * kPosDim + kPosDim - 1)] = 1.0F;
    }
    return result;
}

core::TensorValue graph_constant(
    core::ModuleBuildContext & context,
    std::vector<GraphConstant> & constants,
    const core::TensorShape & shape,
    std::vector<float> values) {
    auto result =
        core::make_tensor(context, GGML_TYPE_F32, shape);
    ggml_set_input(result.tensor);
    constants.push_back(
        GraphConstant{result.tensor, std::move(values)});
    return result;
}

size_t element_count(const core::TensorShape & shape) {
    size_t result = 1;
    for (size_t axis = 0; axis < shape.rank; ++axis) {
        result *= static_cast<size_t>(shape.dims[axis]);
    }
    return result;
}

StateTensor state_tensor(
    core::ModuleBuildContext & context,
    const core::TensorShape & shape) {
    StateTensor result;
    result.value = core::make_tensor(
        context, GGML_TYPE_F32, shape);
    result.input = result.value.tensor;
    ggml_set_input(result.input);
    result.host.assign(element_count(shape), 0.0F);
    return result;
}

AttentionResult attention_weights(
    core::ModuleBuildContext & context,
    std::vector<GraphConstant> & constants,
    const core::TensorValue & input,
    const core::TensorValue & cached_key,
    const core::TensorValue & padding_mask,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix,
    int64_t heads,
    int64_t query_dim) {
    const int64_t seq = input.shape.dims[0];
    const int64_t left_context = cached_key.shape.dims[0];
    const int64_t source_frames = left_context + seq;
    auto projected = linear(
        context,
        input,
        params,
        prefix + ".self_attn_weights.in_proj");
    const auto & position_weight = param(
        params,
        prefix + ".self_attn_weights.linear_pos.weight");
    if (position_weight.host.empty()) {
        throw std::runtime_error(
            "Kroko relative position weight is unavailable");
    }
    auto current_key = modules::SliceModule(
        {2, heads * query_dim, heads * query_dim})
                           .build(context, projected);
    auto all_keys = modules::ConcatModule({0}).build(
        context, cached_key, current_key);
    AttentionResult result;
    result.key = modules::SliceModule(
        {0, seq, left_context})
                     .build(context, all_keys);
    const auto position =
        compact_relative_position(seq, left_context);
    result.weights.reserve(static_cast<size_t>(heads));
    for (int64_t head = 0; head < heads; ++head) {
        auto query = modules::SliceModule(
            {2, head * query_dim, query_dim})
                         .build(context, projected);
        auto key = modules::SliceModule(
            {2, head * query_dim, query_dim})
                       .build(context, all_keys);
        auto position_query = modules::SliceModule(
            {2,
             2 * heads * query_dim + head * kPosHeadDim,
             kPosHeadDim})
                                  .build(context, projected);
        query = transpose(context, query, {1, 0, 2, 3});
        key = transpose(context, key, {1, 2, 0, 3});
        auto scores =
            modules::MatMulModule().build(context, query, key);
        position_query = transpose(
            context, position_query, {1, 0, 2, 3});
        core::TensorValue position_scores;
        for (int64_t dim = 0; dim < kPosHeadDim; ++dim) {
            std::vector<float> table(
                static_cast<size_t>(seq * source_frames), 0.0F);
            const int64_t output_row =
                head * kPosHeadDim + dim;
            for (int64_t target = 0; target < seq; ++target) {
                for (int64_t source = 0;
                     source < source_frames;
                     ++source) {
                    const int64_t relative =
                        (seq - 1 - target) + source;
                    double value = 0.0;
                    for (int64_t p = 0; p < kPosDim; ++p) {
                        value +=
                            static_cast<double>(
                                position[static_cast<size_t>(
                                    relative * kPosDim + p)]) *
                            static_cast<double>(
                                position_weight.host[
                                    static_cast<size_t>(
                                        output_row * kPosDim + p)]);
                    }
                    table[static_cast<size_t>(
                        target * source_frames + source)] =
                        static_cast<float>(value);
                }
            }
            auto relative = graph_constant(
                context,
                constants,
                core::TensorShape::from_dims(
                    {1, seq, source_frames}),
                std::move(table));
            auto query_dim_value = modules::SliceModule(
                {2, dim, 1})
                                       .build(
                                           context,
                                           position_query);
            auto repeated =
                modules::RepeatModule(
                    {core::TensorShape::from_dims(
                        {1, seq, source_frames})})
                    .build(context, query_dim_value);
            auto term = mul(context, repeated, relative);
            position_scores = position_scores.valid()
                ? add(context, position_scores, term)
                : term;
        }
        result.weights.push_back(
            modules::SoftmaxModule().build(
                context,
                add(
                    context,
                    add(context, scores, position_scores),
                    padding_mask)));
    }
    return result;
}

ValueResult self_attention(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    const std::vector<core::TensorValue> & attention,
    const core::TensorValue & cached_value,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix,
    const std::string & module,
    int64_t heads,
    int64_t value_dim) {
    const std::string base = prefix + "." + module;
    auto values = linear(
        context, input, params, base + ".in_proj");
    const int64_t seq = values.shape.dims[0];
    const int64_t left_context = cached_value.shape.dims[0];
    values = modules::ConcatModule({0}).build(
        context, cached_value, values);
    auto new_state = modules::SliceModule(
        {0, seq, left_context})
                         .build(context, values);
    std::vector<core::TensorValue> pieces;
    pieces.reserve(static_cast<size_t>(heads));
    for (int64_t head = 0; head < heads; ++head) {
        auto value = modules::SliceModule(
            {2, head * value_dim, value_dim})
                         .build(context, values);
        value = transpose(context, value, {1, 0, 2, 3});
        auto mixed = modules::MatMulModule().build(
            context, attention[static_cast<size_t>(head)], value);
        pieces.push_back(
            transpose(context, mixed, {1, 0, 2, 3}));
    }
    auto joined = pieces.front();
    for (size_t index = 1; index < pieces.size(); ++index) {
        joined = modules::ConcatModule({2}).build(
            context, joined, pieces[index]);
    }
    return {
        linear(context, joined, params, base + ".out_proj"),
        new_state};
}

core::TensorValue feed_forward(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix,
    const std::string & module) {
    auto result = linear(
        context,
        input,
        params,
        prefix + "." + module + ".in_proj");
    result = swoosh_l(context, result);
    return linear(
        context,
        result,
        params,
        prefix + "." + module + ".out_proj");
}

ValueResult nonlin_attention(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    const core::TensorValue & attention,
    const core::TensorValue & cached_value,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix) {
    const std::string base = prefix + ".nonlin_attention";
    auto projected =
        linear(context, input, params, base + ".in_proj");
    const int64_t hidden = projected.shape.last_dim() / 3;
    auto gate_input =
        modules::SliceModule({2, 0, hidden})
            .build(context, projected);
    auto value =
        modules::SliceModule({2, hidden, hidden})
            .build(context, projected);
    auto output_gate =
        modules::SliceModule({2, 2 * hidden, hidden})
            .build(context, projected);
    value = mul(
        context,
        value,
        modules::TanhModule().build(context, gate_input));
    const int64_t seq = value.shape.dims[0];
    const int64_t left_context = cached_value.shape.dims[0];
    value = modules::ConcatModule({0}).build(
        context, cached_value, value);
    auto new_state = modules::SliceModule(
        {0, seq, left_context})
                         .build(context, value);
    auto joined = transpose(context, value, {1, 0, 2, 3});
    joined = modules::MatMulModule().build(
        context, attention, joined);
    joined = transpose(context, joined, {1, 0, 2, 3});
    return {
        linear(
            context,
            mul(context, joined, output_gate),
            params,
            base + ".out_proj"),
        new_state};
}

core::TensorValue pad_left_1d(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    int64_t amount) {
    std::array<int32_t, 4> left = {0, 0, 0, 0};
    std::array<int32_t, 4> right = {0, 0, 0, 0};
    left[core::logical_axis_to_ggml_axis(3, 2)] =
        static_cast<int32_t>(amount);
    const auto source = contiguous(context, input);
    auto shape = input.shape;
    shape.dims[2] += amount;
    return core::wrap_tensor(
        ggml_pad_ext(
            context.ggml,
            source.tensor,
            left[0],
            right[0],
            left[1],
            right[1],
            left[2],
            right[2],
            left[3],
            right[3]),
        shape,
        GGML_TYPE_F32);
}

ValueResult convolution_module(
    core::ModuleBuildContext & context,
    std::vector<GraphConstant> & constants,
    const core::TensorValue & input,
    const core::TensorValue & cached_value,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix,
    const std::string & module,
    int64_t kernel) {
    const std::string base = prefix + "." + module;
    const int64_t channels = input.shape.last_dim();
    auto projected =
        linear(context, input, params, base + ".in_proj");
    auto value =
        modules::SliceModule({2, 0, channels})
            .build(context, projected);
    auto gate =
        modules::SliceModule({2, channels, channels})
            .build(context, projected);
    value = mul(
        context,
        value,
        modules::SigmoidModule().build(context, gate));
    auto channel_first =
        transpose(context, value, {1, 2, 0, 3});
    const int64_t left_pad = kernel / 2;
    auto causal_input = modules::ConcatModule({2}).build(
        context, cached_value, channel_first);
    auto new_state = contiguous(
        context,
        modules::SliceModule(
            {2, input.shape.dims[0], left_pad})
            .build(context, causal_input));
    const auto & causal_weight = param(
        params,
        base + ".depthwise_conv.causal_conv.weight");
    const auto & causal_bias = param(
        params,
        base + ".depthwise_conv.causal_conv.bias");
    auto causal =
        modules::DepthwiseConv1dModule(
            {channels, left_pad + 1, 1, 0, 1, true})
        .build(
            context,
            causal_input,
            {causal_weight.tensor, causal_bias.tensor});
    const auto & chunk_weight = param(
        params,
        base + ".depthwise_conv.chunkwise_conv.weight");
    const auto & chunk_bias = param(
        params,
        base + ".depthwise_conv.chunkwise_conv.bias");
    auto chunk = modules::DepthwiseConv1dModule(
        {channels, kernel, 1, static_cast<int>(left_pad), 1, true})
                     .build(
                         context,
                         channel_first,
                         {chunk_weight.tensor, chunk_bias.tensor});
    const auto & left = param(
        params,
        base + ".depthwise_conv.chunk_scale_left");
    const auto & right = param(
        params,
        base + ".depthwise_conv.chunk_scale_right");
    if (left.host.size() !=
            static_cast<size_t>(channels * kernel) ||
        right.host.size() !=
            static_cast<size_t>(channels * kernel)) {
        throw std::runtime_error(
            "Kroko chunk convolution scales are invalid");
    }
    const int64_t seq = input.shape.dims[0];
    std::vector<float> chunk_scale(
        static_cast<size_t>(channels * seq), 1.0F);
    for (int64_t channel = 0; channel < channels; ++channel) {
        for (int64_t frame = 0; frame < seq; ++frame) {
            float edge = 0.0F;
            if (seq < kernel) {
                edge += left.host[static_cast<size_t>(
                    channel * kernel + frame)];
                edge += right.host[static_cast<size_t>(
                    channel * kernel + kernel - seq + frame)];
            } else {
                if (frame < kernel) {
                    edge += left.host[static_cast<size_t>(
                        channel * kernel + frame)];
                }
                if (frame >= seq - kernel) {
                    edge += right.host[static_cast<size_t>(
                        channel * kernel + frame - (seq - kernel))];
                }
            }
            chunk_scale[static_cast<size_t>(
                channel * seq + frame)] += edge;
        }
    }
    auto scale_tensor = graph_constant(
        context,
        constants,
        core::TensorShape::from_dims({1, channels, seq}),
        std::move(chunk_scale));
    auto combined = add(
        context, causal, mul(context, chunk, scale_tensor));
    combined = transpose(
        context, combined, {2, 0, 1, 3});
    combined = swoosh_r(context, combined);
    return {
        linear(
            context, combined, params, base + ".out_proj"),
        new_state};
}

LayerResult zipformer_layer(
    core::ModuleBuildContext & context,
    std::vector<GraphConstant> & constants,
    const core::TensorValue & input,
    const LayerStateValue & state,
    const core::TensorValue & padding_mask,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix,
    int64_t heads,
    int64_t query_dim,
    int64_t value_dim,
    int64_t kernel) {
    const auto original = input;
    const auto attention = attention_weights(
        context,
        constants,
        input,
        state.key,
        padding_mask,
        params,
        prefix,
        heads,
        query_dim);
    auto output = add(
        context,
        input,
        feed_forward(
            context, input, params, prefix, "feed_forward1"));
    const auto nonlinear = nonlin_attention(
        context,
        output,
        attention.weights.front(),
        state.nonlin,
        params,
        prefix);
    output = add(
        context, output, nonlinear.output);
    const auto attention1 = self_attention(
        context,
        output,
        attention.weights,
        state.val1,
        params,
        prefix,
        "self_attn1",
        heads,
        value_dim);
    output = add(
        context, output, attention1.output);
    const auto convolution1 = convolution_module(
        context,
        constants,
        output,
        state.conv1,
        params,
        prefix,
        "conv_module1",
        kernel);
    output = add(
        context, output, convolution1.output);
    output = add(
        context,
        output,
        feed_forward(
            context, output, params, prefix, "feed_forward2"));
    output = bypass(
        context,
        original,
        output,
        param(params, prefix + ".bypass_mid.bypass_scale"));
    const auto attention2 = self_attention(
        context,
        output,
        attention.weights,
        state.val2,
        params,
        prefix,
        "self_attn2",
        heads,
        value_dim);
    output = add(
        context, output, attention2.output);
    const auto convolution2 = convolution_module(
        context,
        constants,
        output,
        state.conv2,
        params,
        prefix,
        "conv_module2",
        kernel);
    output = add(
        context, output, convolution2.output);
    output = add(
        context,
        output,
        feed_forward(
            context, output, params, prefix, "feed_forward3"));
    output = bias_norm(
        context, output, params, prefix + ".norm");
    return {
        bypass(
            context,
            original,
            output,
            param(params, prefix + ".bypass.bypass_scale")),
        {
            attention.key,
            nonlinear.state,
            attention1.state,
            attention2.state,
            convolution1.state,
            convolution2.state,
        }};
}

core::TensorValue convert_channels(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    int64_t channels) {
    const int64_t current = input.shape.last_dim();
    if (current == channels) {
        return input;
    }
    if (current > channels) {
        return modules::SliceModule(
            {2, 0, channels})
            .build(context, input);
    }
    std::array<int32_t, 4> left = {0, 0, 0, 0};
    std::array<int32_t, 4> right = {0, 0, 0, 0};
    right[core::logical_axis_to_ggml_axis(3, 2)] =
        static_cast<int32_t>(channels - current);
    const auto source = contiguous(context, input);
    auto shape = input.shape;
    shape.dims[2] = channels;
    return core::wrap_tensor(
        ggml_pad_ext(
            context.ggml,
            source.tensor,
            left[0],
            right[0],
            left[1],
            right[1],
            left[2],
            right[2],
            left[3],
            right[3]),
        shape,
        GGML_TYPE_F32);
}

core::TensorValue downsample(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    const Param & weights) {
    if (weights.host.empty()) {
        throw std::runtime_error(
            "Kroko downsample weights are unavailable");
    }
    const int64_t factor =
        static_cast<int64_t>(weights.host.size());
    const int64_t output_frames =
        (input.shape.dims[0] + factor - 1) / factor;
    std::vector<core::TensorValue> rows;
    rows.reserve(static_cast<size_t>(output_frames));
    for (int64_t frame = 0; frame < output_frames; ++frame) {
        core::TensorValue sum;
        for (int64_t index = 0; index < factor; ++index) {
            const int64_t source_frame = std::min(
                input.shape.dims[0] - 1,
                frame * factor + index);
            auto row = modules::SliceModule(
                {0, source_frame, 1})
                           .build(context, input);
            row = scale(
                context,
                row,
                weights.host[static_cast<size_t>(index)]);
            sum = sum.valid() ? add(context, sum, row) : row;
        }
        rows.push_back(sum);
    }
    auto output = rows.front();
    for (size_t index = 1; index < rows.size(); ++index) {
        output = modules::ConcatModule({0}).build(
            context, output, rows[index]);
    }
    return output;
}

core::TensorValue upsample(
    core::ModuleBuildContext & context,
    const core::TensorValue & input,
    int64_t factor,
    int64_t target_frames) {
    std::vector<core::TensorValue> rows;
    rows.reserve(static_cast<size_t>(target_frames));
    for (int64_t frame = 0;
         frame < input.shape.dims[0] &&
         static_cast<int64_t>(rows.size()) < target_frames;
         ++frame) {
        auto row =
            modules::SliceModule({0, frame, 1})
                .build(context, input);
        for (int64_t repeat = 0;
             repeat < factor &&
             static_cast<int64_t>(rows.size()) < target_frames;
             ++repeat) {
            rows.push_back(row);
        }
    }
    auto output = rows.front();
    for (size_t index = 1; index < rows.size(); ++index) {
        output = modules::ConcatModule({0}).build(
            context, output, rows[index]);
    }
    return output;
}

class ZipformerGraph {
public:
    ZipformerGraph(
        ggml_backend_t backend,
        core::BackendType backend_type,
        const KrokoASRConfig & config,
        const std::unordered_map<std::string, Param> & params)
        : backend_(backend),
          input_frames_(config.chunk_shift / 2) {
        if (input_frames_ <= 0) {
            throw std::runtime_error(
                "Kroko Zipformer input frame count is invalid");
        }
        ggml_init_params init{
            kGraphContextBytes, nullptr, true};
        context_.reset(ggml_init(init));
        if (context_ == nullptr) {
            throw std::runtime_error(
                "failed to initialize Kroko Zipformer graph context");
        }
        core::ModuleBuildContext build{
            context_.get(), "kroko_asr.zipformer", backend_type};
        auto input = core::make_tensor(
            build,
            GGML_TYPE_F32,
            core::TensorShape::from_dims(
                {input_frames_, 1, kInputDim}));
        input_ = input.tensor;
        ggml_set_input(input_);
        auto current = input;
        std::vector<core::TensorValue> stack_outputs;
        int64_t layer_offset = 0;
        for (size_t stack = 0;
             stack < config.encoder_dims.size();
             ++stack) {
            const int64_t channels = config.encoder_dims[stack];
            const int64_t factor =
                config.downsampling_factors[stack];
            current = convert_channels(build, current, channels);
            const auto stack_input = current;
            if (factor > 1) {
                current = downsample(
                    build,
                    current,
                    param(
                        params,
                        "encoder.encoder.encoders." +
                            std::to_string(stack) +
                            ".downsample.weights"));
            }
            const int64_t left_context =
                config.left_context_len[stack];
            StackMask stack_mask;
            stack_mask.value = core::make_tensor(
                build,
                GGML_TYPE_F32,
                core::TensorShape::from_dims(
                    {1,
                     current.shape.dims[0],
                     left_context + current.shape.dims[0]}));
            stack_mask.input = stack_mask.value.tensor;
            ggml_set_input(stack_mask.input);
            stack_mask.frames = current.shape.dims[0];
            stack_mask.left_context = left_context;
            stack_mask.downsampling = factor;
            stack_mask.host.assign(
                element_count(stack_mask.value.shape), 0.0F);
            stack_masks_.push_back(std::move(stack_mask));
            for (int64_t layer = 0;
                 layer < config.num_encoder_layers[stack];
                 ++layer) {
                const std::string prefix =
                    "encoder.encoder.encoders." +
                    std::to_string(stack) +
                    (stack == 0 ? ".layers." : ".encoder.layers.") +
                    std::to_string(layer);
                LayerState layer_state;
                layer_state.key = state_tensor(
                    build,
                    core::TensorShape::from_dims(
                        {left_context,
                         1,
                         config.num_heads[stack] *
                             config.query_head_dims[stack]}));
                layer_state.nonlin = state_tensor(
                    build,
                    core::TensorShape::from_dims(
                        {left_context, 1, 3 * channels / 4}));
                layer_state.val1 = state_tensor(
                    build,
                    core::TensorShape::from_dims(
                        {left_context,
                         1,
                         config.num_heads[stack] *
                             config.value_head_dims[stack]}));
                layer_state.val2 = state_tensor(
                    build,
                    core::TensorShape::from_dims(
                        {left_context,
                         1,
                         config.num_heads[stack] *
                             config.value_head_dims[stack]}));
                layer_state.conv1 = state_tensor(
                    build,
                    core::TensorShape::from_dims(
                        {1,
                         channels,
                         config.cnn_module_kernels[stack] / 2}));
                layer_state.conv2 = state_tensor(
                    build,
                    core::TensorShape::from_dims(
                        {1,
                         channels,
                         config.cnn_module_kernels[stack] / 2}));
                const auto layer_result = zipformer_layer(
                    build,
                    constants_,
                    current,
                    {
                        layer_state.key.value,
                        layer_state.nonlin.value,
                        layer_state.val1.value,
                        layer_state.val2.value,
                        layer_state.conv1.value,
                        layer_state.conv2.value,
                    },
                    stack_masks_.back().value,
                    params,
                    prefix,
                    config.num_heads[stack],
                    config.query_head_dims[stack],
                    config.value_head_dims[stack],
                    config.cnn_module_kernels[stack]);
                current = layer_result.output;
                layer_state.key.output =
                    layer_result.state.key.tensor;
                layer_state.nonlin.output =
                    layer_result.state.nonlin.tensor;
                layer_state.val1.output =
                    layer_result.state.val1.tensor;
                layer_state.val2.output =
                    layer_result.state.val2.tensor;
                layer_state.conv1.output =
                    layer_result.state.conv1.tensor;
                layer_state.conv2.output =
                    layer_result.state.conv2.tensor;
                layer_states_.push_back(std::move(layer_state));
                ++layer_offset;
            }
            if (factor > 1) {
                current = upsample(
                    build,
                    current,
                    factor,
                    stack_input.shape.dims[0]);
                current = bypass(
                    build,
                    stack_input,
                    current,
                    param(
                        params,
                        "encoder.encoder.encoders." +
                            std::to_string(stack) +
                            ".out_combiner.bypass_scale"));
            }
            stack_outputs.push_back(current);
        }
        if (layer_offset != 19) {
            throw std::runtime_error(
                "Kroko Zipformer layer count is not 19");
        }

        std::vector<core::TensorValue> full_pieces;
        full_pieces.push_back(stack_outputs.back());
        int64_t current_dim =
            config.encoder_dims.back();
        for (int64_t stack =
                 static_cast<int64_t>(stack_outputs.size()) - 2;
             stack >= 0;
             --stack) {
            const int64_t stack_dim =
                config.encoder_dims[static_cast<size_t>(stack)];
            if (stack_dim > current_dim) {
                full_pieces.push_back(
                    modules::SliceModule(
                        {2, current_dim, stack_dim - current_dim})
                        .build(
                            build,
                            stack_outputs[
                                static_cast<size_t>(stack)]));
                current_dim = stack_dim;
            }
        }
        current = full_pieces.front();
        for (size_t index = 1;
             index < full_pieces.size();
             ++index) {
            current = modules::ConcatModule({2}).build(
                build, current, full_pieces[index]);
        }
        current = downsample(
            build,
            current,
            param(
                params,
                "encoder.encoder.downsample_output.weights"));
        current = linear(
            build,
            current,
            params,
            "encoder.encoder_proj");
        output_frames_ = current.shape.dims[0];
        output_dim_ = current.shape.dims[2];
        output_ = current.tensor;
        ggml_set_output(output_);
        for (auto & state : layer_states_) {
            ggml_set_output(state.key.output);
            ggml_set_output(state.nonlin.output);
            ggml_set_output(state.val1.output);
            ggml_set_output(state.val2.output);
            ggml_set_output(state.conv1.output);
            ggml_set_output(state.conv2.output);
        }
        graph_ = ggml_new_graph_custom(
            context_.get(), kGraphNodes, false);
        ggml_build_forward_expand(graph_, output_);
        for (const auto & state : layer_states_) {
            ggml_build_forward_expand(graph_, state.key.output);
            ggml_build_forward_expand(
                graph_, state.nonlin.output);
            ggml_build_forward_expand(graph_, state.val1.output);
            ggml_build_forward_expand(graph_, state.val2.output);
            ggml_build_forward_expand(graph_, state.conv1.output);
            ggml_build_forward_expand(graph_, state.conv2.output);
        }
        allocator_ = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend_));
        if (allocator_ == nullptr ||
            !ggml_gallocr_reserve(allocator_, graph_) ||
            !ggml_gallocr_alloc_graph(allocator_, graph_)) {
            throw std::runtime_error(
                "failed to allocate Kroko Zipformer graph");
        }
        for (const auto & constant : constants_) {
            ggml_backend_tensor_set(
                constant.tensor,
                constant.values.data(),
                0,
                constant.values.size() * sizeof(float));
        }
        plan_ =
            core::create_backend_graph_plan_if_host(backend_, graph_);
    }

    ~ZipformerGraph() {
        core::release_backend_graph_resources(backend_, graph_);
        if (plan_ != nullptr) {
            core::free_backend_graph_plan(backend_, plan_);
        }
        if (allocator_ != nullptr) {
            ggml_gallocr_free(allocator_);
        }
    }

    KrokoEncoderChunk run(
        const std::vector<float> & input) {
        if (input.size() !=
            static_cast<size_t>(input_frames_ * kInputDim)) {
            throw std::runtime_error(
                "Kroko Zipformer input shape does not match the package chunk size");
        }
        ggml_backend_tensor_set_async(
            backend_,
            input_,
            input.data(),
            0,
            input.size() * sizeof(float));
        const auto set_state = [&](StateTensor & state) {
            ggml_backend_tensor_set_async(
                backend_,
                state.input,
                state.host.data(),
                0,
                state.host.size() * sizeof(float));
        };
        for (auto & state : layer_states_) {
            set_state(state.key);
            set_state(state.nonlin);
            set_state(state.val1);
            set_state(state.val2);
            set_state(state.conv1);
            set_state(state.conv2);
        }
        for (auto & mask : stack_masks_) {
            const int64_t valid_left = std::min(
                mask.left_context,
                processed_frames_ / mask.downsampling);
            std::fill(
                mask.host.begin(), mask.host.end(), 0.0F);
            for (int64_t target = 0;
                 target < mask.frames;
                 ++target) {
                for (int64_t source = 0;
                     source < mask.left_context - valid_left;
                     ++source) {
                    mask.host[static_cast<size_t>(
                        target *
                            (mask.left_context + mask.frames) +
                        source)] = -1000.0F;
                }
            }
            ggml_backend_tensor_set_async(
                backend_,
                mask.input,
                mask.host.data(),
                0,
                mask.host.size() * sizeof(float));
        }
        for (const auto & constant : constants_) {
            ggml_backend_tensor_set_async(
                backend_,
                constant.tensor,
                constant.values.data(),
                0,
                constant.values.size() * sizeof(float));
        }
        ggml_backend_synchronize(backend_);
        const auto status = core::compute_backend_graph(
            backend_,
            graph_,
            plan_,
            "Kroko Zipformer");
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(
                "Kroko Zipformer graph compute failed");
        }
        KrokoEncoderChunk result;
        result.frames = output_frames_;
        result.channels = output_dim_;
        result.values.resize(
            static_cast<size_t>(output_frames_ * output_dim_));
        ggml_backend_tensor_get_async(
            backend_,
            output_,
            result.values.data(),
            0,
            result.values.size() * sizeof(float));
        const auto get_state = [&](StateTensor & state) {
            ggml_backend_tensor_get_async(
                backend_,
                state.output,
                state.host.data(),
                0,
                state.host.size() * sizeof(float));
        };
        for (auto & state : layer_states_) {
            get_state(state.key);
            get_state(state.nonlin);
            get_state(state.val1);
            get_state(state.val2);
            get_state(state.conv1);
            get_state(state.conv2);
        }
        ggml_backend_synchronize(backend_);
        processed_frames_ += input_frames_;
        return result;
    }

    void reset() {
        processed_frames_ = 0;
        const auto clear = [](StateTensor & value) {
            std::fill(
                value.host.begin(),
                value.host.end(),
                0.0F);
        };
        for (auto & state : layer_states_) {
            clear(state.key);
            clear(state.nonlin);
            clear(state.val1);
            clear(state.val2);
            clear(state.conv1);
            clear(state.conv2);
        }
    }

private:
    ggml_backend_t backend_ = nullptr;
    std::unique_ptr<ggml_context, ContextDeleter> context_;
    std::vector<GraphConstant> constants_;
    std::vector<LayerState> layer_states_;
    std::vector<StackMask> stack_masks_;
    int64_t processed_frames_ = 0;
    int64_t input_frames_ = 0;
    int64_t output_frames_ = 0;
    int64_t output_dim_ = 0;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
    ggml_backend_graph_plan_t plan_ = nullptr;
};

bool needs_host_values(std::string_view name) {
    const auto ends_with = [&](std::string_view suffix) {
        return name.size() >= suffix.size() &&
               name.substr(name.size() - suffix.size()) == suffix;
    };
    return ends_with(".linear_pos.weight") ||
           ends_with(".norm.log_scale") ||
           ends_with(".bypass_scale") ||
           name.find(".downsample.weights") !=
               std::string_view::npos ||
           name.find(".downsample_output.weights") !=
               std::string_view::npos ||
           ends_with(".chunk_scale_left") ||
           ends_with(".chunk_scale_right");
}

bool is_zipformer_tensor(std::string_view name) {
    const auto starts_with = [&](std::string_view prefix) {
        return name.size() >= prefix.size() &&
               name.substr(0, prefix.size()) == prefix;
    };
    return starts_with("encoder.encoder.") ||
           starts_with("encoder.encoder_proj.");
}

}  // namespace

struct KrokoZipformerRuntime::Impl {
    std::shared_ptr<const KrokoASRAssets> assets;
    std::shared_ptr<core::BackendWeightStore> store;
    std::unordered_map<std::string, Param> params;
    std::unique_ptr<ZipformerGraph> graph;
};

KrokoZipformerRuntime::KrokoZipformerRuntime(
    std::shared_ptr<const KrokoASRAssets> assets,
    core::ExecutionContext & execution_context)
    : impl_(std::make_unique<Impl>()) {
    if (assets == nullptr || assets->weights == nullptr) {
        throw std::runtime_error(
            "Kroko Zipformer requires tensor assets");
    }
    impl_->assets = std::move(assets);
    impl_->store = std::make_shared<core::BackendWeightStore>(
        execution_context.backend(),
        execution_context.backend_type(),
        "Kroko Zipformer weights",
        kWeightContextBytes);
    const auto & source = *impl_->assets->weights;
    for (const auto & metadata : source.tensors()) {
        if (!is_zipformer_tensor(metadata.name)) {
            continue;
        }
        Param value;
        value.shape = metadata.shape.empty()
            ? std::vector<int64_t>{1}
            : metadata.shape;
        const bool keep_host_type =
            metadata.shape.size() == 2 &&
            !needs_host_values(metadata.name);
        value.tensor = keep_host_type
            ? impl_->store->load_tensor(
                  source,
                  metadata.name,
                  assets::TensorStorageType::Native,
                  value.shape)
            : impl_->store->load_f32_tensor(
                  source, metadata.name, value.shape);
        if (needs_host_values(metadata.name)) {
            value.host =
                source.require_f32(metadata.name, value.shape);
        }
        impl_->params.emplace(
            metadata.name, std::move(value));
    }
    impl_->store->upload();
    impl_->graph = std::make_unique<ZipformerGraph>(
        execution_context.backend(),
        execution_context.backend_type(),
        impl_->assets->config,
        impl_->params);
}

KrokoZipformerRuntime::~KrokoZipformerRuntime() = default;

KrokoEncoderChunk
KrokoZipformerRuntime::encode_chunk(
    const std::vector<float> & subsampled_features) const {
    if (impl_ == nullptr || impl_->graph == nullptr) {
        throw std::runtime_error(
            "Kroko Zipformer is not initialized");
    }
    return impl_->graph->run(subsampled_features);
}

void KrokoZipformerRuntime::reset() const {
    if (impl_ == nullptr || impl_->graph == nullptr) {
        throw std::runtime_error(
            "Kroko Zipformer is not initialized");
    }
    impl_->graph->reset();
}

}  // namespace engine::models::kroko_asr
