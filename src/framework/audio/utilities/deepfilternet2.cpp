#include "engine/framework/audio/utilities/deepfilternet2.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/fft.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/io/safetensors.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <complex>
#include <cstring>
#include <cmath>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace engine::audio {
namespace {

constexpr size_t kDeepFilterNet2ForwardGraphContextBytes = 128ull * 1024ull * 1024ull;

struct BackendDeleter {
    void operator()(ggml_backend * backend) const noexcept {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct NamedWeight {
    std::vector<int64_t> shape;
    std::vector<float> values;
    core::TensorValue tensor;
};

int64_t numel(const std::vector<int64_t> & shape) {
    if (shape.empty()) {
        return 1;
    }
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, [](int64_t lhs, int64_t rhs) {
        if (rhs <= 0) {
            throw std::runtime_error("DeepFilterNet2 tensor shape must be positive");
        }
        return lhs * rhs;
    });
}

core::TensorShape tensor_shape_from_vector(const std::vector<int64_t> & shape) {
    if (shape.empty() || shape.size() > core::kMaxTensorRank) {
        throw std::runtime_error("DeepFilterNet2 weight rank must be between 1 and 4");
    }
    if (shape.size() == 1) {
        return core::TensorShape::from_dims({shape[0]});
    }
    if (shape.size() == 2) {
        return core::TensorShape::from_dims({shape[0], shape[1]});
    }
    if (shape.size() == 3) {
        return core::TensorShape::from_dims({shape[0], shape[1], shape[2]});
    }
    return core::TensorShape::from_dims({shape[0], shape[1], shape[2], shape[3]});
}

void require_weight(
    const std::unordered_map<std::string, NamedWeight> & weights,
    const std::string & name,
    const std::vector<int64_t> & shape) {
    const auto it = weights.find(name);
    if (it == weights.end()) {
        throw std::runtime_error("DeepFilterNet2 missing required weight: " + name);
    }
    if (it->second.shape != shape) {
        throw std::runtime_error("DeepFilterNet2 required weight shape mismatch: " + name);
    }
}

void validate_deepfilternet2_weights(const std::unordered_map<std::string, NamedWeight> & weights) {
    require_weight(weights, "enc.erb_conv0.1.weight", {64, 1, 3, 3});
    require_weight(weights, "enc.df_conv0.1.weight", {64, 1, 3, 3});
    require_weight(weights, "enc.emb_gru.linear_in.0.weight", {8, 128, 32});
    require_weight(weights, "enc.emb_gru.gru.weight_ih_l0", {768, 256});
    require_weight(weights, "erb_dec.emb_gru.linear_out.0.weight", {8, 32, 64});
    require_weight(weights, "erb_dec.convt2.0.weight", {64, 1, 1, 3});
    require_weight(weights, "df_dec.df_convp.1.weight", {10, 32, 5, 1});
    require_weight(weights, "df_dec.df_gru.gru.weight_ih_l1", {768, 256});
    require_weight(weights, "df_dec.df_out.0.weight", {8, 32, 120});
    require_weight(weights, "mask.erb_inv_fb", {32, 481});
}

void trim_deepfilternet2_host_weights(std::unordered_map<std::string, NamedWeight> & weights) {
    for (auto & [name, weight] : weights) {
        if (name == "mask.erb_inv_fb") {
            continue;
        }
        weight.values.clear();
        weight.values.shrink_to_fit();
    }
}

void add_f32_weight(
    std::unordered_map<std::string, NamedWeight> & weights,
    core::BackendWeightStore & store,
    const std::string & name,
    std::vector<int64_t> shape,
    std::vector<float> values) {
    NamedWeight w;
    w.shape = std::move(shape);
    w.values = std::move(values);
    w.tensor = store.make_f32(tensor_shape_from_vector(w.shape), w.values);
    weights.emplace(name, std::move(w));
}

void add_deepfilternet2_derived_weights(
    std::unordered_map<std::string, NamedWeight> & weights,
    core::BackendWeightStore & store) {
    std::vector<std::string> names;
    names.reserve(weights.size());
    for (const auto & [name, w] : weights) {
        names.push_back(name);
    }
    for (const auto & name : names) {
        const auto it = weights.find(name);
        if (it == weights.end()) {
            throw std::runtime_error("DeepFilterNet2 missing weight while creating derived tensors: " + name);
        }
        const auto & w = it->second;
        if (w.shape.size() == 3) {
            const int64_t groups = w.shape[0];
            const int64_t in_per_group = w.shape[1];
            const int64_t out_per_group = w.shape[2];
            std::vector<float> values(static_cast<size_t>(groups * out_per_group * in_per_group));
            for (int64_t g = 0; g < groups; ++g) {
                for (int64_t o = 0; o < out_per_group; ++o) {
                    for (int64_t i = 0; i < in_per_group; ++i) {
                        values[static_cast<size_t>((g * out_per_group + o) * in_per_group + i)] =
                            w.values[static_cast<size_t>((g * in_per_group + i) * out_per_group + o)];
                    }
                }
            }
            add_f32_weight(weights, store, name + ".__linear", {groups, out_per_group, in_per_group}, std::move(values));
        }
        if (name.size() > 7 && name.rfind(".weight") == name.size() - 7) {
            const auto prefix = name.substr(0, name.size() - 7);
            const auto bias_it = weights.find(prefix + ".bias");
            const auto mean_it = weights.find(prefix + ".running_mean");
            const auto var_it = weights.find(prefix + ".running_var");
            if (bias_it != weights.end() && mean_it != weights.end() && var_it != weights.end() && w.shape.size() == 1) {
                const int64_t channels = w.shape[0];
                std::vector<float> scale(static_cast<size_t>(channels));
                std::vector<float> bias(static_cast<size_t>(channels));
                for (int64_t c = 0; c < channels; ++c) {
                    const float s = w.values[static_cast<size_t>(c)] /
                                    std::sqrt(var_it->second.values[static_cast<size_t>(c)] + 1.0e-5f);
                    scale[static_cast<size_t>(c)] = s;
                    bias[static_cast<size_t>(c)] =
                        bias_it->second.values[static_cast<size_t>(c)] -
                        mean_it->second.values[static_cast<size_t>(c)] * s;
                }
                add_f32_weight(weights, store, prefix + ".__bn_scale", {channels}, std::move(scale));
                add_f32_weight(weights, store, prefix + ".__bn_bias", {channels}, std::move(bias));
            }
        }
    }
}

const NamedWeight & weight(const std::unordered_map<std::string, NamedWeight> & weights, const std::string & name) {
    const auto it = weights.find(name);
    if (it == weights.end()) {
        throw std::runtime_error("DeepFilterNet2 missing weight: " + name);
    }
    return it->second;
}

core::TensorValue graph_pad2d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t left,
    int64_t right,
    int64_t top,
    int64_t bottom) {
    std::array<int32_t, 4> lp = {0, 0, 0, 0};
    std::array<int32_t, 4> rp = {0, 0, 0, 0};
    lp[core::logical_axis_to_ggml_axis(input.shape.rank, 2)] = static_cast<int32_t>(top);
    rp[core::logical_axis_to_ggml_axis(input.shape.rank, 2)] = static_cast<int32_t>(bottom);
    lp[core::logical_axis_to_ggml_axis(input.shape.rank, 3)] = static_cast<int32_t>(left);
    rp[core::logical_axis_to_ggml_axis(input.shape.rank, 3)] = static_cast<int32_t>(right);
    auto shape = input.shape;
    shape.dims[2] += top + bottom;
    shape.dims[3] += left + right;
    return core::wrap_tensor(
        ggml_pad_ext(ctx.ggml, input.tensor, lp[0], rp[0], lp[1], rp[1], lp[2], rp[2], lp[3], rp[3]),
        shape,
        input.type);
}

core::TensorValue graph_contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    return core::ensure_backend_addressable_layout(ctx, input);
}

core::TensorValue graph_reshape(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorShape & shape) {
    return core::reshape_tensor(ctx, graph_contiguous(ctx, input), shape);
}

core::TensorValue graph_transpose(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::array<int, core::kMaxTensorRank> & axes,
    size_t rank) {
    if (rank != input.shape.rank) {
        throw std::runtime_error("DeepFilterNet2 transpose rank mismatch");
    }
    core::TensorShape output_shape = {};
    output_shape.rank = rank;
    std::array<bool, core::kMaxTensorRank> seen = {false, false, false, false};
    std::array<int, core::kMaxTensorRank> ggml_axes = {0, 1, 2, 3};
    for (size_t out_axis = 0; out_axis < rank; ++out_axis) {
        const int in_axis = axes[out_axis];
        if (in_axis < 0 || in_axis >= static_cast<int>(rank) || seen[static_cast<size_t>(in_axis)]) {
            throw std::runtime_error("DeepFilterNet2 transpose axes must be a permutation");
        }
        seen[static_cast<size_t>(in_axis)] = true;
        output_shape.dims[out_axis] = input.shape.dims[static_cast<size_t>(in_axis)];
        const int out_ggml_axis = core::logical_axis_to_ggml_axis(rank, static_cast<int>(out_axis));
        const int in_ggml_axis = core::logical_axis_to_ggml_axis(rank, in_axis);
        ggml_axes[static_cast<size_t>(in_ggml_axis)] = out_ggml_axis;
    }
    return core::wrap_tensor(
        ggml_permute(ctx.ggml, graph_contiguous(ctx, input).tensor, ggml_axes[0], ggml_axes[1], ggml_axes[2], ggml_axes[3]),
        output_shape,
        input.type);
}

core::TensorValue graph_add(core::ModuleBuildContext & ctx, const core::TensorValue & lhs, const core::TensorValue & rhs) {
    return modules::AddModule().build(ctx, lhs, rhs);
}

core::TensorValue graph_mul(core::ModuleBuildContext & ctx, const core::TensorValue & lhs, const core::TensorValue & rhs) {
    return modules::MulModule().build(ctx, lhs, rhs);
}

core::TensorValue graph_scale(core::ModuleBuildContext & ctx, const core::TensorValue & input, float scale) {
    return core::wrap_tensor(ggml_scale(ctx.ggml, graph_contiguous(ctx, input).tensor, scale), input.shape, GGML_TYPE_F32);
}

core::TensorValue graph_scale_bias(core::ModuleBuildContext & ctx, const core::TensorValue & input, float scale, float bias) {
    return core::wrap_tensor(ggml_scale_bias(ctx.ggml, graph_contiguous(ctx, input).tensor, scale, bias), input.shape, GGML_TYPE_F32);
}

core::TensorValue graph_repeat_like(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & target) {
    return core::wrap_tensor(ggml_repeat(ctx.ggml, input.tensor, target.tensor), target.shape, GGML_TYPE_F32);
}

core::TensorValue graph_weight_scalar_like(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & scalar,
    const core::TensorValue & target) {
    auto view = graph_reshape(ctx, scalar, core::TensorShape::from_dims({1, 1, 1, 1}));
    return graph_repeat_like(ctx, view, target);
}

core::TensorValue graph_linear(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NamedWeight & weight_tensor,
    const NamedWeight * bias_tensor,
    int64_t out_features,
    const std::string & label) {
    if (weight_tensor.shape.size() != 2 || weight_tensor.shape[0] != out_features || weight_tensor.shape[1] != input.shape.last_dim()) {
        throw std::runtime_error("DeepFilterNet2 graph linear shape mismatch: " + label);
    }
    std::optional<core::TensorValue> bias;
    if (bias_tensor != nullptr) {
        if (bias_tensor->shape != std::vector<int64_t>{out_features}) {
            throw std::runtime_error("DeepFilterNet2 graph linear bias shape mismatch: " + label);
        }
        bias = bias_tensor->tensor;
    }
    return modules::LinearModule({input.shape.last_dim(), out_features, bias.has_value()}).build(
        ctx,
        input,
        modules::LinearWeights{weight_tensor.tensor, bias});
}

core::TensorValue graph_conv2d_direct(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NamedWeight & weight_tensor,
    const NamedWeight * bias_tensor,
    int64_t stride_t,
    int64_t stride_f,
    int64_t pad_t,
    int64_t pad_f,
    int64_t dilation_t,
    int64_t dilation_f,
    int64_t groups,
    const std::string & label) {
    if (weight_tensor.shape.size() != 4) {
        throw std::runtime_error("DeepFilterNet2 graph conv weight rank mismatch: " + label);
    }
    const int64_t out_c = weight_tensor.shape[0];
    const int64_t in_per_group = weight_tensor.shape[1];
    if (groups <= 0 || input.shape.dims[1] != in_per_group * groups || out_c % groups != 0) {
        throw std::runtime_error("DeepFilterNet2 graph conv group shape mismatch: " + label);
    }
    if (groups == 1) {
        std::optional<core::TensorValue> bias;
        if (bias_tensor != nullptr) {
            bias = bias_tensor->tensor;
        }
        return modules::Conv2dModule({
            input.shape.dims[1],
            out_c,
            weight_tensor.shape[2],
            weight_tensor.shape[3],
            static_cast<int>(stride_t),
            static_cast<int>(stride_f),
            static_cast<int>(pad_t),
            static_cast<int>(pad_f),
            static_cast<int>(dilation_t),
            static_cast<int>(dilation_f),
            bias.has_value(),
        }).build(ctx, input, modules::Conv2dWeights{weight_tensor.tensor, bias});
    }

    const int64_t out_per_group = out_c / groups;
    core::TensorValue output;
    for (int64_t group = 0; group < groups; ++group) {
        auto input_slice = modules::SliceModule({1, group * in_per_group, in_per_group}).build(ctx, input);
        auto weight_slice = modules::SliceModule({0, group * out_per_group, out_per_group}).build(ctx, weight_tensor.tensor);
        std::optional<core::TensorValue> bias_slice;
        if (bias_tensor != nullptr) {
            bias_slice = modules::SliceModule({0, group * out_per_group, out_per_group}).build(ctx, bias_tensor->tensor);
        }
        auto group_output = modules::Conv2dModule({
            in_per_group,
            out_per_group,
            weight_tensor.shape[2],
            weight_tensor.shape[3],
            static_cast<int>(stride_t),
            static_cast<int>(stride_f),
            static_cast<int>(pad_t),
            static_cast<int>(pad_f),
            static_cast<int>(dilation_t),
            static_cast<int>(dilation_f),
            bias_slice.has_value(),
        }).build(ctx, input_slice, modules::Conv2dWeights{weight_slice, bias_slice});
        output = output.valid() ? modules::ConcatModule({1}).build(ctx, output, group_output) : group_output;
    }
    return output;
}

core::TensorValue graph_batch_norm_act(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NamedWeight & scale,
    const NamedWeight & bias,
    bool relu,
    bool sigmoid_output,
    const std::string & label) {
    if (scale.shape != std::vector<int64_t>{input.shape.dims[1]} || bias.shape != std::vector<int64_t>{input.shape.dims[1]}) {
        throw std::runtime_error("DeepFilterNet2 graph batch norm fused shape mismatch: " + label);
    }
    auto scale_view = graph_reshape(ctx, scale.tensor, core::TensorShape::from_dims({1, input.shape.dims[1], 1, 1}));
    auto bias_view = graph_reshape(ctx, bias.tensor, core::TensorShape::from_dims({1, input.shape.dims[1], 1, 1}));
    auto y = graph_add(ctx, graph_mul(ctx, input, graph_repeat_like(ctx, scale_view, input)), graph_repeat_like(ctx, bias_view, input));
    if (relu) {
        y = modules::ReluModule().build(ctx, y);
    }
    if (sigmoid_output) {
        y = modules::SigmoidModule().build(ctx, y);
    }
    return y;
}

struct Tensor4 {
    int64_t b = 0;
    int64_t c = 0;
    int64_t t = 0;
    int64_t f = 0;
    std::vector<float> v;

    Tensor4() = default;
    Tensor4(int64_t batch, int64_t channels, int64_t frames, int64_t bins)
        : b(batch), c(channels), t(frames), f(bins), v(static_cast<size_t>(batch * channels * frames * bins), 0.0f) {}

    float & at(int64_t bi, int64_t ci, int64_t ti, int64_t fi) {
        return v[static_cast<size_t>((((bi * c + ci) * t + ti) * f) + fi)];
    }

    float at(int64_t bi, int64_t ci, int64_t ti, int64_t fi) const {
        return v[static_cast<size_t>((((bi * c + ci) * t + ti) * f) + fi)];
    }
};

Tensor4 tensor4_from_values(
    const std::vector<float> & values,
    const std::vector<int64_t> & shape,
    const std::string & label) {
    if (shape.size() != 4 || shape[0] != 1) {
        throw std::runtime_error("DeepFilterNet2 " + label + " shape must be [1,C,T,F]");
    }
    const int64_t count = shape[0] * shape[1] * shape[2] * shape[3];
    if (static_cast<int64_t>(values.size()) != count) {
        throw std::runtime_error("DeepFilterNet2 " + label + " value count mismatch");
    }
    Tensor4 out(shape[0], shape[1], shape[2], shape[3]);
    out.v = values;
    return out;
}

core::TensorValue graph_conv2d_norm_act(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::unordered_map<std::string, NamedWeight> & weights,
    const std::string & prefix,
    int64_t out_channels,
    int64_t kernel_t,
    int64_t kernel_f,
    int64_t fstride,
    bool separable,
    bool sigmoid_output) {
    const int64_t pad_t = kernel_t - 1;
    const int64_t pad_f = kernel_f / 2;
    const int64_t groups = separable ? std::gcd(input.shape.dims[1], out_channels) : 1;
    const bool has_time_pad = pad_t > 0;
    const int conv_index = has_time_pad ? 1 : 0;
    const bool has_pointwise = separable && groups > 1 && std::max(kernel_t, kernel_f) > 1;
    const int norm_index = conv_index + (has_pointwise ? 2 : 1);
    auto x = has_time_pad ? graph_pad2d(ctx, input, 0, 0, pad_t, 0) : input;
    x = graph_conv2d_direct(
        ctx,
        x,
        weight(weights, prefix + "." + std::to_string(conv_index) + ".weight"),
        nullptr,
        1,
        fstride,
        0,
        pad_f,
        1,
        1,
        groups,
        prefix + "." + std::to_string(conv_index));
    if (has_pointwise) {
        x = graph_conv2d_direct(
            ctx,
            x,
            weight(weights, prefix + "." + std::to_string(conv_index + 1) + ".weight"),
            nullptr,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            prefix + "." + std::to_string(conv_index + 1));
    }
    const std::string norm_prefix = prefix + "." + std::to_string(norm_index);
    return graph_batch_norm_act(
        ctx,
        x,
        weight(weights, norm_prefix + ".__bn_scale"),
        weight(weights, norm_prefix + ".__bn_bias"),
        !sigmoid_output,
        sigmoid_output,
        norm_prefix);
}

core::TensorValue graph_flatten_conv_feature(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    auto x = graph_transpose(ctx, input, {{0, 2, 3, 1}}, 4);
    return graph_reshape(ctx, x, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[2], input.shape.dims[3] * input.shape.dims[1]}));
}

core::TensorValue graph_concat_last(core::ModuleBuildContext & ctx, const core::TensorValue & lhs, const core::TensorValue & rhs) {
    return modules::ConcatModule({2}).build(ctx, lhs, rhs);
}

core::TensorValue graph_grouped_linear(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::unordered_map<std::string, NamedWeight> & weights,
    const std::string & name,
    bool relu) {
    const auto & original = weight(weights, name);
    const std::string transposed_name = name + ".__linear";
    const auto & transposed = weight(weights, transposed_name);
    if (original.shape.size() != 3) {
        throw std::runtime_error("DeepFilterNet2 graph grouped linear weight rank mismatch: " + name);
    }
    const int64_t groups = original.shape[0];
    const int64_t in_per_group = original.shape[1];
    const int64_t out_per_group = original.shape[2];
    if (input.shape.last_dim() != groups * in_per_group) {
        throw std::runtime_error("DeepFilterNet2 graph grouped linear input shape mismatch: " + name);
    }
    core::TensorValue output;
    for (int64_t group = 0; group < groups; ++group) {
        auto input_slice = modules::SliceModule({2, group * in_per_group, in_per_group}).build(ctx, input);
        auto weight_slice = modules::SliceModule({0, group, 1}).build(ctx, transposed.tensor);
        weight_slice = graph_reshape(ctx, weight_slice, core::TensorShape::from_dims({out_per_group, in_per_group}));
        auto group_output = graph_linear(ctx, input_slice, NamedWeight{{out_per_group, in_per_group}, {}, weight_slice}, nullptr, out_per_group, name);
        output = output.valid() ? modules::ConcatModule({2}).build(ctx, output, group_output) : group_output;
    }
    if (relu) {
        output = modules::ReluModule().build(ctx, output);
    }
    return output;
}

core::TensorValue graph_linear_sigmoid(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::unordered_map<std::string, NamedWeight> & weights,
    const std::string & prefix,
    float scale,
    float offset) {
    auto x = graph_linear(
        ctx,
        input,
        weight(weights, prefix + ".0.weight"),
        &weight(weights, prefix + ".0.bias"),
        1,
        prefix);
    x = modules::SigmoidModule().build(ctx, x);
    return graph_scale_bias(ctx, x, scale, offset);
}

core::TensorValue graph_gru_sequence(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::unordered_map<std::string, NamedWeight> & weights,
    const std::string & prefix,
    int layers,
    bool skip_input) {
    auto x = input;
    const auto residual = input;
    constexpr int64_t hidden = 256;
    for (int layer = 0; layer < layers; ++layer) {
        const std::string suffix = "_l" + std::to_string(layer);
        const std::string w_ih_name = prefix + ".weight_ih" + suffix;
        const std::string w_hh_name = prefix + ".weight_hh" + suffix;
        const std::string b_ih_name = prefix + ".bias_ih" + suffix;
        const std::string b_hh_name = prefix + ".bias_hh" + suffix;
        const auto & w_ih = weight(weights, w_ih_name);
        const auto & w_hh = weight(weights, w_hh_name);
        const auto & b_ih = weight(weights, b_ih_name);
        const auto & b_hh = weight(weights, b_hh_name);
        auto h0 = modules::SliceModule({0, 0, hidden}).build(ctx, b_ih.tensor);
        auto h = graph_scale(ctx, graph_reshape(ctx, h0, core::TensorShape::from_dims({1, hidden})), 0.0f);
        core::TensorValue output;
        for (int64_t t = 0; t < x.shape.dims[1]; ++t) {
            auto x_t = modules::SliceModule({1, t, 1}).build(ctx, x);
            x_t = graph_reshape(ctx, x_t, core::TensorShape::from_dims({1, x.shape.last_dim()}));
            auto gate_x = graph_linear(ctx, x_t, w_ih, &b_ih, 3 * hidden, prefix + suffix + ".ih");
            auto gate_h = graph_linear(ctx, h, w_hh, &b_hh, 3 * hidden, prefix + suffix + ".hh");
            auto gx_r = modules::SliceModule({1, 0, hidden}).build(ctx, gate_x);
            auto gx_z = modules::SliceModule({1, hidden, hidden}).build(ctx, gate_x);
            auto gx_n = modules::SliceModule({1, 2 * hidden, hidden}).build(ctx, gate_x);
            auto gh_r = modules::SliceModule({1, 0, hidden}).build(ctx, gate_h);
            auto gh_z = modules::SliceModule({1, hidden, hidden}).build(ctx, gate_h);
            auto gh_n = modules::SliceModule({1, 2 * hidden, hidden}).build(ctx, gate_h);
            auto r = modules::SigmoidModule().build(ctx, graph_add(ctx, gx_r, gh_r));
            auto z = modules::SigmoidModule().build(ctx, graph_add(ctx, gx_z, gh_z));
            auto n = modules::TanhModule().build(ctx, graph_add(ctx, gx_n, graph_mul(ctx, r, gh_n)));
            auto one_minus_z = graph_scale_bias(ctx, z, -1.0f, 1.0f);
            h = graph_add(ctx, graph_mul(ctx, one_minus_z, n), graph_mul(ctx, z, h));
            auto h_frame = graph_reshape(ctx, h, core::TensorShape::from_dims({1, 1, hidden}));
            output = output.valid() ? modules::ConcatModule({1}).build(ctx, output, h_frame) : h_frame;
        }
        x = output;
    }
    if (skip_input) {
        x = graph_add(ctx, x, residual);
    }
    return x;
}

core::TensorValue graph_squeezed_gru(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::unordered_map<std::string, NamedWeight> & weights,
    const std::string & prefix,
    int layers,
    bool relu_in,
    bool skip_input,
    const std::string & linear_out_name) {
    auto x = graph_grouped_linear(ctx, input, weights, prefix + ".linear_in.0.weight", relu_in);
    x = graph_gru_sequence(ctx, x, weights, prefix + ".gru", layers, skip_input);
    if (!linear_out_name.empty()) {
        x = graph_grouped_linear(ctx, x, weights, linear_out_name, true);
    }
    return x;
}

struct GraphEncodedDeepFilterNet2 {
    core::TensorValue e0;
    core::TensorValue e1;
    core::TensorValue e2;
    core::TensorValue e3;
    core::TensorValue emb;
    core::TensorValue c0;
    core::TensorValue lsnr;
};

GraphEncodedDeepFilterNet2 graph_run_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & feat_erb,
    const core::TensorValue & feat_spec,
    const std::unordered_map<std::string, NamedWeight> & weights) {
    GraphEncodedDeepFilterNet2 out;
    out.e0 = graph_conv2d_norm_act(ctx, feat_erb, weights, "enc.erb_conv0", 64, 3, 3, 1, true, false);
    out.e1 = graph_conv2d_norm_act(ctx, out.e0, weights, "enc.erb_conv1", 64, 1, 3, 2, true, false);
    out.e2 = graph_conv2d_norm_act(ctx, out.e1, weights, "enc.erb_conv2", 64, 1, 3, 2, true, false);
    out.e3 = graph_conv2d_norm_act(ctx, out.e2, weights, "enc.erb_conv3", 64, 1, 3, 1, true, false);
    out.c0 = graph_conv2d_norm_act(ctx, feat_spec, weights, "enc.df_conv0", 64, 3, 3, 1, true, false);
    auto c1 = graph_conv2d_norm_act(ctx, out.c0, weights, "enc.df_conv1", 64, 1, 3, 2, true, false);
    auto cemb = graph_grouped_linear(ctx, graph_flatten_conv_feature(ctx, c1), weights, "enc.df_fc_emb.0.weight", true);
    auto emb = graph_concat_last(ctx, graph_flatten_conv_feature(ctx, out.e3), cemb);
    out.emb = graph_squeezed_gru(ctx, emb, weights, "enc.emb_gru", 1, true, false, "");
    out.lsnr = graph_linear_sigmoid(ctx, out.emb, weights, "enc.lsnr_fc", 50.0f, -15.0f);
    return out;
}

core::TensorValue graph_view_embedding_as_conv(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & emb,
    int64_t freq_bins) {
    if (emb.shape.last_dim() % freq_bins != 0) {
        throw std::runtime_error("DeepFilterNet2 graph embedding reshape mismatch");
    }
    const int64_t channels = emb.shape.last_dim() / freq_bins;
    auto x = graph_reshape(ctx, emb, core::TensorShape::from_dims({emb.shape.dims[0], emb.shape.dims[1], freq_bins, channels}));
    return graph_transpose(ctx, x, {{0, 3, 1, 2}}, 4);
}

core::TensorValue graph_conv_transpose2d_depthwise_freq(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const NamedWeight & weight_tensor,
    int64_t stride_f,
    int64_t pad_f,
    int64_t output_pad_f,
    const std::string & label) {
    if (weight_tensor.shape.size() != 4 || weight_tensor.shape[0] != input.shape.dims[1] || weight_tensor.shape[1] != 1 || weight_tensor.shape[2] != 1) {
        throw std::runtime_error("DeepFilterNet2 graph conv transpose depthwise shape mismatch: " + label);
    }
    const int64_t channels = input.shape.dims[1];
    const int64_t kernel_f = weight_tensor.shape[3];
    const int64_t out_f = (input.shape.dims[3] - 1) * stride_f - 2 * pad_f + (kernel_f - 1) + output_pad_f + 1;
    core::TensorValue channel_output;
    for (int64_t c = 0; c < channels; ++c) {
        auto input_c = modules::SliceModule({1, c, 1}).build(ctx, input);
        core::TensorValue freq_output;
        for (int64_t of = 0; of < out_f; ++of) {
            core::TensorValue sum;
            for (int64_t in_f = 0; in_f < input.shape.dims[3]; ++in_f) {
                for (int64_t kf = 0; kf < kernel_f; ++kf) {
                    if (in_f * stride_f - pad_f + kf != of) {
                        continue;
                    }
                    auto src = modules::SliceModule({3, in_f, 1}).build(ctx, input_c);
                    auto w_c = modules::SliceModule({0, c, 1}).build(ctx, weight_tensor.tensor);
                    auto w_k = modules::SliceModule({3, kf, 1}).build(ctx, w_c);
                    auto term = graph_mul(ctx, src, graph_weight_scalar_like(ctx, w_k, src));
                    sum = sum.valid() ? graph_add(ctx, sum, term) : term;
                }
            }
            if (!sum.valid()) {
                auto src = modules::SliceModule({3, 0, 1}).build(ctx, input_c);
                sum = graph_scale(ctx, src, 0.0f);
            }
            freq_output = freq_output.valid() ? modules::ConcatModule({3}).build(ctx, freq_output, sum) : sum;
        }
        channel_output = channel_output.valid() ? modules::ConcatModule({1}).build(ctx, channel_output, freq_output) : freq_output;
    }
    return channel_output;
}

core::TensorValue graph_conv_transpose2d_norm_act(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::unordered_map<std::string, NamedWeight> & weights,
    const std::string & prefix,
    int64_t fstride) {
    auto x = graph_conv_transpose2d_depthwise_freq(ctx, input, weight(weights, prefix + ".0.weight"), fstride, 1, 1, prefix + ".0");
    x = graph_conv2d_direct(ctx, x, weight(weights, prefix + ".1.weight"), nullptr, 1, 1, 0, 0, 1, 1, 1, prefix + ".1");
    return graph_batch_norm_act(
        ctx,
        x,
        weight(weights, prefix + ".2.__bn_scale"),
        weight(weights, prefix + ".2.__bn_bias"),
        true,
        false,
        prefix + ".2");
}

core::TensorValue graph_erb_decoder(
    core::ModuleBuildContext & ctx,
    const GraphEncodedDeepFilterNet2 & enc,
    const std::unordered_map<std::string, NamedWeight> & weights) {
    auto emb = graph_squeezed_gru(ctx, enc.emb, weights, "erb_dec.emb_gru", 2, true, true, "erb_dec.emb_gru.linear_out.0.weight");
    auto emb4 = graph_view_embedding_as_conv(ctx, emb, enc.e3.shape.dims[3]);
    auto e3 = graph_conv2d_norm_act(
        ctx,
        graph_add(ctx, graph_conv2d_norm_act(ctx, enc.e3, weights, "erb_dec.conv3p", 64, 1, 1, 1, true, false), emb4),
        weights,
        "erb_dec.convt3",
        64,
        1,
        3,
        1,
        true,
        false);
    auto e2 = graph_conv_transpose2d_norm_act(
        ctx,
        graph_add(ctx, graph_conv2d_norm_act(ctx, enc.e2, weights, "erb_dec.conv2p", 64, 1, 1, 1, true, false), e3),
        weights,
        "erb_dec.convt2",
        2);
    auto e1 = graph_conv_transpose2d_norm_act(
        ctx,
        graph_add(ctx, graph_conv2d_norm_act(ctx, enc.e1, weights, "erb_dec.conv1p", 64, 1, 1, 1, true, false), e2),
        weights,
        "erb_dec.convt1",
        2);
    return graph_conv2d_norm_act(
        ctx,
        graph_add(ctx, graph_conv2d_norm_act(ctx, enc.e0, weights, "erb_dec.conv0p", 64, 1, 1, 1, true, false), e1),
        weights,
        "erb_dec.conv0_out",
        1,
        1,
        3,
        1,
        true,
        true);
}

std::pair<core::TensorValue, core::TensorValue> graph_df_decoder(
    core::ModuleBuildContext & ctx,
    const GraphEncodedDeepFilterNet2 & enc,
    const std::unordered_map<std::string, NamedWeight> & weights) {
    auto c = graph_squeezed_gru(ctx, enc.emb, weights, "df_dec.df_gru", 2, true, true, "");
    auto alpha = graph_linear_sigmoid(ctx, c, weights, "df_dec.df_fc_a", 1.0f, 0.0f);
    auto out = modules::TanhModule().build(ctx, graph_grouped_linear(ctx, c, weights, "df_dec.df_out.0.weight", false));
    auto pathway = graph_conv2d_norm_act(ctx, enc.c0, weights, "df_dec.df_convp", 10, 5, 1, 1, true, false);
    auto pathway_flat = graph_transpose(ctx, pathway, {{0, 2, 3, 1}}, 4);
    pathway_flat = graph_reshape(ctx, pathway_flat, core::TensorShape::from_dims({pathway.shape.dims[0], pathway.shape.dims[2], 96 * 10}));
    auto coefs = graph_add(ctx, out, pathway_flat);
    return {coefs, alpha};
}

struct DeepFilterNet2GraphOutput {
    DeepFilterNet2Tensor erb_mask;
    DeepFilterNet2Tensor df_coefs;
    DeepFilterNet2Tensor enc_lsnr;
    DeepFilterNet2Tensor df_alpha;
};

class DeepFilterNet2ForwardGraph {
public:
    DeepFilterNet2ForwardGraph(
        const std::unordered_map<std::string, NamedWeight> & weights,
        ggml_backend_t backend,
        core::BackendType backend_type,
        int64_t frames)
        : backend_(backend),
          backend_type_(backend_type),
          frames_(frames) {
        ggml_init_params init{kDeepFilterNet2ForwardGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(init));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize DeepFilterNet2 forward graph context");
        }
        core::ModuleBuildContext build_ctx{ctx_.get(), "deepfilternet2.forward", backend_type_};
        auto feat_erb = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, frames_, 32}));
        auto feat_spec = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 2, frames_, 96}));
        feat_erb_ = feat_erb.tensor;
        feat_spec_ = feat_spec.tensor;
        auto enc = graph_run_encoder(build_ctx, feat_erb, feat_spec, weights);
        auto mask = graph_erb_decoder(build_ctx, enc, weights);
        auto [coefs_flat, alpha] = graph_df_decoder(build_ctx, enc, weights);
        auto coefs = graph_reshape(build_ctx, coefs_flat, core::TensorShape::from_dims({1, frames_, 96, 10}));
        mask_ = mask.tensor;
        coefs_ = coefs.tensor;
        lsnr_ = enc.lsnr.tensor;
        alpha_ = alpha.tensor;
        mask_shape_ = mask.shape;
        coefs_shape_ = coefs.shape;
        lsnr_shape_ = enc.lsnr.shape;
        alpha_shape_ = alpha.shape;
        ggml_set_output(mask_);
        ggml_set_output(coefs_);
        ggml_set_output(lsnr_);
        ggml_set_output(alpha_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, mask_);
        ggml_build_forward_expand(graph_, coefs_);
        ggml_build_forward_expand(graph_, lsnr_);
        ggml_build_forward_expand(graph_, alpha_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate DeepFilterNet2 forward graph");
        }
        plan_ = core::create_backend_graph_plan_if_host(backend_, graph_);
    }

    ~DeepFilterNet2ForwardGraph() {
        core::release_backend_graph_resources(backend_, graph_);
        if (plan_ != nullptr) {
            core::free_backend_graph_plan(backend_, plan_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(int64_t frames) const noexcept {
        return frames_ == frames;
    }

    DeepFilterNet2GraphOutput run(const Tensor4 & feat_erb, const Tensor4 & feat_spec) const {
        ggml_backend_tensor_set(feat_erb_, feat_erb.v.data(), 0, feat_erb.v.size() * sizeof(float));
        ggml_backend_tensor_set(feat_spec_, feat_spec.v.data(), 0, feat_spec.v.size() * sizeof(float));
        const auto status = core::compute_backend_graph(backend_, graph_, plan_, "DeepFilterNet2 forward");
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DeepFilterNet2 forward graph compute failed");
        }
        return {
            read_tensor(mask_, mask_shape_),
            read_tensor(coefs_, coefs_shape_),
            read_tensor(lsnr_, lsnr_shape_),
            read_tensor(alpha_, alpha_shape_),
        };
    }

private:
    DeepFilterNet2Tensor read_tensor(ggml_tensor * tensor, const core::TensorShape & shape) const {
        DeepFilterNet2Tensor out;
        for (size_t i = 0; i < shape.rank; ++i) {
            out.shape.push_back(shape.dims[i]);
        }
        out.values.resize(static_cast<size_t>(shape.num_elements()));
        ggml_backend_tensor_get(tensor, out.values.data(), 0, out.values.size() * sizeof(float));
        return out;
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int64_t frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * feat_erb_ = nullptr;
    ggml_tensor * feat_spec_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * coefs_ = nullptr;
    ggml_tensor * lsnr_ = nullptr;
    ggml_tensor * alpha_ = nullptr;
    core::TensorShape mask_shape_;
    core::TensorShape coefs_shape_;
    core::TensorShape lsnr_shape_;
    core::TensorShape alpha_shape_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_graph_plan_t plan_ = nullptr;
};

std::vector<size_t> deepfilternet2_erb_widths() {
    constexpr size_t sample_rate = 48000;
    constexpr size_t fft_size = 960;
    constexpr size_t bands = 32;
    constexpr size_t min_bins = 2;
    const auto freq_to_erb = [](float freq_hz) {
        return 9.265f * std::log1p(freq_hz / (24.7f * 9.265f));
    };
    const auto erb_to_freq = [](float erb) {
        return 24.7f * 9.265f * (std::exp(erb / 9.265f) - 1.0f);
    };
    const float nyquist = static_cast<float>(sample_rate / 2);
    const float freq_width = static_cast<float>(sample_rate) / static_cast<float>(fft_size);
    const float erb_low = freq_to_erb(0.0f);
    const float erb_high = freq_to_erb(nyquist);
    const float step = (erb_high - erb_low) / static_cast<float>(bands);
    std::vector<size_t> widths(bands, 0);
    int prev_freq = 0;
    int freq_over = 0;
    for (size_t i = 1; i <= bands; ++i) {
        const float freq = erb_to_freq(erb_low + static_cast<float>(i) * step);
        const int fb = static_cast<int>(std::round(freq / freq_width));
        int bins = fb - prev_freq - freq_over;
        if (bins < static_cast<int>(min_bins)) {
            freq_over = static_cast<int>(min_bins) - bins;
            bins = static_cast<int>(min_bins);
        } else {
            freq_over = 0;
        }
        widths[i - 1] = static_cast<size_t>(bins);
        prev_freq = fb;
    }
    widths.back() += 1;
    size_t total = 0;
    for (const size_t width : widths) {
        total += width;
    }
    const size_t freq_bins = fft_size / 2 + 1;
    if (total > freq_bins) {
        widths.back() -= total - freq_bins;
    }
    return widths;
}

std::vector<float> deepfilternet2_window() {
    constexpr int64_t fft_size = 960;
    constexpr int64_t window_half = fft_size / 2;
    constexpr double pi = 3.14159265358979323846264338327950288;
    std::vector<float> window(static_cast<size_t>(fft_size));
    for (int64_t i = 0; i < fft_size; ++i) {
        const double s = std::sin(0.5 * pi * (static_cast<double>(i) + 0.5) / static_cast<double>(window_half));
        window[static_cast<size_t>(i)] = static_cast<float>(std::sin(0.5 * pi * s * s));
    }
    return window;
}

struct DeepFilterNet2Features {
    std::vector<std::complex<float>> spec;
    int64_t frames = 0;
    Tensor4 feat_erb;
    Tensor4 feat_spec;
};

DeepFilterNet2Features make_deepfilternet2_features(const std::vector<float> & waveform) {
    constexpr int64_t fft_size = 960;
    constexpr int64_t hop = 480;
    constexpr int64_t freq_bins = 481;
    constexpr int64_t df_bins = 96;
    constexpr float alpha = 0.99f;
    constexpr float wnorm = 1.0f / (static_cast<float>(fft_size * fft_size) / static_cast<float>(2 * hop));

    std::vector<float> padded = waveform;
    padded.insert(padded.end(), static_cast<size_t>(fft_size), 0.0f);
    const int64_t chunks = static_cast<int64_t>(padded.size()) / hop;
    const auto window = deepfilternet2_window();
    std::vector<float> analysis_mem(static_cast<size_t>(fft_size - hop), 0.0f);
    std::vector<float> frame(static_cast<size_t>(fft_size), 0.0f);
    std::vector<std::complex<float>> all_spec(static_cast<size_t>(chunks * freq_bins));
    std::vector<std::complex<float>> frame_spec(static_cast<size_t>(freq_bins));

    for (int64_t chunk = 0; chunk < chunks; ++chunk) {
        for (int64_t i = 0; i < fft_size - hop; ++i) {
            frame[static_cast<size_t>(i)] = analysis_mem[static_cast<size_t>(i)] * window[static_cast<size_t>(i)];
        }
        for (int64_t i = 0; i < hop; ++i) {
            const int64_t sample_index = chunk * hop + i;
            const float sample = sample_index < static_cast<int64_t>(padded.size())
                ? padded[static_cast<size_t>(sample_index)]
                : 0.0f;
            frame[static_cast<size_t>(fft_size - hop + i)] = sample * window[static_cast<size_t>(fft_size - hop + i)];
            analysis_mem[static_cast<size_t>(i)] = sample;
        }
        real_fft_forward(
            TensorShape{static_cast<size_t>(fft_size)},
            TensorStrideBytes{static_cast<std::ptrdiff_t>(sizeof(float))},
            TensorStrideBytes{static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))},
            0,
            frame.data(),
            frame_spec.data(),
            wnorm,
            1);
        for (int64_t f = 0; f < freq_bins; ++f) {
            all_spec[static_cast<size_t>(chunk * freq_bins + f)] = frame_spec[static_cast<size_t>(f)];
        }
    }

    const int64_t frames = chunks;
    DeepFilterNet2Features out;
    out.frames = frames;
    out.spec = std::move(all_spec);
    out.feat_erb = Tensor4(1, 1, frames, 32);
    out.feat_spec = Tensor4(1, 2, frames, df_bins);
    const auto widths = deepfilternet2_erb_widths();
    std::vector<float> mean_state(32);
    for (int64_t i = 0; i < 32; ++i) {
        mean_state[static_cast<size_t>(i)] = -60.0f + static_cast<float>(i) * (-30.0f / 31.0f);
    }
    std::vector<float> unit_state(static_cast<size_t>(df_bins));
    for (int64_t i = 0; i < df_bins; ++i) {
        unit_state[static_cast<size_t>(i)] = 0.001f + static_cast<float>(i) * ((0.0001f - 0.001f) / static_cast<float>(df_bins - 1));
    }
    for (int64_t t = 0; t < frames; ++t) {
        size_t offset = 0;
        for (size_t band = 0; band < widths.size(); ++band) {
            float energy = 0.0f;
            for (size_t j = 0; j < widths[band]; ++j) {
                const auto value = out.spec[static_cast<size_t>(t * freq_bins) + offset + j];
                energy += (value.real() * value.real() + value.imag() * value.imag()) / static_cast<float>(widths[band]);
            }
            float erb = 10.0f * std::log10(energy + 1.0e-10f);
            mean_state[band] = erb * (1.0f - alpha) + mean_state[band] * alpha;
            erb = (erb - mean_state[band]) / 40.0f;
            out.feat_erb.at(0, 0, t, static_cast<int64_t>(band)) = erb;
            offset += widths[band];
        }
        for (int64_t f = 0; f < df_bins; ++f) {
            auto value = out.spec[static_cast<size_t>(t * freq_bins + f)];
            const float mag = std::sqrt(value.real() * value.real() + value.imag() * value.imag());
            unit_state[static_cast<size_t>(f)] = mag * (1.0f - alpha) + unit_state[static_cast<size_t>(f)] * alpha;
            const float scale = 1.0f / std::sqrt(unit_state[static_cast<size_t>(f)]);
            out.feat_spec.at(0, 0, t, f) = value.real() * scale;
            out.feat_spec.at(0, 1, t, f) = value.imag() * scale;
        }
    }
    return out;
}

void apply_deep_filter(
    std::vector<std::complex<float>> & spec,
    const Tensor4 & mask,
    const DeepFilterNet2Tensor & coefs,
    const DeepFilterNet2Tensor & alpha,
    int64_t frames) {
    constexpr int64_t freq_bins = 481;
    constexpr int64_t df_bins = 96;
    constexpr int64_t order = 5;
    if (alpha.shape != std::vector<int64_t>{1, frames, 1}) {
        throw std::runtime_error("DeepFilterNet2 df alpha shape mismatch");
    }
    const auto widths = deepfilternet2_erb_widths();
    for (int64_t t = 0; t < frames; ++t) {
        size_t offset = 0;
        for (size_t band = 0; band < widths.size(); ++band) {
            const float gain = mask.at(0, 0, t, static_cast<int64_t>(band));
            for (size_t j = 0; j < widths[band]; ++j) {
                const size_t index = static_cast<size_t>(t * freq_bins) + offset + j;
                spec[index] *= gain;
            }
            offset += widths[band];
        }
    }
    const auto original = spec;
    for (int64_t t = 0; t < frames; ++t) {
        const float blend = alpha.values[static_cast<size_t>(t)];
        for (int64_t f = 0; f < df_bins; ++f) {
            std::complex<float> value(0.0f, 0.0f);
            for (int64_t k = 0; k < order; ++k) {
                const int64_t source_t = t + k - (order - 1);
                const std::complex<float> source =
                    source_t < 0 || source_t >= frames
                        ? std::complex<float>(0.0f, 0.0f)
                        : original[static_cast<size_t>(source_t * freq_bins + f)];
                const size_t cbase = static_cast<size_t>(((t * order + k) * 2) * df_bins + f);
                const std::complex<float> coef(coefs.values[cbase], coefs.values[cbase + static_cast<size_t>(df_bins)]);
                value += source * coef;
            }
            const size_t index = static_cast<size_t>(t * freq_bins + f);
            spec[index] = value * blend + original[index] * (1.0f - blend);
        }
    }
}

std::vector<float> synthesize_deepfilternet2(
    std::vector<std::complex<float>> spec,
    int64_t frames,
    int64_t original_samples) {
    constexpr int64_t fft_size = 960;
    constexpr int64_t hop = 480;
    constexpr int64_t freq_bins = 481;
    const auto window = deepfilternet2_window();
    std::vector<float> synthesis_mem(static_cast<size_t>(fft_size - hop), 0.0f);
    std::vector<float> frame(static_cast<size_t>(fft_size), 0.0f);
    std::vector<float> output(static_cast<size_t>(frames * hop + (fft_size - hop)), 0.0f);
    for (int64_t t = 0; t < frames; ++t) {
        real_fft_inverse(
            TensorShape{static_cast<size_t>(fft_size)},
            TensorStrideBytes{static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))},
            TensorStrideBytes{static_cast<std::ptrdiff_t>(sizeof(float))},
            0,
            spec.data() + static_cast<size_t>(t * freq_bins),
            frame.data(),
            1.0f,
            1);
        for (int64_t i = 0; i < fft_size; ++i) {
            frame[static_cast<size_t>(i)] *= window[static_cast<size_t>(i)];
        }
        for (int64_t i = 0; i < hop; ++i) {
            output[static_cast<size_t>(t * hop + i)] = frame[static_cast<size_t>(i)] + synthesis_mem[static_cast<size_t>(i)];
        }
        for (int64_t i = 0; i < fft_size - 2 * hop; ++i) {
            synthesis_mem[static_cast<size_t>(i)] =
                synthesis_mem[static_cast<size_t>(i + hop)] + frame[static_cast<size_t>(hop + i)];
        }
        for (int64_t i = fft_size - 2 * hop; i < fft_size - hop; ++i) {
            synthesis_mem[static_cast<size_t>(i)] = frame[static_cast<size_t>(hop + i)];
        }
    }
    for (int64_t i = 0; i < fft_size - hop; ++i) {
        output[static_cast<size_t>(frames * hop + i)] = synthesis_mem[static_cast<size_t>(i)];
    }
    constexpr int64_t delay = fft_size - hop;
    if (static_cast<int64_t>(output.size()) < delay + original_samples) {
        throw std::runtime_error("DeepFilterNet2 synthesis output is shorter than requested crop");
    }
    return std::vector<float>(
        output.begin() + delay,
        output.begin() + delay + original_samples);
}

}  // namespace

struct DeepFilterNet2ModelState {
    std::unique_ptr<ggml_backend, BackendDeleter> backend;
    core::BackendType backend_type = core::BackendType::Cpu;
    std::shared_ptr<core::BackendWeightStore> store;
    std::unordered_map<std::string, NamedWeight> weights;
    mutable std::unique_ptr<DeepFilterNet2ForwardGraph> forward_graph;
};

DeepFilterNet2WaveformOutput run_mono_48k_whole(
    const DeepFilterNet2ModelState & state,
    const std::vector<float> & waveform) {
    auto features = make_deepfilternet2_features(waveform);
    if (!state.forward_graph || !state.forward_graph->matches(features.frames)) {
        state.forward_graph.reset();
        state.forward_graph = std::make_unique<DeepFilterNet2ForwardGraph>(
            state.weights,
            state.backend.get(),
            state.backend_type,
            features.frames);
    }
    const auto output = state.forward_graph->run(features.feat_erb, features.feat_spec);
    Tensor4 mask = tensor4_from_values(output.erb_mask.values, output.erb_mask.shape, "erb_mask");
    apply_deep_filter(
        features.spec,
        mask,
        output.df_coefs,
        output.df_alpha,
        features.frames);
    return DeepFilterNet2WaveformOutput{
        48000,
        synthesize_deepfilternet2(std::move(features.spec), features.frames, static_cast<int64_t>(waveform.size()))};
}

DeepFilterNet2Model::DeepFilterNet2Model() = default;
DeepFilterNet2Model::~DeepFilterNet2Model() = default;
DeepFilterNet2Model::DeepFilterNet2Model(DeepFilterNet2Model &&) noexcept = default;
DeepFilterNet2Model & DeepFilterNet2Model::operator=(DeepFilterNet2Model &&) noexcept = default;

DeepFilterNet2Model::DeepFilterNet2Model(std::shared_ptr<const DeepFilterNet2ModelState> state)
    : state_(std::move(state)) {
    if (state_ == nullptr || state_->backend == nullptr || state_->store == nullptr || state_->weights.empty()) {
        throw std::runtime_error("DeepFilterNet2 requires loaded named weights");
    }
}

DeepFilterNet2Model DeepFilterNet2Model::load_from_directory(const std::filesystem::path & model_dir) {
    return load_from_directory(model_dir, core::BackendConfig{});
}

DeepFilterNet2Model DeepFilterNet2Model::load_from_directory(
    const std::filesystem::path & model_dir,
    const core::BackendConfig & backend_config) {
    const auto weights_path = model_dir / "deepfilternet2.safetensors";
    const auto index = engine::io::load_safetensors_index(weights_path);
    const auto architecture = index.metadata.find("architecture");
    if (architecture == index.metadata.end() || architecture->second != "DeepFilterNet2") {
        throw std::runtime_error("DeepFilterNet2 safetensors is missing DeepFilterNet2 architecture metadata: " + weights_path.string());
    }

    auto state = std::make_shared<DeepFilterNet2ModelState>();
    state->backend.reset(core::init_backend(backend_config));
    state->backend_type = core::backend_type(state->backend.get());
    state->store = std::make_shared<core::BackendWeightStore>(
        state->backend.get(),
        state->backend_type,
        "DeepFilterNet2",
        32ull * 1024ull * 1024ull);

    auto source = assets::open_tensor_source(weights_path);
    for (const auto & meta : source->tensors()) {
        const auto raw = source->require_tensor_data(meta.name);
        if (raw.metadata.dtype != "F32") {
            throw std::runtime_error("DeepFilterNet2 named runtime supports only F32 weights: " + meta.name);
        }
        const int64_t elements = numel(raw.metadata.shape);
        if (static_cast<int64_t>(raw.bytes.size()) != elements * static_cast<int64_t>(sizeof(float))) {
            throw std::runtime_error("DeepFilterNet2 weight byte size mismatch: " + meta.name);
        }
        NamedWeight weight;
        weight.shape = raw.metadata.shape;
        weight.values.resize(static_cast<size_t>(elements));
        std::memcpy(weight.values.data(), raw.bytes.data(), raw.bytes.size());
        weight.tensor = state->store->make_f32(tensor_shape_from_vector(weight.shape), weight.values);
        state->weights.emplace(meta.name, std::move(weight));
    }
    source->release_storage();
    validate_deepfilternet2_weights(state->weights);
    add_deepfilternet2_derived_weights(state->weights, *state->store);
    state->store->upload();
    trim_deepfilternet2_host_weights(state->weights);
    return DeepFilterNet2Model(std::move(state));
}

DeepFilterNet2Output DeepFilterNet2Model::run_features(
    const std::vector<float> & feat_erb,
    const std::vector<int64_t> & feat_erb_shape,
    const std::vector<float> & feat_spec,
    const std::vector<int64_t> & feat_spec_shape) const {
    if (state_ == nullptr) {
        throw std::runtime_error("DeepFilterNet2 model is not initialized");
    }
    auto erb = tensor4_from_values(feat_erb, feat_erb_shape, "feat_erb");
    auto spec = tensor4_from_values(feat_spec, feat_spec_shape, "feat_spec");
    if (erb.c != 1 || erb.f != 32 || spec.c != 2 || spec.f != 96 || erb.t != spec.t) {
        throw std::runtime_error("DeepFilterNet2 feature input shape mismatch");
    }
    if (!state_->forward_graph || !state_->forward_graph->matches(erb.t)) {
        state_->forward_graph.reset();
        state_->forward_graph = std::make_unique<DeepFilterNet2ForwardGraph>(
            state_->weights,
            state_->backend.get(),
            state_->backend_type,
            erb.t);
    }
    const auto output = state_->forward_graph->run(erb, spec);
    return DeepFilterNet2Output{output.erb_mask, output.df_coefs, output.enc_lsnr, output.df_alpha};
}

DeepFilterNet2WaveformOutput DeepFilterNet2Model::run_mono_48k(const std::vector<float> & waveform) const {
    if (state_ == nullptr) {
        throw std::runtime_error("DeepFilterNet2 model is not initialized");
    }
    if (waveform.empty()) {
        throw std::runtime_error("DeepFilterNet2 waveform input is empty");
    }
    constexpr int64_t segment_samples = 48000;
    constexpr int64_t stride_samples = 36000;
    constexpr int64_t segment_threshold = segment_samples * 2;
    const int64_t original_samples = static_cast<int64_t>(waveform.size());
    if (original_samples <= segment_threshold) {
        return run_mono_48k_whole(*state_, waveform);
    }

    std::vector<float> padded = waveform;
    const int64_t remainder = (original_samples - segment_samples) % stride_samples;
    if (remainder != 0) {
        padded.insert(padded.end(), static_cast<size_t>(stride_samples - remainder), 0.0f);
    }
    const int64_t padded_samples = static_cast<int64_t>(padded.size());
    std::vector<float> output(static_cast<size_t>(padded_samples), 0.0f);
    std::vector<float> weights(static_cast<size_t>(padded_samples), 0.0f);
    for (int64_t current = 0; current + segment_samples <= padded_samples; current += stride_samples) {
        std::vector<float> segment(
            padded.begin() + static_cast<std::ptrdiff_t>(current),
            padded.begin() + static_cast<std::ptrdiff_t>(current + segment_samples));
        const auto segment_output = run_mono_48k_whole(*state_, segment);
        if (static_cast<int64_t>(segment_output.samples.size()) != segment_samples) {
            throw std::runtime_error("DeepFilterNet2 segmented output length mismatch");
        }
        for (int64_t i = 0; i < segment_samples; ++i) {
            float weight = 1.0f;
            if (current > 0 && i < segment_samples - stride_samples) {
                weight = static_cast<float>(i + 1) / static_cast<float>(segment_samples - stride_samples + 1);
            }
            if (current + segment_samples < padded_samples && i >= stride_samples) {
                weight = static_cast<float>(segment_samples - i) / static_cast<float>(segment_samples - stride_samples + 1);
            }
            const size_t out_index = static_cast<size_t>(current + i);
            output[out_index] += segment_output.samples[static_cast<size_t>(i)] * weight;
            weights[out_index] += weight;
        }
    }
    output.resize(static_cast<size_t>(original_samples));
    weights.resize(static_cast<size_t>(original_samples));
    for (size_t i = 0; i < output.size(); ++i) {
        if (weights[i] <= 0.0f) {
            throw std::runtime_error("DeepFilterNet2 segmented synthesis produced an uncovered sample");
        }
        output[i] /= weights[i];
    }
    return DeepFilterNet2WaveformOutput{48000, std::move(output)};
}

}  // namespace engine::audio
