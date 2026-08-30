#include "engine/framework/audio/zipenhancer.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
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

namespace engine::audio {
namespace {

constexpr size_t kZipEnhancerForwardGraphContextBytes = 96ull * 1024ull * 1024ull;
constexpr size_t kZipEnhancerForwardGraphNodes = 65536;
constexpr int64_t kSampleRate = 16000;
constexpr int64_t kNfft = 400;
constexpr int64_t kHop = 100;
constexpr int64_t kWin = 400;
constexpr int64_t kFreqBins = 201;
constexpr int64_t kWindowSamples = kSampleRate * 2;             // 32000, 2 s
constexpr int64_t kStrideSamples = kWindowSamples * 3 / 4;      // 24000, 1.5 s
constexpr int64_t kSegmentThreshold = kWindowSamples * 3;       // 96000, 6 s
constexpr int64_t kDense = 64;
constexpr int64_t kHeads = 4;
constexpr int64_t kQueryHeadDim = 12;
constexpr int64_t kPosHeadDim = 4;
constexpr int64_t kValueHeadDim = 8;
constexpr int64_t kPosDim = 24;
constexpr float kCompress = 0.3f;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct BackendDeleter {
    void operator()(ggml_backend * backend) const noexcept {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

int64_t product(const std::vector<int64_t> & shape) {
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>{});
}

core::TensorShape tensor_shape_from_vector(const std::vector<int64_t> & shape) {
    if (shape.empty()) {
        return core::TensorShape::from_dims({1});
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
    if (shape.size() == 4) {
        return core::TensorShape::from_dims({shape[0], shape[1], shape[2], shape[3]});
    }
    throw std::runtime_error("ZipEnhancer tensor rank exceeds GGML wrapper rank");
}

struct Param {
    std::vector<int64_t> shape;
    std::vector<float> values;
    core::TensorValue tensor;
};

const Param & require_param(const std::unordered_map<std::string, Param> & params, const std::string & name) {
    const auto it = params.find(name);
    if (it == params.end()) {
        throw std::runtime_error("missing ZipEnhancer tensor: " + name);
    }
    return it->second;
}

void require_shape(const Param & p, const std::vector<int64_t> & shape, const std::string & name) {
    if (p.shape != shape) {
        throw std::runtime_error("ZipEnhancer tensor shape mismatch: " + name);
    }
}

bool has_suffix(const std::string & value, const char * suffix) {
    const std::string_view tail(suffix);
    return value.size() >= tail.size() &&
           value.compare(value.size() - tail.size(), tail.size(), tail) == 0;
}

bool zipenhancer_param_needs_host_values(const std::string & name) {
    return has_suffix(name, ".self_attn_weights.linear_pos.weight") ||
           has_suffix(name, ".norm.log_scale") ||
           has_suffix(name, ".downsample_t.bias") ||
           has_suffix(name, ".downsample_f.bias");
}

void trim_zipenhancer_host_params(std::unordered_map<std::string, Param> & params) {
    for (auto & [name, param] : params) {
        if (zipenhancer_param_needs_host_values(name)) {
            continue;
        }
        param.values.clear();
        param.values.shrink_to_fit();
    }
}

struct Tensor4 {
    int64_t b = 0;
    int64_t c = 0;
    int64_t t = 0;
    int64_t f = 0;
    std::vector<float> v;

    Tensor4() = default;
    Tensor4(int64_t batch, int64_t channels, int64_t frames, int64_t freq)
        : b(batch), c(channels), t(frames), f(freq), v(static_cast<size_t>(batch * channels * frames * freq), 0.0f) {}

    float & at(int64_t bi, int64_t ci, int64_t ti, int64_t fi) {
        return v[static_cast<size_t>((((bi * c) + ci) * t + ti) * f + fi)];
    }

    float at(int64_t bi, int64_t ci, int64_t ti, int64_t fi) const {
        return v[static_cast<size_t>((((bi * c) + ci) * t + ti) * f + fi)];
    }
};

modules::Conv2dWeights graph_conv2d_weights(const Param & weight, const Param & bias) {
    modules::Conv2dWeights out;
    out.weight = weight.tensor;
    out.bias = bias.tensor;
    return out;
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
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    lp[core::logical_axis_to_ggml_axis(contiguous.shape.rank, 2)] = static_cast<int32_t>(top);
    rp[core::logical_axis_to_ggml_axis(contiguous.shape.rank, 2)] = static_cast<int32_t>(bottom);
    lp[core::logical_axis_to_ggml_axis(contiguous.shape.rank, 3)] = static_cast<int32_t>(left);
    rp[core::logical_axis_to_ggml_axis(contiguous.shape.rank, 3)] = static_cast<int32_t>(right);
    auto shape = input.shape;
    shape.dims[2] += top + bottom;
    shape.dims[3] += left + right;
    return core::wrap_tensor(
        ggml_pad_ext(ctx.ggml, contiguous.tensor, lp[0], rp[0], lp[1], rp[1], lp[2], rp[2], lp[3], rp[3]),
        shape,
        input.type);
}

core::TensorValue graph_conv2d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::unordered_map<std::string, Param> & params,
    const std::string & weight_name,
    const std::string & bias_name,
    int64_t pad_t,
    int64_t pad_f,
    int64_t stride_t,
    int64_t stride_f,
    int64_t dil_t,
    int64_t dil_f) {
    const auto & weight = require_param(params, weight_name);
    const auto & bias = require_param(params, bias_name);
    if (weight.shape.size() != 4) {
        throw std::runtime_error("ZipEnhancer graph conv2d weight shape mismatch: " + weight_name);
    }
    return modules::Conv2dModule({
        input.shape.dims[1],
        weight.shape[0],
        weight.shape[2],
        weight.shape[3],
        static_cast<int>(stride_t),
        static_cast<int>(stride_f),
        static_cast<int>(pad_t),
        static_cast<int>(pad_f),
        static_cast<int>(dil_t),
        static_cast<int>(dil_f),
        true,
    }).build(ctx, input, graph_conv2d_weights(weight, bias));
}

core::TensorValue graph_instance_norm_prelu(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::unordered_map<std::string, Param> & params,
    const std::string & norm_weight_name,
    const std::string & norm_bias_name,
    const std::string & prelu_weight_name) {
    const auto & norm_weight = require_param(params, norm_weight_name);
    const auto & norm_bias = require_param(params, norm_bias_name);
    const auto & prelu_weight = require_param(params, prelu_weight_name);
    core::validate_shape(norm_weight.tensor, core::TensorShape::from_dims({input.shape.dims[1]}), "ZipEnhancer instance norm weight");
    core::validate_shape(norm_bias.tensor, core::TensorShape::from_dims({input.shape.dims[1]}), "ZipEnhancer instance norm bias");
    core::validate_shape(prelu_weight.tensor, core::TensorShape::from_dims({input.shape.dims[1]}), "ZipEnhancer PReLU weight");

    auto mean_f = modules::ReduceMeanModule({3}).build(ctx, input);
    auto mean = modules::ReduceMeanModule({2}).build(ctx, mean_f);
    auto mean_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, mean.tensor, input.tensor), input.shape, GGML_TYPE_F32);
    auto centered = core::wrap_tensor(ggml_sub(ctx.ggml, input.tensor, mean_rep.tensor), input.shape, GGML_TYPE_F32);
    auto squared = core::wrap_tensor(ggml_sqr(ctx.ggml, centered.tensor), centered.shape, GGML_TYPE_F32);
    auto var_f = modules::ReduceMeanModule({3}).build(ctx, squared);
    auto var = modules::ReduceMeanModule({2}).build(ctx, var_f);
    auto stddev = core::wrap_tensor(
        ggml_sqrt(ctx.ggml, ggml_scale_bias(ctx.ggml, var.tensor, 1.0f, 1.0e-5f)),
        var.shape,
        GGML_TYPE_F32);
    auto stddev_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, stddev.tensor, input.tensor), input.shape, GGML_TYPE_F32);
    auto normed = core::wrap_tensor(ggml_div(ctx.ggml, centered.tensor, stddev_rep.tensor), input.shape, GGML_TYPE_F32);
    auto weight = core::reshape_tensor(ctx, norm_weight.tensor, core::TensorShape::from_dims({1, input.shape.dims[1], 1, 1}));
    auto bias = core::reshape_tensor(ctx, norm_bias.tensor, core::TensorShape::from_dims({1, input.shape.dims[1], 1, 1}));
    auto weight_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, weight.tensor, normed.tensor), normed.shape, GGML_TYPE_F32);
    auto bias_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, bias.tensor, normed.tensor), normed.shape, GGML_TYPE_F32);
    auto affine = core::wrap_tensor(ggml_add(ctx.ggml, ggml_mul(ctx.ggml, normed.tensor, weight_rep.tensor), bias_rep.tensor), normed.shape, GGML_TYPE_F32);

    auto positive = modules::ReluModule().build(ctx, affine);
    auto negative = modules::ReluModule().build(ctx, core::wrap_tensor(ggml_scale(ctx.ggml, affine.tensor, -1.0f), affine.shape, GGML_TYPE_F32));
    auto slope = core::reshape_tensor(ctx, prelu_weight.tensor, core::TensorShape::from_dims({1, input.shape.dims[1], 1, 1}));
    auto slope_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, slope.tensor, affine.tensor), affine.shape, GGML_TYPE_F32);
    auto scaled_negative = core::wrap_tensor(ggml_mul(ctx.ggml, slope_rep.tensor, negative.tensor), affine.shape, GGML_TYPE_F32);
    return core::wrap_tensor(ggml_sub(ctx.ggml, positive.tensor, scaled_negative.tensor), affine.shape, GGML_TYPE_F32);
}

core::TensorValue graph_dense_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix) {
    auto skip = input;
    auto x = input;
    for (int i = 0; i < 4; ++i) {
        const int64_t dilation = int64_t{1} << i;
        const int64_t pad_top = dilation;
        const std::string base = prefix + ".dense_block." + std::to_string(i);
        x = graph_conv2d(
            ctx,
            graph_pad2d(ctx, skip, 1, 1, pad_top, 0),
            params,
            base + ".1.weight",
            base + ".1.bias",
            0,
            0,
            1,
            1,
            dilation,
            1);
        x = graph_instance_norm_prelu(
            ctx,
            x,
            params,
            base + ".2.weight",
            base + ".2.bias",
            base + ".3.weight");
        skip = modules::ConcatModule({1}).build(ctx, x, skip);
    }
    return x;
}

core::TensorValue graph_dense_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::unordered_map<std::string, Param> & params) {
    auto x = graph_conv2d(
        ctx,
        input,
        params,
        "dense_encoder.dense_conv_1.0.weight",
        "dense_encoder.dense_conv_1.0.bias",
        0,
        0,
        1,
        1,
        1,
        1);
    x = graph_instance_norm_prelu(
        ctx,
        x,
        params,
        "dense_encoder.dense_conv_1.1.weight",
        "dense_encoder.dense_conv_1.1.bias",
        "dense_encoder.dense_conv_1.2.weight");
    x = graph_dense_block(ctx, x, params, "dense_encoder.dense_block");
    x = graph_conv2d(
        ctx,
        x,
        params,
        "dense_encoder.dense_conv_2.0.weight",
        "dense_encoder.dense_conv_2.0.bias",
        0,
        1,
        1,
        2,
        1,
        1);
    return graph_instance_norm_prelu(
        ctx,
        x,
        params,
        "dense_encoder.dense_conv_2.1.weight",
        "dense_encoder.dense_conv_2.1.bias",
        "dense_encoder.dense_conv_2.2.weight");
}

std::vector<float> compact_rel_pos(int64_t seq_len) {
    constexpr float kPi = 3.14159265358979323846f;
    const int64_t length = 2 * seq_len - 1;
    std::vector<float> pe(static_cast<size_t>(length * kPosDim), 0.0f);
    const float compression_length = std::sqrt(static_cast<float>(kPosDim));
    const float length_scale = static_cast<float>(kPosDim) / (2.0f * kPi);
    for (int64_t i = 0; i < length; ++i) {
        const float x = static_cast<float>(i - (seq_len - 1));
        const float x_abs = std::fabs(x);
        const float tmp = std::log(x_abs + compression_length) - std::log(compression_length);
        const float x_compressed = compression_length * (x < 0.0f ? -1.0f : (x > 0.0f ? 1.0f : 0.0f)) * tmp;
        const float x_atan = std::atan(x_compressed / length_scale);
        for (int64_t j = 0; j < kPosDim / 2; ++j) {
            const float freq = static_cast<float>(j + 1);
            pe[static_cast<size_t>(i * kPosDim + 2 * j)] = std::cos(x_atan * freq);
            pe[static_cast<size_t>(i * kPosDim + 2 * j + 1)] = std::sin(x_atan * freq);
        }
        pe[static_cast<size_t>(i * kPosDim + kPosDim - 1)] = 1.0f;
    }
    return pe;
}

struct GraphConstant {
    ggml_tensor * tensor = nullptr;
    int64_t seq = 0;
    int64_t head = 0;
    int64_t dim = 0;
    const Param * pos_weight = nullptr;
};

core::TensorValue make_relative_projection_constant(
    core::ModuleBuildContext & ctx,
    std::vector<GraphConstant> & constants,
    int64_t seq,
    int64_t head,
    int64_t dim,
    const Param & pos_weight) {
    const auto shape = core::TensorShape::from_dims({1, seq, seq});
    auto value = core::make_tensor(ctx, GGML_TYPE_F32, shape);
    constants.push_back(GraphConstant{value.tensor, seq, head, dim, &pos_weight});
    return value;
}

core::TensorValue graph_param(const Param & param) {
    return param.tensor;
}

core::TensorValue graph_linear(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix,
    int64_t out_features) {
    const std::string weight_name = prefix + ".weight";
    const std::string bias_name = prefix + ".bias";
    const auto & weight = require_param(params, weight_name);
    const auto & bias = require_param(params, bias_name);
    return modules::LinearModule({input.shape.last_dim(), out_features, true}).build(
        ctx,
        input,
        modules::LinearWeights{graph_param(weight), graph_param(bias)});
}

core::TensorValue graph_add(core::ModuleBuildContext & ctx, const core::TensorValue & lhs, const core::TensorValue & rhs) {
    return modules::AddModule().build(ctx, lhs, rhs);
}

core::TensorValue graph_sub(core::ModuleBuildContext & ctx, const core::TensorValue & lhs, const core::TensorValue & rhs) {
    core::validate_shape(rhs, lhs.shape, "ZipEnhancer sub rhs");
    const auto lhs_contiguous = core::ensure_backend_addressable_layout(ctx, lhs);
    const auto rhs_contiguous = core::ensure_backend_addressable_layout(ctx, rhs);
    return core::wrap_tensor(ggml_sub(ctx.ggml, lhs_contiguous.tensor, rhs_contiguous.tensor), lhs.shape, GGML_TYPE_F32);
}

core::TensorValue graph_mul(core::ModuleBuildContext & ctx, const core::TensorValue & lhs, const core::TensorValue & rhs) {
    return modules::MulModule().build(ctx, lhs, rhs);
}

core::TensorValue graph_scale(core::ModuleBuildContext & ctx, const core::TensorValue & input, float scale) {
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::wrap_tensor(ggml_scale(ctx.ggml, contiguous.tensor, scale), input.shape, GGML_TYPE_F32);
}

core::TensorValue graph_scale_bias(core::ModuleBuildContext & ctx, const core::TensorValue & input, float scale, float bias) {
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::wrap_tensor(ggml_scale_bias(ctx.ggml, contiguous.tensor, scale, bias), input.shape, GGML_TYPE_F32);
}

core::TensorValue graph_reshape(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorShape & shape) {
    return core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, input), shape);
}

core::TensorValue graph_transpose(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::array<int, core::kMaxTensorRank> & axes,
    size_t rank) {
    if (rank != input.shape.rank) {
        throw std::runtime_error("ZipEnhancer transpose rank mismatch");
    }
    core::TensorShape output_shape = {};
    output_shape.rank = rank;
    std::array<bool, core::kMaxTensorRank> seen = {false, false, false, false};
    std::array<int, core::kMaxTensorRank> ggml_axes = {0, 1, 2, 3};
    for (size_t out_axis = 0; out_axis < rank; ++out_axis) {
        const int in_axis = axes[out_axis];
        if (in_axis < 0 || in_axis >= static_cast<int>(rank) || seen[static_cast<size_t>(in_axis)]) {
            throw std::runtime_error("ZipEnhancer transpose axes must be a permutation");
        }
        seen[static_cast<size_t>(in_axis)] = true;
        output_shape.dims[out_axis] = input.shape.dims[static_cast<size_t>(in_axis)];
        const int out_ggml_axis = core::logical_axis_to_ggml_axis(rank, static_cast<int>(out_axis));
        const int in_ggml_axis = core::logical_axis_to_ggml_axis(rank, in_axis);
        ggml_axes[static_cast<size_t>(in_ggml_axis)] = out_ggml_axis;
    }
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::wrap_tensor(
        ggml_permute(ctx.ggml, contiguous.tensor, ggml_axes[0], ggml_axes[1], ggml_axes[2], ggml_axes[3]),
        output_shape,
        input.type);
}

core::TensorValue graph_concat_all(
    core::ModuleBuildContext & ctx,
    const std::vector<core::TensorValue> & tensors,
    int axis) {
    if (tensors.empty()) {
        throw std::runtime_error("ZipEnhancer graph concat requires at least one tensor");
    }
    auto out = tensors.front();
    for (size_t i = 1; i < tensors.size(); ++i) {
        out = modules::ConcatModule({axis}).build(ctx, out, tensors[i]);
    }
    return out;
}

core::TensorValue graph_softplus_shifted(core::ModuleBuildContext & ctx, const core::TensorValue & input, float offset) {
    auto shifted = graph_scale_bias(ctx, input, 1.0f, -offset);
    auto exp_value = core::wrap_tensor(ggml_exp(ctx.ggml, shifted.tensor), input.shape, GGML_TYPE_F32);
    auto plus_one = graph_scale_bias(ctx, exp_value, 1.0f, 1.0f);
    return core::wrap_tensor(ggml_log(ctx.ggml, plus_one.tensor), input.shape, GGML_TYPE_F32);
}

core::TensorValue graph_swoosh_l(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    auto y = graph_softplus_shifted(ctx, input, 4.0f);
    y = graph_add(ctx, y, graph_scale(ctx, input, -0.08f));
    return graph_scale_bias(ctx, y, 1.0f, -0.035f);
}

core::TensorValue graph_swoosh_r(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    auto y = graph_softplus_shifted(ctx, input, 1.0f);
    y = graph_add(ctx, y, graph_scale(ctx, input, -0.08f));
    return graph_scale_bias(ctx, y, 1.0f, -0.313261687f);
}

core::TensorValue graph_broadcast_channel_scale(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & scale,
    const core::TensorValue & target) {
    core::validate_shape(scale, core::TensorShape::from_dims({target.shape.last_dim()}), "ZipEnhancer channel scale");
    std::vector<int64_t> view_dims(target.shape.rank, 1);
    view_dims.back() = target.shape.last_dim();
    auto view = core::reshape_tensor(ctx, scale, tensor_shape_from_vector(view_dims));
    return modules::RepeatModule({target.shape}).build(ctx, view);
}

core::TensorValue graph_bypass(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & orig,
    const core::TensorValue & x,
    const Param & scale) {
    auto scale_rep = graph_broadcast_channel_scale(ctx, scale.tensor, orig);
    auto delta = graph_sub(ctx, x, orig);
    return graph_add(ctx, orig, graph_mul(ctx, delta, scale_rep));
}

std::vector<float> precompute_relative_projection(
    int64_t seq,
    int64_t head,
    int64_t dim,
    const Param & pos_weight,
    const std::vector<float> & pos_emb) {
    require_shape(pos_weight, {kHeads * kPosHeadDim, kPosDim}, "ZipEnhancer relative position weight");
    std::vector<float> table(static_cast<size_t>(seq * seq), 0.0f);
    for (int64_t t1 = 0; t1 < seq; ++t1) {
        for (int64_t t2 = 0; t2 < seq; ++t2) {
            const int64_t rel = (seq - 1 - t1) + t2;
            float sum = 0.0f;
            const int64_t out = head * kPosHeadDim + dim;
            for (int64_t i = 0; i < kPosDim; ++i) {
                sum += pos_emb[static_cast<size_t>(rel * kPosDim + i)] *
                       pos_weight.values[static_cast<size_t>(out * kPosDim + i)];
            }
            table[static_cast<size_t>(t1 * seq + t2)] = sum;
        }
    }
    return table;
}

std::vector<core::TensorValue> graph_attention_weights(
    core::ModuleBuildContext & ctx,
    std::vector<GraphConstant> & constants,
    const core::TensorValue & x,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix) {
    const int64_t seq = x.shape.dims[0];
    const int64_t batch = x.shape.dims[1];
    auto qkp = graph_linear(ctx, x, params, prefix + ".self_attn_weights.in_proj", 112);
    std::vector<core::TensorValue> heads;
    heads.reserve(kHeads);
    const std::string pos_weight_name = prefix + ".self_attn_weights.linear_pos.weight";
    const auto & pos_weight = require_param(params, pos_weight_name);
    for (int64_t h = 0; h < kHeads; ++h) {
        auto q = modules::SliceModule({2, h * kQueryHeadDim, kQueryHeadDim}).build(ctx, qkp);
        auto k = modules::SliceModule({2, kHeads * kQueryHeadDim + h * kQueryHeadDim, kQueryHeadDim}).build(ctx, qkp);
        auto p = modules::SliceModule({2, 2 * kHeads * kQueryHeadDim + h * kPosHeadDim, kPosHeadDim}).build(ctx, qkp);
        q = graph_transpose(ctx, q, {{1, 0, 2, 3}}, 3);
        k = graph_transpose(ctx, k, {{1, 2, 0, 3}}, 3);
        auto scores = modules::MatMulModule().build(ctx, q, k);

        p = graph_transpose(ctx, p, {{1, 0, 2, 3}}, 3);
        core::TensorValue pos_scores;
        for (int64_t d = 0; d < kPosHeadDim; ++d) {
            auto p_d = modules::SliceModule({2, d, 1}).build(ctx, p);
            auto p_rep = modules::RepeatModule({core::TensorShape::from_dims({batch, seq, seq})}).build(ctx, p_d);
            auto rel = make_relative_projection_constant(
                ctx,
                constants,
                seq,
                h,
                d,
                pos_weight);
            rel = core::reshape_tensor(ctx, rel, core::TensorShape::from_dims({1, seq, seq}));
            const auto p_contiguous = core::ensure_backend_addressable_layout(ctx, p_rep);
            const auto rel_contiguous = core::ensure_backend_addressable_layout(ctx, rel);
            auto term = core::wrap_tensor(
                ggml_mul(ctx.ggml, p_contiguous.tensor, rel_contiguous.tensor),
                p_rep.shape,
                GGML_TYPE_F32);
            pos_scores = pos_scores.valid() ? graph_add(ctx, pos_scores, term) : term;
        }
        heads.push_back(modules::SoftmaxModule().build(ctx, graph_add(ctx, scores, pos_scores)));
    }
    return heads;
}

core::TensorValue graph_self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const std::vector<core::TensorValue> & attn,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix,
    const std::string & which) {
    const std::string base = prefix + "." + which;
    auto v = graph_linear(ctx, x, params, base + ".in_proj", kHeads * kValueHeadDim);
    std::vector<core::TensorValue> pieces;
    pieces.reserve(kHeads);
    for (int64_t h = 0; h < kHeads; ++h) {
        auto vh = modules::SliceModule({2, h * kValueHeadDim, kValueHeadDim}).build(ctx, v);
        vh = graph_transpose(ctx, vh, {{1, 0, 2, 3}}, 3);
        auto out = modules::MatMulModule().build(ctx, attn[static_cast<size_t>(h)], vh);
        pieces.push_back(graph_transpose(ctx, out, {{1, 0, 2, 3}}, 3));
    }
    auto ctx_value = graph_concat_all(ctx, pieces, 2);
    return graph_linear(ctx, ctx_value, params, base + ".out_proj", kDense);
}

core::TensorValue graph_feed_forward(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix,
    const std::string & which,
    int64_t hidden) {
    const std::string base = prefix + "." + which;
    auto y = graph_linear(ctx, x, params, base + ".in_proj", hidden);
    y = graph_swoosh_l(ctx, y);
    return graph_linear(ctx, y, params, base + ".out_proj", kDense);
}

core::TensorValue graph_nonlin_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const core::TensorValue & attn,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix) {
    const std::string base = prefix + ".nonlin_attention";
    auto p = graph_linear(ctx, x, params, base + ".in_proj", 144);
    auto lhs = modules::SliceModule({2, 0, 48}).build(ctx, p);
    auto rhs = modules::SliceModule({2, 48, 48}).build(ctx, p);
    auto gate = modules::SliceModule({2, 96, 48}).build(ctx, p);
    auto gated = graph_mul(ctx, rhs, modules::TanhModule().build(ctx, lhs));
    auto gated_bt = graph_transpose(ctx, gated, {{1, 0, 2, 3}}, 3);
    auto mixed = modules::MatMulModule().build(ctx, attn, gated_bt);
    mixed = graph_transpose(ctx, mixed, {{1, 0, 2, 3}}, 3);
    return graph_linear(ctx, graph_mul(ctx, mixed, gate), params, base + ".out_proj", kDense);
}

core::TensorValue graph_depthwise_conv1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Param & weight,
    const Param & bias,
    const std::string & name) {
    require_shape(weight, {kDense, 1, 15}, name + ".weight");
    require_shape(bias, {kDense}, name + ".bias");
    return modules::DepthwiseConv1dModule({kDense, 15, 1, 7, 1, true}).build(
        ctx,
        input,
        modules::DepthwiseConv1dWeights{weight.tensor, bias.tensor});
}

core::TensorValue graph_conv_module(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix,
    const std::string & which) {
    const std::string base = prefix + "." + which;
    auto p = graph_linear(ctx, x, params, base + ".in_proj", 128);
    auto lhs = modules::SliceModule({2, 0, kDense}).build(ctx, p);
    auto rhs = modules::SliceModule({2, kDense, kDense}).build(ctx, p);
    auto gated = graph_mul(ctx, lhs, modules::SigmoidModule().build(ctx, rhs));
    auto channel_first = graph_transpose(ctx, gated, {{1, 2, 0, 3}}, 3);
    auto conv = graph_depthwise_conv1d(
        ctx,
        channel_first,
        require_param(params, base + ".depthwise_conv.weight"),
        require_param(params, base + ".depthwise_conv.bias"),
        base + ".depthwise_conv");
    conv = graph_swoosh_r(ctx, conv);
    auto seq_first = graph_transpose(ctx, conv, {{2, 0, 1, 3}}, 3);
    return graph_linear(ctx, seq_first, params, base + ".out_proj", kDense);
}

core::TensorValue graph_bias_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix) {
    const std::string log_scale_name = prefix + ".norm.log_scale";
    const std::string bias_name = prefix + ".norm.bias";
    const auto & log_scale = require_param(params, log_scale_name);
    const auto & bias = require_param(params, bias_name);
    if (log_scale.values.empty()) {
        throw std::runtime_error("ZipEnhancer BiasNorm log_scale is empty");
    }
    auto bias_view = core::reshape_tensor(ctx, bias.tensor, core::TensorShape::from_dims({1, 1, kDense}));
    auto bias_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, bias_view.tensor, x.tensor), x.shape, GGML_TYPE_F32);
    auto centered = graph_sub(ctx, x, bias_rep);
    auto mean_sq = modules::ReduceMeanModule({2}).build(ctx, core::wrap_tensor(ggml_sqr(ctx.ggml, centered.tensor), x.shape, GGML_TYPE_F32));
    auto denom = core::wrap_tensor(ggml_sqrt(ctx.ggml, mean_sq.tensor), mean_sq.shape, GGML_TYPE_F32);
    auto denom_rep = core::wrap_tensor(ggml_repeat(ctx.ggml, denom.tensor, x.tensor), x.shape, GGML_TYPE_F32);
    auto normalized = core::wrap_tensor(ggml_div(ctx.ggml, x.tensor, denom_rep.tensor), x.shape, GGML_TYPE_F32);
    return graph_scale(ctx, normalized, std::exp(log_scale.values.front()));
}

core::TensorValue graph_zipformer_layer(
    core::ModuleBuildContext & ctx,
    std::vector<GraphConstant> & constants,
    const core::TensorValue & input,
    const std::unordered_map<std::string, Param> & params,
    const std::string & prefix) {
    const auto orig = input;
    auto attn = graph_attention_weights(ctx, constants, input, params, prefix);
    auto x = graph_add(ctx, input, graph_feed_forward(ctx, input, params, prefix, "feed_forward1", 192));
    x = graph_add(ctx, x, graph_nonlin_attention(ctx, x, attn.front(), params, prefix));
    x = graph_add(ctx, x, graph_self_attention(ctx, x, attn, params, prefix, "self_attn1"));
    x = graph_add(ctx, x, graph_conv_module(ctx, x, params, prefix, "conv_module1"));
    x = graph_add(ctx, x, graph_feed_forward(ctx, x, params, prefix, "feed_forward2", 256));
    x = graph_bypass(ctx, orig, x, require_param(params, prefix + ".bypass_mid.bypass_scale"));
    x = graph_add(ctx, x, graph_self_attention(ctx, x, attn, params, prefix, "self_attn2"));
    x = graph_add(ctx, x, graph_conv_module(ctx, x, params, prefix, "conv_module2"));
    x = graph_add(ctx, x, graph_feed_forward(ctx, x, params, prefix, "feed_forward3", 320));
    x = graph_bias_norm(ctx, x, params, prefix);
    return graph_bypass(ctx, orig, x, require_param(params, prefix + ".bypass.bypass_scale"));
}

std::vector<float> softmax_values(const Param & bias) {
    const float max_value = *std::max_element(bias.values.begin(), bias.values.end());
    std::vector<float> values = bias.values;
    float denom = 0.0f;
    for (float & value : values) {
        value = std::exp(value - max_value);
        denom += value;
    }
    for (float & value : values) {
        value /= denom;
    }
    return values;
}

core::TensorValue graph_downsample(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Param & bias) {
    const int64_t ds = static_cast<int64_t>(bias.values.size());
    if (ds <= 0) {
        throw std::runtime_error("ZipEnhancer downsample bias is empty");
    }
    const auto weights = softmax_values(bias);
    const int64_t out_s = (input.shape.dims[0] + ds - 1) / ds;
    std::vector<core::TensorValue> pieces;
    pieces.reserve(static_cast<size_t>(out_s));
    for (int64_t s = 0; s < out_s; ++s) {
        core::TensorValue acc;
        for (int64_t k = 0; k < ds; ++k) {
            const int64_t src_s = std::min(input.shape.dims[0] - 1, s * ds + k);
            auto slice = modules::SliceModule({0, src_s, 1}).build(ctx, input);
            slice = graph_scale(ctx, slice, weights[static_cast<size_t>(k)]);
            acc = acc.valid() ? graph_add(ctx, acc, slice) : slice;
        }
        pieces.push_back(acc);
    }
    return graph_concat_all(ctx, pieces, 0);
}

core::TensorValue graph_upsample_slice(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t factor,
    int64_t target) {
    std::vector<core::TensorValue> pieces;
    pieces.reserve(static_cast<size_t>(input.shape.dims[0] * factor));
    for (int64_t s = 0; s < input.shape.dims[0]; ++s) {
        auto slice = modules::SliceModule({0, s, 1}).build(ctx, input);
        for (int64_t k = 0; k < factor; ++k) {
            pieces.push_back(slice);
        }
    }
    auto out = graph_concat_all(ctx, pieces, 0);
    if (out.shape.dims[0] != target) {
        out = modules::SliceModule({0, 0, target}).build(ctx, out);
    }
    return out;
}

core::TensorValue graph_run_dual_encoder(
    core::ModuleBuildContext & ctx,
    std::vector<GraphConstant> & constants,
    core::TensorValue x,
    const std::unordered_map<std::string, Param> & params,
    const std::string & root) {
    const int64_t frames = x.shape.dims[2];
    const int64_t freq = x.shape.dims[3];
    auto f_in = graph_transpose(ctx, x, {{3, 2, 0, 1}}, 4);
    f_in = graph_reshape(ctx, f_in, core::TensorShape::from_dims({freq, frames, kDense}));
    auto f_out = graph_zipformer_layer(ctx, constants, f_in, params, root + ".f_layers.0");
    f_out = graph_bypass(ctx, f_in, f_out, require_param(params, root + ".bypass_layers.0.bypass_scale"));
    x = graph_reshape(ctx, f_out, core::TensorShape::from_dims({freq, frames, 1, kDense}));
    x = graph_transpose(ctx, x, {{2, 3, 1, 0}}, 4);

    auto t_in = graph_transpose(ctx, x, {{2, 3, 0, 1}}, 4);
    t_in = graph_reshape(ctx, t_in, core::TensorShape::from_dims({frames, freq, kDense}));
    auto t_out = graph_zipformer_layer(ctx, constants, t_in, params, root + ".t_layers.0");
    t_out = graph_bypass(ctx, t_in, t_out, require_param(params, root + ".bypass_layers.1.bypass_scale"));
    x = graph_reshape(ctx, t_out, core::TensorShape::from_dims({frames, freq, 1, kDense}));
    return graph_transpose(ctx, x, {{2, 3, 0, 1}}, 4);
}

core::TensorValue graph_run_downsampled_encoder(
    core::ModuleBuildContext & ctx,
    std::vector<GraphConstant> & constants,
    const core::TensorValue & input,
    const std::unordered_map<std::string, Param> & params,
    const std::string & root,
    int64_t ds) {
    const int64_t frames = input.shape.dims[2];
    const int64_t freq = input.shape.dims[3];
    auto t_seq = graph_transpose(ctx, input, {{2, 3, 0, 1}}, 4);
    t_seq = graph_reshape(ctx, t_seq, core::TensorShape::from_dims({frames, freq, kDense}));
    auto td = graph_downsample(ctx, t_seq, require_param(params, root + ".downsample_t.bias"));
    const int64_t d_frames = td.shape.dims[0];
    auto f_seq = graph_transpose(ctx, td, {{1, 0, 2, 3}}, 3);
    auto fd = graph_downsample(ctx, f_seq, require_param(params, root + ".downsample_f.bias"));
    const int64_t d_freq = fd.shape.dims[0];
    auto x = graph_transpose(ctx, fd, {{2, 1, 0, 3}}, 3);
    x = graph_reshape(ctx, x, core::TensorShape::from_dims({1, kDense, d_frames, d_freq}));
    x = graph_run_dual_encoder(ctx, constants, x, params, root + ".encoder");

    auto f_up = graph_transpose(ctx, x, {{3, 2, 0, 1}}, 4);
    f_up = graph_reshape(ctx, f_up, core::TensorShape::from_dims({d_freq, d_frames, kDense}));
    f_up = graph_upsample_slice(ctx, f_up, ds, freq);
    auto t_up = graph_transpose(ctx, f_up, {{1, 0, 2, 3}}, 3);
    t_up = graph_upsample_slice(ctx, t_up, ds, frames);
    auto orig = graph_transpose(ctx, input, {{2, 3, 0, 1}}, 4);
    orig = graph_reshape(ctx, orig, core::TensorShape::from_dims({frames, freq, kDense}));
    auto combined = graph_bypass(ctx, orig, t_up, require_param(params, root + ".out_combiner.bypass_scale"));
    auto out = graph_transpose(ctx, combined, {{2, 0, 1, 3}}, 3);
    return graph_reshape(ctx, out, core::TensorShape::from_dims({1, kDense, frames, freq}));
}

core::TensorValue graph_tsconformer(
    core::ModuleBuildContext & ctx,
    std::vector<GraphConstant> & constants,
    core::TensorValue x,
    const std::unordered_map<std::string, Param> & params) {
    x = graph_run_dual_encoder(ctx, constants, x, params, "TSConformer.encoders.0");
    x = graph_run_downsampled_encoder(ctx, constants, x, params, "TSConformer.encoders.1", 2);
    x = graph_run_downsampled_encoder(ctx, constants, x, params, "TSConformer.encoders.2", 2);
    x = graph_run_dual_encoder(ctx, constants, x, params, "TSConformer.encoders.3");
    return x;
}

core::TensorValue graph_subpixel_conv(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::unordered_map<std::string, Param> & params,
    const std::string & weight_name,
    const std::string & bias_name) {
    auto conv = graph_conv2d(ctx, input, params, weight_name, bias_name, 0, 1, 1, 1, 1, 1);
    if (conv.shape.dims[1] != kDense * 2) {
        throw std::runtime_error("ZipEnhancer graph subpixel conv channel mismatch: " + weight_name);
    }
    auto x = graph_reshape(
        ctx,
        conv,
        core::TensorShape::from_dims({conv.shape.dims[0], kDense, 2, conv.shape.dims[2] * conv.shape.dims[3]}));
    x = graph_transpose(ctx, x, {{0, 1, 3, 2}}, 4);
    return graph_reshape(ctx, x, core::TensorShape::from_dims({conv.shape.dims[0], kDense, conv.shape.dims[2], conv.shape.dims[3] * 2}));
}

core::TensorValue graph_mapping_decoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::unordered_map<std::string, Param> & params) {
    auto x = graph_dense_block(ctx, input, params, "mask_decoder.dense_block");
    x = graph_subpixel_conv(ctx, x, params, "mask_decoder.mask_conv.0.conv1.weight", "mask_decoder.mask_conv.0.conv1.bias");
    x = graph_instance_norm_prelu(
        ctx,
        x,
        params,
        "mask_decoder.mask_conv.1.weight",
        "mask_decoder.mask_conv.1.bias",
        "mask_decoder.mask_conv.2.weight");
    x = graph_conv2d(ctx, x, params, "mask_decoder.mask_conv.3.weight", "mask_decoder.mask_conv.3.bias", 0, 0, 1, 1, 1, 1);
    return modules::ReluModule().build(ctx, x);
}

std::pair<core::TensorValue, core::TensorValue> graph_phase_decoder_real_imag(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const std::unordered_map<std::string, Param> & params) {
    auto x = graph_dense_block(ctx, input, params, "phase_decoder.dense_block");
    x = graph_subpixel_conv(ctx, x, params, "phase_decoder.phase_conv.0.conv1.weight", "phase_decoder.phase_conv.0.conv1.bias");
    x = graph_instance_norm_prelu(
        ctx,
        x,
        params,
        "phase_decoder.phase_conv.1.weight",
        "phase_decoder.phase_conv.1.bias",
        "phase_decoder.phase_conv.2.weight");
    auto real = graph_conv2d(ctx, x, params, "phase_decoder.phase_conv_r.weight", "phase_decoder.phase_conv_r.bias", 0, 0, 1, 1, 1, 1);
    auto imag = graph_conv2d(ctx, x, params, "phase_decoder.phase_conv_i.weight", "phase_decoder.phase_conv_i.bias", 0, 0, 1, 1, 1, 1);
    return {real, imag};
}

Tensor4 make_model_input(
    const std::vector<float> & mag,
    const std::vector<float> & pha,
    int64_t frames) {
    Tensor4 x(1, 2, frames, kFreqBins);
    for (int64_t f = 0; f < kFreqBins; ++f) {
        for (int64_t t = 0; t < frames; ++t) {
            x.at(0, 0, t, f) = mag[static_cast<size_t>(f * frames + t)];
            x.at(0, 1, t, f) = pha[static_cast<size_t>(f * frames + t)];
        }
    }
    return x;
}

struct ZipEnhancerGraphOutput {
    Tensor4 magnitude;
    Tensor4 phase_real;
    Tensor4 phase_imag;
};

class ZipEnhancerForwardGraph {
public:
    ZipEnhancerForwardGraph(
        const std::unordered_map<std::string, Param> & params,
        ggml_backend_t backend,
        core::BackendType backend_type,
        int64_t frames)
        : backend_(backend),
          backend_type_(backend_type),
          frames_(frames) {
        if (backend_ == nullptr) {
            throw std::runtime_error("ZipEnhancer backend is not initialized");
        }
        if (frames_ <= 0) {
            throw std::runtime_error("ZipEnhancer forward graph frame count must be positive");
        }
        ggml_init_params init{kZipEnhancerForwardGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(init));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize ZipEnhancer forward graph context");
        }

        core::ModuleBuildContext build_ctx{ctx_.get(), "zipenhancer.forward", backend_type_};
        auto input = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 2, frames_, kFreqBins}));
        input_ = input.tensor;
        auto encoded = graph_dense_encoder(build_ctx, input, params);
        auto transformed = graph_tsconformer(build_ctx, constants_, encoded, params);
        auto magnitude = graph_mapping_decoder(build_ctx, transformed, params);
        auto phase = graph_phase_decoder_real_imag(build_ctx, transformed, params);
        magnitude_ = magnitude.tensor;
        phase_real_ = phase.first.tensor;
        phase_imag_ = phase.second.tensor;
        magnitude_shape_ = magnitude.shape;
        phase_shape_ = phase.first.shape;
        ggml_set_output(magnitude_);
        ggml_set_output(phase_real_);
        ggml_set_output(phase_imag_);

        graph_ = ggml_new_graph_custom(ctx_.get(), kZipEnhancerForwardGraphNodes, false);
        ggml_build_forward_expand(graph_, magnitude_);
        ggml_build_forward_expand(graph_, phase_real_);
        ggml_build_forward_expand(graph_, phase_imag_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate ZipEnhancer forward graph");
        }
        int64_t cached_seq = 0;
        std::vector<float> cached_pos_emb;
        for (const auto & constant : constants_) {
            if (constant.tensor == nullptr || constant.pos_weight == nullptr) {
                throw std::runtime_error("ZipEnhancer relative-position constant is incomplete");
            }
            if (constant.seq != cached_seq) {
                cached_seq = constant.seq;
                cached_pos_emb = compact_rel_pos(cached_seq);
            }
            auto values = precompute_relative_projection(
                constant.seq,
                constant.head,
                constant.dim,
                *constant.pos_weight,
                cached_pos_emb);
            ggml_backend_tensor_set(
                constant.tensor,
                values.data(),
                0,
                values.size() * sizeof(float));
        }
        cached_pos_emb.clear();
        cached_pos_emb.shrink_to_fit();
        plan_ = core::create_backend_graph_plan_if_host(backend_, graph_);
    }

    ~ZipEnhancerForwardGraph() {
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

    ZipEnhancerGraphOutput run(const Tensor4 & input) const {
        if (input.b != 1 || input.c != 2 || input.t != frames_ || input.f != kFreqBins) {
            throw std::runtime_error("ZipEnhancer forward graph input shape changed without rebuild");
        }
        ggml_backend_tensor_set(input_, input.v.data(), 0, input.v.size() * sizeof(float));
        const auto status = core::compute_backend_graph(backend_, graph_, plan_, "ZipEnhancer forward");
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ZipEnhancer forward graph compute failed");
        }
        ZipEnhancerGraphOutput output;
        output.magnitude = Tensor4(magnitude_shape_.dims[0], magnitude_shape_.dims[1], magnitude_shape_.dims[2], magnitude_shape_.dims[3]);
        output.phase_real = Tensor4(phase_shape_.dims[0], phase_shape_.dims[1], phase_shape_.dims[2], phase_shape_.dims[3]);
        output.phase_imag = Tensor4(phase_shape_.dims[0], phase_shape_.dims[1], phase_shape_.dims[2], phase_shape_.dims[3]);
        ggml_backend_tensor_get(magnitude_, output.magnitude.v.data(), 0, output.magnitude.v.size() * sizeof(float));
        ggml_backend_tensor_get(phase_real_, output.phase_real.v.data(), 0, output.phase_real.v.size() * sizeof(float));
        ggml_backend_tensor_get(phase_imag_, output.phase_imag.v.data(), 0, output.phase_imag.v.size() * sizeof(float));
        return output;
    }

private:
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int64_t frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::vector<GraphConstant> constants_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * magnitude_ = nullptr;
    ggml_tensor * phase_real_ = nullptr;
    ggml_tensor * phase_imag_ = nullptr;
    core::TensorShape magnitude_shape_;
    core::TensorShape phase_shape_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_graph_plan_t plan_ = nullptr;
};

}  // namespace

struct ZipEnhancerModelState {
    std::unique_ptr<ggml_backend, BackendDeleter> backend;
    core::BackendType backend_type = core::BackendType::Cpu;
    std::shared_ptr<core::BackendWeightStore> store;
    std::unordered_map<std::string, Param> params;
    mutable std::unique_ptr<ZipEnhancerForwardGraph> forward_graph;
};

ZipEnhancerChunkPlan zipenhancer_chunk_plan(int64_t input_samples, const ZipEnhancerOptions & options) noexcept {
    ZipEnhancerChunkPlan plan;
    plan.window_samples = kWindowSamples;
    plan.stride_samples = kStrideSamples;
    if (input_samples <= 0) {
        return plan;
    }
    if (input_samples <= kSegmentThreshold) {
        plan.padded_samples = std::max<int64_t>(input_samples, kWindowSamples);
        plan.output_samples = (options.match_input_length || input_samples < kWindowSamples)
            ? input_samples
            : (input_samples / kHop) * kHop;
        return plan;
    }
    plan.segmented = true;
    const int64_t remainder = (input_samples - kWindowSamples) % kStrideSamples;
    plan.padded_samples = input_samples + (remainder == 0 ? 0 : kStrideSamples - remainder);
    plan.output_samples = input_samples;
    return plan;
}

float zipenhancer_chunk_fade_weight(int64_t position, int64_t overlap_samples, int64_t fade_samples) noexcept {
    if (overlap_samples <= 0) {
        return 1.0f;
    }
    const int64_t fade = std::clamp<int64_t>(fade_samples, 0, overlap_samples);
    const int64_t lead = (overlap_samples - fade) / 2;
    if (position < lead) {
        return 0.0f;
    }
    if (position >= lead + fade) {
        return 1.0f;
    }
    return static_cast<float>(position - lead + 1) / static_cast<float>(fade + 1);
}

float zipenhancer_segment_weight(
    int64_t offset_in_segment,
    int64_t segment_start,
    const ZipEnhancerChunkPlan & plan,
    const ZipEnhancerOptions & options) noexcept {
    const int64_t overlap = plan.window_samples - plan.stride_samples;
    float weight = 1.0f;
    if (segment_start > 0 && offset_in_segment < overlap) {
        weight = zipenhancer_chunk_fade_weight(offset_in_segment, overlap, options.chunk_crossfade_samples);
    }
    if (segment_start + plan.window_samples < plan.padded_samples && offset_in_segment >= plan.stride_samples) {
        weight = zipenhancer_chunk_fade_weight(
            overlap - 1 - (offset_in_segment - plan.stride_samples),
            overlap,
            options.chunk_crossfade_samples);
    }
    return weight;
}

std::vector<float> zipenhancer_chunk_weights(const ZipEnhancerChunkPlan & plan, const ZipEnhancerOptions & options) {
    if (plan.padded_samples <= 0) {
        return {};
    }
    if (!plan.segmented) {
        return std::vector<float>(static_cast<size_t>(plan.padded_samples), 1.0f);
    }
    std::vector<float> weights(static_cast<size_t>(plan.padded_samples), 0.0f);
    for (int64_t current = 0; current + plan.window_samples <= plan.padded_samples; current += plan.stride_samples) {
        for (int64_t i = 0; i < plan.window_samples; ++i) {
            weights[static_cast<size_t>(current + i)] += zipenhancer_segment_weight(i, current, plan, options);
        }
    }
    return weights;
}

ZipEnhancerWaveformOutput denoise_mono_16k_whole(
    const ZipEnhancerModelState & state,
    const std::vector<float> & waveform,
    int64_t requested_samples) {
    if (waveform.empty()) {
        throw std::runtime_error("ZipEnhancer input waveform is empty");
    }
    double energy = 0.0;
    for (const float value : waveform) {
        energy += static_cast<double>(value) * static_cast<double>(value);
    }
    if (!(energy > 0.0)) {
        throw std::runtime_error("ZipEnhancer input waveform has zero energy");
    }
    const float norm_factor = std::sqrt(static_cast<float>(waveform.size()) / static_cast<float>(energy));
    std::vector<float> normalized(waveform.size());
    for (size_t i = 0; i < waveform.size(); ++i) {
        normalized[i] = waveform[i] * norm_factor;
    }

    const STFTConfig stft_config{kNfft, kHop, kWin, true, STFTPadMode::Reflect, STFTFamily::Kokoro};
    const auto & window = get_cached_stft_window(stft_config);
    const auto complex = STFT().compute_complex(normalized, window, 1, static_cast<int64_t>(normalized.size()), stft_config, 0);
    const int64_t frames = complex.shape[2];
    std::vector<float> mag(static_cast<size_t>(kFreqBins * frames));
    std::vector<float> pha(static_cast<size_t>(kFreqBins * frames));
    for (int64_t f = 0; f < kFreqBins; ++f) {
        for (int64_t t = 0; t < frames; ++t) {
            const size_t base = static_cast<size_t>(((f * frames + t) * 2));
            const float re = complex.values[base];
            const float im = complex.values[base + 1];
            const size_t idx = static_cast<size_t>(f * frames + t);
            mag[idx] = std::pow(std::sqrt(re * re + im * im + 1.0e-9f), kCompress);
            pha[idx] = std::atan2(im, re + 1.0e-5f);
        }
    }

    const auto model_input = make_model_input(mag, pha, frames);
    if (!state.forward_graph || !state.forward_graph->matches(frames)) {
        state.forward_graph.reset();
        state.forward_graph = std::make_unique<ZipEnhancerForwardGraph>(
            state.params,
            state.backend.get(),
            state.backend_type,
            frames);
    }
    const auto decoded = state.forward_graph->run(model_input);
    if (decoded.magnitude.f != kFreqBins || decoded.phase_real.f != kFreqBins || decoded.phase_imag.f != kFreqBins ||
        decoded.magnitude.t != frames || decoded.phase_real.t != frames || decoded.phase_imag.t != frames) {
        throw std::runtime_error("ZipEnhancer decoder output shape mismatch");
    }
    std::vector<float> out_complex(static_cast<size_t>(kFreqBins * frames * 2));
    for (int64_t f = 0; f < kFreqBins; ++f) {
        for (int64_t t = 0; t < frames; ++t) {
            const size_t idx = static_cast<size_t>(f * frames + t);
            const float linear_mag = std::pow(std::max(0.0f, decoded.magnitude.at(0, 0, t, f)), 1.0f / kCompress);
            const float real = decoded.phase_real.at(0, 0, t, f);
            const float imag = decoded.phase_imag.at(0, 0, t, f);
            const float phase = std::atan2(imag, real);
            out_complex[idx * 2] = linear_mag * std::cos(phase);
            out_complex[idx * 2 + 1] = linear_mag * std::sin(phase);
        }
    }
    const int64_t output_samples = requested_samples > 0 ? requested_samples : (frames - 1) * kHop;
    auto wav = ISTFT().compute(
        out_complex,
        window,
        1,
        kFreqBins,
        frames,
        output_samples,
        stft_config,
        0);
    for (float & value : wav.values) {
        value /= norm_factor;
    }
    return ZipEnhancerWaveformOutput{kSampleRate, std::move(wav.values)};
}

ZipEnhancerModel::ZipEnhancerModel() = default;
ZipEnhancerModel::~ZipEnhancerModel() = default;
ZipEnhancerModel::ZipEnhancerModel(ZipEnhancerModel &&) noexcept = default;
ZipEnhancerModel & ZipEnhancerModel::operator=(ZipEnhancerModel &&) noexcept = default;

ZipEnhancerModel::ZipEnhancerModel(std::shared_ptr<const ZipEnhancerModelState> state)
    : state_(std::move(state)) {}

ZipEnhancerModel ZipEnhancerModel::load_from_directory(const std::filesystem::path & model_dir) {
    return load_from_directory(model_dir, core::BackendConfig{});
}

ZipEnhancerModel ZipEnhancerModel::load_from_directory(
    const std::filesystem::path & model_dir,
    const core::BackendConfig & backend_config) {
    auto source = assets::open_tensor_source(model_dir / "zipenhancer.safetensors");
    auto state = std::make_shared<ZipEnhancerModelState>();
    state->backend.reset(core::init_backend(backend_config));
    state->backend_type = core::backend_type(state->backend.get());
    state->store = std::make_shared<core::BackendWeightStore>(
        state->backend.get(),
        state->backend_type,
        "ZipEnhancer",
        32ull * 1024ull * 1024ull);
    for (const auto & meta : source->tensors()) {
        const auto raw = source->require_tensor_data(meta.name);
        if (raw.metadata.dtype != "F32") {
            throw std::runtime_error("ZipEnhancer only supports F32 tensors: " + meta.name);
        }
        const int64_t elements = raw.metadata.shape.empty() ? 1 : product(raw.metadata.shape);
        if (static_cast<int64_t>(raw.bytes.size()) != elements * static_cast<int64_t>(sizeof(float))) {
            throw std::runtime_error("ZipEnhancer tensor byte size mismatch: " + meta.name);
        }
        Param param;
        param.shape = raw.metadata.shape;
        param.values.resize(static_cast<size_t>(elements));
        std::memcpy(param.values.data(), raw.bytes.data(), raw.bytes.size());
        param.tensor = state->store->make_f32(tensor_shape_from_vector(param.shape), param.values);
        state->params.emplace(meta.name, std::move(param));
    }
    state->store->upload();
    source->release_storage();
    trim_zipenhancer_host_params(state->params);
    return ZipEnhancerModel(std::move(state));
}

ZipEnhancerWaveformOutput ZipEnhancerModel::denoise_mono_16k(
    const std::vector<float> & waveform,
    const ZipEnhancerOptions & options) const {
    if (!state_) {
        throw std::runtime_error("ZipEnhancerModel is not loaded");
    }
    if (waveform.empty()) {
        throw std::runtime_error("ZipEnhancer input waveform is empty");
    }

    const int64_t original_samples = static_cast<int64_t>(waveform.size());
    const auto plan = zipenhancer_chunk_plan(original_samples, options);
    if (!plan.segmented) {
        if (original_samples < plan.window_samples) {
            std::vector<float> padded = waveform;
            padded.resize(static_cast<size_t>(plan.window_samples), 0.0f);
            auto output = denoise_mono_16k_whole(*state_, padded, plan.window_samples);
            output.samples.resize(static_cast<size_t>(plan.output_samples));
            return output;
        }
        return denoise_mono_16k_whole(*state_, waveform, plan.output_samples);
    }

    std::vector<float> padded = waveform;
    padded.resize(static_cast<size_t>(plan.padded_samples), 0.0f);
    std::vector<float> output(static_cast<size_t>(plan.padded_samples), 0.0f);
    const auto weights = zipenhancer_chunk_weights(plan, options);
    for (int64_t current = 0; current + plan.window_samples <= plan.padded_samples; current += plan.stride_samples) {
        std::vector<float> segment(
            padded.begin() + static_cast<std::ptrdiff_t>(current),
            padded.begin() + static_cast<std::ptrdiff_t>(current + plan.window_samples));
        const auto segment_output = denoise_mono_16k_whole(*state_, segment, plan.window_samples);
        if (static_cast<int64_t>(segment_output.samples.size()) != plan.window_samples) {
            throw std::runtime_error("ZipEnhancer segmented output length mismatch");
        }
        for (int64_t i = 0; i < plan.window_samples; ++i) {
            const float weight = zipenhancer_segment_weight(i, current, plan, options);
            output[static_cast<size_t>(current + i)] += segment_output.samples[static_cast<size_t>(i)] * weight;
        }
    }
    output.resize(static_cast<size_t>(plan.output_samples));
    for (size_t i = 0; i < output.size(); ++i) {
        if (weights[i] <= 0.0f) {
            throw std::runtime_error("ZipEnhancer segmented synthesis produced an uncovered sample");
        }
        output[i] /= weights[i];
    }
    return ZipEnhancerWaveformOutput{kSampleRate, std::move(output)};
}

}  // namespace engine::audio
