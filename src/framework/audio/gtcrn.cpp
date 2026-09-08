#include "engine/framework/audio/gtcrn.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::audio {
namespace {

constexpr int64_t kSampleRate = 16000;
constexpr int64_t kNfft = 512;
constexpr int64_t kHop = 256;
constexpr int64_t kWin = 512;
constexpr int64_t kFreqBins = 257;
constexpr int64_t kErbLowBins = 65;
constexpr int64_t kErbHighBins = 64;
constexpr int64_t kErbBins = 129;
constexpr int64_t kReducedFreqBins = 33;
constexpr int64_t kChannels = 16;
constexpr int64_t kHalfChannels = 8;
constexpr size_t kGraphContextBytes = 96ull * 1024ull * 1024ull;
constexpr float kPi = 3.14159265358979323846F;

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

struct Conv2dLayer {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel_h = 0;
    int64_t kernel_w = 0;
    int stride_h = 1;
    int stride_w = 1;
    int pad_h = 0;
    int pad_w = 0;
    int dilation_h = 1;
    int dilation_w = 1;
    int groups = 1;
    modules::Conv2dWeights conv;
};

struct BatchNormLayer {
    modules::BatchNorm2dEvalWeights norm;
    int64_t channels = 0;
};

struct GruWeights {
    modules::LinearWeights input;
    modules::LinearWeights recurrent;
    int64_t input_size = 0;
    int64_t hidden_size = 0;
};

struct GroupedGruWeights {
    GruWeights rnn1;
    GruWeights rnn2;
};

struct BiGroupedGruWeights {
    GroupedGruWeights forward;
    GroupedGruWeights reverse;
};

struct TRAWeights {
    GruWeights gru;
    modules::LinearWeights fc;
};

struct GTConvBlockWeights {
    Conv2dLayer point_conv1;
    BatchNormLayer point_bn1;
    core::TensorValue point_prelu;
    Conv2dLayer depth_conv;
    BatchNormLayer depth_bn;
    core::TensorValue depth_prelu;
    Conv2dLayer point_conv2;
    BatchNormLayer point_bn2;
    TRAWeights tra;
    int64_t dilation_t = 1;
};

struct ConvBlockWeights {
    Conv2dLayer conv;
    BatchNormLayer bn;
    core::TensorValue prelu;
    bool tanh = false;
};

struct DPGRNNWeights {
    BiGroupedGruWeights intra_rnn;
    modules::LinearWeights intra_fc;
    modules::NormWeights intra_ln;
    GroupedGruWeights inter_rnn;
    modules::LinearWeights inter_fc;
    modules::NormWeights inter_ln;
};

struct GTCRNWeights {
    ~GTCRNWeights();

    std::filesystem::path source_path;
    std::unique_ptr<ggml_backend, BackendDeleter> backend;
    core::BackendType backend_type = core::BackendType::Cpu;
    std::shared_ptr<core::BackendWeightStore> store;
    modules::LinearWeights erb_fc;
    modules::LinearWeights ierb_fc;
    ConvBlockWeights encoder0;
    ConvBlockWeights encoder1;
    std::array<GTConvBlockWeights, 3> encoder_gt;
    DPGRNNWeights dpgrnn1;
    DPGRNNWeights dpgrnn2;
    std::array<GTConvBlockWeights, 3> decoder_gt;
    ConvBlockWeights decoder3;
    ConvBlockWeights decoder4;
};

core::TensorShape shape(std::initializer_list<int64_t> dims) {
    return core::TensorShape::from_dims(dims);
}

int64_t gtcrn_cache_frames_for_slot(int slot) {
    switch (slot) {
        case 0:
        case 5:
            return 2;
        case 1:
        case 4:
            return 4;
        case 2:
        case 3:
            return 10;
        default:
            throw std::runtime_error("GTCRN cache slot out of range");
    }
}

std::vector<float> gtcrn_sqrt_hann_window() {
    std::vector<float> window(static_cast<size_t>(kWin), 0.0F);
    for (int64_t i = 0; i < kWin; ++i) {
        const float hann = 0.5F - 0.5F * std::cos(2.0F * kPi * static_cast<float>(i) / static_cast<float>(kWin));
        window[static_cast<size_t>(i)] = std::sqrt(std::max(0.0F, hann));
    }
    return window;
}

modules::Conv2dWeights load_conv(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    std::initializer_list<int64_t> weight_shape,
    int64_t bias_channels) {
    return {
        store.load_f32_tensor(source, prefix + ".weight", weight_shape),
        store.load_f32_tensor(source, prefix + ".bias", {bias_channels}),
    };
}

modules::Conv2dWeights load_conv_native(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const std::string & weight_name,
    std::initializer_list<int64_t> weight_shape,
    int64_t bias_channels) {
    return {
        store.load_f32_tensor(source, weight_name, weight_shape),
        store.load_f32_tensor(source, prefix + ".bias", {bias_channels}),
    };
}

modules::LinearWeights load_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_features,
    int64_t in_features,
    bool bias) {
    modules::LinearWeights weights;
    weights.weight = store.load_f32_tensor(source, prefix + ".weight", {out_features, in_features});
    if (bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_features});
    }
    return weights;
}

BatchNormLayer load_bn(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    return {
        {
            store.load_f32_tensor(source, prefix + ".__bn_scale", {channels}),
            store.load_f32_tensor(source, prefix + ".__bn_bias", {channels}),
        },
        channels,
    };
}

GruWeights load_gru(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t input_size,
    int64_t hidden_size) {
    GruWeights weights;
    weights.input = {
        store.load_f32_tensor(source, prefix + ".weight_ih_l0", {hidden_size * 3, input_size}),
        store.load_f32_tensor(source, prefix + ".bias_ih_l0", {hidden_size * 3}),
    };
    weights.recurrent = {
        store.load_f32_tensor(source, prefix + ".weight_hh_l0", {hidden_size * 3, hidden_size}),
        store.load_f32_tensor(source, prefix + ".bias_hh_l0", {hidden_size * 3}),
    };
    weights.input_size = input_size;
    weights.hidden_size = hidden_size;
    return weights;
}

GroupedGruWeights load_grouped_gru(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t group_input,
    int64_t group_hidden) {
    return {
        load_gru(store, source, prefix + ".rnn1", group_input, group_hidden),
        load_gru(store, source, prefix + ".rnn2", group_input, group_hidden),
    };
}

BiGroupedGruWeights load_bi_grouped_gru(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t group_input,
    int64_t group_hidden) {
    return {
        load_grouped_gru(store, source, prefix, group_input, group_hidden),
        {
            load_gru(store, source, prefix + ".rnn1_reverse", group_input, group_hidden),
            load_gru(store, source, prefix + ".rnn2_reverse", group_input, group_hidden),
        },
    };
}

TRAWeights load_tra(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix) {
    return {
        load_gru(store, source, prefix + ".att_gru", kHalfChannels, kChannels),
        load_linear(store, source, prefix + ".att_fc", kHalfChannels, kChannels, true),
    };
}

ConvBlockWeights load_conv_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_w,
    int stride_w,
    int pad_w,
    int groups,
    bool transposed,
    bool tanh) {
    ConvBlockWeights block;
    block.conv = {
        in_channels,
        out_channels,
        1,
        kernel_w,
        1,
        stride_w,
        0,
        pad_w,
        1,
        1,
        groups,
        transposed ? load_conv_native(
                         store,
                         source,
                         prefix + ".conv",
                         prefix + ".conv.__stream_weight",
                         {out_channels, in_channels / groups, 1, kernel_w},
                         out_channels)
                   : load_conv(store, source, prefix + ".conv", {out_channels, in_channels / groups, 1, kernel_w}, out_channels),
    };
    block.bn = load_bn(store, source, prefix + ".bn", out_channels);
    if (!tanh) {
        block.prelu = store.load_f32_tensor(source, prefix + ".act.weight", {1});
    }
    block.tanh = tanh;
    return block;
}

GTConvBlockWeights load_gt_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    bool transposed,
    int64_t dilation_t) {
    GTConvBlockWeights block;
    block.point_conv1 = {
        kHalfChannels * 3,
        kChannels,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        transposed ? load_conv_native(
                         store,
                         source,
                         prefix + ".point_conv1",
                         prefix + ".point_conv1.__stream_weight",
                         {kChannels, kHalfChannels * 3, 1, 1},
                         kChannels)
                   : load_conv(store, source, prefix + ".point_conv1", {kChannels, kHalfChannels * 3, 1, 1}, kChannels),
    };
    block.point_conv2 = {
        kChannels,
        kHalfChannels,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        transposed ? load_conv_native(
                         store,
                         source,
                         prefix + ".point_conv2",
                         prefix + ".point_conv2.__stream_weight",
                         {kHalfChannels, kChannels, 1, 1},
                         kHalfChannels)
                   : load_conv(store, source, prefix + ".point_conv2", {kHalfChannels, kChannels, 1, 1}, kHalfChannels),
    };
    block.point_bn1 = load_bn(store, source, prefix + ".point_bn1", kChannels);
    block.point_prelu = store.load_f32_tensor(source, prefix + ".point_act.weight", {1});
    block.depth_conv = {
        kChannels,
        kChannels,
        3,
        3,
        1,
        1,
        0,
        1,
        static_cast<int>(dilation_t),
        1,
        kChannels,
        transposed ? load_conv_native(
                         store,
                         source,
                         prefix + ".depth_conv",
                         prefix + ".depth_conv.__stream_weight",
                         {kChannels, 1, 3, 3},
                         kChannels)
                   : load_conv(store, source, prefix + ".depth_conv", {kChannels, 1, 3, 3}, kChannels),
    };
    block.depth_bn = load_bn(store, source, prefix + ".depth_bn", kChannels);
    block.depth_prelu = store.load_f32_tensor(source, prefix + ".depth_act.weight", {1});
    block.point_bn2 = load_bn(store, source, prefix + ".point_bn2", kHalfChannels);
    block.tra = load_tra(store, source, prefix + ".tra");
    block.dilation_t = dilation_t;
    return block;
}

DPGRNNWeights load_dpgrnn(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix) {
    return {
        load_bi_grouped_gru(store, source, prefix + ".intra_rnn", 8, 4),
        load_linear(store, source, prefix + ".intra_fc", 16, 16, true),
        {
            store.load_f32_tensor(source, prefix + ".intra_ln.weight", {kReducedFreqBins, kChannels}),
            store.load_f32_tensor(source, prefix + ".intra_ln.bias", {kReducedFreqBins, kChannels}),
        },
        load_grouped_gru(store, source, prefix + ".inter_rnn", 8, 8),
        load_linear(store, source, prefix + ".inter_fc", 16, 16, true),
        {
            store.load_f32_tensor(source, prefix + ".inter_ln.weight", {kReducedFreqBins, kChannels}),
            store.load_f32_tensor(source, prefix + ".inter_ln.bias", {kReducedFreqBins, kChannels}),
        },
    };
}

core::TensorValue contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(ctx, value);
}

core::TensorValue add(core::ModuleBuildContext & ctx, const core::TensorValue & lhs, const core::TensorValue & rhs) {
    return modules::AddModule().build(ctx, contiguous(ctx, lhs), contiguous(ctx, rhs));
}

core::TensorValue mul(core::ModuleBuildContext & ctx, const core::TensorValue & lhs, const core::TensorValue & rhs) {
    return modules::MulModule().build(ctx, contiguous(ctx, lhs), contiguous(ctx, rhs));
}

core::TensorValue scale(core::ModuleBuildContext & ctx, const core::TensorValue & input, float value) {
    const auto x = contiguous(ctx, input);
    return core::wrap_tensor(ggml_scale(ctx.ggml, x.tensor, value), x.shape, GGML_TYPE_F32);
}

core::TensorValue sigmoid(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    const auto x = contiguous(ctx, input);
    return modules::SigmoidModule().build(ctx, x);
}

core::TensorValue tanh_act(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    return modules::TanhModule().build(ctx, contiguous(ctx, input));
}

core::TensorValue sqrt_eps(core::ModuleBuildContext & ctx, const core::TensorValue & input, float eps) {
    const auto x = contiguous(ctx, input);
    auto biased = core::wrap_tensor(ggml_scale_bias(ctx.ggml, x.tensor, 1.0F, eps), x.shape, GGML_TYPE_F32);
    return modules::SqrtModule().build(ctx, biased);
}

core::TensorValue repeat_to(core::ModuleBuildContext & ctx, const core::TensorValue & input, const core::TensorShape & target) {
    return modules::RepeatModule({target}).build(ctx, input);
}

core::TensorValue scalar_like(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & scalar,
    const core::TensorShape & target) {
    auto shaped = core::reshape_tensor(ctx, scalar, core::TensorShape::from_dims({1, 1, 1, 1}));
    return repeat_to(ctx, shaped, target);
}

core::TensorValue prelu(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & alpha) {
    auto x = contiguous(ctx, input);
    auto pos = modules::ReluModule().build(ctx, x);
    auto neg = core::wrap_tensor(ggml_sub(ctx.ggml, x.tensor, pos.tensor), x.shape, GGML_TYPE_F32);
    auto a = scalar_like(ctx, alpha, x.shape);
    return add(ctx, pos, mul(ctx, a, neg));
}

core::TensorValue batch_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const BatchNormLayer & bn) {
    return modules::BatchNorm2dEvalModule({bn.channels}).build(ctx, input, bn.norm);
}

core::TensorValue conv2d_grouped(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Conv2dLayer & layer) {
    if (layer.groups == 1) {
        return modules::Conv2dModule({
            layer.in_channels,
            layer.out_channels,
            layer.kernel_h,
            layer.kernel_w,
            layer.stride_h,
            layer.stride_w,
            layer.pad_h,
            layer.pad_w,
            layer.dilation_h,
            layer.dilation_w,
            true,
        }).build(ctx, input, layer.conv);
    }
    if (layer.groups == layer.in_channels && layer.groups == layer.out_channels) {
        return modules::DepthwiseConv2dModule({
            layer.in_channels,
            layer.kernel_h,
            layer.kernel_w,
            layer.stride_h,
            layer.stride_w,
            layer.pad_h,
            layer.pad_w,
            layer.dilation_h,
            layer.dilation_w,
            true,
        }).build(ctx, input, layer.conv);
    }
    if (layer.in_channels % layer.groups != 0 || layer.out_channels % layer.groups != 0) {
        throw std::runtime_error("GTCRN grouped conv shape mismatch");
    }
    const int64_t in_per_group = layer.in_channels / layer.groups;
    const int64_t out_per_group = layer.out_channels / layer.groups;
    core::TensorValue out;
    for (int64_t group = 0; group < layer.groups; ++group) {
        auto in_g = modules::SliceModule({1, group * in_per_group, in_per_group}).build(ctx, input);
        auto w_g = modules::SliceModule({0, group * out_per_group, out_per_group}).build(ctx, layer.conv.weight);
        auto b_g = modules::SliceModule({0, group * out_per_group, out_per_group}).build(ctx, *layer.conv.bias);
        auto y = modules::Conv2dModule({
            in_per_group,
            out_per_group,
            layer.kernel_h,
            layer.kernel_w,
            layer.stride_h,
            layer.stride_w,
            layer.pad_h,
            layer.pad_w,
            layer.dilation_h,
            layer.dilation_w,
            true,
        }).build(ctx, in_g, {w_g, b_g});
        out = out.valid() ? modules::ConcatModule({1}).build(ctx, out, y) : y;
    }
    return out;
}

core::TensorValue conv_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ConvBlockWeights & block) {
    auto y = conv2d_grouped(ctx, input, block.conv);
    y = batch_norm(ctx, y, block.bn);
    return block.tanh ? tanh_act(ctx, y) : prelu(ctx, y, block.prelu);
}

core::TensorValue sfe_freq(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input) {
    auto first = modules::SliceModule({3, 0, 1}).build(ctx, input);
    auto zero = scale(ctx, first, 0.0F);
    auto left_padded = modules::ConcatModule({3}).build(ctx, zero, input);
    auto right_padded = modules::ConcatModule({3}).build(ctx, input, zero);
    core::TensorValue out;
    for (int64_t c = 0; c < input.shape.dims[1]; ++c) {
        auto left = modules::SliceModule({1, c, 1}).build(ctx, modules::SliceModule({3, 0, input.shape.dims[3]}).build(ctx, left_padded));
        auto center = modules::SliceModule({1, c, 1}).build(ctx, input);
        auto right = modules::SliceModule({1, c, 1}).build(ctx, modules::SliceModule({3, 1, input.shape.dims[3]}).build(ctx, right_padded));
        auto triple = modules::ConcatModule({1}).build(ctx, left, center);
        triple = modules::ConcatModule({1}).build(ctx, triple, right);
        out = out.valid() ? modules::ConcatModule({1}).build(ctx, out, triple) : triple;
    }
    return out;
}

core::TensorValue pad_freq(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t left,
    int64_t right) {
    auto zero = scale(ctx, modules::SliceModule({3, 0, 1}).build(ctx, input), 0.0F);
    auto out = input;
    for (int64_t i = 0; i < left; ++i) {
        out = modules::ConcatModule({3}).build(ctx, zero, out);
    }
    for (int64_t i = 0; i < right; ++i) {
        out = modules::ConcatModule({3}).build(ctx, out, zero);
    }
    return out;
}

core::TensorValue conv_transpose2d_freq_stride2(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const Conv2dLayer & layer) {
    core::TensorValue up;
    for (int64_t f = 0; f < input.shape.dims[3]; ++f) {
        auto value = modules::SliceModule({3, f, 1}).build(ctx, input);
        auto zero = scale(ctx, value, 0.0F);
        up = up.valid() ? modules::ConcatModule({3}).build(ctx, up, value) : value;
        up = modules::ConcatModule({3}).build(ctx, up, zero);
    }
    up = pad_freq(ctx, up, 2, 1);
    Conv2dLayer conv = layer;
    conv.stride_w = 1;
    conv.pad_w = 0;
    return conv2d_grouped(ctx, up, conv);
}

core::TensorValue stream_conv_cache_update(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    const core::TensorValue & input) {
    if (cache.shape.dims[2] == 1) {
        return input;
    }
    auto tail = modules::SliceModule({2, 1, cache.shape.dims[2] - 1}).build(ctx, cache);
    return modules::ConcatModule({2}).build(ctx, tail, input);
}

struct GTBlockOutput {
    core::TensorValue y;
    core::TensorValue conv_cache;
    core::TensorValue tra_cache;
};

core::TensorValue gru_step(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & hidden,
    const core::TensorValue & ones,
    const GruWeights & weights) {
    auto input_gates = modules::LinearModule({weights.input_size, weights.hidden_size * 3, true}).build(ctx, input, weights.input);
    auto recurrent_gates = modules::LinearModule({weights.hidden_size, weights.hidden_size * 3, true}).build(ctx, hidden, weights.recurrent);
    auto input_reset = modules::SliceModule({1, 0, weights.hidden_size}).build(ctx, input_gates);
    auto input_update = modules::SliceModule({1, weights.hidden_size, weights.hidden_size}).build(ctx, input_gates);
    auto input_candidate = modules::SliceModule({1, weights.hidden_size * 2, weights.hidden_size}).build(ctx, input_gates);
    auto recurrent_reset = modules::SliceModule({1, 0, weights.hidden_size}).build(ctx, recurrent_gates);
    auto recurrent_update = modules::SliceModule({1, weights.hidden_size, weights.hidden_size}).build(ctx, recurrent_gates);
    auto recurrent_candidate = modules::SliceModule({1, weights.hidden_size * 2, weights.hidden_size}).build(ctx, recurrent_gates);
    auto reset = sigmoid(ctx, add(ctx, input_reset, recurrent_reset));
    auto update = sigmoid(ctx, add(ctx, input_update, recurrent_update));
    auto candidate = tanh_act(ctx, add(ctx, input_candidate, mul(ctx, reset, recurrent_candidate)));
    auto one_minus_update = core::wrap_tensor(ggml_sub(ctx.ggml, ones.tensor, update.tensor), update.shape, GGML_TYPE_F32);
    return add(ctx, mul(ctx, one_minus_update, candidate), mul(ctx, update, hidden));
}

core::TensorValue grouped_gru_step(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & hidden,
    const core::TensorValue & ones8,
    const GroupedGruWeights & weights) {
    auto x1 = modules::SliceModule({1, 0, weights.rnn1.input_size}).build(ctx, input);
    auto x2 = modules::SliceModule({1, weights.rnn1.input_size, weights.rnn2.input_size}).build(ctx, input);
    auto h1 = modules::SliceModule({1, 0, weights.rnn1.hidden_size}).build(ctx, hidden);
    auto h2 = modules::SliceModule({1, weights.rnn1.hidden_size, weights.rnn2.hidden_size}).build(ctx, hidden);
    auto y1 = gru_step(ctx, x1, h1, ones8, weights.rnn1);
    auto y2 = gru_step(ctx, x2, h2, ones8, weights.rnn2);
    return modules::ConcatModule({1}).build(ctx, y1, y2);
}

core::TensorValue grouped_gru_sequence(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bsi,
    const core::TensorValue & initial_hidden,
    const core::TensorValue & ones,
    const GroupedGruWeights & weights,
    bool reverse) {
    const int64_t steps = input_bsi.shape.dims[1];
    auto hidden = initial_hidden;
    std::vector<core::TensorValue> outputs(static_cast<size_t>(steps));
    for (int64_t i = 0; i < steps; ++i) {
        const int64_t step = reverse ? (steps - 1 - i) : i;
        auto x = modules::SliceModule({1, step, 1}).build(ctx, input_bsi);
        x = core::reshape_tensor(ctx, contiguous(ctx, x), shape({input_bsi.shape.dims[0], input_bsi.shape.dims[2]}));
        hidden = grouped_gru_step(ctx, x, hidden, ones, weights);
        outputs[static_cast<size_t>(step)] = hidden;
    }
    core::TensorValue out;
    for (int64_t step = 0; step < steps; ++step) {
        auto y = core::reshape_tensor(ctx, outputs[static_cast<size_t>(step)], shape({input_bsi.shape.dims[0], 1, hidden.shape.dims[1]}));
        out = out.valid() ? modules::ConcatModule({1}).build(ctx, out, y) : y;
    }
    return out;
}

core::TensorValue bi_grouped_gru_sequence(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bsi,
    const core::TensorValue & zero_hidden,
    const core::TensorValue & ones4,
    const BiGroupedGruWeights & weights) {
    auto forward = grouped_gru_sequence(ctx, input_bsi, zero_hidden, ones4, weights.forward, false);
    auto reverse = grouped_gru_sequence(ctx, input_bsi, zero_hidden, ones4, weights.reverse, true);
    auto f1 = modules::SliceModule({2, 0, 4}).build(ctx, forward);
    auto f2 = modules::SliceModule({2, 4, 4}).build(ctx, forward);
    auto r1 = modules::SliceModule({2, 0, 4}).build(ctx, reverse);
    auto r2 = modules::SliceModule({2, 4, 4}).build(ctx, reverse);
    auto g1 = modules::ConcatModule({2}).build(ctx, f1, r1);
    auto g2 = modules::ConcatModule({2}).build(ctx, f2, r2);
    return modules::ConcatModule({2}).build(ctx, g1, g2);
}

core::TensorValue layer_norm_width_hidden(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::NormWeights & weights) {
    auto x = contiguous(ctx, input);
    auto mean_c = modules::ReduceMeanModule({3}).build(ctx, x);
    auto mean = modules::ReduceMeanModule({2}).build(ctx, mean_c);
    mean = repeat_to(ctx, mean, x.shape);
    auto centered = core::wrap_tensor(ggml_sub(ctx.ggml, x.tensor, mean.tensor), x.shape, GGML_TYPE_F32);
    auto var_c = modules::ReduceMeanModule({3}).build(ctx, core::wrap_tensor(ggml_sqr(ctx.ggml, centered.tensor), x.shape, GGML_TYPE_F32));
    auto var = modules::ReduceMeanModule({2}).build(ctx, var_c);
    var = repeat_to(ctx, var, x.shape);
    auto inv_std = core::wrap_tensor(ggml_div(ctx.ggml, repeat_to(ctx, core::reshape_tensor(ctx, weights.weight.value(), shape({1, 1, kReducedFreqBins, kChannels})), x.shape).tensor, sqrt_eps(ctx, var, 1.0e-8F).tensor), x.shape, GGML_TYPE_F32);
    auto normalized = mul(ctx, centered, inv_std);
    auto bias = repeat_to(ctx, core::reshape_tensor(ctx, weights.bias.value(), shape({1, 1, kReducedFreqBins, kChannels})), x.shape);
    return add(ctx, normalized, bias);
}

struct DPOutput {
    core::TensorValue y;
    core::TensorValue inter_cache;
};

DPOutput dpgrnn(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bctf,
    const core::TensorValue & inter_cache,
    const core::TensorValue & zero_hidden8,
    const core::TensorValue & ones4,
    const core::TensorValue & ones8,
    const DPGRNNWeights & weights) {
    auto x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, input_bctf);
    x = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, x);
    auto intra_x = core::reshape_tensor(ctx, contiguous(ctx, x), shape({x.shape.dims[0] * x.shape.dims[1], x.shape.dims[2], x.shape.dims[3]}));
    intra_x = bi_grouped_gru_sequence(ctx, intra_x, zero_hidden8, ones4, weights.intra_rnn);
    intra_x = modules::LinearModule({kChannels, kChannels, true}).build(ctx, intra_x, weights.intra_fc);
    intra_x = core::reshape_tensor(ctx, contiguous(ctx, intra_x), shape({1, 1, kReducedFreqBins, kChannels}));
    intra_x = layer_norm_width_hidden(ctx, intra_x, weights.intra_ln);
    auto intra_out = add(ctx, x, intra_x);

    auto inter_x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, intra_out);
    inter_x = core::reshape_tensor(ctx, contiguous(ctx, inter_x), shape({kReducedFreqBins, 1, kChannels}));
    auto inter_y = grouped_gru_sequence(ctx, inter_x, inter_cache, ones8, weights.inter_rnn, false);
    auto next_cache = core::reshape_tensor(ctx, contiguous(ctx, modules::SliceModule({1, 0, 1}).build(ctx, inter_y)), shape({kReducedFreqBins, kChannels}));
    inter_y = modules::LinearModule({kChannels, kChannels, true}).build(ctx, inter_y, weights.inter_fc);
    inter_y = core::reshape_tensor(ctx, contiguous(ctx, inter_y), shape({1, kReducedFreqBins, 1, kChannels}));
    inter_y = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, inter_y);
    inter_y = layer_norm_width_hidden(ctx, inter_y, weights.inter_ln);
    auto out = add(ctx, intra_out, inter_y);
    out = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, out);
    out = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, out);
    return {out, next_cache};
}

GTBlockOutput gt_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & conv_cache,
    const core::TensorValue & tra_cache,
    const core::TensorValue & ones16,
    const GTConvBlockWeights & weights,
    bool transposed) {
    auto x1 = modules::SliceModule({1, 0, input.shape.dims[1] / 2}).build(ctx, input);
    auto x2 = modules::SliceModule({1, input.shape.dims[1] / 2, input.shape.dims[1] / 2}).build(ctx, input);
    x1 = sfe_freq(ctx, x1);
    auto h1 = conv2d_grouped(ctx, x1, weights.point_conv1);
    h1 = prelu(ctx, batch_norm(ctx, h1, weights.point_bn1), weights.point_prelu);
    const auto depth_input_frame = h1;
    core::TensorValue conv_input = modules::ConcatModule({2}).build(ctx, conv_cache, h1);
    if (transposed) {
        conv_input = pad_freq(ctx, conv_input, 1, 1);
        Conv2dLayer conv = weights.depth_conv;
        conv.pad_w = 0;
        h1 = conv2d_grouped(ctx, conv_input, conv);
    } else {
        h1 = conv2d_grouped(ctx, conv_input, weights.depth_conv);
    }
    auto next_conv_cache = stream_conv_cache_update(ctx, conv_cache, depth_input_frame);
    h1 = prelu(ctx, batch_norm(ctx, h1, weights.depth_bn), weights.depth_prelu);
    h1 = conv2d_grouped(ctx, h1, weights.point_conv2);
    h1 = batch_norm(ctx, h1, weights.point_bn2);

    auto zt = modules::ReduceMeanModule({3}).build(ctx, core::wrap_tensor(ggml_sqr(ctx.ggml, contiguous(ctx, h1).tensor), h1.shape, GGML_TYPE_F32));
    zt = core::reshape_tensor(ctx, contiguous(ctx, modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, zt)), shape({1, kHalfChannels}));
    auto next_tra_cache = gru_step(ctx, zt, tra_cache, ones16, weights.tra.gru);
    auto at = modules::LinearModule({kChannels, kHalfChannels, true}).build(ctx, next_tra_cache, weights.tra.fc);
    at = sigmoid(ctx, at);
    at = repeat_to(ctx, core::reshape_tensor(ctx, at, shape({1, kHalfChannels, 1, 1})), h1.shape);
    h1 = mul(ctx, h1, at);

    core::TensorValue shuffled;
    for (int64_t c = 0; c < kHalfChannels; ++c) {
        auto a = modules::SliceModule({1, c, 1}).build(ctx, h1);
        auto b = modules::SliceModule({1, c, 1}).build(ctx, x2);
        auto pair = modules::ConcatModule({1}).build(ctx, a, b);
        shuffled = shuffled.valid() ? modules::ConcatModule({1}).build(ctx, shuffled, pair) : pair;
    }
    return {shuffled, next_conv_cache, next_tra_cache};
}

core::TensorValue erb_compress(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & feat,
    const modules::LinearWeights & erb_fc) {
    auto low = modules::SliceModule({3, 0, kErbLowBins}).build(ctx, feat);
    auto high = modules::SliceModule({3, kErbLowBins, kFreqBins - kErbLowBins}).build(ctx, feat);
    high = modules::LinearModule({kFreqBins - kErbLowBins, kErbHighBins, false}).build(ctx, high, erb_fc);
    return modules::ConcatModule({3}).build(ctx, low, high);
}

core::TensorValue erb_expand(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & feat,
    const modules::LinearWeights & ierb_fc) {
    auto low = modules::SliceModule({3, 0, kErbLowBins}).build(ctx, feat);
    auto high = modules::SliceModule({3, kErbLowBins, kErbHighBins}).build(ctx, feat);
    high = modules::LinearModule({kErbHighBins, kFreqBins - kErbLowBins, false}).build(ctx, high, ierb_fc);
    return modules::ConcatModule({3}).build(ctx, low, high);
}

struct GraphBuildOutput {
    core::TensorValue output;
    std::array<core::TensorValue, 6> conv_cache_out;
    std::array<core::TensorValue, 6> tra_cache_out;
    std::array<core::TensorValue, 2> inter_cache_out;
};

}  // namespace

struct GTCRNModelState {
    GTCRNWeights weights;
};

class GTCRNFrameGraph {
public:
    explicit GTCRNFrameGraph(const GTCRNModelState & state) : state_(state) {
        ggml_init_params params{};
        params.mem_size = kGraphContextBytes;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ctx_.reset(ggml_init(params));
        if (!ctx_) {
            throw std::runtime_error("failed to create GTCRN frame graph context");
        }

        core::ModuleBuildContext build_ctx{ctx_.get(), "gtcrn.frame", state_.weights.backend_type};
        input_ = core::make_tensor(build_ctx, GGML_TYPE_F32, shape({1, kFreqBins, 1, 2}));
        zero_hidden8_ = core::make_tensor(build_ctx, GGML_TYPE_F32, shape({1, 8}));
        ones4_ = core::make_tensor(build_ctx, GGML_TYPE_F32, shape({1, 4}));
        ones8_ = core::make_tensor(build_ctx, GGML_TYPE_F32, shape({kReducedFreqBins, 8}));
        ones16_ = core::make_tensor(build_ctx, GGML_TYPE_F32, shape({1, 16}));
        for (int i = 0; i < 6; ++i) {
            const int64_t pad = gtcrn_cache_frames_for_slot(i);
            conv_cache_in_[static_cast<size_t>(i)] = core::make_tensor(build_ctx, GGML_TYPE_F32, shape({1, kChannels, pad, kReducedFreqBins}));
            tra_cache_in_[static_cast<size_t>(i)] = core::make_tensor(build_ctx, GGML_TYPE_F32, shape({1, kChannels}));
        }
        for (auto & cache : inter_cache_in_) {
            cache = core::make_tensor(build_ctx, GGML_TYPE_F32, shape({kReducedFreqBins, kChannels}));
        }

        auto real = modules::SliceModule({3, 0, 1}).build(build_ctx, input_);
        real = core::reshape_tensor(build_ctx, contiguous(build_ctx, real), shape({1, kFreqBins, 1}));
        auto imag = modules::SliceModule({3, 1, 1}).build(build_ctx, input_);
        imag = core::reshape_tensor(build_ctx, contiguous(build_ctx, imag), shape({1, kFreqBins, 1}));
        auto mag = sqrt_eps(build_ctx, add(build_ctx, core::wrap_tensor(ggml_sqr(build_ctx.ggml, real.tensor), real.shape, GGML_TYPE_F32), core::wrap_tensor(ggml_sqr(build_ctx.ggml, imag.tensor), imag.shape, GGML_TYPE_F32)), 1.0e-12F);
        auto feat = modules::ConcatModule({1}).build(
            build_ctx,
            core::reshape_tensor(build_ctx, mag, shape({1, 1, kFreqBins, 1})),
            core::reshape_tensor(build_ctx, real, shape({1, 1, kFreqBins, 1})));
        feat = modules::ConcatModule({1}).build(build_ctx, feat, core::reshape_tensor(build_ctx, imag, shape({1, 1, kFreqBins, 1})));
        feat = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(build_ctx, feat);
        feat = erb_compress(build_ctx, feat, state_.weights.erb_fc);
        feat = sfe_freq(build_ctx, feat);
        auto e0 = conv_block(build_ctx, feat, state_.weights.encoder0);
        auto e1 = conv_block(build_ctx, e0, state_.weights.encoder1);
        std::array<core::TensorValue, 5> en_outs = {e0, e1, {}, {}, {}};
        auto gt = gt_block(build_ctx, e1, conv_cache_in_[0], tra_cache_in_[0], ones16_, state_.weights.encoder_gt[0], false);
        en_outs[2] = gt.y;
        conv_cache_out_[0] = gt.conv_cache;
        tra_cache_out_[0] = gt.tra_cache;
        gt = gt_block(build_ctx, gt.y, conv_cache_in_[1], tra_cache_in_[1], ones16_, state_.weights.encoder_gt[1], false);
        en_outs[3] = gt.y;
        conv_cache_out_[1] = gt.conv_cache;
        tra_cache_out_[1] = gt.tra_cache;
        gt = gt_block(build_ctx, gt.y, conv_cache_in_[2], tra_cache_in_[2], ones16_, state_.weights.encoder_gt[2], false);
        en_outs[4] = gt.y;
        conv_cache_out_[2] = gt.conv_cache;
        tra_cache_out_[2] = gt.tra_cache;

        auto dp = dpgrnn(build_ctx, gt.y, inter_cache_in_[0], zero_hidden8_, ones4_, ones8_, state_.weights.dpgrnn1);
        inter_cache_out_[0] = dp.inter_cache;
        dp = dpgrnn(build_ctx, dp.y, inter_cache_in_[1], zero_hidden8_, ones4_, ones8_, state_.weights.dpgrnn2);
        inter_cache_out_[1] = dp.inter_cache;

        gt = gt_block(build_ctx, add(build_ctx, dp.y, en_outs[4]), conv_cache_in_[3], tra_cache_in_[3], ones16_, state_.weights.decoder_gt[0], true);
        conv_cache_out_[3] = gt.conv_cache;
        tra_cache_out_[3] = gt.tra_cache;
        gt = gt_block(build_ctx, add(build_ctx, gt.y, en_outs[3]), conv_cache_in_[4], tra_cache_in_[4], ones16_, state_.weights.decoder_gt[1], true);
        conv_cache_out_[4] = gt.conv_cache;
        tra_cache_out_[4] = gt.tra_cache;
        gt = gt_block(build_ctx, add(build_ctx, gt.y, en_outs[2]), conv_cache_in_[5], tra_cache_in_[5], ones16_, state_.weights.decoder_gt[2], true);
        conv_cache_out_[5] = gt.conv_cache;
        tra_cache_out_[5] = gt.tra_cache;
        auto y = conv_transpose2d_freq_stride2(build_ctx, add(build_ctx, gt.y, en_outs[1]), state_.weights.decoder3.conv);
        y = prelu(build_ctx, batch_norm(build_ctx, y, state_.weights.decoder3.bn), state_.weights.decoder3.prelu);
        y = conv_transpose2d_freq_stride2(build_ctx, add(build_ctx, y, en_outs[0]), state_.weights.decoder4.conv);
        y = tanh_act(build_ctx, batch_norm(build_ctx, y, state_.weights.decoder4.bn));
        auto mask = erb_expand(build_ctx, y, state_.weights.ierb_fc);
        auto mask_real = modules::SliceModule({1, 0, 1}).build(build_ctx, mask);
        auto mask_imag = modules::SliceModule({1, 1, 1}).build(build_ctx, mask);
        auto spec_real = modules::SliceModule({3, 0, 1}).build(build_ctx, input_);
        spec_real = core::reshape_tensor(build_ctx, contiguous(build_ctx, spec_real), shape({1, 1, 1, kFreqBins}));
        auto spec_imag = modules::SliceModule({3, 1, 1}).build(build_ctx, input_);
        spec_imag = core::reshape_tensor(build_ctx, contiguous(build_ctx, spec_imag), shape({1, 1, 1, kFreqBins}));
        auto out_real = add(
            build_ctx,
            mul(build_ctx, spec_real, mask_real),
            scale(build_ctx, mul(build_ctx, spec_imag, mask_imag), -1.0F));
        auto out_imag = add(build_ctx, mul(build_ctx, spec_imag, mask_real), mul(build_ctx, spec_real, mask_imag));
        out_real = core::reshape_tensor(build_ctx, contiguous(build_ctx, out_real), shape({1, kFreqBins, 1, 1}));
        out_imag = core::reshape_tensor(build_ctx, contiguous(build_ctx, out_imag), shape({1, kFreqBins, 1, 1}));
        output_ = modules::ConcatModule({3}).build(build_ctx, out_real, out_imag);
        output_scratch_.resize(static_cast<size_t>(kFreqBins * 2));

        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        for (const auto & cache : conv_cache_out_) {
            ggml_build_forward_expand(graph_, cache.tensor);
        }
        for (const auto & cache : tra_cache_out_) {
            ggml_build_forward_expand(graph_, cache.tensor);
        }
        for (const auto & cache : inter_cache_out_) {
            ggml_build_forward_expand(graph_, cache.tensor);
        }
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), state_.weights.backend.get());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate GTCRN frame graph tensors");
        }
        std::vector<float> zeros8(8, 0.0F);
        std::vector<float> ones4(4, 1.0F);
        std::vector<float> ones8(static_cast<size_t>(kReducedFreqBins * 8), 1.0F);
        std::vector<float> ones16(16, 1.0F);
        core::write_tensor_f32(zero_hidden8_, zeros8);
        core::write_tensor_f32(ones4_, ones4);
        core::write_tensor_f32(ones8_, ones8);
        core::write_tensor_f32(ones16_, ones16);
        plan_ = core::create_backend_graph_plan_if_host(state_.weights.backend.get(), graph_);
    }

    ~GTCRNFrameGraph() {
        if (plan_ != nullptr) {
            core::free_backend_graph_plan(state_.weights.backend.get(), plan_);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    void run(
        const float * input_frame,
        std::array<std::vector<float>, 6> & conv_cache,
        std::array<std::vector<float>, 6> & tra_cache,
        std::array<std::vector<float>, 2> & inter_cache,
        float * output_frame) {
        core::write_tensor_f32(input_, input_frame, static_cast<size_t>(kFreqBins * 2));
        for (int i = 0; i < 6; ++i) {
            core::write_tensor_f32(conv_cache_in_[static_cast<size_t>(i)], conv_cache[static_cast<size_t>(i)]);
            core::write_tensor_f32(tra_cache_in_[static_cast<size_t>(i)], tra_cache[static_cast<size_t>(i)]);
        }
        for (int i = 0; i < 2; ++i) {
            core::write_tensor_f32(inter_cache_in_[static_cast<size_t>(i)], inter_cache[static_cast<size_t>(i)]);
        }
        const auto status = core::compute_backend_graph(state_.weights.backend.get(), graph_, plan_, "GTCRN frame");
        ggml_backend_synchronize(state_.weights.backend.get());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("GTCRN frame GGML graph compute failed");
        }
        core::read_tensor_f32_into(output_.tensor, output_scratch_);
        std::copy(output_scratch_.begin(), output_scratch_.end(), output_frame);
        for (int i = 0; i < 6; ++i) {
            core::read_tensor_f32_into(conv_cache_out_[static_cast<size_t>(i)].tensor, conv_cache[static_cast<size_t>(i)]);
            core::read_tensor_f32_into(tra_cache_out_[static_cast<size_t>(i)].tensor, tra_cache[static_cast<size_t>(i)]);
        }
        for (int i = 0; i < 2; ++i) {
            core::read_tensor_f32_into(inter_cache_out_[static_cast<size_t>(i)].tensor, inter_cache[static_cast<size_t>(i)]);
        }
    }

private:
    const GTCRNModelState & state_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue input_;
    core::TensorValue zero_hidden8_;
    core::TensorValue ones4_;
    core::TensorValue ones8_;
    core::TensorValue ones16_;
    std::array<core::TensorValue, 6> conv_cache_in_;
    std::array<core::TensorValue, 6> tra_cache_in_;
    std::array<core::TensorValue, 2> inter_cache_in_;
    std::array<core::TensorValue, 6> conv_cache_out_;
    std::array<core::TensorValue, 6> tra_cache_out_;
    std::array<core::TensorValue, 2> inter_cache_out_;
    core::TensorValue output_;
    std::vector<float> output_scratch_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_backend_graph_plan_t plan_ = nullptr;
};

GTCRNWeights::~GTCRNWeights() = default;

struct GTCRNStreamingSession::State {
    std::array<std::vector<float>, 6> conv_cache;
    std::array<std::vector<float>, 6> tra_cache;
    std::array<std::vector<float>, 2> inter_cache;
    std::unique_ptr<GTCRNFrameGraph> graph;

    explicit State(const GTCRNModelState & model) {
        reset();
        graph = std::make_unique<GTCRNFrameGraph>(model);
    }

    void reset() {
        for (int i = 0; i < 6; ++i) {
            const int64_t pad = gtcrn_cache_frames_for_slot(i);
            conv_cache[static_cast<size_t>(i)].assign(static_cast<size_t>(kChannels * pad * kReducedFreqBins), 0.0F);
            tra_cache[static_cast<size_t>(i)].assign(static_cast<size_t>(kChannels), 0.0F);
        }
        for (auto & cache : inter_cache) {
            cache.assign(static_cast<size_t>(kReducedFreqBins * kChannels), 0.0F);
        }
    }
};

GTCRNModel::GTCRNModel() = default;
GTCRNModel::~GTCRNModel() = default;
GTCRNModel::GTCRNModel(GTCRNModel &&) noexcept = default;
GTCRNModel & GTCRNModel::operator=(GTCRNModel &&) noexcept = default;

GTCRNModel::GTCRNModel(std::shared_ptr<const GTCRNModelState> state) : state_(std::move(state)) {
    if (state_ == nullptr) {
        throw std::runtime_error("GTCRN model requires state");
    }
}

GTCRNModel GTCRNModel::load_from_safetensors(const std::filesystem::path & checkpoint_path) {
    return load_from_safetensors(checkpoint_path, core::BackendConfig{});
}

GTCRNModel GTCRNModel::load_from_safetensors(
    const std::filesystem::path & checkpoint_path,
    const core::BackendConfig & backend_config) {
    auto source = assets::open_tensor_source(checkpoint_path);
    auto state = std::make_shared<GTCRNModelState>();
    auto & weights = state->weights;
    weights.source_path = checkpoint_path;
    weights.backend.reset(core::init_backend(backend_config));
    weights.backend_type = core::backend_type(weights.backend.get());
    weights.store = std::make_shared<core::BackendWeightStore>(weights.backend.get(), weights.backend_type, "GTCRN", 4ull * 1024ull * 1024ull);
    weights.erb_fc = {weights.store->load_f32_tensor(*source, "erb.erb_fc.weight", {kErbHighBins, kFreqBins - kErbLowBins}), std::nullopt};
    weights.ierb_fc = {weights.store->load_f32_tensor(*source, "erb.ierb_fc.weight", {kFreqBins - kErbLowBins, kErbHighBins}), std::nullopt};
    weights.encoder0 = load_conv_block(*weights.store, *source, "encoder.en_convs.0", 9, kChannels, 5, 2, 2, 1, false, false);
    weights.encoder1 = load_conv_block(*weights.store, *source, "encoder.en_convs.1", kChannels, kChannels, 5, 2, 2, 2, false, false);
    weights.encoder_gt = {
        load_gt_block(*weights.store, *source, "encoder.en_convs.2", false, 1),
        load_gt_block(*weights.store, *source, "encoder.en_convs.3", false, 2),
        load_gt_block(*weights.store, *source, "encoder.en_convs.4", false, 5),
    };
    weights.dpgrnn1 = load_dpgrnn(*weights.store, *source, "dpgrnn1");
    weights.dpgrnn2 = load_dpgrnn(*weights.store, *source, "dpgrnn2");
    weights.decoder_gt = {
        load_gt_block(*weights.store, *source, "decoder.de_convs.0", true, 5),
        load_gt_block(*weights.store, *source, "decoder.de_convs.1", true, 2),
        load_gt_block(*weights.store, *source, "decoder.de_convs.2", true, 1),
    };
    weights.decoder3 = load_conv_block(*weights.store, *source, "decoder.de_convs.3", kChannels, kChannels, 5, 2, 2, 2, true, false);
    weights.decoder4 = load_conv_block(*weights.store, *source, "decoder.de_convs.4", kChannels, 2, 5, 2, 2, 1, true, true);
    weights.store->upload();
    source->release_storage();
    return GTCRNModel(std::move(state));
}

std::unique_ptr<GTCRNStreamingSession> GTCRNModel::create_streaming_session() const {
    if (state_ == nullptr) {
        throw std::runtime_error("GTCRN model is not initialized");
    }
    return std::make_unique<GTCRNStreamingSession>(state_);
}

GTCRNWaveformOutput GTCRNModel::denoise_mono_16k(const std::vector<float> & waveform) const {
    if (waveform.empty()) {
        throw std::runtime_error("GTCRN waveform input is empty");
    }
    const STFTConfig stft_config{kNfft, kHop, kWin, true, STFTPadMode::Reflect, STFTFamily::Default};
    const auto window = gtcrn_sqrt_hann_window();
    auto spec = STFT().compute_complex(waveform, window, 1, static_cast<int64_t>(waveform.size()), stft_config);
    const int64_t frames = spec.shape[2];
    auto session = create_streaming_session();
    std::vector<float> enhanced(spec.values.size(), 0.0F);
    std::array<float, kFreqBins * 2> input_frame{};
    std::array<float, kFreqBins * 2> output_frame{};
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t f = 0; f < kFreqBins; ++f) {
            const size_t src = static_cast<size_t>(((f * frames) + frame) * 2);
            input_frame[static_cast<size_t>(f * 2)] = spec.values[src];
            input_frame[static_cast<size_t>(f * 2 + 1)] = spec.values[src + 1];
        }
        session->process_stft_frame(input_frame.data(), output_frame.data());
        for (int64_t f = 0; f < kFreqBins; ++f) {
            const size_t dst = static_cast<size_t>(((f * frames) + frame) * 2);
            enhanced[dst] = output_frame[static_cast<size_t>(f * 2)];
            enhanced[dst + 1] = output_frame[static_cast<size_t>(f * 2 + 1)];
        }
    }
    auto wav = ISTFT().compute(enhanced, window, 1, kFreqBins, frames, static_cast<int64_t>(waveform.size()), stft_config);
    return {static_cast<int>(kSampleRate), std::move(wav.values)};
}

GTCRNStreamingSession::GTCRNStreamingSession(std::shared_ptr<const GTCRNModelState> state)
    : model_(std::move(state)),
      state_(model_ == nullptr ? nullptr : std::make_unique<State>(*model_)) {
    if (model_ == nullptr || state_ == nullptr) {
        throw std::runtime_error("GTCRN streaming session requires model state");
    }
}

GTCRNStreamingSession::~GTCRNStreamingSession() = default;
GTCRNStreamingSession::GTCRNStreamingSession(GTCRNStreamingSession &&) noexcept = default;
GTCRNStreamingSession & GTCRNStreamingSession::operator=(GTCRNStreamingSession &&) noexcept = default;

void GTCRNStreamingSession::reset() {
    if (state_ == nullptr) {
        throw std::runtime_error("GTCRN streaming session is not initialized");
    }
    state_->reset();
}

void GTCRNStreamingSession::process_stft_frame(const float * spec_frame_257x2, float * out_spec_frame_257x2) {
    if (spec_frame_257x2 == nullptr || out_spec_frame_257x2 == nullptr || state_ == nullptr || !state_->graph) {
        throw std::runtime_error("GTCRN streaming session is not initialized");
    }
    state_->graph->run(spec_frame_257x2, state_->conv_cache, state_->tra_cache, state_->inter_cache, out_spec_frame_257x2);
}

}  // namespace engine::audio
