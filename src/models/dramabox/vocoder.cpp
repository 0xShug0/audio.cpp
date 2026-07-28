#include "engine/models/dramabox/vocoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/optimizations/fast_conv_modules.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::dramabox {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t kNumResblockKernels = 3;
constexpr int64_t kActivationRatio = 2;
constexpr int64_t kActivationKernel = 12;
constexpr size_t kVocoderWeightContextBytes = 1536ull * 1024ull * 1024ull;
constexpr size_t kVocoderGraphContextBytes = 384ull * 1024ull * 1024ull;
constexpr size_t kVocoderGraphNodeCapacity = 65536;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

std::vector<float> reversed_filter(std::vector<float> values) {
    std::reverse(values.begin(), values.end());
    return values;
}

std::vector<float> expand_channel_filter(const std::vector<float> & filter, int64_t channels) {
    if (filter.empty() || channels <= 0) {
        throw std::runtime_error("DramaBox vocoder filter shape mismatch");
    }
    std::vector<float> expanded(static_cast<size_t>(channels * static_cast<int64_t>(filter.size())), 0.0F);
    for (int64_t channel = 0; channel < channels; ++channel) {
        std::copy(
            filter.begin(),
            filter.end(),
            expanded.begin() + static_cast<std::ptrdiff_t>(channel * static_cast<int64_t>(filter.size())));
    }
    return expanded;
}

std::vector<float> expand_activation_filter(const std::vector<float> & filter, int64_t channels) {
    if (static_cast<int64_t>(filter.size()) != kActivationKernel || channels <= 0) {
        throw std::runtime_error("DramaBox vocoder activation filter shape mismatch");
    }
    return expand_channel_filter(filter, channels);
}

float sinc(float x) {
    if (x == 0.0F) {
        return 1.0F;
    }
    constexpr float pi = 3.14159265358979323846F;
    return std::sin(pi * x) / (pi * x);
}

std::vector<float> make_hann_resampler_filter(int64_t ratio) {
    constexpr float rolloff = 0.99F;
    constexpr float lowpass_filter_width = 6.0F;
    const int64_t width = static_cast<int64_t>(std::ceil(lowpass_filter_width / rolloff));
    const int64_t kernel_size = 2 * width * ratio + 1;
    std::vector<float> filter(static_cast<size_t>(kernel_size), 0.0F);
    constexpr float pi = 3.14159265358979323846F;
    for (int64_t i = 0; i < kernel_size; ++i) {
        const float time_axis = (static_cast<float>(i) / static_cast<float>(ratio) - static_cast<float>(width)) * rolloff;
        const float clamped = std::max(-lowpass_filter_width, std::min(lowpass_filter_width, time_axis));
        const float window = std::pow(std::cos(clamped * pi / lowpass_filter_width / 2.0F), 2.0F);
        filter[static_cast<size_t>(i)] = sinc(time_axis) * window * rolloff / static_cast<float>(ratio);
    }
    return filter;
}

DramaBoxVocoderConv1dWeights load_conv1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool use_bias) {
    DramaBoxVocoderConv1dWeights out;
    out.in_channels = in_channels;
    out.out_channels = out_channels;
    out.kernel = kernel;
    out.stride = stride;
    out.padding = padding;
    out.dilation = dilation;
    out.use_bias = use_bias;
    out.conv.weight = store.load_tensor(source, prefix + ".weight", storage_type, {out_channels, in_channels, kernel});
    if (use_bias) {
        out.conv.bias = store.load_tensor(source, prefix + ".bias", assets::TensorStorageType::F32, {out_channels});
    }
    return out;
}

DramaBoxVocoderConvTranspose1dWeights load_conv_transpose1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    bool use_bias) {
    DramaBoxVocoderConvTranspose1dWeights out;
    out.in_channels = in_channels;
    out.out_channels = out_channels;
    out.kernel = kernel;
    out.stride = stride;
    out.padding = padding;
    out.use_bias = use_bias;
    out.conv.weight = store.load_tensor(source, prefix + ".weight", assets::TensorStorageType::F32, {in_channels, out_channels, kernel});
    if (use_bias) {
        out.conv.bias = store.load_tensor(source, prefix + ".bias", assets::TensorStorageType::F32, {out_channels});
    }
    return out;
}

DramaBoxVocoderActivationWeights load_activation(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    const auto alpha_raw = source.require_f32(prefix + ".act.alpha", {channels});
    const auto beta_raw = source.require_f32(prefix + ".act.beta", {channels});
    std::vector<float> alpha(static_cast<size_t>(channels), 0.0F);
    std::vector<float> inv_beta(static_cast<size_t>(channels), 0.0F);
    for (int64_t index = 0; index < channels; ++index) {
        alpha[static_cast<size_t>(index)] = std::exp(alpha_raw[static_cast<size_t>(index)]);
        inv_beta[static_cast<size_t>(index)] = 1.0F / (std::exp(beta_raw[static_cast<size_t>(index)]) + 1.0e-9F);
    }
    DramaBoxVocoderActivationWeights out;
    out.alpha = store.make_from_f32(core::TensorShape::from_dims({channels}), assets::TensorStorageType::F32, std::move(alpha));
    out.inv_beta = store.make_from_f32(core::TensorShape::from_dims({channels}), assets::TensorStorageType::F32, std::move(inv_beta));
    out.up_filter = store.make_from_f32(
        core::TensorShape::from_dims({channels, 1, 1, kActivationKernel}),
        assets::TensorStorageType::F32,
        expand_activation_filter(reversed_filter(source.require_f32(prefix + ".upsample.filter", {1, 1, kActivationKernel})), channels));
    out.down_filter = store.make_from_f32(
        core::TensorShape::from_dims({channels, 1, 1, kActivationKernel}),
        assets::TensorStorageType::F32,
        expand_activation_filter(source.require_f32(prefix + ".downsample.lowpass.filter", {1, 1, kActivationKernel}), channels));
    return out;
}

DramaBoxVocoderResBlockWeights load_resblock(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t channels,
    int64_t kernel) {
    DramaBoxVocoderResBlockWeights out;
    out.convs1.reserve(3);
    out.convs2.reserve(3);
    out.activations.reserve(6);
    const int dilations[3] = {1, 3, 5};
    for (int layer = 0; layer < 3; ++layer) {
        const int dilation = dilations[layer];
        out.convs1.push_back(load_conv1d(
            store,
            source,
            prefix + ".convs1." + std::to_string(layer),
            storage_type,
            channels,
            channels,
            kernel,
            1,
            static_cast<int>((kernel * dilation - dilation) / 2),
            dilation,
            true));
    }
    for (int layer = 0; layer < 3; ++layer) {
        out.convs2.push_back(load_conv1d(
            store,
            source,
            prefix + ".convs2." + std::to_string(layer),
            storage_type,
            channels,
            channels,
            kernel,
            1,
            static_cast<int>((kernel - 1) / 2),
            1,
            true));
    }
    for (int activation = 0; activation < 6; ++activation) {
        const char * group = activation < 3 ? "acts1" : "acts2";
        const int layer = activation < 3 ? activation : activation - 3;
        out.activations.push_back(load_activation(
            store,
            source,
            prefix + "." + group + "." + std::to_string(layer),
            channels));
    }
    return out;
}

DramaBoxBigVganWeights load_bigvgan(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t sample_rate,
    int64_t initial_channel,
    const std::vector<int64_t> & upsample_rates,
    const std::vector<int64_t> & upsample_kernel_sizes) {
    const std::vector<int64_t> resblock_kernels = {3, 7, 11};
    DramaBoxBigVganWeights weights;
    weights.sample_rate = sample_rate;
    weights.num_mels = 128;
    weights.initial_channel = initial_channel;
    weights.upsample_rates = upsample_rates;
    weights.upsample_kernel_sizes = upsample_kernel_sizes;
    weights.conv_pre = load_conv1d(store, source, prefix + ".conv_pre", storage_type, initial_channel, 128, 7, 1, 3, 1, true);
    const int64_t upsample_count = static_cast<int64_t>(upsample_rates.size());
    weights.ups.reserve(static_cast<size_t>(upsample_count));
    weights.resblocks.reserve(static_cast<size_t>(upsample_count * kNumResblockKernels));
    for (int64_t up_index = 0; up_index < upsample_count; ++up_index) {
        const int64_t in_channels = initial_channel / (int64_t{1} << up_index);
        const int64_t out_channels = initial_channel / (int64_t{1} << (up_index + 1));
        weights.ups.push_back(load_conv_transpose1d(
            store,
            source,
            prefix + ".ups." + std::to_string(up_index),
            in_channels,
            out_channels,
            upsample_kernel_sizes[static_cast<size_t>(up_index)],
            upsample_rates[static_cast<size_t>(up_index)],
            (upsample_kernel_sizes[static_cast<size_t>(up_index)] - upsample_rates[static_cast<size_t>(up_index)]) / 2,
            true));
        for (int64_t kernel_index = 0; kernel_index < kNumResblockKernels; ++kernel_index) {
            weights.resblocks.push_back(load_resblock(
                store,
                source,
                prefix + ".resblocks." + std::to_string(up_index * kNumResblockKernels + kernel_index),
                storage_type,
                out_channels,
                resblock_kernels[static_cast<size_t>(kernel_index)]));
        }
    }
    const int64_t post_channels = initial_channel / (int64_t{1} << upsample_count);
    weights.activation_post = load_activation(store, source, prefix + ".act_post", post_channels);
    weights.conv_post = load_conv1d(store, source, prefix + ".conv_post", storage_type, 2, post_channels, 7, 1, 3, 1, false);
    return weights;
}

ggml_tensor * repeat_frame(ggml_context * ctx, ggml_tensor * x, int64_t frame, int64_t count) {
    ggml_tensor * view = ggml_view_2d(ctx, x, 1, x->ne[1], x->nb[1], static_cast<size_t>(frame) * x->nb[0]);
    return ggml_repeat_4d(ctx, view, count, x->ne[1], 1, 1);
}

ggml_tensor * replicate_pad(ggml_context * ctx, ggml_tensor * x, int64_t left, int64_t right) {
    ggml_tensor * out = x;
    if (left > 0) {
        out = ggml_concat(ctx, repeat_frame(ctx, x, 0, left), out, 0);
    }
    if (right > 0) {
        out = ggml_concat(ctx, out, repeat_frame(ctx, x, x->ne[0] - 1, right), 0);
    }
    return out;
}

ggml_tensor * zero_pad_right(ggml_context * ctx, ggml_tensor * x, int64_t right) {
    if (right <= 0) {
        return x;
    }
    ggml_tensor * zero = ggml_scale(ctx, repeat_frame(ctx, x, x->ne[0] - 1, right), 0.0F);
    return ggml_concat(ctx, x, zero, 0);
}

ggml_tensor * conv1d(
    ggml_context * ctx,
    core::BackendType backend_type,
    ggml_tensor * x,
    const DramaBoxVocoderConv1dWeights & conv,
    const std::string & name) {
    core::ModuleBuildContext build_ctx{ctx, name.c_str(), backend_type};
    const auto input = core::wrap_tensor(
        ggml_reshape_3d(ctx, core::has_backend_addressable_layout(x) ? x : ggml_cont(ctx, x), x->ne[0], x->ne[1], 1),
        core::TensorShape::from_dims({1, conv.in_channels, x->ne[0]}),
        GGML_TYPE_F32);
    const modules::FastConv1dModule module(
        {
            conv.in_channels,
            conv.out_channels,
            conv.kernel,
            static_cast<int>(conv.stride),
            static_cast<int>(conv.padding),
            static_cast<int>(conv.dilation),
            conv.use_bias,
        },
        modules::FastConv1dKind::MinittsFast1dIm2col);
    const auto output = module.build(build_ctx, input, conv.conv);
    return ggml_set_name(ggml_reshape_2d(ctx, output.tensor, output.shape.dims[2], output.shape.dims[1]), name.c_str());
}

ggml_tensor * conv_transpose1d(
    ggml_context * ctx,
    core::BackendType backend_type,
    ggml_tensor * x,
    const DramaBoxVocoderConvTranspose1dWeights & conv,
    const std::string & name) {
    core::ModuleBuildContext build_ctx{ctx, name.c_str(), backend_type};
    auto * input = ggml_reshape_3d(ctx, core::has_backend_addressable_layout(x) ? x : ggml_cont(ctx, x), x->ne[0], x->ne[1], 1);
    const auto input_value =
        core::wrap_tensor(input, core::TensorShape::from_dims({1, conv.in_channels, x->ne[0]}), GGML_TYPE_F32);
    const auto module = modules::ConvTranspose1dModule({
        conv.in_channels,
        conv.out_channels,
        conv.kernel,
        static_cast<int>(conv.stride),
        static_cast<int>(conv.padding),
        1,
        conv.use_bias});
    const auto output = module.build(build_ctx, input_value, conv.conv);
    return ggml_set_name(ggml_reshape_2d(ctx, output.tensor, output.shape.dims[2], output.shape.dims[1]), name.c_str());
}

ggml_tensor * depthwise_conv_transpose_filter(ggml_context * ctx, ggml_tensor * x, const core::TensorValue & filter, int64_t stride) {
    ggml_tensor * x3 = ggml_reshape_3d(ctx, core::has_backend_addressable_layout(x) ? x : ggml_cont(ctx, x), 1, x->ne[0], x->ne[1]);
    ggml_tensor * zero3 = ggml_scale(ctx, x3, 0.0F);
    ggml_tensor * interleaved3 = x3;
    for (int64_t index = 1; index < stride; ++index) {
        interleaved3 = ggml_concat(ctx, interleaved3, zero3, 0);
    }
    ggml_tensor * interleaved = ggml_reshape_2d(ctx, interleaved3, x->ne[0] * stride, x->ne[1]);
    interleaved = ggml_view_2d(ctx, interleaved, x->ne[0] * stride - (stride - 1), interleaved->ne[1], interleaved->nb[1], 0);
    ggml_tensor * interleaved_contiguous = core::has_backend_addressable_layout(interleaved) ? interleaved : ggml_cont(ctx, interleaved);
    ggml_tensor * input4 = ggml_reshape_4d(ctx, interleaved_contiguous, interleaved->ne[0], 1, interleaved->ne[1], 1);
    ggml_tensor * y4 = ggml_conv_2d_dw_direct(ctx, filter.tensor, input4, 1, 1, static_cast<int>(filter.shape.dims[3] - 1), 0, 1, 1);
    y4 = core::has_backend_addressable_layout(y4) ? y4 : ggml_cont(ctx, y4);
    return ggml_reshape_2d(ctx, y4, y4->ne[0], y4->ne[2]);
}

ggml_tensor * depthwise_conv_filter(ggml_context * ctx, ggml_tensor * x, const core::TensorValue & filter, int64_t stride) {
    ggml_tensor * x4 = ggml_reshape_4d(ctx, core::has_backend_addressable_layout(x) ? x : ggml_cont(ctx, x), x->ne[0], 1, x->ne[1], 1);
    ggml_tensor * y4 = ggml_conv_2d_dw_direct(ctx, filter.tensor, x4, static_cast<int>(stride), 1, 0, 0, 1, 1);
    y4 = core::has_backend_addressable_layout(y4) ? y4 : ggml_cont(ctx, y4);
    return ggml_reshape_2d(ctx, y4, y4->ne[0], y4->ne[2]);
}

ggml_tensor * activation1d(ggml_context * ctx, ggml_tensor * x, const DramaBoxVocoderActivationWeights & weights, const std::string & name) {
    const int64_t pad = kActivationKernel / kActivationRatio - 1;
    const int64_t pad_left = pad * kActivationRatio + (kActivationKernel - kActivationRatio) / 2;
    const int64_t pad_right = pad * kActivationRatio + (kActivationKernel - kActivationRatio + 1) / 2;
    ggml_tensor * up = replicate_pad(ctx, x, pad, pad);
    up = depthwise_conv_transpose_filter(ctx, up, weights.up_filter, kActivationRatio);
    up = ggml_scale(ctx, up, static_cast<float>(kActivationRatio));
    up = ggml_view_2d(
        ctx,
        up,
        up->ne[0] - pad_left - pad_right,
        up->ne[1],
        up->nb[1],
        static_cast<size_t>(pad_left) * up->nb[0]);
    ggml_tensor * alpha = ggml_set_name(ggml_reshape_2d(ctx, weights.alpha.tensor, 1, x->ne[1]), (name + ".alpha").c_str());
    ggml_tensor * inv_beta = ggml_set_name(ggml_reshape_2d(ctx, weights.inv_beta.tensor, 1, x->ne[1]), (name + ".inv_beta").c_str());
    ggml_tensor * sinusoid = ggml_sin(ctx, ggml_mul(ctx, up, alpha));
    sinusoid = core::has_backend_addressable_layout(sinusoid) ? sinusoid : ggml_cont(ctx, sinusoid);
    ggml_tensor * periodic = ggml_sqr(ctx, sinusoid);
    ggml_tensor * activated = ggml_add(ctx, up, ggml_mul(ctx, periodic, inv_beta));
    const int64_t lowpass_left = kActivationKernel / 2 - 1;
    const int64_t lowpass_right = kActivationKernel / 2;
    ggml_tensor * down = replicate_pad(ctx, activated, lowpass_left, lowpass_right);
    down = depthwise_conv_filter(ctx, down, weights.down_filter, kActivationRatio);
    return ggml_set_name(down, name.c_str());
}

ggml_tensor * amp_block(
    ggml_context * ctx,
    core::BackendType backend_type,
    ggml_tensor * x,
    const DramaBoxVocoderResBlockWeights & block,
    const std::string & name) {
    for (int layer = 0; layer < 3; ++layer) {
        ggml_tensor * xt = activation1d(ctx, x, block.activations[static_cast<size_t>(layer)], name + ".act1." + std::to_string(layer));
        xt = conv1d(ctx, backend_type, xt, block.convs1[static_cast<size_t>(layer)], name + ".convs1." + std::to_string(layer));
        xt = activation1d(ctx, xt, block.activations[static_cast<size_t>(layer + 3)], name + ".act2." + std::to_string(layer));
        xt = conv1d(ctx, backend_type, xt, block.convs2[static_cast<size_t>(layer)], name + ".convs2." + std::to_string(layer));
        x = ggml_set_name(ggml_add(ctx, x, xt), (name + ".residual." + std::to_string(layer)).c_str());
    }
    return x;
}

ggml_tensor * build_bigvgan(
    ggml_context * ctx,
    core::BackendType backend_type,
    const DramaBoxBigVganWeights & weights,
    ggml_tensor * mel,
    bool apply_final_activation,
    bool use_tanh_at_final,
    ggml_tensor ** conv_pre_output = nullptr,
    ggml_tensor ** upsample0_output = nullptr) {
    ggml_tensor * x = conv1d(ctx, backend_type, mel, weights.conv_pre, "conv_pre");
    if (conv_pre_output != nullptr) {
        *conv_pre_output = x;
    }
    for (size_t up_index = 0; up_index < weights.ups.size(); ++up_index) {
        x = conv_transpose1d(ctx, backend_type, x, weights.ups[up_index], "ups." + std::to_string(up_index));
        ggml_tensor * sum = nullptr;
        for (int64_t kernel_index = 0; kernel_index < kNumResblockKernels; ++kernel_index) {
            const size_t block_index = up_index * static_cast<size_t>(kNumResblockKernels) + static_cast<size_t>(kernel_index);
            ggml_tensor * block = amp_block(ctx, backend_type, x, weights.resblocks[block_index], "resblocks." + std::to_string(block_index));
            sum = sum == nullptr ? block : ggml_add(ctx, sum, block);
        }
        x = ggml_set_name(
            ggml_scale(ctx, sum, 1.0F / static_cast<float>(kNumResblockKernels)),
            ("upsample." + std::to_string(up_index)).c_str());
        if (up_index == 0 && upsample0_output != nullptr) {
            *upsample0_output = x;
        }
    }
    x = activation1d(ctx, x, weights.activation_post, "activation_post");
    x = conv1d(ctx, backend_type, x, weights.conv_post, "conv_post");
    if (!apply_final_activation) {
        return ggml_set_name(x, "waveform");
    }
    return ggml_set_name(use_tanh_at_final ? ggml_tanh(ctx, x) : ggml_clamp(ctx, x, -1.0F, 1.0F), "waveform");
}

ggml_tensor * build_stereo_mel_from_wave(
    ggml_context * ctx,
    ggml_tensor * low_wave,
    const DramaBoxVocoderWeights & weights,
    int64_t padded_samples) {
    ggml_tensor * left = ggml_scale(ctx, repeat_frame(ctx, low_wave, 0, 432), 0.0F);
    ggml_tensor * x = ggml_concat(ctx, left, low_wave, 0);
    x = zero_pad_right(ctx, x, padded_samples - low_wave->ne[0]);
    ggml_tensor * x3 = ggml_reshape_3d(ctx, core::has_backend_addressable_layout(x) ? x : ggml_cont(ctx, x), x->ne[0], 1, x->ne[1]);
    ggml_tensor * spec = ggml_conv_1d_fast_1d_im2col(ctx, weights.stft_forward_basis.tensor, x3, 80, 0, 1);
    ggml_tensor * real = ggml_view_3d(ctx, spec, spec->ne[0], 257, spec->ne[2], spec->nb[1], spec->nb[2], 0);
    ggml_tensor * imag = ggml_view_3d(ctx, spec, spec->ne[0], 257, spec->ne[2], spec->nb[1], spec->nb[2], 257 * spec->nb[1]);
    ggml_tensor * mag = ggml_sqrt(
        ctx,
        ggml_add(
            ctx,
            ggml_sqr(ctx, core::has_backend_addressable_layout(real) ? real : ggml_cont(ctx, real)),
            ggml_sqr(ctx, core::has_backend_addressable_layout(imag) ? imag : ggml_cont(ctx, imag))));
    ggml_tensor * mag_for_mel = ggml_cont(ctx, ggml_transpose(ctx, mag));
    ggml_tensor * mel = ggml_mul_mat(ctx, weights.mel_basis.tensor, mag_for_mel);
    mel = ggml_cont(ctx, ggml_transpose(ctx, mel));
    mel = ggml_clamp(ctx, mel, 1.0e-5F, INFINITY);
    mel = ggml_log(ctx, mel);
    mel = core::has_backend_addressable_layout(mel) ? mel : ggml_cont(ctx, mel);
    return ggml_reshape_2d(ctx, mel, mel->ne[0], mel->ne[1] * mel->ne[2]);
}

ggml_tensor * build_resampler_skip(
    ggml_context * ctx,
    ggml_tensor * low_wave,
    const DramaBoxVocoderWeights & weights,
    int64_t padded_samples) {
    const int64_t ratio = 3;
    const int64_t width = 7;
    const int64_t kernel_size = 2 * width * ratio + 1;
    const int64_t pad_left = 2 * width * ratio;
    const int64_t pad_right = kernel_size - ratio;
    ggml_tensor * x = zero_pad_right(ctx, low_wave, padded_samples - low_wave->ne[0]);
    x = replicate_pad(ctx, x, width, width);
    ggml_tensor * up = depthwise_conv_transpose_filter(ctx, x, weights.resampler_filter, ratio);
    up = ggml_scale(ctx, up, static_cast<float>(ratio));
    return ggml_view_2d(
        ctx,
        up,
        up->ne[0] - pad_left - pad_right,
        up->ne[1],
        up->nb[1],
        static_cast<size_t>(pad_left) * up->nb[0]);
}

}  // namespace

DramaBoxVocoderWeights load_dramabox_vocoder_weights(
    const DramaBoxAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type) {
    const auto & source = *assets.audio_weights;
    (void)weight_storage_type;
    constexpr auto vocoder_storage_type = assets::TensorStorageType::F32;
    DramaBoxVocoderWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "dramabox.vocoder.weights",
        weight_context_bytes == 0 ? kVocoderWeightContextBytes : weight_context_bytes);
    weights.vocoder = load_bigvgan(
        *weights.store,
        source,
        "vocoder.vocoder",
        vocoder_storage_type,
        assets.config.vocoder.input_sample_rate,
        1536,
        {5, 2, 2, 2, 2, 2},
        {11, 4, 4, 4, 4, 4});
    weights.bwe = load_bigvgan(
        *weights.store,
        source,
        "vocoder.bwe_generator",
        vocoder_storage_type,
        assets.config.vocoder.output_sample_rate,
        assets.config.vocoder.bwe_initial_channel,
        assets.config.vocoder.bwe_upsample_rates,
        assets.config.vocoder.bwe_upsample_kernel_sizes);
    weights.mel_basis = weights.store->load_tensor(
        source,
        "vocoder.mel_stft.mel_basis",
        assets::TensorStorageType::F32,
        {assets.config.vocoder.num_mels, assets.config.vocoder.n_fft / 2 + 1});
    weights.stft_forward_basis = weights.store->load_tensor(
        source,
        "vocoder.mel_stft.stft_fn.forward_basis",
        assets::TensorStorageType::F32,
        {assets.config.vocoder.n_fft + 2, 1, assets.config.vocoder.n_fft});
    weights.resampler_filter = weights.store->make_from_f32(
        core::TensorShape::from_dims({2, 1, 1, 43}),
        assets::TensorStorageType::F32,
        expand_channel_filter(make_hann_resampler_filter(3), 2));
    weights.store->upload();
    return weights;
}

class DramaBoxVocoderRuntime::VocoderGraph {
public:
    VocoderGraph(core::ExecutionContext & execution, const DramaBoxVocoderWeights & weights, int64_t mel_frames)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          weights_(weights),
          mel_frames_(mel_frames) {
        build();
    }

    ~VocoderGraph() {
        if (backend_ != nullptr && graph_ != nullptr) {
            core::release_backend_graph_resources(backend_type_, backend_, graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(int64_t mel_frames) const noexcept {
        return mel_frames == mel_frames_;
    }

    void run(const std::vector<float> & mel) const {
        const auto input_start = Clock::now();
        core::write_tensor_f32(mel_value_, mel);
        core::set_backend_threads(backend_, threads_);
        debug::timing_log_scalar("dramabox.vocoder16.input_upload_ms", debug::elapsed_ms(input_start, Clock::now()));
        compute(nullptr);
    }

    void run_from_device(const ggml_tensor * mel) const {
        if (mel == nullptr ||
            mel->type != input_->type ||
            mel->ne[0] != input_->ne[0] ||
            mel->ne[1] != input_->ne[1] ||
            mel->ne[2] != input_->ne[2] ||
            mel->ne[3] != input_->ne[3]) {
            throw std::runtime_error("DramaBox vocoder device mel shape mismatch");
        }
        const auto input_start = Clock::now();
        ggml_backend_tensor_copy(mel, input_);
        core::set_backend_threads(backend_, threads_);
        debug::timing_log_scalar("dramabox.vocoder16.device_input_copy_ms", debug::elapsed_ms(input_start, Clock::now()));
        compute(nullptr);
    }

    void compute(std::vector<float> * out) const {
        const auto compute_start = Clock::now();
        if (core::compute_backend_graph(backend_, graph_, nullptr, "dramabox.vocoder16") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DramaBox vocoder graph compute failed");
        }
        ggml_backend_synchronize(backend_);
        debug::timing_log_scalar("dramabox.vocoder16.graph.compute_ms", debug::elapsed_ms(compute_start, Clock::now()));
        if (out != nullptr) {
            const auto read_start = Clock::now();
            core::read_tensor_f32_into(output_, *out);
            debug::timing_log_scalar("dramabox.vocoder16.output_read_ms", debug::elapsed_ms(read_start, Clock::now()));
        }
    }

    int64_t samples() const noexcept {
        return output_->ne[0];
    }

    const ggml_tensor * output_tensor() const noexcept {
        return output_;
    }

private:
    void build() {
        const auto build_start = Clock::now();
        ggml_init_params params{kVocoderGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("DramaBox vocoder ggml context initialization failed");
        }
        input_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F32, 64, mel_frames_, 2, 1);
        mel_value_ = core::wrap_tensor(input_, core::TensorShape::from_dims({1, 2, mel_frames_, 64}), GGML_TYPE_F32);
        ggml_set_input(input_);
        mel_ = ggml_reshape_2d(ctx_.get(), ggml_cont(ctx_.get(), ggml_permute(ctx_.get(), input_, 1, 0, 2, 3)), mel_frames_, 128);
        output_ = build_bigvgan(ctx_.get(), backend_type_, weights_.vocoder, mel_, true, false);
        output_ = core::has_backend_addressable_layout(output_) ? output_ : ggml_cont(ctx_.get(), output_);
        ggml_set_output(output_);
        const auto expand_start = Clock::now();
        graph_ = ggml_new_graph_custom(ctx_.get(), kVocoderGraphNodeCapacity, false);
        ggml_build_forward_expand(graph_, output_);
        debug::timing_log_scalar("dramabox.vocoder16.graph.expand_ms", debug::elapsed_ms(expand_start, Clock::now()));
        debug::timing_log_scalar("dramabox.vocoder16.graph.nodes", static_cast<double>(ggml_graph_n_nodes(graph_)));
        const auto alloc_start = Clock::now();
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("DramaBox vocoder backend buffer allocation failed");
        }
        debug::timing_log_scalar("dramabox.vocoder16.graph.alloc_ms", debug::elapsed_ms(alloc_start, Clock::now()));
        debug::timing_log_scalar("dramabox.vocoder16.graph.build_ms", debug::elapsed_ms(build_start, Clock::now()));
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    const DramaBoxVocoderWeights & weights_;
    int64_t mel_frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * mel_ = nullptr;
    core::TensorValue mel_value_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

class DramaBoxVocoderRuntime::BweGraph {
public:
    BweGraph(core::ExecutionContext & execution, const DramaBoxVocoderWeights & weights, int64_t low_samples)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          weights_(weights),
          low_samples_(low_samples) {
        build();
    }

    ~BweGraph() {
        if (backend_ != nullptr && graph_ != nullptr) {
            core::release_backend_graph_resources(backend_type_, backend_, graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(int64_t low_samples) const noexcept {
        return low_samples == low_samples_;
    }

    void run(const std::vector<float> & low_wave, std::vector<float> & out) const {
        const auto input_start = Clock::now();
        core::write_tensor_f32(input_value_, low_wave);
        core::set_backend_threads(backend_, threads_);
        debug::timing_log_scalar("dramabox.bwe.input_upload_ms", debug::elapsed_ms(input_start, Clock::now()));
        compute(out);
    }

    void run_from_device(const ggml_tensor * low_wave, std::vector<float> & out) const {
        if (low_wave == nullptr ||
            low_wave->type != input_->type ||
            low_wave->ne[0] != input_->ne[0] ||
            low_wave->ne[1] != input_->ne[1] ||
            low_wave->ne[2] != input_->ne[2] ||
            low_wave->ne[3] != input_->ne[3]) {
            throw std::runtime_error("DramaBox BWE device waveform shape mismatch");
        }
        const auto input_start = Clock::now();
        ggml_backend_tensor_copy(low_wave, input_);
        core::set_backend_threads(backend_, threads_);
        debug::timing_log_scalar("dramabox.bwe.device_input_copy_ms", debug::elapsed_ms(input_start, Clock::now()));
        compute(out);
    }

    void compute(std::vector<float> & out) const {
        const auto compute_start = Clock::now();
        if (core::compute_backend_graph(backend_, graph_, nullptr, "dramabox.bwe") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DramaBox BWE graph compute failed");
        }
        ggml_backend_synchronize(backend_);
        debug::timing_log_scalar("dramabox.bwe.graph.compute_ms", debug::elapsed_ms(compute_start, Clock::now()));
        const auto read_start = Clock::now();
        core::read_tensor_f32_into(output_, out);
        debug::timing_log_scalar("dramabox.bwe.output_read_ms", debug::elapsed_ms(read_start, Clock::now()));
    }

private:
    void build() {
        const auto build_start = Clock::now();
        padded_samples_ = low_samples_;
        const int64_t remainder = padded_samples_ % 80;
        if (remainder != 0) {
            padded_samples_ += 80 - remainder;
        }
        output_samples_ = low_samples_ * 3;
        ggml_init_params params{kVocoderGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("DramaBox BWE ggml context initialization failed");
        }
        input_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_F32, low_samples_, 2);
        input_value_ = core::wrap_tensor(input_, core::TensorShape::from_dims({2, low_samples_}), GGML_TYPE_F32);
        ggml_set_input(input_);
        ggml_tensor * bwe_mel = build_stereo_mel_from_wave(ctx_.get(), input_, weights_, padded_samples_);
        ggml_tensor * residual = build_bigvgan(ctx_.get(), backend_type_, weights_.bwe, bwe_mel, false, false);
        ggml_tensor * skip = build_resampler_skip(ctx_.get(), input_, weights_, padded_samples_);
        ggml_tensor * summed = ggml_clamp(ctx_.get(), ggml_add(ctx_.get(), residual, skip), -1.0F, 1.0F);
        output_ = ggml_view_2d(ctx_.get(), summed, output_samples_, summed->ne[1], summed->nb[1], 0);
        output_ = core::has_backend_addressable_layout(output_) ? output_ : ggml_cont(ctx_.get(), output_);
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), kVocoderGraphNodeCapacity, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("DramaBox BWE backend buffer allocation failed");
        }
        debug::timing_log_scalar("dramabox.bwe.graph.build_ms", debug::elapsed_ms(build_start, Clock::now()));
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    const DramaBoxVocoderWeights & weights_;
    int64_t low_samples_ = 0;
    int64_t padded_samples_ = 0;
    int64_t output_samples_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    core::TensorValue input_value_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

DramaBoxVocoderRuntime::DramaBoxVocoderRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const DramaBoxAssets> assets,
    assets::TensorStorageType weight_storage_type)
    : execution_(&execution),
      assets_(std::move(assets)),
      weight_storage_type_(weight_storage_type) {
    if (execution_ == nullptr) {
        throw std::runtime_error("DramaBox vocoder runtime requires execution context");
    }
    if (assets_ == nullptr) {
        throw std::runtime_error("DramaBox vocoder runtime requires assets");
    }
}

DramaBoxVocoderRuntime::~DramaBoxVocoderRuntime() = default;

void DramaBoxVocoderRuntime::prepare(int64_t mel_frames) const {
    if (!weights_) {
        weights_ = std::make_unique<DramaBoxVocoderWeights>(load_dramabox_vocoder_weights(
            *assets_,
            execution_->backend(),
            execution_->backend_type(),
            0,
            weight_storage_type_));
    }
    if (!vocoder_graph_ || !vocoder_graph_->matches(mel_frames)) {
        vocoder_graph_.reset();
        vocoder_graph_ = std::make_unique<VocoderGraph>(*execution_, *weights_, mel_frames);
    }
    const int64_t low_samples = vocoder_graph_->samples();
    if (!bwe_graph_ || !bwe_graph_->matches(low_samples)) {
        bwe_graph_.reset();
        bwe_graph_ = std::make_unique<BweGraph>(*execution_, *weights_, low_samples);
    }
}

DramaBoxVocoderOutput DramaBoxVocoderRuntime::synthesize(const DramaBoxDecodedMel & mel) const {
    if (mel.batch != 1 || mel.channels != 2 || mel.mel_bins != 64 ||
        (mel.device_values == nullptr &&
         static_cast<int64_t>(mel.values.size()) != mel.batch * mel.channels * mel.frames * mel.mel_bins)) {
        throw std::runtime_error("DramaBox vocoder received invalid mel dimensions");
    }
    prepare(mel.frames);
    if (mel.device_values != nullptr) {
        vocoder_graph_->run_from_device(mel.device_values);
    } else {
        vocoder_graph_->run(mel.values);
    }
    DramaBoxVocoderOutput out;
    bwe_graph_->run_from_device(vocoder_graph_->output_tensor(), out.waveform);
    out.channels = 2;
    out.samples = static_cast<int64_t>(out.waveform.size()) / 2;
    out.sample_rate = assets_->config.vocoder.output_sample_rate;
    return out;
}

}  // namespace engine::models::dramabox
