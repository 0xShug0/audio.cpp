#include "engine/framework/modules/bigvgan_vocoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::modules {
namespace {

using Conv1dWeights = BigVganVocoderWeights::Conv1dWeights;
using ConvTranspose1dWeights = BigVganVocoderWeights::ConvTranspose1dWeights;
using ActivationWeights = BigVganVocoderWeights::ActivationWeights;
using ResBlockWeights = BigVganVocoderWeights::ResBlockWeights;

constexpr int64_t kBigVganNumKernels = 3;
constexpr int64_t kBigVganActivationRatio = 2;
constexpr int64_t kBigVganActivationKernel = 12;

int64_t tensor_elements(const std::vector<int64_t> & shape) {
    if (shape.empty()) {
        throw std::runtime_error("BigVGAN tensor shape is empty");
    }
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, [](int64_t lhs, int64_t rhs) {
        if (rhs <= 0) {
            throw std::runtime_error("BigVGAN tensor shape contains non-positive dimension");
        }
        return lhs * rhs;
    });
}

void validate_config(const BigVganVocoderConfig & config) {
    if (config.sampling_rate <= 0 || config.num_mels <= 0 || config.n_fft <= 0 ||
        config.hop_size <= 0 || config.win_size <= 0 || config.upsample_initial_channel <= 0) {
        throw std::runtime_error("BigVGAN config contains non-positive dimensions");
    }
    if (config.upsample_rates.empty() ||
        config.upsample_rates.size() != config.upsample_kernel_sizes.size()) {
        throw std::runtime_error("BigVGAN upsample config is inconsistent");
    }
    if (config.resblock_kernel_sizes.size() != kBigVganNumKernels) {
        throw std::runtime_error("BigVGAN currently requires the reference three-kernel AMPBlock1 config");
    }
    if (config.snake_logscale != true) {
        throw std::runtime_error("BigVGAN SeedVC checkpoints require snake_logscale=true");
    }
}

std::vector<float> fold_weight_norm(
    const std::vector<float> & weight_v,
    const std::vector<float> & weight_g,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2) {
    if (static_cast<int64_t>(weight_v.size()) != dim0 * dim1 * dim2 ||
        static_cast<int64_t>(weight_g.size()) != dim0) {
        throw std::runtime_error("BigVGAN weight-norm tensor shape mismatch");
    }
    std::vector<float> folded(weight_v.size(), 0.0F);
    for (int64_t d0 = 0; d0 < dim0; ++d0) {
        double norm_sq = 0.0;
        const size_t base = static_cast<size_t>(d0 * dim1 * dim2);
        for (int64_t index = 0; index < dim1 * dim2; ++index) {
            const float value = weight_v[base + static_cast<size_t>(index)];
            norm_sq += static_cast<double>(value) * static_cast<double>(value);
        }
        const float scale = weight_g[static_cast<size_t>(d0)] /
            static_cast<float>(std::sqrt(norm_sq));
        for (int64_t index = 0; index < dim1 * dim2; ++index) {
            folded[base + static_cast<size_t>(index)] =
                weight_v[base + static_cast<size_t>(index)] * scale;
        }
    }
    return folded;
}

std::vector<float> reversed_filter(std::vector<float> values) {
    std::reverse(values.begin(), values.end());
    return values;
}

std::vector<float> expand_activation_filter_2d(
    const std::vector<float> & filter,
    int64_t channels) {
    if (static_cast<int64_t>(filter.size()) != kBigVganActivationKernel || channels <= 0) {
        throw std::runtime_error("BigVGAN activation filter expansion shape mismatch");
    }
    std::vector<float> expanded(static_cast<size_t>(channels * kBigVganActivationKernel), 0.0F);
    for (int64_t channel = 0; channel < channels; ++channel) {
        std::copy(
            filter.begin(),
            filter.end(),
            expanded.begin() + static_cast<std::ptrdiff_t>(channel * kBigVganActivationKernel));
    }
    return expanded;
}

std::vector<float> transpose_conv1d_to_conv1d_weight(
    const std::vector<float> & weight,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel) {
    if (static_cast<int64_t>(weight.size()) != in_channels * out_channels * kernel) {
        throw std::runtime_error("BigVGAN transposed-conv weight shape mismatch");
    }
    std::vector<float> converted(static_cast<size_t>(out_channels * in_channels * kernel), 0.0F);
    for (int64_t in_channel = 0; in_channel < in_channels; ++in_channel) {
        for (int64_t out_channel = 0; out_channel < out_channels; ++out_channel) {
            for (int64_t tap = 0; tap < kernel; ++tap) {
                const size_t src = static_cast<size_t>((in_channel * out_channels + out_channel) * kernel + tap);
                const size_t dst =
                    static_cast<size_t>((out_channel * in_channels + in_channel) * kernel + (kernel - 1 - tap));
                converted[dst] = weight[src];
            }
        }
    }
    return converted;
}

Conv1dWeights load_weight_norm_conv1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool use_bias,
    assets::TensorStorageType storage_type) {
    (void) storage_type;
    const auto weight_v = source.require_f32(prefix + ".weight_v", {out_channels, in_channels, kernel});
    const auto weight_g = source.require_f32(prefix + ".weight_g", {out_channels, 1, 1});
    Conv1dWeights out;
    out.in_channels = in_channels;
    out.out_channels = out_channels;
    out.kernel = kernel;
    out.stride = stride;
    out.padding = padding;
    out.dilation = dilation;
    out.use_bias = use_bias;
    out.weight = store.make_from_f32(
        core::TensorShape::from_dims({out_channels, in_channels, kernel}),
        assets::TensorStorageType::F32,
        fold_weight_norm(weight_v, weight_g, out_channels, in_channels, kernel));
    if (use_bias) {
        out.bias = store.make_from_f32(
            core::TensorShape::from_dims({out_channels}),
            assets::TensorStorageType::F32,
            source.require_f32(prefix + ".bias", {out_channels}));
    }
    return out;
}

ConvTranspose1dWeights load_weight_norm_conv_transpose1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    assets::TensorStorageType storage_type) {
    (void) storage_type;
    const auto weight_v = source.require_f32(prefix + ".weight_v", {in_channels, out_channels, kernel});
    const auto weight_g = source.require_f32(prefix + ".weight_g", {in_channels, 1, 1});
    ConvTranspose1dWeights out;
    out.in_channels = in_channels;
    out.out_channels = out_channels;
    out.kernel = kernel;
    out.stride = stride;
    out.padding = padding;
    out.use_bias = true;
    out.conv1d_weight = store.make_from_f32(
        core::TensorShape::from_dims({out_channels, in_channels, kernel}),
        assets::TensorStorageType::F32,
        transpose_conv1d_to_conv1d_weight(
            fold_weight_norm(weight_v, weight_g, in_channels, out_channels, kernel),
            in_channels,
            out_channels,
            kernel));
    out.bias = store.make_from_f32(
        core::TensorShape::from_dims({out_channels}),
        assets::TensorStorageType::F32,
        source.require_f32(prefix + ".bias", {out_channels}));
    return out;
}

ActivationWeights load_activation(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    assets::TensorStorageType storage_type) {
    (void) storage_type;
    const auto alpha_raw = source.require_f32(prefix + ".act.alpha", {channels});
    const auto beta_raw = source.require_f32(prefix + ".act.beta", {channels});
    std::vector<float> alpha(static_cast<size_t>(channels), 0.0F);
    std::vector<float> inv_beta(static_cast<size_t>(channels), 0.0F);
    for (int64_t index = 0; index < channels; ++index) {
        alpha[static_cast<size_t>(index)] = std::exp(alpha_raw[static_cast<size_t>(index)]);
        inv_beta[static_cast<size_t>(index)] =
            1.0F / (std::exp(beta_raw[static_cast<size_t>(index)]) + 1.0e-9F);
    }
    ActivationWeights out;
    out.alpha = store.make_from_f32(core::TensorShape::from_dims({channels}), assets::TensorStorageType::F32, std::move(alpha));
    out.inv_beta = store.make_from_f32(core::TensorShape::from_dims({channels}), assets::TensorStorageType::F32, std::move(inv_beta));
    const auto up_filter = reversed_filter(
        source.require_f32(prefix + ".upsample.filter", {1, 1, kBigVganActivationKernel}));
    const auto down_filter =
        source.require_f32(prefix + ".downsample.lowpass.filter", {1, 1, kBigVganActivationKernel});
    out.up_filter = store.make_from_f32(
        core::TensorShape::from_dims({channels, 1, 1, kBigVganActivationKernel}),
        assets::TensorStorageType::F32,
        expand_activation_filter_2d(up_filter, channels));
    out.down_filter = store.make_from_f32(
        core::TensorShape::from_dims({channels, 1, 1, kBigVganActivationKernel}),
        assets::TensorStorageType::F32,
        expand_activation_filter_2d(down_filter, channels));
    return out;
}

ResBlockWeights load_resblock(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    int64_t kernel,
    assets::TensorStorageType storage_type) {
    (void) storage_type;
    ResBlockWeights out;
    out.convs1.reserve(3);
    out.convs2.reserve(3);
    out.activations.reserve(6);
    const int dilations[3] = {1, 3, 5};
    for (int layer = 0; layer < 3; ++layer) {
        const int dilation = dilations[layer];
        out.convs1.push_back(load_weight_norm_conv1d(
            store,
            source,
            prefix + ".convs1." + std::to_string(layer),
            channels,
            channels,
            kernel,
            1,
            static_cast<int>((kernel * dilation - dilation) / 2),
            dilation,
            true,
            storage_type));
    }
    for (int layer = 0; layer < 3; ++layer) {
        out.convs2.push_back(load_weight_norm_conv1d(
            store,
            source,
            prefix + ".convs2." + std::to_string(layer),
            channels,
            channels,
            kernel,
            1,
            static_cast<int>((kernel - 1) / 2),
            1,
            true,
            storage_type));
    }
    for (int activation = 0; activation < 6; ++activation) {
        out.activations.push_back(load_activation(
            store,
            source,
            prefix + ".activations." + std::to_string(activation),
            channels,
            storage_type));
    }
    return out;
}

ggml_tensor * named(ggml_tensor * tensor, const char * name) {
    return ggml_set_name(tensor, name);
}

ggml_tensor * contiguous_if_needed(ggml_context * ctx, ggml_tensor * tensor) {
    return core::has_backend_addressable_layout(tensor) ? tensor : ggml_cont(ctx, tensor);
}

ggml_tensor * weight_2d(
    ggml_context * ctx,
    const core::TensorValue & value,
    int64_t ne0,
    int64_t ne1,
    const std::string & name) {
    return named(ggml_reshape_2d(ctx, value.tensor, ne0, ne1), name.c_str());
}

ggml_tensor * weight_3d(
    ggml_context * ctx,
    const core::TensorValue & value,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    const std::string & name) {
    return named(ggml_reshape_3d(ctx, value.tensor, ne0, ne1, ne2), name.c_str());
}

ggml_tensor * slice_frames(
    ggml_context * ctx,
    ggml_tensor * x,
    int64_t start,
    int64_t length) {
    return ggml_view_2d(
        ctx,
        x,
        length,
        x->ne[1],
        x->nb[1],
        static_cast<size_t>(start) * x->nb[0]);
}

ggml_tensor * repeat_frame(
    ggml_context * ctx,
    ggml_tensor * x,
    int64_t frame,
    int64_t count) {
    ggml_tensor * view = slice_frames(ctx, x, frame, 1);
    return ggml_repeat_4d(ctx, view, count, x->ne[1], 1, 1);
}

ggml_tensor * replicate_pad(
    ggml_context * ctx,
    ggml_tensor * x,
    int64_t left,
    int64_t right) {
    ggml_tensor * out = x;
    if (left > 0) {
        out = ggml_concat(ctx, repeat_frame(ctx, x, 0, left), out, 0);
    }
    if (right > 0) {
        out = ggml_concat(ctx, out, repeat_frame(ctx, x, x->ne[0] - 1, right), 0);
    }
    return out;
}

ggml_tensor * conv1d(
    ggml_context * ctx,
    ggml_tensor * x,
    const Conv1dWeights & conv,
    const std::string & name) {
    ggml_tensor * weight = weight_3d(ctx, conv.weight, conv.kernel, conv.in_channels, conv.out_channels, name + ".weight");
    ggml_tensor * x3 = ggml_reshape_3d(ctx, contiguous_if_needed(ctx, x), x->ne[0], x->ne[1], 1);
    ggml_tensor * y3 = ggml_conv_1d_fast_1d_im2col(
        ctx,
        weight,
        x3,
        static_cast<int>(conv.stride),
        static_cast<int>(conv.padding),
        static_cast<int>(conv.dilation));
    ggml_tensor * y = ggml_reshape_2d(ctx, y3, y3->ne[0], y3->ne[1]);
    if (conv.use_bias) {
        y = ggml_add(ctx, y, weight_2d(ctx, *conv.bias, 1, conv.out_channels, name + ".bias"));
    }
    return named(y, name.c_str());
}

ggml_tensor * conv_transpose1d(
    ggml_context * ctx,
    ggml_tensor * x,
    const ConvTranspose1dWeights & conv,
    const std::string & name) {
    ggml_tensor * x3 = ggml_reshape_3d(ctx, contiguous_if_needed(ctx, x), 1, x->ne[0], x->ne[1]);
    ggml_tensor * zero3 = ggml_scale(ctx, x3, 0.0F);
    ggml_tensor * interleaved3 = x3;
    for (int64_t index = 1; index < conv.stride; ++index) {
        interleaved3 = ggml_concat(ctx, interleaved3, zero3, 0);
    }
    ggml_tensor * interleaved = ggml_reshape_2d(ctx, interleaved3, x->ne[0] * conv.stride, x->ne[1]);
    interleaved = slice_frames(ctx, interleaved, 0, x->ne[0] * conv.stride - (conv.stride - 1));

    ggml_tensor * weight = weight_3d(ctx, conv.conv1d_weight, conv.kernel, conv.in_channels, conv.out_channels, name + ".weight");
    ggml_tensor * input3 = ggml_reshape_3d(ctx, contiguous_if_needed(ctx, interleaved), interleaved->ne[0], interleaved->ne[1], 1);
    ggml_tensor * y3 = ggml_conv_1d_fast_1d_im2col(
        ctx,
        weight,
        input3,
        1,
        static_cast<int>(conv.kernel - 1 - conv.padding),
        1);
    ggml_tensor * y = ggml_reshape_2d(ctx, y3, y3->ne[0], y3->ne[1]);
    if (conv.use_bias) {
        y = ggml_add(ctx, y, weight_2d(ctx, *conv.bias, 1, conv.out_channels, name + ".bias"));
    }
    return named(y, name.c_str());
}

ggml_tensor * depthwise_conv_transpose_filter(
    ggml_context * ctx,
    ggml_tensor * x,
    const core::TensorValue & filter,
    int64_t stride) {
    if (stride != kBigVganActivationRatio) {
        throw std::runtime_error("BigVGAN activation upsample currently expects ratio 2");
    }
    ggml_tensor * x3 = ggml_reshape_3d(ctx, contiguous_if_needed(ctx, x), 1, x->ne[0], x->ne[1]);
    ggml_tensor * zero3 = ggml_scale(ctx, x3, 0.0F);
    ggml_tensor * pairs = ggml_concat(ctx, x3, zero3, 0);
    ggml_tensor * interleaved = ggml_reshape_2d(ctx, pairs, x->ne[0] * stride, x->ne[1]);
    interleaved = slice_frames(ctx, interleaved, 0, x->ne[0] * stride - (stride - 1));
    ggml_tensor * input4 = ggml_reshape_4d(ctx, contiguous_if_needed(ctx, interleaved), interleaved->ne[0], 1, interleaved->ne[1], 1);
    ggml_tensor * weight = filter.tensor;
    ggml_tensor * y4 = ggml_conv_2d_dw_direct(
        ctx,
        weight,
        input4,
        1,
        1,
        static_cast<int>(filter.shape.dims[3] - 1),
        0,
        1,
        1);
    y4 = contiguous_if_needed(ctx, y4);
    return ggml_reshape_2d(ctx, y4, y4->ne[0], y4->ne[2]);
}

ggml_tensor * depthwise_conv_filter(
    ggml_context * ctx,
    ggml_tensor * x,
    const core::TensorValue & filter,
    int64_t stride) {
    ggml_tensor * x4 = ggml_reshape_4d(ctx, contiguous_if_needed(ctx, x), x->ne[0], 1, x->ne[1], 1);
    ggml_tensor * weight = filter.tensor;
    ggml_tensor * y4 = ggml_conv_2d_dw_direct(
        ctx,
        weight,
        x4,
        static_cast<int>(stride),
        1,
        0,
        0,
        1,
        1);
    y4 = contiguous_if_needed(ctx, y4);
    return ggml_reshape_2d(ctx, y4, y4->ne[0], y4->ne[2]);
}

ggml_tensor * activation1d(
    ggml_context * ctx,
    ggml_tensor * x,
    const ActivationWeights & weights,
    const std::string & name) {
    const int64_t pad = kBigVganActivationKernel / kBigVganActivationRatio - 1;
    const int64_t pad_left =
        pad * kBigVganActivationRatio + (kBigVganActivationKernel - kBigVganActivationRatio) / 2;
    const int64_t pad_right =
        pad * kBigVganActivationRatio + (kBigVganActivationKernel - kBigVganActivationRatio + 1) / 2;
    ggml_tensor * up = replicate_pad(ctx, x, pad, pad);
    up = depthwise_conv_transpose_filter(ctx, up, weights.up_filter, kBigVganActivationRatio);
    up = ggml_scale(ctx, up, static_cast<float>(kBigVganActivationRatio));
    up = slice_frames(ctx, up, pad_left, up->ne[0] - pad_left - pad_right);

    ggml_tensor * alpha = weight_2d(ctx, weights.alpha, 1, x->ne[1], name + ".alpha");
    ggml_tensor * inv_beta = weight_2d(ctx, weights.inv_beta, 1, x->ne[1], name + ".inv_beta");
    ggml_tensor * periodic = ggml_sqr(ctx, ggml_sin(ctx, ggml_mul(ctx, up, alpha)));
    ggml_tensor * activated = ggml_add(ctx, up, ggml_mul(ctx, periodic, inv_beta));

    const int64_t lowpass_left = kBigVganActivationKernel / 2 - 1;
    const int64_t lowpass_right = kBigVganActivationKernel / 2;
    ggml_tensor * down = replicate_pad(ctx, activated, lowpass_left, lowpass_right);
    down = depthwise_conv_filter(ctx, down, weights.down_filter, kBigVganActivationRatio);
    return named(down, name.c_str());
}

ggml_tensor * amp_block(
    ggml_context * ctx,
    ggml_tensor * x,
    const ResBlockWeights & block,
    const std::string & name) {
    for (int layer = 0; layer < 3; ++layer) {
        ggml_tensor * xt = activation1d(ctx, x, block.activations[static_cast<size_t>(2 * layer)], name + ".act1." + std::to_string(layer));
        xt = conv1d(ctx, xt, block.convs1[static_cast<size_t>(layer)], name + ".convs1." + std::to_string(layer));
        xt = activation1d(ctx, xt, block.activations[static_cast<size_t>(2 * layer + 1)], name + ".act2." + std::to_string(layer));
        xt = conv1d(ctx, xt, block.convs2[static_cast<size_t>(layer)], name + ".convs2." + std::to_string(layer));
        x = named(ggml_add(ctx, x, xt), (name + ".residual." + std::to_string(layer)).c_str());
    }
    return x;
}

ggml_tensor * build_bigvgan_graph(
    ggml_context * ctx,
    const BigVganVocoderWeights & weights,
    ggml_tensor * mel) {
    ggml_tensor * x = conv1d(ctx, mel, weights.conv_pre, "conv_pre");
    for (size_t up_index = 0; up_index < weights.ups.size(); ++up_index) {
        x = conv_transpose1d(ctx, x, weights.ups[up_index], "ups." + std::to_string(up_index) + ".0");
        ggml_tensor * sum = nullptr;
        for (int64_t kernel_index = 0; kernel_index < kBigVganNumKernels; ++kernel_index) {
            const size_t block_index = up_index * static_cast<size_t>(kBigVganNumKernels) + static_cast<size_t>(kernel_index);
            ggml_tensor * block = amp_block(ctx, x, weights.resblocks[block_index], "resblocks." + std::to_string(block_index));
            sum = sum == nullptr ? block : ggml_add(ctx, sum, block);
        }
        x = named(ggml_scale(ctx, sum, 1.0F / static_cast<float>(kBigVganNumKernels)), ("upsample." + std::to_string(up_index)).c_str());
    }
    x = activation1d(ctx, x, weights.activation_post, "activation_post");
    x = conv1d(ctx, x, weights.conv_post, "conv_post");
    return named(ggml_clamp(ctx, x, -1.0F, 1.0F), "waveform");
}

class BigVganRunner {
public:
    BigVganRunner(const BigVganVocoderWeights & weights, const core::BackendConfig & backend)
        : weights_(weights), backend_(backend) {
        if (weights_.execution_context == nullptr) {
            throw std::runtime_error("BigVGAN runner requires execution context");
        }
    }

    ~BigVganRunner() {
        release_graph();
    }

    BigVganVocoderOutput run(const std::vector<float> & mel, int64_t frames) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (frames <= 0 || static_cast<int64_t>(mel.size()) != weights_.config.num_mels * frames) {
            throw std::runtime_error("BigVGAN mel size mismatch");
        }
        ensure_graph(frames);
        ggml_backend_tensor_set(mel_, mel.data(), 0, mel.size() * sizeof(float));
        if (engine::core::compute_backend_graph(weights_.execution_context->backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for BigVGAN");
        }
        BigVganVocoderOutput out;
        out.waveform = core::read_tensor_f32(output_);
        out.batch = 1;
        out.samples = output_->ne[0];
        out.sample_rate = weights_.config.sampling_rate;
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
        mel_ = {};
        output_ = nullptr;
        frames_ = 0;
    }

    void ensure_graph(int64_t frames) {
        if (ggml_ != nullptr && frames_ == frames) {
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
            throw std::runtime_error("failed to initialize BigVGAN graph context");
        }
        mel_ = named(ggml_new_tensor_2d(ggml_, GGML_TYPE_F32, frames, weights_.config.num_mels), "mel");
        output_ = build_bigvgan_graph(ggml_, weights_, mel_);
        graph_ = ggml_new_graph_custom(ggml_, 262144, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_.execution_context->backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate BigVGAN graph memory");
        }
        frames_ = frames;
    }

    const BigVganVocoderWeights & weights_;
    core::BackendConfig backend_;
    std::mutex mutex_;
    ggml_context * ggml_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * mel_ = nullptr;
    ggml_tensor * output_ = nullptr;
    int64_t frames_ = 0;
};

}  // namespace

struct BigVganVocoderComponent::State {
    std::unique_ptr<BigVganRunner> runner;
};

BigVganVocoderComponent BigVganVocoderComponent::load_from_safetensors(
    const std::filesystem::path & checkpoint_path,
    core::BackendConfig backend,
    BigVganVocoderConfig config) {
    return load_from_tensor_source(engine::assets::open_tensor_source(checkpoint_path), std::move(backend), std::move(config));
}

BigVganVocoderComponent BigVganVocoderComponent::load_from_tensor_source(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    BigVganVocoderConfig config) {
    validate_config(config);
    if (source == nullptr) {
        throw std::runtime_error("BigVGAN component requires tensor source");
    }
    auto weights = std::make_shared<BigVganVocoderWeights>();
    weights->config = std::move(config);
    weights->source_path = source->source_path();
    weights->execution_context = std::make_shared<core::ExecutionContext>(backend);
    weights->store = std::make_shared<core::BackendWeightStore>(
        weights->execution_context->backend(),
        weights->execution_context->backend_type(),
        "framework.bigvgan.weights",
        1024ull * 1024ull * 1024ull);

    for (const auto & tensor : source->tensors()) {
        weights->parameter_count += tensor_elements(tensor.shape);
        ++weights->loaded_tensor_count;
    }

    weights->conv_pre = load_weight_norm_conv1d(
        *weights->store,
        *source,
        "conv_pre",
        weights->config.upsample_initial_channel,
        weights->config.num_mels,
        7,
        1,
        3,
        1,
        true,
        weights->config.weight_storage_type);

    const int64_t upsample_count = static_cast<int64_t>(weights->config.upsample_rates.size());
    weights->ups.reserve(static_cast<size_t>(upsample_count));
    weights->resblocks.reserve(static_cast<size_t>(upsample_count * kBigVganNumKernels));
    for (int64_t up_index = 0; up_index < upsample_count; ++up_index) {
        const int64_t in_channels = weights->config.upsample_initial_channel / (int64_t{1} << up_index);
        const int64_t out_channels = weights->config.upsample_initial_channel / (int64_t{1} << (up_index + 1));
        const int64_t kernel = weights->config.upsample_kernel_sizes[static_cast<size_t>(up_index)];
        const int64_t stride = weights->config.upsample_rates[static_cast<size_t>(up_index)];
        weights->ups.push_back(load_weight_norm_conv_transpose1d(
            *weights->store,
            *source,
            "ups." + std::to_string(up_index) + ".0",
            in_channels,
            out_channels,
            kernel,
            stride,
            (kernel - stride) / 2,
            weights->config.weight_storage_type));
        for (int64_t kernel_index = 0; kernel_index < kBigVganNumKernels; ++kernel_index) {
            const int64_t block_index = up_index * kBigVganNumKernels + kernel_index;
            weights->resblocks.push_back(load_resblock(
                *weights->store,
                *source,
                "resblocks." + std::to_string(block_index),
                out_channels,
                weights->config.resblock_kernel_sizes[static_cast<size_t>(kernel_index)],
                weights->config.weight_storage_type));
        }
    }

    const int64_t post_channels =
        weights->config.upsample_initial_channel / (int64_t{1} << upsample_count);
    weights->activation_post =
        load_activation(*weights->store, *source, "activation_post", post_channels, weights->config.weight_storage_type);
    weights->conv_post = load_weight_norm_conv1d(
        *weights->store,
        *source,
        "conv_post",
        1,
        post_channels,
        7,
        1,
        3,
        1,
        false,
        weights->config.weight_storage_type);

    weights->store->upload();
    source->release_storage();
    return BigVganVocoderComponent(std::move(weights), backend);
}

BigVganVocoderComponent::BigVganVocoderComponent(
    std::shared_ptr<const BigVganVocoderWeights> weights,
    core::BackendConfig backend)
    : weights_(std::move(weights)),
      backend_(backend),
      state_(std::make_shared<State>()) {
    if (weights_ == nullptr) {
        throw std::runtime_error("BigVGAN component requires weights");
    }
    state_->runner = std::make_unique<BigVganRunner>(*weights_, backend_);
}

const core::BackendConfig & BigVganVocoderComponent::backend() const noexcept {
    return backend_;
}

const std::shared_ptr<const BigVganVocoderWeights> & BigVganVocoderComponent::weights() const noexcept {
    return weights_;
}

int64_t BigVganVocoderComponent::sample_rate() const noexcept {
    return weights_ == nullptr ? 0 : weights_->config.sampling_rate;
}

int64_t BigVganVocoderComponent::num_mels() const noexcept {
    return weights_ == nullptr ? 0 : weights_->config.num_mels;
}

int64_t BigVganVocoderComponent::loaded_tensor_count() const noexcept {
    return weights_ == nullptr ? 0 : weights_->loaded_tensor_count;
}

int64_t BigVganVocoderComponent::parameter_count() const noexcept {
    return weights_ == nullptr ? 0 : weights_->parameter_count;
}

BigVganVocoderOutput BigVganVocoderComponent::synthesize(
    const std::vector<float> & mel,
    int64_t frames) const {
    if (state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("BigVGAN component is not initialized");
    }
    return state_->runner->run(mel, frames);
}

BigVganVocoderOutput BigVganVocoderComponent::synthesize_chunked(
    const std::vector<float> & mel,
    int64_t frames,
    int64_t chunk_frames,
    int64_t overlap_frames) const {
    if (weights_ == nullptr || state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("BigVGAN component is not initialized");
    }
    if (frames <= 0 || chunk_frames <= 0 || overlap_frames < 0 ||
        static_cast<int64_t>(mel.size()) != weights_->config.num_mels * frames) {
        throw std::runtime_error("BigVGAN chunked synthesis received invalid dimensions");
    }
    if (frames <= chunk_frames) {
        return synthesize(mel, frames);
    }

    std::vector<float> waveform;
    int64_t position = 0;
    while (position < frames) {
        const int64_t active_frames = std::min(chunk_frames, frames - position);
        const int64_t chunk_start = std::max<int64_t>(0, position - overlap_frames);
        const int64_t chunk_end = std::min<int64_t>(frames, position + active_frames + overlap_frames);
        const int64_t left_context = position - chunk_start;
        const int64_t right_context = chunk_end - (position + active_frames);
        const int64_t current_frames = chunk_end - chunk_start;

        std::vector<float> chunk_mel(static_cast<size_t>(weights_->config.num_mels * current_frames), 0.0F);
        for (int64_t mel_bin = 0; mel_bin < weights_->config.num_mels; ++mel_bin) {
            const auto src = mel.begin() + static_cast<std::ptrdiff_t>(mel_bin * frames + chunk_start);
            std::copy(
                src,
                src + static_cast<std::ptrdiff_t>(current_frames),
                chunk_mel.begin() + static_cast<std::ptrdiff_t>(mel_bin * current_frames));
        }
        const auto chunk_wave = state_->runner->run(chunk_mel, current_frames).waveform;
        const int64_t sample_start = left_context * weights_->config.hop_size;
        const int64_t sample_end =
            static_cast<int64_t>(chunk_wave.size()) - right_context * weights_->config.hop_size;
        if (sample_start < 0 || sample_end < sample_start || sample_end > static_cast<int64_t>(chunk_wave.size())) {
            throw std::runtime_error("BigVGAN chunked synthesis crop is invalid");
        }
        waveform.insert(
            waveform.end(),
            chunk_wave.begin() + static_cast<std::ptrdiff_t>(sample_start),
            chunk_wave.begin() + static_cast<std::ptrdiff_t>(sample_end));
        position += active_frames;
    }

    BigVganVocoderOutput out;
    out.waveform = std::move(waveform);
    out.batch = 1;
    out.samples = static_cast<int64_t>(out.waveform.size());
    out.sample_rate = weights_->config.sampling_rate;
    return out;
}

void BigVganVocoderComponent::release_runtime_graph() {
    if (state_ != nullptr && state_->runner != nullptr) {
        state_->runner->release_runtime_graph();
    }
}

}  // namespace engine::modules
