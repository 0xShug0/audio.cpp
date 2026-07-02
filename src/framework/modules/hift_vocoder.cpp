#include "engine/framework/modules/hift_vocoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/sampling/torch_random.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::modules {
namespace {

using Conv1dWeights = HiftVocoderWeights::Conv1dWeights;
using ConvTranspose1dWeights = HiftVocoderWeights::ConvTranspose1dWeights;
using LinearWeights = HiftVocoderWeights::LinearWeights;
using ResBlockWeights = HiftVocoderWeights::ResBlockWeights;
using SnakeWeights = HiftVocoderWeights::SnakeWeights;

constexpr int64_t kHiftStftBins = 9;
constexpr int64_t kHiftStftChannels = kHiftStftBins * 2;

int64_t tensor_elements(const std::vector<int64_t> & shape) {
    if (shape.empty()) {
        throw std::runtime_error("HiFT tensor shape is empty");
    }
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, [](int64_t lhs, int64_t rhs) {
        if (rhs <= 0) {
            throw std::runtime_error("HiFT tensor shape contains non-positive dimension");
        }
        return lhs * rhs;
    });
}

void validate_config(const HiftVocoderConfig & config) {
    if (config.in_channels <= 0 || config.base_channels <= 0 || config.nb_harmonics < 0 ||
        config.sampling_rate <= 0 || config.istft_n_fft <= 0 || config.istft_hop <= 0 ||
        config.f0_num_class != 1 || config.f0_in_channels <= 0 || config.f0_cond_channels <= 0) {
        throw std::runtime_error("HiFT config contains invalid dimensions");
    }
    if (config.istft_n_fft != 16) {
        throw std::runtime_error("HiFT framework module currently expects n_fft=16");
    }
    if (config.upsample_rates.empty() ||
        config.upsample_rates.size() != config.upsample_kernel_sizes.size() ||
        config.source_resblock_kernel_sizes.size() != config.upsample_rates.size() ||
        config.source_resblock_dilation_sizes.size() != config.source_resblock_kernel_sizes.size() ||
        config.resblock_dilation_sizes.size() != config.resblock_kernel_sizes.size()) {
        throw std::runtime_error("HiFT config arrays are inconsistent");
    }
    for (const auto rate : config.upsample_rates) {
        if (rate <= 0) {
            throw std::runtime_error("HiFT upsample rate must be positive");
        }
    }
}

Conv1dWeights load_conv1d(
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
    Conv1dWeights conv;
    conv.out_channels = out_channels;
    conv.in_channels = in_channels;
    conv.kernel = kernel;
    conv.stride = stride;
    conv.padding = padding;
    conv.dilation = dilation;
    conv.use_bias = use_bias;
    conv.weight = source.require_f32(prefix + ".weight", {out_channels, in_channels, kernel});
    conv.weight_tensor = store.make_from_f32(
        core::TensorShape::from_dims({out_channels, in_channels, kernel}),
        assets::TensorStorageType::F32,
        conv.weight);
    if (use_bias) {
        conv.bias = source.require_f32(prefix + ".bias", {out_channels});
        conv.bias_tensor = store.make_from_f32(core::TensorShape::from_dims({out_channels}), assets::TensorStorageType::F32, conv.bias);
    }
    return conv;
}

ConvTranspose1dWeights load_conv_transpose1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    bool use_bias,
    assets::TensorStorageType storage_type) {
    (void) storage_type;
    ConvTranspose1dWeights conv;
    conv.in_channels = in_channels;
    conv.out_channels = out_channels;
    conv.kernel = kernel;
    conv.stride = stride;
    conv.padding = padding;
    conv.use_bias = use_bias;
    conv.weight = source.require_f32(prefix + ".weight", {in_channels, out_channels, kernel});
    conv.weight_tensor = store.make_from_f32(
        core::TensorShape::from_dims({in_channels, out_channels, kernel}),
        assets::TensorStorageType::F32,
        conv.weight);
    if (use_bias) {
        conv.bias = source.require_f32(prefix + ".bias", {out_channels});
        conv.bias_tensor = store.make_from_f32(core::TensorShape::from_dims({out_channels}), assets::TensorStorageType::F32, conv.bias);
    }
    return conv;
}

LinearWeights load_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_features,
    int64_t in_features,
    bool use_bias,
    assets::TensorStorageType storage_type) {
    (void) storage_type;
    LinearWeights linear;
    linear.out_features = out_features;
    linear.in_features = in_features;
    linear.use_bias = use_bias;
    linear.weight = source.require_f32(prefix + ".weight", {out_features, in_features});
    linear.weight_tensor = store.make_from_f32(
        core::TensorShape::from_dims({out_features, in_features}),
        assets::TensorStorageType::F32,
        linear.weight);
    if (use_bias) {
        linear.bias = source.require_f32(prefix + ".bias", {out_features});
        linear.bias_tensor = store.make_from_f32(core::TensorShape::from_dims({out_features}), assets::TensorStorageType::F32, linear.bias);
    }
    return linear;
}

SnakeWeights load_snake(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & name,
    int64_t channels,
    assets::TensorStorageType storage_type) {
    (void) storage_type;
    SnakeWeights snake;
    snake.alpha = source.require_f32(name, {channels});
    std::vector<float> inv_alpha(snake.alpha.size(), 0.0F);
    for (size_t index = 0; index < snake.alpha.size(); ++index) {
        inv_alpha[index] = 1.0F / (snake.alpha[index] + 1.0e-9F);
    }
    snake.alpha_tensor = store.make_from_f32(core::TensorShape::from_dims({channels, 1}), assets::TensorStorageType::F32, snake.alpha);
    snake.inv_alpha_tensor =
        store.make_from_f32(core::TensorShape::from_dims({channels, 1}), assets::TensorStorageType::F32, std::move(inv_alpha));
    return snake;
}

ResBlockWeights load_resblock(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    int64_t kernel,
    const std::vector<int64_t> & dilations,
    assets::TensorStorageType storage_type) {
    ResBlockWeights block;
    block.convs1.reserve(dilations.size());
    block.convs2.reserve(dilations.size());
    block.activations1.reserve(dilations.size());
    block.activations2.reserve(dilations.size());
    for (size_t index = 0; index < dilations.size(); ++index) {
        const int64_t dilation = dilations[index];
        block.convs1.push_back(load_conv1d(
            store,
            source,
            prefix + ".convs1." + std::to_string(index),
            channels,
            channels,
            kernel,
            1,
            (kernel * dilation - dilation) / 2,
            dilation,
            true,
            storage_type));
        block.convs2.push_back(load_conv1d(
            store,
            source,
            prefix + ".convs2." + std::to_string(index),
            channels,
            channels,
            kernel,
            1,
            (kernel - 1) / 2,
            1,
            true,
            storage_type));
        block.activations1.push_back(load_snake(
            store,
            source,
            prefix + ".activations1." + std::to_string(index) + ".alpha",
            channels,
            storage_type));
        block.activations2.push_back(load_snake(
            store,
            source,
            prefix + ".activations2." + std::to_string(index) + ".alpha",
            channels,
            storage_type));
    }
    return block;
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

ggml_tensor * reflect_pad_1d(
    ggml_context * ctx,
    ggml_tensor * x,
    int64_t left,
    int64_t right) {
    if (left < 0 || right < 0 || left >= x->ne[0] || right >= x->ne[0]) {
        throw std::runtime_error("HiFT reflect pad size mismatch");
    }
    return ggml_pad_reflect_1d(ctx, x, static_cast<int>(left), static_cast<int>(right));
}

ggml_tensor * conv1d(
    ggml_context * ctx,
    ggml_tensor * x,
    const Conv1dWeights & conv,
    const std::string & name,
    bool allow_pointwise_fastpath) {
    if (allow_pointwise_fastpath &&
        conv.kernel == 1 &&
        conv.stride == 1 &&
        conv.padding == 0 &&
        conv.dilation == 1) {
        ggml_tensor * weight = weight_2d(ctx, conv.weight_tensor, conv.in_channels, conv.out_channels, name + ".weight");
        ggml_tensor * x_cf = contiguous_if_needed(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
        ggml_tensor * y_cf = ggml_mul_mat(ctx, weight, x_cf);
        ggml_tensor * y = contiguous_if_needed(ctx, ggml_permute(ctx, y_cf, 1, 0, 2, 3));
        y = ggml_reshape_2d(ctx, y, x->ne[0], conv.out_channels);
        if (conv.use_bias) {
            y = ggml_add(ctx, y, weight_2d(ctx, conv.bias_tensor, 1, conv.out_channels, name + ".bias"));
        }
        return named(y, name.c_str());
    }

    ggml_tensor * weight =
        weight_3d(ctx, conv.weight_tensor, conv.kernel, conv.in_channels, conv.out_channels, name + ".weight");
    ggml_tensor * x3 = ggml_reshape_3d(ctx, contiguous_if_needed(ctx, x), x->ne[0], x->ne[1], 1);
    ggml_tensor * y3 =
        ggml_conv_1d(ctx, weight, x3, static_cast<int>(conv.stride), static_cast<int>(conv.padding), static_cast<int>(conv.dilation));
    ggml_tensor * y = ggml_reshape_2d(ctx, y3, y3->ne[0], y3->ne[1]);
    if (conv.use_bias) {
        y = ggml_add(ctx, y, weight_2d(ctx, conv.bias_tensor, 1, conv.out_channels, name + ".bias"));
    }
    return named(y, name.c_str());
}

ggml_tensor * conv_transpose1d(
    ggml_context * ctx,
    ggml_tensor * x,
    const ConvTranspose1dWeights & conv,
    const std::string & name) {
    ggml_tensor * weight =
        weight_3d(ctx, conv.weight_tensor, conv.kernel, conv.out_channels, conv.in_channels, name + ".weight");
    ggml_tensor * y_full = ggml_conv_transpose_1d(ctx, weight, contiguous_if_needed(ctx, x), static_cast<int>(conv.stride), 0, 1);
    const int64_t cropped_len = (x->ne[0] - 1) * conv.stride - 2 * conv.padding + conv.kernel;
    ggml_tensor * y =
        ggml_view_2d(ctx, y_full, cropped_len, conv.out_channels, y_full->nb[1], static_cast<size_t>(conv.padding) * sizeof(float));
    if (conv.use_bias) {
        y = ggml_add(ctx, y, weight_2d(ctx, conv.bias_tensor, 1, conv.out_channels, name + ".bias"));
    }
    return named(y, name.c_str());
}

ggml_tensor * snake(
    ggml_context * ctx,
    ggml_tensor * x,
    const SnakeWeights & act,
    const std::string & name) {
    ggml_tensor * alpha = weight_2d(ctx, act.alpha_tensor, 1, x->ne[1], name + ".alpha");
    ggml_tensor * inv = weight_2d(ctx, act.inv_alpha_tensor, 1, x->ne[1], name + ".inv_alpha");
    ggml_tensor * s = ggml_sin(ctx, ggml_mul(ctx, x, alpha));
    s = ggml_mul(ctx, ggml_sqr(ctx, s), inv);
    return named(ggml_add(ctx, x, s), name.c_str());
}

ggml_tensor * resblock(
    ggml_context * ctx,
    ggml_tensor * x,
    const ResBlockWeights & block,
    const std::string & name,
    bool allow_pointwise_fastpath) {
    for (size_t index = 0; index < block.convs1.size(); ++index) {
        ggml_tensor * xt = snake(ctx, x, block.activations1[index], name + ".act1." + std::to_string(index));
        xt = conv1d(ctx, xt, block.convs1[index], name + ".convs1." + std::to_string(index), allow_pointwise_fastpath);
        xt = snake(ctx, xt, block.activations2[index], name + ".act2." + std::to_string(index));
        xt = conv1d(ctx, xt, block.convs2[index], name + ".convs2." + std::to_string(index), allow_pointwise_fastpath);
        x = named(ggml_add(ctx, x, xt), (name + ".residual." + std::to_string(index)).c_str());
    }
    return x;
}

ggml_tensor * build_f0_graph(
    ggml_context * ctx,
    const HiftVocoderWeights & weights,
    ggml_tensor * input) {
    ggml_tensor * x = input;
    for (size_t index = 0; index < weights.f0_predictor.condnet.size(); ++index) {
        x = conv1d(ctx, x, weights.f0_predictor.condnet[index], "f0.condnet." + std::to_string(index), false);
        x = named(ggml_elu(ctx, x), ("f0.elu." + std::to_string(index)).c_str());
    }
    auto linear_as_conv = Conv1dWeights{};
    linear_as_conv.weight = weights.f0_predictor.classifier.weight;
    linear_as_conv.bias = weights.f0_predictor.classifier.bias;
    linear_as_conv.weight_tensor = weights.f0_predictor.classifier.weight_tensor;
    linear_as_conv.bias_tensor = weights.f0_predictor.classifier.bias_tensor;
    linear_as_conv.out_channels = weights.f0_predictor.classifier.out_features;
    linear_as_conv.in_channels = weights.f0_predictor.classifier.in_features;
    linear_as_conv.kernel = 1;
    linear_as_conv.stride = 1;
    linear_as_conv.padding = 0;
    linear_as_conv.dilation = 1;
    linear_as_conv.use_bias = weights.f0_predictor.classifier.use_bias;
    x = conv1d(ctx, x, linear_as_conv, "f0.classifier", false);
    return named(ggml_abs(ctx, x), "f0.output");
}

ggml_tensor * build_backend_graph(
    ggml_context * ctx,
    const HiftVocoderWeights & weights,
    ggml_tensor * speech_in,
    ggml_tensor * source_in,
    bool allow_pointwise_fastpath) {
    ggml_tensor * x = conv1d(ctx, speech_in, weights.conv_pre, "conv_pre", allow_pointwise_fastpath);
    const int64_t num_kernels = static_cast<int64_t>(weights.config.resblock_kernel_sizes.size());
    for (size_t up_index = 0; up_index < weights.ups.size(); ++up_index) {
        x = named(ggml_leaky_relu(ctx, x, weights.config.lrelu_slope, false), ("ups." + std::to_string(up_index) + ".act").c_str());
        x = conv_transpose1d(ctx, x, weights.ups[up_index], "ups." + std::to_string(up_index));
        if (up_index + 1 == weights.ups.size()) {
            x = named(reflect_pad_1d(ctx, x, 1, 0), "reflection_pad");
        }

        ggml_tensor * si =
            conv1d(ctx, source_in, weights.source_downs[up_index], "source_downs." + std::to_string(up_index), allow_pointwise_fastpath);
        si = resblock(ctx, si, weights.source_resblocks[up_index], "source_resblocks." + std::to_string(up_index), allow_pointwise_fastpath);
        x = named(ggml_add(ctx, x, si), ("fusion." + std::to_string(up_index)).c_str());

        ggml_tensor * sum = nullptr;
        for (int64_t kernel_index = 0; kernel_index < num_kernels; ++kernel_index) {
            const int64_t block_index = static_cast<int64_t>(up_index) * num_kernels + kernel_index;
            ggml_tensor * block_out = resblock(
                ctx,
                x,
                weights.resblocks[static_cast<size_t>(block_index)],
                "resblocks." + std::to_string(block_index),
                allow_pointwise_fastpath);
            sum = sum == nullptr ? block_out : ggml_add(ctx, sum, block_out);
        }
        x = named(ggml_scale(ctx, sum, 1.0F / static_cast<float>(num_kernels)), ("upsample." + std::to_string(up_index)).c_str());
    }
    x = named(ggml_leaky_relu(ctx, x, 0.01F, false), "post.act");
    return conv1d(ctx, x, weights.conv_post, "conv_post", allow_pointwise_fastpath);
}

int64_t upsample_scale(const HiftVocoderConfig & config) {
    return std::accumulate(config.upsample_rates.begin(), config.upsample_rates.end(), config.istft_hop, std::multiplies<int64_t>{});
}

std::vector<float> nearest_upsample_f0(const std::vector<float> & f0, int64_t frames, int64_t scale) {
    if (static_cast<int64_t>(f0.size()) != frames) {
        throw std::runtime_error("HiFT F0 shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(frames * scale), 0.0F);
    for (int64_t t = 0; t < frames; ++t) {
        std::fill(
            out.begin() + static_cast<std::ptrdiff_t>(t * scale),
            out.begin() + static_cast<std::ptrdiff_t>((t + 1) * scale),
            f0[static_cast<size_t>(t)]);
    }
    return out;
}

std::vector<float> make_sine_source(
    const HiftVocoderWeights & weights,
    const std::vector<float> & f0_upsampled,
    uint64_t seed,
    uint64_t prior_noise_values,
    const std::vector<float> * source_random_values) {
    constexpr float kPi = 3.14159265358979323846F;
    constexpr float kTwoPi = 2.0F * kPi;
    const int64_t frames = static_cast<int64_t>(f0_upsampled.size());
    const int64_t harmonics = weights.config.nb_harmonics + 1;
    const size_t phase_count = static_cast<size_t>(harmonics);
    const size_t gaussian_count = static_cast<size_t>(harmonics * frames);
    if (source_random_values != nullptr &&
        source_random_values->size() != phase_count + gaussian_count) {
        throw std::runtime_error("HiFT source random value count mismatch");
    }
    std::vector<float> phase_uniform = source_random_values == nullptr
        ? sampling::generate_torch_cuda_uniform(phase_count, seed, prior_noise_values)
        : std::vector<float>(source_random_values->begin(), source_random_values->begin() + static_cast<std::ptrdiff_t>(phase_count));
    std::vector<float> phase(static_cast<size_t>(harmonics), 0.0F);
    for (int64_t h = 0; h < harmonics; ++h) {
        phase[static_cast<size_t>(h)] = phase_uniform[static_cast<size_t>(h)] * kTwoPi - kPi;
    }
    phase[0] = 0.0F;
    const uint64_t gaussian_offset = prior_noise_values + static_cast<uint64_t>(phase_uniform.size());
    std::vector<float> gaussian = source_random_values == nullptr
        ? sampling::generate_torch_cuda_randn(
              gaussian_count,
              seed,
              sampling::TorchRandnPrecision::Float32,
              gaussian_offset)
        : std::vector<float>(source_random_values->begin() + static_cast<std::ptrdiff_t>(phase_count), source_random_values->end());

    std::vector<float> source(static_cast<size_t>(frames), 0.0F);
    std::vector<float> frac(static_cast<size_t>(harmonics), 0.0F);
    for (int64_t t = 0; t < frames; ++t) {
        const float base_f0 = f0_upsampled[static_cast<size_t>(t)];
        const float uv = base_f0 > weights.config.nsf_voiced_threshold ? 1.0F : 0.0F;
        float sum = weights.source_linear.bias[0];
        for (int64_t h = 0; h < harmonics; ++h) {
            frac[static_cast<size_t>(h)] +=
                base_f0 * static_cast<float>(h + 1) / static_cast<float>(weights.config.sampling_rate);
            frac[static_cast<size_t>(h)] -= std::floor(frac[static_cast<size_t>(h)]);
            const float theta = kTwoPi * frac[static_cast<size_t>(h)];
            float wave =
                weights.config.nsf_alpha * std::sin(theta + phase[static_cast<size_t>(h)]);
            const float noise_amp =
                uv * weights.config.nsf_sigma + (1.0F - uv) * weights.config.nsf_alpha / 3.0F;
            wave = wave * uv + noise_amp * gaussian[static_cast<size_t>(h * frames + t)];
            sum += wave * weights.source_linear.weight[static_cast<size_t>(h)];
        }
        source[static_cast<size_t>(t)] = std::tanh(sum);
    }
    return source;
}

std::vector<float> source_stft_bct(
    const HiftVocoderWeights & weights,
    const std::vector<float> & source,
    int64_t & stft_frames,
    std::vector<float> & window) {
    const audio::STFTConfig window_config{
        weights.config.istft_n_fft,
        weights.config.istft_hop,
        weights.config.istft_n_fft,
        true,
        audio::STFTPadMode::Reflect,
        audio::STFTFamily::Kokoro,
    };
    window = audio::get_cached_stft_window(window_config);
    const auto complex = audio::STFT().compute_complex(
        source,
        window,
        1,
        static_cast<int64_t>(source.size()),
        {weights.config.istft_n_fft, weights.config.istft_hop, weights.config.istft_n_fft});
    stft_frames = complex.shape[2];
    std::vector<float> stft(static_cast<size_t>(kHiftStftChannels * stft_frames), 0.0F);
    for (int64_t f = 0; f < kHiftStftBins; ++f) {
        for (int64_t t = 0; t < stft_frames; ++t) {
            const size_t complex_index = static_cast<size_t>((f * stft_frames + t) * 2);
            stft[static_cast<size_t>(f * stft_frames + t)] = complex.values[complex_index];
            stft[static_cast<size_t>((kHiftStftBins + f) * stft_frames + t)] = complex.values[complex_index + 1];
        }
    }
    return stft;
}

std::vector<float> waveform_from_post(
    const HiftVocoderWeights & weights,
    const std::vector<float> & post,
    int64_t frames,
    const std::vector<float> & window) {
    if (static_cast<int64_t>(post.size()) != kHiftStftChannels * frames) {
        throw std::runtime_error("HiFT post shape mismatch");
    }
    std::vector<float> complex_spec(static_cast<size_t>(kHiftStftBins * frames * 2), 0.0F);
    for (int64_t f = 0; f < kHiftStftBins; ++f) {
        for (int64_t t = 0; t < frames; ++t) {
            float magnitude = std::exp(post[static_cast<size_t>(f * frames + t)]);
            magnitude = std::min(magnitude, 1.0e2F);
            const float phase = std::sin(post[static_cast<size_t>((kHiftStftBins + f) * frames + t)]);
            const size_t complex_index = static_cast<size_t>((f * frames + t) * 2);
            complex_spec[complex_index] = magnitude * std::cos(phase);
            complex_spec[complex_index + 1] = magnitude * std::sin(phase);
        }
    }
    const int64_t samples = weights.config.istft_hop * (frames - 1);
    auto wav = audio::ISTFT().compute(
        complex_spec,
        window,
        1,
        kHiftStftBins,
        frames,
        samples,
        {weights.config.istft_n_fft, weights.config.istft_hop, weights.config.istft_n_fft});
    for (float & sample : wav.values) {
        sample = std::clamp(sample, -weights.config.audio_limit, weights.config.audio_limit);
    }
    return wav.values;
}

class F0Runner {
public:
    F0Runner(const HiftVocoderWeights & weights, int64_t frames)
        : weights_(&weights), capacity_frames_(frames) {
        ggml_init_params params{128ull * 1024ull * 1024ull, nullptr, true};
        ctx_ = ggml_init(params);
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize HiFT F0 graph context");
        }
        input_ = named(ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, frames, weights.config.in_channels), "f0.input");
        output_ = build_f0_graph(ctx_, weights, input_);
        graph_ = ggml_new_graph_custom(ctx_, 32768, false);
        ggml_build_forward_expand(graph_, output_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, weights.execution_context->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate HiFT F0 graph");
        }
    }

    ~F0Runner() {
        if (weights_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(weights_->execution_context->backend(), graph_);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
        if (ctx_ != nullptr) {
            ggml_free(ctx_);
        }
    }

    bool supports(const HiftVocoderWeights & weights, int64_t frames) const {
        return weights_ == &weights && capacity_frames_ == frames;
    }

    std::vector<float> run(const std::vector<float> & mel, int64_t frames) {
        if (frames != capacity_frames_) {
            throw std::runtime_error("HiFT F0 graph frame mismatch");
        }
        ggml_backend_tensor_set(input_, mel.data(), 0, mel.size() * sizeof(float));
        if (engine::core::compute_backend_graph(weights_->execution_context->backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for HiFT F0 predictor");
        }
        auto raw = core::read_tensor_f32(output_);
        std::vector<float> f0(static_cast<size_t>(frames), 0.0F);
        for (int64_t t = 0; t < frames; ++t) {
            f0[static_cast<size_t>(t)] = raw[static_cast<size_t>(t)];
        }
        return f0;
    }

private:
    const HiftVocoderWeights * weights_ = nullptr;
    int64_t capacity_frames_ = 0;
    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

class BackendRunner {
public:
    BackendRunner(const HiftVocoderWeights & weights, int64_t frames, int64_t stft_frames)
        : weights_(&weights), capacity_frames_(frames), capacity_stft_frames_(stft_frames) {
        ggml_init_params params{
            512ull * 1024ull * 1024ull +
                static_cast<size_t>(std::max<int64_t>(frames, 1)) * 4ull * 1024ull * 1024ull,
            nullptr,
            true,
        };
        ctx_ = ggml_init(params);
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize HiFT backend graph context");
        }
        speech_in_ = named(ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, frames, weights.config.in_channels), "speech_in");
        source_in_ = named(ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, stft_frames, kHiftStftChannels), "source_in");
        post_ = build_backend_graph(
            ctx_,
            weights,
            speech_in_,
            source_in_,
            weights.execution_context->uses_host_graph_plan());
        graph_ = ggml_new_graph_custom(ctx_, 65536, false);
        ggml_build_forward_expand(graph_, post_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, weights.execution_context->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate HiFT backend graph");
        }
    }

    ~BackendRunner() {
        if (weights_ != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(weights_->execution_context->backend(), graph_);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
        if (ctx_ != nullptr) {
            ggml_free(ctx_);
        }
    }

    bool supports(const HiftVocoderWeights & weights, int64_t frames, int64_t stft_frames) const {
        return weights_ == &weights && capacity_frames_ == frames && capacity_stft_frames_ == stft_frames;
    }

    std::vector<float> run(
        const std::vector<float> & mel,
        int64_t frames,
        const std::vector<float> & source_stft,
        int64_t stft_frames) {
        if (frames != capacity_frames_ || stft_frames != capacity_stft_frames_) {
            throw std::runtime_error("HiFT backend graph frame mismatch");
        }
        ggml_backend_tensor_set(speech_in_, mel.data(), 0, mel.size() * sizeof(float));
        ggml_backend_tensor_set(source_in_, source_stft.data(), 0, source_stft.size() * sizeof(float));
        if (engine::core::compute_backend_graph(weights_->execution_context->backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for HiFT backend");
        }
        return core::read_tensor_f32(post_);
    }

private:
    const HiftVocoderWeights * weights_ = nullptr;
    int64_t capacity_frames_ = 0;
    int64_t capacity_stft_frames_ = 0;
    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * speech_in_ = nullptr;
    ggml_tensor * source_in_ = nullptr;
    ggml_tensor * post_ = nullptr;
};

class HiftRunner {
public:
    HiftRunner(const HiftVocoderWeights & weights, core::BackendConfig backend)
        : weights_(&weights), backend_(backend) {}

    std::vector<float> predict_f0(const std::vector<float> & mel, int64_t frames) {
        std::lock_guard<std::mutex> lock(mutex_);
        return predict_f0_locked(mel, frames);
    }

    HiftVocoderOutput run(
        const std::vector<float> & mel,
        int64_t frames,
        uint64_t seed,
        uint64_t prior_noise_values,
        const std::vector<float> * source_random_values) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto f0 = predict_f0_locked(mel, frames);
        const int64_t scale = upsample_scale(weights_->config);
        const auto f0_upsampled = nearest_upsample_f0(f0, frames, scale);
        const auto source = make_sine_source(*weights_, f0_upsampled, seed, prior_noise_values, source_random_values);
        std::vector<float> window;
        int64_t stft_frames = 0;
        const auto source_stft = source_stft_bct(*weights_, source, stft_frames, window);
        if (backend_runner_ != nullptr && !backend_runner_->supports(*weights_, frames, stft_frames)) {
            backend_runner_.reset();
        }
        if (backend_runner_ == nullptr) {
            backend_runner_ = std::make_unique<BackendRunner>(*weights_, frames, stft_frames);
        }
        const auto post = backend_runner_->run(mel, frames, source_stft, stft_frames);
        HiftVocoderOutput out;
        out.waveform = waveform_from_post(*weights_, post, stft_frames, window);
        out.batch = 1;
        out.samples = static_cast<int64_t>(out.waveform.size());
        out.sample_rate = weights_->config.sampling_rate;
        return out;
    }

    void release_runtime_cache() {
        std::lock_guard<std::mutex> lock(mutex_);
        f0_runner_.reset();
        backend_runner_.reset();
    }

private:
    std::vector<float> predict_f0_locked(const std::vector<float> & mel, int64_t frames) {
        if (frames <= 0 || static_cast<int64_t>(mel.size()) != weights_->config.in_channels * frames) {
            throw std::runtime_error("HiFT mel shape mismatch");
        }
        if (f0_runner_ != nullptr && !f0_runner_->supports(*weights_, frames)) {
            f0_runner_.reset();
        }
        if (f0_runner_ == nullptr) {
            f0_runner_ = std::make_unique<F0Runner>(*weights_, frames);
        }
        return f0_runner_->run(mel, frames);
    }

    const HiftVocoderWeights * weights_ = nullptr;
    core::BackendConfig backend_;
    std::mutex mutex_;
    std::unique_ptr<F0Runner> f0_runner_;
    std::unique_ptr<BackendRunner> backend_runner_;
};

}  // namespace

struct HiftVocoderComponent::State {
    std::unique_ptr<HiftRunner> runner;
};

HiftVocoderComponent HiftVocoderComponent::load_from_safetensors(
    const std::filesystem::path & checkpoint_path,
    core::BackendConfig backend,
    HiftVocoderConfig config) {
    validate_config(config);
    const auto source = assets::open_tensor_source(checkpoint_path);
    auto weights = std::make_shared<HiftVocoderWeights>();
    weights->config = std::move(config);
    weights->source_path = source->source_path();
    weights->execution_context = std::make_shared<core::ExecutionContext>(backend);
    weights->store = std::make_shared<core::BackendWeightStore>(
        weights->execution_context->backend(),
        weights->execution_context->backend_type(),
        "framework.hift.weights",
        768ull * 1024ull * 1024ull);

    for (const auto & tensor : source->tensors()) {
        weights->parameter_count += tensor_elements(tensor.shape);
        ++weights->loaded_tensor_count;
    }

    for (int index = 0; index < 5; ++index) {
        weights->f0_predictor.condnet.push_back(load_conv1d(
            *weights->store,
            *source,
            "f0_predictor.condnet." + std::to_string(index * 2),
            weights->config.f0_cond_channels,
            index == 0 ? weights->config.f0_in_channels : weights->config.f0_cond_channels,
            3,
            1,
            1,
            1,
            true,
            weights->config.weight_storage_type));
    }
    weights->f0_predictor.classifier = load_linear(
        *weights->store,
        *source,
        "f0_predictor.classifier",
        weights->config.f0_num_class,
        weights->config.f0_cond_channels,
        true,
        weights->config.weight_storage_type);

    weights->conv_pre = load_conv1d(
        *weights->store,
        *source,
        "conv_pre",
        weights->config.base_channels,
        weights->config.in_channels,
        7,
        1,
        3,
        1,
        true,
        weights->config.weight_storage_type);

    const int64_t upsample_count = static_cast<int64_t>(weights->config.upsample_rates.size());
    weights->ups.reserve(static_cast<size_t>(upsample_count));
    weights->source_downs.reserve(static_cast<size_t>(upsample_count));
    weights->source_resblocks.reserve(static_cast<size_t>(upsample_count));
    const int64_t num_kernels = static_cast<int64_t>(weights->config.resblock_kernel_sizes.size());
    weights->resblocks.reserve(static_cast<size_t>(upsample_count * num_kernels));

    std::vector<int64_t> downsample_rates = {1};
    if (upsample_count > 1) {
        for (int64_t index = upsample_count - 1; index > 0; --index) {
            downsample_rates.push_back(weights->config.upsample_rates[static_cast<size_t>(index)]);
        }
    }
    std::vector<int64_t> downsample_cum(downsample_rates.size(), 1);
    std::partial_sum(
        downsample_rates.begin(),
        downsample_rates.end(),
        downsample_cum.begin(),
        std::multiplies<int64_t>{});
    std::reverse(downsample_cum.begin(), downsample_cum.end());

    for (int64_t up_index = 0; up_index < upsample_count; ++up_index) {
        const int64_t in_channels = weights->config.base_channels / (int64_t{1} << up_index);
        const int64_t out_channels = weights->config.base_channels / (int64_t{1} << (up_index + 1));
        const int64_t kernel = weights->config.upsample_kernel_sizes[static_cast<size_t>(up_index)];
        const int64_t stride = weights->config.upsample_rates[static_cast<size_t>(up_index)];
        weights->ups.push_back(load_conv_transpose1d(
            *weights->store,
            *source,
            "ups." + std::to_string(up_index),
            in_channels,
            out_channels,
            kernel,
            stride,
            (kernel - stride) / 2,
            true,
            weights->config.weight_storage_type));

        const int64_t source_stride = downsample_cum[static_cast<size_t>(up_index)];
        const int64_t source_kernel = source_stride == 1 ? 1 : source_stride * 2;
        const int64_t source_padding = source_stride == 1 ? 0 : source_stride / 2;
        weights->source_downs.push_back(load_conv1d(
            *weights->store,
            *source,
            "source_downs." + std::to_string(up_index),
            out_channels,
            kHiftStftChannels,
            source_kernel,
            source_stride,
            source_padding,
            1,
            true,
            weights->config.weight_storage_type));

        weights->source_resblocks.push_back(load_resblock(
            *weights->store,
            *source,
            "source_resblocks." + std::to_string(up_index),
            out_channels,
            weights->config.source_resblock_kernel_sizes[static_cast<size_t>(up_index)],
            weights->config.source_resblock_dilation_sizes[static_cast<size_t>(up_index)],
            weights->config.weight_storage_type));

        for (int64_t kernel_index = 0; kernel_index < num_kernels; ++kernel_index) {
            const int64_t block_index = up_index * num_kernels + kernel_index;
            weights->resblocks.push_back(load_resblock(
                *weights->store,
                *source,
                "resblocks." + std::to_string(block_index),
                out_channels,
                weights->config.resblock_kernel_sizes[static_cast<size_t>(kernel_index)],
                weights->config.resblock_dilation_sizes[static_cast<size_t>(kernel_index)],
                weights->config.weight_storage_type));
        }
    }

    const int64_t post_channels = weights->config.base_channels / (int64_t{1} << upsample_count);
    weights->conv_post = load_conv1d(
        *weights->store,
        *source,
        "conv_post",
        kHiftStftChannels,
        post_channels,
        7,
        1,
        3,
        1,
        true,
        weights->config.weight_storage_type);
    weights->source_linear = load_linear(
        *weights->store,
        *source,
        "m_source.l_linear",
        1,
        weights->config.nb_harmonics + 1,
        true,
        weights->config.weight_storage_type);

    weights->store->upload();
    source->release_storage();
    return HiftVocoderComponent(std::move(weights), backend);
}

HiftVocoderComponent::HiftVocoderComponent(
    std::shared_ptr<const HiftVocoderWeights> weights,
    core::BackendConfig backend)
    : weights_(std::move(weights)),
      backend_(backend),
      state_(std::make_shared<State>()) {
    if (weights_ == nullptr) {
        throw std::runtime_error("HiFT component requires weights");
    }
    state_->runner = std::make_unique<HiftRunner>(*weights_, backend_);
}

const core::BackendConfig & HiftVocoderComponent::backend() const noexcept {
    return backend_;
}

const std::shared_ptr<const HiftVocoderWeights> & HiftVocoderComponent::weights() const noexcept {
    return weights_;
}

int64_t HiftVocoderComponent::sample_rate() const noexcept {
    return weights_ == nullptr ? 0 : weights_->config.sampling_rate;
}

int64_t HiftVocoderComponent::num_mels() const noexcept {
    return weights_ == nullptr ? 0 : weights_->config.in_channels;
}

int64_t HiftVocoderComponent::loaded_tensor_count() const noexcept {
    return weights_ == nullptr ? 0 : weights_->loaded_tensor_count;
}

int64_t HiftVocoderComponent::parameter_count() const noexcept {
    return weights_ == nullptr ? 0 : weights_->parameter_count;
}

HiftVocoderOutput HiftVocoderComponent::synthesize(
    const std::vector<float> & mel,
    int64_t frames,
    uint64_t seed,
    uint64_t prior_noise_values,
    const std::vector<float> * source_random_values) const {
    if (state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("HiFT component is not initialized");
    }
    return state_->runner->run(mel, frames, seed, prior_noise_values, source_random_values);
}

std::vector<float> HiftVocoderComponent::predict_f0(const std::vector<float> & mel, int64_t frames) const {
    if (state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("HiFT component is not initialized");
    }
    return state_->runner->predict_f0(mel, frames);
}

void HiftVocoderComponent::release_runtime_cache() const {
    if (state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("HiFT component is not initialized");
    }
    state_->runner->release_runtime_cache();
}

}  // namespace engine::modules
