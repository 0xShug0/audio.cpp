#include "engine/framework/modules/vocoders/hifigan_vocoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/torch_random.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::modules {
namespace {

namespace binding = engine::modules::binding;
using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

int64_t tensor_elements(const std::vector<int64_t> & shape) {
    if (shape.empty()) {
        throw std::runtime_error("HiFi-GAN tensor shape is empty");
    }
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, [](int64_t lhs, int64_t rhs) {
        if (rhs <= 0) {
            throw std::runtime_error("HiFi-GAN tensor shape contains non-positive dimension");
        }
        return lhs * rhs;
    });
}

std::string tensor_name(const HifiGanVocoderConfig & config, const std::string & suffix) {
    return config.tensor_prefix.empty() ? suffix : config.tensor_prefix + "." + suffix;
}

void validate_config(const HifiGanVocoderConfig & config) {
    if (config.sampling_rate <= 0 || config.num_mels <= 0 || config.upsample_initial_channel <= 0 ||
        config.output_channels <= 0) {
        throw std::runtime_error("HiFi-GAN config contains invalid dimensions");
    }
    if (config.upsample_rates.empty() ||
        config.upsample_rates.size() != config.upsample_kernel_sizes.size() ||
        config.resblock_kernel_sizes.empty() ||
        config.resblock_dilation_sizes.size() != config.resblock_kernel_sizes.size()) {
        throw std::runtime_error("HiFi-GAN config arrays are inconsistent");
    }
    for (const auto rate : config.upsample_rates) {
        if (rate <= 0) {
            throw std::runtime_error("HiFi-GAN upsample rate must be positive");
        }
    }
    for (const auto & dilations : config.resblock_dilation_sizes) {
        if (dilations.empty()) {
            throw std::runtime_error("HiFi-GAN resblock dilation list cannot be empty");
        }
    }
    if (config.source.enabled && config.source.harmonic_num < 0) {
        throw std::runtime_error("HiFi-GAN source conditioning harmonic count cannot be negative");
    }
    if (config.global_conditioning.channels < 0) {
        throw std::runtime_error("HiFi-GAN global conditioning channel count cannot be negative");
    }
}

HifiGanVocoderWeights::LinearWeights load_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const HifiGanVocoderConfig & config,
    const std::string & prefix,
    int64_t out_features,
    int64_t in_features,
    bool use_bias) {
    HifiGanVocoderWeights::LinearWeights out;
    out.out_features = out_features;
    out.in_features = in_features;
    out.use_bias = use_bias;
    out.weight = store.load_tensor(
        source,
        tensor_name(config, prefix + ".weight"),
        assets::TensorStorageType::F32,
        {out_features, in_features});
    if (use_bias) {
        out.bias = store.load_tensor(
            source,
            tensor_name(config, prefix + ".bias"),
            assets::TensorStorageType::F32,
            {out_features});
    }
    return out;
}

Conv1dWeights load_conv1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const HifiGanVocoderConfig & config,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    bool use_bias = true) {
    return binding::conv1d_from_source_resolving_weight_norm(
        store,
        source,
        tensor_name(config, prefix),
        config.weight_storage_type,
        out_channels,
        in_channels,
        kernel,
        use_bias);
}

ConvTranspose1dWeights load_conv_transpose1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const HifiGanVocoderConfig & config,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel) {
    return binding::conv_transpose1d_from_source_resolving_weight_norm(
        store,
        source,
        tensor_name(config, prefix),
        config.weight_storage_type,
        in_channels,
        out_channels,
        kernel,
        true);
}

HifiGanVocoderWeights::ResBlockWeights load_resblock(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const HifiGanVocoderConfig & config,
    const std::string & prefix,
    int64_t channels,
    int64_t kernel,
    const std::vector<int64_t> & dilations) {
    HifiGanVocoderWeights::ResBlockWeights out;
    out.convs1.reserve(dilations.size());
    if (config.resblock_kind == HifiGanResBlockKind::PairedConv) {
        out.convs2.reserve(dilations.size());
    }
    for (size_t index = 0; index < dilations.size(); ++index) {
        if (config.resblock_kind == HifiGanResBlockKind::SingleConv) {
            out.convs1.push_back(load_conv1d(
                store,
                source,
                config,
                prefix + ".convs." + std::to_string(index),
                channels,
                channels,
                kernel));
        } else {
            out.convs1.push_back(load_conv1d(
                store,
                source,
                config,
                prefix + ".convs1." + std::to_string(index),
                channels,
                channels,
                kernel));
            out.convs2.push_back(load_conv1d(
                store,
                source,
                config,
                prefix + ".convs2." + std::to_string(index),
                channels,
                channels,
                kernel));
        }
    }
    return out;
}

core::TensorValue crop_time_to(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t frames) {
    if (input.shape.rank != 3) {
        throw std::runtime_error("HiFi-GAN crop expects BCH tensor");
    }
    if (frames <= 0 || frames > input.shape.dims[2]) {
        throw std::runtime_error("HiFi-GAN crop frame count is invalid");
    }
    if (frames == input.shape.dims[2]) {
        return input;
    }
    return SliceModule({2, 0, frames}).build(ctx, input);
}

core::TensorValue hifigan_resblock(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const HifiGanVocoderWeights::ResBlockWeights & weights,
    int64_t channels,
    int64_t kernel,
    const std::vector<int64_t> & dilations,
    HifiGanResBlockKind kind,
    float slope) {
    auto x = input;
    if (weights.convs1.size() != dilations.size() ||
        (kind == HifiGanResBlockKind::PairedConv && weights.convs2.size() != dilations.size())) {
        throw std::runtime_error("HiFi-GAN resblock weight count mismatch");
    }
    for (size_t layer = 0; layer < dilations.size(); ++layer) {
        auto xt = LeakyReluModule({slope}).build(ctx, x);
        xt = Conv1dModule({
                channels,
                channels,
                kernel,
                1,
                static_cast<int>((kernel * dilations[layer] - dilations[layer]) / 2),
                static_cast<int>(dilations[layer]),
                true})
                 .build(ctx, xt, weights.convs1[layer]);
        if (kind == HifiGanResBlockKind::PairedConv) {
            xt = LeakyReluModule({slope}).build(ctx, xt);
            xt = Conv1dModule({
                    channels,
                    channels,
                    kernel,
                    1,
                    static_cast<int>((kernel - 1) / 2),
                    1,
                    true})
                     .build(ctx, xt, weights.convs2[layer]);
        }
        x = AddModule().build(ctx, xt, x);
    }
    return x;
}

core::TensorValue align_conditioning_frames(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & conditioning,
    int64_t frames) {
    if (conditioning.shape.rank != 3) {
        throw std::runtime_error("HiFi-GAN global conditioning expects BCH tensor");
    }
    if (conditioning.shape.dims[2] == frames) {
        return conditioning;
    }
    if (conditioning.shape.dims[2] == 1) {
        return RepeatModule({core::TensorShape::from_dims({conditioning.shape.dims[0], conditioning.shape.dims[1], frames})})
            .build(ctx, conditioning);
    }
    throw std::runtime_error("HiFi-GAN global conditioning frame count mismatch");
}

core::TensorValue hifigan_graph(
    core::ModuleBuildContext & ctx,
    const HifiGanVocoderWeights & weights,
    const core::TensorValue & mel,
    const core::TensorValue * source,
    const core::TensorValue * conditioning) {
    const auto & config = weights.config;
    auto x = Conv1dModule({config.num_mels, config.upsample_initial_channel, 7, 1, 3, 1, true})
                 .build(ctx, mel, weights.conv_pre);
    if (conditioning != nullptr) {
        if (!weights.cond.has_value()) {
            throw std::runtime_error("HiFi-GAN conditioning input provided without conditioning weights");
        }
        auto g = align_conditioning_frames(ctx, *conditioning, x.shape.dims[2]);
        g = Conv1dModule({
                config.global_conditioning.channels,
                config.upsample_initial_channel,
                1,
                1,
                0,
                1,
                config.global_conditioning.use_bias})
                .build(ctx, g, *weights.cond);
        x = AddModule().build(ctx, x, g);
    }
    const int64_t num_kernels = static_cast<int64_t>(config.resblock_kernel_sizes.size());
    for (int64_t up = 0; up < static_cast<int64_t>(config.upsample_rates.size()); ++up) {
        const int64_t in_channels = config.upsample_initial_channel / (int64_t{1} << up);
        const int64_t out_channels = config.upsample_initial_channel / (int64_t{1} << (up + 1));
        const int64_t kernel = config.upsample_kernel_sizes[static_cast<size_t>(up)];
        const int64_t rate = config.upsample_rates[static_cast<size_t>(up)];
        const int64_t padding = (kernel - rate) / 2;
        x = LeakyReluModule({config.leaky_relu_slope}).build(ctx, x);
        const int64_t module_padding = config.lower_padded_conv_transpose_as_crop ? 0 : padding;
        x = ConvTranspose1dModule({
                in_channels,
                out_channels,
                kernel,
                static_cast<int>(rate),
                static_cast<int>(module_padding),
                1,
                true})
                .build(ctx, x, weights.ups[static_cast<size_t>(up)]);
        if (config.lower_padded_conv_transpose_as_crop && padding > 0) {
            const int64_t cropped_frames = x.shape.dims[2] - 2 * padding;
            if (cropped_frames <= 0) {
                throw std::runtime_error("HiFi-GAN padded ConvTranspose1d crop would produce empty output");
            }
            x = SliceModule({2, padding, cropped_frames}).build(ctx, x);
        }
        if (source != nullptr) {
            auto x_source = Conv1dModule({
                    1,
                    out_channels,
                    weights.noise_convs[static_cast<size_t>(up)].weight.shape.dims[2],
                    static_cast<int>(up + 1 < static_cast<int64_t>(config.upsample_rates.size())
                            ? std::accumulate(
                                  config.upsample_rates.begin() + static_cast<std::ptrdiff_t>(up + 1),
                                  config.upsample_rates.end(),
                                  int64_t{1},
                                  std::multiplies<int64_t>{})
                            : 1),
                    static_cast<int>(up + 1 < static_cast<int64_t>(config.upsample_rates.size())
                            ? std::accumulate(
                                  config.upsample_rates.begin() + static_cast<std::ptrdiff_t>(up + 1),
                                  config.upsample_rates.end(),
                                  int64_t{1},
                                  std::multiplies<int64_t>{}) /
                                  2
                            : 0),
                    1,
                    true})
                    .build(ctx, *source, weights.noise_convs[static_cast<size_t>(up)]);
            const int64_t frames = std::min(x.shape.dims[2], x_source.shape.dims[2]);
            x = crop_time_to(ctx, x, frames);
            x_source = crop_time_to(ctx, x_source, frames);
            x = AddModule().build(ctx, x, x_source);
        }
        core::TensorValue sum;
        for (int64_t kernel_index = 0; kernel_index < num_kernels; ++kernel_index) {
            const int64_t block_index = up * num_kernels + kernel_index;
            auto block = hifigan_resblock(
                ctx,
                x,
                weights.resblocks[static_cast<size_t>(block_index)],
                out_channels,
                config.resblock_kernel_sizes[static_cast<size_t>(kernel_index)],
                config.resblock_dilation_sizes[static_cast<size_t>(kernel_index)],
                config.resblock_kind,
                config.leaky_relu_slope);
            sum = kernel_index == 0 ? block : AddModule().build(ctx, sum, block);
        }
        x = core::wrap_tensor(
            ggml_scale(ctx.ggml, core::ensure_backend_addressable_layout(ctx, sum).tensor,
                1.0F / static_cast<float>(num_kernels)),
            sum.shape,
            GGML_TYPE_F32);
    }
    x = LeakyReluModule({config.post_leaky_relu_slope}).build(ctx, x);
    x = Conv1dModule({
            config.upsample_initial_channel / (int64_t{1} << static_cast<int64_t>(config.upsample_rates.size())),
            config.output_channels,
            7,
            1,
            3,
            1,
            config.conv_post_use_bias})
            .build(ctx, x, weights.conv_post);
    return TanhModule().build(ctx, x);
}

int64_t total_upsample(const HifiGanVocoderConfig & config) {
    return std::accumulate(
        config.upsample_rates.begin(),
        config.upsample_rates.end(),
        int64_t{1},
        std::multiplies<int64_t>{});
}

std::vector<float> make_harmonic_source(
    const HifiGanVocoderWeights & weights,
    const std::vector<float> & f0,
    int64_t frames,
    uint64_t seed,
    uint64_t prior_noise_values) {
    constexpr float kPi = 3.14159265358979323846F;
    constexpr float kTwoPi = 2.0F * kPi;
    const int64_t upp = total_upsample(weights.config);
    if (static_cast<int64_t>(f0.size()) != frames) {
        throw std::runtime_error("HiFi-GAN source conditioning F0 shape mismatch");
    }
    const int64_t samples = frames * upp;
    const int64_t harmonics = weights.config.source.harmonic_num + 1;
    const size_t phase_count = static_cast<size_t>(harmonics);
    const size_t gaussian_count = static_cast<size_t>(harmonics * samples);
    std::vector<float> phase_uniform =
        sampling::generate_torch_cuda_uniform(phase_count, seed, prior_noise_values);
    std::vector<float> gaussian = sampling::generate_torch_cuda_randn(
        gaussian_count,
        seed,
        sampling::TorchRandnPrecision::Float32,
        prior_noise_values + static_cast<uint64_t>(phase_uniform.size()));
    std::vector<float> phase(static_cast<size_t>(harmonics), 0.0F);
    for (int64_t h = 0; h < harmonics; ++h) {
        phase[static_cast<size_t>(h)] = phase_uniform[static_cast<size_t>(h)];
    }
    phase[0] = 0.0F;

    float linear_bias = 0.0F;
    if (weights.source_linear.bias.has_value()) {
        const auto bias_values = core::read_tensor_f32(weights.source_linear.bias->tensor);
        linear_bias = bias_values[0];
    }
    const auto linear_weight = core::read_tensor_f32(weights.source_linear.weight.tensor);
    std::vector<float> rad_frame(static_cast<size_t>(frames * harmonics), 0.0F);
    for (int64_t t = 0; t < frames; ++t) {
        for (int64_t h = 0; h < harmonics; ++h) {
            float rad = f0[static_cast<size_t>(t)] * static_cast<float>(h + 1) /
                static_cast<float>(weights.config.sampling_rate);
            rad -= std::floor(rad);
            if (t == 0) {
                rad += phase[static_cast<size_t>(h)];
            }
            rad_frame[static_cast<size_t>(t * harmonics + h)] = rad;
        }
    }

    std::vector<float> cumulative(static_cast<size_t>(frames * harmonics), 0.0F);
    for (int64_t h = 0; h < harmonics; ++h) {
        float sum = 0.0F;
        for (int64_t t = 0; t < frames; ++t) {
            sum += rad_frame[static_cast<size_t>(t * harmonics + h)];
            cumulative[static_cast<size_t>(t * harmonics + h)] = sum * static_cast<float>(upp);
        }
    }

    std::vector<float> cumulative_interp(static_cast<size_t>(samples * harmonics), 0.0F);
    for (int64_t sample = 0; sample < samples; ++sample) {
        const float pos = samples == 1 || frames == 1
            ? 0.0F
            : static_cast<float>(sample) * static_cast<float>(frames - 1) / static_cast<float>(samples - 1);
        const int64_t left = static_cast<int64_t>(std::floor(pos));
        const int64_t right = std::min<int64_t>(left + 1, frames - 1);
        const float mix = pos - static_cast<float>(left);
        for (int64_t h = 0; h < harmonics; ++h) {
            const float a = cumulative[static_cast<size_t>(left * harmonics + h)];
            const float b = cumulative[static_cast<size_t>(right * harmonics + h)];
            float value = a + (b - a) * mix;
            value -= std::floor(value);
            cumulative_interp[static_cast<size_t>(sample * harmonics + h)] = value;
        }
    }

    std::vector<float> rad_nearest(static_cast<size_t>(samples * harmonics), 0.0F);
    for (int64_t sample = 0; sample < samples; ++sample) {
        const int64_t frame = std::min<int64_t>(sample / upp, frames - 1);
        for (int64_t h = 0; h < harmonics; ++h) {
            rad_nearest[static_cast<size_t>(sample * harmonics + h)] =
                rad_frame[static_cast<size_t>(frame * harmonics + h)];
        }
    }

    std::vector<float> sine_phase(static_cast<size_t>(harmonics), 0.0F);
    std::vector<float> source(static_cast<size_t>(samples), 0.0F);
    for (int64_t t = 0; t < samples; ++t) {
        const float base_f0 = f0[static_cast<size_t>(t / upp)];
        const float uv = base_f0 > weights.config.source.voiced_threshold ? 1.0F : 0.0F;
        float sum = linear_bias;
        for (int64_t h = 0; h < harmonics; ++h) {
            float cumsum_shift = 0.0F;
            if (t > 0) {
                const float current = cumulative_interp[static_cast<size_t>(t * harmonics + h)];
                const float previous = cumulative_interp[static_cast<size_t>((t - 1) * harmonics + h)];
                cumsum_shift = current - previous < 0.0F ? -1.0F : 0.0F;
            }
            sine_phase[static_cast<size_t>(h)] += rad_nearest[static_cast<size_t>(t * harmonics + h)] + cumsum_shift;
            float wave = weights.config.source.amplitude * std::sin(sine_phase[static_cast<size_t>(h)] * kTwoPi);
            const float noise_amp = uv * weights.config.source.noise_std +
                (1.0F - uv) * weights.config.source.amplitude / 3.0F;
            wave = wave * uv + noise_amp * gaussian[static_cast<size_t>(t * harmonics + h)];
            sum += wave * linear_weight[static_cast<size_t>(h)];
        }
        source[static_cast<size_t>(t)] = std::tanh(sum);
    }
    return source;
}

class HifiGanRunner {
public:
    HifiGanRunner(const HifiGanVocoderWeights & weights, const core::BackendConfig & backend)
        : weights_(weights), backend_(backend) {
        if (weights_.execution_context == nullptr) {
            throw std::runtime_error("HiFi-GAN runner requires execution context");
        }
    }

    ~HifiGanRunner() {
        release_graph();
    }

    HifiGanVocoderOutput run(
        const std::vector<float> & mel,
        int64_t frames,
        const std::vector<float> * source,
        const std::vector<float> * conditioning,
        int64_t conditioning_frames) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (frames <= 0 || static_cast<int64_t>(mel.size()) != weights_.config.num_mels * frames) {
            throw std::runtime_error("HiFi-GAN mel size mismatch");
        }
        if (conditioning != nullptr &&
            (conditioning_frames <= 0 ||
                static_cast<int64_t>(conditioning->size()) !=
                    weights_.config.global_conditioning.channels * conditioning_frames)) {
            throw std::runtime_error("HiFi-GAN conditioning size mismatch");
        }
        ensure_graph(frames, source != nullptr, conditioning != nullptr ? conditioning_frames : 0);
        ggml_backend_tensor_set(mel_, mel.data(), 0, mel.size() * sizeof(float));
        if (source_ != nullptr) {
            if (source == nullptr || static_cast<int64_t>(source->size()) != source_->ne[0]) {
                throw std::runtime_error("HiFi-GAN source size mismatch");
            }
            ggml_backend_tensor_set(source_, source->data(), 0, source->size() * sizeof(float));
        }
        if (conditioning_ != nullptr) {
            if (conditioning == nullptr) {
                throw std::runtime_error("HiFi-GAN conditioning graph/input mismatch");
            }
            ggml_backend_tensor_set(conditioning_, conditioning->data(), 0, conditioning->size() * sizeof(float));
        }
        core::set_backend_threads(weights_.execution_context->backend(), weights_.execution_context->config().threads);
        if (engine::core::compute_backend_graph(weights_.execution_context->backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for HiFi-GAN");
        }
        ggml_backend_synchronize(weights_.execution_context->backend());
        HifiGanVocoderOutput out;
        out.waveform = core::read_tensor_f32(output_);
        out.batch = 1;
        out.samples = static_cast<int64_t>(out.waveform.size());
        out.sample_rate = weights_.config.sampling_rate;
        return out;
    }

    void release_runtime_graph() {
        std::lock_guard<std::mutex> lock(mutex_);
        release_graph();
    }

private:
    void release_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(weights_.execution_context->backend(), graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
            input_buffer_ = nullptr;
        }
        input_ctx_.reset();
        graph_ctx_.reset();
        mel_ = nullptr;
        source_ = nullptr;
        conditioning_ = nullptr;
        output_ = nullptr;
        frames_ = 0;
        has_source_ = false;
        conditioning_frames_ = 0;
    }

    void ensure_graph(int64_t frames, bool has_source, int64_t conditioning_frames) {
        if (graph_ != nullptr && frames_ == frames && has_source_ == has_source &&
            conditioning_frames_ == conditioning_frames) {
            return;
        }
        release_graph();
        const auto build_start = Clock::now();
        ggml_init_params graph_params{512ull * 1024ull * 1024ull, nullptr, true};
        graph_ctx_.reset(ggml_init(graph_params));
        if (graph_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize HiFi-GAN graph context");
        }
        ggml_init_params input_params{32ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize HiFi-GAN input context");
        }
        core::ModuleBuildContext build{graph_ctx_.get(), "framework.hifigan", weights_.execution_context->backend_type()};
        core::ModuleBuildContext input_build{
            input_ctx_.get(),
            "framework.hifigan.inputs",
            weights_.execution_context->backend_type()};
        mel_ = core::make_tensor(
            input_build,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, weights_.config.num_mels, frames})).tensor;
        ggml_set_input(mel_);
        core::TensorValue source_value;
        const core::TensorValue * source_ptr = nullptr;
        core::TensorValue conditioning_value;
        const core::TensorValue * conditioning_ptr = nullptr;
        if (has_source) {
            source_ = core::make_tensor(
                input_build,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, 1, frames * total_upsample(weights_.config)})).tensor;
            ggml_set_input(source_);
            source_value = core::wrap_tensor(
                source_,
                core::TensorShape::from_dims({1, 1, frames * total_upsample(weights_.config)}),
                GGML_TYPE_F32);
            source_ptr = &source_value;
        }
        if (conditioning_frames > 0) {
            conditioning_ = core::make_tensor(
                input_build,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({
                    1,
                    weights_.config.global_conditioning.channels,
                    conditioning_frames,
                })).tensor;
            ggml_set_input(conditioning_);
            conditioning_value = core::wrap_tensor(
                conditioning_,
                core::TensorShape::from_dims({1, weights_.config.global_conditioning.channels, conditioning_frames}),
                GGML_TYPE_F32);
            conditioning_ptr = &conditioning_value;
        }
        auto x = core::wrap_tensor(
            mel_,
            core::TensorShape::from_dims({1, weights_.config.num_mels, frames}),
            GGML_TYPE_F32);
        x = hifigan_graph(build, weights_, x, source_ptr, conditioning_ptr);
        output_ = core::ensure_backend_addressable_layout(build, x).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(graph_ctx_.get(), 131072, false);
        ggml_build_forward_expand(graph_, output_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), weights_.execution_context->backend());
        if (input_buffer_ == nullptr) {
            release_graph();
            throw std::runtime_error("failed to allocate HiFi-GAN input buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_.execution_context->backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate HiFi-GAN graph memory");
        }
        frames_ = frames;
        has_source_ = has_source;
        conditioning_frames_ = conditioning_frames;
        engine::debug::timing_log_scalar(
            "framework.hifigan.graph.build_ms",
            engine::debug::elapsed_ms(build_start, Clock::now()));
    }

    const HifiGanVocoderWeights & weights_;
    core::BackendConfig backend_;
    std::mutex mutex_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * mel_ = nullptr;
    ggml_tensor * source_ = nullptr;
    ggml_tensor * conditioning_ = nullptr;
    ggml_tensor * output_ = nullptr;
    int64_t frames_ = 0;
    bool has_source_ = false;
    int64_t conditioning_frames_ = 0;
};

}  // namespace

struct HifiGanVocoderComponent::State {
    std::unique_ptr<HifiGanRunner> runner;
};

HifiGanVocoderComponent HifiGanVocoderComponent::load_from_tensor_source(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    HifiGanVocoderConfig config) {
    validate_config(config);
    if (source == nullptr) {
        throw std::runtime_error("HiFi-GAN component requires tensor source");
    }
    auto weights = std::make_shared<HifiGanVocoderWeights>();
    weights->config = std::move(config);
    weights->execution_context = std::make_shared<core::ExecutionContext>(backend);
    weights->store = std::make_shared<core::BackendWeightStore>(
        weights->execution_context->backend(),
        weights->execution_context->backend_type(),
        "framework.hifigan.weights",
        512ull * 1024ull * 1024ull);

    for (const auto & tensor : source->tensors()) {
        if (!weights->config.tensor_prefix.empty() &&
            tensor.name.rfind(weights->config.tensor_prefix + ".", 0) != 0) {
            continue;
        }
        weights->parameter_count += tensor_elements(tensor.shape);
        ++weights->loaded_tensor_count;
    }

    weights->conv_pre = load_conv1d(
        *weights->store,
        *source,
        weights->config,
        "conv_pre",
        weights->config.upsample_initial_channel,
        weights->config.num_mels,
        7);
    if (weights->config.global_conditioning.channels > 0) {
        weights->cond = load_conv1d(
            *weights->store,
            *source,
            weights->config,
            weights->config.global_conditioning.prefix,
            weights->config.upsample_initial_channel,
            weights->config.global_conditioning.channels,
            1,
            weights->config.global_conditioning.use_bias);
    }
    const int64_t upsample_count = static_cast<int64_t>(weights->config.upsample_rates.size());
    const int64_t num_kernels = static_cast<int64_t>(weights->config.resblock_kernel_sizes.size());
    weights->ups.reserve(static_cast<size_t>(upsample_count));
    weights->resblocks.reserve(static_cast<size_t>(upsample_count * num_kernels));
    if (weights->config.source.enabled) {
        weights->noise_convs.reserve(static_cast<size_t>(upsample_count));
    }
    for (int64_t up = 0; up < upsample_count; ++up) {
        const int64_t in_channels = weights->config.upsample_initial_channel / (int64_t{1} << up);
        const int64_t out_channels = weights->config.upsample_initial_channel / (int64_t{1} << (up + 1));
        const int64_t kernel = weights->config.upsample_kernel_sizes[static_cast<size_t>(up)];
        weights->ups.push_back(load_conv_transpose1d(
            *weights->store,
            *source,
            weights->config,
            "ups." + std::to_string(up),
            in_channels,
            out_channels,
            kernel));
        if (weights->config.source.enabled) {
            const int64_t stride_f0 = up + 1 < upsample_count
                ? std::accumulate(
                      weights->config.upsample_rates.begin() + static_cast<std::ptrdiff_t>(up + 1),
                      weights->config.upsample_rates.end(),
                      int64_t{1},
                      std::multiplies<int64_t>{})
                : 1;
            weights->noise_convs.push_back(load_conv1d(
                *weights->store,
                *source,
                weights->config,
                weights->config.source.noise_conv_prefix + "." + std::to_string(up),
                out_channels,
                1,
                up + 1 < upsample_count ? stride_f0 * 2 : 1));
        }
        for (int64_t kernel_index = 0; kernel_index < num_kernels; ++kernel_index) {
            const int64_t block_index = up * num_kernels + kernel_index;
            weights->resblocks.push_back(load_resblock(
                *weights->store,
                *source,
                weights->config,
                "resblocks." + std::to_string(block_index),
                out_channels,
                weights->config.resblock_kernel_sizes[static_cast<size_t>(kernel_index)],
                weights->config.resblock_dilation_sizes[static_cast<size_t>(kernel_index)]));
        }
    }
    const int64_t post_channels = weights->config.upsample_initial_channel / (int64_t{1} << upsample_count);
    weights->conv_post = load_conv1d(
        *weights->store,
        *source,
        weights->config,
        "conv_post",
        weights->config.output_channels,
        post_channels,
        7,
        weights->config.conv_post_use_bias);
    if (weights->config.source.enabled) {
        weights->source_linear = load_linear(
            *weights->store,
            *source,
            weights->config,
            weights->config.source.linear_prefix,
            1,
            weights->config.source.harmonic_num + 1,
            true);
    }

    weights->store->upload();
    if (weights->config.release_source_storage_after_load) {
        source->release_storage();
    }
    return HifiGanVocoderComponent(std::move(weights), backend);
}

HifiGanVocoderComponent::HifiGanVocoderComponent(
    std::shared_ptr<const HifiGanVocoderWeights> weights,
    core::BackendConfig backend)
    : weights_(std::move(weights)),
      backend_(backend),
      state_(std::make_shared<State>()) {
    if (weights_ == nullptr) {
        throw std::runtime_error("HiFi-GAN component requires weights");
    }
    state_->runner = std::make_unique<HifiGanRunner>(*weights_, backend_);
}

const core::BackendConfig & HifiGanVocoderComponent::backend() const noexcept {
    return backend_;
}

const std::shared_ptr<const HifiGanVocoderWeights> & HifiGanVocoderComponent::weights() const noexcept {
    return weights_;
}

int64_t HifiGanVocoderComponent::sample_rate() const noexcept {
    return weights_ == nullptr ? 0 : weights_->config.sampling_rate;
}

int64_t HifiGanVocoderComponent::num_mels() const noexcept {
    return weights_ == nullptr ? 0 : weights_->config.num_mels;
}

int64_t HifiGanVocoderComponent::loaded_tensor_count() const noexcept {
    return weights_ == nullptr ? 0 : weights_->loaded_tensor_count;
}

int64_t HifiGanVocoderComponent::parameter_count() const noexcept {
    return weights_ == nullptr ? 0 : weights_->parameter_count;
}

HifiGanVocoderOutput HifiGanVocoderComponent::synthesize(const HifiGanVocoderRequest & request) const {
    if (weights_ == nullptr || state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("HiFi-GAN component is not initialized");
    }
    if (request.mel == nullptr) {
        throw std::runtime_error("HiFi-GAN request requires mel input");
    }
    if (weights_->config.source.enabled && request.f0 == nullptr) {
        throw std::runtime_error("HiFi-GAN source-conditioned variant requires F0 input");
    }
    if (!weights_->config.source.enabled && request.f0 != nullptr) {
        throw std::runtime_error("HiFi-GAN request provided F0 for variant without source conditioning");
    }
    if (weights_->config.global_conditioning.channels > 0 && request.conditioning == nullptr) {
        throw std::runtime_error("HiFi-GAN globally conditioned variant requires conditioning input");
    }
    if (weights_->config.global_conditioning.channels == 0 && request.conditioning != nullptr) {
        throw std::runtime_error("HiFi-GAN request provided conditioning for variant without conditioning weights");
    }
    std::vector<float> source;
    if (request.f0 != nullptr) {
        source = make_harmonic_source(*weights_, *request.f0, request.frames, request.seed, request.prior_noise_values);
    }
    return state_->runner->run(
        *request.mel,
        request.frames,
        request.f0 != nullptr ? &source : nullptr,
        request.conditioning,
        request.conditioning_frames);
}

HifiGanVocoderOutput HifiGanVocoderComponent::synthesize(
    const std::vector<float> & mel,
    int64_t frames) const {
    return synthesize(HifiGanVocoderRequest{&mel, frames});
}

HifiGanVocoderOutput HifiGanVocoderComponent::synthesize_with_f0(
    const std::vector<float> & mel,
    const std::vector<float> & f0,
    int64_t frames,
    uint64_t seed,
    uint64_t prior_noise_values) const {
    HifiGanVocoderRequest request;
    request.mel = &mel;
    request.frames = frames;
    request.f0 = &f0;
    request.seed = seed;
    request.prior_noise_values = prior_noise_values;
    return synthesize(request);
}

void HifiGanVocoderComponent::release_runtime_graph() const {
    if (state_ != nullptr && state_->runner != nullptr) {
        state_->runner->release_runtime_graph();
    }
}

}  // namespace engine::modules
