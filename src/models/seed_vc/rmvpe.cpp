#include "engine/models/seed_vc/rmvpe.h"

#include "engine/models/seed_vc/assets.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::seed_vc {
namespace {

using engine::core::TensorShape;
using engine::core::TensorValue;

constexpr int64_t kRmvpeMelBins = 128;
constexpr int64_t kRmvpeFeatureDim = 384;
constexpr int64_t kRmvpeHiddenDim = 256;
constexpr int64_t kRmvpeClasses = 360;
constexpr int64_t kRmvpeFeatureActiveFrames = 4096;
constexpr int64_t kRmvpeFeatureOverlapFrames = 2048;
constexpr int64_t kRmvpeFeatureGraphFrames = kRmvpeFeatureActiveFrames + 2 * kRmvpeFeatureOverlapFrames;
constexpr int64_t kRmvpeGruChunkFrames = 512;

struct RmvpeMelFilterbank {
    std::vector<float> values;
    std::vector<int64_t> starts;
    std::vector<int64_t> ends;
};

struct RmvpePreparedMel {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t padded_frames = 0;
};

struct RmvpeGruOutputs {
    TensorValue sequence;
    TensorValue final_hidden;
};

struct SeedVcRmvpeWeights {
    std::shared_ptr<engine::core::ExecutionContext> execution_context;
    std::shared_ptr<engine::core::BackendWeightStore> store;
    std::unordered_map<std::string, TensorValue> tensors;
};

SeedVcRmvpeWeights load_rmvpe_weights(
    std::shared_ptr<const engine::assets::TensorSource> source,
    engine::core::BackendConfig backend,
    engine::assets::TensorStorageType storage_type) {
    if (source == nullptr) {
        throw std::runtime_error("Seed-VC RMVPE requires weights");
    }
    SeedVcRmvpeWeights weights;
    weights.execution_context = std::make_shared<engine::core::ExecutionContext>(backend);
    weights.store = std::make_shared<engine::core::BackendWeightStore>(
        weights.execution_context->backend(),
        weights.execution_context->backend_type(),
        "seed_vc.rmvpe.weights",
        256ull * 1024ull * 1024ull);
    const auto tensors = source->tensors();
    weights.tensors.reserve(tensors.size());
    for (const auto & tensor : tensors) {
        if (tensor.name.size() >= 20 &&
            tensor.name.compare(tensor.name.size() - 20, 20, ".num_batches_tracked") == 0) {
            continue;
        }
        weights.tensors.emplace(
            tensor.name,
            weights.store->load_tensor(*source, tensor.name, storage_type, tensor.shape));
    }
    weights.store->upload();
    return weights;
}

const TensorValue & require_tensor(const SeedVcRmvpeWeights & weights, const std::string & name) {
    const auto it = weights.tensors.find(name);
    if (it == weights.tensors.end()) {
        throw std::runtime_error("Seed-VC RMVPE missing tensor: " + name);
    }
    return it->second;
}

TensorValue contiguous(engine::core::ModuleBuildContext & ctx, const TensorValue & value) {
    return engine::core::ensure_backend_addressable_layout(ctx, value);
}

TensorValue add(engine::core::ModuleBuildContext & ctx, const TensorValue & lhs, const TensorValue & rhs) {
    return engine::modules::AddModule{}.build(ctx, lhs, rhs);
}

TensorValue mul(engine::core::ModuleBuildContext & ctx, const TensorValue & lhs, const TensorValue & rhs) {
    return engine::modules::MulModule{}.build(ctx, lhs, rhs);
}

TensorValue sub(engine::core::ModuleBuildContext & ctx, const TensorValue & lhs, const TensorValue & rhs) {
    return engine::core::wrap_tensor(
        ggml_sub(ctx.ggml, contiguous(ctx, lhs).tensor, contiguous(ctx, rhs).tensor),
        lhs.shape,
        GGML_TYPE_F32);
}

TensorValue sigmoid(engine::core::ModuleBuildContext & ctx, const TensorValue & x) {
    return engine::modules::SigmoidModule{}.build(ctx, x);
}

TensorValue tanh_value(engine::core::ModuleBuildContext & ctx, const TensorValue & x) {
    return engine::modules::TanhModule{}.build(ctx, x);
}

TensorValue slice_axis(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    int axis,
    int64_t start,
    int64_t length) {
    return engine::modules::SliceModule({axis, start, length}).build(ctx, input);
}

engine::modules::Conv2dWeights conv2d_weights(
    const SeedVcRmvpeWeights & weights,
    const std::string & prefix,
    bool bias) {
    return {
        require_tensor(weights, prefix + ".weight"),
        bias ? std::optional<TensorValue>(require_tensor(weights, prefix + ".bias")) : std::nullopt};
}

engine::modules::LinearWeights linear_weights(
    const SeedVcRmvpeWeights & weights,
    const std::string & prefix,
    bool bias) {
    return {
        require_tensor(weights, prefix + ".weight"),
        bias ? std::optional<TensorValue>(require_tensor(weights, prefix + ".bias")) : std::nullopt};
}

engine::modules::LinearWeights linear_weights(
    const SeedVcRmvpeWeights & weights,
    const std::string & weight_name,
    const std::string & bias_name,
    bool bias) {
    return {
        require_tensor(weights, weight_name),
        bias ? std::optional<TensorValue>(require_tensor(weights, bias_name)) : std::nullopt};
}

TensorValue batch_norm2d(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const SeedVcRmvpeWeights & weights,
    const std::string & prefix) {
    const int64_t channels = input.shape.dims[1];
    const auto weight = engine::core::reshape_tensor(
        ctx,
        require_tensor(weights, prefix + ".weight"),
        TensorShape::from_dims({1, channels, 1, 1}));
    const auto bias = engine::core::reshape_tensor(
        ctx,
        require_tensor(weights, prefix + ".bias"),
        TensorShape::from_dims({1, channels, 1, 1}));
    const auto mean = engine::core::reshape_tensor(
        ctx,
        require_tensor(weights, prefix + ".running_mean"),
        TensorShape::from_dims({1, channels, 1, 1}));
    const auto var = engine::core::reshape_tensor(
        ctx,
        require_tensor(weights, prefix + ".running_var"),
        TensorShape::from_dims({1, channels, 1, 1}));
    const auto input_cont = contiguous(ctx, input);
    const auto weight_rep = engine::core::wrap_tensor(
        ggml_repeat(ctx.ggml, weight.tensor, input_cont.tensor),
        input.shape,
        GGML_TYPE_F32);
    const auto bias_rep = engine::core::wrap_tensor(
        ggml_repeat(ctx.ggml, bias.tensor, input_cont.tensor),
        input.shape,
        GGML_TYPE_F32);
    const auto mean_rep = engine::core::wrap_tensor(
        ggml_repeat(ctx.ggml, mean.tensor, input_cont.tensor),
        input.shape,
        GGML_TYPE_F32);
    const auto denom = engine::core::wrap_tensor(
        ggml_sqrt(ctx.ggml, ggml_scale_bias(ctx.ggml, var.tensor, 1.0F, 1.0e-5F)),
        var.shape,
        GGML_TYPE_F32);
    const auto denom_rep = engine::core::wrap_tensor(
        ggml_repeat(ctx.ggml, denom.tensor, input_cont.tensor),
        input.shape,
        GGML_TYPE_F32);
    auto centered = engine::core::wrap_tensor(
        ggml_sub(ctx.ggml, input_cont.tensor, mean_rep.tensor),
        input.shape,
        GGML_TYPE_F32);
    auto normalized = engine::core::wrap_tensor(
        ggml_div(ctx.ggml, centered.tensor, denom_rep.tensor),
        input.shape,
        GGML_TYPE_F32);
    return add(ctx, mul(ctx, normalized, weight_rep), bias_rep);
}

TensorValue relu(engine::core::ModuleBuildContext & ctx, const TensorValue & input) {
    return engine::core::wrap_tensor(
        ggml_relu(ctx.ggml, contiguous(ctx, input).tensor),
        input.shape,
        GGML_TYPE_F32);
}

TensorValue conv_bn_relu(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const SeedVcRmvpeWeights & weights,
    const std::string & conv_prefix,
    const std::string & bn_prefix,
    int64_t in_channels,
    int64_t out_channels) {
    auto x = engine::modules::Conv2dModule({in_channels, out_channels, 3, 3, 1, 1, 1, 1, 1, 1, false})
                 .build(ctx, input, conv2d_weights(weights, conv_prefix, false));
    x = batch_norm2d(ctx, x, weights, bn_prefix);
    return relu(ctx, x);
}

TensorValue conv_block_res(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const SeedVcRmvpeWeights & weights,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels) {
    auto y = conv_bn_relu(ctx, input, weights, prefix + ".conv.0", prefix + ".conv.1", in_channels, out_channels);
    y = conv_bn_relu(ctx, y, weights, prefix + ".conv.3", prefix + ".conv.4", out_channels, out_channels);
    TensorValue residual = input;
    if (in_channels != out_channels) {
        residual = engine::modules::Conv2dModule({in_channels, out_channels, 1, 1, 1, 1, 0, 0, 1, 1, true})
                       .build(ctx, input, conv2d_weights(weights, prefix + ".shortcut", true));
    }
    return add(ctx, y, residual);
}

TensorValue avg_pool2d_2x2(engine::core::ModuleBuildContext & ctx, const TensorValue & input) {
    auto pooled = engine::core::wrap_tensor(
        ggml_pool_2d(ctx.ggml, contiguous(ctx, input).tensor, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0),
        TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], input.shape.dims[2] / 2, input.shape.dims[3] / 2}),
        GGML_TYPE_F32);
    return pooled;
}

TensorValue conv_transpose2d_pytorch_2x(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const SeedVcRmvpeWeights & weights,
    const std::string & prefix,
    int64_t out_channels) {
    const auto weight = require_tensor(weights, prefix + ".weight");
    auto full = engine::core::wrap_tensor(
        ggml_conv_transpose_2d_p0(ctx.ggml, contiguous(ctx, weight).tensor, contiguous(ctx, input).tensor, 2),
        TensorShape::from_dims({input.shape.dims[0], out_channels, input.shape.dims[2] * 2 + 1, input.shape.dims[3] * 2 + 1}),
        GGML_TYPE_F32);
    full = slice_axis(ctx, full, 2, 1, input.shape.dims[2] * 2);
    full = slice_axis(ctx, full, 3, 1, input.shape.dims[3] * 2);
    return full;
}

TensorValue decoder_upsample(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const SeedVcRmvpeWeights & weights,
    const std::string & prefix,
    int64_t out_channels) {
    auto x = conv_transpose2d_pytorch_2x(ctx, input, weights, prefix + ".conv1.0", out_channels);
    x = batch_norm2d(ctx, x, weights, prefix + ".conv1.1");
    return relu(ctx, x);
}

RmvpeGruOutputs build_gru_chunk_graph(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & input,
    const TensorValue & keep,
    const TensorValue & initial_hidden,
    const SeedVcRmvpeWeights & weights,
    const std::string & suffix,
    bool reverse) {
    const int64_t frames = input.shape.dims[0];
    const std::string base = "fc.0.gru.";
    const auto projected_input = engine::modules::LinearModule({kRmvpeFeatureDim, kRmvpeHiddenDim * 3, true}).build(
        ctx,
        input,
        linear_weights(weights, base + "weight_ih_l0" + suffix, base + "bias_ih_l0" + suffix, true));
    TensorValue hidden = initial_hidden;
    std::vector<TensorValue> steps(static_cast<size_t>(frames));
    for (int64_t index = 0; index < frames; ++index) {
        const int64_t t = reverse ? (frames - 1 - index) : index;
        const auto xi = slice_axis(ctx, projected_input, 0, t, 1);
        const auto hi = engine::modules::LinearModule({kRmvpeHiddenDim, kRmvpeHiddenDim * 3, true}).build(
            ctx,
            hidden,
            linear_weights(weights, base + "weight_hh_l0" + suffix, base + "bias_hh_l0" + suffix, true));
        const auto i_r = slice_axis(ctx, xi, 1, 0, kRmvpeHiddenDim);
        const auto i_z = slice_axis(ctx, xi, 1, kRmvpeHiddenDim, kRmvpeHiddenDim);
        const auto i_n = slice_axis(ctx, xi, 1, kRmvpeHiddenDim * 2, kRmvpeHiddenDim);
        const auto h_r = slice_axis(ctx, hi, 1, 0, kRmvpeHiddenDim);
        const auto h_z = slice_axis(ctx, hi, 1, kRmvpeHiddenDim, kRmvpeHiddenDim);
        const auto h_n = slice_axis(ctx, hi, 1, kRmvpeHiddenDim * 2, kRmvpeHiddenDim);
        const auto reset = sigmoid(ctx, add(ctx, i_r, h_r));
        const auto update = sigmoid(ctx, add(ctx, i_z, h_z));
        const auto candidate = tanh_value(ctx, add(ctx, i_n, mul(ctx, reset, h_n)));
        const auto updated = add(ctx, candidate, mul(ctx, update, sub(ctx, hidden, candidate)));
        const auto keep_t = slice_axis(ctx, keep, 0, t, 1);
        const auto keep_rep = engine::core::wrap_tensor(
            ggml_repeat(ctx.ggml, keep_t.tensor, hidden.tensor),
            hidden.shape,
            GGML_TYPE_F32);
        hidden = add(ctx, hidden, mul(ctx, keep_rep, sub(ctx, updated, hidden)));
        steps[static_cast<size_t>(t)] = hidden;
    }
    auto sequence = steps[0];
    for (int64_t t = 1; t < frames; ++t) {
        sequence = engine::modules::ConcatModule({0}).build(ctx, sequence, steps[static_cast<size_t>(t)]);
    }
    return {sequence, hidden};
}

TensorValue build_rmvpe_feature_graph(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & mel,
    const SeedVcRmvpeWeights & weights) {
    auto x = engine::modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, mel);
    x = engine::core::reshape_tensor(ctx, contiguous(ctx, x), TensorShape::from_dims({1, 1, x.shape.dims[1], x.shape.dims[2]}));
    x = batch_norm2d(ctx, x, weights, "unet.encoder.bn");

    std::vector<TensorValue> skips;
    skips.reserve(5);
    int64_t channels = 1;
    int64_t out_channels = 16;
    for (int layer = 0; layer < 5; ++layer) {
        const std::string base = "unet.encoder.layers." + std::to_string(layer);
        for (int block = 0; block < 4; ++block) {
            x = conv_block_res(ctx, x, weights, base + ".conv." + std::to_string(block), channels, out_channels);
            channels = out_channels;
        }
        skips.push_back(x);
        x = avg_pool2d_2x2(ctx, x);
        out_channels *= 2;
    }

    channels = 256;
    for (int layer = 0; layer < 4; ++layer) {
        const std::string base = "unet.intermediate.layers." + std::to_string(layer);
        for (int block = 0; block < 4; ++block) {
            x = conv_block_res(ctx, x, weights, base + ".conv." + std::to_string(block), channels, 512);
            channels = 512;
        }
    }

    channels = 512;
    for (int layer = 0; layer < 5; ++layer) {
        const int64_t dec_out = channels / 2;
        const std::string base = "unet.decoder.layers." + std::to_string(layer);
        x = decoder_upsample(ctx, x, weights, base, dec_out);
        auto skip = skips.back();
        skips.pop_back();
        x = engine::modules::ConcatModule({1}).build(ctx, x, skip);
        channels = dec_out * 2;
        for (int block = 0; block < 4; ++block) {
            x = conv_block_res(ctx, x, weights, base + ".conv2." + std::to_string(block), channels, dec_out);
            channels = dec_out;
        }
    }

    x = engine::modules::Conv2dModule({16, 3, 3, 3, 1, 1, 1, 1, 1, 1, true})
            .build(ctx, x, conv2d_weights(weights, "cnn", true));
    x = engine::modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
    return engine::core::reshape_tensor(ctx, contiguous(ctx, x), TensorShape::from_dims({x.shape.dims[1], kRmvpeFeatureDim}));
}

TensorValue build_rmvpe_head_graph(
    engine::core::ModuleBuildContext & ctx,
    const TensorValue & forward,
    const TensorValue & reverse,
    const SeedVcRmvpeWeights & weights) {
    auto gru = engine::modules::ConcatModule({1}).build(ctx, forward, reverse);
    auto logits = engine::modules::LinearModule({2 * kRmvpeHiddenDim, kRmvpeClasses, true})
                      .build(ctx, gru, linear_weights(weights, "fc.1", true));
    return sigmoid(ctx, logits);
}

std::vector<float> compute_rmvpe_log_mel(
    const std::vector<float> & waveform_16k,
    size_t threads) {
    constexpr int64_t kSampleRate = 16000;
    constexpr int64_t kNfft = 1024;
    constexpr int64_t kHop = 160;
    const engine::audio::STFTConfig stft_config{
        kNfft,
        kHop,
        kNfft,
        true,
        engine::audio::STFTPadMode::Reflect,
        engine::audio::STFTFamily::Kokoro,
    };
    const auto & window = engine::audio::get_cached_stft_window(stft_config);
    const auto magnitude = engine::audio::STFT().compute_magnitude(
        waveform_16k,
        window,
        1,
        static_cast<int64_t>(waveform_16k.size()),
        stft_config,
        threads);
    const int64_t freq_bins = magnitude.shape[1];
    const int64_t frames = magnitude.shape[2];
    static const RmvpeMelFilterbank mel_filter = [] {
        constexpr int64_t freq_bins = kNfft / 2 + 1;
        std::vector<double> mel_points(static_cast<size_t>(kRmvpeMelBins + 2), 0.0);
        const auto hz_to_mel = [](double hz) {
            return 2595.0 * std::log10(1.0 + hz / 700.0);
        };
        const auto mel_to_hz = [](double mel) {
            return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
        };
        const double mel_min = hz_to_mel(30.0);
        const double mel_max = hz_to_mel(8000.0);
        for (int64_t i = 0; i < kRmvpeMelBins + 2; ++i) {
            mel_points[static_cast<size_t>(i)] =
                mel_min + (mel_max - mel_min) * static_cast<double>(i) / static_cast<double>(kRmvpeMelBins + 1);
        }
        std::vector<double> hz_points(static_cast<size_t>(kRmvpeMelBins + 2), 0.0);
        for (int64_t i = 0; i < kRmvpeMelBins + 2; ++i) {
            hz_points[static_cast<size_t>(i)] = mel_to_hz(mel_points[static_cast<size_t>(i)]);
        }
        RmvpeMelFilterbank filter;
        filter.values.assign(static_cast<size_t>(kRmvpeMelBins * freq_bins), 0.0F);
        filter.starts.assign(static_cast<size_t>(kRmvpeMelBins), freq_bins);
        filter.ends.assign(static_cast<size_t>(kRmvpeMelBins), 0);
        for (int64_t mel = 0; mel < kRmvpeMelBins; ++mel) {
            const double left = hz_points[static_cast<size_t>(mel)];
            const double center = hz_points[static_cast<size_t>(mel + 1)];
            const double right = hz_points[static_cast<size_t>(mel + 2)];
            const double lower_width = std::max(center - left, 1.0e-12);
            const double upper_width = std::max(right - center, 1.0e-12);
            for (int64_t freq = 0; freq < freq_bins; ++freq) {
                const double hz = static_cast<double>(kSampleRate) * 0.5 *
                    static_cast<double>(freq) / static_cast<double>(freq_bins - 1);
                const double lower = (hz - left) / lower_width;
                const double upper = (right - hz) / upper_width;
                filter.values[static_cast<size_t>(mel * freq_bins + freq)] =
                    static_cast<float>(std::max(0.0, std::min(lower, upper)));
            }
            const double enorm = 2.0 / std::max(right - left, 1.0e-12);
            for (int64_t freq = 0; freq < freq_bins; ++freq) {
                auto & value = filter.values[static_cast<size_t>(mel * freq_bins + freq)];
                value *= static_cast<float>(enorm);
                if (value != 0.0F) {
                    filter.starts[static_cast<size_t>(mel)] = std::min(filter.starts[static_cast<size_t>(mel)], freq);
                    filter.ends[static_cast<size_t>(mel)] = std::max(filter.ends[static_cast<size_t>(mel)], freq + 1);
                }
            }
            if (filter.starts[static_cast<size_t>(mel)] == freq_bins) {
                filter.starts[static_cast<size_t>(mel)] = 0;
            }
        }
        return filter;
    }();
    std::vector<float> out(static_cast<size_t>(kRmvpeMelBins * frames), 0.0F);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if(kRmvpeMelBins * frames >= 4096)
#endif
    for (int64_t mel = 0; mel < kRmvpeMelBins; ++mel) {
        for (int64_t frame = 0; frame < frames; ++frame) {
            float sum = 0.0F;
            const int64_t start = mel_filter.starts[static_cast<size_t>(mel)];
            const int64_t end = mel_filter.ends[static_cast<size_t>(mel)];
            for (int64_t freq = start; freq < end; ++freq) {
                const float mag = magnitude.values[static_cast<size_t>(freq * frames + frame)];
                sum += mel_filter.values[static_cast<size_t>(mel * freq_bins + freq)] * mag;
            }
            out[static_cast<size_t>(mel * frames + frame)] = std::log(std::max(sum, 1.0e-5F));
        }
    }
    return out;
}

std::vector<float> pad_rmvpe_mel_time_axis(
    const std::vector<float> & mel,
    int64_t frames,
    int64_t padded_frames) {
    if (static_cast<int64_t>(mel.size()) != kRmvpeMelBins * frames || frames <= 0 || padded_frames < frames) {
        throw std::runtime_error("Seed-VC RMVPE mel padding shape mismatch");
    }
    std::vector<float> padded(static_cast<size_t>(kRmvpeMelBins * padded_frames), 0.0F);
    for (int64_t channel = 0; channel < kRmvpeMelBins; ++channel) {
        std::copy_n(
            mel.begin() + static_cast<std::ptrdiff_t>(channel * frames),
            frames,
            padded.begin() + static_cast<std::ptrdiff_t>(channel * padded_frames));
    }
    return padded;
}

RmvpePreparedMel prepare_rmvpe_mel(const std::vector<float> & waveform_16k, size_t threads) {
    auto mel = compute_rmvpe_log_mel(waveform_16k, threads);
    const int64_t frames = static_cast<int64_t>(mel.size()) / kRmvpeMelBins;
    const int64_t padded_frames = 32 * ((frames - 1) / 32 + 1);
    return {pad_rmvpe_mel_time_axis(mel, frames, padded_frames), frames, padded_frames};
}

void copy_mel_window(
    const std::vector<float> & mel,
    int64_t padded_frames,
    int64_t graph_frames,
    int64_t start,
    int64_t frames,
    std::vector<float> & out) {
    if (start < 0 || frames < 0 || start + frames > padded_frames) {
        throw std::runtime_error("Seed-VC RMVPE mel window is out of range");
    }
    out.assign(static_cast<size_t>(kRmvpeMelBins * graph_frames), 0.0F);
    for (int64_t mel_bin = 0; mel_bin < kRmvpeMelBins; ++mel_bin) {
        std::copy_n(
            mel.begin() + static_cast<std::ptrdiff_t>(mel_bin * padded_frames + start),
            frames,
            out.begin() + static_cast<std::ptrdiff_t>(mel_bin * graph_frames));
    }
}

void copy_feature_rows(
    const std::vector<float> & source,
    int64_t source_start,
    int64_t rows,
    std::vector<float> & destination,
    int64_t destination_start) {
    for (int64_t row = 0; row < rows; ++row) {
        std::copy_n(
            source.begin() + static_cast<std::ptrdiff_t>((source_start + row) * kRmvpeFeatureDim),
            kRmvpeFeatureDim,
            destination.begin() + static_cast<std::ptrdiff_t>((destination_start + row) * kRmvpeFeatureDim));
    }
}

void fill_gru_chunk(
    const std::vector<float> & features,
    int64_t start,
    int64_t frames,
    std::vector<float> & chunk,
    std::vector<float> & keep) {
    chunk.assign(static_cast<size_t>(kRmvpeGruChunkFrames * kRmvpeFeatureDim), 0.0F);
    keep.assign(static_cast<size_t>(kRmvpeGruChunkFrames), 0.0F);
    for (int64_t row = 0; row < frames; ++row) {
        std::copy_n(
            features.begin() + static_cast<std::ptrdiff_t>((start + row) * kRmvpeFeatureDim),
            kRmvpeFeatureDim,
            chunk.begin() + static_cast<std::ptrdiff_t>(row * kRmvpeFeatureDim));
        keep[static_cast<size_t>(row)] = 1.0F;
    }
}

void copy_hidden_rows_to_chunk(
    const std::vector<float> & source,
    int64_t start,
    int64_t frames,
    std::vector<float> & chunk) {
    chunk.assign(static_cast<size_t>(kRmvpeGruChunkFrames * kRmvpeHiddenDim), 0.0F);
    for (int64_t row = 0; row < frames; ++row) {
        std::copy_n(
            source.begin() + static_cast<std::ptrdiff_t>((start + row) * kRmvpeHiddenDim),
            kRmvpeHiddenDim,
            chunk.begin() + static_cast<std::ptrdiff_t>(row * kRmvpeHiddenDim));
    }
}

void copy_hidden_rows(
    const std::vector<float> & source,
    int64_t source_start,
    int64_t rows,
    std::vector<float> & destination,
    int64_t destination_start) {
    for (int64_t row = 0; row < rows; ++row) {
        std::copy_n(
            source.begin() + static_cast<std::ptrdiff_t>((source_start + row) * kRmvpeHiddenDim),
            kRmvpeHiddenDim,
            destination.begin() + static_cast<std::ptrdiff_t>((destination_start + row) * kRmvpeHiddenDim));
    }
}

std::vector<float> decode_rmvpe_salience(
    const std::vector<float> & salience,
    int64_t frames,
    float threshold) {
    std::vector<float> f0(static_cast<size_t>(frames), 0.0F);
    for (int64_t frame = 0; frame < frames; ++frame) {
        int center = 0;
        float best = salience[static_cast<size_t>(frame * kRmvpeClasses)];
        for (int cls = 1; cls < kRmvpeClasses; ++cls) {
            const float value = salience[static_cast<size_t>(frame * kRmvpeClasses + cls)];
            if (value > best) {
                best = value;
                center = cls;
            }
        }
        if (best <= threshold) {
            continue;
        }
        float weighted = 0.0F;
        float weight_sum = 0.0F;
        for (int offset = -4; offset <= 4; ++offset) {
            const int cls = center + offset;
            if (cls < 0 || cls >= kRmvpeClasses) {
                continue;
            }
            const float value = salience[static_cast<size_t>(frame * kRmvpeClasses + cls)];
            const float cents = 20.0F * static_cast<float>(cls) + 1997.3794084376191F;
            weighted += value * cents;
            weight_sum += value;
        }
        if (weight_sum > 0.0F) {
            const float cents = weighted / weight_sum;
            f0[static_cast<size_t>(frame)] = 10.0F * std::pow(2.0F, cents / 1200.0F);
        }
    }
    return f0;
}

}  // namespace

struct SeedVcRmvpeF0Extractor::State {
    ~State() {
        release_feature_graph();
        release_forward_graph();
        release_reverse_graph();
        release_head_graph();
    }

    struct FeatureGraph {
        ggml_context * ctx = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        ggml_cgraph * graph = nullptr;
        TensorValue input;
        ggml_tensor * output = nullptr;
        int64_t frames = 0;
    };

    struct GruGraph {
        ggml_context * ctx = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        ggml_cgraph * graph = nullptr;
        TensorValue input;
        TensorValue keep;
        TensorValue hidden;
        ggml_tensor * sequence = nullptr;
        ggml_tensor * final_hidden = nullptr;
    };

    struct HeadGraph {
        ggml_context * ctx = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        ggml_cgraph * graph = nullptr;
        TensorValue forward;
        TensorValue reverse;
        ggml_tensor * output = nullptr;
    };

    static void release_feature_graph(FeatureGraph & graph) {
        if (graph.gallocr != nullptr) {
            ggml_gallocr_free(graph.gallocr);
            graph.gallocr = nullptr;
        }
        if (graph.ctx != nullptr) {
            ggml_free(graph.ctx);
            graph.ctx = nullptr;
        }
        graph.graph = nullptr;
        graph.input = {};
        graph.output = nullptr;
        graph.frames = 0;
    }

    static void release_gru_graph(GruGraph & graph) {
        if (graph.gallocr != nullptr) {
            ggml_gallocr_free(graph.gallocr);
            graph.gallocr = nullptr;
        }
        if (graph.ctx != nullptr) {
            ggml_free(graph.ctx);
            graph.ctx = nullptr;
        }
        graph.graph = nullptr;
        graph.input = {};
        graph.keep = {};
        graph.hidden = {};
        graph.sequence = nullptr;
        graph.final_hidden = nullptr;
    }

    static void release_head_graph(HeadGraph & graph) {
        if (graph.gallocr != nullptr) {
            ggml_gallocr_free(graph.gallocr);
            graph.gallocr = nullptr;
        }
        if (graph.ctx != nullptr) {
            ggml_free(graph.ctx);
            graph.ctx = nullptr;
        }
        graph.graph = nullptr;
        graph.forward = {};
        graph.reverse = {};
        graph.output = nullptr;
    }

    void release_feature_graph() {
        release_feature_graph(feature);
    }

    void release_forward_graph() {
        release_gru_graph(forward);
    }

    void release_reverse_graph() {
        release_gru_graph(reverse);
    }

    void release_head_graph() {
        release_head_graph(head);
    }

    void ensure_feature_graph(const SeedVcRmvpeWeights & weights, int64_t frames) {
        if (frames <= 0 || frames % 32 != 0) {
            throw std::runtime_error("Seed-VC RMVPE feature graph requires positive 32-aligned frame count");
        }
        if (feature.ctx != nullptr && feature.frames == frames) {
            return;
        }
        release_feature_graph();
        if (weights.execution_context == nullptr) {
            throw std::runtime_error("Seed-VC RMVPE requires execution context");
        }
        ggml_init_params params{512ull * 1024ull * 1024ull, nullptr, true};
        feature.ctx = ggml_init(params);
        if (feature.ctx == nullptr) {
            throw std::runtime_error("failed to initialize Seed-VC RMVPE feature graph context");
        }
        engine::core::ModuleBuildContext build_ctx{
            feature.ctx,
            "seed_vc.rmvpe.feature",
            weights.execution_context->backend_type()};
        feature.input = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            TensorShape::from_dims({1, kRmvpeMelBins, frames}));
        ggml_set_input(feature.input.tensor);
        const auto result = build_rmvpe_feature_graph(build_ctx, feature.input, weights);
        feature.output = result.tensor;
        ggml_set_output(feature.output);
        feature.graph = ggml_new_graph_custom(feature.ctx, 16384, false);
        ggml_build_forward_expand(feature.graph, feature.output);
        feature.gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights.execution_context->backend()));
        if (feature.gallocr == nullptr ||
            !ggml_gallocr_reserve(feature.gallocr, feature.graph) ||
            !ggml_gallocr_alloc_graph(feature.gallocr, feature.graph)) {
            release_feature_graph();
            throw std::runtime_error("failed to allocate Seed-VC RMVPE feature graph tensors");
        }
        feature.frames = frames;
    }

    void ensure_gru_graph(const SeedVcRmvpeWeights & weights, GruGraph & target, const std::string & suffix, bool reverse_graph) {
        if (target.ctx != nullptr) {
            return;
        }
        if (weights.execution_context == nullptr) {
            throw std::runtime_error("Seed-VC RMVPE requires execution context");
        }
        ggml_init_params params{128ull * 1024ull * 1024ull, nullptr, true};
        target.ctx = ggml_init(params);
        if (target.ctx == nullptr) {
            throw std::runtime_error("failed to initialize Seed-VC RMVPE GRU graph context");
        }
        engine::core::ModuleBuildContext build_ctx{
            target.ctx,
            reverse_graph ? "seed_vc.rmvpe.gru_reverse" : "seed_vc.rmvpe.gru_forward",
            weights.execution_context->backend_type()};
        target.input = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            TensorShape::from_dims({kRmvpeGruChunkFrames, kRmvpeFeatureDim}));
        target.keep = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            TensorShape::from_dims({kRmvpeGruChunkFrames, 1}));
        target.hidden = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            TensorShape::from_dims({1, kRmvpeHiddenDim}));
        ggml_set_input(target.input.tensor);
        ggml_set_input(target.keep.tensor);
        ggml_set_input(target.hidden.tensor);
        const auto result = build_gru_chunk_graph(
            build_ctx,
            target.input,
            target.keep,
            target.hidden,
            weights,
            suffix,
            reverse_graph);
        target.sequence = result.sequence.tensor;
        target.final_hidden = result.final_hidden.tensor;
        ggml_set_output(target.sequence);
        ggml_set_output(target.final_hidden);
        target.graph = ggml_new_graph_custom(target.ctx, 40960, false);
        ggml_build_forward_expand(target.graph, target.sequence);
        ggml_build_forward_expand(target.graph, target.final_hidden);
        target.gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights.execution_context->backend()));
        if (target.gallocr == nullptr ||
            !ggml_gallocr_reserve(target.gallocr, target.graph) ||
            !ggml_gallocr_alloc_graph(target.gallocr, target.graph)) {
            release_gru_graph(target);
            throw std::runtime_error("failed to allocate Seed-VC RMVPE GRU graph tensors");
        }
    }

    void ensure_head_graph(const SeedVcRmvpeWeights & weights) {
        if (head.ctx != nullptr) {
            return;
        }
        if (weights.execution_context == nullptr) {
            throw std::runtime_error("Seed-VC RMVPE requires execution context");
        }
        ggml_init_params params{64ull * 1024ull * 1024ull, nullptr, true};
        head.ctx = ggml_init(params);
        if (head.ctx == nullptr) {
            throw std::runtime_error("failed to initialize Seed-VC RMVPE head graph context");
        }
        engine::core::ModuleBuildContext build_ctx{
            head.ctx,
            "seed_vc.rmvpe.head",
            weights.execution_context->backend_type()};
        head.forward = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            TensorShape::from_dims({kRmvpeGruChunkFrames, kRmvpeHiddenDim}));
        head.reverse = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            TensorShape::from_dims({kRmvpeGruChunkFrames, kRmvpeHiddenDim}));
        ggml_set_input(head.forward.tensor);
        ggml_set_input(head.reverse.tensor);
        const auto result = build_rmvpe_head_graph(build_ctx, head.forward, head.reverse, weights);
        head.output = result.tensor;
        ggml_set_output(head.output);
        head.graph = ggml_new_graph_custom(head.ctx, 4096, false);
        ggml_build_forward_expand(head.graph, head.output);
        head.gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights.execution_context->backend()));
        if (head.gallocr == nullptr ||
            !ggml_gallocr_reserve(head.gallocr, head.graph) ||
            !ggml_gallocr_alloc_graph(head.gallocr, head.graph)) {
            release_head_graph();
            throw std::runtime_error("failed to allocate Seed-VC RMVPE head graph tensors");
        }
    }

    void ensure_graphs(const SeedVcRmvpeWeights & weights, int64_t feature_frames) {
        ensure_feature_graph(weights, feature_frames);
        ensure_gru_graph(weights, forward, "", false);
        ensure_gru_graph(weights, reverse, "_reverse", true);
        ensure_head_graph(weights);
    }

    std::mutex mutex;
    SeedVcRmvpeWeights weights;
    FeatureGraph feature;
    GruGraph forward;
    GruGraph reverse;
    HeadGraph head;
};

SeedVcRmvpeF0Extractor::SeedVcRmvpeF0Extractor(
    std::shared_ptr<const engine::assets::TensorSource> source,
    engine::core::BackendConfig backend,
    engine::assets::TensorStorageType storage_type)
    : state_(std::make_shared<State>()) {
    state_->weights = load_rmvpe_weights(std::move(source), std::move(backend), storage_type);
}

SeedVcRmvpeF0Extractor::~SeedVcRmvpeF0Extractor() = default;
SeedVcRmvpeF0Extractor::SeedVcRmvpeF0Extractor(SeedVcRmvpeF0Extractor &&) noexcept = default;
SeedVcRmvpeF0Extractor & SeedVcRmvpeF0Extractor::operator=(SeedVcRmvpeF0Extractor &&) noexcept = default;

std::vector<float> SeedVcRmvpeF0Extractor::infer_16k_mono(
    const std::vector<float> & waveform_16k,
    float threshold,
    size_t threads) const {
    if (state_ == nullptr) {
        throw std::runtime_error("Seed-VC RMVPE is not initialized");
    }
    auto timing_start = std::chrono::steady_clock::now();
    const auto mel = prepare_rmvpe_mel(waveform_16k, threads);
    auto timing_end = std::chrono::steady_clock::now();
    engine::debug::timing_log_scalar(
        "seed_vc.rmvpe.mel_ms",
        engine::debug::elapsed_ms(timing_start, timing_end));
    timing_start = timing_end;
    std::lock_guard<std::mutex> lock(state_->mutex);
    const int64_t feature_step = mel.padded_frames <= kRmvpeFeatureActiveFrames
        ? mel.padded_frames
        : kRmvpeFeatureActiveFrames;
    const int64_t feature_graph_frames = mel.padded_frames <= kRmvpeFeatureActiveFrames
        ? mel.padded_frames
        : kRmvpeFeatureGraphFrames;
    state_->ensure_graphs(state_->weights, feature_graph_frames);
    timing_end = std::chrono::steady_clock::now();
    engine::debug::timing_log_scalar(
        "seed_vc.rmvpe.prepare_ms",
        engine::debug::elapsed_ms(timing_start, timing_end));
    timing_start = timing_end;

    const auto backend = state_->weights.execution_context->backend();
    std::vector<float> feature_window;
    std::vector<float> features(static_cast<size_t>(mel.padded_frames * kRmvpeFeatureDim), 0.0F);
    for (int64_t active_start = 0; active_start < mel.padded_frames; active_start += feature_step) {
        const int64_t active_frames =
            std::min<int64_t>(feature_step, mel.padded_frames - active_start);
        const int64_t context_start = std::max<int64_t>(0, active_start - kRmvpeFeatureOverlapFrames);
        const int64_t context_end = std::min<int64_t>(
            mel.padded_frames,
            active_start + active_frames + kRmvpeFeatureOverlapFrames);
        const int64_t context_frames = context_end - context_start;
        const int64_t active_offset = active_start - context_start;
        copy_mel_window(
            mel.values,
            mel.padded_frames,
            feature_graph_frames,
            context_start,
            context_frames,
            feature_window);
        engine::core::write_tensor_f32(state_->feature.input, feature_window);
        if (engine::core::compute_backend_graph(backend, state_->feature.graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for Seed-VC RMVPE feature graph");
        }
        const auto feature_output = engine::core::read_tensor_f32(state_->feature.output);
        copy_feature_rows(feature_output, active_offset, active_frames, features, active_start);
    }

    std::vector<float> feature_chunk;
    std::vector<float> keep;
    std::vector<float> hidden(static_cast<size_t>(kRmvpeHiddenDim), 0.0F);
    std::vector<float> forward_states(static_cast<size_t>(mel.padded_frames * kRmvpeHiddenDim), 0.0F);
    for (int64_t start = 0; start < mel.padded_frames; start += kRmvpeGruChunkFrames) {
        const int64_t chunk_frames = std::min<int64_t>(kRmvpeGruChunkFrames, mel.padded_frames - start);
        fill_gru_chunk(features, start, chunk_frames, feature_chunk, keep);
        engine::core::write_tensor_f32(state_->forward.input, feature_chunk);
        engine::core::write_tensor_f32(state_->forward.keep, keep);
        engine::core::write_tensor_f32(state_->forward.hidden, hidden);
        if (engine::core::compute_backend_graph(backend, state_->forward.graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for Seed-VC RMVPE forward GRU graph");
        }
        const auto sequence = engine::core::read_tensor_f32(state_->forward.sequence);
        copy_hidden_rows(sequence, 0, chunk_frames, forward_states, start);
        hidden = engine::core::read_tensor_f32(state_->forward.final_hidden);
    }

    std::vector<float> salience(static_cast<size_t>(mel.frames * kRmvpeClasses), 0.0F);
    std::vector<float> reverse_hidden(static_cast<size_t>(kRmvpeHiddenDim), 0.0F);
    std::vector<float> forward_chunk;
    for (int64_t end = mel.padded_frames; end > 0;) {
        const int64_t start = std::max<int64_t>(0, end - kRmvpeGruChunkFrames);
        const int64_t chunk_frames = end - start;
        fill_gru_chunk(features, start, chunk_frames, feature_chunk, keep);
        engine::core::write_tensor_f32(state_->reverse.input, feature_chunk);
        engine::core::write_tensor_f32(state_->reverse.keep, keep);
        engine::core::write_tensor_f32(state_->reverse.hidden, reverse_hidden);
        if (engine::core::compute_backend_graph(backend, state_->reverse.graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for Seed-VC RMVPE reverse GRU graph");
        }
        const auto reverse_sequence = engine::core::read_tensor_f32(state_->reverse.sequence);
        reverse_hidden = engine::core::read_tensor_f32(state_->reverse.final_hidden);
        const int64_t salience_rows =
            start < mel.frames ? std::min<int64_t>(chunk_frames, mel.frames - start) : 0;
        if (salience_rows > 0) {
            copy_hidden_rows_to_chunk(forward_states, start, chunk_frames, forward_chunk);
            engine::core::write_tensor_f32(state_->head.forward, forward_chunk);
            engine::core::write_tensor_f32(state_->head.reverse, reverse_sequence);
            if (engine::core::compute_backend_graph(backend, state_->head.graph) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("ggml_backend_graph_compute failed for Seed-VC RMVPE head graph");
            }
            const auto head_output = engine::core::read_tensor_f32(state_->head.output);
            for (int64_t row = 0; row < salience_rows; ++row) {
                std::copy_n(
                    head_output.begin() + static_cast<std::ptrdiff_t>(row * kRmvpeClasses),
                    kRmvpeClasses,
                    salience.begin() + static_cast<std::ptrdiff_t>((start + row) * kRmvpeClasses));
            }
        }
        end = start;
    }
    timing_end = std::chrono::steady_clock::now();
    engine::debug::timing_log_scalar(
        "seed_vc.rmvpe.compute_ms",
        engine::debug::elapsed_ms(timing_start, timing_end));
    timing_start = timing_end;
    auto f0 = decode_rmvpe_salience(salience, mel.frames, threshold);
    timing_end = std::chrono::steady_clock::now();
    engine::debug::timing_log_scalar(
        "seed_vc.rmvpe.decode_ms",
        engine::debug::elapsed_ms(timing_start, timing_end));
    return f0;
}

}  // namespace engine::models::seed_vc
