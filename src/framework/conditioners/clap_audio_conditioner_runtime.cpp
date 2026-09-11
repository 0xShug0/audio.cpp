#include "engine/framework/conditioners/clap_audio_conditioner_runtime.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/attention/types.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace engine::conditioners {
namespace {

namespace binding = engine::modules::binding;

constexpr std::array<int64_t, 4> kDepths = {2, 2, 12, 2};
constexpr std::array<int64_t, 4> kHeads = {4, 8, 16, 32};
constexpr int64_t kWindowSize = 8;
constexpr float kClampMin = 1.0e-10F;
constexpr double kPi = 3.141592653589793238462643383279502884;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

std::string tensor_name(const ClapAudioConfig & config, const std::string & name) {
    if (config.tensor_prefix.empty()) {
        return name;
    }
    return config.tensor_prefix + "." + name;
}

void validate_config(const ClapAudioConfig & config) {
    if (config.sample_rate <= 0 || config.max_samples <= 0 || config.n_fft <= 0 ||
        config.hop_length <= 0 || config.win_length <= 0 || config.mel_bins <= 0 ||
        config.spec_size <= 0 || config.patch_size <= 0 || config.patch_stride <= 0 ||
        config.embed_dim <= 0 || config.output_dim <= 0 || config.class_count <= 0) {
        throw std::runtime_error("CLAP audio config dimensions must be positive");
    }
    if (config.spec_size % config.patch_stride != 0 || config.spec_size % config.mel_bins != 0) {
        throw std::runtime_error("CLAP audio spec size must align with patch stride and mel bins");
    }
}

std::vector<float> require_vector(
    const assets::TensorSource & source,
    const ClapAudioConfig & config,
    const std::string & name,
    std::initializer_list<int64_t> shape) {
    return source.require_f32(tensor_name(config, name), std::vector<int64_t>(shape));
}

std::vector<float> batch_norm_scale(
    const assets::TensorSource & source,
    const ClapAudioConfig & config,
    const std::string & prefix,
    int64_t channels) {
    const auto weight = require_vector(source, config, prefix + ".weight", {channels});
    const auto running_var = require_vector(source, config, prefix + ".running_var", {channels});
    std::vector<float> scale(static_cast<size_t>(channels));
    for (int64_t c = 0; c < channels; ++c) {
        scale[static_cast<size_t>(c)] = weight[static_cast<size_t>(c)] /
            std::sqrt(running_var[static_cast<size_t>(c)] + config.batch_norm_eps);
    }
    return scale;
}

std::vector<float> batch_norm_bias(
    const assets::TensorSource & source,
    const ClapAudioConfig & config,
    const std::string & prefix,
    int64_t channels,
    const std::vector<float> & scale) {
    const auto bias = require_vector(source, config, prefix + ".bias", {channels});
    const auto running_mean = require_vector(source, config, prefix + ".running_mean", {channels});
    std::vector<float> out(static_cast<size_t>(channels));
    for (int64_t c = 0; c < channels; ++c) {
        out[static_cast<size_t>(c)] =
            bias[static_cast<size_t>(c)] - running_mean[static_cast<size_t>(c)] * scale[static_cast<size_t>(c)];
    }
    return out;
}

std::vector<float> relative_position_bias(
    const assets::TensorSource & source,
    const ClapAudioConfig & config,
    const std::string & prefix,
    int64_t heads) {
    const int64_t table_rows = (2 * kWindowSize - 1) * (2 * kWindowSize - 1);
    const auto table = require_vector(
        source,
        config,
        prefix + ".attn.relative_position_bias_table",
        {table_rows, heads});
    constexpr int64_t n = kWindowSize * kWindowSize;
    std::vector<float> out(static_cast<size_t>(heads * n * n), 0.0F);
    for (int64_t qh = 0; qh < kWindowSize; ++qh) {
        for (int64_t qw = 0; qw < kWindowSize; ++qw) {
            for (int64_t kh = 0; kh < kWindowSize; ++kh) {
                for (int64_t kw = 0; kw < kWindowSize; ++kw) {
                    const int64_t rel_h = qh - kh + kWindowSize - 1;
                    const int64_t rel_w = qw - kw + kWindowSize - 1;
                    const int64_t row = rel_h * (2 * kWindowSize - 1) + rel_w;
                    const int64_t q = qh * kWindowSize + qw;
                    const int64_t k = kh * kWindowSize + kw;
                    for (int64_t h = 0; h < heads; ++h) {
                        out[static_cast<size_t>((h * n + q) * n + k)] =
                            table[static_cast<size_t>(row * heads + h)];
                    }
                }
            }
        }
    }
    return out;
}

std::vector<float> shifted_window_mask(int64_t height, int64_t width, int64_t shift) {
    const int64_t windows_h = height / kWindowSize;
    const int64_t windows_w = width / kWindowSize;
    const int64_t windows = windows_h * windows_w;
    constexpr int64_t n = kWindowSize * kWindowSize;
    std::vector<int64_t> regions(static_cast<size_t>(height * width), 0);
    int64_t region = 0;
    const std::array<int64_t, 4> h_starts = {0, height - kWindowSize, height - shift, height};
    const std::array<int64_t, 4> w_starts = {0, width - kWindowSize, width - shift, width};
    for (int hi = 0; hi < 3; ++hi) {
        for (int wi = 0; wi < 3; ++wi) {
            for (int64_t y = h_starts[static_cast<size_t>(hi)]; y < h_starts[static_cast<size_t>(hi + 1)]; ++y) {
                for (int64_t x = w_starts[static_cast<size_t>(wi)]; x < w_starts[static_cast<size_t>(wi + 1)]; ++x) {
                    regions[static_cast<size_t>(y * width + x)] = region;
                }
            }
            ++region;
        }
    }
    std::vector<float> mask(static_cast<size_t>(windows * n * n), 0.0F);
    for (int64_t wh = 0; wh < windows_h; ++wh) {
        for (int64_t ww = 0; ww < windows_w; ++ww) {
            const int64_t window = wh * windows_w + ww;
            std::array<int64_t, n> ids = {};
            for (int64_t y = 0; y < kWindowSize; ++y) {
                for (int64_t x = 0; x < kWindowSize; ++x) {
                    ids[static_cast<size_t>(y * kWindowSize + x)] =
                        regions[static_cast<size_t>((wh * kWindowSize + y) * width + ww * kWindowSize + x)];
                }
            }
            for (int64_t q = 0; q < n; ++q) {
                for (int64_t k = 0; k < n; ++k) {
                    mask[static_cast<size_t>((window * n + q) * n + k)] = ids[static_cast<size_t>(q)] == ids[static_cast<size_t>(k)] ? 0.0F : -100.0F;
                }
            }
        }
    }
    return mask;
}

float cubic_weight(float x) {
    constexpr float a = -0.75F;
    const float ax = std::fabs(x);
    if (ax <= 1.0F) {
        return ((a + 2.0F) * ax - (a + 3.0F)) * ax * ax + 1.0F;
    }
    if (ax < 2.0F) {
        return (((a * ax - 5.0F * a) * ax + 8.0F * a) * ax - 4.0F * a);
    }
    return 0.0F;
}

int64_t clamp_index(int64_t value, int64_t limit) {
    return std::min<int64_t>(std::max<int64_t>(value, 0), limit - 1);
}

std::vector<float> resize_time_bicubic_align_corners(
    const std::vector<float> & input,
    int64_t batch,
    int64_t time,
    int64_t mel,
    int64_t output_time) {
    std::vector<float> out(static_cast<size_t>(batch * output_time * mel), 0.0F);
    if (time == output_time) {
        return input;
    }
    const float scale = output_time > 1 ? static_cast<float>(time - 1) / static_cast<float>(output_time - 1) : 0.0F;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t t = 0; t < output_time; ++t) {
            const float src = scale * static_cast<float>(t);
            const int64_t base = static_cast<int64_t>(std::floor(src));
            for (int64_t m = 0; m < mel; ++m) {
                float sum = 0.0F;
                for (int64_t i = -1; i <= 2; ++i) {
                    const int64_t idx = clamp_index(base + i, time);
                    const float w = cubic_weight(src - static_cast<float>(base + i));
                    sum += input[static_cast<size_t>((b * time + idx) * mel + m)] * w;
                }
                out[static_cast<size_t>((b * output_time + t) * mel + m)] = sum;
            }
        }
    }
    return out;
}

struct ClapBlockWeights {
    modules::NormWeights norm1;
    modules::NormWeights norm2;
    modules::AttentionWeights attention;
    core::TensorValue relative_bias;
    std::optional<core::TensorValue> shift_mask;
    modules::FeedForwardWeights mlp;
};

struct ClapPatchMergingWeights {
    modules::NormWeights norm;
    modules::LinearWeights reduction;
};

struct ClapStageWeights {
    std::vector<ClapBlockWeights> blocks;
    std::optional<ClapPatchMergingWeights> downsample;
};

struct ClapWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    std::vector<float> mel_filterbank;
    modules::Conv1dWeights stft_real;
    modules::Conv1dWeights stft_imag;
    core::TensorValue mel_projection;
    std::vector<float> bn0_scale;
    std::vector<float> bn0_bias;
    modules::Conv2dWeights patch_embed;
    modules::NormWeights patch_norm;
    std::vector<ClapStageWeights> stages;
    modules::NormWeights norm;
    modules::LinearWeights audio_proj0;
    modules::LinearWeights audio_proj2;
};

modules::Conv1dWeights make_stft_conv_weights(
    core::BackendWeightStore & store,
    const ClapAudioConfig & config,
    bool imaginary) {
    const int64_t freq_bins = config.n_fft / 2 + 1;
    const int64_t window_offset = (config.n_fft - config.win_length) / 2;
    const audio::STFTConfig stft_config{
        config.n_fft,
        config.hop_length,
        config.win_length,
        true,
        audio::STFTPadMode::Reflect,
        audio::STFTFamily::Default};
    const auto & window = audio::get_cached_stft_window(stft_config);
    std::vector<float> values(static_cast<size_t>(freq_bins * config.n_fft), 0.0F);
    for (int64_t f = 0; f < freq_bins; ++f) {
        for (int64_t i = 0; i < config.win_length; ++i) {
            const int64_t k = window_offset + i;
            const double phase = -2.0 * kPi * static_cast<double>(k * f) / static_cast<double>(config.n_fft);
            const double component = imaginary ? std::sin(phase) : std::cos(phase);
            values[static_cast<size_t>(f * config.n_fft + k)] =
                static_cast<float>(component) * window[static_cast<size_t>(i)];
        }
    }
    return {
        store.make_f32(core::TensorShape::from_dims({freq_bins, 1, config.n_fft}), std::move(values)),
        std::nullopt};
}

modules::AttentionWeights load_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const ClapAudioConfig & config,
    const std::string & prefix,
    int64_t hidden,
    const ClapAudioRuntimeOptions & options) {
    modules::AttentionWeights weights;
    weights.qkv_weight = store.load_tensor(
        source,
        tensor_name(config, prefix + ".attn.qkv.weight"),
        options.weight_storage_type,
        {3 * hidden, hidden});
    weights.qkv_bias = store.load_f32_tensor(
        source,
        tensor_name(config, prefix + ".attn.qkv.bias"),
        {3 * hidden});
    weights.out_weight = store.load_tensor(
        source,
        tensor_name(config, prefix + ".attn.proj.weight"),
        options.weight_storage_type,
        {hidden, hidden});
    weights.out_bias = store.load_f32_tensor(
        source,
        tensor_name(config, prefix + ".attn.proj.bias"),
        {hidden});
    return weights;
}

ClapBlockWeights load_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const ClapAudioConfig & config,
    int64_t stage,
    int64_t block,
    int64_t resolution,
    const ClapAudioRuntimeOptions & options) {
    const int64_t hidden = config.embed_dim * (1LL << stage);
    const int64_t heads = kHeads[static_cast<size_t>(stage)];
    const std::string prefix = "audio_branch.layers." + std::to_string(stage) + ".blocks." + std::to_string(block);
    ClapBlockWeights weights;
    weights.norm1 = binding::norm_from_source(store, source, tensor_name(config, prefix + ".norm1"), hidden);
    weights.norm2 = binding::norm_from_source(store, source, tensor_name(config, prefix + ".norm2"), hidden);
    weights.attention = load_attention(store, source, config, prefix, hidden, options);
    weights.relative_bias = store.make_f32(
        core::TensorShape::from_dims({1, heads, kWindowSize * kWindowSize, kWindowSize * kWindowSize}),
        relative_position_bias(source, config, prefix, heads));
    if ((block % 2) == 1) {
        weights.shift_mask = store.make_f32(
            core::TensorShape::from_dims({
                (resolution / kWindowSize) * (resolution / kWindowSize),
                1,
                kWindowSize * kWindowSize,
                kWindowSize * kWindowSize,
            }),
            shifted_window_mask(resolution, resolution, kWindowSize / 2));
    }
    weights.mlp = {
        store.load_tensor(
            source,
            tensor_name(config, prefix + ".mlp.fc1.weight"),
            options.weight_storage_type,
            {4 * hidden, hidden}),
        store.load_f32_tensor(source, tensor_name(config, prefix + ".mlp.fc1.bias"), {4 * hidden}),
        store.load_tensor(
            source,
            tensor_name(config, prefix + ".mlp.fc2.weight"),
            options.weight_storage_type,
            {hidden, 4 * hidden}),
        store.load_f32_tensor(source, tensor_name(config, prefix + ".mlp.fc2.bias"), {hidden}),
    };
    return weights;
}

ClapWeights load_weights(
    const assets::TensorSource & source,
    const ClapAudioConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    const ClapAudioRuntimeOptions & options) {
    ClapWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "framework.clap_audio.weights",
        options.weight_context_bytes);
    const auto melw = require_vector(
        source,
        config,
        "audio_branch.logmel_extractor.melW",
        {config.n_fft / 2 + 1, config.mel_bins});
    weights.mel_filterbank.assign(static_cast<size_t>(config.mel_bins * (config.n_fft / 2 + 1)), 0.0F);
    for (int64_t f = 0; f < config.n_fft / 2 + 1; ++f) {
        for (int64_t m = 0; m < config.mel_bins; ++m) {
            weights.mel_filterbank[static_cast<size_t>(m * (config.n_fft / 2 + 1) + f)] =
                melw[static_cast<size_t>(f * config.mel_bins + m)];
        }
    }
    const int64_t freq_bins = config.n_fft / 2 + 1;
    weights.stft_real = make_stft_conv_weights(*weights.store, config, false);
    weights.stft_imag = make_stft_conv_weights(*weights.store, config, true);
    std::vector<float> mel_projection(static_cast<size_t>(freq_bins * config.mel_bins), 0.0F);
    for (int64_t f = 0; f < freq_bins; ++f) {
        for (int64_t m = 0; m < config.mel_bins; ++m) {
            mel_projection[static_cast<size_t>(f * config.mel_bins + m)] =
                weights.mel_filterbank[static_cast<size_t>(m * freq_bins + f)];
        }
    }
    weights.mel_projection = weights.store->make_f32(
        core::TensorShape::from_dims({1, freq_bins, config.mel_bins}),
        std::move(mel_projection));
    const auto bn_scale = batch_norm_scale(source, config, "audio_branch.bn0", config.mel_bins);
    const auto bn_bias = batch_norm_bias(source, config, "audio_branch.bn0", config.mel_bins, bn_scale);
    weights.bn0_scale = bn_scale;
    weights.bn0_bias = bn_bias;
    weights.patch_embed = {
        weights.store->load_tensor(
            source,
            tensor_name(config, "audio_branch.patch_embed.proj.weight"),
            options.weight_storage_type,
            {config.embed_dim, 1, config.patch_size, config.patch_size}),
        weights.store->load_f32_tensor(
            source,
            tensor_name(config, "audio_branch.patch_embed.proj.bias"),
            {config.embed_dim})};
    weights.patch_norm = binding::norm_from_source(
        *weights.store,
        source,
        tensor_name(config, "audio_branch.patch_embed.norm"),
        config.embed_dim);
    int64_t resolution = config.spec_size / config.patch_stride;
    for (int64_t stage = 0; stage < 4; ++stage) {
        ClapStageWeights stage_weights;
        for (int64_t block = 0; block < kDepths[static_cast<size_t>(stage)]; ++block) {
            stage_weights.blocks.push_back(load_block(*weights.store, source, config, stage, block, resolution, options));
        }
        if (stage < 3) {
            const int64_t hidden = config.embed_dim * (1LL << stage);
            const std::string prefix = "audio_branch.layers." + std::to_string(stage) + ".downsample";
            ClapPatchMergingWeights downsample;
            downsample.norm = binding::norm_from_source(
                *weights.store,
                source,
                tensor_name(config, prefix + ".norm"),
                4 * hidden);
            downsample.reduction = {
                weights.store->load_tensor(
                    source,
                    tensor_name(config, prefix + ".reduction.weight"),
                    options.weight_storage_type,
                    {2 * hidden, 4 * hidden}),
                std::nullopt};
            stage_weights.downsample = std::move(downsample);
            resolution /= 2;
        }
        weights.stages.push_back(std::move(stage_weights));
    }
    weights.norm = binding::norm_from_source(
        *weights.store,
        source,
        tensor_name(config, "audio_branch.norm"),
        config.embed_dim * 8);
    weights.audio_proj0 = {
        weights.store->load_tensor(
            source,
            tensor_name(config, "audio_projection.0.weight"),
            options.weight_storage_type,
            {config.output_dim, config.embed_dim * 8}),
        weights.store->load_f32_tensor(
            source,
            tensor_name(config, "audio_projection.0.bias"),
            {config.output_dim})};
    weights.audio_proj2 = {
        weights.store->load_tensor(
            source,
            tensor_name(config, "audio_projection.2.weight"),
            options.weight_storage_type,
            {config.output_dim, config.output_dim}),
        weights.store->load_f32_tensor(
            source,
            tensor_name(config, "audio_projection.2.bias"),
            {config.output_dim})};
    weights.store->upload();
    return weights;
}

std::vector<float> prepare_waveform(const std::vector<float> & waveform, int64_t batch, int64_t samples, int64_t max_samples) {
    if (static_cast<int64_t>(waveform.size()) != batch * samples) {
        throw std::runtime_error("CLAP audio waveform size mismatch");
    }
    std::vector<float> out(static_cast<size_t>(batch * max_samples), 0.0F);
    for (int64_t b = 0; b < batch; ++b) {
        const float * src = waveform.data() + static_cast<size_t>(b * samples);
        float * dst = out.data() + static_cast<size_t>(b * max_samples);
        if (samples >= max_samples) {
            std::copy(src, src + max_samples, dst);
            continue;
        }
        const int64_t repeats = max_samples / samples;
        int64_t offset = 0;
        for (int64_t r = 0; r < repeats; ++r) {
            std::copy(src, src + samples, dst + offset);
            offset += samples;
        }
    }
    return out;
}

std::vector<float> build_logmel_image(
    const std::vector<float> & logmel_bmt,
    int64_t batch,
    const ClapAudioConfig & config,
    const ClapWeights & weights) {
    const int64_t frames = 1 + config.max_samples / config.hop_length;
    if (static_cast<int64_t>(logmel_bmt.size()) != batch * config.mel_bins * frames) {
        throw std::runtime_error("CLAP audio log-mel tensor size mismatch");
    }
    std::vector<float> mel(static_cast<size_t>(batch * frames * config.mel_bins), 0.0F);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t t = 0; t < frames; ++t) {
            for (int64_t m = 0; m < config.mel_bins; ++m) {
                mel[static_cast<size_t>((b * frames + t) * config.mel_bins + m)] =
                    logmel_bmt[static_cast<size_t>(((b * config.mel_bins) + m) * frames + t)];
            }
        }
    }
    mel = resize_time_bicubic_align_corners(mel, batch, frames, config.mel_bins, config.spec_size * (config.spec_size / config.mel_bins));
    const int64_t target_time = config.spec_size * (config.spec_size / config.mel_bins);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t t = 0; t < target_time; ++t) {
            for (int64_t m = 0; m < config.mel_bins; ++m) {
                auto & value = mel[static_cast<size_t>((b * target_time + t) * config.mel_bins + m)];
                value = value * weights.bn0_scale[static_cast<size_t>(m)] + weights.bn0_bias[static_cast<size_t>(m)];
            }
        }
    }
    std::vector<float> image(static_cast<size_t>(batch * config.spec_size * config.spec_size), 0.0F);
    const int64_t freq_ratio = config.spec_size / config.mel_bins;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t m = 0; m < config.mel_bins; ++m) {
            for (int64_t r = 0; r < freq_ratio; ++r) {
                for (int64_t t = 0; t < config.spec_size; ++t) {
                    const int64_t src_t = r * config.spec_size + t;
                    const int64_t out_h = r * config.mel_bins + m;
                    image[static_cast<size_t>((b * config.spec_size + out_h) * config.spec_size + t)] =
                        mel[static_cast<size_t>((b * target_time + src_t) * config.mel_bins + m)];
                }
            }
        }
    }
    return image;
}

core::TensorValue build_logmel_frontend(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & waveform,
    const ClapWeights & weights,
    const ClapAudioConfig & config,
    const ClapAudioRuntimeOptions & options) {
    const int64_t freq_bins = config.n_fft / 2 + 1;
    auto padded = modules::ReflectPad1dModule({config.n_fft / 2, config.n_fft / 2}).build(ctx, waveform);
    auto real = modules::Conv1dModule({1, freq_bins, config.n_fft, static_cast<int>(config.hop_length), 0, 1, false})
                    .build(ctx, padded, weights.stft_real);
    auto imag = modules::Conv1dModule({1, freq_bins, config.n_fft, static_cast<int>(config.hop_length), 0, 1, false})
                    .build(ctx, padded, weights.stft_imag);
    auto power = modules::AddModule().build(
        ctx,
        modules::MulModule().build(ctx, real, real),
        modules::MulModule().build(ctx, imag, imag));
    auto power_btf = modules::TransposeModule({{0, 2, 1, 3}, power.shape.rank}).build(ctx, power);
    auto mel_weights = modules::RepeatModule({
        core::TensorShape::from_dims({waveform.shape.dims[0], freq_bins, config.mel_bins})})
                           .build(ctx, weights.mel_projection);
    auto mel = modules::MatMulModule().build(ctx, power_btf, mel_weights);
    ggml_mul_mat_set_prec(mel.tensor, options.projection_precision);
    auto logmel = core::wrap_tensor(
        ggml_scale(ctx.ggml, ggml_log(ctx.ggml, ggml_clamp(ctx.ggml, mel.tensor, kClampMin, INFINITY)), 10.0F / std::log(10.0F)),
        mel.shape,
        GGML_TYPE_F32);
    return core::ensure_backend_addressable_layout(
        ctx,
        modules::TransposeModule({{0, 2, 1, 3}, logmel.shape.rank}).build(ctx, logmel));
}

core::TensorValue roll_2d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    int64_t shift,
    bool reverse) {
    if (shift == 0) {
        return x;
    }
    const int64_t height = x.shape.dims[1];
    const int64_t width = x.shape.dims[2];
    const int64_t start_h = reverse ? height - shift : shift;
    auto h0 = modules::SliceModule({1, start_h, height - start_h}).build(ctx, x);
    auto h1 = modules::SliceModule({1, 0, start_h}).build(ctx, x);
    auto rolled = modules::ConcatModule({1}).build(ctx, h0, h1);
    const int64_t start_w = reverse ? width - shift : shift;
    auto w0 = modules::SliceModule({2, start_w, width - start_w}).build(ctx, rolled);
    auto w1 = modules::SliceModule({2, 0, start_w}).build(ctx, rolled);
    return modules::ConcatModule({2}).build(ctx, w0, w1);
}

core::TensorValue partition_windows(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    int64_t height,
    int64_t width) {
    const int64_t batch = x.shape.dims[0];
    auto out = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, x),
        core::TensorShape::from_dims({
            batch * (height / kWindowSize),
            kWindowSize,
            width,
            x.shape.dims[3],
        }));
    out = modules::TransposeModule({{0, 2, 1, 3}, out.shape.rank}).build(ctx, out);
    out = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, out),
        core::TensorShape::from_dims({
            batch * (height / kWindowSize) * (width / kWindowSize),
            kWindowSize,
            kWindowSize,
            x.shape.dims[3],
        }));
    out = modules::TransposeModule({{0, 2, 1, 3}, out.shape.rank}).build(ctx, out);
    return core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, out),
        core::TensorShape::from_dims({
            batch * (height / kWindowSize) * (width / kWindowSize),
            kWindowSize * kWindowSize,
            x.shape.dims[3],
        }));
}

core::TensorValue reverse_windows(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & windows,
    int64_t batch,
    int64_t height,
    int64_t width) {
    auto out = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, windows),
        core::TensorShape::from_dims({
            windows.shape.dims[0],
            kWindowSize,
            kWindowSize,
            windows.shape.dims[2],
        }));
    out = modules::TransposeModule({{0, 2, 1, 3}, out.shape.rank}).build(ctx, out);
    out = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, out),
        core::TensorShape::from_dims({
            batch * (height / kWindowSize),
            width,
            kWindowSize,
            windows.shape.dims[2],
        }));
    out = modules::TransposeModule({{0, 2, 1, 3}, out.shape.rank}).build(ctx, out);
    return core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, out),
        core::TensorShape::from_dims({batch, height, width, windows.shape.dims[2]}));
}

core::TensorValue repeat_window_mask(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & mask,
    int64_t batch,
    int64_t heads) {
    std::vector<core::TensorValue> masks;
    masks.reserve(static_cast<size_t>(batch));
    auto head_mask = modules::RepeatModule({
        core::TensorShape::from_dims({mask.shape.dims[0], heads, mask.shape.dims[2], mask.shape.dims[3]})})
                         .build(ctx, mask);
    for (int64_t b = 0; b < batch; ++b) {
        masks.push_back(head_mask);
    }
    auto out = masks.front();
    for (size_t i = 1; i < masks.size(); ++i) {
        out = modules::ConcatModule({0}).build(ctx, out, masks[i]);
    }
    return out;
}

core::TensorValue window_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ClapBlockWeights & weights,
    int64_t heads,
    const ClapAudioRuntimeOptions & options) {
    const int64_t hidden = input.shape.dims[2];
    const int64_t head_dim = hidden / heads;
    auto qkv = modules::LinearModule({hidden, 3 * hidden, true, options.projection_precision})
                   .build(ctx, input, {*weights.attention.qkv_weight, weights.attention.qkv_bias});
    auto q = modules::SliceModule({2, 0, hidden}).build(ctx, qkv);
    auto k = modules::SliceModule({2, hidden, hidden}).build(ctx, qkv);
    auto v = modules::SliceModule({2, 2 * hidden, hidden}).build(ctx, qkv);
    auto to_heads = [&](core::TensorValue value) {
        value = core::ensure_backend_addressable_layout(ctx, value);
        value = core::reshape_tensor(ctx, value, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, head_dim}));
        value = modules::TransposeModule({{0, 2, 1, 3}, value.shape.rank}).build(ctx, value);
        return core::ensure_backend_addressable_layout(ctx, value);
    };
    q = to_heads(q);
    k = to_heads(k);
    v = to_heads(v);
    auto scores = modules::MatMulModule().build(
        ctx,
        q,
        modules::TransposeModule(modules::TransposeConfig{{0, 1, 3, 2}, k.shape.rank}).build(ctx, k));
    ggml_mul_mat_set_prec(scores.tensor, options.attention_precision);
    scores = core::wrap_tensor(
        ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(head_dim))),
        scores.shape,
        GGML_TYPE_F32);
    auto bias = modules::RepeatModule(modules::RepeatConfig{scores.shape}).build(ctx, weights.relative_bias);
    scores = modules::AddModule().build(ctx, scores, bias);
    if (weights.shift_mask.has_value()) {
        auto mask = repeat_window_mask(ctx, *weights.shift_mask, input.shape.dims[0] / weights.shift_mask->shape.dims[0], heads);
        scores = modules::AddModule().build(ctx, scores, mask);
    }
    auto probs = core::wrap_tensor(ggml_soft_max(ctx.ggml, core::ensure_backend_addressable_layout(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
    auto context = modules::MatMulModule().build(ctx, probs, v);
    ggml_mul_mat_set_prec(context.tensor, options.attention_precision);
    context = modules::TransposeModule(modules::TransposeConfig{{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, context),
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], hidden}));
    return modules::LinearModule({hidden, hidden, true, options.projection_precision})
        .build(ctx, context, {weights.attention.out_weight, weights.attention.out_bias});
}

core::TensorValue apply_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ClapBlockWeights & weights,
    int64_t resolution,
    int64_t heads,
    const ClapAudioConfig & config,
    const ClapAudioRuntimeOptions & options) {
    const int64_t batch = input.shape.dims[0];
    const int64_t hidden = input.shape.dims[2];
    auto shortcut = input;
    auto x = modules::LayerNormModule({hidden, config.layer_norm_eps, true, true}).build(ctx, input, weights.norm1);
    x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), core::TensorShape::from_dims({batch, resolution, resolution, hidden}));
    const int64_t shift = weights.shift_mask.has_value() ? kWindowSize / 2 : 0;
    x = roll_2d(ctx, x, shift, false);
    auto windows = partition_windows(ctx, x, resolution, resolution);
    auto attended = window_attention(ctx, windows, weights, heads, options);
    x = reverse_windows(ctx, attended, batch, resolution, resolution);
    x = roll_2d(ctx, x, shift, true);
    x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), core::TensorShape::from_dims({batch, resolution * resolution, hidden}));
    x = modules::ResidualAddModule().build(ctx, shortcut, x);
    auto mlp_input = modules::LayerNormModule({hidden, config.layer_norm_eps, true, true}).build(ctx, x, weights.norm2);
    auto mlp = modules::FeedForwardModule({
        hidden,
        4 * hidden,
        true,
        modules::GeluApproximation::ExactErf,
        options.projection_precision,
    }).build(ctx, mlp_input, weights.mlp);
    return modules::ResidualAddModule().build(ctx, x, mlp);
}

core::TensorValue patch_merge(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t resolution,
    const ClapPatchMergingWeights & weights,
    const ClapAudioConfig & config,
    const ClapAudioRuntimeOptions & options) {
    const int64_t batch = input.shape.dims[0];
    const int64_t hidden = input.shape.dims[2];
    auto x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, input), core::TensorShape::from_dims({batch, resolution, resolution, hidden}));
    auto out = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, x),
        core::TensorShape::from_dims({batch * (resolution / 2), 2, resolution, hidden}));
    out = modules::TransposeModule({{0, 2, 1, 3}, out.shape.rank}).build(ctx, out);
    out = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, out),
        core::TensorShape::from_dims({batch * (resolution / 2) * (resolution / 2), 2, 2, hidden}));
    out = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, out),
        core::TensorShape::from_dims({batch, (resolution / 2) * (resolution / 2), 4 * hidden}));
    out = modules::LayerNormModule({4 * hidden, config.layer_norm_eps, true, true}).build(ctx, out, weights.norm);
    return modules::LinearModule({4 * hidden, 2 * hidden, false, options.projection_precision}).build(ctx, out, weights.reduction);
}

core::TensorValue l2_normalize_last(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    auto squared = core::wrap_tensor(ggml_sqr(ctx.ggml, contiguous.tensor), contiguous.shape, GGML_TYPE_F32);
    auto sum = modules::ReduceSumModule({static_cast<int>(contiguous.shape.rank - 1)}).build(ctx, squared);
    auto denom = core::wrap_tensor(
        ggml_scale_bias(ctx.ggml, ggml_sqrt(ctx.ggml, core::ensure_backend_addressable_layout(ctx, sum).tensor), 1.0F, 1.0e-12F),
        sum.shape,
        GGML_TYPE_F32);
    auto repeated = modules::RepeatModule({contiguous.shape}).build(ctx, denom);
    return core::wrap_tensor(ggml_div(ctx.ggml, contiguous.tensor, repeated.tensor), contiguous.shape, GGML_TYPE_F32);
}

core::TensorValue build_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & images,
    const ClapWeights & weights,
    const ClapAudioConfig & config,
    const ClapAudioRuntimeOptions & options) {
    auto x = modules::Conv2dModule({
        1,
        config.embed_dim,
        config.patch_size,
        config.patch_size,
        static_cast<int>(config.patch_stride),
        static_cast<int>(config.patch_stride),
        0,
        0,
        1,
        1,
        true,
    }).build(ctx, images, weights.patch_embed);
    const int64_t grid = config.spec_size / config.patch_stride;
    x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), core::TensorShape::from_dims({images.shape.dims[0], config.embed_dim, grid * grid}));
    x = modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
    x = modules::LayerNormModule({config.embed_dim, config.layer_norm_eps, true, true}).build(ctx, x, weights.patch_norm);
    int64_t resolution = grid;
    for (int64_t stage = 0; stage < 4; ++stage) {
        for (const auto & block : weights.stages[static_cast<size_t>(stage)].blocks) {
            x = apply_block(ctx, x, block, resolution, kHeads[static_cast<size_t>(stage)], config, options);
        }
        if (weights.stages[static_cast<size_t>(stage)].downsample.has_value()) {
            x = patch_merge(ctx, x, resolution, *weights.stages[static_cast<size_t>(stage)].downsample, config, options);
            resolution /= 2;
        }
    }
    const int64_t hidden = config.embed_dim * 8;
    x = modules::LayerNormModule({hidden, config.layer_norm_eps, true, true}).build(ctx, x, weights.norm);
    x = modules::ReduceMeanModule({1}).build(ctx, x);
    x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), core::TensorShape::from_dims({images.shape.dims[0], hidden}));
    x = modules::LinearModule({hidden, config.output_dim, true, options.projection_precision}).build(ctx, x, weights.audio_proj0);
    x = core::wrap_tensor(ggml_relu(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);
    x = modules::LinearModule({config.output_dim, config.output_dim, true, options.projection_precision}).build(ctx, x, weights.audio_proj2);
    return l2_normalize_last(ctx, x);
}

}  // namespace

struct ClapAudioConditionerRuntime::Impl {
    Impl(
        std::shared_ptr<const assets::TensorSource> source_in,
        core::ExecutionContext & execution_in,
        ClapAudioConfig config_in,
        ClapAudioRuntimeOptions options_in)
        : source(std::move(source_in)),
          execution(execution_in),
          config(std::move(config_in)),
          options(options_in),
          weights(load_weights(*source, config, execution.backend(), execution.backend_type(), options)) {
    }

    ClapAudioEmbedding encode_audio(
        const std::vector<float> & waveform,
        int64_t batch,
        int64_t samples,
        size_t threads) {
        (void)threads;
        const auto padded = prepare_waveform(waveform, batch, samples, config.max_samples);
        ensure_frontend_graph(batch);
        const double frontend_upload_ms = debug::measure_ms([&]() {
            ggml_backend_tensor_set(frontend_input.tensor, padded.data(), 0, padded.size() * sizeof(float));
        });
        const double frontend_compute_ms = debug::measure_ms([&]() {
            ggml_backend_graph_compute(execution.backend(), frontend_graph);
        });
        std::vector<float> logmel;
        const double frontend_read_ms = debug::measure_ms([&]() {
            logmel = core::read_tensor_f32(frontend_output.tensor);
        });
        double image_build_ms = 0.0;
        std::vector<float> image_values;
        image_build_ms = debug::measure_ms([&]() {
            image_values = build_logmel_image(logmel, batch, config, weights);
        });
        debug::timing_log_scalar("controlfoley.clap.frontend_upload_ms", frontend_upload_ms);
        debug::timing_log_scalar("controlfoley.clap.frontend_compute_ms", frontend_compute_ms);
        debug::timing_log_scalar("controlfoley.clap.frontend_read_ms", frontend_read_ms);
        debug::timing_log_scalar("controlfoley.clap.image_build_ms", image_build_ms);
        ensure_graph(batch);
        const double encoder_upload_ms = debug::measure_ms([&]() {
            ggml_backend_tensor_set(input.tensor, image_values.data(), 0, image_values.size() * sizeof(float));
        });
        const double encoder_compute_ms = debug::measure_ms([&]() {
            ggml_backend_graph_compute(execution.backend(), graph);
        });
        ClapAudioEmbedding embedding;
        embedding.batch = batch;
        embedding.features = config.output_dim;
        const double encoder_read_ms = debug::measure_ms([&]() {
            embedding.values = core::read_tensor_f32(output.tensor);
        });
        debug::timing_log_scalar("controlfoley.clap.encoder_upload_ms", encoder_upload_ms);
        debug::timing_log_scalar("controlfoley.clap.encoder_compute_ms", encoder_compute_ms);
        debug::timing_log_scalar("controlfoley.clap.encoder_read_ms", encoder_read_ms);
        return embedding;
    }

    void release_runtime_graphs() {
        frontend_graph = nullptr;
        frontend_input = {};
        frontend_output = {};
        if (frontend_gallocr != nullptr) {
            ggml_gallocr_free(frontend_gallocr);
            frontend_gallocr = nullptr;
        }
        frontend_graph_ctx.reset();
        cached_frontend_batch = 0;
        graph = nullptr;
        input = {};
        output = {};
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
        graph_ctx.reset();
        cached_batch = 0;
    }

    void ensure_frontend_graph(int64_t batch) {
        if (frontend_graph != nullptr && cached_frontend_batch == batch) {
            return;
        }
        frontend_graph = nullptr;
        frontend_input = {};
        frontend_output = {};
        if (frontend_gallocr != nullptr) {
            ggml_gallocr_free(frontend_gallocr);
            frontend_gallocr = nullptr;
        }
        frontend_graph_ctx.reset();
        ggml_init_params params{64ull * 1024ull * 1024ull, nullptr, true};
        frontend_graph_ctx.reset(ggml_init(params));
        if (frontend_graph_ctx == nullptr) {
            throw std::runtime_error("failed to initialize CLAP audio frontend graph context");
        }
        core::ModuleBuildContext ctx{frontend_graph_ctx.get(), "framework.clap_audio.frontend", execution.backend_type()};
        frontend_input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, 1, config.max_samples}));
        frontend_output = build_logmel_frontend(ctx, frontend_input, weights, config, options);
        frontend_graph = ggml_new_graph_custom(frontend_graph_ctx.get(), 131072, false);
        ggml_build_forward_expand(frontend_graph, frontend_output.tensor);
        frontend_gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
        if (frontend_gallocr == nullptr) {
            throw std::runtime_error("failed to create CLAP audio frontend graph allocator");
        }
        if (!ggml_gallocr_reserve(frontend_gallocr, frontend_graph) ||
            !ggml_gallocr_alloc_graph(frontend_gallocr, frontend_graph)) {
            throw std::runtime_error("failed to allocate CLAP audio frontend graph");
        }
        cached_frontend_batch = batch;
    }

    void ensure_graph(int64_t batch) {
        if (graph != nullptr && cached_batch == batch) {
            return;
        }
        release_runtime_graphs();
        ggml_init_params params{options.graph_arena_bytes, nullptr, true};
        graph_ctx.reset(ggml_init(params));
        if (graph_ctx == nullptr) {
            throw std::runtime_error("failed to initialize CLAP audio graph context");
        }
        core::ModuleBuildContext ctx{graph_ctx.get(), "framework.clap_audio", execution.backend_type()};
        input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, 1, config.spec_size, config.spec_size}));
        output = build_encoder(ctx, input, weights, config, options);
        graph = ggml_new_graph_custom(graph_ctx.get(), 1048576, false);
        ggml_build_forward_expand(graph, output.tensor);
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
        if (gallocr == nullptr) {
            throw std::runtime_error("failed to create CLAP audio graph allocator");
        }
        if (!ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
            throw std::runtime_error("failed to allocate CLAP audio graph");
        }
        cached_batch = batch;
    }

    std::shared_ptr<const assets::TensorSource> source;
    core::ExecutionContext & execution;
    ClapAudioConfig config;
    ClapAudioRuntimeOptions options;
    ClapWeights weights;
    int64_t cached_frontend_batch = 0;
    int64_t cached_batch = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> frontend_graph_ctx;
    std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx;
    ggml_gallocr_t frontend_gallocr = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_cgraph * frontend_graph = nullptr;
    ggml_cgraph * graph = nullptr;
    core::TensorValue frontend_input;
    core::TensorValue frontend_output;
    core::TensorValue input;
    core::TensorValue output;
};

ClapAudioConditionerRuntime::ClapAudioConditionerRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    core::ExecutionContext & execution,
    ClapAudioConfig config,
    ClapAudioRuntimeOptions options) {
    if (!source) {
        throw std::runtime_error("CLAP audio runtime requires a tensor source");
    }
    validate_config(config);
    impl_ = std::make_unique<Impl>(std::move(source), execution, std::move(config), options);
}

ClapAudioConditionerRuntime::~ClapAudioConditionerRuntime() = default;

ClapAudioEmbedding ClapAudioConditionerRuntime::encode_audio(
    const std::vector<float> & waveform,
    int64_t batch,
    int64_t samples,
    size_t threads) {
    if (impl_ == nullptr) {
        throw std::runtime_error("CLAP audio runtime is not initialized");
    }
    return impl_->encode_audio(waveform, batch, samples, threads);
}

void ClapAudioConditionerRuntime::release_runtime_graphs() {
    if (impl_ != nullptr) {
        impl_->release_runtime_graphs();
    }
}

}  // namespace engine::conditioners
