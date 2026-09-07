#include "engine/models/higgs_audio_tts/codec.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::higgs_audio_tts {
namespace {

using Clock = std::chrono::steady_clock;

constexpr const char * kCodecPrefix = "tied.embedding.modality_embeddings.0.model.";
constexpr int64_t kCodecCodebooks = 8;
constexpr int64_t kCodecCodebookSize = 1024;
constexpr int64_t kCodecCodebookDim = 64;
constexpr int64_t kCodecHiddenSize = 1024;
constexpr int64_t kAcousticHiddenSize = 256;
constexpr int64_t kSemanticHiddenSize = 768;
constexpr int64_t kCodecProjectInputSize = kAcousticHiddenSize + kSemanticHiddenSize;
constexpr int64_t kResidualUnitsPerBlock = 3;
constexpr int64_t kSemanticResidualUnitsPerBlock = 2;
constexpr int64_t kSemanticIntermediateSize = 3072;
constexpr int64_t kSemanticAttentionHeads = 12;
constexpr int64_t kSemanticLayers = 12;
constexpr int64_t kSemanticConvLayers = 7;
constexpr int64_t kSemanticSampleRate = 16000;
constexpr int64_t kSemanticPadSamples = 160;
constexpr float kSemanticLayerNormEps = 1.0e-5F;

const int64_t kUpsampleRatios[] = {8, 5, 4, 2, 3};
const int64_t kDecoderChannels[] = {1024, 512, 256, 128, 64, 32};
const int64_t kEncoderChannels[] = {64, 128, 256, 512, 1024, 2048};
const int64_t kSemanticConvDim[] = {512, 512, 512, 512, 512, 512, 512};
const int64_t kSemanticConvKernel[] = {10, 3, 3, 3, 3, 2, 2};
const int64_t kSemanticConvStride[] = {5, 2, 2, 2, 2, 2, 2};
constexpr int64_t kDecoderBlockCount = 5;
constexpr int64_t kEncoderBlockCount = 5;
constexpr int64_t kSemanticBlockCount = 2;
constexpr int kCodecSampleRate = 24000;
constexpr int64_t kCodecDecodeCapacityBucketFrames = 32;
constexpr int64_t kCodecHopLength = 960;
constexpr int64_t kCodecPadSamples = kCodecHopLength / 2;
constexpr int64_t kCodecDecodeWindowFrames = 128;
constexpr int64_t kCodecDecodeContextFrames = 32;
constexpr int64_t kCodecFullDecodeMaxFrames = 512;
// The decoder is a non-causal conv stack, so the last frames of a stream see zero padding
// on their right instead of real context and decode as a rising hiss over the final
// ~300 ms (worst on short utterances). Repeating the last frame past the end gives those
// frames context; the extra audio is trimmed again. Eight frames is enough — 16 and 32
// produce identical samples.
constexpr int64_t kCodecTailContextFrames = 8;
constexpr int64_t kResidualDilations[] = {1, 3, 9};

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

std::string codec_name(const std::string & name) { return std::string(kCodecPrefix) + name; }

modules::LinearWeights load_linear(core::BackendWeightStore & store,
                                   const assets::TensorSource & source,
                                   const std::string & name,
                                   assets::TensorStorageType storage_type,
                                   int64_t out_features,
                                   int64_t in_features,
                                   bool use_bias) {
    modules::LinearWeights weights;
    weights.weight = store.load_tensor(
        source, codec_name(name + ".weight"), storage_type, {out_features, in_features});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, codec_name(name + ".bias"), {out_features});
    }
    return weights;
}

modules::Conv1dWeights load_conv1d(core::BackendWeightStore & store,
                                   const assets::TensorSource & source,
                                   const std::string & name,
                                   assets::TensorStorageType storage_type,
                                   int64_t out_channels,
                                   int64_t in_channels,
                                   int64_t kernel_size,
                                   bool use_bias) {
    modules::Conv1dWeights weights;
    weights.weight = store.load_tensor(source,
                                       codec_name(name + ".weight"),
                                       storage_type,
                                       {out_channels, in_channels, kernel_size});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, codec_name(name + ".bias"), {out_channels});
    }
    return weights;
}

modules::ConvTranspose1dWeights load_conv_transpose1d(core::BackendWeightStore & store,
                                                      const assets::TensorSource & source,
                                                      const std::string & name,
                                                      assets::TensorStorageType storage_type,
                                                      int64_t in_channels,
                                                      int64_t out_channels,
                                                      int64_t kernel_size,
                                                      bool use_bias) {
    modules::ConvTranspose1dWeights weights;
    weights.weight = store.load_tensor(source,
                                       codec_name(name + ".weight"),
                                       storage_type,
                                       {in_channels, out_channels, kernel_size});
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, codec_name(name + ".bias"), {out_channels});
    }
    return weights;
}

modules::Snake1dWeights load_snake(core::BackendWeightStore & store,
                                   const assets::TensorSource & source,
                                   const std::string & name,
                                   int64_t channels) {
    const auto raw = source.require_f32(codec_name(name), std::vector<int64_t>{1, channels, 1});
    return {store.make_f32(core::TensorShape::from_dims({channels}), raw)};
}

core::TensorValue require_semantic_tensor(const HiggsCodecWeights & weights,
                                          const std::string & name) {
    const auto it = weights.semantic_model.find(name);
    if (it == weights.semantic_model.end()) {
        throw std::runtime_error("Higgs TTS codec missing semantic tensor: " + name);
    }
    return it->second;
}

modules::NormWeights semantic_norm_weights(const HiggsCodecWeights & weights,
                                           const std::string & prefix) {
    return modules::NormWeights{require_semantic_tensor(weights, prefix + ".weight"),
                                require_semantic_tensor(weights, prefix + ".bias")};
}

modules::LinearWeights semantic_linear_weights(const HiggsCodecWeights & weights,
                                               const std::string & prefix) {
    return modules::LinearWeights{require_semantic_tensor(weights, prefix + ".weight"),
                                  require_semantic_tensor(weights, prefix + ".bias")};
}

modules::Conv1dWeights semantic_conv_weights(const HiggsCodecWeights & weights,
                                             const std::string & prefix,
                                             bool use_bias) {
    modules::Conv1dWeights out;
    out.weight = require_semantic_tensor(weights, prefix + ".weight");
    if (use_bias) {
        out.bias = require_semantic_tensor(weights, prefix + ".bias");
    }
    return out;
}

int64_t
conv1d_output_frames(int64_t input_frames, int64_t kernel, int64_t stride, int64_t padding) {
    return (input_frames + 2 * padding - kernel) / stride + 1;
}

int64_t ceil_div(int64_t value, int64_t divisor) { return (value + divisor - 1) / divisor; }

int64_t semantic_feature_frames(int64_t input_samples) {
    int64_t frames = input_samples;
    for (int64_t index = 0; index < kSemanticConvLayers; ++index) {
        frames = conv1d_output_frames(frames,
                                      kSemanticConvKernel[static_cast<size_t>(index)],
                                      kSemanticConvStride[static_cast<size_t>(index)],
                                      0);
        if (frames <= 0) {
            throw std::runtime_error("Higgs TTS semantic encoder input is too short");
        }
    }
    return ceil_div(frames, 2);
}

int64_t acoustic_encoder_frames(int64_t input_samples) {
    int64_t frames = conv1d_output_frames(input_samples, 7, 1, 3);
    for (int64_t block = 0; block < kEncoderBlockCount; ++block) {
        const int64_t ratio = kUpsampleRatios[static_cast<size_t>(block)];
        frames = conv1d_output_frames(frames, 2 * ratio, ratio, (ratio + 1) / 2);
        if (frames <= 0) {
            throw std::runtime_error("Higgs TTS acoustic encoder input is too short");
        }
    }
    frames = conv1d_output_frames(frames, 3, 1, 1);
    if (frames <= 0) {
        throw std::runtime_error("Higgs TTS acoustic encoder input is too short");
    }
    return frames;
}

std::vector<float> pad_zeros(const std::vector<float> & input, int64_t left, int64_t right) {
    std::vector<float> out(static_cast<size_t>(left + static_cast<int64_t>(input.size()) + right),
                           0.0F);
    std::copy(input.begin(), input.end(), out.begin() + left);
    return out;
}

std::vector<float> resample_mono_if_needed(const std::vector<float> & mono,
                                           int source_sample_rate,
                                           int target_sample_rate) {
    if (source_sample_rate == target_sample_rate) {
        return mono;
    }
    auto out =
        audio::resample_mono_torchaudio_sinc_hann(mono, source_sample_rate, target_sample_rate);
    if (out.empty()) {
        throw std::runtime_error("Higgs TTS codec resampling produced no samples");
    }
    return out;
}

std::vector<float> prepare_mono_audio(const runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("Higgs TTS codec reference audio sample rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error("Higgs TTS codec reference audio channel count must be positive");
    }
    if (audio.samples.empty()) {
        throw std::runtime_error("Higgs TTS codec reference audio is empty");
    }
    if ((audio.samples.size() % static_cast<size_t>(audio.channels)) != 0) {
        throw std::runtime_error("Higgs TTS codec reference audio sample count is "
                                 "not divisible by channels");
    }
    return audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
}

std::vector<float> prepare_codec_audio_24k(const std::vector<float> & mono,
                                           int source_sample_rate) {
    auto audio_24k = resample_mono_if_needed(mono, source_sample_rate, kCodecSampleRate);
    if (static_cast<int64_t>(audio_24k.size()) < kCodecSampleRate) {
        audio_24k.resize(static_cast<size_t>(kCodecSampleRate), 0.0F);
    }
    return audio_24k;
}

std::vector<float> prepare_semantic_audio_16k(const std::vector<float> & mono,
                                              int source_sample_rate) {
    auto semantic_16k = resample_mono_if_needed(mono, source_sample_rate, kSemanticSampleRate);
    return pad_zeros(semantic_16k, kSemanticPadSamples, kSemanticPadSamples);
}

std::vector<float> effective_semantic_pos_conv_weight(const assets::TensorSource & source,
                                                      int64_t out_channels,
                                                      int64_t in_channels,
                                                      int64_t kernel_size) {
    const auto g = source.require_f32(codec_name("semantic_model.encoder.pos_conv_embed."
                                                 "conv.parametrizations.weight.original0"),
                                      {1, 1, kernel_size});
    const auto v = source.require_f32(codec_name("semantic_model.encoder.pos_conv_embed."
                                                 "conv.parametrizations.weight.original1"),
                                      {out_channels, in_channels, kernel_size});
    std::vector<float> weight(v.size());
    for (int64_t k = 0; k < kernel_size; ++k) {
        double sum = 0.0;
        for (int64_t out = 0; out < out_channels; ++out) {
            for (int64_t in = 0; in < in_channels; ++in) {
                const size_t index =
                    static_cast<size_t>((out * in_channels + in) * kernel_size + k);
                sum += static_cast<double>(v[index]) * static_cast<double>(v[index]);
            }
        }
        const double norm = std::sqrt(sum);
        if (norm == 0.0) {
            throw std::runtime_error("Higgs TTS semantic positional-conv weight norm is zero");
        }
        const float scale_value =
            static_cast<float>(static_cast<double>(g[static_cast<size_t>(k)]) / norm);
        for (int64_t out = 0; out < out_channels; ++out) {
            for (int64_t in = 0; in < in_channels; ++in) {
                const size_t index =
                    static_cast<size_t>((out * in_channels + in) * kernel_size + k);
                weight[index] = v[index] * scale_value;
            }
        }
    }
    return weight;
}

void load_semantic_tensor(HiggsCodecWeights & weights,
                          const assets::TensorSource & source,
                          const std::string & name,
                          const std::vector<int64_t> & shape,
                          assets::TensorStorageType storage_type) {
    weights.semantic_model.emplace(
        name,
        weights.store->load_tensor(
            source, codec_name("semantic_model." + name), storage_type, shape));
}

void load_semantic_f32_tensor(HiggsCodecWeights & weights,
                              const assets::TensorSource & source,
                              const std::string & name,
                              const std::vector<int64_t> & shape) {
    weights.semantic_model.emplace(
        name, weights.store->load_f32_tensor(source, codec_name("semantic_model." + name), shape));
}

void load_hubert_semantic_model_weights(HiggsCodecWeights & weights,
                                        const assets::TensorSource & source,
                                        assets::TensorStorageType storage_type) {
    for (int64_t layer = 0; layer < kSemanticConvLayers; ++layer) {
        const std::string prefix = "feature_extractor.conv_layers." + std::to_string(layer);
        load_semantic_tensor(weights,
                             source,
                             prefix + ".conv.weight",
                             {kSemanticConvDim[static_cast<size_t>(layer)],
                              layer == 0 ? 1 : kSemanticConvDim[static_cast<size_t>(layer - 1)],
                              kSemanticConvKernel[static_cast<size_t>(layer)]},
                             storage_type);
        if (layer == 0) {
            load_semantic_f32_tensor(
                weights, source, prefix + ".layer_norm.weight", {kSemanticConvDim[0]});
            load_semantic_f32_tensor(
                weights, source, prefix + ".layer_norm.bias", {kSemanticConvDim[0]});
        }
    }
    load_semantic_f32_tensor(
        weights, source, "feature_projection.layer_norm.weight", {kSemanticConvDim[6]});
    load_semantic_f32_tensor(
        weights, source, "feature_projection.layer_norm.bias", {kSemanticConvDim[6]});
    load_semantic_tensor(weights,
                         source,
                         "feature_projection.projection.weight",
                         {kSemanticHiddenSize, kSemanticConvDim[6]},
                         storage_type);
    load_semantic_f32_tensor(
        weights, source, "feature_projection.projection.bias", {kSemanticHiddenSize});
    load_semantic_f32_tensor(weights, source, "encoder.layer_norm.weight", {kSemanticHiddenSize});
    load_semantic_f32_tensor(weights, source, "encoder.layer_norm.bias", {kSemanticHiddenSize});
    weights.semantic_model.emplace(
        "encoder.pos_conv_embed.conv.weight",
        weights.store->make_f32(
            core::TensorShape::from_dims({kSemanticHiddenSize, kSemanticHiddenSize / 16, 128}),
            effective_semantic_pos_conv_weight(
                source, kSemanticHiddenSize, kSemanticHiddenSize / 16, 128)));
    load_semantic_f32_tensor(
        weights, source, "encoder.pos_conv_embed.conv.bias", {kSemanticHiddenSize});
    for (int64_t layer = 0; layer < kSemanticLayers; ++layer) {
        const std::string prefix = "encoder.layers." + std::to_string(layer);
        load_semantic_f32_tensor(
            weights, source, prefix + ".layer_norm.weight", {kSemanticHiddenSize});
        load_semantic_f32_tensor(
            weights, source, prefix + ".layer_norm.bias", {kSemanticHiddenSize});
        load_semantic_f32_tensor(
            weights, source, prefix + ".final_layer_norm.weight", {kSemanticHiddenSize});
        load_semantic_f32_tensor(
            weights, source, prefix + ".final_layer_norm.bias", {kSemanticHiddenSize});
        load_semantic_tensor(weights,
                             source,
                             prefix + ".attention.q_proj.weight",
                             {kSemanticHiddenSize, kSemanticHiddenSize},
                             storage_type);
        load_semantic_f32_tensor(
            weights, source, prefix + ".attention.q_proj.bias", {kSemanticHiddenSize});
        load_semantic_tensor(weights,
                             source,
                             prefix + ".attention.k_proj.weight",
                             {kSemanticHiddenSize, kSemanticHiddenSize},
                             storage_type);
        load_semantic_f32_tensor(
            weights, source, prefix + ".attention.k_proj.bias", {kSemanticHiddenSize});
        load_semantic_tensor(weights,
                             source,
                             prefix + ".attention.v_proj.weight",
                             {kSemanticHiddenSize, kSemanticHiddenSize},
                             storage_type);
        load_semantic_f32_tensor(
            weights, source, prefix + ".attention.v_proj.bias", {kSemanticHiddenSize});
        load_semantic_tensor(weights,
                             source,
                             prefix + ".attention.out_proj.weight",
                             {kSemanticHiddenSize, kSemanticHiddenSize},
                             storage_type);
        load_semantic_f32_tensor(
            weights, source, prefix + ".attention.out_proj.bias", {kSemanticHiddenSize});
        load_semantic_tensor(weights,
                             source,
                             prefix + ".feed_forward.intermediate_dense.weight",
                             {kSemanticIntermediateSize, kSemanticHiddenSize},
                             storage_type);
        load_semantic_f32_tensor(weights,
                                 source,
                                 prefix + ".feed_forward.intermediate_dense.bias",
                                 {kSemanticIntermediateSize});
        load_semantic_tensor(weights,
                             source,
                             prefix + ".feed_forward.output_dense.weight",
                             {kSemanticHiddenSize, kSemanticIntermediateSize},
                             storage_type);
        load_semantic_f32_tensor(
            weights, source, prefix + ".feed_forward.output_dense.bias", {kSemanticHiddenSize});
    }
}

HiggsCodecResidualUnitWeights load_residual_unit(core::BackendWeightStore & store,
                                                 const assets::TensorSource & source,
                                                 const std::string & prefix,
                                                 assets::TensorStorageType storage_type,
                                                 int64_t channels) {
    HiggsCodecResidualUnitWeights weights;
    weights.snake1 = load_snake(store, source, prefix + ".snake1.alpha", channels);
    weights.conv1 =
        load_conv1d(store, source, prefix + ".conv1", storage_type, channels, channels, 7, true);
    weights.snake2 = load_snake(store, source, prefix + ".snake2.alpha", channels);
    weights.conv2 =
        load_conv1d(store, source, prefix + ".conv2", storage_type, channels, channels, 1, true);
    return weights;
}

HiggsCodecDecoderBlockWeights load_decoder_block(core::BackendWeightStore & store,
                                                 const assets::TensorSource & source,
                                                 int64_t block_index,
                                                 assets::TensorStorageType storage_type) {
    if (block_index < 0 || block_index >= kDecoderBlockCount) {
        throw std::runtime_error("Higgs TTS codec decoder block index is out of range");
    }
    const int64_t in_channels = kDecoderChannels[static_cast<size_t>(block_index)];
    const int64_t out_channels = kDecoderChannels[static_cast<size_t>(block_index + 1)];
    const int64_t ratio = kUpsampleRatios[static_cast<size_t>(block_index)];
    const std::string prefix = "acoustic_decoder.block." + std::to_string(block_index);

    HiggsCodecDecoderBlockWeights weights;
    weights.snake = load_snake(store, source, prefix + ".snake1.alpha", in_channels);
    weights.conv_transpose = load_conv_transpose1d(store,
                                                   source,
                                                   prefix + ".conv_t1",
                                                   storage_type,
                                                   in_channels,
                                                   out_channels,
                                                   2 * ratio,
                                                   true);
    weights.residual_units.reserve(kResidualUnitsPerBlock);
    for (int64_t unit = 0; unit < kResidualUnitsPerBlock; ++unit) {
        weights.residual_units.push_back(
            load_residual_unit(store,
                               source,
                               prefix + ".res_unit" + std::to_string(unit + 1),
                               storage_type,
                               out_channels));
    }
    return weights;
}

HiggsCodecEncoderBlockWeights load_encoder_block(core::BackendWeightStore & store,
                                                 const assets::TensorSource & source,
                                                 int64_t block_index,
                                                 assets::TensorStorageType storage_type) {
    if (block_index < 0 || block_index >= kEncoderBlockCount) {
        throw std::runtime_error("Higgs TTS codec encoder block index is out of range");
    }
    const int64_t in_channels = kEncoderChannels[static_cast<size_t>(block_index)];
    const int64_t out_channels = kEncoderChannels[static_cast<size_t>(block_index + 1)];
    const int64_t ratio = kUpsampleRatios[static_cast<size_t>(block_index)];
    const std::string prefix = "acoustic_encoder.block." + std::to_string(block_index);

    HiggsCodecEncoderBlockWeights weights;
    weights.snake = load_snake(store, source, prefix + ".snake1.alpha", in_channels);
    weights.conv = load_conv1d(
        store, source, prefix + ".conv1", storage_type, out_channels, in_channels, 2 * ratio, true);
    weights.residual_units.reserve(kResidualUnitsPerBlock);
    for (int64_t unit = 0; unit < kResidualUnitsPerBlock; ++unit) {
        weights.residual_units.push_back(
            load_residual_unit(store,
                               source,
                               prefix + ".res_unit" + std::to_string(unit + 1),
                               storage_type,
                               in_channels));
    }
    return weights;
}

HiggsCodecSemanticResidualUnitWeights
load_semantic_residual_unit(core::BackendWeightStore & store,
                            const assets::TensorSource & source,
                            const std::string & prefix,
                            assets::TensorStorageType storage_type) {
    HiggsCodecSemanticResidualUnitWeights weights;
    weights.conv1 = load_conv1d(store,
                                source,
                                prefix + ".conv1",
                                storage_type,
                                kSemanticHiddenSize,
                                kSemanticHiddenSize,
                                3,
                                false);
    weights.conv2 = load_conv1d(store,
                                source,
                                prefix + ".conv2",
                                storage_type,
                                kSemanticHiddenSize,
                                kSemanticHiddenSize,
                                1,
                                false);
    return weights;
}

HiggsCodecSemanticEncoderBlockWeights
load_semantic_encoder_block(core::BackendWeightStore & store,
                            const assets::TensorSource & source,
                            int64_t block_index,
                            assets::TensorStorageType storage_type) {
    if (block_index < 0 || block_index >= kSemanticBlockCount) {
        throw std::runtime_error("Higgs TTS codec semantic encoder block index is out of range");
    }
    const std::string prefix = "encoder_semantic.conv_blocks." + std::to_string(block_index);
    HiggsCodecSemanticEncoderBlockWeights weights;
    weights.residual_units.reserve(kSemanticResidualUnitsPerBlock);
    for (int64_t unit = 0; unit < kSemanticResidualUnitsPerBlock; ++unit) {
        weights.residual_units.push_back(load_semantic_residual_unit(
            store, source, prefix + ".res_units." + std::to_string(unit), storage_type));
    }
    weights.conv = load_conv1d(store,
                               source,
                               prefix + ".conv",
                               storage_type,
                               kSemanticHiddenSize,
                               kSemanticHiddenSize,
                               3,
                               true);
    return weights;
}

HiggsCodecVectorQuantizerWeights load_quantizer(core::BackendWeightStore & store,
                                                const assets::TensorSource & source,
                                                int64_t index,
                                                assets::TensorStorageType storage_type) {
    const std::string prefix = "quantizer.quantizers." + std::to_string(index);
    HiggsCodecVectorQuantizerWeights weights;
    weights.codebook = store.load_tensor(source,
                                         codec_name(prefix + ".codebook.embed"),
                                         storage_type,
                                         {kCodecCodebookSize, kCodecCodebookDim});
    weights.project_in = load_linear(store,
                                     source,
                                     prefix + ".project_in",
                                     storage_type,
                                     kCodecCodebookDim,
                                     kCodecHiddenSize,
                                     true);
    weights.project_out = load_linear(store,
                                      source,
                                      prefix + ".project_out",
                                      storage_type,
                                      kCodecHiddenSize,
                                      kCodecCodebookDim,
                                      true);
    return weights;
}

core::TensorValue
conv_transpose_with_adjusted_output_padding(core::ModuleBuildContext & ctx,
                                            const core::TensorValue & input,
                                            const modules::ConvTranspose1dWeights & weights,
                                            int64_t in_channels,
                                            int64_t out_channels,
                                            int64_t ratio) {
    const int64_t kernel = 2 * ratio;
    const int64_t padding = (ratio + 1) / 2;
    const int64_t output_padding = ratio % 2;
    auto full = modules::ConvTranspose1dModule({
                                                   in_channels,
                                                   out_channels,
                                                   kernel,
                                                   static_cast<int>(ratio),
                                                   0,
                                                   1,
                                                   weights.bias.has_value(),
                                               })
                    .build(ctx, input, weights);
    const int64_t cropped_frames =
        (input.shape.dims[2] - 1) * ratio - 2 * padding + kernel + output_padding;
    if (cropped_frames <= 0 || cropped_frames > full.shape.dims[2]) {
        throw std::runtime_error(
            "Higgs TTS codec adjusted ConvTranspose1d output length is invalid");
    }
    return modules::SliceModule({2, padding, cropped_frames}).build(ctx, full);
}

core::TensorValue dac_snake(core::ModuleBuildContext & ctx,
                            const core::TensorValue & input,
                            const modules::Snake1dWeights & weights,
                            int64_t channels) {
    const auto input_ready = core::ensure_backend_addressable_layout(ctx, input);
    const auto input_f32 =
        input_ready.type == GGML_TYPE_F32
            ? input_ready
            : core::wrap_tensor(ggml_cast(ctx.ggml, input_ready.tensor, GGML_TYPE_F32),
                                input_ready.shape,
                                GGML_TYPE_F32);
    auto alpha =
        core::reshape_tensor(ctx, weights.alpha, core::TensorShape::from_dims({1, channels, 1}));
    auto ax = core::wrap_tensor(
        ggml_mul(ctx.ggml, input_f32.tensor, alpha.tensor), input_f32.shape, GGML_TYPE_F32);
    auto s = core::wrap_tensor(ggml_sin(ctx.ggml, ax.tensor), input_f32.shape, GGML_TYPE_F32);
    auto s2 =
        core::wrap_tensor(ggml_mul(ctx.ggml, s.tensor, s.tensor), input_f32.shape, GGML_TYPE_F32);
    auto denom = core::wrap_tensor(
        ggml_scale_bias(ctx.ggml, alpha.tensor, 1.0F, 1.0e-9F), alpha.shape, GGML_TYPE_F32);
    auto periodic = core::wrap_tensor(
        ggml_div(ctx.ggml, s2.tensor, denom.tensor), input_f32.shape, GGML_TYPE_F32);
    return core::wrap_tensor(
        ggml_add(ctx.ggml, input_f32.tensor, periodic.tensor), input_f32.shape, GGML_TYPE_F32);
}

core::TensorValue residual_unit(core::ModuleBuildContext & ctx,
                                const core::TensorValue & input,
                                const HiggsCodecResidualUnitWeights & weights,
                                int64_t channels,
                                int64_t dilation) {
    auto hidden = dac_snake(ctx, input, weights.snake1, channels);
    hidden = modules::Conv1dModule({channels,
                                    channels,
                                    7,
                                    1,
                                    static_cast<int>(3 * dilation),
                                    static_cast<int>(dilation),
                                    true})
                 .build(ctx, hidden, weights.conv1);
    hidden = dac_snake(ctx, hidden, weights.snake2, channels);
    hidden = modules::Conv1dModule({channels, channels, 1, 1, 0, 1, true})
                 .build(ctx, hidden, weights.conv2);
    return modules::AddModule{}.build(ctx, input, hidden);
}

core::TensorValue contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(ctx, value);
}

core::TensorValue transpose_bct_btc(core::ModuleBuildContext & ctx,
                                    const core::TensorValue & value) {
    return modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, value);
}

core::TensorValue add_same(core::ModuleBuildContext & ctx,
                           const core::TensorValue & lhs,
                           const core::TensorValue & rhs) {
    return modules::AddModule{}.build(ctx, lhs, rhs);
}

core::TensorValue
scale(core::ModuleBuildContext & ctx, const core::TensorValue & value, float factor) {
    return core::wrap_tensor(
        ggml_scale(ctx.ggml, contiguous(ctx, value).tensor, factor), value.shape, GGML_TYPE_F32);
}

core::TensorValue group_norm_affine(core::ModuleBuildContext & ctx,
                                    const core::TensorValue & input,
                                    int64_t groups,
                                    float eps,
                                    const modules::NormWeights & weights) {
    core::TensorValue output;
    if (input.shape.rank == 3 && groups == input.shape.dims[1]) {
        auto input_f32 = input.type == GGML_TYPE_F32
                             ? input
                             : core::wrap_tensor(ggml_cast(ctx.ggml, input.tensor, GGML_TYPE_F32),
                                                 input.shape,
                                                 GGML_TYPE_F32);
        auto mean = modules::ReduceMeanModule({2}).build(ctx, input_f32);
        auto mean_rep = core::wrap_tensor(
            ggml_repeat(ctx.ggml, mean.tensor, input_f32.tensor), input_f32.shape, GGML_TYPE_F32);
        auto centered = core::wrap_tensor(
            ggml_sub(ctx.ggml, input_f32.tensor, mean_rep.tensor), input_f32.shape, GGML_TYPE_F32);
        auto variance = modules::ReduceMeanModule({2}).build(
            ctx, modules::MulModule().build(ctx, centered, centered));
        auto stddev = core::wrap_tensor(
            ggml_sqrt(ctx.ggml, ggml_scale_bias(ctx.ggml, variance.tensor, 1.0F, eps)),
            variance.shape,
            GGML_TYPE_F32);
        auto stddev_rep = core::wrap_tensor(
            ggml_repeat(ctx.ggml, stddev.tensor, input_f32.tensor), input_f32.shape, GGML_TYPE_F32);
        output = core::wrap_tensor(
            ggml_div(ctx.ggml, centered.tensor, stddev_rep.tensor), input_f32.shape, GGML_TYPE_F32);
    } else {
        output = core::wrap_tensor(
            ggml_group_norm(ctx.ggml, input.tensor, groups, eps), input.shape, GGML_TYPE_F32);
    }
    if (weights.weight.has_value()) {
        auto weight = core::reshape_tensor(
            ctx, *weights.weight, core::TensorShape::from_dims({1, input.shape.dims[1], 1}));
        auto repeated = core::wrap_tensor(
            ggml_repeat(ctx.ggml, weight.tensor, output.tensor), output.shape, GGML_TYPE_F32);
        output = core::wrap_tensor(
            ggml_mul(ctx.ggml, output.tensor, repeated.tensor), output.shape, GGML_TYPE_F32);
    }
    if (weights.bias.has_value()) {
        auto bias = core::reshape_tensor(
            ctx, *weights.bias, core::TensorShape::from_dims({1, input.shape.dims[1], 1}));
        auto repeated = core::wrap_tensor(
            ggml_repeat(ctx.ggml, bias.tensor, output.tensor), output.shape, GGML_TYPE_F32);
        output = core::wrap_tensor(
            ggml_add(ctx.ggml, output.tensor, repeated.tensor), output.shape, GGML_TYPE_F32);
    }
    return output;
}

core::TensorValue grouped_pos_conv(core::ModuleBuildContext & ctx,
                                   const core::TensorValue & input_bct,
                                   const modules::Conv1dWeights & weights) {
    constexpr int64_t groups = 16;
    constexpr int64_t channels_per_group = kSemanticHiddenSize / groups;
    const auto input_contiguous = contiguous(ctx, input_bct);
    core::TensorValue out;
    for (int64_t group = 0; group < groups; ++group) {
        auto input_group = modules::SliceModule({1, group * channels_per_group, channels_per_group})
                               .build(ctx, input_contiguous);
        auto weight_group =
            modules::SliceModule({0, group * channels_per_group, channels_per_group})
                .build(ctx, weights.weight);
        modules::Conv1dWeights group_weights{weight_group, std::nullopt};
        if (weights.bias.has_value()) {
            group_weights.bias =
                modules::SliceModule({0, group * channels_per_group, channels_per_group})
                    .build(ctx, *weights.bias);
        }
        auto group_out =
            modules::Conv1dModule(
                {channels_per_group, channels_per_group, 128, 1, 64, 1, weights.bias.has_value()})
                .build(ctx, input_group, group_weights);
        out = out.valid() ? modules::ConcatModule({1}).build(ctx, out, group_out) : group_out;
    }
    out = modules::SliceModule({2, 0, input_bct.shape.dims[2]}).build(ctx, out);
    return modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, out);
}

core::TensorValue semantic_self_attention(core::ModuleBuildContext & ctx,
                                          const core::TensorValue & hidden_btc,
                                          const HiggsCodecWeights & weights,
                                          int64_t layer_index) {
    constexpr int64_t head_dim = kSemanticHiddenSize / kSemanticAttentionHeads;
    const std::string prefix = "encoder.layers." + std::to_string(layer_index) + ".attention";
    auto q = modules::LinearModule({kSemanticHiddenSize, kSemanticHiddenSize, true, GGML_PREC_F32})
                 .build(ctx, hidden_btc, semantic_linear_weights(weights, prefix + ".q_proj"));
    auto k = modules::LinearModule({kSemanticHiddenSize, kSemanticHiddenSize, true, GGML_PREC_F32})
                 .build(ctx, hidden_btc, semantic_linear_weights(weights, prefix + ".k_proj"));
    auto v = modules::LinearModule({kSemanticHiddenSize, kSemanticHiddenSize, true, GGML_PREC_F32})
                 .build(ctx, hidden_btc, semantic_linear_weights(weights, prefix + ".v_proj"));
    q = core::reshape_tensor(ctx,
                             contiguous(ctx, q),
                             core::TensorShape::from_dims({hidden_btc.shape.dims[0],
                                                           hidden_btc.shape.dims[1],
                                                           kSemanticAttentionHeads,
                                                           head_dim}));
    k = core::reshape_tensor(ctx,
                             contiguous(ctx, k),
                             core::TensorShape::from_dims({hidden_btc.shape.dims[0],
                                                           hidden_btc.shape.dims[1],
                                                           kSemanticAttentionHeads,
                                                           head_dim}));
    v = core::reshape_tensor(ctx,
                             contiguous(ctx, v),
                             core::TensorShape::from_dims({hidden_btc.shape.dims[0],
                                                           hidden_btc.shape.dims[1],
                                                           kSemanticAttentionHeads,
                                                           head_dim}));
    q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
    k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
    v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
    const auto k_t = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k);
    auto scores = modules::MatMulModule().build(ctx, q, k_t);
    scores = scale(ctx, scores, static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim))));
    auto attn = core::wrap_tensor(
        ggml_soft_max(ctx.ggml, contiguous(ctx, scores).tensor), scores.shape, GGML_TYPE_F32);
    auto context = modules::MatMulModule().build(ctx, attn, v);
    context = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, context);
    context = core::reshape_tensor(
        ctx,
        contiguous(ctx, context),
        core::TensorShape::from_dims(
            {hidden_btc.shape.dims[0], hidden_btc.shape.dims[1], kSemanticHiddenSize}));
    return modules::LinearModule({kSemanticHiddenSize, kSemanticHiddenSize, true, GGML_PREC_F32})
        .build(ctx, context, semantic_linear_weights(weights, prefix + ".out_proj"));
}

core::TensorValue semantic_feed_forward(core::ModuleBuildContext & ctx,
                                        const core::TensorValue & hidden_btc,
                                        const HiggsCodecWeights & weights,
                                        int64_t layer_index) {
    const std::string prefix = "encoder.layers." + std::to_string(layer_index) + ".feed_forward";
    auto x =
        modules::LinearModule({kSemanticHiddenSize, kSemanticIntermediateSize, true, GGML_PREC_F32})
            .build(
                ctx, hidden_btc, semantic_linear_weights(weights, prefix + ".intermediate_dense"));
    x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
    return modules::LinearModule(
               {kSemanticIntermediateSize, kSemanticHiddenSize, true, GGML_PREC_F32})
        .build(ctx, x, semantic_linear_weights(weights, prefix + ".output_dense"));
}

core::TensorValue downsample_time_by_2(core::ModuleBuildContext & ctx,
                                       const core::TensorValue & hidden_btc,
                                       int64_t target_frames) {
    core::TensorValue out;
    for (int64_t frame = 0; frame < target_frames; ++frame) {
        auto slice = modules::SliceModule({1, frame * 2, 1}).build(ctx, hidden_btc);
        out = out.valid() ? modules::ConcatModule({1}).build(ctx, out, slice) : slice;
    }
    return out;
}

struct HiggsCodecEncodeGraphValues {
    std::array<core::TensorValue, static_cast<size_t>(kCodecCodebooks)> codes = {};
};

core::TensorValue hubert_hidden_state_mean(core::ModuleBuildContext & ctx,
                                           const core::TensorValue & input_values,
                                           const HiggsCodecWeights & weights,
                                           int64_t target_frames) {
    auto hidden = core::reshape_tensor(
        ctx,
        input_values,
        core::TensorShape::from_dims({input_values.shape.dims[0], 1, input_values.shape.dims[1]}));
    int64_t in_channels = 1;
    for (int64_t index = 0; index < kSemanticConvLayers; ++index) {
        const std::string prefix = "feature_extractor.conv_layers." + std::to_string(index);
        hidden = modules::Conv1dModule(
                     {in_channels,
                      kSemanticConvDim[static_cast<size_t>(index)],
                      kSemanticConvKernel[static_cast<size_t>(index)],
                      static_cast<int>(kSemanticConvStride[static_cast<size_t>(index)]),
                      0,
                      1,
                      false})
                     .build(ctx, hidden, semantic_conv_weights(weights, prefix + ".conv", false));
        if (index == 0) {
            hidden = group_norm_affine(ctx,
                                       hidden,
                                       kSemanticConvDim[0],
                                       kSemanticLayerNormEps,
                                       semantic_norm_weights(weights, prefix + ".layer_norm"));
        }
        hidden = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, hidden);
        in_channels = kSemanticConvDim[static_cast<size_t>(index)];
    }
    hidden = transpose_bct_btc(ctx, hidden);
    hidden =
        modules::LayerNormModule({kSemanticConvDim[6], kSemanticLayerNormEps, true, true})
            .build(ctx, hidden, semantic_norm_weights(weights, "feature_projection.layer_norm"));
    hidden =
        modules::LinearModule({kSemanticConvDim[6], kSemanticHiddenSize, true, GGML_PREC_F32})
            .build(ctx, hidden, semantic_linear_weights(weights, "feature_projection.projection"));

    auto pos =
        grouped_pos_conv(ctx,
                         transpose_bct_btc(ctx, hidden),
                         semantic_conv_weights(weights, "encoder.pos_conv_embed.conv", true));
    hidden = add_same(ctx, hidden, transpose_bct_btc(ctx, pos));
    hidden = modules::LayerNormModule({kSemanticHiddenSize, kSemanticLayerNormEps, true, true})
                 .build(ctx, hidden, semantic_norm_weights(weights, "encoder.layer_norm"));

    auto sum = hidden;
    for (int64_t layer = 0; layer < kSemanticLayers; ++layer) {
        const std::string prefix = "encoder.layers." + std::to_string(layer);
        const auto attn_residual = hidden;
        hidden = semantic_self_attention(ctx, hidden, weights, layer);
        hidden = add_same(ctx, attn_residual, hidden);
        hidden = modules::LayerNormModule({kSemanticHiddenSize, kSemanticLayerNormEps, true, true})
                     .build(ctx, hidden, semantic_norm_weights(weights, prefix + ".layer_norm"));
        hidden = add_same(ctx, hidden, semantic_feed_forward(ctx, hidden, weights, layer));
        hidden =
            modules::LayerNormModule({kSemanticHiddenSize, kSemanticLayerNormEps, true, true})
                .build(ctx, hidden, semantic_norm_weights(weights, prefix + ".final_layer_norm"));
        sum = add_same(ctx, sum, hidden);
    }
    auto hidden_mean = scale(ctx, sum, 1.0F / static_cast<float>(kSemanticLayers + 1));
    auto features = downsample_time_by_2(ctx, hidden_mean, target_frames);
    return features;
}

core::TensorValue semantic_residual_unit(core::ModuleBuildContext & ctx,
                                         const core::TensorValue & input,
                                         const HiggsCodecSemanticResidualUnitWeights & weights) {
    auto hidden = modules::EluModule().build(ctx, input);
    hidden = modules::Conv1dModule({kSemanticHiddenSize, kSemanticHiddenSize, 3, 1, 1, 1, false})
                 .build(ctx, hidden, weights.conv1);
    hidden = modules::EluModule().build(ctx, hidden);
    hidden = modules::Conv1dModule({kSemanticHiddenSize, kSemanticHiddenSize, 1, 1, 0, 1, false})
                 .build(ctx, hidden, weights.conv2);
    return modules::AddModule{}.build(ctx, input, hidden);
}

core::TensorValue semantic_encoder(core::ModuleBuildContext & ctx,
                                   const core::TensorValue & hidden_btc,
                                   const HiggsCodecWeights & weights) {
    auto hidden = transpose_bct_btc(ctx, hidden_btc);
    hidden = modules::Conv1dModule({kSemanticHiddenSize, kSemanticHiddenSize, 3, 1, 1, 1, false})
                 .build(ctx, hidden, weights.semantic_encoder_input);
    for (const auto & block : weights.semantic_encoder_blocks) {
        for (const auto & unit : block.residual_units) {
            hidden = semantic_residual_unit(ctx, hidden, unit);
        }
        hidden = modules::Conv1dModule({kSemanticHiddenSize, kSemanticHiddenSize, 3, 1, 1, 1, true})
                     .build(ctx, hidden, block.conv);
    }
    return hidden;
}

core::TensorValue acoustic_encoder(core::ModuleBuildContext & ctx,
                                   const core::TensorValue & waveform,
                                   const HiggsCodecWeights & weights,
                                   int64_t target_frames) {
    auto hidden = core::reshape_tensor(
        ctx, waveform, core::TensorShape::from_dims({1, 1, waveform.shape.dims[0]}));
    hidden = modules::Conv1dModule({1, kEncoderChannels[0], 7, 1, 3, 1, true})
                 .build(ctx, hidden, weights.acoustic_encoder_input);
    for (int64_t block = 0; block < kEncoderBlockCount; ++block) {
        const int64_t in_channels = kEncoderChannels[static_cast<size_t>(block)];
        const int64_t out_channels = kEncoderChannels[static_cast<size_t>(block + 1)];
        const int64_t ratio = kUpsampleRatios[static_cast<size_t>(block)];
        const auto & block_weights = weights.acoustic_encoder_blocks[static_cast<size_t>(block)];
        for (size_t unit_index = 0; unit_index < block_weights.residual_units.size();
             ++unit_index) {
            hidden = residual_unit(ctx,
                                   hidden,
                                   block_weights.residual_units[unit_index],
                                   in_channels,
                                   kResidualDilations[unit_index]);
        }
        hidden = dac_snake(ctx, hidden, block_weights.snake, in_channels);
        hidden = modules::Conv1dModule({in_channels,
                                        out_channels,
                                        2 * ratio,
                                        static_cast<int>(ratio),
                                        static_cast<int>((ratio + 1) / 2),
                                        1,
                                        true})
                     .build(ctx, hidden, block_weights.conv);
    }
    hidden = dac_snake(ctx, hidden, weights.acoustic_encoder_output_snake, kEncoderChannels[5]);
    hidden = modules::Conv1dModule({kEncoderChannels[5], kAcousticHiddenSize, 3, 1, 1, 1, true})
                 .build(ctx, hidden, weights.acoustic_encoder_output);
    if (hidden.shape.dims[2] != target_frames) {
        hidden = modules::SliceModule({2, 0, target_frames}).build(ctx, hidden);
    }
    return hidden;
}

std::array<core::TensorValue, static_cast<size_t>(kCodecCodebooks)>
quantizer_encode(core::ModuleBuildContext & ctx,
                 const core::TensorValue & embeddings_bct,
                 const HiggsCodecWeights & weights) {
    auto residual = embeddings_bct;
    std::array<core::TensorValue, static_cast<size_t>(kCodecCodebooks)> all_codes{};
    for (int64_t codebook = 0; codebook < kCodecCodebooks; ++codebook) {
        auto hidden = transpose_bct_btc(ctx, residual);
        hidden =
            modules::LinearModule({kCodecHiddenSize, kCodecCodebookDim, true, GGML_PREC_F32})
                .build(ctx, hidden, weights.quantizers[static_cast<size_t>(codebook)].project_in);
        auto flat = core::reshape_tensor(
            ctx,
            contiguous(ctx, hidden),
            core::TensorShape::from_dims({hidden.shape.dims[1], kCodecCodebookDim}));
        auto codebook_weight = weights.quantizers[static_cast<size_t>(codebook)].codebook;
        auto codebook_t = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx, codebook_weight);
        auto dot = modules::MatMulModule().build(ctx, flat, codebook_t);
        auto x2 =
            modules::ReduceSumModule({1}).build(ctx, modules::MulModule().build(ctx, flat, flat));
        x2 = core::wrap_tensor(
            ggml_repeat(ctx.ggml, x2.tensor, dot.tensor), dot.shape, GGML_TYPE_F32);
        const auto codebook_weight_f32 =
            codebook_weight.type == GGML_TYPE_F32
                ? codebook_weight
                : core::wrap_tensor(ggml_cast(ctx.ggml, codebook_weight.tensor, GGML_TYPE_F32),
                                    codebook_weight.shape,
                                    GGML_TYPE_F32);
        auto e2 = modules::ReduceSumModule({1}).build(
            ctx, modules::MulModule().build(ctx, codebook_weight_f32, codebook_weight_f32));
        e2 = core::reshape_tensor(ctx, e2, core::TensorShape::from_dims({1, kCodecCodebookSize}));
        e2 = core::wrap_tensor(
            ggml_repeat(ctx.ggml, e2.tensor, dot.tensor), dot.shape, GGML_TYPE_F32);
        auto logits =
            core::wrap_tensor(ggml_scale(ctx.ggml, dot.tensor, 2.0F), dot.shape, GGML_TYPE_F32);
        logits = core::wrap_tensor(
            ggml_sub(ctx.ggml, logits.tensor, x2.tensor), logits.shape, GGML_TYPE_F32);
        logits = core::wrap_tensor(
            ggml_sub(ctx.ggml, logits.tensor, e2.tensor), logits.shape, GGML_TYPE_F32);
        auto ids = core::wrap_tensor(ggml_argmax(ctx.ggml, contiguous(ctx, logits).tensor),
                                     core::TensorShape::from_dims({embeddings_bct.shape.dims[2]}),
                                     GGML_TYPE_I32);
        all_codes[static_cast<size_t>(codebook)] = ids;
        auto quantized = modules::EmbeddingModule({kCodecCodebookSize, kCodecCodebookDim})
                             .build(ctx, ids, codebook_weight);
        quantized =
            modules::LinearModule({kCodecCodebookDim, kCodecHiddenSize, true, GGML_PREC_F32})
                .build(
                    ctx, quantized, weights.quantizers[static_cast<size_t>(codebook)].project_out);
        quantized = core::reshape_tensor(
            ctx,
            contiguous(ctx, quantized),
            core::TensorShape::from_dims({1, embeddings_bct.shape.dims[2], kCodecHiddenSize}));
        quantized = transpose_bct_btc(ctx, quantized);
        residual = core::wrap_tensor(
            ggml_sub(ctx.ggml, residual.tensor, quantized.tensor), residual.shape, GGML_TYPE_F32);
    }
    return all_codes;
}

HiggsCodecEncodeGraphValues codec_encode(core::ModuleBuildContext & ctx,
                                         const core::TensorValue & waveform_24k,
                                         const core::TensorValue & semantic_waveform_16k,
                                         const HiggsCodecWeights & weights,
                                         int64_t target_frames) {
    HiggsCodecEncodeGraphValues out;
    auto semantic =
        hubert_hidden_state_mean(ctx, semantic_waveform_16k, weights, target_frames);
    semantic = semantic_encoder(ctx, semantic, weights);
    if (semantic.shape.dims[2] != target_frames) {
        semantic = modules::SliceModule({2, 0, target_frames}).build(ctx, semantic);
    }
    auto acoustic = acoustic_encoder(ctx, waveform_24k, weights, target_frames);
    auto concat = modules::ConcatModule({1}).build(ctx, acoustic, semantic);
    auto hidden = transpose_bct_btc(ctx, concat);
    hidden = modules::LinearModule({kCodecProjectInputSize, kCodecHiddenSize, true, GGML_PREC_F32})
                 .build(ctx, hidden, weights.codec_project);
    hidden = transpose_bct_btc(ctx, hidden);
    out.codes = quantizer_encode(ctx, hidden, weights);
    return out;
}

core::TensorValue quantizer_decode(core::ModuleBuildContext & ctx,
                                   ggml_tensor * codes,
                                   const HiggsCodecWeights & weights,
                                   int64_t frames) {
    auto codes_value = core::wrap_tensor(
        codes, core::TensorShape::from_dims({frames, kCodecCodebooks}), GGML_TYPE_I32);
    std::vector<core::TensorValue> projected;
    projected.reserve(weights.quantizers.size());
    for (int64_t codebook = 0; codebook < kCodecCodebooks; ++codebook) {
        auto ids = modules::SliceModule({1, codebook, 1}).build(ctx, codes_value);
        ids = core::reshape_tensor(ctx,
                                   core::ensure_backend_addressable_layout(ctx, ids),
                                   core::TensorShape::from_dims({frames}));
        auto hidden =
            modules::EmbeddingModule({kCodecCodebookSize, kCodecCodebookDim})
                .build(ctx, ids, weights.quantizers[static_cast<size_t>(codebook)].codebook);
        hidden =
            modules::LinearModule({kCodecCodebookDim, kCodecHiddenSize, true})
                .build(ctx, hidden, weights.quantizers[static_cast<size_t>(codebook)].project_out);
        projected.push_back(hidden);
    }
    auto sum = projected.front();
    for (size_t index = 1; index < projected.size(); ++index) {
        sum = modules::AddModule{}.build(ctx, sum, projected[index]);
    }
    return sum;
}

core::TensorValue acoustic_decoder(core::ModuleBuildContext & ctx,
                                   const core::TensorValue & quantized,
                                   const HiggsCodecWeights & weights) {
    auto hidden = modules::LinearModule({kCodecHiddenSize, kAcousticHiddenSize, true})
                      .build(ctx, quantized, weights.acoustic_project);
    hidden = core::reshape_tensor(
        ctx, hidden, core::TensorShape::from_dims({1, hidden.shape.dims[0], kAcousticHiddenSize}));
    hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, hidden);
    hidden = modules::Conv1dModule({kAcousticHiddenSize, kDecoderChannels[0], 7, 1, 3, 1, true})
                 .build(ctx, hidden, weights.acoustic_decoder_input);
    for (int64_t block = 0; block < kDecoderBlockCount; ++block) {
        const int64_t in_channels = kDecoderChannels[static_cast<size_t>(block)];
        const int64_t out_channels = kDecoderChannels[static_cast<size_t>(block + 1)];
        const int64_t ratio = kUpsampleRatios[static_cast<size_t>(block)];
        const auto & block_weights = weights.acoustic_decoder_blocks[static_cast<size_t>(block)];
        hidden = dac_snake(ctx, hidden, block_weights.snake, in_channels);
        hidden = conv_transpose_with_adjusted_output_padding(
            ctx, hidden, block_weights.conv_transpose, in_channels, out_channels, ratio);
        for (size_t unit_index = 0; unit_index < block_weights.residual_units.size();
             ++unit_index) {
            hidden = residual_unit(ctx,
                                   hidden,
                                   block_weights.residual_units[unit_index],
                                   out_channels,
                                   kResidualDilations[unit_index]);
        }
    }
    hidden = dac_snake(ctx, hidden, weights.acoustic_decoder_output_snake, kDecoderChannels[5]);
    return modules::Conv1dModule({kDecoderChannels[5], 1, 7, 1, 3, 1, true})
        .build(ctx, hidden, weights.acoustic_decoder_output);
}

} // namespace

class HiggsCodecEncodeGraph {
public:
    HiggsCodecEncodeGraph(const HiggsCodecRuntime * runtime,
                          int64_t acoustic_samples,
                          int64_t semantic_samples,
                          int64_t frames)
        : runtime_(runtime), acoustic_samples_(acoustic_samples),
          semantic_samples_(semantic_samples), frames_(frames) {
        if (runtime_ == nullptr) {
            throw std::runtime_error("Higgs TTS codec encode graph requires runtime");
        }
        if (acoustic_samples_ <= 0 || semantic_samples_ <= 0 || frames_ <= 0) {
            throw std::runtime_error("Higgs TTS codec encode graph requires positive dimensions");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{runtime_->encode_graph_arena_bytes(), nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Higgs TTS codec encode graph context");
        }
        core::ModuleBuildContext build_ctx{
            ctx_.get(), "higgs_audio_tts.codec.encode", runtime_->backend_type()};
        acoustic_input_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_F32, acoustic_samples_);
        semantic_input_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_F32, semantic_samples_);
        ggml_set_input(acoustic_input_);
        ggml_set_input(semantic_input_);
        auto acoustic = core::wrap_tensor(
            acoustic_input_, core::TensorShape::from_dims({acoustic_samples_}), GGML_TYPE_F32);
        auto semantic = core::wrap_tensor(
            semantic_input_, core::TensorShape::from_dims({1, semantic_samples_}), GGML_TYPE_F32);
        auto encoded = codec_encode(
            build_ctx, acoustic, semantic, runtime_->weights(), frames_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        for (size_t codebook = 0; codebook < outputs_.size(); ++codebook) {
            outputs_[codebook] = encoded.codes[codebook].tensor;
            ggml_set_output(outputs_[codebook]);
            ggml_build_forward_expand(graph_, outputs_[codebook]);
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
                gallocr_ = nullptr;
            }
            throw std::runtime_error("failed to allocate Higgs TTS codec encode graph");
        }
        engine::debug::timing_log_scalar("higgs_audio_tts.codec.encode.graph.build_ms",
                                         engine::debug::elapsed_ms(build_start, Clock::now()));
    }

    ~HiggsCodecEncodeGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(const HiggsCodecRuntime & runtime,
                 int64_t acoustic_samples,
                 int64_t semantic_samples,
                 int64_t frames) const {
        return runtime_ == &runtime && acoustic_samples_ == acoustic_samples &&
               semantic_samples_ == semantic_samples && frames_ == frames;
    }

    HiggsCodecEncodeOutput
    run(const std::vector<float> & acoustic, const std::vector<float> & semantic, int64_t frames) {
        if (static_cast<int64_t>(acoustic.size()) != acoustic_samples_ ||
            static_cast<int64_t>(semantic.size()) != semantic_samples_ || frames != frames_) {
            throw std::runtime_error("Higgs TTS codec encode graph shape mismatch");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(
            acoustic_input_, acoustic.data(), 0, acoustic.size() * sizeof(float));
        ggml_backend_tensor_set(
            semantic_input_, semantic.data(), 0, semantic.size() * sizeof(float));
        engine::debug::timing_log_scalar("higgs_audio_tts.codec.encode_input_upload_ms",
                                         engine::debug::elapsed_ms(timing_start, Clock::now()));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        engine::debug::timing_log_scalar("higgs_audio_tts.codec.encode.graph.compute_ms",
                                         engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Higgs TTS codec encode graph compute failed");
        }
        timing_start = Clock::now();
        HiggsCodecEncodeOutput out;
        out.frames = frames;
        out.codebooks = kCodecCodebooks;
        out.codes.resize(static_cast<size_t>(frames * kCodecCodebooks));
        std::vector<int32_t> codebook_codes(static_cast<size_t>(frames_));
        for (int64_t codebook = 0; codebook < kCodecCodebooks; ++codebook) {
            ggml_backend_tensor_get(outputs_[static_cast<size_t>(codebook)],
                                    codebook_codes.data(),
                                    0,
                                    codebook_codes.size() * sizeof(int32_t));
            for (int64_t frame = 0; frame < frames_; ++frame) {
                out.codes[static_cast<size_t>(frame * kCodecCodebooks + codebook)] =
                    codebook_codes[static_cast<size_t>(frame)];
            }
        }
        engine::debug::timing_log_scalar("higgs_audio_tts.codec.encode_output_read_ms",
                                         engine::debug::elapsed_ms(timing_start, Clock::now()));
        return out;
    }

private:
    const HiggsCodecRuntime * runtime_ = nullptr;
    int64_t acoustic_samples_ = 0;
    int64_t semantic_samples_ = 0;
    int64_t frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * acoustic_input_ = nullptr;
    ggml_tensor * semantic_input_ = nullptr;
    std::array<ggml_tensor *, static_cast<size_t>(kCodecCodebooks)> outputs_ = {};
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

class HiggsCodecDecodeGraph {
public:
    HiggsCodecDecodeGraph(const HiggsCodecRuntime * runtime, int64_t frames)
        : runtime_(runtime), capacity_frames_(frames) {
        if (runtime_ == nullptr) {
            throw std::runtime_error("Higgs TTS codec decode graph requires runtime");
        }
        if (capacity_frames_ <= 0) {
            throw std::runtime_error("Higgs TTS codec decode graph requires positive frame count");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{runtime_->decode_graph_arena_bytes(), nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Higgs TTS codec decode graph context");
        }
        core::ModuleBuildContext build_ctx{
            ctx_.get(), "higgs_audio_tts.codec.decode", runtime_->backend_type()};
        codes_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_I32, kCodecCodebooks, capacity_frames_);
        frame_mask_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_F32, 1, capacity_frames_);
        ggml_set_input(codes_);
        ggml_set_input(frame_mask_);
        auto hidden = quantizer_decode(build_ctx, codes_, runtime_->weights(), capacity_frames_);
        const auto mask = core::wrap_tensor(
            frame_mask_, core::TensorShape::from_dims({capacity_frames_, 1}), GGML_TYPE_F32);
        hidden = modules::MulModule{}.build(
            build_ctx,
            hidden,
            core::wrap_tensor(ggml_repeat(build_ctx.ggml, mask.tensor, hidden.tensor),
                              hidden.shape,
                              GGML_TYPE_F32));
        auto audio = acoustic_decoder(build_ctx, hidden, runtime_->weights());
        output_ = audio.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(runtime_->backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            if (gallocr_ != nullptr) {
                ggml_gallocr_free(gallocr_);
                gallocr_ = nullptr;
            }
            throw std::runtime_error("failed to allocate Higgs TTS codec decode graph");
        }
        code_scratch_.assign(static_cast<size_t>(capacity_frames_ * kCodecCodebooks), 0);
        frame_mask_values_.assign(static_cast<size_t>(capacity_frames_), 0.0F);
        engine::debug::timing_log_scalar("higgs_audio_tts.codec.decode.graph.build_ms",
                                         engine::debug::elapsed_ms(build_start, Clock::now()));
    }

    ~HiggsCodecDecodeGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(const HiggsCodecRuntime & runtime, int64_t frames) const {
        return runtime_ == &runtime && frames <= capacity_frames_;
    }

    int64_t capacity_frames() const { return capacity_frames_; }

    HiggsCodecDecodeOutput run(const std::vector<int32_t> & codes, int64_t frames) {
        if (frames <= 0 || frames > capacity_frames_) {
            throw std::runtime_error("Higgs TTS codec decode frame count exceeds graph capacity");
        }
        if (static_cast<int64_t>(codes.size()) != frames * kCodecCodebooks) {
            throw std::runtime_error("Higgs TTS codec decode code matrix shape mismatch");
        }
        for (const int32_t code : codes) {
            if (code < 0 || code >= kCodecCodebookSize) {
                throw std::runtime_error("Higgs TTS codec decode code is outside codebook range");
            }
        }
        std::fill(code_scratch_.begin(), code_scratch_.end(), 0);
        for (int64_t frame = 0; frame < frames; ++frame) {
            const auto src = codes.begin() + static_cast<ptrdiff_t>(frame * kCodecCodebooks);
            const auto dst =
                code_scratch_.begin() + static_cast<ptrdiff_t>(frame * kCodecCodebooks);
            std::copy_n(src, static_cast<size_t>(kCodecCodebooks), dst);
        }
        std::fill(frame_mask_values_.begin(), frame_mask_values_.end(), 0.0F);
        std::fill(frame_mask_values_.begin(),
                  frame_mask_values_.begin() + static_cast<ptrdiff_t>(frames),
                  1.0F);
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(
            codes_, code_scratch_.data(), 0, code_scratch_.size() * sizeof(int32_t));
        ggml_backend_tensor_set(
            frame_mask_, frame_mask_values_.data(), 0, frame_mask_values_.size() * sizeof(float));
        engine::debug::timing_log_scalar("higgs_audio_tts.codec.decode_input_upload_ms",
                                         engine::debug::elapsed_ms(timing_start, Clock::now()));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        engine::debug::timing_log_scalar("higgs_audio_tts.codec.decode.graph.compute_ms",
                                         engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Higgs TTS codec decode graph compute failed");
        }
        HiggsCodecDecodeOutput out;
        out.sample_rate = kCodecSampleRate;
        out.channels = 1;
        out.samples = frames;
        for (int64_t block = 0; block < kDecoderBlockCount; ++block) {
            out.samples *= kUpsampleRatios[static_cast<size_t>(block)];
        }
        out.values.resize(static_cast<size_t>(out.samples));
        timing_start = Clock::now();
        ggml_backend_tensor_get(output_, out.values.data(), 0, out.values.size() * sizeof(float));
        engine::debug::timing_log_scalar("higgs_audio_tts.codec.decode_output_read_ms",
                                         engine::debug::elapsed_ms(timing_start, Clock::now()));
        return out;
    }

private:
    const HiggsCodecRuntime * runtime_ = nullptr;
    int64_t capacity_frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * codes_ = nullptr;
    ggml_tensor * frame_mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    std::vector<int32_t> code_scratch_;
    std::vector<float> frame_mask_values_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

HiggsCodecWeights load_higgs_codec_decode_weights(const HiggsAssets & assets,
                                                  ggml_backend_t backend,
                                                  core::BackendType backend_type,
                                                  size_t weight_context_bytes,
                                                  assets::TensorStorageType weight_storage_type) {
    if (assets.weights == nullptr) {
        throw std::runtime_error("Higgs TTS codec weights require tensor source");
    }
    if (backend == nullptr) {
        throw std::runtime_error("Higgs TTS codec backend is not initialized");
    }
    HiggsCodecWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend, backend_type, "higgs_audio_tts.codec.weights", weight_context_bytes);
    const auto & source = *assets.weights;
    load_hubert_semantic_model_weights(weights, source, weight_storage_type);
    weights.quantizers.reserve(kCodecCodebooks);
    for (int64_t index = 0; index < kCodecCodebooks; ++index) {
        weights.quantizers.push_back(
            load_quantizer(*weights.store, source, index, weight_storage_type));
    }
    weights.acoustic_encoder_input = load_conv1d(*weights.store,
                                                 source,
                                                 "acoustic_encoder.conv1",
                                                 weight_storage_type,
                                                 kEncoderChannels[0],
                                                 1,
                                                 7,
                                                 true);
    weights.acoustic_encoder_blocks.reserve(kEncoderBlockCount);
    for (int64_t block = 0; block < kEncoderBlockCount; ++block) {
        weights.acoustic_encoder_blocks.push_back(
            load_encoder_block(*weights.store, source, block, weight_storage_type));
    }
    weights.acoustic_encoder_output_snake =
        load_snake(*weights.store, source, "acoustic_encoder.snake1.alpha", kEncoderChannels[5]);
    weights.acoustic_encoder_output = load_conv1d(*weights.store,
                                                  source,
                                                  "acoustic_encoder.conv2",
                                                  weight_storage_type,
                                                  kAcousticHiddenSize,
                                                  kEncoderChannels[5],
                                                  3,
                                                  true);
    weights.semantic_encoder_input = load_conv1d(*weights.store,
                                                 source,
                                                 "encoder_semantic.conv",
                                                 weight_storage_type,
                                                 kSemanticHiddenSize,
                                                 kSemanticHiddenSize,
                                                 3,
                                                 false);
    weights.semantic_encoder_blocks.reserve(kSemanticBlockCount);
    for (int64_t block = 0; block < kSemanticBlockCount; ++block) {
        weights.semantic_encoder_blocks.push_back(
            load_semantic_encoder_block(*weights.store, source, block, weight_storage_type));
    }
    weights.codec_project = load_linear(*weights.store,
                                        source,
                                        "fc",
                                        weight_storage_type,
                                        kCodecHiddenSize,
                                        kCodecProjectInputSize,
                                        true);
    weights.acoustic_project = load_linear(*weights.store,
                                           source,
                                           "fc2",
                                           weight_storage_type,
                                           kAcousticHiddenSize,
                                           kCodecHiddenSize,
                                           true);
    weights.acoustic_decoder_input = load_conv1d(*weights.store,
                                                 source,
                                                 "acoustic_decoder.conv1",
                                                 weight_storage_type,
                                                 kDecoderChannels[0],
                                                 kAcousticHiddenSize,
                                                 7,
                                                 true);
    weights.acoustic_decoder_blocks.reserve(kDecoderBlockCount);
    for (int64_t block = 0; block < kDecoderBlockCount; ++block) {
        weights.acoustic_decoder_blocks.push_back(
            load_decoder_block(*weights.store, source, block, weight_storage_type));
    }
    weights.acoustic_decoder_output_snake =
        load_snake(*weights.store, source, "acoustic_decoder.snake1.alpha", kDecoderChannels[5]);
    weights.acoustic_decoder_output = load_conv1d(*weights.store,
                                                  source,
                                                  "acoustic_decoder.conv2",
                                                  weight_storage_type,
                                                  1,
                                                  kDecoderChannels[5],
                                                  7,
                                                  true);
    weights.store->upload();
    return weights;
}

HiggsCodecRuntime::HiggsCodecRuntime(std::shared_ptr<const HiggsAssets> assets,
                                     core::ExecutionContext & execution,
                                     size_t weight_context_bytes,
                                     size_t decode_graph_arena_bytes,
                                     size_t encode_graph_arena_bytes,
                                     assets::TensorStorageType weight_storage_type)
    : assets_(std::move(assets)), backend_(execution.backend()),
      backend_type_(execution.backend_type()), threads_(std::max(1, execution.config().threads)),
      decode_graph_arena_bytes_(decode_graph_arena_bytes),
      encode_graph_arena_bytes_(encode_graph_arena_bytes),
      weights_(nullptr) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Higgs TTS codec runtime requires assets");
    }
    if (assets_->weights == nullptr) {
        throw std::runtime_error("Higgs TTS codec runtime requires tensor source");
    }
    weights_ = std::make_shared<HiggsCodecWeights>(load_higgs_codec_decode_weights(
        *assets_, backend_, backend_type_, weight_context_bytes, weight_storage_type));
    if (decode_graph_arena_bytes_ == 0) {
        throw std::runtime_error("Higgs TTS codec decode graph arena bytes must be non-zero");
    }
    if (encode_graph_arena_bytes_ == 0) {
        throw std::runtime_error("Higgs TTS codec encode graph arena bytes must be non-zero");
    }
}

HiggsCodecRuntime::~HiggsCodecRuntime() = default;

const HiggsCodecWeights & HiggsCodecRuntime::weights() const noexcept { return *weights_; }

ggml_backend_t HiggsCodecRuntime::backend() const noexcept { return backend_; }

core::BackendType HiggsCodecRuntime::backend_type() const noexcept { return backend_type_; }

int HiggsCodecRuntime::threads() const noexcept { return threads_; }

size_t HiggsCodecRuntime::decode_graph_arena_bytes() const noexcept {
    return decode_graph_arena_bytes_;
}

size_t HiggsCodecRuntime::encode_graph_arena_bytes() const noexcept {
    return encode_graph_arena_bytes_;
}

HiggsCodecEncodeOutput
HiggsCodecRuntime::encode_reference(const runtime::AudioBuffer & audio) const {
    const auto mono = prepare_mono_audio(audio);
    const auto acoustic_24k_base = prepare_codec_audio_24k(mono, audio.sample_rate);
    auto semantic_16k = prepare_semantic_audio_16k(mono, audio.sample_rate);
    const int64_t frames = semantic_feature_frames(static_cast<int64_t>(semantic_16k.size()));

    std::vector<float> acoustic_24k = acoustic_24k_base;
    const int64_t acoustic_frames =
        acoustic_encoder_frames(static_cast<int64_t>(acoustic_24k.size()));
    if (acoustic_frames != frames) {
        acoustic_24k = pad_zeros(acoustic_24k_base, kCodecPadSamples, kCodecPadSamples);
        const int64_t padded_frames =
            acoustic_encoder_frames(static_cast<int64_t>(acoustic_24k.size()));
        if (padded_frames != frames) {
            throw std::runtime_error("Higgs TTS codec acoustic and semantic encoder "
                                     "frame counts do not match");
        }
    }

    if (encode_graph_ == nullptr ||
        !encode_graph_->matches(*this,
                                static_cast<int64_t>(acoustic_24k.size()),
                                static_cast<int64_t>(semantic_16k.size()),
                                frames)) {
        encode_graph_.reset();
        encode_graph_ =
            std::make_unique<HiggsCodecEncodeGraph>(this,
                                                    static_cast<int64_t>(acoustic_24k.size()),
                                                    static_cast<int64_t>(semantic_16k.size()),
                                                    frames);
    }
    engine::debug::trace_log_scalar("higgs_audio_tts.codec.encode.input_frames", frames);
    engine::debug::trace_log_f32("higgs_audio_tts.codec.encode.input_acoustic_24k",
                                 {static_cast<int64_t>(acoustic_24k.size())},
                                 acoustic_24k);
    engine::debug::trace_log_f32("higgs_audio_tts.codec.encode.input_semantic_16k",
                                 {static_cast<int64_t>(semantic_16k.size())},
                                 semantic_16k);
    return encode_graph_->run(acoustic_24k, semantic_16k, frames);
}

HiggsCodecDecodeOutput HiggsCodecRuntime::decode_codes(const std::vector<int32_t> & codes,
                                                       int64_t frames,
                                                       int64_t codebooks) const {
    if (frames <= 0) {
        throw std::runtime_error("Higgs TTS codec decode requires positive frame count");
    }
    if (codebooks != kCodecCodebooks) {
        throw std::runtime_error("Higgs TTS codec decode requires exactly 8 codebooks");
    }
    if (static_cast<int64_t>(codes.size()) != frames * codebooks) {
        throw std::runtime_error("Higgs TTS codec decode code count mismatch");
    }
    if (kCodecTailContextFrames <= 0) {
        return decode_codes_impl(codes, frames, codebooks);
    }

    std::vector<int32_t> padded_codes;
    padded_codes.reserve(static_cast<size_t>((frames + kCodecTailContextFrames) * codebooks));
    padded_codes.insert(padded_codes.end(), codes.begin(), codes.end());
    const auto last_frame_begin = codes.end() - static_cast<ptrdiff_t>(codebooks);
    for (int64_t i = 0; i < kCodecTailContextFrames; ++i) {
        padded_codes.insert(padded_codes.end(), last_frame_begin, codes.end());
    }

    auto out = decode_codes_impl(padded_codes, frames + kCodecTailContextFrames, codebooks);
    const int64_t keep_samples = frames * kCodecHopLength;
    if (keep_samples <= 0 || keep_samples > static_cast<int64_t>(out.values.size())) {
        throw std::runtime_error("Higgs TTS codec tail-context decode produced too few samples");
    }
    out.values.resize(static_cast<size_t>(keep_samples));
    out.samples = keep_samples;
    return out;
}

HiggsCodecDecodeOutput HiggsCodecRuntime::decode_codes_impl(const std::vector<int32_t> & codes,
                                                            int64_t frames,
                                                            int64_t codebooks) const {
    if (frames <= 0) {
        throw std::runtime_error("Higgs TTS codec decode requires positive frame count");
    }
    if (codebooks != kCodecCodebooks) {
        throw std::runtime_error("Higgs TTS codec decode requires exactly 8 codebooks");
    }
    if (static_cast<int64_t>(codes.size()) != frames * codebooks) {
        throw std::runtime_error("Higgs TTS codec decode code count mismatch");
    }
    engine::debug::trace_log_scalar("higgs_audio_tts.codec.decode.input_frames", frames);
    engine::debug::trace_log_scalar("higgs_audio_tts.codec.decode.input_codebooks", codebooks);
    engine::debug::trace_log_i32("higgs_audio_tts.codec.decode.input_codes",
                                 {frames, codebooks},
                                 codes);

    auto run_window = [&](const std::vector<int32_t> & window_codes,
                          int64_t window_frames,
                          int64_t min_capacity_frames) -> HiggsCodecDecodeOutput {
        const int64_t bucketed_frames =
            ((std::max(window_frames, min_capacity_frames) +
              kCodecDecodeCapacityBucketFrames - 1) /
             kCodecDecodeCapacityBucketFrames) *
            kCodecDecodeCapacityBucketFrames;
        encode_graph_.reset();
        if (decode_graph_ == nullptr || !decode_graph_->matches(*this, window_frames)) {
            decode_graph_.reset();
            decode_graph_ = std::make_unique<HiggsCodecDecodeGraph>(this, bucketed_frames);
        }
        return decode_graph_->run(window_codes, window_frames);
    };

    if (frames <= kCodecFullDecodeMaxFrames) {
        return run_window(codes, frames, frames);
    }

    HiggsCodecDecodeOutput out;
    out.sample_rate = kCodecSampleRate;
    out.channels = 1;
    out.samples = frames * kCodecHopLength;
    out.values.reserve(static_cast<size_t>(out.samples));

    std::vector<int32_t> window_codes;
    int64_t emitted_frames = 0;
    while (emitted_frames < frames) {
        const int64_t emit_begin = emitted_frames;
        const int64_t emit_end =
            std::min<int64_t>(frames, emit_begin + kCodecDecodeWindowFrames);
        const int64_t window_begin =
            std::max<int64_t>(0, emit_begin - kCodecDecodeContextFrames);
        const int64_t window_end =
            std::min<int64_t>(frames, emit_end + kCodecDecodeContextFrames);
        const int64_t window_frames = window_end - window_begin;
        window_codes.resize(static_cast<size_t>(window_frames * kCodecCodebooks));
        for (int64_t frame = 0; frame < window_frames; ++frame) {
            const auto src =
                codes.begin() +
                static_cast<ptrdiff_t>((window_begin + frame) * kCodecCodebooks);
            auto dst =
                window_codes.begin() + static_cast<ptrdiff_t>(frame * kCodecCodebooks);
            std::copy_n(src, static_cast<size_t>(kCodecCodebooks), dst);
        }

        const auto window = run_window(
            window_codes,
            window_frames,
            kCodecDecodeWindowFrames + 2 * kCodecDecodeContextFrames);
        const int64_t trim_frames = emit_begin - window_begin;
        const int64_t emit_frames = emit_end - emit_begin;
        const int64_t sample_begin = trim_frames * kCodecHopLength;
        const int64_t sample_count = emit_frames * kCodecHopLength;
        if (sample_begin < 0 || sample_count <= 0 ||
            sample_begin + sample_count > static_cast<int64_t>(window.values.size())) {
            throw std::runtime_error("Higgs TTS codec decode window produced invalid length");
        }
        out.values.insert(out.values.end(),
                          window.values.begin() + static_cast<ptrdiff_t>(sample_begin),
                          window.values.begin() +
                              static_cast<ptrdiff_t>(sample_begin + sample_count));
        emitted_frames = emit_end;
    }
    if (static_cast<int64_t>(out.values.size()) != out.samples) {
        throw std::runtime_error("Higgs TTS codec chunked decode output length mismatch");
    }
    return out;
}

void HiggsCodecRuntime::release_encode_graph() {
    encode_graph_.reset();
}

void HiggsCodecRuntime::release_runtime_graphs() {
    release_encode_graph();
    decode_graph_.reset();
}

} // namespace engine::models::higgs_audio_tts
