#include "engine/models/meanvc2/speaker_encoder.h"

#include "engine/models/meanvc2/audio_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::meanvc2 {
namespace {

namespace ecapa = engine::modules::ecapa_tdnn;

constexpr int64_t kWavlmHiddenSize = 1024;
constexpr int64_t kWavlmLayers = 24;
constexpr int64_t kEcapaChannels = 512;
constexpr int64_t kEcapaScale = 8;
constexpr int64_t kEcapaWidth = kEcapaChannels / kEcapaScale;
constexpr int64_t kEcapaMfaChannels = 1536;
constexpr int64_t kEcapaEmbeddingDims = 256;
constexpr float kStatsEps = 1.0e-9F;

engine::modules::WavlmEncoderConfig meanvc2_wavlm_config(
    engine::assets::TensorStorageType weight_storage_type) {
    engine::modules::WavlmEncoderConfig config;
    config.hidden_size = kWavlmHiddenSize;
    config.intermediate_size = 4096;
    config.num_hidden_layers = kWavlmLayers;
    config.output_hidden_layer = kWavlmLayers;
    config.num_attention_heads = 16;
    config.num_buckets = 320;
    config.max_distance = 800;
    config.num_conv_pos_embeddings = 128;
    config.num_conv_pos_embedding_groups = 16;
    config.layer_norm_eps = 1.0e-5F;
    config.normalize_input = true;
    config.conv_feature_layer_norm = true;
    config.transformer_layer_norm_first = true;
    config.weight_storage_type = weight_storage_type;
    return config;
}

ecapa::Conv1dWeights meanvc2_conv1d_weights(
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t padding,
    int64_t dilation,
    bool use_bias = true) {
    ecapa::Conv1dWeights out;
    out.weight_name = prefix + ".weight";
    out.weight_source_shape = {out_channels, in_channels, kernel};
    out.out_channels = out_channels;
    out.in_channels = in_channels;
    out.kernel = kernel;
    out.padding = padding;
    out.dilation = dilation;
    out.use_bias = use_bias;
    out.padding_mode = ecapa::Conv1dPaddingMode::Zero;
    if (use_bias) {
        out.bias_name = prefix + ".bias";
    }
    return out;
}

ecapa::Conv1dWeights meanvc2_linear_as_conv1d_weights(
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    bool use_bias = true) {
    auto out = meanvc2_conv1d_weights(prefix, out_channels, in_channels, 1, 0, 1, use_bias);
    out.weight_source_shape = {out_channels, in_channels};
    return out;
}

ecapa::BatchNorm1dWeights load_meanvc2_batch_norm(
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    return {
        source.require_f32(prefix + ".weight", {channels}),
        source.require_f32(prefix + ".bias", {channels}),
        source.require_f32(prefix + ".running_mean", {channels}),
        source.require_f32(prefix + ".running_var", {channels}),
    };
}

ecapa::TDNNBlockWeights meanvc2_tdnn_weights(
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t padding,
    int64_t dilation) {
    return {
        meanvc2_conv1d_weights(prefix + ".conv", out_channels, in_channels, kernel, padding, dilation),
        load_meanvc2_batch_norm(source, prefix + ".bn", out_channels),
    };
}

ecapa::SERes2NetBlockWeights meanvc2_se_res2_block_weights(
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t dilation) {
    ecapa::SERes2NetBlockWeights block;
    block.tdnn1 = meanvc2_tdnn_weights(
        source,
        prefix + ".Conv1dReluBn1",
        kEcapaChannels,
        kEcapaChannels,
        1,
        0,
        1);
    block.res2net.scale = kEcapaScale;
    block.res2net.width = kEcapaWidth;
    block.res2net.first_chunk_passthrough = false;
    block.res2net.blocks.reserve(static_cast<size_t>(kEcapaScale - 1));
    for (int64_t index = 0; index < kEcapaScale - 1; ++index) {
        block.res2net.blocks.push_back({
            meanvc2_conv1d_weights(
                prefix + ".Res2Conv1dReluBn.convs." + std::to_string(index),
                kEcapaWidth,
                kEcapaWidth,
                3,
                dilation,
                dilation),
            load_meanvc2_batch_norm(
                source,
                prefix + ".Res2Conv1dReluBn.bns." + std::to_string(index),
                kEcapaWidth),
        });
    }
    block.tdnn2 = meanvc2_tdnn_weights(
        source,
        prefix + ".Conv1dReluBn2",
        kEcapaChannels,
        kEcapaChannels,
        1,
        0,
        1);
    block.se = {
        meanvc2_linear_as_conv1d_weights(prefix + ".SE_Connect.linear1", 128, kEcapaChannels),
        meanvc2_linear_as_conv1d_weights(prefix + ".SE_Connect.linear2", kEcapaChannels, 128),
    };
    return block;
}

std::shared_ptr<const ecapa::EcapaWeights> load_meanvc2_ecapa_weights(
    std::shared_ptr<const engine::assets::TensorSource> source) {
    if (source == nullptr) {
        throw std::runtime_error("MeanVC2 ECAPA source is missing");
    }
    auto weights = std::make_shared<ecapa::EcapaWeights>();
    weights->source = source;
    weights->feature_dim = kWavlmHiddenSize;
    weights->embedding_dim = kEcapaEmbeddingDims;
    weights->stats_eps = kStatsEps;
    weights->weight_context_bytes = 512ull * 1024ull * 1024ull;
    weights->graph_context_bytes = 512ull * 1024ull * 1024ull;
    weights->block0 = meanvc2_tdnn_weights(*source, "layer1", kEcapaChannels, kWavlmHiddenSize, 5, 2, 1);
    weights->se_blocks = {
        meanvc2_se_res2_block_weights(*source, "layer2", 2),
        meanvc2_se_res2_block_weights(*source, "layer3", 3),
        meanvc2_se_res2_block_weights(*source, "layer4", 4),
    };
    weights->mfa.conv = meanvc2_conv1d_weights("conv", kEcapaMfaChannels, kEcapaChannels * 3, 1, 0, 1);
    weights->mfa_use_batch_norm = false;
    weights->asp.kind = ecapa::AttentivePoolingKind::Simple;
    weights->asp.tdnn_use_batch_norm = false;
    weights->asp.tdnn.conv = meanvc2_conv1d_weights("pooling.linear1", 128, kEcapaMfaChannels, 1, 0, 1);
    weights->asp.conv = meanvc2_conv1d_weights("pooling.linear2", kEcapaMfaChannels, 128, 1, 0, 1);
    weights->asp_bn = load_meanvc2_batch_norm(*source, "bn", kEcapaMfaChannels * 2);
    weights->fc = meanvc2_linear_as_conv1d_weights("linear", kEcapaEmbeddingDims, kEcapaMfaChannels * 2);
    return weights;
}

std::vector<float> softmax_feature_weights(const std::vector<float> & raw) {
    if (raw.size() != static_cast<size_t>(kWavlmLayers + 1)) {
        throw std::runtime_error("MeanVC2 speaker feature_weight shape mismatch");
    }
    const float max_value = *std::max_element(raw.begin(), raw.end());
    std::vector<float> out(raw.size(), 0.0F);
    double sum = 0.0;
    for (size_t i = 0; i < raw.size(); ++i) {
        out[i] = std::exp(raw[i] - max_value);
        sum += out[i];
    }
    if (sum <= 0.0) {
        throw std::runtime_error("MeanVC2 speaker feature_weight softmax underflow");
    }
    const float inv_sum = static_cast<float>(1.0 / sum);
    for (float & value : out) {
        value *= inv_sum;
    }
    return out;
}

void instance_norm_channels_first(std::vector<float> & values, int64_t frames, int64_t dims) {
    if (frames <= 0 || dims <= 0 || static_cast<int64_t>(values.size()) != frames * dims) {
        throw std::runtime_error("MeanVC2 speaker features require a valid [frames, dims] tensor");
    }
    for (int64_t dim = 0; dim < dims; ++dim) {
        double mean = 0.0;
        for (int64_t frame = 0; frame < frames; ++frame) {
            mean += values[static_cast<size_t>(frame * dims + dim)];
        }
        mean /= static_cast<double>(frames);
        double variance = 0.0;
        for (int64_t frame = 0; frame < frames; ++frame) {
            const double centered = static_cast<double>(values[static_cast<size_t>(frame * dims + dim)]) - mean;
            variance += centered * centered;
        }
        variance /= static_cast<double>(frames);
        const float inv_std = 1.0F / std::sqrt(static_cast<float>(variance) + 1.0e-5F);
        for (int64_t frame = 0; frame < frames; ++frame) {
            float & value = values[static_cast<size_t>(frame * dims + dim)];
            value = (value - static_cast<float>(mean)) * inv_std;
        }
    }
}

}  // namespace

MeanVC2SpeakerEncoderRuntime::MeanVC2SpeakerEncoderRuntime(
    std::shared_ptr<const engine::assets::TensorSource> wavlm_source,
    std::shared_ptr<const engine::assets::TensorSource> ecapa_source,
    engine::core::ExecutionContext & execution_context,
    engine::assets::TensorStorageType weight_storage_type) {
    if (wavlm_source == nullptr || ecapa_source == nullptr) {
        throw std::runtime_error("MeanVC2 speaker encoder requires WavLM and ECAPA tensor sources");
    }
    wavlm_ = engine::modules::WavlmEncoderComponent::load_from_tensor_source(
        *wavlm_source,
        execution_context.config(),
        meanvc2_wavlm_config(weight_storage_type));
    feature_weights_ = softmax_feature_weights(ecapa_source->require_f32("feature_weight", {kWavlmLayers + 1}));
    ecapa_ = std::make_unique<engine::modules::ecapa_tdnn::EcapaRuntime>(
        load_meanvc2_ecapa_weights(ecapa_source),
        std::vector<std::string>{},
        execution_context,
        weight_storage_type);
    wavlm_source->release_storage();
}

MeanVC2SpeakerEncoderRuntime::~MeanVC2SpeakerEncoderRuntime() = default;

MeanVC2SpeakerFeatures MeanVC2SpeakerEncoderRuntime::extract_features(const runtime::AudioBuffer & audio) const {
    const auto prepared = prepare_meanvc2_audio_16k(audio);
    std::vector<int64_t> output_layers;
    output_layers.reserve(static_cast<size_t>(kWavlmLayers + 1));
    for (int64_t layer = 0; layer <= kWavlmLayers; ++layer) {
        output_layers.push_back(layer);
    }
    const auto layers = wavlm_.encode_layers(
        prepared.mono_16k,
        1,
        static_cast<int64_t>(prepared.mono_16k.size()),
        output_layers);
    if (layers.batch != 1 || layers.tokens <= 0 || layers.hidden_size != kWavlmHiddenSize ||
        layers.hidden_states.size() != static_cast<size_t>(kWavlmLayers + 1)) {
        throw std::runtime_error("MeanVC2 speaker WavLM output shape mismatch");
    }

    MeanVC2SpeakerFeatures out;
    out.frames = layers.tokens;
    out.dims = layers.hidden_size;
    out.values.assign(static_cast<size_t>(out.frames * out.dims), 0.0F);
    for (size_t layer = 0; layer < layers.hidden_states.size(); ++layer) {
        const auto & values = layers.hidden_states[layer];
        if (values.size() != out.values.size()) {
            throw std::runtime_error("MeanVC2 speaker WavLM layer size mismatch");
        }
        const float weight = feature_weights_[layer];
        for (size_t i = 0; i < out.values.size(); ++i) {
            out.values[i] += values[i] * weight;
        }
    }
    instance_norm_channels_first(out.values, out.frames, out.dims);
    return out;
}

std::vector<float> MeanVC2SpeakerEncoderRuntime::embed(const runtime::AudioBuffer & audio) const {
    if (ecapa_ == nullptr) {
        throw std::runtime_error("MeanVC2 speaker encoder is not initialized");
    }
    const auto features = extract_features(audio);
    if (features.dims != kWavlmHiddenSize) {
        throw std::runtime_error("MeanVC2 speaker feature dimension mismatch");
    }
    return ecapa_->embed_features(features.values, features.frames);
}

}  // namespace engine::models::meanvc2
