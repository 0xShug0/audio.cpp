#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/io/binary.h"
#include "engine/framework/io/json.h"
#include "engine/models/kokoro_tts/assets.h"

#include "engine/models/kokoro_tts/g2p_en.h"

#include <filesystem>
#include <cmath>
#include <cstring>
#include <string>
#include <stdexcept>
#include <vector>

namespace kokoro_ggml {

namespace {

struct KokoroConfigMetadata {
    int64_t n_token = 178;
    int64_t hidden_dim = 512;
    int64_t style_dim = 128;
    int64_t n_layer = 3;
    int64_t max_dur = 50;
    int64_t n_mels = 80;
    int64_t text_encoder_kernel_size = 5;
    int64_t plbert_hidden_size = 768;
    int64_t plbert_num_attention_heads = 12;
    int64_t plbert_intermediate_size = 2048;
    int64_t plbert_max_position_embeddings = 512;
    int64_t plbert_num_hidden_layers = 12;
    int64_t gen_istft_n_fft = 20;
    int64_t gen_istft_hop_size = 5;
    std::vector<int64_t> upsample_rates = {10, 6};
    std::vector<int64_t> upsample_kernel_sizes = {20, 12};
    std::vector<int64_t> resblock_kernel_sizes = {3, 7, 11};
    std::vector<std::vector<int64_t>> resblock_dilation_sizes = {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}};
    int64_t upsample_initial_channel = 512;
};

}  // namespace

std::vector<float> apply_weight_norm(
    const std::vector<float> & g,
    const std::vector<float> & v,
    int64_t leading,
    int64_t inner_size) {
    std::vector<float> weight(v.size(), 0.0f);
    for (int64_t i = 0; i < leading; ++i) {
        double norm = 0.0;
        for (int64_t j = 0; j < inner_size; ++j) {
            const float value = v[static_cast<size_t>(i * inner_size + j)];
            norm += static_cast<double>(value) * static_cast<double>(value);
        }
        const float scale = g[static_cast<size_t>(i)] / std::sqrt(static_cast<float>(norm) + 1.0e-12f);
        for (int64_t j = 0; j < inner_size; ++j) {
            weight[static_cast<size_t>(i * inner_size + j)] = v[static_cast<size_t>(i * inner_size + j)] * scale;
        }
    }
    return weight;
}

std::vector<float> combine_lstm_biases(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("Kokoro LSTM bias size mismatch");
    }
    std::vector<float> out(lhs.size(), 0.0f);
    for (size_t i = 0; i < lhs.size(); ++i) {
        out[i] = lhs[i] + rhs[i];
    }
    return out;
}

engine::assets::TensorStorageType vector_storage_type(engine::assets::TensorStorageType storage_type) {
    if (storage_type == engine::assets::TensorStorageType::Q4_0 ||
        storage_type == engine::assets::TensorStorageType::Q4_1 ||
        storage_type == engine::assets::TensorStorageType::Q5_0 ||
        storage_type == engine::assets::TensorStorageType::Q5_1 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return engine::assets::TensorStorageType::F32;
    }
    return storage_type;
}

KokoroWeights::LinearWeights load_linear(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_features,
    int64_t in_features,
    bool use_bias,
    engine::assets::TensorStorageType storage_type) {
    KokoroWeights::LinearWeights linear;
    linear.out_features = out_features;
    linear.in_features = in_features;
    linear.use_bias = use_bias;
    linear.weight = store.load_tensor(source, prefix + ".weight", storage_type, {out_features, in_features});
    if (use_bias) {
        linear.bias = store.load_tensor(source, prefix + ".bias", vector_storage_type(storage_type), {out_features});
    }
    return linear;
}

KokoroWeights::HostAffineWeights load_host_affine(
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_features,
    int64_t in_features,
    bool use_bias) {
    KokoroWeights::HostAffineWeights affine;
    affine.out_features = out_features;
    affine.in_features = in_features;
    affine.use_bias = use_bias;
    affine.weight = source.require_f32(prefix + ".weight", {out_features, in_features});
    if (use_bias) {
        affine.bias = source.require_f32(prefix + ".bias", {out_features});
    }
    return affine;
}

KokoroWeights::EmbeddingWeights load_embedding(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & name,
    int64_t num_embeddings,
    int64_t embedding_dim,
    engine::assets::TensorStorageType storage_type) {
    KokoroWeights::EmbeddingWeights embedding;
    embedding.num_embeddings = num_embeddings;
    embedding.embedding_dim = embedding_dim;
    embedding.weight = store.load_tensor(source, name, storage_type, {num_embeddings, embedding_dim});
    return embedding;
}

KokoroWeights::LayerNormWeights load_layer_norm(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    float eps) {
    KokoroWeights::LayerNormWeights norm;
    norm.channels = channels;
    norm.eps = eps;
    norm.weight = store.load_tensor(source, prefix + ".weight", engine::assets::TensorStorageType::Native, {channels});
    norm.bias = store.load_tensor(source, prefix + ".bias", engine::assets::TensorStorageType::Native, {channels});
    return norm;
}

KokoroWeights::WeightNormConv1dWeights load_weight_norm_conv1d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    bool use_bias,
    engine::assets::TensorStorageType storage_type) {
    const int64_t grouped_in_channels = in_channels / groups;
    auto g = source.require_f32( prefix + ".weight_g", {out_channels, 1, 1});
    auto v = source.require_f32( prefix + ".weight_v", {out_channels, grouped_in_channels, kernel});
    KokoroWeights::WeightNormConv1dWeights conv;
    conv.out_channels = out_channels;
    conv.in_channels = in_channels;
    conv.kernel = kernel;
    conv.stride = stride;
    conv.padding = padding;
    conv.dilation = dilation;
    conv.groups = groups;
    conv.use_bias = use_bias;
    conv.weight = store.make_from_f32(
        engine::core::TensorShape::from_dims({out_channels, grouped_in_channels, kernel}),
        storage_type,
        apply_weight_norm(g, v, out_channels, grouped_in_channels * kernel));
    if (use_bias) {
        conv.bias = store.load_tensor(source, prefix + ".bias", vector_storage_type(storage_type), {out_channels});
    }
    return conv;
}

KokoroWeights::Conv1dWeights load_conv1d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups,
    bool use_bias,
    engine::assets::TensorStorageType storage_type) {
    KokoroWeights::Conv1dWeights conv;
    conv.out_channels = out_channels;
    conv.in_channels = in_channels;
    conv.kernel = kernel;
    conv.stride = stride;
    conv.padding = padding;
    conv.dilation = dilation;
    conv.groups = groups;
    conv.use_bias = use_bias;
    conv.weight = store.load_tensor(source, prefix + ".weight", storage_type, {out_channels, in_channels / groups, kernel});
    if (use_bias) {
        conv.bias = store.load_tensor(source, prefix + ".bias", vector_storage_type(storage_type), {out_channels});
    }
    return conv;
}

KokoroWeights::WeightNormConvTranspose1dWeights load_weight_norm_conv_transpose1d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups,
    bool use_bias,
    engine::assets::TensorStorageType storage_type) {
    const int64_t grouped_out_channels = out_channels / groups;
    auto g = source.require_f32( prefix + ".weight_g", {in_channels, 1, 1});
    auto v = source.require_f32( prefix + ".weight_v", {in_channels, grouped_out_channels, kernel});
    KokoroWeights::WeightNormConvTranspose1dWeights conv;
    conv.in_channels = in_channels;
    conv.out_channels = out_channels;
    conv.kernel = kernel;
    conv.stride = stride;
    conv.padding = padding;
    conv.output_padding = output_padding;
    conv.groups = groups;
    conv.use_bias = use_bias;
    auto normalized = apply_weight_norm(g, v, in_channels, grouped_out_channels * kernel);
    conv.weight = store.make_from_f32(
        engine::core::TensorShape::from_dims({in_channels, grouped_out_channels, kernel}),
        storage_type,
        normalized);
    std::vector<float> bias_values;
    if (use_bias) {
        bias_values = source.require_f32(prefix + ".bias", {groups == in_channels ? in_channels : out_channels});
        conv.bias = store.load_tensor(
            source,
            prefix + ".bias",
            vector_storage_type(storage_type),
            {groups == in_channels ? in_channels : out_channels});
    }
    if (groups == in_channels && in_channels == out_channels && grouped_out_channels == 1) {
        std::vector<float> dense(static_cast<size_t>(in_channels * out_channels * kernel), 0.0f);
        for (int64_t channel = 0; channel < in_channels; ++channel) {
            const float * src = normalized.data() + static_cast<ptrdiff_t>(channel * kernel);
            float * dst = dense.data() + static_cast<ptrdiff_t>((channel * out_channels + channel) * kernel);
            std::memcpy(dst, src, static_cast<size_t>(kernel) * sizeof(float));
        }
        conv.dense_weight = store.make_from_f32(
            engine::core::TensorShape::from_dims({in_channels, out_channels, kernel}),
            storage_type,
            std::move(dense));
    }
    if (groups == 1 && output_padding == 0 && kernel == stride * 2 && padding > 0 && padding < stride) {
        auto phase_conv = std::make_shared<KokoroWeights::Conv1dWeights>();
        phase_conv->in_channels = in_channels;
        phase_conv->out_channels = out_channels * stride;
        phase_conv->kernel = 3;
        phase_conv->stride = 1;
        phase_conv->padding = 1;
        phase_conv->dilation = 1;
        phase_conv->groups = 1;
        phase_conv->use_bias = use_bias;
        std::vector<float> phase_weight(
            static_cast<size_t>(phase_conv->out_channels * phase_conv->in_channels * phase_conv->kernel),
            0.0f);
        std::vector<float> phase_bias;
        if (use_bias) {
            phase_bias.assign(static_cast<size_t>(phase_conv->out_channels), 0.0f);
        }
        auto source_weight = [&](int64_t in_channel, int64_t out_channel, int64_t kernel_index) -> float {
            const size_t offset = static_cast<size_t>((in_channel * out_channels + out_channel) * kernel + kernel_index);
            return normalized[offset];
        };
        auto target_weight = [&](int64_t phase_channel, int64_t in_channel, int64_t kernel_index) -> float & {
            const size_t offset = static_cast<size_t>(
                (phase_channel * phase_conv->in_channels + in_channel) * phase_conv->kernel + kernel_index);
            return phase_weight[offset];
        };
        const int64_t split_phase = stride - padding;
        for (int64_t out_channel = 0; out_channel < out_channels; ++out_channel) {
            for (int64_t phase = 0; phase < stride; ++phase) {
                const int64_t phase_channel = out_channel * stride + phase;
                if (use_bias) {
                    phase_bias[static_cast<size_t>(phase_channel)] = bias_values[static_cast<size_t>(out_channel)];
                }
                for (int64_t in_channel = 0; in_channel < in_channels; ++in_channel) {
                    const int64_t center_kernel = padding + phase;
                    target_weight(phase_channel, in_channel, 1) =
                        source_weight(in_channel, out_channel, center_kernel);
                    if (phase < split_phase) {
                        const int64_t previous_kernel = stride + padding + phase;
                        target_weight(phase_channel, in_channel, 0) =
                            source_weight(in_channel, out_channel, previous_kernel);
                    } else {
                        const int64_t next_kernel = phase + padding - stride;
                        target_weight(phase_channel, in_channel, 2) =
                            source_weight(in_channel, out_channel, next_kernel);
                    }
                }
            }
        }
        phase_conv->weight = store.make_from_f32(
            engine::core::TensorShape::from_dims({phase_conv->out_channels, phase_conv->in_channels, phase_conv->kernel}),
            storage_type,
            std::move(phase_weight));
        if (use_bias) {
            phase_conv->bias = store.make_from_f32(
                engine::core::TensorShape::from_dims({phase_conv->out_channels}),
                vector_storage_type(storage_type),
                std::move(phase_bias));
        }
        conv.phase_shuffle_conv = std::move(phase_conv);
    }
    return conv;
}

KokoroWeights::LstmWeights load_lstm(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t input_size,
    int64_t hidden_size,
    engine::assets::TensorStorageType storage_type) {
    KokoroWeights::LstmWeights lstm;
    lstm.input_size = input_size;
    lstm.hidden_size = hidden_size;
    lstm.weight_ih_l0 = store.load_tensor(source, prefix + ".weight_ih_l0", storage_type, {4 * hidden_size, input_size});
    lstm.weight_hh_l0 = store.load_tensor(source, prefix + ".weight_hh_l0", storage_type, {4 * hidden_size, hidden_size});
    lstm.bias_ih_l0 = store.load_tensor(source, prefix + ".bias_ih_l0", vector_storage_type(storage_type), {4 * hidden_size});
    lstm.bias_hh_l0 = store.load_tensor(source, prefix + ".bias_hh_l0", vector_storage_type(storage_type), {4 * hidden_size});
    auto bias_ih = source.require_f32(prefix + ".bias_ih_l0", {4 * hidden_size});
    auto bias_hh = source.require_f32(prefix + ".bias_hh_l0", {4 * hidden_size});
    lstm.combined_bias_l0 = store.make_from_f32(
        engine::core::TensorShape::from_dims({4 * hidden_size}),
        vector_storage_type(storage_type),
        combine_lstm_biases(bias_ih, bias_hh));
    lstm.weight_ih_l0_reverse =
        store.load_tensor(source, prefix + ".weight_ih_l0_reverse", storage_type, {4 * hidden_size, input_size});
    lstm.weight_hh_l0_reverse =
        store.load_tensor(source, prefix + ".weight_hh_l0_reverse", storage_type, {4 * hidden_size, hidden_size});
    lstm.bias_ih_l0_reverse =
        store.load_tensor(source, prefix + ".bias_ih_l0_reverse", vector_storage_type(storage_type), {4 * hidden_size});
    lstm.bias_hh_l0_reverse =
        store.load_tensor(source, prefix + ".bias_hh_l0_reverse", vector_storage_type(storage_type), {4 * hidden_size});
    auto bias_ih_reverse = source.require_f32(prefix + ".bias_ih_l0_reverse", {4 * hidden_size});
    auto bias_hh_reverse = source.require_f32(prefix + ".bias_hh_l0_reverse", {4 * hidden_size});
    lstm.combined_bias_l0_reverse = store.make_from_f32(
        engine::core::TensorShape::from_dims({4 * hidden_size}),
        vector_storage_type(storage_type),
        combine_lstm_biases(bias_ih_reverse, bias_hh_reverse));
    return lstm;
}

namespace {

KokoroWeights::LayerNormWeights load_beta_gamma_layer_norm(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    float eps) {
    KokoroWeights::LayerNormWeights norm;
    norm.channels = channels;
    norm.eps = eps;
    norm.weight = store.load_tensor(source, prefix + ".gamma", engine::assets::TensorStorageType::Native, {channels});
    norm.bias = store.load_tensor(source, prefix + ".beta", engine::assets::TensorStorageType::Native, {channels});
    return norm;
}

KokoroWeights::AdaLayerNormWeights load_ada_layer_norm(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    engine::assets::TensorStorageType storage_type) {
    KokoroWeights::AdaLayerNormWeights norm;
    norm.channels = channels;
    norm.fc = load_linear(store, source, prefix + ".fc", channels * 2, 128, true, storage_type);
    return norm;
}

KokoroWeights::AdaIn1dWeights load_adain1d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    engine::assets::TensorStorageType storage_type) {
    KokoroWeights::AdaIn1dWeights norm;
    norm.channels = channels;
    norm.fc = load_linear(store, source, prefix + ".fc", channels * 2, 128, true, storage_type);
    return norm;
}

KokoroWeights::AlbertLayerWeights load_albert_layer(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden_size,
    int64_t intermediate_size,
    engine::assets::TensorStorageType storage_type) {
    KokoroWeights::AlbertLayerWeights layer;
    layer.attention.query = load_linear(store, source, prefix + ".attention.query", hidden_size, hidden_size, true, storage_type);
    layer.attention.key = load_linear(store, source, prefix + ".attention.key", hidden_size, hidden_size, true, storage_type);
    layer.attention.value = load_linear(store, source, prefix + ".attention.value", hidden_size, hidden_size, true, storage_type);
    layer.attention.dense = load_linear(store, source, prefix + ".attention.dense", hidden_size, hidden_size, true, storage_type);
    layer.attention.layer_norm = load_layer_norm(store, source, prefix + ".attention.LayerNorm", hidden_size, 1.0e-12f);
    layer.ffn = load_linear(store, source, prefix + ".ffn", intermediate_size, hidden_size, true, storage_type);
    layer.ffn_output = load_linear(store, source, prefix + ".ffn_output", hidden_size, intermediate_size, true, storage_type);
    layer.full_layer_layer_norm = load_layer_norm(store, source, prefix + ".full_layer_layer_norm", hidden_size, 1.0e-12f);
    return layer;
}

KokoroWeights::TextEncoderBlockWeights load_text_encoder_block(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    int64_t block_index,
    engine::assets::TensorStorageType conv_storage_type) {
    const std::string prefix = "text_encoder.cnn." + std::to_string(block_index);
    KokoroWeights::TextEncoderBlockWeights block;
    block.conv = load_weight_norm_conv1d(store, source, prefix + ".0", 512, 512, 5, 1, 2, 1, 1, true, conv_storage_type);
    block.layer_norm = load_beta_gamma_layer_norm(store, source, prefix + ".1", 512, 1.0e-5f);
    return block;
}

KokoroWeights::AdainResBlock1dWeights load_adain_resblock(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t dim_in,
    int64_t dim_out,
    bool upsample,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    KokoroWeights::AdainResBlock1dWeights block;
    block.learned_sc = dim_in != dim_out;
    block.use_pool = upsample;
    block.upsample = upsample;
    block.conv1 = load_weight_norm_conv1d(store, source, prefix + ".conv1", dim_out, dim_in, 3, 1, 1, 1, 1, true, conv_storage_type);
    block.conv2 = load_weight_norm_conv1d(store, source, prefix + ".conv2", dim_out, dim_out, 3, 1, 1, 1, 1, true, conv_storage_type);
    block.norm1 = load_adain1d(store, source, prefix + ".norm1", dim_in, matmul_storage_type);
    block.norm2 = load_adain1d(store, source, prefix + ".norm2", dim_out, matmul_storage_type);
    if (block.learned_sc) {
        block.conv1x1 =
            load_weight_norm_conv1d(store, source, prefix + ".conv1x1", dim_out, dim_in, 1, 1, 0, 1, 1, false, conv_storage_type);
    }
    if (block.use_pool) {
        block.pool =
            load_weight_norm_conv_transpose1d(store, source, prefix + ".pool", dim_in, dim_in, 3, 2, 1, 1, dim_in, true, conv_storage_type);
    }
    return block;
}

KokoroWeights::GeneratorResBlockWeights load_generator_resblock(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    int64_t kernel,
    const std::vector<int64_t> & dilations,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    KokoroWeights::GeneratorResBlockWeights block;
    for (size_t i = 0; i < dilations.size(); ++i) {
        block.convs1.push_back(load_weight_norm_conv1d(
            store,
            source,
            prefix + ".convs1." + std::to_string(i),
            channels,
            channels,
            kernel,
            1,
            static_cast<int64_t>((kernel * dilations[i] - dilations[i]) / 2),
            dilations[i],
            1,
            true,
            conv_storage_type));
        block.convs2.push_back(load_weight_norm_conv1d(
            store,
            source,
            prefix + ".convs2." + std::to_string(i),
            channels,
            channels,
            kernel,
            1,
            (kernel - 1) / 2,
            1,
            1,
            true,
            conv_storage_type));
        block.adain1.push_back(load_adain1d(store, source, prefix + ".adain1." + std::to_string(i), channels, matmul_storage_type));
        block.adain2.push_back(load_adain1d(store, source, prefix + ".adain2." + std::to_string(i), channels, matmul_storage_type));
        block.alpha1.push_back(
            store.load_tensor(source, prefix + ".alpha1." + std::to_string(i), engine::assets::TensorStorageType::Native, {1, channels, 1}));
        block.alpha2.push_back(
            store.load_tensor(source, prefix + ".alpha2." + std::to_string(i), engine::assets::TensorStorageType::Native, {1, channels, 1}));
    }
    return block;
}

}  // namespace

KokoroConfigMetadata parse_kokoro_config_metadata(const engine::io::json::Value & root) {
    if (!root.is_object()) {
        throw std::runtime_error("kokoro config root must be an object");
    }
    KokoroConfigMetadata config;
    const auto * n_token = root.find("n_token");
    const auto * hidden_dim = root.find("hidden_dim");
    const auto * style_dim = root.find("style_dim");
    const auto * n_layer = root.find("n_layer");
    const auto * max_dur = root.find("max_dur");
    const auto * n_mels = root.find("n_mels");
    const auto * text_encoder_kernel_size = root.find("text_encoder_kernel_size");
    if (n_token) config.n_token = n_token->as_i64();
    if (hidden_dim) config.hidden_dim = hidden_dim->as_i64();
    if (style_dim) config.style_dim = style_dim->as_i64();
    if (n_layer) config.n_layer = n_layer->as_i64();
    if (max_dur) config.max_dur = max_dur->as_i64();
    if (n_mels) config.n_mels = n_mels->as_i64();
    if (text_encoder_kernel_size) config.text_encoder_kernel_size = text_encoder_kernel_size->as_i64();

    if (const auto * plbert = root.find("plbert")) {
        const auto * hidden_size = plbert->find("hidden_size");
        const auto * num_attention_heads = plbert->find("num_attention_heads");
        const auto * intermediate_size = plbert->find("intermediate_size");
        const auto * max_position_embeddings = plbert->find("max_position_embeddings");
        const auto * num_hidden_layers = plbert->find("num_hidden_layers");
        if (hidden_size) config.plbert_hidden_size = hidden_size->as_i64();
        if (num_attention_heads) config.plbert_num_attention_heads = num_attention_heads->as_i64();
        if (intermediate_size) config.plbert_intermediate_size = intermediate_size->as_i64();
        if (max_position_embeddings) config.plbert_max_position_embeddings = max_position_embeddings->as_i64();
        if (num_hidden_layers) config.plbert_num_hidden_layers = num_hidden_layers->as_i64();
    }

    if (const auto * istftnet = root.find("istftnet")) {
        const auto * gen_istft_n_fft = istftnet->find("gen_istft_n_fft");
        const auto * gen_istft_hop_size = istftnet->find("gen_istft_hop_size");
        const auto * upsample_rates = istftnet->find("upsample_rates");
        const auto * upsample_kernel_sizes = istftnet->find("upsample_kernel_sizes");
        const auto * resblock_kernel_sizes = istftnet->find("resblock_kernel_sizes");
        const auto * upsample_initial_channel = istftnet->find("upsample_initial_channel");
        if (gen_istft_n_fft) config.gen_istft_n_fft = gen_istft_n_fft->as_i64();
        if (gen_istft_hop_size) config.gen_istft_hop_size = gen_istft_hop_size->as_i64();
        if (upsample_rates) {
            config.upsample_rates.clear();
            for (const auto & value : upsample_rates->as_array()) {
                config.upsample_rates.push_back(value.as_i64());
            }
        }
        if (upsample_kernel_sizes) {
            config.upsample_kernel_sizes.clear();
            for (const auto & value : upsample_kernel_sizes->as_array()) {
                config.upsample_kernel_sizes.push_back(value.as_i64());
            }
        }
        if (resblock_kernel_sizes) {
            config.resblock_kernel_sizes.clear();
            for (const auto & value : resblock_kernel_sizes->as_array()) {
                config.resblock_kernel_sizes.push_back(value.as_i64());
            }
        }
        if (upsample_initial_channel) config.upsample_initial_channel = upsample_initial_channel->as_i64();
    }

    return config;
}

std::shared_ptr<const KokoroWeights> load_kokoro_weights(
    const engine::io::json::Value & config_root,
    const engine::assets::TensorSource & source,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes) {
    const auto config = parse_kokoro_config_metadata(config_root);
    auto weights = std::make_shared<KokoroWeights>();
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        backend,
        backend_type,
        "kokoro.weights",
        weight_context_bytes);
    weights->n_token = config.n_token;
    weights->hidden_dim = config.hidden_dim;
    weights->style_dim = config.style_dim;
    weights->n_layer = config.n_layer;
    weights->max_dur = config.max_dur;
    weights->n_mels = config.n_mels;
    weights->context_length = config.plbert_max_position_embeddings;
    weights->max_output_tokens = weights->context_length * weights->max_dur;
    weights->bert.hidden_size = config.plbert_hidden_size;
    weights->bert.embedding_size = 128;
    weights->bert.intermediate_size = config.plbert_intermediate_size;
    weights->bert.max_position_embeddings = config.plbert_max_position_embeddings;
    weights->bert.num_hidden_layers = config.plbert_num_hidden_layers;
    weights->bert.num_attention_heads = config.plbert_num_attention_heads;
    weights->decoder.generator.gen_istft_n_fft = config.gen_istft_n_fft;
    weights->decoder.generator.gen_istft_hop_size = config.gen_istft_hop_size;

    auto & store = *weights->store;

    weights->bert.embeddings.word_embeddings = load_embedding(store, source, "bert.embeddings.word_embeddings.weight", weights->n_token, 128, matmul_storage_type);
    weights->bert.embeddings.position_embeddings = load_embedding(store, source, "bert.embeddings.position_embeddings.weight", weights->context_length, 128, matmul_storage_type);
    weights->bert.embeddings.token_type_embeddings = load_embedding(store, source, "bert.embeddings.token_type_embeddings.weight", 2, 128, matmul_storage_type);
    weights->bert.embeddings.layer_norm = load_layer_norm(store, source, "bert.embeddings.LayerNorm", 128, 1.0e-12f);
    weights->bert.embedding_hidden_mapping_in = load_linear(store, source, "bert.encoder.embedding_hidden_mapping_in", weights->bert.hidden_size, 128, true, matmul_storage_type);
    weights->bert.shared_layer = load_albert_layer(
        store,
        source,
        "bert.encoder.albert_layer_groups.0.albert_layers.0",
        weights->bert.hidden_size,
        weights->bert.intermediate_size,
        matmul_storage_type);
    weights->bert.pooler = load_linear(store, source, "bert.pooler", weights->bert.hidden_size, weights->bert.hidden_size, true, matmul_storage_type);

    weights->bert_encoder = load_linear(store, source, "bert_encoder", weights->hidden_dim, weights->bert.hidden_size, true, matmul_storage_type);

    for (int64_t i = 0; i < weights->n_layer; ++i) {
        weights->text_encoder.cnn.push_back(load_text_encoder_block(store, source, i, conv_storage_type));
    }
    weights->text_encoder.embedding = load_embedding(store, source, "text_encoder.embedding.weight", weights->n_token, weights->hidden_dim, matmul_storage_type);
    weights->text_encoder.lstm = load_lstm(store, source, "text_encoder.lstm", weights->hidden_dim, weights->hidden_dim / 2, matmul_storage_type);

    const std::string predictor_duration_encoder_prefix = "predictor.text_encoder";
    for (int64_t i = 0; i < weights->n_layer; ++i) {
        weights->predictor.duration_encoder.lstms.push_back(load_lstm(
            store,
            source,
            predictor_duration_encoder_prefix + ".lstms." + std::to_string(i * 2),
            weights->hidden_dim + weights->style_dim,
            weights->hidden_dim / 2,
            matmul_storage_type));
        weights->predictor.duration_encoder.ada_layer_norms.push_back(load_ada_layer_norm(store, source, predictor_duration_encoder_prefix + ".lstms." + std::to_string(i * 2 + 1), weights->hidden_dim, matmul_storage_type));
    }
    weights->predictor.lstm =
        load_lstm(store, source, "predictor.lstm", weights->hidden_dim + weights->style_dim, weights->hidden_dim / 2, matmul_storage_type);
    weights->predictor.duration_proj = load_linear(store, source, "predictor.duration_proj.linear_layer", weights->max_dur, weights->hidden_dim, true, matmul_storage_type);
    weights->predictor.shared =
        load_lstm(store, source, "predictor.shared", weights->hidden_dim + weights->style_dim, weights->hidden_dim / 2, matmul_storage_type);
    weights->predictor.f0_blocks.push_back(
        load_adain_resblock(store, source, "predictor.F0.0", weights->hidden_dim, weights->hidden_dim, false, matmul_storage_type, conv_storage_type));
    weights->predictor.f0_blocks.push_back(load_adain_resblock(
        store, source, "predictor.F0.1", weights->hidden_dim, weights->hidden_dim / 2, true, matmul_storage_type, conv_storage_type));
    weights->predictor.f0_blocks.push_back(load_adain_resblock(
        store, source, "predictor.F0.2", weights->hidden_dim / 2, weights->hidden_dim / 2, false, matmul_storage_type, conv_storage_type));
    weights->predictor.n_blocks.push_back(
        load_adain_resblock(store, source, "predictor.N.0", weights->hidden_dim, weights->hidden_dim, false, matmul_storage_type, conv_storage_type));
    weights->predictor.n_blocks.push_back(load_adain_resblock(
        store, source, "predictor.N.1", weights->hidden_dim, weights->hidden_dim / 2, true, matmul_storage_type, conv_storage_type));
    weights->predictor.n_blocks.push_back(load_adain_resblock(
        store, source, "predictor.N.2", weights->hidden_dim / 2, weights->hidden_dim / 2, false, matmul_storage_type, conv_storage_type));
    weights->predictor.f0_proj = load_conv1d(store, source, "predictor.F0_proj", 1, weights->hidden_dim / 2, 1, 1, 0, 1, 1, true, conv_storage_type);
    weights->predictor.n_proj = load_conv1d(store, source, "predictor.N_proj", 1, weights->hidden_dim / 2, 1, 1, 0, 1, 1, true, conv_storage_type);

    weights->decoder.encode = load_adain_resblock(store, source, "decoder.encode", 514, 1024, false, matmul_storage_type, conv_storage_type);
    weights->decoder.decode.push_back(
        load_adain_resblock(store, source, "decoder.decode.0", 1090, 1024, false, matmul_storage_type, conv_storage_type));
    weights->decoder.decode.push_back(
        load_adain_resblock(store, source, "decoder.decode.1", 1090, 1024, false, matmul_storage_type, conv_storage_type));
    weights->decoder.decode.push_back(
        load_adain_resblock(store, source, "decoder.decode.2", 1090, 1024, false, matmul_storage_type, conv_storage_type));
    weights->decoder.decode.push_back(
        load_adain_resblock(store, source, "decoder.decode.3", 1090, 512, true, matmul_storage_type, conv_storage_type));
    weights->decoder.f0_conv = load_weight_norm_conv1d(store, source, "decoder.F0_conv", 1, 1, 3, 2, 1, 1, 1, true, conv_storage_type);
    weights->decoder.n_conv = load_weight_norm_conv1d(store, source, "decoder.N_conv", 1, 1, 3, 2, 1, 1, 1, true, conv_storage_type);
    weights->decoder.asr_res = load_weight_norm_conv1d(store, source, "decoder.asr_res.0", 64, 512, 1, 1, 0, 1, 1, true, conv_storage_type);

    weights->decoder.generator.ups.push_back(
        load_weight_norm_conv_transpose1d(store, source, "decoder.generator.ups.0", 512, 256, 20, 10, 5, 0, 1, true, conv_storage_type));
    weights->decoder.generator.ups.push_back(
        load_weight_norm_conv_transpose1d(store, source, "decoder.generator.ups.1", 256, 128, 12, 6, 3, 0, 1, true, conv_storage_type));
    weights->decoder.generator.noise_convs.push_back(load_conv1d(store, source, "decoder.generator.noise_convs.0", 256, 22, 12, 6, 3, 1, 1, true, conv_storage_type));
    weights->decoder.generator.noise_convs.push_back(load_conv1d(store, source, "decoder.generator.noise_convs.1", 128, 22, 1, 1, 0, 1, 1, true, conv_storage_type));
    weights->decoder.generator.noise_res.push_back(
        load_generator_resblock(store, source, "decoder.generator.noise_res.0", 256, 7, {1, 3, 5}, matmul_storage_type, conv_storage_type));
    weights->decoder.generator.noise_res.push_back(
        load_generator_resblock(store, source, "decoder.generator.noise_res.1", 128, 11, {1, 3, 5}, matmul_storage_type, conv_storage_type));
    weights->decoder.generator.resblocks.push_back(
        load_generator_resblock(store, source, "decoder.generator.resblocks.0", 256, 3, {1, 3, 5}, matmul_storage_type, conv_storage_type));
    weights->decoder.generator.resblocks.push_back(
        load_generator_resblock(store, source, "decoder.generator.resblocks.1", 256, 7, {1, 3, 5}, matmul_storage_type, conv_storage_type));
    weights->decoder.generator.resblocks.push_back(
        load_generator_resblock(store, source, "decoder.generator.resblocks.2", 256, 11, {1, 3, 5}, matmul_storage_type, conv_storage_type));
    weights->decoder.generator.resblocks.push_back(
        load_generator_resblock(store, source, "decoder.generator.resblocks.3", 128, 3, {1, 3, 5}, matmul_storage_type, conv_storage_type));
    weights->decoder.generator.resblocks.push_back(
        load_generator_resblock(store, source, "decoder.generator.resblocks.4", 128, 7, {1, 3, 5}, matmul_storage_type, conv_storage_type));
    weights->decoder.generator.resblocks.push_back(
        load_generator_resblock(store, source, "decoder.generator.resblocks.5", 128, 11, {1, 3, 5}, matmul_storage_type, conv_storage_type));
    weights->decoder.generator.conv_post =
        load_weight_norm_conv1d(store, source, "decoder.generator.conv_post", 22, 128, 7, 1, 3, 1, 1, true, conv_storage_type);
    weights->decoder.generator.source_linear = load_host_affine(source, "decoder.generator.m_source.l_linear", 1, 9, true);

    weights->store->upload();

    return weights;
}

}  // namespace kokoro_ggml

namespace engine::models::kokoro_tts {

namespace {

struct KokoroAssetResources {
    assets::ResourceBundle bundle;
    io::json::Value config;
    io::json::Value voices;
    std::shared_ptr<const assets::TensorSource> weights;
};

KokoroAssetResources load_asset_resources(const std::filesystem::path & model_root) {
    KokoroAssetResources resources;
    resources.bundle = assets::ResourceBundle(std::filesystem::weakly_canonical(model_root));
    resources.bundle.add_model_files({
        {"config", "config.json"},
        {"voices", "voices.json"},
        {"weights", "kokoro-v1_0.safetensors"},
    });
    resources.config = resources.bundle.parse_json("config");
    if (!resources.config.is_object()) {
        throw std::runtime_error("Kokoro config root must be an object");
    }
    resources.voices = resources.bundle.parse_json("voices");
    if (!resources.voices.is_object()) {
        throw std::runtime_error("Kokoro voices.json root must be an object");
    }
    resources.weights = resources.bundle.open_tensor_source("weights");
    return resources;
}

std::string language_code_from_voice_id(const std::string & voice_id) {
    if (voice_id.size() < 2 || (voice_id[1] != 'f' && voice_id[1] != 'm')) {
        throw std::runtime_error("invalid Kokoro voice id: " + voice_id);
    }
    return std::string(1, voice_id[0]);
}

std::vector<float> read_f32_file_exact(const std::filesystem::path & path, size_t expected_values) {
    const auto blob = engine::io::read_binary_blob(path);
    const size_t expected_bytes = expected_values * sizeof(float);
    if (blob.size() != expected_bytes) {
        throw std::runtime_error(
            "unexpected Kokoro voice pack size for " + path.string() +
            ": expected " + std::to_string(expected_bytes) +
            " bytes, got " + std::to_string(blob.size()));
    }
    std::vector<float> values(expected_values, 0.0f);
    std::memcpy(values.data(), blob.data(), expected_bytes);
    return values;
}

}  // namespace

std::shared_ptr<const KokoroAssets> load_kokoro_assets(const std::filesystem::path & model_root) {
    auto resources = load_asset_resources(model_root);
    const auto & root = resources.bundle.model_root();

    auto assets = std::make_shared<KokoroAssets>();
    assets->model_root = root;
    assets->config = std::move(resources.config);
    assets->model_weights = std::move(resources.weights);
    assets->context_length = kokoro_ggml::parse_kokoro_config_metadata(assets->config).plbert_max_position_embeddings;
    assets->english_lexicon_dir = root / "misaki_en";
    assets->english_g2p_us = std::make_shared<const kokoro_ggml::g2p_en::EnglishG2P>(assets->english_lexicon_dir, false);
    assets->english_g2p_gb = std::make_shared<const kokoro_ggml::g2p_en::EnglishG2P>(assets->english_lexicon_dir, true);

    const auto * vocab = assets->config.find("vocab");
    if (vocab == nullptr || !vocab->is_object()) {
        throw std::runtime_error("Kokoro config.json is missing vocab object");
    }
    for (const auto & [symbol, value] : vocab->as_object()) {
        assets->vocab[symbol] = static_cast<int32_t>(value.as_i64());
    }

    for (const auto & [voice_id, value] : resources.voices.as_object()) {
        if (!value.is_object()) {
            throw std::runtime_error("Kokoro voice entry must be an object: " + voice_id);
        }
        KokoroVoicePack pack;
        pack.id = voice_id;
        pack.language_code = language_code_from_voice_id(voice_id);
        const auto & object = value.as_object();
        const auto rows_it = object.find("rows");
        const auto cols_it = object.find("cols");
        const auto path_it = object.find("path");
        if (rows_it == object.end() || cols_it == object.end() || path_it == object.end()) {
            throw std::runtime_error("Kokoro voice entry is missing rows/cols/path: " + voice_id);
        }
        pack.rows = rows_it->second.as_i64();
        pack.cols = cols_it->second.as_i64();
        pack.values = read_f32_file_exact(
            root / "voices" / path_it->second.as_string(),
            static_cast<size_t>(pack.rows * pack.cols));
        assets->voices.emplace(voice_id, std::move(pack));
    }

    return assets;
}

std::shared_ptr<const kokoro_ggml::KokoroWeights> load_kokoro_backend_weights(
    const KokoroAssets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes) {
    if (!assets.model_weights) {
        throw std::runtime_error("Kokoro assets are missing model weights");
    }
    auto weights = kokoro_ggml::load_kokoro_weights(
        assets.config,
        *assets.model_weights,
        backend,
        backend_type,
        matmul_storage_type,
        conv_storage_type,
        weight_context_bytes);
    assets.model_weights->release_storage();
    return weights;
}

}  // namespace engine::models::kokoro_tts
