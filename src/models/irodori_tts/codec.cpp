#include "engine/models/irodori_tts/codec.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::irodori_tts {
namespace {

constexpr size_t kCodecWeightContextBytes = 512ull * 1024ull * 1024ull;

std::vector<float> squeeze_channel_alpha(
    const assets::TensorSource & source,
    const std::string & name,
    int64_t channels) {
    const auto values = source.require_f32(name, {1, channels, 1});
    if (static_cast<int64_t>(values.size()) != channels) {
        throw std::runtime_error("Irodori codec Snake alpha shape mismatch: " + name);
    }
    return values;
}

std::vector<float> squeeze_weight_g(
    const std::vector<float> & values,
    int64_t channels,
    const std::string & prefix) {
    if (static_cast<int64_t>(values.size()) != channels) {
        throw std::runtime_error("Irodori codec weight_g shape mismatch: " + prefix);
    }
    return values;
}

std::vector<float> fold_weight_norm_dim0(
    const std::vector<float> & weight_v,
    const std::vector<float> & weight_g,
    const std::vector<int64_t> & shape,
    const std::string & prefix) {
    if (shape.empty()) {
        throw std::runtime_error("Irodori codec weight norm shape is empty: " + prefix);
    }
    int64_t inner = 1;
    for (size_t i = 1; i < shape.size(); ++i) {
        inner *= shape[i];
    }
    if (static_cast<int64_t>(weight_v.size()) != shape[0] * inner ||
        static_cast<int64_t>(weight_g.size()) != shape[0]) {
        throw std::runtime_error("Irodori codec weight norm tensor size mismatch: " + prefix);
    }
    std::vector<float> out(weight_v.size(), 0.0F);
    for (int64_t row = 0; row < shape[0]; ++row) {
        const size_t base = static_cast<size_t>(row * inner);
        double sum = 0.0;
        for (int64_t i = 0; i < inner; ++i) {
            const float value = weight_v[base + static_cast<size_t>(i)];
            sum += static_cast<double>(value) * static_cast<double>(value);
        }
        if (sum == 0.0) {
            throw std::runtime_error("Irodori codec weight norm row has zero norm: " + prefix);
        }
        const float scale = static_cast<float>(static_cast<double>(weight_g[static_cast<size_t>(row)]) / std::sqrt(sum));
        for (int64_t i = 0; i < inner; ++i) {
            out[base + static_cast<size_t>(i)] = weight_v[base + static_cast<size_t>(i)] * scale;
        }
    }
    return out;
}

modules::Conv1dWeights load_weight_norm_conv1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_size,
    bool use_bias = true) {
    modules::Conv1dWeights weights;
    const auto weight_v = source.require_f32(prefix + ".weight_v", {out_channels, in_channels, kernel_size});
    const auto weight_g = squeeze_weight_g(source.require_f32(prefix + ".weight_g", {out_channels, 1, 1}), out_channels, prefix);
    weights.weight = store.make_from_f32(
        core::TensorShape::from_dims({out_channels, in_channels, kernel_size}),
        storage_type,
        fold_weight_norm_dim0(weight_v, weight_g, {out_channels, in_channels, kernel_size}, prefix));
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    }
    return weights;
}

modules::ConvTranspose1dWeights load_weight_norm_conv_transpose1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_size,
    bool use_bias = true) {
    modules::ConvTranspose1dWeights weights;
    const auto weight_v = source.require_f32(prefix + ".weight_v", {in_channels, out_channels, kernel_size});
    const auto weight_g = squeeze_weight_g(source.require_f32(prefix + ".weight_g", {in_channels, 1, 1}), in_channels, prefix);
    weights.weight = store.make_from_f32(
        core::TensorShape::from_dims({in_channels, out_channels, kernel_size}),
        storage_type,
        fold_weight_norm_dim0(weight_v, weight_g, {in_channels, out_channels, kernel_size}, prefix));
    if (use_bias) {
        weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    }
    return weights;
}

modules::Snake1dWeights load_snake(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & name,
    int64_t channels) {
    return {store.make_f32(core::TensorShape::from_dims({channels}), squeeze_channel_alpha(source, name, channels))};
}

IrodoriCodecResidualUnitWeights load_residual_unit(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t channels,
    int64_t kernel,
    int64_t dilation) {
    IrodoriCodecResidualUnitWeights weights;
    weights.snake0 = load_snake(store, source, prefix + ".block.0.alpha", channels);
    weights.conv0 = load_weight_norm_conv1d(store, source, prefix + ".block.1", storage_type, channels, channels, kernel);
    weights.snake1 = load_snake(store, source, prefix + ".block.2.alpha", channels);
    weights.conv1 = load_weight_norm_conv1d(store, source, prefix + ".block.3", storage_type, channels, channels, 1);
    (void)dilation;
    return weights;
}

IrodoriCodecDecoderBlockWeights load_decoder_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t input_dim,
    int64_t output_dim,
    int64_t stride) {
    IrodoriCodecDecoderBlockWeights weights;
    weights.up_snake = load_snake(store, source, prefix + ".block.0.alpha", input_dim);
    weights.up_conv = load_weight_norm_conv_transpose1d(
        store,
        source,
        prefix + ".block.1",
        storage_type,
        input_dim,
        output_dim,
        2 * stride);
    weights.residual_0 = load_residual_unit(store, source, prefix + ".block.4", storage_type, output_dim, 7, 1);
    weights.residual_1 = load_residual_unit(store, source, prefix + ".block.5", storage_type, output_dim, 7, 3);
    weights.residual_2 = load_residual_unit(store, source, prefix + ".block.8", storage_type, output_dim, 7, 9);
    return weights;
}

IrodoriCodecEncoderBlockWeights load_encoder_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t input_dim,
    int64_t output_dim,
    int64_t stride) {
    IrodoriCodecEncoderBlockWeights weights;
    weights.residual_0 = load_residual_unit(store, source, prefix + ".block.0", storage_type, input_dim, 7, 1);
    weights.residual_1 = load_residual_unit(store, source, prefix + ".block.1", storage_type, input_dim, 7, 3);
    weights.residual_2 = load_residual_unit(store, source, prefix + ".block.2", storage_type, input_dim, 7, 9);
    weights.down_snake = load_snake(store, source, prefix + ".block.3.alpha", input_dim);
    weights.down_conv = load_weight_norm_conv1d(
        store,
        source,
        prefix + ".block.4",
        storage_type,
        output_dim,
        input_dim,
        2 * stride);
    return weights;
}

core::TensorValue conv1d_same(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::Conv1dWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int64_t dilation = 1) {
    const int64_t padding = (kernel - 1) * dilation / 2;
    return modules::Conv1dModule({
        in_channels,
        out_channels,
        kernel,
        1,
        static_cast<int>(padding),
        static_cast<int>(dilation),
        weights.bias.has_value(),
    }).build(ctx, input, weights);
}

core::TensorValue conv1d_norm_padding(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::Conv1dWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int64_t stride,
    int64_t dilation = 1) {
    const int64_t padding = (kernel - stride) * dilation / 2;
    return modules::Conv1dModule({
        in_channels,
        out_channels,
        kernel,
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(dilation),
        weights.bias.has_value(),
    }).build(ctx, input, weights);
}

core::TensorValue residual_unit(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const IrodoriCodecResidualUnitWeights & weights,
    int64_t channels,
    int64_t dilation) {
    auto hidden = modules::Snake1dModule({channels}).build(ctx, input, weights.snake0);
    hidden = conv1d_same(ctx, hidden, weights.conv0, channels, channels, 7, dilation);
    hidden = modules::Snake1dModule({channels}).build(ctx, hidden, weights.snake1);
    hidden = conv1d_same(ctx, hidden, weights.conv1, channels, channels, 1);
    return modules::AddModule{}.build(ctx, hidden, input);
}

core::TensorValue decoder_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const IrodoriCodecDecoderBlockWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t stride) {
    auto hidden = modules::Snake1dModule({in_channels}).build(ctx, input, weights.up_snake);
    const int64_t padding = (stride + 1) / 2;
    hidden = modules::ConvTranspose1dModule({
        in_channels,
        out_channels,
        2 * stride,
        static_cast<int>(stride),
        static_cast<int>(padding),
        1,
        weights.up_conv.bias.has_value(),
    }).build(ctx, hidden, weights.up_conv);
    hidden = residual_unit(ctx, hidden, weights.residual_0, out_channels, 1);
    hidden = residual_unit(ctx, hidden, weights.residual_1, out_channels, 3);
    hidden = residual_unit(ctx, hidden, weights.residual_2, out_channels, 9);
    return hidden;
}

core::TensorValue encoder_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const IrodoriCodecEncoderBlockWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t stride) {
    auto hidden = residual_unit(ctx, input, weights.residual_0, in_channels, 1);
    hidden = residual_unit(ctx, hidden, weights.residual_1, in_channels, 3);
    hidden = residual_unit(ctx, hidden, weights.residual_2, in_channels, 9);
    hidden = modules::Snake1dModule({in_channels}).build(ctx, hidden, weights.down_snake);
    return conv1d_norm_padding(ctx, hidden, weights.down_conv, in_channels, out_channels, 2 * stride, stride);
}

}  // namespace

IrodoriCodecWeights load_irodori_codec_weights(
    const IrodoriAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType conv_storage_type) {
    const auto & source = *assets.codec_weights;
    IrodoriCodecWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "irodori_tts.codec.weights",
        weight_context_bytes == 0 ? kCodecWeightContextBytes : weight_context_bytes);
    weights.encoder_input = load_weight_norm_conv1d(
        *weights.store,
        source,
        "encoder.block.0",
        conv_storage_type,
        assets.codec.encoder_dim,
        1,
        7);
    const int64_t encoder_rates[] = {2, 8, 10, 12};
    int64_t encoder_in_channels = assets.codec.encoder_dim;
    for (int64_t block = 0; block < 4; ++block) {
        const int64_t encoder_out_channels = encoder_in_channels * 2;
        weights.encoder_blocks.push_back(load_encoder_block(
            *weights.store,
            source,
            "encoder.block." + std::to_string(block + 1),
            conv_storage_type,
            encoder_in_channels,
            encoder_out_channels,
            encoder_rates[block]));
        encoder_in_channels = encoder_out_channels;
    }
    weights.encoder_output_snake = load_snake(*weights.store, source, "encoder.block.5.alpha", encoder_in_channels);
    weights.encoder_output = load_weight_norm_conv1d(
        *weights.store,
        source,
        "encoder.block.6",
        conv_storage_type,
        assets.codec.latent_dim,
        encoder_in_channels,
        3);
    weights.quantizer_in_proj = load_weight_norm_conv1d(
        *weights.store,
        source,
        "quantizer.in_proj",
        conv_storage_type,
        2 * assets.codec.codebook_dim,
        assets.codec.latent_dim,
        1);
    weights.quantizer_out_proj = load_weight_norm_conv1d(
        *weights.store,
        source,
        "quantizer.out_proj",
        conv_storage_type,
        assets.codec.latent_dim,
        assets.codec.codebook_dim,
        1);
    weights.decoder_input = load_weight_norm_conv1d(
        *weights.store,
        source,
        "decoder.model.0",
        conv_storage_type,
        assets.codec.decoder_dim,
        assets.codec.latent_dim,
        7);
    const int64_t rates[] = {12, 10, 8, 2};
    int64_t in_channels = assets.codec.decoder_dim;
    for (int64_t block = 0; block < 4; ++block) {
        const int64_t out_channels = in_channels / 2;
        weights.decoder_blocks.push_back(load_decoder_block(
            *weights.store,
            source,
            "decoder.model." + std::to_string(block + 1),
            conv_storage_type,
            in_channels,
            out_channels,
            rates[block]));
        in_channels = out_channels;
    }
    weights.watermark_passthrough.snake =
        load_snake(*weights.store, source, "decoder.wm_model.encoder_block.pre.0.alpha", in_channels);
    weights.watermark_passthrough.conv = load_weight_norm_conv1d(
        *weights.store,
        source,
        "decoder.wm_model.encoder_block.pre.1",
        conv_storage_type,
        1,
        in_channels,
        7);
    weights.store->upload();
    return weights;
}

core::TensorValue build_irodori_codec_decode(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & latent_btd,
    const IrodoriCodecWeights & weights,
    const IrodoriCodecConfig & config) {
    auto hidden = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, latent_btd);
    hidden = modules::Conv1dModule({config.codebook_dim, config.latent_dim, 1, 1, 0, 1, true})
                 .build(ctx, hidden, weights.quantizer_out_proj);
    hidden = conv1d_same(ctx, hidden, weights.decoder_input, config.latent_dim, config.decoder_dim, 7);
    const int64_t rates[] = {12, 10, 8, 2};
    int64_t in_channels = config.decoder_dim;
    for (size_t i = 0; i < weights.decoder_blocks.size(); ++i) {
        const int64_t out_channels = in_channels / 2;
        hidden = decoder_block(
            ctx,
            hidden,
            weights.decoder_blocks[i],
            in_channels,
            out_channels,
            rates[i]);
        in_channels = out_channels;
    }
    hidden = modules::Snake1dModule({in_channels}).build(ctx, hidden, weights.watermark_passthrough.snake);
    hidden = conv1d_same(ctx, hidden, weights.watermark_passthrough.conv, in_channels, 1, 7);
    return modules::TanhModule{}.build(ctx, hidden);
}

core::TensorValue build_irodori_codec_encode(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & waveform_bct,
    const IrodoriCodecWeights & weights,
    const IrodoriCodecConfig & config) {
    auto hidden = conv1d_same(ctx, waveform_bct, weights.encoder_input, 1, config.encoder_dim, 7);
    const int64_t rates[] = {2, 8, 10, 12};
    int64_t in_channels = config.encoder_dim;
    for (size_t i = 0; i < weights.encoder_blocks.size(); ++i) {
        const int64_t out_channels = in_channels * 2;
        hidden = encoder_block(ctx, hidden, weights.encoder_blocks[i], in_channels, out_channels, rates[i]);
        in_channels = out_channels;
    }
    hidden = modules::Snake1dModule({in_channels}).build(ctx, hidden, weights.encoder_output_snake);
    hidden = conv1d_same(ctx, hidden, weights.encoder_output, in_channels, config.latent_dim, 3);
    hidden = modules::Conv1dModule({config.latent_dim, 2 * config.codebook_dim, 1, 1, 0, 1, true})
                 .build(ctx, hidden, weights.quantizer_in_proj);
    hidden = modules::SliceModule({1, 0, config.codebook_dim}).build(ctx, hidden);
    return modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, hidden);
}

}  // namespace engine::models::irodori_tts
