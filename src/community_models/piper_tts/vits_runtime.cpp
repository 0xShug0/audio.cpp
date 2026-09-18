#include "engine/community_models/piper_tts/vits_runtime.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/cache_slots.h"
#include "engine/framework/sampling/noise.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::piper_tts {
namespace {

namespace core = engine::core;
namespace modules = engine::modules;

constexpr size_t kIoArenaBytes = 1ULL * 1024ULL * 1024ULL;
constexpr size_t kDurationGraphArenaBytes = 16ULL * 1024ULL * 1024ULL;
constexpr size_t kDurationFlowGraphArenaBytes = 4ULL * 1024ULL * 1024ULL;
constexpr size_t kDecoderGraphArenaBytes = 16ULL * 1024ULL * 1024ULL;
constexpr size_t kWeightArenaBytes = 16ULL * 1024ULL * 1024ULL;
constexpr int64_t kMaxLatentFrames = 4000;
constexpr float kLeakyReluSlope = 0.1F;

struct GgmlContextDeleter {
    void operator()(ggml_context * context) const noexcept {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

core::TensorValue contiguous(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value) {
    if (core::has_backend_addressable_layout(value.tensor)) {
        return value;
    }
    return core::wrap_tensor(
        ggml_cont(ctx.ggml, value.tensor),
        value.shape,
        value.type);
}

ggml_tensor * contiguous_if_needed(
    ggml_context * context,
    ggml_tensor * tensor) {
    return core::has_backend_addressable_layout(tensor)
        ? tensor
        : ggml_cont(context, tensor);
}

core::TensorValue scaled(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & value,
    float scale) {
    const auto source = contiguous(ctx, value);
    return core::wrap_tensor(
        ggml_scale(ctx.ggml, source.tensor, scale),
        value.shape,
        GGML_TYPE_F32);
}

core::TensorValue add(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    return modules::AddModule().build(ctx, lhs, rhs);
}

core::TensorValue subtract(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    core::validate_shape(rhs, lhs.shape, "Piper TTS subtraction");
    return core::wrap_tensor(
        ggml_sub(ctx.ggml, lhs.tensor, rhs.tensor),
        lhs.shape,
        GGML_TYPE_F32);
}

core::TensorValue multiply(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    return modules::MulModule().build(ctx, lhs, rhs);
}

core::TensorValue matmul(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    const auto left = ggml_is_contiguous(lhs.tensor)
        ? lhs
        : core::wrap_tensor(ggml_cont(ctx.ggml, lhs.tensor), lhs.shape, lhs.type);
    const auto right = ggml_is_contiguous(rhs.tensor)
        ? rhs
        : core::wrap_tensor(ggml_cont(ctx.ggml, rhs.tensor), rhs.shape, rhs.type);
    return modules::MatMulModule().build(ctx, left, right);
}

struct PiperTtsBackendWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    std::unordered_map<std::string, core::TensorValue> tensors;
    std::array<float, 2> duration_affine_mean{};
    std::array<float, 2> duration_affine_scale{};
};

const core::TensorValue & weight(
    const PiperTtsBackendWeights & weights,
    const std::string & name) {
    const auto found = weights.tensors.find(name);
    if (found == weights.tensors.end()) {
        throw std::runtime_error("Piper TTS missing tensor: " + name);
    }
    return found->second;
}

std::shared_ptr<const PiperTtsBackendWeights> load_weights(
    const std::shared_ptr<const PiperTtsAssets> & assets,
    ggml_backend_t backend,
    core::BackendType backend_type) {
    auto out = std::make_shared<PiperTtsBackendWeights>();
    const auto affine_mean = assets->weights->require_f32("dp.flows.0.m", {2, 1});
    const auto affine_scale = assets->weights->require_f32(
        "/dp/flows.0/Exp_output_0", {2, 1});
    std::copy(affine_mean.begin(), affine_mean.end(), out->duration_affine_mean.begin());
    std::copy(affine_scale.begin(), affine_scale.end(), out->duration_affine_scale.begin());
    out->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "piper_tts.weights",
        kWeightArenaBytes);
    const auto metadata = assets->weights->tensors();
    out->tensors.reserve(metadata.size());
    for (const auto & tensor : metadata) {
        out->tensors.emplace(
            tensor.name,
            out->store->load_tensor(
                *assets->weights,
                tensor.name,
                assets::TensorStorageType::Native,
                tensor.shape));
    }
    out->store->upload();
    assets->weights->release_storage();
    return out;
}

modules::ConvTranspose1dWeights conv_transpose_weights(
    const PiperTtsBackendWeights & weights,
    const std::string & prefix) {
    return {
        weight(weights, prefix + ".weight"),
        weight(weights, prefix + ".bias"),
    };
}

core::TensorValue conv1d(
    core::ModuleBuildContext & ctx,
    const PiperTtsBackendWeights & weights,
    const core::TensorValue & input,
    const std::string & prefix,
    int64_t out_channels,
    int64_t kernel,
    int padding,
    int dilation = 1,
    bool use_bias = true) {
    const int64_t in_channels = input.shape.dims[1];
    const int64_t input_frames = input.shape.dims[2];
    const int64_t output_frames =
        input_frames + 2 * padding - dilation * (kernel - 1);
    const auto source = contiguous(ctx, input);
    auto * input_2d = ggml_reshape_2d(
        ctx.ggml,
        source.tensor,
        input_frames,
        in_channels);
    auto * kernel_tensor = weight(weights, prefix + ".weight").tensor;
    ggml_tensor * output = nullptr;
    if (kernel == 1 && padding == 0 && dilation == 1 &&
        ctx.backend_type != core::BackendType::Cpu) {
        auto * kernel_2d = ggml_reshape_2d(
            ctx.ggml,
            kernel_tensor,
            in_channels,
            out_channels);
        auto * input_channels_first = contiguous_if_needed(
            ctx.ggml,
            ggml_permute(ctx.ggml, input_2d, 1, 0, 2, 3));
        auto * output_channels_first =
            ggml_mul_mat(ctx.ggml, kernel_2d, input_channels_first);
        output = ggml_reshape_2d(
            ctx.ggml,
            contiguous_if_needed(
                ctx.ggml,
                ggml_permute(
                    ctx.ggml,
                    output_channels_first,
                    1,
                    0,
                    2,
                    3)),
            output_frames,
            out_channels);
    } else {
        auto * kernel_3d = ggml_reshape_3d(
            ctx.ggml,
            kernel_tensor,
            kernel,
            in_channels,
            out_channels);
        auto * input_3d = ggml_reshape_3d(
            ctx.ggml,
            input_2d,
            input_frames,
            in_channels,
            1);
        auto * output_3d = ggml_conv_1d(
            ctx.ggml,
            kernel_3d,
            input_3d,
            1,
            padding,
            dilation);
        output = ggml_reshape_2d(
            ctx.ggml,
            output_3d,
            output_frames,
            out_channels);
    }
    if (use_bias) {
        auto * bias = ggml_reshape_2d(
            ctx.ggml,
            weight(weights, prefix + ".bias").tensor,
            1,
            out_channels);
        output = ggml_add(ctx.ggml, output, bias);
    }
    return core::wrap_tensor(
        ggml_reshape_3d(
            ctx.ggml,
            output,
            output_frames,
            out_channels,
            1),
        core::TensorShape::from_dims({1, out_channels, output_frames}),
        GGML_TYPE_F32);
}

core::TensorValue channel_layer_norm(
    core::ModuleBuildContext & ctx,
    const PiperTtsBackendWeights & weights,
    const core::TensorValue & input,
    const std::string & prefix) {
    auto btc = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
    btc = modules::LayerNormModule({
        input.shape.dims[1],
        1.0e-5F,
        true,
        true,
    }).build(
        ctx,
        btc,
        {
            weight(weights, prefix + ".gamma"),
            weight(weights, prefix + ".beta"),
        });
    return modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, btc);
}

core::TensorValue depthwise_conv1d(
    core::ModuleBuildContext & ctx,
    const PiperTtsBackendWeights & weights,
    const core::TensorValue & input,
    const std::string & prefix,
    int padding,
    int dilation) {
    const auto source = contiguous(ctx, input);
    auto output = core::wrap_tensor(
        ggml_conv_1d_dw(
            ctx.ggml,
            contiguous(ctx, weight(weights, prefix + ".weight")).tensor,
            source.tensor,
            1,
            padding,
            dilation),
        input.shape,
        GGML_TYPE_F32);
    auto * bias = ggml_reshape_3d(
        ctx.ggml,
        weight(weights, prefix + ".bias").tensor,
        1,
        input.shape.dims[1],
        1);
    return core::wrap_tensor(
        ggml_cont(ctx.ggml, ggml_add(ctx.ggml, output.tensor, bias)),
        input.shape,
        GGML_TYPE_F32);
}

core::TensorValue dds_conv(
    core::ModuleBuildContext & ctx,
    const PiperTtsBackendWeights & weights,
    core::TensorValue value,
    const std::string & prefix) {
    for (int layer = 0; layer < 3; ++layer) {
        const int dilation = layer == 0 ? 1 : (layer == 1 ? 3 : 9);
        auto branch = depthwise_conv1d(
            ctx,
            weights,
            value,
            prefix + ".convs_sep." + std::to_string(layer),
            dilation,
            dilation);
        branch = channel_layer_norm(
            ctx,
            weights,
            branch,
            prefix + ".norms_1." + std::to_string(layer));
        branch = modules::GeluModule().build(ctx, branch);
        branch = conv1d(
            ctx,
            weights,
            branch,
            prefix + ".convs_1x1." + std::to_string(layer),
            value.shape.dims[1],
            1,
            0);
        branch = channel_layer_norm(
            ctx,
            weights,
            branch,
            prefix + ".norms_2." + std::to_string(layer));
        branch = modules::GeluModule().build(ctx, branch);
        value = add(ctx, value, branch);
    }
    return value;
}

core::TensorValue relative_embeddings(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & source,
    int64_t heads,
    int64_t length) {
    auto value = core::reshape_tensor(
        ctx,
        source,
        core::TensorShape::from_dims({1, 1, 9, source.shape.dims[2]}));
    if (length > 5) {
        const int64_t padding = length - 5;
        value = modules::Pad2dModule({0, 0, padding, padding}).build(ctx, value);
    } else {
        value = modules::SliceModule({
            2,
            5 - length,
            2 * length - 1,
        }).build(ctx, value);
    }
    return modules::RepeatModule({
        core::TensorShape::from_dims(
            {1, heads, 2 * length - 1, source.shape.dims[2]}),
    }).build(ctx, value);
}

core::TensorValue relative_to_absolute(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & relative,
    int64_t length) {
    core::TensorValue output;
    for (int64_t row = 0; row < length; ++row) {
        auto value = modules::SliceModule({2, row, 1}).build(ctx, relative);
        value = modules::SliceModule({
            3,
            length - 1 - row,
            length,
        }).build(ctx, value);
        output = output.valid()
            ? modules::ConcatModule({2}).build(ctx, output, value)
            : value;
    }
    return output;
}

core::TensorValue relative_value_output(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & probabilities,
    const core::TensorValue & relative_values,
    int64_t length) {
    core::TensorValue output;
    for (int64_t row = 0; row < length; ++row) {
        const auto probability_row =
            modules::SliceModule({2, row, 1}).build(ctx, probabilities);
        const auto value_window = modules::SliceModule({
            2,
            length - 1 - row,
            length,
        }).build(ctx, relative_values);
        const auto value = matmul(ctx, probability_row, value_window);
        output = output.valid()
            ? modules::ConcatModule({2}).build(ctx, output, value)
            : value;
    }
    return output;
}

core::TensorValue attention(
    core::ModuleBuildContext & ctx,
    const PiperTtsBackendWeights & weights,
    const PiperTtsConfig & config,
    const core::TensorValue & input,
    int layer) {
    const std::string prefix =
        "enc_p.encoder.attn_layers." + std::to_string(layer);
    const int64_t channels = config.hidden_channels;
    const int64_t heads = config.attention_heads;
    const int64_t key_channels = channels / heads;
    const int64_t length = input.shape.dims[2];

    auto q = conv1d(ctx, weights, input, prefix + ".conv_q", channels, 1, 0);
    auto k = conv1d(ctx, weights, input, prefix + ".conv_k", channels, 1, 0);
    auto v = conv1d(ctx, weights, input, prefix + ".conv_v", channels, 1, 0);
    q = core::reshape_tensor(
        ctx,
        contiguous(ctx, q),
        core::TensorShape::from_dims({1, heads, key_channels, length}));
    k = core::reshape_tensor(
        ctx,
        contiguous(ctx, k),
        core::TensorShape::from_dims({1, heads, key_channels, length}));
    v = core::reshape_tensor(
        ctx,
        contiguous(ctx, v),
        core::TensorShape::from_dims({1, heads, key_channels, length}));
    q = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, q);
    k = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k);
    v = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, v);
    q = scaled(ctx, q, 1.0F / std::sqrt(static_cast<float>(key_channels)));

    const auto k_transposed =
        modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, k);
    auto scores = matmul(ctx, q, k_transposed);
    const auto rel_k = relative_embeddings(
        ctx,
        weight(weights, prefix + ".emb_rel_k"),
        heads,
        length);
    const auto rel_k_transposed =
        modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, rel_k);
    const auto relative_logits = matmul(ctx, q, rel_k_transposed);
    scores = add(ctx, scores, relative_to_absolute(ctx, relative_logits, length));
    const auto probabilities = modules::SoftmaxModule().build(ctx, scores);

    auto output = matmul(ctx, probabilities, v);
    const auto rel_v = relative_embeddings(
        ctx,
        weight(weights, prefix + ".emb_rel_v"),
        heads,
        length);
    output = add(
        ctx,
        output,
        relative_value_output(ctx, probabilities, rel_v, length));
    output = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, output);
    output = core::reshape_tensor(
        ctx,
        contiguous(ctx, output),
        core::TensorShape::from_dims({1, channels, length}));
    return conv1d(
        ctx,
        weights,
        output,
        prefix + ".conv_o",
        channels,
        1,
        0);
}

core::TensorValue text_encoder(
    core::ModuleBuildContext & ctx,
    const PiperTtsBackendWeights & weights,
    const PiperTtsConfig & config,
    const core::TensorValue & tokens) {
    auto embedded = modules::EmbeddingModule({
        config.vocab_size,
        config.hidden_channels,
    }).build(
        ctx,
        tokens,
        weight(weights, "enc_p.emb.weight"));
    embedded = scaled(
        ctx,
        embedded,
        std::sqrt(static_cast<float>(config.hidden_channels)));
    auto hidden =
        modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, embedded);
    for (int layer = 0; layer < config.encoder_layers; ++layer) {
        hidden = channel_layer_norm(
            ctx,
            weights,
            add(ctx, hidden, attention(ctx, weights, config, hidden, layer)),
            "enc_p.encoder.norm_layers_1." + std::to_string(layer));
        auto ffn = conv1d(
            ctx,
            weights,
            hidden,
            "enc_p.encoder.ffn_layers." + std::to_string(layer) + ".conv_1",
            config.filter_channels,
            3,
            1);
        ffn = modules::ReluModule().build(ctx, ffn);
        ffn = conv1d(
            ctx,
            weights,
            ffn,
            "enc_p.encoder.ffn_layers." + std::to_string(layer) + ".conv_2",
            config.hidden_channels,
            3,
            1);
        hidden = channel_layer_norm(
            ctx,
            weights,
            add(ctx, hidden, ffn),
            "enc_p.encoder.norm_layers_2." + std::to_string(layer));
    }
    return hidden;
}

core::TensorValue duration_conditioning(
    core::ModuleBuildContext & ctx,
    const PiperTtsBackendWeights & weights,
    const PiperTtsConfig & config,
    const core::TensorValue & hidden) {
    auto value = conv1d(
        ctx,
        weights,
        hidden,
        "dp.pre",
        config.hidden_channels,
        1,
        0);
    value = dds_conv(ctx, weights, value, "dp.convs");
    return conv1d(
        ctx,
        weights,
        value,
        "dp.proj",
        config.hidden_channels,
        1,
        0);
}

core::TensorValue duration_flow_parameters(
    core::ModuleBuildContext & ctx,
    const PiperTtsBackendWeights & weights,
    const core::TensorValue & x0,
    const core::TensorValue & conditioning,
    int flow) {
    const std::string prefix = "dp.flows." + std::to_string(flow);
    auto value = conv1d(ctx, weights, x0, prefix + ".pre", 192, 1, 0);
    value = add(ctx, value, conditioning);
    value = dds_conv(ctx, weights, value, prefix + ".convs");
    return conv1d(ctx, weights, value, prefix + ".proj", 29, 1, 0);
}

core::TensorValue flip_channels(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & reverse_indices) {
    const int64_t channels = input.shape.dims[1];
    const int64_t frames = input.shape.dims[2];
    const auto matrix = core::reshape_tensor(
        ctx,
        contiguous(ctx, input),
        core::TensorShape::from_dims({channels, frames}));
    auto gathered = core::wrap_tensor(
        ggml_get_rows(ctx.ggml, matrix.tensor, reverse_indices.tensor),
        core::TensorShape::from_dims({channels, frames}),
        input.type);
    return core::reshape_tensor(ctx, contiguous(ctx, gathered), input.shape);
}

core::TensorValue wavenet(
    core::ModuleBuildContext & ctx,
    const PiperTtsBackendWeights & weights,
    const PiperTtsConfig & config,
    core::TensorValue hidden,
    const std::string & prefix) {
    core::TensorValue output = scaled(ctx, hidden, 0.0F);
    for (int layer = 0; layer < config.flow_layers; ++layer) {
        const auto in = conv1d(
            ctx,
            weights,
            hidden,
            prefix + ".in_layers." + std::to_string(layer),
            2 * config.hidden_channels,
            5,
            2,
            1);
        auto tanh_part = modules::SliceModule({
            1,
            0,
            config.hidden_channels,
        }).build(ctx, in);
        auto sigmoid_part = modules::SliceModule({
            1,
            config.hidden_channels,
            config.hidden_channels,
        }).build(ctx, in);
        tanh_part = modules::TanhModule().build(ctx, tanh_part);
        sigmoid_part = modules::SigmoidModule().build(ctx, sigmoid_part);
        const auto acts = multiply(ctx, tanh_part, sigmoid_part);
        const int64_t output_channels =
            layer + 1 < config.flow_layers
            ? 2 * config.hidden_channels
            : config.hidden_channels;
        const auto res_skip = conv1d(
            ctx,
            weights,
            acts,
            prefix + ".res_skip_layers." + std::to_string(layer),
            output_channels,
            1,
            0);
        if (layer + 1 < config.flow_layers) {
            hidden = add(
                ctx,
                hidden,
                modules::SliceModule({
                    1,
                    0,
                    config.hidden_channels,
                }).build(ctx, res_skip));
            output = add(
                ctx,
                output,
                modules::SliceModule({
                    1,
                    config.hidden_channels,
                    config.hidden_channels,
                }).build(ctx, res_skip));
        } else {
            output = add(ctx, output, res_skip);
        }
    }
    return output;
}

core::TensorValue reverse_flow(
    core::ModuleBuildContext & ctx,
    const PiperTtsBackendWeights & weights,
    const PiperTtsConfig & config,
    core::TensorValue input,
    const core::TensorValue & reverse_indices) {
    const int64_t half = config.inter_channels / 2;
    for (int flow = static_cast<int>(config.flow_count) - 1; flow >= 0; --flow) {
        input = flip_channels(ctx, input, reverse_indices);
        const std::string prefix =
            "flow.flows." + std::to_string(flow * 2);
        const auto x0 =
            modules::SliceModule({1, 0, half}).build(ctx, input);
        auto x1 =
            modules::SliceModule({1, half, half}).build(ctx, input);
        auto hidden = conv1d(
            ctx,
            weights,
            x0,
            prefix + ".pre",
            config.hidden_channels,
            1,
            0);
        hidden = wavenet(
            ctx,
            weights,
            config,
            hidden,
            prefix + ".enc");
        const auto mean = conv1d(
            ctx,
            weights,
            hidden,
            prefix + ".post",
            half,
            1,
            0);
        x1 = subtract(ctx, x1, mean);
        input = modules::ConcatModule({1}).build(ctx, x0, x1);
    }
    return input;
}

core::TensorValue leaky_relu(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    float slope) {
    const auto source = contiguous(ctx, input);
    return core::wrap_tensor(
        ggml_leaky_relu(ctx.ggml, source.tensor, slope, false),
        input.shape,
        GGML_TYPE_F32);
}

core::TensorValue resblock(
    core::ModuleBuildContext & ctx,
    const PiperTtsBackendWeights & weights,
    core::TensorValue input,
    const std::string & prefix,
    int64_t kernel,
    const std::vector<int64_t> & dilations) {
    for (size_t layer = 0; layer < dilations.size(); ++layer) {
        auto value = leaky_relu(ctx, input, kLeakyReluSlope);
        const int dilation = static_cast<int>(dilations[layer]);
        value = conv1d(
            ctx,
            weights,
            value,
            prefix + ".convs." + std::to_string(layer),
            input.shape.dims[1],
            kernel,
            static_cast<int>((kernel * dilation - dilation) / 2),
            dilation);
        input = add(ctx, input, value);
    }
    return input;
}

core::TensorValue conv_transpose1d(
    core::ModuleBuildContext & ctx,
    const PiperTtsBackendWeights & weights,
    const core::TensorValue & input,
    const std::string & prefix,
    int64_t out_channels,
    int64_t kernel,
    int stride,
    int64_t padding) {
    const int64_t input_frames = input.shape.dims[2];
    const int64_t output_frames = input_frames * stride;
    if (ctx.backend_type == core::BackendType::Cpu && padding > 0) {
        const auto source = contiguous(ctx, input);
        auto * input_2d = ggml_reshape_2d(
            ctx.ggml,
            source.tensor,
            input_frames,
            input.shape.dims[1]);
        auto * output = ggml_conv_transpose_1d(
            ctx.ggml,
            weight(weights, prefix + ".weight").tensor,
            input_2d,
            stride,
            0,
            1);
        const int64_t full_frames =
            (input_frames - 1) * stride + kernel;
        output = ggml_reshape_2d(
            ctx.ggml,
            output,
            full_frames,
            out_channels);
        output = ggml_view_2d(
            ctx.ggml,
            output,
            output_frames,
            out_channels,
            ggml_row_size(output->type, full_frames),
            ggml_row_size(output->type, padding));
        output = contiguous_if_needed(ctx.ggml, output);
        auto * bias = ggml_reshape_2d(
            ctx.ggml,
            weight(weights, prefix + ".bias").tensor,
            1,
            out_channels);
        output = ggml_add(ctx.ggml, output, bias);
        return core::wrap_tensor(
            ggml_reshape_3d(
                ctx.ggml,
                output,
                output_frames,
                out_channels,
                1),
            core::TensorShape::from_dims(
                {1, out_channels, output_frames}),
            GGML_TYPE_F32);
    }

    auto output = modules::ConvTranspose1dModule({
        input.shape.dims[1],
        out_channels,
        kernel,
        stride,
        0,
        1,
        true,
    }).build(
        ctx,
        input,
        conv_transpose_weights(weights, prefix));
    return modules::SliceModule({
        2,
        padding,
        output_frames,
    }).build(ctx, output);
}

core::TensorValue decoder(
    core::ModuleBuildContext & ctx,
    const PiperTtsBackendWeights & weights,
    const PiperTtsConfig & config,
    const core::TensorValue & latent) {
    auto hidden = conv1d(
        ctx,
        weights,
        latent,
        "dec.conv_pre",
        config.upsample_initial_channels,
        7,
        3);
    for (size_t stage = 0; stage < config.upsample_rates.size(); ++stage) {
        hidden = leaky_relu(ctx, hidden, kLeakyReluSlope);
        const int64_t out_channels =
            config.upsample_initial_channels /
            (int64_t{1} << static_cast<int64_t>(stage + 1));
        const int rate = static_cast<int>(config.upsample_rates[stage]);
        const int64_t kernel = config.upsample_kernel_sizes[stage];
        const int64_t padding = (kernel - rate) / 2;
        hidden = conv_transpose1d(
            ctx,
            weights,
            hidden,
            "dec.ups." + std::to_string(stage),
            out_channels,
            kernel,
            rate,
            padding);

        core::TensorValue sum;
        for (size_t block = 0; block < config.resblock_kernel_sizes.size(); ++block) {
            const size_t index =
                stage * config.resblock_kernel_sizes.size() + block;
            auto value = resblock(
                ctx,
                weights,
                hidden,
                "dec.resblocks." + std::to_string(index),
                config.resblock_kernel_sizes[block],
                config.resblock_dilations[block]);
            sum = sum.valid() ? add(ctx, sum, value) : value;
        }
        hidden = scaled(
            ctx,
            sum,
            1.0F / static_cast<float>(config.resblock_kernel_sizes.size()));
    }
    hidden = leaky_relu(ctx, hidden, 0.01F);
    hidden = conv1d(
        ctx,
        weights,
        hidden,
        "dec.conv_post",
        1,
        7,
        3,
        1,
        false);
    return modules::TanhModule().build(ctx, hidden);
}

struct GraphResources {
    ~GraphResources() {
        core::free_backend_graph_plan(backend, plan);
        core::release_backend_graph_resources(backend, graph);
        if (allocator != nullptr) {
            ggml_gallocr_free(allocator);
        }
        if (io_buffer != nullptr) {
            ggml_backend_buffer_free(io_buffer);
        }
    }

    std::unique_ptr<ggml_context, GgmlContextDeleter> io_context;
    std::unique_ptr<ggml_context, GgmlContextDeleter> graph_context;
    ggml_backend_buffer_t io_buffer = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_cgraph * graph = nullptr;
};

void allocate_graph(GraphResources & resources) {
    resources.io_buffer =
        ggml_backend_alloc_ctx_tensors(resources.io_context.get(), resources.backend);
    if (resources.io_buffer == nullptr) {
        throw std::runtime_error("Piper TTS failed to allocate graph input buffer");
    }
    resources.allocator =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(resources.backend));
    if (resources.allocator == nullptr ||
        !ggml_gallocr_reserve(resources.allocator, resources.graph) ||
        !ggml_gallocr_alloc_graph(resources.allocator, resources.graph)) {
        throw std::runtime_error("Piper TTS failed to allocate backend graph");
    }
    core::validate_backend_graph_supported(
        resources.backend,
        resources.graph,
        "Piper TTS");
    resources.plan =
        core::create_backend_graph_plan_if_host(resources.backend, resources.graph);
}

void compute_graph(GraphResources & resources, const char * label) {
    const auto status = core::compute_backend_graph(
        resources.backend,
        resources.graph,
        resources.plan,
        label);
    ggml_backend_synchronize(resources.backend);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(label) + " graph compute failed");
    }
}

struct DurationGraph : GraphResources {
    int64_t token_count = 0;
    ggml_tensor * tokens = nullptr;
    ggml_tensor * stats = nullptr;
    ggml_tensor * conditioning = nullptr;
};

std::unique_ptr<DurationGraph> build_duration_graph(
    const PiperTtsBackendWeights & weights,
    const PiperTtsConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    int64_t token_count) {
    auto out = std::make_unique<DurationGraph>();
    out->backend = backend;
    out->token_count = token_count;
    out->io_context.reset(ggml_init({kIoArenaBytes, nullptr, true}));
    out->graph_context.reset(ggml_init({kDurationGraphArenaBytes, nullptr, true}));
    if (out->io_context == nullptr || out->graph_context == nullptr) {
        throw std::runtime_error("Piper TTS failed to create duration graph contexts");
    }
    core::ModuleBuildContext io_ctx{
        out->io_context.get(),
        "piper_tts.duration.io",
        backend_type,
    };
    core::ModuleBuildContext ctx{
        out->graph_context.get(),
        "piper_tts.duration",
        backend_type,
    };
    auto tokens = core::make_tensor(
        io_ctx,
        GGML_TYPE_I32,
        core::TensorShape::from_dims({1, token_count}));
    ggml_set_input(tokens.tensor);
    const auto hidden = text_encoder(ctx, weights, config, tokens);
    auto stats = conv1d(
        ctx,
        weights,
        hidden,
        "enc_p.proj",
        2 * config.inter_channels,
        1,
        0);
    auto conditioning = duration_conditioning(ctx, weights, config, hidden);
    stats = contiguous(ctx, stats);
    conditioning = contiguous(ctx, conditioning);
    out->tokens = tokens.tensor;
    out->stats = stats.tensor;
    out->conditioning = conditioning.tensor;
    ggml_set_output(out->stats);
    ggml_set_output(out->conditioning);
    out->graph = ggml_new_graph_custom(ctx.ggml, 131072, false);
    ggml_build_forward_expand(out->graph, out->stats);
    ggml_build_forward_expand(out->graph, out->conditioning);
    allocate_graph(*out);
    return out;
}

struct DurationFlowGraph : GraphResources {
    int64_t token_count = 0;
    ggml_tensor * x0 = nullptr;
    ggml_tensor * conditioning = nullptr;
    ggml_tensor * parameters = nullptr;
};

std::unique_ptr<DurationFlowGraph> build_duration_flow_graph(
    const PiperTtsBackendWeights & weights,
    ggml_backend_t backend,
    core::BackendType backend_type,
    int64_t token_count,
    int flow) {
    auto out = std::make_unique<DurationFlowGraph>();
    out->backend = backend;
    out->token_count = token_count;
    out->io_context.reset(ggml_init({kIoArenaBytes, nullptr, true}));
    out->graph_context.reset(ggml_init({kDurationFlowGraphArenaBytes, nullptr, true}));
    if (out->io_context == nullptr || out->graph_context == nullptr) {
        throw std::runtime_error("Piper TTS failed to create duration-flow graph contexts");
    }
    core::ModuleBuildContext io_ctx{
        out->io_context.get(), "piper_tts.duration_flow.io", backend_type};
    core::ModuleBuildContext ctx{
        out->graph_context.get(), "piper_tts.duration_flow", backend_type};
    auto x0 = core::make_tensor(
        io_ctx, GGML_TYPE_F32,
        core::TensorShape::from_dims({1, 1, token_count}));
    auto conditioning = core::make_tensor(
        io_ctx, GGML_TYPE_F32,
        core::TensorShape::from_dims({1, 192, token_count}));
    ggml_set_input(x0.tensor);
    ggml_set_input(conditioning.tensor);
    auto parameters = contiguous(
        ctx,
        duration_flow_parameters(ctx, weights, x0, conditioning, flow));
    out->x0 = x0.tensor;
    out->conditioning = conditioning.tensor;
    out->parameters = parameters.tensor;
    ggml_set_output(out->parameters);
    out->graph = ggml_new_graph_custom(ctx.ggml, 32768, false);
    ggml_build_forward_expand(out->graph, out->parameters);
    allocate_graph(*out);
    return out;
}

struct DecoderGraph : GraphResources {
    int64_t latent_frames = 0;
    ggml_tensor * latent = nullptr;
    ggml_tensor * reverse_indices = nullptr;
    ggml_tensor * waveform = nullptr;
};

std::unique_ptr<DecoderGraph> build_decoder_graph(
    const PiperTtsBackendWeights & weights,
    const PiperTtsConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    int64_t latent_frames) {
    auto out = std::make_unique<DecoderGraph>();
    out->backend = backend;
    out->latent_frames = latent_frames;
    out->io_context.reset(ggml_init({kIoArenaBytes, nullptr, true}));
    out->graph_context.reset(ggml_init({kDecoderGraphArenaBytes, nullptr, true}));
    if (out->io_context == nullptr || out->graph_context == nullptr) {
        throw std::runtime_error("Piper TTS failed to create decoder graph contexts");
    }
    core::ModuleBuildContext io_ctx{
        out->io_context.get(),
        "piper_tts.decoder.io",
        backend_type,
    };
    core::ModuleBuildContext ctx{
        out->graph_context.get(),
        "piper_tts.decoder",
        backend_type,
    };
    auto latent = core::make_tensor(
        io_ctx,
        GGML_TYPE_F32,
        core::TensorShape::from_dims(
            {1, config.inter_channels, latent_frames}));
    auto reverse_indices = core::make_tensor(
        io_ctx,
        GGML_TYPE_I32,
        core::TensorShape::from_dims({config.inter_channels}));
    ggml_set_input(latent.tensor);
    ggml_set_input(reverse_indices.tensor);
    auto waveform = decoder(
        ctx,
        weights,
        config,
        reverse_flow(ctx, weights, config, latent, reverse_indices));
    waveform = contiguous(ctx, waveform);
    out->latent = latent.tensor;
    out->reverse_indices = reverse_indices.tensor;
    out->waveform = waveform.tensor;
    ggml_set_output(out->waveform);
    out->graph = ggml_new_graph_custom(ctx.ggml, 131072, false);
    ggml_build_forward_expand(out->graph, out->waveform);
    allocate_graph(*out);
    std::vector<int32_t> indices;
    indices.reserve(static_cast<size_t>(config.inter_channels));
    for (int64_t channel = config.inter_channels - 1; channel >= 0; --channel) {
        indices.push_back(static_cast<int32_t>(channel));
    }
    core::write_tensor_i32(reverse_indices, indices);
    return out;
}

struct ExpandedPrior {
    int64_t frames = 0;
    std::vector<float> latent;
};

float inverse_rational_quadratic(
    float input,
    const float * raw_widths,
    const float * raw_heights,
    const float * raw_derivatives) {
    constexpr int bins = 10;
    constexpr float bound = 5.0F;
    constexpr float min_bin = 1.0e-3F;
    constexpr float min_derivative = 1.0e-3F;
    if (input < -bound || input > bound) {
        return input;
    }
    auto normalized = [](const float * raw, float * values) {
        float maximum = raw[0];
        for (int i = 1; i < bins; ++i) maximum = std::max(maximum, raw[i]);
        float sum = 0.0F;
        for (int i = 0; i < bins; ++i) {
            values[i] = std::exp(raw[i] - maximum);
            sum += values[i];
        }
        for (int i = 0; i < bins; ++i) {
            values[i] = min_bin + (1.0F - min_bin * bins) * values[i] / sum;
        }
    };
    float widths[bins];
    float heights[bins];
    normalized(raw_widths, widths);
    normalized(raw_heights, heights);
    float derivatives[bins + 1];
    derivatives[0] = 1.0F;
    derivatives[bins] = 1.0F;
    for (int i = 1; i < bins; ++i) {
        derivatives[i] = min_derivative + std::log1p(std::exp(raw_derivatives[i - 1]));
    }
    float cumulative_width = -bound;
    float cumulative_height = -bound;
    int bin = bins - 1;
    for (int i = 0; i < bins; ++i) {
        const float next = cumulative_height + 2.0F * bound * heights[i];
        if (input < next + 1.0e-6F) {
            bin = i;
            break;
        }
        cumulative_width += 2.0F * bound * widths[i];
        cumulative_height = next;
    }
    const float width = 2.0F * bound * widths[bin];
    const float height = 2.0F * bound * heights[bin];
    const float delta = height / width;
    const float d0 = derivatives[bin];
    const float d1 = derivatives[bin + 1];
    const float shifted = input - cumulative_height;
    const float a = shifted * (d0 + d1 - 2.0F * delta) + height * (delta - d0);
    const float b = height * d0 - shifted * (d0 + d1 - 2.0F * delta);
    const float c = -delta * shifted;
    const float discriminant = std::max(0.0F, b * b - 4.0F * a * c);
    const float root = (2.0F * c) / (-b - std::sqrt(discriminant));
    return cumulative_width + std::clamp(root, 0.0F, 1.0F) * width;
}

ExpandedPrior expand_prior(
    const PiperTtsConfig & config,
    const std::vector<float> & stats,
    const std::vector<float> & log_duration,
    const PiperTtsGenerationOptions & options) {
    const int64_t token_count = static_cast<int64_t>(log_duration.size());
    const size_t expected_stats = static_cast<size_t>(
        2 * config.inter_channels * token_count);
    if (stats.size() != expected_stats || token_count <= 0) {
        throw std::runtime_error("Piper TTS duration graph returned invalid output");
    }
    std::vector<int64_t> durations(static_cast<size_t>(token_count), 0);
    int64_t total = 0;
    for (int64_t token = 0; token < token_count; ++token) {
        const double value =
            std::exp(static_cast<double>(log_duration[static_cast<size_t>(token)])) /
            static_cast<double>(options.speaking_rate);
        if (!std::isfinite(value) || value < 0.0) {
            throw std::runtime_error("Piper TTS duration predictor produced a non-finite duration");
        }
        if (value > static_cast<double>(kMaxLatentFrames)) {
            throw std::runtime_error(
                "Piper TTS expanded latent exceeds the 4000-frame limit");
        }
        const int64_t duration = static_cast<int64_t>(std::ceil(value));
        if (duration > kMaxLatentFrames - total) {
            throw std::runtime_error(
                "Piper TTS expanded latent exceeds the 4000-frame limit");
        }
        durations[static_cast<size_t>(token)] = duration;
        total += duration;
    }
    if (total <= 0) {
        throw std::runtime_error("Piper TTS duration predictor produced no audio frames");
    }

    auto noise = sampling::generate_normal_noise(
        static_cast<size_t>(config.inter_channels * total),
        options.seed);
    ExpandedPrior out;
    out.frames = total;
    out.latent.resize(noise.size());
    int64_t frame = 0;
    for (int64_t token = 0; token < token_count; ++token) {
        for (int64_t offset = 0; offset < durations[static_cast<size_t>(token)]; ++offset) {
            for (int64_t channel = 0; channel < config.inter_channels; ++channel) {
                const size_t stat_index =
                    static_cast<size_t>(channel * token_count + token);
                const size_t log_index =
                    static_cast<size_t>((channel + config.inter_channels) * token_count + token);
                const size_t output_index =
                    static_cast<size_t>(channel * total + frame);
                const float mean = stats[stat_index];
                const float log_scale = stats[log_index];
                const float latent =
                    mean +
                    noise[output_index] *
                        std::exp(log_scale) *
                        options.variation;
                if (!std::isfinite(latent)) {
                    throw std::runtime_error("Piper TTS prior expansion produced a non-finite latent");
                }
                out.latent[output_index] = latent;
            }
            ++frame;
        }
    }
    return out;
}

}  // namespace

struct PiperVitsRuntime::State {
    struct BackendOwner {
        ggml_backend_t value = nullptr;
        ~BackendOwner() {
            if (value != nullptr) {
                ggml_backend_free(value);
            }
        }
    };

    State(
        std::shared_ptr<const PiperTtsAssets> assets_in,
        core::BackendConfig backend_config)
        : assets(std::move(assets_in)),
          threads(std::max(1, backend_config.threads)) {
        if (assets == nullptr) {
            throw std::runtime_error("Piper TTS native runtime requires assets");
        }
        backend_config.threads = threads;
        backend.value = core::init_backend(backend_config);
        backend_type = core::backend_type(backend.value);
        core::set_backend_threads(backend.value, threads);
        weights = load_weights(assets, backend.value, backend_type);
        if (backend_type == core::BackendType::Cuda) {
            // Host duration alignment prevents small TF32 differences from being
            // amplified by monotone expansion while the expensive decoder stays on CUDA.
            core::BackendConfig duration_config{
                core::BackendType::Cpu,
                0,
                threads,
            };
            duration_backend.value = core::init_backend(duration_config);
            core::set_backend_threads(duration_backend.value, threads);
            duration_weights = load_weights(
                assets,
                duration_backend.value,
                core::BackendType::Cpu);
        }
    }

    DurationGraph & duration_graph(int64_t token_count) {
        if (auto * found = duration_graphs.find(token_count)) {
            engine::debug::trace_log_scalar(
                "piper_tts.duration_graph.cache_hit",
                true);
            return **found;
        }
        engine::debug::trace_log_scalar(
            "piper_tts.duration_graph.cache_hit",
            false);
        const auto duration_backend_value =
            duration_backend.value != nullptr
            ? duration_backend.value
            : backend.value;
        const auto duration_backend_type =
            duration_backend.value != nullptr
            ? core::BackendType::Cpu
            : backend_type;
        const auto & selected_weights =
            duration_weights != nullptr
            ? duration_weights
            : weights;
        duration_graphs.put(
            token_count,
            build_duration_graph(
                *selected_weights,
                assets->config,
                duration_backend_value,
                duration_backend_type,
                token_count));
        auto * created = duration_graphs.find(token_count);
        if (created == nullptr) {
            throw std::runtime_error("Piper TTS duration graph cache insert failed");
        }
        return **created;
    }

    DecoderGraph & decoder_graph(int64_t frames) {
        if (auto * found = decoder_graphs.find(frames)) {
            engine::debug::trace_log_scalar(
                "piper_tts.decoder_graph.cache_hit",
                true);
            return **found;
        }
        engine::debug::trace_log_scalar(
            "piper_tts.decoder_graph.cache_hit",
            false);
        decoder_graphs.put(
            frames,
            build_decoder_graph(
                *weights,
                assets->config,
                backend.value,
                backend_type,
                frames));
        auto * created = decoder_graphs.find(frames);
        if (created == nullptr) {
            throw std::runtime_error("Piper TTS decoder graph cache insert failed");
        }
        return **created;
    }

    DurationFlowGraph & duration_flow_graph(int64_t token_count, int flow) {
        const int slot = flow == 7 ? 0 : (flow == 5 ? 1 : 2);
        if (flow != 7 && flow != 5 && flow != 3) {
            throw std::runtime_error("Piper TTS invalid duration flow");
        }
        auto & cache = duration_flow_graphs[static_cast<size_t>(slot)];
        if (auto * found = cache.find(token_count)) {
            return **found;
        }
        cache.put(
            token_count,
            build_duration_flow_graph(
                *weights, backend.value, backend_type, token_count, flow));
        auto * created = cache.find(token_count);
        if (created == nullptr) {
            throw std::runtime_error("Piper TTS duration-flow graph cache insert failed");
        }
        return **created;
    }

    std::shared_ptr<const PiperTtsAssets> assets;
    int threads = 1;
    core::BackendType backend_type = core::BackendType::Cpu;
    BackendOwner backend;
    std::shared_ptr<const PiperTtsBackendWeights> weights;
    BackendOwner duration_backend;
    std::shared_ptr<const PiperTtsBackendWeights> duration_weights;
    runtime::CacheSlots<int64_t, std::unique_ptr<DurationGraph>> duration_graphs{4};
    std::array<runtime::CacheSlots<int64_t, std::unique_ptr<DurationFlowGraph>>, 3>
        duration_flow_graphs;
    runtime::CacheSlots<int64_t, std::unique_ptr<DecoderGraph>> decoder_graphs{2};
};

PiperVitsRuntime::PiperVitsRuntime(
    std::shared_ptr<const PiperTtsAssets> assets,
    core::BackendConfig backend_config)
    : state_(std::make_unique<State>(std::move(assets), backend_config)) {}

PiperVitsRuntime::~PiperVitsRuntime() = default;

runtime::AudioBuffer PiperVitsRuntime::synthesize(
    const std::vector<int32_t> & token_ids,
    const PiperTtsGenerationOptions & options) {
    if (token_ids.empty()) {
        throw std::runtime_error("Piper TTS requires at least one text token");
    }
    const auto total_start = std::chrono::steady_clock::now();
    auto duration_start = std::chrono::steady_clock::now();
    auto & duration = state_->duration_graph(
        static_cast<int64_t>(token_ids.size()));
    core::write_tensor_i32(
        core::wrap_tensor(
            duration.tokens,
            core::TensorShape::from_dims(
                {1, static_cast<int64_t>(token_ids.size())}),
            GGML_TYPE_I32),
        token_ids);
    compute_graph(duration, "Piper TTS duration");
    const auto stats = core::read_tensor_f32(duration.stats);
    const auto conditioning = core::read_tensor_f32(duration.conditioning);
    engine::debug::timing_log_scalar(
        "piper_tts.duration_ms",
        engine::debug::elapsed_ms(duration_start));

    const int64_t token_count = static_cast<int64_t>(token_ids.size());
    auto z = sampling::generate_normal_noise(
        static_cast<size_t>(2 * token_count),
        options.seed);
    for (float & value : z) {
        value *= options.duration_variation;
    }
    auto flip = [&]() {
        for (int64_t token = 0; token < token_count; ++token) {
            std::swap(z[static_cast<size_t>(token)],
                      z[static_cast<size_t>(token_count + token)]);
        }
    };
    flip();
    for (const int flow : {7, 5, 3}) {
        auto & graph = state_->duration_flow_graph(token_count, flow);
        core::write_tensor_f32(
            core::wrap_tensor(
                graph.x0,
                core::TensorShape::from_dims({1, 1, token_count}),
                GGML_TYPE_F32),
            std::vector<float>(z.begin(), z.begin() + token_count));
        core::write_tensor_f32(
            core::wrap_tensor(
                graph.conditioning,
                core::TensorShape::from_dims({1, 192, token_count}),
                GGML_TYPE_F32),
            conditioning);
        compute_graph(graph, "Piper TTS duration flow");
        const auto parameters = core::read_tensor_f32(graph.parameters);
        for (int64_t token = 0; token < token_count; ++token) {
            float widths[10];
            float heights[10];
            float derivatives[9];
            for (int i = 0; i < 10; ++i) {
                widths[i] = parameters[static_cast<size_t>(i * token_count + token)] /
                    std::sqrt(192.0F);
                heights[i] = parameters[static_cast<size_t>((10 + i) * token_count + token)] /
                    std::sqrt(192.0F);
            }
            for (int i = 0; i < 9; ++i) {
                derivatives[i] = parameters[static_cast<size_t>((20 + i) * token_count + token)];
            }
            z[static_cast<size_t>(token_count + token)] = inverse_rational_quadratic(
                z[static_cast<size_t>(token_count + token)],
                widths,
                heights,
                derivatives);
        }
        flip();
    }
    std::vector<float> log_duration(
        z.begin(), z.begin() + token_count);
    for (float & value : log_duration) {
        value = (value - state_->weights->duration_affine_mean[0]) *
            state_->weights->duration_affine_scale[0];
    }
    auto prior = expand_prior(
        state_->assets->config,
        stats,
        log_duration,
        options);
    auto decoder_start = std::chrono::steady_clock::now();
    auto & decoder_graph = state_->decoder_graph(prior.frames);
    core::write_tensor_f32(
        core::wrap_tensor(
            decoder_graph.latent,
            core::TensorShape::from_dims({
                1,
                state_->assets->config.inter_channels,
                prior.frames,
            }),
            GGML_TYPE_F32),
        prior.latent);
    compute_graph(decoder_graph, "Piper TTS decoder");
    runtime::AudioBuffer out;
    out.sample_rate = static_cast<int>(state_->assets->config.sample_rate);
    out.channels = 1;
    out.samples = core::read_tensor_f32(decoder_graph.waveform);
    engine::debug::timing_log_scalar(
        "piper_tts.decoder_ms",
        engine::debug::elapsed_ms(decoder_start));
    engine::debug::trace_log_scalar(
        "piper_tts.token_count",
        static_cast<int64_t>(token_ids.size()));
    engine::debug::trace_log_scalar(
        "piper_tts.latent_frames",
        prior.frames);
    engine::debug::trace_log_scalar(
        "piper_tts.output_samples",
        static_cast<int64_t>(out.samples.size()));
    engine::debug::timing_log_scalar(
        "session.wall_ms",
        engine::debug::elapsed_ms(total_start));
    return out;
}

void apply_piper_tts_edge_fade(
    std::vector<float> & samples,
    int sample_rate,
    float milliseconds) {
    if (samples.empty() || sample_rate <= 0 || milliseconds <= 0.0F) {
        return;
    }
    const size_t frames = std::min(
        samples.size() / 2,
        static_cast<size_t>(std::llround(
            static_cast<double>(sample_rate) *
            static_cast<double>(milliseconds) /
            1000.0)));
    if (frames == 0) {
        return;
    }
    for (size_t index = 0; index < frames; ++index) {
        const float gain = frames == 1
            ? 1.0F
            : static_cast<float>(index) /
                static_cast<float>(frames - 1);
        samples[index] *= gain;
        samples[samples.size() - 1 - index] *= gain;
    }
}

}  // namespace engine::models::piper_tts
