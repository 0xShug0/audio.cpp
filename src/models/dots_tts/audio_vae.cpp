#include "engine/models/dots_tts/audio_vae.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/recurrent_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::dots_tts {
namespace {

constexpr int64_t kActivationRatio = 2;
constexpr int64_t kActivationKernel = 12;
constexpr int64_t kActivationPhaseKernel = kActivationKernel / kActivationRatio;
constexpr int64_t kEncoderBaseChannels = 12;
constexpr int64_t kProjectionKernel = 3;
constexpr int64_t kResStackLayers = 6;
constexpr int64_t kResStackDilationBase = 2;
constexpr int64_t kDefaultLookahead = 2;
constexpr float kLeakyReluSlope = 0.2F;
constexpr float kResStackLeakyReluSlope = 0.01F;
constexpr size_t kWeightContextBytes = 256ull * 1024ull * 1024ull;
constexpr size_t kGraphContextBytes = 512ull * 1024ull * 1024ull;

using engine::modules::AddModule;
using engine::modules::AliasFreeActivationModule;
using engine::modules::AliasFreeActivationWeights;
using engine::modules::ConcatModule;
using engine::modules::Conv1dModule;
using engine::modules::Conv1dWeights;
using engine::modules::ConvTranspose1dModule;
using engine::modules::ConvTranspose1dWeights;
using engine::modules::LinearModule;
using engine::modules::LinearWeights;
using engine::modules::LSTMSequenceModule;
using engine::modules::LSTMSequenceWeights;
using engine::modules::LSTMStackWeights;
using engine::modules::MulModule;
using engine::modules::SliceModule;
using engine::modules::StreamingConv1dModule;
using engine::modules::StreamingPadMode;
using engine::modules::TransposeModule;

int64_t tensor_elements(const std::vector<int64_t> & shape) {
    if (shape.empty()) {
        throw std::runtime_error("DotTTS AudioVAE tensor shape is empty");
    }
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, [](int64_t lhs, int64_t rhs) {
        if (rhs <= 0) {
            throw std::runtime_error("DotTTS AudioVAE tensor shape contains non-positive dimension");
        }
        return lhs * rhs;
    });
}

int64_t int_pow(int64_t base, int64_t exponent) {
    int64_t out = 1;
    for (int64_t i = 0; i < exponent; ++i) {
        out *= base;
    }
    return out;
}

std::vector<float> frame_major_to_channel_major(
    const std::vector<float> & values,
    int64_t frames,
    int64_t channels) {
    if (frames <= 0 || channels <= 0 || static_cast<int64_t>(values.size()) != frames * channels) {
        throw std::runtime_error("DotTTS AudioVAE latent layout conversion shape mismatch");
    }
    std::vector<float> out(values.size(), 0.0F);
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t channel = 0; channel < channels; ++channel) {
            out[static_cast<size_t>(channel * frames + frame)] =
                values[static_cast<size_t>(frame * channels + channel)];
        }
    }
    return out;
}

using ActivationWeights = AliasFreeActivationWeights;

struct ResBlockWeights {
    std::vector<Conv1dWeights> convs1;
    std::vector<Conv1dWeights> convs2;
    std::vector<ActivationWeights> activations;
};

core::TensorValue amp_block_bct(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ResBlockWeights & block,
    int64_t channels,
    int64_t kernel) {
    auto x = input;
    const int dilations[3] = {1, 3, 5};
    const AliasFreeActivationModule activation({channels, kActivationKernel, kActivationRatio});
    for (int64_t layer = 0; layer < 3; ++layer) {
        auto xt = activation.build(ctx, x, block.activations[static_cast<size_t>(2 * layer)]);
        xt = StreamingConv1dModule({
            channels,
            channels,
            kernel,
            1,
            dilations[layer],
            true,
            StreamingPadMode::Constant,
        }).build(ctx, xt, block.convs1[static_cast<size_t>(layer)]);
        xt = activation.build(ctx, xt, block.activations[static_cast<size_t>(2 * layer + 1)]);
        xt = StreamingConv1dModule({
            channels,
            channels,
            kernel,
            1,
            1,
            true,
            StreamingPadMode::Constant,
        }).build(ctx, xt, block.convs2[static_cast<size_t>(layer)]);
        x = AddModule().build(ctx, x, xt);
    }
    return x;
}

struct ResStackWeights {
    std::vector<Conv1dWeights> convs1;
    std::vector<Conv1dWeights> convs2;
};

struct DotsAudioVaeWeights {
    DotsVocoderConfig config;
    std::shared_ptr<core::ExecutionContext> execution_context;
    std::shared_ptr<core::BackendWeightStore> store;
    std::vector<Conv1dWeights> encoder_convs;
    std::vector<ResStackWeights> encoder_resstacks;
    LinearWeights enc_mi_in;
    LSTMStackWeights enc_mi_lstm;
    LinearWeights enc_mi_out;
    Conv1dWeights pre_proj;
    Conv1dWeights post_proj;
    LinearWeights dec_mi_in;
    LSTMStackWeights dec_mi_lstm;
    LinearWeights dec_mi_out;
    Conv1dWeights decoder_conv_pre;
    std::vector<ConvTranspose1dWeights> decoder_ups;
    std::vector<ResBlockWeights> decoder_resblocks;
    ActivationWeights decoder_activation_post;
    Conv1dWeights decoder_conv_post;
    int64_t loaded_tensor_count = 0;
    int64_t parameter_count = 0;
};

struct EncoderGraphOutputs {
    core::TensorValue latent_distribution;
};

ActivationWeights load_alias_free_activation(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    auto alpha = source.require_f32(prefix + ".act.alpha", {channels});
    auto inv_beta = source.require_f32(prefix + ".act.beta", {channels});
    for (auto & value : alpha) {
        value = std::exp(value);
    }
    for (auto & value : inv_beta) {
        value = 1.0F / (std::exp(value) + 1.0e-9F);
    }

    auto expand_filter = [&](const std::vector<float> & filter, const std::vector<int64_t> & shape, const std::string & name) {
        const bool expanded = shape == std::vector<int64_t>{channels, 1, kActivationKernel};
        if (expanded) {
            if (static_cast<int64_t>(filter.size()) != channels * kActivationKernel) {
                throw std::runtime_error(name + " expanded activation filter shape mismatch");
            }
            return filter;
        }
        if (static_cast<int64_t>(filter.size()) != kActivationKernel) {
            throw std::runtime_error(name + " activation filter shape mismatch");
        }
        std::vector<float> out(static_cast<size_t>(channels * kActivationKernel), 0.0F);
        for (int64_t channel = 0; channel < channels; ++channel) {
            std::copy(
                filter.begin(),
                filter.end(),
                out.begin() + static_cast<std::ptrdiff_t>(channel * kActivationKernel));
        }
        return out;
    };
    auto reverse_taps = [](std::vector<float> values) {
        if (values.size() % static_cast<size_t>(kActivationKernel) != 0) {
            throw std::runtime_error("DotTTS AudioVAE activation filter size mismatch");
        }
        for (size_t offset = 0; offset < values.size(); offset += static_cast<size_t>(kActivationKernel)) {
            std::reverse(
                values.begin() + static_cast<std::ptrdiff_t>(offset),
                values.begin() + static_cast<std::ptrdiff_t>(offset + static_cast<size_t>(kActivationKernel)));
        }
        return values;
    };
    auto phase_filter = [&](const std::vector<float> & filter, int64_t phase) {
        std::vector<float> out(static_cast<size_t>(channels * kActivationPhaseKernel), 0.0F);
        for (int64_t channel = 0; channel < channels; ++channel) {
            for (int64_t index = 0; index < kActivationPhaseKernel; ++index) {
                out[static_cast<size_t>(channel * kActivationPhaseKernel + index)] =
                    filter[static_cast<size_t>(channel * kActivationKernel + phase + index * kActivationRatio)];
            }
        }
        return out;
    };

    const std::string up_name = prefix + ".upsample.filter";
    const std::string down_name = prefix + ".downsample.lowpass.filter";
    const auto up_shape = source.require_metadata(up_name).shape;
    const auto down_shape = source.require_metadata(down_name).shape;
    const auto up_filter = expand_filter(reverse_taps(source.require_f32(up_name, up_shape)), up_shape, up_name);
    const auto down_filter = expand_filter(source.require_f32(down_name, down_shape), down_shape, down_name);
    return {
        store.make_f32(core::TensorShape::from_dims({channels}), alpha),
        store.make_f32(core::TensorShape::from_dims({channels}), inv_beta),
        store.make_f32(core::TensorShape::from_dims({channels, 1, 1, kActivationPhaseKernel}), phase_filter(up_filter, 1)),
        store.make_f32(core::TensorShape::from_dims({channels, 1, 1, kActivationPhaseKernel}), phase_filter(up_filter, 0)),
        store.make_f32(core::TensorShape::from_dims({channels, 1, 1, kActivationKernel}), down_filter),
    };
}

ResStackWeights load_resstack(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    assets::TensorStorageType storage_type) {
    ResStackWeights out;
    out.convs1.reserve(kResStackLayers);
    out.convs2.reserve(kResStackLayers);
    for (int64_t layer = 0; layer < kResStackLayers; ++layer) {
        out.convs1.push_back(engine::modules::binding::conv1d_from_source_resolving_weight_norm(
            store,
            source,
            prefix + ".layers." + std::to_string(layer) + ".2",
            storage_type,
            channels,
            channels,
            3,
            true));
        out.convs2.push_back(engine::modules::binding::conv1d_from_source_resolving_weight_norm(
            store,
            source,
            prefix + ".layers." + std::to_string(layer) + ".5",
            storage_type,
            channels,
            channels,
            3,
            true));
    }
    return out;
}

ResBlockWeights load_resblock(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    int64_t kernel,
    assets::TensorStorageType storage_type) {
    ResBlockWeights out;
    out.convs1.reserve(3);
    out.convs2.reserve(3);
    out.activations.reserve(6);
    for (int64_t layer = 0; layer < 3; ++layer) {
        out.convs1.push_back(engine::modules::binding::conv1d_from_source_resolving_weight_norm(
            store,
            source,
            prefix + ".convs1." + std::to_string(layer),
            storage_type,
            channels,
            channels,
            kernel,
            true));
        out.convs2.push_back(engine::modules::binding::conv1d_from_source_resolving_weight_norm(
            store,
            source,
            prefix + ".convs2." + std::to_string(layer),
            storage_type,
            channels,
            channels,
            kernel,
            true));
    }
    for (int64_t layer = 0; layer < 6; ++layer) {
        out.activations.push_back(load_alias_free_activation(
            store,
            source,
            prefix + ".activations." + std::to_string(layer),
            channels));
    }
    return out;
}

core::TensorValue apply_slstm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const LSTMStackWeights & weights,
    int64_t hidden) {
    auto x = input;
    const auto residual = input;
    for (const auto & layer : weights.layers) {
        auto zero_state = core::wrap_tensor(
            ggml_scale(ctx.ggml, SliceModule({0, 0, 1}).build(ctx, x).tensor, 0.0F),
            core::TensorShape::from_dims({1, hidden}),
            GGML_TYPE_F32);
        auto out = LSTMSequenceModule({hidden, hidden, false}).build(ctx, x, zero_state, zero_state, LSTMSequenceWeights{layer});
        x = out.sequence;
    }
    return AddModule().build(ctx, x, residual);
}

EncoderGraphOutputs build_encoder_graph(
    core::ModuleBuildContext & ctx,
    const DotsAudioVaeWeights & weights,
    const core::TensorValue & waveform) {
    const int64_t initial_left_pad = kProjectionKernel - 1;
    auto * initial_padded = ggml_pad_ext(
        ctx.ggml,
        core::ensure_backend_addressable_layout(ctx, waveform).tensor,
        static_cast<int>(initial_left_pad),
        0,
        0,
        0,
        0,
        0,
        0,
        0);
    auto x = Conv1dModule({
        1,
        kEncoderBaseChannels,
        kProjectionKernel,
        1,
        0,
        1,
        true,
    }).build(
        ctx,
        core::wrap_tensor(
            initial_padded,
            core::TensorShape::from_dims({waveform.shape.dims[0], waveform.shape.dims[1], waveform.shape.dims[2] + initial_left_pad}),
            GGML_TYPE_F32),
        weights.encoder_convs[0]);
    EncoderGraphOutputs out;
    x = core::wrap_tensor(
        ggml_leaky_relu(ctx.ggml, core::ensure_backend_addressable_layout(ctx, x).tensor, kLeakyReluSlope, false),
        x.shape,
        GGML_TYPE_F32);
    for (size_t stage = 0; stage < weights.config.downsample_rates.size(); ++stage) {
        const int64_t in_channels = weights.config.downsample_channels[stage];
        const int64_t out_channels = weights.config.downsample_channels[stage + 1];
        const int64_t factor = weights.config.downsample_rates[stage];
        const int64_t downsample_kernel = factor * 2;
        const int64_t downsample_left_pad = downsample_kernel - 1;
        auto * downsample_padded = ggml_pad_ext(
            ctx.ggml,
            core::ensure_backend_addressable_layout(ctx, x).tensor,
            static_cast<int>(downsample_left_pad),
            0,
            0,
            0,
            0,
            0,
            0,
            0);
        x = Conv1dModule({
            in_channels,
            out_channels,
            downsample_kernel,
            static_cast<int>(factor),
            0,
            1,
            true,
        }).build(
            ctx,
            core::wrap_tensor(
                downsample_padded,
                core::TensorShape::from_dims({x.shape.dims[0], x.shape.dims[1], x.shape.dims[2] + downsample_left_pad}),
                GGML_TYPE_F32),
            weights.encoder_convs[stage + 1]);
        const auto & stack = weights.encoder_resstacks[stage];
        for (int64_t layer = 0; layer < kResStackLayers; ++layer) {
            auto xt = core::wrap_tensor(
                ggml_leaky_relu(ctx.ggml, core::ensure_backend_addressable_layout(ctx, x).tensor, kResStackLeakyReluSlope, false),
                x.shape,
                GGML_TYPE_F32);
            const int64_t dilation = int_pow(kResStackDilationBase, layer);
            const int64_t conv1_left_pad = dilation * 2;
            auto * conv1_padded = ggml_pad_ext(
                ctx.ggml,
                core::ensure_backend_addressable_layout(ctx, xt).tensor,
                static_cast<int>(conv1_left_pad),
                0,
                0,
                0,
                0,
                0,
                0,
                0);
            xt = Conv1dModule({
                out_channels,
                out_channels,
                3,
                1,
                0,
                static_cast<int>(dilation),
                true,
            }).build(
                ctx,
                core::wrap_tensor(
                    conv1_padded,
                    core::TensorShape::from_dims({xt.shape.dims[0], xt.shape.dims[1], xt.shape.dims[2] + conv1_left_pad}),
                    GGML_TYPE_F32),
                stack.convs1[static_cast<size_t>(layer)]);
            xt = core::wrap_tensor(
                ggml_leaky_relu(ctx.ggml, core::ensure_backend_addressable_layout(ctx, xt).tensor, kResStackLeakyReluSlope, false),
                xt.shape,
                GGML_TYPE_F32);
            auto * conv2_padded = ggml_pad_ext(
                ctx.ggml,
                core::ensure_backend_addressable_layout(ctx, xt).tensor,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0);
            xt = Conv1dModule({
                out_channels,
                out_channels,
                3,
                1,
                0,
                1,
                true,
            }).build(
                ctx,
                core::wrap_tensor(
                    conv2_padded,
                    core::TensorShape::from_dims({xt.shape.dims[0], xt.shape.dims[1], xt.shape.dims[2] + 2}),
                    GGML_TYPE_F32),
                stack.convs2[static_cast<size_t>(layer)]);
            x = AddModule().build(ctx, x, xt);
        }
        x = core::wrap_tensor(
            ggml_leaky_relu(ctx.ggml, core::ensure_backend_addressable_layout(ctx, x).tensor, kLeakyReluSlope, false),
            x.shape,
            GGML_TYPE_F32);
    }
    x = Conv1dModule({
        weights.config.downsample_channels.back(),
        weights.config.latent_dim,
        2 * kDefaultLookahead + 1,
        1,
        kDefaultLookahead,
        1,
        true,
    }).build(ctx, x, weights.encoder_convs.back());
    x = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x)),
        core::TensorShape::from_dims({x.shape.dims[2], x.shape.dims[1]}));
    x = LinearModule({weights.config.latent_dim, weights.config.latent_dim * 4, true}).build(ctx, x, weights.enc_mi_in);
    x = apply_slstm(ctx, x, weights.enc_mi_lstm, weights.config.latent_dim * 4);
    x = LinearModule({weights.config.latent_dim * 4, weights.config.latent_dim, true}).build(ctx, x, weights.enc_mi_out);
    x = TransposeModule({{0, 2, 1, 3}, 3}).build(
        ctx,
        core::reshape_tensor(
            ctx,
            core::ensure_backend_addressable_layout(ctx, x),
            core::TensorShape::from_dims({1, x.shape.dims[0], x.shape.dims[1]})));
    out.latent_distribution = Conv1dModule({
        weights.config.latent_dim,
        weights.config.latent_dim * 2,
        1,
        1,
        0,
        1,
        true,
    }).build(ctx, x, weights.pre_proj);
    return out;
}

core::TensorValue build_decoder_graph(
    core::ModuleBuildContext & ctx,
    const DotsAudioVaeWeights & weights,
    const core::TensorValue & latents) {
    auto x = Conv1dModule({
        weights.config.latent_dim,
        weights.config.latent_dim,
        1,
        1,
        0,
        1,
        true,
    }).build(ctx, latents, weights.post_proj);
    x = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x)),
        core::TensorShape::from_dims({x.shape.dims[2], x.shape.dims[1]}));
    x = LinearModule({weights.config.latent_dim, weights.config.latent_dim * 4, true}).build(ctx, x, weights.dec_mi_in);
    x = apply_slstm(ctx, x, weights.dec_mi_lstm, weights.config.latent_dim * 4);
    x = core::ensure_backend_addressable_layout(ctx, x);
    x = LinearModule({weights.config.latent_dim * 4, weights.config.latent_dim, true}).build(ctx, x, weights.dec_mi_out);
    x = core::ensure_backend_addressable_layout(ctx, x);
    return core::ensure_backend_addressable_layout(
        ctx,
        TransposeModule({{0, 2, 1, 3}, 3}).build(
            ctx,
            core::reshape_tensor(
                ctx,
                core::ensure_backend_addressable_layout(ctx, x),
                core::TensorShape::from_dims({1, x.shape.dims[0], x.shape.dims[1]}))));
}

core::TensorValue build_decoder_waveform_graph(
    core::ModuleBuildContext & ctx,
    const DotsAudioVaeWeights & weights,
    const core::TensorValue & decoder_input) {
    auto x = Conv1dModule({
        weights.config.latent_dim,
        weights.config.upsample_initial_channel,
        2 * kDefaultLookahead + 1,
        1,
        kDefaultLookahead,
        1,
        true,
    }).build(ctx, decoder_input, weights.decoder_conv_pre);
    for (size_t stage = 0; stage < weights.config.upsample_rates.size(); ++stage) {
        const int64_t in_channels = weights.config.upsample_initial_channel / (int64_t{1} << stage);
        const int64_t out_channels = weights.config.upsample_initial_channel / (int64_t{1} << (stage + 1));
        x = ConvTranspose1dModule({
            in_channels,
            out_channels,
            weights.config.upsample_kernel_sizes[stage],
            static_cast<int>(weights.config.upsample_rates[stage]),
            0,
            1,
            true,
        }).build(ctx, x, weights.decoder_ups[stage]);
        x = SliceModule({2, 0, x.shape.dims[2] - weights.config.upsample_rates[stage]}).build(ctx, x);
        std::optional<core::TensorValue> sum;
        for (size_t kernel_index = 0; kernel_index < weights.config.resblock_kernel_sizes.size(); ++kernel_index) {
            const size_t block_index = stage * weights.config.resblock_kernel_sizes.size() + kernel_index;
            auto y = amp_block_bct(
                ctx,
                x,
                weights.decoder_resblocks[block_index],
                out_channels,
                weights.config.resblock_kernel_sizes[kernel_index]);
            sum = sum.has_value() ? AddModule().build(ctx, *sum, y) : std::make_optional(std::move(y));
        }
        if (!sum.has_value()) {
            throw std::runtime_error("DotTTS AudioVAE decoder stage has no residual blocks");
        }
        x = core::wrap_tensor(
            ggml_scale(ctx.ggml, core::ensure_backend_addressable_layout(ctx, *sum).tensor, 1.0F / static_cast<float>(weights.config.resblock_kernel_sizes.size())),
            sum->shape,
            GGML_TYPE_F32);
    }
    x = AliasFreeActivationModule({x.shape.dims[1], kActivationKernel, kActivationRatio}).build(
        ctx,
        x,
        weights.decoder_activation_post);
    x = StreamingConv1dModule({
        x.shape.dims[1],
        1,
        7,
        1,
        1,
        false,
        StreamingPadMode::Constant,
    }).build(ctx, x, weights.decoder_conv_post);
    if (weights.config.use_tanh_at_final) {
        x = engine::modules::TanhModule().build(ctx, x);
    } else {
        x = core::wrap_tensor(
            ggml_clamp(ctx.ggml, core::ensure_backend_addressable_layout(ctx, x).tensor, -1.0F, 1.0F),
            x.shape,
            GGML_TYPE_F32);
    }
    return core::ensure_backend_addressable_layout(ctx, x);
}

struct DecoderStreamFrontendOutputs {
    core::TensorValue decoder_input;
    std::vector<core::TensorValue> hidden;
    std::vector<core::TensorValue> cell;
};

struct DecoderStreamFrontendValues {
    std::vector<float> decoder_input;
    std::vector<std::vector<float>> hidden;
    std::vector<std::vector<float>> cell;
};

DecoderStreamFrontendOutputs build_decoder_stream_frontend_graph(
    core::ModuleBuildContext & ctx,
    const DotsAudioVaeWeights & weights,
    const core::TensorValue & latents,
    const std::vector<core::TensorValue> & initial_hidden,
    const std::vector<core::TensorValue> & initial_cell) {
    auto x = Conv1dModule({
        weights.config.latent_dim,
        weights.config.latent_dim,
        1,
        1,
        0,
        1,
        true,
    }).build(ctx, latents, weights.post_proj);
    x = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x)),
        core::TensorShape::from_dims({x.shape.dims[2], x.shape.dims[1]}));
    const int64_t hidden = weights.config.latent_dim * 4;
    x = LinearModule({weights.config.latent_dim, hidden, true}).build(ctx, x, weights.dec_mi_in);
    const auto residual = x;
    DecoderStreamFrontendOutputs out;
    out.hidden.reserve(weights.dec_mi_lstm.layers.size());
    out.cell.reserve(weights.dec_mi_lstm.layers.size());
    for (size_t layer = 0; layer < weights.dec_mi_lstm.layers.size(); ++layer) {
        auto sequence = LSTMSequenceModule({hidden, hidden, false}).build(
            ctx,
            x,
            initial_hidden[layer],
            initial_cell[layer],
            LSTMSequenceWeights{weights.dec_mi_lstm.layers[layer]});
        x = sequence.sequence;
        out.hidden.push_back(sequence.hidden);
        out.cell.push_back(sequence.cell);
    }
    x = AddModule().build(ctx, x, residual);
    x = LinearModule({hidden, weights.config.latent_dim, true}).build(ctx, x, weights.dec_mi_out);
    out.decoder_input = core::ensure_backend_addressable_layout(
        ctx,
        TransposeModule({{0, 2, 1, 3}, 3}).build(
            ctx,
            core::reshape_tensor(
                ctx,
                core::ensure_backend_addressable_layout(ctx, x),
                core::TensorShape::from_dims({1, x.shape.dims[0], x.shape.dims[1]}))));
    return out;
}

class EncoderRunner {
public:
    explicit EncoderRunner(std::shared_ptr<const DotsAudioVaeWeights> weights)
        : weights_(std::move(weights)) {}

    ~EncoderRunner() { release_graph(); }

    DotsEncoderLatents run(const std::vector<float> & waveform) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (waveform.empty()) {
            throw std::runtime_error("DotTTS AudioVAE encoder received empty waveform");
        }
        ensure_graph(static_cast<int64_t>(waveform.size()));
        ggml_backend_tensor_set(waveform_, waveform.data(), 0, waveform.size() * sizeof(float));
        if (core::compute_graph(*weights_->execution_context, graph_, plan_, "dots_tts.audio_vae.encoder") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DotTTS AudioVAE encoder graph compute failed");
        }
        DotsEncoderLatents out;
        out.channels = output_->ne[1];
        out.frames = output_->ne[0];
        out.values = core::read_tensor_f32(output_);
        return out;
    }

    void release_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(weights_->execution_context->backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
            ggml_ = nullptr;
        }
        plan_.reset();
        graph_ = nullptr;
        waveform_ = nullptr;
        output_ = nullptr;
        samples_ = 0;
    }

private:
    void ensure_graph(int64_t samples) {
        if (ggml_ != nullptr && samples_ == samples) {
            return;
        }
        release_graph();
        ggml_init_params params{kGraphContextBytes, nullptr, true};
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize DotTTS AudioVAE encoder graph context");
        }
        core::ModuleBuildContext build_ctx{ggml_, "dots_tts.audio_vae.encoder", weights_->execution_context->backend_type()};
        auto input = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, samples}));
        waveform_ = input.tensor;
        auto outputs = build_encoder_graph(build_ctx, *weights_, input);
        output_ = outputs.latent_distribution.tensor;
        graph_ = ggml_new_graph_custom(ggml_, 262144, false);
        ggml_build_forward_expand(graph_, output_);
        core::validate_backend_graph_supported(weights_->execution_context->backend(), graph_, "dots_tts.audio_vae.encoder");
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_->execution_context->backend()));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate DotTTS AudioVAE encoder graph memory");
        }
        core::prepare_host_graph_plan(*weights_->execution_context, graph_, plan_);
        samples_ = samples;
    }

    std::shared_ptr<const DotsAudioVaeWeights> weights_;
    std::mutex mutex_;
    ggml_context * ggml_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    core::HostGraphPlan plan_;
    ggml_tensor * waveform_ = nullptr;
    ggml_tensor * output_ = nullptr;
    int64_t samples_ = 0;
};

class DecoderRunner {
public:
    explicit DecoderRunner(std::shared_ptr<const DotsAudioVaeWeights> weights)
        : weights_(std::move(weights)) {}

    ~DecoderRunner() { release_graph(); }

    DotsDecodedAudio run(const std::vector<float> & latents, int64_t frames) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (frames <= 0 || static_cast<int64_t>(latents.size()) != weights_->config.latent_dim * frames) {
            throw std::runtime_error("DotTTS AudioVAE decoder latent size mismatch");
        }
        ensure_graph(frames);
        ggml_backend_tensor_set(latents_, latents.data(), 0, latents.size() * sizeof(float));
        if (core::compute_graph(*weights_->execution_context, graph_, plan_, "dots_tts.audio_vae.decoder") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DotTTS AudioVAE decoder graph compute failed");
        }
        DotsDecodedAudio out;
        out.sample_rate = weights_->config.sample_rate;
        out.samples = core::read_tensor_f32(output_);
        return out;
    }

    void release_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(weights_->execution_context->backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
            ggml_ = nullptr;
        }
        plan_.reset();
        graph_ = nullptr;
        latents_ = nullptr;
        output_ = nullptr;
        frames_ = 0;
    }

private:
    void ensure_graph(int64_t frames) {
        if (ggml_ != nullptr && frames_ == frames) {
            return;
        }
        release_graph();
        ggml_init_params params{kGraphContextBytes, nullptr, true};
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize DotTTS AudioVAE decoder graph context");
        }
        core::ModuleBuildContext build_ctx{ggml_, "dots_tts.audio_vae.decoder", weights_->execution_context->backend_type()};
        auto input = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, weights_->config.latent_dim, frames}));
        latents_ = input.tensor;
        auto decoder_input = build_decoder_graph(build_ctx, *weights_, input);
        auto output = build_decoder_waveform_graph(build_ctx, *weights_, decoder_input);
        output_ = output.tensor;
        graph_ = ggml_new_graph_custom(ggml_, 524288, false);
        ggml_build_forward_expand(graph_, output_);
        core::validate_backend_graph_supported(weights_->execution_context->backend(), graph_, "dots_tts.audio_vae.decoder");
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_->execution_context->backend()));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate DotTTS AudioVAE decoder graph memory");
        }
        core::prepare_host_graph_plan(*weights_->execution_context, graph_, plan_);
        frames_ = frames;
    }

    std::shared_ptr<const DotsAudioVaeWeights> weights_;
    std::mutex mutex_;
    ggml_context * ggml_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    core::HostGraphPlan plan_;
    ggml_tensor * latents_ = nullptr;
    ggml_tensor * output_ = nullptr;
    int64_t frames_ = 0;
};

class DecoderWindowRunner {
public:
    explicit DecoderWindowRunner(std::shared_ptr<const DotsAudioVaeWeights> weights)
        : weights_(std::move(weights)) {}

    ~DecoderWindowRunner() { release_graph(); }

    DotsDecodedAudio run(const std::vector<float> & decoder_window, int64_t frames) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (frames <= 0 || static_cast<int64_t>(decoder_window.size()) != weights_->config.latent_dim * frames) {
            throw std::runtime_error("DotTTS AudioVAE decoder window size mismatch");
        }
        ensure_graph(frames);
        ggml_backend_tensor_set(window_, decoder_window.data(), 0, decoder_window.size() * sizeof(float));
        if (core::compute_graph(*weights_->execution_context, graph_, plan_, "dots_tts.audio_vae.stream.window_decoder") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DotTTS AudioVAE streaming decoder graph compute failed");
        }
        DotsDecodedAudio out;
        out.sample_rate = weights_->config.sample_rate;
        out.samples = core::read_tensor_f32(output_);
        return out;
    }

    void release_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(weights_->execution_context->backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
            ggml_ = nullptr;
        }
        plan_.reset();
        graph_ = nullptr;
        window_ = nullptr;
        output_ = nullptr;
        frames_ = 0;
    }

private:
    void ensure_graph(int64_t frames) {
        if (ggml_ != nullptr && frames_ == frames) {
            return;
        }
        release_graph();
        ggml_init_params params{kGraphContextBytes, nullptr, true};
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize DotTTS AudioVAE streaming decoder graph context");
        }
        core::ModuleBuildContext build_ctx{ggml_, "dots_tts.audio_vae.stream.window_decoder", weights_->execution_context->backend_type()};
        auto input = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, weights_->config.latent_dim, frames}));
        window_ = input.tensor;
        auto output = build_decoder_waveform_graph(build_ctx, *weights_, input);
        output_ = output.tensor;
        graph_ = ggml_new_graph_custom(ggml_, 524288, false);
        ggml_build_forward_expand(graph_, output_);
        core::validate_backend_graph_supported(weights_->execution_context->backend(), graph_, "dots_tts.audio_vae.stream.window_decoder");
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_->execution_context->backend()));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate DotTTS AudioVAE streaming decoder graph memory");
        }
        core::prepare_host_graph_plan(*weights_->execution_context, graph_, plan_);
        frames_ = frames;
    }

    std::shared_ptr<const DotsAudioVaeWeights> weights_;
    std::mutex mutex_;
    ggml_context * ggml_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    core::HostGraphPlan plan_;
    ggml_tensor * window_ = nullptr;
    ggml_tensor * output_ = nullptr;
    int64_t frames_ = 0;
};

class StreamFrontendRunner {
public:
    explicit StreamFrontendRunner(std::shared_ptr<const DotsAudioVaeWeights> weights)
        : weights_(std::move(weights)) {}

    ~StreamFrontendRunner() { release_graph(); }

    DecoderStreamFrontendValues run(
        const std::vector<float> & latents,
        int64_t frames,
        const std::vector<float> & hidden,
        const std::vector<float> & cell) {
        std::lock_guard<std::mutex> lock(mutex_);
        const int64_t hidden_size = weights_->config.latent_dim * 4;
        const int64_t layers = static_cast<int64_t>(weights_->dec_mi_lstm.layers.size());
        if (frames <= 0 || static_cast<int64_t>(latents.size()) != weights_->config.latent_dim * frames ||
            static_cast<int64_t>(hidden.size()) != layers * hidden_size ||
            static_cast<int64_t>(cell.size()) != layers * hidden_size) {
            throw std::runtime_error("DotTTS AudioVAE streaming frontend tensor size mismatch");
        }
        ensure_graph(frames);
        ggml_backend_tensor_set(latents_, latents.data(), 0, latents.size() * sizeof(float));
        ggml_backend_tensor_set(hidden_, hidden.data(), 0, hidden.size() * sizeof(float));
        ggml_backend_tensor_set(cell_, cell.data(), 0, cell.size() * sizeof(float));
        if (core::compute_graph(*weights_->execution_context, graph_, plan_, "dots_tts.audio_vae.stream.frontend") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DotTTS AudioVAE streaming frontend graph compute failed");
        }
        DecoderStreamFrontendValues out;
        out.decoder_input = core::read_tensor_f32(decoder_input_);
        out.hidden.reserve(hidden_outputs_.size());
        out.cell.reserve(cell_outputs_.size());
        for (size_t layer = 0; layer < hidden_outputs_.size(); ++layer) {
            out.hidden.push_back(core::read_tensor_f32(hidden_outputs_[layer]));
            out.cell.push_back(core::read_tensor_f32(cell_outputs_[layer]));
        }
        return out;
    }

    void release_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(weights_->execution_context->backend(), graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
            ggml_ = nullptr;
        }
        plan_.reset();
        graph_ = nullptr;
        latents_ = nullptr;
        hidden_ = nullptr;
        cell_ = nullptr;
        decoder_input_ = nullptr;
        hidden_inputs_.clear();
        cell_inputs_.clear();
        hidden_outputs_.clear();
        cell_outputs_.clear();
        frames_ = 0;
    }

private:
    void ensure_graph(int64_t frames) {
        if (ggml_ != nullptr && frames_ == frames) {
            return;
        }
        release_graph();
        ggml_init_params params{kGraphContextBytes, nullptr, true};
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize DotTTS AudioVAE streaming frontend graph context");
        }
        core::ModuleBuildContext build_ctx{ggml_, "dots_tts.audio_vae.stream.frontend", weights_->execution_context->backend_type()};
        const int64_t hidden_size = weights_->config.latent_dim * 4;
        auto input = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, weights_->config.latent_dim, frames}));
        auto state_hidden = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({static_cast<int64_t>(weights_->dec_mi_lstm.layers.size()), hidden_size}));
        auto state_cell = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({static_cast<int64_t>(weights_->dec_mi_lstm.layers.size()), hidden_size}));
        latents_ = input.tensor;
        hidden_ = state_hidden.tensor;
        cell_ = state_cell.tensor;
        hidden_inputs_.reserve(weights_->dec_mi_lstm.layers.size());
        cell_inputs_.reserve(weights_->dec_mi_lstm.layers.size());
        for (size_t layer = 0; layer < weights_->dec_mi_lstm.layers.size(); ++layer) {
            hidden_inputs_.push_back(SliceModule({0, static_cast<int64_t>(layer), 1}).build(build_ctx, state_hidden));
            cell_inputs_.push_back(SliceModule({0, static_cast<int64_t>(layer), 1}).build(build_ctx, state_cell));
        }
        auto output = build_decoder_stream_frontend_graph(build_ctx, *weights_, input, hidden_inputs_, cell_inputs_);
        decoder_input_ = output.decoder_input.tensor;
        for (const auto & hidden_output : output.hidden) {
            hidden_outputs_.push_back(hidden_output.tensor);
        }
        for (const auto & cell_output : output.cell) {
            cell_outputs_.push_back(cell_output.tensor);
        }
        graph_ = ggml_new_graph_custom(ggml_, 131072, false);
        ggml_set_output(decoder_input_);
        ggml_build_forward_expand(graph_, decoder_input_);
        for (auto * hidden_output : hidden_outputs_) {
            ggml_set_output(hidden_output);
            ggml_build_forward_expand(graph_, hidden_output);
        }
        for (auto * cell_output : cell_outputs_) {
            ggml_set_output(cell_output);
            ggml_build_forward_expand(graph_, cell_output);
        }
        core::validate_backend_graph_supported(weights_->execution_context->backend(), graph_, "dots_tts.audio_vae.stream.frontend");
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_->execution_context->backend()));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate DotTTS AudioVAE streaming frontend graph memory");
        }
        core::prepare_host_graph_plan(*weights_->execution_context, graph_, plan_);
        frames_ = frames;
    }

    std::shared_ptr<const DotsAudioVaeWeights> weights_;
    std::mutex mutex_;
    ggml_context * ggml_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    core::HostGraphPlan plan_;
    ggml_tensor * latents_ = nullptr;
    ggml_tensor * hidden_ = nullptr;
    ggml_tensor * cell_ = nullptr;
    ggml_tensor * decoder_input_ = nullptr;
    std::vector<core::TensorValue> hidden_inputs_;
    std::vector<core::TensorValue> cell_inputs_;
    std::vector<ggml_tensor *> hidden_outputs_;
    std::vector<ggml_tensor *> cell_outputs_;
    int64_t frames_ = 0;
};

void validate_config(const DotsVocoderConfig & config) {
    if (config.sample_rate <= 0 || config.latent_dim <= 0 || config.upsample_initial_channel <= 0 || config.mi_num_layers <= 0) {
        throw std::runtime_error("DotTTS AudioVAE config contains non-positive dimensions");
    }
    if (!config.causal || !config.causal_encoder) {
        throw std::runtime_error("DotTTS AudioVAE currently expects causal encoder and decoder checkpoints");
    }
    if (config.downsample_channels.size() != config.downsample_rates.size() + 1 ||
        config.upsample_rates.size() != config.upsample_kernel_sizes.size() ||
        config.resblock_kernel_sizes.size() != 3 ||
        config.resblock_dilation_sizes.size() != 3 ||
        config.activation != "snakebeta" ||
        config.resblock != "1") {
        throw std::runtime_error("DotTTS AudioVAE config is not supported by the native component");
    }
}

std::shared_ptr<DotsAudioVaeWeights> load_weights(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    DotsVocoderConfig config,
    assets::TensorStorageType weight_storage_type,
    assets::TensorStorageType conv_weight_storage_type) {
    validate_config(config);
    if (source == nullptr) {
        throw std::runtime_error("DotTTS AudioVAE requires tensor source");
    }
    auto weights = std::make_shared<DotsAudioVaeWeights>();
    weights->config = std::move(config);
    weights->execution_context = std::make_shared<core::ExecutionContext>(backend);
    weights->store = std::make_shared<core::BackendWeightStore>(
        weights->execution_context->backend(),
        weights->execution_context->backend_type(),
        "dots_tts.audio_vae.weights",
        kWeightContextBytes);
    for (const auto & tensor : source->tensors()) {
        weights->parameter_count += tensor_elements(tensor.shape);
        ++weights->loaded_tensor_count;
    }

    weights->encoder_convs.reserve(weights->config.downsample_rates.size() + 2);
    weights->encoder_convs.push_back(engine::modules::binding::conv1d_from_source_resolving_weight_norm(
        *weights->store,
        *source,
        "audio_encoder.generator.0.layer",
        conv_weight_storage_type,
        kEncoderBaseChannels,
        1,
        kProjectionKernel,
        true));
    int64_t generator_index = 2;
    for (size_t stage = 0; stage < weights->config.downsample_rates.size(); ++stage) {
        const int64_t in_channels = weights->config.downsample_channels[stage];
        const int64_t out_channels = weights->config.downsample_channels[stage + 1];
        const int64_t factor = weights->config.downsample_rates[stage];
        weights->encoder_convs.push_back(engine::modules::binding::conv1d_from_source_resolving_weight_norm(
            *weights->store,
            *source,
            "audio_encoder.generator." + std::to_string(generator_index) + ".layer",
            conv_weight_storage_type,
            out_channels,
            in_channels,
            factor * 2,
            true));
        weights->encoder_resstacks.push_back(load_resstack(
            *weights->store,
            *source,
            "audio_encoder.generator." + std::to_string(generator_index + 1),
            out_channels,
            conv_weight_storage_type));
        generator_index += 3;
    }
    weights->encoder_convs.push_back(engine::modules::binding::conv1d_from_source_resolving_weight_norm(
        *weights->store,
        *source,
        "audio_encoder.generator." + std::to_string(generator_index) + ".layer",
        conv_weight_storage_type,
        weights->config.latent_dim,
        weights->config.downsample_channels.back(),
        2 * kDefaultLookahead + 1,
        true));

    const int64_t mi_hidden = weights->config.latent_dim * 4;
    weights->enc_mi_in = engine::modules::binding::linear_from_source(
        *weights->store,
        *source,
        "enc_mi_layer.0",
        weight_storage_type,
        mi_hidden,
        weights->config.latent_dim,
        true);
    weights->enc_mi_lstm = engine::modules::binding::lstm_stack_from_source(
        *weights->store,
        *source,
        "enc_mi_layer.1.lstm",
        weights->config.mi_num_layers,
        mi_hidden,
        mi_hidden,
        weight_storage_type);
    weights->enc_mi_out = engine::modules::binding::linear_from_source(
        *weights->store,
        *source,
        "enc_mi_layer.2",
        weight_storage_type,
        weights->config.latent_dim,
        mi_hidden,
        true);
    weights->pre_proj = engine::modules::binding::conv1d_from_source(
        *weights->store,
        *source,
        "pre_proj",
        conv_weight_storage_type,
        weights->config.latent_dim * 2,
        weights->config.latent_dim,
        1,
        true);
    weights->post_proj = engine::modules::binding::conv1d_from_source(
        *weights->store,
        *source,
        "post_proj",
        conv_weight_storage_type,
        weights->config.latent_dim,
        weights->config.latent_dim,
        1,
        true);
    weights->dec_mi_in = engine::modules::binding::linear_from_source(
        *weights->store,
        *source,
        "dec_mi_layer.0",
        weight_storage_type,
        mi_hidden,
        weights->config.latent_dim,
        true);
    weights->dec_mi_lstm = engine::modules::binding::lstm_stack_from_source(
        *weights->store,
        *source,
        "dec_mi_layer.1.lstm",
        weights->config.mi_num_layers,
        mi_hidden,
        mi_hidden,
        weight_storage_type);
    weights->dec_mi_out = engine::modules::binding::linear_from_source(
        *weights->store,
        *source,
        "dec_mi_layer.2",
        weight_storage_type,
        weights->config.latent_dim,
        mi_hidden,
        true);
    weights->decoder_conv_pre = engine::modules::binding::conv1d_from_source_resolving_weight_norm(
        *weights->store,
        *source,
        "decoder.conv_pre",
        conv_weight_storage_type,
        weights->config.upsample_initial_channel,
        weights->config.latent_dim,
        2 * kDefaultLookahead + 1,
        true);

    for (size_t stage = 0; stage < weights->config.upsample_rates.size(); ++stage) {
        const int64_t in_channels = weights->config.upsample_initial_channel / (int64_t{1} << stage);
        const int64_t out_channels = weights->config.upsample_initial_channel / (int64_t{1} << (stage + 1));
        weights->decoder_ups.push_back(engine::modules::binding::conv_transpose1d_from_source_resolving_weight_norm(
            *weights->store,
            *source,
            "decoder.ups." + std::to_string(stage) + ".0",
            conv_weight_storage_type,
            in_channels,
            out_channels,
            weights->config.upsample_kernel_sizes[stage],
            true));
        for (size_t kernel_index = 0; kernel_index < weights->config.resblock_kernel_sizes.size(); ++kernel_index) {
            weights->decoder_resblocks.push_back(load_resblock(
                *weights->store,
                *source,
                "decoder.resblocks." + std::to_string(stage * weights->config.resblock_kernel_sizes.size() + kernel_index),
                out_channels,
                weights->config.resblock_kernel_sizes[kernel_index],
                conv_weight_storage_type));
        }
    }
    const int64_t post_channels = weights->config.upsample_initial_channel / (int64_t{1} << weights->config.upsample_rates.size());
    weights->decoder_activation_post = load_alias_free_activation(
        *weights->store,
        *source,
        "decoder.activation_post",
        post_channels);
    weights->decoder_conv_post = engine::modules::binding::conv1d_from_source_resolving_weight_norm(
        *weights->store,
        *source,
        "decoder.conv_post",
        conv_weight_storage_type,
        1,
        post_channels,
        7,
        weights->config.use_bias_at_final);

    weights->store->upload();
    source->release_storage();
    return weights;
}

int64_t audio_vae_hop_size(const DotsVocoderConfig & config) {
    int64_t hop = 1;
    for (const int64_t factor : config.downsample_rates) {
        hop *= factor;
    }
    return hop;
}

struct RationalFrames {
    int64_t num = 0;
    int64_t den = 1;

    void add(int64_t value_num, int64_t value_den) {
        const int64_t next_num = num * value_den + value_num * den;
        const int64_t next_den = den * value_den;
        const int64_t divisor = std::gcd(std::llabs(next_num), std::llabs(next_den));
        num = next_num / divisor;
        den = next_den / divisor;
    }

    void divide(int64_t value) {
        den *= value;
        const int64_t divisor = std::gcd(std::llabs(num), std::llabs(den));
        num /= divisor;
        den /= divisor;
    }

    int64_t ceil() const {
        return (num + den - 1) / den;
    }
};

int64_t activation_left_context() {
    const int64_t total_left = (kActivationKernel - 1) + (kActivationKernel - 1);
    return (total_left + kActivationRatio - 1) / kActivationRatio;
}

int64_t amp_block_left_context(int64_t kernel) {
    int64_t left = 0;
    const int64_t dilations[3] = {1, 3, 5};
    for (int64_t layer = 0; layer < 3; ++layer) {
        left += activation_left_context() + dilations[layer] * (kernel - 1);
        left += activation_left_context() + (kernel - 1);
    }
    return left;
}

int64_t decoder_stream_left_context(const DotsVocoderConfig & config) {
    RationalFrames left;
    left.add(kDefaultLookahead, 1);
    RationalFrames scale;
    scale.num = 1;
    scale.den = 1;
    for (size_t stage = 0; stage < config.upsample_rates.size(); ++stage) {
        left.add(scale.num, scale.den);
        scale.divide(config.upsample_rates[stage]);
        int64_t stage_context = 0;
        for (const int64_t kernel : config.resblock_kernel_sizes) {
            stage_context = std::max(stage_context, amp_block_left_context(kernel));
        }
        left.add(scale.num * stage_context, scale.den);
    }
    left.add(scale.num * activation_left_context(), scale.den);
    left.add(scale.num * 6, scale.den);
    return left.ceil();
}

int64_t decoder_stream_lookahead() {
    return kDefaultLookahead;
}

int64_t stream_window_size(const DotsVocoderConfig & config, int64_t chunk_frames) {
    if (chunk_frames <= 0) {
        throw std::runtime_error("DotTTS AudioVAE stream chunk_frames must be positive");
    }
    return chunk_frames + decoder_stream_lookahead() + decoder_stream_left_context(config);
}

void update_stream_window(
    std::vector<float> & window,
    const std::vector<float> & decoder_input,
    int64_t latent_dim,
    int64_t window_frames,
    int64_t valid_frames) {
    const int64_t chunk_frames = static_cast<int64_t>(decoder_input.size()) / latent_dim;
    if (chunk_frames >= window_frames) {
        throw std::runtime_error("DotTTS AudioVAE stream decoder window must be larger than chunk_frames");
    }
    const int64_t new_valid = std::min(valid_frames + chunk_frames, window_frames);
    const int64_t start = std::max<int64_t>(valid_frames + chunk_frames - window_frames, 0);
    std::vector<float> combined(static_cast<size_t>((window_frames + chunk_frames) * latent_dim), 0.0F);
    for (int64_t channel = 0; channel < latent_dim; ++channel) {
        std::copy(
            window.begin() + static_cast<std::ptrdiff_t>(channel * window_frames),
            window.begin() + static_cast<std::ptrdiff_t>((channel + 1) * window_frames),
            combined.begin() + static_cast<std::ptrdiff_t>(channel * (window_frames + chunk_frames)));
        std::copy(
            decoder_input.begin() + static_cast<std::ptrdiff_t>(channel * chunk_frames),
            decoder_input.begin() + static_cast<std::ptrdiff_t>((channel + 1) * chunk_frames),
            combined.begin() + static_cast<std::ptrdiff_t>(channel * (window_frames + chunk_frames) + valid_frames));
    }
    std::fill(window.begin(), window.end(), 0.0F);
    for (int64_t channel = 0; channel < latent_dim; ++channel) {
        std::copy(
            combined.begin() + static_cast<std::ptrdiff_t>(channel * (window_frames + chunk_frames) + start),
            combined.begin() + static_cast<std::ptrdiff_t>(channel * (window_frames + chunk_frames) + start + new_valid),
            window.begin() + static_cast<std::ptrdiff_t>(channel * window_frames));
    }
}

std::vector<float> slice_stream_audio(
    const std::vector<float> & audio_window,
    const DotsVocoderConfig & config,
    int64_t window_frames,
    int64_t & total_frames,
    int64_t & emitted_frames,
    int64_t appended_frames,
    bool final) {
    total_frames += appended_frames;
    const int64_t stable_end = final ? total_frames : std::max<int64_t>(0, total_frames - decoder_stream_lookahead());
    if (stable_end <= emitted_frames) {
        return {};
    }
    const int64_t valid_frames = std::min(total_frames, window_frames);
    const int64_t window_start = total_frames - valid_frames;
    if (emitted_frames < window_start) {
        throw std::runtime_error("DotTTS AudioVAE stream decoder window is too short for fixed-graph decoding");
    }
    const int64_t local_start = emitted_frames - window_start;
    const int64_t local_end = stable_end - window_start;
    const int64_t sample_start = local_start * audio_vae_hop_size(config);
    const int64_t sample_end = local_end * audio_vae_hop_size(config);
    if (sample_start < 0 || sample_end < sample_start || sample_end > static_cast<int64_t>(audio_window.size())) {
        throw std::runtime_error("DotTTS AudioVAE stream output slice is outside the decoder window");
    }
    emitted_frames = stable_end;
    return std::vector<float>(
        audio_window.begin() + static_cast<std::ptrdiff_t>(sample_start),
        audio_window.begin() + static_cast<std::ptrdiff_t>(sample_end));
}

}  // namespace

struct DotsAudioVaeStreamState::Impl {
    int64_t chunk_frames = 0;
    int64_t window_frames = 0;
    int64_t total_frames = 0;
    int64_t emitted_frames = 0;
    std::vector<float> hidden;
    std::vector<float> cell;
    std::vector<float> decoder_window;
};

DotsAudioVaeStreamState::DotsAudioVaeStreamState() : impl_(std::make_unique<Impl>()) {}
DotsAudioVaeStreamState::~DotsAudioVaeStreamState() = default;
DotsAudioVaeStreamState::DotsAudioVaeStreamState(DotsAudioVaeStreamState &&) noexcept = default;
DotsAudioVaeStreamState & DotsAudioVaeStreamState::operator=(DotsAudioVaeStreamState &&) noexcept = default;

struct DotsAudioVaeComponent::Impl {
    std::shared_ptr<const DotsAudioVaeWeights> weights;
    std::unique_ptr<EncoderRunner> encoder;
    std::unique_ptr<DecoderRunner> decoder;
    std::unique_ptr<StreamFrontendRunner> stream_frontend;
    std::unique_ptr<DecoderWindowRunner> stream_decoder;
};

DotsAudioVaeComponent DotsAudioVaeComponent::load_from_tensor_source(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    DotsVocoderConfig config,
    assets::TensorStorageType weight_storage_type,
    assets::TensorStorageType conv_weight_storage_type) {
    DotsAudioVaeComponent component;
    component.impl_ = std::make_unique<Impl>();
    component.impl_->weights = load_weights(
        std::move(source),
        backend,
        std::move(config),
        weight_storage_type,
        conv_weight_storage_type);
    component.impl_->encoder = std::make_unique<EncoderRunner>(component.impl_->weights);
    component.impl_->decoder = std::make_unique<DecoderRunner>(component.impl_->weights);
    component.impl_->stream_frontend = std::make_unique<StreamFrontendRunner>(component.impl_->weights);
    component.impl_->stream_decoder = std::make_unique<DecoderWindowRunner>(component.impl_->weights);
    return component;
}

DotsAudioVaeComponent::DotsAudioVaeComponent() = default;
DotsAudioVaeComponent::DotsAudioVaeComponent(DotsAudioVaeComponent &&) noexcept = default;
DotsAudioVaeComponent & DotsAudioVaeComponent::operator=(DotsAudioVaeComponent &&) noexcept = default;
DotsAudioVaeComponent::~DotsAudioVaeComponent() = default;

int64_t DotsAudioVaeComponent::sample_rate() const noexcept {
    return impl_ == nullptr || impl_->weights == nullptr ? 0 : impl_->weights->config.sample_rate;
}

int64_t DotsAudioVaeComponent::hop_size() const noexcept {
    if (impl_ == nullptr || impl_->weights == nullptr) {
        return 0;
    }
    return audio_vae_hop_size(impl_->weights->config);
}

bool DotsAudioVaeComponent::is_loaded() const noexcept {
    return impl_ != nullptr && impl_->weights != nullptr;
}

DotsEncoderLatents DotsAudioVaeComponent::extract_latents(const std::vector<float> & waveform) const {
    if (impl_ == nullptr || impl_->encoder == nullptr) {
        throw std::runtime_error("DotTTS AudioVAE component is not initialized");
    }
    return impl_->encoder->run(waveform);
}

DotsDecodedAudio DotsAudioVaeComponent::decode_latents(const std::vector<float> & latents, int64_t frames) const {
    if (impl_ == nullptr || impl_->decoder == nullptr) {
        throw std::runtime_error("DotTTS AudioVAE component is not initialized");
    }
    auto channel_major = frame_major_to_channel_major(latents, frames, impl_->weights->config.latent_dim);
    return impl_->decoder->run(
        channel_major,
        frames);
}

DotsAudioVaeStreamState DotsAudioVaeComponent::create_stream_state(int64_t chunk_frames) const {
    if (impl_ == nullptr || impl_->weights == nullptr) {
        throw std::runtime_error("DotTTS AudioVAE component is not initialized");
    }
    if (chunk_frames <= 0) {
        throw std::runtime_error("DotTTS AudioVAE stream chunk_frames must be positive");
    }
    DotsAudioVaeStreamState state;
    state.impl_->chunk_frames = chunk_frames;
    state.impl_->window_frames = stream_window_size(impl_->weights->config, chunk_frames);
    const int64_t hidden_size = impl_->weights->config.latent_dim * 4;
    const int64_t layers = static_cast<int64_t>(impl_->weights->dec_mi_lstm.layers.size());
    state.impl_->hidden.assign(static_cast<size_t>(layers * hidden_size), 0.0F);
    state.impl_->cell.assign(static_cast<size_t>(layers * hidden_size), 0.0F);
    state.impl_->decoder_window.assign(static_cast<size_t>(impl_->weights->config.latent_dim * state.impl_->window_frames), 0.0F);
    return state;
}

DotsDecodedAudio DotsAudioVaeComponent::stream_step(
    const std::vector<float> & latents,
    int64_t frames,
    DotsAudioVaeStreamState & state) const {
    if (impl_ == nullptr || impl_->weights == nullptr || impl_->stream_frontend == nullptr || impl_->stream_decoder == nullptr || state.impl_ == nullptr) {
        throw std::runtime_error("DotTTS AudioVAE stream state is not initialized");
    }
    if (frames <= 0 || static_cast<int64_t>(latents.size()) != impl_->weights->config.latent_dim * frames) {
        throw std::runtime_error("DotTTS AudioVAE stream latent size mismatch");
    }
    const int64_t valid_frames = std::min(state.impl_->total_frames, state.impl_->window_frames);
    auto frontend = impl_->stream_frontend->run(
        frame_major_to_channel_major(latents, frames, impl_->weights->config.latent_dim),
        frames,
        state.impl_->hidden,
        state.impl_->cell);
    const int64_t hidden_size = impl_->weights->config.latent_dim * 4;
    for (size_t layer = 0; layer < frontend.hidden.size(); ++layer) {
        std::copy(
            frontend.hidden[layer].begin(),
            frontend.hidden[layer].end(),
            state.impl_->hidden.begin() + static_cast<std::ptrdiff_t>(layer * hidden_size));
        std::copy(
            frontend.cell[layer].begin(),
            frontend.cell[layer].end(),
            state.impl_->cell.begin() + static_cast<std::ptrdiff_t>(layer * hidden_size));
    }
    update_stream_window(
        state.impl_->decoder_window,
        frontend.decoder_input,
        impl_->weights->config.latent_dim,
        state.impl_->window_frames,
        valid_frames);
    auto window_audio = impl_->stream_decoder->run(state.impl_->decoder_window, state.impl_->window_frames);
    window_audio.samples = slice_stream_audio(
        window_audio.samples,
        impl_->weights->config,
        state.impl_->window_frames,
        state.impl_->total_frames,
        state.impl_->emitted_frames,
        frames,
        false);
    return window_audio;
}

DotsDecodedAudio DotsAudioVaeComponent::flush_stream(DotsAudioVaeStreamState & state) const {
    if (impl_ == nullptr || impl_->weights == nullptr || impl_->stream_decoder == nullptr || state.impl_ == nullptr) {
        throw std::runtime_error("DotTTS AudioVAE stream state is not initialized");
    }
    auto window_audio = impl_->stream_decoder->run(state.impl_->decoder_window, state.impl_->window_frames);
    window_audio.samples = slice_stream_audio(
        window_audio.samples,
        impl_->weights->config,
        state.impl_->window_frames,
        state.impl_->total_frames,
        state.impl_->emitted_frames,
        0,
        true);
    return window_audio;
}

void DotsAudioVaeComponent::release_runtime_graphs() {
    if (impl_ == nullptr) {
        return;
    }
    if (impl_->encoder != nullptr) {
        impl_->encoder->release_graph();
    }
    if (impl_->decoder != nullptr) {
        impl_->decoder->release_graph();
    }
    if (impl_->stream_frontend != nullptr) {
        impl_->stream_frontend->release_graph();
    }
    if (impl_->stream_decoder != nullptr) {
        impl_->stream_decoder->release_graph();
    }
}

}  // namespace engine::models::dots_tts
