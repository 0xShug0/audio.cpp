#include "engine/models/fish_audio/codec.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention_modules.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include "../common/constant_tensor_cache.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace engine::models::fish_audio {
namespace {

namespace binding = engine::modules::binding;

constexpr int64_t kCodecDim = 1024;
constexpr int64_t kCodecTransformerHeads = 16;
constexpr int64_t kCodecHeadDim = 64;
constexpr int64_t kCodecIntermediate = 3072;
constexpr int64_t kCodecTransformerLayers = 8;
constexpr float kCodecNormEps = 1.0e-5F;
constexpr float kConvNextNormEps = 1.0e-6F;
constexpr float kCodecRopeTheta = 10000.0F;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept {
        if (alloc != nullptr) {
            ggml_gallocr_free(alloc);
        }
    }
};

std::vector<int64_t> dims_vector(const core::TensorShape & shape) {
    std::vector<int64_t> out;
    out.reserve(shape.rank);
    for (size_t i = 0; i < shape.rank; ++i) {
        out.push_back(shape.dims[i]);
    }
    return out;
}

std::vector<float> prepare_codec_mono(
    const runtime::AudioBuffer & audio,
    int target_sample_rate_hz) {
    auto mono = engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    if (audio.sample_rate != target_sample_rate_hz) {
        mono = engine::audio::resample_mono_torchaudio_sinc_hann(
            mono,
            audio.sample_rate,
            target_sample_rate_hz);
    }
    return mono;
}

struct CodecTransformerLayerWeights {
    modules::NormWeights attention_norm;
    modules::AttentionWeights attention;
    modules::LayerScaleWeights attention_scale;
    modules::NormWeights ffn_norm;
    modules::QwenMLPWeights feed_forward;
    modules::LayerScaleWeights ffn_scale;
};

struct CodecTransformerWeights {
    std::vector<CodecTransformerLayerWeights> layers;
    modules::NormWeights norm;
};

struct ResidualUnitWeights {
    modules::Snake1dWeights snake1;
    modules::Conv1dWeights conv1;
    modules::Snake1dWeights snake2;
    modules::Conv1dWeights conv2;
};

struct EncoderBlockWeights {
    ResidualUnitWeights residual1;
    ResidualUnitWeights residual3;
    ResidualUnitWeights residual9;
    modules::Snake1dWeights snake;
    modules::Conv1dWeights conv;
    std::optional<CodecTransformerWeights> transformer;
};

struct DecoderBlockWeights {
    modules::Snake1dWeights snake;
    modules::ConvTranspose1dWeights conv;
    ResidualUnitWeights residual1;
    ResidualUnitWeights residual3;
    ResidualUnitWeights residual9;
};

struct ConvNeXtBlockWeights {
    modules::DepthwiseConv1dWeights dwconv;
    modules::NormWeights norm;
    modules::LinearWeights pwconv1;
    modules::LinearWeights pwconv2;
    modules::LayerScaleWeights gamma;
};

struct QuantizerUnitWeights {
    modules::Conv1dWeights in_proj;
    modules::Conv1dWeights out_proj;
    core::TensorValue codebook;
    core::TensorValue normalized_codebook;
};

struct FishCodecWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::Conv1dWeights encoder_first;
    std::vector<EncoderBlockWeights> encoder_blocks;
    modules::Snake1dWeights encoder_final_snake;
    modules::Conv1dWeights encoder_final;

    std::vector<std::pair<modules::Conv1dWeights, ConvNeXtBlockWeights>> downsample;
    CodecTransformerWeights pre_module;
    QuantizerUnitWeights semantic_quantizer;
    std::vector<QuantizerUnitWeights> residual_quantizers;
    CodecTransformerWeights post_module;
    std::vector<std::pair<modules::ConvTranspose1dWeights, ConvNeXtBlockWeights>> upsample;

    modules::Conv1dWeights decoder_first;
    std::vector<DecoderBlockWeights> decoder_blocks;
    modules::Snake1dWeights decoder_final_snake;
    modules::Conv1dWeights decoder_final;
};

int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

std::vector<float> normalized_rows(const std::vector<float> & values, int64_t rows, int64_t cols) {
    if (static_cast<int64_t>(values.size()) != rows * cols) {
        throw std::runtime_error("Fish Audio normalized_rows shape mismatch");
    }
    std::vector<float> out(values.size(), 0.0F);
    for (int64_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (int64_t col = 0; col < cols; ++col) {
            const float value = values[static_cast<size_t>(row * cols + col)];
            sum += static_cast<double>(value) * static_cast<double>(value);
        }
        const float inv = sum > 0.0 ? static_cast<float>(1.0 / std::sqrt(sum)) : 0.0F;
        for (int64_t col = 0; col < cols; ++col) {
            const size_t index = static_cast<size_t>(row * cols + col);
            out[index] = values[index] * inv;
        }
    }
    return out;
}

core::TensorValue slice_frames(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t start, int64_t frames) {
    if (frames <= 0) {
        throw std::runtime_error("Fish Audio codec slice_frames requires positive frames");
    }
    return modules::SliceModule({2, start, frames}).build(ctx, input);
}

core::TensorValue zero_prefix_like(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t frames) {
    if (frames <= 0) {
        throw std::runtime_error("Fish Audio zero_prefix_like requires positive frames");
    }
    auto first = modules::SliceModule({2, 0, 1}).build(ctx, input);
    if (ctx.backend_type == core::BackendType::Cpu) {
        first = core::ensure_backend_addressable_layout(ctx, first);
    }
    first = core::wrap_tensor(ggml_scale(ctx.ggml, first.tensor, 0.0F), first.shape, GGML_TYPE_F32);
    return modules::RepeatModule({core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], frames})})
        .build(ctx, first);
}

core::TensorValue zero_suffix_like(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t frames) {
    if (frames <= 0) {
        throw std::runtime_error("Fish Audio zero_suffix_like requires positive frames");
    }
    auto last = modules::SliceModule({2, input.shape.dims[2] - 1, 1}).build(ctx, input);
    if (ctx.backend_type == core::BackendType::Cpu) {
        last = core::ensure_backend_addressable_layout(ctx, last);
    }
    last = core::wrap_tensor(ggml_scale(ctx.ggml, last.tensor, 0.0F), last.shape, GGML_TYPE_F32);
    return modules::RepeatModule({core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], frames})})
        .build(ctx, last);
}

int64_t extra_padding_for_conv1d(int64_t frames, int64_t effective_kernel, int64_t stride, int64_t left_pad) {
    const double n_frames = (static_cast<double>(frames - effective_kernel + left_pad) / static_cast<double>(stride)) + 1.0;
    const int64_t ideal_length =
        (static_cast<int64_t>(std::ceil(n_frames)) - 1) * stride + (effective_kernel - left_pad);
    return ideal_length - frames;
}

core::TensorValue causal_pad(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t left_pad, int64_t right_pad) {
    if (left_pad < 0) {
        throw std::runtime_error("Fish Audio causal conv requires non-negative left padding");
    }
    if (right_pad < 0) {
        throw std::runtime_error("Fish Audio causal conv requires non-negative right padding");
    }
    if (left_pad == 0 && right_pad == 0) {
        return input;
    }
    auto out = input;
    if (left_pad > 0) {
        out = modules::ConcatModule({2}).build(ctx, zero_prefix_like(ctx, input, left_pad), out);
    }
    if (right_pad > 0) {
        out = modules::ConcatModule({2}).build(ctx, out, zero_suffix_like(ctx, input, right_pad));
    }
    return out;
}

core::TensorValue causal_conv1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::Conv1dWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int stride,
    int dilation,
    bool use_bias) {
    const int64_t effective_kernel = (kernel - 1) * dilation + 1;
    const int64_t left_pad = effective_kernel - stride;
    const int64_t right_pad = extra_padding_for_conv1d(input.shape.dims[2], effective_kernel, stride, left_pad);
    auto padded = causal_pad(ctx, input, left_pad, right_pad);
    return modules::Conv1dModule({
        in_channels,
        out_channels,
        kernel,
        stride,
        0,
        dilation,
        use_bias,
    }).build(ctx, padded, weights);
}

core::TensorValue causal_depthwise_conv1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::DepthwiseConv1dWeights & weights,
    int64_t channels,
    int64_t kernel,
    int stride,
    int dilation,
    bool use_bias) {
    const int64_t effective_kernel = (kernel - 1) * dilation + 1;
    const int64_t left_pad = effective_kernel - stride;
    const int64_t right_pad = extra_padding_for_conv1d(input.shape.dims[2], effective_kernel, stride, left_pad);
    auto padded = causal_pad(ctx, input, left_pad, right_pad);
    return modules::DepthwiseConv1dModule({
        channels,
        kernel,
        stride,
        0,
        dilation,
        use_bias,
    }).build(ctx, padded, weights);
}

core::TensorValue causal_conv_transpose1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::ConvTranspose1dWeights & weights,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel,
    int stride,
    bool use_bias) {
    auto out = modules::ConvTranspose1dModule({
        in_channels,
        out_channels,
        kernel,
        stride,
        0,
        1,
        use_bias,
    }).build(ctx, input, weights);
    const int64_t pad = kernel - stride;
    const int64_t padding_right = static_cast<int64_t>(std::ceil(static_cast<double>(pad)));
    const int64_t padding_left = pad - padding_right;
    return slice_frames(ctx, out, padding_left, out.shape.dims[2] - padding_left - padding_right);
}

core::TensorValue l2_normalize_last(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    auto squared = modules::MulModule{}.build(ctx, input, input);
    auto sum = modules::ReduceSumModule({static_cast<int>(input.shape.rank - 1)}).build(ctx, squared);
    auto shifted = core::wrap_tensor(ggml_scale_bias(ctx.ggml, sum.tensor, 1.0F, 1.0e-12F), sum.shape, GGML_TYPE_F32);
    auto denom = modules::SqrtModule{}.build(ctx, shifted);
    auto repeated = modules::RepeatModule({input.shape}).build(ctx, denom);
    return core::wrap_tensor(ggml_div(ctx.ggml, input.tensor, repeated.tensor), input.shape, GGML_TYPE_F32);
}

core::TensorValue build_mlp(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::QwenMLPWeights & weights) {
    auto gate = modules::LinearModule({kCodecDim, kCodecIntermediate, false, GGML_PREC_F32})
                    .build(ctx, input, weights.gate_proj);
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule({kCodecDim, kCodecIntermediate, false, GGML_PREC_F32})
                  .build(ctx, input, weights.up_proj);
    auto hidden = modules::MulModule{}.build(ctx, gate, up);
    return modules::LinearModule({kCodecIntermediate, kCodecDim, false, GGML_PREC_F32})
        .build(ctx, hidden, weights.down_proj);
}

core::TensorValue reshape_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], kCodecTransformerHeads, kCodecHeadDim}));
}

core::TensorValue attention_from_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    const core::TensorValue & attention_mask) {
    auto q = modules::TransposeModule({{0, 2, 1, 3}, q_heads.shape.rank}).build(ctx, q_heads);
    auto k = modules::TransposeModule({{0, 2, 1, 3}, k_heads.shape.rank}).build(ctx, k_heads);
    auto v = modules::TransposeModule({{0, 2, 1, 3}, v_heads.shape.rank}).build(ctx, v_heads);
    q = core::wrap_tensor(ggml_cont(ctx.ggml, q.tensor), q.shape, q.type);
    k = core::wrap_tensor(ggml_cont(ctx.ggml, k.tensor), k.shape, k.type);
    v = core::wrap_tensor(ggml_cont(ctx.ggml, v.tensor), v.shape, v.type);
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q.tensor,
        k.tensor,
        v.tensor,
        attention_mask.tensor,
        1.0F / std::sqrt(static_cast<float>(kCodecHeadDim)),
        0.0F,
        0.0F);
    ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
    return core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[2], q.shape.dims[1], kCodecHeadDim}),
        GGML_TYPE_F32);
}

core::TensorValue build_transformer_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue & attention_mask,
    const CodecTransformerLayerWeights & weights) {
    auto normed = modules::RMSNormModule({kCodecDim, kCodecNormEps, true, false}).build(ctx, input, weights.attention_norm);
    auto q = modules::LinearModule({kCodecDim, kCodecDim, false, GGML_PREC_F32})
                 .build(ctx, normed, {weights.attention.q_weight, std::nullopt});
    auto k = modules::LinearModule({kCodecDim, kCodecDim, false, GGML_PREC_F32})
                 .build(ctx, normed, {weights.attention.k_weight, std::nullopt});
    auto v = modules::LinearModule({kCodecDim, kCodecDim, false, GGML_PREC_F32})
                 .build(ctx, normed, {weights.attention.v_weight, std::nullopt});
    q = modules::RoPEModule({kCodecHeadDim, GGML_ROPE_TYPE_NORMAL, kCodecRopeTheta}).build(ctx, reshape_heads(ctx, q), positions);
    k = modules::RoPEModule({kCodecHeadDim, GGML_ROPE_TYPE_NORMAL, kCodecRopeTheta}).build(ctx, reshape_heads(ctx, k), positions);
    v = reshape_heads(ctx, v);
    auto context = attention_from_heads(ctx, q, k, v, attention_mask);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], kCodecDim}));
    auto attn = modules::LinearModule({kCodecDim, kCodecDim, false, GGML_PREC_F32})
                    .build(ctx, context, {weights.attention.out_weight, std::nullopt});
    attn = modules::LayerScaleModule{}.build(ctx, attn, weights.attention_scale);
    auto hidden = modules::AddModule{}.build(ctx, input, attn);
    auto ffn_in = modules::RMSNormModule({kCodecDim, kCodecNormEps, true, false}).build(ctx, hidden, weights.ffn_norm);
    auto ff = build_mlp(ctx, ffn_in, weights.feed_forward);
    ff = modules::LayerScaleModule{}.build(ctx, ff, weights.ffn_scale);
    return modules::AddModule{}.build(ctx, hidden, ff);
}

core::TensorValue make_positions(
    core::ModuleBuildContext &,
    common::ConstantTensorCache & constants,
    int64_t frames) {
    std::vector<int32_t> values(static_cast<size_t>(frames));
    for (int64_t i = 0; i < frames; ++i) {
        values[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    return constants.make_tensor(core::TensorShape::from_dims({frames}), GGML_TYPE_I32, values.data(), values.size() * sizeof(int32_t));
}

core::TensorValue make_causal_mask(
    core::ModuleBuildContext &,
    common::ConstantTensorCache & constants,
    int64_t frames,
    int64_t window_size) {
    std::vector<ggml_fp16_t> values(static_cast<size_t>(frames * frames), ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
    for (int64_t row = 0; row < frames; ++row) {
        const int64_t begin = window_size > 0 ? std::max<int64_t>(0, row - window_size + 1) : 0;
        for (int64_t col = begin; col <= row; ++col) {
            values[static_cast<size_t>(row * frames + col)] = ggml_fp32_to_fp16(0.0F);
        }
    }
    return constants.make_tensor(core::TensorShape::from_dims({frames, frames}), GGML_TYPE_F16, values.data(), values.size() * sizeof(ggml_fp16_t));
}

core::TensorValue build_window_transformer(
    core::ModuleBuildContext & ctx,
    common::ConstantTensorCache & constants,
    const core::TensorValue & input_bct,
    const CodecTransformerWeights & weights,
    int64_t window_size) {
    auto x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input_bct);
    auto positions = make_positions(ctx, constants, x.shape.dims[1]);
    auto mask = make_causal_mask(ctx, constants, x.shape.dims[1], window_size);
    for (const auto & layer : weights.layers) {
        x = build_transformer_layer(ctx, x, positions, mask, layer);
    }
    x = modules::RMSNormModule({kCodecDim, kCodecNormEps, true, false}).build(ctx, x, weights.norm);
    return modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
}

core::TensorValue build_residual_unit(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ResidualUnitWeights & weights,
    int64_t channels,
    int dilation) {
    auto y = modules::Snake1dModule({channels}).build(ctx, input, weights.snake1);
    y = causal_conv1d(ctx, y, weights.conv1, channels, channels, 7, 1, dilation, true);
    y = modules::Snake1dModule({channels}).build(ctx, y, weights.snake2);
    y = causal_conv1d(ctx, y, weights.conv2, channels, channels, 1, 1, 1, true);
    core::TensorValue x = input;
    if (x.shape.dims[2] != y.shape.dims[2]) {
        x = slice_frames(ctx, x, 0, y.shape.dims[2]);
    }
    return modules::AddModule{}.build(ctx, x, y);
}

core::TensorValue build_convnext(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ConvNeXtBlockWeights & weights,
    int64_t channels) {
    auto y = causal_depthwise_conv1d(ctx, input, weights.dwconv, channels, 7, 1, 1, true);
    y = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, y);
    y = modules::LayerNormModule({channels, kConvNextNormEps, true, true}).build(ctx, y, weights.norm);
    y = modules::LinearModule({channels, channels * 4, true, GGML_PREC_F32}).build(ctx, y, weights.pwconv1);
    y = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, y);
    y = modules::LinearModule({channels * 4, channels, true, GGML_PREC_F32}).build(ctx, y, weights.pwconv2);
    y = modules::LayerScaleModule{}.build(ctx, y, weights.gamma);
    y = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, y);
    core::TensorValue x = input;
    if (x.shape.dims[2] != y.shape.dims[2]) {
        x = slice_frames(ctx, x, 0, y.shape.dims[2]);
    }
    return modules::AddModule{}.build(ctx, x, y);
}

core::TensorValue build_encoder(
    core::ModuleBuildContext & ctx,
    common::ConstantTensorCache & constants,
    const core::TensorValue & input,
    const FishCodecWeights & weights) {
    auto x = causal_conv1d(ctx, input, weights.encoder_first, 1, 64, 7, 1, 1, true);
    int64_t channels = 64;
    const int strides[] = {2, 4, 8, 8};
    for (size_t index = 0; index < weights.encoder_blocks.size(); ++index) {
        const auto & block = weights.encoder_blocks[index];
        x = build_residual_unit(ctx, x, block.residual1, channels, 1);
        x = build_residual_unit(ctx, x, block.residual3, channels, 3);
        x = build_residual_unit(ctx, x, block.residual9, channels, 9);
        x = modules::Snake1dModule({channels}).build(ctx, x, block.snake);
        x = causal_conv1d(ctx, x, block.conv, channels, channels * 2, 2 * strides[index], strides[index], 1, true);
        channels *= 2;
        if (block.transformer.has_value()) {
            x = build_window_transformer(ctx, constants, x, *block.transformer, 512);
        }
    }
    x = modules::Snake1dModule({channels}).build(ctx, x, weights.encoder_final_snake);
    return causal_conv1d(ctx, x, weights.encoder_final, channels, kCodecDim, 3, 1, 1, true);
}

core::TensorValue build_decoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const FishCodecWeights & weights) {
    auto x = causal_conv1d(ctx, input, weights.decoder_first, kCodecDim, 1536, 7, 1, 1, true);
    int64_t channels = 1536;
    const int strides[] = {8, 8, 4, 2};
    for (size_t index = 0; index < weights.decoder_blocks.size(); ++index) {
        const auto & block = weights.decoder_blocks[index];
        x = modules::Snake1dModule({channels}).build(ctx, x, block.snake);
        x = causal_conv_transpose1d(ctx, x, block.conv, channels, channels / 2, 2 * strides[index], strides[index], true);
        channels /= 2;
        x = build_residual_unit(ctx, x, block.residual1, channels, 1);
        x = build_residual_unit(ctx, x, block.residual3, channels, 3);
        x = build_residual_unit(ctx, x, block.residual9, channels, 9);
    }
    x = modules::Snake1dModule({channels}).build(ctx, x, weights.decoder_final_snake);
    x = causal_conv1d(ctx, x, weights.decoder_final, channels, 1, 7, 1, 1, true);
    return modules::TanhModule{}.build(ctx, x);
}

core::TensorValue build_quantizer_out(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & ids_bt,
    const QuantizerUnitWeights & weights,
    int64_t codebook_size) {
    auto emb_btd = modules::CodebookLookupModule({codebook_size, 8}).build(ctx, ids_bt, weights.codebook);
    auto emb_bdt = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, emb_btd);
    return modules::Conv1dModule({8, kCodecDim, 1, 1, 0, 1, true}).build(ctx, emb_bdt, weights.out_proj);
}

core::TensorValue build_decode_quantizer(
    core::ModuleBuildContext & ctx,
    common::ConstantTensorCache & constants,
    const std::vector<core::TensorValue> & code_inputs,
    const FishCodecWeights & weights) {
    auto latent = build_quantizer_out(ctx, code_inputs[0], weights.semantic_quantizer, 4096);
    for (size_t index = 0; index < weights.residual_quantizers.size(); ++index) {
        auto residual = build_quantizer_out(ctx, code_inputs[index + 1], weights.residual_quantizers[index], 1024);
        latent = modules::AddModule{}.build(ctx, latent, residual);
    }
    latent = build_window_transformer(ctx, constants, latent, weights.post_module, 128);
    for (const auto & stage : weights.upsample) {
        latent = causal_conv_transpose1d(ctx, latent, stage.first, kCodecDim, kCodecDim, 2, 2, true);
        latent = build_convnext(ctx, latent, stage.second, kCodecDim);
    }
    return latent;
}

core::TensorValue build_encode_quantizer(
    core::ModuleBuildContext & ctx,
    common::ConstantTensorCache & constants,
    const core::TensorValue & encoder_latent,
    const FishCodecWeights & weights,
    std::vector<ggml_tensor *> & code_outputs,
    std::vector<std::pair<std::string, core::TensorValue>> & trace_outputs) {
    auto x = encoder_latent;
    for (const auto & stage : weights.downsample) {
        x = causal_conv1d(ctx, x, stage.first, kCodecDim, kCodecDim, 2, 2, 1, true);
        x = build_convnext(ctx, x, stage.second, kCodecDim);
    }
    trace_outputs.push_back({"fish_audio.codec.after_downsample", x});
    x = build_window_transformer(ctx, constants, x, weights.pre_module, 128);
    trace_outputs.push_back({"fish_audio.codec.after_pre_module", x});

    auto residual = x;
    auto quantize_one = [&](const QuantizerUnitWeights & quantizer, int64_t codebook_size) {
        auto projected = modules::Conv1dModule({kCodecDim, 8, 1, 1, 0, 1, true}).build(ctx, residual, quantizer.in_proj);
        auto projected_btd = l2_normalize_last(ctx, modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, projected));
        auto logits = modules::LinearModule({8, codebook_size, false, GGML_PREC_F32})
                          .build(ctx, projected_btd, {quantizer.normalized_codebook, std::nullopt});
        auto flat_logits = core::reshape_tensor(
            ctx,
            core::ensure_backend_addressable_layout(ctx, logits),
            core::TensorShape::from_dims({logits.shape.dims[1], codebook_size}));
        auto * ids_raw = ggml_argmax(ctx.ggml, flat_logits.tensor);
        ggml_set_output(ids_raw);
        code_outputs.push_back(ids_raw);
        auto ids = core::reshape_tensor(
            ctx,
            core::wrap_tensor(ids_raw, core::TensorShape::from_dims({logits.shape.dims[1]}), GGML_TYPE_I32),
            core::TensorShape::from_dims({1, logits.shape.dims[1]}));
        auto quantized = build_quantizer_out(ctx, ids, quantizer, codebook_size);
        residual = core::wrap_tensor(ggml_sub(ctx.ggml, residual.tensor, quantized.tensor), residual.shape, GGML_TYPE_F32);
    };
    quantize_one(weights.semantic_quantizer, 4096);
    for (const auto & quantizer : weights.residual_quantizers) {
        quantize_one(quantizer, 1024);
    }
    return x;
}

modules::Snake1dWeights load_snake(core::BackendWeightStore & store, const assets::TensorSource & source, const std::string & name, int64_t channels) {
    return {store.make_f32(
        core::TensorShape::from_dims({channels}),
        source.require_f32(name + ".alpha", {1, channels, 1}))};
}

CodecTransformerWeights load_transformer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t layers) {
    CodecTransformerWeights out;
    out.layers.reserve(static_cast<size_t>(layers));
    for (int64_t layer = 0; layer < layers; ++layer) {
        const std::string layer_prefix = prefix + ".layers." + std::to_string(layer);
        CodecTransformerLayerWeights weights;
        weights.attention_norm = binding::norm_weight_from_source(store, source, layer_prefix + ".attention_norm", kCodecDim);
        weights.attention.q_weight = store.load_tensor(source, layer_prefix + ".attention.q_proj.weight", storage_type, {kCodecDim, kCodecDim});
        weights.attention.k_weight = store.load_tensor(source, layer_prefix + ".attention.k_proj.weight", storage_type, {kCodecDim, kCodecDim});
        weights.attention.v_weight = store.load_tensor(source, layer_prefix + ".attention.v_proj.weight", storage_type, {kCodecDim, kCodecDim});
        weights.attention.out_weight = store.load_tensor(source, layer_prefix + ".attention.wo.weight", storage_type, {kCodecDim, kCodecDim});
        weights.attention_scale = binding::layer_scale_from_named_source(store, source, layer_prefix + ".attention_layer_scale.gamma");
        weights.ffn_norm = binding::norm_weight_from_source(store, source, layer_prefix + ".ffn_norm", kCodecDim);
        weights.feed_forward.gate_proj.weight = store.load_tensor(source, layer_prefix + ".feed_forward.w1.weight", storage_type, {kCodecIntermediate, kCodecDim});
        weights.feed_forward.down_proj.weight = store.load_tensor(source, layer_prefix + ".feed_forward.w2.weight", storage_type, {kCodecDim, kCodecIntermediate});
        weights.feed_forward.up_proj.weight = store.load_tensor(source, layer_prefix + ".feed_forward.w3.weight", storage_type, {kCodecIntermediate, kCodecDim});
        weights.ffn_scale = binding::layer_scale_from_named_source(store, source, layer_prefix + ".ffn_layer_scale.gamma");
        out.layers.push_back(std::move(weights));
    }
    out.norm = binding::norm_weight_from_source(store, source, prefix + ".norm", kCodecDim);
    return out;
}

ResidualUnitWeights load_residual_unit(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t channels) {
    ResidualUnitWeights out;
    out.snake1 = load_snake(store, source, prefix + ".block.0", channels);
    out.conv1 = binding::conv1d_from_named_source(
        store,
        source,
        prefix + ".block.1.conv.weight",
        prefix + ".block.1.conv.bias",
        storage_type);
    out.snake2 = load_snake(store, source, prefix + ".block.2", channels);
    out.conv2 = binding::conv1d_from_named_source(
        store,
        source,
        prefix + ".block.3.conv.weight",
        prefix + ".block.3.conv.bias",
        storage_type);
    return out;
}

ConvNeXtBlockWeights load_convnext(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t channels) {
    ConvNeXtBlockWeights out;
    out.dwconv = binding::depthwise_conv1d_from_source(
        store,
        source,
        prefix + ".dwconv.conv",
        storage_type,
        channels,
        7,
        true);
    out.norm = binding::norm_from_source(store, source, prefix + ".norm", channels);
    out.pwconv1 = binding::linear_from_source(store, source, prefix + ".pwconv1", storage_type, channels * 4, channels, true);
    out.pwconv2 = binding::linear_from_source(store, source, prefix + ".pwconv2", storage_type, channels, channels * 4, true);
    out.gamma = binding::layer_scale_from_named_source(store, source, prefix + ".gamma");
    return out;
}

QuantizerUnitWeights load_quantizer_unit(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t codebook_size) {
    QuantizerUnitWeights out;
    out.in_proj = binding::conv1d_from_named_source(
        store,
        source,
        prefix + ".in_proj.weight",
        prefix + ".in_proj.bias",
        storage_type);
    out.out_proj = binding::conv1d_from_named_source(
        store,
        source,
        prefix + ".out_proj.weight",
        prefix + ".out_proj.bias",
        storage_type);
    const auto codebook = source.require_f32(prefix + ".codebook.weight", {codebook_size, 8});
    out.codebook = store.make_from_f32(core::TensorShape::from_dims({codebook_size, 8}), storage_type, codebook);
    out.normalized_codebook = store.make_from_f32(
        core::TensorShape::from_dims({codebook_size, 8}),
        storage_type,
        normalized_rows(codebook, codebook_size, 8));
    return out;
}

std::shared_ptr<FishCodecWeights> load_weights(
    const FishAudioAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType matmul_storage_type,
    assets::TensorStorageType conv_storage_type) {
    auto weights = std::make_shared<FishCodecWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(backend, backend_type, "Fish Audio codec", weight_context_bytes);
    auto & store = *weights->store;
    const auto & source = *assets.codec_weights;

    weights->encoder_first = binding::conv1d_from_named_source(
        store,
        source,
        "encoder.block.0.conv.weight",
        "encoder.block.0.conv.bias",
        conv_storage_type);
    int64_t encoder_channels = 64;
    for (int64_t block_index = 0; block_index < 4; ++block_index) {
        const std::string prefix = "encoder.block." + std::to_string(block_index + 1) + ".block";
        EncoderBlockWeights block;
        block.residual1 = load_residual_unit(store, source, prefix + ".0", conv_storage_type, encoder_channels);
        block.residual3 = load_residual_unit(store, source, prefix + ".1", conv_storage_type, encoder_channels);
        block.residual9 = load_residual_unit(store, source, prefix + ".2", conv_storage_type, encoder_channels);
        block.snake = load_snake(store, source, prefix + ".3", encoder_channels);
        block.conv = binding::conv1d_from_named_source(
            store,
            source,
            prefix + ".4.conv.weight",
            prefix + ".4.conv.bias",
            conv_storage_type);
        encoder_channels *= 2;
        if (block_index == 3) {
            block.transformer = load_transformer(store, source, prefix + ".5", matmul_storage_type, 4);
        }
        weights->encoder_blocks.push_back(std::move(block));
    }
    weights->encoder_final_snake = load_snake(store, source, "encoder.block.5", kCodecDim);
    weights->encoder_final = binding::conv1d_from_named_source(
        store,
        source,
        "encoder.block.6.conv.weight",
        "encoder.block.6.conv.bias",
        conv_storage_type);

    for (int64_t i = 0; i < 2; ++i) {
        const std::string prefix = "quantizer.downsample." + std::to_string(i);
        weights->downsample.push_back({
            binding::conv1d_from_named_source(
                store,
                source,
                prefix + ".0.conv.weight",
                prefix + ".0.conv.bias",
                conv_storage_type),
            load_convnext(store, source, prefix + ".1", matmul_storage_type, kCodecDim),
        });
    }
    weights->pre_module = load_transformer(store, source, "quantizer.pre_module", matmul_storage_type, kCodecTransformerLayers);
    weights->semantic_quantizer = load_quantizer_unit(store, source, "quantizer.semantic_quantizer.quantizers.0", matmul_storage_type, 4096);
    for (int64_t i = 0; i < assets.config.codec.quantizer_codebooks; ++i) {
        weights->residual_quantizers.push_back(
            load_quantizer_unit(store, source, "quantizer.quantizer.quantizers." + std::to_string(i), matmul_storage_type, 1024));
    }
    weights->post_module = load_transformer(store, source, "quantizer.post_module", matmul_storage_type, kCodecTransformerLayers);
    for (int64_t i = 0; i < 2; ++i) {
        const std::string prefix = "quantizer.upsample." + std::to_string(i);
        weights->upsample.push_back({
            binding::conv_transpose1d_from_named_source(
                store,
                source,
                prefix + ".0.conv.weight",
                prefix + ".0.conv.bias",
                conv_storage_type),
            load_convnext(store, source, prefix + ".1", matmul_storage_type, kCodecDim),
        });
    }

    weights->decoder_first = binding::conv1d_from_named_source(
        store,
        source,
        "decoder.model.0.conv.weight",
        "decoder.model.0.conv.bias",
        conv_storage_type);
    int64_t decoder_channels = 1536;
    for (int64_t block_index = 0; block_index < 4; ++block_index) {
        const std::string prefix = "decoder.model." + std::to_string(block_index + 1) + ".block";
        DecoderBlockWeights block;
        block.snake = load_snake(store, source, prefix + ".0", decoder_channels);
        block.conv = binding::conv_transpose1d_from_named_source(
            store,
            source,
            prefix + ".1.conv.weight",
            prefix + ".1.conv.bias",
            conv_storage_type);
        decoder_channels /= 2;
        block.residual1 = load_residual_unit(store, source, prefix + ".2", conv_storage_type, decoder_channels);
        block.residual3 = load_residual_unit(store, source, prefix + ".3", conv_storage_type, decoder_channels);
        block.residual9 = load_residual_unit(store, source, prefix + ".4", conv_storage_type, decoder_channels);
        weights->decoder_blocks.push_back(std::move(block));
    }
    weights->decoder_final_snake = load_snake(store, source, "decoder.model.5", 96);
    weights->decoder_final = binding::conv1d_from_named_source(
        store,
        source,
        "decoder.model.6.conv.weight",
        "decoder.model.6.conv.bias",
        conv_storage_type);

    store.upload();
    return weights;
}

struct DecodeGraph {
    DecodeGraph(
        std::shared_ptr<const FishAudioAssets> assets,
        std::shared_ptr<const FishCodecWeights> weights,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        int64_t frames)
        : assets_(std::move(assets)),
          weights_(std::move(weights)),
          backend_(execution_context.backend()),
          backend_type_(execution_context.backend_type()),
          threads_(std::max(1, execution_context.config().threads)),
          frame_capacity_(frames),
          constants_(backend_, threads_, "Fish Audio codec decode constants") {
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Fish Audio codec decode graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "fish_audio.codec.decode", backend_type_};
        constants_.begin_graph();
        for (int64_t codebook = 0; codebook < assets_->config.codec.total_codebooks; ++codebook) {
            auto ids = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, frame_capacity_}));
            ggml_set_input(ids.tensor);
            code_inputs_.push_back(ids);
        }
        auto latent = build_decode_quantizer(ctx, constants_, code_inputs_, *weights_);
        auto waveform = build_decoder(ctx, latent, *weights_);
        output_ = waveform.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false);
        ggml_build_forward_expand(graph_, output_);
        constants_.finish_graph();
        constants_.ensure_uploaded();
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            throw std::runtime_error("failed to allocate Fish Audio codec decode graph");
        }
    }

    ~DecodeGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
    }

    bool matches(int64_t frames, ggml_backend_t backend, int threads) const {
        return frame_capacity_ >= frames && backend_ == backend && threads_ == std::max(1, threads);
    }

    runtime::AudioBuffer run(const FishAudioCodes & codes) {
        const int64_t codebooks = assets_->config.codec.total_codebooks;
        if (codes.codebooks != codebooks || codes.frames <= 0 ||
            static_cast<int64_t>(codes.codes.size()) != codebooks * codes.frames) {
            std::ostringstream oss;
            oss << "Fish Audio codec decode code shape mismatch: expected_codebooks=" << codebooks
                << " actual_codebooks=" << codes.codebooks
                << " frames=" << codes.frames
                << " values=" << codes.codes.size()
                << " expected_values=" << (codebooks * codes.frames);
            throw std::runtime_error(oss.str());
        }
        if (codes.frames > frame_capacity_) {
            throw std::runtime_error("Fish Audio codec decode request exceeds graph capacity");
        }
        for (int64_t codebook = 0; codebook < codebooks; ++codebook) {
            std::vector<int32_t> padded(static_cast<size_t>(frame_capacity_), 0);
            for (int64_t frame = 0; frame < codes.frames; ++frame) {
                int32_t value = codes.codes[static_cast<size_t>(codebook * codes.frames + frame)];
                if (codebook == 0) {
                    value = std::clamp<int32_t>(value, 0, 4095);
                } else {
                    value = std::clamp<int32_t>(value, 0, 1023);
                }
                padded[static_cast<size_t>(frame)] = value;
            }
            core::write_tensor_i32(code_inputs_[static_cast<size_t>(codebook)], padded);
        }
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Fish Audio codec decode graph compute failed");
        }
        auto values = core::read_tensor_f32(output_);
        const int64_t expected_samples = codes.frames * assets_->config.codec.frame_length;
        if (static_cast<int64_t>(values.size()) > expected_samples) {
            values.resize(static_cast<size_t>(expected_samples));
        }
        return runtime::AudioBuffer{assets_->config.codec.sample_rate, 1, std::move(values)};
    }

private:
    std::shared_ptr<const FishAudioAssets> assets_;
    std::shared_ptr<const FishCodecWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    int64_t frame_capacity_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::vector<core::TensorValue> code_inputs_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
    common::ConstantTensorCache constants_;
};

struct EncodeGraph {
    EncodeGraph(
        std::shared_ptr<const FishAudioAssets> assets,
        std::shared_ptr<const FishCodecWeights> weights,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        int64_t samples,
        int64_t frames)
        : assets_(std::move(assets)),
          weights_(std::move(weights)),
          backend_(execution_context.backend()),
          backend_type_(execution_context.backend_type()),
          threads_(std::max(1, execution_context.config().threads)),
          sample_capacity_(samples),
          frame_capacity_(frames),
          constants_(backend_, threads_, "Fish Audio codec encode constants") {
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Fish Audio codec encode graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "fish_audio.codec.encode", backend_type_};
        constants_.begin_graph();
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, sample_capacity_}));
        ggml_set_input(input_.tensor);
        auto encoded = build_encoder(ctx, constants_, input_, *weights_);
        trace_outputs_.push_back({"fish_audio.codec.encoder_latent", encoded});
        build_encode_quantizer(ctx, constants_, encoded, *weights_, code_outputs_, trace_outputs_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 1048576, false);
        for (const auto & trace_output : trace_outputs_) {
            ggml_set_output(trace_output.second.tensor);
            ggml_build_forward_expand(graph_, trace_output.second.tensor);
        }
        for (ggml_tensor * code_output : code_outputs_) {
            ggml_build_forward_expand(graph_, code_output);
        }
        constants_.finish_graph();
        constants_.ensure_uploaded();
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            throw std::runtime_error("failed to allocate Fish Audio codec encode graph");
        }
    }

    ~EncodeGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
    }

    bool matches(int64_t samples, int64_t frames, ggml_backend_t backend, int threads) const {
        return sample_capacity_ >= samples &&
            frame_capacity_ >= frames &&
            backend_ == backend &&
            threads_ == std::max(1, threads);
    }

    FishAudioCodes run(const runtime::AudioBuffer & audio) {
        auto mono = prepare_codec_mono(audio, assets_->config.codec.sample_rate);
        const int64_t original_samples = static_cast<int64_t>(mono.size());
        const int64_t padded_samples = ceil_div(original_samples, assets_->config.codec.frame_length) * assets_->config.codec.frame_length;
        const int64_t frames = ceil_div(original_samples, assets_->config.codec.frame_length);
        if (padded_samples > sample_capacity_ || frames > frame_capacity_) {
            throw std::runtime_error("Fish Audio codec encode request exceeds graph capacity");
        }
        mono.resize(static_cast<size_t>(sample_capacity_), 0.0F);
        core::write_tensor_f32(input_, mono);
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Fish Audio codec encode graph compute failed");
        }
        if (engine::debug::trace_log_enabled()) {
            for (const auto & trace_output : trace_outputs_) {
                engine::debug::trace_log_f32(
                    trace_output.first,
                    dims_vector(trace_output.second.shape),
                    core::read_tensor_f32(trace_output.second.tensor));
            }
        }
        FishAudioCodes out;
        out.codebooks = static_cast<int64_t>(code_outputs_.size());
        out.frames = frames;
        out.codes.resize(static_cast<size_t>(out.codebooks * out.frames));
        for (int64_t codebook = 0; codebook < out.codebooks; ++codebook) {
            auto values = core::read_tensor_i32(code_outputs_[static_cast<size_t>(codebook)]);
            for (int64_t frame = 0; frame < out.frames; ++frame) {
                out.codes[static_cast<size_t>(codebook * out.frames + frame)] = values[static_cast<size_t>(frame)];
            }
        }
        engine::debug::trace_log_i32(
            "fish_audio.codec.reference_codes",
            {out.codebooks, out.frames},
            out.codes);
        return out;
    }

private:
    std::shared_ptr<const FishAudioAssets> assets_;
    std::shared_ptr<const FishCodecWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    int64_t sample_capacity_ = 0;
    int64_t frame_capacity_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue input_;
    std::vector<ggml_tensor *> code_outputs_;
    std::vector<std::pair<std::string, core::TensorValue>> trace_outputs_;
    ggml_cgraph * graph_ = nullptr;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
    common::ConstantTensorCache constants_;
};

}  // namespace

class FishAudioCodecRuntime::Impl {
public:
    Impl(
        std::shared_ptr<const FishAudioAssets> assets,
        core::BackendConfig backend,
        int threads,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType matmul_weight_storage_type,
        assets::TensorStorageType conv_weight_storage_type)
        : assets_(std::move(assets)),
          execution_(std::move(backend)),
          threads_(std::max(1, threads)),
          graph_arena_bytes_(graph_arena_bytes) {
        weights_ = load_weights(
            *assets_,
            execution_.backend(),
            execution_.backend_type(),
            weight_context_bytes,
            matmul_weight_storage_type,
            conv_weight_storage_type);
    }

    FishAudioCodes encode_reference(const runtime::AudioBuffer & audio) {
        auto mono = prepare_codec_mono(audio, assets_->config.codec.sample_rate);
        const int64_t samples = ceil_div(static_cast<int64_t>(mono.size()), assets_->config.codec.frame_length) *
            assets_->config.codec.frame_length;
        const int64_t frames = ceil_div(static_cast<int64_t>(mono.size()), assets_->config.codec.frame_length);
        if (encode_graph_ == nullptr || !encode_graph_->matches(samples, frames, execution_.backend(), threads_)) {
            encode_graph_ = std::make_unique<EncodeGraph>(assets_, weights_, execution_, graph_arena_bytes_, samples, frames);
        }
        return encode_graph_->run(audio);
    }

    runtime::AudioBuffer decode(const FishAudioCodes & codes) {
        if (decode_graph_ == nullptr || !decode_graph_->matches(codes.frames, execution_.backend(), threads_)) {
            decode_graph_ = std::make_unique<DecodeGraph>(assets_, weights_, execution_, graph_arena_bytes_, codes.frames);
        }
        return decode_graph_->run(codes);
    }

    void release_encode_graph() {
        encode_graph_.reset();
    }

    void release_runtime_graphs() {
        encode_graph_.reset();
        decode_graph_.reset();
    }

private:
    std::shared_ptr<const FishAudioAssets> assets_;
    core::ExecutionContext execution_;
    int threads_ = 1;
    size_t graph_arena_bytes_ = 0;
    std::shared_ptr<const FishCodecWeights> weights_;
    std::unique_ptr<EncodeGraph> encode_graph_;
    std::unique_ptr<DecodeGraph> decode_graph_;
};

FishAudioCodecRuntime::FishAudioCodecRuntime(
    std::shared_ptr<const FishAudioAssets> assets,
    core::BackendConfig backend,
    int threads,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType matmul_weight_storage_type,
    assets::TensorStorageType conv_weight_storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          std::move(backend),
          threads,
          graph_arena_bytes,
          weight_context_bytes,
          matmul_weight_storage_type,
          conv_weight_storage_type)) {}

FishAudioCodecRuntime::~FishAudioCodecRuntime() = default;

FishAudioCodes FishAudioCodecRuntime::encode_reference(const runtime::AudioBuffer & audio) {
    return impl_->encode_reference(audio);
}

runtime::AudioBuffer FishAudioCodecRuntime::decode(const FishAudioCodes & codes) {
    return impl_->decode(codes);
}

void FishAudioCodecRuntime::release_encode_graph() {
    impl_->release_encode_graph();
}

void FishAudioCodecRuntime::release_runtime_graphs() {
    impl_->release_runtime_graphs();
}

}  // namespace engine::models::fish_audio
