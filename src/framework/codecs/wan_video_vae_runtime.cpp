#include "engine/framework/codecs/wan_video_vae_runtime.h"

#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::codecs {
namespace {

engine::core::TensorValue ensure_f32(engine::core::ModuleBuildContext &ctx,
                                     const engine::core::TensorValue &value) {
  if (value.type == GGML_TYPE_F32) {
    return value;
  }
  return engine::core::wrap_tensor(
      ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F32), value.shape,
      GGML_TYPE_F32);
}

engine::core::TensorValue pad_3d(engine::core::ModuleBuildContext &ctx,
                                 const engine::core::TensorValue &input,
                                 int64_t left, int64_t right, int64_t top,
                                 int64_t bottom, int64_t front, int64_t back) {
  engine::core::validate_rank_between(input, 4, 4, "Wan VAE 3D input");
  if (left < 0 || right < 0 || top < 0 || bottom < 0 || front < 0 || back < 0) {
    throw std::runtime_error("Wan VAE 3D padding values must be non-negative");
  }
  const auto contiguous =
      engine::core::ensure_backend_addressable_layout(ctx, input);
  auto output_shape = input.shape;
  output_shape.dims[1] += front + back;
  output_shape.dims[2] += top + bottom;
  output_shape.dims[3] += left + right;
  return engine::core::wrap_tensor(
      ggml_pad_ext(ctx.ggml, contiguous.tensor, static_cast<int>(left),
                   static_cast<int>(right), static_cast<int>(top),
                   static_cast<int>(bottom), static_cast<int>(front),
                   static_cast<int>(back), 0, 0),
      output_shape, input.type);
}

engine::core::TensorValue
video_to_frame_batch(engine::core::ModuleBuildContext &ctx,
                     const engine::core::TensorValue &input) {
  return engine::modules::TransposeModule({{1, 0, 2, 3}, 4}).build(ctx, input);
}

engine::core::TensorValue
frame_batch_to_video(engine::core::ModuleBuildContext &ctx,
                     const engine::core::TensorValue &input) {
  return engine::modules::TransposeModule({{1, 0, 2, 3}, 4}).build(ctx, input);
}

void validate_architecture_config(const WanVideoVAEArchitectureConfig &config) {
  if (config.dim <= 0 || config.latent_channels <= 0 ||
      config.num_res_blocks < 0) {
    throw std::runtime_error("Wan VAE architecture dimensions must be valid");
  }
  if (config.dim_mult.empty()) {
    throw std::runtime_error(
        "Wan VAE architecture requires at least one dim multiplier");
  }
  for (const int64_t mult : config.dim_mult) {
    if (mult <= 0) {
      throw std::runtime_error("Wan VAE dim multipliers must be positive");
    }
  }
  if (config.temporal_downsample.size() + 1 != config.dim_mult.size()) {
    throw std::runtime_error(
        "Wan VAE temporal_downsample must have dim_mult.size() - 1 entries");
  }
}

std::vector<int64_t> encoder_dims(const WanVideoVAEArchitectureConfig &config) {
  std::vector<int64_t> dims;
  dims.reserve(config.dim_mult.size() + 1);
  dims.push_back(config.dim);
  for (const int64_t mult : config.dim_mult) {
    dims.push_back(config.dim * mult);
  }
  return dims;
}

std::vector<int64_t> decoder_dims(const WanVideoVAEArchitectureConfig &config) {
  std::vector<int64_t> dims;
  dims.reserve(config.dim_mult.size() + 1);
  dims.push_back(config.dim * config.dim_mult.back());
  for (auto it = config.dim_mult.rbegin(); it != config.dim_mult.rend(); ++it) {
    dims.push_back(config.dim * *it);
  }
  return dims;
}

bool has_attention_at_scale(const WanVideoVAEArchitectureConfig &config,
                            float scale) {
  return std::find_if(config.attention_scales.begin(),
                      config.attention_scales.end(), [scale](float value) {
                        return std::fabs(value - scale) < 1.0e-6F;
                      }) != config.attention_scales.end();
}

WanVideoVAEResampleMode
downsample_mode(const WanVideoVAEArchitectureConfig &config, size_t index) {
  return config.temporal_downsample.at(index)
             ? WanVideoVAEResampleMode::Downsample3d
             : WanVideoVAEResampleMode::Downsample2d;
}

WanVideoVAEResampleMode
upsample_mode(const WanVideoVAEArchitectureConfig &config, size_t index) {
  const size_t reverse_index = config.temporal_downsample.size() - 1 - index;
  return config.temporal_downsample.at(reverse_index)
             ? WanVideoVAEResampleMode::Upsample3d
             : WanVideoVAEResampleMode::Upsample2d;
}

int64_t conv2d_output_dim(int64_t input, int64_t kernel, int stride,
                          int padding, int dilation) {
  return (input + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

int64_t conv3d_output_dim(int64_t input, int64_t kernel, int stride,
                          int padding, int dilation) {
  return (input + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

bool use_cuda_wan_vae_fast_path(
    const engine::core::ModuleBuildContext &ctx,
    bool enabled) noexcept {
  return enabled && ctx.backend_type == engine::core::BackendType::Cuda;
}

engine::core::TensorValue
wan_vae_add_2d_bias(engine::core::ModuleBuildContext &ctx,
                    const engine::core::TensorValue &output,
                    int64_t out_channels,
                    const std::optional<engine::core::TensorValue> &bias,
                    bool use_cuda_fast_lowerings) {
  if (!bias.has_value()) {
    return output;
  }
  engine::core::validate_shape(
      *bias, engine::core::TensorShape::from_dims({out_channels}), "bias");
  const auto output_contiguous =
      engine::core::ensure_backend_addressable_layout(ctx, output);
  const auto bias_view = engine::core::reshape_tensor(
      ctx, ensure_f32(ctx, *bias),
      engine::core::TensorShape::from_dims({1, out_channels, 1, 1}));
  auto *bias_tensor = use_cuda_wan_vae_fast_path(ctx, use_cuda_fast_lowerings)
                          ? bias_view.tensor
                          : ggml_repeat(ctx.ggml, bias_view.tensor,
                                        output_contiguous.tensor);
  return engine::core::wrap_tensor(
      ggml_add(ctx.ggml, output_contiguous.tensor, bias_tensor), output.shape,
      GGML_TYPE_F32);
}

engine::core::TensorValue
wan_vae_add_3d_bias(engine::core::ModuleBuildContext &ctx,
                    const engine::core::TensorValue &output,
                    int64_t out_channels,
                    const std::optional<engine::core::TensorValue> &bias,
                    bool use_cuda_fast_lowerings) {
  if (!bias.has_value()) {
    return output;
  }
  engine::core::validate_shape(
      *bias, engine::core::TensorShape::from_dims({out_channels}), "bias");
  const auto output_contiguous =
      engine::core::ensure_backend_addressable_layout(ctx, output);
  const auto bias_view = engine::core::reshape_tensor(
      ctx, ensure_f32(ctx, *bias),
      engine::core::TensorShape::from_dims({out_channels, 1, 1, 1}));
  auto *bias_tensor = use_cuda_wan_vae_fast_path(ctx, use_cuda_fast_lowerings)
                          ? bias_view.tensor
                          : ggml_repeat(ctx.ggml, bias_view.tensor,
                                        output_contiguous.tensor);
  if (output_contiguous.type == GGML_TYPE_F16) {
    return engine::core::wrap_tensor(
        ggml_add_cast(ctx.ggml, output_contiguous.tensor, bias_tensor,
                      GGML_TYPE_F32),
        output.shape, GGML_TYPE_F32);
  }
  return engine::core::wrap_tensor(
      ggml_add(ctx.ggml, output_contiguous.tensor, bias_tensor), output.shape,
      GGML_TYPE_F32);
}

engine::core::TensorValue
wan_vae_conv2d_linear(engine::core::ModuleBuildContext &ctx,
                      const engine::core::TensorValue &input,
                      const engine::modules::Conv2dWeights &weights,
                      const engine::modules::Conv2dConfig &config) {
  if (ctx.ggml == nullptr) {
    throw std::runtime_error("ModuleBuildContext.ggml is null");
  }
  engine::core::validate_rank_between(input, 4, 4, "Wan VAE Conv2D input");
  engine::core::validate_shape(
      input,
      engine::core::TensorShape::from_dims(
          {input.shape.dims[0], config.in_channels, input.shape.dims[2],
           input.shape.dims[3]}),
      "Wan VAE Conv2D input");
  engine::core::validate_shape(
      weights.weight,
      engine::core::TensorShape::from_dims(
          {config.out_channels, config.in_channels, config.kernel_height,
           config.kernel_width}),
      "Wan VAE Conv2D weight");

  const auto output_shape = engine::core::TensorShape::from_dims({
      input.shape.dims[0],
      config.out_channels,
      conv2d_output_dim(input.shape.dims[2], config.kernel_height,
                        config.stride_height, config.padding_height,
                        config.dilation_height),
      conv2d_output_dim(input.shape.dims[3], config.kernel_width,
                        config.stride_width, config.padding_width,
                        config.dilation_width),
  });

  if (!use_cuda_wan_vae_fast_path(ctx, config.cuda_fast_lowering) ||
      (weights.weight.type != GGML_TYPE_F32 &&
       weights.weight.type != GGML_TYPE_F16)) {
    auto no_bias_config = config;
    no_bias_config.use_bias = false;
    const engine::modules::Conv2dWeights no_bias_weights{weights.weight,
                                                         std::nullopt};
    return engine::modules::Conv2dModule(no_bias_config)
        .build(ctx, input, no_bias_weights);
  }

  constexpr int64_t kMaxCudaFastPathConv2dRows = 4'000'000;
  const int64_t rows = output_shape.dims[0] * output_shape.dims[2] * output_shape.dims[3];
  if (config.cuda_large_shape_lowering &&
      rows > kMaxCudaFastPathConv2dRows && input.shape.dims[0] > 1) {
    const int64_t max_frames =
        std::max<int64_t>(1, kMaxCudaFastPathConv2dRows / (output_shape.dims[2] * output_shape.dims[3]));
    std::vector<engine::core::TensorValue> chunks;
    chunks.reserve(static_cast<size_t>((input.shape.dims[0] + max_frames - 1) / max_frames));
    for (int64_t start = 0; start < input.shape.dims[0]; start += max_frames) {
      const int64_t count = std::min<int64_t>(max_frames, input.shape.dims[0] - start);
      const auto chunk = engine::modules::SliceModule({0, start, count}).build(ctx, input);
      chunks.push_back(wan_vae_conv2d_linear(ctx, chunk, weights, config));
    }
    auto output = chunks.front();
    for (size_t i = 1; i < chunks.size(); ++i) {
      output = engine::modules::ConcatModule({0, config.cuda_fast_lowering}).build(ctx, output, chunks[i]);
    }
    return output;
  }

  const auto input_contiguous =
      engine::core::ensure_backend_addressable_layout(ctx, ensure_f32(ctx, input));
  const auto weight_contiguous =
      engine::core::ensure_backend_addressable_layout(ctx, weights.weight);
  auto *im2col = ggml_im2col(ctx.ggml, weight_contiguous.tensor,
                             input_contiguous.tensor, config.stride_width,
                             config.stride_height, config.padding_width,
                             config.padding_height, config.dilation_width,
                             config.dilation_height, true, GGML_TYPE_F16);
  ggml_im2col_2d_set_lowering(im2col, GGML_IM2COL_2D_LOWERING_CUDA_N_K3_PAD1_X8);
  auto *mm = ggml_mul_mat(
      ctx.ggml,
      ggml_reshape_2d(ctx.ggml, im2col, im2col->ne[0],
                      im2col->ne[3] * im2col->ne[2] * im2col->ne[1]),
      ggml_reshape_2d(ctx.ggml, weight_contiguous.tensor,
                      weight_contiguous.tensor->ne[0] *
                          weight_contiguous.tensor->ne[1] *
                          weight_contiguous.tensor->ne[2],
                      weight_contiguous.tensor->ne[3]));
  if (config.cuda_tile_f16_accum_output_lowering) {
    ggml_mul_mat_set_lowering(mm, GGML_MUL_MAT_LOWERING_CUDA_TILE_F16_ACCUM_OUTPUT);
  }
  auto *raw = ggml_reshape_4d(ctx.ggml, mm, im2col->ne[1], im2col->ne[2],
                              im2col->ne[3], weight_contiguous.tensor->ne[3]);
  raw = ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, raw, 0, 1, 3, 2));
  return engine::core::wrap_tensor(raw, output_shape, GGML_TYPE_F32);
}

engine::core::TensorValue
wan_vae_conv2d(engine::core::ModuleBuildContext &ctx,
               const engine::core::TensorValue &input,
               const engine::modules::Conv2dWeights &weights,
               const engine::modules::Conv2dConfig &config) {
  if (config.use_bias && !weights.bias.has_value()) {
    throw std::runtime_error("bias is required when Wan VAE Conv2D uses bias");
  }
  return wan_vae_add_2d_bias(
      ctx, wan_vae_conv2d_linear(ctx, input, weights, config),
      config.out_channels, config.use_bias ? weights.bias : std::nullopt,
      config.cuda_fast_lowering);
}

engine::core::TensorValue
wan_vae_conv3d_linear(engine::core::ModuleBuildContext &ctx,
                      const engine::core::TensorValue &input,
                      const engine::modules::Conv3dWeights &weights,
                      const engine::modules::Conv3dConfig &config) {
  if (ctx.ggml == nullptr) {
    throw std::runtime_error("ModuleBuildContext.ggml is null");
  }
  engine::core::validate_rank_between(input, 4, 4, "Wan VAE Conv3D input");
  if (input.shape.dims[0] % config.in_channels != 0) {
    throw std::runtime_error(
        "Wan VAE Conv3D input first dimension must be batch * in_channels");
  }
  const int64_t batch = input.shape.dims[0] / config.in_channels;
  engine::core::validate_shape(
      weights.weight,
      engine::core::TensorShape::from_dims(
          {config.out_channels * config.in_channels, config.kernel_depth,
           config.kernel_height, config.kernel_width}),
      "Wan VAE Conv3D weight");

  const auto output_shape = engine::core::TensorShape::from_dims({
      batch * config.out_channels,
      conv3d_output_dim(input.shape.dims[1], config.kernel_depth,
                        config.stride_depth, config.padding_depth,
                        config.dilation_depth),
      conv3d_output_dim(input.shape.dims[2], config.kernel_height,
                        config.stride_height, config.padding_height,
                        config.dilation_height),
      conv3d_output_dim(input.shape.dims[3], config.kernel_width,
                        config.stride_width, config.padding_width,
                        config.dilation_width),
  });

  constexpr int64_t kMaxCudaFastPathConvRows = 4'000'000;
  const int64_t rows = output_shape.dims[1] * output_shape.dims[2] * output_shape.dims[3];
  if (use_cuda_wan_vae_fast_path(ctx, config.cuda_fast_lowering) &&
      config.kernel_depth == 1 && config.stride_depth == 1 &&
      config.padding_depth == 0 && config.dilation_depth == 1 &&
      config.cuda_large_shape_lowering &&
      rows > kMaxCudaFastPathConvRows && input.shape.dims[1] > 1) {
    const int64_t max_frames =
        std::max<int64_t>(1, kMaxCudaFastPathConvRows / (output_shape.dims[2] * output_shape.dims[3]));
    std::vector<engine::core::TensorValue> chunks;
    chunks.reserve(static_cast<size_t>((input.shape.dims[1] + max_frames - 1) / max_frames));
    for (int64_t start = 0; start < input.shape.dims[1]; start += max_frames) {
      const int64_t count = std::min<int64_t>(max_frames, input.shape.dims[1] - start);
      const auto chunk = engine::modules::SliceModule({1, start, count}).build(ctx, input);
      chunks.push_back(wan_vae_conv3d_linear(ctx, chunk, weights, config));
    }
    auto output = chunks.front();
    for (size_t i = 1; i < chunks.size(); ++i) {
      output = engine::modules::ConcatModule({1, config.cuda_fast_lowering}).build(ctx, output, chunks[i]);
    }
    return output;
  }

  if (!use_cuda_wan_vae_fast_path(ctx, config.cuda_fast_lowering) ||
      (weights.weight.type != GGML_TYPE_F32 &&
       weights.weight.type != GGML_TYPE_F16)) {
    auto no_bias_config = config;
    no_bias_config.use_bias = false;
    const engine::modules::Conv3dWeights no_bias_weights{weights.weight,
                                                         std::nullopt};
    return engine::modules::Conv3dModule(no_bias_config)
        .build(ctx, input, no_bias_weights);
  }

  const auto input_contiguous =
      engine::core::ensure_backend_addressable_layout(ctx, ensure_f32(ctx, input));
  const auto weight_contiguous =
      engine::core::ensure_backend_addressable_layout(ctx, weights.weight);
  auto *im2col = ggml_im2col_3d(
      ctx.ggml, weight_contiguous.tensor, input_contiguous.tensor,
      config.in_channels, config.stride_width, config.stride_height,
      config.stride_depth, config.padding_width, config.padding_height,
      config.padding_depth, config.dilation_width, config.dilation_height,
      config.dilation_depth, GGML_TYPE_F16);
  ggml_im2col_3d_set_lowering(im2col, GGML_IM2COL_3D_LOWERING_CUDA_N1_K3_NOPAD_X8);
  auto *mm = ggml_mul_mat(
      ctx.ggml,
      ggml_reshape_2d(ctx.ggml, im2col, im2col->ne[0],
                      im2col->ne[3] * im2col->ne[2] * im2col->ne[1]),
      ggml_reshape_2d(ctx.ggml, weight_contiguous.tensor,
                      weight_contiguous.tensor->ne[0] *
                          weight_contiguous.tensor->ne[1] *
                          weight_contiguous.tensor->ne[2] *
                          config.in_channels,
                      config.out_channels));
  if (config.cuda_tile_f16_accum_output_lowering) {
    ggml_mul_mat_set_lowering(mm, GGML_MUL_MAT_LOWERING_CUDA_TILE_F16_ACCUM_OUTPUT);
  }
  auto *raw = ggml_reshape_4d(ctx.ggml, mm, im2col->ne[1] * im2col->ne[2],
                              output_shape.dims[1], batch, config.out_channels);
  raw = ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, raw, 0, 1, 3, 2));
  raw = ggml_reshape_4d(ctx.ggml, raw, output_shape.dims[3],
                        output_shape.dims[2], output_shape.dims[1],
                        config.out_channels * batch);
  return engine::core::wrap_tensor(raw, output_shape, GGML_TYPE_F32);
}

engine::core::TensorValue
wan_vae_silu_rms_norm3d(engine::core::ModuleBuildContext &ctx,
                        const engine::core::TensorValue &input,
                        const WanVideoRMSNorm3dWeights &weights,
                        const WanVideoRMSNorm3dConfig &config) {
  if (use_cuda_wan_vae_fast_path(ctx, config.cuda_fast_lowering)) {
    engine::core::validate_shape(input,
                                 engine::core::TensorShape::from_dims(
                                     {config.channels, input.shape.dims[1],
                                      input.shape.dims[2], input.shape.dims[3]}),
                                 "Wan video RMSNorm input");
    engine::core::validate_shape(
        weights.gamma, engine::core::TensorShape::from_dims({config.channels}),
        "Wan video RMSNorm gamma");
    const auto input_f32 = ensure_f32(ctx, input);
    const auto contiguous =
        engine::core::ensure_backend_addressable_layout(ctx, input_f32);
    auto *raw = ggml_rms_norm_channels_silu(ctx.ggml, contiguous.tensor,
                                            ensure_f32(ctx, weights.gamma).tensor,
                                            config.eps);
    ggml_rms_norm_channels_set_lowering(raw, GGML_RMS_NORM_CHANNELS_LOWERING_CUDA_COALESCED);
    return engine::core::wrap_tensor(raw, input.shape, GGML_TYPE_F32);
  }
  auto hidden = WanVideoRMSNorm3dModule(config).build(ctx, input, weights);
  return engine::modules::SiluModule{}.build(ctx, hidden);
}

engine::core::TensorValue wan_vae_silu_rms_norm3d_add_bias(
    engine::core::ModuleBuildContext &ctx,
    const engine::core::TensorValue &input,
    const std::optional<engine::core::TensorValue> &bias,
    const WanVideoRMSNorm3dWeights &weights,
    const WanVideoRMSNorm3dConfig &config) {
  if (bias.has_value() && use_cuda_wan_vae_fast_path(ctx, config.cuda_fast_lowering)) {
    engine::core::validate_shape(input,
                                 engine::core::TensorShape::from_dims(
                                     {config.channels, input.shape.dims[1],
                                      input.shape.dims[2], input.shape.dims[3]}),
                                 "Wan video RMSNorm input");
    engine::core::validate_shape(
        *bias, engine::core::TensorShape::from_dims({config.channels}),
        "bias");
    engine::core::validate_shape(
        weights.gamma, engine::core::TensorShape::from_dims({config.channels}),
        "Wan video RMSNorm gamma");
    const auto input_f32 = ensure_f32(ctx, input);
    const auto contiguous =
        engine::core::ensure_backend_addressable_layout(ctx, input_f32);
    auto *raw = ggml_rms_norm_channels_add_bias_silu(
        ctx.ggml, contiguous.tensor, ensure_f32(ctx, *bias).tensor,
        ensure_f32(ctx, weights.gamma).tensor, config.eps);
    ggml_rms_norm_channels_set_lowering(raw, GGML_RMS_NORM_CHANNELS_LOWERING_CUDA_COALESCED);
    return engine::core::wrap_tensor(raw, input.shape, GGML_TYPE_F32);
  }
  return wan_vae_silu_rms_norm3d(
      ctx, wan_vae_add_3d_bias(ctx, input, config.channels, bias, config.cuda_fast_lowering), weights,
      config);
}

engine::core::TensorValue wan_vae_causal_conv3d_linear(
    engine::core::ModuleBuildContext &ctx,
    const engine::core::TensorValue &input,
    const engine::modules::Conv3dWeights &weights,
    const engine::modules::Conv3dConfig &config,
    const engine::core::TensorValue *cache) {
  auto conv_config = config;
  auto conv_input = input;
  int64_t front_pad = 2 * config.padding_depth;
  if (cache != nullptr) {
    engine::core::validate_rank_between(*cache, 4, 4,
                                        "Wan causal Conv3D cache");
    if (cache->shape.dims[0] != input.shape.dims[0] ||
        cache->shape.dims[2] != input.shape.dims[2] ||
        cache->shape.dims[3] != input.shape.dims[3]) {
      throw std::runtime_error("Wan causal Conv3D cache shape mismatch");
    }
    if (cache->shape.dims[1] > front_pad) {
      throw std::runtime_error("Wan causal Conv3D cache is longer than the "
                               "configured front padding");
    }
    front_pad -= cache->shape.dims[1];
    const bool cache_type_supported = cache->type == GGML_TYPE_F32 || cache->type == GGML_TYPE_F16;
    const bool input_type_supported = input.type == GGML_TYPE_F32 || input.type == GGML_TYPE_F16;
    const bool spatial_gemm_supported =
        use_cuda_wan_vae_fast_path(ctx, config.cuda_fast_lowering) && weights.weight.type == GGML_TYPE_F32 &&
        cache_type_supported && input_type_supported &&
        config.kernel_depth == 3 && config.kernel_height == 3 &&
        config.kernel_width == 3 && config.stride_depth == 1 &&
        config.stride_height == 1 && config.stride_width == 1 &&
        config.padding_depth == 1 && config.padding_height == 1 &&
        config.padding_width == 1 && config.dilation_depth == 1 &&
        config.dilation_height == 1 && config.dilation_width == 1 &&
        input.shape.dims[0] == config.in_channels &&
        cache->shape.dims[1] <= 2;
    if (spatial_gemm_supported) {
      const auto cache_f32 =
          engine::core::ensure_backend_addressable_layout(ctx, *cache);
      const auto input_f32 =
          engine::core::ensure_backend_addressable_layout(ctx, input);
      const auto weight_contiguous =
          engine::core::ensure_backend_addressable_layout(ctx, weights.weight);
      auto *raw = ggml_conv_3d_concat_pad_spatial_gemm_ex(
          ctx.ggml, cache_f32.tensor, input_f32.tensor, weight_contiguous.tensor,
          config.padding_width, config.padding_width, config.padding_height,
          config.padding_height, static_cast<int>(front_pad), 0,
          GGML_TYPE_F16);
      ggml_conv_3d_concat_pad_spatial_gemm_set_lowering(
          raw,
          config.cuda_large_shape_lowering
              ? GGML_CONV_3D_CONCAT_PAD_SPATIAL_GEMM_LOWERING_CUDA_TILED_C48
              : GGML_CONV_3D_CONCAT_PAD_SPATIAL_GEMM_LOWERING_CUDA_C48);
      const auto output_shape = engine::core::TensorShape::from_dims(
          {config.out_channels, input.shape.dims[1], input.shape.dims[2],
           input.shape.dims[3]});
      return engine::core::wrap_tensor(raw, output_shape, GGML_TYPE_F16);
    }
    const auto concat_cache = cache->type == input.type
        ? *cache
        : engine::core::wrap_tensor(ggml_cast(ctx.ggml, cache->tensor, input.type), cache->shape, input.type);
    conv_input = engine::modules::ConcatModule({1, config.cuda_fast_lowering}).build(ctx, concat_cache, input);
  }
  const auto padded =
      pad_3d(ctx, conv_input, config.padding_width, config.padding_width,
             config.padding_height, config.padding_height, front_pad, 0);
  conv_config.padding_depth = 0;
  conv_config.padding_height = 0;
  conv_config.padding_width = 0;
  return wan_vae_conv3d_linear(ctx, padded, weights, conv_config);
}

void validate_attention_weight_presence(
    const std::optional<WanVideoVAEAttentionBlockWeights> &weights,
    bool required) {
  if (required && !weights.has_value()) {
    throw std::runtime_error(
        "Wan VAE stage block requires attention weights at this scale");
  }
  if (!required && weights.has_value()) {
    throw std::runtime_error(
        "Wan VAE stage block has unexpected attention weights at this scale");
  }
}

const engine::core::TensorValue *next_cache_input(
    WanVideoVAECacheBuildState &cache) {
  const engine::core::TensorValue *value = nullptr;
  if (cache.input_caches != nullptr) {
    if (cache.cursor >= cache.input_caches->size()) {
      throw std::runtime_error("Wan VAE cache input count mismatch");
    }
    value = &(*cache.input_caches)[cache.cursor];
  }
  return value;
}

void push_cache_output(engine::core::ModuleBuildContext &ctx,
                       WanVideoVAECacheBuildState &cache,
                       engine::core::TensorValue value) {
  if (cache.output_caches != nullptr) {
    cache.output_caches->push_back(
        cache.addressable_output_caches
            ? engine::core::ensure_backend_addressable_layout(ctx, value)
            : value);
  }
  ++cache.cursor;
}

engine::core::TensorValue zero_like(engine::core::ModuleBuildContext &ctx,
                                    const engine::core::TensorValue &input) {
  return engine::core::wrap_tensor(ggml_scale(ctx.ggml, input.tensor, 0.0F),
                                   input.shape, GGML_TYPE_F32);
}

engine::core::TensorValue last_time_frames(
    engine::core::ModuleBuildContext &ctx,
    const engine::core::TensorValue &input,
    int64_t frames) {
  if (frames <= 0 || input.shape.dims[1] < frames) {
    throw std::runtime_error("Wan VAE cache slice shape mismatch");
  }
  return engine::modules::SliceModule(
             {1, input.shape.dims[1] - frames, frames})
      .build(ctx, input);
}

engine::core::TensorValue make_two_frame_conv_cache(
    engine::core::ModuleBuildContext &ctx,
    const engine::core::TensorValue &input,
    const engine::core::TensorValue *previous_cache,
    bool raw_single_frame_inputs) {
  if (input.shape.dims[1] >= 2) {
    return last_time_frames(ctx, input, 2);
  }
  if (input.shape.dims[1] != 1) {
    throw std::runtime_error("Wan VAE conv cache source has no frames");
  }
  if (raw_single_frame_inputs) {
    return last_time_frames(ctx, input, 1);
  }
  engine::core::TensorValue prefix;
  if (previous_cache != nullptr) {
    prefix = last_time_frames(ctx, *previous_cache, 1);
    if (prefix.type != input.type) {
      prefix = engine::core::wrap_tensor(ggml_cast(ctx.ggml, prefix.tensor, input.type),
                                         prefix.shape, input.type);
    }
  } else {
    prefix = zero_like(ctx, input);
  }
  return engine::modules::ConcatModule({1}).build(ctx, prefix, input);
}

engine::core::TensorValue make_one_frame_cache(
    engine::core::ModuleBuildContext &ctx,
    const engine::core::TensorValue &input) {
  return last_time_frames(ctx, input, 1);
}

engine::core::TensorValue build_cached_causal_conv3d(
    engine::core::ModuleBuildContext &ctx,
    const engine::core::TensorValue &input,
    const engine::modules::Conv3dWeights &weights,
    const engine::modules::Conv3dConfig &config,
    WanVideoVAECacheBuildState &cache) {
  if (config.padding_depth <= 0) {
    return WanCausalConv3dModule(config).build(ctx, input, weights);
  }
  const auto *previous = next_cache_input(cache);
  constexpr int64_t kMaxCudaFastPathConvRows = 4'000'000;
  const int64_t rows = input.shape.dims[1] * input.shape.dims[2] * input.shape.dims[3];
  const bool can_split_cached_conv =
      use_cuda_wan_vae_fast_path(ctx, config.cuda_fast_lowering) &&
      (weights.weight.type == GGML_TYPE_F32 || weights.weight.type == GGML_TYPE_F16) &&
      config.kernel_depth == 3 && config.stride_depth == 1 &&
      config.padding_depth == 1 && config.dilation_depth == 1 &&
      config.stride_height == 1 && config.stride_width == 1 &&
      config.dilation_height == 1 && config.dilation_width == 1 &&
      config.cuda_large_shape_lowering &&
      rows > kMaxCudaFastPathConvRows && input.shape.dims[1] > 1;
  if (can_split_cached_conv) {
    const int64_t max_frames =
        std::max<int64_t>(1, kMaxCudaFastPathConvRows / (input.shape.dims[2] * input.shape.dims[3]));
    std::vector<engine::core::TensorValue> chunks;
    chunks.reserve(static_cast<size_t>((input.shape.dims[1] + max_frames - 1) / max_frames));
    engine::core::TensorValue local_previous_storage;
    const engine::core::TensorValue *local_previous = previous;
    for (int64_t start = 0; start < input.shape.dims[1]; start += max_frames) {
      const int64_t count = std::min<int64_t>(max_frames, input.shape.dims[1] - start);
      const auto chunk = engine::modules::SliceModule({1, start, count}).build(ctx, input);
      chunks.push_back(WanCausalConv3dModule(config).build(ctx, chunk, weights, local_previous));
      local_previous_storage = make_two_frame_conv_cache(ctx, chunk, local_previous, false);
      local_previous = &local_previous_storage;
    }
    auto output = chunks.front();
    for (size_t i = 1; i < chunks.size(); ++i) {
      output = engine::modules::ConcatModule({1, config.cuda_fast_lowering}).build(ctx, output, chunks[i]);
    }
    push_cache_output(ctx, cache, make_two_frame_conv_cache(
                                  ctx, input, previous,
                                  cache.raw_single_frame_inputs));
    return output;
  }
  auto output = WanCausalConv3dModule(config).build(ctx, input, weights, previous);
  push_cache_output(ctx, cache, make_two_frame_conv_cache(
                                    ctx, input, previous,
                                    cache.raw_single_frame_inputs));
  return output;
}

} // namespace

WanVideoVAELatentScaleModule::WanVideoVAELatentScaleModule(
    WanVideoVAEConfig config, WanVideoVAELatentTransform transform)
    : config_(std::move(config)), transform_(transform) {
  if (config_.latent_channels <= 0) {
    throw std::runtime_error("Wan VAE latent channels must be positive");
  }
}

engine::core::TensorValue WanVideoVAELatentScaleModule::build(
    engine::core::ModuleBuildContext &ctx,
    const engine::core::TensorValue &input,
    const engine::core::TensorValue &mean,
    const engine::core::TensorValue &std) const {
  engine::core::validate_rank_between(input, 4, 4, "Wan VAE latent input");
  if (input.shape.dims[0] != config_.latent_channels) {
    throw std::runtime_error("Wan VAE latent input channel count mismatch");
  }
  engine::core::validate_shape(
      mean, engine::core::TensorShape::from_dims({config_.latent_channels}),
      "Wan VAE latent mean");
  engine::core::validate_shape(
      std, engine::core::TensorShape::from_dims({config_.latent_channels}),
      "Wan VAE latent std");

  const auto input_f32 = ensure_f32(ctx, input);
  const auto mean_view = engine::core::reshape_tensor(
      ctx, ensure_f32(ctx, mean),
      engine::core::TensorShape::from_dims({config_.latent_channels, 1, 1, 1}));
  const auto std_view = engine::core::reshape_tensor(
      ctx, ensure_f32(ctx, std),
      engine::core::TensorShape::from_dims({config_.latent_channels, 1, 1, 1}));
  const auto mean_full = engine::core::wrap_tensor(
      ggml_repeat(ctx.ggml, mean_view.tensor, input_f32.tensor),
      input_f32.shape, GGML_TYPE_F32);
  const auto std_full = engine::core::wrap_tensor(
      ggml_repeat(ctx.ggml, std_view.tensor, input_f32.tensor), input_f32.shape,
      GGML_TYPE_F32);

  if (transform_ == WanVideoVAELatentTransform::Encode) {
    const auto centered = engine::core::wrap_tensor(
        ggml_sub(ctx.ggml, input_f32.tensor, mean_full.tensor), input_f32.shape,
        GGML_TYPE_F32);
    return engine::modules::MulModule{}.build(ctx, centered, std_full);
  }

  const auto unscaled = engine::core::wrap_tensor(
      ggml_div(ctx.ggml, input_f32.tensor, std_full.tensor), input_f32.shape,
      GGML_TYPE_F32);
  return engine::modules::AddModule{}.build(ctx, unscaled, mean_full);
}

WanVideoVAEConfig WanVideoVAELatentScaleModule::s2v_14b_config() {
  return WanVideoVAEConfig{};
}

WanCausalConv3dModule::WanCausalConv3dModule(
    engine::modules::Conv3dConfig config)
    : config_(config) {
  if (config_.padding_depth < 0 || config_.padding_height < 0 ||
      config_.padding_width < 0) {
    throw std::runtime_error("Wan causal Conv3D padding must be non-negative");
  }
}

engine::core::TensorValue WanCausalConv3dModule::build(
    engine::core::ModuleBuildContext &ctx,
    const engine::core::TensorValue &input,
    const engine::modules::Conv3dWeights &weights) const {
  return build(ctx, input, weights, nullptr);
}

engine::core::TensorValue
WanCausalConv3dModule::build(engine::core::ModuleBuildContext &ctx,
                             const engine::core::TensorValue &input,
                             const engine::modules::Conv3dWeights &weights,
                             const engine::core::TensorValue *cache) const {
  if (config_.use_bias && !weights.bias.has_value()) {
    throw std::runtime_error("bias is required when Wan causal Conv3D uses bias");
  }
  return wan_vae_add_3d_bias(
      ctx, wan_vae_causal_conv3d_linear(ctx, input, weights, config_, cache),
      config_.out_channels, config_.use_bias ? weights.bias : std::nullopt,
      config_.cuda_fast_lowering);
}

WanVideoRMSNorm3dModule::WanVideoRMSNorm3dModule(WanVideoRMSNorm3dConfig config)
    : config_(config) {
  if (config_.channels <= 0) {
    throw std::runtime_error("Wan video RMSNorm channels must be positive");
  }
  if (config_.eps < 0.0F) {
    throw std::runtime_error("Wan video RMSNorm eps must not be negative");
  }
}

engine::core::TensorValue
WanVideoRMSNorm3dModule::build(engine::core::ModuleBuildContext &ctx,
                               const engine::core::TensorValue &input,
                               const WanVideoRMSNorm3dWeights &weights) const {
  engine::core::validate_shape(input,
                               engine::core::TensorShape::from_dims(
                                   {config_.channels, input.shape.dims[1],
                                    input.shape.dims[2], input.shape.dims[3]}),
                               "Wan video RMSNorm input");
  engine::core::validate_shape(
      weights.gamma, engine::core::TensorShape::from_dims({config_.channels}),
      "Wan video RMSNorm gamma");

  const auto input_f32 = ensure_f32(ctx, input);
  if (use_cuda_wan_vae_fast_path(ctx, config_.cuda_fast_lowering)) {
    auto contiguous =
        engine::core::ensure_backend_addressable_layout(ctx, input_f32);
    auto *raw = ggml_rms_norm_channels(ctx.ggml, contiguous.tensor,
                                       ensure_f32(ctx, weights.gamma).tensor,
                                       config_.eps);
    ggml_rms_norm_channels_set_lowering(raw, GGML_RMS_NORM_CHANNELS_LOWERING_CUDA_COALESCED);
    return engine::core::wrap_tensor(raw, input.shape, GGML_TYPE_F32);
  }
  auto transposed =
      engine::modules::TransposeModule({{3, 1, 2, 0}, 4}).build(ctx, input_f32);
  transposed = engine::core::ensure_backend_addressable_layout(ctx, transposed);
  auto normalized = engine::core::wrap_tensor(
      ggml_rms_norm(ctx.ggml, transposed.tensor, config_.eps),
      transposed.shape, GGML_TYPE_F32);
  normalized =
      engine::modules::TransposeModule({{3, 1, 2, 0}, 4}).build(ctx, normalized);
  normalized = engine::core::ensure_backend_addressable_layout(ctx, normalized);
  const auto gamma = engine::core::reshape_tensor(
      ctx, ensure_f32(ctx, weights.gamma),
      engine::core::TensorShape::from_dims({config_.channels, 1, 1, 1}));
  return engine::core::wrap_tensor(
      ggml_mul(ctx.ggml, normalized.tensor,
               ggml_repeat(ctx.ggml, gamma.tensor, normalized.tensor)),
      normalized.shape, GGML_TYPE_F32);
}

WanVideoVAEResidualBlockModule::WanVideoVAEResidualBlockModule(
    WanVideoVAEResidualBlockConfig config)
    : config_(config) {
  if (config_.in_channels <= 0 || config_.out_channels <= 0 ||
      config_.kernel_size <= 0) {
    throw std::runtime_error(
        "Wan VAE residual block dimensions must be positive");
  }
}

engine::core::TensorValue WanVideoVAEResidualBlockModule::build(
    engine::core::ModuleBuildContext &ctx,
    const engine::core::TensorValue &input,
    const WanVideoVAEResidualBlockWeights &weights) const {
  engine::core::validate_shape(input,
                               engine::core::TensorShape::from_dims(
                                   {config_.in_channels, input.shape.dims[1],
                                    input.shape.dims[2], input.shape.dims[3]}),
                               "Wan VAE residual block input");
  engine::core::TensorValue shortcut = input;
  if (config_.in_channels != config_.out_channels) {
    if (!weights.shortcut.has_value()) {
      throw std::runtime_error("Wan VAE residual block requires shortcut "
                               "weights when channel counts differ");
    }
    shortcut = WanCausalConv3dModule({config_.in_channels, config_.out_channels,
                                      1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, true,
                                      config_.cuda_fast_lowering,
                                      config_.cuda_large_shape_lowering,
                                      config_.cuda_tile_f16_accum_output_lowering})
                   .build(ctx, input, *weights.shortcut);
  }

  auto hidden = wan_vae_silu_rms_norm3d(
      ctx, input, weights.norm1,
      {config_.in_channels, config_.rms_norm_eps, config_.cuda_fast_lowering});
  hidden = wan_vae_causal_conv3d_linear(
      ctx, hidden, weights.conv1,
      {config_.in_channels, config_.out_channels, config_.kernel_size,
       config_.kernel_size, config_.kernel_size, 1, 1, 1,
       static_cast<int>(config_.kernel_size / 2),
       static_cast<int>(config_.kernel_size / 2),
       static_cast<int>(config_.kernel_size / 2), 1, 1, 1, true,
       config_.cuda_fast_lowering,
       config_.cuda_large_shape_lowering,
       config_.cuda_tile_f16_accum_output_lowering},
      nullptr);
  hidden = wan_vae_silu_rms_norm3d_add_bias(
      ctx, hidden, weights.conv1.bias, weights.norm2,
      {config_.out_channels, config_.rms_norm_eps, config_.cuda_fast_lowering});
  hidden = WanCausalConv3dModule(
               {config_.out_channels, config_.out_channels, config_.kernel_size,
                config_.kernel_size, config_.kernel_size, 1, 1, 1,
                static_cast<int>(config_.kernel_size / 2),
                static_cast<int>(config_.kernel_size / 2),
                static_cast<int>(config_.kernel_size / 2), 1, 1, 1, true,
                config_.cuda_fast_lowering,
                config_.cuda_large_shape_lowering,
                config_.cuda_tile_f16_accum_output_lowering})
               .build(ctx, hidden, weights.conv2);
  return engine::modules::AddModule{}.build(ctx, hidden, shortcut);
}

engine::core::TensorValue WanVideoVAEResidualBlockModule::build_cached(
    engine::core::ModuleBuildContext &ctx,
    const engine::core::TensorValue &input,
    const WanVideoVAEResidualBlockWeights &weights,
    WanVideoVAECacheBuildState &cache) const {
  engine::core::validate_shape(input,
                               engine::core::TensorShape::from_dims(
                                   {config_.in_channels, input.shape.dims[1],
                                    input.shape.dims[2], input.shape.dims[3]}),
                               "Wan VAE residual block input");
  engine::core::TensorValue shortcut = input;
  if (config_.in_channels != config_.out_channels) {
    if (!weights.shortcut.has_value()) {
      throw std::runtime_error("Wan VAE residual block requires shortcut "
                               "weights when channel counts differ");
    }
    shortcut = WanCausalConv3dModule({config_.in_channels, config_.out_channels,
                                      1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, true,
                                      config_.cuda_fast_lowering,
                                      config_.cuda_large_shape_lowering,
                                      config_.cuda_tile_f16_accum_output_lowering})
                   .build(ctx, input, *weights.shortcut);
  }

  auto hidden = wan_vae_silu_rms_norm3d(
      ctx, input, weights.norm1,
      {config_.in_channels, config_.rms_norm_eps, config_.cuda_fast_lowering});
  const auto conv1_input = hidden;
  const auto *conv1_cache = next_cache_input(cache);
  hidden = wan_vae_causal_conv3d_linear(
      ctx, hidden, weights.conv1,
      {config_.in_channels, config_.out_channels, config_.kernel_size,
       config_.kernel_size, config_.kernel_size, 1, 1, 1,
       static_cast<int>(config_.kernel_size / 2),
       static_cast<int>(config_.kernel_size / 2),
       static_cast<int>(config_.kernel_size / 2), 1, 1, 1, true,
       config_.cuda_fast_lowering,
       config_.cuda_large_shape_lowering,
       config_.cuda_tile_f16_accum_output_lowering},
      conv1_cache);
  push_cache_output(ctx, cache,
                    make_two_frame_conv_cache(ctx, conv1_input, conv1_cache,
                                              cache.raw_single_frame_inputs));
  hidden = wan_vae_silu_rms_norm3d_add_bias(
      ctx, hidden, weights.conv1.bias, weights.norm2,
      {config_.out_channels, config_.rms_norm_eps, config_.cuda_fast_lowering});
  hidden = build_cached_causal_conv3d(
      ctx, hidden, weights.conv2,
      {config_.out_channels, config_.out_channels, config_.kernel_size,
       config_.kernel_size, config_.kernel_size, 1, 1, 1,
       static_cast<int>(config_.kernel_size / 2),
       static_cast<int>(config_.kernel_size / 2),
       static_cast<int>(config_.kernel_size / 2), 1, 1, 1, true,
       config_.cuda_fast_lowering,
       config_.cuda_large_shape_lowering,
       config_.cuda_tile_f16_accum_output_lowering},
      cache);
  return engine::modules::AddModule{}.build(ctx, hidden, shortcut);
}

WanVideoVAEResampleModule::WanVideoVAEResampleModule(
    WanVideoVAEResampleConfig config)
    : config_(config) {
  if (config_.channels <= 0) {
    throw std::runtime_error("Wan VAE resample channels must be positive");
  }
  if ((config_.mode == WanVideoVAEResampleMode::Upsample2d ||
       config_.mode == WanVideoVAEResampleMode::Upsample3d) &&
      config_.channels % 2 != 0) {
    throw std::runtime_error("Wan VAE upsample requires an even channel count");
  }
}

engine::core::TensorValue WanVideoVAEResampleModule::build_initial(
    engine::core::ModuleBuildContext &ctx,
    const engine::core::TensorValue &input,
    const WanVideoVAEResampleWeights &weights) const {
  engine::core::validate_shape(input,
                               engine::core::TensorShape::from_dims(
                                   {config_.channels, input.shape.dims[1],
                                    input.shape.dims[2], input.shape.dims[3]}),
                               "Wan VAE resample input");
  if (config_.mode == WanVideoVAEResampleMode::None) {
    return input;
  }

  auto frame_batch = video_to_frame_batch(ctx, input);
  if (config_.mode == WanVideoVAEResampleMode::Upsample2d ||
      config_.mode == WanVideoVAEResampleMode::Upsample3d) {
    frame_batch = engine::modules::NearestUpsample2dModule(
                      {
                          frame_batch.shape.dims[2] * 2,
                          frame_batch.shape.dims[3] * 2,
                      })
                      .build(ctx, frame_batch);
    frame_batch =
        wan_vae_conv2d(ctx, frame_batch, weights.spatial,
                       {config_.channels, config_.channels / 2, 3, 3, 1, 1, 1,
                        1, 1, 1, true, config_.cuda_fast_lowering,
                        config_.cuda_large_shape_lowering,
                        config_.cuda_tile_f16_accum_output_lowering});
    return frame_batch_to_video(ctx, frame_batch);
  }

  frame_batch =
      engine::modules::Pad2dModule({0, 1, 0, 1}).build(ctx, frame_batch);
  frame_batch =
      wan_vae_conv2d(ctx, frame_batch, weights.spatial,
                     {config_.channels, config_.channels, 3, 3, 2, 2, 0, 0, 1,
                      1, true, config_.cuda_fast_lowering,
                      config_.cuda_large_shape_lowering,
                      config_.cuda_tile_f16_accum_output_lowering});
  return frame_batch_to_video(ctx, frame_batch);
}

engine::core::TensorValue WanVideoVAEResampleModule::build_cached(
    engine::core::ModuleBuildContext &ctx,
    const engine::core::TensorValue &input,
    const WanVideoVAEResampleWeights &weights,
    WanVideoVAECacheBuildState &cache) const {
  engine::core::validate_shape(input,
                               engine::core::TensorShape::from_dims(
                                   {config_.channels, input.shape.dims[1],
                                    input.shape.dims[2], input.shape.dims[3]}),
                               "Wan VAE resample input");
  if (config_.mode != WanVideoVAEResampleMode::Downsample3d) {
    if (config_.mode != WanVideoVAEResampleMode::Upsample3d) {
      return build_initial(ctx, input, weights);
    }
    if (!weights.time.has_value()) {
      throw std::runtime_error("Wan VAE upsample3d requires time_conv weights");
    }
    const auto *previous = next_cache_input(cache);
    auto temporal = input;
    if (previous == nullptr) {
      push_cache_output(ctx, cache,
                        make_two_frame_conv_cache(ctx, zero_like(ctx, input),
                                                  nullptr,
                                                  cache.raw_single_frame_inputs));
    } else {
      temporal =
          WanCausalConv3dModule({config_.channels, config_.channels * 2, 3, 1,
                                 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, true,
                                 config_.cuda_fast_lowering,
                                 config_.cuda_large_shape_lowering,
                                 config_.cuda_tile_f16_accum_output_lowering})
              .build(ctx, input, *weights.time, previous);
      push_cache_output(ctx, cache,
                        make_two_frame_conv_cache(
                            ctx, input, previous,
                            cache.raw_single_frame_inputs));
      auto first =
          engine::modules::SliceModule({0, 0, config_.channels})
              .build(ctx, temporal);
      auto second =
          engine::modules::SliceModule({0, config_.channels, config_.channels})
              .build(ctx, temporal);
      if (temporal.shape.dims[1] == 1) {
        temporal = engine::modules::ConcatModule({1, config_.cuda_fast_lowering}).build(ctx, first, second);
      } else {
        std::vector<engine::core::TensorValue> frames;
        frames.reserve(static_cast<size_t>(temporal.shape.dims[1] * 2));
        for (int64_t frame = 0; frame < temporal.shape.dims[1]; ++frame) {
          frames.push_back(engine::modules::SliceModule({1, frame, 1})
                               .build(ctx, first));
          frames.push_back(engine::modules::SliceModule({1, frame, 1})
                               .build(ctx, second));
        }
        temporal = frames.front();
        for (size_t i = 1; i < frames.size(); ++i) {
          temporal = engine::modules::ConcatModule({1, config_.cuda_fast_lowering}).build(ctx, temporal, frames[i]);
        }
      }
    }

    auto frame_batch = video_to_frame_batch(ctx, temporal);
    frame_batch = engine::modules::NearestUpsample2dModule(
                      {
                          frame_batch.shape.dims[2] * 2,
                          frame_batch.shape.dims[3] * 2,
                      })
                      .build(ctx, frame_batch);
    frame_batch =
        wan_vae_conv2d(ctx, frame_batch, weights.spatial,
                       {config_.channels, config_.channels / 2, 3, 3, 1, 1, 1,
                        1, 1, 1, true, config_.cuda_fast_lowering,
                        config_.cuda_large_shape_lowering,
                        config_.cuda_tile_f16_accum_output_lowering});
    return frame_batch_to_video(ctx, frame_batch);
  }
  if (!weights.time.has_value()) {
    throw std::runtime_error("Wan VAE downsample3d requires time_conv weights");
  }

  auto frame_batch = video_to_frame_batch(ctx, input);
  frame_batch =
      engine::modules::Pad2dModule({0, 1, 0, 1}).build(ctx, frame_batch);
  frame_batch =
      wan_vae_conv2d(ctx, frame_batch, weights.spatial,
                     {config_.channels, config_.channels, 3, 3, 2, 2, 0, 0, 1,
                      1, true, config_.cuda_fast_lowering,
                      config_.cuda_large_shape_lowering,
                      config_.cuda_tile_f16_accum_output_lowering});
  auto spatial = frame_batch_to_video(ctx, frame_batch);
  const auto *previous = next_cache_input(cache);
  auto output = spatial;
  if (previous != nullptr) {
      const auto previous_spatial_type = previous->type == spatial.type
          ? *previous
          : engine::core::wrap_tensor(ggml_cast(ctx.ggml, previous->tensor, spatial.type), previous->shape, spatial.type);
      output = WanCausalConv3dModule({config_.channels, config_.channels, 3, 1,
                                      1, 2, 1, 1, 0, 0, 0, 1, 1, 1, true,
                                      config_.cuda_fast_lowering,
                                      config_.cuda_large_shape_lowering,
                                      config_.cuda_tile_f16_accum_output_lowering})
                 .build(ctx,
                        engine::modules::ConcatModule({1, config_.cuda_fast_lowering}).build(ctx, previous_spatial_type, spatial),
                        *weights.time);
  }
  push_cache_output(ctx, cache, make_one_frame_cache(ctx, spatial));
  return output;
}

WanVideoVAEAttentionBlockModule::WanVideoVAEAttentionBlockModule(
    WanVideoVAEAttentionBlockConfig config)
    : config_(config) {
  if (config_.channels <= 0) {
    throw std::runtime_error("Wan VAE attention channels must be positive");
  }
}

engine::core::TensorValue WanVideoVAEAttentionBlockModule::build(
    engine::core::ModuleBuildContext &ctx,
    const engine::core::TensorValue &input,
    const WanVideoVAEAttentionBlockWeights &weights) const {
  engine::core::validate_shape(input,
                               engine::core::TensorShape::from_dims(
                                   {config_.channels, input.shape.dims[1],
                                    input.shape.dims[2], input.shape.dims[3]}),
                               "Wan VAE attention input");
  const int64_t frames = input.shape.dims[1];
  const int64_t height = input.shape.dims[2];
  const int64_t width = input.shape.dims[3];
  const int64_t spatial = height * width;

  auto hidden =
      WanVideoRMSNorm3dModule({config_.channels, config_.rms_norm_eps, config_.cuda_fast_lowering})
          .build(ctx, input, weights.norm);
  hidden = video_to_frame_batch(ctx, hidden);
  auto qkv =
      wan_vae_conv2d(ctx, hidden, weights.qkv,
                     {config_.channels, 3 * config_.channels, 1, 1, 1, 1, 0, 0,
                      1, 1, true, config_.cuda_fast_lowering,
                      config_.cuda_large_shape_lowering,
                      config_.cuda_tile_f16_accum_output_lowering});
  auto q = engine::modules::SliceModule({1, 0, config_.channels}).build(ctx, qkv);
  auto k = engine::modules::SliceModule({1, config_.channels, config_.channels}).build(ctx, qkv);
  auto v = engine::modules::SliceModule({1, 2 * config_.channels, config_.channels}).build(ctx, qkv);
  q = engine::core::reshape_tensor(
      ctx, engine::core::ensure_backend_addressable_layout(ctx, q),
      engine::core::TensorShape::from_dims({frames, config_.channels, spatial}));
  k = engine::core::reshape_tensor(
      ctx, engine::core::ensure_backend_addressable_layout(ctx, k),
      engine::core::TensorShape::from_dims({frames, config_.channels, spatial}));
  v = engine::core::reshape_tensor(
      ctx, engine::core::ensure_backend_addressable_layout(ctx, v),
      engine::core::TensorShape::from_dims({frames, config_.channels, spatial}));
  q = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, q);
  v = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, v);
  auto scores = engine::modules::MatMulModule{}.build(
      ctx, engine::core::ensure_backend_addressable_layout(ctx, q),
      engine::core::ensure_backend_addressable_layout(ctx, k));
  scores = engine::core::wrap_tensor(
      ggml_scale(ctx.ggml, scores.tensor,
                 1.0F / std::sqrt(static_cast<float>(config_.channels))),
      scores.shape, GGML_TYPE_F32);
  auto probs = engine::modules::SoftmaxModule{}.build(ctx, scores);
  auto attended = engine::modules::MatMulModule{}.build(
      ctx, engine::core::ensure_backend_addressable_layout(ctx, probs),
      engine::core::ensure_backend_addressable_layout(ctx, v));
  attended =
      engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, attended);
  attended = engine::core::reshape_tensor(
      ctx, engine::core::ensure_backend_addressable_layout(ctx, attended),
      engine::core::TensorShape::from_dims(
          {frames, config_.channels, height, width}));
  attended =
      wan_vae_conv2d(ctx, attended, weights.proj,
                     {config_.channels, config_.channels, 1, 1, 1, 1, 0, 0, 1,
                      1, true, config_.cuda_fast_lowering,
                      config_.cuda_large_shape_lowering,
                      config_.cuda_tile_f16_accum_output_lowering});
  return engine::modules::AddModule{}.build(
      ctx, frame_batch_to_video(ctx, attended), input);
}

WanVideoVAEEncoderModule::WanVideoVAEEncoderModule(
    WanVideoVAEArchitectureConfig config)
    : config_(std::move(config)) {
  validate_architecture_config(config_);
}

engine::core::TensorValue WanVideoVAEEncoderModule::build_initial(
    engine::core::ModuleBuildContext &ctx,
    const engine::core::TensorValue &input,
    const WanVideoVAEEncoderWeights &weights) const {
  engine::core::validate_shape(
      input,
      engine::core::TensorShape::from_dims(
          {3, input.shape.dims[1], input.shape.dims[2], input.shape.dims[3]}),
      "Wan VAE encoder input");
  if (weights.stages.size() != config_.dim_mult.size()) {
    throw std::runtime_error("Wan VAE encoder stage count mismatch");
  }
  const auto dims = encoder_dims(config_);
  auto hidden = WanCausalConv3dModule(
                    {3, dims[0], 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, true,
                     config_.use_cuda_fast_lowerings,
                     config_.use_cuda_large_shape_lowerings,
                     config_.use_cuda_tile_f16_accum_output_lowering})
                    .build(ctx, input, weights.conv_in);

  float scale = 1.0F;
  for (size_t stage_index = 0; stage_index < config_.dim_mult.size();
       ++stage_index) {
    const auto &stage = weights.stages[stage_index];
    if (stage.blocks.size() != static_cast<size_t>(config_.num_res_blocks)) {
      throw std::runtime_error("Wan VAE encoder residual block count mismatch");
    }
    int64_t in_channels = dims[stage_index];
    const int64_t out_channels = dims[stage_index + 1];
    const bool use_attention = has_attention_at_scale(config_, scale);
    for (const auto &block : stage.blocks) {
      hidden = WanVideoVAEResidualBlockModule(
                   {in_channels, out_channels, 3, config_.rms_norm_eps,
                    config_.use_cuda_fast_lowerings,
                    config_.use_cuda_large_shape_lowerings,
                    config_.use_cuda_tile_f16_accum_output_lowering})
                   .build(ctx, hidden, block.residual);
      validate_attention_weight_presence(block.attention, use_attention);
      if (use_attention) {
        hidden = WanVideoVAEAttentionBlockModule(
                     {out_channels, config_.rms_norm_eps,
                      config_.use_cuda_fast_lowerings,
                      config_.use_cuda_large_shape_lowerings,
                      config_.use_cuda_tile_f16_accum_output_lowering})
                     .build(ctx, hidden, *block.attention);
      }
      in_channels = out_channels;
    }
    if (stage_index + 1 != config_.dim_mult.size()) {
      if (!stage.resample.has_value()) {
        throw std::runtime_error(
            "Wan VAE encoder stage requires resample weights");
      }
      hidden = WanVideoVAEResampleModule(
                   {out_channels, downsample_mode(config_, stage_index),
                    config_.use_cuda_fast_lowerings,
                    config_.use_cuda_large_shape_lowerings,
                    config_.use_cuda_tile_f16_accum_output_lowering})
                   .build_initial(ctx, hidden, *stage.resample);
      scale /= 2.0F;
    } else if (stage.resample.has_value()) {
      throw std::runtime_error(
          "Wan VAE encoder final stage must not have resample weights");
    }
  }

  const int64_t channels = dims.back();
  hidden = WanVideoVAEResidualBlockModule(
               {channels, channels, 3, config_.rms_norm_eps,
                config_.use_cuda_fast_lowerings,
                config_.use_cuda_large_shape_lowerings,
                config_.use_cuda_tile_f16_accum_output_lowering})
               .build(ctx, hidden, weights.middle1);
  hidden = WanVideoVAEAttentionBlockModule({channels, config_.rms_norm_eps,
                                            config_.use_cuda_fast_lowerings,
                                            config_.use_cuda_large_shape_lowerings,
                                            config_.use_cuda_tile_f16_accum_output_lowering})
               .build(ctx, hidden, weights.middle_attention);
  hidden = WanVideoVAEResidualBlockModule(
               {channels, channels, 3, config_.rms_norm_eps,
                config_.use_cuda_fast_lowerings,
                config_.use_cuda_large_shape_lowerings,
                config_.use_cuda_tile_f16_accum_output_lowering})
               .build(ctx, hidden, weights.middle2);
  hidden = wan_vae_silu_rms_norm3d(
      ctx, hidden, weights.head_norm,
      {channels, config_.rms_norm_eps, config_.use_cuda_fast_lowerings});
  hidden = WanCausalConv3dModule({channels, 2 * config_.latent_channels, 3, 3,
                                  3, 1, 1, 1, 1, 1, 1, 1, 1, 1, true,
                                  config_.use_cuda_fast_lowerings,
                                  config_.use_cuda_large_shape_lowerings,
                                  config_.use_cuda_tile_f16_accum_output_lowering})
               .build(ctx, hidden, weights.head);
  hidden = WanCausalConv3dModule({2 * config_.latent_channels,
                                  2 * config_.latent_channels, 1, 1, 1, 1, 1, 1,
                                  0, 0, 0, 1, 1, 1, true,
                                  config_.use_cuda_fast_lowerings,
                                  config_.use_cuda_large_shape_lowerings,
                                  config_.use_cuda_tile_f16_accum_output_lowering})
               .build(ctx, hidden, weights.quant_conv);
  auto mu = engine::modules::SliceModule({0, 0, config_.latent_channels})
                .build(ctx, hidden);
  if (!config_.scale_encoded_latents) {
    return mu;
  }
  return WanVideoVAELatentScaleModule(
             WanVideoVAEConfig{config_.latent_channels},
             WanVideoVAELatentTransform::Encode)
      .build(ctx, mu, weights.latent_mean, weights.latent_std);
}

engine::core::TensorValue WanVideoVAEEncoderModule::build_cached(
    engine::core::ModuleBuildContext &ctx,
    const engine::core::TensorValue &input,
    const WanVideoVAEEncoderWeights &weights,
    WanVideoVAECacheBuildState &cache) const {
  engine::core::validate_shape(
      input,
      engine::core::TensorShape::from_dims(
          {3, input.shape.dims[1], input.shape.dims[2], input.shape.dims[3]}),
      "Wan VAE encoder input");
  if (weights.stages.size() != config_.dim_mult.size()) {
    throw std::runtime_error("Wan VAE encoder stage count mismatch");
  }
  const auto dims = encoder_dims(config_);
  auto hidden = build_cached_causal_conv3d(
      ctx, input, weights.conv_in,
      {3, dims[0], 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, true,
       config_.use_cuda_fast_lowerings,
       config_.use_cuda_large_shape_lowerings,
       config_.use_cuda_tile_f16_accum_output_lowering},
      cache);

  float scale = 1.0F;
  for (size_t stage_index = 0; stage_index < config_.dim_mult.size();
       ++stage_index) {
    const auto &stage = weights.stages[stage_index];
    if (stage.blocks.size() != static_cast<size_t>(config_.num_res_blocks)) {
      throw std::runtime_error("Wan VAE encoder residual block count mismatch");
    }
    int64_t in_channels = dims[stage_index];
    const int64_t out_channels = dims[stage_index + 1];
    const bool use_attention = has_attention_at_scale(config_, scale);
    for (const auto &block : stage.blocks) {
      hidden = WanVideoVAEResidualBlockModule(
                   {in_channels, out_channels, 3, config_.rms_norm_eps,
                    config_.use_cuda_fast_lowerings,
                    config_.use_cuda_large_shape_lowerings,
                    config_.use_cuda_tile_f16_accum_output_lowering})
                   .build_cached(ctx, hidden, block.residual, cache);
      validate_attention_weight_presence(block.attention, use_attention);
      if (use_attention) {
        hidden = WanVideoVAEAttentionBlockModule(
                     {out_channels, config_.rms_norm_eps,
                      config_.use_cuda_fast_lowerings,
                      config_.use_cuda_large_shape_lowerings,
                      config_.use_cuda_tile_f16_accum_output_lowering})
                     .build(ctx, hidden, *block.attention);
      }
      in_channels = out_channels;
    }
    if (stage_index + 1 != config_.dim_mult.size()) {
      if (!stage.resample.has_value()) {
        throw std::runtime_error(
            "Wan VAE encoder stage requires resample weights");
      }
      hidden = WanVideoVAEResampleModule(
                   {out_channels, downsample_mode(config_, stage_index),
                    config_.use_cuda_fast_lowerings,
                    config_.use_cuda_large_shape_lowerings,
                    config_.use_cuda_tile_f16_accum_output_lowering})
                   .build_cached(ctx, hidden, *stage.resample, cache);
      scale /= 2.0F;
    } else if (stage.resample.has_value()) {
      throw std::runtime_error(
          "Wan VAE encoder final stage must not have resample weights");
    }
  }

  const int64_t channels = dims.back();
  hidden = WanVideoVAEResidualBlockModule(
               {channels, channels, 3, config_.rms_norm_eps,
                config_.use_cuda_fast_lowerings,
                config_.use_cuda_large_shape_lowerings,
                config_.use_cuda_tile_f16_accum_output_lowering})
               .build_cached(ctx, hidden, weights.middle1, cache);
  hidden = WanVideoVAEAttentionBlockModule({channels, config_.rms_norm_eps,
                                            config_.use_cuda_fast_lowerings,
                                            config_.use_cuda_large_shape_lowerings,
                                            config_.use_cuda_tile_f16_accum_output_lowering})
               .build(ctx, hidden, weights.middle_attention);
  hidden = WanVideoVAEResidualBlockModule(
               {channels, channels, 3, config_.rms_norm_eps,
                config_.use_cuda_fast_lowerings,
                config_.use_cuda_large_shape_lowerings,
                config_.use_cuda_tile_f16_accum_output_lowering})
               .build_cached(ctx, hidden, weights.middle2, cache);
  hidden = wan_vae_silu_rms_norm3d(
      ctx, hidden, weights.head_norm,
      {channels, config_.rms_norm_eps, config_.use_cuda_fast_lowerings});
  hidden = build_cached_causal_conv3d(
      ctx, hidden, weights.head,
      {channels, 2 * config_.latent_channels, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1,
       1, true, config_.use_cuda_fast_lowerings,
       config_.use_cuda_large_shape_lowerings,
       config_.use_cuda_tile_f16_accum_output_lowering},
      cache);
  hidden = WanCausalConv3dModule({2 * config_.latent_channels,
                                  2 * config_.latent_channels, 1, 1, 1, 1, 1, 1,
                                  0, 0, 0, 1, 1, 1, true,
                                  config_.use_cuda_fast_lowerings,
                                  config_.use_cuda_large_shape_lowerings,
                                  config_.use_cuda_tile_f16_accum_output_lowering})
               .build(ctx, hidden, weights.quant_conv);
  auto mu = engine::modules::SliceModule({0, 0, config_.latent_channels})
                .build(ctx, hidden);
  if (!config_.scale_encoded_latents) {
    return mu;
  }
  return WanVideoVAELatentScaleModule(
             WanVideoVAEConfig{config_.latent_channels},
             WanVideoVAELatentTransform::Encode)
      .build(ctx, mu, weights.latent_mean, weights.latent_std);
}

WanVideoVAEDecoderModule::WanVideoVAEDecoderModule(
    WanVideoVAEArchitectureConfig config)
    : config_(std::move(config)) {
  validate_architecture_config(config_);
}

engine::core::TensorValue WanVideoVAEDecoderModule::build_initial(
    engine::core::ModuleBuildContext &ctx,
    const engine::core::TensorValue &input,
    const WanVideoVAEDecoderWeights &weights) const {
  engine::core::validate_shape(
      input,
      engine::core::TensorShape::from_dims(
          {config_.latent_channels, input.shape.dims[1], input.shape.dims[2],
           input.shape.dims[3]}),
      "Wan VAE decoder input");
  if (weights.stages.size() != config_.dim_mult.size()) {
    throw std::runtime_error("Wan VAE decoder stage count mismatch");
  }
  auto hidden = config_.scale_decoded_latents
                    ? WanVideoVAELatentScaleModule(WanVideoVAEConfig{config_.latent_channels},
                                                   WanVideoVAELatentTransform::Decode)
                          .build(ctx, input, weights.latent_mean, weights.latent_std)
                    : input;
  hidden =
      WanCausalConv3dModule({config_.latent_channels, config_.latent_channels,
                             1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, true,
                             config_.use_cuda_fast_lowerings,
                             config_.use_cuda_large_shape_lowerings,
                             config_.use_cuda_tile_f16_accum_output_lowering})
          .build(ctx, hidden, weights.quant_conv);

  const auto dims = decoder_dims(config_);
  hidden = WanCausalConv3dModule({config_.latent_channels, dims[0], 3, 3, 3, 1,
                                  1, 1, 1, 1, 1, 1, 1, 1, true,
                                  config_.use_cuda_fast_lowerings,
                                  config_.use_cuda_large_shape_lowerings,
                                  config_.use_cuda_tile_f16_accum_output_lowering})
               .build(ctx, hidden, weights.conv_in);

  hidden = WanVideoVAEResidualBlockModule(
               {dims[0], dims[0], 3, config_.rms_norm_eps,
                config_.use_cuda_fast_lowerings,
                config_.use_cuda_large_shape_lowerings,
                config_.use_cuda_tile_f16_accum_output_lowering})
               .build(ctx, hidden, weights.middle1);
  hidden = WanVideoVAEAttentionBlockModule({dims[0], config_.rms_norm_eps,
                                            config_.use_cuda_fast_lowerings,
                                            config_.use_cuda_large_shape_lowerings,
                                            config_.use_cuda_tile_f16_accum_output_lowering})
               .build(ctx, hidden, weights.middle_attention);
  hidden = WanVideoVAEResidualBlockModule(
               {dims[0], dims[0], 3, config_.rms_norm_eps,
                config_.use_cuda_fast_lowerings,
                config_.use_cuda_large_shape_lowerings,
                config_.use_cuda_tile_f16_accum_output_lowering})
               .build(ctx, hidden, weights.middle2);

  const auto scale_divisor_shift =
      static_cast<int>(std::max<size_t>(config_.dim_mult.size(), 2) - 2);
  float scale = 1.0F / static_cast<float>(1ll << scale_divisor_shift);
  for (size_t stage_index = 0; stage_index < config_.dim_mult.size();
       ++stage_index) {
    const auto &stage = weights.stages[stage_index];
    if (stage.blocks.size() !=
        static_cast<size_t>(config_.num_res_blocks + 1)) {
      throw std::runtime_error("Wan VAE decoder residual block count mismatch");
    }
    int64_t in_channels = dims[stage_index];
    if (stage_index == 1 || stage_index == 2 || stage_index == 3) {
      in_channels /= 2;
    }
    const int64_t out_channels = dims[stage_index + 1];
    const bool use_attention = has_attention_at_scale(config_, scale);
    for (const auto &block : stage.blocks) {
      hidden = WanVideoVAEResidualBlockModule(
                   {in_channels, out_channels, 3, config_.rms_norm_eps,
                    config_.use_cuda_fast_lowerings,
                    config_.use_cuda_large_shape_lowerings,
                    config_.use_cuda_tile_f16_accum_output_lowering})
                   .build(ctx, hidden, block.residual);
      validate_attention_weight_presence(block.attention, use_attention);
      if (use_attention) {
        hidden = WanVideoVAEAttentionBlockModule(
                     {out_channels, config_.rms_norm_eps,
                      config_.use_cuda_fast_lowerings,
                      config_.use_cuda_large_shape_lowerings,
                      config_.use_cuda_tile_f16_accum_output_lowering})
                     .build(ctx, hidden, *block.attention);
      }
      in_channels = out_channels;
    }
    if (stage_index + 1 != config_.dim_mult.size()) {
      if (!stage.resample.has_value()) {
        throw std::runtime_error(
            "Wan VAE decoder stage requires resample weights");
      }
      hidden = WanVideoVAEResampleModule(
                   {out_channels, upsample_mode(config_, stage_index),
                    config_.use_cuda_fast_lowerings,
                    config_.use_cuda_large_shape_lowerings,
                    config_.use_cuda_tile_f16_accum_output_lowering})
                   .build_initial(ctx, hidden, *stage.resample);
      scale *= 2.0F;
    } else if (stage.resample.has_value()) {
      throw std::runtime_error(
          "Wan VAE decoder final stage must not have resample weights");
    }
  }

  const int64_t channels = dims.back();
  hidden = wan_vae_silu_rms_norm3d(
      ctx, hidden, weights.head_norm,
      {channels, config_.rms_norm_eps, config_.use_cuda_fast_lowerings});
  return WanCausalConv3dModule(
             {channels, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, true,
              config_.use_cuda_fast_lowerings,
              config_.use_cuda_large_shape_lowerings,
              config_.use_cuda_tile_f16_accum_output_lowering})
      .build(ctx, hidden, weights.head);
}

engine::core::TensorValue WanVideoVAEDecoderModule::build_cached(
    engine::core::ModuleBuildContext &ctx,
    const engine::core::TensorValue &input,
    const WanVideoVAEDecoderWeights &weights,
    WanVideoVAECacheBuildState &cache) const {
  engine::core::validate_shape(
      input,
      engine::core::TensorShape::from_dims(
          {config_.latent_channels, input.shape.dims[1], input.shape.dims[2],
           input.shape.dims[3]}),
      "Wan VAE decoder input");
  if (weights.stages.size() != config_.dim_mult.size()) {
    throw std::runtime_error("Wan VAE decoder stage count mismatch");
  }
  auto hidden = config_.scale_decoded_latents
                    ? WanVideoVAELatentScaleModule(WanVideoVAEConfig{config_.latent_channels},
                                                   WanVideoVAELatentTransform::Decode)
                          .build(ctx, input, weights.latent_mean, weights.latent_std)
                    : input;
  hidden =
      WanCausalConv3dModule({config_.latent_channels, config_.latent_channels,
                             1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, true,
                             config_.use_cuda_fast_lowerings,
                             config_.use_cuda_large_shape_lowerings,
                             config_.use_cuda_tile_f16_accum_output_lowering})
          .build(ctx, hidden, weights.quant_conv);

  const auto dims = decoder_dims(config_);
  hidden = build_cached_causal_conv3d(
      ctx, hidden, weights.conv_in,
      {config_.latent_channels, dims[0], 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       true, config_.use_cuda_fast_lowerings,
       config_.use_cuda_large_shape_lowerings,
       config_.use_cuda_tile_f16_accum_output_lowering},
      cache);

  hidden = WanVideoVAEResidualBlockModule(
               {dims[0], dims[0], 3, config_.rms_norm_eps,
                config_.use_cuda_fast_lowerings,
                config_.use_cuda_large_shape_lowerings,
                config_.use_cuda_tile_f16_accum_output_lowering})
               .build_cached(ctx, hidden, weights.middle1, cache);
  hidden = WanVideoVAEAttentionBlockModule({dims[0], config_.rms_norm_eps,
                                            config_.use_cuda_fast_lowerings,
                                            config_.use_cuda_large_shape_lowerings,
                                            config_.use_cuda_tile_f16_accum_output_lowering})
               .build(ctx, hidden, weights.middle_attention);
  hidden = WanVideoVAEResidualBlockModule(
               {dims[0], dims[0], 3, config_.rms_norm_eps,
                config_.use_cuda_fast_lowerings,
                config_.use_cuda_large_shape_lowerings,
                config_.use_cuda_tile_f16_accum_output_lowering})
               .build_cached(ctx, hidden, weights.middle2, cache);

  const auto scale_divisor_shift =
      static_cast<int>(std::max<size_t>(config_.dim_mult.size(), 2) - 2);
  float scale = 1.0F / static_cast<float>(1ll << scale_divisor_shift);
  for (size_t stage_index = 0; stage_index < config_.dim_mult.size();
       ++stage_index) {
    const auto &stage = weights.stages[stage_index];
    if (stage.blocks.size() !=
        static_cast<size_t>(config_.num_res_blocks + 1)) {
      throw std::runtime_error("Wan VAE decoder residual block count mismatch");
    }
    int64_t in_channels = dims[stage_index];
    if (stage_index == 1 || stage_index == 2 || stage_index == 3) {
      in_channels /= 2;
    }
    const int64_t out_channels = dims[stage_index + 1];
    const bool use_attention = has_attention_at_scale(config_, scale);
    for (const auto &block : stage.blocks) {
      hidden = WanVideoVAEResidualBlockModule(
                   {in_channels, out_channels, 3, config_.rms_norm_eps,
                    config_.use_cuda_fast_lowerings,
                    config_.use_cuda_large_shape_lowerings,
                    config_.use_cuda_tile_f16_accum_output_lowering})
                   .build_cached(ctx, hidden, block.residual, cache);
      validate_attention_weight_presence(block.attention, use_attention);
      if (use_attention) {
        hidden = WanVideoVAEAttentionBlockModule(
                     {out_channels, config_.rms_norm_eps,
                      config_.use_cuda_fast_lowerings,
                      config_.use_cuda_large_shape_lowerings,
                      config_.use_cuda_tile_f16_accum_output_lowering})
                     .build(ctx, hidden, *block.attention);
      }
      in_channels = out_channels;
    }
    if (stage_index + 1 != config_.dim_mult.size()) {
      if (!stage.resample.has_value()) {
        throw std::runtime_error(
            "Wan VAE decoder stage requires resample weights");
      }
      hidden = WanVideoVAEResampleModule(
                   {out_channels, upsample_mode(config_, stage_index),
                    config_.use_cuda_fast_lowerings,
                    config_.use_cuda_large_shape_lowerings,
                    config_.use_cuda_tile_f16_accum_output_lowering})
                   .build_cached(ctx, hidden, *stage.resample, cache);
      scale *= 2.0F;
    } else if (stage.resample.has_value()) {
      throw std::runtime_error(
          "Wan VAE decoder final stage must not have resample weights");
    }
  }

  const int64_t channels = dims.back();
  hidden = wan_vae_silu_rms_norm3d(
      ctx, hidden, weights.head_norm,
      {channels, config_.rms_norm_eps, config_.use_cuda_fast_lowerings});
  return build_cached_causal_conv3d(
      ctx, hidden, weights.head,
      {channels, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, true,
       config_.use_cuda_fast_lowerings,
       config_.use_cuda_large_shape_lowerings,
       config_.use_cuda_tile_f16_accum_output_lowering}, cache);
}

} // namespace engine::codecs
