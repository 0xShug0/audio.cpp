#include "engine/community_models/outetts/dac.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace engine::models::outetts {
namespace {

struct GgmlContextDeleter {
  void operator()(ggml_context *ctx) const noexcept {
    if (ctx != nullptr) {
      ggml_free(ctx);
    }
  }
};

struct SnakeWeights {
  core::TensorValue alpha;
};
struct ConvWeights {
  modules::Conv1dWeights value;
  int64_t in_channels = 0;
  int64_t out_channels = 0;
  int64_t kernel = 0;
};
struct ConvTransposeWeights {
  modules::ConvTranspose1dWeights value;
  int64_t in_channels = 0;
  int64_t out_channels = 0;
  int64_t kernel = 0;
};
struct ResidualWeights {
  SnakeWeights snake1;
  ConvWeights conv1;
  SnakeWeights snake2;
  ConvWeights conv2;
};
struct DecoderBlockWeights {
  SnakeWeights snake;
  ConvTransposeWeights up;
  std::vector<ResidualWeights> residuals;
  int stride = 1;
};
struct EncoderBlockWeights {
  std::vector<ResidualWeights> residuals;
  SnakeWeights snake;
  ConvWeights down;
  int stride = 1;
};
struct HostQuantizerWeights {
  std::vector<float> codebook;
  std::vector<float> in_weight;
  std::vector<float> in_bias;
  std::vector<float> out_weight;
  std::vector<float> out_bias;
};
struct QuantizerWeights {
  core::TensorValue codebook;
  ConvWeights out_proj;
};
struct DacWeights {
  std::shared_ptr<core::BackendWeightStore> store;
  std::vector<QuantizerWeights> quantizers;
  ConvWeights first;
  std::vector<DecoderBlockWeights> blocks;
  SnakeWeights final_snake;
  ConvWeights final_conv;
  ConvWeights encoder_first;
  std::vector<EncoderBlockWeights> encoder_blocks;
  SnakeWeights encoder_final_snake;
  ConvWeights encoder_final_conv;
  std::vector<HostQuantizerWeights> host_quantizers;
};

std::vector<float> fold_weight_norm(const std::vector<float> &v,
                                    const std::vector<float> &g, int64_t dim0,
                                    int64_t dim1, int64_t kernel) {
  std::vector<float> out(v.size());
  for (int64_t i = 0; i < dim0; ++i) {
    const size_t base = static_cast<size_t>(i * dim1 * kernel);
    double norm2 = 0.0;
    for (int64_t j = 0; j < dim1 * kernel; ++j) {
      const double x = v[base + static_cast<size_t>(j)];
      norm2 += x * x;
    }
    const float scale =
        g[static_cast<size_t>(i)] / static_cast<float>(std::sqrt(norm2));
    for (int64_t j = 0; j < dim1 * kernel; ++j) {
      out[base + static_cast<size_t>(j)] =
          v[base + static_cast<size_t>(j)] * scale;
    }
  }
  return out;
}

ConvWeights load_conv(core::BackendWeightStore &store,
                      const assets::TensorSource &source,
                      const std::string &prefix, int64_t out_channels,
                      int64_t in_channels, int64_t kernel,
                      assets::TensorStorageType storage_type) {
  const auto v = source.require_f32(prefix + ".weight_v",
                                    {out_channels, in_channels, kernel});
  const auto g = source.require_f32(prefix + ".weight_g", {out_channels, 1, 1});
  ConvWeights out;
  out.in_channels = in_channels;
  out.out_channels = out_channels;
  out.kernel = kernel;
  out.value.weight = store.make_from_f32(
      core::TensorShape::from_dims({out_channels, in_channels, kernel}),
      storage_type, fold_weight_norm(v, g, out_channels, in_channels, kernel));
  out.value.bias =
      store.load_f32_tensor(source, prefix + ".bias", {out_channels});
  return out;
}

ConvTransposeWeights load_conv_transpose(
    core::BackendWeightStore &store, const assets::TensorSource &source,
    const std::string &prefix, int64_t in_channels, int64_t out_channels,
    int64_t kernel, assets::TensorStorageType storage_type) {
  const auto v = source.require_f32(prefix + ".weight_v",
                                    {in_channels, out_channels, kernel});
  const auto g = source.require_f32(prefix + ".weight_g", {in_channels, 1, 1});
  ConvTransposeWeights out;
  out.in_channels = in_channels;
  out.out_channels = out_channels;
  out.kernel = kernel;
  out.value.weight = store.make_from_f32(
      core::TensorShape::from_dims({in_channels, out_channels, kernel}),
      storage_type, fold_weight_norm(v, g, in_channels, out_channels, kernel));
  out.value.bias =
      store.load_f32_tensor(source, prefix + ".bias", {out_channels});
  return out;
}

SnakeWeights load_snake(core::BackendWeightStore &store,
                        const assets::TensorSource &source,
                        const std::string &name, int64_t channels) {
  return {store.make_from_f32(core::TensorShape::from_dims({channels}),
                              assets::TensorStorageType::F32,
                              source.require_f32(name, {1, channels, 1}))};
}

ResidualWeights load_residual(core::BackendWeightStore &store,
                              const assets::TensorSource &source,
                              const std::string &prefix, int64_t channels,
                              assets::TensorStorageType storage_type) {
  ResidualWeights out;
  out.snake1 = load_snake(store, source, prefix + ".block.0.alpha", channels);
  out.conv1 = load_conv(store, source, prefix + ".block.1", channels, channels,
                        7, storage_type);
  out.snake2 = load_snake(store, source, prefix + ".block.2.alpha", channels);
  out.conv2 = load_conv(store, source, prefix + ".block.3", channels, channels,
                        1, storage_type);
  return out;
}

DacWeights load_weights(const OuteTTSAssets &assets,
                        core::ExecutionContext &execution, size_t context_bytes,
                        assets::TensorStorageType storage_type) {
  const auto &source = *assets.dac_weights;
  DacWeights out;
  out.store = std::make_shared<core::BackendWeightStore>(
      execution.backend(), execution.backend_type(), "outetts.dac.weights",
      context_bytes);
  for (int i = 0; i < 2; ++i) {
    const std::string p = "quantizer.quantizers." + std::to_string(i);
    QuantizerWeights q;
    q.codebook =
        out.store->load_tensor(source, p + ".codebook.weight",
                               assets::TensorStorageType::F32, {1024, 8});
    q.out_proj = load_conv(*out.store, source, p + ".out_proj", 1024, 8, 1,
                           storage_type);
    out.quantizers.push_back(std::move(q));
  }
  out.first = load_conv(*out.store, source, "decoder.model.0", 1536, 1024, 7,
                        storage_type);
  const int strides[] = {8, 5, 4, 2};
  int64_t in_channels = 1536;
  for (int stage = 0; stage < 4; ++stage) {
    const int64_t out_channels = in_channels / 2;
    const std::string p = "decoder.model." + std::to_string(stage + 1);
    DecoderBlockWeights block;
    block.stride = strides[stage];
    block.snake =
        load_snake(*out.store, source, p + ".block.0.alpha", in_channels);
    block.up =
        load_conv_transpose(*out.store, source, p + ".block.1", in_channels,
                            out_channels, 2 * strides[stage], storage_type);
    for (int residual = 0; residual < 3; ++residual) {
      block.residuals.push_back(load_residual(
          *out.store, source, p + ".block." + std::to_string(residual + 2),
          out_channels, storage_type));
    }
    out.blocks.push_back(std::move(block));
    in_channels = out_channels;
  }
  out.final_snake = load_snake(*out.store, source, "decoder.model.5.alpha", 96);
  out.final_conv =
      load_conv(*out.store, source, "decoder.model.6", 1, 96, 7, storage_type);

  out.encoder_first =
      load_conv(*out.store, source, "encoder.block.0", 64, 1, 7, storage_type);
  const int encoder_strides[] = {2, 4, 5, 8};
  int64_t encoder_channels = 64;
  for (int stage = 0; stage < 4; ++stage) {
    const int64_t out_channels = encoder_channels * 2;
    const std::string p = "encoder.block." + std::to_string(stage + 1);
    EncoderBlockWeights block;
    block.stride = encoder_strides[stage];
    for (int residual_index = 0; residual_index < 3; ++residual_index) {
      block.residuals.push_back(load_residual(
          *out.store, source, p + ".block." + std::to_string(residual_index),
          encoder_channels, storage_type));
    }
    block.snake =
        load_snake(*out.store, source, p + ".block.3.alpha", encoder_channels);
    block.down =
        load_conv(*out.store, source, p + ".block.4", out_channels,
                  encoder_channels, 2 * encoder_strides[stage], storage_type);
    out.encoder_blocks.push_back(std::move(block));
    encoder_channels = out_channels;
  }
  out.encoder_final_snake =
      load_snake(*out.store, source, "encoder.block.5.alpha", 1024);
  out.encoder_final_conv = load_conv(*out.store, source, "encoder.block.6",
                                     1024, 1024, 3, storage_type);

  for (int i = 0; i < 2; ++i) {
    const std::string p = "quantizer.quantizers." + std::to_string(i);
    HostQuantizerWeights q;
    q.codebook = source.require_f32(p + ".codebook.weight", {1024, 8});
    q.in_weight = fold_weight_norm(
        source.require_f32(p + ".in_proj.weight_v", {8, 1024, 1}),
        source.require_f32(p + ".in_proj.weight_g", {8, 1, 1}), 8, 1024, 1);
    q.in_bias = source.require_f32(p + ".in_proj.bias", {8});
    q.out_weight = fold_weight_norm(
        source.require_f32(p + ".out_proj.weight_v", {1024, 8, 1}),
        source.require_f32(p + ".out_proj.weight_g", {1024, 1, 1}), 1024, 8, 1);
    q.out_bias = source.require_f32(p + ".out_proj.bias", {1024});
    out.host_quantizers.push_back(std::move(q));
  }
  out.store->upload();
  return out;
}

core::TensorValue snake(core::ModuleBuildContext &ctx,
                        const core::TensorValue &input,
                        const SnakeWeights &weights) {
  const int64_t channels = input.shape.dims[1];
  auto alpha = core::reshape_tensor(
      ctx, weights.alpha, core::TensorShape::from_dims({1, channels, 1}));
  alpha = core::wrap_tensor(ggml_repeat(ctx.ggml, alpha.tensor, input.tensor),
                            input.shape, GGML_TYPE_F32);
  auto ax = core::wrap_tensor(ggml_mul(ctx.ggml, input.tensor, alpha.tensor),
                              input.shape, GGML_TYPE_F32);
  auto s = core::wrap_tensor(ggml_sin(ctx.ggml, ax.tensor), input.shape,
                             GGML_TYPE_F32);
  auto s2 = core::wrap_tensor(ggml_mul(ctx.ggml, s.tensor, s.tensor),
                              input.shape, GGML_TYPE_F32);
  auto denom =
      core::wrap_tensor(ggml_scale_bias(ctx.ggml, alpha.tensor, 1.0F, 1.0e-9F),
                        input.shape, GGML_TYPE_F32);
  auto frac = core::wrap_tensor(ggml_div(ctx.ggml, s2.tensor, denom.tensor),
                                input.shape, GGML_TYPE_F32);
  return core::wrap_tensor(ggml_add(ctx.ggml, input.tensor, frac.tensor),
                           input.shape, GGML_TYPE_F32);
}

core::TensorValue conv(core::ModuleBuildContext &ctx,
                       const core::TensorValue &input,
                       const ConvWeights &weights, int padding,
                       int dilation = 1) {
  return modules::Conv1dModule({
                                   weights.in_channels,
                                   weights.out_channels,
                                   weights.kernel,
                                   1,
                                   padding,
                                   dilation,
                                   true,
                               })
      .build(ctx, input, weights.value);
}

core::TensorValue strided_conv(core::ModuleBuildContext &ctx,
                               const core::TensorValue &input,
                               const ConvWeights &weights, int stride,
                               int padding) {
  return modules::Conv1dModule({
                                   weights.in_channels,
                                   weights.out_channels,
                                   weights.kernel,
                                   stride,
                                   padding,
                                   1,
                                   true,
                               })
      .build(ctx, input, weights.value);
}

core::TensorValue residual(core::ModuleBuildContext &ctx,
                           const core::TensorValue &input,
                           const ResidualWeights &weights, int dilation) {
  auto x = snake(ctx, input, weights.snake1);
  x = conv(ctx, x, weights.conv1, 3 * dilation, dilation);
  x = snake(ctx, x, weights.snake2);
  x = conv(ctx, x, weights.conv2, 0);
  return modules::AddModule{}.build(ctx, input, x);
}

std::vector<float> load_reference_audio(const runtime::AudioBuffer &input) {
  if (input.sample_rate <= 0 || input.channels <= 0 || input.samples.empty()) {
    throw std::runtime_error(
        "OuteTTS voice cloning requires non-empty reference audio");
  }
  auto samples = audio::mixdown_interleaved_to_mono_average(
      input.samples, input.channels, audio::MonoMixAccumulation::Float64);
  if (input.sample_rate != 24000) {
    audio::SoxrResampleOptions options;
    options.output_length_policy = audio::SoxrOutputLengthPolicy::ActualOutput;
    options.reject_empty_output = true;
    options.warning_context = "OuteTTS reference audio";
    options.fallback_description = "linear reference-audio resampling";
    samples = audio::resample_mono_soxr_or_linear(samples, input.sample_rate,
                                                  24000, options);
  }
  constexpr size_t max_samples = 20u * 24000u;
  if (samples.size() > max_samples) {
    throw std::runtime_error(
        "OuteTTS reference audio is longer than the supported 20 seconds");
  }
  return samples;
}

std::vector<float> prepare_reference_audio(const std::vector<float> &input) {
  auto samples = input;
  double sum_sq = 0.0;
  for (const float sample : samples)
    sum_sq += static_cast<double>(sample) * sample;
  const float rms = static_cast<float>(
      std::sqrt(sum_sq / static_cast<double>(samples.size())));
  if (rms > 1.0e-6F) {
    const float scale = std::pow(10.0F, -18.0F / 20.0F) / rms;
    for (float &sample : samples)
      sample *= scale;
  }
  float peak = 0.0F;
  for (const float sample : samples)
    peak = std::max(peak, std::fabs(sample));
  const float peak_limit = std::pow(10.0F, -1.0F / 20.0F);
  if (peak > peak_limit) {
    const float scale = peak_limit / peak;
    for (float &sample : samples)
      sample *= scale;
  }
  samples.resize(((samples.size() + 319u) / 320u) * 320u, 0.0F);
  return samples;
}

std::pair<std::vector<int32_t>, std::vector<int32_t>>
quantize_reference(const std::vector<float> &frame_major,
                   const std::vector<HostQuantizerWeights> &quantizers) {
  constexpr int hidden_size = 1024;
  constexpr int codebook_size = 1024;
  constexpr int codebook_dim = 8;
  if (frame_major.empty() || frame_major.size() % hidden_size != 0 ||
      quantizers.size() != 2) {
    throw std::runtime_error(
        "OuteTTS DAC encoder produced invalid acoustic latents");
  }
  const size_t frames = frame_major.size() / hidden_size;
  std::vector<float> residual_values = frame_major;
  std::vector<int32_t> outputs[2] = {
      std::vector<int32_t>(frames),
      std::vector<int32_t>(frames),
  };
  for (size_t q_index = 0; q_index < quantizers.size(); ++q_index) {
    const auto &q = quantizers[q_index];
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t frame = 0; frame < static_cast<int64_t>(frames); ++frame) {
      float projected[codebook_dim];
      float projected_norm_sq = 0.0F;
      const size_t frame_offset = static_cast<size_t>(frame) * hidden_size;
      for (int d = 0; d < codebook_dim; ++d) {
        float value = q.in_bias[static_cast<size_t>(d)];
        const size_t weight_offset = static_cast<size_t>(d) * hidden_size;
        for (int h = 0; h < hidden_size; ++h) {
          value += q.in_weight[weight_offset + static_cast<size_t>(h)] *
                   residual_values[frame_offset + static_cast<size_t>(h)];
        }
        projected[d] = value;
        projected_norm_sq += value * value;
      }
      const float projected_norm = std::sqrt(projected_norm_sq) + 1.0e-12F;
      int best_code = 0;
      float best_distance = std::numeric_limits<float>::infinity();
      for (int code = 0; code < codebook_size; ++code) {
        const size_t code_offset = static_cast<size_t>(code) * codebook_dim;
        float embedding_norm_sq = 0.0F;
        for (int d = 0; d < codebook_dim; ++d) {
          const float value = q.codebook[code_offset + static_cast<size_t>(d)];
          embedding_norm_sq += value * value;
        }
        const float embedding_norm = std::sqrt(embedding_norm_sq) + 1.0e-12F;
        float distance = 0.0F;
        for (int d = 0; d < codebook_dim; ++d) {
          const float difference =
              projected[d] / projected_norm -
              q.codebook[code_offset + static_cast<size_t>(d)] / embedding_norm;
          distance += difference * difference;
        }
        if (distance < best_distance) {
          best_distance = distance;
          best_code = code;
        }
      }
      outputs[q_index][static_cast<size_t>(frame)] = best_code;
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t frame = 0; frame < static_cast<int64_t>(frames); ++frame) {
      const int code = outputs[q_index][static_cast<size_t>(frame)];
      const size_t code_offset = static_cast<size_t>(code) * codebook_dim;
      const size_t frame_offset = static_cast<size_t>(frame) * hidden_size;
      for (int out = 0; out < hidden_size; ++out) {
        float value = q.out_bias[static_cast<size_t>(out)];
        const size_t weight_offset = static_cast<size_t>(out) * codebook_dim;
        for (int d = 0; d < codebook_dim; ++d) {
          value += q.out_weight[weight_offset + static_cast<size_t>(d)] *
                   q.codebook[code_offset + static_cast<size_t>(d)];
        }
        residual_values[frame_offset + static_cast<size_t>(out)] -= value;
      }
    }
  }
  return {std::move(outputs[0]), std::move(outputs[1])};
}

void normalize_decoded_audio(std::vector<float> &audio) {
  if (audio.empty()) {
    return;
  }
  double sum_sq = 0.0;
  for (const float sample : audio) {
    sum_sq += static_cast<double>(sample) * static_cast<double>(sample);
  }
  const float rms =
      static_cast<float>(std::sqrt(sum_sq / static_cast<double>(audio.size())));
  if (rms > 1.0e-6F) {
    const float target_rms = std::pow(10.0F, -18.0F / 20.0F);
    const float scale = target_rms / rms;
    for (float &sample : audio) {
      sample *= scale;
    }
  }
  float peak = 0.0F;
  for (const float sample : audio) {
    peak = std::max(peak, std::fabs(sample));
  }
  const float peak_limit = std::pow(10.0F, -1.0F / 20.0F);
  if (peak > peak_limit && peak > 1.0e-6F) {
    const float scale = peak_limit / peak;
    for (float &sample : audio) {
      sample *= scale;
    }
  }
}

} // namespace

struct OuteTTSDacDecoder::Impl {
  Impl(std::shared_ptr<const OuteTTSAssets> assets_in,
       core::ExecutionContext &execution_in, size_t weight_context_bytes,
       size_t graph_context_bytes_in, assets::TensorStorageType storage_type)
      : assets(std::move(assets_in)), execution(execution_in),
        graph_context_bytes(graph_context_bytes_in),
        weights(load_weights(*assets, execution, weight_context_bytes,
                             storage_type)) {}

  OuteTTSDacDecoder::EncodedReference
  encode_reference(const runtime::AudioBuffer &input_audio) {
    auto feature_samples = load_reference_audio(input_audio);
    auto samples = prepare_reference_audio(feature_samples);
    const int64_t sample_count = static_cast<int64_t>(samples.size());
    ggml_init_params params{graph_context_bytes, nullptr, true};
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx(ggml_init(params));
    if (!ctx) {
      throw std::runtime_error(
          "failed to create OuteTTS DAC encoder graph context");
    }
    core::ModuleBuildContext build{ctx.get(), "outetts.dac.encode"};
    auto *pcm =
        ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, sample_count, 1, 1);
    auto x = core::wrap_tensor(
        pcm, core::TensorShape::from_dims({1, 1, sample_count}), GGML_TYPE_F32);
    x = conv(build, x, weights.encoder_first, 3);
    for (const auto &block : weights.encoder_blocks) {
      x = residual(build, x, block.residuals[0], 1);
      x = residual(build, x, block.residuals[1], 3);
      x = residual(build, x, block.residuals[2], 9);
      x = snake(build, x, block.snake);
      x = strided_conv(build, x, block.down, block.stride,
                       (block.stride + 1) / 2);
    }
    x = snake(build, x, weights.encoder_final_snake);
    x = conv(build, x, weights.encoder_final_conv, 1);
    x = core::ensure_backend_addressable_layout(build, x);
    ggml_set_output(x.tensor);
    auto *graph = ggml_new_graph_custom(ctx.get(), 65536, false);
    ggml_build_forward_expand(graph, x.tensor);
    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(execution.backend()));
    if (allocator == nullptr || !ggml_gallocr_reserve(allocator, graph) ||
        !ggml_gallocr_alloc_graph(allocator, graph)) {
      if (allocator != nullptr)
        ggml_gallocr_free(allocator);
      throw std::runtime_error("failed to allocate OuteTTS DAC encoder graph");
    }
    ggml_backend_tensor_set(pcm, samples.data(), 0,
                            samples.size() * sizeof(float));
    core::set_backend_threads(execution.backend(),
                              std::max(1, execution.config().threads));
    const auto status = core::compute_backend_graph(execution.backend(), graph);
    ggml_backend_synchronize(execution.backend());
    if (status != GGML_STATUS_SUCCESS) {
      core::release_backend_graph_resources(execution.backend(), graph);
      ggml_gallocr_free(allocator);
      throw std::runtime_error("OuteTTS DAC encoder graph compute failed");
    }
    const int64_t frames = x.shape.dims[2];
    if (x.shape.rank != 3 || x.shape.dims[0] != 1 || x.shape.dims[1] != 1024 ||
        frames <= 0) {
      core::release_backend_graph_resources(execution.backend(), graph);
      ggml_gallocr_free(allocator);
      throw std::runtime_error(
          "OuteTTS DAC encoder produced an unexpected shape");
    }
    std::vector<float> channel_major(static_cast<size_t>(frames) * 1024u);
    ggml_backend_tensor_get(x.tensor, channel_major.data(), 0,
                            channel_major.size() * sizeof(float));
    core::release_backend_graph_resources(execution.backend(), graph);
    ggml_gallocr_free(allocator);

    std::vector<float> frame_major(channel_major.size());
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t frame = 0; frame < frames; ++frame) {
      for (int64_t channel = 0; channel < 1024; ++channel) {
        frame_major[static_cast<size_t>(frame * 1024 + channel)] =
            channel_major[static_cast<size_t>(channel * frames + frame)];
      }
    }
    auto codes = quantize_reference(frame_major, weights.host_quantizers);
    OuteTTSDacDecoder::EncodedReference result;
    // OuteTTS derives conditioning features from the resampled source audio,
    // while the DAC encoder receives a separately loudness-normalized copy.
    result.samples = std::move(feature_samples);
    result.codebook1 = std::move(codes.first);
    result.codebook2 = std::move(codes.second);
    return result;
  }

  runtime::AudioBuffer decode(const std::vector<int32_t> &c1,
                              const std::vector<int32_t> &c2) {
    const int64_t frames = static_cast<int64_t>(std::min(c1.size(), c2.size()));
    if (frames <= 0) {
      throw std::runtime_error(
          "OuteTTS DAC requires at least one complete code pair");
    }
    ggml_init_params params{graph_context_bytes, nullptr, true};
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx(ggml_init(params));
    if (!ctx) {
      throw std::runtime_error("failed to create OuteTTS DAC graph context");
    }
    core::ModuleBuildContext build{ctx.get(), "outetts.dac.decode"};
    auto *ids1 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, frames, 1);
    auto *ids2 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, frames, 1);
    auto id_value1 = core::wrap_tensor(
        ids1, core::TensorShape::from_dims({1, frames}), GGML_TYPE_I32);
    auto id_value2 = core::wrap_tensor(
        ids2, core::TensorShape::from_dims({1, frames}), GGML_TYPE_I32);
    auto e1 = modules::EmbeddingModule({1024, 8}).build(
        build, id_value1, weights.quantizers[0].codebook);
    auto e2 = modules::EmbeddingModule({1024, 8}).build(
        build, id_value2, weights.quantizers[1].codebook);
    e1 = modules::TransposeModule({{0, 2, 1}, 3}).build(build, e1);
    e2 = modules::TransposeModule({{0, 2, 1}, 3}).build(build, e2);
    auto z1 = conv(build, core::ensure_backend_addressable_layout(build, e1),
                   weights.quantizers[0].out_proj, 0);
    auto z2 = conv(build, core::ensure_backend_addressable_layout(build, e2),
                   weights.quantizers[1].out_proj, 0);
    auto x = modules::AddModule{}.build(build, z1, z2);
    x = conv(build, x, weights.first, 3);
    for (const auto &block : weights.blocks) {
      x = snake(build, x, block.snake);
      auto upsampled = modules::ConvTranspose1dModule({
                                                          block.up.in_channels,
                                                          block.up.out_channels,
                                                          block.up.kernel,
                                                          block.stride,
                                                          0,
                                                          1,
                                                          true,
                                                      })
                           .build(build, x, block.up.value);
      const int padding = static_cast<int>(std::ceil(block.stride / 2.0));
      const int64_t target_frames = upsampled.shape.dims[2] - 2 * padding;
      x = modules::SliceModule({2, padding, target_frames})
              .build(build, upsampled);
      x = residual(build, x, block.residuals[0], 1);
      x = residual(build, x, block.residuals[1], 3);
      x = residual(build, x, block.residuals[2], 9);
    }
    x = snake(build, x, weights.final_snake);
    x = conv(build, x, weights.final_conv, 3);
    x = modules::TanhModule{}.build(build, x);
    x = core::ensure_backend_addressable_layout(build, x);
    ggml_set_output(x.tensor);
    auto *graph = ggml_new_graph_custom(ctx.get(), 65536, false);
    ggml_build_forward_expand(graph, x.tensor);
    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(execution.backend()));
    if (allocator == nullptr || !ggml_gallocr_reserve(allocator, graph) ||
        !ggml_gallocr_alloc_graph(allocator, graph)) {
      if (allocator != nullptr)
        ggml_gallocr_free(allocator);
      throw std::runtime_error("failed to allocate OuteTTS DAC graph");
    }
    ggml_backend_tensor_set(ids1, c1.data(), 0,
                            static_cast<size_t>(frames) * sizeof(int32_t));
    ggml_backend_tensor_set(ids2, c2.data(), 0,
                            static_cast<size_t>(frames) * sizeof(int32_t));
    core::set_backend_threads(execution.backend(),
                              std::max(1, execution.config().threads));
    const auto status = core::compute_backend_graph(execution.backend(), graph);
    ggml_backend_synchronize(execution.backend());
    if (status != GGML_STATUS_SUCCESS) {
      core::release_backend_graph_resources(execution.backend(), graph);
      ggml_gallocr_free(allocator);
      throw std::runtime_error("OuteTTS DAC graph compute failed");
    }
    runtime::AudioBuffer audio;
    audio.sample_rate = 24000;
    audio.channels = 1;
    audio.samples.resize(static_cast<size_t>(x.shape.dims[2]));
    ggml_backend_tensor_get(x.tensor, audio.samples.data(), 0,
                            audio.samples.size() * sizeof(float));
    normalize_decoded_audio(audio.samples);
    core::release_backend_graph_resources(execution.backend(), graph);
    ggml_gallocr_free(allocator);
    return audio;
  }

  std::shared_ptr<const OuteTTSAssets> assets;
  core::ExecutionContext &execution;
  size_t graph_context_bytes;
  DacWeights weights;
};

OuteTTSDacDecoder::OuteTTSDacDecoder(
    std::shared_ptr<const OuteTTSAssets> assets,
    core::ExecutionContext &execution_context, size_t weight_context_bytes,
    size_t graph_context_bytes, assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(std::move(assets), execution_context,
                                   weight_context_bytes, graph_context_bytes,
                                   weight_storage_type)) {}

OuteTTSDacDecoder::~OuteTTSDacDecoder() = default;

runtime::AudioBuffer
OuteTTSDacDecoder::decode(const std::vector<int32_t> &codebook1,
                          const std::vector<int32_t> &codebook2) {
  return impl_->decode(codebook1, codebook2);
}

OuteTTSDacDecoder::EncodedReference
OuteTTSDacDecoder::encode_reference(const runtime::AudioBuffer &audio) {
  return impl_->encode_reference(audio);
}

} // namespace engine::models::outetts
