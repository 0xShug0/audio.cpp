#include "engine/models/irodori_tts/codec.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
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

std::vector<float> squeeze_channel_alpha(const assets::TensorSource &source,
                                         const std::string &name,
                                         int64_t channels) {
  const auto values = source.require_f32(name, {1, channels, 1});
  if (static_cast<int64_t>(values.size()) != channels) {
    throw std::runtime_error("Irodori codec Snake alpha shape mismatch: " +
                             name);
  }
  return values;
}

std::vector<float> squeeze_weight_g(const std::vector<float> &values,
                                    int64_t channels,
                                    const std::string &prefix) {
  if (static_cast<int64_t>(values.size()) != channels) {
    throw std::runtime_error("Irodori codec weight_g shape mismatch: " +
                             prefix);
  }
  return values;
}

std::vector<float> fold_weight_norm_dim0(const std::vector<float> &weight_v,
                                         const std::vector<float> &weight_g,
                                         const std::vector<int64_t> &shape,
                                         const std::string &prefix) {
  if (shape.empty()) {
    throw std::runtime_error("Irodori codec weight norm shape is empty: " +
                             prefix);
  }
  int64_t inner = 1;
  for (size_t i = 1; i < shape.size(); ++i) {
    inner *= shape[i];
  }
  if (static_cast<int64_t>(weight_v.size()) != shape[0] * inner ||
      static_cast<int64_t>(weight_g.size()) != shape[0]) {
    throw std::runtime_error(
        "Irodori codec weight norm tensor size mismatch: " + prefix);
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
      throw std::runtime_error("Irodori codec weight norm row has zero norm: " +
                               prefix);
    }
    const float scale = static_cast<float>(
        static_cast<double>(weight_g[static_cast<size_t>(row)]) /
        std::sqrt(sum));
    for (int64_t i = 0; i < inner; ++i) {
      out[base + static_cast<size_t>(i)] =
          weight_v[base + static_cast<size_t>(i)] * scale;
    }
  }
  return out;
}

modules::Conv1dWeights load_weight_norm_conv1d(
    core::BackendWeightStore &store, const assets::TensorSource &source,
    const std::string &prefix, assets::TensorStorageType storage_type,
    int64_t out_channels, int64_t in_channels, int64_t kernel_size,
    bool use_bias = true) {
  modules::Conv1dWeights weights;
  const auto weight_v = source.require_f32(
      prefix + ".weight_v", {out_channels, in_channels, kernel_size});
  const auto weight_g = squeeze_weight_g(
      source.require_f32(prefix + ".weight_g", {out_channels, 1, 1}),
      out_channels, prefix);
  weights.weight = store.make_from_f32(
      core::TensorShape::from_dims({out_channels, in_channels, kernel_size}),
      storage_type,
      fold_weight_norm_dim0(weight_v, weight_g,
                            {out_channels, in_channels, kernel_size}, prefix));
  if (use_bias) {
    weights.bias =
        store.load_f32_tensor(source, prefix + ".bias", {out_channels});
  }
  return weights;
}

modules::ConvTranspose1dWeights load_weight_norm_conv_transpose1d(
    core::BackendWeightStore &store, const assets::TensorSource &source,
    const std::string &prefix, assets::TensorStorageType storage_type,
    int64_t in_channels, int64_t out_channels, int64_t kernel_size,
    bool use_bias = true) {
  modules::ConvTranspose1dWeights weights;
  const auto weight_v = source.require_f32(
      prefix + ".weight_v", {in_channels, out_channels, kernel_size});
  const auto weight_g = squeeze_weight_g(
      source.require_f32(prefix + ".weight_g", {in_channels, 1, 1}),
      in_channels, prefix);
  weights.weight = store.make_from_f32(
      core::TensorShape::from_dims({in_channels, out_channels, kernel_size}),
      storage_type,
      fold_weight_norm_dim0(weight_v, weight_g,
                            {in_channels, out_channels, kernel_size}, prefix));
  if (use_bias) {
    weights.bias =
        store.load_f32_tensor(source, prefix + ".bias", {out_channels});
  }
  return weights;
}

modules::Snake1dWeights load_snake(core::BackendWeightStore &store,
                                   const assets::TensorSource &source,
                                   const std::string &name, int64_t channels) {
  return {store.make_f32(core::TensorShape::from_dims({channels}),
                         squeeze_channel_alpha(source, name, channels))};
}

IrodoriCodecResidualUnitWeights load_residual_unit(
    core::BackendWeightStore &store, const assets::TensorSource &source,
    const std::string &prefix, assets::TensorStorageType storage_type,
    int64_t channels, int64_t kernel, int64_t dilation) {
  IrodoriCodecResidualUnitWeights weights;
  weights.snake0 =
      load_snake(store, source, prefix + ".block.0.alpha", channels);
  weights.conv0 =
      load_weight_norm_conv1d(store, source, prefix + ".block.1", storage_type,
                              channels, channels, kernel);
  weights.snake1 =
      load_snake(store, source, prefix + ".block.2.alpha", channels);
  weights.conv1 = load_weight_norm_conv1d(store, source, prefix + ".block.3",
                                          storage_type, channels, channels, 1);
  (void)dilation;
  return weights;
}

IrodoriCodecDecoderBlockWeights load_decoder_block(
    core::BackendWeightStore &store, const assets::TensorSource &source,
    const std::string &prefix, assets::TensorStorageType storage_type,
    int64_t input_dim, int64_t output_dim, int64_t stride) {
  IrodoriCodecDecoderBlockWeights weights;
  weights.up_snake =
      load_snake(store, source, prefix + ".block.0.alpha", input_dim);
  weights.up_conv = load_weight_norm_conv_transpose1d(
      store, source, prefix + ".block.1", storage_type, input_dim, output_dim,
      2 * stride);
  weights.residual_0 = load_residual_unit(store, source, prefix + ".block.4",
                                          storage_type, output_dim, 7, 1);
  weights.residual_1 = load_residual_unit(store, source, prefix + ".block.5",
                                          storage_type, output_dim, 7, 3);
  weights.residual_2 = load_residual_unit(store, source, prefix + ".block.8",
                                          storage_type, output_dim, 7, 9);
  return weights;
}

IrodoriCodecEncoderBlockWeights load_encoder_block(
    core::BackendWeightStore &store, const assets::TensorSource &source,
    const std::string &prefix, assets::TensorStorageType storage_type,
    int64_t input_dim, int64_t output_dim, int64_t stride) {
  IrodoriCodecEncoderBlockWeights weights;
  weights.residual_0 = load_residual_unit(store, source, prefix + ".block.0",
                                          storage_type, input_dim, 7, 1);
  weights.residual_1 = load_residual_unit(store, source, prefix + ".block.1",
                                          storage_type, input_dim, 7, 3);
  weights.residual_2 = load_residual_unit(store, source, prefix + ".block.2",
                                          storage_type, input_dim, 7, 9);
  weights.down_snake =
      load_snake(store, source, prefix + ".block.3.alpha", input_dim);
  weights.down_conv =
      load_weight_norm_conv1d(store, source, prefix + ".block.4", storage_type,
                              output_dim, input_dim, 2 * stride);
  return weights;
}

core::TensorValue residual_unit(core::ModuleBuildContext &ctx,
                                const core::TensorValue &input,
                                const IrodoriCodecResidualUnitWeights &weights,
                                int64_t channels, int64_t dilation) {
  auto hidden =
      modules::Snake1dModule({channels}).build(ctx, input, weights.snake0);
  hidden = modules::Conv1dModule({
                                     channels,
                                     channels,
                                     7,
                                     1,
                                     static_cast<int>((7 - 1) * dilation / 2),
                                     static_cast<int>(dilation),
                                     weights.conv0.bias.has_value(),
                                 })
               .build(ctx, hidden, weights.conv0);
  hidden =
      modules::Snake1dModule({channels}).build(ctx, hidden, weights.snake1);
  hidden = modules::Conv1dModule({
                                     channels,
                                     channels,
                                     1,
                                     1,
                                     0,
                                     1,
                                     weights.conv1.bias.has_value(),
                                 })
               .build(ctx, hidden, weights.conv1);
  return modules::AddModule{}.build(ctx, hidden, input);
}

core::TensorValue decoder_block(core::ModuleBuildContext &ctx,
                                const core::TensorValue &input,
                                const IrodoriCodecDecoderBlockWeights &weights,
                                int64_t in_channels, int64_t out_channels,
                                int64_t stride) {
  auto hidden =
      modules::Snake1dModule({in_channels}).build(ctx, input, weights.up_snake);
  const int64_t padding = (stride + 1) / 2;
  const bool crop_vulkan_output = ctx.backend_type == core::BackendType::Vulkan;
  hidden = modules::ConvTranspose1dModule({
                                              in_channels,
                                              out_channels,
                                              2 * stride,
                                              static_cast<int>(stride),
                                              crop_vulkan_output
                                                  ? 0
                                                  : static_cast<int>(padding),
                                              1,
                                              weights.up_conv.bias.has_value(),
                                          })
               .build(ctx, hidden, weights.up_conv);
  if (crop_vulkan_output) {
    hidden = modules::SliceModule({2, padding,
                                   hidden.shape.dims[2] - 2 * padding})
                 .build(ctx, hidden);
  }
  hidden = residual_unit(ctx, hidden, weights.residual_0, out_channels, 1);
  hidden = residual_unit(ctx, hidden, weights.residual_1, out_channels, 3);
  hidden = residual_unit(ctx, hidden, weights.residual_2, out_channels, 9);
  return hidden;
}

core::TensorValue encoder_block(core::ModuleBuildContext &ctx,
                                const core::TensorValue &input,
                                const IrodoriCodecEncoderBlockWeights &weights,
                                int64_t in_channels, int64_t out_channels,
                                int64_t stride) {
  auto hidden = residual_unit(ctx, input, weights.residual_0, in_channels, 1);
  hidden = residual_unit(ctx, hidden, weights.residual_1, in_channels, 3);
  hidden = residual_unit(ctx, hidden, weights.residual_2, in_channels, 9);
  hidden = modules::Snake1dModule({in_channels})
               .build(ctx, hidden, weights.down_snake);
  return modules::Conv1dModule({
                                   in_channels,
                                   out_channels,
                                   2 * stride,
                                   static_cast<int>(stride),
                                   static_cast<int>((2 * stride - stride) / 2),
                                   1,
                                   weights.down_conv.bias.has_value(),
                               })
      .build(ctx, hidden, weights.down_conv);
}

} // namespace

IrodoriCodecWeights
load_irodori_codec_weights(const IrodoriAssets &assets, ggml_backend_t backend,
                           core::BackendType backend_type,
                           size_t weight_context_bytes,
                           assets::TensorStorageType conv_storage_type) {
  const auto &source = *assets.codec_weights;
  IrodoriCodecWeights weights;
  weights.store = std::make_shared<core::BackendWeightStore>(
      backend, backend_type, "irodori_tts.codec.weights",
      weight_context_bytes == 0 ? kCodecWeightContextBytes
                                : weight_context_bytes);
  weights.encoder_input = load_weight_norm_conv1d(
      *weights.store, source, "encoder.block.0", conv_storage_type,
      assets.codec.encoder_dim, 1, 7);
  const int64_t encoder_rates[] = {2, 8, 10, 12};
  int64_t encoder_in_channels = assets.codec.encoder_dim;
  for (int64_t block = 0; block < 4; ++block) {
    const int64_t encoder_out_channels = encoder_in_channels * 2;
    weights.encoder_blocks.push_back(load_encoder_block(
        *weights.store, source, "encoder.block." + std::to_string(block + 1),
        conv_storage_type, encoder_in_channels, encoder_out_channels,
        encoder_rates[block]));
    encoder_in_channels = encoder_out_channels;
  }
  weights.encoder_output_snake = load_snake(
      *weights.store, source, "encoder.block.5.alpha", encoder_in_channels);
  weights.encoder_output = load_weight_norm_conv1d(
      *weights.store, source, "encoder.block.6", conv_storage_type,
      assets.codec.latent_dim, encoder_in_channels, 3);
  weights.quantizer_in_proj = load_weight_norm_conv1d(
      *weights.store, source, "quantizer.in_proj", conv_storage_type,
      2 * assets.codec.codebook_dim, assets.codec.latent_dim, 1);
  weights.quantizer_out_proj = load_weight_norm_conv1d(
      *weights.store, source, "quantizer.out_proj", conv_storage_type,
      assets.codec.latent_dim, assets.codec.codebook_dim, 1);
  weights.decoder_input = load_weight_norm_conv1d(
      *weights.store, source, "decoder.model.0", conv_storage_type,
      assets.codec.decoder_dim, assets.codec.latent_dim, 7);
  const int64_t rates[] = {12, 10, 8, 2};
  int64_t in_channels = assets.codec.decoder_dim;
  for (int64_t block = 0; block < 4; ++block) {
    const int64_t out_channels = in_channels / 2;
    weights.decoder_blocks.push_back(load_decoder_block(
        *weights.store, source, "decoder.model." + std::to_string(block + 1),
        conv_storage_type, in_channels, out_channels, rates[block]));
    in_channels = out_channels;
  }
  weights.watermark_passthrough.snake =
      load_snake(*weights.store, source,
                 "decoder.wm_model.encoder_block.pre.0.alpha", in_channels);
  weights.watermark_passthrough.conv = load_weight_norm_conv1d(
      *weights.store, source, "decoder.wm_model.encoder_block.pre.1",
      conv_storage_type, 1, in_channels, 7);
  weights.store->upload();
  return weights;
}

core::TensorValue build_irodori_codec_decode(
    core::ModuleBuildContext &ctx, const core::TensorValue &latent_btd,
    const IrodoriCodecWeights &weights, const IrodoriCodecConfig &config) {
  auto hidden = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, latent_btd);
  hidden = modules::Conv1dModule(
               {config.codebook_dim, config.latent_dim, 1, 1, 0, 1, true})
               .build(ctx, hidden, weights.quantizer_out_proj);
  hidden = modules::Conv1dModule({
                                     config.latent_dim,
                                     config.decoder_dim,
                                     7,
                                     1,
                                     3,
                                     1,
                                     weights.decoder_input.bias.has_value(),
                                 })
               .build(ctx, hidden, weights.decoder_input);
  const int64_t rates[] = {12, 10, 8, 2};
  int64_t in_channels = config.decoder_dim;
  for (size_t i = 0; i < weights.decoder_blocks.size(); ++i) {
    const int64_t out_channels = in_channels / 2;
    hidden = decoder_block(ctx, hidden, weights.decoder_blocks[i], in_channels,
                           out_channels, rates[i]);
    in_channels = out_channels;
  }
  hidden = modules::Snake1dModule({in_channels})
               .build(ctx, hidden, weights.watermark_passthrough.snake);
  hidden = modules::Conv1dModule(
               {
                   in_channels,
                   1,
                   7,
                   1,
                   3,
                   1,
                   weights.watermark_passthrough.conv.bias.has_value(),
               })
               .build(ctx, hidden, weights.watermark_passthrough.conv);
  return modules::TanhModule{}.build(ctx, hidden);
}

core::TensorValue build_irodori_codec_encode(
    core::ModuleBuildContext &ctx, const core::TensorValue &waveform_bct,
    const IrodoriCodecWeights &weights, const IrodoriCodecConfig &config) {
  auto hidden =
      modules::Conv1dModule({
                                1,
                                config.encoder_dim,
                                7,
                                1,
                                3,
                                1,
                                weights.encoder_input.bias.has_value(),
                            })
          .build(ctx, waveform_bct, weights.encoder_input);
  const int64_t rates[] = {2, 8, 10, 12};
  int64_t in_channels = config.encoder_dim;
  for (size_t i = 0; i < weights.encoder_blocks.size(); ++i) {
    const int64_t out_channels = in_channels * 2;
    hidden = encoder_block(ctx, hidden, weights.encoder_blocks[i], in_channels,
                           out_channels, rates[i]);
    in_channels = out_channels;
  }
  hidden = modules::Snake1dModule({in_channels})
               .build(ctx, hidden, weights.encoder_output_snake);
  hidden = modules::Conv1dModule({
                                     in_channels,
                                     config.latent_dim,
                                     3,
                                     1,
                                     1,
                                     1,
                                     weights.encoder_output.bias.has_value(),
                                 })
               .build(ctx, hidden, weights.encoder_output);
  hidden = modules::Conv1dModule(
               {config.latent_dim, 2 * config.codebook_dim, 1, 1, 0, 1, true})
               .build(ctx, hidden, weights.quantizer_in_proj);
  hidden = modules::SliceModule({1, 0, config.codebook_dim}).build(ctx, hidden);
  return modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, hidden);
}

namespace {

struct GgmlContextDeleter {
  void operator()(ggml_context *ctx) const noexcept {
    if (ctx != nullptr) {
      ggml_free(ctx);
    }
  }
};

std::vector<float> reflect_pad_right_to_multiple(std::vector<float> samples,
                                                 int64_t multiple) {
  if (samples.empty() || multiple <= 0) {
    throw std::runtime_error("Irodori-TTS reference audio is invalid");
  }
  const int64_t length = static_cast<int64_t>(samples.size());
  const int64_t pad = (multiple - (length % multiple)) % multiple;
  if (pad == 0) {
    return samples;
  }
  if (pad >= length) {
    throw std::runtime_error(
        "Irodori-TTS reference audio is too short for DACVAE reflect padding");
  }
  samples.reserve(static_cast<size_t>(length + pad));
  for (int64_t i = 0; i < pad; ++i) {
    samples.push_back(samples[static_cast<size_t>(length - 2 - i)]);
  }
  return samples;
}

std::vector<float> lfilter_biquad(const std::vector<float> &input, double b0,
                                  double b1, double b2, double a1, double a2) {
  std::vector<float> output(input.size(), 0.0F);
  double x1 = 0.0;
  double x2 = 0.0;
  double y1 = 0.0;
  double y2 = 0.0;
  for (size_t i = 0; i < input.size(); ++i) {
    const double x0 = static_cast<double>(input[i]);
    const double y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    output[i] = static_cast<float>(y0);
    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;
  }
  return output;
}

double bs1770_loudness_48k(const std::vector<float> &mono) {
  constexpr int kSampleRate = 48000;
  constexpr int kMinSamples = kSampleRate / 2;
  constexpr int kBlockSamples = static_cast<int>(0.4 * kSampleRate);
  constexpr int kStrideSamples = kBlockSamples / 4;
  constexpr double kAbsoluteGate = -70.0;
  constexpr double kMinLoudness = -70.0;
  std::vector<float> working = mono;
  if (static_cast<int>(working.size()) < kMinSamples) {
    working.resize(kMinSamples, 0.0F);
  }
  working = lfilter_biquad(working, 1.5351828863637502, -2.691804030199196,
                           1.198426263333146, -1.6906995865986896,
                           0.7325047060963897);
  working = lfilter_biquad(working, 0.9950442970178917, -1.9900885940357833,
                           0.9950442970178917, -1.990076284018423,
                           0.9901009040531438);
  const int64_t block_count =
      (static_cast<int64_t>(working.size()) - kBlockSamples) / kStrideSamples +
      1;
  std::vector<double> block_energy(static_cast<size_t>(block_count));
#ifdef _OPENMP
#pragma omp parallel for if(block_count >= 8)
#endif
  for (int64_t block = 0; block < block_count; ++block) {
    const int64_t start = block * kStrideSamples;
    double sum_sq = 0.0;
    for (int i = 0; i < kBlockSamples; ++i) {
      const double sample = working[static_cast<size_t>(start + i)];
      sum_sq += sample * sample;
    }
    block_energy[static_cast<size_t>(block)] =
        sum_sq / static_cast<double>(kBlockSamples);
  }
  if (block_energy.empty()) {
    return kMinLoudness;
  }
  auto loudness_from_energy = [](double energy) {
    return -0.691 + 10.0 * std::log10(std::max(
                               energy, std::numeric_limits<double>::min()));
  };
  double gated_sum = 0.0;
  int gated_count = 0;
  std::vector<double> block_loudness(block_energy.size(), kMinLoudness);
  for (size_t i = 0; i < block_energy.size(); ++i) {
    block_loudness[i] = loudness_from_energy(block_energy[i]);
    if (block_loudness[i] > kAbsoluteGate) {
      gated_sum += block_energy[i];
      ++gated_count;
    }
  }
  if (gated_count == 0) {
    return kMinLoudness;
  }
  const double relative_gate =
      loudness_from_energy(gated_sum / static_cast<double>(gated_count)) - 10.0;
  double relative_sum = 0.0;
  int relative_count = 0;
  for (size_t i = 0; i < block_energy.size(); ++i) {
    if (block_loudness[i] > kAbsoluteGate &&
        block_loudness[i] > relative_gate) {
      relative_sum += block_energy[i];
      ++relative_count;
    }
  }
  if (relative_count == 0) {
    return kMinLoudness;
  }
  return std::max(
      loudness_from_energy(relative_sum / static_cast<double>(relative_count)),
      kMinLoudness);
}

void normalize_reference_audio_in_place(std::vector<float> &mono,
                                        double target_db) {
  constexpr double kGainFactor = 0.11512925464970229;
  const double loudness = bs1770_loudness_48k(mono);
  const double gain = std::exp((target_db - loudness) * kGainFactor);
  float peak = 0.0F;
  const int64_t samples = static_cast<int64_t>(mono.size());
#ifdef _OPENMP
#pragma omp parallel for reduction(max: peak) if(samples >= 4096)
#endif
  for (int64_t i = 0; i < samples; ++i) {
    float &sample = mono[static_cast<size_t>(i)];
    sample = static_cast<float>(static_cast<double>(sample) * gain);
    peak = std::max(peak, std::abs(sample));
  }
  if (std::isfinite(peak) && peak > 1.0F) {
    const float peak_gain = 1.0F / peak;
#ifdef _OPENMP
#pragma omp parallel for if(samples >= 4096)
#endif
    for (int64_t i = 0; i < samples; ++i) {
      mono[static_cast<size_t>(i)] *= peak_gain;
    }
  }
}

} // namespace

class IrodoriCodec::Impl {
public:
  Impl(std::shared_ptr<const IrodoriAssets> assets,
       core::ExecutionContext &execution_context, size_t graph_arena_bytes,
       size_t weight_context_bytes,
       assets::TensorStorageType weight_storage_type)
      : assets_(std::move(assets)),
        weights_(load_irodori_codec_weights(
            *assets_, execution_context.backend(),
            execution_context.backend_type(), weight_context_bytes,
            weight_storage_type)),
        backend_(execution_context.backend()),
        backend_type_(execution_context.backend_type()),
        threads_(std::max(1, execution_context.config().threads)),
        graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr) {
      throw std::runtime_error(
          "Irodori-TTS codec graph runner requires assets");
    }
  }

  runtime::AudioBuffer decode(const std::vector<float> &latent,
                              int64_t latent_steps, int64_t target_samples) {
    const bool graph_rebuild =
        graph_ == nullptr || graph_->latent_steps() != latent_steps;
    if (graph_rebuild) {
      graph_.reset();
      graph_ = std::make_unique<Graph>(*this, latent_steps, graph_arena_bytes_);
    }
    debug::trace_log_scalar("irodori_tts.codec_decode.graph_rebuild",
                             graph_rebuild);
    return graph_->run(latent, target_samples);
  }

  std::vector<float> encode_reference(const runtime::AudioBuffer &audio,
                                      int64_t &latent_steps_out) {
    auto mono = engine::audio::mixdown_interleaved_to_mono_average(
        audio.samples, audio.channels);
    if (audio.sample_rate != assets_->codec.sample_rate) {
      engine::audio::TorchaudioSincHannResampleOptions options;
      options.kernel_mode = engine::audio::TorchaudioSincHannKernelMode::
          Float32ComputationStoredAsFloat32;
      options.accumulation =
          engine::audio::TorchaudioSincHannAccumulation::Float32;
      mono = engine::audio::resample_mono_torchaudio_sinc_hann(
          mono, audio.sample_rate, assets_->codec.sample_rate, options);
    }
    normalize_reference_audio_in_place(mono, -16.0);
    mono = reflect_pad_right_to_multiple(std::move(mono),
                                         assets_->codec.hop_length);
    const int64_t padded_samples = static_cast<int64_t>(mono.size());
    const bool graph_rebuild =
        encode_graph_ == nullptr ||
        encode_graph_->padded_samples() != padded_samples;
    if (graph_rebuild) {
      encode_graph_.reset();
      encode_graph_ = std::make_unique<EncodeGraph>(*this, padded_samples,
                                                    graph_arena_bytes_);
    }
    debug::trace_log_scalar("irodori_tts.codec_encode.graph_rebuild",
                             graph_rebuild);
    return encode_graph_->run(mono, latent_steps_out);
  }

  void release_graphs() {
    graph_.reset();
    encode_graph_.reset();
  }

private:
  class EncodeGraph {
  public:
    EncodeGraph(IrodoriCodec::Impl &owner, int64_t padded_samples,
                size_t graph_arena_bytes)
        : owner_(&owner), padded_samples_(padded_samples) {
      const auto &config = owner.assets_->codec;
      ggml_init_params params{graph_arena_bytes, nullptr, true};
      ctx_.reset(ggml_init(params));
      if (ctx_ == nullptr) {
        throw std::runtime_error(
            "failed to initialize Irodori-TTS codec encode graph context");
      }
      core::ModuleBuildContext build_ctx{ctx_.get(), "irodori_tts.codec_encode",
                                         owner.backend_type_};
      waveform_ = core::make_tensor(
          build_ctx, GGML_TYPE_F32,
          core::TensorShape::from_dims({1, 1, padded_samples_}));
      ggml_set_input(waveform_.tensor);
      auto output = build_irodori_codec_encode(build_ctx, waveform_,
                                               owner.weights_, config);
      output_ = core::ensure_backend_addressable_layout(build_ctx, output);
      ggml_set_output(output_.tensor);
      graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
      ggml_build_forward_expand(graph_, output_.tensor);
      gallocr_ = ggml_gallocr_new(
          ggml_backend_get_default_buffer_type(owner.backend_));
      if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) ||
          !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
        throw std::runtime_error(
            "failed to allocate Irodori-TTS codec encode graph");
      }
    }

    ~EncodeGraph() {
      engine::core::release_backend_graph_resources(owner_->backend_, graph_);
      if (gallocr_ != nullptr) {
        ggml_gallocr_free(gallocr_);
      }
    }

    int64_t padded_samples() const noexcept { return padded_samples_; }

    std::vector<float> run(const std::vector<float> &mono,
                           int64_t &latent_steps_out) {
      if (static_cast<int64_t>(mono.size()) != padded_samples_) {
        throw std::runtime_error(
            "Irodori-TTS codec encode audio size mismatch");
      }
      core::write_tensor_f32(waveform_, mono);
      core::set_backend_threads(owner_->backend_, owner_->threads_);
      const ggml_status status = core::compute_backend_graph(
          owner_->backend_, graph_, nullptr, "irodori_tts.codec_encode");
      ggml_backend_synchronize(owner_->backend_);
      if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(
            "Irodori-TTS codec encode graph compute failed");
      }
      latent_steps_out = output_.shape.dims[1];
      return core::read_tensor_f32(output_.tensor);
    }

  private:
    IrodoriCodec::Impl *owner_ = nullptr;
    int64_t padded_samples_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue waveform_;
    core::TensorValue output_;
    ggml_cgraph *graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
  };

  class Graph {
  public:
    Graph(IrodoriCodec::Impl &owner, int64_t latent_steps,
          size_t graph_arena_bytes)
        : owner_(&owner), latent_steps_(latent_steps) {
      const auto &config = owner.assets_->config;
      ggml_init_params params{graph_arena_bytes, nullptr, true};
      ctx_.reset(ggml_init(params));
      if (ctx_ == nullptr) {
        throw std::runtime_error(
            "failed to initialize Irodori-TTS codec graph context");
      }
      core::ModuleBuildContext build_ctx{ctx_.get(), "irodori_tts.codec_decode",
                                         owner.backend_type_};
      latent_ = core::make_tensor(
          build_ctx, GGML_TYPE_F32,
          core::TensorShape::from_dims({1, latent_steps_, config.latent_dim}));
      ggml_set_input(latent_.tensor);
      auto output = build_irodori_codec_decode(
          build_ctx, latent_, owner.weights_, owner.assets_->codec);
      output_ = core::ensure_backend_addressable_layout(build_ctx, output);
      ggml_set_output(output_.tensor);
      graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
      ggml_build_forward_expand(graph_, output_.tensor);
      gallocr_ = ggml_gallocr_new(
          ggml_backend_get_default_buffer_type(owner.backend_));
      if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) ||
          !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
        throw std::runtime_error("failed to allocate Irodori-TTS codec graph");
      }
    }

    ~Graph() {
      engine::core::release_backend_graph_resources(owner_->backend_, graph_);
      if (gallocr_ != nullptr) {
        ggml_gallocr_free(gallocr_);
      }
    }

    int64_t latent_steps() const noexcept { return latent_steps_; }

    runtime::AudioBuffer run(const std::vector<float> &latent,
                             int64_t target_samples) {
      const auto &config = owner_->assets_->config;
      if (static_cast<int64_t>(latent.size()) !=
          latent_steps_ * config.latent_dim) {
        throw std::runtime_error("Irodori-TTS codec latent size mismatch");
      }
      core::write_tensor_f32(latent_, latent);
      core::set_backend_threads(owner_->backend_, owner_->threads_);
      const ggml_status status = core::compute_backend_graph(
          owner_->backend_, graph_, nullptr, "irodori_tts.codec_decode");
      ggml_backend_synchronize(owner_->backend_);
      if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Irodori-TTS codec graph compute failed");
      }
      auto samples = core::read_tensor_f32(output_.tensor);
      if (target_samples > 0 &&
          static_cast<int64_t>(samples.size()) > target_samples) {
        samples.resize(static_cast<size_t>(target_samples));
      }
      return runtime::AudioBuffer{
          owner_->assets_->codec.sample_rate,
          1,
          std::move(samples),
      };
    }

  private:
    IrodoriCodec::Impl *owner_ = nullptr;
    int64_t latent_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    core::TensorValue latent_;
    core::TensorValue output_;
    ggml_cgraph *graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
  };

  std::shared_ptr<const IrodoriAssets> assets_;
  IrodoriCodecWeights weights_;
  ggml_backend_t backend_ = nullptr;
  core::BackendType backend_type_ = core::BackendType::Cpu;
  int threads_ = 1;
  size_t graph_arena_bytes_ = 0;
  std::unique_ptr<Graph> graph_;
  std::unique_ptr<EncodeGraph> encode_graph_;
};

IrodoriCodec::IrodoriCodec(std::shared_ptr<const IrodoriAssets> assets,
                           core::ExecutionContext &execution_context,
                           size_t graph_arena_bytes,
                           size_t weight_context_bytes,
                           assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(std::move(assets), execution_context,
                                   graph_arena_bytes, weight_context_bytes,
                                   weight_storage_type)) {}

IrodoriCodec::~IrodoriCodec() = default;

runtime::AudioBuffer IrodoriCodec::decode(const std::vector<float> &latent,
                                          int64_t latent_steps,
                                          int64_t target_samples) {
  return impl_->decode(latent, latent_steps, target_samples);
}

std::vector<float>
IrodoriCodec::encode_reference(const runtime::AudioBuffer &audio,
                               int64_t &latent_steps_out) {
  return impl_->encode_reference(audio, latent_steps_out);
}

void IrodoriCodec::release_graphs() { impl_->release_graphs(); }

} // namespace engine::models::irodori_tts
