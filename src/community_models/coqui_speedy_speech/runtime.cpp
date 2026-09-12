#include "engine/community_models/coqui_speedy_speech/runtime.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/vocoders/hifigan_vocoder.h"
#include "engine/framework/modules/weight_binding.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::coqui_speedy_speech {
namespace {
namespace core = engine::core;
namespace mod = engine::modules;
namespace binding = engine::modules::binding;

constexpr size_t kWeightArena = 48ULL * 1024ULL * 1024ULL;
constexpr size_t kIoArena = 4ULL * 1024ULL * 1024ULL;
constexpr size_t kGraphArena = 96ULL * 1024ULL * 1024ULL;
constexpr int64_t kMaxMelFrames = 4096;

struct ContextDeleter {
  void operator()(ggml_context *p) const noexcept {
    if (p)
      ggml_free(p);
  }
};

struct ConvBn {
  mod::Conv1dWeights conv;
  mod::BatchNorm1dEvalWeights norm;
};
struct ResidualBlock {
  std::array<ConvBn, 2> layers;
  int dilation = 1;
};
struct Weights {
  std::shared_ptr<core::BackendWeightStore> store;
  core::TensorValue embedding;
  mod::Conv1dWeights encoder_pre;
  std::vector<ResidualBlock> encoder;
  mod::Conv1dWeights encoder_post0;
  mod::BatchNorm1dEvalWeights encoder_post_norm;
  mod::Conv1dWeights encoder_post3;
  mod::Conv1dWeights duration_conv1, duration_conv2, duration_proj;
  mod::NormWeights duration_norm1, duration_norm2;
  std::vector<ResidualBlock> decoder;
  mod::Conv1dWeights decoder_post;
  std::array<ConvBn, 2> decoder_postnet;
  mod::Conv1dWeights mel_projection;
};

mod::BatchNorm1dEvalWeights load_bn(core::BackendWeightStore &store,
                                    const engine::assets::TensorSource &source,
                                    const std::string &prefix,
                                    int64_t channels) {
  return {
      store.load_f32_tensor(source, prefix + ".scale", {channels}),
      store.load_f32_tensor(source, prefix + ".offset", {channels}),
  };
}

ConvBn load_conv_bn(core::BackendWeightStore &store,
                    const engine::assets::TensorSource &source,
                    const std::string &prefix, int64_t channels,
                    int64_t kernel) {
  ConvBn out;
  out.conv =
      binding::conv1d_from_source(store, source, prefix + ".conv1d",
                                  engine::assets::TensorStorageType::Native,
                                  channels, channels, kernel, true);
  out.norm = load_bn(store, source, prefix + ".norm", channels);
  return out;
}

std::shared_ptr<const Weights> load_weights(const Assets &assets,
                                            core::ExecutionContext &execution) {
  auto out = std::make_shared<Weights>();
  out->store = std::make_shared<core::BackendWeightStore>(
      execution.backend(), execution.backend_type(),
      "coqui_speedy_speech.weights", kWeightArena);
  auto &store = *out->store;
  const auto &source = *assets.weights;
  constexpr int64_t c = 128;
  out->embedding =
      store.load_tensor(source, "acoustic.emb.weight",
                        engine::assets::TensorStorageType::Native, {130, c});
  out->encoder_pre = binding::conv1d_from_source(
      store, source, "acoustic.encoder.encoder.prenet.0",
      engine::assets::TensorStorageType::Native, c, c, 1, true);
  for (size_t i = 0; i < assets.config.encoder_dilations.size(); ++i) {
    ResidualBlock block;
    block.dilation = static_cast<int>(assets.config.encoder_dilations[i]);
    for (int j = 0; j < 2; ++j)
      block.layers[static_cast<size_t>(j)] = load_conv_bn(
          store, source,
          "acoustic.encoder.encoder.res_conv_block.res_blocks." +
              std::to_string(i) + ".conv_bn_blocks." + std::to_string(j),
          c, 4);
    out->encoder.push_back(std::move(block));
  }
  out->encoder_post0 = binding::conv1d_from_source(
      store, source, "acoustic.encoder.encoder.postnet.0",
      engine::assets::TensorStorageType::Native, c, c, 1, true);
  out->encoder_post_norm =
      load_bn(store, source, "acoustic.encoder.encoder.postnet.2", c);
  out->encoder_post3 = binding::conv1d_from_source(
      store, source, "acoustic.encoder.encoder.postnet.3",
      engine::assets::TensorStorageType::Native, c, c, 1, true);
  out->duration_conv1 = binding::conv1d_from_source(
      store, source, "acoustic.duration_predictor.conv_1",
      engine::assets::TensorStorageType::Native, 256, c, 3, true);
  out->duration_conv2 = binding::conv1d_from_source(
      store, source, "acoustic.duration_predictor.conv_2",
      engine::assets::TensorStorageType::Native, 256, 256, 3, true);
  out->duration_proj = binding::conv1d_from_source(
      store, source, "acoustic.duration_predictor.proj",
      engine::assets::TensorStorageType::Native, 1, 256, 1, true);
  out->duration_norm1 = {
      store.load_f32_tensor(source, "acoustic.duration_predictor.norm_1.gamma",
                            {256}),
      store.load_f32_tensor(source, "acoustic.duration_predictor.norm_1.beta",
                            {256})};
  out->duration_norm2 = {
      store.load_f32_tensor(source, "acoustic.duration_predictor.norm_2.gamma",
                            {256}),
      store.load_f32_tensor(source, "acoustic.duration_predictor.norm_2.beta",
                            {256})};
  for (size_t i = 0; i < assets.config.decoder_dilations.size(); ++i) {
    ResidualBlock block;
    block.dilation = static_cast<int>(assets.config.decoder_dilations[i]);
    for (int j = 0; j < 2; ++j)
      block.layers[static_cast<size_t>(j)] = load_conv_bn(
          store, source,
          "acoustic.decoder.decoder.res_conv_block.res_blocks." +
              std::to_string(i) + ".conv_bn_blocks." + std::to_string(j),
          c, 4);
    out->decoder.push_back(std::move(block));
  }
  out->decoder_post = binding::conv1d_from_source(
      store, source, "acoustic.decoder.decoder.post_conv",
      engine::assets::TensorStorageType::Native, c, c, 1, true);
  for (int j = 0; j < 2; ++j)
    out->decoder_postnet[static_cast<size_t>(j)] =
        load_conv_bn(store, source,
                     "acoustic.decoder.decoder.postnet.0.conv_bn_blocks." +
                         std::to_string(j),
                     c, 4);
  out->mel_projection = binding::conv1d_from_source(
      store, source, "acoustic.decoder.decoder.postnet.1",
      engine::assets::TensorStorageType::Native, 80, c, 1, true);
  store.upload();
  return out;
}

core::TensorValue contiguous(core::ModuleBuildContext &ctx,
                             core::TensorValue x) {
  return core::has_backend_addressable_layout(x.tensor)
             ? x
             : core::wrap_tensor(ggml_cont(ctx.ggml, x.tensor), x.shape,
                                 x.type);
}
core::TensorValue add(core::ModuleBuildContext &ctx, const core::TensorValue &a,
                      const core::TensorValue &b) {
  return mod::AddModule{}.build(ctx, a, b);
}
core::TensorValue conv(core::ModuleBuildContext &ctx, core::TensorValue x,
                       const mod::Conv1dWeights &w, int64_t in, int64_t out,
                       int kernel, int dilation = 1) {
  return mod::Conv1dModule({in, out, kernel, 1, 0, dilation, true})
      .build(ctx, contiguous(ctx, x), w);
}
core::TensorValue pad_time(core::ModuleBuildContext &ctx, core::TensorValue x,
                           int left, int right) {
  auto shape = x.shape;
  shape.dims[2] += left + right;
  return core::wrap_tensor(ggml_pad_ext(ctx.ggml, contiguous(ctx, x).tensor,
                                        left, right, 0, 0, 0, 0, 0, 0),
                           shape, GGML_TYPE_F32);
}
core::TensorValue conv_bn(core::ModuleBuildContext &ctx, core::TensorValue x,
                          const ConvBn &w, int dilation) {
  x = conv(ctx, x, w.conv, 128, 128, 4, dilation);
  const int total = dilation * 3;
  x = pad_time(ctx, x, total / 2, total - total / 2);
  x = mod::ReluModule{}.build(ctx, x);
  return mod::BatchNorm1dEvalModule({128}).build(ctx, x, w.norm);
}
core::TensorValue residual_stack(core::ModuleBuildContext &ctx,
                                 core::TensorValue x,
                                 const std::vector<ResidualBlock> &blocks) {
  for (const auto &block : blocks) {
    auto y = conv_bn(ctx, x, block.layers[0], block.dilation);
    y = conv_bn(ctx, y, block.layers[1], block.dilation);
    x = add(ctx, x, y);
  }
  return x;
}
core::TensorValue channel_norm(core::ModuleBuildContext &ctx,
                               core::TensorValue x,
                               const mod::NormWeights &weights,
                               int64_t channels) {
  auto btc = mod::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
  btc = mod::LayerNormModule({channels, 1.0e-5F, true, true})
            .build(ctx, btc, weights);
  return mod::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, btc);
}
core::TensorValue encode(core::ModuleBuildContext &ctx, const Weights &w,
                         const core::TensorValue &tokens) {
  auto x = mod::EmbeddingModule({130, 128}).build(ctx, tokens, w.embedding);
  x = mod::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
  const auto residual = x;
  x = mod::ReluModule{}.build(ctx, conv(ctx, x, w.encoder_pre, 128, 128, 1));
  x = residual_stack(ctx, x, w.encoder);
  x = add(ctx, x, residual);
  x = mod::ReluModule{}.build(ctx, conv(ctx, x, w.encoder_post0, 128, 128, 1));
  x = mod::BatchNorm1dEvalModule({128}).build(ctx, x, w.encoder_post_norm);
  return conv(ctx, x, w.encoder_post3, 128, 128, 1);
}
core::TensorValue durations(core::ModuleBuildContext &ctx, const Weights &w,
                            core::TensorValue x) {
  // Glow-TTS DurationPredictor uses Conv1d(..., padding=kernel_size / 2),
  // i.e. zero padding is applied to the input. Padding the convolution output
  // is not equivalent at the sequence boundaries because it drops the bias
  // and all valid boundary responses.
  x = conv(ctx, pad_time(ctx, x, 1, 1), w.duration_conv1, 128, 256, 3);
  x = mod::ReluModule{}.build(ctx, x);
  x = channel_norm(ctx, x, w.duration_norm1, 256);
  x = conv(ctx, pad_time(ctx, x, 1, 1), w.duration_conv2, 256, 256, 3);
  x = mod::ReluModule{}.build(ctx, x);
  x = channel_norm(ctx, x, w.duration_norm2, 256);
  return conv(ctx, x, w.duration_proj, 256, 1, 1);
}
core::TensorValue decode_mel(core::ModuleBuildContext &ctx, const Weights &w,
                             core::TensorValue x) {
  x = residual_stack(ctx, x, w.decoder);
  x = add(ctx, conv(ctx, x, w.decoder_post, 128, 128, 1), x);
  x = conv_bn(ctx, x, w.decoder_postnet[0], 1);
  x = conv_bn(ctx, x, w.decoder_postnet[1], 1);
  return conv(ctx, x, w.mel_projection, 128, 80, 1);
}

struct Graph {
  ~Graph() {
    if (plan.active())
      plan.reset();
    if (allocator)
      ggml_gallocr_free(allocator);
    if (io_buffer)
      ggml_backend_buffer_free(io_buffer);
  }
  std::unique_ptr<ggml_context, ContextDeleter> io, arena;
  ggml_backend_buffer_t io_buffer = nullptr;
  ggml_gallocr_t allocator = nullptr;
  core::HostGraphPlan plan;
  ggml_cgraph *graph = nullptr;
  ggml_tensor *input = nullptr;
  ggml_tensor *first = nullptr;
  ggml_tensor *second = nullptr;
};

void allocate(Graph &g, core::ExecutionContext &execution, const char *label) {
  g.io_buffer = ggml_backend_alloc_ctx_tensors(g.io.get(), execution.backend());
  g.allocator = ggml_gallocr_new(
      ggml_backend_get_default_buffer_type(execution.backend()));
  if (!g.io_buffer || !g.allocator ||
      !ggml_gallocr_reserve(g.allocator, g.graph) ||
      !ggml_gallocr_alloc_graph(g.allocator, g.graph))
    throw std::runtime_error(std::string(label) + " allocation failed");
  core::validate_backend_graph_supported(execution.backend(), g.graph, label);
  core::prepare_host_graph_plan(execution, g.graph, g.plan);
}

std::unique_ptr<Graph> encoder_graph(core::ExecutionContext &execution,
                                     const Weights &w, int64_t count) {
  auto g = std::make_unique<Graph>();
  g->io.reset(ggml_init({kIoArena, nullptr, true}));
  g->arena.reset(ggml_init({kGraphArena, nullptr, true}));
  core::ModuleBuildContext io{g->io.get(), "speedy.encoder.io",
                              execution.backend_type()};
  core::ModuleBuildContext ctx{g->arena.get(), "speedy.encoder",
                               execution.backend_type()};
  auto input = core::make_tensor(io, GGML_TYPE_I32,
                                 core::TensorShape::from_dims({1, count}));
  ggml_set_input(input.tensor);
  auto hidden = contiguous(ctx, encode(ctx, w, input));
  auto duration = contiguous(ctx, durations(ctx, w, hidden));
  g->input = input.tensor;
  g->first = hidden.tensor;
  g->second = duration.tensor;
  ggml_set_output(g->first);
  ggml_set_output(g->second);
  g->graph = ggml_new_graph_custom(ctx.ggml, 8192, false);
  ggml_build_forward_expand(g->graph, g->first);
  ggml_build_forward_expand(g->graph, g->second);
  allocate(*g, execution, "Coqui SpeedySpeech encoder");
  return g;
}

std::unique_ptr<Graph> decoder_graph(core::ExecutionContext &execution,
                                     const Weights &w, int64_t frames) {
  auto g = std::make_unique<Graph>();
  g->io.reset(ggml_init({kIoArena, nullptr, true}));
  g->arena.reset(ggml_init({kGraphArena, nullptr, true}));
  core::ModuleBuildContext io{g->io.get(), "speedy.decoder.io",
                              execution.backend_type()};
  core::ModuleBuildContext ctx{g->arena.get(), "speedy.decoder",
                               execution.backend_type()};
  auto input = core::make_tensor(
      io, GGML_TYPE_F32, core::TensorShape::from_dims({1, 128, frames}));
  ggml_set_input(input.tensor);
  auto mel = contiguous(ctx, decode_mel(ctx, w, input));
  g->input = input.tensor;
  g->first = mel.tensor;
  ggml_set_output(g->first);
  g->graph = ggml_new_graph_custom(ctx.ggml, 8192, false);
  ggml_build_forward_expand(g->graph, g->first);
  allocate(*g, execution, "Coqui SpeedySpeech decoder");
  return g;
}

void compute(Graph &g, core::ExecutionContext &execution, const char *label) {
  if (core::compute_graph(execution, g.graph, g.plan, label) !=
      GGML_STATUS_SUCCESS)
    throw std::runtime_error(std::string(label) + " compute failed");
  ggml_backend_synchronize(execution.backend());
}

std::vector<float> expand(const std::vector<float> &encoded,
                          const std::vector<float> &log_duration,
                          int64_t tokens, float speaking_rate,
                          int64_t &frames) {
  std::vector<int64_t> ds(static_cast<size_t>(tokens));
  frames = 0;
  for (int64_t i = 0; i < tokens; ++i) {
    const float raw =
        (std::exp(log_duration[static_cast<size_t>(i)]) - 1.0F) / speaking_rate;
    const int64_t d = std::max<int64_t>(1, std::llround(raw));
    if (frames + d > kMaxMelFrames)
      throw std::runtime_error("SpeedySpeech duration limit exceeded");
    ds[static_cast<size_t>(i)] = d;
    frames += d;
  }
  std::vector<float> out(static_cast<size_t>(128 * frames));
  int64_t target = 0;
  for (int64_t t = 0; t < tokens; ++t)
    for (int64_t n = 0; n < ds[static_cast<size_t>(t)]; ++n, ++target)
      for (int64_t c = 0; c < 128; ++c) {
        const float exponent = static_cast<float>(2 * (c / 2)) / 128.0F;
        const float angle =
            static_cast<float>(target) * std::pow(10000.0F, exponent);
        const float pos = (c % 2 == 0) ? std::sin(angle) : std::cos(angle);
        out[static_cast<size_t>(c * frames + target)] =
            encoded[static_cast<size_t>(c * tokens + t)] * std::sqrt(128.0F) +
            pos;
      }
  return out;
}
} // namespace

struct NativeRuntime::State {
  explicit State(std::shared_ptr<const Assets> a, core::BackendConfig config)
      : assets(std::move(a)), execution(config),
        weights(load_weights(*assets, execution)),
        vocoder(mod::HifiGanVocoderComponent::load_from_tensor_source(
            assets->weights, config,
            {22050,
             80,
             128,
             1,
             {8, 8, 2, 2},
             {16, 16, 4, 4},
             {3, 7, 11},
             {{1, 3, 5}, {1, 3, 5}, {1, 3, 5}},
             mod::HifiGanResBlockKind::PairedConv,
             0.1F,
             0.01F,
             true,
             engine::assets::TensorStorageType::Native,
             "vocoder",
             {},
             {},
             false,
             true})) {}
  std::shared_ptr<const Assets> assets;
  core::ExecutionContext execution;
  std::shared_ptr<const Weights> weights;
  mod::HifiGanVocoderComponent vocoder;
};

NativeRuntime::NativeRuntime(std::shared_ptr<const Assets> assets,
                             core::BackendConfig backend)
    : state_(std::make_unique<State>(std::move(assets), backend)) {}
NativeRuntime::~NativeRuntime() = default;

engine::runtime::AudioBuffer
NativeRuntime::synthesize(const std::vector<int32_t> &tokens,
                          float speaking_rate) {
  if (tokens.empty() || speaking_rate <= 0.0F)
    throw std::runtime_error("invalid SpeedySpeech request");
  auto eg = encoder_graph(state_->execution, *state_->weights,
                          static_cast<int64_t>(tokens.size()));
  core::write_tensor_i32(
      core::wrap_tensor(eg->input,
                        core::TensorShape::from_dims(
                            {1, static_cast<int64_t>(tokens.size())}),
                        GGML_TYPE_I32),
      tokens);
  compute(*eg, state_->execution, "Coqui SpeedySpeech encoder");
  const auto encoded = core::read_tensor_f32(eg->first);
  const auto duration = core::read_tensor_f32(eg->second);
  int64_t frames = 0;
  const auto expanded =
      expand(encoded, duration, static_cast<int64_t>(tokens.size()),
             speaking_rate, frames);
  auto dg = decoder_graph(state_->execution, *state_->weights, frames);
  core::write_tensor_f32(
      core::wrap_tensor(dg->input,
                        core::TensorShape::from_dims({1, 128, frames}),
                        GGML_TYPE_F32),
      expanded);
  compute(*dg, state_->execution, "Coqui SpeedySpeech decoder");
  const auto mel = core::read_tensor_f32(dg->first);
  const auto wave = state_->vocoder.synthesize(mel, frames);
  engine::runtime::AudioBuffer out;
  out.sample_rate = wave.sample_rate;
  out.channels = 1;
  out.samples = wave.waveform;
  for (float &value : out.samples)
    value = std::clamp(value, -1.0F, 1.0F);
  return out;
}

} // namespace engine::community_models::coqui_speedy_speech
