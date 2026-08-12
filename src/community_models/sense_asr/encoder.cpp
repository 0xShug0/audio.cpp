#include "engine/community_models/sense_asr/encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::sense_asr {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kEncoderGraphNodes = 32768;
constexpr size_t kWeightContextBytes = 64 * 1024 * 1024;
constexpr float kLayerNormEpsilon = 1.0e-5F;

struct LinearWeights {
  engine::core::TensorValue weight;
  engine::core::TensorValue bias;
};

struct SanmBlockWeights {
  engine::core::TensorValue norm1_weight;
  engine::core::TensorValue norm1_bias;
  LinearWeights linear_q_k_v;
  LinearWeights linear_out;
  engine::core::TensorValue fsmn_block;
  LinearWeights w_1;
  LinearWeights w_2;
  engine::core::TensorValue norm2_weight;
  engine::core::TensorValue norm2_bias;
};

LinearWeights load_linear(engine::core::BackendWeightStore &store,
                          const engine::assets::TensorSource &source,
                          const std::string &prefix, int64_t input_size,
                          int64_t output_size,
                          engine::assets::TensorStorageType storage_type) {
  return {
      store.load_tensor(source, prefix + ".weight", storage_type,
                        {output_size, input_size}),
      store.load_f32_tensor(source, prefix + ".bias", {output_size}),
  };
}

SanmBlockWeights load_sanm_block(engine::core::BackendWeightStore &store,
                                 const engine::assets::TensorSource &source,
                                 const std::string &prefix, int64_t input_size,
                                 const SenseAsrEncoderConfig &config,
                                 engine::assets::TensorStorageType storage_type) {
  SanmBlockWeights weights;
  weights.norm1_weight = store.load_f32_tensor(source, prefix + ".norm1.weight",
                                               {input_size});
  weights.norm1_bias =
      store.load_f32_tensor(source, prefix + ".norm1.bias", {input_size});
  weights.linear_q_k_v =
      load_linear(store, source, prefix + ".self_attn.linear_q_k_v",
                  input_size, 3 * config.d_model, storage_type);
  weights.linear_out =
      load_linear(store, source, prefix + ".self_attn.linear_out",
                  config.d_model, config.d_model, storage_type);
  weights.fsmn_block = store.load_f32_tensor(
      source, prefix + ".self_attn.fsmn_block.weight",
      {config.kernel_size, config.d_model});
  weights.w_1 = load_linear(store, source, prefix + ".feed_forward.w_1",
                            config.d_model, config.ffn_dim, storage_type);
  weights.w_2 = load_linear(store, source, prefix + ".feed_forward.w_2",
                            config.ffn_dim, config.d_model, storage_type);
  weights.norm2_weight = store.load_f32_tensor(source, prefix + ".norm2.weight",
                                               {config.d_model});
  weights.norm2_bias =
      store.load_f32_tensor(source, prefix + ".norm2.bias", {config.d_model});
  return weights;
}

struct EncoderWeights {
  std::unique_ptr<engine::core::BackendWeightStore> store;
  SanmBlockWeights stem;
  std::vector<SanmBlockWeights> main_layers;
  engine::core::TensorValue after_norm_weight;
  engine::core::TensorValue after_norm_bias;
  std::vector<SanmBlockWeights> timestamp_layers;
  engine::core::TensorValue tp_norm_weight;
  engine::core::TensorValue tp_norm_bias;
  LinearWeights ctc_lo;
  std::vector<float> embed_rows;
};

std::unique_ptr<EncoderWeights>
load_encoder_weights(const SenseAsrAssets &assets,
                     engine::core::ExecutionContext &execution_context,
                     engine::assets::TensorStorageType storage_type) {
  auto weights = std::make_unique<EncoderWeights>();
  weights->store = std::make_unique<engine::core::BackendWeightStore>(
      execution_context.backend(), execution_context.backend_type(),
      "SenseVoice-Small encoder weights", kWeightContextBytes);
  const auto &source = *assets.model_weights;
  const auto &config = assets.config.encoder;
  const std::string stem_root = "encoder.encoders0.0";
  const std::string main_root = "encoder.encoders.";
  const std::string tp_root = "encoder.tp_encoders.";

  weights->stem =
      load_sanm_block(*weights->store, source, stem_root, config.input_size,
                      config, storage_type);
  weights->main_layers.reserve(static_cast<size_t>(config.num_blocks - 1));
  for (int64_t index = 0; index < config.num_blocks - 1; ++index) {
    weights->main_layers.push_back(load_sanm_block(
        *weights->store, source, main_root + std::to_string(index),
        config.d_model, config, storage_type));
  }
  weights->after_norm_weight = weights->store->load_f32_tensor(
      source, "encoder.after_norm.weight", {config.d_model});
  weights->after_norm_bias = weights->store->load_f32_tensor(
      source, "encoder.after_norm.bias", {config.d_model});
  weights->timestamp_layers.reserve(
      static_cast<size_t>(config.timestamp_prediction_layers));
  for (int64_t index = 0; index < config.timestamp_prediction_layers; ++index) {
    weights->timestamp_layers.push_back(load_sanm_block(
        *weights->store, source, tp_root + std::to_string(index),
        config.d_model, config, storage_type));
  }
  weights->tp_norm_weight = weights->store->load_f32_tensor(
      source, "encoder.tp_norm.weight", {config.d_model});
  weights->tp_norm_bias = weights->store->load_f32_tensor(
      source, "encoder.tp_norm.bias", {config.d_model});
  weights->ctc_lo =
      load_linear(*weights->store, source, "ctc.ctc_lo", config.d_model,
                  config.vocab_size, storage_type);

  const auto embed =
      source.require_f32("embed.weight", {16, config.input_size});
  if (static_cast<int64_t>(embed.size()) != 16 * config.input_size) {
    throw std::runtime_error(
        "SenseVoice-Small embed.weight has an unexpected size");
  }
  weights->embed_rows = embed;

  weights->store->upload();
  return weights;
}

ggml_tensor *lin(ggml_context *ctx, ggml_tensor *weight, ggml_tensor *bias,
                 ggml_tensor *x) {
  auto *y = ggml_mul_mat(ctx, weight, x);
  return bias != nullptr ? ggml_add(ctx, y, bias) : y;
}

ggml_tensor *lnorm(ggml_context *ctx, ggml_tensor *gamma, ggml_tensor *beta,
                   ggml_tensor *x) {
  return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, kLayerNormEpsilon),
                                gamma),
                  beta);
}

ggml_tensor *sanm_attn(ggml_context *ctx, const SanmBlockWeights &weights,
                       const SenseAsrEncoderConfig &config, ggml_tensor *x,
                       int64_t frames) {
  const int64_t d_model = config.d_model;
  const int64_t heads = config.attention_heads;
  const int64_t head_dim = d_model / heads;
  const int64_t kernel = config.kernel_size;

  ggml_tensor *qkv =
      lin(ctx, weights.linear_q_k_v.weight.tensor,
          weights.linear_q_k_v.bias.tensor, x);
  const size_t qkv_row_stride = qkv->nb[1];
  ggml_tensor *q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, d_model, frames,
                                                qkv_row_stride, 0));
  ggml_tensor *k = ggml_cont(
      ctx, ggml_view_2d(ctx, qkv, d_model, frames, qkv_row_stride,
                        static_cast<size_t>(d_model) * sizeof(float)));
  ggml_tensor *v = ggml_cont(
      ctx, ggml_view_2d(ctx, qkv, d_model, frames, qkv_row_stride,
                        2 * static_cast<size_t>(d_model) * sizeof(float)));

  const int64_t pad = (kernel - 1) / 2;
  ggml_tensor *padded =
      ggml_pad_ext(ctx, v, 0, 0, pad, pad, 0, 0, 0, 0);
  ggml_tensor *fsmn = v;
  for (int64_t j = 0; j < kernel; ++j) {
    ggml_tensor *shifted =
        ggml_cont(ctx, ggml_view_2d(ctx, padded, d_model, frames,
                                    padded->nb[1], j * padded->nb[1]));
    ggml_tensor *wj = ggml_view_1d(ctx, weights.fsmn_block.tensor, d_model,
                                   j * weights.fsmn_block.tensor->nb[1]);
    fsmn = ggml_add(ctx, fsmn, ggml_mul(ctx, shifted, wj));
  }

  q = ggml_permute(ctx, ggml_reshape_3d(ctx, q, head_dim, heads, frames), 0, 2,
                   1, 3);
  k = ggml_permute(ctx, ggml_reshape_3d(ctx, k, head_dim, heads, frames), 0, 2,
                   1, 3);
  ggml_tensor *value_heads = ggml_cont(
      ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, head_dim, heads, frames),
                        1, 2, 0, 3));
  ggml_tensor *scores =
      ggml_soft_max(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, k, q),
                                    1.0F / std::sqrt(static_cast<float>(head_dim))));
  ggml_tensor *attention = ggml_cont_2d(
      ctx, ggml_permute(ctx, ggml_mul_mat(ctx, value_heads, scores), 0, 2, 1,
                        3),
      d_model, frames);
  return ggml_add(
      ctx,
      lin(ctx, weights.linear_out.weight.tensor,
          weights.linear_out.bias.tensor, attention),
      fsmn);
}

ggml_tensor *sanm_layer(ggml_context *ctx, const SanmBlockWeights &weights,
                        const SenseAsrEncoderConfig &config, ggml_tensor *x,
                        int64_t frames, bool residual) {
  ggml_tensor *input = x;
  ggml_tensor *h =
      lnorm(ctx, weights.norm1_weight.tensor, weights.norm1_bias.tensor, x);
  ggml_tensor *attention = sanm_attn(ctx, weights, config, h, frames);
  x = residual ? ggml_add(ctx, input, attention) : attention;
  ggml_tensor *r = x;
  h = lnorm(ctx, weights.norm2_weight.tensor, weights.norm2_bias.tensor, x);
  h = lin(ctx, weights.w_1.weight.tensor, weights.w_1.bias.tensor, h);
  h = ggml_relu(ctx, h);
  h = lin(ctx, weights.w_2.weight.tensor, weights.w_2.bias.tensor, h);
  return ggml_add(ctx, r, h);
}

std::vector<float> make_posenc_input(int64_t frames, int64_t channels) {
  if (channels <= 2 || channels % 2 != 0) {
    throw std::runtime_error(
        "SenseVoice positional encoding channel shape is invalid");
  }
  const int64_t half = channels / 2;
  const double increment = std::log(10000.0) / (static_cast<double>(half) - 1.0);
  std::vector<float> values(static_cast<size_t>(frames * channels), 0.0F);
  for (int64_t frame = 0; frame < frames; ++frame) {
    const double position = static_cast<double>(frame + 1);
    for (int64_t index = 0; index < half; ++index) {
      const double inverse_timescale = std::exp(static_cast<double>(index) * -increment);
      const double phase = position * inverse_timescale;
      const size_t base = static_cast<size_t>(frame * channels + index);
      values[base] = static_cast<float>(std::sin(phase));
      values[base + static_cast<size_t>(half)] = static_cast<float>(std::cos(phase));
    }
  }
  return values;
}

} // namespace

struct SenseAsrEncoderRuntime::Impl {
  struct Graph {
    int64_t frames = 0;
    ggml_backend_t backend = nullptr;
    ggml_context *ggml = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph *graph = nullptr;
    engine::core::HostGraphPlan host_plan;
    engine::core::TensorValue input;
    ggml_tensor *output = nullptr;

    ~Graph() {
      host_plan.reset();
      if (backend != nullptr && graph != nullptr) {
        engine::core::release_backend_graph_resources(backend, graph);
      }
      if (allocator != nullptr) {
        ggml_gallocr_free(allocator);
      }
      if (ggml != nullptr) {
        ggml_free(ggml);
      }
    }
  };

  Impl(std::shared_ptr<const SenseAsrAssets> assets_value,
       engine::core::ExecutionContext &execution_context_value,
       size_t graph_arena_bytes_value,
       engine::assets::TensorStorageType weight_storage)
      : assets(std::move(assets_value)),
        execution_context(&execution_context_value),
        graph_arena_bytes(graph_arena_bytes_value) {
    if (assets == nullptr || assets->model_weights == nullptr) {
      throw std::runtime_error(
          "SenseVoice encoder requires model assets and weights");
    }
    if (graph_arena_bytes == 0) {
      throw std::runtime_error(
          "SenseVoice encoder graph arena must be non-zero");
    }
    query_tokens = assets->config.encoder.query_tokens;
    weights =
        load_encoder_weights(*assets, *execution_context, weight_storage);
  }

  void ensure_graph(int64_t frames) {
    const auto &config = assets->config.encoder;
    const int64_t nq = static_cast<int64_t>(query_tokens.size());
    const int64_t total = nq + frames;
    if (total > config.max_frames) {
      throw std::runtime_error(
          "SenseVoice encoder input exceeds positional capacity");
    }
    if (cached_graph != nullptr && cached_graph->frames == frames &&
        cached_graph->backend == execution_context->backend()) {
      return;
    }

    const auto build_start = Clock::now();
    auto next = std::make_unique<Graph>();
    next->frames = frames;
    next->backend = execution_context->backend();
    ggml_init_params params{};
    params.mem_size = graph_arena_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    next->ggml = ggml_init(params);
    if (next->ggml == nullptr) {
      throw std::runtime_error(
          "failed to initialize SenseVoice encoder graph context");
    }

    auto *x = ggml_new_tensor_2d(next->ggml, GGML_TYPE_F32,
                                 config.input_size, total);
    ggml_set_input(x);
    next->input = engine::core::wrap_tensor(
        x, engine::core::TensorShape::from_dims(
               {total, config.input_size}));

    ggml_tensor *h =
        sanm_layer(next->ggml, weights->stem, config, x, total, false);
    for (size_t index = 0; index < weights->main_layers.size(); ++index) {
      h = sanm_layer(next->ggml, weights->main_layers[index], config, h, total,
                     true);
    }
    h = lnorm(next->ggml, weights->after_norm_weight.tensor,
              weights->after_norm_bias.tensor, h);
    for (size_t index = 0; index < weights->timestamp_layers.size(); ++index) {
      h = sanm_layer(next->ggml, weights->timestamp_layers[index], config, h,
                     total, true);
    }
    h = lnorm(next->ggml, weights->tp_norm_weight.tensor,
              weights->tp_norm_bias.tensor, h);
    next->output =
        lin(next->ggml, weights->ctc_lo.weight.tensor,
            weights->ctc_lo.bias.tensor, h);
    ggml_set_output(next->output);

    next->graph =
        ggml_new_graph_custom(next->ggml, kEncoderGraphNodes, false);
    ggml_build_forward_expand(next->graph, next->output);
    engine::core::validate_backend_graph_supported(next->backend, next->graph,
                                                   "SenseVoice encoder");
    next->allocator =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(next->backend));
    if (next->allocator == nullptr ||
        !ggml_gallocr_reserve(next->allocator, next->graph) ||
        !ggml_gallocr_alloc_graph(next->allocator, next->graph)) {
      throw std::runtime_error(
          "failed to allocate SenseVoice encoder graph tensors");
    }
    engine::core::prepare_host_graph_plan(*execution_context, next->graph,
                                          next->host_plan);

    cached_graph = std::move(next);
    engine::debug::timing_log_scalar(
        "sense_asr.encoder.graph_build_ms",
        engine::debug::elapsed_ms(build_start, Clock::now()));
  }

  SenseAsrEncoderOutput encode(const SenseAsrAudioFeatures &features) {
    const auto &config = assets->config.encoder;
    if (features.frames < 0 || features.feature_dim != config.input_size) {
      throw std::runtime_error("SenseVoice encoder input shape is invalid");
    }
    if (static_cast<int64_t>(features.values.size()) !=
        features.frames * features.feature_dim) {
      throw std::runtime_error(
          "SenseVoice encoder input value count mismatch");
    }
    if (features.frames <= 0) {
      return SenseAsrEncoderOutput{};
    }

    const auto encode_start = Clock::now();
    const int64_t nq = static_cast<int64_t>(query_tokens.size());
    const int64_t total = nq + features.frames;
    ensure_graph(features.frames);

    std::vector<float> input(static_cast<size_t>(total) * config.input_size, 0.0F);
    for (int64_t i = 0; i < nq; ++i) {
      const int32_t token = query_tokens[static_cast<size_t>(i)];
      if (token < 0 || token >= 16) {
        throw std::runtime_error(
            "SenseVoice query token is outside the embedding table");
      }
      const float *row =
          weights->embed_rows.data() + static_cast<size_t>(token) * static_cast<size_t>(config.input_size);
      std::copy(row, row + config.input_size,
                input.data() + static_cast<size_t>(i) * static_cast<size_t>(config.input_size));
    }
    std::copy(features.values.begin(), features.values.end(),
              input.begin() + static_cast<ptrdiff_t>(nq) * config.input_size);

    const float scale = std::sqrt(static_cast<float>(config.d_model));
    for (auto &value : input) {
      value *= scale;
    }
    const auto positions =
        make_posenc_input(total, config.input_size);
    for (size_t i = 0; i < input.size(); ++i) {
      input[i] += positions[i];
    }

    engine::core::write_tensor_f32(cached_graph->input, input);
    engine::core::set_backend_threads(execution_context->backend(),
                                      execution_context->config().threads);
    // Bypass host graph plan to match reference behavior exactly
    const auto status = ggml_backend_graph_compute(execution_context->backend(),
                                                   cached_graph->graph);
    if (status != GGML_STATUS_SUCCESS) {
      throw std::runtime_error("SenseVoice encoder graph execution failed");
    }

    SenseAsrEncoderOutput output;
    output.logits = engine::core::read_tensor_f32(cached_graph->output);
    output.frames = total;
    output.vocab_size = config.vocab_size;
    engine::debug::timing_log_scalar(
        "sense_asr.encoder_ms",
        engine::debug::elapsed_ms(encode_start, Clock::now()));
    return output;
  }

  std::shared_ptr<const SenseAsrAssets> assets;
  engine::core::ExecutionContext *execution_context = nullptr;
  size_t graph_arena_bytes = 0;
  std::vector<int32_t> query_tokens;
  std::unique_ptr<EncoderWeights> weights;
  std::unique_ptr<Graph> cached_graph;
};

SenseAsrEncoderRuntime::SenseAsrEncoderRuntime(
    std::shared_ptr<const SenseAsrAssets> assets,
    engine::core::ExecutionContext &execution_context,
    size_t graph_arena_bytes,
    engine::assets::TensorStorageType weight_storage)
    : impl_(std::make_unique<Impl>(std::move(assets), execution_context,
                                   graph_arena_bytes, weight_storage)) {}

SenseAsrEncoderRuntime::~SenseAsrEncoderRuntime() = default;
SenseAsrEncoderRuntime::SenseAsrEncoderRuntime(
    SenseAsrEncoderRuntime &&) noexcept = default;
SenseAsrEncoderRuntime &SenseAsrEncoderRuntime::operator=(
    SenseAsrEncoderRuntime &&) noexcept = default;

void SenseAsrEncoderRuntime::prepare_capacity(int64_t frames) {
  if (impl_ == nullptr) {
    throw std::runtime_error("SenseVoice encoder runtime is moved from");
  }
  impl_->ensure_graph(frames);
}

void SenseAsrEncoderRuntime::set_query_tokens(
    std::vector<int32_t> query_tokens_value) {
  if (impl_ == nullptr) {
    throw std::runtime_error("SenseVoice encoder runtime is moved from");
  }
  if (query_tokens_value.empty()) {
    throw std::runtime_error("SenseVoice query tokens must not be empty");
  }
  for (int32_t token : query_tokens_value) {
    if (token < 0 || token >= 16) {
      throw std::runtime_error(
          "SenseVoice query token is outside the embedding table");
    }
  }
  impl_->query_tokens = std::move(query_tokens_value);
}

SenseAsrEncoderOutput
SenseAsrEncoderRuntime::encode(const SenseAsrAudioFeatures &features) {
  if (impl_ == nullptr) {
    throw std::runtime_error("SenseVoice encoder runtime is moved from");
  }
  return impl_->encode(features);
}

} // namespace engine::community_models::sense_asr
