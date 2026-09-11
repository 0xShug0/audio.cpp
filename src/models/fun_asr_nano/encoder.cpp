#include "engine/models/fun_asr_nano/encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/speech_encoders/sanm.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::fun_asr_nano {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kEncoderGraphNodes = 131072;
constexpr size_t kWeightContextBytes = 32 * 1024 * 1024;
constexpr float kLayerNormEpsilon = 1.0e-5F;

engine::modules::LinearWeights
load_linear(engine::core::BackendWeightStore &store,
            const engine::assets::TensorSource &source,
            const std::string &prefix, int64_t input_size, int64_t output_size,
            engine::assets::TensorStorageType storage_type) {
  return {
      store.load_tensor(source, prefix + ".weight", storage_type,
                        {output_size, input_size}),
      store.load_f32_tensor(source, prefix + ".bias", {output_size}),
  };
}

engine::modules::NormWeights
load_norm(engine::core::BackendWeightStore &store,
          const engine::assets::TensorSource &source, const std::string &prefix,
          int64_t size) {
  return {
      store.load_f32_tensor(source, prefix + ".weight", {size}),
      store.load_f32_tensor(source, prefix + ".bias", {size}),
  };
}

engine::modules::SanmBlockWeightsView
load_sanm_block(engine::core::BackendWeightStore &store,
                const engine::assets::TensorSource &source,
                const std::string &prefix, int64_t input_size,
                const FunAsrNanoEncoderConfig &config,
                engine::assets::TensorStorageType storage_type) {
  return {
      load_norm(store, source, prefix + ".self_attn_layer_norm", input_size),
      load_linear(store, source, prefix + ".self_attn.q_proj", input_size,
                  config.d_model, storage_type),
      load_linear(store, source, prefix + ".self_attn.k_proj", input_size,
                  config.d_model, storage_type),
      load_linear(store, source, prefix + ".self_attn.v_proj", input_size,
                  config.d_model, storage_type),
      load_linear(store, source, prefix + ".self_attn.out_proj", config.d_model,
                  config.d_model, storage_type),
      store.load_f32_tensor(source, prefix + ".fsmn.conv.weight",
                            {config.d_model, 1, config.kernel_size}),
      load_norm(store, source, prefix + ".final_layer_norm", config.d_model),
      load_linear(store, source, prefix + ".fc1", config.d_model,
                  config.ffn_dim, storage_type),
      load_linear(store, source, prefix + ".fc2", config.ffn_dim,
                  config.d_model, storage_type),
  };
}

struct EncoderWeights {
  std::unique_ptr<engine::core::BackendWeightStore> store;
  engine::modules::SanmBlockWeightsView stem;
  std::vector<engine::modules::SanmBlockWeightsView> main_layers;
  engine::modules::NormWeights main_norm;
  std::vector<engine::modules::SanmBlockWeightsView> timestamp_layers;
  engine::modules::NormWeights timestamp_norm;
};

std::unique_ptr<EncoderWeights>
load_encoder_weights(const FunAsrNanoAssets &assets,
                     engine::core::ExecutionContext &execution_context,
                     engine::assets::TensorStorageType storage_type) {
  auto weights = std::make_unique<EncoderWeights>();
  weights->store = std::make_unique<engine::core::BackendWeightStore>(
      execution_context.backend(), execution_context.backend_type(),
      "Fun-ASR-Nano encoder weights", kWeightContextBytes);
  const auto &source = *assets.model_weights;
  const auto &config = assets.config.encoder;
  const std::string root = "model.audio_tower.";

  weights->stem = load_sanm_block(*weights->store, source, root + "stem",
                                  config.input_size, config, storage_type);
  weights->main_layers.reserve(static_cast<size_t>(config.layers - 1));
  for (int64_t index = 0; index < config.layers - 1; ++index) {
    weights->main_layers.push_back(load_sanm_block(
        *weights->store, source, root + "layers." + std::to_string(index),
        config.d_model, config, storage_type));
  }
  weights->main_norm =
      load_norm(*weights->store, source, root + "layer_norm", config.d_model);
  weights->timestamp_layers.reserve(
      static_cast<size_t>(config.timestamp_prediction_layers));
  for (int64_t index = 0; index < config.timestamp_prediction_layers; ++index) {
    weights->timestamp_layers.push_back(load_sanm_block(
        *weights->store, source,
        root + "timestamp_prediction_layers." + std::to_string(index),
        config.d_model, config, storage_type));
  }
  weights->timestamp_norm =
      load_norm(*weights->store, source,
                root + "timestamp_prediction_layer_norm", config.d_model);
  weights->store->upload();
  return weights;
}

std::vector<float> make_sinusoidal_positions(int64_t frames, int64_t channels) {
  if (frames <= 0 || channels <= 2 || channels % 2 != 0) {
    throw std::runtime_error(
        "Fun-ASR-Nano sinusoidal position shape is invalid");
  }
  const int64_t half = channels / 2;
  const float increment = std::log(10000.0F) / static_cast<float>(half - 1);
  std::vector<float> values(static_cast<size_t>(frames * channels), 0.0F);
  for (int64_t frame = 0; frame < frames; ++frame) {
    const float position = static_cast<float>(frame + 1);
    for (int64_t index = 0; index < half; ++index) {
      const float inverse_timescale =
          std::exp(-increment * static_cast<float>(index));
      const float phase = position * inverse_timescale;
      const size_t base = static_cast<size_t>(frame * channels + index);
      values[base] = std::sin(phase);
      values[base + static_cast<size_t>(half)] = std::cos(phase);
    }
  }
  return values;
}

engine::modules::SanmBlockConfig
block_config(const FunAsrNanoEncoderConfig &config, int64_t input_size) {
  engine::modules::SanmBlockConfig result;
  result.input_size = input_size;
  result.model_size = config.d_model;
  result.num_heads = config.attention_heads;
  result.ffn_size = config.ffn_dim;
  result.fsmn_kernel_size = config.kernel_size;
  result.layer_norm_eps = kLayerNormEpsilon;
  result.attention_lowering =
      engine::modules::ScaledDotProductAttentionLowering::Explicit;
  return result;
}

} // namespace

struct FunAsrNanoEncoderRuntime::Impl {
  struct Graph {
    int64_t frames = 0;
    ggml_backend_t backend = nullptr;
    ggml_context *ggml = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph *graph = nullptr;
    engine::core::HostGraphPlan host_plan;
    engine::core::TensorValue input;
    engine::core::TensorValue positions;
    engine::core::TensorValue output;
    std::vector<std::pair<std::string, engine::core::TensorValue>> checkpoints;

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

  Impl(std::shared_ptr<const FunAsrNanoAssets> assets_value,
       engine::core::ExecutionContext &execution_context_value,
       size_t graph_arena_bytes_value,
       engine::assets::TensorStorageType weight_storage)
      : assets(std::move(assets_value)),
        execution_context(&execution_context_value),
        graph_arena_bytes(graph_arena_bytes_value) {
    if (assets == nullptr || assets->model_weights == nullptr) {
      throw std::runtime_error(
          "Fun-ASR-Nano encoder requires model assets and weights");
    }
    if (graph_arena_bytes == 0) {
      throw std::runtime_error(
          "Fun-ASR-Nano encoder graph arena must be non-zero");
    }
    weights = load_encoder_weights(*assets, *execution_context, weight_storage);
  }

  void ensure_graph(int64_t frames) {
    if (frames <= 0) {
      throw std::runtime_error(
          "Fun-ASR-Nano encoder graph requires positive frames");
    }
    const auto &config = assets->config.encoder;
    if (frames >= config.max_position_embeddings) {
      throw std::runtime_error(
          "Fun-ASR-Nano encoder input exceeds positional capacity");
    }
    if (cached_graph != nullptr && cached_graph->frames == frames &&
        cached_graph->backend == execution_context->backend()) {
      engine::debug::trace_log_scalar("fun_asr_nano.encoder.graph_cache_hit",
                                      true);
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
          "failed to initialize Fun-ASR-Nano encoder graph context");
    }

    engine::core::ModuleBuildContext context{next->ggml, "fun_asr_nano.encoder",
                                             execution_context->backend_type()};
    const auto input_shape =
        engine::core::TensorShape::from_dims({1, frames, config.input_size});
    next->input =
        engine::core::make_tensor(context, GGML_TYPE_F32, input_shape);
    ggml_set_input(next->input.tensor);
    ggml_set_output(next->input.tensor);
    next->positions =
        engine::core::make_tensor(context, GGML_TYPE_F32, input_shape);
    ggml_set_input(next->positions.tensor);
    ggml_set_output(next->positions.tensor);

    auto hidden = engine::core::wrap_tensor(
        ggml_scale(context.ggml, next->input.tensor,
                   std::sqrt(static_cast<float>(config.d_model))),
        input_shape, GGML_TYPE_F32);
    hidden =
        engine::modules::AddModule{}.build(context, hidden, next->positions);
    hidden = engine::modules::sanm_projection_block(
        context, hidden, weights->stem,
        block_config(config, config.input_size));
    next->checkpoints.push_back({"stem", hidden});

    const auto residual_config = block_config(config, config.d_model);
    for (size_t index = 0; index < weights->main_layers.size(); ++index) {
      hidden = engine::modules::sanm_residual_block(
          context, hidden, weights->main_layers[index], residual_config);
      if (index == 0 || index == 24 || index == 48) {
        next->checkpoints.push_back(
            {"main_layer_" + std::to_string(index), hidden});
      }
    }
    hidden = engine::modules::sanm_layer_norm(
        context, hidden, weights->main_norm, kLayerNormEpsilon);
    next->checkpoints.push_back({"main_layer_norm", hidden});

    for (size_t index = 0; index < weights->timestamp_layers.size(); ++index) {
      hidden = engine::modules::sanm_residual_block(
          context, hidden, weights->timestamp_layers[index], residual_config);
      if (index == 0 || index == 10 || index == 19) {
        next->checkpoints.push_back(
            {"timestamp_layer_" + std::to_string(index), hidden});
      }
    }
    next->output = engine::modules::sanm_layer_norm(
        context, hidden, weights->timestamp_norm, kLayerNormEpsilon);
    next->checkpoints.push_back({"final", next->output});
    for (const auto &checkpoint : next->checkpoints) {
      ggml_set_output(checkpoint.second.tensor);
    }

    next->graph = ggml_new_graph_custom(next->ggml, kEncoderGraphNodes, false);
    ggml_build_forward_expand(next->graph, next->output.tensor);
    engine::core::validate_backend_graph_supported(next->backend, next->graph,
                                                   "Fun-ASR-Nano encoder");
    next->allocator =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(next->backend));
    if (next->allocator == nullptr ||
        !ggml_gallocr_reserve(next->allocator, next->graph) ||
        !ggml_gallocr_alloc_graph(next->allocator, next->graph)) {
      throw std::runtime_error(
          "failed to allocate Fun-ASR-Nano encoder graph tensors");
    }
    engine::core::write_tensor_f32(
        next->positions, make_sinusoidal_positions(frames, config.input_size));
    engine::core::prepare_host_graph_plan(*execution_context, next->graph,
                                          next->host_plan);

    cached_graph = std::move(next);
    engine::debug::timing_log_scalar(
        "fun_asr_nano.encoder.graph_build_ms",
        engine::debug::elapsed_ms(build_start, Clock::now()));
    engine::debug::trace_log_scalar("fun_asr_nano.encoder.graph_cache_hit",
                                    false);
    engine::debug::trace_log_scalar("fun_asr_nano.encoder.graph_frames",
                                    frames);
  }

  FunAsrNanoEncoderEmbeddings encode(const FunAsrNanoAudioFeatures &features,
                                     bool capture_stages) {
    const auto &config = assets->config.encoder;
    if (features.frames <= 0 || features.feature_dim != config.input_size) {
      throw std::runtime_error("Fun-ASR-Nano encoder input shape is invalid");
    }
    if (features.frames >= config.max_position_embeddings) {
      throw std::runtime_error(
          "Fun-ASR-Nano encoder input exceeds positional capacity");
    }
    if (features.valid_frames != features.frames) {
      throw std::runtime_error(
          "Fun-ASR-Nano encoder currently requires unpadded features");
    }
    if (static_cast<int64_t>(features.values.size()) !=
        features.frames * features.feature_dim) {
      throw std::runtime_error(
          "Fun-ASR-Nano encoder input value count mismatch");
    }

    const auto encode_start = Clock::now();
    ensure_graph(features.frames);
    engine::core::write_tensor_f32(cached_graph->input, features.values);
    const auto status = engine::core::compute_graph(
        *execution_context, cached_graph->graph, cached_graph->host_plan,
        "Fun-ASR-Nano encoder");
    if (status != GGML_STATUS_SUCCESS) {
      throw std::runtime_error("Fun-ASR-Nano encoder graph execution failed");
    }

    FunAsrNanoEncoderEmbeddings output;
    output.values = engine::core::read_tensor_f32(cached_graph->output.tensor);
    output.frames = features.frames;
    output.valid_frames = features.valid_frames;
    output.hidden_size = config.d_model;
    if (capture_stages) {
      output.stages.reserve(cached_graph->checkpoints.size());
      for (const auto &checkpoint : cached_graph->checkpoints) {
        output.stages.push_back(
            {checkpoint.first,
             engine::core::read_tensor_f32(checkpoint.second.tensor)});
      }
    }
    engine::debug::timing_log_scalar(
        "fun_asr_nano.encoder_ms",
        engine::debug::elapsed_ms(encode_start, Clock::now()));
    return output;
  }

  std::shared_ptr<const FunAsrNanoAssets> assets;
  engine::core::ExecutionContext *execution_context = nullptr;
  size_t graph_arena_bytes = 0;
  std::unique_ptr<EncoderWeights> weights;
  std::unique_ptr<Graph> cached_graph;
};

FunAsrNanoEncoderRuntime::FunAsrNanoEncoderRuntime(
    std::shared_ptr<const FunAsrNanoAssets> assets,
    engine::core::ExecutionContext &execution_context, size_t graph_arena_bytes,
    engine::assets::TensorStorageType weight_storage)
    : impl_(std::make_unique<Impl>(std::move(assets), execution_context,
                                   graph_arena_bytes, weight_storage)) {}

FunAsrNanoEncoderRuntime::~FunAsrNanoEncoderRuntime() = default;
FunAsrNanoEncoderRuntime::FunAsrNanoEncoderRuntime(
    FunAsrNanoEncoderRuntime &&) noexcept = default;
FunAsrNanoEncoderRuntime &FunAsrNanoEncoderRuntime::operator=(
    FunAsrNanoEncoderRuntime &&) noexcept = default;

void FunAsrNanoEncoderRuntime::prepare_capacity(int64_t frames) {
  if (impl_ == nullptr) {
    throw std::runtime_error("Fun-ASR-Nano encoder runtime is moved from");
  }
  impl_->ensure_graph(frames);
}

FunAsrNanoEncoderEmbeddings
FunAsrNanoEncoderRuntime::encode(const FunAsrNanoAudioFeatures &features,
                                 bool capture_stages) {
  if (impl_ == nullptr) {
    throw std::runtime_error("Fun-ASR-Nano encoder runtime is moved from");
  }
  return impl_->encode(features, capture_stages);
}

} // namespace engine::models::fun_asr_nano
