#include "engine/models/fun_asr_nano/adaptor.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::fun_asr_nano {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kAdaptorGraphNodes = 16384;
constexpr size_t kWeightContextBytes = 8 * 1024 * 1024;
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

struct AdaptorLayerWeights {
  engine::modules::NormWeights attention_norm;
  engine::modules::LinearWeights q_proj;
  engine::modules::LinearWeights k_proj;
  engine::modules::LinearWeights v_proj;
  engine::modules::LinearWeights out_proj;
  engine::modules::NormWeights final_norm;
  engine::modules::LinearWeights fc1;
  engine::modules::LinearWeights fc2;
};

struct AdaptorWeights {
  std::unique_ptr<engine::core::BackendWeightStore> store;
  engine::modules::LinearWeights projector_1;
  engine::modules::LinearWeights projector_2;
  std::vector<AdaptorLayerWeights> layers;
};

AdaptorLayerWeights
load_adaptor_layer(engine::core::BackendWeightStore &store,
                   const engine::assets::TensorSource &source,
                   const std::string &prefix,
                   const FunAsrNanoAdaptorConfig &config,
                   engine::assets::TensorStorageType storage_type) {
  return {
      load_norm(store, source, prefix + ".self_attn_layer_norm",
                config.d_model),
      load_linear(store, source, prefix + ".self_attn.q_proj", config.d_model,
                  config.d_model, storage_type),
      load_linear(store, source, prefix + ".self_attn.k_proj", config.d_model,
                  config.d_model, storage_type),
      load_linear(store, source, prefix + ".self_attn.v_proj", config.d_model,
                  config.d_model, storage_type),
      load_linear(store, source, prefix + ".self_attn.out_proj", config.d_model,
                  config.d_model, storage_type),
      load_norm(store, source, prefix + ".final_layer_norm", config.d_model),
      load_linear(store, source, prefix + ".fc1", config.d_model,
                  config.ffn_dim, storage_type),
      load_linear(store, source, prefix + ".fc2", config.ffn_dim,
                  config.d_model, storage_type),
  };
}

std::unique_ptr<AdaptorWeights>
load_adaptor_weights(const FunAsrNanoAssets &assets,
                     engine::core::ExecutionContext &execution_context,
                     engine::assets::TensorStorageType storage_type) {
  auto weights = std::make_unique<AdaptorWeights>();
  weights->store = std::make_unique<engine::core::BackendWeightStore>(
      execution_context.backend(), execution_context.backend_type(),
      "Fun-ASR-Nano adaptor weights", kWeightContextBytes);
  const auto &source = *assets.model_weights;
  const auto &encoder = assets.config.encoder;
  const auto &config = assets.config.adaptor;
  const std::string projector_root = "model.multi_modal_projector.";
  weights->projector_1 = load_linear(
      *weights->store, source, projector_root + "linear_1", encoder.d_model,
      assets.config.projector_hidden_size, storage_type);
  weights->projector_2 = load_linear(
      *weights->store, source, projector_root + "linear_2",
      assets.config.projector_hidden_size, config.d_model, storage_type);

  const std::string current_root = "model.audio_adaptor.blocks.";
  const std::string legacy_root = "model.multi_modal_projector.blocks.";
  const std::string block_root =
      source.has_tensor(current_root + "0.self_attn.q_proj.weight")
          ? current_root
          : legacy_root;
  weights->layers.reserve(static_cast<size_t>(config.layers));
  for (int64_t index = 0; index < config.layers; ++index) {
    weights->layers.push_back(load_adaptor_layer(
        *weights->store, source, block_root + std::to_string(index), config,
        storage_type));
  }
  weights->store->upload();
  return weights;
}

engine::core::TensorValue linear(engine::core::ModuleBuildContext &context,
                                 const engine::core::TensorValue &input,
                                 const engine::modules::LinearWeights &weights,
                                 int64_t input_size, int64_t output_size) {
  return engine::modules::LinearModule(
             {input_size, output_size, true, GGML_PREC_F32})
      .build(context, input, weights);
}

engine::core::TensorValue
layer_norm(engine::core::ModuleBuildContext &context,
           const engine::core::TensorValue &input,
           const engine::modules::NormWeights &weights, int64_t hidden_size) {
  return engine::modules::LayerNormModule(
             {hidden_size, kLayerNormEpsilon, true, true})
      .build(context, input, weights);
}

engine::core::TensorValue split_heads(engine::core::ModuleBuildContext &context,
                                      const engine::core::TensorValue &input,
                                      int64_t heads, int64_t head_dim) {
  const auto contiguous =
      engine::core::ensure_backend_addressable_layout(context, input);
  const auto reshaped = engine::core::reshape_tensor(
      context, contiguous,
      engine::core::TensorShape::from_dims(
          {input.shape.dims[0], input.shape.dims[1], heads, head_dim}));
  return engine::modules::TransposeModule({{0, 2, 1, 3}, 4})
      .build(context, reshaped);
}

engine::core::TensorValue merge_heads(engine::core::ModuleBuildContext &context,
                                      const engine::core::TensorValue &input,
                                      int64_t hidden_size) {
  const auto contiguous =
      engine::core::ensure_backend_addressable_layout(context, input);
  return engine::core::reshape_tensor(
      context, contiguous,
      engine::core::TensorShape::from_dims(
          {input.shape.dims[0], input.shape.dims[1], hidden_size}));
}

engine::core::TensorValue
adaptor_block(engine::core::ModuleBuildContext &context,
              const engine::core::TensorValue &input,
              const AdaptorLayerWeights &weights,
              const FunAsrNanoAdaptorConfig &config) {
  const int64_t head_dim = config.d_model / config.attention_heads;
  auto normalized =
      layer_norm(context, input, weights.attention_norm, config.d_model);
  auto q = split_heads(context,
                       linear(context, normalized, weights.q_proj,
                              config.d_model, config.d_model),
                       config.attention_heads, head_dim);
  auto k = split_heads(context,
                       linear(context, normalized, weights.k_proj,
                              config.d_model, config.d_model),
                       config.attention_heads, head_dim);
  auto v = split_heads(context,
                       linear(context, normalized, weights.v_proj,
                              config.d_model, config.d_model),
                       config.attention_heads, head_dim);
  auto attention =
      engine::modules::ScaledDotProductAttentionModule(
          {head_dim,
           engine::modules::ScaledDotProductAttentionLowering::Explicit,
           GGML_PREC_F32, engine::modules::AttentionCausality::NonCausal})
          .build(context, q, k, v);
  attention = merge_heads(context, attention, config.d_model);
  attention = linear(context, attention, weights.out_proj, config.d_model,
                     config.d_model);
  auto hidden =
      engine::modules::ResidualAddModule{}.build(context, input, attention);

  normalized = layer_norm(context, hidden, weights.final_norm, config.d_model);
  auto feed_forward =
      linear(context, normalized, weights.fc1, config.d_model, config.ffn_dim);
  feed_forward = engine::modules::ReluModule{}.build(context, feed_forward);
  feed_forward = linear(context, feed_forward, weights.fc2, config.ffn_dim,
                        config.d_model);
  return engine::modules::ResidualAddModule{}.build(context, hidden,
                                                    feed_forward);
}

} // namespace

struct FunAsrNanoAdaptorRuntime::Impl {
  struct Graph {
    int64_t frames = 0;
    ggml_backend_t backend = nullptr;
    ggml_context *ggml = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph *graph = nullptr;
    engine::core::HostGraphPlan host_plan;
    engine::core::TensorValue input;
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
          "Fun-ASR-Nano adaptor requires model assets and weights");
    }
    if (graph_arena_bytes == 0) {
      throw std::runtime_error(
          "Fun-ASR-Nano adaptor graph arena must be non-zero");
    }
    const auto &config = assets->config.adaptor;
    if (config.d_model % config.attention_heads != 0) {
      throw std::runtime_error(
          "Fun-ASR-Nano adaptor width must be divisible by its head count");
    }
    weights = load_adaptor_weights(*assets, *execution_context, weight_storage);
  }

  void ensure_graph(int64_t frames) {
    if (frames <= 0) {
      throw std::runtime_error(
          "Fun-ASR-Nano adaptor graph requires positive frames");
    }
    if (cached_graph != nullptr && cached_graph->frames == frames &&
        cached_graph->backend == execution_context->backend()) {
      engine::debug::trace_log_scalar("fun_asr_nano.adaptor.graph_cache_hit",
                                      true);
      return;
    }

    const auto build_start = Clock::now();
    const auto &encoder = assets->config.encoder;
    const auto &config = assets->config.adaptor;
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
          "failed to initialize Fun-ASR-Nano adaptor graph context");
    }

    engine::core::ModuleBuildContext context{next->ggml, "fun_asr_nano.adaptor",
                                             execution_context->backend_type()};
    const auto input_shape =
        engine::core::TensorShape::from_dims({1, frames, encoder.d_model});
    next->input =
        engine::core::make_tensor(context, GGML_TYPE_F32, input_shape);
    ggml_set_input(next->input.tensor);
    ggml_set_output(next->input.tensor);

    auto linear_1 =
        linear(context, next->input, weights->projector_1, encoder.d_model,
               assets->config.projector_hidden_size);
    auto linear_1_checkpoint = engine::core::wrap_tensor(
        ggml_dup(context.ggml, linear_1.tensor), linear_1.shape, linear_1.type);
    next->checkpoints.push_back({"linear_1", linear_1_checkpoint});
    auto hidden = engine::modules::ReluModule{}.build(context, linear_1);
    hidden = linear(context, hidden, weights->projector_2,
                    assets->config.projector_hidden_size, config.d_model);
    auto linear_2_checkpoint = engine::core::wrap_tensor(
        ggml_dup(context.ggml, hidden.tensor), hidden.shape, hidden.type);
    next->checkpoints.push_back({"linear_2", linear_2_checkpoint});
    for (size_t index = 0; index < weights->layers.size(); ++index) {
      hidden = adaptor_block(context, hidden, weights->layers[index], config);
      auto checkpoint =
          index + 1 == weights->layers.size()
              ? hidden
              : engine::core::wrap_tensor(ggml_dup(context.ggml, hidden.tensor),
                                          hidden.shape, hidden.type);
      next->checkpoints.push_back(
          {"block_" + std::to_string(index), checkpoint});
    }
    next->output = hidden;
    next->checkpoints.push_back({"packed_valid", next->output});
    for (const auto &checkpoint : next->checkpoints) {
      ggml_set_output(checkpoint.second.tensor);
    }

    next->graph = ggml_new_graph_custom(next->ggml, kAdaptorGraphNodes, false);
    for (const auto &checkpoint : next->checkpoints) {
      ggml_build_forward_expand(next->graph, checkpoint.second.tensor);
    }
    ggml_build_forward_expand(next->graph, next->output.tensor);
    engine::core::validate_backend_graph_supported(next->backend, next->graph,
                                                   "Fun-ASR-Nano adaptor");
    next->allocator =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(next->backend));
    if (next->allocator == nullptr ||
        !ggml_gallocr_reserve(next->allocator, next->graph) ||
        !ggml_gallocr_alloc_graph(next->allocator, next->graph)) {
      throw std::runtime_error(
          "failed to allocate Fun-ASR-Nano adaptor graph tensors");
    }
    engine::core::prepare_host_graph_plan(*execution_context, next->graph,
                                          next->host_plan);
    cached_graph = std::move(next);
    engine::debug::timing_log_scalar(
        "fun_asr_nano.adaptor.graph_build_ms",
        engine::debug::elapsed_ms(build_start, Clock::now()));
    engine::debug::trace_log_scalar("fun_asr_nano.adaptor.graph_cache_hit",
                                    false);
    engine::debug::trace_log_scalar("fun_asr_nano.adaptor.graph_frames",
                                    frames);
  }

  FunAsrNanoAdaptorEmbeddings
  adapt(const FunAsrNanoEncoderEmbeddings &encoder_embeddings,
        const std::vector<int32_t> &mask, bool capture_stages) {
    const auto &encoder = assets->config.encoder;
    const auto &config = assets->config.adaptor;
    if (encoder_embeddings.frames <= 0 ||
        encoder_embeddings.hidden_size != encoder.d_model) {
      throw std::runtime_error("Fun-ASR-Nano adaptor input shape is invalid");
    }
    if (encoder_embeddings.frames >
        std::numeric_limits<int64_t>::max() / encoder.d_model) {
      throw std::runtime_error("Fun-ASR-Nano adaptor input shape is invalid");
    }
    if (static_cast<int64_t>(encoder_embeddings.values.size()) !=
        encoder_embeddings.frames * encoder.d_model) {
      throw std::runtime_error(
          "Fun-ASR-Nano adaptor input value count mismatch");
    }
    if (static_cast<int64_t>(mask.size()) != encoder_embeddings.frames) {
      throw std::runtime_error("Fun-ASR-Nano adaptor mask size mismatch");
    }

    bool saw_padding = false;
    int64_t valid_frames = 0;
    for (const int32_t value : mask) {
      if (value == 0) {
        saw_padding = true;
      } else if (value == 1 && !saw_padding) {
        ++valid_frames;
      } else if (value == 1) {
        throw std::runtime_error(
            "Fun-ASR-Nano adaptor mask must be right-padded");
      } else {
        throw std::runtime_error(
            "Fun-ASR-Nano adaptor mask values must be zero or one");
      }
    }
    if (valid_frames <= 0 || valid_frames != encoder_embeddings.valid_frames) {
      throw std::runtime_error(
          "Fun-ASR-Nano adaptor mask does not match valid frames");
    }

    const auto adapt_start = Clock::now();
    ensure_graph(valid_frames);
    const size_t valid_value_count =
        static_cast<size_t>(valid_frames * encoder.d_model);
    std::vector<float> valid_values(encoder_embeddings.values.begin(),
                                    encoder_embeddings.values.begin() +
                                        valid_value_count);
    engine::core::write_tensor_f32(cached_graph->input, valid_values);
    const auto status = engine::core::compute_graph(
        *execution_context, cached_graph->graph, cached_graph->host_plan,
        "Fun-ASR-Nano adaptor");
    if (status != GGML_STATUS_SUCCESS) {
      throw std::runtime_error("Fun-ASR-Nano adaptor graph execution failed");
    }

    FunAsrNanoAdaptorEmbeddings output;
    output.values = engine::core::read_tensor_f32(cached_graph->output.tensor);
    output.tokens = valid_frames;
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
        "fun_asr_nano.adaptor_ms",
        engine::debug::elapsed_ms(adapt_start, Clock::now()));
    return output;
  }

  std::shared_ptr<const FunAsrNanoAssets> assets;
  engine::core::ExecutionContext *execution_context = nullptr;
  size_t graph_arena_bytes = 0;
  std::unique_ptr<AdaptorWeights> weights;
  std::unique_ptr<Graph> cached_graph;
};

FunAsrNanoAdaptorRuntime::FunAsrNanoAdaptorRuntime(
    std::shared_ptr<const FunAsrNanoAssets> assets,
    engine::core::ExecutionContext &execution_context, size_t graph_arena_bytes,
    engine::assets::TensorStorageType weight_storage)
    : impl_(std::make_unique<Impl>(std::move(assets), execution_context,
                                   graph_arena_bytes, weight_storage)) {}

FunAsrNanoAdaptorRuntime::~FunAsrNanoAdaptorRuntime() = default;
FunAsrNanoAdaptorRuntime::FunAsrNanoAdaptorRuntime(
    FunAsrNanoAdaptorRuntime &&) noexcept = default;
FunAsrNanoAdaptorRuntime &FunAsrNanoAdaptorRuntime::operator=(
    FunAsrNanoAdaptorRuntime &&) noexcept = default;

void FunAsrNanoAdaptorRuntime::prepare_capacity(int64_t valid_frames) {
  if (impl_ == nullptr) {
    throw std::runtime_error("Fun-ASR-Nano adaptor runtime is moved from");
  }
  impl_->ensure_graph(valid_frames);
}

FunAsrNanoAdaptorEmbeddings FunAsrNanoAdaptorRuntime::adapt(
    const FunAsrNanoEncoderEmbeddings &encoder_embeddings,
    const std::vector<int32_t> &mask, bool capture_stages) {
  if (impl_ == nullptr) {
    throw std::runtime_error("Fun-ASR-Nano adaptor runtime is moved from");
  }
  return impl_->adapt(encoder_embeddings, mask, capture_stages);
}

} // namespace engine::models::fun_asr_nano
