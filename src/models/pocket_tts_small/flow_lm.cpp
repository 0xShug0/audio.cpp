#include "engine/models/pocket_tts_small/flow_lm.h"
#include "engine/models/pocket_tts_small/assets.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/attention/grouped_query_attention.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace engine::models::pocket_tts_small {

modules::StreamingTransformerEncoderBlockOutputs build_transformer(
    const pocket_tts_small::FlowLMConfig & config,
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const modules::TransformerEncoderBlockWeights & weights,
    const std::optional<core::TensorValue> & prefix_key,
    const std::optional<core::TensorValue> & prefix_value,
    const std::optional<core::TensorValue> & attention_mask,
    const std::optional<core::TensorValue> & cache_slots = std::nullopt) {
    const int64_t head_dim = config.attention_size / config.num_heads;
    const int64_t kv_width = config.num_kv_heads * head_dim;
    const int64_t batch = input.shape.dims[0];
    const int64_t steps = input.shape.dims[1];
    auto normalized = modules::LayerNormModule({config.hidden_size, config.norm_eps, true, true})
        .build(ctx, input, weights.norm1);
    const auto & attention = weights.self_attention;
    auto q = modules::LinearModule({config.hidden_size, config.attention_size, false})
        .build(ctx, normalized, {attention.q_weight, attention.q_bias});
    auto k = modules::LinearModule({config.hidden_size, kv_width, false})
        .build(ctx, normalized, {attention.k_weight, attention.k_bias});
    auto v = modules::LinearModule({config.hidden_size, kv_width, false})
        .build(ctx, normalized, {attention.v_weight, attention.v_bias});
    q = core::reshape_tensor(ctx, q, core::TensorShape::from_dims({batch, steps, config.num_heads, head_dim}));
    k = core::reshape_tensor(ctx, k, core::TensorShape::from_dims({batch, steps, config.num_kv_heads, head_dim}));
    v = core::reshape_tensor(ctx, v, core::TensorShape::from_dims({batch, steps, config.num_kv_heads, head_dim}));
    const modules::RoPEModule rope({head_dim, GGML_ROPE_TYPE_NORMAL, 10000.0F});
    q = positions.shape.rank == 2 ? rope.build_batched(ctx, q, positions) : rope.build(ctx, q, positions);
    k = positions.shape.rank == 2 ? rope.build_batched(ctx, k, positions) : rope.build(ctx, k, positions);
    core::TensorValue all_k, all_v;
    if (cache_slots) {
        const modules::FastKVSetRowsModule append;
        all_k = append.build(ctx, *prefix_key, k, *cache_slots);
        all_v = append.build(ctx, *prefix_value, v, *cache_slots);
    } else {
        all_k = prefix_key ? modules::ConcatModule({1}).build(ctx, *prefix_key, k) : k;
        all_v = prefix_value ? modules::ConcatModule({1}).build(ctx, *prefix_value, v) : v;
    }
    auto mask = attention_mask;
    if (config.flash_attention && mask && mask->type != GGML_TYPE_F16)
        mask = core::wrap_tensor(ggml_cast(ctx.ggml, mask->tensor, GGML_TYPE_F16), mask->shape, GGML_TYPE_F16);
    const modules::TransposeModule to_heads({{0, 2, 1, 3}, 4});
    auto context = modules::GroupedQueryAttentionModule({
        head_dim, config.flash_attention ? modules::GroupedQueryAttentionLowering::FlashGroupedViewKV
                                        : modules::GroupedQueryAttentionLowering::ManualRepeat,
        GGML_PREC_F32, modules::AttentionCausality::NonCausal,
    }).build(ctx, to_heads.build(ctx, q), to_heads.build(ctx, all_k), to_heads.build(ctx, all_v), mask);
    context = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, context),
        core::TensorShape::from_dims({batch, steps, config.attention_size}));
    auto update = modules::LinearModule({config.attention_size, config.hidden_size, false})
        .build(ctx, context, {attention.out_weight, attention.out_bias});
    if (weights.layer_scale1.has_value()) {
        update = modules::LayerScaleModule().build(ctx, update, *weights.layer_scale1);
    }
    auto output = modules::ResidualAddModule().build(ctx, input, update);
    normalized = modules::LayerNormModule({config.hidden_size, config.norm_eps, true, true})
        .build(ctx, output, weights.norm2);
    update = modules::FeedForwardModule({config.hidden_size, config.intermediate_size, false,
        modules::GeluApproximation::Tanh}).build(ctx, normalized, weights.feed_forward);
    if (weights.layer_scale2.has_value()) {
        update = modules::LayerScaleModule().build(ctx, update, *weights.layer_scale2);
    }
    return {modules::ResidualAddModule().build(ctx, output, update), k, v};
}

core::TensorValue build_sample(
    const pocket_tts_small::FlowLMConfig & config, core::ModuleBuildContext & ctx,
    const core::TensorValue & condition, const core::TensorValue & noise,
    const modules::ConditionedFlowMLPWeights & weights) {
    const auto output = modules::ConditionedFlowMLPModule({config.latent_size, config.flow_hidden_size,
        config.condition_size, config.latent_size, config.flow_depth, config.flow_eps, true})
        .build(ctx, condition, noise, weights);
    return config.sampler == pocket_tts_small::FlowSamplerKind::Drifting
        ? output : modules::ResidualAddModule().build(ctx, noise, output);
}

namespace {

modules::TransformerEncoderBlockWeights make_transformer_layer_weights(
    core::ModuleBuildContext & ctx,
    const models::pocket_tts_small::BackendWeights & weights,
    const FlowLMConfig & config,
    int64_t layer) {
    return make_transformer_block_weights(
        ctx,
        weights.flow.transformer_layers.at(static_cast<size_t>(layer)),
        config.hidden_size,
        config.attention_size,
        config.num_kv_heads * (config.attention_size / config.num_heads));
}

modules::ConditionedFlowMLPWeights make_flow_net_weights(
    const models::pocket_tts_small::BackendWeights & weights,
    const FlowLMConfig & config) {
    if (static_cast<int64_t>(weights.flow.flow_net.residual_layers.size()) != config.flow_depth) {
        throw std::runtime_error("PocketTTS FlowNet residual layer count does not match config");
    }
    return weights.flow.flow_net;
}

std::vector<float> replace_bos_sentinel_if_needed(
    const std::vector<float> & bos_embedding,
    const std::vector<float> & input_latent,
    int64_t latent_size) {
    if (static_cast<int64_t>(input_latent.size()) != latent_size) {
        throw std::runtime_error("PocketTTS input latent has unexpected size");
    }
    bool has_nan = false;
    for (float value : input_latent) {
        if (std::isnan(value)) {
            has_nan = true;
            break;
        }
    }
    if (!has_nan) {
        return input_latent;
    }
    if (bos_embedding.size() != static_cast<size_t>(latent_size)) {
        throw std::runtime_error("FlowLM BOS embedding size does not match latent_dim");
    }
    return bos_embedding;
}

core::TensorValue make_cache_prefix_view(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache_tensor,
    int64_t prefix_steps,
    int64_t num_heads,
    int64_t head_dim) {
    const int64_t step_elems = num_heads * head_dim;
    auto flat = core::wrap_tensor(
        ggml_view_1d(
            ctx.ggml,
            cache_tensor.tensor,
            prefix_steps * step_elems,
            0),
        core::TensorShape::from_dims({prefix_steps * step_elems}),
        GGML_TYPE_F32);
    return core::reshape_tensor(
        ctx,
        flat,
        core::TensorShape::from_dims({1, prefix_steps, num_heads, head_dim}));
}

void write_prompt_attention_mask(
    std::vector<float> & mask,
    int64_t valid_prefix_steps,
    int64_t prefix_capacity,
    int64_t valid_prompt_steps,
    int64_t prompt_capacity) {
    const int64_t total_key_capacity = prefix_capacity + prompt_capacity;
    if (valid_prefix_steps < 0 || valid_prompt_steps < 0 || valid_prompt_steps > prompt_capacity
        || valid_prefix_steps > prefix_capacity || prefix_capacity < 0) {
        throw std::runtime_error("FlowLM prompt attention mask received invalid capacities");
    }
    mask.assign(static_cast<size_t>(prompt_capacity * total_key_capacity), -INFINITY);
    for (int64_t q = 0; q < prompt_capacity; ++q) {
        const size_t row_offset = static_cast<size_t>(q * total_key_capacity);
        const int64_t max_key = prefix_capacity + q;
        if (q >= valid_prompt_steps) {
            mask[row_offset + static_cast<size_t>(max_key)] = 0.0F;
            continue;
        }
        for (int64_t k = 0; k < valid_prefix_steps; ++k) {
            mask[row_offset + static_cast<size_t>(k)] = 0.0F;
        }
        for (int64_t k = prefix_capacity; k <= max_key; ++k) {
            mask[row_offset + static_cast<size_t>(k)] = 0.0F;
        }
    }
}

}  // namespace

class FlowLMWeightsRuntime {
public:
    const BackendWeights * const source;
    const size_t context_bytes;
    using Weights = FlowLMWeights;

    FlowLMWeightsRuntime(
        const FlowLMConfig & config,
        int threads,
        const models::pocket_tts_small::BackendWeights & weights,
        size_t view_context_bytes) : source(&weights), context_bytes(view_context_bytes) {
        (void) threads;
        weights_.input_linear = weights.flow.input_linear;
        weights_.transformer_layers.reserve(static_cast<size_t>(config.layers));
        ggml_ctx_ = ggml_init({view_context_bytes, nullptr, true});
        if (ggml_ctx_ == nullptr) {
            throw std::runtime_error("Failed to initialize FlowLM weights view context");
        }
        backend_type_ = weights.backend_type;
        core::ModuleBuildContext ctx = core::ModuleBuildContext{ggml_ctx_, "flow_lm_weights_runtime", backend_type_};
        for (int64_t layer = 0; layer < config.layers; ++layer) {
            weights_.transformer_layers.push_back(make_transformer_layer_weights(ctx, weights, config, layer));
        }
        weights_.out_norm = weights.flow.out_norm;
        weights_.out_eos = weights.flow.out_eos;
        weights_.z_projection = weights.flow.z_projection;
        weights_.flow_net = make_flow_net_weights(weights, config);
        const auto & bos = weights.host.bos_emb;
        if (static_cast<int64_t>(bos.size()) != config.latent_size) {
            throw std::runtime_error("flow_lm.bos_emb must have shape [latent_dim]");
        }
        bos_embedding_ = bos;
    }

    ~FlowLMWeightsRuntime() {
        if (ggml_ctx_ != nullptr) {
            ggml_free(ggml_ctx_);
        }
    }

    FlowLMWeightsRuntime(const FlowLMWeightsRuntime &) = delete;
    FlowLMWeightsRuntime & operator=(const FlowLMWeightsRuntime &) = delete;

    const Weights & weights() const noexcept {
        return weights_;
    }

    const std::vector<float> & bos_embedding() const noexcept {
        return bos_embedding_;
    }

    core::BackendType backend_type() const noexcept {
        return backend_type_;
    }

private:
    ggml_context * ggml_ctx_ = nullptr;
    Weights weights_;
    std::vector<float> bos_embedding_;
    core::BackendType backend_type_ = core::BackendType::Cpu;
};

class FlowLMDecodeRuntime {
public:
    const ggml_backend_t backend;
    const int threads;
    const int64_t batch_size;
    const int64_t cache_steps;
    const size_t graph_context_bytes;

    FlowLMDecodeRuntime(const FlowLMConfig & config, ggml_backend_t device, int workers,
        std::shared_ptr<const FlowLMWeightsRuntime> weights, int64_t batch, int64_t capacity, size_t graph_bytes)
        : backend(device), threads(workers), batch_size(batch), cache_steps(capacity),
          graph_context_bytes(graph_bytes), config_(config), weights_(std::move(weights)),
          context_(ggml_init({graph_bytes, nullptr, true}), ggml_free),
          allocator_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(device)), ggml_gallocr_free) {
        if (!context_ || !allocator_) throw std::runtime_error("Failed to create batched FlowLM graph");
        core::ModuleBuildContext ctx{context_.get(), "pocket_tts_small.batch", weights_->backend_type()};
        const auto & w = weights_->weights();
        latent_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, 1, config_.latent_size}));
        noise_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, config_.latent_size}));
        positions_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({batch, 1}));
        slots_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({batch}));
        mask_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, 1, 1, capacity}));
        std::vector<core::TensorValue> keys, values;
        const auto head_dim = config_.attention_size / config_.num_heads;
        auto x = modules::LinearModule({config_.latent_size, config_.hidden_size, w.input_linear.bias.has_value()})
            .build(ctx, latent_, w.input_linear);
        for (int64_t layer = 0; layer < config_.layers; ++layer) {
            const auto shape = core::TensorShape::from_dims({batch, capacity, config_.num_kv_heads, head_dim});
            keys.push_back(core::make_tensor(ctx, config_.kv_cache_type, shape));
            values.push_back(core::make_tensor(ctx, config_.kv_cache_type, shape));
            auto out = build_transformer(config_, ctx, x, positions_, w.transformer_layers[layer],
                keys.back(), values.back(), mask_, slots_);
            x = out.output;
        }
        runtime::TransformerKVCacheOptions cache_options;
        cache_options.lazy_import_scratch = true;
        cache_options.allow_f16_storage = config_.kv_cache_type == GGML_TYPE_F16;
        cache_ = runtime::TransformerBatchedKVCache(capacity, batch, config_.num_kv_heads * head_dim,
            std::move(keys), std::move(values), cache_options);
        auto condition = modules::LayerNormModule({config_.hidden_size, config_.norm_eps, true, true})
            .build(ctx, x, w.out_norm);
        condition = core::reshape_tensor(ctx, condition, core::TensorShape::from_dims({batch, config_.hidden_size}));
        if (w.z_projection) condition = modules::LinearModule({config_.hidden_size, config_.condition_size,
            w.z_projection->bias.has_value()}).build(ctx, condition, *w.z_projection);
        const auto eos = modules::LinearModule({config_.condition_size, 1, true}).build(ctx, condition, w.out_eos);
        output_ = modules::ConcatModule({1}).build(ctx, build_sample(config_, ctx, condition, noise_, w.flow_net), eos);
        graph_ = ggml_new_graph_custom(context_.get(), 32768, false);
        ggml_set_output(output_.tensor);
        ggml_build_forward_expand(graph_, output_.tensor);
        for (auto input : {latent_, noise_, positions_, slots_, mask_}) {
            ggml_set_input(input.tensor);
            ggml_set_output(input.tensor);
        }
        for (int64_t layer = 0; layer < config_.layers; ++layer) {
            ggml_set_output(cache_.key_tensor(layer).tensor);
            ggml_set_output(cache_.value_tensor(layer).tensor);
        }
        if (!ggml_gallocr_reserve(allocator_.get(), graph_) || !ggml_gallocr_alloc_graph(allocator_.get(), graph_))
            throw std::runtime_error("Failed to allocate batched FlowLM graph");
        if (core::uses_host_graph_plan(backend)) {
            core::set_backend_threads(backend, threads);
            host_plan_.backend = backend;
            host_plan_.plan = core::create_backend_graph_plan_if_host(backend, graph_);
            if (!host_plan_.active()) throw std::runtime_error("Failed to prepare batched FlowLM host graph plan");
        }
        debug::trace_log_scalar("pocket_tts_small.flow_lm.decode.graph_buffer_mb",
            double(ggml_gallocr_get_buffer_size(allocator_.get(), 0)) / (1024 * 1024));
        positions_buffer_.resize(batch);
        slots_buffer_.resize(batch);
        mask_buffer_.resize(batch * capacity);
        advances_.resize(batch);
    }

    ~FlowLMDecodeRuntime() {
        core::release_backend_graph_resources(backend, graph_, true);
    }

    void import_states(const std::vector<FlowLMState> & states) {
        runtime::TransformerBatchedKVState packed;
        packed.batch_size = batch_size;
        packed.current_end_by_batch.resize(batch_size, 0);
        packed.valid_steps_by_batch.resize(batch_size, 0);
        packed.layers.resize(config_.layers);
        for (size_t row = 0; row < states.size(); ++row) {
            packed.current_end_by_batch[row] = states[row].current_end;
            packed.valid_steps_by_batch[row] = states[row].current_end;
            for (int64_t layer = 0; layer < config_.layers; ++layer) {
                const auto & from = states[row].layers.at(layer);
                auto & to = packed.layers[layer];
                to.key.insert(to.key.end(), from.key.begin(), from.key.end());
                to.value.insert(to.value.end(), from.value.begin(), from.value.end());
            }
        }
        cache_.import_state(packed);
        input_write_ms = graph_ms = output_read_ms = 0.0;
    }

    const std::vector<float> & step(const std::vector<float> & inputs,
        const std::vector<float> & noises, const std::vector<int64_t> & active) {
        std::fill(mask_buffer_.begin(), mask_buffer_.end(), -INFINITY);
        std::fill(advances_.begin(), advances_.end(), 0);
        for (int64_t row = 0; row < batch_size; ++row) {
            const auto position = cache_.current_end_by_batch()[row];
            positions_buffer_[row] = static_cast<int32_t>(position);
            slots_buffer_[row] = static_cast<int32_t>(row * cache_steps + std::min(position, cache_steps - 1));
            const size_t start = row * cache_steps;
            std::fill_n(mask_buffer_.begin() + start, std::min(cache_.valid_steps_by_batch()[row] + 1, cache_steps), 0.0F);
            if (row < static_cast<int64_t>(active.size()) && active[row]) {
                if (position >= cache_steps) throw std::runtime_error("FlowLM decode cache capacity exceeded");
                advances_[row] = 1;
            }
        }
        input_write_ms += debug::measure_ms([&]() {
            core::write_tensor_f32(latent_, inputs);
            core::write_tensor_f32(noise_, noises);
            core::write_tensor_i32(positions_, positions_buffer_);
            core::write_tensor_i32(slots_, slots_buffer_);
            core::write_tensor_f32(mask_, mask_buffer_);
        });
        core::set_backend_threads(backend, threads);
        graph_ms += debug::measure_ms([&]() {
            if (core::compute_backend_graph(backend, graph_, host_plan_.plan) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("FlowLM decode computation failed");
        });
        cache_.advance_after_direct_append(advances_);
        output_read_ms += debug::measure_ms([&]() { core::read_tensor_f32_into(output_.tensor, results_); });
        return results_;
    }

    const FlowLMWeightsRuntime & weights() const { return *weights_; }
    double input_write_ms = 0.0;
    double graph_ms = 0.0;
    double output_read_ms = 0.0;

private:
    FlowLMConfig config_;
    std::shared_ptr<const FlowLMWeightsRuntime> weights_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, decltype(&ggml_gallocr_free)> allocator_;
    runtime::TransformerBatchedKVCache cache_;
    core::HostGraphPlan host_plan_;
    ggml_cgraph * graph_ = nullptr;
    core::TensorValue latent_, noise_, positions_, slots_, mask_, output_;
    std::vector<float> mask_buffer_, results_;
    std::vector<int32_t> positions_buffer_, slots_buffer_;
    std::vector<int64_t> advances_;
};

class FlowLMStepRuntime {
public:
    FlowLMStepRuntime(
        const FlowLMConfig & config,
        ggml_backend_t backend,
        int threads,
        std::shared_ptr<const FlowLMWeightsRuntime> weights_runtime,
        int64_t cache_steps,
        int64_t prompt_steps,
        int64_t prompt_prefix_steps,
        size_t graph_context_bytes)
        : config_(config),
          backend_(backend),
          threads_(threads),
          weights_runtime_(std::move(weights_runtime)),
          cache_steps_(std::max<int64_t>(1, cache_steps)),
          prompt_steps_(std::max<int64_t>(0, prompt_steps)),
          prompt_prefix_steps_(std::max<int64_t>(0, prompt_prefix_steps)),
          head_dim_(config.attention_size / config.num_heads) {
        if (weights_runtime_ == nullptr) {
            throw std::runtime_error("FlowLM step runtime requires weights runtime");
        }
        const auto & weights = weights_runtime_->weights();
        if (config_.native_cpu) {
#ifdef ENGINE_HAVE_NEON
            if (!core::is_host_backend(backend)) throw std::runtime_error("Native FlowLM requires the CPU backend");
            native_ = std::make_unique<NativeFlowLM>(config_, weights, cache_steps_,
                config_.native_threads > 0 ? config_.native_threads : threads);
            return;
#else
            throw std::runtime_error("Native FlowLM requires a build with ENGINE_ENABLE_NEON=ON");
#endif
        }

        ggml_ctx_ = ggml_init({graph_context_bytes, nullptr, true});
        if (ggml_ctx_ == nullptr) {
            throw std::runtime_error("Failed to initialize FlowLM step ggml context");
        }

        core::ModuleBuildContext ctx{ggml_ctx_, "flow_lm_step_runtime", weights_runtime_->backend_type()};

        if (prompt_steps_ > 0) {
            prompt_positions_buffer_.resize(static_cast<size_t>(prompt_steps_));
            prompt_embeddings_buffer_.assign(
                static_cast<size_t>(prompt_steps_ * config_.hidden_size),
                0.0F);
            prompt_embeddings_ = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, prompt_steps_, config_.hidden_size}));
            prompt_positions_ = core::make_tensor(
                ctx,
                GGML_TYPE_I32,
                core::TensorShape::from_dims({prompt_steps_}));
            prompt_slots_ = core::make_tensor(ctx, GGML_TYPE_I32,
                core::TensorShape::from_dims({prompt_steps_}));
            prompt_slots_buffer_.resize(prompt_steps_);
            write_prompt_attention_mask(
                prompt_attention_mask_buffer_,
                0,
                prompt_prefix_steps_,
                0,
                prompt_steps_);
            prompt_attention_mask_ = core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, 1, prompt_steps_, prompt_prefix_steps_ + prompt_steps_}));
        }

        const int64_t prefill_steps = std::max<int64_t>(1, prompt_prefix_steps_ + prompt_steps_);
        std::vector<core::TensorValue> step_cache_keys;
        std::vector<core::TensorValue> step_cache_values;
        step_cache_keys.reserve(static_cast<size_t>(config_.layers));
        step_cache_values.reserve(static_cast<size_t>(config_.layers));

        for (int64_t layer = 0; layer < config_.layers; ++layer) {
            step_cache_keys.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, prefill_steps, config_.num_kv_heads, head_dim_})));
            step_cache_values.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, prefill_steps, config_.num_kv_heads, head_dim_})));
        }
        step_cache_ = runtime::TransformerKVCache(
            prefill_steps,
            config_.num_kv_heads * head_dim_,
            std::move(step_cache_keys),
            std::move(step_cache_values));
        if (prompt_steps_ > 0) {
            prompt_keys_.reserve(static_cast<size_t>(config_.layers));
            prompt_values_.reserve(static_cast<size_t>(config_.layers));
            auto x_prompt = prompt_embeddings_;
            for (int64_t layer = 0; layer < config_.layers; ++layer) {
                std::optional<core::TensorValue> prefix_k = std::nullopt;
                std::optional<core::TensorValue> prefix_v = std::nullopt;
                if (prompt_prefix_steps_ > 0) {
                    prefix_k = make_cache_prefix_view(
                        ctx,
                        step_cache_.key_tensor(static_cast<size_t>(layer)),
                        prompt_prefix_steps_,
                        config_.num_kv_heads,
                        head_dim_);
                    prefix_v = make_cache_prefix_view(
                        ctx,
                        step_cache_.value_tensor(static_cast<size_t>(layer)),
                        prompt_prefix_steps_,
                        config_.num_kv_heads,
                        head_dim_);
                }

                auto outputs = build_transformer(
                        config_, ctx, x_prompt, prompt_positions_, weights.transformer_layers[static_cast<size_t>(layer)],
                        prefix_k, prefix_v, prompt_attention_mask_);
                x_prompt = outputs.output;
                prompt_keys_.push_back(outputs.key);
                prompt_values_.push_back(outputs.value);
            }
            prompt_output_ = x_prompt;
        }

        params_buffer_ = ggml_backend_alloc_ctx_tensors(ggml_ctx_, backend_);
        if (params_buffer_ == nullptr) {
            // Under VRAM pressure the CUDA buffer alloc fails and returns null;
            // the write_tensor calls below would then trip GGML_ASSERT and
            // SIGABRT the whole server. Fail the request instead (see #55).
            release_runtime();
            throw std::runtime_error("FlowLM step parameter buffer allocation failed (out of memory)");
        }
        if (prompt_steps_ > 0) {
            core::write_tensor_f32(prompt_embeddings_, prompt_embeddings_buffer_);
            core::write_tensor_i32(prompt_positions_, std::vector<int32_t>(static_cast<size_t>(prompt_steps_), 0));
            core::write_tensor_f32(prompt_attention_mask_, prompt_attention_mask_buffer_);
        }
        core::set_backend_threads(backend_, threads_);
        if (prompt_steps_ > 0) {
            prompt_graph_ = ggml_new_graph_custom(ggml_ctx_, 32768, false);
            ggml_build_forward_expand(prompt_graph_, prompt_output_.tensor);
            const modules::FastKVSetRowsModule append;
            for (size_t layer = 0; layer < prompt_keys_.size(); ++layer) {
                const auto key = append.build_block(ctx, step_cache_.key_tensor(layer), prompt_keys_[layer], prompt_slots_);
                const auto value = append.build_block(ctx, step_cache_.value_tensor(layer), prompt_values_[layer], prompt_slots_);
                ggml_build_forward_expand(prompt_graph_, key.tensor);
                ggml_build_forward_expand(prompt_graph_, value.tensor);
            }
            prompt_galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
            if (prompt_galloc_ == nullptr ||
                !ggml_gallocr_reserve(prompt_galloc_, prompt_graph_) ||
                !ggml_gallocr_alloc_graph(prompt_galloc_, prompt_graph_)) {
                release_runtime();
                throw std::runtime_error("FlowLM prompt graph allocation failed (out of memory)");
            }
            if (engine::core::uses_host_graph_plan(backend_)) {
                prompt_plan_ = engine::core::create_backend_graph_plan_if_host(backend_, prompt_graph_);
                if (prompt_plan_ == nullptr) {
                    release_runtime();
                    throw std::runtime_error("Failed to create FlowLM prompt graph plan");
                }
            }
        }
    }

    ~FlowLMStepRuntime() {
        release_runtime();
    }

    // Free every owned backend resource and null it, so the constructor can
    // call this on an allocation-failure throw path (a throw from the
    // constructor skips the destructor, which would otherwise leak the ggml
    // context and any buffers allocated before the failure). Idempotent.
    void release_runtime() {
        core::release_backend_graph_resources(backend_, prompt_graph_, true);
        if (prompt_plan_ != nullptr) {
            engine::core::free_backend_graph_plan(backend_, prompt_plan_);
            prompt_plan_ = nullptr;
        }
        if (prompt_galloc_ != nullptr) {
            ggml_gallocr_free(prompt_galloc_);
            prompt_galloc_ = nullptr;
        }
        if (params_buffer_ != nullptr) {
            ggml_backend_buffer_free(params_buffer_);
            params_buffer_ = nullptr;
        }
        if (ggml_ctx_ != nullptr) {
            ggml_free(ggml_ctx_);
            ggml_ctx_ = nullptr;
        }
    }

    void apply_prompt_from_state(
        const std::vector<float> & input_embeddings,
        int64_t steps,
        const FlowLMState & state) {
#ifdef ENGINE_HAVE_NEON
        if (native_) {
            native_->apply_prompt(input_embeddings, steps, state);
            return;
        }
#endif
        if (prompt_steps_ <= 0) {
            throw std::runtime_error("FlowLM prompt runtime was not initialized with prompt support");
        }
        if (steps <= 0 || steps > prompt_steps_) {
            throw std::runtime_error("FlowLM prompt runtime step count exceeds prepared capacity");
        }
        if (state.current_end < 0 || state.current_end > prompt_prefix_steps_) {
            throw std::runtime_error("FlowLM prompt runtime prefix state exceeds prepared capacity");
        }
        if (state.current_end + steps > cache_steps_) {
            throw std::runtime_error("FlowLM prompt runtime total cache steps exceed prepared capacity");
        }

        if (input_embeddings.size() != static_cast<size_t>(steps * config_.hidden_size)) {
            throw std::runtime_error("FlowLM prompt runtime input embedding count mismatch");
        }
        std::fill(prompt_embeddings_buffer_.begin(), prompt_embeddings_buffer_.end(), 0.0F);
        std::copy(input_embeddings.begin(), input_embeddings.end(), prompt_embeddings_buffer_.begin());
        core::write_tensor_f32(prompt_embeddings_, prompt_embeddings_buffer_);
        for (int64_t i = 0; i < steps; ++i) {
            prompt_positions_buffer_[static_cast<size_t>(i)] = static_cast<int32_t>(state.current_end + i);
        }
        for (int64_t i = steps; i < prompt_steps_; ++i) {
            prompt_positions_buffer_[static_cast<size_t>(i)] = 0;
        }
        core::write_tensor_i32(prompt_positions_, prompt_positions_buffer_);
        for (int64_t i = 0; i < prompt_steps_; ++i)
            prompt_slots_buffer_[i] = static_cast<int32_t>(state.current_end + i);
        core::write_tensor_i32(prompt_slots_, prompt_slots_buffer_);
        write_prompt_attention_mask(
            prompt_attention_mask_buffer_,
            state.current_end,
            prompt_prefix_steps_,
            steps,
            prompt_steps_);
        core::write_tensor_f32(prompt_attention_mask_, prompt_attention_mask_buffer_);

        step_cache_.import_state(state);
        compute_graph(prompt_graph_, prompt_plan_);
        step_cache_.advance_after_direct_append(steps);
    }

    FlowLMStepResult run_native_step(
        const std::vector<float> & input_latent,
        const std::vector<float> & noise) {
#ifdef ENGINE_HAVE_NEON
        return native_->step(replace_bos_sentinel_if_needed(weights_runtime_->bos_embedding(), input_latent,
            config_.latent_size), noise);
#else
        (void) input_latent;
        (void) noise;
        throw std::runtime_error("Native FlowLM requires a build with ENGINE_ENABLE_NEON=ON");
#endif
    }

    FlowLMState export_state() const {
#ifdef ENGINE_HAVE_NEON
        if (native_) return native_->export_state();
#endif
        return step_cache_.export_state();
    }

private:
    void compute_graph(ggml_cgraph * graph, ggml_backend_graph_plan_t plan) {
        engine::core::compute_backend_graph(backend_, graph, plan);
    }

    FlowLMConfig config_;
    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    std::shared_ptr<const FlowLMWeightsRuntime> weights_runtime_;
#ifdef ENGINE_HAVE_NEON
    std::unique_ptr<NativeFlowLM> native_;
#endif
    int64_t cache_steps_ = 1;
    int64_t prompt_steps_ = 0;
    int64_t prompt_prefix_steps_ = 0;
    int64_t head_dim_ = 0;
    ggml_context * ggml_ctx_ = nullptr;
    ggml_backend_buffer_t params_buffer_ = nullptr;
    ggml_cgraph * prompt_graph_ = nullptr;
    ggml_gallocr_t prompt_galloc_ = nullptr;
    ggml_backend_graph_plan_t prompt_plan_ = nullptr;
    core::TensorValue prompt_embeddings_;
    std::vector<float> prompt_embeddings_buffer_;
    core::TensorValue prompt_positions_;
    core::TensorValue prompt_slots_;
    std::vector<int32_t> prompt_slots_buffer_;
    core::TensorValue prompt_attention_mask_;
    std::vector<int32_t> prompt_positions_buffer_;
    std::vector<float> prompt_attention_mask_buffer_;
    std::vector<core::TensorValue> prompt_keys_;
    std::vector<core::TensorValue> prompt_values_;
    core::TensorValue prompt_output_;
    runtime::TransformerKVCache step_cache_;
};

FlowLM::FlowLM(FlowLMConfig config) : config_(std::move(config)) {
    if (config_.attention_size == 0) config_.attention_size = config_.hidden_size;
    if (config_.condition_size == 0) config_.condition_size = config_.hidden_size;
    if (config_.num_kv_heads == 0) config_.num_kv_heads = config_.num_heads;
    if (config_.num_heads <= 0 || config_.num_kv_heads <= 0 || config_.attention_size <= 0
        || config_.attention_size % config_.num_heads != 0 || config_.num_heads % config_.num_kv_heads != 0
        || (config_.attention_size / config_.num_heads) % 2 != 0) {
        throw std::runtime_error("PocketTTS FlowLM requires divisible query/KV heads and an even RoPE head dimension");
    }
}

const FlowLMConfig & FlowLM::config() const noexcept {
    return config_;
}

std::shared_ptr<const FlowLMWeightsRuntime> FlowLM::create_weights_runtime(
    ggml_backend_t backend,
    int threads,
    const models::pocket_tts_small::BackendWeights & weights,
    size_t weights_view_context_bytes) const {
    (void) backend;
    return std::make_shared<FlowLMWeightsRuntime>(config_, threads, weights, weights_view_context_bytes);
}

FlowLMState FlowLM::make_empty_state() const {
    FlowLMState state;
    state.current_end = 0;
    state.layers.resize(static_cast<size_t>(config_.layers));
    return state;
}

FlowLMState FlowLM::make_state(const models::pocket_tts_small::VoiceStateAssets & voice_assets) const {
    if (voice_assets.transformer_layers.empty()) {
        throw std::runtime_error("PocketTTS voice assets do not contain FlowLM cache layers");
    }
    if (static_cast<int64_t>(voice_assets.transformer_layers.size()) != config_.layers) {
        throw std::runtime_error("PocketTTS voice assets layer count does not match FlowLM configuration");
    }

    FlowLMState state;
    state.layers.reserve(voice_assets.transformer_layers.size());
    state.current_end = voice_assets.transformer_layers.front().offset;
    const int64_t head_elems = config_.num_kv_heads * (config_.attention_size / config_.num_heads);

    for (const auto & layer : voice_assets.transformer_layers) {
        if (layer.offset != state.current_end) {
            throw std::runtime_error("PocketTTS voice cache offsets are inconsistent across layers");
        }
        if (layer.offset < 0 || layer.offset > layer.cached_steps) {
            throw std::runtime_error("PocketTTS voice cache offset must be within cached_steps");
        }
        const size_t expected_elems = static_cast<size_t>(layer.cached_steps * head_elems);
        if (layer.key.size() != expected_elems || layer.value.size() != expected_elems) {
            throw std::runtime_error("PocketTTS voice cache tensor sizes do not match cached_steps * num_kv_heads * head_dim");
        }
        const size_t valid_elems = static_cast<size_t>(layer.offset * head_elems);
        FlowLMCacheState cache;
        cache.valid_steps = layer.offset;
        cache.key.assign(layer.key.begin(), layer.key.begin() + static_cast<std::ptrdiff_t>(valid_elems));
        cache.value.assign(layer.value.begin(), layer.value.begin() + static_cast<std::ptrdiff_t>(valid_elems));
        state.layers.push_back(std::move(cache));
    }
    return state;
}

std::shared_ptr<FlowLMStepRuntime> FlowLM::create_step_runtime(
    ggml_backend_t backend,
    int threads,
    std::shared_ptr<const FlowLMWeightsRuntime> weights,
    int64_t cache_steps,
    int64_t prompt_steps,
    int64_t prompt_prefix_steps,
    size_t step_graph_context_bytes) const {
    return std::make_shared<FlowLMStepRuntime>(
        config_,
        backend,
        threads,
        std::move(weights),
        cache_steps,
        prompt_steps,
        prompt_prefix_steps,
        step_graph_context_bytes);
}

void FlowLM::apply_prompt(
    FlowLMStepRuntime & runtime,
    const std::vector<float> & input_embeddings,
    int64_t steps,
    const FlowLMState & state) const {
    runtime.apply_prompt_from_state(input_embeddings, steps, state);
}

FlowLMState FlowLM::prompt_state(
    ggml_backend_t backend,
    int threads,
    const models::pocket_tts_small::Assets & manifest,
    const models::pocket_tts_small::BackendWeights & weights,
    const std::vector<float> & input_embeddings,
    int64_t steps,
    const FlowLMState & state,
    size_t weights_view_context_bytes,
    size_t step_graph_context_bytes) const {
    FlowLMState prompted_state;
    const double prompt_ms = engine::debug::measure_ms([&]() {
        (void) manifest;
        auto & cache = prompt_cache_;
        const bool weights_hit = cache.weights_runtime && cache.backend == backend && cache.threads == threads
            && cache.weights == &weights && cache.weights_context_bytes == weights_view_context_bytes;
        if (!weights_hit) {
            cache = {};
            cache.weights_runtime = create_weights_runtime(backend, threads, weights, weights_view_context_bytes);
            cache.backend = backend;
            cache.threads = threads;
            cache.weights = &weights;
            cache.weights_context_bytes = weights_view_context_bytes;
        }
        const bool cache_hit = cache.runtime && cache.steps >= steps && cache.prefix_steps >= state.current_end
            && cache.graph_context_bytes == step_graph_context_bytes;
        if (!cache_hit) {
            const auto graph_steps = std::max(cache.steps, steps);
            const auto graph_prefix = std::max(cache.prefix_steps, state.current_end);
            cache.runtime.reset();
            cache.runtime = create_step_runtime(backend, threads, cache.weights_runtime,
                graph_prefix + graph_steps, graph_steps, graph_prefix, step_graph_context_bytes);
            cache.steps = graph_steps;
            cache.prefix_steps = graph_prefix;
            cache.graph_context_bytes = step_graph_context_bytes;
        }
        debug::trace_log_scalar("pocket_tts_small.flow_lm.voice_prompt_cache_hit", cache_hit ? 1 : 0);
        cache.runtime->apply_prompt_from_state(input_embeddings, steps, state);
        prompted_state = cache.runtime->export_state();
    });
    engine::debug::timing_log_scalar("pocket_tts_small.flow_lm.prompt_ms", prompt_ms);
    return prompted_state;
}

FlowLMStepResult FlowLM::run_native_step(
    FlowLMStepRuntime & runtime,
    const std::vector<float> & input_latent,
    const std::vector<float> & noise) const {
    return runtime.run_native_step(input_latent, noise);
}

namespace {

std::vector<float> sample_trunc_normal(std::mt19937 & rng, int64_t count, float stddev, float clamp) {
    std::normal_distribution<float> dist(0.0f, stddev);
    std::vector<float> values(static_cast<size_t>(count));
    for (float & value : values) {
        do {
            value = dist(rng);
        } while (value < -clamp || value > clamp);
    }
    return values;
}

std::vector<float> sample_normal(std::mt19937 & rng, int64_t count, float stddev) {
    std::normal_distribution<float> dist(0.0f, stddev);
    std::vector<float> values(static_cast<size_t>(count));
    for (float & value : values) {
        value = dist(rng);
    }
    return values;
}

void validate_generation_inputs(
    const FlowLMConfig & flow_config,
    const std::vector<float> & text_embeddings,
    const AcousticGenerationConfig & config) {
    if (config.max_steps <= 0) {
        throw std::runtime_error("PocketTTS acoustic max_steps must be positive");
    }
    if (config.temperature <= 0.0F) {
        throw std::runtime_error("PocketTTS acoustic temperature must be positive");
    }
    if (!config.noise_schedule.empty() && config.noise_schedule.size() % static_cast<size_t>(flow_config.latent_size) != 0) {
        throw std::runtime_error("PocketTTS acoustic noise_schedule must be a multiple of latent_size");
    }
    if (!config.noise_schedule.empty()) {
        const size_t scheduled_steps =
            config.noise_schedule.size() / static_cast<size_t>(flow_config.latent_size);
        if (scheduled_steps < static_cast<size_t>(config.max_steps)) {
            throw std::runtime_error("PocketTTS acoustic noise_schedule must provide at least max_steps latent noise vectors");
        }
    }
    if (text_embeddings.size() % static_cast<size_t>(flow_config.hidden_size) != 0) {
        throw std::runtime_error("PocketTTS acoustic text embeddings must be a multiple of hidden_size");
    }
}

std::vector<float> sample_noise_for_step(const FlowLMConfig & flow_config, AcousticRowState & state) {
    if (!state.config.noise_schedule.empty()) {
        const size_t start = static_cast<size_t>(state.step) * static_cast<size_t>(flow_config.latent_size);
        return std::vector<float>(
            state.config.noise_schedule.begin() + static_cast<ptrdiff_t>(start),
            state.config.noise_schedule.begin() + static_cast<ptrdiff_t>(start + static_cast<size_t>(flow_config.latent_size)));
    }
    if (state.config.noise_clamp > 0.0F) {
        return sample_trunc_normal(
            state.rng,
            flow_config.latent_size,
            std::sqrt(state.config.temperature),
            state.config.noise_clamp);
    }
    return sample_normal(
        state.rng,
        flow_config.latent_size,
        std::sqrt(state.config.temperature));
}

}  // namespace

AcousticModel::AcousticModel(FlowLMConfig config) : flow_lm_(std::move(config)) {}

const FlowLMConfig & AcousticModel::config() const noexcept {
    return flow_lm_.config();
}

std::shared_ptr<FlowLMDecodeRuntime> AcousticModel::prepare_decode_runtime(
    ggml_backend_t backend, int threads, const BackendWeights & weights,
    int64_t batch, int64_t capacity, size_t weights_context_bytes, size_t graph_context_bytes) const {
    const bool same_context = decode_runtime_ && decode_runtime_->backend == backend && decode_runtime_->threads == threads
        && decode_runtime_->weights().source == &weights && decode_runtime_->weights().context_bytes == weights_context_bytes
        && decode_runtime_->graph_context_bytes == graph_context_bytes;
    const bool graph_hit = same_context && decode_runtime_->batch_size >= batch && decode_runtime_->cache_steps >= capacity;
    debug::trace_log_scalar("pocket_tts_small.flow_lm.decode.graph_cache_hit", graph_hit ? 1 : 0);
    if (!graph_hit) {
        const auto graph_batch = same_context ? std::max(batch, decode_runtime_->batch_size) : batch;
        const auto graph_steps = same_context ? std::max(capacity, decode_runtime_->cache_steps) : capacity;
        if (graph_batch * graph_steps > INT32_MAX) throw std::runtime_error("Batched FlowLM cache exceeds I32 row indexing");
        decode_runtime_.reset();
        decode_runtime_ = std::make_shared<FlowLMDecodeRuntime>(flow_lm_.config(), backend, threads,
            flow_lm_.create_weights_runtime(backend, threads, weights, weights_context_bytes), graph_batch,
            graph_steps, graph_context_bytes);
    }
    return decode_runtime_;
}

AcousticPreparedRuntime AcousticModel::prepare_runtime(
    ggml_backend_t backend,
    int threads,
    const models::pocket_tts_small::Assets & manifest,
    const models::pocket_tts_small::BackendWeights & weights,
    const std::vector<float> & text_embeddings,
    const FlowLMState & initial_state,
    const AcousticGenerationConfig & config,
    int64_t prompt_capacity,
    int64_t initial_cache_capacity,
    int max_steps_capacity,
    size_t flow_weights_view_context_bytes,
    size_t flow_step_graph_context_bytes) const {
    if (config.max_steps <= 0) {
        throw std::runtime_error("PocketTTS acoustic max_steps must be positive");
    }
    if (text_embeddings.size() % static_cast<size_t>(flow_lm_.config().hidden_size) != 0) {
        throw std::runtime_error("PocketTTS acoustic text embeddings must be a multiple of hidden_size");
    }
    const int64_t prompt_steps = static_cast<int64_t>(text_embeddings.size() / static_cast<size_t>(flow_lm_.config().hidden_size));
    AcousticPreparedRuntime runtime;
    const int64_t initial_cache_steps = initial_state.current_end;
    if (prompt_capacity < prompt_steps) {
        throw std::runtime_error("PocketTTS acoustic prompt_capacity must cover prompt_steps");
    }
    if (initial_cache_capacity < initial_cache_steps) {
        throw std::runtime_error("PocketTTS acoustic initial_cache_capacity must cover initial_cache_steps");
    }
    if (max_steps_capacity < config.max_steps) {
        throw std::runtime_error("PocketTTS acoustic max_steps_capacity must cover max_steps");
    }
    const int64_t total_cache_steps =
        initial_cache_capacity + prompt_capacity + static_cast<int64_t>(max_steps_capacity);
    if (!flow_lm_.config().native_cpu) {
        runtime.backend = backend;
        runtime.threads = threads;
        runtime.weights_context_bytes = flow_weights_view_context_bytes;
        runtime.graph_context_bytes = flow_step_graph_context_bytes;
        runtime.decode_runtime = prepare_decode_runtime(backend, threads, weights, 1, total_cache_steps,
            flow_weights_view_context_bytes, flow_step_graph_context_bytes);
        runtime_cache_.prompt_capacity = std::max(runtime_cache_.prompt_capacity, prompt_capacity);
        runtime_cache_.max_steps_capacity = std::max(runtime_cache_.max_steps_capacity, max_steps_capacity);
        return runtime;
    }
    const bool cache_hit =
        runtime_cache_.runtime != nullptr
        && runtime_cache_.backend == backend
        && runtime_cache_.threads == threads
        && runtime_cache_.manifest == &manifest
        && runtime_cache_.prompt_capacity >= prompt_capacity
        // The step runtime bakes the voice-prefix slot layout at creation: prompt KV
        // destinations start at initial_cache_capacity and the prefix attention view
        // spans exactly that many slots. Reusing a runtime built for a different prefix
        // length misaligns the KV slots against the cache validity accounting, so the
        // prefix capacity must match exactly (not merely fit).
        && runtime_cache_.initial_cache_capacity == initial_cache_capacity
        && runtime_cache_.max_steps_capacity >= max_steps_capacity
        && runtime_cache_.flow_weights_view_context_bytes == flow_weights_view_context_bytes
        && runtime_cache_.flow_step_graph_context_bytes == flow_step_graph_context_bytes;
    if (!cache_hit) {
        const bool weights_hit =
            runtime_cache_.weights_runtime != nullptr
            && runtime_cache_.backend == backend
            && runtime_cache_.threads == threads
            && runtime_cache_.manifest == &manifest
            && runtime_cache_.flow_weights_view_context_bytes == flow_weights_view_context_bytes;
        runtime_cache_.backend = backend;
        runtime_cache_.threads = threads;
        runtime_cache_.manifest = &manifest;
        runtime_cache_.prompt_capacity = prompt_capacity;
        runtime_cache_.initial_cache_capacity = initial_cache_capacity;
        runtime_cache_.max_steps_capacity = max_steps_capacity;
        runtime_cache_.flow_weights_view_context_bytes = flow_weights_view_context_bytes;
        runtime_cache_.flow_step_graph_context_bytes = flow_step_graph_context_bytes;
        if (!weights_hit) {
            runtime_cache_.weights_runtime =
                flow_lm_.create_weights_runtime(backend, threads, weights, flow_weights_view_context_bytes);
        }
        runtime_cache_.runtime = flow_lm_.create_step_runtime(
            backend,
            threads,
            runtime_cache_.weights_runtime,
            total_cache_steps,
            prompt_capacity,
            initial_cache_capacity,
            flow_step_graph_context_bytes);
    }
    runtime.step_runtime = runtime_cache_.runtime;
    return runtime;
}

AcousticStreamState AcousticModel::start_rows(
    const AcousticPreparedRuntime & runtime, const Assets & manifest, const BackendWeights & weights,
    const std::vector<std::vector<float>> & text_embeddings, const std::vector<FlowLMState> & initial_states,
    const std::vector<AcousticGenerationConfig> & configs) const {
    if (configs.empty() || configs.size() != text_embeddings.size() || configs.size() != initial_states.size())
        throw std::runtime_error("Acoustic inputs must have the same nonzero row count");
    if (runtime.step_runtime && configs.size() != 1)
        throw std::runtime_error("Native acoustic generation supports one request at a time");
    if (!runtime.decode_runtime && !runtime.step_runtime)
        throw std::runtime_error("PocketTTS acoustic runtime is not initialized");
    AcousticStreamState state;
    state.runtime = runtime;
    state.rows.resize(configs.size());
    std::vector<FlowLMState> prompted;
    for (size_t row = 0; row < configs.size(); ++row) {
        validate_generation_inputs(flow_lm_.config(), text_embeddings[row], configs[row]);
        const auto steps = text_embeddings[row].size() / flow_lm_.config().hidden_size;
        if (runtime.decode_runtime) {
            prompted.push_back(flow_lm_.prompt_state(runtime.backend, runtime.threads, manifest, weights,
                text_embeddings[row], steps, initial_states[row], runtime.weights_context_bytes, runtime.graph_context_bytes));
        } else {
            flow_lm_.apply_prompt(*runtime.step_runtime, text_embeddings[row], steps, initial_states[row]);
        }
        state.rows[row].config = configs[row];
        state.rows[row].rng.seed(configs[row].seed);
        state.rows[row].current_input.assign(flow_lm_.config().latent_size, std::numeric_limits<float>::quiet_NaN());
    }
    const int64_t batch = runtime.decode_runtime ? runtime.decode_runtime->batch_size : 1;
    if (runtime.decode_runtime) runtime.decode_runtime->import_states(prompted);
    state.inputs.resize(batch * flow_lm_.config().latent_size);
    state.noise.resize(state.inputs.size());
    state.active.resize(batch);
    return state;
}

std::vector<std::optional<FlowLMStepResult>> AcousticModel::next_rows(AcousticStreamState & state) const {
    std::fill(state.active.begin(), state.active.end(), 0);
    std::fill(state.inputs.begin(), state.inputs.end(), 0.0F);
    std::fill(state.noise.begin(), state.noise.end(), 0.0F);
    const auto latent_size = flow_lm_.config().latent_size;
    for (size_t row = 0; row < state.rows.size(); ++row) {
        auto & item = state.rows[row];
        if (item.done || item.step >= item.config.max_steps) continue;
        state.active[row] = 1;
        const auto noise = sample_noise_for_step(flow_lm_.config(), item);
        std::copy(noise.begin(), noise.end(), state.noise.begin() + row * latent_size);
        const auto input = state.runtime.decode_runtime
            ? replace_bos_sentinel_if_needed(state.runtime.decode_runtime->weights().bos_embedding(), item.current_input, latent_size)
            : item.current_input;
        std::copy(input.begin(), input.end(), state.inputs.begin() + row * latent_size);
    }
    if (std::none_of(state.active.begin(), state.active.end(), [](int64_t value) { return value != 0; })) return {};
    std::vector<std::optional<FlowLMStepResult>> results(state.rows.size());
    const std::vector<float> * output = nullptr;
    if (state.runtime.decode_runtime) {
        output = &state.runtime.decode_runtime->step(state.inputs, state.noise, state.active);
    } else {
        results.front() = flow_lm_.run_native_step(*state.runtime.step_runtime, state.inputs, state.noise);
    }
    for (size_t row = 0; row < state.rows.size(); ++row) {
        if (!state.active[row]) continue;
        auto & item = state.rows[row];
        auto & result = results[row];
        if (output) {
            const auto offset = row * (latent_size + 1);
            result = FlowLMStepResult{{output->begin() + offset, output->begin() + offset + latent_size}, (*output)[offset + latent_size]};
        }
        if (item.step >= item.config.min_eos_steps && result->eos_logit > item.config.eos_threshold && item.eos_step < 0)
            item.eos_step = item.step;
        if (item.eos_step >= 0 && item.step >= item.eos_step + item.config.frames_after_eos) {
            item.done = true;
            result.reset();
        } else {
            item.current_input = result->next_latent;
            ++item.step;
        }
    }
    return results;
}

std::vector<AcousticModelResult> AcousticModel::generate_rows(
    const AcousticPreparedRuntime & runtime, const Assets & manifest, const BackendWeights & weights,
    const std::vector<std::vector<float>> & text_embeddings, const std::vector<FlowLMState> & initial_states,
    const std::vector<AcousticGenerationConfig> & configs) const {
    std::vector<AcousticModelResult> results(configs.size());
    const auto started = std::chrono::steady_clock::now();
    auto state = start_rows(runtime, manifest, weights, text_embeddings, initial_states, configs);
    while (true) {
        const auto steps = next_rows(state);
        if (steps.empty()) break;
        for (size_t row = 0; row < steps.size(); ++row) {
            if (!steps[row]) continue;
            auto & result = results[row];
            result.latents.insert(result.latents.end(), steps[row]->next_latent.begin(), steps[row]->next_latent.end());
            result.eos_logits.push_back(steps[row]->eos_logit);
            ++result.generated_steps;
        }
    }
    for (const auto & result : results) if (result.generated_steps == 0)
        throw std::runtime_error("Acoustic generation produced an empty row");
    if (runtime.decode_runtime) {
        debug::timing_log_scalar("pocket_tts_small.flow_lm.step.input_write_ms", runtime.decode_runtime->input_write_ms);
        debug::timing_log_scalar("pocket_tts_small.flow_lm.step.graph.compute_ms", runtime.decode_runtime->graph_ms);
        debug::timing_log_scalar("pocket_tts_small.flow_lm.step.output_read_ms", runtime.decode_runtime->output_read_ms);
    }
    debug::timing_log_scalar("pocket_tts_small.acoustic.generate_ms", debug::elapsed_ms(started));
    return results;
}

AcousticModelResult AcousticModel::generate(
    const AcousticPreparedRuntime & runtime, const Assets & manifest, const BackendWeights & weights,
    const std::vector<float> & text_embeddings, const FlowLMState & initial_state,
    const AcousticGenerationConfig & config) const {
    return generate_rows(runtime, manifest, weights, {text_embeddings}, {initial_state}, {config}).front();
}

AcousticStreamState AcousticModel::start_stream(
    const AcousticPreparedRuntime & runtime, const Assets & manifest, const BackendWeights & weights,
    const std::vector<float> & text_embeddings, const FlowLMState & initial_state,
    const AcousticGenerationConfig & config) const {
    return start_rows(runtime, manifest, weights, {text_embeddings}, {initial_state}, {config});
}

std::optional<FlowLMStepResult> AcousticModel::next_stream_step(AcousticStreamState & state) const {
    auto results = next_rows(state);
    return results.empty() ? std::nullopt : std::move(results.front());
}

std::vector<AcousticModelResult> AcousticModel::generate_batch(
    ggml_backend_t backend, int threads, const Assets & manifest, const BackendWeights & weights,
    const std::vector<std::vector<float>> & text_embeddings, const std::vector<FlowLMState> & initial_states,
    const std::vector<AcousticGenerationConfig> & configs, int64_t cache_capacity,
    size_t weights_context_bytes, size_t graph_context_bytes) const {
    if (configs.empty() || configs.size() != text_embeddings.size() || configs.size() != initial_states.size())
        throw std::runtime_error("Acoustic inputs must have the same nonzero row count");
    if (flow_lm_.config().native_cpu) throw std::runtime_error("Batched acoustic generation requires GGML flow");
    int64_t required = 0;
    for (size_t row = 0; row < configs.size(); ++row) {
        validate_generation_inputs(flow_lm_.config(), text_embeddings[row], configs[row]);
        required = std::max(required, initial_states[row].current_end +
            static_cast<int64_t>(text_embeddings[row].size()) / flow_lm_.config().hidden_size + configs[row].max_steps);
    }
    if (cache_capacity < required) throw std::runtime_error("Batch cache capacity does not cover all request bounds");
    AcousticPreparedRuntime runtime;
    runtime.backend = backend;
    runtime.threads = threads;
    runtime.weights_context_bytes = weights_context_bytes;
    runtime.graph_context_bytes = graph_context_bytes;
    runtime.decode_runtime = prepare_decode_runtime(backend, threads, weights, configs.size(), cache_capacity,
        weights_context_bytes, graph_context_bytes);
    return generate_rows(runtime, manifest, weights, text_embeddings, initial_states, configs);
}

void AcousticModel::clear_runtime_cache() const noexcept {
    runtime_cache_ = {};
    decode_runtime_.reset();
}

int64_t AcousticModel::prepared_prompt_capacity() const noexcept {
    return runtime_cache_.runtime || decode_runtime_ ? runtime_cache_.prompt_capacity : 0;
}

int AcousticModel::prepared_max_steps_capacity() const noexcept {
    return runtime_cache_.runtime || decode_runtime_ ? runtime_cache_.max_steps_capacity : 0;
}

}  // namespace engine::models::pocket_tts_small
