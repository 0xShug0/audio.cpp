#include "engine/models/mulacover/ar.h"
#include "engine/models/mulacover/conditioning.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/hf_sampler.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <limits>
#include <stdexcept>

namespace engine::models::mulacover {
namespace {
using core::TensorShape;
using core::TensorValue;
namespace binding = modules::binding;

struct DecoderWeights {
    modules::DecoderStackWeights stack;
    modules::NormWeights final_norm;
    std::optional<modules::LinearWeights> lm_head;
};

DecoderWeights bind_decoder(core::BackendWeightStore & store,
    const assets::TensorSource & source, const modules::DecoderStackConfig & config,
    const std::string & prefix, assets::TensorStorageType storage) {
    // Llama3 scaled RoPE; GGML consumes frequency divisors.
    std::vector<float> factors;
    for (int64_t i = 0; i < config.head_dim; i += 2) {
        const float frequency = std::pow(config.rope_theta, -static_cast<float>(i) / config.head_dim);
        const float wavelength = 2.0F * 3.14159265358979323846F / frequency;
        const float smooth = (8192.0F / wavelength - 1.0F) / 3.0F;
        const float scaled = wavelength > 8192.0F ? frequency / 32.0F : wavelength >= 2048.0F
            ? (1.0F - smooth) * frequency / 32.0F + smooth * frequency : frequency;
        factors.push_back(frequency / scaled);
    }
    const auto rope = store.make_f32(TensorShape::from_dims({config.head_dim / 2}), std::move(factors));
    DecoderWeights weights;
    for (int64_t i = 0; i < config.layers; ++i) {
        const auto layer = prefix + ".layers." + std::to_string(i);
        modules::DecoderLayerWeights w;
        w.input_norm = binding::norm_from_named_source(store, source, layer + ".sa_norm.scale", std::nullopt);
        w.post_norm = binding::norm_from_named_source(store, source, layer + ".mlp_norm.scale", std::nullopt);
        w.self_attention.q_weight = store.load_tensor(source, layer + ".attn.q_proj.weight", storage,
            {config.num_attention_heads * config.head_dim, config.hidden_size});
        w.self_attention.k_weight = store.load_tensor(source, layer + ".attn.k_proj.weight", storage,
            {config.num_key_value_heads * config.head_dim, config.hidden_size});
        w.self_attention.v_weight = store.load_tensor(source, layer + ".attn.v_proj.weight", storage,
            {config.num_key_value_heads * config.head_dim, config.hidden_size});
        w.self_attention.out_weight = store.load_tensor(source, layer + ".attn.output_proj.weight", storage,
            {config.hidden_size, config.hidden_size});
        w.mlp.gate_proj = binding::linear_from_source(store, source, layer + ".mlp.w1", storage,
            config.intermediate_size, config.hidden_size, false);
        w.mlp.up_proj = binding::linear_from_source(store, source, layer + ".mlp.w3", storage,
            config.intermediate_size, config.hidden_size, false);
        w.mlp.down_proj = binding::linear_from_source(store, source, layer + ".mlp.w2", storage,
            config.hidden_size, config.intermediate_size, false);
        w.rope_frequency_factors = rope;
        weights.stack.layers.push_back(std::move(w));
    }
    weights.final_norm = binding::norm_from_named_source(store, source, prefix + ".norm.scale", std::nullopt);
    return weights;
}
}  // namespace

struct AutoregressiveRuntime::Impl {
    enum class AudioOperation { Embed, SumFrame, Project, Head };

    // Fixed-shape graphs retained by the session. Codebook heads share scratch.
    struct AudioGraph {
        core::ExecutionContext & execution;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
        core::HostGraphPlan plan;
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
        ggml_cgraph * graph = nullptr;
        ggml_tensor * input = nullptr;
        ggml_tensor * output = nullptr;

        AudioGraph(Impl & owner, AudioOperation operation, int64_t batch, int64_t steps = 1, int64_t codebook = 0)
            : execution(owner.execution) {
            context.reset(ggml_init({2ull << 20, nullptr, true}));
            if (!context) throw std::runtime_error("MuLaCover audio graph context allocation failed");
            core::ModuleBuildContext ctx{context.get(), "mulacover.audio", execution.backend_type()};
            const auto & config = owner.assets.mula_config;
            TensorValue value;
            if (operation == AudioOperation::Embed || operation == AudioOperation::SumFrame) {
                const auto count = operation == AudioOperation::Embed ? 1 : config.audio_num_codebooks;
                input = ggml_new_tensor_1d(ctx.ggml, GGML_TYPE_I32, batch * count);
                value = modules::EmbeddingModule({config.audio_num_codebooks * config.audio_vocab_size,
                    config.backbone.hidden_size}).build(ctx,
                    core::wrap_tensor(input, TensorShape::from_dims({batch * count}), GGML_TYPE_I32), owner.audio_embeddings);
                value = core::reshape_tensor(ctx, value, TensorShape::from_dims({batch, count, config.backbone.hidden_size}));
                if (count > 1) {
                    value = modules::ReduceSumModule({1}).build(ctx, value);
                }
            } else {
                value = core::make_tensor(ctx, GGML_TYPE_F32,
                    TensorShape::from_dims({batch, steps, config.backbone.hidden_size}));
                input = value.tensor;
                if (operation == AudioOperation::Project) {
                    value = modules::LinearModule({config.backbone.hidden_size, config.decoder.hidden_size, false})
                        .build(ctx, value, owner.projection);
                } else {
                    auto selected = modules::SliceModule({0, codebook, 1}).build(ctx, owner.audio_head);
                    selected = core::reshape_tensor(ctx, selected,
                        TensorShape::from_dims({config.decoder.hidden_size, config.audio_vocab_size}));
                    value = modules::MatMulModule{}.build(ctx,
                        core::reshape_tensor(ctx, value, TensorShape::from_dims({batch, config.decoder.hidden_size})), selected);
                }
            }
            if (owner.compute_type == GGML_TYPE_BF16) value = modules::Bf16RoundModule{}.build(ctx, value);
            else if (owner.compute_type == GGML_TYPE_F16) {
                value = core::wrap_tensor(ggml_cast(ctx.ggml, ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F16), GGML_TYPE_F32),
                    value.shape, GGML_TYPE_F32);
            }
            ggml_set_input(input);
            output = value.tensor;
            ggml_set_output(output);
            graph = ggml_new_graph_custom(ctx.ggml, 1024, false);
            ggml_build_forward_expand(graph, output);
            core::validate_backend_graph_supported(execution.backend(), graph, "mulacover.audio");
            auto * scratch = owner.head_allocator.get();
            if (operation != AudioOperation::Head) {
                allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
                scratch = allocator.get();
            }
            if (!scratch || !ggml_gallocr_alloc_graph(scratch, graph)) {
                throw std::runtime_error("MuLaCover audio graph workspace allocation failed");
            }
            plan.backend = execution.backend();
            plan.plan = core::create_backend_graph_plan_if_host(execution.backend(), graph);
            if (operation != AudioOperation::Head) {
                debug::timing_log_scalar("mulacover.audio.graph_buffer_mb",
                    static_cast<double>(ggml_gallocr_get_buffer_size(scratch, 0)) / (1024 * 1024));
            }
        }
        ~AudioGraph() {
            plan.reset();
            core::release_backend_graph_resources(execution.backend(), graph, true);
        }
        template <typename T>
        std::vector<float> run(const std::vector<T> & values) {
            if (values.size() * sizeof(T) != ggml_nbytes(input)) {
                throw std::invalid_argument("MuLaCover audio graph input size mismatch");
            }
            ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(T));
            if (core::compute_backend_graph(execution.backend(), graph, plan.plan, "mulacover.audio") != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("MuLaCover audio graph compute failed");
            }
            return core::read_tensor_f32(output);
        }
    };

    struct DecoderResult {
        std::vector<float> hidden, logits;
    };

    struct DecoderRuntime {
        struct Graph {
            DecoderRuntime & runtime;
            int64_t steps, attention_rows;
            bool prefill;
            std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
            std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
            core::HostGraphPlan plan;
            ggml_cgraph * graph = nullptr;
            ggml_tensor * input = nullptr, *positions = nullptr, *mask = nullptr, *slot = nullptr;
            ggml_tensor * hidden = nullptr, *logits = nullptr;
            std::vector<int32_t> position_values, slots;
            std::vector<ggml_fp16_t> mask_values;

            Graph(DecoderRuntime & runtime, int64_t steps, int64_t attention_rows, bool prefill)
                : runtime(runtime), steps(steps), attention_rows(attention_rows), prefill(prefill) {
                auto & owner = runtime.owner;
                const auto & config = runtime.config;
                context.reset(ggml_init({32ull << 20, nullptr, true}));
                if (!context) throw std::runtime_error("MuLaCover decoder graph context allocation failed");
                core::ModuleBuildContext ctx{context.get(), runtime.name.c_str(), owner.execution.backend_type()};
                auto x = core::make_tensor(ctx, GGML_TYPE_F32,
                    TensorShape::from_dims({runtime.batch, steps, config.hidden_size}));
                input = x.tensor;
                positions = ggml_new_tensor_1d(ctx.ggml, GGML_TYPE_I32, steps);
                auto position_value = core::wrap_tensor(positions, TensorShape::from_dims({steps}), GGML_TYPE_I32);
                mask = ggml_new_tensor_4d(ctx.ggml, GGML_TYPE_F16, attention_rows, steps, 1, runtime.batch);
                auto attention_mask = core::wrap_tensor(mask,
                    TensorShape::from_dims({runtime.batch, 1, steps, attention_rows}), GGML_TYPE_F16);
                if (!prefill) slot = ggml_new_tensor_1d(ctx.ggml, GGML_TYPE_I32, runtime.batch);
                for (auto * value : {input, positions, mask}) ggml_set_input(value);
                if (slot) ggml_set_input(slot);
                graph = ggml_new_graph_custom(ctx.ggml, 65536, false);
                const modules::DecoderLayerModule layer_module(modules::decoder_layer_config_from_stack(config));
                for (size_t layer = 0; layer < runtime.weights.stack.layers.size(); ++layer) {
                    if (runtime.is_backbone) x = owner.conditioning.before_layer(ctx, x, position_value, layer);
                    if (prefill) {
                        auto output = layer_module.build(ctx, x, position_value, runtime.weights.stack.layers[layer],
                            std::nullopt, std::nullopt, attention_mask);
                        x = output.output;
                        for (int64_t lane = 0; lane < runtime.batch; ++lane) {
                            auto key = modules::SliceModule({0, lane, 1}).build(ctx, output.key);
                            auto value = modules::SliceModule({0, lane, 1}).build(ctx, output.value);
                            auto key_dest = engine::runtime::view_transformer_kv_cache_steps(ctx, runtime.cache.key_tensor(layer),
                                0, steps, config.num_key_value_heads, config.head_dim, "MuLaCover prefill key", owner.compute_type, lane);
                            auto value_dest = engine::runtime::view_transformer_kv_cache_steps(ctx, runtime.cache.value_tensor(layer),
                                0, steps, config.num_key_value_heads, config.head_dim, "MuLaCover prefill value", owner.compute_type, lane);
                            ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, key.tensor, key_dest.tensor));
                            ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, value.tensor, value_dest.tensor));
                        }
                    } else {
                        const auto cache_slot = core::wrap_tensor(slot, TensorShape::from_dims({runtime.batch}), GGML_TYPE_I32);
                        x = layer_module.build_with_static_cache_tail_batched(ctx, graph, x, position_value,
                            runtime.weights.stack.layers[layer], runtime.cache.key_tensor(layer), runtime.cache.value_tensor(layer),
                            cache_slot, attention_mask).output;
                    }
                    if (runtime.is_backbone) x = owner.conditioning.after_layer(ctx, x, position_value, layer);
                }
                x = modules::SliceModule({1, steps - 1, 1}).build(ctx, x);
                x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false, false, false,
                    config.activation_cast.norm_intermediate_type}).build(ctx, x, runtime.weights.final_norm);
                const auto round = [&](TensorValue value) {
                    if (owner.compute_type == GGML_TYPE_BF16) return modules::Bf16RoundModule{}.build(ctx, value);
                    if (owner.compute_type == GGML_TYPE_F16) {
                        return core::wrap_tensor(ggml_cast(ctx.ggml, ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F16), GGML_TYPE_F32),
                            value.shape, GGML_TYPE_F32);
                    }
                    return value;
                };
                x = round(x);
                hidden = x.tensor;
                ggml_set_output(hidden);
                ggml_build_forward_expand(graph, hidden);
                if (runtime.is_backbone) {
                    logits = round(modules::LinearModule({config.hidden_size, owner.assets.mula_config.audio_vocab_size, false})
                        .build(ctx, x, *runtime.weights.lm_head)).tensor;
                    ggml_set_output(logits);
                    ggml_build_forward_expand(graph, logits);
                }
                core::validate_backend_graph_supported(owner.execution.backend(), graph, runtime.name.c_str());
                position_values.resize(static_cast<size_t>(steps));
                mask_values.resize(static_cast<size_t>(runtime.batch * steps * attention_rows));
                slots.resize(static_cast<size_t>(runtime.batch));
                if (prefill) allocate_prefill_workspace();
            }
            ~Graph() {
                plan.reset();
                core::release_backend_graph_resources(runtime.owner.execution.backend(), graph, true);
            }

            void allocate_prefill_workspace() {
                if (allocator) return;
                auto & execution = runtime.owner.execution;
                allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
                if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
                    throw std::runtime_error("MuLaCover prefill workspace allocation failed");
                }
                plan.backend = execution.backend();
                plan.plan = core::create_backend_graph_plan_if_host(execution.backend(), graph);
                debug::timing_log_scalar(runtime.name + ".prefill.graph_buffer_mb",
                    static_cast<double>(ggml_gallocr_get_buffer_size(allocator.get(), 0)) / (1024 * 1024));
            }

            void release_prefill_workspace() {
                plan.reset();
                core::release_backend_graph_resources(runtime.owner.execution.backend(), graph, true);
                allocator.reset();
                for (auto * tensor = ggml_get_first_tensor(context.get()); tensor;
                    tensor = ggml_get_next_tensor(context.get(), tensor)) {
                    tensor->data = nullptr;
                    tensor->buffer = nullptr;
                    tensor->extra = nullptr;
                }
                debug::timing_log_scalar(runtime.name + ".prefill.graph_buffer_mb", 0.0);
            }

            DecoderResult run(const std::vector<float> & embeddings) {
                if (embeddings.size() * sizeof(float) != ggml_nbytes(input)) {
                    throw std::invalid_argument("MuLaCover decoder embedding shape mismatch");
                }
                const int64_t offset = prefill ? 0 : runtime.cache.valid_steps();
                std::iota(position_values.begin(), position_values.end(), static_cast<int32_t>(offset));
                std::fill(mask_values.begin(), mask_values.end(), ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
                for (int64_t lane = 0; lane < runtime.batch; ++lane) for (int64_t row = 0; row < steps; ++row) {
                    std::fill_n(mask_values.data() + (lane * steps + row) * attention_rows, offset + row + 1, ggml_fp32_to_fp16(0.0F));
                }
                ggml_backend_tensor_set(input, embeddings.data(), 0, embeddings.size() * sizeof(float));
                ggml_backend_tensor_set(positions, position_values.data(), 0, position_values.size() * sizeof(int32_t));
                ggml_backend_tensor_set(mask, mask_values.data(), 0, mask_values.size() * sizeof(ggml_fp16_t));
                if (slot) {
                    for (int64_t lane = 0; lane < runtime.batch; ++lane) slots[lane] = static_cast<int32_t>(lane * runtime.capacity + offset);
                    ggml_backend_tensor_set(slot, slots.data(), 0, slots.size() * sizeof(int32_t));
                }
                auto & execution = runtime.owner.execution;
                auto * host_plan = prefill ? plan.plan : runtime.decode_plan.plan;
                if (core::compute_backend_graph(execution.backend(), graph, host_plan, runtime.name.c_str()) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("MuLaCover decoder graph execution failed");
                }
                runtime.cache.advance_after_direct_append(steps);
                DecoderResult result;
                result.hidden = core::read_tensor_f32(hidden);
                if (logits) result.logits = core::read_tensor_f32(logits);
                return result;
            }
        };

        Impl & owner;
        bool is_backbone;
        std::string name;
        modules::DecoderStackConfig config;
        const DecoderWeights & weights;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> cache_context{nullptr, ggml_free};
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> cache_buffer{nullptr, ggml_backend_buffer_free};
        runtime::TransformerBatchedKVCache cache;
        int64_t batch = 0, capacity = 0;
        std::unique_ptr<Graph> prefix;
        std::map<int64_t, std::unique_ptr<Graph>> decode_graphs;
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> decode_allocator{nullptr, ggml_gallocr_free};
        core::HostGraphPlan decode_plan;
        Graph * active = nullptr;

        DecoderRuntime(Impl & owner, bool is_backbone)
            : owner(owner), is_backbone(is_backbone), name(is_backbone ? "mulacover.backbone" : "mulacover.depth"),
              config(is_backbone ? owner.backbone_config : owner.depth_config),
              weights(is_backbone ? owner.backbone_weights : owner.depth_weights) {}
        ~DecoderRuntime() {
            decode_plan.reset();
            prefix.reset();
            decode_graphs.clear();
        }

        DecoderResult prefill(const std::vector<float> & embeddings, int64_t batch_size, int64_t steps, int64_t requested_capacity) {
            if (steps <= 0 || steps > requested_capacity || requested_capacity > owner.assets.mula_config.max_seq_len) {
                throw std::invalid_argument("MuLaCover prefill exceeds its bounded cache");
            }
            if (batch != batch_size || capacity < requested_capacity) {
                decode_plan.reset();
                active = nullptr;
                prefix.reset();
                decode_graphs.clear();
                decode_allocator.reset();
                cache = {};
                cache_buffer.reset();
                cache_context.reset(ggml_init({2ull << 20, nullptr, true}));
                if (!cache_context) throw std::runtime_error("MuLaCover KV context allocation failed");
                capacity = requested_capacity;
                batch = batch_size;
                core::ModuleBuildContext ctx{cache_context.get(), name.c_str(), owner.execution.backend_type()};
                std::vector<TensorValue> keys, values;
                for (int64_t layer = 0; layer < config.layers; ++layer) {
                    const auto shape = TensorShape::from_dims({batch, capacity, config.num_key_value_heads, config.head_dim});
                    keys.push_back(core::make_tensor(ctx, owner.compute_type, shape));
                    values.push_back(core::make_tensor(ctx, owner.compute_type, shape));
                }
                cache_buffer.reset(ggml_backend_alloc_ctx_tensors(cache_context.get(), owner.execution.backend()));
                if (!cache_buffer) throw std::runtime_error("MuLaCover KV buffer allocation failed");
                runtime::TransformerKVCacheOptions options;
                options.allow_f16_storage = owner.compute_type == GGML_TYPE_F16;
                options.allow_bf16_storage = owner.compute_type == GGML_TYPE_BF16;
                options.lazy_import_scratch = true;
                cache = runtime::TransformerBatchedKVCache(capacity, batch, config.num_key_value_heads * config.head_dim,
                    std::move(keys), std::move(values), options);
                debug::timing_log_scalar(name + ".kv.type", std::string_view(ggml_type_name(owner.compute_type)));
                debug::timing_log_scalar(name + ".kv.buffer_mb",
                    static_cast<double>(ggml_backend_buffer_get_size(cache_buffer.get())) / (1024 * 1024));
            }
            cache.clear_on_backend();
            if (!prefix || prefix->steps != steps) {
                prefix.reset();
                prefix = std::make_unique<Graph>(*this, steps, steps, true);
            }
            prefix->allocate_prefill_workspace();
            auto result = prefix->run(embeddings);
            if (is_backbone) prefix->release_prefill_workspace();
            return result;
        }

        DecoderResult decode(const std::vector<float> & embeddings, int64_t batch_size) {
            if (batch_size != batch || cache.valid_steps() >= capacity) {
                throw std::invalid_argument("MuLaCover decode exceeds its active cache or batch size");
            }
            const int64_t rows = is_backbone ? std::min(capacity, ((cache.valid_steps() + 128) / 128) * 128) : capacity;
            auto & graph = decode_graphs[rows];
            if (!graph) graph = std::make_unique<Graph>(*this, 1, rows, false);
            if (active != graph.get()) {
                decode_plan.reset();
                if (active) core::release_backend_graph_resources(owner.execution.backend(), active->graph, true);
                for (auto * tensor = ggml_get_first_tensor(graph->context.get()); tensor;
                    tensor = ggml_get_next_tensor(graph->context.get(), tensor)) {
                    tensor->data = nullptr;
                    tensor->buffer = nullptr;
                    tensor->extra = nullptr;
                }
                if (!decode_allocator) decode_allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(owner.execution.backend())));
                if (!decode_allocator || !ggml_gallocr_alloc_graph(decode_allocator.get(), graph->graph)) {
                    throw std::runtime_error("MuLaCover decode workspace allocation failed");
                }
                decode_plan.backend = owner.execution.backend();
                decode_plan.plan = core::create_backend_graph_plan_if_host(owner.execution.backend(), graph->graph);
                active = graph.get();
                debug::timing_log_scalar(name + ".decode.graph_buffer_mb",
                    static_cast<double>(ggml_gallocr_get_buffer_size(decode_allocator.get(), 0)) / (1024 * 1024));
            }
            return graph->run(embeddings);
        }
    };

    core::ExecutionContext & execution;
    const MuLaCoverAssets & assets;
    ggml_type compute_type;
    ConditioningRuntime conditioning;
    core::BackendWeightStore store;
    PromptWeights prompt;
    TensorValue audio_embeddings, audio_head;
    modules::LinearWeights projection;
    modules::DecoderStackConfig backbone_config, depth_config;
    DecoderWeights backbone_weights, depth_weights;
    std::unique_ptr<DecoderRuntime> backbone, depth;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> head_allocator{nullptr, ggml_gallocr_free};
    std::unique_ptr<AudioGraph> embed, sum_frame, project_step, project_prefix;
    std::vector<std::unique_ptr<AudioGraph>> heads;
    uint64_t conditioning_revision = 0;
    int64_t graph_batch = 0;
    int64_t backbone_capacity = 0;

    Impl(core::ExecutionContext & execution, const MuLaCoverAssets & assets, assets::TensorStorageType storage)
        : execution(execution), assets(assets),
          compute_type(storage == assets::TensorStorageType::F32 || execution.backend_type() == core::BackendType::Cpu
              ? GGML_TYPE_F32 : execution.backend_type() == core::BackendType::Vulkan ? GGML_TYPE_F16 : GGML_TYPE_BF16),
          conditioning(execution, assets.mula_weights,
              assets.mula_config.backbone.layers - assets.mula_config.peft_layer_num,
              assets.mula_config.qwen_dim, assets.mula_config.train_tag, storage,
              compute_type),
          store(execution.backend(), execution.backend_type(), "mulacover.ar.weights", 8ull << 20) {
        const auto & c = assets.mula_config;
        const auto & source = *assets.mula_weights;
        prompt.compute_type = compute_type;
        prompt.text = store.load_tensor(source, "text_embeddings.weight", storage, {c.text_vocab_size, c.backbone.hidden_size});
        prompt.unconditional = store.load_tensor(source, "unconditional_text_embedding.weight", storage, {1, c.backbone.hidden_size});
        prompt.muq = binding::linear_from_source(store, source, "muq_linear", storage, c.backbone.hidden_size, c.muq_dim, true);
        prompt.qwen = binding::linear_from_source(store, source, "qwen_linear", storage, c.backbone.hidden_size, c.qwen_dim, true);
        audio_embeddings = store.load_tensor(source, "audio_embeddings.weight", storage,
            {c.audio_num_codebooks * c.audio_vocab_size, c.backbone.hidden_size});
        audio_head = store.load_tensor(source, "audio_head", storage,
            {c.audio_num_codebooks - 1, c.decoder.hidden_size, c.audio_vocab_size});
        projection = binding::linear_from_source(store, source, "projection", storage, c.decoder.hidden_size, c.backbone.hidden_size, false);
        backbone_weights = bind_decoder(store, source, c.backbone, "backbone", storage);
        backbone_weights.lm_head = binding::linear_from_source(store, source, "codebook0_head", storage,
            c.audio_vocab_size, c.backbone.hidden_size, false);
        depth_weights = bind_decoder(store, source, c.decoder, "decoder", storage);
        store.upload();
        debug::timing_log_scalar("mulacover.ar.compute_type", std::string_view(ggml_type_name(compute_type)));
        backbone_config = c.backbone;
        depth_config = c.decoder;
        for (auto * config : {&backbone_config, &depth_config}) {
            auto & cast = config->activation_cast;
            cast.enabled = compute_type != GGML_TYPE_F32;
            cast.type = compute_type;
            cast.fused_round = compute_type == GGML_TYPE_BF16;
            cast.after_input_norm = cast.after_qkv_projection = cast.after_rope = true;
            cast.after_attention = cast.after_attention_output = cast.after_residual = true;
            cast.after_ffn_norm = cast.after_mlp_projection = cast.after_mlp_silu = true;
            cast.after_mlp_mul = cast.after_output = true;
            if (cast.enabled) {
                cast.norm_intermediate_type = compute_type;
            }
        }
    }

    GeneratedFrames generate(const PromptEmbeddings & p, const GenerationOptions & options,
        uint64_t seed, int64_t max_frames, const std::vector<int32_t> * teacher) {
        const auto & c = assets.mula_config;
        const int64_t batch = options.guidance_scale > 1.0F ? 2 : 1;
        if (p.batch_size != batch || p.dims != c.backbone.hidden_size || p.steps <= 0 || max_frames <= 0 ||
            p.steps + max_frames > c.max_seq_len || options.top_k <= 0 || options.top_k > c.audio_vocab_size ||
            !(options.temperature > 0) || !std::isfinite(options.temperature)) {
            throw std::invalid_argument("MuLaCover invalid generation shape, sampling options or context capacity");
        }
        if (teacher && teacher->size() != static_cast<size_t>(max_frames * c.audio_num_codebooks)) {
            throw std::invalid_argument("MuLaCover teacher codes must cover every requested frame");
        }
        if (teacher) for (auto code : *teacher) {
            if (code < 0 || code >= c.audio_vocab_size) throw std::invalid_argument("MuLaCover teacher code outside vocabulary");
        }
        if (conditioning_revision != conditioning.revision()) {
            backbone.reset();
            backbone_capacity = 0;
            conditioning_revision = conditioning.revision();
        }
        if (!backbone) backbone = std::make_unique<DecoderRuntime>(*this, true);
        if (!depth) depth = std::make_unique<DecoderRuntime>(*this, false);
        if (graph_batch != batch) {
            backbone_capacity = 0;
            embed.reset(); sum_frame.reset(); project_step.reset(); project_prefix.reset(); heads.clear();
            head_allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
            embed = std::make_unique<AudioGraph>(*this, AudioOperation::Embed, batch);
            sum_frame = std::make_unique<AudioGraph>(*this, AudioOperation::SumFrame, batch);
            project_step = std::make_unique<AudioGraph>(*this, AudioOperation::Project, batch);
            project_prefix = std::make_unique<AudioGraph>(*this, AudioOperation::Project, batch, 2);
            for (int64_t q = 1; q < c.audio_num_codebooks; ++q) {
                heads.push_back(std::make_unique<AudioGraph>(*this, AudioOperation::Head, batch, 1, q - 1));
            }
            debug::timing_log_scalar("mulacover.depth_head.graph_buffer_mb",
                static_cast<double>(ggml_gallocr_get_buffer_size(head_allocator.get(), 0)) / (1024 * 1024));
            graph_batch = batch;
        }
        const auto cache_steps = p.steps + max_frames;
        backbone_capacity = std::max(backbone_capacity, cache_steps);
        const double kv_bytes = 2.0 * batch * c.backbone.layers * c.backbone.num_key_value_heads * c.backbone.head_dim *
            backbone_capacity * ggml_type_size(compute_type);
        debug::timing_log_scalar("mulacover.backbone.kv_capacity_mb", kv_bytes / (1024 * 1024));
        debug::timing_log_scalar("mulacover.backbone.cache_steps", backbone_capacity);
        auto current = backbone->prefill(p.values, batch, p.steps, cache_steps);
        const auto policy = sampling::resolve_torch_cuda_sampling_policy(execution.backend_type(), execution.config().device,
            "mulacover.cuda_sampling_policy", "MuLaCover");
        sampling::HfSamplerScratch scratch;
        sampling::HfSamplingOptions sampling_options;
        sampling_options.top_k = options.top_k;
        sampling_options.temperature = 1.0F;
        sampling::TorchCudaSamplingPolicy sampling_policy = policy;
        // Preserve the reference CUDA tensor-iterator draws on every backend.
        sampling_policy.cuda_fast_path = true;
        std::mt19937 unused_rng(static_cast<uint32_t>(seed));
        sampling::HfSampler sampler;
        std::vector<float> guided(static_cast<size_t>(c.audio_vocab_size));
        uint64_t calls = 0;
        const auto round_scalar = [&](float value) {
            if (compute_type == GGML_TYPE_BF16) return ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
            if (compute_type == GGML_TYPE_F16) return ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
            return value;
        };
        const auto sample = [&](const std::vector<float> & logits) {
            for (int64_t v = 0; v < c.audio_vocab_size; ++v) {
                const float score = batch == 2 ? round_scalar(logits[c.audio_vocab_size + v] +
                    round_scalar(round_scalar(logits[v] - logits[c.audio_vocab_size + v]) * options.guidance_scale)) : logits[v];
                guided[v] = round_scalar(score / options.temperature);
            }
            const sampling::HfTorchSamplingState state{&sampling_policy, seed, calls++};
            return sampler.sample(guided, {}, sampling_options, scratch, unused_rng, &state, "MuLaCover");
        };
        GeneratedFrames out;
        out.codebooks = c.audio_num_codebooks;
        for (int64_t f = 0; f < max_frames; ++f) {
            std::vector<int32_t> frame(static_cast<size_t>(c.audio_num_codebooks));
            int32_t token = sample(current.logits);
            if (teacher) token = (*teacher)[f * c.audio_num_codebooks];
            frame[0] = token;
            std::vector<int32_t> ids(static_cast<size_t>(batch), token);
            auto embedding = embed->run(ids);
            std::vector<float> prefix(static_cast<size_t>(batch * 2 * c.backbone.hidden_size));
            for (int64_t b = 0; b < batch; ++b) {
                std::copy_n(current.hidden.data() + b * c.backbone.hidden_size, c.backbone.hidden_size,
                    prefix.data() + b * 2 * c.backbone.hidden_size);
                std::copy_n(embedding.data() + b * c.backbone.hidden_size, c.backbone.hidden_size,
                    prefix.data() + (b * 2 + 1) * c.backbone.hidden_size);
            }
            auto decoded = depth->prefill(
                project_prefix->run(prefix), batch, 2, c.audio_num_codebooks);
            for (int64_t q = 1; q < c.audio_num_codebooks; ++q) {
                token = sample(heads[q - 1]->run(decoded.hidden));
                if (teacher) token = (*teacher)[f * c.audio_num_codebooks + q];
                frame[q] = token;
                if (q + 1 < c.audio_num_codebooks) {
                    std::fill(ids.begin(), ids.end(), static_cast<int32_t>(q * c.audio_vocab_size + token));
                    decoded = depth->decode(project_step->run(embed->run(ids)), batch);
                }
            }
            if (std::any_of(frame.begin(), frame.end(), [&](int32_t code) { return code >= assets.generation_config.audio_eos_id; })) break;
            for (auto code : frame) if (code < 0 || code >= assets.generation_config.audio_eos_id - 1) {
                throw std::runtime_error("MuLaCover generated a padding or invalid audio code");
            }
            out.codes.insert(out.codes.end(), frame.begin(), frame.end());
            ++out.frames;
            if (f + 1 < max_frames) {
                std::vector<int32_t> all_ids;
                for (int64_t b = 0; b < batch; ++b) for (int64_t q = 0; q < c.audio_num_codebooks; ++q) {
                    all_ids.push_back(static_cast<int32_t>(q * c.audio_vocab_size + frame[q]));
                }
                current = backbone->decode(sum_frame->run(all_ids), batch);
            }
        }
        out.rng_offset_blocks = calls * sampling::torch_cuda_tensor_iterator_offset_blocks(c.audio_vocab_size, policy);
        out.rng_policy = policy;
        return out;
    }
};

AutoregressiveRuntime::AutoregressiveRuntime(core::ExecutionContext & execution, const MuLaCoverAssets & assets, assets::TensorStorageType storage)
    : impl_(std::make_unique<Impl>(execution, assets, storage)) {}
AutoregressiveRuntime::~AutoregressiveRuntime() = default;
const PromptWeights & AutoregressiveRuntime::prompt_weights() const { return impl_->prompt; }
void AutoregressiveRuntime::prepare(const SymbolicInputs & symbolic, const std::vector<float> & style, bool cfg_enabled) {
    const int64_t batch = cfg_enabled ? 2 : 1;
    if (impl_->conditioning.context_length() != symbolic.length || impl_->conditioning.batch_size() != batch) {
        impl_->backbone.reset();
    }
    impl_->conditioning.prepare(symbolic, style, cfg_enabled);
}
GeneratedFrames AutoregressiveRuntime::generate(const PromptEmbeddings & prompt, const GenerationOptions & options,
    uint64_t seed, int64_t max_frames, const std::vector<int32_t> * teacher_codes) {
    return impl_->generate(prompt, options, seed, max_frames, teacher_codes);
}

}  // namespace engine::models::mulacover
