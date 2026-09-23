#include "engine/framework/decoders/moss_tts_delay/backbone.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/transformers/qwen_decoder.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/kv_cache.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::decoders {
namespace {

namespace modules = engine::modules;
namespace binding = engine::modules::binding;
using Clock = std::chrono::steady_clock;

constexpr float kMaskedAttentionBias = std::numeric_limits<float>::lowest();

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct VoiceGenBackboneLayerWeights {
    core::TensorValue input_norm;
    core::TensorValue q_proj;
    core::TensorValue k_proj;
    core::TensorValue v_proj;
    core::TensorValue o_proj;
    core::TensorValue q_norm;
    core::TensorValue k_norm;
    core::TensorValue post_norm;
    core::TensorValue gate_proj;
    core::TensorValue up_proj;
    core::TensorValue down_proj;
};

struct MossTtsDelayBackboneWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue embed_tokens;
    std::vector<VoiceGenBackboneLayerWeights> layers;
    core::TensorValue norm;
};

void validate_weight_storage_type(assets::TensorStorageType storage_type) {
    switch (storage_type) {
        case assets::TensorStorageType::Native:
        case assets::TensorStorageType::F32:
        case assets::TensorStorageType::F16:
        case assets::TensorStorageType::BF16:
        case assets::TensorStorageType::Q8_0:
            return;
        default:
            throw std::runtime_error(
                "MOSS delay backbone weight_type supports only native, f32, f16, bf16, and q8_0");
    }
}

MossTtsDelayBackboneWeights load_backbone_weights(
    const MossTtsDelayConfig & model_config,
    const assets::TensorSource & source,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    validate_weight_storage_type(storage_type);
    const auto & config = model_config.backbone;
    MossTtsDelayBackboneWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "moss_tts_delay.backbone.weights",
        weight_context_bytes);
    weights.embed_tokens = weights.store->load_tensor(
        source,
        "language_model.embed_tokens.weight",
        storage_type,
        {config.vocab_size, config.hidden_size});
    const int64_t dim = config.head_dim;
    weights.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = "language_model.layers." + std::to_string(layer);
        VoiceGenBackboneLayerWeights w;
        w.input_norm = weights.store->load_f32_tensor(source, prefix + ".input_layernorm.weight", {config.hidden_size});
        w.q_proj = weights.store->load_tensor(
            source,
            prefix + ".self_attn.q_proj.weight",
            storage_type,
            {config.num_attention_heads * dim, config.hidden_size});
        w.k_proj = weights.store->load_tensor(
            source,
            prefix + ".self_attn.k_proj.weight",
            storage_type,
            {config.num_key_value_heads * dim, config.hidden_size});
        w.v_proj = weights.store->load_tensor(
            source,
            prefix + ".self_attn.v_proj.weight",
            storage_type,
            {config.num_key_value_heads * dim, config.hidden_size});
        w.o_proj = weights.store->load_tensor(
            source,
            prefix + ".self_attn.o_proj.weight",
            storage_type,
            {config.hidden_size, config.num_attention_heads * dim});
        w.q_norm = weights.store->load_f32_tensor(source, prefix + ".self_attn.q_norm.weight", {dim});
        w.k_norm = weights.store->load_f32_tensor(source, prefix + ".self_attn.k_norm.weight", {dim});
        w.post_norm = weights.store->load_f32_tensor(
            source,
            prefix + ".post_attention_layernorm.weight",
            {config.hidden_size});
        w.gate_proj = weights.store->load_tensor(
            source,
            prefix + ".mlp.gate_proj.weight",
            storage_type,
            {config.intermediate_size, config.hidden_size});
        w.up_proj = weights.store->load_tensor(
            source,
            prefix + ".mlp.up_proj.weight",
            storage_type,
            {config.intermediate_size, config.hidden_size});
        w.down_proj = weights.store->load_tensor(
            source,
            prefix + ".mlp.down_proj.weight",
            storage_type,
            {config.hidden_size, config.intermediate_size});
        weights.layers.push_back(std::move(w));
    }
    weights.norm = weights.store->load_f32_tensor(source, "language_model.norm.weight", {config.hidden_size});
    weights.store->upload();
    return weights;
}

modules::QwenDecoderLayerWeights qwen_layer_weights(const VoiceGenBackboneLayerWeights & weights) {
    modules::QwenDecoderLayerWeights out;
    out.input_norm = {weights.input_norm, std::nullopt};
    out.self_attention.q_weight = weights.q_proj;
    out.self_attention.k_weight = weights.k_proj;
    out.self_attention.v_weight = weights.v_proj;
    out.self_attention.out_weight = weights.o_proj;
    out.q_norm = {weights.q_norm, std::nullopt};
    out.k_norm = {weights.k_norm, std::nullopt};
    out.post_norm = {weights.post_norm, std::nullopt};
    out.mlp.gate_proj = {weights.gate_proj, std::nullopt};
    out.mlp.up_proj = {weights.up_proj, std::nullopt};
    out.mlp.down_proj = {weights.down_proj, std::nullopt};
    return out;
}

modules::QwenDecoderLayerConfig qwen_layer_config(
    const MossTtsDelayBackboneConfig & config,
    ggml_type cache_type) {
    modules::QwenDecoderLayerConfig out;
    out.hidden_size = config.hidden_size;
    out.num_attention_heads = config.num_attention_heads;
    out.num_key_value_heads = config.num_key_value_heads;
    out.head_dim = config.head_dim;
    out.intermediate_size = config.intermediate_size;
    out.rms_norm_eps = config.rms_norm_eps;
    out.rope_theta = config.rope_theta;
    out.attention_precision = GGML_PREC_F32;
    out.use_qk_norm = true;
    out.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::ManualRepeat;
    out.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGrouped;
    out.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    if (cache_type != GGML_TYPE_F32) {
        out.runtime.static_cache.set_rows_mode =
            modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;
    }
    return out;
}

}  // namespace

struct MossTtsDelayBackboneRuntime::Impl {
    MossTtsDelayConfig config;
    // Kept alive because the weight store borrows from it.
    std::shared_ptr<const assets::TensorSource> weights_source;
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    int threads = 1;
    size_t graph_arena_bytes = 0;
    ggml_type cache_type = GGML_TYPE_F32;
    MossTtsDelayBackboneWeights weights;

    // Cached-generation step graph (built once by begin_generation, reused every step).
    std::unique_ptr<ggml_context, GgmlContextDeleter> step_ctx;
    ggml_cgraph * step_graph = nullptr;
    ggml_backend_buffer_t step_buffer = nullptr;
    ggml_tensor * step_token = nullptr;
    ggml_tensor * step_bias = nullptr;
    ggml_tensor * step_positions = nullptr;
    ggml_tensor * step_cache_slot = nullptr;
    ggml_tensor * step_mask = nullptr;
    ggml_tensor * step_hidden = nullptr;
    runtime::TransformerKVCache step_cache;
    std::vector<ggml_fp16_t> mask_host;
    double step_graph_build_ms = 0.0;
    double step_input_upload_ms = 0.0;
    double step_mask_upload_ms = 0.0;
    double step_graph_compute_ms = 0.0;
    double step_output_read_ms = 0.0;
    int64_t step_calls = 0;
    double prefill_graph_build_ms = 0.0;
    size_t prefill_ctx_reserved_bytes = 0;
    size_t prefill_ctx_used_bytes = 0;
    double prefill_input_upload_ms = 0.0;
    double prefill_graph_compute_ms = 0.0;
    double prefill_output_read_ms = 0.0;
    int64_t prefill_calls = 0;

    int64_t release_step_graph() {
        const int64_t released_steps = step_cache.cache_steps();
        if (step_graph != nullptr && backend != nullptr) {
            core::release_backend_graph_resources(backend, step_graph);
            step_graph = nullptr;
        }
        if (step_buffer != nullptr) {
            ggml_backend_buffer_free(step_buffer);
            step_buffer = nullptr;
        }
        step_ctx.reset();
        step_token = nullptr;
        step_bias = nullptr;
        step_positions = nullptr;
        step_cache_slot = nullptr;
        step_mask = nullptr;
        step_hidden = nullptr;
        step_cache = runtime::TransformerKVCache();
        std::vector<ggml_fp16_t>().swap(mask_host);
        return released_steps;
    }

    ~Impl() {
        release_step_graph();
    }
};

MossTtsDelayBackboneRuntime::MossTtsDelayBackboneRuntime(
    MossTtsDelayConfig config,
    std::shared_ptr<const assets::TensorSource> weights,
    core::ExecutionContext & execution_context,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type,
    ggml_type cache_type)
    : impl_(std::make_unique<Impl>()) {
    if (weights == nullptr) {
        throw std::runtime_error("MOSS delay backbone requires model weights");
    }
    impl_->backend = execution_context.backend();
    if (impl_->backend == nullptr) {
        throw std::runtime_error("MOSS delay backbone backend is not initialized");
    }
    impl_->backend_type = execution_context.backend_type();
    impl_->threads = execution_context.config().threads;
    impl_->graph_arena_bytes = graph_arena_bytes;
    impl_->cache_type = cache_type;
    impl_->weights = load_backbone_weights(
        config,
        *weights,
        impl_->backend,
        impl_->backend_type,
        weight_context_bytes,
        weight_storage_type);
    impl_->config = std::move(config);
    impl_->weights_source = std::move(weights);
}

MossTtsDelayBackboneRuntime::~MossTtsDelayBackboneRuntime() = default;

int64_t MossTtsDelayBackboneRuntime::hidden_size() const noexcept {
    return impl_->config.backbone.hidden_size;
}

void MossTtsDelayBackboneRuntime::build_step_graph(int64_t cache_steps) const {
    auto & impl = *impl_;
    const auto graph_build_start = Clock::now();
    const auto & config = impl.config.backbone;
    const auto & weights = impl.weights;
    const int64_t dim = config.head_dim;

    ggml_init_params params{impl.graph_arena_bytes, nullptr, true};
    impl.step_ctx.reset(ggml_init(params));
    if (impl.step_ctx == nullptr) {
        throw std::runtime_error("failed to initialize MOSS delay backbone step graph context");
    }
    ggml_context * gctx = impl.step_ctx.get();
    core::ModuleBuildContext ctx{gctx, "moss_tts_delay.backbone.step", impl.backend_type};

    auto token_input = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, 1}));
    ggml_set_input(token_input.tensor);
    auto bias_input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, config.hidden_size}));
    ggml_set_input(bias_input.tensor);
    auto positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
    ggml_set_input(positions.tensor);
    auto cache_slot = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
    ggml_set_input(cache_slot.tensor);
    auto attention_mask =
        core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, 1, cache_steps}));
    ggml_set_input(attention_mask.tensor);

    std::vector<core::TensorValue> cache_keys;
    std::vector<core::TensorValue> cache_values;
    cache_keys.reserve(static_cast<size_t>(config.num_hidden_layers));
    cache_values.reserve(static_cast<size_t>(config.num_hidden_layers));

    impl.step_graph = ggml_new_graph_custom(gctx, 65536, false);
    const modules::QwenDecoderLayerModule layer_module(qwen_layer_config(config, impl.cache_type));

    auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                 .build(ctx, token_input, weights.embed_tokens);
    x = modules::AddModule{}.build(ctx, x, bias_input);
    for (const auto & layer : weights.layers) {
        auto cache_key = core::make_tensor(
            ctx, impl.cache_type,
            core::TensorShape::from_dims({1, cache_steps, config.num_key_value_heads, dim}));
        auto cache_value = core::make_tensor(
            ctx, impl.cache_type,
            core::TensorShape::from_dims({1, cache_steps, config.num_key_value_heads, dim}));
        cache_keys.push_back(cache_key);
        cache_values.push_back(cache_value);
        auto out = layer_module.build_with_static_cache_tail(
            ctx,
            impl.step_graph,
            x,
            positions,
            qwen_layer_weights(layer),
            cache_key,
            cache_value,
            cache_slot,
            attention_mask);
        x = out.output;
    }
    x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
            .build(ctx, x, binding::norm_data(ctx, weights.norm));
    x = core::ensure_backend_addressable_layout(ctx, x);
    auto hidden = modules::ReshapeModule({
        core::TensorShape::from_dims({1, config.hidden_size}),
    }).build(ctx, x);
    hidden = core::ensure_backend_addressable_layout(ctx, hidden);
    ggml_set_output(hidden.tensor);

    ggml_build_forward_expand(impl.step_graph, hidden.tensor);

    impl.step_buffer = ggml_backend_alloc_ctx_tensors(gctx, impl.backend);
    if (impl.step_buffer == nullptr) {
        throw std::runtime_error("failed to allocate MOSS delay backbone step graph");
    }

    impl.step_token = token_input.tensor;
    impl.step_bias = bias_input.tensor;
    impl.step_positions = positions.tensor;
    impl.step_cache_slot = cache_slot.tensor;
    impl.step_mask = attention_mask.tensor;
    impl.step_hidden = hidden.tensor;
    runtime::TransformerKVCacheOptions cache_options;
    cache_options.allow_f16_storage = impl.cache_type == GGML_TYPE_F16;
    cache_options.allow_bf16_storage = impl.cache_type == GGML_TYPE_BF16;
    cache_options.lazy_import_scratch = true;
    impl.step_cache = runtime::TransformerKVCache(
        cache_steps,
        config.num_key_value_heads * config.head_dim,
        std::move(cache_keys),
        std::move(cache_values),
        cache_options);
    impl.mask_host.assign(static_cast<size_t>(cache_steps), ggml_fp32_to_fp16(kMaskedAttentionBias));
    impl.step_graph_build_ms += engine::debug::elapsed_ms(graph_build_start);
}

void MossTtsDelayBackboneRuntime::begin_generation(int64_t max_positions) const {
    if (max_positions <= 0) {
        throw std::runtime_error("MOSS delay backbone begin_generation requires max_positions > 0");
    }
    auto & impl = *impl_;
    if (impl.step_graph == nullptr || impl.step_cache.cache_steps() < max_positions) {
        impl.release_step_graph();
        build_step_graph(max_positions);
    }
    // Zero not-yet-written rows so masked cache entries can never inject NaNs.
    impl.step_cache.clear_on_backend();
}

std::vector<float> MossTtsDelayBackboneRuntime::step(int32_t token_id, const std::vector<float> & audio_bias_row) const {
    std::vector<float> hidden_state;
    step_into(token_id, audio_bias_row, hidden_state);
    return hidden_state;
}

void MossTtsDelayBackboneRuntime::step_into(
    int32_t token_id,
    const std::vector<float> & audio_bias_row,
    std::vector<float> & hidden_state) const {
    auto & impl = *impl_;
    if (impl.step_graph == nullptr) {
        throw std::runtime_error("MOSS delay backbone step called before begin_generation");
    }
    const int64_t hidden = impl.config.backbone.hidden_size;
    if (static_cast<int64_t>(audio_bias_row.size()) != hidden) {
        throw std::runtime_error("MOSS delay backbone step audio bias row size does not match hidden_size");
    }
    if (impl.step_cache.valid_steps() >= impl.step_cache.cache_steps()) {
        throw std::runtime_error("MOSS delay backbone step exceeds cache capacity");
    }
    const int32_t position = static_cast<int32_t>(impl.step_cache.current_end());
    const int32_t cache_slot = static_cast<int32_t>(impl.step_cache.valid_steps());
    auto timing_start = Clock::now();
    ggml_backend_tensor_set(impl.step_token, &token_id, 0, sizeof(int32_t));
    ggml_backend_tensor_set(impl.step_bias, audio_bias_row.data(), 0, audio_bias_row.size() * sizeof(float));
    ggml_backend_tensor_set(impl.step_positions, &position, 0, sizeof(int32_t));
    ggml_backend_tensor_set(impl.step_cache_slot, &cache_slot, 0, sizeof(int32_t));
    impl.step_input_upload_ms += engine::debug::elapsed_ms(timing_start);
#ifdef _OPENMP
#pragma omp parallel for if(impl.step_cache.cache_steps() >= 4096)
#endif
    for (int64_t i = 0; i < impl.step_cache.cache_steps(); ++i) {
        impl.mask_host[static_cast<size_t>(i)] =
            ggml_fp32_to_fp16((i <= cache_slot) ? 0.0F : kMaskedAttentionBias);
    }
    timing_start = Clock::now();
    ggml_backend_tensor_set(impl.step_mask, impl.mask_host.data(), 0, impl.mask_host.size() * sizeof(ggml_fp16_t));
    impl.step_mask_upload_ms += engine::debug::elapsed_ms(timing_start);

    timing_start = Clock::now();
    core::set_backend_threads(impl.backend, impl.threads);
    const ggml_status status = ggml_backend_graph_compute(impl.backend, impl.step_graph);
    ggml_backend_synchronize(impl.backend);
    impl.step_graph_compute_ms += engine::debug::elapsed_ms(timing_start);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("MOSS delay backbone step graph compute failed");
    }
    hidden_state.resize(static_cast<size_t>(hidden));
    timing_start = Clock::now();
    ggml_backend_tensor_get(impl.step_hidden, hidden_state.data(), 0, hidden_state.size() * sizeof(float));
    impl.step_output_read_ms += engine::debug::elapsed_ms(timing_start);
    impl.step_cache.advance_after_direct_append(1);
    ++impl.step_calls;
}

std::vector<float> MossTtsDelayBackboneRuntime::prefill(
    const std::vector<int32_t> & token_ids,
    const std::vector<float> & audio_bias) const {
    auto & impl = *impl_;
    if (impl.step_graph == nullptr) {
        throw std::runtime_error("MOSS delay backbone prefill called before begin_generation");
    }
    const auto & config = impl.config.backbone;
    const int64_t steps = static_cast<int64_t>(token_ids.size());
    if (steps <= 0) {
        throw std::runtime_error("MOSS delay backbone prefill requires a non-empty prompt");
    }
    if (steps > impl.step_cache.cache_steps()) {
        throw std::runtime_error("MOSS delay backbone prefill prompt exceeds cache capacity");
    }
    if (static_cast<int64_t>(audio_bias.size()) != steps * config.hidden_size) {
        throw std::runtime_error("MOSS delay backbone prefill audio bias size does not match [steps, hidden]");
    }
    const int64_t dim = config.head_dim;
    const int64_t kv_heads = config.num_key_value_heads;

    auto timing_start = Clock::now();
    ggml_init_params params{impl.graph_arena_bytes, nullptr, true};
    std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx(ggml_init(params));
    if (graph_ctx == nullptr) {
        throw std::runtime_error("failed to initialize MOSS delay backbone prefill context");
    }
    core::ModuleBuildContext ctx{graph_ctx.get(), "moss_tts_delay.backbone.prefill", impl.backend_type};

    auto token_input = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, steps}));
    ggml_set_input(token_input.tensor);
    auto positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
    ggml_set_input(positions.tensor);
    auto attention_mask =
        core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({1, 1, steps, steps}));
    ggml_set_input(attention_mask.tensor);
    auto bias_input =
        core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, steps, config.hidden_size}));
    ggml_set_input(bias_input.tensor);

    auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                 .build(ctx, token_input, impl.weights.embed_tokens);
    x = modules::AddModule{}.build(ctx, x, bias_input);
    std::vector<core::TensorValue> layer_keys;
    std::vector<core::TensorValue> layer_values;
    layer_keys.reserve(impl.weights.layers.size());
    layer_values.reserve(impl.weights.layers.size());
    const modules::QwenDecoderLayerModule layer_module(qwen_layer_config(config, impl.cache_type));
    for (const auto & layer : impl.weights.layers) {
        auto out = layer_module.build(
            ctx,
            x,
            positions,
            qwen_layer_weights(layer),
            std::nullopt,
            std::nullopt,
            attention_mask);
        x = out.output;
        if (!out.key.valid() || !out.value.valid()) {
            throw std::runtime_error("MOSS delay backbone prefill decoder did not return K/V state");
        }
        layer_keys.push_back(out.key);
        layer_values.push_back(out.value);
    }
    x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
            .build(ctx, x, binding::norm_data(ctx, impl.weights.norm));
    x = core::ensure_backend_addressable_layout(ctx, x);
    auto hidden = modules::ReshapeModule({
        core::TensorShape::from_dims({steps, config.hidden_size}),
    }).build(ctx, x);
    hidden = core::ensure_backend_addressable_layout(ctx, hidden);
    ggml_set_output(hidden.tensor);

    // Copy each layer's K/V straight into the generation cache rows [0, steps) as part of the
    // graph. Doing it on-device (instead of reading the intermediates back to the host) is both
    // faster and immune to the graph allocator reusing those tensors' storage after compute.
    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx.get(), 65536, false);
    ggml_build_forward_expand(graph, hidden.tensor);
    for (size_t layer = 0; layer < impl.weights.layers.size(); ++layer) {
        ggml_tensor * cache_key = impl.step_cache.key_tensor(layer).tensor;
        ggml_tensor * cache_value = impl.step_cache.value_tensor(layer).tensor;
        ggml_tensor * key_view = ggml_view_4d(
            graph_ctx.get(), cache_key, dim, kv_heads, steps, 1,
            cache_key->nb[1], cache_key->nb[2], cache_key->nb[3], 0);
        ggml_tensor * value_view = ggml_view_4d(
            graph_ctx.get(), cache_value, dim, kv_heads, steps, 1,
            cache_value->nb[1], cache_value->nb[2], cache_value->nb[3], 0);
        ggml_build_forward_expand(graph, ggml_cpy(graph_ctx.get(), layer_keys[layer].tensor, key_view));
        ggml_build_forward_expand(graph, ggml_cpy(graph_ctx.get(), layer_values[layer].tensor, value_view));
    }

    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl.backend));
    if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
        throw std::runtime_error("failed to allocate MOSS delay backbone prefill graph");
    }
    impl.prefill_graph_build_ms += engine::debug::elapsed_ms(timing_start);
    impl.prefill_ctx_reserved_bytes = ggml_get_mem_size(graph_ctx.get());
    impl.prefill_ctx_used_bytes = ggml_used_mem(graph_ctx.get());

    timing_start = Clock::now();
    ggml_backend_tensor_set(token_input.tensor, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
    std::vector<int32_t> position_host(static_cast<size_t>(steps));
#ifdef _OPENMP
#pragma omp parallel for if(steps >= 4096)
#endif
    for (int64_t i = 0; i < steps; ++i) {
        position_host[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    ggml_backend_tensor_set(positions.tensor, position_host.data(), 0, position_host.size() * sizeof(int32_t));
    std::vector<ggml_fp16_t> mask_host(
        static_cast<size_t>(steps * steps),
        ggml_fp32_to_fp16(kMaskedAttentionBias));
#ifdef _OPENMP
#pragma omp parallel for if(steps * steps >= 4096)
#endif
    for (int64_t q = 0; q < steps; ++q) {
        for (int64_t k = 0; k <= q; ++k) {
            mask_host[static_cast<size_t>(q * steps + k)] = ggml_fp32_to_fp16(0.0F);
        }
    }
    ggml_backend_tensor_set(attention_mask.tensor, mask_host.data(), 0, mask_host.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(bias_input.tensor, audio_bias.data(), 0, audio_bias.size() * sizeof(float));
    impl.prefill_input_upload_ms += engine::debug::elapsed_ms(timing_start);

    timing_start = Clock::now();
    core::set_backend_threads(impl.backend, impl.threads);
    const ggml_status status = ggml_backend_graph_compute(impl.backend, graph);
    ggml_backend_synchronize(impl.backend);
    impl.prefill_graph_compute_ms += engine::debug::elapsed_ms(timing_start);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(gallocr);
        throw std::runtime_error("MOSS delay backbone prefill graph compute failed");
    }

    std::vector<float> last_hidden(static_cast<size_t>(config.hidden_size));
    timing_start = Clock::now();
    ggml_backend_tensor_get(
        hidden.tensor,
        last_hidden.data(),
        static_cast<size_t>((steps - 1) * config.hidden_size) * sizeof(float),
        last_hidden.size() * sizeof(float));
    impl.prefill_output_read_ms += engine::debug::elapsed_ms(timing_start);
    ggml_gallocr_free(gallocr);
    impl.step_cache.advance_after_direct_append(steps);
    ++impl.prefill_calls;
    return last_hidden;
}

int64_t MossTtsDelayBackboneRuntime::cached_positions() const noexcept {
    return impl_->step_cache.valid_steps();
}

int64_t MossTtsDelayBackboneRuntime::release_cached_step_graph() const {
    return impl_->release_step_graph();
}

void MossTtsDelayBackboneRuntime::reset_timing() const {
    auto & impl = *impl_;
    impl.step_graph_build_ms = 0.0;
    impl.step_input_upload_ms = 0.0;
    impl.step_mask_upload_ms = 0.0;
    impl.step_graph_compute_ms = 0.0;
    impl.step_output_read_ms = 0.0;
    impl.step_calls = 0;
    impl.prefill_graph_build_ms = 0.0;
    impl.prefill_input_upload_ms = 0.0;
    impl.prefill_graph_compute_ms = 0.0;
    impl.prefill_output_read_ms = 0.0;
    impl.prefill_calls = 0;
}

void MossTtsDelayBackboneRuntime::log_timing() const {
    const auto & impl = *impl_;
    engine::debug::timing_log_scalar("moss_tts_delay.backbone.step.graph.build_ms", impl.step_graph_build_ms);
    engine::debug::timing_log_context_reservation("moss_tts_delay.backbone.step.graph", impl.step_ctx.get());
    engine::debug::timing_log_scalar("moss_tts_delay.backbone.step.input_upload_ms", impl.step_input_upload_ms);
    engine::debug::timing_log_scalar("moss_tts_delay.backbone.step.mask_upload_ms", impl.step_mask_upload_ms);
    engine::debug::timing_log_scalar("moss_tts_delay.backbone.step.graph.compute_ms", impl.step_graph_compute_ms);
    engine::debug::timing_log_scalar("moss_tts_delay.backbone.step.output_read_ms", impl.step_output_read_ms);
    engine::debug::trace_log_scalar("moss_tts_delay.backbone.step.calls", impl.step_calls);
    engine::debug::timing_log_scalar("moss_tts_delay.backbone.prefill.graph.build_ms", impl.prefill_graph_build_ms);
    if (impl.prefill_ctx_reserved_bytes > 0) {
        constexpr double kMiB = 1024.0 * 1024.0;
        engine::debug::timing_log_scalar(
            "moss_tts_delay.backbone.prefill.graph.ctx_reserved_mb",
            static_cast<double>(impl.prefill_ctx_reserved_bytes) / kMiB);
        engine::debug::timing_log_scalar(
            "moss_tts_delay.backbone.prefill.graph.ctx_used_mb",
            static_cast<double>(impl.prefill_ctx_used_bytes) / kMiB);
    }
    engine::debug::timing_log_scalar("moss_tts_delay.backbone.prefill.input_upload_ms", impl.prefill_input_upload_ms);
    engine::debug::timing_log_scalar("moss_tts_delay.backbone.prefill.graph.compute_ms", impl.prefill_graph_compute_ms);
    engine::debug::timing_log_scalar("moss_tts_delay.backbone.prefill.output_read_ms", impl.prefill_output_read_ms);
    engine::debug::trace_log_scalar("moss_tts_delay.backbone.prefill.calls", impl.prefill_calls);
}

}  // namespace engine::decoders
