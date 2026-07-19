#include "engine/models/higgs_tts/ar.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/attention/qwen_causal_decoder.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_projection_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::higgs_tts {
namespace {

namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;
constexpr int64_t kLayerwisePrefillMinSteps = 2048;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

modules::QwenDecoderStackConfig make_higgs_qwen_stack_config(const HiggsTextConfig & config) {
    modules::QwenDecoderStackConfig out;
    out.hidden_size = config.hidden_size;
    out.num_attention_heads = config.num_attention_heads;
    out.num_key_value_heads = config.num_key_value_heads;
    out.head_dim = config.head_dim;
    out.intermediate_size = config.intermediate_size;
    out.layers = config.num_hidden_layers;
    out.rms_norm_eps = config.rms_norm_eps;
    out.rope_theta = config.rope_theta;
    out.attention_precision = GGML_PREC_F32;
    out.projection_precision = GGML_PREC_DEFAULT;
    out.qkv_layout = modules::QwenDecoderQKVLayout::Separate;
    out.use_qk_norm = true;
    out.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    out.runtime.static_cache.transpose_context = false;
    return out;
}

class HiggsQwenDecoderComponent {
public:
    HiggsQwenDecoderComponent(const HiggsTextConfig & config, bool packed_qkv)
        : stack_config_(make_higgs_qwen_stack_config(config)),
          layer_config_(modules::qwen_decoder_layer_config_from_stack(stack_config_)),
          layer_module_([&] {
              layer_config_.qkv_layout = packed_qkv
                  ? modules::QwenDecoderQKVLayout::PackedQKV
                  : modules::QwenDecoderQKVLayout::Separate;
              return layer_config_;
          }()) {}

    modules::QwenDecoderLayerOutputs build_prefill_layer(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const modules::QwenDecoderLayerWeights & weights,
        const core::TensorValue & attention_mask,
        const std::optional<core::TensorValue> & prefix_key = std::nullopt,
        const std::optional<core::TensorValue> & prefix_value = std::nullopt) const {
        return layer_module_.build(ctx, input, positions, weights, prefix_key, prefix_value, attention_mask);
    }

    modules::QwenDecoderLayerOutputs build_decode_layer(
        core::ModuleBuildContext & ctx,
        ggml_cgraph * graph,
        const core::TensorValue & input,
        const core::TensorValue & positions,
        const modules::QwenDecoderLayerWeights & weights,
        const core::TensorValue & cache_key,
        const core::TensorValue & cache_value,
        const core::TensorValue & cache_slot,
        const core::TensorValue & attention_mask) const {
        return layer_module_.build_with_static_cache_tail(
            ctx,
            graph,
            input,
            positions,
            weights,
            cache_key,
            cache_value,
            cache_slot,
            attention_mask);
    }

private:
    modules::QwenDecoderStackConfig stack_config_;
    modules::QwenDecoderLayerConfig layer_config_;
    modules::QwenDecoderLayerModule layer_module_;
};

core::TensorValue higgs_cache_view(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    int64_t start,
    int64_t steps,
    int64_t heads,
    int64_t head_dim) {
    if (start < 0 || steps <= 0 || start + steps > cache.shape.dims[1]) {
        throw std::runtime_error("Higgs TTS AR cache view range is invalid");
    }
    return core::wrap_tensor(
        ggml_view_4d(
            ctx.ggml,
            cache.tensor,
            head_dim,
            heads,
            steps,
            1,
            cache.tensor->nb[1],
            cache.tensor->nb[2],
            cache.tensor->nb[3],
            static_cast<size_t>(start) * cache.tensor->nb[2]),
        core::TensorShape::from_dims({1, steps, heads, head_dim}),
        cache.type);
}

modules::QwenDecoderLayerWeights load_layer_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const HiggsTextConfig & config,
    int64_t layer_index,
    assets::TensorStorageType storage_type) {
    const std::string prefix = "body.layers." + std::to_string(layer_index);
    const int64_t q_out = config.num_attention_heads * config.head_dim;
    const int64_t kv_out = config.num_key_value_heads * config.head_dim;
    modules::QwenDecoderLayerWeights weights;
    weights.input_norm = {
        store.load_f32_tensor(source, prefix + ".input_layernorm.weight", {config.hidden_size}),
        std::nullopt,
    };
    {
        const auto q = source.require_tensor(
            prefix + ".self_attn.q_proj.weight",
            storage_type,
            {q_out, config.hidden_size});
        const auto k = source.require_tensor(
            prefix + ".self_attn.k_proj.weight",
            storage_type,
            {kv_out, config.hidden_size});
        const auto v = source.require_tensor(
            prefix + ".self_attn.v_proj.weight",
            storage_type,
            {kv_out, config.hidden_size});
        if (q.type != k.type || q.type != v.type) {
            throw std::runtime_error("Higgs TTS packed QKV weights require matching storage types");
        }
        std::vector<std::byte> packed;
        packed.reserve(q.bytes.size() + k.bytes.size() + v.bytes.size());
        packed.insert(packed.end(), q.bytes.begin(), q.bytes.end());
        packed.insert(packed.end(), k.bytes.begin(), k.bytes.end());
        packed.insert(packed.end(), v.bytes.begin(), v.bytes.end());
        weights.self_attention.qkv_weight = store.make_tensor(
            core::TensorShape::from_dims({q_out + 2 * kv_out, config.hidden_size}),
            q.type,
            packed.data(),
            packed.size());
    }
    weights.self_attention.out_weight = store.load_tensor(
        source,
        prefix + ".self_attn.o_proj.weight",
        storage_type,
        {config.hidden_size, q_out});
    weights.q_norm = {
        store.load_f32_tensor(source, prefix + ".self_attn.q_norm.weight", {config.head_dim}),
        std::nullopt,
    };
    weights.k_norm = {
        store.load_f32_tensor(source, prefix + ".self_attn.k_norm.weight", {config.head_dim}),
        std::nullopt,
    };
    weights.post_norm = {
        store.load_f32_tensor(source, prefix + ".post_attention_layernorm.weight", {config.hidden_size}),
        std::nullopt,
    };
    {
        const auto gate = source.require_tensor(
            prefix + ".mlp.gate_proj.weight",
            storage_type,
            {config.intermediate_size, config.hidden_size});
        const auto up = source.require_tensor(
            prefix + ".mlp.up_proj.weight",
            storage_type,
            {config.intermediate_size, config.hidden_size});
        if (gate.type != up.type) {
            throw std::runtime_error("Higgs TTS packed gate/up weights require matching storage types");
        }
        std::vector<std::byte> packed;
        packed.reserve(gate.bytes.size() + up.bytes.size());
        packed.insert(packed.end(), gate.bytes.begin(), gate.bytes.end());
        packed.insert(packed.end(), up.bytes.begin(), up.bytes.end());
        weights.mlp.gate_up_proj = modules::LinearWeights{
            store.make_tensor(
                core::TensorShape::from_dims({config.intermediate_size * 2, config.hidden_size}),
                gate.type,
                packed.data(),
                packed.size()),
            std::nullopt,
        };
    }
    weights.mlp.down_proj = {
        store.load_tensor(
            source,
            prefix + ".mlp.down_proj.weight",
            storage_type,
            {config.hidden_size, config.intermediate_size}),
        std::nullopt,
    };
    return weights;
}

HiggsQwenDecoderStackWeights load_decoder_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const HiggsTextConfig & config,
    assets::TensorStorageType storage_type) {
    HiggsQwenDecoderStackWeights weights;
    weights.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        weights.layers.push_back(load_layer_weights(store, source, config, layer, storage_type));
    }
    return weights;
}

core::TensorValue build_higgs_decode_code_embedding(
    core::ModuleBuildContext & ctx,
    const HiggsARWeights & weights,
    const HiggsConfig & config,
    ggml_tensor * fused_code_ids) {
    auto code_ids = core::wrap_tensor(
        fused_code_ids,
        core::TensorShape::from_dims({config.audio.num_codebooks}),
        GGML_TYPE_I32);
    auto code = modules::EmbeddingModule(
                    {config.audio.num_codebooks * config.audio.vocab_size, config.text.hidden_size})
                    .build(ctx, code_ids, weights.modality_embedding);
    code = modules::ReduceSumModule({0}).build(ctx, code);
    return core::reshape_tensor(ctx, code, core::TensorShape::from_dims({1, 1, config.text.hidden_size}));
}

core::TensorValue build_higgs_prefill_input_embedding(
    core::ModuleBuildContext & ctx,
    const HiggsARWeights & weights,
    const HiggsConfig & config,
    ggml_tensor * text_tokens,
    ggml_tensor * fused_code_ids,
    ggml_tensor * text_gate,
    ggml_tensor * code_gate,
    int64_t steps) {
    auto text_ids = core::wrap_tensor(text_tokens, core::TensorShape::from_dims({steps}), GGML_TYPE_I32);
    auto text = modules::EmbeddingModule({config.text.vocab_size, config.text.hidden_size})
                    .build(ctx, text_ids, weights.text_embedding);
    text = core::reshape_tensor(ctx, text, core::TensorShape::from_dims({1, steps, config.text.hidden_size}));

    auto code_ids = core::wrap_tensor(
        fused_code_ids,
        core::TensorShape::from_dims({steps, config.audio.num_codebooks}),
        GGML_TYPE_I32);
    auto code = modules::EmbeddingModule(
                    {config.audio.num_codebooks * config.audio.vocab_size, config.text.hidden_size})
                    .build(ctx, code_ids, weights.modality_embedding);
    code = modules::ReduceSumModule({1}).build(ctx, code);
    code = core::reshape_tensor(ctx, code, core::TensorShape::from_dims({1, steps, config.text.hidden_size}));

    auto text_gate_value = core::wrap_tensor(
        text_gate,
        core::TensorShape::from_dims({1, steps, 1}),
        GGML_TYPE_F32);
    auto code_gate_value = core::wrap_tensor(
        code_gate,
        core::TensorShape::from_dims({1, steps, 1}),
        GGML_TYPE_F32);
    text_gate_value = core::wrap_tensor(
        ggml_repeat(ctx.ggml, text_gate_value.tensor, text.tensor), text.shape, GGML_TYPE_F32);
    code_gate_value = core::wrap_tensor(
        ggml_repeat(ctx.ggml, code_gate_value.tensor, code.tensor), code.shape, GGML_TYPE_F32);
    return modules::AddModule{}.build(
        ctx,
        modules::MulModule{}.build(ctx, text, text_gate_value),
        modules::MulModule{}.build(ctx, code, code_gate_value));
}

core::TensorValue build_modality_logits(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden,
    const HiggsARWeights & weights,
    const HiggsConfig & config) {
    const int64_t out_features = config.audio.num_codebooks * config.audio.vocab_size;
    const bool use_fast_projection =
        ctx.backend_type == core::BackendType::Cuda && hidden.shape.rank == 3 &&
        hidden.shape.dims[1] == 1 && out_features % 4 == 0;
    auto logits =
        use_fast_projection
            ? modules::FastPackedProjection4Module({config.text.hidden_size, out_features, GGML_PREC_DEFAULT})
                  .build(ctx, hidden, {weights.modality_embedding, std::nullopt})
            : modules::LinearModule({config.text.hidden_size, out_features, false})
                  .build(ctx, hidden, {weights.modality_embedding, std::nullopt});
    return core::reshape_tensor(
        ctx,
        logits,
        core::TensorShape::from_dims({config.audio.num_codebooks, config.audio.vocab_size}));
}

}  // namespace

HiggsARWeights load_higgs_ar_weights(
    const HiggsAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type) {
    if (assets.weights == nullptr) {
        throw std::runtime_error("Higgs TTS AR weights require tensor source");
    }
    if (backend == nullptr) {
        throw std::runtime_error("Higgs TTS AR backend is not initialized");
    }
    const auto & config = assets.config;
    const auto & source = *assets.weights;
    HiggsARWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "higgs_tts.ar.weights",
        weight_context_bytes);
    weights.text_embedding = weights.store->load_tensor(
        source,
        "tied.embedding.text_embedding.weight",
        weight_storage_type,
        {config.text.vocab_size, config.text.hidden_size});
    weights.modality_embedding = weights.store->load_tensor(
        source,
        "tied.embedding.modality_embeddings.0.embedding.weight",
        weight_storage_type,
        {config.audio.num_codebooks * config.audio.vocab_size, config.text.hidden_size});
    weights.decoder = load_decoder_weights(*weights.store, source, config.text, weight_storage_type);
    weights.norm = weights.store->load_f32_tensor(source, "body.norm.weight", {config.text.hidden_size});
    weights.packed_qkv = true;
    weights.store->upload();
    return weights;
}

void HiggsARDecodeTiming::add(const HiggsARDecodeTiming & other) noexcept {
    input_upload_ms += other.input_upload_ms;
    mask_upload_ms += other.mask_upload_ms;
    graph_compute_ms += other.graph_compute_ms;
    output_read_ms += other.output_read_ms;
    steps += other.steps;
}

HiggsARRuntime::HiggsARRuntime(
    std::shared_ptr<const HiggsAssets> assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : assets_(std::move(assets)),
      backend_(execution.backend()),
      backend_type_(execution.backend_type()),
      device_(execution.config().device),
      threads_(std::max(1, execution.config().threads)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Higgs TTS AR runtime requires assets");
    }
    if (assets_->weights == nullptr) {
        throw std::runtime_error("Higgs TTS AR runtime requires tensor source");
    }
    weights_ = std::make_shared<HiggsARWeights>(
        load_higgs_ar_weights(*assets_, backend_, backend_type_, weight_context_bytes, weight_storage_type));
}

const HiggsAssets & HiggsARRuntime::assets() const noexcept {
    return *assets_;
}

const HiggsARWeights & HiggsARRuntime::weights() const noexcept {
    return *weights_;
}

ggml_backend_t HiggsARRuntime::backend() const noexcept {
    return backend_;
}

core::BackendType HiggsARRuntime::backend_type() const noexcept {
    return backend_type_;
}

int HiggsARRuntime::device() const noexcept {
    return device_;
}

int HiggsARRuntime::threads() const noexcept {
    return threads_;
}

struct HiggsARKVCache::Impl {
    Impl(std::shared_ptr<HiggsARRuntime> input_runtime, int64_t input_cache_steps)
        : runtime(std::move(input_runtime)),
          cache_steps(input_cache_steps) {
        if (runtime == nullptr) {
            throw std::runtime_error("Higgs TTS AR KV cache requires runtime");
        }
        if (cache_steps <= 0) {
            throw std::runtime_error("Higgs TTS AR KV cache requires positive capacity");
        }
        const auto & config = runtime->assets().config;
        const auto & tensor_weights = runtime->weights();
        const int64_t dim = config.text.head_dim;
        cache_layer_count = tensor_weights.decoder.layers.size();
        ggml_init_params params{4 * 1024 * 1024, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize Higgs TTS AR KV cache context");
        }
        core::ModuleBuildContext build_ctx{ctx.get(), "higgs_tts.ar.kv_cache", runtime->backend_type()};
        std::vector<core::TensorValue> key_tensors;
        std::vector<core::TensorValue> value_tensors;
        key_tensors.reserve(tensor_weights.decoder.layers.size());
        value_tensors.reserve(tensor_weights.decoder.layers.size());
        for (size_t layer = 0; layer < tensor_weights.decoder.layers.size(); ++layer) {
            key_tensors.push_back(core::make_tensor(
                build_ctx,
                GGML_TYPE_F16,
                core::TensorShape::from_dims({1, cache_steps, config.text.num_key_value_heads, dim})));
            value_tensors.push_back(core::make_tensor(
                build_ctx,
                GGML_TYPE_F16,
                core::TensorShape::from_dims({1, cache_steps, config.text.num_key_value_heads, dim})));
        }
        cache = runtime::TransformerKVCache(
            cache_steps,
            config.text.num_key_value_heads * dim,
            std::move(key_tensors),
            std::move(value_tensors));
        buffer = ggml_backend_alloc_ctx_tensors(ctx.get(), runtime->backend());
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate Higgs TTS AR KV cache");
        }
    }

    ~Impl() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }

    bool can_run(const HiggsARRuntime & candidate_runtime, int64_t required_steps) const {
        return runtime.get() == &candidate_runtime && cache_steps >= required_steps;
    }

    void reset() {
        runtime::TransformerKVState state;
        state.current_end = 0;
        state.layers.resize(cache_layer_count);
        cache.import_state(state);
    }

    void retain_prefix(int64_t prefix_steps) {
        cache.retain_prefix(prefix_steps);
    }

    void import_state(const runtime::TransformerKVState & state) {
        cache.import_state(state);
    }

    runtime::TransformerKVState export_state() const {
        return cache.export_state();
    }

    void advance_after_direct_append(int64_t steps) {
        cache.advance_after_direct_append(steps);
    }

    std::shared_ptr<HiggsARRuntime> runtime;
    int64_t cache_steps = 0;
    size_t cache_layer_count = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    runtime::TransformerKVCache cache;
    ggml_backend_buffer_t buffer = nullptr;
};

HiggsARKVCache::HiggsARKVCache(std::shared_ptr<HiggsARRuntime> runtime, int64_t cache_steps)
    : impl_(std::make_unique<Impl>(std::move(runtime), cache_steps)) {}

HiggsARKVCache::~HiggsARKVCache() = default;

bool HiggsARKVCache::can_run(const HiggsARRuntime & runtime, int64_t required_steps) const {
    return impl_->can_run(runtime, required_steps);
}

int64_t HiggsARKVCache::cache_steps() const {
    return impl_->cache.cache_steps();
}

int64_t HiggsARKVCache::valid_steps() const {
    return impl_->cache.valid_steps();
}

int64_t HiggsARKVCache::current_end() const {
    return impl_->cache.current_end();
}

void HiggsARKVCache::reset() {
    impl_->reset();
}

void HiggsARKVCache::retain_prefix(int64_t prefix_steps) {
    impl_->retain_prefix(prefix_steps);
}

void HiggsARKVCache::import_state(const runtime::TransformerKVState & state) {
    impl_->import_state(state);
}

runtime::TransformerKVState HiggsARKVCache::export_state() const {
    return impl_->export_state();
}

void HiggsARKVCache::advance_after_direct_append(int64_t steps) {
    impl_->advance_after_direct_append(steps);
}

const core::TensorValue & HiggsARKVCache::key_tensor(size_t layer) const {
    return impl_->cache.key_tensor(layer);
}

const core::TensorValue & HiggsARKVCache::value_tensor(size_t layer) const {
    return impl_->cache.value_tensor(layer);
}

struct HiggsARDecodeGraph::Impl {
    Impl(
        std::shared_ptr<HiggsARRuntime> input_runtime,
        int64_t input_cache_steps,
        HiggsARKVCache & input_cache,
        size_t graph_arena_bytes)
        : runtime(std::move(input_runtime)),
          cache(&input_cache),
          cache_steps(input_cache_steps) {
        if (runtime == nullptr) {
            throw std::runtime_error("Higgs TTS AR decode graph requires runtime");
        }
        if (cache == nullptr || !cache->can_run(*runtime, cache_steps)) {
            throw std::runtime_error("Higgs TTS AR decode graph requires matching KV cache");
        }
        if (cache_steps <= 0) {
            throw std::runtime_error("Higgs TTS AR decode graph requires positive cache capacity");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize Higgs TTS AR decode graph context");
        }
        const auto & config = runtime->assets().config;
        const auto & tensor_weights = runtime->weights();
        core::ModuleBuildContext build_ctx{ctx.get(), "higgs_tts.ar.decode", runtime->backend_type()};

        fused_code_ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, config.audio.num_codebooks);
        auto x = build_higgs_decode_code_embedding(
            build_ctx,
            tensor_weights,
            config,
            fused_code_ids);

        positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
        auto positions_value = core::wrap_tensor(positions, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        cache_slot = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, 1);
        auto cache_slot_value = core::wrap_tensor(cache_slot, core::TensorShape::from_dims({1}), GGML_TYPE_I64);
        attention_mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, cache_steps, 1, 1, 1);
        auto attention_mask_value = core::wrap_tensor(
            attention_mask,
            core::TensorShape::from_dims({1, 1, 1, cache_steps}),
            GGML_TYPE_F16);

        graph = ggml_new_graph_custom(ctx.get(), 65536, false);
        const HiggsQwenDecoderComponent decoder(config.text, tensor_weights.packed_qkv);
        for (size_t layer_index = 0; layer_index < tensor_weights.decoder.layers.size(); ++layer_index) {
            auto out = decoder.build_decode_layer(
                build_ctx,
                graph,
                x,
                positions_value,
                tensor_weights.decoder.layers[layer_index],
                cache->key_tensor(layer_index),
                cache->value_tensor(layer_index),
                cache_slot_value,
                attention_mask_value);
            x = out.output;
        }

        x = modules::RMSNormModule({config.text.hidden_size, config.text.rms_norm_eps, true, false})
                .build(build_ctx, x, {tensor_weights.norm, std::nullopt});
        auto logits = build_modality_logits(build_ctx, x, tensor_weights, config);
        logits_output = logits.tensor;
        ggml_set_output(logits_output);
        ggml_build_forward_expand(graph, logits_output);

        buffer = ggml_backend_alloc_ctx_tensors(ctx.get(), runtime->backend());
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate Higgs TTS AR decode graph");
        }
        core::set_backend_threads(runtime->backend(), runtime->threads());
        fused_code_ids_values.assign(static_cast<size_t>(config.audio.num_codebooks), 0);
        attention_mask_values.assign(static_cast<size_t>(cache_steps), ggml_fp32_to_fp16(-INFINITY));
        engine::debug::timing_log_scalar(
            "higgs_tts.ar.decode.graph.build_ms",
            engine::debug::elapsed_ms(build_start, Clock::now()));
    }

    ~Impl() {
        engine::core::release_backend_graph_resources(runtime->backend(), graph);
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }

    bool can_run(const HiggsARRuntime & candidate_runtime, int64_t required_steps) const {
        return runtime.get() == &candidate_runtime && cache != nullptr && cache->can_run(candidate_runtime, required_steps);
    }

    int64_t cache_steps_value() const {
        return cache_steps;
    }

    void import_prefill_state(const runtime::TransformerKVState & state) {
        cache->import_state(state);
    }

    void reset_timing() noexcept {
        input_upload_ms = 0.0;
        mask_upload_ms = 0.0;
        graph_compute_ms = 0.0;
        output_read_ms = 0.0;
        steps = 0;
    }

    void begin_decode_run() {
        reset_timing();
        std::fill(attention_mask_values.begin(), attention_mask_values.end(), ggml_fp32_to_fp16(-INFINITY));
        const int64_t visible_steps = std::min(cache->valid_steps(), cache_steps);
        std::fill(
            attention_mask_values.begin(),
            attention_mask_values.begin() + static_cast<std::ptrdiff_t>(visible_steps),
            ggml_fp32_to_fp16(0.0F));
        ggml_backend_tensor_set(
            attention_mask,
            attention_mask_values.data(),
            0,
            attention_mask_values.size() * sizeof(ggml_fp16_t));
    }

    HiggsARDecodeTiming timing() const {
        return {input_upload_ms, mask_upload_ms, graph_compute_ms, output_read_ms, steps};
    }

    void run_step_into(const HiggsARDecodeInput & input, HiggsARDecodeOutput & output, bool log_timing) {
        const auto & config = runtime->assets().config;
        if (cache->valid_steps() >= cache_steps) {
            throw std::runtime_error("Higgs TTS AR decode cache exhausted");
        }
        if (!input.use_last_codes) {
            throw std::runtime_error("Higgs TTS AR decode graph expects code-token steps after prefill");
        }
        if (static_cast<int64_t>(input.last_codes.size()) != config.audio.num_codebooks) {
            throw std::runtime_error("Higgs TTS AR decode last codebook row shape mismatch");
        }

        auto timing_start = Clock::now();
        for (int64_t codebook = 0; codebook < config.audio.num_codebooks; ++codebook) {
            const int32_t code = input.last_codes[static_cast<size_t>(codebook)];
            if (code < 0 || code >= config.audio.vocab_size) {
                throw std::runtime_error("Higgs TTS AR decode codebook token is outside vocabulary");
            }
            fused_code_ids_values[static_cast<size_t>(codebook)] =
                static_cast<int32_t>(code + codebook * config.audio.vocab_size);
        }
        ggml_backend_tensor_set(
            fused_code_ids,
            fused_code_ids_values.data(),
            0,
            fused_code_ids_values.size() * sizeof(int32_t));

        const int32_t position = static_cast<int32_t>(cache->current_end());
        ggml_backend_tensor_set(positions, &position, 0, sizeof(int32_t));
        const int64_t cache_slot_value = cache->valid_steps();
        ggml_backend_tensor_set(cache_slot, &cache_slot_value, 0, sizeof(int64_t));
        const double input_upload_delta_ms = engine::debug::elapsed_ms(timing_start, Clock::now());
        input_upload_ms += input_upload_delta_ms;
        if (log_timing) {
            engine::debug::timing_log_scalar("higgs_tts.ar.decode.step0.input_upload_ms", input_upload_delta_ms);
        }
        timing_start = Clock::now();
        attention_mask_values[static_cast<size_t>(cache_slot_value)] = ggml_fp32_to_fp16(0.0F);
        ggml_backend_tensor_set(
            attention_mask,
            attention_mask_values.data() + cache_slot_value,
            static_cast<size_t>(cache_slot_value) * sizeof(ggml_fp16_t),
            sizeof(ggml_fp16_t));
        const double mask_upload_delta_ms = engine::debug::elapsed_ms(timing_start, Clock::now());
        mask_upload_ms += mask_upload_delta_ms;
        if (log_timing) {
            engine::debug::timing_log_scalar("higgs_tts.ar.decode.step0.mask_upload_ms", mask_upload_delta_ms);
        }

        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime->backend(), graph);
        if (log_timing || engine::debug::timing_log_enabled()) {
            ggml_backend_synchronize(runtime->backend());
        }
        const double graph_compute_delta_ms = engine::debug::elapsed_ms(timing_start, Clock::now());
        graph_compute_ms += graph_compute_delta_ms;
        if (log_timing) {
            engine::debug::timing_log_scalar("higgs_tts.ar.decode.step0.graph.compute_ms", graph_compute_delta_ms);
        }
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Higgs TTS AR decode graph compute failed");
        }

        output.codebook_logits.resize(static_cast<size_t>(config.audio.num_codebooks * config.audio.vocab_size));
        timing_start = Clock::now();
        ggml_backend_tensor_get(
            logits_output,
            output.codebook_logits.data(),
            0,
            output.codebook_logits.size() * sizeof(float));
        const double output_read_delta_ms = engine::debug::elapsed_ms(timing_start, Clock::now());
        output_read_ms += output_read_delta_ms;
        if (log_timing) {
            engine::debug::timing_log_scalar("higgs_tts.ar.decode.step0.output_read_ms", output_read_delta_ms);
        }

        cache->advance_after_direct_append(1);
        ++steps;
    }

    std::shared_ptr<HiggsARRuntime> runtime;
    HiggsARKVCache * cache = nullptr;
    int64_t cache_steps = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    ggml_tensor * fused_code_ids = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * cache_slot = nullptr;
    ggml_tensor * attention_mask = nullptr;
    ggml_tensor * logits_output = nullptr;
    std::vector<int32_t> fused_code_ids_values;
    std::vector<ggml_fp16_t> attention_mask_values;
    ggml_cgraph * graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    double input_upload_ms = 0.0;
    double mask_upload_ms = 0.0;
    double graph_compute_ms = 0.0;
    double output_read_ms = 0.0;
    int64_t steps = 0;
};

struct HiggsARPrefillGraph::Impl {
    Impl(
        std::shared_ptr<HiggsARRuntime> input_runtime,
        int64_t input_prompt_steps,
        int64_t input_start_step,
        HiggsARKVCache * input_cache,
        size_t graph_arena_bytes)
        : runtime(std::move(input_runtime)),
          target_cache(input_cache),
          prompt_steps(input_prompt_steps),
          start_step(input_start_step),
          run_steps(input_prompt_steps - input_start_step),
          prefill_cache_steps(input_prompt_steps),
          layerwise(input_prompt_steps >= kLayerwisePrefillMinSteps && input_start_step == 0),
          graph_arena_bytes(graph_arena_bytes) {
        if (runtime == nullptr) {
            throw std::runtime_error("Higgs TTS AR prefill graph requires runtime");
        }
        if (prompt_steps <= 0) {
            throw std::runtime_error("Higgs TTS AR prefill graph requires positive prompt length");
        }
        if (start_step < 0 || start_step >= prompt_steps) {
            throw std::runtime_error("Higgs TTS AR prefill graph start step is outside the prompt");
        }
        if (start_step > 0 && target_cache == nullptr) {
            throw std::runtime_error("Higgs TTS AR suffix prefill requires a target KV cache");
        }
        if (layerwise) {
            engine::debug::timing_log_scalar("higgs_tts.ar.prefill.graph.build_ms", 0.0);
            return;
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize Higgs TTS AR prefill graph context");
        }
        const auto & config = runtime->assets().config;
        const auto & tensor_weights = runtime->weights();
        core::ModuleBuildContext build_ctx{ctx.get(), "higgs_tts.ar.prefill", runtime->backend_type()};

        text_tokens = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, run_steps);
        fused_code_ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, config.audio.num_codebooks, run_steps);
        text_gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, run_steps, 1);
        code_gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, run_steps, 1);
        auto x = build_higgs_prefill_input_embedding(
            build_ctx,
            tensor_weights,
            config,
            text_tokens,
            fused_code_ids,
            text_gate,
            code_gate,
            run_steps);
        positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, run_steps);
        auto positions_value =
            core::wrap_tensor(positions, core::TensorShape::from_dims({run_steps}), GGML_TYPE_I32);
        attention_mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, prefill_cache_steps, run_steps, 1, 1);
        auto attention_mask_value = core::wrap_tensor(
            attention_mask,
            core::TensorShape::from_dims({1, 1, run_steps, prefill_cache_steps}),
            GGML_TYPE_F16);
        graph = ggml_new_graph_custom(ctx.get(), 262144, false);
        keys.reserve(tensor_weights.decoder.layers.size());
        values.reserve(tensor_weights.decoder.layers.size());
        const HiggsQwenDecoderComponent decoder(config.text, tensor_weights.packed_qkv);
        for (size_t layer_index = 0; layer_index < tensor_weights.decoder.layers.size(); ++layer_index) {
            std::optional<core::TensorValue> prefix_key;
            std::optional<core::TensorValue> prefix_value;
            if (start_step > 0) {
                prefix_key = higgs_cache_view(
                    build_ctx,
                    target_cache->key_tensor(layer_index),
                    0,
                    start_step,
                    config.text.num_key_value_heads,
                    config.text.head_dim);
                prefix_value = higgs_cache_view(
                    build_ctx,
                    target_cache->value_tensor(layer_index),
                    0,
                    start_step,
                    config.text.num_key_value_heads,
                    config.text.head_dim);
            }
            auto out = decoder.build_prefill_layer(
                build_ctx,
                x,
                positions_value,
                tensor_weights.decoder.layers[layer_index],
                attention_mask_value,
                prefix_key,
                prefix_value);
            x = out.output;
            if (target_cache != nullptr) {
                auto key_dest = higgs_cache_view(
                    build_ctx,
                    target_cache->key_tensor(layer_index),
                    start_step,
                    run_steps,
                    config.text.num_key_value_heads,
                    config.text.head_dim);
                auto value_dest = higgs_cache_view(
                    build_ctx,
                    target_cache->value_tensor(layer_index),
                    start_step,
                    run_steps,
                    config.text.num_key_value_heads,
                    config.text.head_dim);
                ggml_build_forward_expand(graph, ggml_cpy(ctx.get(), out.key.tensor, key_dest.tensor));
                ggml_build_forward_expand(graph, ggml_cpy(ctx.get(), out.value.tensor, value_dest.tensor));
            } else {
                keys.push_back(out.key.tensor);
                values.push_back(out.value.tensor);
            }
        }
        x = modules::SliceModule({1, run_steps - 1, 1}).build(build_ctx, x);
        x = modules::RMSNormModule({config.text.hidden_size, config.text.rms_norm_eps, true, false})
                .build(build_ctx, x, {tensor_weights.norm, std::nullopt});
        auto logits = build_modality_logits(build_ctx, x, tensor_weights, config);
        logits_output = logits.tensor;
        ggml_set_output(logits_output);
        ggml_build_forward_expand(graph, logits_output);

        buffer = ggml_backend_alloc_ctx_tensors(ctx.get(), runtime->backend());
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate Higgs TTS AR prefill graph");
        }
        text_token_values.assign(static_cast<size_t>(run_steps), 0);
        fused_code_id_values.assign(static_cast<size_t>(run_steps * config.audio.num_codebooks), 0);
        text_gate_values.assign(static_cast<size_t>(run_steps), 0.0F);
        code_gate_values.assign(static_cast<size_t>(run_steps), 0.0F);
        positions_values = modules::qwen_position_ids(run_steps, start_step);
        attention_mask_values = modules::qwen_causal_suffix_mask_values(1, run_steps, start_step);
        engine::debug::timing_log_scalar(
            "higgs_tts.ar.prefill.graph.build_ms",
            engine::debug::elapsed_ms(build_start, Clock::now()));
    }

    ~Impl() {
        engine::core::release_backend_graph_resources(runtime->backend(), graph);
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }

    bool matches(
        const HiggsARRuntime & candidate_runtime,
        int64_t candidate_prompt_steps,
        int64_t candidate_start_step) const {
        return runtime.get() == &candidate_runtime &&
            prompt_steps == candidate_prompt_steps &&
            start_step == candidate_start_step;
    }

    struct EmbeddingGraph {
        EmbeddingGraph(const HiggsARRuntime & runtime, int64_t steps, size_t arena_bytes)
            : runtime(&runtime), steps(steps) {
            const auto & config = runtime.assets().config;
            ggml_init_params params{arena_bytes, nullptr, true};
            ctx.reset(ggml_init(params));
            if (ctx == nullptr) {
                throw std::runtime_error("failed to initialize Higgs TTS AR embedding graph context");
            }
            core::ModuleBuildContext build_ctx{ctx.get(), "higgs_tts.ar.prefill.embedding", runtime.backend_type()};
            text_tokens = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, steps);
            fused_code_ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, config.audio.num_codebooks, steps);
            text_gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, steps, 1);
            code_gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, steps, 1);
            output = build_higgs_prefill_input_embedding(
                         build_ctx,
                         runtime.weights(),
                         config,
                         text_tokens,
                         fused_code_ids,
                         text_gate,
                         code_gate,
                         steps)
                         .tensor;
            graph = ggml_new_graph_custom(ctx.get(), 32768, false);
            ggml_set_output(output);
            ggml_build_forward_expand(graph, output);
            buffer = ggml_backend_alloc_ctx_tensors(ctx.get(), runtime.backend());
            if (buffer == nullptr) {
                throw std::runtime_error("failed to allocate Higgs TTS AR embedding graph");
            }
        }

        ~EmbeddingGraph() {
            engine::core::release_backend_graph_resources(runtime->backend(), graph);
            if (buffer != nullptr) {
                ggml_backend_buffer_free(buffer);
            }
        }

        std::vector<float> run(const HiggsARPrefillInput & input) {
            const auto & config = runtime->assets().config;
            ggml_backend_tensor_set(text_tokens, input.text_tokens.data(), 0, input.text_tokens.size() * sizeof(int32_t));
            ggml_backend_tensor_set(
                fused_code_ids,
                input.fused_code_ids.data(),
                0,
                input.fused_code_ids.size() * sizeof(int32_t));
            ggml_backend_tensor_set(text_gate, input.text_gate.data(), 0, input.text_gate.size() * sizeof(float));
            ggml_backend_tensor_set(code_gate, input.code_gate.data(), 0, input.code_gate.size() * sizeof(float));
            core::set_backend_threads(runtime->backend(), runtime->threads());
            const ggml_status status = engine::core::compute_backend_graph(runtime->backend(), graph);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Higgs TTS AR embedding graph compute failed");
            }
            std::vector<float> hidden(static_cast<size_t>(steps * config.text.hidden_size));
            ggml_backend_tensor_get(output, hidden.data(), 0, hidden.size() * sizeof(float));
            return hidden;
        }

        const HiggsARRuntime * runtime = nullptr;
        int64_t steps = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
        ggml_tensor * text_tokens = nullptr;
        ggml_tensor * fused_code_ids = nullptr;
        ggml_tensor * text_gate = nullptr;
        ggml_tensor * code_gate = nullptr;
        ggml_tensor * output = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
    };

    struct LayerGraph {
        LayerGraph(
            const HiggsARRuntime & runtime,
            const modules::QwenDecoderLayerWeights & layer,
            int64_t steps,
            size_t arena_bytes)
            : runtime(&runtime), steps(steps) {
            const auto & config = runtime.assets().config;
            ggml_init_params params{arena_bytes, nullptr, true};
            ctx.reset(ggml_init(params));
            if (ctx == nullptr) {
                throw std::runtime_error("failed to initialize Higgs TTS AR layer prefill graph context");
            }
            core::ModuleBuildContext build_ctx{ctx.get(), "higgs_tts.ar.prefill.layer", runtime.backend_type()};
            auto x = core::make_tensor(
                build_ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, steps, config.text.hidden_size}));
            input = x.tensor;
            positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, steps);
            auto positions_value =
                core::wrap_tensor(positions, core::TensorShape::from_dims({steps}), GGML_TYPE_I32);
            attention_mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, steps, steps, 1, 1);
            auto attention_mask_value = core::wrap_tensor(
                attention_mask,
                core::TensorShape::from_dims({1, 1, steps, steps}),
                GGML_TYPE_F16);
            const HiggsQwenDecoderComponent decoder(config.text, runtime.weights().packed_qkv);
            auto out = decoder.build_prefill_layer(
                build_ctx,
                x,
                positions_value,
                layer,
                attention_mask_value);
            output = out.output.tensor;
            key = out.key.tensor;
            value = out.value.tensor;
            graph = ggml_new_graph_custom(ctx.get(), 65536, false);
            ggml_set_output(output);
            ggml_build_forward_expand(graph, output);
            buffer = ggml_backend_alloc_ctx_tensors(ctx.get(), runtime.backend());
            if (buffer == nullptr) {
                throw std::runtime_error("failed to allocate Higgs TTS AR layer prefill graph");
            }

            const auto position_values = modules::qwen_position_ids(steps);
            ggml_backend_tensor_set(positions, position_values.data(), 0, position_values.size() * sizeof(int32_t));
            auto mask = modules::qwen_causal_prefill_mask_values(1, steps);
            ggml_backend_tensor_set(attention_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        }

        ~LayerGraph() {
            engine::core::release_backend_graph_resources(runtime->backend(), graph);
            if (buffer != nullptr) {
                ggml_backend_buffer_free(buffer);
            }
        }

        struct Output {
            std::vector<float> hidden;
            std::vector<float> key;
            std::vector<float> value;
        };

        Output run(const std::vector<float> & hidden) {
            const auto & config = runtime->assets().config;
            const size_t hidden_values = static_cast<size_t>(steps * config.text.hidden_size);
            if (hidden.size() != hidden_values) {
                throw std::runtime_error("Higgs TTS AR layer prefill input size mismatch");
            }
            ggml_backend_tensor_set(input, hidden.data(), 0, hidden.size() * sizeof(float));
            core::set_backend_threads(runtime->backend(), runtime->threads());
            const ggml_status status = engine::core::compute_backend_graph(runtime->backend(), graph);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Higgs TTS AR layer prefill graph compute failed");
            }
            Output out;
            out.hidden.resize(hidden_values);
            ggml_backend_tensor_get(output, out.hidden.data(), 0, out.hidden.size() * sizeof(float));
            const size_t layer_values = static_cast<size_t>(
                steps * config.text.num_key_value_heads * config.text.head_dim);
            out.key.resize(layer_values);
            out.value.resize(layer_values);
            ggml_backend_tensor_get(key, out.key.data(), 0, out.key.size() * sizeof(float));
            ggml_backend_tensor_get(value, out.value.data(), 0, out.value.size() * sizeof(float));
            return out;
        }

        const HiggsARRuntime * runtime = nullptr;
        int64_t steps = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
        ggml_tensor * input = nullptr;
        ggml_tensor * positions = nullptr;
        ggml_tensor * attention_mask = nullptr;
        ggml_tensor * output = nullptr;
        ggml_tensor * key = nullptr;
        ggml_tensor * value = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
    };

    struct FinalGraph {
        FinalGraph(const HiggsARRuntime & runtime, size_t arena_bytes) : runtime(&runtime) {
            const auto & config = runtime.assets().config;
            ggml_init_params params{arena_bytes, nullptr, true};
            ctx.reset(ggml_init(params));
            if (ctx == nullptr) {
                throw std::runtime_error("failed to initialize Higgs TTS AR final prefill graph context");
            }
            core::ModuleBuildContext build_ctx{ctx.get(), "higgs_tts.ar.prefill.final", runtime.backend_type()};
            auto x = core::make_tensor(
                build_ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, 1, config.text.hidden_size}));
            input = x.tensor;
            x = modules::RMSNormModule({config.text.hidden_size, config.text.rms_norm_eps, true, false})
                    .build(build_ctx, x, {runtime.weights().norm, std::nullopt});
            auto logits = build_modality_logits(build_ctx, x, runtime.weights(), config);
            output = logits.tensor;
            graph = ggml_new_graph_custom(ctx.get(), 8192, false);
            ggml_set_output(output);
            ggml_build_forward_expand(graph, output);
            buffer = ggml_backend_alloc_ctx_tensors(ctx.get(), runtime.backend());
            if (buffer == nullptr) {
                throw std::runtime_error("failed to allocate Higgs TTS AR final prefill graph");
            }
        }

        ~FinalGraph() {
            engine::core::release_backend_graph_resources(runtime->backend(), graph);
            if (buffer != nullptr) {
                ggml_backend_buffer_free(buffer);
            }
        }

        std::vector<float> run(const std::vector<float> & hidden) {
            const auto & config = runtime->assets().config;
            if (static_cast<int64_t>(hidden.size()) != config.text.hidden_size) {
                throw std::runtime_error("Higgs TTS AR final prefill input size mismatch");
            }
            ggml_backend_tensor_set(input, hidden.data(), 0, hidden.size() * sizeof(float));
            core::set_backend_threads(runtime->backend(), runtime->threads());
            const ggml_status status = engine::core::compute_backend_graph(runtime->backend(), graph);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Higgs TTS AR final prefill graph compute failed");
            }
            std::vector<float> logits(static_cast<size_t>(
                config.audio.num_codebooks * config.audio.vocab_size));
            ggml_backend_tensor_get(output, logits.data(), 0, logits.size() * sizeof(float));
            return logits;
        }

        const HiggsARRuntime * runtime = nullptr;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
        ggml_tensor * input = nullptr;
        ggml_tensor * output = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
    };

    HiggsARPrefillOutput run_layerwise(const HiggsARPrefillInput & input) {
        const auto & config = runtime->assets().config;
        auto hidden = EmbeddingGraph(*runtime, prompt_steps, graph_arena_bytes).run(input);
        HiggsARPrefillOutput out;
        out.kv_state.current_end = prompt_steps;
        out.kv_state.layers.resize(runtime->weights().decoder.layers.size());
        for (size_t layer = 0; layer < runtime->weights().decoder.layers.size(); ++layer) {
            LayerGraph graph(
                *runtime,
                runtime->weights().decoder.layers[layer],
                prompt_steps,
                graph_arena_bytes);
            auto layer_out = graph.run(hidden);
            hidden = std::move(layer_out.hidden);
            auto & state = out.kv_state.layers[layer];
            state.valid_steps = prompt_steps;
            state.key = std::move(layer_out.key);
            state.value = std::move(layer_out.value);
        }
        std::vector<float> last_hidden(static_cast<size_t>(config.text.hidden_size));
        const auto last_begin = hidden.begin() +
            static_cast<std::ptrdiff_t>((prompt_steps - 1) * config.text.hidden_size);
        std::copy(
            last_begin,
            last_begin + static_cast<std::ptrdiff_t>(config.text.hidden_size),
            last_hidden.begin());
        out.output.codebook_logits = FinalGraph(*runtime, graph_arena_bytes).run(last_hidden);
        return out;
    }

    HiggsARPrefillOutput run(const HiggsARPrefillInput & input, int64_t candidate_start_step) {
        const auto & config = runtime->assets().config;
        if (input.steps != prompt_steps ||
            static_cast<int64_t>(input.text_tokens.size()) != prompt_steps ||
            static_cast<int64_t>(input.fused_code_ids.size()) != prompt_steps * config.audio.num_codebooks ||
            static_cast<int64_t>(input.text_gate.size()) != prompt_steps ||
            static_cast<int64_t>(input.code_gate.size()) != prompt_steps) {
            throw std::runtime_error("Higgs TTS AR prefill graph input shape mismatch");
        }
        if (candidate_start_step != start_step) {
            throw std::runtime_error("Higgs TTS AR prefill graph start step mismatch");
        }
        if (start_step > 0 &&
            (target_cache == nullptr || target_cache->valid_steps() < start_step ||
             target_cache->current_end() != start_step)) {
            throw std::runtime_error("Higgs TTS AR suffix prefill requires the retained prefix in KV cache");
        }
        if (layerwise) {
            return run_layerwise(input);
        }
        for (int64_t step = 0; step < run_steps; ++step) {
            const int64_t source_step = start_step + step;
            text_token_values[static_cast<size_t>(step)] =
                input.text_tokens[static_cast<size_t>(source_step)];
            text_gate_values[static_cast<size_t>(step)] =
                input.text_gate[static_cast<size_t>(source_step)];
            code_gate_values[static_cast<size_t>(step)] =
                input.code_gate[static_cast<size_t>(source_step)];
            for (int64_t codebook = 0; codebook < config.audio.num_codebooks; ++codebook) {
                fused_code_id_values[static_cast<size_t>(step * config.audio.num_codebooks + codebook)] =
                    input.fused_code_ids[static_cast<size_t>(source_step * config.audio.num_codebooks + codebook)];
            }
        }
        ggml_backend_tensor_set(text_tokens, text_token_values.data(), 0, text_token_values.size() * sizeof(int32_t));
        ggml_backend_tensor_set(
            fused_code_ids,
            fused_code_id_values.data(),
            0,
            fused_code_id_values.size() * sizeof(int32_t));
        ggml_backend_tensor_set(text_gate, text_gate_values.data(), 0, text_gate_values.size() * sizeof(float));
        ggml_backend_tensor_set(code_gate, code_gate_values.data(), 0, code_gate_values.size() * sizeof(float));
        ggml_backend_tensor_set(positions, positions_values.data(), 0, positions_values.size() * sizeof(int32_t));
        ggml_backend_tensor_set(
            attention_mask,
            attention_mask_values.data(),
            0,
            attention_mask_values.size() * sizeof(ggml_fp16_t));

        core::set_backend_threads(runtime->backend(), runtime->threads());
        const ggml_status status = engine::core::compute_backend_graph(runtime->backend(), graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Higgs TTS AR prefill graph compute failed");
        }
        HiggsARPrefillOutput out;
        out.output.codebook_logits.resize(static_cast<size_t>(config.audio.num_codebooks * config.audio.vocab_size));
        ggml_backend_tensor_get(
            logits_output,
            out.output.codebook_logits.data(),
            0,
            out.output.codebook_logits.size() * sizeof(float));
        if (target_cache != nullptr) {
            target_cache->advance_after_direct_append(run_steps);
            out.wrote_cache = true;
            out.kv_state.current_end = prompt_steps;
            return out;
        }
        out.kv_state.current_end = prompt_steps;
        out.kv_state.layers.resize(keys.size());
        const size_t layer_values = static_cast<size_t>(
            prompt_steps * config.text.num_key_value_heads * config.text.head_dim);
        for (size_t layer = 0; layer < keys.size(); ++layer) {
            auto & state = out.kv_state.layers[layer];
            state.valid_steps = prompt_steps;
            state.key.resize(layer_values);
            state.value.resize(layer_values);
            ggml_backend_tensor_get(keys[layer], state.key.data(), 0, state.key.size() * sizeof(float));
            ggml_backend_tensor_get(values[layer], state.value.data(), 0, state.value.size() * sizeof(float));
        }
        return out;
    }

    std::shared_ptr<HiggsARRuntime> runtime;
    HiggsARKVCache * target_cache = nullptr;
    int64_t prompt_steps = 0;
    int64_t start_step = 0;
    int64_t run_steps = 0;
    int64_t prefill_cache_steps = 0;
    bool layerwise = false;
    size_t graph_arena_bytes = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    ggml_tensor * text_tokens = nullptr;
    ggml_tensor * fused_code_ids = nullptr;
    ggml_tensor * text_gate = nullptr;
    ggml_tensor * code_gate = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * attention_mask = nullptr;
    ggml_tensor * logits_output = nullptr;
    std::vector<ggml_tensor *> keys;
    std::vector<ggml_tensor *> values;
    std::vector<int32_t> text_token_values;
    std::vector<int32_t> fused_code_id_values;
    std::vector<float> text_gate_values;
    std::vector<float> code_gate_values;
    std::vector<int32_t> positions_values;
    std::vector<ggml_fp16_t> attention_mask_values;
    ggml_cgraph * graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
};

HiggsARPrefillGraph::HiggsARPrefillGraph(
    std::shared_ptr<HiggsARRuntime> runtime,
    int64_t prompt_steps,
    int64_t start_step,
    HiggsARKVCache * cache,
    size_t graph_arena_bytes)
    : impl_(std::make_unique<Impl>(std::move(runtime), prompt_steps, start_step, cache, graph_arena_bytes)) {}

HiggsARPrefillGraph::~HiggsARPrefillGraph() = default;

bool HiggsARPrefillGraph::matches(
    const HiggsARRuntime & runtime,
    int64_t prompt_steps,
    int64_t start_step) const {
    return impl_->matches(runtime, prompt_steps, start_step);
}

HiggsARPrefillOutput HiggsARPrefillGraph::run(const HiggsARPrefillInput & input, int64_t start_step) {
    return impl_->run(input, start_step);
}


HiggsARDecodeGraph::HiggsARDecodeGraph(
    std::shared_ptr<HiggsARRuntime> runtime,
    int64_t cache_steps,
    HiggsARKVCache & cache,
    size_t graph_arena_bytes)
    : impl_(std::make_unique<Impl>(std::move(runtime), cache_steps, cache, graph_arena_bytes)) {}

HiggsARDecodeGraph::~HiggsARDecodeGraph() = default;

bool HiggsARDecodeGraph::can_run(const HiggsARRuntime & runtime, int64_t required_steps) const {
    return impl_->can_run(runtime, required_steps);
}

int64_t HiggsARDecodeGraph::cache_steps() const {
    return impl_->cache_steps_value();
}

void HiggsARDecodeGraph::import_prefill_state(const runtime::TransformerKVState & state) {
    impl_->import_prefill_state(state);
}

void HiggsARDecodeGraph::begin_decode_run() {
    impl_->begin_decode_run();
}

HiggsARDecodeTiming HiggsARDecodeGraph::timing() const {
    return impl_->timing();
}

void HiggsARDecodeGraph::run_step_into(
    const HiggsARDecodeInput & input,
    HiggsARDecodeOutput & output,
    bool log_timing) {
    impl_->run_step_into(input, output, log_timing);
}

}  // namespace engine::models::higgs_tts
