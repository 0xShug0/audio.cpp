#include "engine/community_models/outetts/llama.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/attention/qwen_causal_decoder.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/torch_random.h"
#include "../common/constant_tensor_cache.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

namespace engine::models::outetts {
namespace {

namespace binding = modules::binding;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct LayerWeights {
    assets::TensorDataF32 input_norm;
    modules::AttentionWeights attention;
    assets::TensorDataF32 post_norm;
    modules::LinearWeights gate;
    modules::LinearWeights up;
    modules::LinearWeights down;
};

struct ModelWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue embedding;
    core::TensorValue lm_head;
    core::TensorValue rope_factors;
    std::vector<LayerWeights> layers;
    assets::TensorDataF32 norm;
};

struct PrefillOutput {
    std::vector<float> logits;
    runtime::TransformerKVState state;
};

std::vector<float> llama3_rope_factors(const OuteTTSConfig & config) {
    constexpr double pi = 3.14159265358979323846;
    const auto & scaling = config.rope_scaling;
    const double low_wavelength = static_cast<double>(scaling.original_max_position_embeddings) /
        static_cast<double>(scaling.low_freq_factor);
    const double high_wavelength = static_cast<double>(scaling.original_max_position_embeddings) /
        static_cast<double>(scaling.high_freq_factor);
    std::vector<float> factors(static_cast<size_t>(config.head_dim / 2), 1.0F);
    for (int64_t i = 0; i < config.head_dim / 2; ++i) {
        const double inv_freq = 1.0 / std::pow(
            static_cast<double>(config.rope_theta),
            static_cast<double>(2 * i) / static_cast<double>(config.head_dim));
        const double wavelength = 2.0 * pi / inv_freq;
        double scaled = inv_freq;
        if (wavelength > low_wavelength) {
            scaled = inv_freq / static_cast<double>(scaling.factor);
        } else if (wavelength >= high_wavelength) {
            const double smooth =
                (static_cast<double>(scaling.original_max_position_embeddings) / wavelength -
                 static_cast<double>(scaling.low_freq_factor)) /
                (static_cast<double>(scaling.high_freq_factor) -
                 static_cast<double>(scaling.low_freq_factor));
            scaled = (1.0 - smooth) * inv_freq / static_cast<double>(scaling.factor) + smooth * inv_freq;
        }
        // ggml divides its base inverse frequency by this tensor. This matches
        // llama.cpp's rope_freqs.weight representation for Llama-3 scaling.
        factors[static_cast<size_t>(i)] = static_cast<float>(inv_freq / scaled);
    }
    return factors;
}

ModelWeights load_weights(
    const OuteTTSAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & c = assets.config;
    const auto & source = *assets.model_weights;
    ModelWeights out;
    out.store = std::make_shared<core::BackendWeightStore>(
        backend, backend_type, "outetts.llama.weights", context_bytes);
    out.embedding = out.store->load_tensor(
        source, "model.embed_tokens.weight", storage_type, {c.vocab_size, c.hidden_size});
    out.lm_head = out.embedding;
    out.rope_factors = out.store->make_from_f32(
        core::TensorShape::from_dims({c.head_dim / 2}),
        assets::TensorStorageType::F32,
        llama3_rope_factors(c));
    out.layers.reserve(static_cast<size_t>(c.num_hidden_layers));
    for (int64_t i = 0; i < c.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i);
        LayerWeights layer;
        layer.input_norm = source.require_f32_tensor(p + ".input_layernorm.weight", {c.hidden_size});
        layer.attention.q_weight = out.store->load_tensor(
            source, p + ".self_attn.q_proj.weight", storage_type,
            {c.num_attention_heads * c.head_dim, c.hidden_size});
        layer.attention.k_weight = out.store->load_tensor(
            source, p + ".self_attn.k_proj.weight", storage_type,
            {c.num_key_value_heads * c.head_dim, c.hidden_size});
        layer.attention.v_weight = out.store->load_tensor(
            source, p + ".self_attn.v_proj.weight", storage_type,
            {c.num_key_value_heads * c.head_dim, c.hidden_size});
        layer.attention.out_weight = out.store->load_tensor(
            source, p + ".self_attn.o_proj.weight", storage_type,
            {c.hidden_size, c.num_attention_heads * c.head_dim});
        layer.post_norm = source.require_f32_tensor(p + ".post_attention_layernorm.weight", {c.hidden_size});
        layer.gate.weight = out.store->load_tensor(
            source, p + ".mlp.gate_proj.weight", storage_type, {c.intermediate_size, c.hidden_size});
        layer.up.weight = out.store->load_tensor(
            source, p + ".mlp.up_proj.weight", storage_type, {c.intermediate_size, c.hidden_size});
        layer.down.weight = out.store->load_tensor(
            source, p + ".mlp.down_proj.weight", storage_type, {c.hidden_size, c.intermediate_size});
        out.layers.push_back(std::move(layer));
    }
    out.norm = source.require_f32_tensor("model.norm.weight", {c.hidden_size});
    out.store->upload();
    return out;
}

modules::QwenCausalDecoderConfig decoder_config(const OuteTTSConfig & c) {
    modules::QwenCausalDecoderConfig out;
    out.stack.hidden_size = c.hidden_size;
    out.stack.intermediate_size = c.intermediate_size;
    out.stack.num_attention_heads = c.num_attention_heads;
    out.stack.num_key_value_heads = c.num_key_value_heads;
    out.stack.head_dim = c.head_dim;
    out.stack.layers = c.num_hidden_layers;
    out.stack.rms_norm_eps = c.rms_norm_eps;
    out.stack.rope_theta = c.rope_theta;
    // Hugging Face Llama weights use split-half rotary pairs. llama.cpp's
    // dedicated Llama converter permutes Q/K and then uses NORMAL RoPE, while
    // audio.cpp preserves the source tensor layout in safetensors and GGUF.
    out.stack.rope_type = GGML_ROPE_TYPE_NEOX;
    out.stack.use_qk_norm = false;
    out.stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    out.logits_size = c.vocab_size;
    return out;
}

modules::QwenCausalDecoderWeights graph_weights(
    const ModelWeights & weights,
    common::ConstantTensorCache & constants) {
    modules::QwenCausalDecoderWeights out;
    out.stack.layers.reserve(weights.layers.size());
    for (const auto & source : weights.layers) {
        modules::QwenDecoderLayerWeights layer;
        layer.input_norm = binding::norm_data(constants, source.input_norm);
        layer.self_attention = source.attention;
        layer.post_norm = binding::norm_data(constants, source.post_norm);
        layer.mlp.gate_proj = source.gate;
        layer.mlp.up_proj = source.up;
        layer.mlp.down_proj = source.down;
        layer.rope_frequency_factors = weights.rope_factors;
        out.stack.layers.push_back(std::move(layer));
    }
    out.final_norm = binding::norm_data(constants, weights.norm);
    out.lm_head.weight = weights.lm_head;
    return out;
}

struct SamplingScratch {
    std::vector<int32_t> repeated_ids;
    std::vector<size_t> order;
    std::vector<float> probabilities;
};

void apply_repetition_penalty(
    std::vector<float> & logits,
    const std::vector<int32_t> & ids,
    int64_t window,
    float penalty,
    std::vector<int32_t> & repeated_ids) {
    if (penalty == 1.0F || window == 0) {
        return;
    }
    const size_t begin = ids.size() > static_cast<size_t>(window) ? ids.size() - static_cast<size_t>(window) : 0;
    repeated_ids.clear();
    for (auto it = ids.begin() + static_cast<std::ptrdiff_t>(begin); it != ids.end(); ++it) {
        if (std::find(repeated_ids.begin(), repeated_ids.end(), *it) == repeated_ids.end()) {
            repeated_ids.push_back(*it);
        }
    }
    for (const int32_t id : repeated_ids) {
        if (id < 0 || static_cast<size_t>(id) >= logits.size()) {
            continue;
        }
        float & value = logits[static_cast<size_t>(id)];
        value = value <= 0.0F ? value * penalty : value / penalty;
    }
}

int32_t sample_token(
    const std::vector<float> & logits,
    const OuteTTSGenerateOptions & o,
    std::mt19937 & rng,
    const sampling::TorchCudaSamplingPolicy & sampling_policy,
    uint64_t call_index,
    SamplingScratch & scratch) {
    if (!(o.temperature > 0.0F) || !std::isfinite(o.temperature)) {
        return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    auto & order = scratch.order;
    order.resize(logits.size());
    std::iota(order.begin(), order.end(), 0);
    const size_t kept_by_top_k =
        o.top_k > 0 && static_cast<size_t>(o.top_k) < order.size()
            ? static_cast<size_t>(o.top_k)
            : order.size();
    const auto by_logit_desc = [&](size_t a, size_t b) {
        return logits[a] > logits[b];
    };
    if (kept_by_top_k < order.size()) {
        std::partial_sort(
            order.begin(),
            order.begin() + static_cast<std::ptrdiff_t>(kept_by_top_k),
            order.end(),
            by_logit_desc);
        order.resize(kept_by_top_k);
    } else {
        std::sort(order.begin(), order.end(), by_logit_desc);
    }
    const float max_logit = logits[order.front()];
    auto & probabilities = scratch.probabilities;
    probabilities.assign(order.size(), 0.0F);
    double sum = 0.0;
    for (size_t i = 0; i < order.size(); ++i) {
        probabilities[i] = std::exp((logits[order[i]] - max_logit) / o.temperature);
        sum += probabilities[i];
    }
    for (float & probability : probabilities) {
        probability = static_cast<float>(probability / sum);
    }
    const float max_probability = probabilities.front();
    float cumulative = 0.0F;
    size_t kept = 0;
    for (const float probability : probabilities) {
        if (probability < max_probability * o.min_p && kept > 0) {
            break;
        }
        cumulative += probability;
        ++kept;
        if (o.top_p < 1.0F && cumulative >= o.top_p) {
            break;
        }
    }
    order.resize(std::max<size_t>(1, kept));
    probabilities.resize(order.size());
    if (sampling_policy.cuda_fast_path) {
        double best_rank = -std::numeric_limits<double>::infinity();
        int32_t best_token = -1;
        for (size_t index = 0; index < order.size(); ++index) {
            const float exponential =
                sampling::torch_cuda_tensor_iterator_exponential_element(
                    o.seed,
                    static_cast<uint64_t>(logits.size()),
                    static_cast<uint64_t>(order[index]),
                    call_index,
                    sampling_policy.multiprocessor_count,
                    sampling_policy.max_threads_per_multiprocessor);
            const double rank =
                static_cast<double>(probabilities[index]) /
                static_cast<double>(exponential);
            if (rank > best_rank) {
                best_rank = rank;
                best_token = static_cast<int32_t>(order[index]);
            }
        }
        if (best_token < 0) {
            throw std::runtime_error(
                "OuteTTS CUDA sampler failed to select a token");
        }
        return best_token;
    }
    std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());
    return static_cast<int32_t>(order[static_cast<size_t>(distribution(rng))]);
}

class CachedStepGraph {
public:
    CachedStepGraph(
        const OuteTTSConfig & config,
        const ModelWeights & weights,
        common::ConstantTensorCache & constants,
        ggml_backend_t backend,
        int threads,
        int64_t capacity)
        : config_(config), weights_(&weights), constants_(&constants), backend_(backend), threads_(threads), capacity_(capacity) {
        ggml_init_params params{1536ull * 1024ull * 1024ull, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (!ctx_) throw std::runtime_error("failed to create OuteTTS cached-step context");
        core::ModuleBuildContext build{ctx_.get(), "outetts.llama.cached_step"};
        input_id_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_I32, 1, 1);
        auto id = core::wrap_tensor(input_id_, core::TensorShape::from_dims({1, 1}), GGML_TYPE_I32);
        auto x = modules::EmbeddingModule({config_.vocab_size, config_.hidden_size}).build(build, id, weights_->embedding);
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        cache_slot_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, capacity_, 1, 1, 1);
        auto position = core::wrap_tensor(positions_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto slot = core::wrap_tensor(cache_slot_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto mask = core::wrap_tensor(mask_, core::TensorShape::from_dims({1, 1, 1, capacity_}), GGML_TYPE_F16);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        constants_->begin_graph();
        auto output = modules::QwenCausalDecoderModule(decoder_config(config_)).build_static_cache_tail(
            build, graph_, x, position, graph_weights(*weights_, *constants_), capacity_, mask, slot);
        cache_ = std::move(output.cache);
        logits_ = output.logits.tensor;
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, logits_);
        constants_->finish_graph();
        constants_->ensure_uploaded();
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_);
        if (buffer_ == nullptr) throw std::runtime_error("failed to allocate OuteTTS cached-step graph");
        mask_values_.assign(static_cast<size_t>(capacity_), ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
    }

    ~CachedStepGraph() {
        core::release_backend_graph_resources(backend_, graph_);
        if (buffer_ != nullptr) ggml_backend_buffer_free(buffer_);
    }

    int64_t capacity() const noexcept { return capacity_; }

    void import_state(const runtime::TransformerKVState & state) { cache_.import_state(state); }

    void run(int32_t token, std::vector<float> & logits) {
        if (cache_.valid_steps() >= capacity_) throw std::runtime_error("OuteTTS cached-step capacity exceeded");
        ggml_backend_tensor_set(input_id_, &token, 0, sizeof(token));
        const int32_t position = static_cast<int32_t>(cache_.current_end());
        const int32_t slot = static_cast<int32_t>(cache_.valid_steps());
        ggml_backend_tensor_set(positions_, &position, 0, sizeof(position));
        ggml_backend_tensor_set(cache_slot_, &slot, 0, sizeof(slot));
        modules::write_qwen_cached_step_mask(mask_, mask_values_, capacity_, cache_.valid_steps(), cache_.valid_steps());
        core::set_backend_threads(backend_, threads_);
        const auto status = core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("OuteTTS cached-step graph compute failed");
        if (logits.size() != static_cast<size_t>(config_.vocab_size)) {
            logits.resize(static_cast<size_t>(config_.vocab_size));
        }
        ggml_backend_tensor_get(logits_, logits.data(), 0, logits.size() * sizeof(float));
        cache_.advance_after_direct_append(1);
    }

private:
    OuteTTSConfig config_;
    const ModelWeights * weights_ = nullptr;
    common::ConstantTensorCache * constants_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    int64_t capacity_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_id_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * cache_slot_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    runtime::TransformerKVCache cache_;
    std::vector<ggml_fp16_t> mask_values_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

}  // namespace

struct OuteTTSLlamaRuntime::Impl {
    Impl(
        std::shared_ptr<const OuteTTSAssets> assets_in,
        core::BackendType backend_type,
        int device,
        int threads_in,
        size_t weight_context_bytes,
        size_t constant_context_bytes,
        assets::TensorStorageType storage_type)
        : assets(std::move(assets_in)),
          threads(std::max(1, threads_in)),
          sampling_policy(
              backend_type == core::BackendType::Cuda
                  ? sampling::resolve_torch_cuda_sampling_policy(
                        backend_type,
                        device,
                        "outetts",
                        "OuteTTS",
                        sampling::TorchCudaSamplingPolicyFailureMode::StrictCuda)
                  : sampling::TorchCudaSamplingPolicy{}) {
        if (assets == nullptr) {
            throw std::runtime_error("OuteTTS Llama runtime requires assets");
        }
        backend = core::init_backend({backend_type, device, threads});
        weights = load_weights(*assets, backend, backend_type, weight_context_bytes, storage_type);
        constants = std::make_unique<common::ConstantTensorCache>(
            backend, threads, "outetts.llama.constants", constant_context_bytes);
    }

    ~Impl() {
        // CachedStepGraph owns backend buffers and asks the backend to release
        // graph resources in its destructor. Destroy it before its constants,
        // weights, and backend. The previous order left a live graph holding a
        // freed backend and caused a Linux CUDA segfault during session teardown.
        step_graph.reset();
        constants.reset();
        weights.store.reset();
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }

    PrefillOutput prefill(const std::vector<int32_t> & ids) const {
        using Clock = std::chrono::steady_clock;
        const auto total_start = Clock::now();
        const auto & c = assets->config;
        const int64_t steps = static_cast<int64_t>(ids.size());
        // This correctness-first graph uses full-sequence prefill. A cached-step
        // graph can replace it without changing weights, sampling, or package layout.
        const size_t arena = std::max<size_t>(
            1024ull * 1024ull * 1024ull,
            static_cast<size_t>(steps) * static_cast<size_t>(steps) * 256ull + 512ull * 1024ull * 1024ull);
        ggml_init_params params{arena, nullptr, true};
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx(ggml_init(params));
        if (!ctx) {
            throw std::runtime_error("failed to create OuteTTS Llama graph context");
        }
        core::ModuleBuildContext build{ctx.get(), "outetts.llama.prefill"};
        auto * ids_tensor = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, steps, 1);
        auto ids_value = core::wrap_tensor(ids_tensor, core::TensorShape::from_dims({1, steps}), GGML_TYPE_I32);
        auto x = modules::EmbeddingModule({c.vocab_size, c.hidden_size})
                     .build(build, ids_value, weights.embedding);
        auto * positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, steps);
        auto position_value = core::wrap_tensor(positions, core::TensorShape::from_dims({steps}), GGML_TYPE_I32);
        auto * mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, steps, steps, 1, 1);
        auto mask_value = core::wrap_tensor(
            mask, core::TensorShape::from_dims({1, 1, steps, steps}), GGML_TYPE_F16);
        constants->begin_graph();
        auto output = modules::QwenCausalDecoderModule(decoder_config(c)).build(
            build, x, position_value, graph_weights(weights, *constants), std::nullopt, mask_value);
        std::vector<ggml_tensor *> keys;
        std::vector<ggml_tensor *> values;
        keys.reserve(output.state.layers.size());
        values.reserve(output.state.layers.size());
        for (const auto & layer : output.state.layers) {
            if (!layer.key.has_value() || !layer.value.has_value()) {
                throw std::runtime_error("OuteTTS prefill did not produce K/V state");
            }
            auto * key = ggml_cpy(ctx.get(), layer.key->tensor, ggml_dup_tensor(ctx.get(), layer.key->tensor));
            auto * value = ggml_cpy(ctx.get(), layer.value->tensor, ggml_dup_tensor(ctx.get(), layer.value->tensor));
            ggml_set_output(key);
            ggml_set_output(value);
            keys.push_back(key);
            values.push_back(value);
        }
        auto * logits = output.logits.tensor;
        ggml_set_output(logits);
        auto * graph = ggml_new_graph_custom(ctx.get(), 65536, false);
        for (auto * key : keys) ggml_build_forward_expand(graph, key);
        for (auto * value : values) ggml_build_forward_expand(graph, value);
        ggml_build_forward_expand(graph, logits);
        constants->finish_graph();
        constants->ensure_uploaded();
        ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (allocator == nullptr || !ggml_gallocr_reserve(allocator, graph) || !ggml_gallocr_alloc_graph(allocator, graph)) {
            if (allocator != nullptr) {
                ggml_gallocr_free(allocator);
            }
            throw std::runtime_error("failed to allocate OuteTTS Llama graph");
        }
        const auto build_end = Clock::now();
        const auto position_values = modules::qwen_position_ids(steps);
        const auto mask_values = modules::qwen_causal_prefill_mask_values(1, steps);
        ggml_backend_tensor_set(ids_tensor, ids.data(), 0, ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(positions, position_values.data(), 0, position_values.size() * sizeof(int32_t));
        ggml_backend_tensor_set(mask, mask_values.data(), 0, mask_values.size() * sizeof(ggml_fp16_t));
        core::set_backend_threads(backend, threads);
        const auto status = core::compute_backend_graph(backend, graph);
        ggml_backend_synchronize(backend);
        const auto compute_end = Clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            core::release_backend_graph_resources(backend, graph);
            ggml_gallocr_free(allocator);
            throw std::runtime_error("OuteTTS Llama graph compute failed");
        }
        PrefillOutput result;
        result.logits.resize(static_cast<size_t>(c.vocab_size));
        ggml_backend_tensor_get(logits, result.logits.data(), 0, result.logits.size() * sizeof(float));
        result.state.current_end = steps;
        result.state.layers.resize(keys.size());
        const size_t layer_elements = static_cast<size_t>(steps * c.num_key_value_heads * c.head_dim);
        for (size_t layer = 0; layer < keys.size(); ++layer) {
            auto & state = result.state.layers[layer];
            state.valid_steps = steps;
            state.key.resize(layer_elements);
            state.value.resize(layer_elements);
            ggml_backend_tensor_get(keys[layer], state.key.data(), 0, layer_elements * sizeof(float));
            ggml_backend_tensor_get(values[layer], state.value.data(), 0, layer_elements * sizeof(float));
        }
        const auto read_end = Clock::now();
        core::release_backend_graph_resources(backend, graph);
        ggml_gallocr_free(allocator);
        const auto release_end = Clock::now();
        debug::trace_log_scalar("outetts.llama.prefill.graph_rebuilt", true);
        debug::trace_log_scalar("outetts.llama.prefill.graph_reused", false);
        debug::trace_log_scalar("outetts.llama.prefill.tokens", steps);
        debug::timing_log_scalar(
            "outetts.llama.prefill.graph_build_ms",
            debug::elapsed_ms(total_start, build_end));
        debug::timing_log_scalar(
            "outetts.llama.prefill.compute_ms",
            debug::elapsed_ms(build_end, compute_end));
        debug::timing_log_scalar(
            "outetts.llama.prefill.read_ms",
            debug::elapsed_ms(compute_end, read_end));
        debug::timing_log_scalar(
            "outetts.llama.prefill.release_ms",
            debug::elapsed_ms(read_end, release_end));
        return result;
    }

    CachedStepGraph & ensure_step_graph(int64_t capacity) {
        const auto build_start = std::chrono::steady_clock::now();
        const bool rebuilt =
            step_graph == nullptr || step_graph->capacity() < capacity;
        if (rebuilt) {
            step_graph = std::make_unique<CachedStepGraph>(
                assets->config, weights, *constants, backend, threads, capacity);
        }
        debug::trace_log_scalar("outetts.llama.step.graph_rebuilt", rebuilt);
        debug::trace_log_scalar("outetts.llama.step.graph_reused", !rebuilt);
        debug::trace_log_scalar(
            "outetts.llama.step.capacity", step_graph->capacity());
        debug::timing_log_scalar(
            "outetts.llama.step.graph_build_ms",
            rebuilt ? debug::elapsed_ms(build_start) : 0.0);
        return *step_graph;
    }

    int64_t release_cached_step_graph() {
        if (step_graph == nullptr) {
            return 0;
        }
        const int64_t capacity = step_graph->capacity();
        step_graph.reset();
        return capacity;
    }

    std::shared_ptr<const OuteTTSAssets> assets;
    ggml_backend_t backend = nullptr;
    int threads = 1;
    sampling::TorchCudaSamplingPolicy sampling_policy;
    ModelWeights weights;
    std::unique_ptr<common::ConstantTensorCache> constants;
    std::unique_ptr<CachedStepGraph> step_graph;
};

OuteTTSLlamaRuntime::OuteTTSLlamaRuntime(
    std::shared_ptr<const OuteTTSAssets> assets,
    core::BackendType backend_type,
    int device,
    int threads,
    size_t weight_context_bytes,
    size_t constant_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(assets), backend_type, device, threads,
          weight_context_bytes, constant_context_bytes, weight_storage_type)) {}

OuteTTSLlamaRuntime::~OuteTTSLlamaRuntime() = default;

std::string_view outetts_stop_reason_name(OuteTTSStopReason reason) noexcept {
    switch (reason) {
    case OuteTTSStopReason::Eos:
        return "eos";
    case OuteTTSStopReason::AudioEnd:
        return "audio_end";
    case OuteTTSStopReason::MaxTokens:
        return "max_tokens";
    case OuteTTSStopReason::ContextLimit:
        return "context_limit";
    }
    return "unknown";
}

OuteTTSGenerateResult OuteTTSLlamaRuntime::generate(
    const std::vector<int32_t> & prompt,
    const OuteTTSGenerateOptions & options,
    int32_t eos_id,
    int32_t audio_end_id) const {
    if (prompt.empty()) {
        throw std::runtime_error("OuteTTS generation requires a prompt");
    }
    if (options.max_new_tokens <= 0 || options.repetition_window < 0 || options.repetition_penalty <= 0.0F) {
        throw std::runtime_error("invalid OuteTTS generation options");
    }
    const auto total_start = std::chrono::steady_clock::now();
    std::vector<int32_t> all = prompt;
    OuteTTSGenerateResult result;
    result.tokens.reserve(static_cast<size_t>(options.max_new_tokens));
    std::mt19937 rng(options.seed);
    OuteTTSGenerateOptions sampling_options = options;
    auto prefill = impl_->prefill(prompt);
    const int64_t capacity = std::min<int64_t>(
        impl_->assets->generation.max_length,
        static_cast<int64_t>(prompt.size()) + options.max_new_tokens);
    constexpr int64_t kStepGraphCapacityQuantum = 256;
    const int64_t reusable_capacity = std::min<int64_t>(
        impl_->assets->generation.max_length,
        ((capacity + kStepGraphCapacityQuantum - 1) /
         kStepGraphCapacityQuantum) *
            kStepGraphCapacityQuantum);
    auto & step = impl_->ensure_step_graph(reusable_capacity);
    step.import_state(prefill.state);
    std::vector<float> logits = std::move(prefill.logits);
    double sample_ms = 0.0;
    double cached_step_compute_ms = 0.0;
    SamplingScratch sampling_scratch;
    for (int64_t i = 0; i < options.max_new_tokens; ++i) {
        if (static_cast<int64_t>(all.size()) >= impl_->assets->generation.max_length) {
            result.stop_reason = OuteTTSStopReason::ContextLimit;
            break;
        }
        const auto sample_start = std::chrono::steady_clock::now();
        apply_repetition_penalty(
            logits, all, options.repetition_window, options.repetition_penalty,
            sampling_scratch.repeated_ids);
        const int32_t token = sample_token(
            logits, sampling_options, rng,
            impl_->sampling_policy, static_cast<uint64_t>(i),
            sampling_scratch);
        sample_ms += debug::elapsed_ms(sample_start);
        result.tokens.push_back(token);
        if (token == eos_id) {
            result.stop_reason = OuteTTSStopReason::Eos;
            break;
        }
        if (token == audio_end_id) {
            result.stop_reason = OuteTTSStopReason::AudioEnd;
            break;
        }
        // Do not execute one unused cached step after consuming the caller's
        // final generation token.
        if (i + 1 >= options.max_new_tokens) {
            break;
        }
        all.push_back(token);
        const auto step_start = std::chrono::steady_clock::now();
        step.run(token, logits);
        cached_step_compute_ms += debug::elapsed_ms(step_start);
    }
    debug::trace_log_scalar(
        "outetts.llama.generated_tokens",
        static_cast<int64_t>(result.tokens.size()));
    debug::trace_log_scalar("outetts.llama.stop_reason",
                            outetts_stop_reason_name(result.stop_reason));
    debug::timing_log_scalar("outetts.llama.sample_ms", sample_ms);
    debug::timing_log_scalar("outetts.llama.cached_step_compute_ms",
                             cached_step_compute_ms);
    debug::timing_log_scalar("outetts.llama.generate_total_ms",
                             debug::elapsed_ms(total_start));
    return result;
}

int64_t OuteTTSLlamaRuntime::release_cached_step_graph() {
    return impl_->release_cached_step_graph();
}

}  // namespace engine::models::outetts
