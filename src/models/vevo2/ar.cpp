#include "engine/models/vevo2/ar.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/framework/sampling/torch_random.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::vevo2 {
namespace {

namespace assets = engine::assets;
namespace core = engine::core;
namespace modules = engine::modules;
namespace runtime = engine::runtime;

using Clock = std::chrono::steady_clock;

constexpr int64_t kMinNewTokens = 15;

struct SortedSamplerScore {
    size_t index = 0;
    float score = 0.0F;
    float weight = 0.0F;
};

struct SamplerScratch {
    std::vector<uint32_t> seen_marks;
    uint32_t seen_stamp = 1;
    std::vector<float> scores;
    std::vector<float> threshold_values;
    std::vector<SortedSamplerScore> sorted;
};

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

std::shared_ptr<const Vevo2Assets> require_assets(std::shared_ptr<const Vevo2Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Vevo2 AR runtime requires assets");
    }
    return assets;
}

int64_t ar_head_dim(const Vevo2ARConfig & config) {
    if (config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 ||
        config.hidden_size % config.num_attention_heads != 0) {
        throw std::runtime_error("Vevo2 AR config has invalid attention dimensions");
    }
    if (config.num_attention_heads % config.num_key_value_heads != 0) {
        throw std::runtime_error("Vevo2 AR attention heads must be divisible by KV heads");
    }
    return config.hidden_size / config.num_attention_heads;
}

}  // namespace

struct Vevo2ARWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    struct Layer {
        modules::NormWeights input_norm;
        modules::LinearWeights q_proj;
        modules::LinearWeights k_proj;
        modules::LinearWeights v_proj;
        modules::LinearWeights o_proj;
        modules::NormWeights post_norm;
        modules::LinearWeights gate_proj;
        modules::LinearWeights up_proj;
        modules::LinearWeights down_proj;
    };
    std::vector<Layer> layers;
    modules::NormWeights norm;
    core::TensorValue lm_head;
};

namespace {

struct Vevo2ARLayerOutput {
    core::TensorValue output;
    core::TensorValue key;
    core::TensorValue value;
};

struct Vevo2ARPrefillOutput {
    std::vector<float> logits;
    runtime::TransformerKVState kv_state;
};

using TorchCudaSamplingPolicy = engine::sampling::TorchCudaSamplingPolicy;

std::shared_ptr<const Vevo2ARWeights> load_ar_weights(
    const Vevo2Assets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type,
    const assets::TensorSource & source) {
    const auto & config = assets.config.ar;
    const int64_t dim = ar_head_dim(config);
    auto weights = std::make_shared<Vevo2ARWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "vevo2.ar.weights",
        weight_context_bytes);
    weights->token_embedding = weights->store->load_tensor(
        source,
        "model.embed_tokens.weight",
        storage_type,
        {config.vocab_size, config.hidden_size});
    weights->layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer);
        Vevo2ARWeights::Layer layer_weights;
        layer_weights.input_norm = modules::binding::norm_weight_from_source(*weights->store, source, prefix + ".input_layernorm", config.hidden_size);
        layer_weights.q_proj = modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".self_attn.q_proj",
            storage_type,
            config.num_attention_heads * dim,
            config.hidden_size,
            true);
        layer_weights.k_proj = modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".self_attn.k_proj",
            storage_type,
            config.num_key_value_heads * dim,
            config.hidden_size,
            true);
        layer_weights.v_proj = modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".self_attn.v_proj",
            storage_type,
            config.num_key_value_heads * dim,
            config.hidden_size,
            true);
        layer_weights.o_proj = modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".self_attn.o_proj",
            storage_type,
            config.hidden_size,
            config.num_attention_heads * dim,
            false);
        layer_weights.post_norm = modules::binding::norm_weight_from_source(*weights->store, source, prefix + ".post_attention_layernorm", config.hidden_size);
        layer_weights.gate_proj = modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".mlp.gate_proj",
            storage_type,
            config.intermediate_size,
            config.hidden_size,
            false);
        layer_weights.up_proj = modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".mlp.up_proj",
            storage_type,
            config.intermediate_size,
            config.hidden_size,
            false);
        layer_weights.down_proj = modules::binding::linear_from_source(
            *weights->store,
            source,
            prefix + ".mlp.down_proj",
            storage_type,
            config.hidden_size,
            config.intermediate_size,
            false);
        weights->layers.push_back(std::move(layer_weights));
    }
    weights->norm = modules::binding::norm_weight_from_source(*weights->store, source, "model.norm", config.hidden_size);
    if (!config.tie_word_embeddings) {
        throw std::runtime_error("Vevo2 AR currently expects tied token and LM embeddings");
    }
    weights->lm_head = weights->token_embedding;
    weights->store->upload();
    return weights;
}

core::TensorValue reshape_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t heads, int64_t dim) {
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue repeat_kv_heads(core::ModuleBuildContext & ctx, const core::TensorValue & input, int64_t repeats) {
    if (repeats == 1) {
        return input;
    }
    std::vector<core::TensorValue> heads;
    heads.reserve(static_cast<size_t>(input.shape.dims[1] * repeats));
    for (int64_t head = 0; head < input.shape.dims[1]; ++head) {
        auto one = modules::SliceModule({1, head, 1}).build(ctx, input);
        for (int64_t rep = 0; rep < repeats; ++rep) {
            heads.push_back(one);
        }
    }
    auto output = heads.front();
    for (size_t i = 1; i < heads.size(); ++i) {
        output = modules::ConcatModule({1}).build(ctx, output, heads[i]);
    }
    return output;
}

core::TensorValue attention_from_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const std::optional<core::TensorValue> & attention_mask = std::nullopt) {
    const modules::MatMulModule matmul;
    auto scores = matmul.build(ctx, q_heads, modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    core::TensorValue attn;
    if (attention_mask.has_value()) {
        scores = core::ensure_backend_addressable_layout(ctx, scores);
        attn = core::wrap_tensor(
            ggml_soft_max_ext(
                ctx.ggml,
                scores.tensor,
                attention_mask->tensor,
                1.0F / std::sqrt(static_cast<float>(dim)),
                0.0F),
            scores.shape,
            GGML_TYPE_F32);
    } else {
        scores = core::wrap_tensor(
            ggml_scale(ctx.ggml, scores.tensor, 1.0F / std::sqrt(static_cast<float>(dim))),
            scores.shape,
            GGML_TYPE_F32);
        scores = core::wrap_tensor(ggml_diag_mask_inf(ctx.ggml, scores.tensor, 0), scores.shape, GGML_TYPE_F32);
        scores = core::ensure_backend_addressable_layout(ctx, scores);
        attn = core::wrap_tensor(ggml_soft_max(ctx.ggml, scores.tensor), scores.shape, GGML_TYPE_F32);
    }
    return matmul.build(ctx, attn, v_heads);
}

core::TensorValue cache_view(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    int64_t start,
    int64_t steps,
    int64_t heads,
    int64_t dim) {
    if (start < 0 || steps <= 0 || start + steps > cache.shape.dims[1]) {
        throw std::runtime_error("Vevo2 AR cache view range is invalid");
    }
    return core::wrap_tensor(
        ggml_view_4d(
            ctx.ggml,
            cache.tensor,
            dim,
            heads,
            steps,
            1,
            cache.tensor->nb[1],
            cache.tensor->nb[2],
            cache.tensor->nb[3],
            static_cast<size_t>(start) * cache.tensor->nb[2]),
        core::TensorShape::from_dims({1, steps, heads, dim}),
        GGML_TYPE_F32);
}

core::TensorValue flash_attention_from_grouped_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const core::TensorValue & attention_mask) {
    if (!core::has_backend_addressable_layout(q_heads.tensor) ||
        !core::has_backend_addressable_layout(k_heads.tensor) ||
        !core::has_backend_addressable_layout(v_heads.tensor)) {
        throw std::runtime_error("Vevo2 AR flash attention expects contiguous Q/K/V heads");
    }
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q_heads.tensor,
        k_heads.tensor,
        v_heads.tensor,
        attention_mask.tensor,
        1.0F / std::sqrt(static_cast<float>(dim)),
        0.0F,
        0.0F);
    ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
    return core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q_heads.shape.dims[0], q_heads.shape.dims[2], q_heads.shape.dims[1], dim}),
        GGML_TYPE_F32);
}

core::TensorValue mlp(core::ModuleBuildContext & ctx, const core::TensorValue & input, const Vevo2ARWeights::Layer & weights, const Vevo2ARConfig & config) {
    auto gate = modules::LinearModule({config.hidden_size, config.intermediate_size, weights.gate_proj.bias.has_value()})
                    .build(ctx, input, weights.gate_proj);
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule({config.hidden_size, config.intermediate_size, weights.up_proj.bias.has_value()})
                  .build(ctx, input, weights.up_proj);
    return modules::LinearModule({config.intermediate_size, config.hidden_size, weights.down_proj.bias.has_value()})
        .build(ctx, modules::MulModule{}.build(ctx, gate, up), weights.down_proj);
}

Vevo2ARLayerOutput decoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const Vevo2ARWeights::Layer & weights,
    const Vevo2ARConfig & config) {
    const int64_t dim = ar_head_dim(config);
    const int64_t kv_repeats = config.num_attention_heads / config.num_key_value_heads;
    auto x_norm = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                      .build(ctx, input, weights.input_norm);
    auto q = modules::LinearModule({config.hidden_size, config.num_attention_heads * dim, weights.q_proj.bias.has_value()})
                 .build(ctx, x_norm, weights.q_proj);
    auto k = modules::LinearModule({config.hidden_size, config.num_key_value_heads * dim, weights.k_proj.bias.has_value()})
                 .build(ctx, x_norm, weights.k_proj);
    auto v = modules::LinearModule({config.hidden_size, config.num_key_value_heads * dim, weights.v_proj.bias.has_value()})
                 .build(ctx, x_norm, weights.v_proj);
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, reshape_heads(ctx, q, config.num_attention_heads, dim), positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, reshape_heads(ctx, k, config.num_key_value_heads, dim), positions);
    v = reshape_heads(ctx, v, config.num_key_value_heads, dim);

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = repeat_kv_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k), kv_repeats);
    auto v_heads = repeat_kv_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v), kv_repeats);
    auto context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim);
    context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_attention_heads * dim}));
    auto x = modules::AddModule{}.build(
        ctx,
        input,
        modules::LinearModule({config.num_attention_heads * dim, config.hidden_size, weights.o_proj.bias.has_value()})
            .build(ctx, context, weights.o_proj));
    auto ff_in = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                     .build(ctx, x, weights.post_norm);
    return {modules::AddModule{}.build(ctx, x, mlp(ctx, ff_in, weights, config)), k, v};
}

Vevo2ARLayerOutput decoder_layer_with_static_cache_tail(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const Vevo2ARWeights::Layer & weights,
    const Vevo2ARConfig & config,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const core::TensorValue & attention_mask) {
    const int64_t dim = ar_head_dim(config);
    const int64_t scratch_slot = cache_key.shape.dims[1] - 1;
    auto x_norm = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                      .build(ctx, input, weights.input_norm);
    auto q = modules::LinearModule({config.hidden_size, config.num_attention_heads * dim, weights.q_proj.bias.has_value()})
                 .build(ctx, x_norm, weights.q_proj);
    auto k = modules::LinearModule({config.hidden_size, config.num_key_value_heads * dim, weights.k_proj.bias.has_value()})
                 .build(ctx, x_norm, weights.k_proj);
    auto v = modules::LinearModule({config.hidden_size, config.num_key_value_heads * dim, weights.v_proj.bias.has_value()})
                 .build(ctx, x_norm, weights.v_proj);
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, reshape_heads(ctx, q, config.num_attention_heads, dim), positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, reshape_heads(ctx, k, config.num_key_value_heads, dim), positions);
    v = reshape_heads(ctx, v, config.num_key_value_heads, dim);

    auto key_tail = cache_view(ctx, cache_key, scratch_slot, 1, config.num_key_value_heads, dim);
    auto value_tail = cache_view(ctx, cache_value, scratch_slot, 1, config.num_key_value_heads, dim);
    ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, k.tensor, key_tail.tensor));
    ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, v.tensor, value_tail.tensor));

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, cache_key.shape.rank}).build(ctx, cache_key);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, cache_value.shape.rank}).build(ctx, cache_value);
    k_heads = core::wrap_tensor(ggml_cont(ctx.ggml, k_heads.tensor), k_heads.shape, k_heads.type);
    v_heads = core::wrap_tensor(ggml_cont(ctx.ggml, v_heads.tensor), v_heads.shape, v_heads.type);
    auto context = flash_attention_from_grouped_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({1, 1, config.num_attention_heads * dim}));
    auto x = modules::AddModule{}.build(
        ctx,
        input,
        modules::LinearModule({config.num_attention_heads * dim, config.hidden_size, weights.o_proj.bias.has_value()})
            .build(ctx, context, weights.o_proj));
    auto ff_in = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                     .build(ctx, x, weights.post_norm);
    return {modules::AddModule{}.build(ctx, x, mlp(ctx, ff_in, weights, config)), k, v};
}

bool is_eos(int32_t token, int32_t tokenizer_eos) {
    return token == tokenizer_eos;
}

void apply_repetition_penalty(
    std::vector<float> & logits,
    const std::vector<int32_t> & history,
    float penalty,
    SamplerScratch & scratch) {
    if (penalty == 1.0F) {
        return;
    }
    if (!(penalty > 0.0F)) {
        throw std::runtime_error("Vevo2 AR repetition penalty must be positive");
    }
    if (scratch.seen_marks.size() < logits.size()) {
        scratch.seen_marks.assign(logits.size(), 0);
        scratch.seen_stamp = 1;
    } else if (scratch.seen_stamp == std::numeric_limits<uint32_t>::max()) {
        std::fill(scratch.seen_marks.begin(), scratch.seen_marks.end(), 0);
        scratch.seen_stamp = 1;
    }
    const uint32_t stamp = scratch.seen_stamp++;
    for (const int32_t token : history) {
        if (token < 0 || static_cast<size_t>(token) >= logits.size()) {
            continue;
        }
        const size_t index = static_cast<size_t>(token);
        if (scratch.seen_marks[index] == stamp) {
            continue;
        }
        scratch.seen_marks[index] = stamp;
        float & value = logits[index];
        value = value < 0.0F ? value * penalty : value / penalty;
    }
}

void mask_min_new_token_eos(std::vector<float> & logits, int generated_tokens, int32_t eos_token) {
    if (generated_tokens >= kMinNewTokens || eos_token < 0 || static_cast<size_t>(eos_token) >= logits.size()) {
        return;
    }
    logits[static_cast<size_t>(eos_token)] = -std::numeric_limits<float>::infinity();
}

int32_t sample_token(
    const std::vector<float> & logits,
    int top_k,
    float top_p,
    float temperature,
    uint64_t seed,
    int step,
    const TorchCudaSamplingPolicy & sampling_policy,
    SamplerScratch & scratch) {
    if (!(temperature > 0.0F)) {
        throw std::runtime_error("Vevo2 AR sampler temperature must be positive");
    }
    if (logits.empty()) {
        throw std::runtime_error("Vevo2 AR sampler requires logits");
    }
    if (top_k == 1) {
        return static_cast<int32_t>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }
    auto & scores = scratch.scores;
    scores.resize(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        scores[i] = logits[i] / temperature;
    }

    if (top_k > 0 && static_cast<size_t>(top_k) < scores.size()) {
        auto & threshold_values = scratch.threshold_values;
        threshold_values.assign(scores.begin(), scores.end());
        auto nth = threshold_values.begin() + static_cast<std::ptrdiff_t>(top_k - 1);
        std::nth_element(threshold_values.begin(), nth, threshold_values.end(), std::greater<float>());
        const float threshold = *nth;
        for (float & score : scores) {
            if (score < threshold) {
                score = -std::numeric_limits<float>::infinity();
            }
        }
    }

    float max_score = -std::numeric_limits<float>::infinity();
    for (const float score : scores) {
        if (std::isfinite(score)) {
            max_score = std::max(max_score, score);
        }
    }
    if (!std::isfinite(max_score)) {
        throw std::runtime_error("Vevo2 AR sampler probabilities are invalid");
    }

    if (top_p > 0.0F && top_p < 1.0F) {
        auto & sorted = scratch.sorted;
        sorted.clear();
        sorted.reserve(scores.size());
        float total = 0.0F;
        for (size_t index = 0; index < scores.size(); ++index) {
            if (!std::isfinite(scores[index])) {
                continue;
            }
            const float weight = std::exp(scores[index] - max_score);
            sorted.push_back({index, scores[index], weight});
            total += weight;
        }
        if (!(total > 0.0F) || !std::isfinite(total)) {
            throw std::runtime_error("Vevo2 AR sampler top-p probabilities are invalid");
        }
        std::sort(sorted.begin(), sorted.end(), [](const SortedSamplerScore & lhs, const SortedSamplerScore & rhs) {
            if (lhs.score == rhs.score) {
                return lhs.index < rhs.index;
            }
            return lhs.score < rhs.score;
        });
        float cumulative = 0.0F;
        const float remove_mass = 1.0F - top_p;
        const size_t keep_from = sorted.empty() ? 0 : sorted.size() - 1;
        for (size_t i = 0; i < sorted.size(); ++i) {
            cumulative += sorted[i].weight / total;
            if (i < keep_from && cumulative <= remove_mass) {
                scores[sorted[i].index] = -std::numeric_limits<float>::infinity();
            }
        }
    }

    max_score = -std::numeric_limits<float>::infinity();
    for (const float score : scores) {
        if (std::isfinite(score)) {
            max_score = std::max(max_score, score);
        }
    }
    if (!std::isfinite(max_score)) {
        throw std::runtime_error("Vevo2 AR sampler kept probabilities are invalid");
    }
    float kept_total = 0.0F;
    for (const float score : scores) {
        if (std::isfinite(score)) {
            kept_total += std::exp(score - max_score);
        }
    }
    if (!(kept_total > 0.0F) || !std::isfinite(kept_total)) {
        throw std::runtime_error("Vevo2 AR sampler kept probabilities are invalid");
    }
    double best_rank = -std::numeric_limits<double>::infinity();
    int32_t best_token = -1;
    for (size_t index = 0; index < scores.size(); ++index) {
        if (!std::isfinite(scores[index])) {
            continue;
        }
        const float probability = std::exp(scores[index] - max_score) / kept_total;
        if (!(probability > 0.0F)) {
            continue;
        }
        const float exponential = engine::sampling::torch_cuda_tensor_iterator_exponential_element(
            seed,
            static_cast<uint64_t>(scores.size()),
            static_cast<uint64_t>(index),
            static_cast<uint64_t>(step),
            sampling_policy.multiprocessor_count,
            sampling_policy.max_threads_per_multiprocessor);
        const double rank = static_cast<double>(probability) / static_cast<double>(exponential);
        if (rank > best_rank) {
            best_rank = rank;
            best_token = static_cast<int32_t>(index);
        }
    }
    if (best_token < 0) {
        throw std::runtime_error("Vevo2 AR sampler has no kept token");
    }
    return best_token;
}

}  // namespace

struct Vevo2ARPrefillGraph {
    Vevo2ARPrefillGraph(
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        size_t graph_context_bytes,
        std::shared_ptr<const Vevo2ARWeights> weights,
        const Vevo2ARConfig & config,
        int64_t prompt_steps)
        : backend(backend),
          backend_type(backend_type),
          threads(threads),
          weights(std::move(weights)),
          prompt_steps(prompt_steps) {
        if (backend == nullptr || this->weights == nullptr) {
            throw std::runtime_error("Vevo2 AR prefill graph requires backend and weights");
        }
        if (prompt_steps <= 0) {
            throw std::runtime_error("Vevo2 AR prefill graph requires positive prompt length");
        }
        ggml_init_params params{graph_context_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize Vevo2 AR prefill graph context");
        }
        core::ModuleBuildContext build_ctx{ctx.get(), "vevo2.ar.prefill", backend_type};
        token_ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, prompt_steps);
        auto ids = core::wrap_tensor(token_ids, core::TensorShape::from_dims({prompt_steps}), GGML_TYPE_I32);
        auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                     .build(build_ctx, ids, this->weights->token_embedding);
        x = core::reshape_tensor(build_ctx, x, core::TensorShape::from_dims({1, prompt_steps, config.hidden_size}));
        positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, prompt_steps);
        auto position_values = core::wrap_tensor(positions, core::TensorShape::from_dims({prompt_steps}), GGML_TYPE_I32);
        for (const auto & layer : this->weights->layers) {
            auto out = decoder_layer(build_ctx, x, position_values, layer, config);
            x = out.output;
            keys.push_back(out.key.tensor);
            values.push_back(out.value.tensor);
        }
        x = modules::SliceModule({1, prompt_steps - 1, 1}).build(build_ctx, x);
        x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                .build(build_ctx, x, this->weights->norm);
        logits = modules::LinearModule({config.hidden_size, config.vocab_size, false})
                     .build(build_ctx, x, {this->weights->lm_head, std::nullopt})
                     .tensor;
        ggml_set_output(logits);
        graph = ggml_new_graph_custom(ctx.get(), 65536, false);
        ggml_build_forward_expand(graph, logits);
        buffer = ggml_backend_alloc_ctx_tensors(ctx.get(), backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate Vevo2 AR prefill graph");
        }
        std::vector<int32_t> pos(static_cast<size_t>(prompt_steps), 0);
        for (int64_t i = 0; i < prompt_steps; ++i) {
            pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));
    }

    ~Vevo2ARPrefillGraph() {
        engine::core::release_backend_graph_resources(backend, graph);
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }

    bool matches(const Vevo2ARWeights & other_weights, int64_t steps) const noexcept {
        return weights.get() == &other_weights && prompt_steps == steps;
    }

    Vevo2ARPrefillOutput run(const std::vector<int32_t> & ids, const Vevo2ARConfig & config) {
        if (static_cast<int64_t>(ids.size()) != prompt_steps) {
            throw std::runtime_error("Vevo2 AR prefill token id count mismatch");
        }
        ggml_backend_tensor_set(token_ids, ids.data(), 0, ids.size() * sizeof(int32_t));
        core::set_backend_threads(backend, threads);
        const ggml_status status = engine::core::compute_backend_graph(backend, graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Vevo2 AR prefill graph compute failed");
        }
        Vevo2ARPrefillOutput out;
        out.logits.resize(static_cast<size_t>(config.vocab_size));
        ggml_backend_tensor_get(logits, out.logits.data(), 0, out.logits.size() * sizeof(float));
        out.kv_state.current_end = prompt_steps;
        out.kv_state.layers.resize(keys.size());
        const size_t layer_values = static_cast<size_t>(prompt_steps * config.num_key_value_heads * ar_head_dim(config));
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

    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    int threads = 1;
    std::shared_ptr<const Vevo2ARWeights> weights;
    int64_t prompt_steps = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    ggml_tensor * token_ids = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * logits = nullptr;
    std::vector<ggml_tensor *> keys;
    std::vector<ggml_tensor *> values;
    ggml_cgraph * graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
};

struct Vevo2ARDecodeGraph {
    Vevo2ARDecodeGraph(
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        size_t graph_context_bytes,
        std::shared_ptr<const Vevo2ARWeights> weights,
        const Vevo2ARConfig & config,
        int64_t cache_steps)
        : backend(backend),
          backend_type(backend_type),
          threads(threads),
          weights(std::move(weights)),
          cache_steps(cache_steps) {
        if (backend == nullptr || this->weights == nullptr) {
            throw std::runtime_error("Vevo2 AR decode graph requires backend and weights");
        }
        if (cache_steps <= 0) {
            throw std::runtime_error("Vevo2 AR decode graph requires positive cache length");
        }
        const int64_t dim = ar_head_dim(config);
        ggml_init_params params{graph_context_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize Vevo2 AR decode graph context");
        }
        core::ModuleBuildContext build_ctx{ctx.get(), "vevo2.ar.decode", backend_type};
        token_id = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
        auto token = core::wrap_tensor(token_id, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                     .build(build_ctx, token, this->weights->token_embedding);
        x = core::reshape_tensor(build_ctx, x, core::TensorShape::from_dims({1, 1, config.hidden_size}));
        positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
        auto position_values = core::wrap_tensor(positions, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        attention_mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, cache_steps + 1, 1, 1, 1);
        auto mask = core::wrap_tensor(
            attention_mask,
            core::TensorShape::from_dims({1, 1, 1, cache_steps + 1}),
            GGML_TYPE_F16);
        graph = ggml_new_graph_custom(ctx.get(), 65536, false);
        std::vector<core::TensorValue> cache_keys;
        std::vector<core::TensorValue> cache_values;
        for (const auto & layer : this->weights->layers) {
            cache_keys.push_back(core::make_tensor(
                build_ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps + 1, config.num_key_value_heads, dim})));
            cache_values.push_back(core::make_tensor(
                build_ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps + 1, config.num_key_value_heads, dim})));
            auto out = decoder_layer_with_static_cache_tail(
                build_ctx,
                graph,
                x,
                position_values,
                layer,
                config,
                cache_keys.back(),
                cache_values.back(),
                mask);
            x = out.output;
            key_sources.push_back(ggml_view_1d(ctx.get(), out.key.tensor, config.num_key_value_heads * dim, 0));
            value_sources.push_back(ggml_view_1d(ctx.get(), out.value.tensor, config.num_key_value_heads * dim, 0));
        }
        step_cache = runtime::TransformerKVCache(
            cache_steps + 1,
            config.num_key_value_heads * dim,
            std::move(cache_keys),
            std::move(cache_values));
        build_transfer_views(config.num_key_value_heads * dim);
        x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                .build(build_ctx, x, this->weights->norm);
        logits = modules::LinearModule({config.hidden_size, config.vocab_size, false})
                     .build(build_ctx, x, {this->weights->lm_head, std::nullopt})
                     .tensor;
        ggml_set_output(logits);
        ggml_build_forward_expand(graph, logits);
        buffer = ggml_backend_alloc_ctx_tensors(ctx.get(), backend);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate Vevo2 AR decode graph");
        }
        attention_mask_values.assign(static_cast<size_t>(cache_steps + 1), ggml_fp32_to_fp16(-INFINITY));
    }

    ~Vevo2ARDecodeGraph() {
        engine::core::release_backend_graph_resources(backend, graph);
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }

    bool can_run(const Vevo2ARWeights & other_weights, int64_t required_steps) const noexcept {
        return weights.get() == &other_weights && cache_steps >= required_steps;
    }

    void import_state(const runtime::TransformerKVState & state) {
        step_cache.import_state(state);
    }

    std::vector<float> run_step(int32_t token, const Vevo2ARConfig & config) {
        if (step_cache.valid_steps() >= cache_steps) {
            throw std::runtime_error("Vevo2 AR decode cache exhausted");
        }
        ggml_backend_tensor_set(token_id, &token, 0, sizeof(int32_t));
        const int32_t position = static_cast<int32_t>(step_cache.current_end());
        ggml_backend_tensor_set(positions, &position, 0, sizeof(int32_t));
        const auto masked = ggml_fp32_to_fp16(-INFINITY);
        const auto visible = ggml_fp32_to_fp16(0.0F);
        std::fill(attention_mask_values.begin(), attention_mask_values.end(), masked);
        for (int64_t i = 0; i < step_cache.valid_steps(); ++i) {
            attention_mask_values[static_cast<size_t>(i)] = visible;
        }
        attention_mask_values[static_cast<size_t>(cache_steps)] = visible;
        ggml_backend_tensor_set(
            attention_mask,
            attention_mask_values.data(),
            0,
            attention_mask_values.size() * sizeof(ggml_fp16_t));
        core::set_backend_threads(backend, threads);
        const ggml_status status = engine::core::compute_backend_graph(backend, graph);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Vevo2 AR decode graph compute failed");
        }
        std::vector<float> out(static_cast<size_t>(config.vocab_size));
        ggml_backend_tensor_get(logits, out.data(), 0, out.size() * sizeof(float));
        const size_t dst_slot = static_cast<size_t>(step_cache.valid_steps());
        for (size_t layer = 0; layer < key_sources.size(); ++layer) {
            ggml_backend_tensor_copy(key_sources[layer], key_destinations[dst_slot][layer]);
            ggml_backend_tensor_copy(value_sources[layer], value_destinations[dst_slot][layer]);
        }
        step_cache.advance_after_direct_append(1);
        return out;
    }

    void build_transfer_views(int64_t step_elems) {
        key_destinations.assign(static_cast<size_t>(cache_steps), {});
        value_destinations.assign(static_cast<size_t>(cache_steps), {});
        for (int64_t slot = 0; slot < cache_steps; ++slot) {
            const size_t byte_offset = static_cast<size_t>(slot * step_elems) * sizeof(float);
            auto & key_slot = key_destinations[static_cast<size_t>(slot)];
            auto & value_slot = value_destinations[static_cast<size_t>(slot)];
            key_slot.reserve(key_sources.size());
            value_slot.reserve(value_sources.size());
            for (size_t layer = 0; layer < key_sources.size(); ++layer) {
                key_slot.push_back(ggml_view_1d(ctx.get(), step_cache.key_tensor(layer).tensor, step_elems, byte_offset));
                value_slot.push_back(ggml_view_1d(ctx.get(), step_cache.value_tensor(layer).tensor, step_elems, byte_offset));
            }
        }
    }

    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    int threads = 1;
    std::shared_ptr<const Vevo2ARWeights> weights;
    int64_t cache_steps = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    ggml_tensor * token_id = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * attention_mask = nullptr;
    ggml_tensor * logits = nullptr;
    std::vector<ggml_tensor *> key_sources;
    std::vector<ggml_tensor *> value_sources;
    std::vector<std::vector<ggml_tensor *>> key_destinations;
    std::vector<std::vector<ggml_tensor *>> value_destinations;
    std::vector<ggml_fp16_t> attention_mask_values;
    runtime::TransformerKVCache step_cache;
    ggml_cgraph * graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
};

Vevo2AutoregressiveRuntime::Vevo2AutoregressiveRuntime(
    std::shared_ptr<const Vevo2Assets> assets,
    core::ExecutionContext & execution_context,
    size_t weight_context_bytes,
    size_t prefill_graph_context_bytes,
    size_t decode_graph_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : assets_(require_assets(std::move(assets))),
      execution_context_(execution_context),
      prefill_graph_context_bytes_(prefill_graph_context_bytes),
      decode_graph_context_bytes_(decode_graph_context_bytes),
      tokenizer_(assets_),
      weight_source_(assets_->ar_weights),
      weights_(load_ar_weights(
          *assets_,
          execution_context_.backend(),
          execution_context_.backend_type(),
          weight_context_bytes,
          weight_storage_type,
          *weight_source_)) {
    weight_source_->release_storage();
}

Vevo2AutoregressiveRuntime::~Vevo2AutoregressiveRuntime() = default;

Vevo2TokenSequence Vevo2AutoregressiveRuntime::generate_content_style(
    const Vevo2PromptParts & prompt,
    const Vevo2GenerationOptions & generation) const {
    const auto total_start = Clock::now();
    const auto tokenize_start = Clock::now();
    const auto tokenized = tokenizer_.tokenize_prompt(prompt.full_prompt);
    const double tokenize_ms = engine::debug::elapsed_ms(tokenize_start);
    last_prompt_tokens_ = static_cast<int64_t>(tokenized.input_ids.size());
    if (last_prompt_tokens_ <= 0) {
        throw std::runtime_error("Vevo2 AR requires non-empty prompt tokens");
    }
    double prefill_graph_build_ms = 0.0;
    if (prefill_graph_ == nullptr || !prefill_graph_->matches(*weights_, last_prompt_tokens_)) {
        const auto build_start = Clock::now();
        prefill_graph_ = std::make_unique<Vevo2ARPrefillGraph>(
            execution_context_.backend(),
            execution_context_.backend_type(),
            execution_context_.config().threads,
            prefill_graph_context_bytes_,
            weights_,
            assets_->config.ar,
            last_prompt_tokens_);
        prefill_graph_build_ms = engine::debug::elapsed_ms(build_start);
    }
    const int64_t required_cache_steps = last_prompt_tokens_ + generation.max_new_tokens;
    double decode_graph_build_ms = 0.0;
    if (decode_graph_ == nullptr || !decode_graph_->can_run(*weights_, required_cache_steps)) {
        const auto build_start = Clock::now();
        decode_graph_ = std::make_unique<Vevo2ARDecodeGraph>(
            execution_context_.backend(),
            execution_context_.backend_type(),
            execution_context_.config().threads,
            decode_graph_context_bytes_,
            weights_,
            assets_->config.ar,
            required_cache_steps);
        decode_graph_build_ms = engine::debug::elapsed_ms(build_start);
    }

    const auto prefill_start = Clock::now();
    auto prefill = prefill_graph_->run(tokenized.input_ids, assets_->config.ar);
    const double prefill_run_ms = engine::debug::elapsed_ms(prefill_start);
    const auto import_start = Clock::now();
    decode_graph_->import_state(prefill.kv_state);
    const double import_ms = engine::debug::elapsed_ms(import_start);
    std::vector<int32_t> history = tokenized.input_ids;
    std::vector<int32_t> generated_ids;
    generated_ids.reserve(static_cast<size_t>(generation.max_new_tokens));
    const auto rng_start = Clock::now();
    const TorchCudaSamplingPolicy sampling_policy = engine::sampling::resolve_torch_cuda_sampling_policy(
        execution_context_.backend_type(),
        execution_context_.config().device,
        "vevo2.ar.cuda_sampling_policy",
        "Vevo2",
        engine::sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault);
    const double rng_ms = engine::debug::elapsed_ms(rng_start);
    std::vector<float> logits = std::move(prefill.logits);
    double sampling_ms = 0.0;
    double decode_step_ms = 0.0;
    SamplerScratch sampler_scratch;
    for (int step = 0; step < generation.max_new_tokens; ++step) {
        const auto sample_start = Clock::now();
        apply_repetition_penalty(logits, history, generation.repetition_penalty, sampler_scratch);
        mask_min_new_token_eos(logits, static_cast<int>(generated_ids.size()), tokenizer_.eos_token_id());
        const int32_t token = sample_token(
            logits,
            generation.top_k,
            generation.top_p,
            generation.temperature,
            generation.seed,
            step,
            sampling_policy,
            sampler_scratch);
        sampling_ms += engine::debug::elapsed_ms(sample_start);
        generated_ids.push_back(token);
        history.push_back(token);
        if (step + 1 >= kMinNewTokens && is_eos(token, tokenizer_.eos_token_id())) {
            break;
        }
        const auto decode_start = Clock::now();
        logits = decode_graph_->run_step(token, assets_->config.ar);
        decode_step_ms += engine::debug::elapsed_ms(decode_start);
    }

    const auto parse_start = Clock::now();
    auto out = parse_content_style_tokens(tokenizer_.decode(generated_ids, false));
    const double parse_ms = engine::debug::elapsed_ms(parse_start);
    engine::debug::timing_log_scalar("vevo2.ar.tokenize_ms", tokenize_ms);
    engine::debug::timing_log_scalar("vevo2.ar.prefill.graph.build_ms", prefill_graph_build_ms);
    engine::debug::timing_log_scalar("vevo2.ar.decode.graph.build_ms", decode_graph_build_ms);
    engine::debug::timing_log_scalar("vevo2.ar.prefill_run_ms", prefill_run_ms);
    engine::debug::timing_log_scalar("vevo2.ar.kv_import_ms", import_ms);
    engine::debug::timing_log_scalar("vevo2.ar.rng_ms", rng_ms);
    engine::debug::timing_log_scalar("vevo2.ar.sampling_ms", sampling_ms);
    engine::debug::timing_log_scalar("vevo2.ar.decode_step_ms", decode_step_ms);
    engine::debug::timing_log_scalar("vevo2.ar.parse_ms", parse_ms);
    engine::debug::timing_log_scalar("vevo2.ar.total_ms", engine::debug::elapsed_ms(total_start));
    return out;
}

int64_t Vevo2AutoregressiveRuntime::last_prompt_tokens() const noexcept {
    return last_prompt_tokens_;
}

int32_t Vevo2AutoregressiveRuntime::eos_token_id() const noexcept {
    return tokenizer_.eos_token_id();
}

int32_t Vevo2AutoregressiveRuntime::pad_token_id() const noexcept {
    return tokenizer_.pad_token_id();
}

const Vevo2ARConfig & Vevo2AutoregressiveRuntime::config() const noexcept {
    return assets_->config.ar;
}

bool Vevo2AutoregressiveRuntime::weights_uploaded() const noexcept {
    return weights_ != nullptr;
}

}  // namespace engine::models::vevo2
