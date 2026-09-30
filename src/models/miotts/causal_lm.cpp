#include "engine/models/miotts/causal_lm.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/kv_cache.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::miotts {
namespace {

namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct TextLayerWeights {
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

struct MioTTSQwen3Weights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    std::vector<TextLayerWeights> layers;
    core::TensorValue norm;
    core::TensorValue lm_head;
};

struct DecoderLayerOutputs {
    core::TensorValue output;
    core::TensorValue key;
    core::TensorValue value;
};

struct SamplingLogits {
    std::vector<float> speech;
    float eos = -std::numeric_limits<float>::infinity();
};

struct PrefillOutput {
    SamplingLogits logits;
    runtime::TransformerKVState kv_state;
};

struct SamplingCandidate {
    int32_t token = 0;
    float score = 0.0F;
};

struct SamplingScratch {
    std::vector<SamplingCandidate> candidates;
    std::vector<double> probabilities;
};

int64_t head_dim(const MioTTSConfig & config) {
    if (config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 || config.head_dim <= 0) {
        throw std::runtime_error("MioTTS lm attention config is invalid");
    }
    return config.head_dim;
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
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

core::TensorValue prompt_embeddings(
    core::ModuleBuildContext & ctx,
    const MioTTSQwen3Weights & weights,
    const MioTTSConfig & config,
    ggml_tensor * token_ids,
    int64_t prompt_steps) {
    auto ids = core::wrap_tensor(token_ids, core::TensorShape::from_dims({prompt_steps}), GGML_TYPE_I32);
    auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_size}).build(ctx, ids, weights.token_embedding);
    return core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, prompt_steps, config.hidden_size}));
}

int64_t candidate_logit_count(const MioTTSConfig & config) {
    return config.speech_token_count + 1;
}

std::vector<int32_t> build_candidate_token_ids(const MioTTSConfig & config) {
    std::vector<int32_t> ids(static_cast<size_t>(candidate_logit_count(config)));
    for (int64_t i = 0; i < config.speech_token_count; ++i) {
        ids[static_cast<size_t>(i)] = static_cast<int32_t>(config.speech_token_start_id + i);
    }
    ids[static_cast<size_t>(config.speech_token_count)] = static_cast<int32_t>(config.eos_token_id);
    return ids;
}

core::TensorValue candidate_lm_head(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & hidden,
    const MioTTSQwen3Weights & weights,
    const MioTTSConfig & config,
    ggml_tensor * candidate_ids) {
    auto ids = core::wrap_tensor(
        candidate_ids,
        core::TensorShape::from_dims({candidate_logit_count(config)}),
        GGML_TYPE_I32);
    auto selected_weight = modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                               .build(ctx, ids, weights.lm_head);
    return modules::LinearModule({config.hidden_size, candidate_logit_count(config), false})
        .build(ctx, hidden, {selected_weight, std::nullopt});
}

core::TensorValue flash_attention_from_grouped_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const core::TensorValue & attention_mask) {
    const auto q_contiguous = core::ensure_backend_addressable_layout(ctx, q_heads);
    const auto k_contiguous = core::ensure_backend_addressable_layout(ctx, k_heads);
    const auto v_contiguous = core::ensure_backend_addressable_layout(ctx, v_heads);
    auto * flash = ggml_flash_attn_ext(
        ctx.ggml,
        q_contiguous.tensor,
        k_contiguous.tensor,
        v_contiguous.tensor,
        attention_mask.tensor,
        1.0F / std::sqrt(static_cast<float>(dim)),
        0.0F,
        0.0F);
    ggml_flash_attn_ext_set_prec(flash, GGML_PREC_F32);
    return core::wrap_tensor(
        flash,
        core::TensorShape::from_dims({q_contiguous.shape.dims[0], q_contiguous.shape.dims[2], q_contiguous.shape.dims[1], dim}),
        GGML_TYPE_F32);
}

DecoderLayerOutputs decoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const TextLayerWeights & weights,
    const MioTTSConfig & config,
    const std::optional<core::TensorValue> & prefix_key = std::nullopt,
    const std::optional<core::TensorValue> & prefix_value = std::nullopt,
    const std::optional<core::TensorValue> & attention_mask = std::nullopt) {
    const int64_t dim = head_dim(config);
    const int64_t kv_repeats = config.num_attention_heads / config.num_key_value_heads;
    const modules::LinearModule q_proj({config.hidden_size, config.num_attention_heads * dim, false});
    const modules::LinearModule k_proj({config.hidden_size, config.num_key_value_heads * dim, false});
    const modules::LinearModule v_proj({config.hidden_size, config.num_key_value_heads * dim, false});
    const modules::LinearModule o_proj({config.num_attention_heads * dim, config.hidden_size, false});
    const modules::RMSNormModule hidden_norm({config.hidden_size, config.rms_norm_eps, true, false});
    const modules::RMSNormModule head_norm({dim, config.rms_norm_eps, true, false});

    auto x_norm = hidden_norm.build(ctx, input, {weights.input_norm, std::nullopt});
    auto q = q_proj.build(ctx, x_norm, {weights.q_proj, std::nullopt});
    auto k = k_proj.build(ctx, x_norm, {weights.k_proj, std::nullopt});
    auto v = v_proj.build(ctx, x_norm, {weights.v_proj, std::nullopt});
    q = head_norm.build(ctx, reshape_heads(ctx, q, config.num_attention_heads, dim), {weights.q_norm, std::nullopt});
    k = head_norm.build(ctx, reshape_heads(ctx, k, config.num_key_value_heads, dim), {weights.k_norm, std::nullopt});
    v = reshape_heads(ctx, v, config.num_key_value_heads, dim);
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, q, positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, k, positions);

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto all_k = prefix_key.has_value() ? modules::ConcatModule({1}).build(ctx, *prefix_key, k) : k;
    auto all_v = prefix_value.has_value() ? modules::ConcatModule({1}).build(ctx, *prefix_value, v) : v;
    auto k_heads = repeat_kv_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, all_k.shape.rank}).build(ctx, all_k), kv_repeats);
    auto v_heads = repeat_kv_heads(ctx, modules::TransposeModule({{0, 2, 1, 3}, all_v.shape.rank}).build(ctx, all_v), kv_repeats);
    auto context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(
        ctx,
        context,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_attention_heads * dim}));
    auto x = modules::AddModule{}.build(ctx, input, o_proj.build(ctx, context, {weights.o_proj, std::nullopt}));

    auto ff_in = hidden_norm.build(ctx, x, {weights.post_norm, std::nullopt});
    auto gate = modules::LinearModule({config.hidden_size, config.intermediate_size, false})
                    .build(ctx, ff_in, {weights.gate_proj, std::nullopt});
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule({config.hidden_size, config.intermediate_size, false})
                  .build(ctx, ff_in, {weights.up_proj, std::nullopt});
    auto gated = modules::MulModule{}.build(ctx, gate, up);
    auto ff = modules::LinearModule({config.intermediate_size, config.hidden_size, false})
                  .build(ctx, gated, {weights.down_proj, std::nullopt});
    return {modules::AddModule{}.build(ctx, x, ff), k, v};
}

DecoderLayerOutputs decoder_layer_with_static_cache(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const TextLayerWeights & weights,
    const MioTTSConfig & config,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const core::TensorValue & cache_slot,
    const core::TensorValue & attention_mask) {
    const int64_t dim = head_dim(config);
    const modules::LinearModule q_proj({config.hidden_size, config.num_attention_heads * dim, false});
    const modules::LinearModule k_proj({config.hidden_size, config.num_key_value_heads * dim, false});
    const modules::LinearModule v_proj({config.hidden_size, config.num_key_value_heads * dim, false});
    const modules::LinearModule o_proj({config.num_attention_heads * dim, config.hidden_size, false});
    const modules::RMSNormModule hidden_norm({config.hidden_size, config.rms_norm_eps, true, false});

    auto x_norm = hidden_norm.build(ctx, input, {weights.input_norm, std::nullopt});
    auto q = q_proj.build(ctx, x_norm, {weights.q_proj, std::nullopt});
    auto k = k_proj.build(ctx, x_norm, {weights.k_proj, std::nullopt});
    auto v = v_proj.build(ctx, x_norm, {weights.v_proj, std::nullopt});
    q = modules::RMSNormModule({dim, config.rms_norm_eps, true, false})
            .build(ctx, reshape_heads(ctx, q, config.num_attention_heads, dim), {weights.q_norm, std::nullopt});
    k = modules::RMSNormModule({dim, config.rms_norm_eps, true, false})
            .build(ctx, reshape_heads(ctx, k, config.num_key_value_heads, dim), {weights.k_norm, std::nullopt});
    v = reshape_heads(ctx, v, config.num_key_value_heads, dim);
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, q, positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NEOX, config.rope_theta}).build(ctx, k, positions);

    const modules::FastKVSetRowsModule set_rows;
    auto updated_cache_key = set_rows.build(ctx, cache_key, k, cache_slot);
    auto updated_cache_value = set_rows.build(ctx, cache_value, v, cache_slot);

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, updated_cache_key.shape.rank}).build(ctx, updated_cache_key);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, updated_cache_value.shape.rank}).build(ctx, updated_cache_value);
    auto context = flash_attention_from_grouped_heads(ctx, q_heads, k_heads, v_heads, dim, attention_mask);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({1, 1, config.num_attention_heads * dim}));
    auto x = modules::AddModule{}.build(ctx, input, o_proj.build(ctx, context, {weights.o_proj, std::nullopt}));

    auto ff_in = hidden_norm.build(ctx, x, {weights.post_norm, std::nullopt});
    auto gate = modules::LinearModule({config.hidden_size, config.intermediate_size, false})
                    .build(ctx, ff_in, {weights.gate_proj, std::nullopt});
    gate = modules::SiluModule{}.build(ctx, gate);
    auto up = modules::LinearModule({config.hidden_size, config.intermediate_size, false})
                  .build(ctx, ff_in, {weights.up_proj, std::nullopt});
    auto gated = modules::MulModule{}.build(ctx, gate, up);
    auto ff = modules::LinearModule({config.intermediate_size, config.hidden_size, false})
                  .build(ctx, gated, {weights.down_proj, std::nullopt});
    return {modules::AddModule{}.build(ctx, x, ff), k, v};
}

MioTTSQwen3Weights load_weights(
    const MioTTSAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config;
    const auto & source = *assets.model_weights;
    MioTTSQwen3Weights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "miotts.lm.weights",
        weight_context_bytes);
    weights.token_embedding = weights.store->load_tensor(
        source,
        "model.embed_tokens.weight",
        storage_type,
        {config.vocab_size, config.hidden_size});
    weights.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    const int64_t dim = head_dim(config);
    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer);
        TextLayerWeights w;
        w.input_norm = weights.store->load_f32_tensor(source, prefix + ".input_layernorm.weight", {config.hidden_size});
        w.q_proj = weights.store->load_tensor(source, prefix + ".self_attn.q_proj.weight", storage_type, {config.num_attention_heads * dim, config.hidden_size});
        w.k_proj = weights.store->load_tensor(source, prefix + ".self_attn.k_proj.weight", storage_type, {config.num_key_value_heads * dim, config.hidden_size});
        w.v_proj = weights.store->load_tensor(source, prefix + ".self_attn.v_proj.weight", storage_type, {config.num_key_value_heads * dim, config.hidden_size});
        w.o_proj = weights.store->load_tensor(source, prefix + ".self_attn.o_proj.weight", storage_type, {config.hidden_size, config.num_attention_heads * dim});
        w.q_norm = weights.store->load_f32_tensor(source, prefix + ".self_attn.q_norm.weight", {dim});
        w.k_norm = weights.store->load_f32_tensor(source, prefix + ".self_attn.k_norm.weight", {dim});
        w.post_norm = weights.store->load_f32_tensor(source, prefix + ".post_attention_layernorm.weight", {config.hidden_size});
        w.gate_proj = weights.store->load_tensor(source, prefix + ".mlp.gate_proj.weight", storage_type, {config.intermediate_size, config.hidden_size});
        w.up_proj = weights.store->load_tensor(source, prefix + ".mlp.up_proj.weight", storage_type, {config.intermediate_size, config.hidden_size});
        w.down_proj = weights.store->load_tensor(source, prefix + ".mlp.down_proj.weight", storage_type, {config.hidden_size, config.intermediate_size});
        weights.layers.push_back(std::move(w));
    }
    weights.norm = weights.store->load_f32_tensor(source, "model.norm.weight", {config.hidden_size});
    weights.lm_head = weights.token_embedding;
    weights.store->upload();
    return weights;
}

int32_t argmax_token(
    const SamplingLogits & logits,
    const MioTTSConfig & config) {
    if (logits.speech.empty()) {
        throw std::runtime_error("MioTTS lm cannot select from empty logits");
    }
    if (static_cast<int64_t>(logits.speech.size()) != config.speech_token_count) {
        throw std::runtime_error("MioTTS lm speech token range is invalid");
    }
    int64_t best = 0;
    for (int64_t i = 1; i < config.speech_token_count; ++i) {
        if (logits.speech[static_cast<size_t>(i)] > logits.speech[static_cast<size_t>(best)]) {
            best = i;
        }
    }
    if (logits.eos > logits.speech[static_cast<size_t>(best)]) {
        return static_cast<int32_t>(config.eos_token_id);
    }
    return static_cast<int32_t>(config.speech_token_start_id + best);
}

bool is_eos(const MioTTSConfig & config, int32_t token) {
    return static_cast<int64_t>(token) == config.eos_token_id ||
        static_cast<int64_t>(token) == config.pad_token_id;
}

void apply_repetition_penalty(
    SamplingLogits & logits,
    const MioTTSConfig & config,
    const std::vector<int32_t> & history,
    float penalty) {
    if (penalty == 1.0F) {
        return;
    }
    if (!(penalty > 0.0F)) {
        throw std::runtime_error("MioTTS repetition_penalty must be positive");
    }
    for (const int32_t token : history) {
        float * value = nullptr;
        const int64_t speech_index = static_cast<int64_t>(token) - config.speech_token_start_id;
        if (speech_index >= 0 && speech_index < config.speech_token_count) {
            value = &logits.speech[static_cast<size_t>(speech_index)];
        } else if (static_cast<int64_t>(token) == config.eos_token_id) {
            value = &logits.eos;
        }
        if (value != nullptr) {
            *value = *value < 0.0F ? *value * penalty : *value / penalty;
        }
    }
}

void apply_openai_penalties(
    SamplingLogits & logits,
    const MioTTSConfig & config,
    const std::vector<int32_t> & generated,
    float presence_penalty,
    float frequency_penalty) {
    if (presence_penalty == 0.0F && frequency_penalty == 0.0F) {
        return;
    }
    if (presence_penalty < 0.0F || frequency_penalty < 0.0F) {
        throw std::runtime_error("MioTTS presence_penalty and frequency_penalty must be non-negative");
    }
    std::vector<int32_t> counts(static_cast<size_t>(config.speech_token_count), 0);
    for (const int32_t token : generated) {
        const int64_t index = static_cast<int64_t>(token) - config.speech_token_start_id;
        if (index >= 0 && index < config.speech_token_count) {
            ++counts[static_cast<size_t>(index)];
        }
    }
    for (int64_t i = 0; i < config.speech_token_count; ++i) {
        const int32_t count = counts[static_cast<size_t>(i)];
        if (count == 0) {
            continue;
        }
        logits.speech[static_cast<size_t>(i)] -= presence_penalty + frequency_penalty * static_cast<float>(count);
    }
}

int32_t sample_speech_token(
    const SamplingLogits & logits,
    const MioTTSConfig & config,
    const MioTTSGenerationOptions & options,
    std::mt19937 & rng,
    SamplingScratch & scratch) {
    if (!options.do_sample || options.top_k == 1 || options.temperature == 0.0F) {
        return argmax_token(logits, config);
    }
    if (!(options.temperature > 0.0F)) {
        throw std::runtime_error("MioTTS temperature must be non-negative");
    }
    if (!(options.top_p >= 0.0F && options.top_p <= 1.0F)) {
        throw std::runtime_error("MioTTS top_p must be in [0, 1]");
    }
    auto candidate_order = [](const SamplingCandidate & lhs, const SamplingCandidate & rhs) {
        if (lhs.score == rhs.score) {
            return lhs.token < rhs.token;
        }
        return lhs.score > rhs.score;
    };
    auto & candidates = scratch.candidates;
    candidates.clear();
    candidates.reserve(static_cast<size_t>(config.speech_token_count + 1));
    for (int64_t i = 0; i < config.speech_token_count; ++i) {
        const int64_t token = config.speech_token_start_id + i;
        candidates.push_back({static_cast<int32_t>(token), logits.speech[static_cast<size_t>(i)] / options.temperature});
    }
    candidates.push_back({static_cast<int32_t>(config.eos_token_id), logits.eos / options.temperature});
    if (options.top_k > 0 && static_cast<size_t>(options.top_k) < candidates.size()) {
        const auto top_end = candidates.begin() + static_cast<std::ptrdiff_t>(options.top_k);
        std::nth_element(candidates.begin(), top_end, candidates.end(), candidate_order);
        std::sort(candidates.begin(), top_end, candidate_order);
        candidates.erase(top_end, candidates.end());
    } else {
        std::sort(candidates.begin(), candidates.end(), candidate_order);
    }
    const float max_score = candidates.front().score;
    double total = 0.0;
    auto & probabilities = scratch.probabilities;
    probabilities.assign(candidates.size(), 0.0);
    for (size_t i = 0; i < candidates.size(); ++i) {
        probabilities[i] = std::exp(static_cast<double>(candidates[i].score - max_score));
        total += probabilities[i];
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        throw std::runtime_error("MioTTS sampler produced invalid probability mass");
    }
    for (double & prob : probabilities) {
        prob /= total;
    }
    if (options.top_p < 1.0F) {
        double cumulative = 0.0;
        size_t keep = probabilities.size();
        for (size_t i = 0; i < probabilities.size(); ++i) {
            cumulative += probabilities[i];
            if (cumulative >= static_cast<double>(options.top_p)) {
                keep = i + 1;
                break;
            }
        }
        candidates.resize(keep);
        probabilities.resize(keep);
        double kept_total = 0.0;
        for (const double prob : probabilities) {
            kept_total += prob;
        }
        if (!(kept_total > 0.0)) {
            throw std::runtime_error("MioTTS top-p removed all probability mass");
        }
        for (double & prob : probabilities) {
            prob /= kept_total;
        }
    }
    std::discrete_distribution<size_t> distribution(probabilities.begin(), probabilities.end());
    return candidates[distribution(rng)].token;
}

int32_t codec_token_id_to_codec_index(const MioTTSConfig & config, int32_t token_id) {
    const int64_t index = static_cast<int64_t>(token_id) - config.speech_token_start_id;
    if (index < 0 || index >= config.speech_token_count) {
        throw std::runtime_error("MioTTS generated token is outside the speech token range");
    }
    return static_cast<int32_t>(index);
}

class MioTTSQwen3WeightsRuntime {
public:
    MioTTSQwen3WeightsRuntime(
        std::shared_ptr<const MioTTSAssets> assets,
        core::ExecutionContext & execution,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets_(std::move(assets)),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          weights_(std::make_shared<MioTTSQwen3Weights>(load_weights(*assets_, backend_, backend_type_, weight_context_bytes, storage_type))) {
        if (assets_ == nullptr) {
            throw std::runtime_error("MioTTS lm weights runtime requires assets");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("MioTTS lm backend is not initialized");
        }
    }

    const MioTTSAssets & assets() const noexcept {
        return *assets_;
    }

    const MioTTSQwen3Weights & weights() const noexcept {
        return *weights_;
    }

    ggml_backend_t backend() const noexcept {
        return backend_;
    }

    core::BackendType backend_type() const noexcept {
        return backend_type_;
    }

    int threads() const noexcept {
        return threads_;
    }

private:
    std::shared_ptr<const MioTTSAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    std::shared_ptr<const MioTTSQwen3Weights> weights_;
};

class PrefillGraph {
public:
    PrefillGraph(
        std::shared_ptr<MioTTSQwen3WeightsRuntime> runtime,
        int64_t prompt_steps,
        size_t graph_arena_bytes)
        : runtime_(std::move(runtime)),
          prompt_steps_(prompt_steps) {
        if (prompt_steps_ <= 0) {
            throw std::runtime_error("MioTTS lm prefill requires positive prompt length");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MioTTS lm prefill graph context");
        }
        const auto & config = runtime_->assets().config;
        const auto & weights = runtime_->weights();
        core::ModuleBuildContext ctx{ctx_.get(), "miotts.lm.prefill", runtime_->backend_type()};
        token_ids_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
        auto x = prompt_embeddings(
            ctx,
            weights,
            config,
            token_ids_,
            prompt_steps_);
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, prompt_steps_);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({prompt_steps_}), GGML_TYPE_I32);

        for (const auto & layer : weights.layers) {
            auto out = decoder_layer(ctx, x, positions, layer, config);
            x = out.output;
            keys_.push_back(out.key.tensor);
            values_.push_back(out.value.tensor);
        }
        x = modules::SliceModule({1, prompt_steps_ - 1, 1}).build(ctx, x);
        x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                .build(ctx, x, {weights.norm, std::nullopt});
        candidate_ids_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, candidate_logit_count(config));
        auto logits = candidate_lm_head(ctx, x, weights, config, candidate_ids_);
        logits_ = logits.tensor;
        ggml_set_output(logits_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, logits_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate MioTTS lm prefill graph");
        }
        std::vector<int32_t> pos(static_cast<size_t>(prompt_steps_), 0);
        for (int64_t i = 0; i < prompt_steps_; ++i) {
            pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        ggml_backend_tensor_set(positions_, pos.data(), 0, pos.size() * sizeof(int32_t));
        const auto candidate_ids = build_candidate_token_ids(config);
        ggml_backend_tensor_set(candidate_ids_, candidate_ids.data(), 0, candidate_ids.size() * sizeof(int32_t));
        debug::timing_log_scalar("miotts.lm.prefill.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("miotts.lm.prefill_prompt_steps", prompt_steps_);
    }

    ~PrefillGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool matches(const MioTTSQwen3WeightsRuntime & runtime, int64_t prompt_steps) const {
        return runtime_.get() == &runtime && prompt_steps_ == prompt_steps;
    }

    PrefillOutput run(const std::vector<int32_t> & token_ids) {
        const auto & config = runtime_->assets().config;
        if (static_cast<int64_t>(token_ids.size()) != prompt_steps_) {
            throw std::runtime_error("MioTTS lm prefill token id count mismatch");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(token_ids_, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
        debug::timing_log_scalar("miotts.lm.prefill_input_upload_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        debug::timing_log_scalar("miotts.lm.prefill.graph.compute_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MioTTS lm prefill graph compute failed");
        }
        PrefillOutput out;
        out.logits.speech.resize(static_cast<size_t>(config.speech_token_count));
        timing_start = Clock::now();
        ggml_backend_tensor_get(
            logits_,
            out.logits.speech.data(),
            0,
            out.logits.speech.size() * sizeof(float));
        ggml_backend_tensor_get(
            logits_,
            &out.logits.eos,
            static_cast<size_t>(config.speech_token_count) * sizeof(float),
            sizeof(float));
        out.kv_state.current_end = prompt_steps_;
        out.kv_state.layers.resize(keys_.size());
        const size_t layer_values = static_cast<size_t>(prompt_steps_ * config.num_key_value_heads * head_dim(config));
        for (size_t layer = 0; layer < keys_.size(); ++layer) {
            auto & state = out.kv_state.layers[layer];
            state.valid_steps = prompt_steps_;
            state.key.resize(layer_values);
            state.value.resize(layer_values);
            ggml_backend_tensor_get(keys_[layer], state.key.data(), 0, state.key.size() * sizeof(float));
            ggml_backend_tensor_get(values_[layer], state.value.data(), 0, state.value.size() * sizeof(float));
        }
        debug::timing_log_scalar("miotts.lm.prefill_output_read_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        return out;
    }

private:
    std::shared_ptr<MioTTSQwen3WeightsRuntime> runtime_;
    int64_t prompt_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ids_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * candidate_ids_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

class DecodeGraph {
public:
    DecodeGraph(std::shared_ptr<MioTTSQwen3WeightsRuntime> runtime, int64_t cache_steps, size_t graph_arena_bytes)
        : runtime_(std::move(runtime)),
          cache_steps_(cache_steps) {
        if (cache_steps_ <= 0) {
            throw std::runtime_error("MioTTS lm decode requires positive cache length");
        }
        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MioTTS lm decode graph context");
        }
        const auto & config = runtime_->assets().config;
        const auto & weights = runtime_->weights();
        const int64_t dim = head_dim(config);
        core::ModuleBuildContext ctx{ctx_.get(), "miotts.lm.decode", runtime_->backend_type()};
        token_id_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto token_id = core::wrap_tensor(token_id_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                     .build(ctx, token_id, weights.token_embedding);
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, 1, config.hidden_size}));
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        cache_slot_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto cache_slot = core::wrap_tensor(cache_slot_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, cache_steps_, 1, 1, 1);
        auto attention_mask = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({1, 1, 1, cache_steps_}),
            GGML_TYPE_F16);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        std::vector<core::TensorValue> cache_keys;
        std::vector<core::TensorValue> cache_values;
        for (const auto & layer : weights.layers) {
            cache_keys.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_, config.num_key_value_heads, dim})));
            cache_values.push_back(core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({1, cache_steps_, config.num_key_value_heads, dim})));
            auto out = decoder_layer_with_static_cache(
                ctx,
                x,
                positions,
                layer,
                config,
                cache_keys.back(),
                cache_values.back(),
                cache_slot,
                attention_mask);
            x = out.output;
        }
        step_cache_ = runtime::TransformerKVCache(
            cache_steps_,
            config.num_key_value_heads * dim,
            std::move(cache_keys),
            std::move(cache_values));
        x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                .build(ctx, x, {weights.norm, std::nullopt});
        candidate_ids_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, candidate_logit_count(config));
        auto logits = candidate_lm_head(ctx, x, weights, config, candidate_ids_);
        logits_ = logits.tensor;
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, logits_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), runtime_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate MioTTS lm decode graph");
        }
        const auto candidate_ids = build_candidate_token_ids(config);
        ggml_backend_tensor_set(candidate_ids_, candidate_ids.data(), 0, candidate_ids.size() * sizeof(int32_t));
        attention_mask_values_.assign(static_cast<size_t>(cache_steps_), ggml_fp32_to_fp16(-INFINITY));
        debug::timing_log_scalar("miotts.lm.decode.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("miotts.lm.decode_cache_steps", cache_steps_);
    }

    ~DecodeGraph() {
        engine::core::release_backend_graph_resources(runtime_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool can_run(const MioTTSQwen3WeightsRuntime & runtime, int64_t required_steps) const {
        return runtime_.get() == &runtime && cache_steps_ >= required_steps;
    }

    void import_state(const runtime::TransformerKVState & state) {
        step_cache_.import_state(state);
    }

    void reset_timing() noexcept {
        input_upload_ms_ = 0.0;
        mask_upload_ms_ = 0.0;
        graph_compute_ms_ = 0.0;
        logits_read_ms_ = 0.0;
    }

    void run_step_into(int32_t token, SamplingLogits & logits) {
        const auto & config = runtime_->assets().config;
        if (step_cache_.valid_steps() >= cache_steps_) {
            throw std::runtime_error("MioTTS lm decode cache exhausted");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(token_id_, &token, 0, sizeof(int32_t));
        const int32_t position = static_cast<int32_t>(step_cache_.current_end());
        ggml_backend_tensor_set(positions_, &position, 0, sizeof(int32_t));
        const int32_t cache_slot = static_cast<int32_t>(step_cache_.valid_steps());
        ggml_backend_tensor_set(cache_slot_, &cache_slot, 0, sizeof(int32_t));
        input_upload_ms_ += engine::debug::elapsed_ms(timing_start, Clock::now());
        const auto masked = ggml_fp32_to_fp16(-INFINITY);
        const auto visible = ggml_fp32_to_fp16(0.0F);
        std::fill(attention_mask_values_.begin(), attention_mask_values_.end(), masked);
        for (int64_t i = 0; i < step_cache_.valid_steps(); ++i) {
            attention_mask_values_[static_cast<size_t>(i)] = visible;
        }
        attention_mask_values_[static_cast<size_t>(cache_slot)] = visible;
        timing_start = Clock::now();
        ggml_backend_tensor_set(
            attention_mask_,
            attention_mask_values_.data(),
            0,
            attention_mask_values_.size() * sizeof(ggml_fp16_t));
        mask_upload_ms_ += engine::debug::elapsed_ms(timing_start, Clock::now());
        core::set_backend_threads(runtime_->backend(), runtime_->threads());
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(runtime_->backend(), graph_);
        ggml_backend_synchronize(runtime_->backend());
        graph_compute_ms_ += engine::debug::elapsed_ms(timing_start, Clock::now());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MioTTS lm decode graph compute failed");
        }
        logits.speech.resize(static_cast<size_t>(config.speech_token_count));
        timing_start = Clock::now();
        ggml_backend_tensor_get(
            logits_,
            logits.speech.data(),
            0,
            logits.speech.size() * sizeof(float));
        ggml_backend_tensor_get(
            logits_,
            &logits.eos,
            static_cast<size_t>(config.speech_token_count) * sizeof(float),
            sizeof(float));
        logits_read_ms_ += engine::debug::elapsed_ms(timing_start, Clock::now());
        step_cache_.advance_after_direct_append(1);
    }

    void log_timing() const {
        debug::timing_log_scalar("miotts.lm.decode.input_upload_ms", input_upload_ms_);
        debug::timing_log_scalar("miotts.lm.decode.mask_upload_ms", mask_upload_ms_);
        debug::timing_log_scalar("miotts.lm.decode.graph.compute_ms", graph_compute_ms_);
        debug::timing_log_scalar("miotts.lm.decode.logits_read_ms", logits_read_ms_);
    }

private:
    std::shared_ptr<MioTTSQwen3WeightsRuntime> runtime_;
    int64_t cache_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_id_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * cache_slot_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * candidate_ids_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_fp16_t> attention_mask_values_;
    runtime::TransformerKVCache step_cache_;
    double input_upload_ms_ = 0.0;
    double mask_upload_ms_ = 0.0;
    double graph_compute_ms_ = 0.0;
    double logits_read_ms_ = 0.0;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

}  // namespace

struct MioTTSQwen3Runtime::Impl {
    Impl(
        std::shared_ptr<const MioTTSAssets> assets,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : weights(std::make_shared<MioTTSQwen3WeightsRuntime>(
              std::move(assets),
              execution,
              weight_context_bytes,
              storage_type)),
          prefill_graph_arena_bytes(prefill_graph_arena_bytes),
          decode_graph_arena_bytes(decode_graph_arena_bytes) {}

    MioTTSGeneratedTokens generate(
        const MioTTSPrompt & prompt,
        const MioTTSGenerationOptions & options) {
        const auto & config = weights->assets().config;
        if (prompt.input_ids.empty()) {
            throw std::runtime_error("MioTTS lm prompt is empty");
        }
        if (options.max_tokens <= 0) {
            throw std::runtime_error("MioTTS max_tokens must be positive");
        }
        const int64_t prompt_steps = static_cast<int64_t>(prompt.input_ids.size());
        if (prompt_steps + options.max_tokens > config.max_position_embeddings) {
            throw std::runtime_error("MioTTS lm request exceeds max_position_embeddings");
        }
        auto timing_start = Clock::now();
        debug::timing_log_scalar("miotts.lm.prompt_prepare_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (prefill_graph == nullptr || !prefill_graph->matches(*weights, prompt_steps)) {
            prefill_graph = std::make_unique<PrefillGraph>(
                weights,
                prompt_steps,
                prefill_graph_arena_bytes);
        } else {
            debug::timing_log_scalar("miotts.lm.prefill.graph.build_ms", 0.0);
            debug::trace_log_scalar("miotts.lm.prefill_prompt_steps", prompt_steps);
        }
        timing_start = Clock::now();
        auto prefill = prefill_graph->run(prompt.input_ids);
        debug::timing_log_scalar("miotts.lm.prefill_total_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        const int64_t required_cache_steps = prompt_steps + options.max_tokens;
        if (decode_graph == nullptr || !decode_graph->can_run(*weights, required_cache_steps)) {
            decode_graph = std::make_unique<DecodeGraph>(weights, required_cache_steps, decode_graph_arena_bytes);
        } else {
            debug::timing_log_scalar("miotts.lm.decode.graph.build_ms", 0.0);
            debug::trace_log_scalar("miotts.lm.decode_cache_steps", required_cache_steps);
        }
        decode_graph->import_state(prefill.kv_state);
        decode_graph->reset_timing();

        MioTTSGeneratedTokens out;
        out.token_ids.reserve(static_cast<size_t>(options.max_tokens));
        out.codec_tokens.reserve(static_cast<size_t>(options.max_tokens));
        SamplingLogits logits = std::move(prefill.logits);
        std::vector<int32_t> history = prompt.input_ids;
        history.reserve(prompt.input_ids.size() + static_cast<size_t>(options.max_tokens));
        std::mt19937 rng(options.seed);
        SamplingScratch sampling_scratch;
        double sampling_ms = 0.0;
        double decode_step_ms = 0.0;
        timing_start = Clock::now();
        for (int64_t step = 0; step < options.max_tokens; ++step) {
            const auto sampling_start = Clock::now();
            apply_repetition_penalty(logits, config, history, options.repetition_penalty);
            apply_openai_penalties(
                logits,
                config,
                out.token_ids,
                options.presence_penalty,
                options.frequency_penalty);
            const int32_t token = sample_speech_token(logits, config, options, rng, sampling_scratch);
            sampling_ms += engine::debug::elapsed_ms(sampling_start, Clock::now());
            if (is_eos(config, token)) {
                break;
            }
            out.token_ids.push_back(token);
            out.codec_tokens.push_back(codec_token_id_to_codec_index(config, token));
            history.push_back(token);
            const auto step_start = Clock::now();
            decode_graph->run_step_into(token, logits);
            decode_step_ms += engine::debug::elapsed_ms(step_start, Clock::now());
        }
        debug::timing_log_scalar("miotts.lm.sampling_ms", sampling_ms);
        debug::timing_log_scalar("miotts.lm.decode_step_ms", decode_step_ms);
        decode_graph->log_timing();
        debug::timing_log_scalar("miotts.lm.decode_total_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        return out;
    }

    std::shared_ptr<MioTTSQwen3WeightsRuntime> weights;
    size_t prefill_graph_arena_bytes = 0;
    size_t decode_graph_arena_bytes = 0;
    std::unique_ptr<PrefillGraph> prefill_graph;
    std::unique_ptr<DecodeGraph> decode_graph;
};

MioTTSQwen3Runtime::MioTTSQwen3Runtime(
    std::shared_ptr<const MioTTSAssets> assets,
    core::ExecutionContext & execution,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          prefill_graph_arena_bytes,
          decode_graph_arena_bytes,
          weight_context_bytes,
          weight_storage_type)) {}

MioTTSQwen3Runtime::~MioTTSQwen3Runtime() = default;

MioTTSGeneratedTokens MioTTSQwen3Runtime::generate(
    const MioTTSPrompt & prompt,
    const MioTTSGenerationOptions & options) {
    return impl_->generate(prompt, options);
}

}  // namespace engine::models::miotts
