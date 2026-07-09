#include "engine/models/hviske_asr/decoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/kv_cache.h"

#include "../../framework/modules/attention/attention_internal.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::hviske_asr {
namespace {

namespace ai = engine::modules::attention::internal;

using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct DecoderLayerOutputs {
    engine::core::TensorValue output;
    engine::core::TensorValue key;
    engine::core::TensorValue value;
    engine::core::TensorValue cross_key;
    engine::core::TensorValue cross_value;
};

struct PrefillOutput {
    std::vector<float> logits;
};

struct TokenLogProb {
    int32_t token = 0;
    float log_prob = -std::numeric_limits<float>::infinity();
};

struct BeamState {
    std::vector<int32_t> tokens;
    double score = 0.0;
    std::vector<float> logits;
    int64_t slot = 0;
    int64_t valid_steps = 0;
};

struct BeamCandidate {
    size_t parent = 0;
    int32_t token = 0;
    double score = 0.0;
    bool eos = false;
};

struct FinishedBeam {
    std::vector<int32_t> tokens;
    double score = 0.0;
    int64_t length = 0;
};

int64_t head_dim(const HviskeDecoderConfig & config) {
    if (config.heads <= 0 || config.hidden_size % config.heads != 0) {
        throw std::runtime_error("Hviske decoder attention config is invalid");
    }
    return config.hidden_size / config.heads;
}

int32_t argmax_index(const std::vector<float> & values) {
    if (values.empty()) {
        throw std::runtime_error("Hviske decoder cannot select from empty logits");
    }
    return static_cast<int32_t>(
        std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

bool is_eos(const HviskeDecoderConfig & config, int32_t token) {
    return token == static_cast<int32_t>(config.eos_token_id);
}

bool better_log_prob(const TokenLogProb & lhs, const TokenLogProb & rhs) {
    if (lhs.log_prob != rhs.log_prob) {
        return lhs.log_prob > rhs.log_prob;
    }
    return lhs.token < rhs.token;
}

bool better_candidate(const BeamCandidate & lhs, const BeamCandidate & rhs) {
    if (lhs.score != rhs.score) {
        return lhs.score > rhs.score;
    }
    return lhs.token < rhs.token;
}

double normalized_beam_score(double score, int64_t length, float length_penalty) {
    const int64_t safe_length = std::max<int64_t>(1, length);
    return score / std::pow(static_cast<double>(safe_length), static_cast<double>(length_penalty));
}

bool better_finished_beam(const FinishedBeam & lhs, const FinishedBeam & rhs, float length_penalty) {
    const double lhs_score = normalized_beam_score(lhs.score, lhs.length, length_penalty);
    const double rhs_score = normalized_beam_score(rhs.score, rhs.length, length_penalty);
    if (lhs_score != rhs_score) {
        return lhs_score > rhs_score;
    }
    return lhs.score > rhs.score;
}

std::vector<TokenLogProb> top_log_probs(const std::vector<float> & logits, size_t k, bool already_log_probs = false) {
    if (logits.empty() || k == 0) {
        return {};
    }
    float max_logit = -std::numeric_limits<float>::infinity();
    std::vector<TokenLogProb> top;
    top.reserve(std::min(k, logits.size()));
    const auto keep_top_logit = [&](TokenLogProb candidate) {
        if (top.size() == k && !better_log_prob(candidate, top.back())) {
            return;
        }
        auto pos = std::lower_bound(top.begin(), top.end(), candidate, better_log_prob);
        if (top.size() < k) {
            top.insert(pos, candidate);
        } else {
            top.insert(pos, candidate);
            top.pop_back();
        }
    };
    for (size_t i = 0; i < logits.size(); ++i) {
        const float value = logits[i];
        max_logit = std::max(max_logit, value);
        keep_top_logit(TokenLogProb{static_cast<int32_t>(i), value});
    }
    if (already_log_probs) {
        return top;
    }
    float sum = 0.0f;
    for (const float value : logits) {
        sum += std::exp(value - max_logit);
    }
    const float log_sum = max_logit + std::log(sum);
    for (auto & candidate : top) {
        candidate.log_prob -= log_sum;
    }
    return top;
}

engine::core::TensorValue decoder_classifier_output(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const HviskeDecoderWeights & weights,
    const HviskeDecoderConfig & config) {
    auto logits = engine::modules::LinearModule({config.hidden_size, config.vocab_size, true})
                      .build(ctx, input, weights.classifier);
    if (!config.output_log_probs) {
        return logits;
    }
    const auto contiguous = engine::core::ensure_backend_addressable_layout(ctx, logits);
    const auto probs = engine::core::wrap_tensor(
        ggml_soft_max(ctx.ggml, contiguous.tensor),
        contiguous.shape,
        GGML_TYPE_F32);
    return engine::core::wrap_tensor(
        ggml_log(ctx.ggml, probs.tensor),
        probs.shape,
        GGML_TYPE_F32);
}

int32_t sample_token(const std::vector<float> & logits, const HviskeDecodingOptions & options, std::mt19937 & rng) {
    if (logits.empty()) {
        throw std::runtime_error("Hviske decoder cannot sample from empty logits");
    }
    if (!std::isfinite(options.temperature) || options.temperature <= 0.0f) {
        throw std::runtime_error("Hviske decoder temperature must be finite and positive");
    }
    if (!std::isfinite(options.top_p) || options.top_p <= 0.0f || options.top_p > 1.0f) {
        throw std::runtime_error("Hviske decoder top_p must be finite and within (0, 1]");
    }
    const size_t candidate_count = options.top_k > 0
        ? std::min(static_cast<size_t>(options.top_k), logits.size())
        : logits.size();
    std::vector<TokenLogProb> candidates;
    candidates.reserve(candidate_count);

    if (options.top_k > 0) {
        candidates = top_log_probs(logits, candidate_count);
        float max_logit = -std::numeric_limits<float>::infinity();
        for (const auto & candidate : candidates) {
            max_logit = std::max(max_logit, logits[static_cast<size_t>(candidate.token)]);
        }
        double sum = 0.0;
        for (auto & candidate : candidates) {
            const float scaled = (logits[static_cast<size_t>(candidate.token)] - max_logit) / options.temperature;
            const double weight = std::exp(static_cast<double>(scaled));
            candidate.log_prob = static_cast<float>(std::log(weight));
            sum += weight;
        }
        const float log_sum = static_cast<float>(std::log(sum));
        for (auto & candidate : candidates) {
            candidate.log_prob -= log_sum;
        }
    } else {
        float max_logit = -std::numeric_limits<float>::infinity();
        for (const float value : logits) {
            max_logit = std::max(max_logit, value);
        }
        double sum = 0.0;
        for (const float value : logits) {
            sum += std::exp(static_cast<double>((value - max_logit) / options.temperature));
        }
        const float log_sum = max_logit / options.temperature + static_cast<float>(std::log(sum));
        for (size_t i = 0; i < logits.size(); ++i) {
            candidates.push_back(TokenLogProb{
                static_cast<int32_t>(i),
                logits[i] / options.temperature - log_sum,
            });
        }
        std::sort(candidates.begin(), candidates.end(), better_log_prob);
    }

    std::vector<double> weights;
    weights.reserve(candidates.size());
    double total = 0.0;
    for (const auto & candidate : candidates) {
        const double weight = std::exp(static_cast<double>(candidate.log_prob));
        weights.push_back(weight);
        total += weight;
    }
    if (total <= 0.0 || !std::isfinite(total)) {
        throw std::runtime_error("Hviske decoder sampling distribution is invalid");
    }

    if (options.top_p < 1.0f) {
        double cumulative = 0.0;
        size_t keep = 0;
        for (; keep < weights.size(); ++keep) {
            cumulative += weights[keep] / total;
            if (cumulative >= static_cast<double>(options.top_p)) {
                ++keep;
                break;
            }
        }
        keep = std::max<size_t>(1, std::min(keep, weights.size()));
        candidates.resize(keep);
        weights.resize(keep);
    }

    std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
    return candidates[distribution(rng)].token;
}

engine::core::TensorValue attention_from_heads(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & q_heads,
    const engine::core::TensorValue & k_heads,
    const engine::core::TensorValue & v_heads,
    int64_t dim,
    const engine::core::TensorValue & attention_mask) {
    auto broadcast_matmul = [&](const engine::core::TensorValue & lhs, const engine::core::TensorValue & rhs) {
        if (lhs.shape.rank != rhs.shape.rank ||
            lhs.shape.dims[lhs.shape.rank - 1] != rhs.shape.dims[rhs.shape.rank - 2]) {
            throw std::runtime_error("Hviske attention matmul shape mismatch");
        }
        for (size_t axis = 0; axis + 2 < lhs.shape.rank; ++axis) {
            if (rhs.shape.dims[axis] <= 0 || lhs.shape.dims[axis] % rhs.shape.dims[axis] != 0) {
                throw std::runtime_error("Hviske attention matmul broadcast mismatch");
            }
        }
        auto rhs_transposed = engine::modules::TransposeModule({{0, 1, 3, 2}, rhs.shape.rank}).build(ctx, rhs);
        if (!engine::core::has_backend_addressable_layout(rhs_transposed.tensor) || ggml_is_transposed(rhs_transposed.tensor)) {
            rhs_transposed = engine::core::wrap_tensor(ggml_cont(ctx.ggml, rhs_transposed.tensor), rhs_transposed.shape, rhs_transposed.type);
        }
        auto output_shape = lhs.shape;
        output_shape.dims[lhs.shape.rank - 1] = rhs.shape.dims[rhs.shape.rank - 1];
        return engine::core::wrap_tensor(ggml_mul_mat(ctx.ggml, rhs_transposed.tensor, lhs.tensor), output_shape, GGML_TYPE_F32);
    };
    auto scores = broadcast_matmul(
        q_heads,
        engine::modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    scores = engine::core::ensure_backend_addressable_layout(ctx, scores);
    auto attn = engine::core::wrap_tensor(
        ggml_soft_max_ext(
            ctx.ggml,
            scores.tensor,
            attention_mask.tensor,
            1.0F / std::sqrt(static_cast<float>(dim)),
            0.0F),
        scores.shape,
        GGML_TYPE_F32);
    return broadcast_matmul(attn, v_heads);
}

engine::core::TensorValue decoder_embeddings(
    engine::core::ModuleBuildContext & ctx,
    const HviskeDecoderWeights & weights,
    const HviskeDecoderConfig & config,
    ggml_tensor * token_ids,
    ggml_tensor * positions,
    int64_t steps) {
    auto token_tensor = engine::core::wrap_tensor(
        token_ids,
        engine::core::TensorShape::from_dims({1, steps}),
        GGML_TYPE_I32);
    auto position_tensor = engine::core::wrap_tensor(
        positions,
        engine::core::TensorShape::from_dims({1, steps}),
        GGML_TYPE_I32);
    auto token_emb = engine::modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                         .build(ctx, token_tensor, weights.token_embedding);
    auto pos_emb = engine::modules::EmbeddingModule({config.max_sequence_length, config.hidden_size})
                       .build(ctx, position_tensor, weights.position_embedding);
    auto hidden = engine::modules::AddModule().build(ctx, token_emb, pos_emb);
    return engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true})
        .build(ctx, hidden, weights.embedding_norm);
}

engine::core::TensorValue decoder_beam_embeddings(
    engine::core::ModuleBuildContext & ctx,
    const HviskeDecoderWeights & weights,
    const HviskeDecoderConfig & config,
    ggml_tensor * token_ids,
    ggml_tensor * positions,
    int64_t batch) {
    auto token_tensor = engine::core::wrap_tensor(
        token_ids,
        engine::core::TensorShape::from_dims({batch}),
        GGML_TYPE_I32);
    auto position_tensor = engine::core::wrap_tensor(
        positions,
        engine::core::TensorShape::from_dims({batch}),
        GGML_TYPE_I32);
    auto token_emb = engine::modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                         .build(ctx, token_tensor, weights.token_embedding);
    auto pos_emb = engine::modules::EmbeddingModule({config.max_sequence_length, config.hidden_size})
                       .build(ctx, position_tensor, weights.position_embedding);
    auto hidden = engine::modules::AddModule().build(ctx, token_emb, pos_emb);
    hidden = engine::core::reshape_tensor(
        ctx,
        engine::core::ensure_backend_addressable_layout(ctx, hidden),
        engine::core::TensorShape::from_dims({batch, 1, config.hidden_size}));
    return engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true})
        .build(ctx, hidden, weights.embedding_norm);
}

DecoderLayerOutputs decoder_layer(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & encoder_hidden,
    const HviskeDecoderLayerWeights & weights,
    const HviskeDecoderConfig & config,
    const engine::core::TensorValue & self_attention_mask,
    const engine::core::TensorValue & cross_attention_mask) {
    const int64_t dim = head_dim(config);
    const engine::modules::LinearModule q_proj({config.hidden_size, config.hidden_size, true});
    const engine::modules::LinearModule k_proj({config.hidden_size, config.hidden_size, true});
    const engine::modules::LinearModule v_proj({config.hidden_size, config.hidden_size, true});
    const engine::modules::LinearModule out_proj({config.hidden_size, config.hidden_size, true});

    auto x_norm = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true})
                      .build(ctx, input, weights.self_norm);
    auto q = q_proj.build(ctx, x_norm, {weights.self_attn.q_weight, weights.self_attn.q_bias});
    auto k = k_proj.build(ctx, x_norm, {weights.self_attn.k_weight, weights.self_attn.k_bias});
    auto v = v_proj.build(ctx, x_norm, {weights.self_attn.v_weight, weights.self_attn.v_bias});
    q = ai::reshape_heads(ctx, engine::core::ensure_backend_addressable_layout(ctx, q), config.heads, dim);
    k = ai::reshape_heads(ctx, engine::core::ensure_backend_addressable_layout(ctx, k), config.heads, dim);
    v = ai::reshape_heads(ctx, engine::core::ensure_backend_addressable_layout(ctx, v), config.heads, dim);
    auto q_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, self_attention_mask);
    context = engine::modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = engine::core::ensure_backend_addressable_layout(ctx, context);
    context = engine::core::reshape_tensor(
        ctx,
        context,
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.hidden_size}));
    auto hidden = engine::modules::AddModule().build(
        ctx,
        input,
        out_proj.build(ctx, context, {weights.self_attn.out_weight, weights.self_attn.out_bias}));

    auto cross_norm = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true})
                          .build(ctx, hidden, weights.cross_norm);
    auto cq = q_proj.build(ctx, cross_norm, {weights.cross_attn.q_weight, weights.cross_attn.q_bias});
    auto ck = k_proj.build(ctx, encoder_hidden, {weights.cross_attn.k_weight, weights.cross_attn.k_bias});
    auto cv = v_proj.build(ctx, encoder_hidden, {weights.cross_attn.v_weight, weights.cross_attn.v_bias});
    cq = ai::reshape_heads(ctx, engine::core::ensure_backend_addressable_layout(ctx, cq), config.heads, dim);
    ck = ai::reshape_heads(ctx, engine::core::ensure_backend_addressable_layout(ctx, ck), config.heads, dim);
    cv = ai::reshape_heads(ctx, engine::core::ensure_backend_addressable_layout(ctx, cv), config.heads, dim);
    auto cq_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, cq.shape.rank}).build(ctx, cq);
    auto ck_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, ck.shape.rank}).build(ctx, ck);
    auto cv_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, cv.shape.rank}).build(ctx, cv);
    auto cross = attention_from_heads(ctx, cq_heads, ck_heads, cv_heads, dim, cross_attention_mask);
    cross = engine::modules::TransposeModule({{0, 2, 1, 3}, cross.shape.rank}).build(ctx, cross);
    cross = engine::core::ensure_backend_addressable_layout(ctx, cross);
    cross = engine::core::reshape_tensor(
        ctx,
        cross,
        engine::core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.hidden_size}));
    hidden = engine::modules::AddModule().build(
        ctx,
        hidden,
        out_proj.build(ctx, cross, {weights.cross_attn.out_weight, weights.cross_attn.out_bias}));

    auto ff = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true})
                  .build(ctx, hidden, weights.ff_norm);
    ff = engine::modules::LinearModule({config.hidden_size, config.intermediate_size, true})
             .build(ctx, ff, weights.ff_in);
    ff = engine::modules::ReluModule().build(ctx, ff);
    ff = engine::modules::LinearModule({config.intermediate_size, config.hidden_size, true})
             .build(ctx, ff, weights.ff_out);
    return {engine::modules::AddModule().build(ctx, hidden, ff), k, v, ck, cv};
}

DecoderLayerOutputs decoder_layer_with_cache_tail(
    engine::core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & cross_key,
    const engine::core::TensorValue & cross_value,
    const HviskeDecoderLayerWeights & weights,
    const HviskeDecoderConfig & config,
    const engine::core::TensorValue & cache_key,
    const engine::core::TensorValue & cache_value,
    const engine::core::TensorValue & self_attention_mask,
    const engine::core::TensorValue & cross_attention_mask) {
    const int64_t dim = head_dim(config);
    const int64_t scratch_slot = cache_key.shape.dims[1] - 1;
    const engine::modules::LinearModule q_proj({config.hidden_size, config.hidden_size, true});
    const engine::modules::LinearModule k_proj({config.hidden_size, config.hidden_size, true});
    const engine::modules::LinearModule v_proj({config.hidden_size, config.hidden_size, true});
    const engine::modules::LinearModule out_proj({config.hidden_size, config.hidden_size, true});

    auto x_norm = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true})
                      .build(ctx, input, weights.self_norm);
    auto q = q_proj.build(ctx, x_norm, {weights.self_attn.q_weight, weights.self_attn.q_bias});
    auto k = k_proj.build(ctx, x_norm, {weights.self_attn.k_weight, weights.self_attn.k_bias});
    auto v = v_proj.build(ctx, x_norm, {weights.self_attn.v_weight, weights.self_attn.v_bias});
    q = ai::reshape_heads(ctx, engine::core::ensure_backend_addressable_layout(ctx, q), config.heads, dim);
    k = ai::reshape_heads(ctx, engine::core::ensure_backend_addressable_layout(ctx, k), config.heads, dim);
    v = ai::reshape_heads(ctx, engine::core::ensure_backend_addressable_layout(ctx, v), config.heads, dim);
    auto key_tail = engine::runtime::view_transformer_kv_cache_steps(
        ctx,
        cache_key,
        scratch_slot,
        1,
        config.heads,
        dim,
        "Hviske decoder");
    auto value_tail = engine::runtime::view_transformer_kv_cache_steps(
        ctx,
        cache_value,
        scratch_slot,
        1,
        config.heads,
        dim,
        "Hviske decoder");
    ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, k.tensor, key_tail.tensor));
    ggml_build_forward_expand(graph, ggml_cpy(ctx.ggml, v.tensor, value_tail.tensor));
    auto q_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, cache_key.shape.rank}).build(ctx, cache_key);
    auto v_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, cache_value.shape.rank}).build(ctx, cache_value);
    q_heads = engine::core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    k_heads = engine::core::wrap_tensor(ggml_cont(ctx.ggml, k_heads.tensor), k_heads.shape, k_heads.type);
    v_heads = engine::core::wrap_tensor(ggml_cont(ctx.ggml, v_heads.tensor), v_heads.shape, v_heads.type);
    auto context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, self_attention_mask);
    context = engine::modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = engine::core::ensure_backend_addressable_layout(ctx, context);
    context = engine::core::reshape_tensor(ctx, context, engine::core::TensorShape::from_dims({1, 1, config.hidden_size}));
    auto hidden = engine::modules::AddModule().build(
        ctx,
        input,
        out_proj.build(ctx, context, {weights.self_attn.out_weight, weights.self_attn.out_bias}));

    auto cross_norm = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true})
                          .build(ctx, hidden, weights.cross_norm);
    auto cq = q_proj.build(ctx, cross_norm, {weights.cross_attn.q_weight, weights.cross_attn.q_bias});
    cq = ai::reshape_heads(ctx, engine::core::ensure_backend_addressable_layout(ctx, cq), config.heads, dim);
    auto cq_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, cq.shape.rank}).build(ctx, cq);
    auto ck_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, cross_key.shape.rank}).build(ctx, cross_key);
    auto cv_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, cross_value.shape.rank}).build(ctx, cross_value);
    auto cross = attention_from_heads(ctx, cq_heads, ck_heads, cv_heads, dim, cross_attention_mask);
    cross = engine::modules::TransposeModule({{0, 2, 1, 3}, cross.shape.rank}).build(ctx, cross);
    cross = engine::core::ensure_backend_addressable_layout(ctx, cross);
    cross = engine::core::reshape_tensor(ctx, cross, engine::core::TensorShape::from_dims({1, 1, config.hidden_size}));
    hidden = engine::modules::AddModule().build(
        ctx,
        hidden,
        out_proj.build(ctx, cross, {weights.cross_attn.out_weight, weights.cross_attn.out_bias}));

    auto ff = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true})
                  .build(ctx, hidden, weights.ff_norm);
    ff = engine::modules::LinearModule({config.hidden_size, config.intermediate_size, true})
             .build(ctx, ff, weights.ff_in);
    ff = engine::modules::ReluModule().build(ctx, ff);
    ff = engine::modules::LinearModule({config.intermediate_size, config.hidden_size, true})
             .build(ctx, ff, weights.ff_out);
    return {engine::modules::AddModule().build(ctx, hidden, ff), k, v, {}, {}};
}

DecoderLayerOutputs decoder_layer_with_beam_cache_tail(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & cross_key,
    const engine::core::TensorValue & cross_value,
    const HviskeDecoderLayerWeights & weights,
    const HviskeDecoderConfig & config,
    const engine::core::TensorValue & cache_key,
    const engine::core::TensorValue & cache_value,
    const engine::core::TensorValue & cache_slots,
    const engine::core::TensorValue & self_attention_mask,
    const engine::core::TensorValue & cross_attention_mask) {
    const int64_t batch = input.shape.dims[0];
    const int64_t cache_steps = cache_key.shape.dims[1];
    const int64_t dim = head_dim(config);
    const int64_t row_elems = config.heads * dim;
    const engine::modules::LinearModule q_proj({config.hidden_size, config.hidden_size, true});
    const engine::modules::LinearModule k_proj({config.hidden_size, config.hidden_size, true});
    const engine::modules::LinearModule v_proj({config.hidden_size, config.hidden_size, true});
    const engine::modules::LinearModule out_proj({config.hidden_size, config.hidden_size, true});

    auto x_norm = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true})
                      .build(ctx, input, weights.self_norm);
    auto q = q_proj.build(ctx, x_norm, {weights.self_attn.q_weight, weights.self_attn.q_bias});
    auto k = k_proj.build(ctx, x_norm, {weights.self_attn.k_weight, weights.self_attn.k_bias});
    auto v = v_proj.build(ctx, x_norm, {weights.self_attn.v_weight, weights.self_attn.v_bias});
    q = ai::reshape_heads(ctx, engine::core::ensure_backend_addressable_layout(ctx, q), config.heads, dim);
    k = ai::reshape_heads(ctx, engine::core::ensure_backend_addressable_layout(ctx, k), config.heads, dim);
    v = ai::reshape_heads(ctx, engine::core::ensure_backend_addressable_layout(ctx, v), config.heads, dim);

    auto flat_key_cache = engine::core::reshape_tensor(
        ctx,
        cache_key,
        engine::core::TensorShape::from_dims({batch * cache_steps, row_elems}));
    auto flat_value_cache = engine::core::reshape_tensor(
        ctx,
        cache_value,
        engine::core::TensorShape::from_dims({batch * cache_steps, row_elems}));
    auto flat_key_row = engine::core::reshape_tensor(
        ctx,
        engine::core::ensure_backend_addressable_layout(ctx, k),
        engine::core::TensorShape::from_dims({batch, row_elems}));
    auto flat_value_row = engine::core::reshape_tensor(
        ctx,
        engine::core::ensure_backend_addressable_layout(ctx, v),
        engine::core::TensorShape::from_dims({batch, row_elems}));
    auto updated_key = engine::core::wrap_tensor(
        ggml_set_rows(ctx.ggml, flat_key_cache.tensor, flat_key_row.tensor, cache_slots.tensor),
        flat_key_cache.shape,
        GGML_TYPE_F32);
    auto updated_value = engine::core::wrap_tensor(
        ggml_set_rows(ctx.ggml, flat_value_cache.tensor, flat_value_row.tensor, cache_slots.tensor),
        flat_value_cache.shape,
        GGML_TYPE_F32);
    updated_key = engine::core::reshape_tensor(ctx, updated_key, cache_key.shape);
    updated_value = engine::core::reshape_tensor(ctx, updated_value, cache_value.shape);

    auto q_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, updated_key.shape.rank}).build(ctx, updated_key);
    auto v_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, updated_value.shape.rank}).build(ctx, updated_value);
    q_heads = engine::core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    k_heads = engine::core::wrap_tensor(ggml_cont(ctx.ggml, k_heads.tensor), k_heads.shape, k_heads.type);
    v_heads = engine::core::wrap_tensor(ggml_cont(ctx.ggml, v_heads.tensor), v_heads.shape, v_heads.type);
    auto context = attention_from_heads(ctx, q_heads, k_heads, v_heads, dim, self_attention_mask);
    context = engine::modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = engine::core::ensure_backend_addressable_layout(ctx, context);
    context = engine::core::reshape_tensor(
        ctx,
        context,
        engine::core::TensorShape::from_dims({batch, 1, config.hidden_size}));
    auto hidden = engine::modules::AddModule().build(
        ctx,
        input,
        out_proj.build(ctx, context, {weights.self_attn.out_weight, weights.self_attn.out_bias}));

    auto cross_norm = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true})
                          .build(ctx, hidden, weights.cross_norm);
    auto cq = q_proj.build(ctx, cross_norm, {weights.cross_attn.q_weight, weights.cross_attn.q_bias});
    cq = ai::reshape_heads(ctx, engine::core::ensure_backend_addressable_layout(ctx, cq), config.heads, dim);
    auto cq_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, cq.shape.rank}).build(ctx, cq);
    auto ck_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, cross_key.shape.rank}).build(ctx, cross_key);
    auto cv_heads = engine::modules::TransposeModule({{0, 2, 1, 3}, cross_value.shape.rank}).build(ctx, cross_value);
    cq_heads = engine::core::wrap_tensor(ggml_cont(ctx.ggml, cq_heads.tensor), cq_heads.shape, cq_heads.type);
    ck_heads = engine::core::wrap_tensor(ggml_cont(ctx.ggml, ck_heads.tensor), ck_heads.shape, ck_heads.type);
    cv_heads = engine::core::wrap_tensor(ggml_cont(ctx.ggml, cv_heads.tensor), cv_heads.shape, cv_heads.type);
    auto cross = attention_from_heads(ctx, cq_heads, ck_heads, cv_heads, dim, cross_attention_mask);
    cross = engine::modules::TransposeModule({{0, 2, 1, 3}, cross.shape.rank}).build(ctx, cross);
    cross = engine::core::ensure_backend_addressable_layout(ctx, cross);
    cross = engine::core::reshape_tensor(
        ctx,
        cross,
        engine::core::TensorShape::from_dims({batch, 1, config.hidden_size}));
    hidden = engine::modules::AddModule().build(
        ctx,
        hidden,
        out_proj.build(ctx, cross, {weights.cross_attn.out_weight, weights.cross_attn.out_bias}));

    auto ff = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true})
                  .build(ctx, hidden, weights.ff_norm);
    ff = engine::modules::LinearModule({config.hidden_size, config.intermediate_size, true})
             .build(ctx, ff, weights.ff_in);
    ff = engine::modules::ReluModule().build(ctx, ff);
    ff = engine::modules::LinearModule({config.intermediate_size, config.hidden_size, true})
             .build(ctx, ff, weights.ff_out);
    return {engine::modules::AddModule().build(ctx, hidden, ff), k, v, {}, {}};
}

std::vector<float> make_self_prefill_mask(int64_t steps) {
    std::vector<float> mask(static_cast<size_t>(steps * steps), 0.0f);
    for (int64_t q = 0; q < steps; ++q) {
        for (int64_t k = q + 1; k < steps; ++k) {
            mask[static_cast<size_t>(q * steps + k)] = -std::numeric_limits<float>::infinity();
        }
    }
    return mask;
}

std::vector<float> make_cross_mask(int64_t query_steps, int64_t encoder_frames, int64_t valid_frames) {
    std::vector<float> mask(static_cast<size_t>(query_steps * encoder_frames), 0.0f);
    const int64_t valid = std::clamp<int64_t>(valid_frames, 0, encoder_frames);
    for (int64_t q = 0; q < query_steps; ++q) {
        for (int64_t k = valid; k < encoder_frames; ++k) {
            mask[static_cast<size_t>(q * encoder_frames + k)] = -1.0e9f;
        }
    }
    return mask;
}

}  // namespace

struct HviskeDecoderRuntime::PrefillGraph {
    PrefillGraph(
        std::shared_ptr<const HviskeAssets> assets,
        std::shared_ptr<const HviskeWeights> weights,
        engine::core::ExecutionContext & execution_context,
        int64_t prompt_steps,
        int64_t encoder_frames,
        size_t graph_arena_bytes)
        : assets_(std::move(assets)),
          weights_(std::move(weights)),
          execution_context_(&execution_context),
          prompt_steps_(prompt_steps),
          encoder_frames_(encoder_frames) {
        const auto build_start = Clock::now();
        const auto & config = assets_->config.decoder;
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("Failed to initialize Hviske decoder prefill ggml context");
        }
        engine::core::ModuleBuildContext ctx{ctx_.get(), "hviske_asr.decoder.prefill", execution_context_->backend_type()};
        token_ids_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_I32, prompt_steps_, 1);
        positions_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_I32, prompt_steps_, 1);
        encoder_hidden_ = ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, config.hidden_size, encoder_frames_, 1);
        self_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F32, prompt_steps_, prompt_steps_, 1, 1);
        cross_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F32, encoder_frames_, prompt_steps_, 1, 1);

        auto x = decoder_embeddings(ctx, weights_->decoder, config, token_ids_, positions_, prompt_steps_);
        auto encoder = engine::core::wrap_tensor(
            encoder_hidden_,
            engine::core::TensorShape::from_dims({1, encoder_frames_, config.hidden_size}),
            GGML_TYPE_F32);
        auto self_mask = engine::core::wrap_tensor(
            self_mask_,
            engine::core::TensorShape::from_dims({1, 1, prompt_steps_, prompt_steps_}),
            GGML_TYPE_F32);
        auto cross_mask = engine::core::wrap_tensor(
            cross_mask_,
            engine::core::TensorShape::from_dims({1, 1, prompt_steps_, encoder_frames_}),
            GGML_TYPE_F32);
        for (const auto & layer : weights_->decoder.layers) {
            auto out = decoder_layer(ctx, x, encoder, layer, config, self_mask, cross_mask);
            x = out.output;
            keys_.push_back(out.key.tensor);
            values_.push_back(out.value.tensor);
            cross_keys_.push_back(out.cross_key.tensor);
            cross_values_.push_back(out.cross_value.tensor);
        }
        build_step_source_views(config.heads * head_dim(config));
        x = engine::modules::SliceModule({1, prompt_steps_ - 1, 1}).build(ctx, x);
        x = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true})
                .build(ctx, x, weights_->decoder.final_norm);
        auto logits = decoder_classifier_output(ctx, x, weights_->decoder, config);
        logits_ = logits.tensor;
        ggml_set_output(logits_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
        ggml_build_forward_expand(graph_, logits_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), execution_context_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("Failed to allocate Hviske decoder prefill graph");
        }
        std::vector<int32_t> positions(static_cast<size_t>(prompt_steps_), 0);
        for (int64_t i = 0; i < prompt_steps_; ++i) {
            positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        ggml_backend_tensor_set(positions_, positions.data(), 0, positions.size() * sizeof(int32_t));
        const auto self_mask_values = make_self_prefill_mask(prompt_steps_);
        ggml_backend_tensor_set(self_mask_, self_mask_values.data(), 0, self_mask_values.size() * sizeof(float));
        const double build_ms = engine::debug::elapsed_ms(build_start, Clock::now());
        debug::timing_log_scalar("hviske_asr.decoder.prefill.graph_build_ms", build_ms);
        debug::timing_log_scalar("hviske_asr.decoder.prefill.graph_rebuild_ms", build_ms);
        debug::trace_log_scalar("hviske_asr.decoder.prefill_prompt_steps", prompt_steps_);
        debug::trace_log_scalar("hviske_asr.decoder.prefill_encoder_frames", encoder_frames_);
    }

    ~PrefillGraph() {
        engine::core::release_backend_graph_resources(execution_context_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool matches(const HviskeWeights & weights, ggml_backend_t backend, int64_t prompt_steps, int64_t encoder_frames) const {
        return weights_.get() == &weights &&
            execution_context_->backend() == backend &&
            prompt_steps_ == prompt_steps &&
            encoder_frames_ == encoder_frames;
    }

    PrefillOutput run(const std::vector<int32_t> & prompt_ids, const HviskeEncodedAudio & encoded) {
        const auto & config = assets_->config.decoder;
        if (static_cast<int64_t>(prompt_ids.size()) != prompt_steps_) {
            throw std::runtime_error("Hviske decoder prefill prompt length mismatch");
        }
        if (encoded.frames != encoder_frames_ ||
            encoded.hidden_size != config.hidden_size ||
            static_cast<int64_t>(encoded.values.size()) != encoder_frames_ * config.hidden_size) {
            throw std::runtime_error("Hviske decoder prefill encoder shape mismatch");
        }
        const auto upload_start = Clock::now();
        ggml_backend_tensor_set(token_ids_, prompt_ids.data(), 0, prompt_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(encoder_hidden_, encoded.values.data(), 0, encoded.values.size() * sizeof(float));
        const auto cross_mask = make_cross_mask(prompt_steps_, encoder_frames_, encoded.valid_frames);
        ggml_backend_tensor_set(cross_mask_, cross_mask.data(), 0, cross_mask.size() * sizeof(float));
        debug::timing_log_scalar("hviske_asr.decoder.prefill_input_upload_ms", engine::debug::elapsed_ms(upload_start, Clock::now()));

        engine::core::set_backend_threads(execution_context_->backend(), execution_context_->config().threads);
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(execution_context_->backend(), graph_, nullptr, "Hviske decoder prefill");
        ggml_backend_synchronize(execution_context_->backend());
        debug::timing_log_scalar("hviske_asr.decoder.prefill.graph.compute_ms", engine::debug::elapsed_ms(compute_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Hviske decoder prefill graph compute failed");
        }

        const auto read_start = Clock::now();
        PrefillOutput out;
        out.logits.resize(static_cast<size_t>(config.vocab_size));
        ggml_backend_tensor_get(logits_, out.logits.data(), 0, out.logits.size() * sizeof(float));
        debug::timing_log_scalar("hviske_asr.decoder.prefill_output_read_ms", engine::debug::elapsed_ms(read_start, Clock::now()));
        return out;
    }

    int64_t prompt_steps() const noexcept {
        return prompt_steps_;
    }

    int64_t encoder_frames() const noexcept {
        return encoder_frames_;
    }

    size_t layer_count() const noexcept {
        return keys_.size();
    }

    ggml_tensor * key_step_source(size_t step, size_t layer) const {
        return key_step_sources_.at(step).at(layer);
    }

    ggml_tensor * value_step_source(size_t step, size_t layer) const {
        return value_step_sources_.at(step).at(layer);
    }

    ggml_tensor * cross_key(size_t layer) const {
        return cross_keys_.at(layer);
    }

    ggml_tensor * cross_value(size_t layer) const {
        return cross_values_.at(layer);
    }

    ggml_tensor * cross_key_frame_source(size_t frame, size_t layer) const {
        return cross_key_frame_sources_.at(frame).at(layer);
    }

    ggml_tensor * cross_value_frame_source(size_t frame, size_t layer) const {
        return cross_value_frame_sources_.at(frame).at(layer);
    }

private:
    void build_step_source_views(int64_t step_elems) {
        key_step_sources_.assign(static_cast<size_t>(prompt_steps_), {});
        value_step_sources_.assign(static_cast<size_t>(prompt_steps_), {});
        for (int64_t step = 0; step < prompt_steps_; ++step) {
            auto & key_slot = key_step_sources_[static_cast<size_t>(step)];
            auto & value_slot = value_step_sources_[static_cast<size_t>(step)];
            key_slot.reserve(keys_.size());
            value_slot.reserve(values_.size());
            for (size_t layer = 0; layer < keys_.size(); ++layer) {
                const size_t key_offset = static_cast<size_t>(step) * keys_[layer]->nb[2];
                const size_t value_offset = static_cast<size_t>(step) * values_[layer]->nb[2];
                key_slot.push_back(ggml_view_1d(ctx_.get(), keys_[layer], step_elems, key_offset));
                value_slot.push_back(ggml_view_1d(ctx_.get(), values_[layer], step_elems, value_offset));
            }
        }
        cross_key_frame_sources_.assign(static_cast<size_t>(encoder_frames_), {});
        cross_value_frame_sources_.assign(static_cast<size_t>(encoder_frames_), {});
        for (int64_t frame = 0; frame < encoder_frames_; ++frame) {
            auto & key_slot = cross_key_frame_sources_[static_cast<size_t>(frame)];
            auto & value_slot = cross_value_frame_sources_[static_cast<size_t>(frame)];
            key_slot.reserve(cross_keys_.size());
            value_slot.reserve(cross_values_.size());
            for (size_t layer = 0; layer < cross_keys_.size(); ++layer) {
                const size_t key_offset = static_cast<size_t>(frame) * cross_keys_[layer]->nb[2];
                const size_t value_offset = static_cast<size_t>(frame) * cross_values_[layer]->nb[2];
                key_slot.push_back(ggml_view_1d(ctx_.get(), cross_keys_[layer], step_elems, key_offset));
                value_slot.push_back(ggml_view_1d(ctx_.get(), cross_values_[layer], step_elems, value_offset));
            }
        }
    }

    std::shared_ptr<const HviskeAssets> assets_;
    std::shared_ptr<const HviskeWeights> weights_;
    engine::core::ExecutionContext * execution_context_ = nullptr;
    int64_t prompt_steps_ = 0;
    int64_t encoder_frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ids_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * encoder_hidden_ = nullptr;
    ggml_tensor * self_mask_ = nullptr;
    ggml_tensor * cross_mask_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    std::vector<ggml_tensor *> cross_keys_;
    std::vector<ggml_tensor *> cross_values_;
    std::vector<std::vector<ggml_tensor *>> key_step_sources_;
    std::vector<std::vector<ggml_tensor *>> value_step_sources_;
    std::vector<std::vector<ggml_tensor *>> cross_key_frame_sources_;
    std::vector<std::vector<ggml_tensor *>> cross_value_frame_sources_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

struct HviskeDecoderRuntime::DecodeGraph {
    DecodeGraph(
        std::shared_ptr<const HviskeAssets> assets,
        std::shared_ptr<const HviskeWeights> weights,
        engine::core::ExecutionContext & execution_context,
        int64_t encoder_frames,
        int64_t cache_steps,
        size_t graph_arena_bytes)
        : assets_(std::move(assets)),
          weights_(std::move(weights)),
          execution_context_(&execution_context),
          encoder_frames_(encoder_frames),
          cache_steps_(cache_steps) {
        const auto build_start = Clock::now();
        const auto & config = assets_->config.decoder;
        const int64_t dim = head_dim(config);
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("Failed to initialize Hviske decoder decode ggml context");
        }
        engine::core::ModuleBuildContext ctx{ctx_.get(), "hviske_asr.decoder.decode", execution_context_->backend_type()};
        token_id_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_I32, 1, 1);
        position_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_I32, 1, 1);
        self_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F32, cache_steps_ + 1, 1, 1, 1);
        cross_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F32, encoder_frames_, 1, 1, 1);
        auto x = decoder_embeddings(ctx, weights_->decoder, config, token_id_, position_, 1);
        auto self_mask = engine::core::wrap_tensor(
            self_mask_,
            engine::core::TensorShape::from_dims({1, 1, 1, cache_steps_ + 1}),
            GGML_TYPE_F32);
        auto cross_mask = engine::core::wrap_tensor(
            cross_mask_,
            engine::core::TensorShape::from_dims({1, 1, 1, encoder_frames_}),
            GGML_TYPE_F32);
        graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false);
        std::vector<engine::core::TensorValue> cache_keys;
        std::vector<engine::core::TensorValue> cache_values;
        std::vector<engine::core::TensorValue> cross_keys;
        std::vector<engine::core::TensorValue> cross_values;
        for (const auto & layer : weights_->decoder.layers) {
            cache_keys.push_back(engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, cache_steps_ + 1, config.heads, dim})));
            cache_values.push_back(engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, cache_steps_ + 1, config.heads, dim})));
            cross_keys.push_back(engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, encoder_frames_, config.heads, dim})));
            cross_values.push_back(engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, encoder_frames_, config.heads, dim})));
            auto out = decoder_layer_with_cache_tail(
                ctx,
                graph_,
                x,
                cross_keys.back(),
                cross_values.back(),
                layer,
                config,
                cache_keys.back(),
                cache_values.back(),
                self_mask,
                cross_mask);
            x = out.output;
            key_sources_.push_back(ggml_view_1d(ctx_.get(), out.key.tensor, config.heads * dim, 0));
            value_sources_.push_back(ggml_view_1d(ctx_.get(), out.value.tensor, config.heads * dim, 0));
        }
        kv_cache_ = engine::runtime::TransformerKVCache(
            cache_steps_ + 1,
            config.heads * dim,
            std::move(cache_keys),
            std::move(cache_values));
        cross_cache_ = engine::runtime::TransformerKVCache(
            encoder_frames_,
            config.heads * dim,
            std::move(cross_keys),
            std::move(cross_values));
        build_transfer_views(config.heads * dim);
        x = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true})
                .build(ctx, x, weights_->decoder.final_norm);
        auto logits = decoder_classifier_output(ctx, x, weights_->decoder, config);
        logits_ = logits.tensor;
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, logits_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), execution_context_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("Failed to allocate Hviske decoder decode graph");
        }
        self_mask_values_.assign(static_cast<size_t>(cache_steps_ + 1), -std::numeric_limits<float>::infinity());
        const double build_ms = engine::debug::elapsed_ms(build_start, Clock::now());
        debug::timing_log_scalar("hviske_asr.decoder.decode.graph_build_ms", build_ms);
        debug::timing_log_scalar("hviske_asr.decoder.decode.graph_rebuild_ms", build_ms);
        debug::trace_log_scalar("hviske_asr.decoder.decode_cache_steps", cache_steps_);
        debug::trace_log_scalar("hviske_asr.decoder.decode_encoder_frames", encoder_frames_);
    }

    ~DecodeGraph() {
        engine::core::release_backend_graph_resources(execution_context_->backend(), graph_);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool can_run(const HviskeWeights & weights, ggml_backend_t backend, int64_t encoder_frames, int64_t required_steps) const {
        return weights_.get() == &weights &&
            execution_context_->backend() == backend &&
            encoder_frames_ == encoder_frames &&
            cache_steps_ >= required_steps;
    }

    void import_state(
        const PrefillGraph & prefill,
        const HviskeEncodedAudio & encoded) {
        const auto import_start = Clock::now();
        const auto & config = assets_->config.decoder;
        if (encoded.frames != encoder_frames_ ||
            encoded.hidden_size != config.hidden_size ||
            static_cast<int64_t>(encoded.values.size()) != encoder_frames_ * config.hidden_size) {
            throw std::runtime_error("Hviske decoder decode encoder shape mismatch");
        }
        if (prefill.encoder_frames() != encoder_frames_ ||
            prefill.layer_count() != key_sources_.size() ||
            prefill.prompt_steps() > cache_steps_) {
            throw std::runtime_error("Hviske decoder decode prefill state shape mismatch");
        }
        kv_cache_.retain_prefix(0);
        for (int64_t step = 0; step < prefill.prompt_steps(); ++step) {
            const size_t slot = static_cast<size_t>(step);
            for (size_t layer = 0; layer < key_sources_.size(); ++layer) {
                ggml_backend_tensor_copy(prefill.key_step_source(slot, layer), key_destinations_[slot][layer]);
                ggml_backend_tensor_copy(prefill.value_step_source(slot, layer), value_destinations_[slot][layer]);
            }
        }
        kv_cache_.advance_after_direct_append(prefill.prompt_steps());
        for (size_t layer = 0; layer < key_sources_.size(); ++layer) {
            ggml_backend_tensor_copy(prefill.cross_key(layer), cross_cache_.key_tensor(layer).tensor);
            ggml_backend_tensor_copy(prefill.cross_value(layer), cross_cache_.value_tensor(layer).tensor);
        }
        const auto cross_mask = make_cross_mask(1, encoder_frames_, encoded.valid_frames);
        ggml_backend_tensor_set(cross_mask_, cross_mask.data(), 0, cross_mask.size() * sizeof(float));
        debug::timing_log_scalar("hviske_asr.decoder.decode.import_state_ms", engine::debug::elapsed_ms(import_start, Clock::now()));
    }

    std::vector<float> run_step(int32_t token) {
        const auto & config = assets_->config.decoder;
        if (kv_cache_.valid_steps() >= cache_steps_) {
            throw std::runtime_error("Hviske decoder decode cache exhausted");
        }
        ggml_backend_tensor_set(token_id_, &token, 0, sizeof(int32_t));
        const int32_t position = static_cast<int32_t>(kv_cache_.current_end());
        ggml_backend_tensor_set(position_, &position, 0, sizeof(int32_t));

        std::fill(self_mask_values_.begin(), self_mask_values_.end(), -std::numeric_limits<float>::infinity());
        for (int64_t i = 0; i < kv_cache_.valid_steps(); ++i) {
            self_mask_values_[static_cast<size_t>(i)] = 0.0f;
        }
        self_mask_values_[static_cast<size_t>(cache_steps_)] = 0.0f;
        ggml_backend_tensor_set(
            self_mask_,
            self_mask_values_.data(),
            0,
            self_mask_values_.size() * sizeof(float));
        engine::core::set_backend_threads(execution_context_->backend(), execution_context_->config().threads);
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(execution_context_->backend(), graph_, nullptr, "Hviske decoder decode");
        ggml_backend_synchronize(execution_context_->backend());
        debug::timing_log_scalar("hviske_asr.decoder.decode.step.graph.compute_ms", engine::debug::elapsed_ms(compute_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Hviske decoder decode graph compute failed");
        }

        std::vector<float> logits(static_cast<size_t>(config.vocab_size));
        ggml_backend_tensor_get(logits_, logits.data(), 0, logits.size() * sizeof(float));
        const size_t dst_slot = static_cast<size_t>(kv_cache_.valid_steps());
        for (size_t layer = 0; layer < key_sources_.size(); ++layer) {
            ggml_backend_tensor_copy(key_sources_[layer], key_destinations_[dst_slot][layer]);
            ggml_backend_tensor_copy(value_sources_[layer], value_destinations_[dst_slot][layer]);
        }
        kv_cache_.advance_after_direct_append(1);
        return logits;
    }

private:
    void build_transfer_views(int64_t step_elems) {
        key_destinations_.assign(static_cast<size_t>(cache_steps_), {});
        value_destinations_.assign(static_cast<size_t>(cache_steps_), {});
        for (int64_t slot = 0; slot < cache_steps_; ++slot) {
            const size_t byte_offset = static_cast<size_t>(slot * step_elems) * sizeof(float);
            auto & key_slot = key_destinations_[static_cast<size_t>(slot)];
            auto & value_slot = value_destinations_[static_cast<size_t>(slot)];
            key_slot.reserve(key_sources_.size());
            value_slot.reserve(value_sources_.size());
            for (size_t layer = 0; layer < key_sources_.size(); ++layer) {
                key_slot.push_back(ggml_view_1d(ctx_.get(), kv_cache_.key_tensor(layer).tensor, step_elems, byte_offset));
                value_slot.push_back(ggml_view_1d(ctx_.get(), kv_cache_.value_tensor(layer).tensor, step_elems, byte_offset));
            }
        }
    }

    std::shared_ptr<const HviskeAssets> assets_;
    std::shared_ptr<const HviskeWeights> weights_;
    engine::core::ExecutionContext * execution_context_ = nullptr;
    int64_t encoder_frames_ = 0;
    int64_t cache_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_id_ = nullptr;
    ggml_tensor * position_ = nullptr;
    ggml_tensor * self_mask_ = nullptr;
    ggml_tensor * cross_mask_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_tensor *> key_sources_;
    std::vector<ggml_tensor *> value_sources_;
    std::vector<std::vector<ggml_tensor *>> key_destinations_;
    std::vector<std::vector<ggml_tensor *>> value_destinations_;
    std::vector<float> self_mask_values_;
    engine::runtime::TransformerKVCache kv_cache_;
    engine::runtime::TransformerKVCache cross_cache_;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
};

struct HviskeDecoderRuntime::BeamDecodeGraph {
    struct StepOutput {
        std::vector<float> logits;
    };

    struct BatchOutput {
        std::vector<StepOutput> steps;
    };

    BeamDecodeGraph(
        std::shared_ptr<const HviskeAssets> assets,
        std::shared_ptr<const HviskeWeights> weights,
        engine::core::ExecutionContext & execution_context,
        int64_t encoder_frames,
        int64_t cache_steps,
        int64_t beam_count,
        size_t graph_arena_bytes)
        : assets_(std::move(assets)),
          weights_(std::move(weights)),
          execution_context_(&execution_context),
          encoder_frames_(encoder_frames),
          cache_steps_(cache_steps),
          beam_count_(beam_count),
          beam_slots_(2 * beam_count) {
        const auto build_start = Clock::now();
        const auto & config = assets_->config.decoder;
        const int64_t dim = head_dim(config);
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("Failed to initialize Hviske decoder beam decode ggml context");
        }
        engine::core::ModuleBuildContext ctx{ctx_.get(), "hviske_asr.decoder.beam_decode", execution_context_->backend_type()};
        token_ids_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, beam_count_);
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, beam_count_);
        cache_slots_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, beam_count_);
        self_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F32, cache_steps_, 1, 1, beam_count_);
        cross_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F32, encoder_frames_, 1, 1, beam_count_);

        for (int64_t bank = 0; bank < 2; ++bank) {
            auto & keys = bank_keys_[static_cast<size_t>(bank)];
            auto & values = bank_values_[static_cast<size_t>(bank)];
            keys.reserve(weights_->decoder.layers.size());
            values.reserve(weights_->decoder.layers.size());
            for (size_t layer = 0; layer < weights_->decoder.layers.size(); ++layer) {
                keys.push_back(engine::core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    engine::core::TensorShape::from_dims({beam_count_, cache_steps_, config.heads, dim})));
                values.push_back(engine::core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    engine::core::TensorShape::from_dims({beam_count_, cache_steps_, config.heads, dim})));
            }
        }
        cross_keys_.reserve(weights_->decoder.layers.size());
        cross_values_.reserve(weights_->decoder.layers.size());
        for (size_t layer = 0; layer < weights_->decoder.layers.size(); ++layer) {
            cross_keys_.push_back(engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, encoder_frames_, config.heads, dim})));
            cross_values_.push_back(engine::core::make_tensor(
                ctx,
                GGML_TYPE_F32,
                engine::core::TensorShape::from_dims({1, encoder_frames_, config.heads, dim})));
        }

        build_prefix_views(config.heads * dim);
        build_bank_graph(ctx, 0);
        build_bank_graph(ctx, 1);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), execution_context_->backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("Failed to allocate Hviske decoder beam decode graph");
        }
        token_values_.assign(static_cast<size_t>(beam_count_), 0);
        position_values_.assign(static_cast<size_t>(beam_count_), 0);
        cache_slot_values_.assign(static_cast<size_t>(beam_count_), 0);
        self_mask_values_.assign(static_cast<size_t>(beam_count_ * cache_steps_), -std::numeric_limits<float>::infinity());
        cross_mask_values_.assign(static_cast<size_t>(beam_count_ * encoder_frames_), 0.0f);
        const double build_ms = engine::debug::elapsed_ms(build_start, Clock::now());
        debug::timing_log_scalar("hviske_asr.decoder.beam_decode.graph_build_ms", build_ms);
        debug::timing_log_scalar("hviske_asr.decoder.beam_decode.graph_rebuild_ms", build_ms);
        debug::trace_log_scalar("hviske_asr.decoder.beam_decode.beam_count", beam_count_);
        debug::trace_log_scalar("hviske_asr.decoder.beam_decode.cache_steps", cache_steps_);
        debug::trace_log_scalar("hviske_asr.decoder.beam_decode.encoder_frames", encoder_frames_);
    }

    ~BeamDecodeGraph() {
        for (auto & graph : bank_graphs_) {
            engine::core::release_backend_graph_resources(execution_context_->backend(), graph.graph);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool can_run(
        const HviskeWeights & weights,
        ggml_backend_t backend,
        int64_t encoder_frames,
        int64_t required_steps,
        int64_t beam_count) const {
        return weights_.get() == &weights &&
            execution_context_->backend() == backend &&
            encoder_frames_ == encoder_frames &&
            cache_steps_ >= required_steps &&
            beam_count_ == beam_count;
    }

    void initialize_from_prefill(const PrefillGraph & prefill, const HviskeEncodedAudio & encoded, int64_t slot_count) {
        const auto init_start = Clock::now();
        prefix_copy_ms_ = 0.0;
        input_upload_ms_ = 0.0;
        graph_compute_ms_ = 0.0;
        output_read_ms_ = 0.0;
        step_count_ = 0;
        if (slot_count <= 0 || slot_count > beam_slots_ ||
            encoded.frames != encoder_frames_ ||
            prefill.encoder_frames() != encoder_frames_ ||
            prefill.prompt_steps() > cache_steps_ ||
            prefill.layer_count() != bank_keys_[0].size()) {
            throw std::runtime_error("Hviske decoder beam prefill state shape mismatch");
        }
        const int64_t prompt_steps = prefill.prompt_steps();
        for (int64_t slot = 0; slot < slot_count; ++slot) {
            for (int64_t step = 0; step < prompt_steps; ++step) {
                for (size_t layer = 0; layer < bank_keys_[0].size(); ++layer) {
                    ggml_backend_tensor_copy(
                        prefill.key_step_source(static_cast<size_t>(step), layer),
                        beam_key_step_views_[static_cast<size_t>(slot)][static_cast<size_t>(step)][layer]);
                    ggml_backend_tensor_copy(
                        prefill.value_step_source(static_cast<size_t>(step), layer),
                        beam_value_step_views_[static_cast<size_t>(slot)][static_cast<size_t>(step)][layer]);
                }
            }
        }
        for (size_t layer = 0; layer < bank_keys_[0].size(); ++layer) {
            ggml_backend_tensor_copy(prefill.cross_key(layer), cross_keys_[layer].tensor);
            ggml_backend_tensor_copy(prefill.cross_value(layer), cross_values_[layer].tensor);
        }
        const auto cross_mask = make_cross_mask(1, encoder_frames_, encoded.valid_frames);
        std::fill(cross_mask_values_.begin(), cross_mask_values_.end(), 0.0f);
        for (int64_t row = 0; row < beam_count_; ++row) {
            std::copy(
                cross_mask.begin(),
                cross_mask.end(),
                cross_mask_values_.begin() + static_cast<std::ptrdiff_t>(row * encoder_frames_));
        }
        ggml_backend_tensor_set(cross_mask_, cross_mask_values_.data(), 0, cross_mask_values_.size() * sizeof(float));
        debug::timing_log_scalar("hviske_asr.decoder.beam_decode.import_state_ms", engine::debug::elapsed_ms(init_start, Clock::now()));
    }

    BatchOutput run_batch_from_beams(
        const std::vector<int64_t> & parent_slots,
        const std::vector<int64_t> & child_slots,
        int64_t valid_steps,
        const std::vector<int32_t> & tokens) {
        const size_t active = parent_slots.size();
        if (active == 0 || child_slots.size() != active || tokens.size() != active) {
            throw std::runtime_error("Hviske decoder beam batch input shape mismatch");
        }
        if (active > static_cast<size_t>(beam_count_)) {
            throw std::runtime_error("Hviske decoder beam batch exceeds graph capacity");
        }
        if (valid_steps < 0 || valid_steps >= cache_steps_) {
            throw std::runtime_error("Hviske decoder beam decode cache exhausted");
        }
        const auto prefix_start = Clock::now();
        const int64_t child_bank = child_slots.front() / beam_count_;
        if (child_bank < 0 || child_bank > 1) {
            throw std::runtime_error("Hviske decoder beam child bank is out of range");
        }
        for (size_t row = 0; row < active; ++row) {
            if (parent_slots[row] < 0 || parent_slots[row] >= beam_slots_ ||
                child_slots[row] < 0 || child_slots[row] >= beam_slots_ ||
                child_slots[row] / beam_count_ != child_bank ||
                child_slots[row] % beam_count_ != static_cast<int64_t>(row)) {
                throw std::runtime_error("Hviske decoder beam slot layout mismatch");
            }
            for (size_t layer = 0; layer < bank_keys_[0].size(); ++layer) {
                copy_beam_prefix(parent_slots[row], child_slots[row], valid_steps, layer);
            }
            token_values_[row] = tokens[row];
            position_values_[row] = static_cast<int32_t>(valid_steps);
        }
        for (size_t row = active; row < static_cast<size_t>(beam_count_); ++row) {
            const int64_t child_slot = child_bank * beam_count_ + static_cast<int64_t>(row);
            for (size_t layer = 0; layer < bank_keys_[0].size(); ++layer) {
                copy_beam_prefix(child_slots.front(), child_slot, valid_steps, layer);
            }
            token_values_[row] = tokens.front();
            position_values_[row] = static_cast<int32_t>(valid_steps);
        }
        prefix_copy_ms_ += engine::debug::elapsed_ms(prefix_start, Clock::now());

        const auto upload_start = Clock::now();
        std::fill(self_mask_values_.begin(), self_mask_values_.end(), -std::numeric_limits<float>::infinity());
        for (int64_t row = 0; row < beam_count_; ++row) {
            auto * row_values = self_mask_values_.data() + static_cast<std::ptrdiff_t>(row * cache_steps_);
            for (int64_t step = 0; step <= valid_steps; ++step) {
                row_values[static_cast<size_t>(step)] = 0.0f;
            }
            cache_slot_values_[static_cast<size_t>(row)] = static_cast<int32_t>(row * cache_steps_ + valid_steps);
        }
        ggml_backend_tensor_set(token_ids_, token_values_.data(), 0, token_values_.size() * sizeof(int32_t));
        ggml_backend_tensor_set(positions_, position_values_.data(), 0, position_values_.size() * sizeof(int32_t));
        ggml_backend_tensor_set(cache_slots_, cache_slot_values_.data(), 0, cache_slot_values_.size() * sizeof(int32_t));
        ggml_backend_tensor_set(self_mask_, self_mask_values_.data(), 0, self_mask_values_.size() * sizeof(float));
        input_upload_ms_ += engine::debug::elapsed_ms(upload_start, Clock::now());

        auto & graph = bank_graphs_[static_cast<size_t>(child_bank)];
        engine::core::set_backend_threads(execution_context_->backend(), execution_context_->config().threads);
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(execution_context_->backend(), graph.graph, nullptr, "Hviske decoder beam decode");
        ggml_backend_synchronize(execution_context_->backend());
        const double compute_ms = engine::debug::elapsed_ms(compute_start, Clock::now());
        graph_compute_ms_ += compute_ms;
        ++step_count_;
        debug::timing_log_scalar("hviske_asr.decoder.beam_decode.step.graph.compute_ms", compute_ms);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Hviske decoder beam decode graph compute failed");
        }

        const auto read_start = Clock::now();
        const auto & config = assets_->config.decoder;
        std::vector<float> logits(static_cast<size_t>(beam_count_ * config.vocab_size));
        ggml_backend_tensor_get(graph.logits, logits.data(), 0, logits.size() * sizeof(float));
        BatchOutput out;
        out.steps.reserve(active);
        for (size_t row = 0; row < active; ++row) {
            StepOutput step;
            step.logits.assign(
                logits.begin() + static_cast<std::ptrdiff_t>(row * static_cast<size_t>(config.vocab_size)),
                logits.begin() + static_cast<std::ptrdiff_t>((row + 1) * static_cast<size_t>(config.vocab_size)));
            out.steps.push_back(std::move(step));
        }
        output_read_ms_ += engine::debug::elapsed_ms(read_start, Clock::now());
        return out;
    }

    void log_loop_timing() const {
        debug::timing_log_scalar("hviske_asr.decoder.beam_decode.loop_steps", step_count_);
        debug::timing_log_scalar("hviske_asr.decoder.beam_decode.prefix_copy_ms", prefix_copy_ms_);
        debug::timing_log_scalar("hviske_asr.decoder.beam_decode.input_upload_ms", input_upload_ms_);
        debug::timing_log_scalar("hviske_asr.decoder.beam_decode.graph_compute_ms", graph_compute_ms_);
        debug::timing_log_scalar("hviske_asr.decoder.beam_decode.output_read_ms", output_read_ms_);
    }

private:
    struct BankGraph {
        ggml_cgraph * graph = nullptr;
        ggml_tensor * logits = nullptr;
    };

    void build_prefix_views(int64_t step_elems) {
        beam_key_prefix_views_.assign(static_cast<size_t>(beam_slots_), {});
        beam_value_prefix_views_.assign(static_cast<size_t>(beam_slots_), {});
        beam_key_step_views_.assign(static_cast<size_t>(beam_slots_), {});
        beam_value_step_views_.assign(static_cast<size_t>(beam_slots_), {});
        for (int64_t slot = 0; slot < beam_slots_; ++slot) {
            const int64_t bank = slot / beam_count_;
            const int64_t row = slot % beam_count_;
            auto & key_steps = beam_key_prefix_views_[static_cast<size_t>(slot)];
            auto & value_steps = beam_value_prefix_views_[static_cast<size_t>(slot)];
            key_steps.assign(static_cast<size_t>(cache_steps_ + 1), {});
            value_steps.assign(static_cast<size_t>(cache_steps_ + 1), {});
            auto & key_step_views = beam_key_step_views_[static_cast<size_t>(slot)];
            auto & value_step_views = beam_value_step_views_[static_cast<size_t>(slot)];
            key_step_views.assign(static_cast<size_t>(cache_steps_), {});
            value_step_views.assign(static_cast<size_t>(cache_steps_), {});
            for (int64_t steps = 1; steps <= cache_steps_; ++steps) {
                auto & key_layers = key_steps[static_cast<size_t>(steps)];
                auto & value_layers = value_steps[static_cast<size_t>(steps)];
                key_layers.reserve(bank_keys_[static_cast<size_t>(bank)].size());
                value_layers.reserve(bank_values_[static_cast<size_t>(bank)].size());
                const int64_t elems = steps * step_elems;
                const size_t byte_offset = static_cast<size_t>(row * cache_steps_ * step_elems) * sizeof(float);
                for (size_t layer = 0; layer < bank_keys_[static_cast<size_t>(bank)].size(); ++layer) {
                    key_layers.push_back(ggml_view_1d(ctx_.get(), bank_keys_[static_cast<size_t>(bank)][layer].tensor, elems, byte_offset));
                    value_layers.push_back(ggml_view_1d(ctx_.get(), bank_values_[static_cast<size_t>(bank)][layer].tensor, elems, byte_offset));
                }
            }
            for (int64_t step = 0; step < cache_steps_; ++step) {
                auto & key_layers = key_step_views[static_cast<size_t>(step)];
                auto & value_layers = value_step_views[static_cast<size_t>(step)];
                key_layers.reserve(bank_keys_[static_cast<size_t>(bank)].size());
                value_layers.reserve(bank_values_[static_cast<size_t>(bank)].size());
                const size_t byte_offset = static_cast<size_t>((row * cache_steps_ + step) * step_elems) * sizeof(float);
                for (size_t layer = 0; layer < bank_keys_[static_cast<size_t>(bank)].size(); ++layer) {
                    key_layers.push_back(ggml_view_1d(ctx_.get(), bank_keys_[static_cast<size_t>(bank)][layer].tensor, step_elems, byte_offset));
                    value_layers.push_back(ggml_view_1d(ctx_.get(), bank_values_[static_cast<size_t>(bank)][layer].tensor, step_elems, byte_offset));
                }
            }
        }
    }

    void copy_beam_prefix(int64_t parent_slot, int64_t child_slot, int64_t valid_steps, size_t layer) {
        if (valid_steps <= 0 || parent_slot == child_slot) {
            return;
        }
        ggml_backend_tensor_copy(
            beam_key_prefix_views_[static_cast<size_t>(parent_slot)][static_cast<size_t>(valid_steps)][layer],
            beam_key_prefix_views_[static_cast<size_t>(child_slot)][static_cast<size_t>(valid_steps)][layer]);
        ggml_backend_tensor_copy(
            beam_value_prefix_views_[static_cast<size_t>(parent_slot)][static_cast<size_t>(valid_steps)][layer],
            beam_value_prefix_views_[static_cast<size_t>(child_slot)][static_cast<size_t>(valid_steps)][layer]);
    }

    void build_bank_graph(engine::core::ModuleBuildContext & ctx, int64_t bank) {
        const auto & config = assets_->config.decoder;
        auto x = decoder_beam_embeddings(ctx, weights_->decoder, config, token_ids_, positions_, beam_count_);
        auto self_mask = engine::core::wrap_tensor(
            self_mask_,
            engine::core::TensorShape::from_dims({beam_count_, 1, 1, cache_steps_}),
            GGML_TYPE_F32);
        auto cross_mask = engine::core::wrap_tensor(
            cross_mask_,
            engine::core::TensorShape::from_dims({beam_count_, 1, 1, encoder_frames_}),
            GGML_TYPE_F32);
        auto cache_slots = engine::core::wrap_tensor(
            cache_slots_,
            engine::core::TensorShape::from_dims({beam_count_}),
            GGML_TYPE_I32);
        for (size_t layer = 0; layer < weights_->decoder.layers.size(); ++layer) {
            auto out = decoder_layer_with_beam_cache_tail(
                ctx,
                x,
                cross_keys_[layer],
                cross_values_[layer],
                weights_->decoder.layers[layer],
                config,
                bank_keys_[static_cast<size_t>(bank)][layer],
                bank_values_[static_cast<size_t>(bank)][layer],
                cache_slots,
                self_mask,
                cross_mask);
            x = out.output;
        }
        x = engine::modules::LayerNormModule({config.hidden_size, 1.0e-5f, true, true})
                .build(ctx, x, weights_->decoder.final_norm);
        auto logits = decoder_classifier_output(ctx, x, weights_->decoder, config);
        BankGraph graph;
        graph.logits = logits.tensor;
        ggml_set_output(graph.logits);
        graph.graph = ggml_new_graph_custom(ctx_.get(), 131072, false);
        ggml_build_forward_expand(graph.graph, graph.logits);
        bank_graphs_[static_cast<size_t>(bank)] = graph;
    }

    std::shared_ptr<const HviskeAssets> assets_;
    std::shared_ptr<const HviskeWeights> weights_;
    engine::core::ExecutionContext * execution_context_ = nullptr;
    int64_t encoder_frames_ = 0;
    int64_t cache_steps_ = 0;
    int64_t beam_count_ = 0;
    int64_t beam_slots_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_ids_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * cache_slots_ = nullptr;
    ggml_tensor * self_mask_ = nullptr;
    ggml_tensor * cross_mask_ = nullptr;
    std::array<std::vector<engine::core::TensorValue>, 2> bank_keys_;
    std::array<std::vector<engine::core::TensorValue>, 2> bank_values_;
    std::vector<engine::core::TensorValue> cross_keys_;
    std::vector<engine::core::TensorValue> cross_values_;
    std::vector<std::vector<std::vector<ggml_tensor *>>> beam_key_prefix_views_;
    std::vector<std::vector<std::vector<ggml_tensor *>>> beam_value_prefix_views_;
    std::vector<std::vector<std::vector<ggml_tensor *>>> beam_key_step_views_;
    std::vector<std::vector<std::vector<ggml_tensor *>>> beam_value_step_views_;
    std::array<BankGraph, 2> bank_graphs_;
    std::vector<int32_t> token_values_;
    std::vector<int32_t> position_values_;
    std::vector<int32_t> cache_slot_values_;
    std::vector<float> self_mask_values_;
    std::vector<float> cross_mask_values_;
    double prefix_copy_ms_ = 0.0;
    double input_upload_ms_ = 0.0;
    double graph_compute_ms_ = 0.0;
    double output_read_ms_ = 0.0;
    int64_t step_count_ = 0;
    ggml_backend_buffer_t buffer_ = nullptr;
};

HviskeDecoderRuntime::HviskeDecoderRuntime(
    std::shared_ptr<const HviskeAssets> assets,
    std::shared_ptr<const HviskeWeights> weights,
    engine::core::ExecutionContext & execution_context,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes)
    : assets_(std::move(assets)),
      weights_(std::move(weights)),
      execution_context_(&execution_context),
      prefill_graph_arena_bytes_(prefill_graph_arena_bytes),
      decode_graph_arena_bytes_(decode_graph_arena_bytes) {
    if (assets_ == nullptr || weights_ == nullptr) {
        throw std::runtime_error("Hviske decoder requires assets and weights");
    }
    if (prefill_graph_arena_bytes_ == 0 || decode_graph_arena_bytes_ == 0) {
        throw std::runtime_error("Hviske decoder graph arenas must be non-zero");
    }
}

HviskeDecoderRuntime::~HviskeDecoderRuntime() = default;
HviskeDecoderRuntime::HviskeDecoderRuntime(HviskeDecoderRuntime &&) noexcept = default;
HviskeDecoderRuntime & HviskeDecoderRuntime::operator=(HviskeDecoderRuntime &&) noexcept = default;

HviskeDecoderResult HviskeDecoderRuntime::generate(
    const std::vector<int32_t> & prompt_ids,
    const HviskeEncodedAudio & encoded,
    const HviskeDecodingOptions & options) {
    if (execution_context_ == nullptr) {
        throw std::runtime_error("Hviske decoder execution context is null");
    }
    const auto & config = assets_->config.decoder;
    if (prompt_ids.empty()) {
        throw std::runtime_error("Hviske decoder requires non-empty prompt");
    }
    const int64_t max_new_tokens = options.max_new_tokens;
    if (max_new_tokens <= 0) {
        throw std::runtime_error("Hviske decoder max_new_tokens must be positive");
    }
    if (options.num_beams <= 0) {
        throw std::runtime_error("Hviske decoder num_beams must be positive");
    }
    if (!std::isfinite(options.length_penalty) || options.length_penalty <= 0.0f) {
        throw std::runtime_error("Hviske decoder length_penalty must be finite and positive");
    }
    if (options.top_k < 0) {
        throw std::runtime_error("Hviske decoder top_k must be non-negative");
    }
    if (options.do_sample && options.num_beams != 1) {
        throw std::runtime_error("Hviske decoder supports do_sample only with num_beams=1");
    }
    const int64_t prompt_steps = static_cast<int64_t>(prompt_ids.size());
    if (prompt_steps + max_new_tokens > config.max_sequence_length) {
        throw std::runtime_error("Hviske decoder request exceeds max_sequence_length");
    }
    const bool rebuild_prefill = prefill_graph_ == nullptr ||
        !prefill_graph_->matches(*weights_, execution_context_->backend(), prompt_steps, encoded.frames);
    if (rebuild_prefill) {
        prefill_graph_ = std::make_unique<PrefillGraph>(
            assets_,
            weights_,
            *execution_context_,
            prompt_steps,
            encoded.frames,
            prefill_graph_arena_bytes_);
    } else {
        debug::timing_log_scalar("hviske_asr.decoder.prefill.graph_rebuild_ms", 0.0);
    }
    const auto generate_start = Clock::now();
    const auto prefill_start = Clock::now();
    const auto prefill = prefill_graph_->run(prompt_ids, encoded);
    debug::timing_log_scalar("hviske_asr.decoder.prefill.total_ms", engine::debug::elapsed_ms(prefill_start, Clock::now()));
    HviskeDecoderResult result;
    result.token_ids.reserve(static_cast<size_t>(max_new_tokens));
    const auto decode_loop_start = Clock::now();
    if (options.num_beams == 1) {
        const int64_t required_cache_steps = prompt_steps + max_new_tokens;
        const bool rebuild_decode = decode_graph_ == nullptr ||
            !decode_graph_->can_run(*weights_, execution_context_->backend(), encoded.frames, required_cache_steps);
        if (rebuild_decode) {
            decode_graph_ = std::make_unique<DecodeGraph>(
                assets_,
                weights_,
                *execution_context_,
                encoded.frames,
                required_cache_steps,
                decode_graph_arena_bytes_);
        } else {
            debug::timing_log_scalar("hviske_asr.decoder.decode.graph_rebuild_ms", 0.0);
        }
        decode_graph_->import_state(*prefill_graph_, encoded);
        std::mt19937 rng(options.seed);
        int32_t token = options.do_sample ? sample_token(prefill.logits, options, rng) : argmax_index(prefill.logits);
        for (int64_t step = 0; step < max_new_tokens; ++step) {
            if (is_eos(config, token)) {
                break;
            }
            result.token_ids.push_back(token);
            if (step + 1 >= max_new_tokens) {
                break;
            }
            const auto logits = decode_graph_->run_step(token);
            token = options.do_sample ? sample_token(logits, options, rng) : argmax_index(logits);
        }
    } else {
        const int64_t beam_count_i64 = options.num_beams;
        const int64_t required_cache_steps = prompt_steps + max_new_tokens;
        const bool rebuild_beam_decode = beam_decode_graph_ == nullptr ||
            !beam_decode_graph_->can_run(
                *weights_,
                execution_context_->backend(),
                encoded.frames,
                required_cache_steps,
                beam_count_i64);
        if (rebuild_beam_decode) {
            beam_decode_graph_ = std::make_unique<BeamDecodeGraph>(
                assets_,
                weights_,
                *execution_context_,
                encoded.frames,
                required_cache_steps,
                beam_count_i64,
                decode_graph_arena_bytes_);
        } else {
            debug::timing_log_scalar("hviske_asr.decoder.beam_decode.graph_rebuild_ms", 0.0);
        }
        beam_decode_graph_->initialize_from_prefill(*prefill_graph_, encoded, 1);
        BeamState initial;
        initial.logits = prefill.logits;
        initial.slot = 0;
        initial.valid_steps = prompt_steps;
        std::vector<BeamState> active;
        active.push_back(std::move(initial));
        std::vector<FinishedBeam> finished;
        const size_t beam_count = static_cast<size_t>(options.num_beams);
        const size_t candidates_per_beam = beam_count * 2;
        int active_bank = 0;
        double beam_score_ms = 0.0;
        double beam_select_ms = 0.0;
        double beam_batch_ms = 0.0;

        for (int64_t step = 0; step < max_new_tokens && !active.empty(); ++step) {
            const auto score_start = Clock::now();
            std::vector<BeamCandidate> candidates;
            candidates.reserve(active.size() * candidates_per_beam);
            for (size_t beam = 0; beam < active.size(); ++beam) {
                for (const auto & token_log_prob : top_log_probs(active[beam].logits, candidates_per_beam, config.output_log_probs)) {
                    candidates.push_back(BeamCandidate{
                        beam,
                        token_log_prob.token,
                        active[beam].score + static_cast<double>(token_log_prob.log_prob),
                        is_eos(config, token_log_prob.token),
                    });
                }
            }
            std::sort(candidates.begin(), candidates.end(), better_candidate);
            beam_score_ms += engine::debug::elapsed_ms(score_start, Clock::now());

            const auto select_start = Clock::now();
            std::vector<BeamState> next_active;
            next_active.reserve(beam_count);
            const int next_bank = 1 - active_bank;
            std::vector<int64_t> parent_slots;
            std::vector<int64_t> child_slots;
            std::vector<int32_t> next_tokens;
            parent_slots.reserve(beam_count);
            child_slots.reserve(beam_count);
            next_tokens.reserve(beam_count);
            for (size_t rank = 0; rank < candidates.size(); ++rank) {
                const auto & candidate = candidates[rank];
                const auto & parent = active[candidate.parent];
                const int64_t candidate_length = static_cast<int64_t>(parent.tokens.size()) + 1;
                if (candidate.eos) {
                    if (rank < beam_count) {
                        std::vector<int32_t> candidate_tokens = parent.tokens;
                        finished.push_back(FinishedBeam{std::move(candidate_tokens), candidate.score, candidate_length});
                    }
                    continue;
                }
                if (step + 1 >= max_new_tokens) {
                    std::vector<int32_t> candidate_tokens = parent.tokens;
                    candidate_tokens.push_back(candidate.token);
                    finished.push_back(FinishedBeam{std::move(candidate_tokens), candidate.score, candidate_length});
                    continue;
                }
                if (next_active.size() >= beam_count) {
                    continue;
                }
                BeamState next;
                next.tokens = parent.tokens;
                next.tokens.push_back(candidate.token);
                next.score = candidate.score;
                const int64_t child_slot = static_cast<int64_t>(next_bank * static_cast<int>(beam_count) + static_cast<int>(next_active.size()));
                parent_slots.push_back(parent.slot);
                child_slots.push_back(child_slot);
                next_tokens.push_back(candidate.token);
                next.slot = child_slot;
                next.valid_steps = parent.valid_steps + 1;
                next_active.push_back(std::move(next));
            }
            beam_select_ms += engine::debug::elapsed_ms(select_start, Clock::now());
            if (!next_active.empty()) {
                const int64_t parent_valid_steps = next_active.front().valid_steps - 1;
                const auto batch_start = Clock::now();
                const auto batch = beam_decode_graph_->run_batch_from_beams(
                    parent_slots,
                    child_slots,
                    parent_valid_steps,
                    next_tokens);
                beam_batch_ms += engine::debug::elapsed_ms(batch_start, Clock::now());
                if (batch.steps.size() != next_active.size()) {
                    throw std::runtime_error("Hviske decoder beam batch output size mismatch");
                }
                for (size_t beam = 0; beam < next_active.size(); ++beam) {
                    next_active[beam].logits = batch.steps[beam].logits;
                }
            }
            active = std::move(next_active);
            active_bank = next_bank;

            if (finished.size() > beam_count) {
                std::sort(finished.begin(), finished.end(), [&](const FinishedBeam & lhs, const FinishedBeam & rhs) {
                    return better_finished_beam(lhs, rhs, options.length_penalty);
                });
                finished.resize(beam_count);
            }
            if (finished.size() >= beam_count && !active.empty()) {
                double worst_finished_score = std::numeric_limits<double>::infinity();
                for (const auto & beam : finished) {
                    worst_finished_score = std::min(
                        worst_finished_score,
                        normalized_beam_score(beam.score, beam.length, options.length_penalty));
                }
                double best_active_score = -std::numeric_limits<double>::infinity();
                int64_t active_length = 1;
                for (const auto & beam : active) {
                    best_active_score = std::max(best_active_score, beam.score);
                    active_length = std::max(active_length, static_cast<int64_t>(beam.tokens.size()));
                }
                const double best_attainable_score =
                    normalized_beam_score(best_active_score, active_length, options.length_penalty);
                if (worst_finished_score >= best_attainable_score) {
                    active.clear();
                    break;
                }
            }

            if (finished.size() >= beam_count && active.empty()) {
                break;
            }
        }

        if (finished.empty()) {
            for (auto & beam : active) {
                finished.push_back(FinishedBeam{
                    std::move(beam.tokens),
                    beam.score,
                    static_cast<int64_t>(beam.tokens.size()),
                });
            }
        }
        if (!finished.empty()) {
            auto best = std::max_element(finished.begin(), finished.end(), [&](const FinishedBeam & lhs, const FinishedBeam & rhs) {
                const double lhs_score = normalized_beam_score(lhs.score, lhs.length, options.length_penalty);
                const double rhs_score = normalized_beam_score(rhs.score, rhs.length, options.length_penalty);
                if (lhs_score != rhs_score) {
                    return lhs_score < rhs_score;
                }
                return lhs.score < rhs.score;
            });
            result.token_ids = std::move(best->tokens);
        }
        debug::timing_log_scalar("hviske_asr.decoder.beam_score_ms", beam_score_ms);
        debug::timing_log_scalar("hviske_asr.decoder.beam_select_ms", beam_select_ms);
        debug::timing_log_scalar("hviske_asr.decoder.beam_batch_ms", beam_batch_ms);
        beam_decode_graph_->log_loop_timing();
    }
    debug::timing_log_scalar("hviske_asr.decoder.decode_loop_ms", engine::debug::elapsed_ms(decode_loop_start, Clock::now()));
    debug::timing_log_scalar("hviske_asr.decoder.generate_ms", engine::debug::elapsed_ms(generate_start, Clock::now()));
    debug::trace_log_scalar("hviske_asr.decoder.generated_tokens", result.token_ids.size());
    debug::trace_log_scalar("hviske_asr.decoder.num_beams", options.num_beams);
    debug::trace_log_scalar("hviske_asr.decoder.do_sample", options.do_sample ? 1 : 0);
    return result;
}

}  // namespace engine::models::hviske_asr
