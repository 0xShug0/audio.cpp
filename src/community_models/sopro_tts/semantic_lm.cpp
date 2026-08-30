#include "engine/community_models/sopro_tts/semantic_lm.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/transformers/qwen_causal_decode_runtime.h"
#include "engine/framework/modules/transformers/qwen_causal_decoder.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::sopro_tts {
namespace {

namespace binding = engine::modules::binding;

constexpr float kRmsNormEps = 1.0e-6F;   // sopro.nn.layers.RMSNorm default
constexpr float kMaskedLogit = -1.0e9F;  // sampling.sample_next_token

// StylePrefixEncoder runs on the host: eight learned queries cross-attend to at
// most a few hundred reference frames, which is under 0.1 GFLOP.
struct SoproStylePrefixWeights {
    std::vector<float> queries;     // [tokens, dim]
    std::vector<float> kv_norm;     // [dim]
    std::vector<float> q_proj;      // [dim, dim]
    std::vector<float> k_proj;
    std::vector<float> v_proj;
    std::vector<float> out_proj;
    std::vector<float> out_norm;    // [dim]
};

struct SoproSemanticLMHostWeights {
    std::vector<float> text_embedding;      // [text_vocab, dim]
    std::vector<float> semantic_embedding;  // [semantic_vocab + 2, dim], fused
    SoproStylePrefixWeights style_prefix;
};

struct SoproSemanticLMBackendWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::core::TensorValue token_embedding;
    engine::modules::QwenDecoderStackWeights stack;
    engine::modules::NormWeights final_norm;
    engine::modules::LinearWeights token_head;
};

void rms_norm(std::vector<float> & values, const std::vector<float> & weight) {
    const size_t dim = weight.size();
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum += static_cast<double>(values[i]) * static_cast<double>(values[i]);
    }
    const double inv = 1.0 / std::sqrt(sum / static_cast<double>(dim) + kRmsNormEps);
    for (size_t i = 0; i < dim; ++i) {
        values[i] = static_cast<float>(values[i] * inv * weight[i]);
    }
}

// out[row] = weight @ in[row]; weight is [out_dim, in_dim] row-major.
void matmul_rows(
    const std::vector<float> & weight,
    const float * input,
    int64_t rows,
    int64_t in_dim,
    int64_t out_dim,
    float * output) {
#ifdef _OPENMP
#pragma omp parallel for if (rows > 8)
#endif
    for (int64_t row = 0; row < rows; ++row) {
        const float * source = input + row * in_dim;
        float * target = output + row * out_dim;
        for (int64_t o = 0; o < out_dim; ++o) {
            const float * w = weight.data() + static_cast<size_t>(o * in_dim);
            double sum = 0.0;
            for (int64_t i = 0; i < in_dim; ++i) {
                sum += static_cast<double>(w[i]) * static_cast<double>(source[i]);
            }
            target[o] = static_cast<float>(sum);
        }
    }
}

// Fuse LayerScale into the projection that feeds the residual: the branch is
// scale * W @ y with no bias, so scaling the rows of W is exact.
std::vector<float> scale_rows(std::vector<float> weight, const std::vector<float> & scale, int64_t in_dim) {
    for (size_t row = 0; row < scale.size(); ++row) {
        float * values = weight.data() + static_cast<size_t>(static_cast<int64_t>(row) * in_dim);
        const float factor = scale[row];
        for (int64_t i = 0; i < in_dim; ++i) {
            values[i] *= factor;
        }
    }
    return weight;
}

engine::modules::QwenCausalDecoderConfig make_decoder_config(
    const SoproModelConfig & config,
    engine::core::BackendType backend_type) {
    engine::modules::QwenCausalDecoderConfig out;
    out.stack.hidden_size = config.ar_model_dim;
    out.stack.num_attention_heads = config.ar_heads;
    out.stack.num_key_value_heads = config.ar_kv_heads;
    out.stack.head_dim = config.ar_head_dim();
    out.stack.intermediate_size = config.ar_ffn_dim();
    out.stack.layers = config.ar_blocks;
    out.stack.rms_norm_eps = kRmsNormEps;
    out.stack.rope_theta = 10000.0F;
    // sopro.nn.layers.rotate_half splits the head in halves, which is ggml's
    // NEOX rotary layout.
    out.stack.rope_type = GGML_ROPE_TYPE_NEOX;
    out.stack.attention_precision = GGML_PREC_DEFAULT;
    out.stack.projection_precision = GGML_PREC_DEFAULT;
    out.stack.use_qk_norm = config.ar_qk_rms_norm;
    out.stack.qkv_layout = engine::modules::QwenDecoderQKVLayout::PackedQKV;
    out.stack.runtime.mlp.mode = engine::modules::QwenDecoderMLPMode::PackedGateUp;
    out.stack.runtime.attention.prefill_mode = engine::modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.attention.static_mode = engine::modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.stack.runtime.static_cache.update_mode = engine::modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    out.logits_size = config.semantic_vocab_size + 2;
    out.logits_mode = engine::modules::QwenCausalDecoderLogitsMode::LastStep;
    out.use_lm_head_bias = true;  // SemanticLM.token_head is a biased Linear
    out.lm_head_precision = GGML_PREC_DEFAULT;
    (void) backend_type;
    return out;
}

engine::modules::QwenDecoderLayerWeights load_layer(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const SoproModelConfig & config,
    engine::assets::TensorStorageType storage_type,
    int64_t layer) {
    const std::string prefix = "ar_prior.temporal.layers." + std::to_string(layer);
    const int64_t dim = config.ar_model_dim;
    const int64_t head_dim = config.ar_head_dim();
    const int64_t q_out = config.ar_heads * head_dim;
    const int64_t kv_out = config.ar_kv_heads * head_dim;
    const int64_t ffn = config.ar_ffn_dim();

    engine::modules::QwenDecoderLayerWeights out;
    out.input_norm = binding::norm_weight_from_source(store, source, prefix + ".attn_norm", dim);
    auto qkv = source.require_f32(prefix + ".attn.q_proj.weight", {q_out, dim});
    const auto k_rows = source.require_f32(prefix + ".attn.k_proj.weight", {kv_out, dim});
    const auto v_rows = source.require_f32(prefix + ".attn.v_proj.weight", {kv_out, dim});
    qkv.insert(qkv.end(), k_rows.begin(), k_rows.end());
    qkv.insert(qkv.end(), v_rows.begin(), v_rows.end());
    out.self_attention.qkv_weight = store.make_from_f32(
        engine::core::TensorShape::from_dims({q_out + kv_out * 2, dim}), storage_type, std::move(qkv));
    out.self_attention.out_weight = store.make_from_f32(
        engine::core::TensorShape::from_dims({dim, q_out}),
        storage_type,
        scale_rows(
            source.require_f32(prefix + ".attn.out_proj.weight", {dim, q_out}),
            source.require_f32(prefix + ".attn_scale.scale", {dim}),
            q_out));
    if (config.ar_qk_rms_norm) {
        out.q_norm = binding::norm_weight_from_source(store, source, prefix + ".attn.q_norm", head_dim);
        out.k_norm = binding::norm_weight_from_source(store, source, prefix + ".attn.k_norm", head_dim);
    }
    out.post_norm = binding::norm_weight_from_source(store, source, prefix + ".ffn_norm", dim);
    auto gate_up = source.require_f32(prefix + ".ffn.gate_proj.weight", {ffn, dim});
    const auto up_rows = source.require_f32(prefix + ".ffn.up_proj.weight", {ffn, dim});
    gate_up.insert(gate_up.end(), up_rows.begin(), up_rows.end());
    out.mlp.gate_up_proj = engine::modules::LinearWeights{
        store.make_from_f32(
            engine::core::TensorShape::from_dims({ffn * 2, dim}), storage_type, std::move(gate_up)),
        std::nullopt};
    out.mlp.down_proj = engine::modules::LinearWeights{
        store.make_from_f32(
            engine::core::TensorShape::from_dims({dim, ffn}),
            storage_type,
            scale_rows(
                source.require_f32(prefix + ".ffn.down_proj.weight", {dim, ffn}),
                source.require_f32(prefix + ".ffn_scale.scale", {dim}),
                ffn)),
        std::nullopt};
    return out;
}

SoproSemanticLMHostWeights load_host_weights(
    const engine::assets::TensorSource & source,
    const SoproModelConfig & config) {
    SoproSemanticLMHostWeights out;
    const int64_t dim = config.ar_model_dim;
    const int64_t latent = config.latent_dim;
    const int64_t semantic_rows = config.semantic_vocab_size + 2;

    out.text_embedding = source.require_f32("text_tok_emb.weight", {config.text_vocab_size, dim});

    // embed_semantic = sem_in_proj(semantic_tok_emb(id)); fold the projection
    // into the table so every later lookup is a plain row gather.
    const auto table = source.require_f32("semantic_tok_emb.weight", {semantic_rows, latent});
    const auto projection = source.require_f32("sem_in_proj.weight", {dim, latent});
    const auto bias = source.require_f32("sem_in_proj.bias", {dim});
    out.semantic_embedding.assign(static_cast<size_t>(semantic_rows * dim), 0.0F);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t row = 0; row < semantic_rows; ++row) {
        const float * source_row = table.data() + static_cast<size_t>(row * latent);
        float * target = out.semantic_embedding.data() + static_cast<size_t>(row * dim);
        for (int64_t o = 0; o < dim; ++o) {
            const float * w = projection.data() + static_cast<size_t>(o * latent);
            double sum = bias[static_cast<size_t>(o)];
            for (int64_t i = 0; i < latent; ++i) {
                sum += static_cast<double>(w[i]) * static_cast<double>(source_row[i]);
            }
            target[o] = static_cast<float>(sum);
        }
    }

    auto & style = out.style_prefix;
    style.queries = source.require_f32("style_prefix.queries", {config.style_prefix_tokens, dim});
    style.kv_norm = source.require_f32("style_prefix.kv_norm.weight", {dim});
    style.q_proj = source.require_f32("style_prefix.q_proj.weight", {dim, dim});
    style.k_proj = source.require_f32("style_prefix.k_proj.weight", {dim, dim});
    style.v_proj = source.require_f32("style_prefix.v_proj.weight", {dim, dim});
    style.out_proj = source.require_f32("style_prefix.out_proj.weight", {dim, dim});
    style.out_norm = source.require_f32("style_prefix.out_norm.weight", {dim});
    return out;
}

// StylePrefixEncoder.forward for a single sequence.
std::vector<float> build_style_prefix(
    const SoproStylePrefixWeights & weights,
    const std::vector<float> & reference,  // [steps, dim]
    int64_t steps,
    int64_t tokens,
    int64_t dim,
    int64_t heads) {
    const int64_t head_dim = dim / heads;
    std::vector<float> out(static_cast<size_t>(tokens * dim), 0.0F);
    if (steps == 0) {
        // The reference path returns out_norm(queries) when there is nothing
        // to attend to.
        for (int64_t t = 0; t < tokens; ++t) {
            std::vector<float> row(
                weights.queries.begin() + static_cast<ptrdiff_t>(t * dim),
                weights.queries.begin() + static_cast<ptrdiff_t>((t + 1) * dim));
            rms_norm(row, weights.out_norm);
            std::copy(row.begin(), row.end(), out.begin() + static_cast<ptrdiff_t>(t * dim));
        }
        return out;
    }

    std::vector<float> kv(static_cast<size_t>(steps * dim), 0.0F);
    for (int64_t s = 0; s < steps; ++s) {
        std::vector<float> row(
            reference.begin() + static_cast<ptrdiff_t>(s * dim),
            reference.begin() + static_cast<ptrdiff_t>((s + 1) * dim));
        rms_norm(row, weights.kv_norm);
        std::copy(row.begin(), row.end(), kv.begin() + static_cast<ptrdiff_t>(s * dim));
    }

    std::vector<float> q(static_cast<size_t>(tokens * dim), 0.0F);
    std::vector<float> k(static_cast<size_t>(steps * dim), 0.0F);
    std::vector<float> v(static_cast<size_t>(steps * dim), 0.0F);
    matmul_rows(weights.q_proj, weights.queries.data(), tokens, dim, dim, q.data());
    matmul_rows(weights.k_proj, kv.data(), steps, dim, dim, k.data());
    matmul_rows(weights.v_proj, kv.data(), steps, dim, dim, v.data());

    const auto scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
    std::vector<float> context(static_cast<size_t>(tokens * dim), 0.0F);
    std::vector<float> scores(static_cast<size_t>(steps), 0.0F);
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t h = 0; h < heads; ++h) {
            const float * q_head = q.data() + static_cast<size_t>(t * dim + h * head_dim);
            float max_score = -std::numeric_limits<float>::infinity();
            for (int64_t s = 0; s < steps; ++s) {
                const float * k_head = k.data() + static_cast<size_t>(s * dim + h * head_dim);
                double sum = 0.0;
                for (int64_t d = 0; d < head_dim; ++d) {
                    sum += static_cast<double>(q_head[d]) * static_cast<double>(k_head[d]);
                }
                const auto value = static_cast<float>(sum) * scale;
                scores[static_cast<size_t>(s)] = value;
                max_score = std::max(max_score, value);
            }
            double total = 0.0;
            for (auto & score : scores) {
                score = std::exp(score - max_score);
                total += score;
            }
            float * target = context.data() + static_cast<size_t>(t * dim + h * head_dim);
            for (int64_t s = 0; s < steps; ++s) {
                const float weight = static_cast<float>(scores[static_cast<size_t>(s)] / total);
                const float * v_head = v.data() + static_cast<size_t>(s * dim + h * head_dim);
                for (int64_t d = 0; d < head_dim; ++d) {
                    target[d] += weight * v_head[d];
                }
            }
        }
    }

    std::vector<float> projected(static_cast<size_t>(tokens * dim), 0.0F);
    matmul_rows(weights.out_proj, context.data(), tokens, dim, dim, projected.data());
    for (int64_t t = 0; t < tokens; ++t) {
        std::vector<float> row(static_cast<size_t>(dim), 0.0F);
        for (int64_t d = 0; d < dim; ++d) {
            row[static_cast<size_t>(d)] =
                weights.queries[static_cast<size_t>(t * dim + d)] +
                projected[static_cast<size_t>(t * dim + d)];
        }
        rms_norm(row, weights.out_norm);
        std::copy(row.begin(), row.end(), out.begin() + static_cast<ptrdiff_t>(t * dim));
    }
    return out;
}

}  // namespace

int32_t sample_next_token(
    std::vector<float> & logits,
    float temperature,
    float top_p,
    int64_t top_k,
    int32_t bos_id,
    int32_t eos_id,
    bool allow_eos,
    std::mt19937_64 & rng) {
    const auto vocab = static_cast<int64_t>(logits.size());
    if (vocab <= 0) {
        throw std::runtime_error("Sopro semantic LM produced empty logits");
    }
    logits[static_cast<size_t>(bos_id)] = kMaskedLogit;
    if (!allow_eos) {
        logits[static_cast<size_t>(eos_id)] = kMaskedLogit;
    }
    if (temperature <= 0.0F) {
        return static_cast<int32_t>(
            std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }
    const float inv_temperature = 1.0F / std::max(1.0e-5F, temperature);
    float max_logit = -std::numeric_limits<float>::infinity();
    for (auto & value : logits) {
        value *= inv_temperature;
        max_logit = std::max(max_logit, value);
    }
    std::vector<float> probs(logits.size(), 0.0F);
    double total = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        total += probs[i];
    }
    for (auto & value : probs) {
        value = static_cast<float>(value / total);
    }

    if (top_k > 0 && top_k < vocab) {
        std::vector<float> sorted(probs);
        std::nth_element(
            sorted.begin(), sorted.begin() + static_cast<ptrdiff_t>(top_k - 1), sorted.end(),
            std::greater<float>());
        const float kth = sorted[static_cast<size_t>(top_k - 1)];
        double sum = 0.0;
        for (auto & value : probs) {
            if (value < kth) {
                value = 0.0F;
            }
            sum += value;
        }
        const auto inv = static_cast<float>(1.0 / std::max(sum, 1.0e-8));
        for (auto & value : probs) {
            value *= inv;
        }
    }

    if (top_p < 1.0F) {
        const float threshold = std::min(std::max(top_p, 0.0F), 1.0F);
        std::vector<int32_t> order(static_cast<size_t>(vocab));
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            return probs[static_cast<size_t>(a)] > probs[static_cast<size_t>(b)];
        });
        // remove[i] = cdf[i - 1] > p: the first token above the threshold is
        // kept, everything past it is dropped.
        std::vector<float> nucleus(probs.size(), 0.0F);
        double cumulative = 0.0;
        double sum = 0.0;
        for (size_t rank = 0; rank < order.size(); ++rank) {
            const auto index = static_cast<size_t>(order[rank]);
            if (rank == 0 || cumulative <= threshold) {
                nucleus[index] = probs[index];
                sum += probs[index];
            }
            cumulative += probs[index];
        }
        const auto inv = static_cast<float>(1.0 / std::max(sum, 1.0e-8));
        for (size_t i = 0; i < probs.size(); ++i) {
            probs[i] = nucleus[i] * inv;
        }
    }

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double draw = uniform(rng);
    double cumulative = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumulative += probs[i];
        if (draw < cumulative) {
            return static_cast<int32_t>(i);
        }
    }
    for (int64_t i = vocab - 1; i >= 0; --i) {
        if (probs[static_cast<size_t>(i)] > 0.0F) {
            return static_cast<int32_t>(i);
        }
    }
    throw std::runtime_error("Sopro semantic LM sampling found no candidate token");
}

class SoproSemanticLMRuntime::Impl {
public:
    Impl(
        const SoproTTSAssets & assets,
        engine::core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType weight_storage_type)
        : config_(assets.config.model) {
        const auto & source = *assets.model_weights;
        host_ = load_host_weights(source, config_);

        backend_.store = std::make_shared<engine::core::BackendWeightStore>(
            execution.backend(), execution.backend_type(), "sopro_tts.semantic_lm.weights",
            weight_context_bytes);
        auto & store = *backend_.store;
        backend_.token_embedding = store.load_tensor(
            source, "text_tok_emb.weight", weight_storage_type,
            {config_.text_vocab_size, config_.ar_model_dim});
        backend_.stack.layers.reserve(static_cast<size_t>(config_.ar_blocks));
        for (int64_t layer = 0; layer < config_.ar_blocks; ++layer) {
            backend_.stack.layers.push_back(
                load_layer(store, source, config_, weight_storage_type, layer));
        }
        backend_.final_norm = binding::norm_weight_from_source(
            store, source, "ar_prior.out_norm", config_.ar_model_dim);
        backend_.token_head = binding::linear_from_source(
            store, source, "ar_prior.token_head", weight_storage_type,
            config_.semantic_vocab_size + 2, config_.ar_model_dim, true);
        store.upload();

        engine::modules::QwenCausalDecodeRuntimeConfig runtime_config;
        runtime_config.trace_name = "sopro_tts.semantic_lm";
        runtime_config.decoder = make_decoder_config(config_, execution.backend_type());
        runtime_config.prefill_graph_arena_bytes = prefill_graph_arena_bytes;
        runtime_config.decode_graph_arena_bytes = decode_graph_arena_bytes;
        runtime_config.output_mode = engine::modules::QwenCausalDecodeOutputMode::Logits;
        runtime_config.return_hidden = false;

        engine::modules::QwenCausalDecodeRuntimeWeights runtime_weights;
        runtime_weights.token_embedding = backend_.token_embedding;
        runtime_weights.stack = backend_.stack;
        runtime_weights.final_norm = backend_.final_norm;
        runtime_weights.lm_head = backend_.token_head;
        decoder_ = std::make_unique<engine::modules::QwenCausalDecodeRuntime>(
            execution, std::move(runtime_config), std::move(runtime_weights));
    }

    std::vector<int32_t> generate(
        const std::vector<int32_t> & text_ids,
        const std::vector<int32_t> & style_tokens,
        const std::vector<int32_t> & prompt_tokens,
        const SoproSemanticLMOptions & options,
        std::mt19937_64 & rng) {
        const int64_t dim = config_.ar_model_dim;
        const auto text_steps = std::min<int64_t>(
            static_cast<int64_t>(text_ids.size()), config_.max_text_len);
        const int64_t style_steps = config_.style_prefix_tokens;
        const auto prompt_steps = static_cast<int64_t>(prompt_tokens.size());
        const int64_t steps = style_steps + text_steps + prompt_steps + 1;

        std::vector<float> prefix(static_cast<size_t>(steps * dim), 0.0F);
        {
            std::vector<float> style_reference(
                static_cast<size_t>(static_cast<int64_t>(style_tokens.size()) * dim), 0.0F);
            for (size_t i = 0; i < style_tokens.size(); ++i) {
                copy_semantic_row(style_tokens[i], style_reference.data() + i * static_cast<size_t>(dim));
            }
            const auto style = build_style_prefix(
                host_.style_prefix, style_reference, static_cast<int64_t>(style_tokens.size()),
                style_steps, dim, config_.ar_heads);
            std::copy(style.begin(), style.end(), prefix.begin());
        }
        int64_t offset = style_steps;
        for (int64_t i = 0; i < text_steps; ++i) {
            const int32_t id = text_ids[static_cast<size_t>(i)];
            if (id < 0 || id >= config_.text_vocab_size) {
                throw std::runtime_error("Sopro text token id is out of range");
            }
            const float * row = host_.text_embedding.data() + static_cast<size_t>(id * dim);
            std::copy(row, row + dim, prefix.begin() + static_cast<ptrdiff_t>((offset + i) * dim));
        }
        offset += text_steps;
        for (int64_t i = 0; i < prompt_steps; ++i) {
            copy_semantic_row(
                prompt_tokens[static_cast<size_t>(i)],
                prefix.data() + static_cast<size_t>((offset + i) * dim));
        }
        offset += prompt_steps;
        copy_semantic_row(
            static_cast<int32_t>(config_.semantic_bos_id()),
            prefix.data() + static_cast<size_t>(offset * dim));

        const int64_t max_steps = std::max<int64_t>(1, options.max_steps);
        auto prefill = decoder_->prefill_embeddings(prefix, steps);
        decoder_->start_decode_embeddings(prefill.state, steps + max_steps);

        const auto bos_id = static_cast<int32_t>(config_.semantic_bos_id());
        const auto eos_id = static_cast<int32_t>(config_.semantic_eos_id());
        const int64_t min_steps = std::max<int64_t>(1, options.min_steps);
        std::vector<int32_t> tokens;
        tokens.reserve(static_cast<size_t>(max_steps));
        std::vector<float> logits = std::move(prefill.logits);
        std::vector<float> embedding(static_cast<size_t>(dim), 0.0F);
        for (int64_t step = 0; step < max_steps; ++step) {
            const bool allow_eos = (step + 1) >= min_steps;
            int32_t token = sample_next_token(
                logits, options.temperature, options.top_p, options.top_k,
                bos_id, eos_id, allow_eos, rng);
            if (allow_eos && token == eos_id) {
                break;
            }
            token = std::min<int32_t>(
                std::max<int32_t>(token, 0), static_cast<int32_t>(config_.semantic_vocab_size - 1));
            tokens.push_back(token);
            if (step + 1 >= max_steps) {
                break;
            }
            copy_semantic_row(token, embedding.data());
            logits = std::move(decoder_->decode_embedding(embedding).logits);
        }
        return tokens;
    }

    void release_runtime_graphs() {
        decoder_->release_runtime_graphs();
    }

private:
    void copy_semantic_row(int32_t id, float * target) const {
        if (id < 0 || id >= config_.semantic_vocab_size + 2) {
            throw std::runtime_error("Sopro semantic token id is out of range");
        }
        const float * row =
            host_.semantic_embedding.data() + static_cast<size_t>(id * config_.ar_model_dim);
        std::copy(row, row + config_.ar_model_dim, target);
    }

    const SoproModelConfig & config_;
    SoproSemanticLMHostWeights host_;
    SoproSemanticLMBackendWeights backend_;
    std::unique_ptr<engine::modules::QwenCausalDecodeRuntime> decoder_;
};

SoproSemanticLMRuntime::SoproSemanticLMRuntime(
    const SoproTTSAssets & assets,
    engine::core::ExecutionContext & execution_context,
    size_t prefill_graph_arena_bytes,
    size_t decode_graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
          assets, execution_context, prefill_graph_arena_bytes, decode_graph_arena_bytes,
          weight_context_bytes, weight_storage_type)) {}

SoproSemanticLMRuntime::~SoproSemanticLMRuntime() = default;

std::vector<int32_t> SoproSemanticLMRuntime::generate(
    const std::vector<int32_t> & text_ids,
    const std::vector<int32_t> & style_tokens,
    const std::vector<int32_t> & prompt_tokens,
    const SoproSemanticLMOptions & options,
    std::mt19937_64 & rng) const {
    return impl_->generate(text_ids, style_tokens, prompt_tokens, options, rng);
}

void SoproSemanticLMRuntime::release_runtime_graphs() {
    impl_->release_runtime_graphs();
}

}  // namespace engine::community_models::sopro_tts
