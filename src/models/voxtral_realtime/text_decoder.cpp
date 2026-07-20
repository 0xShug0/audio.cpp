#include "engine/models/voxtral_realtime/text_decoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/grouped_query_attention.h"
#include "engine/framework/modules/attention/qwen_causal_decoder.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/bounded_static_kv_decode.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/framework/sampling/torch_random.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::voxtral_realtime {
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

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept {
        if (alloc != nullptr) {
            ggml_gallocr_free(alloc);
        }
    }
};

struct TextLayerWeights {
    core::TensorValue input_norm;
    core::TensorValue q_weight;
    core::TensorValue k_weight;
    core::TensorValue v_weight;
    core::TensorValue o_weight;
    core::TensorValue post_norm;
    core::TensorValue ada1_weight;
    core::TensorValue ada2_weight;
    core::TensorValue gate_weight;
    core::TensorValue up_weight;
    core::TensorValue down_weight;
};

struct TextWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    std::vector<TextLayerWeights> layers;
    core::TensorValue norm;
    core::TensorValue lm_head;
};

assets::TensorStorageType normalize_weight_storage(assets::TensorStorageType storage_type) {
    switch (storage_type) {
        case assets::TensorStorageType::Native:
        case assets::TensorStorageType::F32:
        case assets::TensorStorageType::F16:
        case assets::TensorStorageType::BF16:
        case assets::TensorStorageType::Q8_0:
            return storage_type;
        default:
            throw std::runtime_error("VoxTral text_decoder_weight_type supports only native, f32, f16, bf16, and q8_0");
    }
}

std::shared_ptr<TextWeights> load_weights(
    const VoxtralRealtimeAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config.text;
    const auto & source = *assets.model_weights;
    const auto resolved = normalize_weight_storage(storage_type);
    auto weights = std::make_shared<TextWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "voxtral_realtime.text_decoder.weights",
        weight_context_bytes);
    auto & store = *weights->store;
    weights->token_embedding = store.load_tensor(
        source,
        "language_model.model.embed_tokens.weight",
        resolved,
        {config.vocab_size, config.hidden_size});
    weights->layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string prefix = "language_model.model.layers." + std::to_string(layer);
        TextLayerWeights w;
        w.input_norm = store.load_f32_tensor(source, prefix + ".input_layernorm.weight", {config.hidden_size});
        w.q_weight = store.load_tensor(source, prefix + ".self_attn.q_proj.weight", resolved,
                                       {config.num_attention_heads * config.head_dim, config.hidden_size});
        w.k_weight = store.load_tensor(source, prefix + ".self_attn.k_proj.weight", resolved,
                                       {config.num_key_value_heads * config.head_dim, config.hidden_size});
        w.v_weight = store.load_tensor(source, prefix + ".self_attn.v_proj.weight", resolved,
                                       {config.num_key_value_heads * config.head_dim, config.hidden_size});
        w.o_weight = store.load_tensor(source, prefix + ".self_attn.o_proj.weight", resolved,
                                       {config.hidden_size, config.num_attention_heads * config.head_dim});
        w.post_norm = store.load_f32_tensor(source, prefix + ".post_attention_layernorm.weight", {config.hidden_size});
        w.ada1_weight = store.load_tensor(source, prefix + ".ada_rms_norm.linear1.weight", resolved,
                                          {32, config.hidden_size});
        w.ada2_weight = store.load_tensor(source, prefix + ".ada_rms_norm.linear2.weight", resolved,
                                          {config.hidden_size, 32});
        w.gate_weight = store.load_tensor(source, prefix + ".mlp.gate_proj.weight", resolved,
                                          {config.intermediate_size, config.hidden_size});
        w.up_weight = store.load_tensor(source, prefix + ".mlp.up_proj.weight", resolved,
                                        {config.intermediate_size, config.hidden_size});
        w.down_weight = store.load_tensor(source, prefix + ".mlp.down_proj.weight", resolved,
                                          {config.hidden_size, config.intermediate_size});
        weights->layers.push_back(std::move(w));
    }
    weights->norm = store.load_f32_tensor(source, "language_model.model.norm.weight", {config.hidden_size});
    weights->lm_head = weights->token_embedding;
    const auto upload_start = Clock::now();
    store.upload();
    engine::debug::timing_log_scalar(
        "voxtral_realtime.text_decoder.weights.upload_ms",
        engine::debug::elapsed_ms(upload_start));
    return weights;
}

std::vector<ggml_fp16_t> make_text_mask(int64_t steps, int64_t window) {
    std::vector<ggml_fp16_t> values(static_cast<size_t>(steps * steps), ggml_fp32_to_fp16(-INFINITY));
    for (int64_t row = 0; row < steps; ++row) {
        const int64_t begin = std::max<int64_t>(0, row - window + 1);
        for (int64_t col = begin; col <= row; ++col) {
            values[static_cast<size_t>(row * steps + col)] = ggml_fp32_to_fp16(0.0F);
        }
    }
    return values;
}

std::vector<float> time_embedding(int64_t num_delay_tokens, int64_t hidden_size) {
    if (hidden_size % 2 != 0) {
        throw std::runtime_error("VoxTral time embedding requires an even hidden size");
    }
    std::vector<float> values(static_cast<size_t>(hidden_size));
    const double half = static_cast<double>(hidden_size / 2);
    for (int64_t i = 0; i < hidden_size / 2; ++i) {
        const double inv = std::exp(-std::log(10000.0) * static_cast<double>(i) / half);
        const double value = static_cast<double>(num_delay_tokens) * inv;
        values[static_cast<size_t>(i)] = static_cast<float>(std::cos(value));
        values[static_cast<size_t>(i + hidden_size / 2)] = static_cast<float>(std::sin(value));
    }
    return values;
}

int32_t argmax_index(const float * values, size_t count) {
    if (count == 0) {
        throw std::runtime_error("VoxTral text decoder cannot select from empty logits");
    }
    return static_cast<int32_t>(std::distance(values, std::max_element(values, values + count)));
}

struct LogitCandidate {
    int32_t token = 0;
    float score = 0.0F;
    double weight = 0.0;
};

class TokenSampler {
public:
    int32_t select(
        const std::vector<float> & logits,
        const VoxtralRealtimeGenerationOptions & options,
        uint64_t seed,
        uint64_t & call_index) {
        if (logits.empty()) {
            throw std::runtime_error("VoxTral text decoder cannot select from empty logits");
        }
        if (!options.do_sample || options.top_k == 1) {
            return argmax_index(logits.data(), logits.size());
        }
        build_candidates(logits, options);
        std::sort(candidates_.begin(), candidates_.end(), by_token_asc);
        double total = 0.0;
        for (const auto & candidate : candidates_) {
            total += candidate.weight;
        }
        if (!(total > 0.0) || !std::isfinite(total)) {
            throw std::runtime_error("VoxTral text decoder sampling kept probability mass is invalid");
        }
        const float uniform = engine::sampling::generate_torch_cuda_uniform(1, seed, call_index)[0];
        const double threshold = static_cast<double>(uniform) * total;
        double cumulative = 0.0;
        int32_t selected = candidates_.back().token;
        for (const auto & candidate : candidates_) {
            cumulative += candidate.weight;
            if (threshold <= cumulative) {
                selected = candidate.token;
                break;
            }
        }
        ++call_index;
        return selected;
    }

private:
    void build_candidates(
        const std::vector<float> & logits,
        const VoxtralRealtimeGenerationOptions & options) {
        const size_t vocab = logits.size();
        const size_t top_k = options.top_k <= 0
            ? vocab
            : std::min<size_t>(static_cast<size_t>(options.top_k), vocab);
        float top_k_threshold = -std::numeric_limits<float>::infinity();
        if (top_k < vocab) {
            top_k_heap_.clear();
            top_k_heap_.reserve(top_k);
            for (float logit : logits) {
                const float score = logit / options.temperature;
                if (top_k_heap_.size() < top_k) {
                    top_k_heap_.push_back(score);
                    std::push_heap(top_k_heap_.begin(), top_k_heap_.end(), std::greater<float>{});
                } else if (score > top_k_heap_.front()) {
                    std::pop_heap(top_k_heap_.begin(), top_k_heap_.end(), std::greater<float>{});
                    top_k_heap_.back() = score;
                    std::push_heap(top_k_heap_.begin(), top_k_heap_.end(), std::greater<float>{});
                }
            }
            top_k_threshold = top_k_heap_.front();
        }
        candidates_.clear();
        candidates_.reserve(top_k);
        float max_score = -std::numeric_limits<float>::infinity();
        for (size_t token = 0; token < vocab; ++token) {
            const float score = logits[token] / options.temperature;
            if (score >= top_k_threshold) {
                candidates_.push_back(LogitCandidate{static_cast<int32_t>(token), score, 0.0});
                max_score = std::max(max_score, score);
            }
        }
        if (candidates_.empty() || !std::isfinite(max_score)) {
            throw std::runtime_error("VoxTral text decoder sampling filter produced no candidates");
        }
        double total = 0.0;
        for (auto & candidate : candidates_) {
            candidate.weight = std::exp(static_cast<double>(candidate.score - max_score));
            total += candidate.weight;
        }
        if (!(total > 0.0) || !std::isfinite(total)) {
            throw std::runtime_error("VoxTral text decoder sampling distribution is invalid");
        }
        if (options.top_p < 1.0F) {
            std::sort(candidates_.begin(), candidates_.end(), by_score_asc);
            double cumulative = 0.0;
            const double remove_mass = 1.0 - static_cast<double>(options.top_p);
            const size_t keep_from = candidates_.size() > 1 ? candidates_.size() - 1 : 0;
            size_t write = 0;
            for (size_t i = 0; i < candidates_.size(); ++i) {
                cumulative += candidates_[i].weight / total;
                if (i < keep_from && cumulative <= remove_mass) {
                    continue;
                }
                candidates_[write++] = candidates_[i];
            }
            candidates_.resize(write);
        }
    }

    static bool by_score_asc(const LogitCandidate & lhs, const LogitCandidate & rhs) {
        if (lhs.score == rhs.score) {
            return lhs.token < rhs.token;
        }
        return lhs.score < rhs.score;
    }

    static bool by_token_asc(const LogitCandidate & lhs, const LogitCandidate & rhs) {
        return lhs.token < rhs.token;
    }

    std::vector<LogitCandidate> candidates_;
    std::vector<float> top_k_heap_;
};

struct TextAttentionOutput {
    core::TensorValue output;
    core::TensorValue key;
    core::TensorValue value;
};

TextAttentionOutput build_text_attention(
    core::ModuleBuildContext & ctx,
    const VoxtralRealtimeTextConfig & config,
    const TextLayerWeights & weights,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue & attention_mask) {
    auto q = modules::LinearModule({config.hidden_size, config.num_attention_heads * config.head_dim, false})
                 .build(ctx, input, {weights.q_weight, std::nullopt});
    q = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, q),
        core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[1], config.num_attention_heads, config.head_dim}));
    auto k = modules::LinearModule({config.hidden_size, config.num_key_value_heads * config.head_dim, false})
                 .build(ctx, input, {weights.k_weight, std::nullopt});
    k = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, k),
        core::TensorShape::from_dims({k.shape.dims[0], k.shape.dims[1], config.num_key_value_heads, config.head_dim}));
    auto v = modules::LinearModule({config.hidden_size, config.num_key_value_heads * config.head_dim, false})
                 .build(ctx, input, {weights.v_weight, std::nullopt});
    v = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, v),
        core::TensorShape::from_dims({v.shape.dims[0], v.shape.dims[1], config.num_key_value_heads, config.head_dim}));
    const modules::RoPEModule rope({config.head_dim, GGML_ROPE_TYPE_NEOX, config.rope_theta});
    q = rope.build(ctx, q, positions);
    k = rope.build(ctx, k, positions);
    auto cache_key = core::ensure_backend_addressable_layout(ctx, k);
    auto cache_value = core::ensure_backend_addressable_layout(ctx, v);
    q = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    k = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    v = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = modules::GroupedQueryAttentionModule({
        config.head_dim,
        modules::GroupedQueryAttentionLowering::FlashGrouped,
        GGML_PREC_F32,
    }).build(ctx, q, k, v, attention_mask);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_attention_heads * config.head_dim}));
    return {
        modules::LinearModule({config.num_attention_heads * config.head_dim, config.hidden_size, false})
            .build(ctx, context, {weights.o_weight, std::nullopt}),
        cache_key,
        cache_value,
    };
}

TextAttentionOutput build_text_attention_static(
    core::ModuleBuildContext & ctx,
    const VoxtralRealtimeTextConfig & config,
    const TextLayerWeights & weights,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue & attention_mask,
    const core::TensorValue & cache_key,
    const core::TensorValue & cache_value,
    const core::TensorValue & cache_slot) {
    auto q = modules::LinearModule({config.hidden_size, config.num_attention_heads * config.head_dim, false})
                 .build(ctx, input, {weights.q_weight, std::nullopt});
    q = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, q),
        core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[1], config.num_attention_heads, config.head_dim}));
    auto k = modules::LinearModule({config.hidden_size, config.num_key_value_heads * config.head_dim, false})
                 .build(ctx, input, {weights.k_weight, std::nullopt});
    k = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, k),
        core::TensorShape::from_dims({k.shape.dims[0], k.shape.dims[1], config.num_key_value_heads, config.head_dim}));
    auto v = modules::LinearModule({config.hidden_size, config.num_key_value_heads * config.head_dim, false})
                 .build(ctx, input, {weights.v_weight, std::nullopt});
    v = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, v),
        core::TensorShape::from_dims({v.shape.dims[0], v.shape.dims[1], config.num_key_value_heads, config.head_dim}));
    const modules::RoPEModule rope({config.head_dim, GGML_ROPE_TYPE_NEOX, config.rope_theta});
    q = rope.build(ctx, q, positions);
    k = rope.build(ctx, k, positions);
    k = core::ensure_backend_addressable_layout(ctx, k);
    v = core::ensure_backend_addressable_layout(ctx, v);
    const modules::FastKVSetRowsModule set_rows;
    auto updated_key = set_rows.build(ctx, cache_key, k, cache_slot);
    auto updated_value = set_rows.build(ctx, cache_value, v, cache_slot);
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    q_heads = core::wrap_tensor(ggml_cont(ctx.ggml, q_heads.tensor), q_heads.shape, q_heads.type);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, updated_key.shape.rank}).build(ctx, updated_key);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, updated_value.shape.rank}).build(ctx, updated_value);
    k_heads = core::wrap_tensor(ggml_cont(ctx.ggml, k_heads.tensor), k_heads.shape, k_heads.type);
    v_heads = core::wrap_tensor(ggml_cont(ctx.ggml, v_heads.tensor), v_heads.shape, v_heads.type);
    auto context = modules::GroupedQueryAttentionModule({
        config.head_dim,
        modules::GroupedQueryAttentionLowering::FlashGrouped,
        GGML_PREC_F32,
    }).build(ctx, q_heads, k_heads, v_heads, attention_mask);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({1, 1, config.num_attention_heads * config.head_dim}));
    return {
        modules::LinearModule({config.num_attention_heads * config.head_dim, config.hidden_size, false})
            .build(ctx, context, {weights.o_weight, std::nullopt}),
        k,
        v,
    };
}

core::TensorValue build_text_mlp(
    core::ModuleBuildContext & ctx,
    const VoxtralRealtimeTextConfig & config,
    const TextLayerWeights & weights,
    const core::TensorValue & input) {
    auto gate = modules::LinearModule({config.hidden_size, config.intermediate_size, false})
                    .build(ctx, input, {weights.gate_weight, std::nullopt});
    gate = modules::SiluModule().build(ctx, gate);
    auto up = modules::LinearModule({config.hidden_size, config.intermediate_size, false})
                  .build(ctx, input, {weights.up_weight, std::nullopt});
    auto hidden = modules::MulModule().build(ctx, gate, up);
    return modules::LinearModule({config.intermediate_size, config.hidden_size, false})
        .build(ctx, hidden, {weights.down_weight, std::nullopt});
}

class FullContextGraph {
public:
    struct Output {
        int32_t token = 0;
        runtime::TransformerKVState kv_state;
    };

    FullContextGraph(
        std::shared_ptr<const VoxtralRealtimeAssets> assets,
        std::shared_ptr<const TextWeights> weights,
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        int64_t capacity,
        int64_t audio_tokens,
        size_t arena_bytes)
        : assets_(std::move(assets)),
          weights_(std::move(weights)),
          backend_(backend),
          backend_type_(backend_type),
          threads_(threads),
          capacity_(capacity),
          audio_tokens_(audio_tokens) {
        const auto build_start = Clock::now();
        if (capacity_ <= 0) {
            throw std::runtime_error("VoxTral text decoder graph requires positive capacity");
        }
        ggml_init_params params{arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VoxTral text decoder graph context");
        }
        const auto & config = assets_->config.text;
        core::ModuleBuildContext ctx{ctx_.get(), "voxtral_realtime.text_decoder.full_context", backend_type_};
        token_ids_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, capacity_);
        audio_embeddings_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_F32, config.hidden_size, capacity_);
        time_embedding_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_F32, config.hidden_size);
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, capacity_);
        mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, capacity_, capacity_, 1, 1);
        step_index_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, 1);
        auto ids = core::wrap_tensor(token_ids_, core::TensorShape::from_dims({capacity_}), GGML_TYPE_I32);
        auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_size}).build(ctx, ids, weights_->token_embedding);
        auto audio = core::wrap_tensor(audio_embeddings_, core::TensorShape::from_dims({capacity_, config.hidden_size}), GGML_TYPE_F32);
        x = modules::AddModule().build(ctx, x, audio);
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, capacity_, config.hidden_size}));
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({capacity_}), GGML_TYPE_I32);
        auto mask = core::wrap_tensor(mask_, core::TensorShape::from_dims({1, 1, capacity_, capacity_}), GGML_TYPE_F16);
        auto t_cond = core::wrap_tensor(time_embedding_, core::TensorShape::from_dims({config.hidden_size}), GGML_TYPE_F32);
        for (const auto & layer : weights_->layers) {
            auto attn_in = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                               .build(ctx, x, {layer.input_norm, std::nullopt});
            auto attn = build_text_attention(ctx, config, layer, attn_in, positions, mask);
            x = modules::AddModule().build(ctx, x, attn.output);
            auto * key_readback = ggml_cpy(ctx_.get(), attn.key.tensor, ggml_dup_tensor(ctx_.get(), attn.key.tensor));
            auto * value_readback = ggml_cpy(ctx_.get(), attn.value.tensor, ggml_dup_tensor(ctx_.get(), attn.value.tensor));
            ggml_set_output(key_readback);
            ggml_set_output(value_readback);
            keys_.push_back(key_readback);
            values_.push_back(value_readback);
            auto mlp_in = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                              .build(ctx, x, {layer.post_norm, std::nullopt});
            auto ada = modules::LinearModule({config.hidden_size, 32, false})
                           .build(ctx, t_cond, {layer.ada1_weight, std::nullopt});
            ada = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, ada);
            ada = modules::LinearModule({32, config.hidden_size, false})
                      .build(ctx, ada, {layer.ada2_weight, std::nullopt});
            auto one_plus = core::wrap_tensor(ggml_scale_bias(ctx.ggml, ada.tensor, 1.0F, 1.0F), ada.shape, GGML_TYPE_F32);
            one_plus = core::reshape_tensor(ctx, one_plus, core::TensorShape::from_dims({1, 1, config.hidden_size}));
            one_plus = core::wrap_tensor(ggml_repeat(ctx.ggml, one_plus.tensor, mlp_in.tensor), mlp_in.shape, GGML_TYPE_F32);
            mlp_in = modules::MulModule().build(ctx, mlp_in, one_plus);
            x = modules::AddModule().build(ctx, x, build_text_mlp(ctx, config, layer, mlp_in));
        }
        x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                .build(ctx, x, {weights_->norm, std::nullopt});
        logits_ = modules::LinearModule({config.hidden_size, config.vocab_size, false})
                      .build(ctx, x, {weights_->lm_head, std::nullopt})
                      .tensor;
        auto logits_value = core::wrap_tensor(
            logits_,
            core::TensorShape::from_dims({1, capacity_, config.vocab_size}),
            GGML_TYPE_F32);
        logits_value = core::reshape_tensor(
            ctx,
            logits_value,
            core::TensorShape::from_dims({capacity_, config.vocab_size}));
        auto step_index = core::wrap_tensor(step_index_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        selected_logits_ = ggml_get_rows(ctx_.get(), logits_value.tensor, step_index.tensor);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        for (auto * key : keys_) {
            ggml_build_forward_expand(graph_, key);
        }
        for (auto * value : values_) {
            ggml_build_forward_expand(graph_, value);
        }
        ggml_set_output(selected_logits_);
        ggml_build_forward_expand(graph_, selected_logits_);
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_.get(), graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            throw std::runtime_error("failed to allocate VoxTral text decoder graph");
        }
        const auto positions_values = modules::qwen_position_ids(capacity_);
        ggml_backend_tensor_set(positions_, positions_values.data(), 0, positions_values.size() * sizeof(int32_t));
        auto mask_values = make_text_mask(capacity_, config.sliding_window);
        ggml_backend_tensor_set(mask_, mask_values.data(), 0, mask_values.size() * sizeof(ggml_fp16_t));
        engine::debug::timing_log_scalar(
            "voxtral_realtime.text_prefill.graph_build_ms",
            engine::debug::elapsed_ms(build_start));
    }

    ~FullContextGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
    }

    bool can_run(int64_t required_capacity, int64_t audio_tokens) const {
        return capacity_ == required_capacity && audio_tokens_ == audio_tokens;
    }

    Output run(
        const std::vector<int32_t> & token_ids,
        const std::vector<float> & audio_embeddings,
        int64_t audio_tokens,
        int64_t num_delay_tokens,
        const VoxtralRealtimeGenerationOptions & options,
        uint64_t & sample_call_index,
        TokenSampler & sampler) {
        const auto total_start = Clock::now();
        const auto & config = assets_->config.text;
        if (static_cast<int64_t>(token_ids.size()) > capacity_) {
            throw std::runtime_error("VoxTral text decoder token count exceeds graph capacity");
        }
        std::vector<int32_t> padded_ids(static_cast<size_t>(capacity_), static_cast<int32_t>(config.pad_token_id));
        std::copy(token_ids.begin(), token_ids.end(), padded_ids.begin());
        std::vector<float> padded_audio(static_cast<size_t>(capacity_ * config.hidden_size), 0.0F);
        const int64_t copy_tokens = std::min<int64_t>(audio_tokens, capacity_);
        if (static_cast<int64_t>(audio_embeddings.size()) < audio_tokens * config.hidden_size) {
            throw std::runtime_error("VoxTral text decoder audio embedding value count mismatch");
        }
        std::memcpy(
            padded_audio.data(),
            audio_embeddings.data(),
            static_cast<size_t>(copy_tokens * config.hidden_size) * sizeof(float));
        const auto t = time_embedding(num_delay_tokens, config.hidden_size);
        const auto upload_start = Clock::now();
        ggml_backend_tensor_set(token_ids_, padded_ids.data(), 0, padded_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(audio_embeddings_, padded_audio.data(), 0, padded_audio.size() * sizeof(float));
        ggml_backend_tensor_set(time_embedding_, t.data(), 0, t.size() * sizeof(float));
        const int32_t step_index = static_cast<int32_t>(token_ids.empty() ? 0 : token_ids.size() - 1);
        ggml_backend_tensor_set(step_index_, &step_index, 0, sizeof(step_index));
        engine::debug::timing_log_scalar(
            "voxtral_realtime.text_prefill.input_upload_ms",
            engine::debug::elapsed_ms(upload_start));
        core::set_backend_threads(backend_, threads_);
        const auto compute_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        engine::debug::timing_log_scalar(
            "voxtral_realtime.text_prefill.graph_compute_ms",
            engine::debug::elapsed_ms(compute_start));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VoxTral text decoder graph compute failed");
        }
        Output out;
        logits_values_.resize(static_cast<size_t>(config.vocab_size));
        const auto logits_read_start = Clock::now();
        ggml_backend_tensor_get(
            selected_logits_,
            logits_values_.data(),
            0,
            logits_values_.size() * sizeof(float));
        engine::debug::timing_log_scalar(
            "voxtral_realtime.text_prefill.logits_read_ms",
            engine::debug::elapsed_ms(logits_read_start));
        const auto sample_start = Clock::now();
        out.token = sampler.select(logits_values_, options, options.seed, sample_call_index);
        engine::debug::timing_log_scalar(
            "voxtral_realtime.text_prefill.sample_ms",
            engine::debug::elapsed_ms(sample_start));
        out.kv_state.current_end = static_cast<int64_t>(token_ids.size());
        out.kv_state.layers.resize(keys_.size());
        const size_t layer_values = static_cast<size_t>(
            capacity_ * config.num_key_value_heads * config.head_dim);
        const auto kv_read_start = Clock::now();
        for (size_t layer = 0; layer < keys_.size(); ++layer) {
            auto & state_layer = out.kv_state.layers[layer];
            state_layer.valid_steps = capacity_;
            state_layer.key = core::read_tensor_f32(keys_[layer]);
            state_layer.value = core::read_tensor_f32(values_[layer]);
            if (state_layer.key.size() != layer_values || state_layer.value.size() != layer_values) {
                throw std::runtime_error("VoxTral text decoder prefill K/V export size mismatch");
            }
        }
        engine::debug::timing_log_scalar(
            "voxtral_realtime.text_prefill.kv_read_ms",
            engine::debug::elapsed_ms(kv_read_start));
        engine::debug::timing_log_scalar("voxtral_realtime.text_prefill.prompt_tokens", token_ids.size());
        engine::debug::timing_log_scalar("voxtral_realtime.text_prefill.audio_tokens", audio_tokens);
        engine::debug::timing_log_scalar("voxtral_realtime.text_prefill.total_ms", engine::debug::elapsed_ms(total_start));
        return out;
    }

private:
    std::shared_ptr<const VoxtralRealtimeAssets> assets_;
    std::shared_ptr<const TextWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    int64_t capacity_ = 0;
    int64_t audio_tokens_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
    ggml_tensor * token_ids_ = nullptr;
    ggml_tensor * audio_embeddings_ = nullptr;
    ggml_tensor * time_embedding_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * step_index_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    ggml_tensor * selected_logits_ = nullptr;
    std::vector<ggml_tensor *> keys_;
    std::vector<ggml_tensor *> values_;
    std::vector<float> logits_values_;
    ggml_cgraph * graph_ = nullptr;
};

class DecodeStepGraph {
public:
    DecodeStepGraph(
        std::shared_ptr<const VoxtralRealtimeAssets> assets,
        std::shared_ptr<const TextWeights> weights,
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        int64_t cache_steps,
        size_t arena_bytes)
        : assets_(std::move(assets)),
          weights_(std::move(weights)),
          backend_(backend),
          backend_type_(backend_type),
          threads_(threads),
          cache_steps_(cache_steps) {
        if (cache_steps_ <= 0) {
            throw std::runtime_error("VoxTral text decoder step graph requires positive cache length");
        }
        ggml_init_params state_params{8ull * 1024ull * 1024ull, nullptr, true};
        state_ctx_.reset(ggml_init(state_params));
        if (state_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VoxTral text decoder step state context");
        }
        ggml_init_params graph_params{arena_bytes, nullptr, true};
        graph_ctx_.reset(ggml_init(graph_params));
        if (graph_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VoxTral text decoder step graph context");
        }
        const auto & config = assets_->config.text;
        token_id_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_I32, 1);
        audio_embedding_ = ggml_new_tensor_2d(state_ctx_.get(), GGML_TYPE_F32, config.hidden_size, 1);
        time_embedding_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_F32, config.hidden_size);
        positions_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_I32, 1);
        cache_slot_ = ggml_new_tensor_1d(state_ctx_.get(), GGML_TYPE_I32, 1);
        attention_mask_ = ggml_new_tensor_4d(state_ctx_.get(), GGML_TYPE_F16, cache_steps_, 1, 1, 1);
        std::vector<core::TensorValue> cache_keys;
        std::vector<core::TensorValue> cache_values;
        cache_keys.reserve(weights_->layers.size());
        cache_values.reserve(weights_->layers.size());
        for (size_t layer = 0; layer < weights_->layers.size(); ++layer) {
            cache_keys.push_back(core::wrap_tensor(
                ggml_new_tensor_4d(
                    state_ctx_.get(),
                    GGML_TYPE_F32,
                    config.head_dim,
                    config.num_key_value_heads,
                    cache_steps_,
                    1),
                core::TensorShape::from_dims({1, cache_steps_, config.num_key_value_heads, config.head_dim}),
                GGML_TYPE_F32));
            cache_values.push_back(core::wrap_tensor(
                ggml_new_tensor_4d(
                    state_ctx_.get(),
                    GGML_TYPE_F32,
                    config.head_dim,
                    config.num_key_value_heads,
                    cache_steps_,
                    1),
                core::TensorShape::from_dims({1, cache_steps_, config.num_key_value_heads, config.head_dim}),
                GGML_TYPE_F32));
        }
        state_buffer_ = ggml_backend_alloc_ctx_tensors(state_ctx_.get(), backend_);
        if (state_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate VoxTral text decoder step state tensors");
        }

        core::ModuleBuildContext ctx{graph_ctx_.get(), "voxtral_realtime.text_decoder.step", backend_type_};
        auto ids = core::wrap_tensor(token_id_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto x = modules::EmbeddingModule({config.vocab_size, config.hidden_size}).build(ctx, ids, weights_->token_embedding);
        auto audio = core::wrap_tensor(audio_embedding_, core::TensorShape::from_dims({1, config.hidden_size}), GGML_TYPE_F32);
        x = modules::AddModule().build(ctx, x, audio);
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, 1, config.hidden_size}));
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto cache_slot = core::wrap_tensor(cache_slot_, core::TensorShape::from_dims({1}), GGML_TYPE_I32);
        auto attention_mask = core::wrap_tensor(
            attention_mask_,
            core::TensorShape::from_dims({1, 1, 1, cache_steps_}),
            GGML_TYPE_F16);
        auto t_cond = core::wrap_tensor(time_embedding_, core::TensorShape::from_dims({config.hidden_size}), GGML_TYPE_F32);
        for (size_t layer_index = 0; layer_index < weights_->layers.size(); ++layer_index) {
            const auto & layer = weights_->layers[layer_index];
            auto attn_in = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                               .build(ctx, x, {layer.input_norm, std::nullopt});
            auto attn = build_text_attention_static(
                ctx,
                config,
                layer,
                attn_in,
                positions,
                attention_mask,
                cache_keys[layer_index],
                cache_values[layer_index],
                cache_slot);
            x = modules::AddModule().build(ctx, x, attn.output);
            auto mlp_in = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                              .build(ctx, x, {layer.post_norm, std::nullopt});
            auto ada = modules::LinearModule({config.hidden_size, 32, false})
                           .build(ctx, t_cond, {layer.ada1_weight, std::nullopt});
            ada = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, ada);
            ada = modules::LinearModule({32, config.hidden_size, false})
                      .build(ctx, ada, {layer.ada2_weight, std::nullopt});
            auto one_plus = core::wrap_tensor(ggml_scale_bias(ctx.ggml, ada.tensor, 1.0F, 1.0F), ada.shape, GGML_TYPE_F32);
            one_plus = core::reshape_tensor(ctx, one_plus, core::TensorShape::from_dims({1, 1, config.hidden_size}));
            one_plus = core::wrap_tensor(ggml_repeat(ctx.ggml, one_plus.tensor, mlp_in.tensor), mlp_in.shape, GGML_TYPE_F32);
            mlp_in = modules::MulModule().build(ctx, mlp_in, one_plus);
            x = modules::AddModule().build(ctx, x, build_text_mlp(ctx, config, layer, mlp_in));
        }
        x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false})
                .build(ctx, x, {weights_->norm, std::nullopt});
        logits_ = modules::LinearModule({config.hidden_size, config.vocab_size, false})
                      .build(ctx, x, {weights_->lm_head, std::nullopt})
                      .tensor;
        ggml_set_output(logits_);
        graph_ = ggml_new_graph_custom(graph_ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, logits_);
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_.get(), graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            throw std::runtime_error("failed to allocate VoxTral text decoder step graph tensors");
        }
        cache_ = runtime::TransformerKVCache(
            cache_steps_,
            config.num_key_value_heads * config.head_dim,
            std::move(cache_keys),
            std::move(cache_values));
        attention_mask_values_.assign(static_cast<size_t>(cache_steps_), ggml_fp32_to_fp16(-INFINITY));
        ggml_backend_tensor_set(
            attention_mask_,
            attention_mask_values_.data(),
            0,
            attention_mask_values_.size() * sizeof(ggml_fp16_t));
    }

    ~DecodeStepGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        gallocr_.reset();
        if (state_buffer_ != nullptr) {
            ggml_backend_buffer_free(state_buffer_);
            state_buffer_ = nullptr;
        }
    }

    int64_t cache_steps() const noexcept {
        return cache_steps_;
    }

    runtime::TransformerKVState export_state() const {
        return cache_.export_state();
    }

    void import_state(const runtime::TransformerKVState & state) {
        if (state.layers.empty()) {
            throw std::runtime_error("VoxTral text decoder step cache requires prefill state");
        }
        if (state.layers.front().valid_steps > cache_steps_) {
            throw std::runtime_error("VoxTral text decoder prefill state exceeds sliding cache");
        }
        cache_.import_state(state);
        attention_mask_prefix_uploaded_ = false;
    }

    void advance_cache_after_direct_append(int64_t steps) {
        cache_.advance_after_direct_append(steps);
    }

    int32_t run_step(
        const runtime::BoundedStaticKVDecodeStep & step,
        int32_t token,
        const VoxtralRealtimeAudioEmbeddings & audio_embeddings,
        int64_t num_delay_tokens,
        bool single_audio_embedding,
        const VoxtralRealtimeGenerationOptions & options,
        uint64_t & sample_call_index,
        TokenSampler & sampler) {
        const auto & config = assets_->config.text;
        const int64_t position = step.position;
        if (position < 0) {
            throw std::runtime_error("VoxTral text decoder audio embedding position is out of range");
        }
        const int64_t audio_index = single_audio_embedding ? 0 : position;
        if (audio_index >= audio_embeddings.tokens) {
            throw std::runtime_error("VoxTral text decoder audio embedding position is out of range");
        }
        ggml_backend_tensor_set(token_id_, &token, 0, sizeof(token));
        ggml_backend_tensor_set(
            audio_embedding_,
            audio_embeddings.values.data() + static_cast<ptrdiff_t>(audio_index * config.hidden_size),
            0,
            static_cast<size_t>(config.hidden_size) * sizeof(float));
        if (uploaded_time_embedding_delay_tokens_ != num_delay_tokens) {
            time_embedding_values_ = time_embedding(num_delay_tokens, config.hidden_size);
            ggml_backend_tensor_set(
                time_embedding_,
                time_embedding_values_.data(),
                0,
                time_embedding_values_.size() * sizeof(float));
            uploaded_time_embedding_delay_tokens_ = num_delay_tokens;
        }
        const int32_t position_i32 = static_cast<int32_t>(position);
        ggml_backend_tensor_set(positions_, &position_i32, 0, sizeof(position_i32));
        ggml_backend_tensor_set(cache_slot_, &step.cache_slot, 0, sizeof(step.cache_slot));
        write_attention_mask(step);
        core::set_backend_threads(backend_, threads_);
        const ggml_status status = core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VoxTral text decoder step graph compute failed");
        }
        logits_values_.resize(static_cast<size_t>(config.vocab_size));
        ggml_backend_tensor_get(logits_, logits_values_.data(), 0, logits_values_.size() * sizeof(float));
        const int32_t next_token = sampler.select(logits_values_, options, options.seed, sample_call_index);
        return next_token;
    }

private:
    void write_attention_mask(const runtime::BoundedStaticKVDecodeStep & step) {
        const auto visible = ggml_fp32_to_fp16(0.0F);
        if (!attention_mask_prefix_uploaded_) {
            const auto masked = ggml_fp32_to_fp16(-INFINITY);
            std::fill(attention_mask_values_.begin(), attention_mask_values_.end(), masked);
            for (int64_t i = 0; i < step.valid_steps; ++i) {
                attention_mask_values_[static_cast<size_t>(i)] = visible;
            }
            attention_mask_values_[static_cast<size_t>(step.cache_slot)] = visible;
            ggml_backend_tensor_set(
                attention_mask_,
                attention_mask_values_.data(),
                0,
                attention_mask_values_.size() * sizeof(ggml_fp16_t));
            attention_mask_prefix_uploaded_ = true;
            return;
        }
        if (step.valid_steps < step.cache_steps) {
            attention_mask_values_[static_cast<size_t>(step.cache_slot)] = visible;
            ggml_backend_tensor_set(
                attention_mask_,
                attention_mask_values_.data() + step.cache_slot,
                static_cast<size_t>(step.cache_slot) * sizeof(ggml_fp16_t),
                sizeof(ggml_fp16_t));
        }
    }

    std::shared_ptr<const VoxtralRealtimeAssets> assets_;
    std::shared_ptr<const TextWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    int64_t cache_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> state_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx_;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
    ggml_backend_buffer_t state_buffer_ = nullptr;
    ggml_tensor * token_id_ = nullptr;
    ggml_tensor * audio_embedding_ = nullptr;
    ggml_tensor * time_embedding_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * cache_slot_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    std::vector<ggml_fp16_t> attention_mask_values_;
    bool attention_mask_prefix_uploaded_ = false;
    std::vector<float> time_embedding_values_;
    std::vector<float> logits_values_;
    int64_t uploaded_time_embedding_delay_tokens_ = std::numeric_limits<int64_t>::min();
    runtime::TransformerKVCache cache_;
    ggml_cgraph * graph_ = nullptr;
};

}  // namespace

struct VoxtralRealtimeTextDecoderRuntime::Impl {
    Impl(
        std::shared_ptr<const VoxtralRealtimeAssets> assets,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets(std::move(assets)),
          backend(execution.backend()),
          backend_type(execution.backend_type()),
          threads(std::max(1, execution.config().threads)),
          graph_arena_bytes(std::max(prefill_graph_arena_bytes, decode_graph_arena_bytes)) {
        if (this->assets == nullptr) {
            throw std::runtime_error("VoxTral text decoder requires assets");
        }
        weights = load_weights(*this->assets, backend, backend_type, weight_context_bytes, storage_type);
        decode_runtime = runtime::BoundedStaticKVDecodeRuntime<DecodeStepGraph>({
            this->assets->config.text.sliding_window,
            256,
            "VoxTral text decoder bounded KV",
        });
    }

    VoxtralRealtimeGeneratedTokens generate(
        const VoxtralRealtimePrompt & prompt,
        const VoxtralRealtimeAudioEmbeddings & audio_embeddings,
        const VoxtralRealtimeGenerationOptions & options) {
        const auto & config = assets->config.text;
        if (prompt.input_ids.empty()) {
            throw std::runtime_error("VoxTral text decoder prompt is empty");
        }
        if (audio_embeddings.hidden_size != config.hidden_size) {
            throw std::runtime_error("VoxTral text decoder audio embedding hidden size mismatch");
        }
        if (options.max_new_tokens <= 0) {
            return {};
        }
        const auto total_start = Clock::now();
        uint64_t sample_call_index = 0;
        const int64_t prompt_steps = static_cast<int64_t>(prompt.input_ids.size());
        const auto decode_graph_factory = [this](int64_t cache_steps) {
            auto graph = std::make_unique<DecodeStepGraph>(
                assets,
                weights,
                backend,
                backend_type,
                threads,
                cache_steps,
                graph_arena_bytes);
            engine::debug::timing_log_scalar("voxtral_realtime.text_decoder.decode_cache_steps", cache_steps);
            return graph;
        };
        bool prefill_rebuilt = false;
        if (prefill_graph == nullptr || !prefill_graph->can_run(prompt_steps, audio_embeddings.tokens)) {
            prefill_rebuilt = true;
            prefill_graph = std::make_unique<FullContextGraph>(
                assets,
                weights,
                backend,
                backend_type,
                threads,
                prompt_steps,
                audio_embeddings.tokens,
                graph_arena_bytes);
        }
        const auto prefill_start = Clock::now();
        auto prefill = prefill_graph->run(
            prompt.input_ids,
            audio_embeddings.values,
            audio_embeddings.tokens,
            prompt.num_delay_tokens,
            options,
            sample_call_index,
            sampler);
        const double prefill_ms = engine::debug::elapsed_ms(prefill_start);
        bool decode_rebuilt = decode_runtime.prepare_for_prefill(prompt_steps, decode_graph_factory);
        int64_t decode_rebuilds = decode_rebuilt ? 1 : 0;
        decode_runtime.import_state(prefill.kv_state);
        int32_t token = prefill.token;
        VoxtralRealtimeGeneratedTokens out;
        double step_ms = 0.0;
        for (int64_t step = 0; step < options.max_new_tokens; ++step) {
            if (token == config.eos_token_id) {
                break;
            }
            out.token_ids.push_back(token);
            if (step + 1 < options.max_new_tokens) {
                const auto step_start = Clock::now();
                const auto current_step = decode_runtime.next_step();
                const bool grew_decode_graph =
                    decode_runtime.grow_for_next_step(current_step.valid_steps + 1, decode_graph_factory);
                decode_rebuilt = decode_rebuilt || grew_decode_graph;
                if (grew_decode_graph) {
                    ++decode_rebuilds;
                }
                token = decode_runtime.graph().run_step(
                    decode_runtime.next_step(),
                    token,
                    audio_embeddings,
                    prompt.num_delay_tokens,
                    false,
                    options,
                    sample_call_index,
                    sampler);
                decode_runtime.advance_after_direct_append(1);
                step_ms += engine::debug::elapsed_ms(step_start);
            }
        }
        engine::debug::timing_log_scalar("voxtral_realtime.text_decoder.prompt_tokens", prompt_steps);
        engine::debug::timing_log_scalar("voxtral_realtime.text_decoder.audio_tokens", audio_embeddings.tokens);
        engine::debug::timing_log_scalar("voxtral_realtime.text_decoder.max_new_tokens", options.max_new_tokens);
        engine::debug::timing_log_scalar("voxtral_realtime.text_decoder.generated_tokens", out.token_ids.size());
        engine::debug::timing_log_scalar("voxtral_realtime.text_decoder.prefill_rebuilt", prefill_rebuilt);
        engine::debug::timing_log_scalar("voxtral_realtime.text_decoder.decode_rebuilt", decode_rebuilt);
        engine::debug::timing_log_scalar("voxtral_realtime.text_decoder.decode_rebuilds", decode_rebuilds);
        engine::debug::timing_log_scalar("voxtral_realtime.text_decoder.prefill_ms", prefill_ms);
        engine::debug::timing_log_scalar("voxtral_realtime.text_decoder.step_total_ms", step_ms);
        engine::debug::timing_log_scalar("voxtral_realtime.text_decoder.total_ms", engine::debug::elapsed_ms(total_start));
        return out;
    }

    int32_t begin_stream(
        const VoxtralRealtimePrompt & prompt,
        const VoxtralRealtimeAudioEmbeddings & audio_embeddings,
        const VoxtralRealtimeGenerationOptions & options) {
        if (prompt.input_ids.empty()) {
            throw std::runtime_error("VoxTral streaming text decoder prompt is empty");
        }
        if (prompt.input_ids.size() != static_cast<size_t>(audio_embeddings.tokens)) {
            throw std::runtime_error("VoxTral streaming text decoder prompt/audio token count mismatch");
        }
        const auto total_start = Clock::now();
        stream_sample_call_index_ = 0;
        const auto decode_graph_factory = [this](int64_t cache_steps) {
            auto graph = std::make_unique<DecodeStepGraph>(
                assets,
                weights,
                backend,
                backend_type,
                threads,
                cache_steps,
                graph_arena_bytes);
            engine::debug::timing_log_scalar("voxtral_realtime.text_decoder.decode_cache_steps", cache_steps);
            return graph;
        };
        bool prefill_rebuilt = false;
        if (prefill_graph == nullptr || !prefill_graph->can_run(static_cast<int64_t>(prompt.input_ids.size()), audio_embeddings.tokens)) {
            prefill_rebuilt = true;
            prefill_graph = std::make_unique<FullContextGraph>(
                assets,
                weights,
                backend,
                backend_type,
                threads,
                static_cast<int64_t>(prompt.input_ids.size()),
                audio_embeddings.tokens,
                graph_arena_bytes);
        }
        const auto prefill_start = Clock::now();
        auto prefill = prefill_graph->run(
            prompt.input_ids,
            audio_embeddings.values,
            audio_embeddings.tokens,
            prompt.num_delay_tokens,
            options,
            stream_sample_call_index_,
            sampler);
        const double prefill_ms = engine::debug::elapsed_ms(prefill_start);
        const bool decode_rebuilt = decode_runtime.prepare_for_prefill(
            static_cast<int64_t>(prompt.input_ids.size()),
            decode_graph_factory);
        decode_runtime.import_state(prefill.kv_state);
        prefill_graph.reset();
        engine::debug::timing_log_scalar("voxtral_realtime.text_decoder.stream.begin.prompt_tokens", prompt.input_ids.size());
        engine::debug::timing_log_scalar("voxtral_realtime.text_decoder.stream.begin.audio_tokens", audio_embeddings.tokens);
        engine::debug::timing_log_scalar("voxtral_realtime.text_decoder.stream.begin.prefill_rebuilt", prefill_rebuilt);
        engine::debug::timing_log_scalar("voxtral_realtime.text_decoder.stream.begin.decode_rebuilt", decode_rebuilt);
        engine::debug::timing_log_scalar("voxtral_realtime.text_decoder.stream.begin.prefill_ms", prefill_ms);
        engine::debug::timing_log_scalar(
            "voxtral_realtime.text_decoder.stream.begin.total_ms",
            engine::debug::elapsed_ms(total_start));
        return prefill.token;
    }

    int32_t stream_step(
        int32_t previous_token,
        const VoxtralRealtimeAudioEmbeddings & audio_embeddings,
        int64_t num_delay_tokens,
        const VoxtralRealtimeGenerationOptions & options) {
        if (!decode_runtime.has_graph()) {
            throw std::runtime_error("VoxTral streaming text decoder requires begin_stream before stream_step");
        }
        if (audio_embeddings.tokens != 1) {
            throw std::runtime_error("VoxTral streaming text decoder expects one audio token per steady chunk");
        }
        const auto decode_graph_factory = [this](int64_t cache_steps) {
            auto graph = std::make_unique<DecodeStepGraph>(
                assets,
                weights,
                backend,
                backend_type,
                threads,
                cache_steps,
                graph_arena_bytes);
            engine::debug::timing_log_scalar("voxtral_realtime.text_decoder.decode_cache_steps", cache_steps);
            return graph;
        };
        const auto current_step = decode_runtime.next_step();
        decode_runtime.grow_for_next_step(current_step.valid_steps + 1, decode_graph_factory);
        const int32_t token = decode_runtime.graph().run_step(
            decode_runtime.next_step(),
            previous_token,
            audio_embeddings,
            num_delay_tokens,
            true,
            options,
            stream_sample_call_index_,
            sampler);
        decode_runtime.advance_after_direct_append(1);
        return token;
    }

    bool is_eos(int32_t token) const {
        return token == assets->config.text.eos_token_id;
    }

    std::shared_ptr<const VoxtralRealtimeAssets> assets;
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    int threads = 1;
    size_t graph_arena_bytes = 0;
    std::shared_ptr<const TextWeights> weights;
    std::unique_ptr<FullContextGraph> prefill_graph;
    runtime::BoundedStaticKVDecodeRuntime<DecodeStepGraph> decode_runtime;
    TokenSampler sampler;
    uint64_t stream_sample_call_index_ = 0;
};

VoxtralRealtimeTextDecoderRuntime::VoxtralRealtimeTextDecoderRuntime(
    std::shared_ptr<const VoxtralRealtimeAssets> assets,
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

VoxtralRealtimeTextDecoderRuntime::~VoxtralRealtimeTextDecoderRuntime() = default;

VoxtralRealtimeGeneratedTokens VoxtralRealtimeTextDecoderRuntime::generate(
    const VoxtralRealtimePrompt & prompt,
    const VoxtralRealtimeAudioEmbeddings & audio_embeddings,
    const VoxtralRealtimeGenerationOptions & options) {
    return impl_->generate(prompt, audio_embeddings, options);
}

int32_t VoxtralRealtimeTextDecoderRuntime::begin_stream(
    const VoxtralRealtimePrompt & prompt,
    const VoxtralRealtimeAudioEmbeddings & audio_embeddings,
    const VoxtralRealtimeGenerationOptions & options) {
    return impl_->begin_stream(prompt, audio_embeddings, options);
}

int32_t VoxtralRealtimeTextDecoderRuntime::stream_step(
    int32_t previous_token,
    const VoxtralRealtimeAudioEmbeddings & audio_embeddings,
    int64_t num_delay_tokens,
    const VoxtralRealtimeGenerationOptions & options) {
    return impl_->stream_step(previous_token, audio_embeddings, num_delay_tokens, options);
}

bool VoxtralRealtimeTextDecoderRuntime::is_eos(int32_t token) const {
    return impl_->is_eos(token);
}

}  // namespace engine::models::voxtral_realtime
