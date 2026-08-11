#include "engine/models/muscriptor/decoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/transformers/qwen_causal_decoder.h"
#include "engine/framework/sampling/hf_sampler.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::muscriptor {
namespace {

namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

constexpr int64_t kGraphNodes = 262144;
constexpr int64_t kReservedVocabStart = 1393;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

int64_t head_dim(const MuScriptorConfig & config) {
    if (config.dim % config.num_heads != 0) {
        throw std::runtime_error("MuScriptor hidden size must be divisible by attention heads");
    }
    return config.dim / config.num_heads;
}

int64_t decode_attention_bucket(int64_t required_steps, int64_t max_steps) {
    if (required_steps <= 0 || max_steps <= 0 || required_steps > max_steps) {
        throw std::runtime_error("MuScriptor decode attention bucket shape is invalid");
    }
    constexpr int64_t kMinBucket = 1024;
    constexpr int64_t kBucketMultiple = 512;
    const int64_t bucket = required_steps <= kMinBucket
        ? kMinBucket
        : ((required_steps + kBucketMultiple - 1) / kBucketMultiple) * kBucketMultiple;
    return std::min(bucket, max_steps);
}

assets::TensorStorageType supported_weight_storage(assets::TensorStorageType storage_type) {
    if (storage_type == assets::TensorStorageType::Native ||
        storage_type == assets::TensorStorageType::F32 ||
        storage_type == assets::TensorStorageType::F16 ||
        storage_type == assets::TensorStorageType::BF16 ||
        storage_type == assets::TensorStorageType::Q8_0) {
        return storage_type;
    }
    throw std::runtime_error("muscriptor.weight_type supports native, f32, f16, bf16, and q8_0");
}

std::vector<float> sinusoidal_position_embedding(int64_t steps, int64_t dim, int64_t offset = 0) {
    if (steps <= 0 || dim <= 0 || dim % 2 != 0) {
        throw std::runtime_error("MuScriptor positional embedding shape is invalid");
    }
    const int64_t half = dim / 2;
    const double log_base = std::log(10000.0);
    const double denom = static_cast<double>(half - 1);
    std::vector<double> inv_freq(static_cast<size_t>(half), 0.0);
    for (int64_t i = 0; i < half; ++i) {
        inv_freq[static_cast<size_t>(i)] = std::exp(-log_base * static_cast<double>(i) / denom);
    }
    std::vector<float> values(static_cast<size_t>(steps * dim), 0.0F);
    for (int64_t t = 0; t < steps; ++t) {
        const double position = static_cast<double>(offset + t);
        for (int64_t i = 0; i < half; ++i) {
            const double phase = position * inv_freq[static_cast<size_t>(i)];
            values[static_cast<size_t>(t * dim + i)] = static_cast<float>(std::cos(phase));
            values[static_cast<size_t>(t * dim + half + i)] = static_cast<float>(std::sin(phase));
        }
    }
    return values;
}

void fill_masked_logits(
    std::vector<float> & out,
    const float * logits,
    int64_t logits_size,
    const std::vector<uint8_t> & forbidden_mask) {
    const int64_t vocab = std::min<int64_t>(kReservedVocabStart, logits_size);
    out.resize(static_cast<size_t>(vocab));
    std::copy_n(logits, static_cast<size_t>(vocab), out.begin());
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < out.size() && i < forbidden_mask.size(); ++i) {
        if (forbidden_mask[i] != 0) {
            out[i] = neg_inf;
        }
    }
}

void fill_guided_logits(
    std::vector<float> & out,
    const float * conditional,
    const float * unconditional,
    int64_t logits_size,
    float guidance_scale,
    const std::vector<uint8_t> & forbidden_mask) {
    const int64_t vocab = std::min<int64_t>(kReservedVocabStart, logits_size);
    out.resize(static_cast<size_t>(vocab));
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < vocab; ++i) {
        out[static_cast<size_t>(i)] = unconditional[static_cast<size_t>(i)] +
            (conditional[static_cast<size_t>(i)] - unconditional[static_cast<size_t>(i)]) * guidance_scale;
    }
    for (size_t i = 0; i < out.size() && i < forbidden_mask.size(); ++i) {
        if (forbidden_mask[i] != 0) {
            out[i] = neg_inf;
        }
    }
}

void log_softmax_in_place(std::vector<float> & scores) {
    float max_score = -std::numeric_limits<float>::infinity();
    for (const float value : scores) {
        if (value > max_score) {
            max_score = value;
        }
    }
    if (!std::isfinite(max_score)) {
        throw std::runtime_error("MuScriptor beam search has no finite logits");
    }
    double sum = 0.0;
    for (const float value : scores) {
        if (std::isfinite(value)) {
            sum += std::exp(static_cast<double>(value - max_score));
        }
    }
    const float log_sum = static_cast<float>(std::log(sum));
    for (float & value : scores) {
        value = std::isfinite(value) ? value - max_score - log_sum : -std::numeric_limits<float>::infinity();
    }
}

struct TokenScore {
    int32_t token = 0;
    float score = -std::numeric_limits<float>::infinity();
};

std::vector<TokenScore> top_k_scores(std::vector<float> scores, int64_t k) {
    if (k <= 0 || scores.empty()) {
        throw std::runtime_error("MuScriptor top-k requires a positive k");
    }
    log_softmax_in_place(scores);
    std::vector<TokenScore> out;
    out.reserve(static_cast<size_t>(std::min<int64_t>(k, static_cast<int64_t>(scores.size()))));
    for (int64_t token = 0; token < static_cast<int64_t>(scores.size()); ++token) {
        const float score = scores[static_cast<size_t>(token)];
        if (!std::isfinite(score)) {
            continue;
        }
        out.push_back({static_cast<int32_t>(token), score});
    }
    const auto keep = static_cast<size_t>(std::min<int64_t>(k, static_cast<int64_t>(out.size())));
    if (keep == 0) {
        throw std::runtime_error("MuScriptor top-k found no valid tokens");
    }
    std::partial_sort(
        out.begin(),
        out.begin() + static_cast<ptrdiff_t>(keep),
        out.end(),
        [](const TokenScore & lhs, const TokenScore & rhs) {
            return lhs.score > rhs.score;
        });
    out.resize(keep);
    return out;
}

core::TensorValue view_batched_kv_cache_steps(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    int64_t start,
    int64_t steps,
    const MuScriptorConfig & config,
    const char * label,
    ggml_type view_type = GGML_TYPE_F32) {
    const int64_t dim = head_dim(config);
    if (cache.shape.rank != 4 ||
        cache.shape.dims[1] < start + steps ||
        cache.shape.dims[2] != config.num_heads ||
        cache.shape.dims[3] != dim) {
        throw std::runtime_error(std::string(label) + " batched KV cache view shape mismatch");
    }
    return core::wrap_tensor(
        ggml_view_4d(
            ctx.ggml,
            cache.tensor,
            dim,
            config.num_heads,
            steps,
            cache.shape.dims[0],
            cache.tensor->nb[1],
            cache.tensor->nb[2],
            cache.tensor->nb[3],
            static_cast<size_t>(start) * cache.tensor->nb[2]),
        core::TensorShape::from_dims({cache.shape.dims[0], steps, config.num_heads, dim}),
        view_type);
}

class DecodeKVCache {
public:
    DecodeKVCache() = default;

    DecodeKVCache(
        int64_t cache_steps,
        int64_t step_elems,
        std::vector<core::TensorValue> keys,
        std::vector<core::TensorValue> values,
        ggml_type storage_type)
        : cache_steps_(cache_steps),
          storage_type_(storage_type),
          keys_(std::move(keys)),
          values_(std::move(values)) {
        if (cache_steps_ <= 0 || step_elems <= 0) {
            throw std::runtime_error("MuScriptor decode KV cache requires positive shape");
        }
        if (keys_.size() != values_.size()) {
            throw std::runtime_error("MuScriptor decode KV cache key/value layer count mismatch");
        }
        for (size_t layer = 0; layer < keys_.size(); ++layer) {
            core::validate_shape(keys_[layer], values_[layer].shape, "muscriptor decode KV cache");
            if (keys_[layer].type != storage_type_ || values_[layer].type != storage_type_) {
                throw std::runtime_error("MuScriptor decode KV cache storage type mismatch");
            }
        }
    }

    void adopt_direct_prefix(int64_t steps) {
        if (steps < 0 || steps > cache_steps_) {
            throw std::runtime_error("MuScriptor decode KV cache direct prefix exceeds capacity");
        }
        valid_steps_ = steps;
        current_end_ = steps;
    }

    void advance_after_direct_append(int64_t steps) {
        if (steps <= 0) {
            return;
        }
        if (valid_steps_ + steps > cache_steps_) {
            throw std::runtime_error("MuScriptor decode KV cache direct append exceeds capacity");
        }
        valid_steps_ += steps;
        current_end_ += steps;
    }

    int64_t valid_steps() const noexcept {
        return valid_steps_;
    }

    int64_t current_end() const noexcept {
        return current_end_;
    }

    const core::TensorValue & key_tensor(size_t layer) const {
        return keys_.at(layer);
    }

    const core::TensorValue & value_tensor(size_t layer) const {
        return values_.at(layer);
    }

    size_t layer_count() const noexcept {
        return keys_.size();
    }

private:
    int64_t cache_steps_ = 0;
    int64_t valid_steps_ = 0;
    int64_t current_end_ = 0;
    ggml_type storage_type_ = GGML_TYPE_F32;
    std::vector<core::TensorValue> keys_;
    std::vector<core::TensorValue> values_;
};

core::TensorValue build_transformer_layer(
    core::ModuleBuildContext & ctx,
    ggml_cgraph * graph,
    const core::TensorValue & input,
    const MuScriptorLayerWeights & weights,
    const MuScriptorConfig & config,
    std::optional<core::TensorValue> cache_key,
    std::optional<core::TensorValue> cache_value,
    std::optional<core::TensorValue> cache_slot,
    std::optional<int64_t> attention_steps,
    std::optional<core::TensorValue> attention_mask,
    MuScriptorPerfMode perf_mode,
    core::TensorValue * key_out,
    core::TensorValue * value_out) {
    const int64_t dim = head_dim(config);
    const int64_t batch = input.shape.dims[0];
    const int64_t steps = input.shape.dims[1];
    auto x_norm = modules::LayerNormModule({config.dim, 1.0e-5F, true, true}).build(ctx, input, weights.norm1);
    auto qkv = modules::LinearModule({config.dim, config.dim * 3, false}).build(ctx, x_norm, weights.in_proj);
    auto q = modules::SliceModule({2, 0, config.dim}).build(ctx, qkv);
    auto k = modules::SliceModule({2, config.dim, config.dim}).build(ctx, qkv);
    auto v = modules::SliceModule({2, config.dim * 2, config.dim}).build(ctx, qkv);
    q = core::ensure_backend_addressable_layout(ctx, q);
    k = core::ensure_backend_addressable_layout(ctx, k);
    v = core::ensure_backend_addressable_layout(ctx, v);
    q = core::reshape_tensor(ctx, q, core::TensorShape::from_dims({batch, steps, config.num_heads, dim}));
    k = core::reshape_tensor(ctx, k, core::TensorShape::from_dims({batch, steps, config.num_heads, dim}));
    v = core::reshape_tensor(ctx, v, core::TensorShape::from_dims({batch, steps, config.num_heads, dim}));

    core::TensorValue attn_k = k;
    core::TensorValue attn_v = v;
    if (cache_key.has_value() && cache_value.has_value()) {
        if (!cache_slot.has_value()) {
            throw std::runtime_error("MuScriptor cached decode requires cache_slot");
        }
        const modules::FastKVSetRowsModule set_rows({modules::FastKVSetRowsMode::BackendViewOptimized});
        attn_k = set_rows.build(ctx, *cache_key, k, *cache_slot);
        attn_v = set_rows.build(ctx, *cache_value, v, *cache_slot);
        if (attention_steps.has_value()) {
            if (*attention_steps <= 0 || *attention_steps > attn_k.shape.dims[1]) {
                throw std::runtime_error("MuScriptor decode attention cache view shape is invalid");
            }
            attn_k = view_batched_kv_cache_steps(
                ctx,
                attn_k,
                0,
                *attention_steps,
                config,
                "MuScriptor decode key attention view",
                attn_k.type);
            attn_v = view_batched_kv_cache_steps(
                ctx,
                attn_v,
                0,
                *attention_steps,
                config,
                "MuScriptor decode value attention view",
                attn_v.type);
        }
    }

    if (key_out != nullptr) {
        *key_out = attn_k;
    }
    if (value_out != nullptr) {
        *value_out = attn_v;
    }
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    q_heads = core::ensure_backend_addressable_layout(ctx, q_heads);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, attn_k.shape.rank}).build(ctx, attn_k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, attn_v.shape.rank}).build(ctx, attn_v);
    const bool prefill = !cache_key.has_value();
    const bool use_flash_attention = perf_mode == MuScriptorPerfMode::FlashAttention && attention_mask.has_value();
    auto context = modules::ScaledDotProductAttentionModule({
        dim,
        use_flash_attention
            ? modules::ScaledDotProductAttentionLowering::FlashPreserveViews
            : modules::ScaledDotProductAttentionLowering::Explicit,
        GGML_PREC_F32,
        prefill ? modules::AttentionCausality::Causal : modules::AttentionCausality::NonCausal,
    }).build(ctx, q_heads, k_heads, v_heads, attention_mask);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({batch, steps, config.dim}));
    auto attn = modules::LinearModule({config.dim, config.dim, false}).build(ctx, context, weights.out_proj);
    auto x = modules::ResidualAddModule().build(ctx, attn, input);
    auto ffn_norm = modules::LayerNormModule({config.dim, 1.0e-5F, true, true}).build(ctx, x, weights.norm2);
    auto ffn = modules::FeedForwardModule({config.dim, config.dim * 4, false, modules::GeluApproximation::ExactErf})
                   .build(ctx, ffn_norm, weights.mlp);
    if (graph != nullptr && cache_key.has_value()) {
        ggml_build_forward_expand(graph, attn_k.tensor);
        ggml_build_forward_expand(graph, attn_v.tensor);
    }
    return modules::ResidualAddModule().build(ctx, ffn, x);
}

MuScriptorWeights load_weights(
    const MuScriptorAssets & assets,
    core::ExecutionContext & execution,
    const MuScriptorDecoderOptions & options) {
    const auto & config = assets.config;
    const auto & source = *assets.model_weights;
    MuScriptorWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "muscriptor.weights",
        options.weight_context_bytes);
    const auto storage = supported_weight_storage(options.weight_type);
    weights.mel_projection.weight = weights.store->load_tensor(
        source,
        "condition_provider.conditioners.self_wav.output_proj.weight",
        storage,
        {config.dim, config.n_mels});
    weights.mel_projection.bias = weights.store->load_f32_tensor(
        source,
        "condition_provider.conditioners.self_wav.output_proj.bias",
        {config.dim});
    weights.instrument_embedding = weights.store->load_f32_tensor(
        source,
        "condition_provider.conditioners.instrument_group.embed.weight",
        {1001, config.dim});
    weights.dataset_embedding = weights.store->load_f32_tensor(
        source,
        "condition_provider.conditioners.dataset_name.embed.weight",
        {5, config.dim});
    weights.token_embedding = weights.store->load_f32_tensor(source, "emb.0.weight", {config.card + 1, config.dim});
    weights.layers.reserve(static_cast<size_t>(config.num_layers));
    for (int64_t layer = 0; layer < config.num_layers; ++layer) {
        const std::string prefix = "transformer.layers." + std::to_string(layer) + ".";
        MuScriptorLayerWeights layer_weights;
        layer_weights.in_proj.weight = weights.store->load_tensor(
            source,
            prefix + "self_attn.in_proj_weight",
            storage,
            {config.dim * 3, config.dim});
        layer_weights.out_proj.weight = weights.store->load_tensor(
            source,
            prefix + "self_attn.out_proj.weight",
            storage,
            {config.dim, config.dim});
        layer_weights.norm1.weight = weights.store->load_f32_tensor(source, prefix + "norm1.weight", {config.dim});
        layer_weights.norm1.bias = weights.store->load_f32_tensor(source, prefix + "norm1.bias", {config.dim});
        layer_weights.norm2.weight = weights.store->load_f32_tensor(source, prefix + "norm2.weight", {config.dim});
        layer_weights.norm2.bias = weights.store->load_f32_tensor(source, prefix + "norm2.bias", {config.dim});
        layer_weights.mlp.fc1_weight = weights.store->load_tensor(source, prefix + "linear1.weight", storage, {config.dim * 4, config.dim});
        layer_weights.mlp.fc2_weight = weights.store->load_tensor(source, prefix + "linear2.weight", storage, {config.dim, config.dim * 4});
        weights.layers.push_back(std::move(layer_weights));
    }
    weights.output_norm.weight = weights.store->load_f32_tensor(source, "out_norm.weight", {config.dim});
    weights.output_norm.bias = weights.store->load_f32_tensor(source, "out_norm.bias", {config.dim});
    weights.output_head.weight = weights.store->load_tensor(source, "linears.0.weight", storage, {config.card, config.dim});
    weights.store->upload();
    return weights;
}

}  // namespace

class MuScriptorDecodeCacheStorage {
public:
    MuScriptorDecodeCacheStorage(
        core::ExecutionContext & execution,
        const MuScriptorConfig & config,
        int64_t cache_steps,
        int64_t batch,
        bool enable_reorder_scratch,
        ggml_type storage_type)
        : execution_(execution),
          cache_steps_(cache_steps),
          batch_(batch),
          reorder_scratch_enabled_(enable_reorder_scratch),
          storage_type_(storage_type) {
        const int64_t dim = head_dim(config);
        ggml_init_params state_params{reorder_scratch_enabled_ ? 64ull * 1024ull * 1024ull : 4ull * 1024ull * 1024ull, nullptr, true};
        state_ctx_.reset(ggml_init(state_params));
        if (state_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MuScriptor decode state context");
        }
        core::ModuleBuildContext state_ctx{state_ctx_.get(), "muscriptor.decode.state", execution_.backend_type()};
        std::vector<core::TensorValue> keys;
        std::vector<core::TensorValue> values;
        keys.reserve(static_cast<size_t>(config.num_layers));
        values.reserve(static_cast<size_t>(config.num_layers));
        for (int64_t layer = 0; layer < config.num_layers; ++layer) {
            keys.push_back(core::make_tensor(
                state_ctx,
                storage_type_,
                core::TensorShape::from_dims({batch_, cache_steps_, config.num_heads, dim})));
            values.push_back(core::make_tensor(
                state_ctx,
                storage_type_,
                core::TensorShape::from_dims({batch_, cache_steps_, config.num_heads, dim})));
        }
        if (reorder_scratch_enabled_) {
            reorder_key_scratch_ = core::make_tensor(
                state_ctx,
                storage_type_,
                core::TensorShape::from_dims({batch_, cache_steps_, config.num_heads, dim}));
            reorder_value_scratch_ = core::make_tensor(
                state_ctx,
                storage_type_,
                core::TensorShape::from_dims({batch_, cache_steps_, config.num_heads, dim}));
        }
        const auto state_alloc_start = Clock::now();
        state_buffer_ = ggml_backend_alloc_ctx_tensors(state_ctx_.get(), execution_.backend());
        if (state_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate MuScriptor decode state tensors");
        }
        engine::debug::timing_log_scalar("muscriptor.decode.cache.state_alloc_ms", engine::debug::elapsed_ms(state_alloc_start, Clock::now()));
        cache_ = DecodeKVCache(
            cache_steps_,
            batch_ * config.num_heads * dim,
            std::move(keys),
            std::move(values),
            storage_type_);
        if (reorder_scratch_enabled_) {
            build_reorder_prefix_views(config);
        }
    }

    ~MuScriptorDecodeCacheStorage() {
        if (state_buffer_ != nullptr) {
            ggml_backend_buffer_free(state_buffer_);
        }
    }

    bool matches(int64_t required_steps, int64_t batch, bool enable_reorder_scratch, ggml_type storage_type) const {
        return cache_steps_ >= required_steps &&
            batch_ == batch &&
            reorder_scratch_enabled_ == enable_reorder_scratch &&
            storage_type_ == storage_type;
    }

    void adopt_direct_prefix(int64_t steps) {
        cache_.adopt_direct_prefix(steps);
    }

    void advance_after_direct_append(int64_t steps) {
        cache_.advance_after_direct_append(steps);
    }

    int64_t cache_steps() const noexcept {
        return cache_steps_;
    }

    int64_t batch() const noexcept {
        return batch_;
    }

    int64_t valid_steps() const noexcept {
        return cache_.valid_steps();
    }

    int64_t current_end() const noexcept {
        return cache_.current_end();
    }

    const core::TensorValue & key_tensor(size_t layer) const {
        return cache_.key_tensor(layer);
    }

    const core::TensorValue & value_tensor(size_t layer) const {
        return cache_.value_tensor(layer);
    }

    size_t layer_count() const noexcept {
        return cache_.layer_count();
    }

    void copy_batch_prefixes(
        const std::vector<int64_t> & parent_rows,
        const std::vector<int64_t> & child_rows,
        int64_t valid_steps) {
        const auto reorder_start = Clock::now();
        ++profile_reorder_calls_;
        if (valid_steps <= 0) {
            profile_reorder_ms_ += engine::debug::elapsed_ms(reorder_start, Clock::now());
            return;
        }
        if (parent_rows.size() != child_rows.size() || static_cast<int64_t>(parent_rows.size()) > batch_) {
            throw std::runtime_error("MuScriptor decode beam prefix copy shape mismatch");
        }
        changed_prefix_copies_.clear();
        changed_prefix_copies_.reserve(parent_rows.size());
        for (size_t i = 0; i < parent_rows.size(); ++i) {
            const int64_t parent = parent_rows[i];
            const int64_t child = child_rows[i];
            if (parent < 0 || parent >= batch_ || child < 0 || child >= batch_) {
                throw std::runtime_error("MuScriptor decode beam prefix copy row is out of range");
            }
            if (parent != child) {
                changed_prefix_copies_.push_back({static_cast<size_t>(parent), static_cast<size_t>(child)});
            }
        }
        if (changed_prefix_copies_.empty()) {
            profile_reorder_ms_ += engine::debug::elapsed_ms(reorder_start, Clock::now());
            return;
        }
        if (!reorder_scratch_enabled_) {
            throw std::runtime_error("MuScriptor decode beam reorder scratch is not enabled");
        }
        ++profile_reorder_changed_calls_;
        profile_reorder_changed_rows_ += changed_prefix_copies_.size();
        for (size_t layer = 0; layer < cache_.layer_count(); ++layer) {
            for (size_t index = 0; index < changed_prefix_copies_.size(); ++index) {
                const auto [parent, child] = changed_prefix_copies_[index];
                (void)child;
                ggml_backend_tensor_copy(
                    key_prefix_views_[parent][static_cast<size_t>(valid_steps)][layer],
                    scratch_key_prefix_views_[index][static_cast<size_t>(valid_steps)]);
                ggml_backend_tensor_copy(
                    value_prefix_views_[parent][static_cast<size_t>(valid_steps)][layer],
                    scratch_value_prefix_views_[index][static_cast<size_t>(valid_steps)]);
            }
            for (size_t index = 0; index < changed_prefix_copies_.size(); ++index) {
                const auto [parent, child] = changed_prefix_copies_[index];
                (void)parent;
                ggml_backend_tensor_copy(
                    scratch_key_prefix_views_[index][static_cast<size_t>(valid_steps)],
                    key_prefix_views_[child][static_cast<size_t>(valid_steps)][layer]);
                ggml_backend_tensor_copy(
                    scratch_value_prefix_views_[index][static_cast<size_t>(valid_steps)],
                    value_prefix_views_[child][static_cast<size_t>(valid_steps)][layer]);
            }
        }
        profile_reorder_ms_ += engine::debug::elapsed_ms(reorder_start, Clock::now());
    }

    void reset_reorder_profile() {
        profile_reorder_ms_ = 0.0;
        profile_reorder_calls_ = 0;
        profile_reorder_changed_calls_ = 0;
        profile_reorder_changed_rows_ = 0;
    }

    double reorder_ms() const noexcept {
        return profile_reorder_ms_;
    }

    int64_t reorder_calls() const noexcept {
        return profile_reorder_calls_;
    }

    int64_t reorder_changed_calls() const noexcept {
        return profile_reorder_changed_calls_;
    }

    int64_t reorder_changed_rows() const noexcept {
        return profile_reorder_changed_rows_;
    }

private:
    void build_reorder_prefix_views(const MuScriptorConfig & config) {
        const int64_t dim = head_dim(config);
        const int64_t step_elems = config.num_heads * dim;
        key_prefix_views_.assign(static_cast<size_t>(batch_), {});
        value_prefix_views_.assign(static_cast<size_t>(batch_), {});
        scratch_key_prefix_views_.assign(static_cast<size_t>(batch_), {});
        scratch_value_prefix_views_.assign(static_cast<size_t>(batch_), {});
        const size_t element_size = ggml_type_size(storage_type_);
        for (int64_t row = 0; row < batch_; ++row) {
            auto & key_steps = key_prefix_views_[static_cast<size_t>(row)];
            auto & value_steps = value_prefix_views_[static_cast<size_t>(row)];
            key_steps.assign(static_cast<size_t>(cache_steps_ + 1), {});
            value_steps.assign(static_cast<size_t>(cache_steps_ + 1), {});
            auto & scratch_key_steps = scratch_key_prefix_views_[static_cast<size_t>(row)];
            auto & scratch_value_steps = scratch_value_prefix_views_[static_cast<size_t>(row)];
            scratch_key_steps.assign(static_cast<size_t>(cache_steps_ + 1), nullptr);
            scratch_value_steps.assign(static_cast<size_t>(cache_steps_ + 1), nullptr);
            const size_t byte_offset = static_cast<size_t>(row * cache_steps_ * step_elems) * element_size;
            for (int64_t steps = 1; steps <= cache_steps_; ++steps) {
                auto & key_layers = key_steps[static_cast<size_t>(steps)];
                auto & value_layers = value_steps[static_cast<size_t>(steps)];
                key_layers.reserve(cache_.layer_count());
                value_layers.reserve(cache_.layer_count());
                const int64_t elems = steps * step_elems;
                for (size_t layer = 0; layer < cache_.layer_count(); ++layer) {
                    key_layers.push_back(ggml_view_1d(
                        state_ctx_.get(),
                        cache_.key_tensor(layer).tensor,
                        elems,
                        byte_offset));
                    value_layers.push_back(ggml_view_1d(
                        state_ctx_.get(),
                        cache_.value_tensor(layer).tensor,
                        elems,
                        byte_offset));
                }
                scratch_key_steps[static_cast<size_t>(steps)] =
                    ggml_view_1d(state_ctx_.get(), reorder_key_scratch_.tensor, elems, byte_offset);
                scratch_value_steps[static_cast<size_t>(steps)] =
                    ggml_view_1d(state_ctx_.get(), reorder_value_scratch_.tensor, elems, byte_offset);
            }
        }
    }

    core::ExecutionContext & execution_;
    int64_t cache_steps_ = 0;
    int64_t batch_ = 1;
    bool reorder_scratch_enabled_ = false;
    ggml_type storage_type_ = GGML_TYPE_F32;
    DecodeKVCache cache_;
    core::TensorValue reorder_key_scratch_;
    core::TensorValue reorder_value_scratch_;
    std::vector<std::pair<size_t, size_t>> changed_prefix_copies_;
    std::vector<std::vector<std::vector<ggml_tensor *>>> key_prefix_views_;
    std::vector<std::vector<std::vector<ggml_tensor *>>> value_prefix_views_;
    std::vector<std::vector<ggml_tensor *>> scratch_key_prefix_views_;
    std::vector<std::vector<ggml_tensor *>> scratch_value_prefix_views_;
    double profile_reorder_ms_ = 0.0;
    int64_t profile_reorder_calls_ = 0;
    int64_t profile_reorder_changed_calls_ = 0;
    int64_t profile_reorder_changed_rows_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> state_ctx_;
    ggml_backend_buffer_t state_buffer_ = nullptr;
};

class MuScriptorDecoderRuntime::ConditionGraph {
public:
    ConditionGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<MuScriptorWeights> weights,
        const MuScriptorConfig & config,
        int64_t frames,
        int64_t instrument_steps,
        int64_t batch,
        size_t arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          frames_(frames),
          instrument_steps_(instrument_steps),
          batch_(batch) {
        const auto start = Clock::now();
        ggml_init_params params{arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MuScriptor condition graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "muscriptor.condition", execution_.backend_type()};
        log_mel_ = ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, config.n_mels, frames_, batch_);
        ggml_set_input(log_mel_);
        mel_mask_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_I32, frames_, batch_);
        ggml_set_input(mel_mask_);
        instrument_ids_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_I32, instrument_steps_, batch_);
        ggml_set_input(instrument_ids_);
        dataset_id_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, batch_);
        ggml_set_input(dataset_id_);

        auto mel = core::wrap_tensor(log_mel_, core::TensorShape::from_dims({batch_, frames_, config.n_mels}), GGML_TYPE_F32);
        auto mel_embed = modules::LinearModule({config.n_mels, config.dim, true}).build(ctx, mel, weights_->mel_projection);
        mel_embed = modules::MaskingModule().build(
            ctx,
            mel_embed,
            core::wrap_tensor(mel_mask_, core::TensorShape::from_dims({batch_, frames_}), GGML_TYPE_I32));
        auto instrument = modules::EmbeddingModule({1001, config.dim}).build(
            ctx,
            core::wrap_tensor(instrument_ids_, core::TensorShape::from_dims({batch_, instrument_steps_}), GGML_TYPE_I32),
            weights_->instrument_embedding);
        auto dataset = modules::EmbeddingModule({5, config.dim}).build(
            ctx,
            core::wrap_tensor(dataset_id_, core::TensorShape::from_dims({batch_}), GGML_TYPE_I32),
            weights_->dataset_embedding);
        dataset = core::reshape_tensor(ctx, dataset, core::TensorShape::from_dims({batch_, 1, config.dim}));
        auto text = modules::ConcatModule({1}).build(ctx, mel_embed, dataset);
        output_ = modules::ConcatModule({1}).build(ctx, text, instrument).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), kGraphNodes, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MuScriptor condition graph");
        }
        engine::debug::timing_log_scalar("muscriptor.condition.graph.build_ms", engine::debug::elapsed_ms(start, Clock::now()));
    }

    ~ConditionGraph() {
        engine::core::release_backend_graph_resources(execution_.backend(), graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(const MuScriptorWeights & weights, int64_t frames, int64_t instrument_steps, int64_t batch) const {
        return weights_.get() == &weights && frames_ == frames && instrument_steps_ == instrument_steps && batch_ == batch;
    }

    MuScriptorConditioning run(
        const std::vector<float> & log_mel,
        const std::vector<int32_t> & mel_mask,
        const std::vector<int32_t> & instrument_ids,
        const std::vector<int32_t> & dataset_ids,
        int64_t dim) {
        if (static_cast<int64_t>(log_mel.size()) != batch_ * frames_ * weights_->mel_projection.weight.shape.dims[1]) {
            throw std::runtime_error("MuScriptor condition log-mel size mismatch");
        }
        if (static_cast<int64_t>(mel_mask.size()) != batch_ * frames_) {
            throw std::runtime_error("MuScriptor condition mask size mismatch");
        }
        if (static_cast<int64_t>(instrument_ids.size()) != batch_ * instrument_steps_) {
            throw std::runtime_error("MuScriptor condition instrument id count mismatch");
        }
        if (static_cast<int64_t>(dataset_ids.size()) != batch_) {
            throw std::runtime_error("MuScriptor condition dataset id count mismatch");
        }
        ggml_backend_tensor_set(log_mel_, log_mel.data(), 0, log_mel.size() * sizeof(float));
        ggml_backend_tensor_set(mel_mask_, mel_mask.data(), 0, mel_mask.size() * sizeof(int32_t));
        ggml_backend_tensor_set(instrument_ids_, instrument_ids.data(), 0, instrument_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(dataset_id_, dataset_ids.data(), 0, dataset_ids.size() * sizeof(int32_t));
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        const auto start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        engine::debug::timing_log_scalar("muscriptor.condition.graph.compute_ms", engine::debug::elapsed_ms(start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MuScriptor condition graph compute failed");
        }
        MuScriptorConditioning out;
        out.batch = batch_;
        out.steps = frames_ + instrument_steps_ + 1;
        out.dim = dim;
        out.values.resize(static_cast<size_t>(out.batch * out.steps * dim));
        const auto read_start = Clock::now();
        ggml_backend_tensor_get(output_, out.values.data(), 0, out.values.size() * sizeof(float));
        engine::debug::timing_log_scalar("muscriptor.condition.output_read_ms", engine::debug::elapsed_ms(read_start, Clock::now()));
        return out;
    }

private:
    core::ExecutionContext & execution_;
    std::shared_ptr<MuScriptorWeights> weights_;
    int64_t frames_ = 0;
    int64_t instrument_steps_ = 0;
    int64_t batch_ = 1;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * log_mel_ = nullptr;
    ggml_tensor * mel_mask_ = nullptr;
    ggml_tensor * instrument_ids_ = nullptr;
    ggml_tensor * dataset_id_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

class MuScriptorDecoderRuntime::PrefillGraph {
public:
    struct Output {
        std::vector<float> logits;
        int64_t current_end = 0;
    };

    PrefillGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<MuScriptorWeights> weights,
        const MuScriptorConfig & config,
        int64_t condition_steps,
        int64_t token_steps,
        int64_t batch,
        const std::vector<core::TensorValue> & target_keys,
        const std::vector<core::TensorValue> & target_values,
        MuScriptorPerfMode perf_mode,
        size_t arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          condition_steps_(condition_steps),
          token_steps_(token_steps),
          total_steps_(condition_steps + token_steps),
          batch_(batch),
          target_key0_(target_keys.empty() ? nullptr : target_keys.front().tensor),
          perf_mode_(perf_mode) {
        const auto start = Clock::now();
        if (target_keys.size() != weights_->layers.size() || target_values.size() != weights_->layers.size()) {
            throw std::runtime_error("MuScriptor prefill target cache layer count mismatch");
        }
        ggml_init_params params{arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MuScriptor prefill graph context");
        }
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MuScriptor prefill input context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "muscriptor.prefill", execution_.backend_type()};
        condition_ = ggml_new_tensor_3d(input_ctx_.get(), GGML_TYPE_F32, config.dim, condition_steps_, batch_);
        ggml_set_input(condition_);
        token_ids_ = ggml_new_tensor_2d(input_ctx_.get(), GGML_TYPE_I32, token_steps_, batch_);
        ggml_set_input(token_ids_);
        positions_ = ggml_new_tensor_3d(input_ctx_.get(), GGML_TYPE_F32, config.dim, total_steps_, batch_);
        ggml_set_input(positions_);
        auto condition = core::wrap_tensor(condition_, core::TensorShape::from_dims({batch_, condition_steps_, config.dim}), GGML_TYPE_F32);
        auto tokens = modules::EmbeddingModule({config.card + 1, config.dim}).build(
            ctx,
            core::wrap_tensor(token_ids_, core::TensorShape::from_dims({batch_, token_steps_}), GGML_TYPE_I32),
            weights_->token_embedding);
        auto x = modules::ConcatModule({1}).build(ctx, condition, tokens);
        x = modules::AddModule().build(ctx, x, core::wrap_tensor(positions_, core::TensorShape::from_dims({batch_, total_steps_, config.dim}), GGML_TYPE_F32));
        graph_ = ggml_new_graph_custom(ctx_.get(), kGraphNodes, false);
        for (size_t layer_index = 0; layer_index < weights_->layers.size(); ++layer_index) {
            core::TensorValue key;
            core::TensorValue value;
            x = build_transformer_layer(
                ctx,
                graph_,
                x,
                weights_->layers[layer_index],
                config,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                perf_mode_,
                &key,
                &value);
            auto key_dest = view_batched_kv_cache_steps(
                ctx,
                target_keys[layer_index],
                0,
                total_steps_,
                config,
                "MuScriptor prefill key target",
                target_keys[layer_index].type);
            auto value_dest = view_batched_kv_cache_steps(
                ctx,
                target_values[layer_index],
                0,
                total_steps_,
                config,
                "MuScriptor prefill value target",
                target_values[layer_index].type);
            ggml_build_forward_expand(graph_, ggml_cpy(ctx_.get(), key.tensor, key_dest.tensor));
            ggml_build_forward_expand(graph_, ggml_cpy(ctx_.get(), value.tensor, value_dest.tensor));
        }
        x = modules::LayerNormModule({config.dim, 1.0e-5F, true, true}).build(ctx, x, weights_->output_norm);
        auto tail = modules::SliceModule({1, total_steps_ - 1, 1}).build(ctx, x);
        logits_ = modules::LinearModule({config.dim, config.card, false}).build(ctx, tail, weights_->output_head).tensor;
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, logits_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate MuScriptor prefill input tensors");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MuScriptor prefill graph");
        }
        const auto one_row_pos = sinusoidal_position_embedding(total_steps_, config.dim);
        std::vector<float> pos(static_cast<size_t>(batch_ * total_steps_ * config.dim));
        for (int64_t batch = 0; batch < batch_; ++batch) {
            std::copy(
                one_row_pos.begin(),
                one_row_pos.end(),
                pos.begin() + static_cast<ptrdiff_t>(batch * total_steps_ * config.dim));
        }
        ggml_backend_tensor_set(positions_, pos.data(), 0, pos.size() * sizeof(float));
        engine::debug::timing_log_scalar("muscriptor.prefill.graph.build_ms", engine::debug::elapsed_ms(start, Clock::now()));
    }

    ~PrefillGraph() {
        engine::core::release_backend_graph_resources(execution_.backend(), graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
        }
    }

    bool matches(const MuScriptorWeights & weights, int64_t condition_steps, int64_t token_steps, int64_t batch, ggml_tensor * target_key0) const {
        return weights_.get() == &weights &&
               condition_steps_ == condition_steps &&
               token_steps_ == token_steps &&
               batch_ == batch &&
               target_key0_ == target_key0;
    }

    Output run(const MuScriptorConditioning & condition, const std::vector<int32_t> & token_ids, const MuScriptorConfig & config) {
        if (condition.batch != batch_ || condition.steps != condition_steps_ || condition.dim != config.dim) {
            throw std::runtime_error("MuScriptor prefill condition shape mismatch");
        }
        if (static_cast<int64_t>(token_ids.size()) != batch_ * token_steps_) {
            throw std::runtime_error("MuScriptor prefill token count mismatch");
        }
        ggml_backend_tensor_set(condition_, condition.values.data(), 0, condition.values.size() * sizeof(float));
        ggml_backend_tensor_set(token_ids_, token_ids.data(), 0, token_ids.size() * sizeof(int32_t));
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        const auto start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        engine::debug::timing_log_scalar("muscriptor.prefill.graph.compute_ms", engine::debug::elapsed_ms(start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MuScriptor prefill graph compute failed");
        }
        Output out;
        out.logits.resize(static_cast<size_t>(batch_ * config.card));
        const auto logits_read_start = Clock::now();
        ggml_backend_tensor_get(logits_, out.logits.data(), 0, out.logits.size() * sizeof(float));
        engine::debug::timing_log_scalar("muscriptor.prefill.logits_read_ms", engine::debug::elapsed_ms(logits_read_start, Clock::now()));
        out.current_end = total_steps_;
        engine::debug::timing_log_scalar("muscriptor.prefill.kv_read_ms", 0.0);
        return out;
    }

private:
    core::ExecutionContext & execution_;
    std::shared_ptr<MuScriptorWeights> weights_;
    int64_t condition_steps_ = 0;
    int64_t token_steps_ = 0;
    int64_t total_steps_ = 0;
    int64_t batch_ = 1;
    ggml_tensor * target_key0_ = nullptr;
    MuScriptorPerfMode perf_mode_ = MuScriptorPerfMode::Exact;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    ggml_tensor * condition_ = nullptr;
    ggml_tensor * token_ids_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
};

class MuScriptorDecoderRuntime::DecodeGraph {
public:
    DecodeGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<MuScriptorWeights> weights,
        MuScriptorDecodeCacheStorage & cache_storage,
        const MuScriptorConfig & config,
        int64_t attention_steps,
        MuScriptorPerfMode perf_mode,
        size_t arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          cache_storage_(cache_storage),
          cache_steps_(cache_storage.cache_steps()),
          attention_steps_(attention_steps),
          batch_(cache_storage.batch()),
          shared_attention_mask_(perf_mode == MuScriptorPerfMode::FlashAttention && execution_.backend_type() == core::BackendType::Cuda),
          perf_mode_(perf_mode),
          position_values_(static_cast<size_t>(cache_storage.batch() * cache_storage.cache_steps() * config.dim), 0.0F),
          attention_mask_values_(static_cast<size_t>((shared_attention_mask_ ? 1 : batch_) * attention_steps_), ggml_fp32_to_fp16(-INFINITY)) {
        const auto start = Clock::now();
        ggml_init_params params{arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MuScriptor decode graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "muscriptor.decode", execution_.backend_type()};
        token_id_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, batch_);
        ggml_set_input(token_id_);
        position_ = ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, config.dim, 1, batch_);
        ggml_set_input(position_);
        cache_slot_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, batch_);
        ggml_set_input(cache_slot_);
        const int64_t mask_batch = shared_attention_mask_ ? 1 : batch_;
        attention_mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, attention_steps_, 1, 1, mask_batch);
        ggml_set_input(attention_mask_);
        auto x = modules::EmbeddingModule({config.card + 1, config.dim}).build(
            ctx,
            core::wrap_tensor(token_id_, core::TensorShape::from_dims({batch_}), GGML_TYPE_I32),
            weights_->token_embedding);
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({batch_, 1, config.dim}));
        x = modules::AddModule().build(ctx, x, core::wrap_tensor(position_, core::TensorShape::from_dims({batch_, 1, config.dim}), GGML_TYPE_F32));
        graph_ = ggml_new_graph_custom(ctx_.get(), kGraphNodes, false);
        auto mask = core::wrap_tensor(attention_mask_, core::TensorShape::from_dims({mask_batch, 1, 1, attention_steps_}), GGML_TYPE_F16);
        for (int64_t layer = 0; layer < config.num_layers; ++layer) {
            x = build_transformer_layer(
                ctx,
                graph_,
                x,
                weights_->layers[static_cast<size_t>(layer)],
                config,
                cache_storage_.key_tensor(static_cast<size_t>(layer)),
                cache_storage_.value_tensor(static_cast<size_t>(layer)),
                core::wrap_tensor(cache_slot_, core::TensorShape::from_dims({batch_}), GGML_TYPE_I32),
                attention_steps_,
                mask,
                perf_mode_,
                nullptr,
                nullptr);
        }
        x = modules::LayerNormModule({config.dim, 1.0e-5F, true, true}).build(ctx, x, weights_->output_norm);
        logits_ = modules::LinearModule({config.dim, config.card, false}).build(ctx, x, weights_->output_head).tensor;
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, logits_);
        const auto one_row_pos = sinusoidal_position_embedding(cache_steps_, config.dim);
        for (int64_t batch = 0; batch < batch_; ++batch) {
            std::copy(
                one_row_pos.begin(),
                one_row_pos.end(),
                position_values_.begin() + static_cast<ptrdiff_t>(batch * cache_steps_ * config.dim));
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MuScriptor decode graph");
        }
        cache_slot_values_.assign(static_cast<size_t>(batch_), 0);
        position_upload_values_.assign(static_cast<size_t>(batch_ * config.dim), 0.0F);
        engine::debug::trace_log_scalar("muscriptor.decode.cache_steps", cache_steps_);
        engine::debug::trace_log_scalar("muscriptor.decode.attention_steps", attention_steps_);
        engine::debug::timing_log_scalar("muscriptor.decode.graph.build_ms", engine::debug::elapsed_ms(start, Clock::now()));
    }

    ~DecodeGraph() {
        engine::core::release_backend_graph_resources(execution_.backend(), graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool can_run(
        const MuScriptorWeights & weights,
        const MuScriptorDecodeCacheStorage & cache_storage,
        int64_t attention_steps) const {
        return weights_.get() == &weights && &cache_storage_ == &cache_storage && attention_steps_ == attention_steps;
    }

    void adopt_direct_prefill(int64_t steps) {
        cache_storage_.adopt_direct_prefix(steps);
        std::fill(attention_mask_values_.begin(), attention_mask_values_.end(), ggml_fp32_to_fp16(-INFINITY));
    }

    int64_t cache_steps() const noexcept {
        return cache_steps_;
    }

    int64_t attention_steps() const noexcept {
        return attention_steps_;
    }

    void reset_profile() {
        profile_runs_ = 0;
        profile_input_upload_ms_ = 0.0;
        profile_mask_upload_ms_ = 0.0;
        profile_graph_compute_ms_ = 0.0;
        profile_logits_read_ms_ = 0.0;
        cache_storage_.reset_reorder_profile();
    }

    void log_profile() const {
        engine::debug::trace_log_scalar("muscriptor.decode.runs", profile_runs_);
        engine::debug::timing_log_scalar("muscriptor.decode.input_upload_ms", profile_input_upload_ms_);
        engine::debug::timing_log_scalar("muscriptor.decode.mask_upload_ms", profile_mask_upload_ms_);
        engine::debug::timing_log_scalar("muscriptor.decode.graph.compute_ms", profile_graph_compute_ms_);
        engine::debug::timing_log_scalar("muscriptor.decode.logits_read_ms", profile_logits_read_ms_);
        engine::debug::timing_log_scalar("muscriptor.decode.cache_reorder_ms", cache_storage_.reorder_ms());
        engine::debug::trace_log_scalar("muscriptor.decode.cache_reorder_calls", cache_storage_.reorder_calls());
        engine::debug::trace_log_scalar(
            "muscriptor.decode.cache_reorder_changed_calls",
            cache_storage_.reorder_changed_calls());
        engine::debug::trace_log_scalar(
            "muscriptor.decode.cache_reorder_changed_rows",
            cache_storage_.reorder_changed_rows());
    }

    std::vector<float> run_batch(const std::vector<int32_t> & tokens, const MuScriptorConfig & config) {
        if (static_cast<int64_t>(tokens.size()) != batch_) {
            throw std::runtime_error("MuScriptor decode token batch mismatch");
        }
        if (cache_storage_.valid_steps() >= cache_steps_) {
            throw std::runtime_error("MuScriptor decode cache exhausted");
        }
        if (cache_storage_.valid_steps() >= attention_steps_) {
            throw std::runtime_error("MuScriptor decode attention graph is too small for current cache position");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(token_id_, tokens.data(), 0, tokens.size() * sizeof(int32_t));
        for (int64_t batch = 0; batch < batch_; ++batch) {
            cache_slot_values_[static_cast<size_t>(batch)] = static_cast<int32_t>(batch * cache_steps_ + cache_storage_.valid_steps());
        }
        ggml_backend_tensor_set(cache_slot_, cache_slot_values_.data(), 0, cache_slot_values_.size() * sizeof(int32_t));
        if (cache_storage_.current_end() < 0 || cache_storage_.current_end() >= cache_steps_) {
            throw std::runtime_error("MuScriptor decode position exceeds cache capacity");
        }
        for (int64_t batch = 0; batch < batch_; ++batch) {
            const auto source = position_values_.begin() +
                static_cast<ptrdiff_t>((batch * cache_steps_ + cache_storage_.current_end()) * config.dim);
            std::copy(source, source + config.dim, position_upload_values_.begin() + static_cast<ptrdiff_t>(batch * config.dim));
        }
        ggml_backend_tensor_set(
            position_,
            position_upload_values_.data(),
            0,
            position_upload_values_.size() * sizeof(float));
        profile_input_upload_ms_ += engine::debug::elapsed_ms(timing_start, Clock::now());
        timing_start = Clock::now();
        if (shared_attention_mask_) {
            for (int64_t step = 0; step <= cache_storage_.valid_steps(); ++step) {
                attention_mask_values_[static_cast<size_t>(step)] = ggml_fp32_to_fp16(0.0F);
            }
        } else {
            std::fill(attention_mask_values_.begin(), attention_mask_values_.end(), ggml_fp32_to_fp16(-INFINITY));
            for (int64_t batch = 0; batch < batch_; ++batch) {
                const size_t base = static_cast<size_t>(batch * attention_steps_);
                for (int64_t step = 0; step <= cache_storage_.valid_steps(); ++step) {
                    attention_mask_values_[base + static_cast<size_t>(step)] = ggml_fp32_to_fp16(0.0F);
                }
            }
        }
        ggml_backend_tensor_set(attention_mask_, attention_mask_values_.data(), 0, attention_mask_values_.size() * sizeof(ggml_fp16_t));
        profile_mask_upload_ms_ += engine::debug::elapsed_ms(timing_start, Clock::now());
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        profile_graph_compute_ms_ += engine::debug::elapsed_ms(timing_start, Clock::now());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MuScriptor decode graph compute failed");
        }
        cache_storage_.advance_after_direct_append(1);
        std::vector<float> logits(static_cast<size_t>(batch_ * config.card));
        timing_start = Clock::now();
        ggml_backend_tensor_get(logits_, logits.data(), 0, logits.size() * sizeof(float));
        profile_logits_read_ms_ += engine::debug::elapsed_ms(timing_start, Clock::now());
        ++profile_runs_;
        return logits;
    }

private:
    core::ExecutionContext & execution_;
    std::shared_ptr<MuScriptorWeights> weights_;
    MuScriptorDecodeCacheStorage & cache_storage_;
    int64_t cache_steps_ = 0;
    int64_t attention_steps_ = 0;
    int64_t batch_ = 1;
    bool shared_attention_mask_ = false;
    MuScriptorPerfMode perf_mode_ = MuScriptorPerfMode::Exact;
    std::vector<float> position_values_;
    std::vector<ggml_fp16_t> attention_mask_values_;
    std::vector<int32_t> cache_slot_values_;
    std::vector<float> position_upload_values_;
    int64_t profile_runs_ = 0;
    double profile_input_upload_ms_ = 0.0;
    double profile_mask_upload_ms_ = 0.0;
    double profile_graph_compute_ms_ = 0.0;
    double profile_logits_read_ms_ = 0.0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * token_id_ = nullptr;
    ggml_tensor * position_ = nullptr;
    ggml_tensor * cache_slot_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

MuScriptorDecoderRuntime::MuScriptorDecoderRuntime(
    std::shared_ptr<const MuScriptorAssets> assets,
    core::ExecutionContext & execution,
    MuScriptorDecoderOptions options)
    : assets_(std::move(assets)),
      execution_(execution),
      weights_(std::make_shared<MuScriptorWeights>(load_weights(*assets_, execution_, options))),
      options_(options) {
    assets_->model_weights->release_storage();
}

MuScriptorDecoderRuntime::~MuScriptorDecoderRuntime() = default;

MuScriptorConditioning MuScriptorDecoderRuntime::condition(
    const std::vector<float> & log_mel,
    const std::vector<int32_t> & mel_mask,
    const std::vector<int32_t> & instrument_ids,
    int32_t dataset_id) {
    return condition_batch(log_mel, mel_mask, instrument_ids, std::vector<int32_t>{dataset_id}, 1);
}

MuScriptorConditioning MuScriptorDecoderRuntime::condition_batch(
    const std::vector<float> & log_mel,
    const std::vector<int32_t> & mel_mask,
    const std::vector<int32_t> & instrument_ids,
    const std::vector<int32_t> & dataset_ids,
    int64_t batch) {
    const int64_t frames = static_cast<int64_t>(log_mel.size()) / assets_->config.n_mels;
    if (batch <= 0 || frames % batch != 0) {
        throw std::runtime_error("MuScriptor condition batch shape is invalid");
    }
    const int64_t per_sample_frames = frames / batch;
    const int64_t instrument_steps = static_cast<int64_t>(instrument_ids.size()) / batch;
    if (instrument_steps <= 0 || static_cast<int64_t>(instrument_ids.size()) != instrument_steps * batch) {
        throw std::runtime_error("MuScriptor condition instrument batch shape is invalid");
    }
    if (condition_graph_ == nullptr || !condition_graph_->matches(*weights_, per_sample_frames, instrument_steps, batch)) {
        condition_graph_ = std::make_unique<ConditionGraph>(
            execution_,
            weights_,
            assets_->config,
            per_sample_frames,
            instrument_steps,
            batch,
            options_.condition_graph_arena_bytes);
    } else {
        engine::debug::timing_log_scalar("muscriptor.condition.graph.build_ms", 0.0);
    }
    return condition_graph_->run(log_mel, mel_mask, instrument_ids, dataset_ids, assets_->config.dim);
}

MuScriptorGeneratedChunk MuScriptorDecoderRuntime::generate_greedy(
    const MuScriptorConditioning & conditioning,
    const std::vector<int32_t> & prompt,
    int64_t max_tokens,
    int32_t eos_id,
    const std::vector<int32_t> & forbidden_tokens) {
    MuScriptorGenerationOptions options;
    return generate(conditioning, nullptr, prompt, max_tokens, eos_id, forbidden_tokens, options);
}

MuScriptorGeneratedChunk MuScriptorDecoderRuntime::generate(
    const MuScriptorConditioning & conditioning,
    const MuScriptorConditioning * null_conditioning,
    const std::vector<int32_t> & prompt,
    int64_t max_tokens,
    int32_t eos_id,
    const std::vector<int32_t> & forbidden_tokens,
    const MuScriptorGenerationOptions & options) {
    MuScriptorConditioning active = conditioning;
    if (options.guidance_scale != 1.0F) {
        if (null_conditioning == nullptr ||
            null_conditioning->batch != conditioning.batch ||
            null_conditioning->steps != conditioning.steps ||
            null_conditioning->dim != conditioning.dim) {
            throw std::runtime_error("MuScriptor CFG generation requires matching null conditioning");
        }
        active.batch = conditioning.batch * 2;
        active.values.resize(static_cast<size_t>(active.batch * active.steps * active.dim));
        std::copy(conditioning.values.begin(), conditioning.values.end(), active.values.begin());
        std::copy(
            null_conditioning->values.begin(),
            null_conditioning->values.end(),
            active.values.begin() + static_cast<ptrdiff_t>(conditioning.values.size()));
    }
    return generate_batch(active, std::vector<std::vector<int32_t>>{prompt}, max_tokens, eos_id, forbidden_tokens, options).front();
}

std::vector<MuScriptorGeneratedChunk> MuScriptorDecoderRuntime::generate_batch(
    const MuScriptorConditioning & conditioning,
    const std::vector<std::vector<int32_t>> & prompts,
    int64_t max_tokens,
    int32_t eos_id,
    const std::vector<int32_t> & forbidden_tokens,
    const MuScriptorGenerationOptions & options) {
    if (max_tokens <= 0) {
        throw std::runtime_error("MuScriptor max_tokens must be positive");
    }
    if (options.num_beams <= 0) {
        throw std::runtime_error("MuScriptor num_beams must be positive");
    }
    if (!(options.temperature >= 0.0F) || !std::isfinite(options.temperature)) {
        throw std::runtime_error("MuScriptor temperature must be finite and non-negative");
    }
    if (!std::isfinite(options.guidance_scale)) {
        throw std::runtime_error("MuScriptor guidance_scale must be finite");
    }
    const bool cfg_enabled = options.guidance_scale != 1.0F;
    if (conditioning.batch <= 0 || conditioning.steps <= 0 || conditioning.dim != assets_->config.dim) {
        throw std::runtime_error("MuScriptor generation conditioning shape is invalid");
    }
    if (cfg_enabled && conditioning.batch % 2 != 0) {
        throw std::runtime_error("MuScriptor CFG batch must contain conditional and null rows");
    }
    int64_t batch = cfg_enabled ? conditioning.batch / 2 : conditioning.batch;
    if (options.num_beams > 1) {
        if (batch != 1 || cfg_enabled || options.use_sampling) {
            throw std::runtime_error("MuScriptor beam search supports one non-CFG non-sampling request per graph batch");
        }
        batch = options.num_beams;
    }
    if (static_cast<int64_t>(prompts.size()) != (cfg_enabled ? conditioning.batch / 2 : conditioning.batch)) {
        throw std::runtime_error("MuScriptor generation prompt batch mismatch");
    }
    const int64_t prompt_steps = prompts.empty() ? 0 : static_cast<int64_t>(prompts.front().size());
    for (const auto & prompt : prompts) {
        if (static_cast<int64_t>(prompt.size()) != prompt_steps) {
            throw std::runtime_error("MuScriptor batched generation requires equal prompt lengths");
        }
    }
    const int64_t token_steps = prompt_steps + 1;
    std::vector<int32_t> input(static_cast<size_t>(batch * token_steps), assets_->config.card);
    for (int64_t row = 0; row < batch; ++row) {
        const auto & prompt = prompts[static_cast<size_t>(options.num_beams > 1 ? 0 : row)];
        for (int64_t i = 0; i < prompt_steps; ++i) {
            input[static_cast<size_t>(row * token_steps + i + 1)] = prompt[static_cast<size_t>(i)];
        }
    }
    if (cfg_enabled) {
        input.resize(static_cast<size_t>(conditioning.batch * token_steps));
        for (int64_t row = 0; row < batch; ++row) {
            std::copy_n(
                input.begin() + static_cast<ptrdiff_t>(row * token_steps),
                static_cast<size_t>(token_steps),
                input.begin() + static_cast<ptrdiff_t>((row + batch) * token_steps));
        }
        batch = conditioning.batch;
    }
    MuScriptorConditioning active_conditioning = conditioning;
    if (options.num_beams > 1) {
        active_conditioning.batch = batch;
        active_conditioning.values.resize(static_cast<size_t>(batch * conditioning.steps * conditioning.dim));
        for (int64_t row = 0; row < batch; ++row) {
            std::copy(
                conditioning.values.begin(),
                conditioning.values.end(),
                active_conditioning.values.begin() + static_cast<ptrdiff_t>(row * conditioning.steps * conditioning.dim));
        }
    }

    std::vector<uint8_t> forbidden_mask(static_cast<size_t>(kReservedVocabStart), 0);
    for (const int32_t token : forbidden_tokens) {
        if (token >= 0 && token < kReservedVocabStart) {
            forbidden_mask[static_cast<size_t>(token)] = 1;
        }
    }

    const int64_t logical_batch = cfg_enabled ? conditioning.batch / 2 : (options.num_beams > 1 ? 1 : batch);
    const int64_t max_cache_steps = active_conditioning.steps + max_tokens;
    const int64_t max_prefill_steps = active_conditioning.steps + token_steps;
    if (max_prefill_steps > max_cache_steps) {
        throw std::runtime_error("MuScriptor prompt consumes the decode cache");
    }
    const bool enable_reorder_scratch = options.num_beams > 1;
    const ggml_type decode_cache_type =
        options_.perf_mode == MuScriptorPerfMode::FlashAttention &&
            execution_.backend_type() == core::BackendType::Cuda
        ? GGML_TYPE_F16
        : GGML_TYPE_F32;
    if (decode_cache_ == nullptr || !decode_cache_->matches(
            max_cache_steps,
            batch,
            enable_reorder_scratch,
            decode_cache_type)) {
        decode_graph_.reset();
        prefill_graph_.reset();
        decode_cache_ = std::make_unique<MuScriptorDecodeCacheStorage>(
            execution_,
            assets_->config,
            max_cache_steps,
            batch,
            enable_reorder_scratch,
            decode_cache_type);
    }
    auto ensure_decode_graph = [&](int64_t required_attention_steps) {
        const int64_t attention_steps = decode_attention_bucket(required_attention_steps, max_cache_steps);
        if (decode_graph_ == nullptr || !decode_graph_->can_run(*weights_, *decode_cache_, attention_steps)) {
            decode_graph_ = std::make_unique<DecodeGraph>(
                execution_,
                weights_,
                *decode_cache_,
                assets_->config,
                attention_steps,
                options_.perf_mode,
                options_.decode_graph_arena_bytes);
        }
    };
    ensure_decode_graph(max_prefill_steps);
    auto run_decode = [&](const std::vector<int32_t> & next_tokens) {
        ensure_decode_graph(decode_cache_->valid_steps() + 1);
        return decode_graph_->run_batch(next_tokens, assets_->config);
    };
    auto run_prefill = [&]() {
        std::vector<core::TensorValue> target_keys;
        std::vector<core::TensorValue> target_values;
        target_keys.reserve(decode_cache_->layer_count());
        target_values.reserve(decode_cache_->layer_count());
        for (size_t layer = 0; layer < decode_cache_->layer_count(); ++layer) {
            target_keys.push_back(decode_cache_->key_tensor(layer));
            target_values.push_back(decode_cache_->value_tensor(layer));
        }
        ggml_tensor * target_key0 = target_keys.empty() ? nullptr : target_keys.front().tensor;
        if (prefill_graph_ == nullptr ||
            !prefill_graph_->matches(*weights_, active_conditioning.steps, token_steps, batch, target_key0)) {
            prefill_graph_ = std::make_unique<PrefillGraph>(
                execution_,
                weights_,
                assets_->config,
                active_conditioning.steps,
                token_steps,
                batch,
                target_keys,
                target_values,
                options_.perf_mode,
                options_.prefill_graph_arena_bytes);
        } else {
            engine::debug::timing_log_scalar("muscriptor.prefill.graph.build_ms", 0.0);
        }
        auto prefill = prefill_graph_->run(active_conditioning, input, assets_->config);
        decode_graph_->adopt_direct_prefill(prefill.current_end);
        return prefill;
    };

    sampling::HfSamplingOptions sampling_options;
    sampling_options.do_sample = true;
    sampling_options.temperature = options.temperature;
    sampling_options.top_k = 0;
    sampling_options.top_p = 1.0F;
    sampling_options.repetition_penalty = 1.0F;
    sampling::HfSampler sampler;
    auto choose_token = [&](const std::vector<float> & scores,
                            sampling::HfSamplerScratch & scratch,
                            std::mt19937 & rng,
                            sampling::HfTorchSamplingState & torch_state) {
        if (!options.use_sampling || options.temperature <= 0.0F) {
            return sampling::HfLogitsProcessor::argmax(scores.data(), scores.size(), "MuScriptor greedy");
        }
        const int32_t token =
            sampler.sample(scores, {}, sampling_options, scratch, rng, &torch_state, "MuScriptor sampling");
        ++torch_state.call_index;
        return token;
    };

    const auto start = Clock::now();
    std::vector<MuScriptorGeneratedChunk> out(static_cast<size_t>(logical_batch));
    for (int64_t row = 0; row < logical_batch; ++row) {
        out[static_cast<size_t>(row)].tokens = prompts[static_cast<size_t>(row)];
    }
    auto prefill = run_prefill();
    std::vector<float> logits = std::move(prefill.logits);
    decode_graph_->reset_profile();

    if (options.num_beams > 1) {
        double beam_select_ms = 0.0;
        struct Beam {
            std::vector<int32_t> tokens;
            std::vector<float> logits;
            float score = 0.0F;
            bool ended = false;
            int64_t row = 0;
        };
        std::vector<Beam> beams;
        beams.reserve(static_cast<size_t>(options.num_beams));
        std::vector<float> beam_scores_scratch;
        beam_scores_scratch.reserve(static_cast<size_t>(kReservedVocabStart));
        for (int64_t row = 0; row < options.num_beams; ++row) {
            std::vector<float> row_logits(logits.begin() + static_cast<ptrdiff_t>(row * assets_->config.card),
                                          logits.begin() + static_cast<ptrdiff_t>((row + 1) * assets_->config.card));
            beams.push_back({prompts.front(), std::move(row_logits), 0.0F, false, row});
        }
        for (int64_t offset = prompt_steps; offset < max_tokens; ++offset) {
            bool all_done = true;
            for (const auto & beam : beams) {
                all_done = all_done && beam.ended;
            }
            if (all_done) {
                break;
            }
            struct Candidate {
                Beam beam;
                float normalized = -std::numeric_limits<float>::infinity();
            };
            const auto select_start = Clock::now();
            std::vector<Candidate> candidates;
            for (const auto & beam : beams) {
                if (beam.ended) {
                    candidates.push_back({beam, beam.score});
                    continue;
                }
                fill_masked_logits(
                    beam_scores_scratch,
                    beam.logits.data(),
                    static_cast<int64_t>(beam.logits.size()),
                    forbidden_mask);
                const auto next_tokens = top_k_scores(beam_scores_scratch, options.num_beams);
                for (const auto & next : next_tokens) {
                    Beam candidate;
                    candidate.tokens = beam.tokens;
                    candidate.score = beam.score + next.score;
                    candidate.ended = next.token == eos_id;
                    if (!candidate.ended) {
                        candidate.tokens.push_back(next.token);
                    } else {
                        candidate.logits = beam.logits;
                    }
                    candidate.row = beam.row;
                    const int64_t length = std::max<int64_t>(1, offset + 1);
                    const float length_penalty = 1.0F / std::pow(static_cast<float>(length), 0.75F);
                    candidates.push_back({std::move(candidate), candidate.score * length_penalty});
                }
            }
            const auto keep = static_cast<size_t>(std::min<int64_t>(options.num_beams, static_cast<int64_t>(candidates.size())));
            std::partial_sort(
                candidates.begin(),
                candidates.begin() + static_cast<ptrdiff_t>(keep),
                candidates.end(),
                [](const Candidate & lhs, const Candidate & rhs) {
                    return lhs.normalized > rhs.normalized;
                });
            std::vector<Beam> next_beams;
            next_beams.reserve(keep);
            std::vector<int32_t> next_tokens(static_cast<size_t>(options.num_beams), eos_id);
            std::vector<uint8_t> row_taken(static_cast<size_t>(options.num_beams), 0);
            std::vector<int64_t> copy_parent_rows;
            std::vector<int64_t> copy_child_rows;
            copy_parent_rows.reserve(keep);
            copy_child_rows.reserve(keep);
            for (size_t i = 0; i < keep; ++i) {
                auto beam = std::move(candidates[i].beam);
                const int64_t parent_row = beam.row;
                int64_t assigned_row = -1;
                if (row_taken[static_cast<size_t>(parent_row)] == 0) {
                    assigned_row = parent_row;
                } else {
                    for (int64_t row = 0; row < options.num_beams; ++row) {
                        if (row_taken[static_cast<size_t>(row)] == 0) {
                            assigned_row = row;
                            break;
                        }
                    }
                }
                if (assigned_row < 0) {
                    throw std::runtime_error("MuScriptor beam search could not assign a cache row");
                }
                row_taken[static_cast<size_t>(assigned_row)] = 1;
                if (assigned_row != parent_row) {
                    copy_parent_rows.push_back(parent_row);
                    copy_child_rows.push_back(assigned_row);
                }
                beam.row = assigned_row;
                if (!beam.ended) {
                    next_tokens[static_cast<size_t>(assigned_row)] = beam.tokens.back();
                }
                next_beams.push_back(std::move(beam));
            }
            beam_select_ms += engine::debug::elapsed_ms(select_start, Clock::now());
            decode_cache_->copy_batch_prefixes(copy_parent_rows, copy_child_rows, decode_cache_->valid_steps());
            beams = std::move(next_beams);
            if (offset + 1 >= max_tokens) {
                break;
            }
            const auto next_logits = run_decode(next_tokens);
            for (size_t i = 0; i < beams.size(); ++i) {
                const int64_t row = beams[i].row;
                beams[i].logits.assign(
                    next_logits.begin() + static_cast<ptrdiff_t>(row * assets_->config.card),
                    next_logits.begin() + static_cast<ptrdiff_t>((row + 1) * assets_->config.card));
            }
        }
        if (beams.empty()) {
            throw std::runtime_error("MuScriptor beam search produced no beams");
        }
        const auto best = std::max_element(
            beams.begin(),
            beams.end(),
            [](const Beam & lhs, const Beam & rhs) {
                return lhs.score < rhs.score;
            });
        out.front().tokens = best->tokens;
        out.front().emitted_eos = best->ended;
        engine::debug::timing_log_scalar("muscriptor.decode.beam_select_ms", beam_select_ms);
        decode_graph_->log_profile();
        engine::debug::timing_log_scalar("muscriptor.decode.total_ms", engine::debug::elapsed_ms(start, Clock::now()));
        return out;
    }

    std::mt19937 rng(static_cast<uint32_t>(options.seed));
    sampling::HfSamplerScratch sampler_scratch;
    sampler_scratch.reserve_vocab(static_cast<size_t>(kReservedVocabStart));
    const auto sampling_policy = sampling::resolve_torch_cuda_sampling_policy(
        execution_.backend_type(),
        execution_.config().device,
        "muscriptor.decode.cuda_sampling_policy",
        "MuScriptor",
        sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault);
    sampling::HfTorchSamplingState torch_state;
    torch_state.policy = &sampling_policy;
    torch_state.seed = options.seed;
    torch_state.call_index = 0;

    std::vector<uint8_t> done(static_cast<size_t>(logical_batch), 0);
    std::vector<float> row_scores;
    row_scores.reserve(static_cast<size_t>(kReservedVocabStart));
    std::vector<int32_t> next_tokens(static_cast<size_t>(batch), eos_id);
    double token_select_ms = 0.0;
    for (int64_t step = prompt_steps; step < max_tokens; ++step) {
        bool all_done = true;
        for (const uint8_t value : done) {
            all_done = all_done && value != 0;
        }
        if (all_done) {
            break;
        }
        std::fill(next_tokens.begin(), next_tokens.end(), eos_id);
        for (int64_t row = 0; row < logical_batch; ++row) {
            const auto select_start = Clock::now();
            const auto cond_begin = logits.data() + static_cast<ptrdiff_t>(row * assets_->config.card);
            if (cfg_enabled) {
                const auto uncond_begin = logits.data() + static_cast<ptrdiff_t>((row + logical_batch) * assets_->config.card);
                fill_guided_logits(
                    row_scores,
                    cond_begin,
                    uncond_begin,
                    assets_->config.card,
                    options.guidance_scale,
                    forbidden_mask);
            } else {
                fill_masked_logits(row_scores, cond_begin, assets_->config.card, forbidden_mask);
            }
            const int32_t token = choose_token(row_scores, sampler_scratch, rng, torch_state);
            token_select_ms += engine::debug::elapsed_ms(select_start, Clock::now());
            if (done[static_cast<size_t>(row)] == 0 && token == eos_id) {
                done[static_cast<size_t>(row)] = 1;
                out[static_cast<size_t>(row)].emitted_eos = true;
            } else if (done[static_cast<size_t>(row)] == 0) {
                out[static_cast<size_t>(row)].tokens.push_back(token);
            }
            next_tokens[static_cast<size_t>(row)] = token;
            if (cfg_enabled) {
                next_tokens[static_cast<size_t>(row + logical_batch)] = token;
            }
        }
        if (step + 1 < max_tokens) {
            logits = run_decode(next_tokens);
        }
    }
    engine::debug::timing_log_scalar("muscriptor.decode.token_select_ms", token_select_ms);
    decode_graph_->log_profile();
    engine::debug::timing_log_scalar("muscriptor.decode.total_ms", engine::debug::elapsed_ms(start, Clock::now()));
    return out;
}

}  // namespace engine::models::muscriptor
