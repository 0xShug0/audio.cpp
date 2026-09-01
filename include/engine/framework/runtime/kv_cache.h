#pragma once

#include "engine/framework/core/module.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::runtime {

struct KVLayerState {
    int64_t valid_steps = 0;
    std::vector<float> key;
    std::vector<float> value;
};

struct TransformerKVState {
    int64_t current_end = 0;
    std::vector<KVLayerState> layers;
};

struct TransformerKVCacheOptions {
    bool allow_f16_storage = false;
    bool allow_bf16_storage = false;
};

class TransformerKVCache {
public:
    TransformerKVCache() = default;
    TransformerKVCache(
        int64_t cache_steps,
        int64_t step_elems,
        std::vector<core::TensorValue> keys,
        std::vector<core::TensorValue> values);
    TransformerKVCache(
        int64_t cache_steps,
        int64_t step_elems,
        std::vector<core::TensorValue> keys,
        std::vector<core::TensorValue> values,
        TransformerKVCacheOptions options);

    void import_state(const TransformerKVState & state);
    TransformerKVState export_state() const;

    void advance_after_direct_append(int64_t steps);
    void retain_prefix(int64_t prefix_steps);

    int64_t valid_steps() const noexcept;
    int64_t current_end() const noexcept;
    int64_t cache_steps() const noexcept;

    const core::TensorValue & key_tensor(size_t layer) const;
    const core::TensorValue & value_tensor(size_t layer) const;

    void trace_log_state(const std::string & name, int64_t num_heads, int64_t head_dim) const;

private:
    struct LayerCache {
        core::TensorValue key_tensor;
        core::TensorValue value_tensor;
        std::vector<float> import_key_scratch;
        std::vector<float> import_value_scratch;
    };

    int64_t cache_steps_ = 0;
    int64_t step_elems_ = 0;
    int64_t valid_steps_ = 0;
    int64_t current_end_ = 0;
    TransformerKVCacheOptions options_;
    std::vector<LayerCache> layers_;
};

struct BatchedKVLayerState {
    int64_t valid_steps = 0;
    std::vector<float> key;
    std::vector<float> value;
};

struct TransformerBatchedKVState {
    int64_t batch_size = 0;
    int64_t current_end = 0;
    // 可选的 per-member 结束位置（大小 == batch_size）。空 = 均匀（current_end 生效）。
    std::vector<int64_t> current_ends;
    std::vector<BatchedKVLayerState> layers;
};

class TransformerBatchedKVCache {
public:
    TransformerBatchedKVCache() = default;
    TransformerBatchedKVCache(
        int64_t cache_steps,
        int64_t batch_size,
        int64_t row_elems,
        std::vector<core::TensorValue> keys,
        std::vector<core::TensorValue> values);
    TransformerBatchedKVCache(
        int64_t cache_steps,
        int64_t batch_size,
        int64_t row_elems,
        std::vector<core::TensorValue> keys,
        std::vector<core::TensorValue> values,
        TransformerKVCacheOptions options);

    void import_state(const TransformerBatchedKVState & state);
    TransformerBatchedKVState export_state() const;

    void advance_after_direct_append(int64_t steps);

    int64_t batch_size() const noexcept;
    int64_t valid_steps() const noexcept;
    int64_t current_end() const noexcept;
    int64_t cache_steps() const noexcept;

    // --- per-member 结束位置（不同序列可处于不同位置）---
    int64_t member_end(int64_t batch) const noexcept;
    void set_member_end(int64_t batch, int64_t end) noexcept;
    void advance_member(int64_t batch, int64_t steps) noexcept;
    // 返回 per-member ends（空=均匀，调用方回退到 cache_slots）
    const std::vector<int64_t> & member_ends_for_mask() const noexcept { return member_ends_; }

private:
    struct LayerCache {
        core::TensorValue key_tensor;
        core::TensorValue value_tensor;
        std::vector<float> import_key_scratch;
        std::vector<float> import_value_scratch;
    };

    int64_t cache_steps_ = 0;
    int64_t batch_size_ = 0;
    int64_t row_elems_ = 0;
    int64_t valid_steps_ = 0;
    int64_t current_end_ = 0;
    // per-member 结束位置；空 = 均匀（用 current_end_）
    std::vector<int64_t> member_ends_;
    TransformerKVCacheOptions options_;
    std::vector<LayerCache> layers_;
};

core::TensorValue view_transformer_kv_cache_steps(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    int64_t start,
    int64_t steps,
    int64_t heads,
    int64_t head_dim,
    const char * label,
    ggml_type view_type = GGML_TYPE_F32);

}  // namespace engine::runtime
