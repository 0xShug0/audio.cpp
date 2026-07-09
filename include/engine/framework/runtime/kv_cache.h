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

class TransformerKVCache {
public:
    TransformerKVCache() = default;
    TransformerKVCache(
        int64_t cache_steps,
        int64_t step_elems,
        std::vector<core::TensorValue> keys,
        std::vector<core::TensorValue> values);

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
    std::vector<LayerCache> layers_;
};

core::TensorValue view_transformer_kv_cache_steps(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    int64_t start,
    int64_t steps,
    int64_t heads,
    int64_t head_dim,
    const char * label);

}  // namespace engine::runtime
