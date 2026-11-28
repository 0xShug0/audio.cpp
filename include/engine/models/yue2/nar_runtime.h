#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/models/yue2/ar_runtime.h"
#include "engine/models/yue2/assets.h"
#include "engine/models/yue2/types.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace engine::models::yue2 {

// Prefills the AR state for one NAR chunk. The second argument is how many
// leading tokens every chunk of the song shares.
using Yue2NARPrefillState = std::function<Yue2ARDevicePrefixState(const std::vector<int32_t> &, size_t)>;

class Yue2NARRuntime {
public:
    Yue2NARRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const Yue2Assets> assets,
        assets::TensorStorageType weight_type,
        size_t weight_context_bytes,
        size_t graph_arena_bytes,
        bool allow_flash_attention = true,
        int64_t attention_tile_rows = 0,
        int64_t window_frames = 0,
        int64_t overlap_frames = 0);
    ~Yue2NARRuntime();

    std::vector<float> synthesize(
        const std::vector<int32_t> & prefix,
        const std::vector<int32_t> & codec,
        const Yue2NARPrefillState & prefill_state,
        const std::vector<float> & noise,
        uint64_t seed,
        int64_t ode_steps,
        int64_t context);

    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::yue2
