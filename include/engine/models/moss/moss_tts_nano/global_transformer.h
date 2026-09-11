#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/moss/moss_tts_nano/assets.h"
#include "engine/models/moss/moss_tts_nano/types.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::models::moss_tts_nano {

class MossTTSNanoGlobalTransformerRuntime {
public:
    MossTTSNanoGlobalTransformerRuntime(
        std::shared_ptr<const MossTTSNanoAssets> assets,
        core::ExecutionContext & execution_context,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType weight_storage_type);
    ~MossTTSNanoGlobalTransformerRuntime();

    void prepare_prefill(int64_t prompt_rows);
    void prepare_decode(int64_t cache_rows);
    std::vector<float> last_hidden(const std::vector<int32_t> & rows, int64_t row_count, int64_t row_width);
    int64_t graph_builds() const noexcept;

private:
    struct Weights;
    struct Graph;
    std::shared_ptr<const MossTTSNanoAssets> assets_;
    core::ExecutionContext & execution_context_;
    std::shared_ptr<Weights> weights_;
    size_t prefill_graph_arena_bytes_ = 0;
    int64_t decode_capacity_ = 0;
    int64_t graph_builds_ = 0;
    std::unique_ptr<Graph> graph_;
};

}  // namespace engine::models::moss_tts_nano
