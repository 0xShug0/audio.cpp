#pragma once

#include "engine/models/midashenglm_gen/assets.h"
#include "engine/models/midashenglm_gen/tokenizer_text.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <vector>

namespace engine::models::midashenglm_gen {

struct MiDashengLmGenPromptEncoderOutput {
    std::vector<float> embeddings;
    std::vector<int32_t> attention_mask;
    int64_t batch = 0;
    int64_t tokens = 0;
    int64_t hidden = 0;
};

struct MiDashengLmGenPromptEncoderWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::core::TensorValue token_embedding;
};

class MiDashengLmGenPromptEncoderRuntime {
public:
    MiDashengLmGenPromptEncoderRuntime(
        std::shared_ptr<const MiDashengLmGenAssets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type);
    ~MiDashengLmGenPromptEncoderRuntime();

    MiDashengLmGenPromptEncoderRuntime(const MiDashengLmGenPromptEncoderRuntime &) = delete;
    MiDashengLmGenPromptEncoderRuntime & operator=(const MiDashengLmGenPromptEncoderRuntime &) = delete;

    void prepare(int64_t batch, int64_t tokens);
    MiDashengLmGenPromptEncoderOutput encode(const MiDashengLmGenPromptBatch & input);
    void release_graphs();

private:
    class EncodeGraph;

    std::shared_ptr<const MiDashengLmGenAssets> assets_;
    engine::core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::shared_ptr<const MiDashengLmGenPromptEncoderWeights> weights_;
    std::unique_ptr<EncodeGraph> graph_;
};

}  // namespace engine::models::midashenglm_gen
