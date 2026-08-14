#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/transformers/qwen_causal_decode_runtime.h"
#include "engine/community_models/minimax_music3/assets.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::models::minimax_music3 {

// The global Qwen3-8B language model. The conditional and unconditional CFG sequences run
// as two shared-weight decode runtimes with independent KV states. Logits come from the
// sliced head: row 0 is the audio end token, rows 1.. are the semantic codes.
class MiniMaxMusic3LmRuntime final {
public:
    MiniMaxMusic3LmRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const assets::TensorSource> source,
        const MiniMaxMusic3Config & config,
        size_t weight_context_bytes);
    ~MiniMaxMusic3LmRuntime();

    struct StepResult {
        std::vector<float> cond_logits;    // [lm_logits]
        std::vector<float> uncond_logits;  // [lm_logits]
        std::vector<float> last_hidden;    // [2 * hidden]: conditional row then unconditional row
    };

    // Prefill both branches and enter decode mode sized for required_cache_steps.
    StepResult prefill(
        const std::vector<int32_t> & cond_ids,
        const std::vector<int32_t> & uncond_ids,
        int64_t required_cache_steps);

    // Advance both branches by one step with the same frame feedback embedding.
    StepResult decode_embedding(const std::vector<float> & embedding);

    // The token embedding table, shared with the depth decoder for code lookups.
    core::TensorValue token_embedding() const;

    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::minimax_music3
