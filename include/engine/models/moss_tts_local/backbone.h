#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/moss_tts_local/assets.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::moss_tts_local {

// Qwen3 backbone (transformer.*) runtime: loads the language-model weights and runs
// a prefill forward that returns the final hidden states. Text tokens are embedded in
// the graph; the summed audio-codebook contribution is supplied as a precomputed bias
// so the depth transformer and the generator can share a single embedding table.
class MossBackboneRuntime {
public:
    MossBackboneRuntime(
        std::shared_ptr<const MossTTSLocalAssets> assets,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type);
    ~MossBackboneRuntime();

    MossBackboneRuntime(const MossBackboneRuntime &) = delete;
    MossBackboneRuntime & operator=(const MossBackboneRuntime &) = delete;

    int64_t hidden_size() const noexcept;

    // Runs the backbone over a prefill sequence of text token ids and returns the
    // final hidden states as [steps, hidden_size] row-major float.
    std::vector<float> forward_prefill(const std::vector<int32_t> & token_ids) const;

    // Same as forward_prefill, but adds the per-position audio-codebook embedding sum
    // (audio_bias, [steps * hidden_size] row-major) to the text embedding before the
    // decoder stack. This mirrors MossTTSLocalModel._build_inputs_embeds.
    std::vector<float> forward_prefill_fused(
        const std::vector<int32_t> & token_ids,
        const std::vector<float> & audio_bias) const;

private:
    std::vector<float> run_prefill(const std::vector<int32_t> & token_ids, const float * audio_bias) const;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::moss_tts_local
