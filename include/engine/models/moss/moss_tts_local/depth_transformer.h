#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/moss/moss_tts_local/assets.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::moss_tts_local {

// Single-layer GPT-2-J depth transformer (local_transformer.*). Each generation frame
// seeds it with the backbone hidden state and then feeds back one audio-codebook
// embedding at a time; the returned final hidden state drives the RVQ code heads and
// the assistant/end gate. The layer count is tiny, so the weights are kept in f32 and
// the prefix is recomputed each step instead of maintaining a KV cache. Forward graphs
// for every step count the generation loop uses (1..num_codebooks) are prebuilt once
// and reused; forward() runs 12 times per frame, so per-call graph rebuilds dominated
// the generation loop's host-side cost.
class MossDepthTransformer {
public:
    MossDepthTransformer(
        std::shared_ptr<const MossTTSLocalAssets> assets,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        size_t weight_context_bytes);
    ~MossDepthTransformer();

    MossDepthTransformer(const MossDepthTransformer &) = delete;
    MossDepthTransformer & operator=(const MossDepthTransformer &) = delete;

    int64_t hidden_size() const noexcept;

    // Runs the local transformer over inputs_embeds ([steps, hidden_size] row-major
    // float) and returns the final-step hidden state as [hidden_size] float.
    std::vector<float> forward(const std::vector<float> & inputs_embeds, int64_t steps) const;
    void reset_timing() const;
    void log_timing() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::moss_tts_local
