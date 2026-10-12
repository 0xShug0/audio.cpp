#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/moss/moss_tts_nano/assets.h"
#include "engine/models/moss/moss_tts_nano/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::moss_tts_nano {

// Incremental global transformer runtime for MOSS-TTS-Nano.
//
// The model is generated one audio frame at a time, so the runtime keeps a
// persistent per-layer K/V cache and forwards a single row per step:
//
//     begin_generation(max_positions);
//     hidden = prefill(prompt rows);      // prompt K/V land in the cache
//     for (each frame) step_into(slot token, codes, hidden);
//
// The previous implementation rebuilt the whole transcript forward on every
// step (O(n^2) work) and additionally sized the graph for
// prompt_rows + max_new_frames regardless of the real output length.
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

    // Ensures the single-row decode graph and the K/V cache can hold max_positions rows. The
    // existing cache contents are kept when the current graph is already large enough, so an
    // unchanged prompt prefix can be reused across utterances.
    void begin_generation(int64_t max_positions);
    // Forwards prompt rows [reuse_rows, row_count) and stores their K/V at the same slots.
    // Rows [0, reuse_rows) must already be cached (identical tokens), which lets a repeated
    // template prefix skip its forward pass. Returns the last row's hidden state.
    std::vector<float> prefill(
        const std::vector<int32_t> & rows,
        int64_t row_count,
        int64_t row_width,
        int64_t reuse_rows = 0);
    // Forwards one new row (assistant slot token + one code per codebook) and
    // appends its K/V. `hidden_state` is replaced with the new row's hidden state.
    void step_into(int32_t text_token, const std::vector<int32_t> & code_row, std::vector<float> & hidden_state);
    // Drops every cached row, e.g. when the next prompt shares nothing with the previous one.
    void reset_cache();
    // Rows currently stored in the cache.
    int64_t cached_positions() const noexcept;
    // Cache capacity in rows (prompt rows + max_new_frames).
    int64_t cache_capacity() const noexcept;
    int64_t graph_builds() const noexcept;

private:
    struct Weights;
    struct DecodeGraph;

    std::shared_ptr<const MossTTSNanoAssets> assets_;
    core::ExecutionContext & execution_context_;
    std::shared_ptr<Weights> weights_;
    size_t prefill_graph_arena_bytes_ = 0;
    size_t decode_graph_arena_bytes_ = 0;
    int64_t graph_builds_ = 0;
    std::unique_ptr<DecodeGraph> decode_;
};

}  // namespace engine::models::moss_tts_nano
