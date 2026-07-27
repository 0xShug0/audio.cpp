#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/voxtral_realtime/assets.h"
#include "engine/models/voxtral_realtime/types.h"

#include <cstddef>
#include <memory>

namespace engine::models::voxtral_realtime {

class VoxtralRealtimeTextDecoderRuntime {
public:
    struct Impl;

    VoxtralRealtimeTextDecoderRuntime(
        std::shared_ptr<const VoxtralRealtimeAssets> assets,
        core::ExecutionContext & execution,
        size_t prefill_graph_arena_bytes,
        size_t decode_graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type,
        int64_t stream_decode_cache_steps);
    ~VoxtralRealtimeTextDecoderRuntime();

    VoxtralRealtimeGeneratedTokens generate(
        const VoxtralRealtimePrompt & prompt,
        const VoxtralRealtimeAudioEmbeddings & audio_embeddings,
        const VoxtralRealtimeGenerationOptions & options);
    int32_t begin_stream(
        const VoxtralRealtimePrompt & prompt,
        const VoxtralRealtimeAudioEmbeddings & audio_embeddings,
        const VoxtralRealtimeGenerationOptions & options);
    // `audio_row` selects one row of a possibly batched encoder pass; each call still advances
    // the decoder by exactly one 80 ms step.
    int32_t stream_step(
        int32_t previous_token,
        const VoxtralRealtimeAudioEmbeddings & audio_embeddings,
        int64_t audio_row,
        int64_t num_delay_tokens,
        const VoxtralRealtimeGenerationOptions & options);
    bool is_eos(int32_t token) const;
    // Emits the summed upload/compute/readback/sample split of every streaming decode step.
    void log_stream_timings() const;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::voxtral_realtime
