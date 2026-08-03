#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/voxtral_realtime/assets.h"
#include "engine/models/voxtral_realtime/types.h"

#include <cstddef>
#include <memory>

namespace engine::models::voxtral_realtime {

class VoxtralRealtimeAudioEncoderRuntime {
public:
    struct Impl;

    VoxtralRealtimeAudioEncoderRuntime(
        std::shared_ptr<const VoxtralRealtimeAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type);
    ~VoxtralRealtimeAudioEncoderRuntime();

    VoxtralRealtimeAudioEmbeddings encode(const VoxtralRealtimeFeatures & features);
    VoxtralRealtimeAudioEncoderStreamState make_stream_state() const;
    VoxtralRealtimeAudioEmbeddings encode_stream_chunk(
        const VoxtralRealtimeFeatures & features,
        VoxtralRealtimeAudioEncoderStreamState & state);
    // Emits the summed upload/compute/readback split of every streaming chunk seen so far.
    void log_stream_timings() const;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::voxtral_realtime
