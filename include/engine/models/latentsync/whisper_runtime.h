#pragma once

#include "engine/models/latentsync/whisper_encoder.h"
#include "engine/framework/core/execution_context.h"
#include <memory>

namespace engine::models::latentsync {

class LatentSyncWhisperRuntime {
public:
    LatentSyncWhisperRuntime(core::ExecutionContext & execution, const assets::TensorSource & source,
        modules::WhisperEmbeddingConfig config, assets::TensorStorageType weight_type);
    ~LatentSyncWhisperRuntime();
    std::vector<float> encode_mel(const std::vector<float> & mel);
    std::vector<float> encode_audio(const std::vector<float> & mono_16khz, size_t threads);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Features are [audio_time, five_hidden_levels, 384]; output is [frames, 50, 384].
std::vector<float> latentsync_audio_windows(const std::vector<float> & features,
    int64_t first_frame, int64_t frames, double fps);

}  // namespace engine::models::latentsync
