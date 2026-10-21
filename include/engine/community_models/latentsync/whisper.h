#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/speech_encoders/whisper_embedding.h"

#include <memory>

namespace engine::models::latentsync {

modules::WhisperEmbeddingWeights load_whisper_encoder_weights(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    const modules::WhisperEmbeddingConfig & config, assets::TensorStorageType type);

class LatentSyncWhisperEncoder {
public:
    explicit LatentSyncWhisperEncoder(modules::WhisperEmbeddingConfig config);

    // Returns [batch, time, embedding_layer, channels], before ln_post.
    core::TensorValue build(
        core::ModuleBuildContext & ctx,
        const core::TensorValue & mel,
        const modules::WhisperEmbeddingWeights & weights) const;

private:
    modules::WhisperEmbeddingConfig config_;
};

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
