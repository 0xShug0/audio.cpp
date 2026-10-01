#pragma once

#include "engine/framework/modules/speech_encoders/whisper_embedding.h"
#include "engine/framework/core/backend_weight_store.h"

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

}  // namespace engine::models::latentsync
