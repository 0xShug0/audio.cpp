#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/runtime/session.h"
#include "engine/community_models/vieneu_v3_turbo/assets.h"
#include "engine/community_models/vieneu_v3_turbo/types.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::models::vieneu_v3_turbo {

class VieNeuSpeakerEncoderGraph;
struct VieNeuSpeakerEncoderWeights;

struct VieNeuSpeakerFeatures {
    std::vector<float> values;
    int64_t mel_bins = 0;
    int64_t frames = 0;
};

audio::AudioTensor compute_vieneu_speaker_mel(const runtime::AudioBuffer & audio, int threads);

class VieNeuSpeakerEncoderRuntime {
public:
    VieNeuSpeakerEncoderRuntime(
        std::shared_ptr<const VieNeuTTSAssets> assets,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        engine::assets::TensorStorageType conv_weight_storage_type);
    ~VieNeuSpeakerEncoderRuntime();

    VieNeuSpeakerEmbedding encode(const runtime::AudioBuffer & audio) const;
    VieNeuSpeakerFeatures extract_features(const runtime::AudioBuffer & audio) const;
    VieNeuSpeakerEmbedding encode_features(const VieNeuSpeakerFeatures & features) const;

private:
    std::shared_ptr<const VieNeuTTSAssets> assets_;
    std::shared_ptr<const VieNeuSpeakerEncoderWeights> weights_;
    core::ExecutionContext * execution_context_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    mutable std::unique_ptr<VieNeuSpeakerEncoderGraph> graph_;
};

}  // namespace engine::models::vieneu_v3_turbo
