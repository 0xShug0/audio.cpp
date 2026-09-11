#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session.h"
#include "engine/community_models/vietneu_tts/assets.h"
#include "engine/community_models/vietneu_tts/types.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::models::vietneu_tts {

class VietneuSpeakerEncoderGraph;
struct VietneuSpeakerEncoderWeights;

struct VietneuSpeakerFeatures {
    std::vector<float> values;
    int64_t mel_bins = 0;
    int64_t frames = 0;
};

class VietneuSpeakerEncoderRuntime {
public:
    VietneuSpeakerEncoderRuntime(
        std::shared_ptr<const VietneuTTSAssets> assets,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        engine::assets::TensorStorageType conv_weight_storage_type);
    ~VietneuSpeakerEncoderRuntime();

    VietneuSpeakerEmbedding encode(const runtime::AudioBuffer & audio) const;
    VietneuSpeakerFeatures extract_features(const runtime::AudioBuffer & audio) const;
    VietneuSpeakerEmbedding encode_features(const VietneuSpeakerFeatures & features) const;

private:
    std::shared_ptr<const VietneuTTSAssets> assets_;
    std::shared_ptr<const VietneuSpeakerEncoderWeights> weights_;
    core::ExecutionContext * execution_context_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    mutable std::unique_ptr<VietneuSpeakerEncoderGraph> graph_;
};

}  // namespace engine::models::vietneu_tts
