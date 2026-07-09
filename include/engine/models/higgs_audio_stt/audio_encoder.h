#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/higgs_audio_stt/assets.h"
#include "engine/models/higgs_audio_stt/types.h"

#include <cstddef>
#include <memory>

namespace engine::models::higgs_audio_stt {

class HiggsAudioSTTAudioEncoderGraph;
struct HiggsAudioSTTAudioEncoderWeights;

class HiggsAudioSTTAudioEncoderRuntime {
public:
    HiggsAudioSTTAudioEncoderRuntime(
        std::shared_ptr<const HiggsAudioSTTAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        assets::TensorStorageType weight_storage_type);
    ~HiggsAudioSTTAudioEncoderRuntime();

    HiggsAudioSTTAudioEmbeddings encode(const HiggsAudioSTTAudioFeatures & features);

private:
    std::shared_ptr<const HiggsAudioSTTAssets> assets_;
    std::shared_ptr<const HiggsAudioSTTAudioEncoderWeights> weights_;
    core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::unique_ptr<HiggsAudioSTTAudioEncoderGraph> graph_;
};

}  // namespace engine::models::higgs_audio_stt
