#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/speaker_encoders/ecapa_tdnn_runtime.h"
#include "engine/framework/modules/speech_encoders/wavlm_encoder.h"
#include "engine/framework/runtime/session.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::meanvc2 {

struct MeanVC2SpeakerFeatures {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t dims = 0;
};

class MeanVC2SpeakerEncoderRuntime final {
public:
    MeanVC2SpeakerEncoderRuntime(
        std::shared_ptr<const engine::assets::TensorSource> wavlm_source,
        std::shared_ptr<const engine::assets::TensorSource> ecapa_source,
        engine::core::ExecutionContext & execution_context,
        engine::assets::TensorStorageType weight_storage_type);
    ~MeanVC2SpeakerEncoderRuntime();

    MeanVC2SpeakerFeatures extract_features(const runtime::AudioBuffer & audio) const;
    std::vector<float> embed(const runtime::AudioBuffer & audio) const;

private:
    engine::modules::WavlmEncoderComponent wavlm_;
    std::vector<float> feature_weights_;
    std::unique_ptr<engine::modules::ecapa_tdnn::EcapaRuntime> ecapa_;
};

}  // namespace engine::models::meanvc2
