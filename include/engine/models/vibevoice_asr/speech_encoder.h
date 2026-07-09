#pragma once

#include "engine/models/vibevoice_asr/assets.h"
#include "engine/models/vibevoice_asr/connector.h"
#include "engine/models/vibevoice_asr/speech_tokenizer.h"
#include "engine/models/vibevoice_asr/types.h"

#include <cstddef>
#include <memory>
#include <optional>

namespace engine::models::vibevoice_asr {

class VibeVoiceASRSpeechEncoder {
public:
    VibeVoiceASRSpeechEncoder(
        std::shared_ptr<const VibeVoiceAssets> assets,
        core::BackendType backend_type,
        int device,
        int threads,
        size_t tokenizer_weight_context_bytes,
        size_t connector_weight_context_bytes,
        assets::TensorStorageType tokenizer_weight_storage_type,
        assets::TensorStorageType connector_weight_storage_type);

    VibeVoiceASRSpeechFeatures encode(const runtime::AudioBuffer & audio, uint64_t seed) const;

private:
    std::shared_ptr<const VibeVoiceAssets> assets_;
    std::optional<sampling::TorchCudaSamplingPolicy> sampling_policy_;
    VibeVoiceTokenizerWeightsRuntime tokenizer_;
    VibeVoiceConnectorWeightsRuntime connector_;
};

}  // namespace engine::models::vibevoice_asr
