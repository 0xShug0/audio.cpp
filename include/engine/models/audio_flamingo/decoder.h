#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/audio_flamingo/assets.h"
#include "engine/models/audio_flamingo/types.h"

#include <memory>

namespace engine::models::audio_flamingo {

class AudioFlamingoQwen2DecoderRuntime {
public:
    AudioFlamingoQwen2DecoderRuntime(std::shared_ptr<const AudioFlamingoAssets> assets,
        core::ExecutionContext & execution, assets::TensorStorageType storage_type);
    ~AudioFlamingoQwen2DecoderRuntime();
    AudioFlamingoGeneratedTokens generate(const AudioFlamingoPrompt & prompt,
        const AudioFlamingoAudioProjectorOutput & audio, const AudioFlamingoGenerationOptions & options);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::audio_flamingo
