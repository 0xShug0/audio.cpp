#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/audio_flamingo_next/assets.h"
#include "engine/models/audio_flamingo_next/types.h"

#include <memory>

namespace engine::models::audio_flamingo_next {

class AFNextQwen2DecoderRuntime {
public:
    AFNextQwen2DecoderRuntime(std::shared_ptr<const AFNextAssets> assets,
        core::ExecutionContext & execution, assets::TensorStorageType storage_type);
    ~AFNextQwen2DecoderRuntime();
    AFNextGeneratedTokens generate(const AFNextPrompt & prompt,
        const AFNextAudioProjectorOutput & audio, const AFNextGenerationOptions & options);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::audio_flamingo_next
