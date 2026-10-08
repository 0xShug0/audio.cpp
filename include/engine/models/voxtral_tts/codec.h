#pragma once

#include "engine/models/voxtral_tts/assets.h"
#include "engine/framework/core/execution_context.h"

#include <memory>

namespace engine::models::voxtral_tts {

constexpr int64_t kCodecInitialFrames = 5;
constexpr int64_t kCodecChunkFrames = 25;
constexpr int64_t kCodecContextFrames = 25;

class VoxtralCausalCodecDecoderRuntime {
public:
    VoxtralCausalCodecDecoderRuntime(const VoxtralTtsAssets & assets, core::ExecutionContext & execution,
                                    assets::TensorStorageType storage);
    ~VoxtralCausalCodecDecoderRuntime();
    std::vector<float> decode(const std::vector<int32_t> & codes, bool initial);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::voxtral_tts
