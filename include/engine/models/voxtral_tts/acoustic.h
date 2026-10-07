#pragma once

#include "engine/models/voxtral_tts/assets.h"
#include "engine/framework/core/execution_context.h"

#include <memory>

namespace engine::models::voxtral_tts {

class VoxtralAcousticFlowRuntime {
public:
    VoxtralAcousticFlowRuntime(const VoxtralTtsAssets & assets, core::ExecutionContext & execution,
                              assets::TensorStorageType storage);
    ~VoxtralAcousticFlowRuntime();
    std::vector<int32_t> sample(const std::vector<float> & hidden, const std::vector<float> & noise,
                                int steps, float guidance);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::voxtral_tts
