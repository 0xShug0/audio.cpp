#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/hviske_asr_v6/assets.h"
#include "engine/models/hviske_asr_v6/frontend.h"

#include <memory>
#include <vector>

namespace engine::models::hviske_asr_v6 {

class HviskeV6WhisperRoPEEncoderRuntime {
public:
    HviskeV6WhisperRoPEEncoderRuntime(std::shared_ptr<const HviskeV6Assets> assets,
        core::ExecutionContext & execution, assets::TensorStorageType storage);
    ~HviskeV6WhisperRoPEEncoderRuntime();
    std::vector<float> encode(const HviskeV6Features & features);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::hviske_asr_v6
