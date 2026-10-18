#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <vector>

namespace engine::models::smart_turn {

class SmartTurnWhisperTinyRuntime {
public:
    SmartTurnWhisperTinyRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        core::ExecutionContext & execution);
    ~SmartTurnWhisperTinyRuntime();

    float completion_probability(const std::vector<float> & audio_16k);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::smart_turn
