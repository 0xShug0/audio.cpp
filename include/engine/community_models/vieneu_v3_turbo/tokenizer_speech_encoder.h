#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session.h"
#include "engine/community_models/vieneu_v3_turbo/assets.h"
#include "engine/community_models/vieneu_v3_turbo/types.h"

#include <cstddef>
#include <memory>

namespace engine::core {
class ConstantTensorCache;
}

namespace engine::models {

namespace vieneu_v3_turbo {

struct Qwen3SpeechTokenizerEncoderWeights;
class Qwen3SpeechTokenizerEncoderGraph;

class Qwen3SpeechTokenizerEncoderRuntime {
public:
    Qwen3SpeechTokenizerEncoderRuntime(
        std::shared_ptr<const VieNeuTTSAssets> assets,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        engine::assets::TensorStorageType linear_weight_storage_type,
        engine::assets::TensorStorageType conv_weight_storage_type);
    ~Qwen3SpeechTokenizerEncoderRuntime();

    Qwen3SpeechCodes encode(const runtime::AudioBuffer & audio) const;

private:
    std::shared_ptr<const VieNeuTTSAssets> assets_;
    std::shared_ptr<const Qwen3SpeechTokenizerEncoderWeights> weights_;
    core::ExecutionContext * execution_context_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::unique_ptr<core::ConstantTensorCache> constants_;
    mutable std::unique_ptr<Qwen3SpeechTokenizerEncoderGraph> graph_;
};

}  // namespace vieneu_v3_turbo
}  // namespace engine::models
