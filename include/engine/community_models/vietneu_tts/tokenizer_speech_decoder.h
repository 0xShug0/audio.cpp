#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session.h"
#include "engine/community_models/vietneu_tts/assets.h"
#include "engine/community_models/vietneu_tts/types.h"

#include <cstddef>
#include <memory>

namespace engine::models {

namespace common {
class ConstantTensorCache;
}

namespace vietneu_tts {

struct Qwen3SpeechTokenizerDecoderWeights;
class Qwen3SpeechTokenizerDecoderGraph;

class Qwen3SpeechTokenizerDecoderRuntime {
public:
    Qwen3SpeechTokenizerDecoderRuntime(
        std::shared_ptr<const VietneuTTSAssets> assets,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        size_t constant_context_bytes,
        engine::assets::TensorStorageType linear_weight_storage_type,
        engine::assets::TensorStorageType conv_weight_storage_type);
    ~Qwen3SpeechTokenizerDecoderRuntime();

    runtime::AudioBuffer decode(const Qwen3SpeechCodes & codec_codes) const;
    runtime::AudioBuffer decode_and_trim_reference(
        const Qwen3SpeechCodes & reference_codes,
        const Qwen3SpeechCodes & generated_codes) const;

private:
    std::shared_ptr<const VietneuTTSAssets> assets_;
    core::ExecutionContext * execution_context_ = nullptr;
    std::shared_ptr<const Qwen3SpeechTokenizerDecoderWeights> weights_;
    size_t graph_arena_bytes_ = 0;
    std::unique_ptr<common::ConstantTensorCache> constants_;
    mutable std::unique_ptr<Qwen3SpeechTokenizerDecoderGraph> graph_;
};

}  // namespace vietneu_tts
}  // namespace engine::models
