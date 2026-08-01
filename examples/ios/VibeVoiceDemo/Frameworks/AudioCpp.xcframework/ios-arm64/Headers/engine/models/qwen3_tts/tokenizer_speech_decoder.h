#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/qwen3_tts/assets.h"
#include "engine/models/qwen3_tts/types.h"

#include <cstddef>
#include <memory>

namespace engine::core {
class ConstantTensorCache;
}

namespace engine::models {

namespace qwen3_tts {

struct Qwen3SpeechTokenizerDecoderWeights;
class Qwen3SpeechTokenizerDecoderGraph;

class Qwen3SpeechTokenizerDecoderRuntime {
public:
    Qwen3SpeechTokenizerDecoderRuntime(
        std::shared_ptr<const Qwen3TTSAssets> assets,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        size_t constant_context_bytes,
        engine::assets::TensorStorageType linear_weight_storage_type,
        engine::assets::TensorStorageType conv_weight_storage_type,
        Qwen3TTSPerfMode perf_mode);
    ~Qwen3SpeechTokenizerDecoderRuntime();

    runtime::AudioBuffer decode(const Qwen3SpeechCodes & codec_codes) const;
    runtime::AudioBuffer decode_and_trim_reference(
        const Qwen3SpeechCodes & reference_codes,
        const Qwen3SpeechCodes & generated_codes) const;

private:
    std::shared_ptr<const Qwen3TTSAssets> assets_;
    core::ExecutionContext * execution_context_ = nullptr;
    std::shared_ptr<const Qwen3SpeechTokenizerDecoderWeights> weights_;
    size_t graph_arena_bytes_ = 0;
    Qwen3TTSPerfMode perf_mode_ = Qwen3TTSPerfMode::Standard;
    std::unique_ptr<core::ConstantTensorCache> constants_;
    mutable std::unique_ptr<Qwen3SpeechTokenizerDecoderGraph> graph_;
};

}  // namespace qwen3_tts
}  // namespace engine::models
