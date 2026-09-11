#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/model.h"
#include "engine/models/firered_audio/assets.h"
#include "engine/models/firered_audio/audio_encoder.h"
#include "engine/models/firered_audio/qwen35_runtime.h"
#include "engine/models/firered_audio/tokenizer.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::firered_audio {

struct FireRedAudioUnderstandingRequest {
    std::string prompt;
    engine::runtime::AudioBuffer audio;
    std::string language = "zh";
    int64_t max_new_tokens = 300;
    bool enable_thinking = false;
    bool do_sample = false;
    int64_t top_k = 20;
    float top_p = 0.8F;
    float temperature = 0.7F;
    float repetition_penalty = 1.0F;
    uint32_t seed = 1234;
};

struct FireRedAudioUnderstandingResult {
    std::string text;
    std::string language;
};

class FireRedAudioUnderstandingRuntime {
public:
    FireRedAudioUnderstandingRuntime(
        std::shared_ptr<const FireRedAudioAssets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type,
        bool mem_saver);
    ~FireRedAudioUnderstandingRuntime();

    FireRedAudioUnderstandingRuntime(const FireRedAudioUnderstandingRuntime &) = delete;
    FireRedAudioUnderstandingRuntime & operator=(const FireRedAudioUnderstandingRuntime &) = delete;

    FireRedAudioUnderstandingResult generate(const FireRedAudioUnderstandingRequest & request);
    void release_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::firered_audio
