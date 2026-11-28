#pragma once

#include "engine/models/m5_tts/assets.h"
#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/core/execution_context.h"

#include <array>
#include <functional>
#include <memory>
#include <vector>

namespace engine::models::m5_tts {

using AudioFrame = std::array<int32_t, 16>;
// Sampling is separate from AR execution. Callers can select generated or
// externally supplied tokens without changing either decoder's mathematics.
using TokenSelector = std::function<int32_t(int64_t frame, int64_t codebook,
    const std::vector<float> & logits, bool allow_eos)>;

class M5LlamaRuntime {
public:
    M5LlamaRuntime(const ModelAssets & assets, core::ExecutionContext & execution,
        assets::TensorStorageType storage, core::AttentionPreference attention);
    ~M5LlamaRuntime();

    std::vector<float> text_embeddings(const std::vector<int32_t> & ids);
    std::vector<float> audio_embedding(const AudioFrame & frame);
    void begin(const std::vector<float> & prompt_embeddings, int64_t max_frames);
    AudioFrame next_frame(const TokenSelector & select);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::m5_tts
