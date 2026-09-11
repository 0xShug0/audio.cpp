#pragma once

#include "engine/community_models/glm_tts/assets.h"
#include "engine/community_models/glm_tts/prompt.h"
#include "engine/framework/core/backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::glm_tts {

struct GlmTTSGenerateOptions {
    int64_t max_new_tokens = 0;
    int64_t top_k = 25;
    float top_p = 0.8F;
    float temperature = 1.0F;
    uint32_t seed = 0;
    bool restrict_output_head = true;
};

struct GlmTTSGenerateResult {
    std::vector<int32_t> speech_tokens;
    bool stopped_on_end_of_audio = false;
};

class GlmTTSLlamaRuntime {
public:
    GlmTTSLlamaRuntime(
        std::shared_ptr<const GlmTTSAssets> assets,
        core::BackendType backend_type,
        int device,
        int threads,
        assets::TensorStorageType weight_storage_type =
            assets::TensorStorageType::Native,
        size_t weight_context_bytes = 8ull * 1024ull * 1024ull * 1024ull,
        size_t constant_context_bytes = 256ull * 1024ull * 1024ull);
    ~GlmTTSLlamaRuntime();

    GlmTTSLlamaRuntime(const GlmTTSLlamaRuntime &) = delete;
    GlmTTSLlamaRuntime & operator=(const GlmTTSLlamaRuntime &) = delete;

    GlmTTSGenerateResult generate(
        const GlmTTSPrompt & prompt,
        const GlmTTSGenerateOptions & options) const;
    void release_runtime_graph();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::glm_tts
