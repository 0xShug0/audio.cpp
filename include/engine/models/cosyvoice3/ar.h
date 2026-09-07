#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/cosyvoice3/assets.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::cosyvoice3 {

struct CosyVoice3ArRequest {
    std::vector<int32_t> prompt_text_tokens;
    std::vector<int32_t> target_text_tokens;
    std::vector<int32_t> prompt_speech_tokens;
    uint32_t seed = 1986;
    int64_t top_k = 25;
    int64_t min_tokens = -1;
    int64_t max_tokens = -1;
};

struct CosyVoice3ArOutput {
    std::vector<int32_t> speech_tokens;
};

class CosyVoice3ArRuntime {
public:
    CosyVoice3ArRuntime(
        std::shared_ptr<const CosyVoice3Assets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type);
    ~CosyVoice3ArRuntime();

    CosyVoice3ArRuntime(const CosyVoice3ArRuntime &) = delete;
    CosyVoice3ArRuntime & operator=(const CosyVoice3ArRuntime &) = delete;

    CosyVoice3ArOutput generate(const CosyVoice3ArRequest & request);
    void release_graphs();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::cosyvoice3
