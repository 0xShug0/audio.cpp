#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/community_models/outetts/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace engine::models::outetts {

struct OuteTTSGenerateOptions {
    int64_t max_new_tokens = 4096;
    float temperature = 0.4F;
    float repetition_penalty = 1.1F;
    int64_t repetition_window = 64;
    int64_t top_k = 40;
    float top_p = 0.9F;
    float min_p = 0.05F;
    uint32_t seed = 0;
};

enum class OuteTTSStopReason {
    Eos,
    AudioEnd,
    MaxTokens,
    ContextLimit,
};

std::string_view outetts_stop_reason_name(OuteTTSStopReason reason) noexcept;

struct OuteTTSGenerateResult {
    std::vector<int32_t> tokens;
    OuteTTSStopReason stop_reason = OuteTTSStopReason::MaxTokens;
};

class OuteTTSLlamaRuntime final {
public:
    OuteTTSLlamaRuntime(
        std::shared_ptr<const OuteTTSAssets> assets,
        core::BackendType backend_type,
        int device,
        int threads,
        size_t weight_context_bytes = 4ull * 1024ull * 1024ull * 1024ull,
        size_t constant_context_bytes = 256ull * 1024ull * 1024ull,
        assets::TensorStorageType weight_storage_type = assets::TensorStorageType::Native);
    ~OuteTTSLlamaRuntime();

    OuteTTSLlamaRuntime(const OuteTTSLlamaRuntime &) = delete;
    OuteTTSLlamaRuntime & operator=(const OuteTTSLlamaRuntime &) = delete;

    OuteTTSGenerateResult generate(
        const std::vector<int32_t> & prompt,
        const OuteTTSGenerateOptions & options,
        int32_t eos_id,
        int32_t audio_end_id) const;

    // Releases the reusable cached-step graph while keeping model weights
    // resident. Returns the released KV-cache capacity in tokens.
    int64_t release_cached_step_graph();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::outetts
