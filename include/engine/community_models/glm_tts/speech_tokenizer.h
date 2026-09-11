#pragma once

#include "engine/community_models/glm_tts/assets.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/model.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::glm_tts {

class GlmTTSSpeechTokenizer {
public:
    GlmTTSSpeechTokenizer(
        std::shared_ptr<const GlmTTSAssets> assets,
        core::BackendConfig backend,
        assets::TensorStorageType weight_storage_type =
            assets::TensorStorageType::Native,
        size_t weight_context_bytes = 3ull * 1024ull * 1024ull * 1024ull,
        size_t graph_context_bytes = 2ull * 1024ull * 1024ull * 1024ull);
    ~GlmTTSSpeechTokenizer();

    GlmTTSSpeechTokenizer(const GlmTTSSpeechTokenizer &) = delete;
    GlmTTSSpeechTokenizer & operator=(const GlmTTSSpeechTokenizer &) = delete;

    std::vector<int32_t> encode(
        const runtime::AudioBuffer & audio) const;
    void release_runtime_graph() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::glm_tts
