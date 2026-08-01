#pragma once

#include "engine/community_models/glm_tts/assets.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::glm_tts {

struct GlmTTSFlowInput {
    std::vector<int32_t> speech_tokens;
    std::vector<float> prompt_mel;
    std::vector<float> speaker_embedding;
    std::vector<float> initial_noise;
    int64_t prompt_frames = 0;
    int inference_steps = 10;
    float cfg_rate = 0.7F;
};

struct GlmTTSFlowOutput {
    std::vector<float> mel;
    int64_t frames = 0;
    int64_t prompt_frames = 0;
};

class GlmTTSFlowRuntime {
public:
    GlmTTSFlowRuntime() = default;
    GlmTTSFlowRuntime(
        std::shared_ptr<const assets::TensorSource> source,
        core::BackendConfig backend,
        assets::TensorStorageType storage_type,
        GlmTTSFlowConfig config);
    ~GlmTTSFlowRuntime();

    GlmTTSFlowRuntime(GlmTTSFlowRuntime &&) noexcept;
    GlmTTSFlowRuntime & operator=(GlmTTSFlowRuntime &&) noexcept;
    GlmTTSFlowRuntime(const GlmTTSFlowRuntime &) = delete;
    GlmTTSFlowRuntime & operator=(const GlmTTSFlowRuntime &) = delete;

    GlmTTSFlowOutput generate(const GlmTTSFlowInput & input) const;
    void release_graph();

private:
    struct State;

    GlmTTSFlowConfig config_;
    std::shared_ptr<State> state_;
};

}  // namespace engine::models::glm_tts
