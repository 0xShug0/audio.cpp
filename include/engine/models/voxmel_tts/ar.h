#pragma once

#include "engine/models/voxmel_tts/assets.h"
#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <vector>

namespace engine::models::voxmel_tts {

struct QwenState {
    std::vector<float> hidden;
    float stop_logit = 0;
    int64_t branches = 0;
};

class VoxMelQwenRuntime {
public:
    VoxMelQwenRuntime(const ModelAssets & assets, core::ExecutionContext & execution,
        assets::TensorStorageType storage, core::AttentionPreference attention);
    ~VoxMelQwenRuntime();

    std::vector<float> text_embeddings(const std::vector<int32_t> & ids);
    std::vector<float> coarse_embeddings(const std::vector<float> & coarse);
    QwenState begin(const std::vector<std::vector<float>> & prefixes, int64_t max_frames);
    QwenState advance(const std::vector<float> & coarse);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::voxmel_tts
