#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/firered_audio/assets.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace engine::models::firered_audio {

std::vector<float> firered_cosine_time_schedule(int64_t steps);
std::vector<float> firered_timestep_embedding(float timestep, int64_t dim = 256);

class FireRedAudioFlowRuntime {
public:
    FireRedAudioFlowRuntime(
        std::shared_ptr<const FireRedAudioAssets> assets,
        engine::core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        engine::assets::TensorStorageType storage_type);
    ~FireRedAudioFlowRuntime();

    FireRedAudioFlowRuntime(const FireRedAudioFlowRuntime &) = delete;
    FireRedAudioFlowRuntime & operator=(const FireRedAudioFlowRuntime &) = delete;

    std::vector<float> run(const std::vector<float> & x_in, const std::vector<float> & time_embed, int64_t batch);
    int64_t input_channels() const noexcept;
    void release_graph();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::firered_audio
