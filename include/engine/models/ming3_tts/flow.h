#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/ming3_tts/assets.h"

#include <memory>
#include <vector>

namespace engine::models::ming3_tts {

class MingFlowRuntime {
public:
    MingFlowRuntime(std::shared_ptr<const assets::TensorSource> source,
        const MingFlowConfig & config, core::ExecutionContext & execution);
    ~MingFlowRuntime();

    std::vector<float> predict_velocity(const std::vector<float> & latent, const std::vector<float> & conditioning,
        const std::vector<float> & history, float time);

    std::vector<float> sample(const std::vector<float> & conditioning, const std::vector<float> & history,
        const std::vector<float> & noise, const std::vector<float> & sde_noise,
        float cfg = 2.0f, float sigma = 0.25f, float temperature = 0.0f);
    std::vector<float> aggregate(const std::vector<float> & latent);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::ming3_tts
