#pragma once

#include "engine/models/seed_vc/assets.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::seed_vc {

struct SeedVcV1CfmEstimatorInput {
    std::vector<float> x;
    std::vector<float> prompt;
    std::vector<float> cond;
    std::vector<float> style;
    std::vector<float> timestep;
    int64_t batch = 0;
    int64_t frames = 0;
};

struct SeedVcV1CfmEstimatorOutput {
    std::vector<float> velocity;
    int64_t batch = 0;
    int64_t channels = 0;
    int64_t frames = 0;
};

struct SeedVcV1CfmInferenceInput {
    std::vector<float> mu;
    std::vector<float> prompt;
    std::vector<float> style;
    std::vector<float> initial_noise;
    int64_t batch = 0;
    int64_t frames = 0;
    int64_t prompt_frames = 0;
    int num_inference_steps = 0;
    float temperature = 1.0F;
    float inference_cfg_rate = 0.5F;
};

class SeedVcV1CfmEstimator {
public:
    SeedVcV1CfmEstimator() = default;
    SeedVcV1CfmEstimator(
        std::shared_ptr<const engine::assets::TensorSource> source,
        engine::core::BackendConfig backend,
        engine::assets::TensorStorageType storage_type,
        SeedVcV1DitConfig config,
        SeedVcV1WavenetConfig wavenet_config,
        int64_t style_dim);
    ~SeedVcV1CfmEstimator();

    SeedVcV1CfmEstimator(SeedVcV1CfmEstimator &&) noexcept;
    SeedVcV1CfmEstimator & operator=(SeedVcV1CfmEstimator &&) noexcept;
    SeedVcV1CfmEstimator(const SeedVcV1CfmEstimator &) = delete;
    SeedVcV1CfmEstimator & operator=(const SeedVcV1CfmEstimator &) = delete;

    SeedVcV1CfmEstimatorOutput run(const SeedVcV1CfmEstimatorInput & input) const;
    SeedVcV1CfmEstimatorOutput infer(const SeedVcV1CfmInferenceInput & input) const;

private:
    struct State;

    SeedVcV1DitConfig config_;
    SeedVcV1WavenetConfig wavenet_config_;
    int64_t style_dim_ = 0;
    std::shared_ptr<State> state_;
};

}  // namespace engine::models::seed_vc
