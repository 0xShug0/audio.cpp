#pragma once

#include "engine/models/seed_vc/assets.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::seed_vc {

struct SeedVcV2CfmEstimatorInput {
    std::vector<float> x;
    std::vector<float> prompt;
    std::vector<float> cond;
    std::vector<float> style;
    std::vector<float> timestep;
    int64_t batch = 0;
    int64_t frames = 0;
};

struct SeedVcV2CfmEstimatorOutput {
    std::vector<float> velocity;
    int64_t batch = 0;
    int64_t channels = 0;
    int64_t frames = 0;
};

struct SeedVcV2CfmInferenceInput {
    std::vector<float> mu;
    std::vector<float> prompt;
    std::vector<float> style;
    std::vector<float> initial_noise;
    int64_t batch = 0;
    int64_t frames = 0;
    int64_t prompt_frames = 0;
    int num_inference_steps = 0;
    float temperature = 1.0F;
    float intelligibility_cfg_rate = 0.7F;
    float similarity_cfg_rate = 0.7F;
    bool random_voice = false;
};

class SeedVcV2CfmEstimator {
public:
    SeedVcV2CfmEstimator() = default;
    SeedVcV2CfmEstimator(
        std::shared_ptr<const engine::assets::TensorSource> source,
        engine::core::BackendConfig backend,
        engine::assets::TensorStorageType storage_type,
        SeedVcV2DitConfig config);
    ~SeedVcV2CfmEstimator();

    SeedVcV2CfmEstimator(SeedVcV2CfmEstimator &&) noexcept;
    SeedVcV2CfmEstimator & operator=(SeedVcV2CfmEstimator &&) noexcept;
    SeedVcV2CfmEstimator(const SeedVcV2CfmEstimator &) = delete;
    SeedVcV2CfmEstimator & operator=(const SeedVcV2CfmEstimator &) = delete;

    SeedVcV2CfmEstimatorOutput run(const SeedVcV2CfmEstimatorInput & input) const;
    SeedVcV2CfmEstimatorOutput infer(const SeedVcV2CfmInferenceInput & input) const;

private:
    struct State;

    SeedVcV2DitConfig config_;
    std::shared_ptr<State> state_;
};

}  // namespace engine::models::seed_vc
