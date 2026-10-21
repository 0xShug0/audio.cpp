#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace engine::models::latentsync {

struct LatentSyncDdimConfig {
    int64_t training_steps = 1000;
    float beta_start = 0.00085F;
    float beta_end = 0.012F;
    int64_t steps_offset = 1;
};

// Official epsilon-prediction DDIM schedule: scaled-linear betas, eta=0,
// no sample clipping, and final alpha equal to the first cumulative alpha.
class LatentSyncDdimSchedule {
public:
    LatentSyncDdimSchedule(LatentSyncDdimConfig config, int64_t inference_steps);
    const std::vector<int64_t> & timesteps() const noexcept { return timesteps_; }
    void step(std::vector<float> & sample, const std::vector<float> & noise, size_t index) const;

private:
    std::vector<float> alphas_;
    std::vector<int64_t> timesteps_;
    int64_t stride_;
};

std::vector<float> latentsync_timestep_features(int64_t timestep, int64_t channels, int64_t batch);

}  // namespace engine::models::latentsync
