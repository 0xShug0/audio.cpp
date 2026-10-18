#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine::models::kugelaudio {

// Order-two SDE-DPM-Solver++, cosine VP schedule and v-prediction.
class SpeechScheduler {
public:
    explicit SpeechScheduler(int training_steps = 1000);
    void reset(int inference_steps);
    const std::vector<int64_t> & timesteps() const { return timesteps_; }
    const std::vector<float> & sigmas() const { return sigmas_; }
    void step(std::vector<float> & sample, const std::vector<float> & prediction,
              const std::vector<float> & variance_noise);

private:
    std::vector<float> training_sigmas_;
    std::vector<float> sigmas_;
    std::vector<int64_t> timesteps_;
    std::vector<float> previous_prediction_;
    size_t step_ = 0;
};

}  // namespace engine::models::kugelaudio
