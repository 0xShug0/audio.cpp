#include "engine/community_models/latentsync/sampling.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::latentsync {

LatentSyncDdimSchedule::LatentSyncDdimSchedule(LatentSyncDdimConfig config, int64_t inference_steps) {
    if (config.training_steps < 2 || inference_steps <= 0 || inference_steps > config.training_steps ||
        config.beta_start <= 0 || config.beta_end >= 1 || config.beta_start > config.beta_end) {
        throw std::runtime_error("Invalid LatentSync DDIM schedule");
    }
    stride_ = config.training_steps / inference_steps;
    const float begin = std::sqrt(config.beta_start), end = std::sqrt(config.beta_end);
    const float increment = (end - begin) / static_cast<float>(config.training_steps - 1);
    double cumulative = 1.0;
    for (int64_t i = 0; i < config.training_steps; ++i) {
        // Match torch.linspace's endpoint-relative construction for the second half.
        const float root_beta = i < config.training_steps / 2
            ? begin + increment * static_cast<float>(i)
            : end - increment * static_cast<float>(config.training_steps - i - 1);
        const float alpha = 1.0F - root_beta * root_beta;
        cumulative *= alpha;
        alphas_.push_back(static_cast<float>(cumulative));
    }
    for (int64_t i = inference_steps; i > 0; --i) {
        const int64_t timestep = (i - 1) * stride_ + config.steps_offset;
        if (timestep < 0 || timestep >= config.training_steps) {
            throw std::runtime_error("LatentSync DDIM timestep is outside the training schedule");
        }
        timesteps_.push_back(timestep);
    }
}

void LatentSyncDdimSchedule::step(std::vector<float> & sample,
    const std::vector<float> & noise, size_t index) const {
    if (sample.size() != noise.size() || index >= timesteps_.size()) {
        throw std::runtime_error("Invalid LatentSync DDIM step input");
    }
    const int64_t timestep = timesteps_[index], previous = timestep - stride_;
    const float alpha = alphas_[static_cast<size_t>(timestep)];
    const float previous_alpha = previous >= 0 ? alphas_[static_cast<size_t>(previous)] : alphas_.front();
    const float sqrt_alpha = std::sqrt(alpha), sqrt_beta = std::sqrt(1.0F - alpha);
    const float sqrt_previous = std::sqrt(previous_alpha), direction = std::sqrt(1.0F - previous_alpha);
    for (size_t i = 0; i < sample.size(); ++i) {
        const float original = (sample[i] - sqrt_beta * noise[i]) / sqrt_alpha;
        sample[i] = sqrt_previous * original + direction * noise[i];
    }
}

std::vector<float> latentsync_timestep_features(int64_t timestep, int64_t channels, int64_t batch) {
    if (channels <= 0 || channels % 2 != 0 || batch <= 0) {
        throw std::runtime_error("Invalid LatentSync timestep embedding dimensions");
    }
    std::vector<float> result(static_cast<size_t>(channels * batch));
    const int64_t half = channels / 2;
    for (int64_t i = 0; i < half; ++i) {
        const float exponent = -std::log(10000.0F) * static_cast<float>(i) / static_cast<float>(half);
        const float angle = static_cast<float>(timestep) * std::exp(exponent);
        result[static_cast<size_t>(i)] = std::cos(angle);
        result[static_cast<size_t>(i + half)] = std::sin(angle);
    }
    for (int64_t i = 1; i < batch; ++i) {
        std::copy_n(result.begin(), channels, result.begin() + i * channels);
    }
    return result;
}

}  // namespace engine::models::latentsync
