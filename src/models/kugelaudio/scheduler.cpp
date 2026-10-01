#include "engine/models/kugelaudio/scheduler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::kugelaudio {

SpeechScheduler::SpeechScheduler(int training_steps) {
    if (training_steps <= 0) {
        throw std::runtime_error("KugelAudio diffusion training steps must be positive");
    }
    double cumulative = 1.0;
    for (int i = 0; i < training_steps; ++i) {
        const double t0 = static_cast<double>(i) / training_steps;
        const double t1 = static_cast<double>(i + 1) / training_steps;
        const double c0 = std::cos((t0 + 0.008) / 1.008 * 3.14159265358979323846 / 2);
        const double c1 = std::cos((t1 + 0.008) / 1.008 * 3.14159265358979323846 / 2);
        const float beta = static_cast<float>(std::min(1.0 - c1 * c1 / (c0 * c0), 0.999));
        cumulative *= 1.0F - beta;
        const float alpha = static_cast<float>(cumulative);
        training_sigmas_.push_back(std::sqrt((1.0F - alpha) / alpha));
    }
}

void SpeechScheduler::reset(int inference_steps) {
    if (inference_steps <= 0 || inference_steps > static_cast<int>(training_sigmas_.size())) {
        throw std::runtime_error("KugelAudio inference steps must be within the training schedule");
    }
    timesteps_.clear();
    sigmas_.clear();
    previous_prediction_.clear();
    step_ = 0;
    for (int i = inference_steps; i > 0; --i) {
        const auto t = static_cast<int64_t>(std::nearbyint(
            static_cast<double>(training_sigmas_.size() - 1) * i / inference_steps));
        timesteps_.push_back(t);
        sigmas_.push_back(training_sigmas_[static_cast<size_t>(t)]);
    }
    sigmas_.push_back(0.0F);
}

void SpeechScheduler::step(std::vector<float> & sample, const std::vector<float> & prediction,
                           const std::vector<float> & variance_noise) {
    if (step_ >= timesteps_.size() || sample.size() != prediction.size() || sample.size() != variance_noise.size()) {
        throw std::runtime_error("KugelAudio scheduler step or tensor sizes are invalid");
    }
    const float current = sigmas_[step_];
    const float next = sigmas_[step_ + 1];
    const float alpha_s = 1.0F / std::sqrt(current * current + 1.0F);
    const float sigma_s = current * alpha_s;
    const float alpha_t = 1.0F / std::sqrt(next * next + 1.0F);
    const float sigma_t = next * alpha_t;
    const float lambda_s = std::log(alpha_s) - std::log(sigma_s);
    const float h = std::log(alpha_t) - std::log(sigma_t) - lambda_s;
    const float decay = std::exp(-h);
    const float variance = 1.0F - std::exp(-2.0F * h);
    const float sample_scale = sigma_t / sigma_s * decay;
    const float prediction_scale = alpha_t * variance;
    const float noise_scale = sigma_t * std::sqrt(variance);
    float derivative_scale = 0.0F;
    const bool second_order = step_ > 0 && step_ + 1 < timesteps_.size();
    if (second_order) {
        const float previous = sigmas_[step_ - 1];
        const float alpha_previous = 1.0F / std::sqrt(previous * previous + 1.0F);
        const float sigma_previous = previous * alpha_previous;
        const float h0 = lambda_s - (std::log(alpha_previous) - std::log(sigma_previous));
        derivative_scale = 1.0F / (h0 / h);
    }
    if (!second_order) {
        previous_prediction_.resize(sample.size());
    }
    for (size_t i = 0; i < sample.size(); ++i) {
        const float x0 = alpha_s * sample[i] - sigma_s * prediction[i];
        float result = sample_scale * sample[i] + prediction_scale * x0;
        if (second_order) {
            const float derivative = derivative_scale * (x0 - previous_prediction_[i]);
            result += 0.5F * prediction_scale * derivative;
        }
        sample[i] = result + noise_scale * variance_noise[i];
        previous_prediction_[i] = x0;
    }
    ++step_;
}

}  // namespace engine::models::kugelaudio
