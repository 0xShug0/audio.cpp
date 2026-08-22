#include "engine/models/audiosr/ddim.h"

#include "engine/framework/sampling/torch_random.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace engine::models::audiosr {
namespace {

constexpr double kCosineS = 8.0e-3;
constexpr double kPi = 3.141592653589793238462643383279502884;

std::vector<float> make_cosine_alphas_cumprod(int64_t timesteps) {
    if (timesteps <= 0) {
        throw std::runtime_error("AudioSR diffusion timesteps must be positive");
    }
    std::vector<double> alphas(static_cast<size_t>(timesteps + 1));
    for (int64_t i = 0; i <= timesteps; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(timesteps) + kCosineS;
        const double value = std::cos(t / (1.0 + kCosineS) * kPi * 0.5);
        alphas[static_cast<size_t>(i)] = value * value;
    }
    const double first = alphas.front();
    std::vector<float> out(static_cast<size_t>(timesteps));
    for (int64_t i = 0; i < timesteps; ++i) {
        out[static_cast<size_t>(i)] = static_cast<float>(alphas[static_cast<size_t>(i + 1)] / first);
    }
    return out;
}

std::vector<int64_t> make_ddim_timesteps(int64_t steps, int64_t timesteps) {
    if (steps <= 0 || timesteps <= 0) {
        throw std::runtime_error("AudioSR DDIM step counts must be positive");
    }
    const int64_t stride = timesteps / steps;
    if (stride <= 0) {
        throw std::runtime_error("AudioSR DDIM steps exceed diffusion timesteps");
    }
    std::vector<int64_t> out;
    for (int64_t t = 0; t < timesteps; t += stride) {
        out.push_back(t + 1);
    }
    if (static_cast<int64_t>(out.size()) != steps) {
        throw std::runtime_error("AudioSR DDIM uniform schedule size mismatch");
    }
    return out;
}

struct DdimSchedule {
    std::vector<float> alphas;
    std::vector<float> alphas_prev;
    std::vector<float> sqrt_one_minus_alphas;
    std::vector<float> sigmas;
    std::vector<int64_t> timesteps;
};

DdimSchedule make_schedule(int64_t steps, int64_t timesteps, float eta) {
    auto alphas_cumprod = make_cosine_alphas_cumprod(timesteps);
    auto selected = make_ddim_timesteps(steps, timesteps);
    DdimSchedule schedule;
    schedule.timesteps = selected;
    schedule.alphas.resize(selected.size());
    schedule.alphas_prev.resize(selected.size());
    schedule.sqrt_one_minus_alphas.resize(selected.size());
    schedule.sigmas.resize(selected.size());
    for (size_t i = 0; i < selected.size(); ++i) {
        const int64_t step = selected[i];
        if (step < 0 || step >= static_cast<int64_t>(alphas_cumprod.size())) {
            throw std::runtime_error("AudioSR DDIM timestep is outside diffusion schedule");
        }
        const float alpha = alphas_cumprod[static_cast<size_t>(step)];
        const float alpha_prev = i == 0
            ? alphas_cumprod.front()
            : alphas_cumprod[static_cast<size_t>(selected[i - 1])];
        schedule.alphas[i] = alpha;
        schedule.alphas_prev[i] = alpha_prev;
        schedule.sqrt_one_minus_alphas[i] = std::sqrt(std::max(0.0F, 1.0F - alpha));
        const float sigma_arg =
            (1.0F - alpha_prev) / std::max(1.0F - alpha, 1.0e-20F) *
            (1.0F - alpha / std::max(alpha_prev, 1.0e-20F));
        schedule.sigmas[i] = eta * std::sqrt(std::max(0.0F, sigma_arg));
    }
    return schedule;
}

void ensure_same_latent_shape(const AudioSRLatent & lhs, const AudioSRLatent & rhs, const char * what) {
    if (lhs.channels != rhs.channels || lhs.height != rhs.height || lhs.width != rhs.width ||
        lhs.values.size() != rhs.values.size()) {
        throw std::runtime_error(std::string("AudioSR DDIM latent shape mismatch for ") + what);
    }
}

}  // namespace

AudioSRDdimSampler::AudioSRDdimSampler(AudioSRConfig config, engine::sampling::TorchCudaSamplingPolicy rng_policy)
    : config_(config), rng_policy_(rng_policy) {}

AudioSRLatent AudioSRDdimSampler::sample(
    AudioSRUnetRuntime & unet,
    const AudioSRLatent & condition,
    const AudioSRLatent & unconditional_condition,
    int64_t steps,
    float guidance_scale,
    float eta,
    uint32_t seed) const {
    uint64_t rng_offset_blocks = 0;
    return sample(
        unet,
        condition,
        unconditional_condition,
        steps,
        guidance_scale,
        eta,
        seed,
        rng_offset_blocks);
}

AudioSRLatent AudioSRDdimSampler::sample(
    AudioSRUnetRuntime & unet,
    const AudioSRLatent & condition,
    const AudioSRLatent & unconditional_condition,
    int64_t steps,
    float guidance_scale,
    float eta,
    uint32_t seed,
    uint64_t & rng_offset_blocks) const {
    if (condition.channels <= 0 || condition.height <= 0 || condition.width <= 0) {
        throw std::runtime_error("AudioSR DDIM condition shape is invalid");
    }
    if (condition.channels != unconditional_condition.channels ||
        condition.height != unconditional_condition.height ||
        condition.width != unconditional_condition.width) {
        throw std::runtime_error("AudioSR DDIM condition and unconditional condition shape mismatch");
    }
    if (!std::isfinite(eta) || eta < 0.0F) {
        throw std::runtime_error("AudioSR DDIM eta must be finite and non-negative");
    }
    const auto schedule = make_schedule(steps, config_.diffusion_timesteps, eta);
    AudioSRLatent latent;
    latent.channels = config_.latent_channels;
    latent.height = condition.height;
    latent.width = condition.width;
    const size_t latent_count = static_cast<size_t>(latent.channels * latent.height * latent.width);
    latent.values = engine::sampling::generate_torch_cuda_tensor_iterator_randn(
        latent_count,
        seed,
        rng_offset_blocks,
        rng_policy_,
        engine::sampling::TorchRandnPrecision::Float32);
    rng_offset_blocks += engine::sampling::torch_cuda_tensor_iterator_offset_blocks(
        static_cast<uint64_t>(latent_count),
        rng_policy_);

    for (int64_t i = 0; i < steps; ++i) {
        const int64_t schedule_index = steps - i - 1;
        const int64_t timestep = schedule.timesteps[static_cast<size_t>(schedule_index)];
        auto cond_v = unet.predict_v(latent, timestep, condition);
        auto uncond_v = unet.predict_v(latent, timestep, unconditional_condition);
        if (cond_v.size() != latent_count || uncond_v.size() != latent_count) {
            throw std::runtime_error("AudioSR DDIM UNet output size mismatch");
        }

        AudioSRLatent next;
        next.channels = latent.channels;
        next.height = latent.height;
        next.width = latent.width;
        next.values.resize(latent_count);
        const float alpha = schedule.alphas[static_cast<size_t>(schedule_index)];
        const float alpha_prev = schedule.alphas_prev[static_cast<size_t>(schedule_index)];
        const float sigma = schedule.sigmas[static_cast<size_t>(schedule_index)];
        const float sqrt_alpha = std::sqrt(std::max(0.0F, alpha));
        const float sqrt_alpha_prev = std::sqrt(std::max(0.0F, alpha_prev));
        const float sqrt_one_minus_alpha = schedule.sqrt_one_minus_alphas[static_cast<size_t>(schedule_index)];
        const float dir_scale = std::sqrt(std::max(0.0F, 1.0F - alpha_prev - sigma * sigma));
        std::vector<float> noise;
        if (sigma != 0.0F) {
            noise = engine::sampling::generate_torch_cuda_tensor_iterator_randn(
                latent_count,
                seed,
                rng_offset_blocks,
                rng_policy_,
                engine::sampling::TorchRandnPrecision::Float32);
            rng_offset_blocks += engine::sampling::torch_cuda_tensor_iterator_offset_blocks(
                static_cast<uint64_t>(latent_count),
                rng_policy_);
        }

        for (size_t j = 0; j < latent_count; ++j) {
            const float model_v = uncond_v[j] + guidance_scale * (cond_v[j] - uncond_v[j]);
            const float eps = sqrt_alpha * model_v + sqrt_one_minus_alpha * latent.values[j];
            const float pred_x0 = sqrt_alpha * latent.values[j] - sqrt_one_minus_alpha * model_v;
            const float noise_term = sigma == 0.0F ? 0.0F : sigma * noise[j];
            next.values[j] = sqrt_alpha_prev * pred_x0 + dir_scale * eps + noise_term;
        }
        ensure_same_latent_shape(latent, next, "step update");
        latent = std::move(next);
    }
    return latent;
}

}  // namespace engine::models::audiosr
