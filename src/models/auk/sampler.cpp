#include "engine/models/auk/sampler.h"

#include <cmath>
#include <stdexcept>

namespace engine::models::auk {

std::vector<float> build_time_grid(int64_t steps, bool use_sway, float sway_coef) {
    if (steps <= 0) {
        throw std::runtime_error("AuK sampler needs at least one step");
    }
    std::vector<float> grid(static_cast<size_t>(steps) + 1, 0.0F);
    for (int64_t index = 0; index <= steps; ++index) {
        grid[static_cast<size_t>(index)] = static_cast<float>(index) / static_cast<float>(steps);
    }
    if (use_sway) {
        for (auto & value : grid) {
            const double t = value;
            value = static_cast<float>(t + sway_coef * (std::cos(M_PI / 2.0 * t) - 1.0 + t));
        }
    }
    return grid;
}

AukSamplerResult sample_latents(
    AukDit & dit,
    const AukDitInputs & conditioning,
    const std::vector<float> & noise,
    const AukSamplerOptions & options) {
    const int64_t latent_dim = static_cast<int64_t>(noise.size()) / std::max<int64_t>(conditioning.gen_frames, 1);
    if (conditioning.gen_frames <= 0 || latent_dim <= 0) {
        throw std::runtime_error("AuK sampler received an empty target");
    }

    const auto grid = build_time_grid(options.steps, options.use_sway, options.sway_coef);
    const bool guided = options.cfg_strength >= 1e-5F;

    AukDitInputs conditional = conditioning;
    conditional.drop_text = false;
    conditional.noised = noise;

    // The unconditional branch drops BOTH the text and the reference audio. The
    // reference is zeroed at the input, not at the embedding, because the embedding
    // carries biases.
    AukDitInputs unconditional = conditional;
    unconditional.drop_text = true;
    unconditional.reference.assign(conditioning.reference.size(), 0.0F);

    std::vector<float> state = noise;
    for (size_t step = 0; step + 1 < grid.size(); ++step) {
        const float t = grid[step];
        const float dt = grid[step + 1] - grid[step];

        conditional.timestep = t;
        conditional.noised = state;
        auto cond = dit.forward(conditional).velocity;

        if (guided) {
            unconditional.timestep = t;
            unconditional.noised = state;
            const auto uncond = dit.forward(unconditional).velocity;
            for (size_t index = 0; index < cond.size(); ++index) {
                cond[index] = cond[index] + (cond[index] - uncond[index]) * options.cfg_strength;
            }
        }
        for (size_t index = 0; index < state.size(); ++index) {
            state[index] += dt * cond[index];
        }
    }

    AukSamplerResult result;
    result.latent_dim = latent_dim;
    result.frames = conditioning.ref_frames + conditioning.gen_frames;
    result.time_grid = grid;
    result.latents.reserve(static_cast<size_t>(result.frames * latent_dim));
    result.latents.insert(result.latents.end(), conditioning.reference.begin(), conditioning.reference.end());
    result.latents.insert(result.latents.end(), state.begin(), state.end());
    return result;
}

}  // namespace engine::models::auk
