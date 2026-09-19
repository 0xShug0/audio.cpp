#pragma once

// AuK's conditional flow-matching sampler: an Euler ODE over the DiT's velocity
// field, with classifier-free guidance (cfm_edit.py:143).
//
// The integration is ordinary Euler -- torchdiffeq's `method="euler"`, i.e.
// x += dt * v -- over a time grid that is uniform unless sway sampling warps it.

#include "engine/models/auk/dit.h"

#include <cstdint>
#include <vector>

namespace engine::models::auk {

struct AukSamplerOptions {
    int64_t steps = 32;
    float cfg_strength = 2.0F;
    // ⚠ Applied to the WHOLE grid including its endpoints:
    //   t += coef * (cos(pi/2 * t) - 1 + t)
    // It concentrates steps near t=0. Absent means a uniform grid.
    bool use_sway = false;
    float sway_coef = -1.0F;
};

struct AukSamplerResult {
    std::vector<float> latents;   // [ref_frames + gen_frames, latent_dim], ref prepended
    int64_t frames = 0;
    int64_t latent_dim = 0;
    std::vector<float> time_grid;
};

// `noise` is the caller's y0, [gen_frames, latent_dim]. It is an argument rather than
// drawn here because torch's RNG cannot be reproduced: a parity test injects the
// reference's own draw, and only then is a difference attributable to the sampler.
AukSamplerResult sample_latents(
    AukDit & dit,
    const AukDitInputs & conditioning,
    const std::vector<float> & noise,
    const AukSamplerOptions & options);

std::vector<float> build_time_grid(int64_t steps, bool use_sway, float sway_coef);

}  // namespace engine::models::auk
