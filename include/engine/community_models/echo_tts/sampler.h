#pragma once

#include "engine/community_models/echo_tts/config.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace engine::models::echo_tts {

// One denoiser evaluation. `x` is (sequence_length * latent_size) and `lanes` is
// 1 (conditional only) or 3 (cond, text-uncond, speaker-uncond, concatenated
// along the batch axis). The result holds `lanes` velocity fields of the same
// per-lane size.
using EchoDenoiseFn = std::function<std::vector<float>(
    const std::vector<float> & x, float t, int lanes)>;

// Returns the timestep schedule used by
// inference.py::sample_euler_cfg_independent_guidances:
//     linspace(1, 0, num_steps + 1) * 0.999
// The 0.999 scale exists so that temporal rescaling can be applied on the first
// step; it is not a rounding artifact and changes the trajectory if dropped.
std::vector<float> euler_timestep_schedule(int num_steps);

// True when classifier-free guidance is active at timestep t. Mirrors the
// upstream inclusive comparison on both ends.
bool cfg_active(float t, float cfg_min_t, float cfg_max_t);

// Combines the three CFG lanes into a single velocity, following upstream's
// independent-guidance form:
//     v = v_cond
//         + w_text    * (v_cond - v_uncond_text)
//         + w_speaker * (v_cond - v_uncond_speaker)
std::vector<float> combine_cfg_lanes(
    const std::vector<float> & lanes,
    int64_t lane_elements,
    float cfg_scale_text,
    float cfg_scale_speaker);

// Runs the sampler loop. `denoise` supplies the model evaluation and
// `initial_noise` the starting latent, both injected so this can be tested
// without a backend. `on_kv_rescale`, when set, is invoked at the timestep where
// upstream undoes speaker KV scaling.
std::vector<float> run_euler_sampler(
    const EchoSamplerOptions & options,
    int64_t sequence_length,
    int64_t latent_size,
    std::vector<float> initial_noise,
    const EchoDenoiseFn & denoise,
    const std::function<void()> & on_kv_rescale = {});

}  // namespace engine::models::echo_tts
