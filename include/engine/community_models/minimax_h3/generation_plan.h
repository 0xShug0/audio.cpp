#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/community_models/minimax_h3/assets.h"
#include "engine/community_models/minimax_h3/types.h"

#include <cstdint>
#include <vector>

namespace engine::models::minimax_h3 {

using H3Config = MiniMaxH3Config;

struct InitialLatents {
    std::vector<float> video;
    std::vector<float> audio;
};

std::vector<float> h3_sigmas(int64_t steps, float shift);
std::vector<float> build_step_timestep_features(
    const H3Config & cfg,
    const std::vector<float> & audio_sigmas,
    const std::vector<float> & video_sigmas);
std::vector<float> build_step_timestep_values(
    const H3Config & cfg,
    const std::vector<float> & audio_sigmas,
    const std::vector<float> & video_sigmas);
std::vector<float> build_step_adaln_curve_inputs(
    const H3Config & cfg,
    const std::vector<float> & timestep_values,
    const std::vector<float> & table);
InitialLatents generate_initial_latents(
    const H3Config & cfg,
    const engine::core::ExecutionContext & execution,
    uint32_t seed);
H3Config resolve_generation_config(const H3Config & base, const MiniMaxH3GenerateRequest & request);

}  // namespace engine::models::minimax_h3
