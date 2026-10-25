#pragma once

#include "engine/framework/modules/linear_module.h"
#include "engine/framework/sampling/flow_sampler_runtime.h"
#include "engine/models/irodori_tts/assets.h"
#include "engine/models/irodori_tts/types.h"

#include <array>
#include <vector>

namespace engine::models::irodori_tts {

core::TensorValue build_irodori_meanflow_conditioning(
    core::ModuleBuildContext &ctx, const core::TensorValue &interval_embedding,
    const core::TensorValue &time_condition,
    const std::array<modules::LinearWeights, 3> &weights,
    const IrodoriModelConfig &config);

std::vector<float> sample_irodori_meanflow(
    const IrodoriGenerationOptions &options, int64_t patched_steps,
    int64_t patched_dim, bool mem_saver,
    modules::FlowSamplerSingleBranchDenoiserConfig denoiser);

} // namespace engine::models::irodori_tts
