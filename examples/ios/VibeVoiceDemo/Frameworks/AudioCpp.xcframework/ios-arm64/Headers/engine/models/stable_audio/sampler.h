#pragma once

#include "engine/framework/sampling/torch_random.h"
#include "engine/models/stable_audio/assets.h"
#include "engine/models/stable_audio/types.h"

#include <cstdint>
#include <vector>

namespace engine::models::stable_audio {

struct StableAudioSamplingState {
    int64_t audio_sample_size = 0;
    int64_t latent_sample_size = 0;
    int64_t batch = 0;
    int64_t channels = 0;
    int64_t schedule_points = 0;
    std::vector<float> schedule;
    std::vector<float> padding_mask;
    std::vector<float> noise;
    std::vector<float> local_add_conditioning;
};

StableAudioSamplingState prepare_stable_audio_sampling_state(
    const StableAudioConfig & config,
    const StableAudioRequest & request,
    const engine::sampling::TorchCudaSamplingPolicy & rng_policy);

}  // namespace engine::models::stable_audio
