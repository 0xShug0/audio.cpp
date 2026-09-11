#pragma once

#include "engine/models/audiosr/assets.h"
#include "engine/models/audiosr/autoencoder.h"
#include "engine/models/audiosr/unet.h"
#include "engine/framework/sampling/torch_random.h"

#include <cstdint>
#include <vector>

namespace engine::models::audiosr {

class AudioSRDdimSampler {
public:
    AudioSRDdimSampler(AudioSRConfig config, engine::sampling::TorchCudaSamplingPolicy rng_policy);

    AudioSRLatent sample(
        AudioSRUnetRuntime & unet,
        const AudioSRLatent & condition,
        const AudioSRLatent & unconditional_condition,
        int64_t steps,
        float guidance_scale,
        float eta,
        uint32_t seed) const;

    AudioSRLatent sample(
        AudioSRUnetRuntime & unet,
        const AudioSRLatent & condition,
        const AudioSRLatent & unconditional_condition,
        int64_t steps,
        float guidance_scale,
        float eta,
        uint32_t seed,
        uint64_t & rng_offset_blocks) const;

private:
    AudioSRConfig config_;
    engine::sampling::TorchCudaSamplingPolicy rng_policy_;
};

}  // namespace engine::models::audiosr
