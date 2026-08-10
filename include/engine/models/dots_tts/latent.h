#pragma once

#include "engine/models/dots_tts/assets.h"
#include "engine/framework/sampling/torch_random.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::dots_tts {

struct DotsLatentMatrix {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t dims = 0;
};

class DotsLatentCodec {
public:
    explicit DotsLatentCodec(std::shared_ptr<const DotsAssets> assets);

    DotsLatentMatrix sample_from_encoder_latents(
        const std::vector<float> & latents,
        int64_t channels,
        int64_t frames,
        uint64_t seed,
        uint64_t offset_blocks,
        const engine::sampling::TorchCudaSamplingPolicy & sampling_policy) const;

    DotsLatentMatrix normalize(const DotsLatentMatrix & input) const;
    DotsLatentMatrix denormalize(const DotsLatentMatrix & input) const;

private:
    std::shared_ptr<const DotsAssets> assets_;
    std::vector<float> mean_;
    std::vector<float> std_;
};

}  // namespace engine::models::dots_tts
