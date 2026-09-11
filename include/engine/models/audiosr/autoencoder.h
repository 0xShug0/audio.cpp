#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/models/audiosr/assets.h"

#include <memory>
#include <vector>

namespace engine::models::audiosr {

struct AudioSRLatent {
    std::vector<float> values;
    int64_t channels = 0;
    int64_t height = 0;
    int64_t width = 0;
};

class AudioSRAutoencoderRuntime {
public:
    AudioSRAutoencoderRuntime(
        std::shared_ptr<const AudioSRAssets> assets,
        engine::core::ExecutionContext & execution,
        engine::assets::TensorStorageType weight_type);
    ~AudioSRAutoencoderRuntime();

    AudioSRLatent encode_condition(const std::vector<float> & mel, int64_t frames, uint32_t seed);
    std::vector<float> decode_first_stage(const AudioSRLatent & latent);
    void release_encoder_graph();
    void release_decoder_graph();
    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::audiosr
