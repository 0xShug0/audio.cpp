#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/audiosr/assets.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace engine::models::audiosr {

struct AudioSROptions {
    int num_inference_steps = 50;
    float guidance_scale = 3.5F;
    float ddim_eta = 1.0F;
    float audio_chunk_duration_sec = 15.0F;
    float audio_chunk_overlap_sec = 2.0F;
    uint32_t seed = 42;
};

class AudioSRPipelineRuntime {
public:
    AudioSRPipelineRuntime(
        std::shared_ptr<const AudioSRAssets> assets,
        engine::core::ExecutionContext & execution,
        engine::assets::TensorStorageType weight_type);
    ~AudioSRPipelineRuntime();

    engine::runtime::AudioBuffer run(
        const engine::runtime::AudioBuffer & input,
        const AudioSROptions & options);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

AudioSROptions parse_audiosr_options(const std::unordered_map<std::string, std::string> & options);

}  // namespace engine::models::audiosr
