#pragma once

#include "engine/models/ming3_audio/assets.h"
#include "engine/framework/core/execution_context.h"

namespace engine::models::ming3_audio {

struct TalkerPrompt {
    std::vector<int32_t> ids;
    int64_t speaker_position = -1;
    int64_t latent_position = -1;
};

struct TalkerGenerationOptions {
    uint64_t seed = 42;
    int64_t max_tokens = 1000;
    float guidance_scale = 2.0f;
    float sigma = 0.25f;
    float temperature = 0.0f;
};

class Talker {
public:
    Talker(core::ExecutionContext & execution, std::shared_ptr<const assets::TensorSource> source,
           const TalkerConfig & config);
    ~Talker();

    std::vector<float> prepare_inputs(const TalkerPrompt & prompt, const std::vector<float> & speaker = {},
                                      const std::vector<float> & reference_latents = {});
    std::vector<float> generate(const std::vector<float> & embeddings, const TalkerGenerationOptions & options,
                                uint64_t & random_offset, const std::vector<float> & initial_history = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::ming3_audio
