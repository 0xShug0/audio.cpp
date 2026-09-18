#pragma once

#include "engine/community_models/piper_tts/assets.h"
#include "engine/framework/runtime/session.h"

#include <memory>
#include <vector>

namespace engine::models::piper_tts {

struct PiperTtsGenerationOptions {
    float speaking_rate = 1.0F;
    float variation = 0.667F;
    float duration_variation = 0.8F;
    uint32_t seed = 1234;
};

class PiperVitsRuntime {
public:
    PiperVitsRuntime(
        std::shared_ptr<const PiperTtsAssets> assets,
        core::BackendConfig backend_config);
    ~PiperVitsRuntime();

    runtime::AudioBuffer synthesize(
        const std::vector<int32_t> & token_ids,
        const PiperTtsGenerationOptions & options);

private:
    struct State;
    std::unique_ptr<State> state_;
};

void apply_piper_tts_edge_fade(
    std::vector<float> & samples,
    int sample_rate,
    float milliseconds = 5.0F);

}  // namespace engine::models::piper_tts
