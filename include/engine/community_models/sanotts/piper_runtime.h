#pragma once

#include "engine/community_models/sanotts/assets.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::sanotts {

struct SanoTtsPiperGenerationOptions {
    /** Multiplier on the voice's tuned duration_length_scale; larger is
     *  slower. The decoder is deterministic -- there is no seed. */
    float speaking_rate = 1.0F;
};

class SanoTtsPiperRuntime {
public:
    SanoTtsPiperRuntime(
        std::shared_ptr<const SanoTtsAssets> assets,
        core::BackendConfig backend_config);
    ~SanoTtsPiperRuntime();

    runtime::AudioBuffer synthesize(
        const std::vector<int32_t> & token_ids,
        const SanoTtsPiperGenerationOptions & options);

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace engine::models::sanotts
