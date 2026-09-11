#pragma once

#include "engine/community_models/inflect_v2/assets.h"
#include "engine/framework/runtime/session.h"

#include <memory>
#include <vector>

namespace engine::models::inflect_v2 {

struct InflectV2GenerationOptions {
    float speaking_rate = 1.0F;
    float variation = 0.667F;
    uint32_t seed = 0;
};

class InflectV2NativeRuntime {
public:
    InflectV2NativeRuntime(
        std::shared_ptr<const InflectV2Assets> assets,
        core::BackendConfig backend_config);
    ~InflectV2NativeRuntime();

    runtime::AudioBuffer synthesize(
        const std::vector<int32_t> & token_ids,
        const InflectV2GenerationOptions & options);

private:
    struct State;
    std::unique_ptr<State> state_;
};

void apply_inflect_v2_edge_fade(
    std::vector<float> & samples,
    int sample_rate,
    float milliseconds = 5.0F);

}  // namespace engine::models::inflect_v2
