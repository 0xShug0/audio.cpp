#pragma once

#include "engine/framework/neon/execution.h"
#include "engine/models/moonshine_asr/config.h"
#include "engine/models/moonshine_asr/neon_weights.h"

#include <array>

namespace engine::models::moonshine_asr {

class MoonshineNeonDecoder {
public:
    MoonshineNeonDecoder(MoonshineDecoderConfig config, MoonshineNeonDecoderWeights weights);
    void prepare_memory(const std::vector<float> &memory, int capacity, int threads);
    const std::vector<float> &run_step(int32_t token, int32_t position, int threads);

private:
    struct Cache {
        std::vector<float> key, value;
    };
    MoonshineDecoderConfig config_;
    neon::ExecutionPlan cross_plan_, step_plan_;
    std::vector<std::array<neon::ExecutionPlan::Value, 2>> cross_results_, self_results_;
    std::vector<Cache> self_cache_;
    std::vector<neon::TensorView> step_inputs_;
    int32_t position_ = 0;
    int32_t token_ = 1;
    int next_position_ = 0, capacity_ = 0;
};

} // namespace engine::models::moonshine_asr
