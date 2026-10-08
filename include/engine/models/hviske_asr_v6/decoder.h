#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/hviske_asr_v6/assets.h"

#include <memory>
#include <vector>

namespace engine::models::hviske_asr_v6 {

struct HviskeV6DecodingOptions {
    int64_t max_tokens = 440;
    int64_t num_beams = 2;
    float length_penalty = 1.0f;
    std::vector<int32_t> control_tokens;
};

class HviskeV6Qwen3DecoderRuntime {
public:
    HviskeV6Qwen3DecoderRuntime(std::shared_ptr<const HviskeV6Assets> assets,
        core::ExecutionContext & execution, assets::TensorStorageType storage);
    ~HviskeV6Qwen3DecoderRuntime();
    std::vector<int32_t> generate(const std::vector<float> & audio, const HviskeV6DecodingOptions & options);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::hviske_asr_v6
