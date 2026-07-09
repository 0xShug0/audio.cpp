#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/hviske_asr/assets.h"
#include "engine/models/hviske_asr/frontend.h"
#include "engine/models/hviske_asr/weights.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::hviske_asr {

struct HviskeEncodedAudio {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t valid_frames = 0;
    int64_t hidden_size = 0;
};

class HviskeEncoderRuntime {
public:
    HviskeEncoderRuntime(
        std::shared_ptr<const HviskeAssets> assets,
        std::shared_ptr<const HviskeWeights> weights,
        engine::core::ExecutionContext & execution_context,
        size_t graph_arena_bytes);
    ~HviskeEncoderRuntime();

    HviskeEncoderRuntime(const HviskeEncoderRuntime &) = delete;
    HviskeEncoderRuntime & operator=(const HviskeEncoderRuntime &) = delete;
    HviskeEncoderRuntime(HviskeEncoderRuntime &&) noexcept;
    HviskeEncoderRuntime & operator=(HviskeEncoderRuntime &&) noexcept;

    void prepare_capacity(int64_t input_frames, int64_t input_features);
    HviskeEncodedAudio encode(const HviskeFrontendFeatures & features);

private:
    struct Graph;

    void ensure_graph(int64_t input_frames, int64_t input_features);

    std::shared_ptr<const HviskeAssets> assets_;
    std::shared_ptr<const HviskeWeights> weights_;
    engine::core::ExecutionContext * execution_context_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::unique_ptr<Graph> graph_;
    std::vector<float> time_major_input_;
    std::vector<float> output_scratch_;
};

}  // namespace engine::models::hviske_asr
