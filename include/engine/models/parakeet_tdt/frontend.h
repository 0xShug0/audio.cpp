#pragma once

#include "engine/framework/runtime/session.h"
#include "engine/models/parakeet_tdt/assets.h"

#include <memory>
#include <vector>

namespace engine::core {
class ExecutionContext;
}

namespace engine::models::parakeet_tdt {

struct ParakeetFrontendGraph;
struct ParakeetFrontendBatch {
    std::vector<float> features;
    std::vector<int32_t> attention_mask;
    std::vector<int32_t> feature_lengths;
    int64_t batch = 0;
    int64_t frames = 0;
    int64_t feature_dim = 0;
};

struct ParakeetFrontendScratch {
    ParakeetFrontendScratch();
    ~ParakeetFrontendScratch();
    ParakeetFrontendScratch(ParakeetFrontendScratch &&) noexcept;
    ParakeetFrontendScratch & operator=(ParakeetFrontendScratch &&) noexcept;
    ParakeetFrontendScratch(const ParakeetFrontendScratch &) = delete;
    ParakeetFrontendScratch & operator=(const ParakeetFrontendScratch &) = delete;

    std::vector<float> padded;
    std::vector<int64_t> audio_lengths;
    std::vector<float> power;
    std::unique_ptr<ParakeetFrontendGraph> graph;
};

void compute_parakeet_frontend(
    const std::vector<runtime::AudioBuffer> & audio,
    const std::vector<int64_t> * audio_lengths_override,
    const ParakeetFeatureExtractorConfig & config,
    const core::ExecutionContext & execution_context,
    ParakeetFrontendBatch & output,
    ParakeetFrontendScratch & scratch);

}  // namespace engine::models::parakeet_tdt
