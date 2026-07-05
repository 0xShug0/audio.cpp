#pragma once

#include "engine/models/parakeet_tdt/assets.h"
#include "engine/models/parakeet_tdt/frontend.h"
#include "engine/models/parakeet_tdt/weights.h"

#include <memory>
#include <vector>

namespace engine::core {
class ExecutionContext;
}

namespace engine::models::parakeet_tdt {

struct ParakeetPreEncodeLinearGraph;

struct ParakeetPreEncodeBatch {
    std::vector<float> hidden;
    int64_t frames = 0;
    int64_t valid_frames = 0;
};

struct ParakeetPreEncodeScratch {
    std::unique_ptr<ParakeetPreEncodeLinearGraph> graph;

    ParakeetPreEncodeScratch();
    ~ParakeetPreEncodeScratch();
    ParakeetPreEncodeScratch(ParakeetPreEncodeScratch && other) noexcept;
    ParakeetPreEncodeScratch & operator=(ParakeetPreEncodeScratch && other) noexcept;
    ParakeetPreEncodeScratch(const ParakeetPreEncodeScratch &) = delete;
    ParakeetPreEncodeScratch & operator=(const ParakeetPreEncodeScratch &) = delete;
};

void compute_parakeet_pre_encode(
    const ParakeetFrontendBatch & frontend,
    const ParakeetAssets & assets,
    const ParakeetTDTWeights & weights,
    engine::core::ExecutionContext & execution_context,
    ParakeetPreEncodeBatch & output,
    ParakeetPreEncodeScratch & scratch);

std::vector<float> compute_parakeet_relative_positional_encoding(
    int64_t batch,
    int64_t hidden_size,
    int64_t frames,
    int64_t max_position_embeddings);

}  // namespace engine::models::parakeet_tdt
