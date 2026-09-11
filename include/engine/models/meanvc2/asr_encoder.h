#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/meanvc2/audio_pipeline.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::meanvc2 {

struct MeanVC2AsrEncoderWeights;
struct MeanVC2AsrEncoderGraph;

class MeanVC2AsrEncoderRuntime final {
public:
    MeanVC2AsrEncoderRuntime(
        std::shared_ptr<const engine::assets::TensorSource> source,
        engine::core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        engine::assets::TensorStorageType weight_storage_type);
    ~MeanVC2AsrEncoderRuntime();

    void reset();
    std::vector<float> encode_windows(const std::vector<MeanVC2FbankWindow> & windows);

private:
    engine::core::ExecutionContext & execution_context_;
    size_t graph_context_bytes_ = 0;
    std::shared_ptr<const engine::assets::TensorSource> source_;
    std::shared_ptr<const MeanVC2AsrEncoderWeights> weights_;
    mutable std::unique_ptr<MeanVC2AsrEncoderGraph> graph_;
    std::vector<float> attention_cache_;
    std::vector<float> conv_cache_;
};

}  // namespace engine::models::meanvc2
