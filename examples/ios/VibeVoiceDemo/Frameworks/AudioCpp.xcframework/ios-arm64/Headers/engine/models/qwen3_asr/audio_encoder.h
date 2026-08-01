#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/qwen3_asr/assets.h"
#include "engine/models/qwen3_asr/types.h"

#include <cstddef>
#include <memory>

namespace engine::models::qwen3_asr {

class Qwen3ASRAudioEncoderGraph;
struct Qwen3ASRAudioEncoderWeights;

class Qwen3ASRAudioEncoderRuntime {
public:
    Qwen3ASRAudioEncoderRuntime(
        std::shared_ptr<const Qwen3ASRAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        assets::TensorStorageType weight_storage_type);
    ~Qwen3ASRAudioEncoderRuntime();

    Qwen3ASRAudioEmbeddings encode(const Qwen3ASRAudioFeatures & features);

private:
    std::shared_ptr<const Qwen3ASRAssets> assets_;
    std::shared_ptr<const Qwen3ASRAudioEncoderWeights> weights_;
    core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    std::unique_ptr<Qwen3ASRAudioEncoderGraph> graph_;
};

}  // namespace engine::models::qwen3_asr
