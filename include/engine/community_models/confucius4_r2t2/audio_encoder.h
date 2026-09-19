#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/community_models/confucius4_r2t2/assets.h"
#include "engine/community_models/confucius4_r2t2/types.h"

#include <cstddef>
#include <memory>

namespace engine::community_models::confucius4_r2t2 {

class R2T2ASRAudioEncoderGraph;
struct R2T2ASRAudioEncoderWeights;

class R2T2ASRAudioEncoderRuntime {
public:
    R2T2ASRAudioEncoderRuntime(
        std::shared_ptr<const R2T2ASRAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        assets::TensorStorageType weight_storage_type);
    ~R2T2ASRAudioEncoderRuntime();

    // Actual retained graph capacity, including capacity retained after shrink.
    int64_t graph_capacity_frames() const noexcept { return graph_capacity_frames_; }

    R2T2ASRAudioEmbeddings encode(const R2T2ASRAudioFeatures & features, bool reuse_graph = false);

private:
    std::shared_ptr<const R2T2ASRAssets> assets_;
    std::shared_ptr<const R2T2ASRAudioEncoderWeights> weights_;
    core::ExecutionContext * execution_ = nullptr;
    size_t graph_arena_bytes_ = 0;
    int64_t graph_capacity_frames_ = 0;
    std::unique_ptr<R2T2ASRAudioEncoderGraph> graph_;
};

}  // namespace engine::community_models::confucius4_r2t2
