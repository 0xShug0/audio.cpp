#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/demucs/assets.h"

#include <memory>
#include <vector>

namespace engine::models::demucs {

struct HTDemucsChunkTiming {
    double frontend_ms = 0.0;
    double graph_ms = 0.0;
    double postprocess_ms = 0.0;
    double graph_rebuild_ms = 0.0;
};

class HTDemucsPipeline {
public:
    HTDemucsPipeline(
        std::shared_ptr<const HTDemucsSubmodelAssets> assets,
        core::ExecutionContext & execution_context,
        assets::TensorStorageType weight_storage_type);
    ~HTDemucsPipeline();

    const HTDemucsConfig & config() const noexcept;
    const std::vector<float> & separate_chunk(std::vector<float> & chunk_planar);
    const HTDemucsChunkTiming & last_timing() const noexcept;

private:
    class Impl;

    std::shared_ptr<const HTDemucsSubmodelAssets> assets_;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::demucs
