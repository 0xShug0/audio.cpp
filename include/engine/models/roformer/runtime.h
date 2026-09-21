#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/roformer/assets.h"

#include <memory>
#include <functional>
#include <vector>

namespace engine::models::roformer {

class RoformerRuntime {
public:
    RoformerRuntime(
        std::shared_ptr<const RoformerAssets> assets,
        core::ExecutionContext & execution_context,
        assets::TensorStorageType weight_storage_type);
    ~RoformerRuntime();

    const RoformerArchitectureConfig & config() const noexcept;
    const std::vector<float> & separate_chunk(const std::vector<float> & chunk_planar);
    // Source may run on a CPU worker while sink runs on the caller thread.
    // Their captured mutable state must be independent. Sink is called in order.
    void process_chunks(size_t count,
        const std::function<void(size_t, std::vector<float> &)> & source,
        const std::function<void(size_t, const std::vector<float> &)> & sink);

private:
    class Impl;

    std::shared_ptr<const RoformerAssets> assets_;
    std::unique_ptr<Impl> impl_;
    size_t fft_threads_ = 1;
};

}  // namespace engine::models::roformer
