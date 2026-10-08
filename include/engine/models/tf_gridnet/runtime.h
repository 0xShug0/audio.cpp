#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/tf_gridnet/assets.h"

#include <memory>
#include <vector>

namespace engine::models::tf_gridnet {

class TFGridNetBiLSTMAttentionRuntime {
public:
    TFGridNetBiLSTMAttentionRuntime(std::shared_ptr<const TFGridNetAssets> assets,
        core::ExecutionContext & execution, assets::TensorStorageType storage);
    ~TFGridNetBiLSTMAttentionRuntime();
    std::vector<std::vector<float>> separate(const std::vector<float> & interleaved);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::tf_gridnet
