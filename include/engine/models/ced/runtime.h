#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/ced/assets.h"

namespace engine::models::ced {

class CedViTRuntime {
public:
    CedViTRuntime(std::shared_ptr<const assets::TensorSource> source,
                  CedConfig config, core::ExecutionContext & execution,
                  assets::TensorStorageType storage, bool flash_attention);
    ~CedViTRuntime();
    CedViTRuntime(const CedViTRuntime &) = delete;
    CedViTRuntime & operator=(const CedViTRuntime &) = delete;

    std::vector<float> classify(const std::vector<float> & mel, int64_t frames);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::ced
