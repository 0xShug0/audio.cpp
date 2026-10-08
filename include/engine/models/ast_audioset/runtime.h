#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/ast_audioset/assets.h"

#include <memory>
#include <vector>

namespace engine::models::ast_audioset {

class Runtime {
public:
    Runtime(
        std::shared_ptr<const assets::TensorSource> source,
        Config config,
        core::ExecutionContext & execution,
        assets::TensorStorageType storage_type,
        bool use_flash_attention);
    ~Runtime();

    Runtime(const Runtime &) = delete;
    Runtime & operator=(const Runtime &) = delete;

    std::vector<float> classify(const std::vector<float> & features);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::ast_audioset
