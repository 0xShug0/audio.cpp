#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/apollo/assets.h"

#include <memory>
#include <vector>

namespace engine::models::apollo {

class ApolloRuntime {
public:
    ApolloRuntime(std::shared_ptr<const ApolloAssets> assets,
                  core::ExecutionContext & execution,
                  assets::TensorStorageType storage);
    ~ApolloRuntime();

    std::vector<float> restore(const std::vector<float> & waveform);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::apollo
