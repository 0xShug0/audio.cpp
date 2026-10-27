#pragma once

#include "engine/models/magenta_realtime_2/assets.h"
#include "engine/framework/core/execution_context.h"

#include <array>
#include <memory>
#include <vector>

namespace engine::models::magenta_realtime_2 {

struct MagentaSamplingOptions {
    float temperature = 1.3F;
    int top_k = 40;
    uint32_t seed = 0;
};

class DepthformerRuntime {
public:
    DepthformerRuntime(core::ExecutionContext & execution, std::shared_ptr<const MagentaAssets> assets,
                      engine::assets::TensorStorageType storage);
    ~DepthformerRuntime();

    void reset(const MagentaSamplingOptions & options);
    // Inputs use the upstream unique-token indexing, including its seven reserved values.
    void condition(const std::array<int32_t, 12> & style, const std::array<int32_t, 132> & controls);
    std::array<int32_t, 12> step();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::magenta_realtime_2
