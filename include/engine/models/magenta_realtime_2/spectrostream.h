#pragma once

#include "engine/models/magenta_realtime_2/assets.h"
#include "engine/framework/core/execution_context.h"

#include <array>
#include <memory>
#include <vector>

namespace engine::models::magenta_realtime_2 {

class SpectroStreamRuntime {
public:
    SpectroStreamRuntime(core::ExecutionContext & execution, std::shared_ptr<const MagentaAssets> assets);
    ~SpectroStreamRuntime();

    void reset();
    // At most four consecutive 25 Hz frames, returning interleaved 48 kHz stereo.
    std::vector<float> decode(const std::vector<std::array<int32_t, 12>> & codes);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::magenta_realtime_2
