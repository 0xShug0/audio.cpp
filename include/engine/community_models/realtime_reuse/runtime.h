#pragma once

#include "engine/community_models/realtime_reuse/assets.h"
#include "engine/framework/core/execution_context.h"

#include <array>
#include <memory>
#include <vector>

namespace engine::models::realtime_reuse {

struct Config {
    int exit_layer = 8;
    int lookahead_frames = 0;
};

class Runtime {
public:
    Runtime(std::shared_ptr<const Assets> assets, core::ExecutionContext & execution,
            assets::TensorStorageType storage, Config config);
    ~Runtime();
    void reset();
    std::array<std::vector<float>, 3> infer(const std::vector<float> & features,
        int64_t frames, int64_t bins, bool streaming, bool first = false);
    const Config & config() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::realtime_reuse
