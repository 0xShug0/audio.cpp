#pragma once

#include "engine/models/mossformer2/assets.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <vector>

namespace engine::models::mossformer2 {

class MossFormer2RestorationRuntime {
public:
    MossFormer2RestorationRuntime(std::shared_ptr<const MossFormer2Assets> assets, core::ExecutionContext & execution);
    ~MossFormer2RestorationRuntime();
    std::vector<float> restore(const std::vector<float> & input, int64_t window, int64_t overlap);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::mossformer2
