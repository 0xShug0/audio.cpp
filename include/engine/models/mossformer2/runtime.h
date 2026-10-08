#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/mossformer2/assets.h"

#include <memory>
#include <vector>

namespace engine::models::mossformer2 {

class MossFormer2GatedFSMNRuntime {
public:
    MossFormer2GatedFSMNRuntime(std::shared_ptr<const MossFormer2Assets> assets, core::ExecutionContext & execution);
    ~MossFormer2GatedFSMNRuntime();
    std::vector<std::vector<float>> separate(const std::vector<float> & input);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::mossformer2
