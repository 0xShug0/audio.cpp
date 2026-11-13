#pragma once

#include "engine/models/mossformer2/assets.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <vector>

namespace engine::models::mossformer2 {

class MossFormer2VocoderRuntime {
public:
    MossFormer2VocoderRuntime(std::shared_ptr<const MossFormer2Assets> assets, core::ExecutionContext & execution);
    ~MossFormer2VocoderRuntime();
    std::vector<float> synthesize(const std::vector<float> & features, int64_t frames);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::mossformer2
