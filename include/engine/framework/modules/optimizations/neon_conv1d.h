#pragma once

#include "engine/framework/modules/conv_modules.h"

#include <memory>

namespace engine::modules {

// Owns packed NEON convolutions referenced by a CPU GGML graph. Keep this
// object alive until that graph is released. Other graph operations stay GGML.
class NeonConv1dGraph {
public:
    explicit NeonConv1dGraph(int threads);
    ~NeonConv1dGraph();
    core::TensorValue build(core::ModuleBuildContext & ctx, const core::TensorValue & input,
                            const Conv1dWeights & weights, const Conv1dConfig & config);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::modules
