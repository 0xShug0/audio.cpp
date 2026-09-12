#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/xtts_v2/assets.h"
#include "engine/models/xtts_v2/audio_features.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::xtts_v2 {

struct XttsV2ConditioningLatent {
    std::vector<float> values;  // frame-major [32, 1024]
    int64_t frames = 32;
    int64_t dims = 1024;
};

class XttsV2ConditioningRuntime {
public:
    XttsV2ConditioningRuntime(
        const XttsV2Assets & assets,
        core::ExecutionContext & execution,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        assets::TensorStorageType matmul_storage_type,
        assets::TensorStorageType conv_storage_type);
    ~XttsV2ConditioningRuntime();

    XttsV2ConditioningLatent encode(const XttsV2MelFeatures & mel);
    const std::vector<float> & mel_stats() const noexcept;

private:
    struct Weights;
    class Graph;
    core::ExecutionContext & execution_;
    size_t graph_context_bytes_ = 0;
    std::shared_ptr<const Weights> weights_;
    std::unique_ptr<Graph> graph_;
};

}  // namespace engine::models::xtts_v2
