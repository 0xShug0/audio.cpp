#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <vector>

namespace engine::models::pulsevad {

class PulseVADRuntime {
public:
    PulseVADRuntime(std::shared_ptr<const assets::TensorSource> source,
                    core::ExecutionContext & context,
                    assets::TensorStorageType storage_type);
    ~PulseVADRuntime();

    std::vector<float> infer_features(const std::vector<float> & features);
    std::vector<float> extract_features(const std::vector<float> & window) const;

private:
    class Graph;
    std::unique_ptr<Graph> graph_;
    std::vector<float> window_;
    audio::SparseMelFilterbank filterbank_;
};

}  // namespace engine::models::pulsevad
