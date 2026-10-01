#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"

#include <vector>
#include <memory>
#include <cstddef>

namespace engine::models::kugelaudio {

struct ConnectorWeights {
    modules::LinearWeights fc1;
    modules::NormWeights norm;
    modules::LinearWeights fc2;
};

struct DiffusionLayerWeights {
    modules::NormWeights norm;
    modules::LinearWeights modulation;
    modules::LinearWeights gate;
    modules::LinearWeights up;
    modules::LinearWeights down;
};

struct DiffusionWeights {
    modules::LinearWeights input;
    modules::LinearWeights condition;
    modules::LinearWeights time1;
    modules::LinearWeights time2;
    std::vector<DiffusionLayerWeights> layers;
    modules::LinearWeights final_modulation;
    modules::LinearWeights output;
    core::TensorValue ones;
};

struct DiffusionConfig {
    int64_t hidden_size = 3584;
    int64_t latent_size = 64;
    int64_t intermediate_size = 10752;
    int64_t layers = 4;
    float norm_eps = 1e-5F;
};

ConnectorWeights load_connector_weights(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    assets::TensorStorageType storage, int64_t latent_size, int64_t hidden_size);

DiffusionWeights load_diffusion_weights(
    core::BackendWeightStore & store, const assets::TensorSource & source,
    assets::TensorStorageType storage, const DiffusionConfig & config);

core::TensorValue build_connector(
    core::ModuleBuildContext & ctx, const core::TensorValue & input,
    const ConnectorWeights & weights, int64_t hidden_size);

// time_features is the schedule's precomputed 256-channel cosine/sine embedding.
core::TensorValue build_diffusion_head(
    core::ModuleBuildContext & ctx, const core::TensorValue & latent,
    const core::TensorValue & condition, const core::TensorValue & time_features,
    const DiffusionWeights & weights, const DiffusionConfig & config);

class DiffusionHead {
public:
    DiffusionHead(const assets::TensorSource & source, core::ExecutionContext & execution,
                  assets::TensorStorageType storage, DiffusionConfig config = {});
    ~DiffusionHead();
    void begin_frame(const std::vector<float> & condition, const std::vector<int64_t> & timesteps);
    std::vector<float> predict(const std::vector<float> & latent, size_t step);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::kugelaudio
