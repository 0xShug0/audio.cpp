#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/models/universr/assets.h"

#include <string>

namespace engine::models::universr {

class UniverSRRuntime {
public:
    UniverSRRuntime(std::shared_ptr<const UniverSRAssets> assets,
        core::ExecutionContext & execution, assets::TensorStorageType storage);
    ~UniverSRRuntime();

    void prepare_condition(const std::vector<float> & low_spectrum, int64_t frames, int sample_rate_khz);
    std::vector<float> vector_field(const std::vector<float> & noise, float time, bool conditional);
    std::vector<float> integrate_flow(const std::vector<float> & initial_noise, int steps,
        const std::string & method, float guidance_scale);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::universr
