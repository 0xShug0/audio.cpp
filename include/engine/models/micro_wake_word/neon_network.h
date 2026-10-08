#pragma once

#include "engine/models/micro_wake_word/runtime.h"
#include "engine/framework/neon/execution.h"

namespace engine::models::micro_wake_word {

class NeonNetwork {
public:
    NeonNetwork(const assets::TensorSource &source, const io::json::Value &config,
                const std::vector<NetworkTensorInfo> &tensors, int threads);
    float infer(const std::vector<float> &input);
    void reset();

private:
    neon::ExecutionPlan plan_;
    neon::Shape input_shape_{1};
    int threads_;
    float output_scale_ = 1.0F;
    std::vector<neon::Tensor> states_;
    std::vector<neon::ExecutionPlan::Value> state_outputs_;
    std::vector<neon::TensorView> inputs_;
};

} // namespace engine::models::micro_wake_word
