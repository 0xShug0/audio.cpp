#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/community_models/minimax_h3/types.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace engine::models::minimax_h3 {

struct MiniMaxH3SamplerTargetSpec {
    engine::core::TensorShape packed_state_shape;
    engine::core::TensorShape prediction_shape;
    int64_t active_row_start = 0;
};

struct MiniMaxH3SamplerGraphSpec {
    MiniMaxH3SamplerTargetSpec primary;
    MiniMaxH3SamplerTargetSpec secondary;
};

struct MiniMaxH3SamplerInput {
    ggml_tensor * primary_state = nullptr;
    ggml_tensor * secondary_state = nullptr;
    ggml_tensor * primary_velocity = nullptr;
    ggml_tensor * secondary_velocity = nullptr;
    float primary_sigma = 0.0F;
    float secondary_sigma = 0.0F;
    float primary_sigma_delta = 0.0F;
    float secondary_sigma_delta = 0.0F;
};

struct MiniMaxH3SamplerOutput {
    std::vector<float> next_primary;
    std::vector<float> next_secondary;
};

class MiniMaxH3SamplerGraph {
public:
    MiniMaxH3SamplerGraph(
        engine::core::ExecutionContext & execution,
        const MiniMaxH3SamplerGraphSpec & spec,
        MiniMaxH3SamplerMode mode);
    ~MiniMaxH3SamplerGraph();

    void run(const MiniMaxH3SamplerInput & input, MiniMaxH3SamplerOutput & output);

    double device_copy_ms() const;
    double graph_compute_ms() const;
    double output_read_ms() const;
    double history_update_ms() const;

private:
    engine::core::ExecutionContext & execution_;
    const MiniMaxH3SamplerGraphSpec spec_;
    const MiniMaxH3SamplerMode mode_;
    ggml_context * ctx_ = nullptr;
    ggml_context * input_ctx_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    engine::core::HostGraphPlan plan_;
    engine::core::TensorValue primary_state_t_;
    engine::core::TensorValue secondary_state_t_;
    engine::core::TensorValue primary_velocity_t_;
    engine::core::TensorValue secondary_velocity_t_;
    engine::core::TensorValue previous_primary_denoised_t_;
    engine::core::TensorValue previous_secondary_denoised_t_;
    engine::core::TensorValue previous_primary_velocity_t_;
    engine::core::TensorValue previous_secondary_velocity_t_;
    engine::core::TensorValue primary_sigma_t_;
    engine::core::TensorValue secondary_sigma_t_;
    engine::core::TensorValue primary_state_coeff_t_;
    engine::core::TensorValue primary_velocity_coeff_t_;
    engine::core::TensorValue primary_denoised_coeff_t_;
    engine::core::TensorValue primary_previous_denoised_coeff_t_;
    engine::core::TensorValue primary_previous_velocity_coeff_t_;
    engine::core::TensorValue secondary_state_coeff_t_;
    engine::core::TensorValue secondary_velocity_coeff_t_;
    engine::core::TensorValue secondary_denoised_coeff_t_;
    engine::core::TensorValue secondary_previous_denoised_coeff_t_;
    engine::core::TensorValue secondary_previous_velocity_coeff_t_;
    engine::core::TensorValue denoised_primary_t_;
    engine::core::TensorValue denoised_secondary_t_;
    engine::core::TensorValue next_primary_t_;
    engine::core::TensorValue next_secondary_t_;
    float previous_primary_sigma_ = 0.0F;
    float previous_secondary_sigma_ = 0.0F;
    bool has_history_ = false;
    double device_copy_ms_ = 0.0;
    double graph_compute_ms_ = 0.0;
    double output_read_ms_ = 0.0;
    double history_update_ms_ = 0.0;
};

}  // namespace engine::models::minimax_h3
