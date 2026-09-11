#pragma once

#include "engine/community_models/minimax_h3/dit_denoiser.h"
#include "engine/community_models/minimax_h3/types.h"

#include <cstddef>
#include <vector>

namespace engine::models::minimax_h3 {

class MiniMaxH3SpectrumForecaster {
public:
    MiniMaxH3SpectrumForecaster(
        const MiniMaxH3GenerateRequest & request,
        int64_t steps,
        size_t feature_size);

    bool should_run_full(int64_t step);
    void update(float coordinate, const std::vector<float> & feature);
    void predict(
        float coordinate,
        int64_t row_width,
        int64_t audio_start,
        int64_t audio_rows,
        int64_t video_start,
        int64_t video_rows,
        std::vector<float> & out);

    int64_t full_steps() const;
    int64_t forecast_steps() const;

private:
    size_t feature_size_ = 0;
    int64_t steps_ = 0;
    int64_t warmup_steps_ = 1;
    float current_window_ = 2.0F;
    float flex_window_ = 0.75F;
    int64_t cached_steps_in_window_ = 0;
    int64_t full_steps_ = 0;
    int64_t forecast_steps_ = 0;
    int degree_ = 1;
    int history_size_ = 8;
    float ridge_lambda_ = 0.1F;
    std::vector<float> coordinates_;
    std::vector<float> features_;
    std::vector<float> design_;
    std::vector<float> cholesky_;
    bool factorization_valid_ = false;

    int basis_size() const;
    void build_basis(float coordinate, float * out) const;
    void fit_if_needed();
    std::vector<float> linear_weights(float coordinate) const;
    std::vector<float> spectral_weights(float coordinate);
    std::vector<float> combined_weights(float coordinate, float blend_weight);
};

}  // namespace engine::models::minimax_h3
