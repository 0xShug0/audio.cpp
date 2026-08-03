#pragma once

#include <cstdint>
#include <vector>

namespace engine::models::sortformer_diar {

struct SortformerFeatureBatch {
    std::vector<float> time_major;
    int64_t frames = 0;
    int64_t valid_frames = 0;
};

struct SortformerPostprocessConfig {
    float threshold = 0.5f;
    int64_t min_frames = 0;
    int64_t pad_frames = 0;
};

struct SortformerFixedContextContract {
    double session_len_sec = 90.0;
    int64_t feature_frames = 0;
    int64_t encoder_frames = 0;
};

struct SortformerRunTimings {
    double frontend_ms = 0.0;
    double log_mel_ms = 0.0;
    double feature_normalizer_ms = 0.0;
    double graph_ensure_ms = 0.0;
    double graph_prepare_ms = 0.0;
    double encoder_compute_ms = 0.0;
    double encoder_readback_ms = 0.0;
    double encoder_ms = 0.0;
    double postprocess_ms = 0.0;
    double wall_ms = 0.0;
};

}  // namespace engine::models::sortformer_diar
