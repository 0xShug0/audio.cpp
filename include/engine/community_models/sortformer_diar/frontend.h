#pragma once

#include "engine/community_models/sortformer_diar/assets.h"
#include "engine/framework/runtime/session_base.h"

#include <cstdint>
#include <vector>

namespace engine::community_models::sortformer_diar {

struct SortformerV2FeatureBatch {
    int64_t frames = 0;
    int64_t valid_frames = 0;
    std::vector<float> time_major;
};

SortformerV2FeatureBatch compute_sortformer_v2_features(
    const runtime::AudioBuffer & audio,
    const SortformerV2Assets & assets,
    int64_t threads);

SortformerV2FeatureBatch compute_sortformer_v2_stream_features(
    const std::vector<float> & mono_samples,
    const SortformerV2Assets & assets,
    int64_t threads,
    bool initial_window,
    int64_t expected_frames = -1);

}  // namespace engine::community_models::sortformer_diar
