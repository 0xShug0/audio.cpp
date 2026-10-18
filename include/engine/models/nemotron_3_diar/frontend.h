#pragma once

#include "engine/models/nemotron_3_diar/assets.h"
#include "engine/framework/runtime/session.h"

#include <cstdint>
#include <vector>

namespace engine::models::nemotron_3_diar {

struct StreamWindow;

struct FeatureBatch {
    int64_t batch = 0;
    int64_t feature_frames = 0;
    int64_t encoder_frames = 0;
    std::vector<int64_t> valid_feature_frames;
    std::vector<int64_t> valid_encoder_frames;
    std::vector<float> stacked;
};

audio::NemoMelFrontend make_frontend(const Assets & assets);

FeatureBatch compute_features(
    const std::vector<runtime::AudioBuffer> & audio,
    const Assets & assets,
    int64_t threads);

FeatureBatch compute_stream_features(
    const std::vector<StreamWindow> & windows,
    const Assets & assets,
    int64_t threads);

}  // namespace engine::models::nemotron_3_diar
