#pragma once

#include "engine/framework/modules/speech_encoders/hubert_encoder.h"

#include <cstdint>
#include <vector>

namespace engine::community_models::liveavatar {

struct WanS2VAudioFeature {
  std::vector<float> values;
  int64_t layers = 0;
  int64_t frames = 0;
  int64_t dims = 0;
};

struct WanS2VAudioBucketConfig {
  int64_t source_fps = 30;
  int64_t target_fps = 16;
  int64_t batch_frames = 80;
  int64_t sample_radius = 0;
};

struct WanS2VAudioConditionerConfig {
  int64_t feature_fps = 50;
  int64_t video_fps = 30;
  int64_t target_fps = 16;
  int64_t batch_frames = 80;
  int64_t sample_radius = 0;
};

struct WanS2VAudioBuckets {
  std::vector<float> values;
  int64_t batch = 1;
  int64_t layers = 0;
  int64_t dims = 0;
  int64_t frames = 0;
  int64_t repeats = 0;
};

WanS2VAudioFeature
wan_s2v_linear_interpolate_audio_features(const WanS2VAudioFeature &input,
                                          int64_t input_fps, int64_t output_fps,
                                          int64_t output_frames = 0);

WanS2VAudioBuckets
wan_s2v_bucket_audio_features(const WanS2VAudioFeature &input,
                              const WanS2VAudioBucketConfig &config);

WanS2VAudioBuckets
wan_s2v_prepare_audio_encoder_output(const WanS2VAudioFeature &hidden_states,
                                     const WanS2VAudioConditionerConfig &config);

WanS2VAudioFeature wan_s2v_audio_feature_from_hubert_layers(
    const modules::HubertEncoderLayerOutput &layer_output);

} // namespace engine::community_models::liveavatar
