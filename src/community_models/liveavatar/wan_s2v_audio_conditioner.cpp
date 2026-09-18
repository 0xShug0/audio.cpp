#include "wan_s2v_audio_conditioner.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace engine::community_models::liveavatar {
namespace {

void validate_audio_feature(const WanS2VAudioFeature &feature,
                            const char *name) {
  if (feature.layers <= 0 || feature.frames <= 0 || feature.dims <= 0) {
    throw std::runtime_error(std::string(name) + " shape must be positive");
  }
  if (static_cast<int64_t>(feature.values.size()) !=
      feature.layers * feature.frames * feature.dims) {
    throw std::runtime_error(std::string(name) +
                             " value count does not match shape");
  }
}

size_t feature_index(const WanS2VAudioFeature &feature, int64_t layer,
                     int64_t frame, int64_t dim) {
  return static_cast<size_t>((layer * feature.frames + frame) * feature.dims +
                             dim);
}

std::vector<int64_t> get_sample_indices(int64_t original_fps,
                                        int64_t total_frames,
                                        int64_t target_fps,
                                        int64_t num_sample) {
  if (original_fps <= 0 || target_fps <= 0 || total_frames <= 0 ||
      num_sample <= 0) {
    throw std::runtime_error(
        "Wan S2V audio sample-index config must be positive");
  }
  const double required_duration =
      static_cast<double>(num_sample) / static_cast<double>(target_fps);
  const int64_t required_origin_frames = static_cast<int64_t>(
      std::ceil(required_duration * static_cast<double>(original_fps)));
  if (required_origin_frames > total_frames) {
    throw std::runtime_error(
        "Wan S2V audio bucket length exceeds available frames");
  }

  const double start_time = 0.0;
  const double end_time = start_time + required_duration;
  std::vector<int64_t> indices(static_cast<size_t>(num_sample), 0);
  for (int64_t i = 0; i < num_sample; ++i) {
    const double t = start_time + (end_time - start_time) *
                                      static_cast<double>(i) /
                                      static_cast<double>(num_sample);
    const int64_t index = static_cast<int64_t>(
        std::nearbyint(t * static_cast<double>(original_fps)));
    indices[static_cast<size_t>(i)] =
        std::clamp<int64_t>(index, 0, total_frames - 1);
  }
  return indices;
}

} // namespace

WanS2VAudioFeature
wan_s2v_linear_interpolate_audio_features(const WanS2VAudioFeature &input,
                                          int64_t input_fps, int64_t output_fps,
                                          int64_t output_frames) {
  validate_audio_feature(input, "Wan S2V audio input");
  if (input_fps <= 0 || output_fps <= 0) {
    throw std::runtime_error(
        "Wan S2V audio interpolation FPS values must be positive");
  }
  if (output_frames == 0) {
    output_frames = static_cast<int64_t>(static_cast<double>(input.frames) *
                                         static_cast<double>(output_fps) /
                                         static_cast<double>(input_fps));
  }
  if (output_frames <= 0) {
    throw std::runtime_error(
        "Wan S2V audio interpolation output frame count must be positive");
  }

  WanS2VAudioFeature output;
  output.layers = input.layers;
  output.frames = output_frames;
  output.dims = input.dims;
  output.values.resize(
      static_cast<size_t>(output.layers * output.frames * output.dims), 0.0F);

  for (int64_t layer = 0; layer < input.layers; ++layer) {
    for (int64_t out_frame = 0; out_frame < output_frames; ++out_frame) {
      const double src_pos = output_frames == 1
                                 ? 0.0
                                 : static_cast<double>(out_frame) *
                                       static_cast<double>(input.frames - 1) /
                                       static_cast<double>(output_frames - 1);
      const int64_t lo = static_cast<int64_t>(std::floor(src_pos));
      const int64_t hi = std::min<int64_t>(lo + 1, input.frames - 1);
      const float alpha = static_cast<float>(src_pos - static_cast<double>(lo));
      for (int64_t dim = 0; dim < input.dims; ++dim) {
        const float lhs = input.values[feature_index(input, layer, lo, dim)];
        const float rhs = input.values[feature_index(input, layer, hi, dim)];
        output.values[feature_index(output, layer, out_frame, dim)] =
            lhs + (rhs - lhs) * alpha;
      }
    }
  }
  return output;
}

WanS2VAudioBuckets
wan_s2v_bucket_audio_features(const WanS2VAudioFeature &input,
                              const WanS2VAudioBucketConfig &config) {
  validate_audio_feature(input, "Wan S2V audio bucket input");
  if (config.source_fps <= 0 || config.target_fps <= 0 ||
      config.batch_frames <= 0 || config.sample_radius < 0) {
    throw std::runtime_error("Wan S2V audio bucket config is invalid");
  }
  const double scale = static_cast<double>(config.source_fps) /
                       static_cast<double>(config.target_fps);
  const int64_t repeats =
      static_cast<int64_t>(static_cast<double>(input.frames) /
                           (static_cast<double>(config.batch_frames) * scale)) +
      1;
  const int64_t bucket_frames = repeats * config.batch_frames;
  const int64_t padded_audio =
      static_cast<int64_t>(
          std::ceil(static_cast<double>(repeats * config.batch_frames) /
                    static_cast<double>(config.target_fps) *
                    static_cast<double>(config.source_fps))) -
      input.frames;
  const auto indices =
      get_sample_indices(config.source_fps, input.frames + padded_audio,
                         config.target_fps, bucket_frames);
  const int64_t sample_stride = config.source_fps / config.target_fps;
  if (sample_stride <= 0) {
    throw std::runtime_error("Wan S2V audio sample stride must be positive");
  }

  const int64_t samples_per_frame = 2 * config.sample_radius + 1;
  WanS2VAudioBuckets output;
  output.layers = input.layers;
  output.dims = input.dims * samples_per_frame;
  output.frames = bucket_frames;
  output.repeats = repeats;
  output.values.resize(static_cast<size_t>(output.batch * output.layers *
                                           output.dims * output.frames),
                       0.0F);

  for (int64_t out_frame = 0; out_frame < bucket_frames; ++out_frame) {
    const int64_t center = indices[static_cast<size_t>(out_frame)];
    if (center >= input.frames) {
      continue;
    }
    int64_t sample_slot = 0;
    for (int64_t offset = -config.sample_radius * sample_stride;
         offset <= config.sample_radius * sample_stride;
         offset += sample_stride) {
      const int64_t in_frame =
          std::clamp<int64_t>(center + offset, 0, input.frames - 1);
      for (int64_t layer = 0; layer < input.layers; ++layer) {
        for (int64_t dim = 0; dim < input.dims; ++dim) {
          const size_t out_index = static_cast<size_t>(
              ((layer * output.dims + sample_slot * input.dims + dim) *
               output.frames) +
              out_frame);
          output.values[out_index] =
              input.values[feature_index(input, layer, in_frame, dim)];
        }
      }
      ++sample_slot;
    }
  }
  return output;
}

WanS2VAudioBuckets
wan_s2v_prepare_audio_encoder_output(const WanS2VAudioFeature &hidden_states,
                                     const WanS2VAudioConditionerConfig &config) {
  if (config.feature_fps <= 0 || config.video_fps <= 0 ||
      config.target_fps <= 0 || config.batch_frames <= 0 ||
      config.sample_radius < 0) {
    throw std::runtime_error("Wan S2V audio encoder config is invalid");
  }
  const auto interpolated = wan_s2v_linear_interpolate_audio_features(
      hidden_states, config.feature_fps, config.video_fps);
  return wan_s2v_bucket_audio_features(
      interpolated,
      WanS2VAudioBucketConfig{config.video_fps, config.target_fps,
                              config.batch_frames, config.sample_radius});
}

WanS2VAudioFeature wan_s2v_audio_feature_from_hubert_layers(
    const modules::HubertEncoderLayerOutput &layer_output) {
  if (layer_output.batch != 1 || layer_output.tokens <= 0 ||
      layer_output.hidden_size <= 0) {
    throw std::runtime_error(
        "Wan S2V audio requires batch-1 positive HuBERT layer outputs");
  }
  if (layer_output.layer_indices.size() != layer_output.hidden_states.size() ||
      layer_output.hidden_states.empty()) {
    throw std::runtime_error(
        "Wan S2V audio HuBERT layer output count mismatch");
  }
  WanS2VAudioFeature feature;
  feature.layers = static_cast<int64_t>(layer_output.hidden_states.size());
  feature.frames = layer_output.tokens;
  feature.dims = layer_output.hidden_size;
  feature.values.reserve(
      static_cast<size_t>(feature.layers * feature.frames * feature.dims));
  const int64_t per_layer =
      layer_output.batch * layer_output.tokens * layer_output.hidden_size;
  for (const auto &hidden : layer_output.hidden_states) {
    if (static_cast<int64_t>(hidden.size()) != per_layer) {
      throw std::runtime_error(
          "Wan S2V audio HuBERT layer output shape mismatch");
    }
    feature.values.insert(feature.values.end(), hidden.begin(), hidden.end());
  }
  return feature;
}

} // namespace engine::community_models::liveavatar
