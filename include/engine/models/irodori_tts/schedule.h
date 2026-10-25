#pragma once

#include "engine/framework/runtime/errors.h"
#include "engine/framework/sampling/flow_sampler_runtime.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>

namespace engine::models::irodori_tts {

inline std::vector<modules::FlowSamplerScheduleStep>
make_irodori_meanflow_schedule(int64_t steps) {
  if (steps <= 0) {
    throw runtime::InvalidRequestError("Irodori-TTS MeanFlow requires positive steps");
  }
  std::vector<modules::FlowSamplerScheduleStep> schedule;
  schedule.reserve(static_cast<size_t>(steps));
  const float spacing = 1.0F / static_cast<float>(steps);
  const auto time_at = [&](int64_t index) {
    return index < (steps + 1) / 2
               ? 1.0F - spacing * static_cast<float>(index)
               : spacing * static_cast<float>(steps - index);
  };
  for (int64_t step = 0; step < steps; ++step) {
    const float t = time_at(step);
    const float next = time_at(step + 1);
    schedule.push_back({step, t, next, 0.0F, 0.0F});
  }
  return schedule;
}

// Matches Irodori-TTS rf.py: warp ascending u, then reverse to noise -> data t.
inline std::vector<modules::FlowSamplerScheduleStep>
make_irodori_rf_schedule(int64_t steps, std::string mode = "linear",
                         float sway_coeff = -1.0F) {
  using runtime::InvalidRequestError;
  if (steps <= 0) {
    throw InvalidRequestError("Irodori-TTS RF schedule requires positive steps");
  }
  const auto first = mode.find_first_not_of(" \t\r\n");
  mode = first == std::string::npos ? "" :
      mode.substr(first, mode.find_last_not_of(" \t\r\n") - first + 1);
  std::transform(mode.begin(), mode.end(), mode.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (mode != "linear" && mode != "sway") {
    throw InvalidRequestError("Irodori-TTS t_schedule_mode must be linear or sway");
  }
  if (!std::isfinite(sway_coeff)) {
    throw InvalidRequestError("Irodori-TTS sway_coeff must be finite");
  }
  auto time_at = [&](int64_t index) {
    float u = static_cast<float>(index) / static_cast<float>(steps);
    if (mode == "sway") {
      u = std::clamp(u + sway_coeff * (std::cos(1.5707963267948966F * u) + u - 1.0F),
                     0.0F, 1.0F);
    }
    return (1.0F - u) * 0.999F;
  };
  std::vector<modules::FlowSamplerScheduleStep> schedule;
  schedule.reserve(static_cast<size_t>(steps));
  float t = time_at(0);
  for (int64_t step = 0; step < steps; ++step) {
    const float next = time_at(step + 1);
    if (!(t > next)) {
      throw InvalidRequestError(
          "Irodori-TTS t_schedule must be strictly decreasing; adjust num_inference_steps or sway_coeff");
    }
    schedule.push_back({step, t, next, 0.0F, 0.0F});
    t = next;
  }
  return schedule;
}

} // namespace engine::models::irodori_tts
