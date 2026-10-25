#pragma once

#include "engine/framework/runtime/errors.h"
#include "engine/framework/sampling/flow_sampler_runtime.h"
#include "engine/framework/sampling/sway_schedule.h"

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
  return modules::make_rf_schedule(steps, "Irodori-TTS", mode, sway_coeff);
}

} // namespace engine::models::irodori_tts
