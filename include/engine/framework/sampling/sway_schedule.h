#pragma once

#include "engine/framework/runtime/errors.h"
#include "flow_sampler_runtime.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>

namespace engine::modules {

// Create a linear or sway timestep schedule for RF sampling.
inline std::vector<FlowSamplerScheduleStep>
make_rf_schedule(int64_t steps, std::string model,
                 std::string mode = "linear", float sway_coeff = -1.0F,
                 float init_scale = 0.999F) {
    using runtime::InvalidRequestError;
    if (steps <= 0) {
        throw InvalidRequestError(model + " RF Schedule requires positive steps");
    }
    const auto first = mode.find_first_not_of(" \t\r\n");
    mode = first == std::string::npos ? "" :
        mode.substr(first, mode.find_last_not_of(" \t\r\n") - first + 1);
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (mode != "linear" && mode != "sway") {
        throw InvalidRequestError(model + " t_schedule_mode must be linear or sway");
    }
    if (!std::isfinite(sway_coeff)) {
        throw InvalidRequestError(model + " sway_coeff must be finite");
    }
    auto time_at = [&](int64_t index) {
        float u = static_cast<float>(index) / static_cast<float>(steps);
        if (mode == "sway") {
        u = std::clamp(u + sway_coeff * (std::cos(1.5707963267948966F * u) + u - 1.0F),
                        0.0F, 1.0F);
        }
        return (1.0F - u) * init_scale;
    };
    std::vector<FlowSamplerScheduleStep> schedule;
    schedule.reserve(static_cast<size_t>(steps));
    float t = time_at(0);
    for (int64_t step = 0; step < steps; ++step) {
        const float next = time_at(step + 1);
        if (!(t > next)) {
            throw runtime::InvalidRequestError(
                model + " t_schedule must be strictly decreasing; adjust num_inference_steps or sway_coeff");
        }
        schedule.push_back({step, t, next, 0.0F, 0.0F});
        t = next;
    }
    return schedule;
}

} // namespace engine::modules
