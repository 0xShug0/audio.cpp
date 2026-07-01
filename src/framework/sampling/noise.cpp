#include "engine/framework/sampling/noise.h"

#include <algorithm>
#include <random>

namespace engine::sampling {

std::vector<float> generate_normal_noise(std::size_t count, uint32_t seed, float scale) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0F, scale);
    std::vector<float> values(count);
    for (auto & value : values) {
        value = dist(rng);
    }
    return values;
}

void clamp_noise(std::vector<float> & noise, float min_value, float max_value) {
    for (auto & value : noise) {
        value = std::max(min_value, std::min(max_value, value));
    }
}

}  // namespace engine::sampling
