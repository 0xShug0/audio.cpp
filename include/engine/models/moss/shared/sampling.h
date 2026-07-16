#pragma once

#include "engine/framework/sampling/torch_random.h"

#include <cstdint>
#include <random>
#include <string_view>
#include <vector>

namespace engine::models::moss {

int32_t argmax_index(const std::vector<float> & logits, std::string_view context);

void apply_repetition_penalty(
    std::vector<float> & logits,
    const std::vector<int32_t> & previous_token_ids,
    float penalty,
    std::string_view context);

int32_t sample_index(
    const std::vector<float> & logits,
    int top_k,
    float top_p,
    float temperature,
    std::mt19937 & rng,
    std::string_view context,
    const engine::sampling::TorchCudaSamplingPolicy * sampling_policy = nullptr,
    uint64_t seed = 0,
    uint64_t call_index = 0);

}  // namespace engine::models::moss
