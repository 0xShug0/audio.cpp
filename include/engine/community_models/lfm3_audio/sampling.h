#pragma once

// LFM3-Audio samplers, in the upstream processor's order: repetition penalty
// (audio: the unique codes of the last frames of that codebook), suppressed
// ids to -inf (text), then temperature and top-k, and a multinomial draw.
// Greedy (first argmax, after the penalty) when the temperature is 0, or
// unset with no top-k. Draws go through std::discrete_distribution, so they
// are deterministic per platform only.

#include "engine/community_models/lfm3_audio/depth.h"
#include "engine/framework/sampling/hf_sampler.h"

#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace engine::community_models::lfm3_audio {

struct Lfm3AudioSampling {
    std::optional<float> temperature = 0.4f;  // nullopt: unset (greedy unless top_k is set)
    int64_t top_k = 0;                        // 0: unset
    float repetition_penalty = 1.2f;          // 1: off
    int64_t penalty_window = 64;              // frames of history per codebook
};

struct Lfm3TextSampling {
    std::optional<float> temperature = 0.0f;  // nullopt: unset (greedy unless top_k is set)
    int64_t top_k = 0;                        // 0: unset
};

class Lfm3AudioSampler {
public:
    Lfm3AudioSampler(const Lfm3AudioSampling & sampling, uint64_t seed);
    Lfm3Codes next_order();  // Fisher-Yates with j = rng() % (i+1): portable, unlike std::shuffle
    int32_t pick(int64_t codebook, std::vector<float> & logits);
    void commit(const Lfm3Codes & frame);  // non-EOS frames only; keeps last penalty_window per codebook

private:
    Lfm3AudioSampling sampling_;
    sampling::HfSamplingOptions options_;
    std::mt19937 rng_;  // one stream for orders and draws (HF uses one global RNG)
    sampling::HfSampler sampler_;
    sampling::HfSamplerScratch scratch_;
    std::array<std::vector<int32_t>, 12> history_;  // oldest first
};

class Lfm3TextSampler {
public:
    Lfm3TextSampler(const Lfm3TextSampling & sampling, uint64_t seed);  // own stream, seed mixed as Lfm2CodeSampler
    int32_t pick(std::vector<float> & logits, const std::vector<int32_t> & suppress);

private:
    sampling::HfSamplingOptions options_;
    std::mt19937 rng_;
    sampling::HfSampler sampler_;
    sampling::HfSamplerScratch scratch_;
};

}  // namespace engine::community_models::lfm3_audio
