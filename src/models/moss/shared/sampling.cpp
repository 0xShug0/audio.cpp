#include "engine/models/moss/shared/sampling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace engine::models::moss {

int32_t argmax_index(const std::vector<float> & logits, std::string_view context) {
    if (logits.empty()) {
        throw std::runtime_error(std::string(context) + " sampler received empty logits");
    }
    size_t best = 0;
    for (size_t index = 1; index < logits.size(); ++index) {
        if (logits[index] > logits[best]) {
            best = index;
        }
    }
    return static_cast<int32_t>(best);
}

void apply_repetition_penalty(
    std::vector<float> & logits,
    const std::vector<int32_t> & previous_token_ids,
    float penalty,
    std::string_view context) {
    if (penalty == 1.0F || previous_token_ids.empty()) {
        return;
    }
    if (penalty <= 0.0F) {
        throw std::runtime_error(std::string(context) + " repetition penalty must be positive");
    }
    std::unordered_set<int32_t> seen;
    seen.reserve(previous_token_ids.size());
    for (const int32_t token : previous_token_ids) {
        if (token < 0 || token >= static_cast<int32_t>(logits.size())) {
            continue;
        }
        if (!seen.insert(token).second) {
            continue;
        }
        float & value = logits[static_cast<size_t>(token)];
        value = value < 0.0F ? value * penalty : value / penalty;
    }
}

int32_t sample_index(
    const std::vector<float> & logits,
    int top_k,
    float top_p,
    float temperature,
    std::mt19937 & rng,
    std::string_view context,
    const engine::sampling::TorchCudaSamplingPolicy * sampling_policy,
    uint64_t seed,
    uint64_t call_index) {
    if (temperature <= 0.0F) {
        throw std::runtime_error(std::string(context) + " sampler temperature must be positive");
    }
    std::vector<int32_t> indices;
    indices.reserve(logits.size());
    for (size_t index = 0; index < logits.size(); ++index) {
        if (std::isfinite(logits[index])) {
            indices.push_back(static_cast<int32_t>(index));
        }
    }
    if (indices.empty()) {
        throw std::runtime_error(std::string(context) + " sampler has no finite logits");
    }
    if (top_k > 0 && static_cast<int>(indices.size()) > top_k) {
        std::vector<int32_t> ranked = indices;
        const auto keep = ranked.begin() + top_k - 1;
        std::nth_element(ranked.begin(), keep, ranked.end(), [&](int32_t lhs, int32_t rhs) {
            return logits[static_cast<size_t>(lhs)] > logits[static_cast<size_t>(rhs)];
        });
        const float kth_logit = logits[static_cast<size_t>(*keep)];
        indices.erase(
            std::remove_if(
                indices.begin(),
                indices.end(),
                [&](int32_t index) {
                    return logits[static_cast<size_t>(index)] < kth_logit;
                }),
            indices.end());
    }
    std::sort(indices.begin(), indices.end(), [&](int32_t lhs, int32_t rhs) {
        const float lhs_logit = logits[static_cast<size_t>(lhs)];
        const float rhs_logit = logits[static_cast<size_t>(rhs)];
        if (lhs_logit == rhs_logit) {
            return lhs < rhs;
        }
        return lhs_logit > rhs_logit;
    });

    const float max_logit = logits[static_cast<size_t>(indices.front())] / temperature;
    std::vector<double> weights;
    weights.reserve(indices.size());
    double total = 0.0;
    for (const int32_t index : indices) {
        const double weight = std::exp(static_cast<double>(logits[static_cast<size_t>(index)] / temperature - max_logit));
        weights.push_back(weight);
        total += weight;
    }
    if (top_p > 0.0F && top_p < 1.0F) {
        double cumulative = 0.0;
        size_t keep = weights.size();
        for (size_t index = 0; index < weights.size(); ++index) {
            cumulative += weights[index] / total;
            if (cumulative > top_p) {
                keep = index + 1;
                break;
            }
        }
        indices.resize(keep);
        weights.resize(keep);
    }
    if (sampling_policy != nullptr && sampling_policy->cuda_fast_path) {
        double best_rank = -std::numeric_limits<double>::infinity();
        int32_t best_token = -1;
        for (size_t index = 0; index < indices.size(); ++index) {
            const float exponential = engine::sampling::torch_cuda_tensor_iterator_exponential_element(
                seed,
                static_cast<uint64_t>(logits.size()),
                static_cast<uint64_t>(indices[index]),
                call_index,
                sampling_policy->multiprocessor_count,
                sampling_policy->max_threads_per_multiprocessor);
            const double rank = weights[index] / static_cast<double>(exponential);
            if (rank > best_rank) {
                best_rank = rank;
                best_token = indices[index];
            }
        }
        if (best_token < 0) {
            throw std::runtime_error(std::string(context) + " CUDA sampler failed to select a token");
        }
        return best_token;
    }
    std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
    return indices[distribution(rng)];
}

}  // namespace engine::models::moss
