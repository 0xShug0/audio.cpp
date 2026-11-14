#include "engine/community_models/zonos3/sampler.h"

#include <algorithm>
#include <cmath>
#include <functional>

namespace engine::models::zonos3 {

Zonos3Sampler::Zonos3Sampler(const Zonos3SamplingOptions & options, int codebooks, int vocab, int codebook_size,
                             int pad_id, int stop_id)
    : options_(options),
      rng_(options.seed),
      codebooks_(codebooks),
      vocab_(vocab),
      codebook_size_(codebook_size),
      pad_id_(pad_id),
      stop_id_(stop_id),
      history_(static_cast<size_t>(codebooks)),
      logit_(static_cast<size_t>(vocab)),
      scratch_(static_cast<size_t>(vocab)),
      seen_(static_cast<size_t>(vocab), 0) {
    p_.reserve(static_cast<size_t>(vocab));
    candidates_.reserve(static_cast<size_t>(vocab));
    order_.reserve(static_cast<size_t>(vocab));
    keep_.reserve(static_cast<size_t>(vocab));
}

void Zonos3Sampler::accept(const std::vector<int32_t> & frame) {
    const int history_codebooks = options_.repetition_penalty == 1.0f ? 0
        : (options_.repetition_codebooks < 0 ? codebooks_ : options_.repetition_codebooks);
    for (int cb = 0; cb < history_codebooks && cb < codebooks_ && cb < static_cast<int>(frame.size()); ++cb) {
        auto & history = history_[static_cast<size_t>(cb)];
        history.push_back(frame[static_cast<size_t>(cb)]);
        while (static_cast<int>(history.size()) > options_.repetition_window) history.pop_front();
    }
    ++generated_;
}

int Zonos3Sampler::sample(const float * logits, int cb) {
    std::copy(logits, logits + vocab_, logit_.begin());
    // The audio pad is never a target; the stop id is the only special id, and in codebook c it waits for c frames.
    for (int i = codebook_size_; i < vocab_; ++i) {
        if (i == pad_id_ || i != stop_id_ || generated_ < cb) logit_[static_cast<size_t>(i)] = -INFINITY;
    }
    const int penalized = options_.repetition_codebooks < 0 ? codebooks_ : options_.repetition_codebooks;
    if (options_.repetition_penalty != 1.0f && cb < penalized) {
        if (++seen_generation_ == 0) {
            std::fill(seen_.begin(), seen_.end(), 0u);
            seen_generation_ = 1;
        }
        for (const int32_t t : history_[static_cast<size_t>(cb)]) {
            if (t >= 0 && t < codebook_size_ && seen_[static_cast<size_t>(t)] != seen_generation_) {
                seen_[static_cast<size_t>(t)] = seen_generation_;
                float & l = logit_[static_cast<size_t>(t)];
                l = l > 0.0f ? l / options_.repetition_penalty : l * options_.repetition_penalty;
            }
        }
    }
    if (options_.temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_; ++i) {
            if (logit_[static_cast<size_t>(i)] > logit_[static_cast<size_t>(best)]) best = i;
        }
        return best;
    }
    for (auto & l : logit_) l /= options_.temperature;

    candidates_.clear();
    if (options_.top_k > 0 && options_.top_k < vocab_) {
        std::copy(logit_.begin(), logit_.end(), scratch_.begin());
        std::nth_element(scratch_.begin(), scratch_.begin() + (options_.top_k - 1), scratch_.end(), std::greater<float>());
        const float kth = scratch_[static_cast<size_t>(options_.top_k - 1)];
        for (int i = 0; i < vocab_; ++i) {
            if (logit_[static_cast<size_t>(i)] >= kth) candidates_.push_back(i);
        }
    } else {
        for (int i = 0; i < vocab_; ++i) candidates_.push_back(i);
    }

    float max_logit = -INFINITY;
    for (const int i : candidates_) max_logit = std::max(max_logit, logit_[static_cast<size_t>(i)]);
    p_.resize(candidates_.size());
    double sum = 0.0;
    for (size_t j = 0; j < candidates_.size(); ++j) {
        p_[j] = std::exp(logit_[static_cast<size_t>(candidates_[j])] - max_logit);
        sum += p_[j];
    }
    for (auto & x : p_) x = static_cast<float>(x / sum);

    order_.resize(candidates_.size());
    for (size_t j = 0; j < order_.size(); ++j) order_[j] = static_cast<int>(j);
    std::sort(order_.begin(), order_.end(), [&](int a, int b) { return p_[static_cast<size_t>(a)] > p_[static_cast<size_t>(b)]; });

    keep_.assign(candidates_.size(), 1);
    if (options_.top_p > 0.0f && options_.top_p < 1.0f) {
        double before = 0.0;
        for (const int r : order_) {
            if (before > options_.top_p) keep_[static_cast<size_t>(r)] = 0;
            before += p_[static_cast<size_t>(r)];
        }
    }
    if (options_.min_p > 0.0f) {
        const float threshold = options_.min_p * p_[static_cast<size_t>(order_[0])];
        for (size_t j = 0; j < p_.size(); ++j) {
            if (p_[j] < threshold) keep_[j] = 0;
        }
    }
    double kept = 0.0;
    for (size_t j = 0; j < p_.size(); ++j) {
        if (!keep_[j]) p_[j] = 0.0f;
        kept += p_[j];
    }
    if (kept <= 0.0) return candidates_[static_cast<size_t>(order_[0])];
    std::uniform_real_distribution<double> uniform(0.0, kept);
    const double r = uniform(rng_);
    double acc = 0.0;
    for (size_t j = 0; j < p_.size(); ++j) {
        acc += p_[j];
        if (acc >= r) return candidates_[j];
    }
    return candidates_[static_cast<size_t>(order_[0])];
}

}  // namespace engine::models::zonos3
