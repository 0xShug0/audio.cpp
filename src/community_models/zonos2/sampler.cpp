#include "engine/community_models/zonos2/sampler.h"

#include <algorithm>
#include <cmath>
#include <functional>

namespace engine::models::zonos2 {

Zonos2Sampler::Zonos2Sampler(const Zonos2SamplingOptions & options, int codebooks, int vocab, int codebook_size)
    : options_(options),
      rng_(options.seed),
      vocab_(vocab),
      codebook_size_(codebook_size),
      penalized_codebooks_(options.repetition_codebooks < 0 ? codebooks : std::min(options.repetition_codebooks, codebooks)),
      history_(static_cast<size_t>(codebooks)),
      logit_(static_cast<size_t>(vocab)),
      scratch_(static_cast<size_t>(vocab)),
      seen_(static_cast<size_t>(vocab), 0) {
    if (options_.repetition_penalty <= 1.0f || options_.repetition_window <= 0) penalized_codebooks_ = 0;
    p_.reserve(static_cast<size_t>(vocab));
    candidates_.reserve(static_cast<size_t>(vocab));
    order_.reserve(static_cast<size_t>(vocab));
}

void Zonos2Sampler::accept(const std::vector<int32_t> & frame) {
    for (int cb = 0; cb < penalized_codebooks_ && cb < static_cast<int>(frame.size()); ++cb) {
        auto & history = history_[static_cast<size_t>(cb)];
        history.push_back(frame[static_cast<size_t>(cb)]);
        while (static_cast<int>(history.size()) > options_.repetition_window) history.pop_front();
    }
}

int Zonos2Sampler::sample(const float * logits, int cb) {
    std::copy(logits, logits + vocab_, logit_.begin());
    if (cb < penalized_codebooks_) {
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
    auto argmax = [&] {
        int best = 0;
        for (int i = 1; i < vocab_; ++i) {
            if (logit_[static_cast<size_t>(i)] > logit_[static_cast<size_t>(best)]) best = i;
        }
        return best;
    };
    if (options_.temperature <= 0.0f) return argmax();
    const float temperature = std::max(options_.temperature, 1e-8f);
    for (auto & l : logit_) l /= temperature;

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
        p_[j] = std::exp(static_cast<double>(logit_[static_cast<size_t>(candidates_[j])] - max_logit));
        sum += p_[j];
    }
    for (auto & x : p_) x = static_cast<double>(static_cast<float>(x / sum));

    auto renormalize = [&] {
        double kept = 0.0;
        for (const double x : p_) kept += x;
        if (kept > 0.0) {
            for (auto & x : p_) x /= kept;
        }
        return kept;
    };
    if (options_.top_p > 0.0f && options_.top_p < 1.0f) {
        order_.resize(candidates_.size());
        for (size_t j = 0; j < order_.size(); ++j) order_[j] = static_cast<int>(j);
        std::stable_sort(order_.begin(), order_.end(),
                         [&](int a, int b) { return p_[static_cast<size_t>(a)] > p_[static_cast<size_t>(b)]; });
        double before = 0.0;
        for (const int r : order_) {
            const double p = p_[static_cast<size_t>(r)];
            if (before > options_.top_p) p_[static_cast<size_t>(r)] = 0.0;
            before += p;
        }
        renormalize();
    }
    if (options_.min_p > 0.0f) {
        const double threshold = static_cast<double>(options_.min_p) * *std::max_element(p_.begin(), p_.end());
        for (auto & x : p_) {
            if (x < threshold) x = 0.0;
        }
        renormalize();
    }
    double total = 0.0;
    for (const double x : p_) total += x;
    if (!(total > 0.0)) return argmax();
    std::uniform_real_distribution<double> uniform(0.0, total);
    const double r = uniform(rng_);
    double acc = 0.0;
    int last = -1;
    for (size_t j = 0; j < p_.size(); ++j) {
        if (p_[j] <= 0.0) continue;
        acc += p_[j];
        last = candidates_[j];
        if (acc >= r) return last;
    }
    return last;
}

}  // namespace engine::models::zonos2
