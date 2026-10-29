#include "engine/community_models/lfm3_audio/sampling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::community_models::lfm3_audio {
namespace {

constexpr int64_t kCodebooks = 12;

// The text stream's seed: splitmix64 of the request seed with "text" in its
// top bytes, so text and audio draws of one seed come from different streams.
uint64_t text_seed(uint64_t seed) {
    uint64_t z = (seed ^ 0x7465787400000000ull) + 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

void require_finite(const std::vector<float> & logits, const char * what) {
    if (logits.empty() || !std::all_of(logits.begin(), logits.end(), [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error(std::string("LFM3-Audio ") + what + " logits are empty or non-finite");
    }
}

// The upstream rule: greedy when the temperature is 0, or unset with no top-k; an unset temperature with top-k
// samples at 1.
sampling::HfSamplingOptions hf_options(std::optional<float> temperature, int64_t top_k, float repetition_penalty, const char * what) {
    if ((temperature && (!std::isfinite(*temperature) || *temperature < 0.0f)) || top_k < 0 || !std::isfinite(repetition_penalty) ||
        repetition_penalty <= 0.0f) {
        throw std::invalid_argument(std::string("LFM3-Audio ") + what + " sampling needs temperature >= 0, top_k >= 0 and repetition_penalty > 0");
    }
    sampling::HfSamplingOptions out;
    out.do_sample = !(temperature ? *temperature == 0.0f : top_k == 0);
    out.temperature = out.do_sample ? temperature.value_or(1.0f) : 1.0f;
    out.top_k = top_k;
    out.repetition_penalty = repetition_penalty;
    return out;
}

}  // namespace

Lfm3AudioSampler::Lfm3AudioSampler(const Lfm3AudioSampling & sampling, uint64_t seed)
    : sampling_(sampling),
      options_(hf_options(sampling.temperature, sampling.top_k, sampling.repetition_penalty, "audio")),
      rng_(static_cast<uint32_t>(seed)) {
    if (sampling.penalty_window < 0) {
        throw std::invalid_argument("LFM3-Audio audio repetition penalty window must be >= 0");
    }
}

Lfm3Codes Lfm3AudioSampler::next_order() {
    Lfm3Codes order;
    std::iota(order.begin(), order.end(), 0);
    for (size_t i = order.size() - 1; i > 0; --i) {
        std::swap(order[i], order[rng_() % (i + 1)]);
    }
    return order;
}

int32_t Lfm3AudioSampler::pick(int64_t codebook, std::vector<float> & logits) {
    if (codebook < 0 || codebook >= kCodebooks) {
        throw std::invalid_argument("LFM3-Audio audio codebook " + std::to_string(codebook) + " is outside [0, 12)");
    }
    require_finite(logits, "audio code");
    scratch_.reserve_vocab(logits.size());
    return sampler_.sample(logits, history_[static_cast<size_t>(codebook)], options_, scratch_, rng_, nullptr, "lfm3_audio audio code");
}

void Lfm3AudioSampler::commit(const Lfm3Codes & frame) {
    const auto window = static_cast<size_t>(sampling_.penalty_window);
    for (size_t k = 0; k < frame.size(); ++k) {
        if (frame[k] < 0) {
            throw std::invalid_argument("LFM3-Audio audio history got a negative code");
        }
        auto & history = history_[k];
        history.push_back(frame[k]);
        if (history.size() > window) {
            history.erase(history.begin(), history.end() - static_cast<std::ptrdiff_t>(window));
        }
    }
}

Lfm3TextSampler::Lfm3TextSampler(const Lfm3TextSampling & sampling, uint64_t seed)
    : options_(hf_options(sampling.temperature, sampling.top_k, 1.0f, "text")),
      rng_(static_cast<uint32_t>(text_seed(seed))) {}

int32_t Lfm3TextSampler::pick(std::vector<float> & logits, const std::vector<int32_t> & suppress) {
    require_finite(logits, "text");
    for (const auto id : suppress) {
        if (id < 0 || static_cast<size_t>(id) >= logits.size()) {
            throw std::invalid_argument("LFM3-Audio suppressed token " + std::to_string(id) + " is outside the vocabulary");
        }
        logits[static_cast<size_t>(id)] = -std::numeric_limits<float>::infinity();
    }
    scratch_.reserve_vocab(logits.size());
    return sampler_.sample(logits, {}, options_, scratch_, rng_, nullptr, "lfm3_audio text token");
}

}  // namespace engine::community_models::lfm3_audio
