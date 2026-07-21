#include "engine/models/higgs_audio_tts/sampler.h"

#include "engine/framework/sampling/torch_random.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace engine::models::higgs_audio_tts {
namespace {

constexpr float kGreedyTemperatureThreshold = 1.0e-5F;

struct SamplerScratch {
    std::vector<float> & scores;
    std::vector<float> & probs;
    std::vector<int64_t> & order;
    std::vector<int64_t> & kept;
};

void require_cuda_sampling_policy(const HiggsCudaSamplingPolicy & policy) {
    if (policy.multiprocessor_count <= 0 || policy.max_threads_per_multiprocessor <= 0) {
        throw std::runtime_error("Higgs TTS stochastic sampler requires CUDA "
                                 "sampling device properties");
    }
}

uint32_t rotl32(uint32_t value, int shift) {
    return static_cast<uint32_t>((value << shift) | (value >> (32 - shift)));
}

uint32_t fmix32(uint32_t value) {
    value ^= value >> 16;
    value *= 0x85EBCA6Bu;
    value ^= value >> 13;
    value *= 0xC2B2AE35u;
    value ^= value >> 16;
    return value;
}

uint32_t murmur3_mix(uint32_t hash, uint32_t key) {
    key *= 0xCC9E2D51u;
    key = rotl32(key, 15);
    key *= 0x1B873593u;
    hash ^= key;
    hash = rotl32(hash, 13);
    hash = hash * 5u + 0xE6546B64u;
    return hash;
}

uint32_t sglang_murmur_hash32(uint64_t seed, uint32_t position, uint32_t column) {
    uint32_t hash = 0;
    hash = murmur3_mix(hash, static_cast<uint32_t>(seed & 0xFFFFFFFFull));
    hash = murmur3_mix(hash, static_cast<uint32_t>((seed >> 32) & 0xFFFFFFFFull));
    hash = murmur3_mix(hash, position);
    hash = murmur3_mix(hash, column);
    hash ^= 16u;
    return fmix32(hash);
}

int32_t argmax_row(const float * logits, int64_t vocab_size) {
    int32_t best = 0;
    float best_value = logits[0];
    for (int64_t i = 1; i < vocab_size; ++i) {
        const float value = logits[i];
        if (value > best_value) {
            best_value = value;
            best = static_cast<int32_t>(i);
        }
    }
    return best;
}

float finite_max(const std::vector<float> & scores, const std::vector<int64_t> & candidates) {
    float max_score = -std::numeric_limits<float>::infinity();
    if (candidates.empty()) {
        for (const float score : scores) {
            if (std::isfinite(score)) {
                max_score = std::max(max_score, score);
            }
        }
    } else {
        for (const int64_t index : candidates) {
            const float score = scores[static_cast<size_t>(index)];
            if (std::isfinite(score)) {
                max_score = std::max(max_score, score);
            }
        }
    }
    if (!std::isfinite(max_score)) {
        throw std::runtime_error("Higgs TTS sampler kept no finite logits");
    }
    return max_score;
}

void scores_to_probs(SamplerScratch & scratch, int64_t vocab_size) {
    const float max_score = finite_max(scratch.scores, scratch.kept);
    double total = 0.0;
    scratch.probs.assign(static_cast<size_t>(vocab_size), 0.0F);
    if (scratch.kept.empty()) {
        for (int64_t i = 0; i < vocab_size; ++i) {
            const float score = scratch.scores[static_cast<size_t>(i)];
            if (std::isfinite(score)) {
                const float value =
                    static_cast<float>(std::exp(static_cast<double>(score - max_score)));
                scratch.probs[static_cast<size_t>(i)] = value;
                total += static_cast<double>(value);
            }
        }
    } else {
        for (const int64_t index : scratch.kept) {
            const float value = static_cast<float>(std::exp(
                static_cast<double>(scratch.scores[static_cast<size_t>(index)] - max_score)));
            scratch.probs[static_cast<size_t>(index)] = value;
            total += static_cast<double>(value);
        }
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        throw std::runtime_error("Higgs TTS sampler probability mass is invalid");
    }
    const float inv_total = static_cast<float>(1.0 / total);
    for (float & prob : scratch.probs) {
        prob *= inv_total;
    }
}

void renormalize_probs(SamplerScratch & scratch) {
    double total = 0.0;
    if (scratch.kept.empty()) {
        for (const float prob : scratch.probs) {
            total += static_cast<double>(prob);
        }
    } else {
        for (const int64_t index : scratch.kept) {
            total += static_cast<double>(scratch.probs[static_cast<size_t>(index)]);
        }
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        throw std::runtime_error("Higgs TTS sampler probability mass is invalid");
    }
    const float inv_total = static_cast<float>(1.0 / total);
    if (scratch.kept.empty()) {
        for (float & prob : scratch.probs) {
            prob *= inv_total;
        }
    } else {
        for (const int64_t index : scratch.kept) {
            scratch.probs[static_cast<size_t>(index)] *= inv_total;
        }
    }
}

void apply_top_k_to_probs(SamplerScratch & scratch, int64_t vocab_size, int64_t top_k) {
    scratch.kept.clear();
    if (top_k <= 0 || top_k >= vocab_size) {
        return;
    }
    scratch.order.resize(static_cast<size_t>(vocab_size));
    std::iota(scratch.order.begin(), scratch.order.end(), 0);
    auto kth = scratch.order.begin() + static_cast<std::ptrdiff_t>(top_k - 1);
    std::nth_element(
        scratch.order.begin(), kth, scratch.order.end(), [&](int64_t lhs, int64_t rhs) {
            return scratch.probs[static_cast<size_t>(lhs)] >
                   scratch.probs[static_cast<size_t>(rhs)];
        });
    const float threshold = scratch.probs[static_cast<size_t>(*kth)];
    scratch.kept.reserve(static_cast<size_t>(top_k));
    for (int64_t i = 0; i < vocab_size; ++i) {
        auto & prob = scratch.probs[static_cast<size_t>(i)];
        if (prob < threshold) {
            prob = 0.0F;
        } else {
            scratch.kept.push_back(i);
        }
    }
    renormalize_probs(scratch);
}

void apply_top_k_to_scores(SamplerScratch & scratch, int64_t vocab_size, int64_t top_k) {
    scratch.kept.clear();
    if (top_k <= 0 || top_k >= vocab_size) {
        return;
    }
    scratch.order.resize(static_cast<size_t>(vocab_size));
    std::iota(scratch.order.begin(), scratch.order.end(), 0);
    auto kth = scratch.order.begin() + static_cast<std::ptrdiff_t>(top_k - 1);
    std::nth_element(
        scratch.order.begin(), kth, scratch.order.end(), [&](int64_t lhs, int64_t rhs) {
            return scratch.scores[static_cast<size_t>(lhs)] >
                   scratch.scores[static_cast<size_t>(rhs)];
        });
    const float threshold = scratch.scores[static_cast<size_t>(*kth)];
    scratch.kept.reserve(static_cast<size_t>(top_k));
    for (int64_t i = 0; i < vocab_size; ++i) {
        if (scratch.scores[static_cast<size_t>(i)] >= threshold) {
            scratch.kept.push_back(i);
        }
    }
}

void apply_top_p_to_probs(SamplerScratch & scratch, int64_t vocab_size, float top_p) {
    if (!(top_p < 1.0F)) {
        return;
    }
    if (!(top_p > 0.0F)) {
        throw std::runtime_error("Higgs TTS sampler top_p must be positive");
    }
    if (scratch.kept.empty()) {
        scratch.kept.reserve(static_cast<size_t>(vocab_size));
        for (int64_t i = 0; i < vocab_size; ++i) {
            if (scratch.probs[static_cast<size_t>(i)] > 0.0F) {
                scratch.kept.push_back(i);
            }
        }
    }
    if (scratch.kept.empty()) {
        throw std::runtime_error("Higgs TTS sampler top-p kept no probabilities");
    }
    std::stable_sort(scratch.kept.begin(), scratch.kept.end(), [&](int64_t lhs, int64_t rhs) {
        return scratch.probs[static_cast<size_t>(lhs)] > scratch.probs[static_cast<size_t>(rhs)];
    });

    float cumulative = 0.0F;
    float threshold = scratch.probs[static_cast<size_t>(scratch.kept.front())];
    for (const int64_t index : scratch.kept) {
        threshold = scratch.probs[static_cast<size_t>(index)];
        cumulative += threshold;
        if (cumulative >= top_p) {
            break;
        }
    }

    size_t kept_count = 0;
    for (const int64_t index : scratch.kept) {
        auto & prob = scratch.probs[static_cast<size_t>(index)];
        if (prob < threshold) {
            prob = 0.0F;
        } else {
            scratch.kept[kept_count++] = index;
        }
    }
    scratch.kept.resize(kept_count);
    renormalize_probs(scratch);
}

double sglang_gumbel_from_hash(uint32_t hashed) {
    constexpr double kUint32Max = static_cast<double>(std::numeric_limits<uint32_t>::max());
    const double x = static_cast<double>(hashed) / kUint32Max;
    const double log_x = std::max(std::log(x), std::numeric_limits<double>::lowest());
    return -std::log(-log_x);
}

int32_t sample_seeded_sglang_gumbel(const std::vector<float> & probs,
                                    const std::vector<int64_t> & candidates,
                                    uint64_t seed,
                                    uint64_t position) {
    double best_rank = -std::numeric_limits<double>::infinity();
    int32_t best = -1;
    const auto sample_one = [&](int64_t i) {
        const float prob = probs[static_cast<size_t>(i)];
        if (!(prob > 0.0F)) {
            return;
        }
        const double logprob = std::log(static_cast<double>(prob));
        const uint32_t hashed = sglang_murmur_hash32(
            seed, static_cast<uint32_t>(position & 0xFFFFFFFFull), static_cast<uint32_t>(i));
        const double rank = logprob + sglang_gumbel_from_hash(hashed);
        if (rank > best_rank) {
            best_rank = rank;
            best = static_cast<int32_t>(i);
        }
    };
    if (candidates.empty()) {
        for (int64_t i = 0; i < static_cast<int64_t>(probs.size()); ++i) {
            sample_one(i);
        }
    } else {
        for (const int64_t index : candidates) {
            sample_one(index);
        }
    }
    if (best < 0) {
        throw std::runtime_error("Higgs TTS sampler failed to select a codebook token");
    }
    return best;
}

int32_t sample_unseeded_torch_multinomial(const std::vector<float> & probs,
                                          const std::vector<int64_t> & candidates,
                                          uint64_t seed,
                                          uint64_t call_index,
                                          const HiggsCudaSamplingPolicy & policy,
                                          std::mt19937 & fallback_rng) {
    const int64_t vocab_size = static_cast<int64_t>(probs.size());
    if (!policy.cuda_fast_path) {
        std::vector<double> weights;
        if (candidates.empty()) {
            weights.reserve(probs.size());
            for (float prob : probs) {
                weights.push_back(static_cast<double>(std::max(prob, 0.0F)));
            }
            std::discrete_distribution<int64_t> distribution(weights.begin(), weights.end());
            return static_cast<int32_t>(distribution(fallback_rng));
        }
        weights.reserve(candidates.size());
        for (const int64_t index : candidates) {
            weights.push_back(static_cast<double>(std::max(probs[static_cast<size_t>(index)], 0.0F)));
        }
        std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
        return static_cast<int32_t>(candidates[distribution(fallback_rng)]);
    }
    require_cuda_sampling_policy(policy);

    double best_rank = -std::numeric_limits<double>::infinity();
    int32_t best = -1;
    const auto sample_one = [&](int64_t i) {
        const float prob = probs[static_cast<size_t>(i)];
        if (!(prob > 0.0F)) {
            return;
        }
        const float exponential = engine::sampling::torch_cuda_tensor_iterator_exponential_element(
            seed,
            static_cast<uint64_t>(vocab_size),
            static_cast<uint64_t>(i),
            call_index,
            policy.multiprocessor_count,
            policy.max_threads_per_multiprocessor);
        const double rank = static_cast<double>(prob) / static_cast<double>(exponential);
        if (rank > best_rank) {
            best_rank = rank;
            best = static_cast<int32_t>(i);
        }
    };
    if (candidates.empty()) {
        for (int64_t i = 0; i < vocab_size; ++i) {
            sample_one(i);
        }
    } else {
        for (const int64_t index : candidates) {
            sample_one(index);
        }
    }
    if (best < 0) {
        throw std::runtime_error("Higgs TTS sampler failed to select a codebook token");
    }
    return best;
}

int32_t sample_codebook_row(const float * logits,
                            int64_t vocab_size,
                            const HiggsSamplingOptions & options,
                            uint64_t call_index,
                            SamplerScratch & scratch) {
    if (logits == nullptr || vocab_size <= 0) {
        throw std::runtime_error("Higgs TTS sampler requires logits");
    }
    if (options.temperature <= kGreedyTemperatureThreshold ||
        (options.top_k.has_value() && *options.top_k == 1)) {
        return argmax_row(logits, vocab_size);
    }
    if (!(options.temperature > 0.0F) || !std::isfinite(options.temperature)) {
        throw std::runtime_error("Higgs TTS sampler temperature must be finite and positive");
    }

    scratch.kept.clear();
    scratch.scores.resize(static_cast<size_t>(vocab_size));
    for (int64_t i = 0; i < vocab_size; ++i) {
        scratch.scores[static_cast<size_t>(i)] = logits[i] / options.temperature;
    }
    bool top_k_applied_to_scores = false;
    if (options.top_k.has_value()) {
        if (*options.top_k < 0) {
            throw std::runtime_error("Higgs TTS sampler top_k must be non-negative");
        }
        const int64_t top_k = std::min<int64_t>(*options.top_k, vocab_size);
        if (top_k > 0 && top_k < vocab_size) {
            apply_top_k_to_scores(scratch, vocab_size, top_k);
            top_k_applied_to_scores = true;
        }
    }
    scores_to_probs(scratch, vocab_size);
    if (options.top_k.has_value() && !top_k_applied_to_scores) {
        apply_top_k_to_probs(scratch, vocab_size, std::min<int64_t>(*options.top_k, vocab_size));
    }
    if (options.top_p.has_value()) {
        apply_top_p_to_probs(scratch, vocab_size, *options.top_p);
    }
    if (options.has_seed) {
        return sample_seeded_sglang_gumbel(
            scratch.probs, scratch.kept, options.seed & 0x7FFFFFFFull, call_index);
    }
    if (options.fallback_rng == nullptr) {
        throw std::runtime_error("Higgs TTS sampler fallback RNG is missing");
    }
    return sample_unseeded_torch_multinomial(
        scratch.probs, scratch.kept, options.seed, call_index, options.cuda_policy, *options.fallback_rng);
}

} // namespace

HiggsCodebookSampler::HiggsCodebookSampler(int64_t num_codebooks, int64_t codebook_vocab_size)
    : num_codebooks_(num_codebooks), codebook_vocab_size_(codebook_vocab_size) {
    if (num_codebooks_ <= 0 || codebook_vocab_size_ <= 0) {
        throw std::runtime_error("Higgs TTS sampler requires positive codebook dimensions");
    }
    scratch_scores_.reserve(static_cast<size_t>(codebook_vocab_size_));
    scratch_probs_.reserve(static_cast<size_t>(codebook_vocab_size_));
    scratch_order_.reserve(static_cast<size_t>(codebook_vocab_size_));
    scratch_kept_.reserve(static_cast<size_t>(codebook_vocab_size_));
    scratch_codes_.reserve(static_cast<size_t>(num_codebooks_));
}

HiggsSamplerState HiggsCodebookSampler::make_state() const {
    HiggsSamplerState state;
    state.num_codebooks = num_codebooks_;
    state.last_codes.assign(static_cast<size_t>(num_codebooks_), 0);
    return state;
}

const std::vector<int32_t> & HiggsCodebookSampler::step(const float * logits,
                                                        int64_t logits_count,
                                                        HiggsSamplerState & state,
                                                        HiggsSamplingOptions & options) {
    if (state.num_codebooks != num_codebooks_) {
        throw std::runtime_error("Higgs TTS sampler state codebook count mismatch");
    }
    if (logits_count != num_codebooks_ * codebook_vocab_size_) {
        throw std::runtime_error("Higgs TTS sampler logits shape mismatch");
    }

    if (state.generation_done) {
        scratch_codes_.assign(static_cast<size_t>(num_codebooks_), kHiggsStopCode);
        return scratch_codes_;
    }

    scratch_codes_.assign(static_cast<size_t>(num_codebooks_), 0);
    SamplerScratch scratch{scratch_scores_, scratch_probs_, scratch_order_, scratch_kept_};
    for (int64_t codebook = 0; codebook < num_codebooks_; ++codebook) {
        const float * row = logits + static_cast<size_t>(codebook * codebook_vocab_size_);
        scratch_codes_[static_cast<size_t>(codebook)] =
            sample_codebook_row(row,
                                codebook_vocab_size_,
                                options,
                                static_cast<uint64_t>(state.step_count * num_codebooks_ + codebook),
                                scratch);
    }

    if (state.delay_count < num_codebooks_) {
        const int64_t next_codebook = state.delay_count + 1;
        if (next_codebook < num_codebooks_) {
            for (int64_t codebook = next_codebook; codebook < num_codebooks_; ++codebook) {
                scratch_codes_[static_cast<size_t>(codebook)] = kHiggsBocId;
            }
        }
        state.delay_count += 1;
    } else if (state.eoc_countdown.has_value()) {
        *state.eoc_countdown -= 1;
        if (*state.eoc_countdown <= 0) {
            state.generation_done = true;
        }
    } else if (scratch_codes_.front() == kHiggsEocId) {
        if (num_codebooks_ <= 2) {
            state.generation_done = true;
        } else {
            state.eoc_countdown = num_codebooks_ - 2;
        }
    }

    state.step_count += 1;
    if (!state.generation_done) {
        state.last_codes = scratch_codes_;
    }
    return scratch_codes_;
}

} // namespace engine::models::higgs_audio_tts
