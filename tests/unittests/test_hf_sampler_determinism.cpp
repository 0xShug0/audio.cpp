// Pins the Torch-CPU multinomial stream of engine::sampling::HfTokenSampler.
//
// hf_sampler.cpp used to rebuild a Mersenne Twister and fast-forward it by
// call_index * vocab * 2 words on every single sample, which is quadratic in the number of
// generated tokens and is paid on Metal by FireRedTTS3, MagpieTTS, MuScriptor and
// PersonaPlex. It now keeps a small set of generators alive across calls instead.
//
// The oracle below is a verbatim copy of the *old* rebuild-and-rewind algorithm. Every
// assertion in this file compares the shipping sampler against it, so any change that
// alters the RNG stream for a given seed fails here rather than silently changing
// generated audio.

#include "engine/framework/sampling/hf_sampler.h"
#include "engine/framework/sampling/torch_random.h"

#include "test_assert.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::test::require;
using engine::test::require_eq;

// Verbatim copy of hf_sampler.cpp's TorchCpuMt19937 as it stood before the cache landed.
class ReferenceMt19937 {
public:
    explicit ReferenceMt19937(uint64_t seed) {
        state_[0] = static_cast<uint32_t>(seed & 0xffffffffU);
        for (int index = 1; index < kStateN; ++index) {
            state_[index] =
                1812433253U * (state_[index - 1] ^ (state_[index - 1] >> 30U)) + static_cast<uint32_t>(index);
        }
        left_ = 1;
        next_ = 0;
    }

    uint32_t random() {
        if (--left_ == 0) {
            next_state();
        }
        uint32_t y = state_[next_++];
        y ^= y >> 11U;
        y ^= (y << 7U) & 0x9d2c5680U;
        y ^= (y << 15U) & 0xefc60000U;
        y ^= y >> 18U;
        return y;
    }

    uint64_t random64() {
        const uint32_t high = random();
        const uint32_t low = random();
        return (static_cast<uint64_t>(high) << 32U) | static_cast<uint64_t>(low);
    }

    void discard(uint64_t count) {
        for (uint64_t index = 0; index < count; ++index) {
            (void)random();
        }
    }

private:
    static constexpr int kStateN = 624;
    static constexpr int kStateM = 397;
    static constexpr uint32_t kMatrixA = 0x9908b0dfU;
    static constexpr uint32_t kUpperMask = 0x80000000U;
    static constexpr uint32_t kLowerMask = 0x7fffffffU;

    static uint32_t mix_bits(uint32_t lhs, uint32_t rhs) {
        return (lhs & kUpperMask) | (rhs & kLowerMask);
    }

    static uint32_t twist(uint32_t lhs, uint32_t rhs) {
        return (mix_bits(lhs, rhs) >> 1U) ^ ((rhs & 1U) != 0U ? kMatrixA : 0U);
    }

    void next_state() {
        uint32_t * p = state_.data();
        left_ = kStateN;
        next_ = 0;

        for (int index = kStateN - kStateM + 1; --index != 0; ++p) {
            *p = p[kStateM] ^ twist(p[0], p[1]);
        }
        for (int index = kStateM; --index != 0; ++p) {
            *p = p[kStateM - kStateN] ^ twist(p[0], p[1]);
        }
        *p = p[kStateM - kStateN] ^ twist(p[0], state_[0]);
    }

    int left_ = 1;
    uint32_t next_ = 0;
    std::array<uint32_t, kStateN> state_{};
};

float reference_exponential(ReferenceMt19937 & rng) {
    constexpr uint64_t mask = (uint64_t{1} << std::numeric_limits<double>::digits) - 1U;
    constexpr double divisor = 1.0 / static_cast<double>(uint64_t{1} << std::numeric_limits<double>::digits);
    const double uniform = static_cast<double>(rng.random64() & mask) * divisor;
    return static_cast<float>(-std::log1p(-uniform));
}

// Verbatim copy of the old sample_torch_cpu_multinomial: rebuild, fast-forward, sample.
int32_t reference_sample(const std::vector<float> & scores, uint64_t seed, uint64_t call_index) {
    ReferenceMt19937 rng(seed);
    rng.discard(call_index * static_cast<uint64_t>(scores.size()) * 2ULL);

    float max_score = -std::numeric_limits<float>::infinity();
    for (const float score : scores) {
        if (std::isfinite(score)) {
            max_score = std::max(max_score, score);
        }
    }
    if (!std::isfinite(max_score)) {
        throw std::runtime_error("reference sampler has no finite logits");
    }
    double best_rank = -std::numeric_limits<double>::infinity();
    int32_t best_token = -1;
    for (size_t token = 0; token < scores.size(); ++token) {
        const float exponential = reference_exponential(rng);
        if (!std::isfinite(scores[token])) {
            continue;
        }
        const double weight = std::exp(static_cast<double>(scores[token] - max_score));
        const double rank = weight / static_cast<double>(exponential);
        if (rank > best_rank) {
            best_rank = rank;
            best_token = static_cast<int32_t>(token);
        }
    }
    if (best_token < 0) {
        throw std::runtime_error("reference sampler failed to select a token");
    }
    return best_token;
}

int32_t engine_sample(const std::vector<float> & scores, uint64_t seed, uint64_t call_index) {
    // cuda_fast_path stays false, which is what the policy resolves to on every non-CUDA
    // backend, so this is exactly the path Metal takes.
    engine::sampling::TorchCudaSamplingPolicy policy;
    engine::sampling::HfTorchSamplingState torch_state;
    torch_state.policy = &policy;
    torch_state.seed = seed;
    torch_state.call_index = call_index;
    engine::sampling::HfSamplerScratch scratch;
    std::mt19937 unused_fallback(0);
    return engine::sampling::HfTokenSampler::sample_from_processed_scores(
        scores,
        scratch,
        unused_fallback,
        &torch_state,
        "hf sampler determinism test");
}

// A logits row with enough spread that the sampled token genuinely varies with the RNG.
std::vector<float> make_scores(size_t vocab_size, uint32_t shape_seed) {
    std::mt19937 shaper(shape_seed);
    std::uniform_real_distribution<float> spread(-4.0F, 4.0F);
    std::vector<float> scores(vocab_size, 0.0F);
    for (float & score : scores) {
        score = spread(shaper);
    }
    return scores;
}

size_t distinct_count(const std::vector<int32_t> & tokens) {
    return std::set<int32_t>(tokens.begin(), tokens.end()).size();
}

// A sequential walk is the case the cache is built for: it must consume the identical
// stream the rebuild-and-rewind algorithm did.
void test_sequential_walk_matches_rebuild_and_rewind() {
    constexpr uint64_t seed = 20240301;
    constexpr size_t vocab_size = 97;
    constexpr uint64_t steps = 64;
    const auto scores = make_scores(vocab_size, 11);

    std::vector<int32_t> engine_tokens;
    std::vector<int32_t> reference_tokens;
    engine_tokens.reserve(static_cast<size_t>(steps));
    reference_tokens.reserve(static_cast<size_t>(steps));
    for (uint64_t call_index = 0; call_index < steps; ++call_index) {
        engine_tokens.push_back(engine_sample(scores, seed, call_index));
        reference_tokens.push_back(reference_sample(scores, seed, call_index));
    }

    for (size_t index = 0; index < engine_tokens.size(); ++index) {
        require_eq(
            engine_tokens[index],
            reference_tokens[index],
            "sequential walk call_index " + std::to_string(index));
    }
    // Guards against the test passing because every call returns the same token.
    require(
        distinct_count(engine_tokens) >= size_t{8},
        "sequential walk produced only " + std::to_string(distinct_count(engine_tokens)) +
            " distinct tokens; the fixture is not exercising the RNG");
}

// Out-of-order call_index values must miss the cache and rebuild, still reproducing the
// exact stream position the old code would have reached.
void test_random_access_matches_rebuild_and_rewind() {
    constexpr uint64_t seed = 777;
    constexpr size_t vocab_size = 41;
    const auto scores = make_scores(vocab_size, 22);
    const std::vector<uint64_t> call_indices = {5, 2, 9, 0, 5, 7, 1, 9, 3, 0};

    for (const uint64_t call_index : call_indices) {
        require_eq(
            engine_sample(scores, seed, call_index),
            reference_sample(scores, seed, call_index),
            "random access call_index " + std::to_string(call_index));
    }
}

// Several generations running on one thread must not contaminate one another. The first
// group fits in the cache so every call is a hit; the second deliberately exceeds it so
// every call is an eviction and a rebuild. Both must reproduce the reference stream.
void test_interleaved_seeds_stay_independent() {
    constexpr size_t vocab_size = 53;
    const auto scores = make_scores(vocab_size, 33);
    const std::vector<uint64_t> fitting_seeds = {1, 2, 3};
    const std::vector<uint64_t> thrashing_seeds = {11, 12, 13, 14, 15, 16, 17};

    for (uint64_t call_index = 0; call_index < 12; ++call_index) {
        for (const uint64_t seed : fitting_seeds) {
            require_eq(
                engine_sample(scores, seed, call_index),
                reference_sample(scores, seed, call_index),
                "interleaved seed " + std::to_string(seed) + " call_index " + std::to_string(call_index));
        }
    }
    for (uint64_t call_index = 0; call_index < 6; ++call_index) {
        for (const uint64_t seed : thrashing_seeds) {
            require_eq(
                engine_sample(scores, seed, call_index),
                reference_sample(scores, seed, call_index),
                "thrashing seed " + std::to_string(seed) + " call_index " + std::to_string(call_index));
        }
    }
}

// Filtered-out tokens are -inf but still consume two RNG words each. If that invariant
// ever breaks, the cached generator drifts and this fails.
void test_filtered_scores_still_advance_the_stream() {
    constexpr uint64_t seed = 4242;
    constexpr size_t vocab_size = 128;
    auto scores = make_scores(vocab_size, 44);
    for (size_t index = 0; index < scores.size(); ++index) {
        if (index % 3 != 0) {
            scores[index] = -std::numeric_limits<float>::infinity();
        }
    }

    std::vector<int32_t> tokens;
    for (uint64_t call_index = 0; call_index < 32; ++call_index) {
        const int32_t token = engine_sample(scores, seed, call_index);
        require_eq(
            token,
            reference_sample(scores, seed, call_index),
            "filtered scores call_index " + std::to_string(call_index));
        require(token % 3 == 0, "sampler selected a filtered-out token");
        tokens.push_back(token);
    }
    require(distinct_count(tokens) >= size_t{5}, "filtered fixture is not exercising the RNG");
}

// Two identical walks in the same process — one on a cold cache slot, one on a warm one —
// must return the identical token sequence.
void test_repeated_walk_is_reproducible() {
    constexpr uint64_t seed = 99991;
    constexpr size_t vocab_size = 71;
    constexpr uint64_t steps = 40;
    const auto scores = make_scores(vocab_size, 55);

    std::vector<int32_t> first;
    for (uint64_t call_index = 0; call_index < steps; ++call_index) {
        first.push_back(engine_sample(scores, seed, call_index));
    }
    // Evict the warm slot with unrelated traffic before replaying.
    const auto filler = make_scores(23, 66);
    for (uint64_t seed_offset = 0; seed_offset < 8; ++seed_offset) {
        (void)engine_sample(filler, 500 + seed_offset, seed_offset);
    }
    std::vector<int32_t> second;
    for (uint64_t call_index = 0; call_index < steps; ++call_index) {
        second.push_back(engine_sample(scores, seed, call_index));
    }

    require_eq(second.size(), first.size(), "repeated walk length");
    for (size_t index = 0; index < first.size(); ++index) {
        require_eq(second[index], first[index], "repeated walk call_index " + std::to_string(index));
    }
    require(distinct_count(first) >= size_t{8}, "repeated walk fixture is not exercising the RNG");
}

// Vocabulary size is part of the stream position. A generation that changes it mid-walk
// (a different sampler head) must not reuse the previous generator.
void test_vocab_size_change_invalidates_the_stream_position() {
    constexpr uint64_t seed = 31337;
    const auto small = make_scores(19, 77);
    const auto large = make_scores(83, 88);

    for (uint64_t call_index = 0; call_index < 10; ++call_index) {
        require_eq(
            engine_sample(small, seed, call_index),
            reference_sample(small, seed, call_index),
            "small vocab call_index " + std::to_string(call_index));
        require_eq(
            engine_sample(large, seed, call_index),
            reference_sample(large, seed, call_index),
            "large vocab call_index " + std::to_string(call_index));
    }
}

}  // namespace

int main() {
    try {
        test_sequential_walk_matches_rebuild_and_rewind();
        test_random_access_matches_rebuild_and_rewind();
        test_interleaved_seeds_stay_independent();
        test_filtered_scores_still_advance_the_stream();
        test_repeated_walk_is_reproducible();
        test_vocab_size_change_invalidates_the_stream_position();
        std::cout << "hf_sampler_determinism_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "hf_sampler_determinism_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
