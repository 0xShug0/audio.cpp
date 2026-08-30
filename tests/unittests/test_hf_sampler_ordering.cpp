// Pins the warper order of engine::sampling::HfSampler: repetition penalty, then
// temperature, then top-k, then top-p — the HuggingFace order (hf_sampler.cpp:452-454).
//
// The order is load-bearing, not cosmetic: evaluating the nucleus before the temperature
// divides admits a different set of tokens, and the difference grows as the temperature
// moves away from 1.0. docs/reviews/06 F6.8 flagged Fish Audio for using the reversed
// order; that model deliberately reproduces fish-speech's `logits_to_probs`, so the two
// orders coexist on purpose. This test fixes the shared sampler's order so nobody
// "harmonises" them by changing the wrong one.

#include "engine/framework/sampling/hf_sampler.h"

#include "test_assert.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using engine::test::require;
using engine::test::require_close;
using engine::test::require_eq;

// Hand-computed fixture.
//
//   logits      = [2, 1, 0, -1], temperature = 0.5, top_p = 0.9
//
// Temperature first (the HF order, and this repo's):
//   scaled      = [4, 2, 0, -2]
//   softmax     = [0.864665, 0.117057, 0.015842, 0.002144]
//   apply_top_p removes from the tail while the removed mass stays <= 1 - top_p = 0.1:
//     -1 -> 0.002144 (<= 0.1, removed), 0 -> 0.017986 (<= 0.1, removed),
//      1 -> 0.135043 (>  0.1, kept)
//   survivors   = {0, 1}
//   renormalised p(token 0) = 1 / (1 + e^-2) = 0.8807971
//
// Top-p first at T = 1 (the reversed order):
//   softmax     = [0.643914, 0.236883, 0.087144, 0.032058]
//     -1 -> 0.032058 (<= 0.1, removed), 0 -> 0.119202 (> 0.1, kept)
//   survivors   = {0, 1, 2}
constexpr float kTemperature = 0.5F;
constexpr float kTopP = 0.9F;
constexpr float kExpectedTopProbability = 0.8807971F;

const std::vector<float> & fixture_logits() {
    static const std::vector<float> logits = {2.0F, 1.0F, 0.0F, -1.0F};
    return logits;
}

bool is_removed(float score) {
    return score == -std::numeric_limits<float>::infinity();
}

void test_temperature_before_top_p_keeps_two_tokens() {
    std::vector<float> scores = fixture_logits();
    engine::sampling::HfSamplerScratch scratch;
    engine::sampling::HfLogitsProcessor::apply_temperature(scores, kTemperature);
    engine::sampling::HfLogitsProcessor::apply_top_p(scores, kTopP, 1, scratch);

    require_close(scores[0], 4.0F, 0.0F, "temperature-first scaled logit 0");
    require_close(scores[1], 2.0F, 0.0F, "temperature-first scaled logit 1");
    require(!is_removed(scores[0]), "temperature-first removed token 0");
    require(!is_removed(scores[1]), "temperature-first removed token 1");
    require(is_removed(scores[2]), "temperature-first kept token 2 inside the nucleus");
    require(is_removed(scores[3]), "temperature-first kept token 3 inside the nucleus");
}

void test_top_p_before_temperature_keeps_three_tokens() {
    std::vector<float> scores = fixture_logits();
    engine::sampling::HfSamplerScratch scratch;
    engine::sampling::HfLogitsProcessor::apply_top_p(scores, kTopP, 1, scratch);
    engine::sampling::HfLogitsProcessor::apply_temperature(scores, kTemperature);

    require(!is_removed(scores[0]), "top-p-first removed token 0");
    require(!is_removed(scores[1]), "top-p-first removed token 1");
    // The whole point: at T = 0.5 the reversed order admits a token the HF order rejects.
    require(!is_removed(scores[2]), "top-p-first rejected token 2; the two orders no longer differ");
    require(is_removed(scores[3]), "top-p-first kept token 3 inside the nucleus");
    require_close(scores[2], 0.0F, 0.0F, "top-p-first scaled logit 2");
}

void test_sampler_end_to_end_uses_the_temperature_first_nucleus() {
    engine::sampling::HfSampler sampler;
    engine::sampling::HfSamplerScratch scratch;
    engine::sampling::HfSamplingOptions options;
    options.do_sample = true;
    options.temperature = kTemperature;
    options.top_p = kTopP;
    std::mt19937 rng(1234);

    constexpr int64_t draws = 20000;
    std::vector<int64_t> counts(fixture_logits().size(), 0);
    for (int64_t draw = 0; draw < draws; ++draw) {
        const int32_t token = sampler.sample(
            fixture_logits(),
            {},
            options,
            scratch,
            rng,
            nullptr,
            "hf sampler ordering test");
        require(token >= 0 && token < 4, "sampler returned an out-of-range token");
        ++counts[static_cast<size_t>(token)];
    }

    require_eq(counts[2], int64_t{0}, "token 2 was sampled from outside the nucleus");
    require_eq(counts[3], int64_t{0}, "token 3 was sampled from outside the nucleus");
    require(counts[1] > 0, "token 1 never sampled; the nucleus collapsed to a single token");

    const float observed = static_cast<float>(counts[0]) / static_cast<float>(draws);
    // 20 000 draws give a standard error of 0.0023, so 0.01 is a ~4.3 sigma band.
    require_close(observed, kExpectedTopProbability, 0.01F, "post-temperature top token frequency");
}

void test_argmax_ignores_the_warpers_when_sampling_is_off() {
    engine::sampling::HfSampler sampler;
    engine::sampling::HfSamplerScratch scratch;
    engine::sampling::HfSamplingOptions options;
    options.do_sample = false;
    options.temperature = kTemperature;
    options.top_p = kTopP;
    std::mt19937 rng(1);

    const int32_t token = sampler.sample(
        fixture_logits(),
        {},
        options,
        scratch,
        rng,
        nullptr,
        "hf sampler ordering test");
    require_eq(token, int32_t{0}, "greedy decode did not pick the maximum logit");
}

}  // namespace

int main() {
    try {
        test_temperature_before_top_p_keeps_two_tokens();
        test_top_p_before_temperature_keeps_three_tokens();
        test_sampler_end_to_end_uses_the_temperature_first_nucleus();
        test_argmax_ignores_the_warpers_when_sampling_is_off();
        std::cout << "hf_sampler_ordering_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "hf_sampler_ordering_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
