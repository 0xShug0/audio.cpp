// The LFM3-Audio samplers on hand-made logits: the greedy rule, the
// repetition penalty (once per unique code of that codebook's window), top-k
// ties, suppression and portable MaskGIT orders.
#include "../lfm2_audio/lfm2_audio_test_package.h"
#include "engine/community_models/lfm3_audio/sampling.h"
#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace engine::community_models::lfm3_audio;
using engine::test::require;
using engine::test::require_eq;
using lfm2_audio_test::require_throws;

Lfm3AudioSampling audio(std::optional<float> temperature, int64_t top_k = 0, float penalty = 1.0f, int64_t window = 64) {
    Lfm3AudioSampling out;
    out.temperature = temperature;
    out.top_k = top_k;
    out.repetition_penalty = penalty;
    out.penalty_window = window;
    return out;
}

Lfm3Codes frame(int32_t code) {
    Lfm3Codes out;
    out.fill(code);
    return out;
}

int32_t pick(Lfm3AudioSampler & sampler, int64_t codebook, std::vector<float> logits) {
    return sampler.pick(codebook, logits);
}

std::set<int32_t> draws(Lfm3AudioSampler & sampler, const std::vector<float> & logits, int count) {
    std::set<int32_t> seen;
    for (int i = 0; i < count; ++i) seen.insert(pick(sampler, 0, logits));
    return seen;
}

void test_orders() {
    // Fisher-Yates over raw mt19937 output: the same on libc++ and libstdc++.
    Lfm3AudioSampler a(audio(0.4f), 0);
    Lfm3AudioSampler b(audio(0.4f), 0);
    Lfm3AudioSampler c(audio(0.4f), 1);
    const Lfm3Codes golden0 = {11, 4, 0, 5, 10, 1, 2, 9, 6, 3, 7, 8};
    const Lfm3Codes golden1 = {9, 2, 0, 10, 11, 8, 6, 7, 5, 4, 3, 1};
    std::vector<Lfm3Codes> seen;
    for (int i = 0; i < 50; ++i) {
        const auto order = a.next_order();
        require(order == b.next_order(), "same seed must give the same orders");
        auto sorted = order;
        std::sort(sorted.begin(), sorted.end());
        for (int32_t k = 0; k < 12; ++k) require_eq(sorted[static_cast<size_t>(k)], k, "order is a permutation");
        if (i == 0) {
            std::cout << "seed 0 first order:";
            for (const auto k : order) std::cout << " " << k;
            std::cout << "\n";
            require(order == golden0, "seed 0's first order must match the portable golden value");
        }
        seen.push_back(order);
    }
    const auto first_c = c.next_order();
    std::cout << "seed 1 first order:";
    for (const auto k : first_c) std::cout << " " << k;
    std::cout << "\n";
    require(first_c == golden1, "seed 1's first order must match the portable golden value");
    std::sort(seen.begin(), seen.end());
    require(std::unique(seen.begin(), seen.end()) - seen.begin() > 45, "orders must vary from frame to frame");
}

void test_greedy_rule() {
    const std::vector<float> logits = {0.5f, 2.0f, 2.0f, -1.0f};
    for (uint64_t seed = 0; seed < 5; ++seed) {
        Lfm3AudioSampler zero(audio(0.0f), seed);
        Lfm3AudioSampler unset(audio(std::nullopt), seed);
        require_eq(pick(zero, 0, logits), 1, "T = 0 picks the first max");
        require_eq(pick(unset, 0, logits), 1, "T unset, no top_k picks the first max");
    }
    // T unset with top_k samples at T = 1 among the top k (both tied maxima appear).
    Lfm3AudioSampler topk(audio(std::nullopt, 2), 3);
    const auto seen = draws(topk, logits, 200);
    require(seen == std::set<int32_t>({1, 2}), "T unset with top_k 2 samples the two tied maxima");
}

void test_repetition_penalty() {
    // Once per unique code: 2.0 / 1.2 = 1.67 > 1.5 keeps code 0; three applications (1.16) would not.
    Lfm3AudioSampler once(audio(0.0f, 0, 1.2f), 0);
    for (int i = 0; i < 3; ++i) once.commit(frame(0));
    require_eq(pick(once, 0, {2.0f, 1.5f}), 0, "penalty applies once per unique code");
    require_eq(pick(once, 0, {2.0f, 1.7f}), 1, "a positive score is divided");
    require_eq(pick(once, 0, {-1.0f, -1.1f}), 1, "a negative score is multiplied");

    Lfm3AudioSampler per(audio(0.0f, 0, 1.2f), 0);
    Lfm3Codes f = frame(5);
    f[3] = 0;
    per.commit(f);
    require_eq(pick(per, 0, {2.0f, 1.7f}), 0, "codebook 0 has no code 0 in its history");
    require_eq(pick(per, 3, {2.0f, 1.7f}), 1, "codebook 3 penalizes its code 0");

    Lfm3AudioSampler window(audio(0.0f, 0, 1.2f, 4), 0);
    window.commit(frame(0));
    for (int i = 0; i < 3; ++i) window.commit(frame(5));
    require_eq(pick(window, 0, {2.0f, 1.7f}), 1, "code 0 is in the last 4 frames");
    window.commit(frame(5));
    require_eq(pick(window, 0, {2.0f, 1.7f}), 0, "code 0 fell out of the last 4 frames");

    Lfm3AudioSampler off(audio(0.0f, 0, 1.0f), 0);
    off.commit(frame(0));
    require_eq(pick(off, 0, {2.0f, 1.9f}), 0, "penalty 1 is off");
}

void test_top_k_ties() {
    // top_k 1 keeps every logit equal to the largest one.
    Lfm3AudioSampler sampler(audio(1.0f, 1), 9);
    require(draws(sampler, {1.0f, 3.0f, 3.0f, 0.0f}, 200) == std::set<int32_t>({1, 2}), "top-k keeps ties");
    Lfm3AudioSampler wide(audio(1.0f), 9);
    require(draws(wide, {0.0f, 0.1f, 0.2f, 0.3f}, 400).size() == 4, "temperature sampling covers the vocabulary");
}

void test_text() {
    const std::vector<float> logits = {1.0f, 3.0f, 2.0f, 2.9f, -1.0f};
    Lfm3TextSampler greedy({}, 0);
    auto row = logits;
    require_eq(greedy.pick(row, {}), 1, "text default is greedy");
    row = logits;
    require_eq(greedy.pick(row, {1}), 3, "a suppressed id is never picked");
    require(std::isinf(row[1]) && row[1] < 0, "suppression writes -inf");

    Lfm3TextSampling sampled;
    sampled.temperature = 0.8f;
    auto run = [&](uint64_t seed) {
        Lfm3TextSampler sampler(sampled, seed);
        std::vector<int32_t> out;
        for (int i = 0; i < 40; ++i) {
            auto r = logits;
            out.push_back(sampler.pick(r, {}));
        }
        return out;
    };
    require(run(7) == run(7), "text sampling is deterministic per seed");
}

void test_errors() {
    require_throws([] { Lfm3AudioSampler s(audio(-0.1f), 0); }, "negative temperature");
    require_throws([] { Lfm3AudioSampler s(audio(0.4f, -1), 0); }, "negative top_k");
    require_throws([] { Lfm3AudioSampler s(audio(0.4f, 0, 0.0f), 0); }, "zero repetition penalty");
    require_throws([] { Lfm3AudioSampler s(audio(0.4f, 0, 1.2f, -1), 0); }, "negative window");
    Lfm3AudioSampler sampler(audio(0.0f), 0);
    require_throws([&] { pick(sampler, 12, {1.0f}); }, "codebook out of range");
    require_throws([&] { pick(sampler, 0, {1.0f, NAN}); }, "non-finite logits");
    require_throws([&] { pick(sampler, 0, {}); }, "empty logits");
    require_throws([&] { sampler.commit(frame(-1)); }, "negative code");
    Lfm3TextSampler text({}, 0);
    std::vector<float> row = {1.0f, 2.0f};
    require_throws([&] { text.pick(row, {2}); }, "suppressed id out of range");
}

}  // namespace

int main() {
    try {
        test_orders();
        test_greedy_rule();
        test_repetition_penalty();
        test_top_k_ties();
        test_text();
        test_errors();
        std::cout << "PASS: lfm3_audio sampling\n";
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
