#include "engine/framework/sampling/hf_sampler.h"

#include "test_assert.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using engine::sampling::HfSampler;
using engine::sampling::HfSamplerScratch;
using engine::sampling::HfSamplingOptions;
using engine::sampling::HfTorchSamplingState;
using engine::sampling::TorchCudaSamplingPolicy;

std::vector<float> make_logits(size_t vocab_size, int step) {
    std::vector<float> logits(vocab_size);
    for (size_t index = 0; index < vocab_size; ++index) {
        const double x = static_cast<double>(index + 1) * 0.013 + static_cast<double>(step) * 0.071;
        logits[index] = static_cast<float>(std::sin(x) * 3.0 + std::cos(x * 0.37) * 1.5);
    }
    return logits;
}

std::vector<int32_t> make_history(size_t vocab_size, int step) {
    std::vector<int32_t> history;
    history.reserve(80);
    for (int index = 0; index < 80; ++index) {
        history.push_back(static_cast<int32_t>((index * 37 + step * 11) % static_cast<int>(vocab_size)));
    }
    return history;
}

int32_t reference_argmax(const std::vector<float> & scores) {
    size_t best = 0;
    for (size_t index = 1; index < scores.size(); ++index) {
        if (scores[index] > scores[best]) {
            best = index;
        }
    }
    return static_cast<int32_t>(best);
}

void reference_repetition_penalty(
    std::vector<float> & scores,
    const std::vector<int32_t> & history,
    float penalty) {
    if (penalty == 1.0F) {
        return;
    }
    std::unordered_set<int32_t> seen;
    for (const int32_t token : history) {
        if (token < 0 || static_cast<size_t>(token) >= scores.size()) {
            continue;
        }
        if (!seen.insert(token).second) {
            continue;
        }
        float & score = scores[static_cast<size_t>(token)];
        score = score < 0.0F ? score * penalty : score / penalty;
    }
}

void reference_top_k(std::vector<float> & scores, int64_t top_k, int64_t min_tokens_to_keep) {
    if (top_k <= 0 || scores.empty()) {
        return;
    }
    const int64_t keep = std::min<int64_t>(
        static_cast<int64_t>(scores.size()),
        std::max(top_k, min_tokens_to_keep));
    if (keep >= static_cast<int64_t>(scores.size())) {
        return;
    }
    std::vector<int32_t> order(scores.size());
    for (size_t index = 0; index < scores.size(); ++index) {
        order[index] = static_cast<int32_t>(index);
    }
    auto kth = order.begin() + static_cast<std::ptrdiff_t>(keep - 1);
    std::nth_element(order.begin(), kth, order.end(), [&](int32_t lhs, int32_t rhs) {
        return scores[static_cast<size_t>(lhs)] > scores[static_cast<size_t>(rhs)];
    });
    const float threshold = scores[static_cast<size_t>(*kth)];
    for (float & score : scores) {
        if (score < threshold) {
            score = -std::numeric_limits<float>::infinity();
        }
    }
}

void reference_top_p(std::vector<float> & scores, float top_p, int64_t min_tokens_to_keep) {
    if (!(top_p < 1.0F) || scores.empty()) {
        return;
    }
    std::vector<int32_t> sorted;
    sorted.reserve(scores.size());
    for (size_t index = 0; index < scores.size(); ++index) {
        if (std::isfinite(scores[index])) {
            sorted.push_back(static_cast<int32_t>(index));
        }
    }
    if (sorted.empty()) {
        return;
    }
    std::sort(sorted.begin(), sorted.end(), [&](int32_t lhs, int32_t rhs) {
        const float lhs_score = scores[static_cast<size_t>(lhs)];
        const float rhs_score = scores[static_cast<size_t>(rhs)];
        if (lhs_score == rhs_score) {
            return lhs < rhs;
        }
        return lhs_score < rhs_score;
    });
    const float max_score = scores[static_cast<size_t>(sorted.back())];
    std::vector<double> weights(sorted.size());
    double total = 0.0;
    for (size_t index = 0; index < sorted.size(); ++index) {
        weights[index] = std::exp(static_cast<double>(scores[static_cast<size_t>(sorted[index])] - max_score));
        total += weights[index];
    }
    double cumulative = 0.0;
    const double remove_mass = 1.0 - static_cast<double>(top_p);
    const size_t min_keep = std::min<size_t>(static_cast<size_t>(min_tokens_to_keep), sorted.size());
    const size_t protected_from = sorted.size() - min_keep;
    for (size_t index = 0; index < sorted.size(); ++index) {
        cumulative += weights[index] / total;
        if (index < protected_from && cumulative <= remove_mass) {
            scores[static_cast<size_t>(sorted[index])] = -std::numeric_limits<float>::infinity();
        }
    }
}

int32_t reference_sample(
    const std::vector<float> & logits,
    const std::vector<int32_t> & history,
    const HfSamplingOptions & options,
    std::mt19937 & fallback_rng,
    const HfTorchSamplingState * torch_state) {
    std::vector<float> scores = logits;
    reference_repetition_penalty(scores, history, options.repetition_penalty);
    if (!options.do_sample) {
        return reference_argmax(scores);
    }
    if (options.temperature != 1.0F) {
        for (float & score : scores) {
            score /= options.temperature;
        }
    }
    reference_top_k(scores, options.top_k, options.min_tokens_to_keep);
    reference_top_p(scores, options.top_p, options.min_tokens_to_keep);

    std::vector<int32_t> candidates;
    std::vector<double> weights;
    candidates.reserve(scores.size());
    float max_score = -std::numeric_limits<float>::infinity();
    for (size_t index = 0; index < scores.size(); ++index) {
        if (std::isfinite(scores[index])) {
            candidates.push_back(static_cast<int32_t>(index));
            max_score = std::max(max_score, scores[index]);
        }
    }
    weights.reserve(candidates.size());
    for (const int32_t token : candidates) {
        weights.push_back(std::exp(static_cast<double>(scores[static_cast<size_t>(token)] - max_score)));
    }
    if (torch_state != nullptr && torch_state->policy != nullptr && torch_state->policy->cuda_fast_path) {
        double best_rank = -std::numeric_limits<double>::infinity();
        int32_t best_token = -1;
        for (size_t index = 0; index < candidates.size(); ++index) {
            const int32_t token = candidates[index];
            const float exponential = engine::sampling::torch_cuda_tensor_iterator_exponential_element(
                torch_state->seed,
                static_cast<uint64_t>(scores.size()),
                static_cast<uint64_t>(token),
                torch_state->call_index,
                torch_state->policy->multiprocessor_count,
                torch_state->policy->max_threads_per_multiprocessor);
            const double rank = weights[index] / static_cast<double>(exponential);
            if (rank > best_rank) {
                best_rank = rank;
                best_token = token;
            }
        }
        return best_token;
    }
    std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
    return candidates[distribution(fallback_rng)];
}

template <typename Fn>
double elapsed_ms(Fn && fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void test_matches_reference_fallback_sampler() {
    HfSampler sampler;
    HfSamplerScratch scratch;
    HfSamplingOptions options;
    options.do_sample = true;
    options.temperature = 0.82F;
    options.top_k = 64;
    options.top_p = 0.86F;
    options.min_tokens_to_keep = 2;
    options.repetition_penalty = 1.15F;
    for (int step = 0; step < 24; ++step) {
        const auto logits = make_logits(4096, step);
        const auto history = make_history(logits.size(), step);
        std::mt19937 reference_rng(1234 + step);
        std::mt19937 optimized_rng(1234 + step);
        const int32_t expected = reference_sample(logits, history, options, reference_rng, nullptr);
        const int32_t actual = sampler.sample(logits, history, options, scratch, optimized_rng, nullptr, "fallback");
        engine::test::require_eq(actual, expected, "fallback sampled token");
    }
}

void test_matches_reference_torch_sampler() {
    TorchCudaSamplingPolicy policy;
    policy.cuda_fast_path = true;
    policy.multiprocessor_count = 128;
    policy.max_threads_per_multiprocessor = 1536;
    HfTorchSamplingState torch_state;
    torch_state.policy = &policy;
    torch_state.seed = 5678;

    HfSampler sampler;
    HfSamplerScratch scratch;
    HfSamplingOptions options;
    options.do_sample = true;
    options.temperature = 0.75F;
    options.top_k = 0;
    options.top_p = 0.9F;
    options.min_tokens_to_keep = 1;
    options.repetition_penalty = 1.05F;
    for (int step = 0; step < 24; ++step) {
        torch_state.call_index = static_cast<uint64_t>(step);
        const auto logits = make_logits(4096, step);
        const auto history = make_history(logits.size(), step);
        std::mt19937 reference_rng(1);
        std::mt19937 optimized_rng(1);
        const int32_t expected = reference_sample(logits, history, options, reference_rng, &torch_state);
        const int32_t actual = sampler.sample(logits, history, options, scratch, optimized_rng, &torch_state, "torch");
        engine::test::require_eq(actual, expected, "torch sampled token");
    }
}

void test_matches_python_hf_processor_reference_values() {
    std::vector<float> scores{0.2F, -0.3F, 1.4F, 0.8F, 0.1F, 1.2F, -0.6F};
    HfSamplerScratch scratch;
    engine::sampling::HfLogitsProcessor::apply_repetition_penalty(
        scores,
        std::vector<int32_t>{2, 2, 4},
        1.2F,
        scratch);
    engine::sampling::HfLogitsProcessor::apply_temperature(scores, 0.5F);
    engine::sampling::HfLogitsProcessor::apply_top_k(scores, 3, 1, scratch);
    engine::sampling::HfLogitsProcessor::apply_top_p(scores, 0.75F, 1, scratch);

    const std::vector<float> expected{
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        2.3333332538604736F,
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        2.4000000953674316F,
        -std::numeric_limits<float>::infinity(),
    };
    for (size_t index = 0; index < scores.size(); ++index) {
        if (std::isfinite(expected[index])) {
            engine::test::require_close(scores[index], expected[index], 1.0e-6F, "python HF processor score");
        } else {
            engine::test::require(std::isinf(scores[index]) && scores[index] < 0.0F, "python HF processor mask");
        }
    }
}

void test_matches_python_hf_cuda_multinomial_reference_sequence() {
    TorchCudaSamplingPolicy policy;
    policy.cuda_fast_path = true;
    policy.multiprocessor_count = 128;
    policy.max_threads_per_multiprocessor = 1536;
    HfTorchSamplingState torch_state;
    torch_state.policy = &policy;
    torch_state.seed = 1;
    torch_state.call_index = 0;

    HfSampler sampler;
    HfSamplerScratch scratch;
    HfSamplingOptions options;
    options.do_sample = true;
    options.temperature = 0.82F;
    options.top_k = 64;
    options.top_p = 0.86F;
    options.min_tokens_to_keep = 2;
    options.repetition_penalty = 1.15F;
    std::mt19937 fallback_rng(1);
    const std::vector<int32_t> expected{2537, 2526, 2513, 3960, 3934, 2507, 3946, 3933, 3921, 2478};
    for (size_t step = 0; step < expected.size(); ++step) {
        torch_state.call_index = static_cast<uint64_t>(step);
        const int generator_step = static_cast<int>(3 + step);
        const int32_t actual = sampler.sample(
            make_logits(4096, generator_step),
            make_history(4096, generator_step),
            options,
            scratch,
            fallback_rng,
            &torch_state,
            "python HF cuda");
        engine::test::require_eq(actual, expected[step], "python HF CUDA multinomial token sequence");
    }
}

void test_matches_python_hf_cpu_multinomial_reference_sequence() {
    HfTorchSamplingState torch_state;
    torch_state.seed = 1;

    HfSampler sampler;
    HfSamplerScratch scratch;
    HfSamplingOptions options;
    options.do_sample = true;
    options.temperature = 0.82F;
    options.top_k = 64;
    options.top_p = 0.86F;
    options.min_tokens_to_keep = 2;
    options.repetition_penalty = 1.15F;
    std::mt19937 fallback_rng(1);
    const std::vector<int32_t> expected{3974, 2514, 2514, 3946, 2491, 2490, 3922, 2477, 2487, 2479};
    for (size_t step = 0; step < expected.size(); ++step) {
        torch_state.call_index = static_cast<uint64_t>(step);
        const int generator_step = static_cast<int>(3 + step);
        const int32_t actual = sampler.sample(
            make_logits(4096, generator_step),
            make_history(4096, generator_step),
            options,
            scratch,
            fallback_rng,
            &torch_state,
            "python HF cpu");
        engine::test::require_eq(actual, expected[step], "python HF CPU multinomial token sequence");
    }
}

void test_greedy_fast_path_matches_reference() {
    HfSampler sampler;
    HfSamplerScratch scratch;
    HfSamplingOptions options;
    options.do_sample = false;
    options.repetition_penalty = 1.0F;
    std::mt19937 fallback_rng(1234);
    const auto logits = make_logits(8192, 7);
    const std::vector<int32_t> history;
    const int32_t expected = reference_sample(logits, history, options, fallback_rng, nullptr);
    const int32_t actual = sampler.sample(logits, history, options, scratch, fallback_rng, nullptr, "greedy");
    engine::test::require_eq(actual, expected, "greedy token");
}

void test_no_processor_sampling_fast_path_matches_reference() {
    HfSampler sampler;
    HfSamplerScratch scratch;
    HfSamplingOptions options;
    options.do_sample = true;
    options.temperature = 1.0F;
    options.top_k = 0;
    options.top_p = 1.0F;
    options.repetition_penalty = 1.0F;
    const auto logits = make_logits(8192, 9);
    const std::vector<int32_t> history;
    std::mt19937 reference_rng(4321);
    std::mt19937 optimized_rng(4321);
    const int32_t expected = reference_sample(logits, history, options, reference_rng, nullptr);
    const int32_t actual = sampler.sample(logits, history, options, scratch, optimized_rng, nullptr, "plain sample");
    engine::test::require_eq(actual, expected, "plain sampled token");
}

void test_no_processor_torch_fast_path_ignores_previous_candidates() {
    TorchCudaSamplingPolicy policy;
    policy.cuda_fast_path = true;
    policy.multiprocessor_count = 128;
    policy.max_threads_per_multiprocessor = 1536;
    HfTorchSamplingState torch_state;
    torch_state.policy = &policy;
    torch_state.seed = 77;

    HfSampler sampler;
    HfSamplerScratch scratch;
    HfSamplingOptions filtered_options;
    filtered_options.do_sample = true;
    filtered_options.temperature = 0.8F;
    filtered_options.top_k = 8;
    filtered_options.top_p = 0.75F;
    filtered_options.repetition_penalty = 1.1F;
    const auto first_logits = make_logits(2048, 3);
    const auto first_history = make_history(first_logits.size(), 3);
    std::mt19937 filtered_rng(1);
    torch_state.call_index = 0;
    (void)sampler.sample(first_logits, first_history, filtered_options, scratch, filtered_rng, &torch_state, "filtered");

    HfSamplingOptions plain_options;
    plain_options.do_sample = true;
    plain_options.temperature = 1.0F;
    plain_options.top_k = 0;
    plain_options.top_p = 1.0F;
    plain_options.repetition_penalty = 1.0F;
    const auto plain_logits = make_logits(2048, 4);
    std::mt19937 reference_rng(2);
    std::mt19937 optimized_rng(2);
    torch_state.call_index = 1;
    const int32_t expected = reference_sample(plain_logits, {}, plain_options, reference_rng, &torch_state);
    const int32_t actual = sampler.sample(plain_logits, {}, plain_options, scratch, optimized_rng, &torch_state, "plain torch");
    engine::test::require_eq(actual, expected, "plain torch stale candidates");
}

void test_direct_torch_sampler_ignores_previous_score_buffer() {
    TorchCudaSamplingPolicy policy;
    policy.cuda_fast_path = true;
    policy.multiprocessor_count = 128;
    policy.max_threads_per_multiprocessor = 1536;
    HfTorchSamplingState torch_state;
    torch_state.policy = &policy;
    torch_state.seed = 88;

    HfSamplerScratch scratch;
    std::mt19937 fallback_rng(1);
    const auto first_scores = make_logits(1024, 1);
    torch_state.call_index = 0;
    (void)engine::sampling::HfTokenSampler::sample_from_processed_scores(
        first_scores,
        scratch,
        fallback_rng,
        &torch_state,
        "first");

    const auto second_scores = make_logits(1024, 2);
    std::mt19937 reference_rng(2);
    std::mt19937 optimized_rng(2);
    HfSamplingOptions options;
    options.do_sample = true;
    torch_state.call_index = 1;
    const int32_t expected = reference_sample(second_scores, {}, options, reference_rng, &torch_state);
    const int32_t actual = engine::sampling::HfTokenSampler::sample_from_processed_scores(
        second_scores,
        scratch,
        optimized_rng,
        &torch_state,
        "second");
    engine::test::require_eq(actual, expected, "direct torch stale score buffer");
}

void benchmark_sampler_path() {
    constexpr int steps = 80;
    constexpr size_t vocab_size = 32768;
    const std::vector<float> logits = make_logits(vocab_size, 0);
    const std::vector<int32_t> history = make_history(vocab_size, 0);

    TorchCudaSamplingPolicy policy;
    policy.cuda_fast_path = true;
    policy.multiprocessor_count = 128;
    policy.max_threads_per_multiprocessor = 1536;
    HfTorchSamplingState torch_state;
    torch_state.policy = &policy;
    torch_state.seed = 4321;

    HfSamplingOptions options;
    options.do_sample = true;
    options.temperature = 0.8F;
    options.top_k = 0;
    options.top_p = 0.9F;
    options.min_tokens_to_keep = 1;
    options.repetition_penalty = 1.1F;

    std::mt19937 reference_rng(1);
    const double reference_ms = elapsed_ms([&] {
        for (int step = 0; step < steps; ++step) {
            torch_state.call_index = static_cast<uint64_t>(step);
            (void)reference_sample(logits, history, options, reference_rng, &torch_state);
        }
    });

    HfSampler sampler;
    HfSamplerScratch scratch;
    std::mt19937 optimized_rng(1);
    const double optimized_ms = elapsed_ms([&] {
        for (int step = 0; step < steps; ++step) {
            torch_state.call_index = static_cast<uint64_t>(step);
            (void)sampler.sample(logits, history, options, scratch, optimized_rng, &torch_state, "bench");
        }
    });

    std::cout << "hf_sampler_reference_ms " << reference_ms << '\n';
    std::cout << "hf_sampler_optimized_ms " << optimized_ms << '\n';
    std::cout << "hf_sampler_speedup " << (reference_ms / optimized_ms) << '\n';
}

}  // namespace

int main() {
    try {
        test_greedy_fast_path_matches_reference();
        test_matches_python_hf_processor_reference_values();
        test_matches_python_hf_cuda_multinomial_reference_sequence();
        test_matches_python_hf_cpu_multinomial_reference_sequence();
        test_no_processor_sampling_fast_path_matches_reference();
        test_no_processor_torch_fast_path_ignores_previous_candidates();
        test_direct_torch_sampler_ignores_previous_score_buffer();
        test_matches_reference_fallback_sampler();
        test_matches_reference_torch_sampler();
        benchmark_sampler_path();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "hf_sampler_test passed\n";
    return 0;
}
