#include "../../../src/community_models/fish_drama3/generator.cpp"

#include <iostream>

namespace {
int32_t original_sample(const std::vector<float> & logits, float temperature, int top_k,
                        float top_p, std::mt19937 & random) {
    std::vector<int32_t> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    const auto count = std::min(logits.size(), static_cast<size_t>(top_k));
    std::partial_sort(order.begin(), order.begin() + count, order.end(), [&](int32_t a, int32_t b) {
        return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
    });
    order.resize(count);
    if (temperature == 0 || top_k == 1) return order.front();
    const double maximum = logits[order.front()] / temperature;
    std::vector<double> probabilities(order.size());
    double total = 0;
    for (float logit : logits) total += std::exp(logit / temperature - maximum);
    for (size_t rank = 0; rank < order.size(); ++rank)
        probabilities[rank] = std::exp(logits[order[rank]] / temperature - maximum);
    double cumulative = 0;
    for (size_t rank = 0; rank < order.size(); ++rank) {
        cumulative += probabilities[rank] / total;
        if (rank > 0 && (cumulative > top_p || rank >= static_cast<size_t>(top_k))) probabilities[rank] = 0;
    }
    std::discrete_distribution<size_t> choice(probabilities.begin(), probabilities.end());
    return order[choice(random)];
}
}

int main() {
    try {
        engine::debug::configure_logging({true, std::nullopt});
        const auto started = std::chrono::steady_clock::now();
        std::mt19937 source(1729);
        std::normal_distribution<float> normal(0, 3);
        size_t draws = 0;
        for (size_t size : {size_t(1), size_t(17), size_t(65), size_t(257), size_t(4096), size_t(4097)}) {
            for (bool tied : {false, true}) {
                std::vector<float> logits(size);
                for (auto & logit : logits) logit = tied ? float(source() % 7) : normal(source);
                for (const float temperature : {0.0F, 0.01F, 0.7F, 1.0F, 1.5F}) {
                    for (const int top_k : {1, 30, 64, 128, int(size + 4)}) {
                        for (const float top_p : {0.01F, 0.9F, 0.95F, 1.0F}) {
                            std::mt19937 expected(1234), actual(1234);
                            engine::models::fish_drama3::SamplerScratch scratch;
                            for (int repeat = 0; repeat < 32; ++repeat) {
                                const auto a = original_sample(logits, temperature, top_k, top_p, expected);
                                const auto b = engine::models::fish_drama3::sample(logits, temperature, top_k, top_p, actual, scratch);
                                if (a != b || expected != actual) throw std::runtime_error("Drama 3 sampler changed token or RNG state");
                                ++draws;
                            }
                        }
                    }
                }
            }
        }
        std::cout << "draws=" << draws << " token_and_rng_matches=true\n";
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
