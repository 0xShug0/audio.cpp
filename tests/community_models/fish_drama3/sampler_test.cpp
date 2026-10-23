#include "../../../src/community_models/fish_drama3/generator.cpp"

#include <ggml-alloc.h>
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

void test_device_top_p_boundary(engine::core::BackendType backend_type) {
    engine::core::ExecutionContext execution({backend_type, 0, 1});
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    if (!context) throw std::runtime_error("sampler test context allocation failed");
    auto * ctx = context.get();
    auto * probs = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    auto * top_p = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    auto * top_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    auto * uniform = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(probs);
    ggml_set_input(top_p);
    ggml_set_input(top_k);
    ggml_set_input(uniform);
    auto * cumulative = ggml_cumsum(ctx, probs);
    auto * ranks = ggml_arange(ctx, 0, 4, 1);
    auto * first = ggml_clamp(ctx, ggml_scale_bias(ctx, ranks, -1, 1), 0, 1);
    auto * keep = ggml_scale_bias(ctx, ggml_clamp(ctx, ggml_sgn(ctx, ggml_sub(ctx, cumulative, top_p)), 0, 1), -1, 1);
    keep = ggml_clamp(ctx, ggml_add(ctx, keep, first), 0, 1);
    keep = ggml_mul(ctx, keep, ggml_clamp(ctx, ggml_add(ctx, ggml_scale(ctx, ranks, -1), top_k), 0, 1));
    auto * cdf = ggml_cumsum(ctx, ggml_mul(ctx, probs, keep));
    auto * total = ggml_view_1d(ctx, cdf, 1, 3 * sizeof(float));
    auto * threshold = ggml_mul(ctx, total, uniform);
    auto * selected = ggml_cast(
        ctx, ggml_sum(ctx, ggml_clamp(ctx, ggml_sgn(ctx, ggml_add(ctx, ggml_scale(ctx, cdf, -1), threshold)), 0, 1)),
        GGML_TYPE_I32);
    ggml_set_output(keep);
    ggml_set_output(selected);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, keep);
    ggml_build_forward_expand(graph, selected);
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())), ggml_gallocr_free);
    if (!ggml_gallocr_alloc_graph(allocator.get(), graph)) {
        throw std::runtime_error("sampler test graph allocation failed");
    }
    const std::array<float, 4> probabilities{0.25F, 0.25F, 0.25F, 0.25F};
    for (float cutoff : {0.125F, std::nextafter(0.5F, 0.0F), 0.5F, std::nextafter(0.5F, 1.0F), 1.0F}) {
        for (float k : {1.0F, 2.0F, 4.0F}) {
            for (float draw : {0.0F, 0.5F, 0.75F}) {
                ggml_backend_tensor_set(probs, probabilities.data(), 0, sizeof(probabilities));
                ggml_backend_tensor_set(top_p, &cutoff, 0, sizeof(cutoff));
                ggml_backend_tensor_set(top_k, &k, 0, sizeof(k));
                ggml_backend_tensor_set(uniform, &draw, 0, sizeof(draw));
                if (ggml_backend_graph_compute(execution.backend(), graph) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("sampler boundary graph failed");
                }
                std::array<float, 4> actual{};
                ggml_backend_tensor_get(keep, actual.data(), 0, sizeof(actual));
                std::array<float, 4> expected_cdf{};
                float mass = 0;
                for (size_t rank = 0; rank < probabilities.size(); ++rank) {
                    const float prefix = 0.25F * static_cast<float>(rank + 1);
                    const float expected = (rank == 0 || prefix <= cutoff) && rank < k ? 1.0F : 0.0F;
                    if (actual[rank] != expected) {
                        throw std::runtime_error("device top-p mask disagrees at rank " + std::to_string(rank) +
                                                 ", top_p=" + std::to_string(cutoff) +
                                                 ", mask=" + std::to_string(actual[rank]));
                    }
                    mass += expected * 0.25F;
                    expected_cdf[rank] = mass;
                }
                int32_t choice = -1;
                ggml_backend_tensor_get(selected, &choice, 0, sizeof(choice));
                const auto expected_choice =
                    std::lower_bound(expected_cdf.begin(), expected_cdf.end(), mass * draw) - expected_cdf.begin();
                if (choice != expected_choice)
                    throw std::runtime_error("device sampler CDF boundary disagrees with host");
            }
        }
    }
    engine::core::release_backend_graph_resources(execution.backend(), graph, true);
}
}

int main(int argc, char ** argv) {
    try {
        engine::debug::configure_logging({true, std::nullopt});
        const auto started = std::chrono::steady_clock::now();
        const auto backend = argc > 2 && std::string(argv[1]) == "--backend"
            ? std::string(argv[2]) : std::string("cpu");
        if (backend != "cpu" && backend != "cuda" && backend != "vulkan") {
            throw std::runtime_error("sampler test backend must be cpu, cuda, or vulkan");
        }
        test_device_top_p_boundary(backend == "cuda" ? engine::core::BackendType::Cuda
            : backend == "vulkan" ? engine::core::BackendType::Vulkan : engine::core::BackendType::Cpu);
        std::cout << "device_top_p_boundary_matches=true backend=" << backend << '\n';
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
