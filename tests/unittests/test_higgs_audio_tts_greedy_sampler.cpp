#include "engine/models/higgs_audio_tts/sampler.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void test_greedy_sampling_ignores_non_finite_logits() {
    using engine::models::higgs_audio_tts::HiggsCodebookSampler;
    using engine::models::higgs_audio_tts::HiggsSamplingOptions;

    HiggsCodebookSampler sampler(1, 4);
    auto state = sampler.make_state();
    HiggsSamplingOptions options;
    options.temperature = 1.0F;
    options.top_k = 1;

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> logits = {nan, -inf, 1.0F, 2.0F};
    const auto & codes = sampler.step(logits.data(), logits.size(), state, options);
    if (codes.size() != 1 || codes.front() != 3) {
        throw std::runtime_error("Higgs greedy sampler did not select the highest finite logit");
    }
}

void test_greedy_sampling_rejects_all_non_finite_logits() {
    using engine::models::higgs_audio_tts::HiggsCodebookSampler;
    using engine::models::higgs_audio_tts::HiggsSamplingOptions;

    HiggsCodebookSampler sampler(1, 3);
    auto state = sampler.make_state();
    HiggsSamplingOptions options;
    options.temperature = 1.0F;
    options.top_k = 1;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> logits = {nan, inf, -inf};

    try {
        (void)sampler.step(logits.data(), logits.size(), state, options);
    } catch (const std::runtime_error & error) {
        if (std::string(error.what()).find("sampler kept no finite logits") != std::string::npos) {
            return;
        }
        throw;
    }
    throw std::runtime_error("Higgs greedy sampler accepted logits without a finite value");
}

} // namespace

int main() {
    test_greedy_sampling_ignores_non_finite_logits();
    test_greedy_sampling_rejects_all_non_finite_logits();
    std::cout << "higgs_audio_tts_greedy_sampler_test passed\n";
    return 0;
}
