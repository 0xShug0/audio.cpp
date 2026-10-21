#include "engine/models/higgs_audio_tts/sampler.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void test_top_k_ignores_non_finite_logits() {
    using engine::models::higgs_audio_tts::HiggsCodebookSampler;
    using engine::models::higgs_audio_tts::HiggsSamplingOptions;

    HiggsCodebookSampler sampler(1, 4);
    auto state = sampler.make_state();
    HiggsSamplingOptions options;
    options.temperature = 1.0F;
    options.top_k = 3;
    options.has_seed = true;
    options.seed = 123;

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> logits = {nan, 0.0F, 1.0F, 2.0F};
    const auto & codes = sampler.step(logits.data(), logits.size(), state, options);

    if (codes.size() != 1 || codes.front() < 1 || codes.front() > 3) {
        throw std::runtime_error("Higgs top-k sampling selected a non-finite logit");
    }
}

void test_top_k_reports_when_all_logits_are_non_finite() {
    using engine::models::higgs_audio_tts::HiggsCodebookSampler;
    using engine::models::higgs_audio_tts::HiggsSamplingOptions;

    HiggsCodebookSampler sampler(1, 3);
    auto state = sampler.make_state();
    HiggsSamplingOptions options;
    options.temperature = 1.0F;
    options.top_k = 2;
    options.has_seed = true;
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
    throw std::runtime_error("Higgs sampler accepted logits without a finite value");
}

} // namespace

int main() {
    test_top_k_ignores_non_finite_logits();
    test_top_k_reports_when_all_logits_are_non_finite();
    std::cout << "higgs_audio_tts_sampler_test passed\n";
    return 0;
}
