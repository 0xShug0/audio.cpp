#include "engine/community_models/mms_forced_aligner/emissions.h"
#include "test_assert.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::test::require;
using engine::test::require_close;
using engine::test::require_eq;

void test_normalize_zero_mean_unit_variance() {
    std::vector<float> waveform;
    for (int i = 0; i < 4000; ++i) {
        waveform.push_back(static_cast<float>(i % 97) / 31.0F);
    }
    const auto out = engine::community_models::mms_forced_aligner::mms_normalize_waveform_16k(waveform);
    double sum = 0.0;
    double sum_sq = 0.0;
    for (const float sample : out) {
        sum += sample;
        sum_sq += static_cast<double>(sample) * static_cast<double>(sample);
    }
    const double mean = sum / static_cast<double>(out.size());
    const double variance = sum_sq / static_cast<double>(out.size()) - mean * mean;
    require_close(static_cast<float>(mean), 0.0F, 1.0e-5F, "normalized mean");
    require_close(static_cast<float>(variance), 1.0F, 1.0e-3F, "normalized variance");
}

void test_normalize_constant_waveform_finite() {
    const std::vector<float> waveform(1000, 0.25F);
    const auto out = engine::community_models::mms_forced_aligner::mms_normalize_waveform_16k(waveform);
    require(out.size() == waveform.size(), "constant waveform length");
    for (const float sample : out) {
        require(std::isfinite(sample), "constant waveform must stay finite");
        require_close(sample, 0.0F, 1.0e-6F, "constant waveform normalizes to zero");
    }
}

void test_normalize_rejects_empty() {
    bool threw = false;
    try {
        (void) engine::community_models::mms_forced_aligner::mms_normalize_waveform_16k({});
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "empty waveform must be rejected");
}

void test_log_softmax_rows_sum_to_one() {
    std::vector<float> logits(3 * 5);
    for (size_t i = 0; i < logits.size(); ++i) {
        logits[i] = static_cast<float>(i % 7) - 3.0F;
    }
    const auto out = engine::community_models::mms_forced_aligner::mms_log_softmax_and_star(logits.data(), 3, 5);
    require_eq(static_cast<int64_t>(out.size()), int64_t{18}, "output width frames*(classes+1)");
    for (int64_t frame = 0; frame < 3; ++frame) {
        const float * row = out.data() + frame * 6;
        double sum = 0.0;
        for (int64_t cls = 0; cls < 5; ++cls) {
            sum += std::exp(row[cls]);
        }
        require_close(static_cast<float>(sum), 1.0F, 1.0e-5F, "row sums to one over real classes");
        require_close(row[5], 0.0F, 0.0F, "star column is exactly zero");
    }
}

void test_log_softmax_extreme_values() {
    std::vector<float> logits = {1000.0F, -1000.0F, 0.0F};
    const auto out = engine::community_models::mms_forced_aligner::mms_log_softmax_and_star(logits.data(), 1, 3);
    require_close(out[0], 0.0F, 1.0e-5F, "max class log-prob");
    require_close(out[1], -2000.0F, 0.5F, "min class log-prob");
    require(out[3] == 0.0F, "star column");
}

}  // namespace

int main() {
    try {
        test_normalize_zero_mean_unit_variance();
        test_normalize_constant_waveform_finite();
        test_normalize_rejects_empty();
        test_log_softmax_rows_sum_to_one();
        test_log_softmax_extreme_values();
        std::cout << "mms_emissions_test passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "mms_emissions_test: %s\n", error.what());
        return 1;
    }
}
