#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/flashsr.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef ENGINE_REPO_ROOT
#define ENGINE_REPO_ROOT "."
#endif

#ifndef ENGINE_TEST_ASSET_ROOT
#define ENGINE_TEST_ASSET_ROOT "tests/assets"
#endif

namespace {

std::filesystem::path repo_path(const std::string & relative) {
    return std::filesystem::path(ENGINE_REPO_ROOT) / relative;
}

std::filesystem::path asset_path(const std::string & relative) {
    return std::filesystem::path(ENGINE_TEST_ASSET_ROOT) / relative;
}

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    const std::vector<float> & actual,
    const engine::assets::TensorDataF32 & expected,
    float max_allowed,
    double mean_allowed,
    const std::string & label) {
    require(expected.shape.rank == 2 && expected.shape.dims[0] == 1, label + " expected shape mismatch");
    require(actual.size() == expected.values.size(), label + " size mismatch");
    float max_diff = 0.0f;
    size_t max_index = 0;
    double mean_diff = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected.values[i]);
        mean_diff += static_cast<double>(diff);
        if (diff > max_diff) {
            max_diff = diff;
            max_index = i;
        }
    }
    mean_diff /= static_cast<double>(actual.size());
    if (max_diff > max_allowed || mean_diff > mean_allowed) {
        std::ostringstream oss;
        oss << label << " mismatch: max_diff=" << max_diff
            << " mean_diff=" << mean_diff
            << " index=" << max_index
            << " expected=" << expected.values[max_index]
            << " actual=" << actual[max_index];
        throw std::runtime_error(oss.str());
    }
}

void run_case(int case_index) {
    const auto model = engine::audio::FlashSrModel::load_from_directory(
        repo_path("assets/framework/audio_utilities/flashsr"));
    const auto fixture = engine::assets::open_tensor_source(
        asset_path("framework/audio_utilities/flashsr/flashsr_case" + std::to_string(case_index) + ".safetensors"));
    const auto input = fixture->require_f32_tensor("audio_values");
    const auto expected = fixture->require_f32_tensor("reconstruction");
    require(input.shape.rank == 2 && input.shape.dims[0] == 1, "FlashSR input shape mismatch");
    // The checked-in fixtures were captured from the upstream reference, which
    // peak-normalises every file to the ceiling. Ask for that behaviour
    // explicitly so this stays a parity test; input-level preservation is
    // covered by enhancement_alignment_test.
    engine::audio::FlashSrOptions reference_parity;
    reference_parity.preserve_input_level = false;
    const auto output = model.super_resolve_mono_16k(input.values, reference_parity);
    require(output.sample_rate == 48000, "FlashSR sample rate mismatch");
    require_close(output.samples, expected, 2.0e-4f, 2.0e-5, "case " + std::to_string(case_index));
}

}  // namespace

int main() {
    try {
        run_case(0);
        run_case(1);
        std::cout << "flashsr_utility_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "flashsr_utility_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
