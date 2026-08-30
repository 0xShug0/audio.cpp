#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/rnnoise.h"

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

engine::core::BackendConfig cuda_backend() {
    engine::core::BackendConfig backend;
#ifdef GGML_USE_CUDA
    backend.type = engine::core::BackendType::Cuda;
#elif defined(GGML_USE_VULKAN)
    backend.type = engine::core::BackendType::Vulkan;
#else
    backend.type = engine::core::BackendType::Cpu;
#endif
    backend.device = 0;
    backend.threads = 1;
    return backend;
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
    const std::vector<float> & expected,
    float max_allowed,
    double mean_allowed,
    const std::string & label) {
    require(actual.size() == expected.size(), label + " size mismatch");
    float max_diff = 0.0f;
    size_t max_index = 0;
    double mean_diff = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float diff = std::fabs(actual[i] - expected[i]);
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
            << " expected=" << expected[max_index]
            << " actual=" << actual[max_index];
        throw std::runtime_error(oss.str());
    }
}

void run_case(const engine::audio::RnnoiseModel & model, const std::string & checkpoint, int case_index) {
    const auto fixture = engine::assets::open_tensor_source(
        asset_path("framework/audio_utilities/rnnoise/" + checkpoint + "_case" + std::to_string(case_index) + ".safetensors"));
    const auto features = fixture->require_f32_tensor("features");
    const auto expected_gains = fixture->require_f32_tensor("gains");
    const auto expected_vad = fixture->require_f32_tensor("vad");
    require(features.shape.rank == 2, "features fixture rank mismatch");
    require(expected_gains.shape.rank == 2, "gains fixture rank mismatch");
    require(expected_vad.shape.rank == 2, "vad fixture rank mismatch");
    const int64_t frames = features.shape.dims[0];
    const int64_t feature_size = features.shape.dims[1];
    const auto actual = model.infer_features(features.values, frames, feature_size);
    require(actual.frames == frames, "RNNoise frame count mismatch");
    require(actual.gain_bands == expected_gains.shape.dims[1], "RNNoise gain band count mismatch");
    require_close(actual.gains, expected_gains.values, 2.5e-5f, 2.0e-6, checkpoint + " gains case " + std::to_string(case_index));
    require_close(actual.vad, expected_vad.values, 2.5e-5f, 2.0e-6, checkpoint + " vad case " + std::to_string(case_index));
}

void run_waveform_case(const engine::audio::RnnoiseModel & model, const std::string & checkpoint, int case_index) {
    const auto fixture = engine::assets::open_tensor_source(
        asset_path("framework/audio_utilities/rnnoise/" + checkpoint + "_waveform_case" + std::to_string(case_index) + ".safetensors"));
    const auto input = fixture->require_f32_tensor("waveform_input");
    const auto expected_output = fixture->require_f32_tensor("waveform_output");
    const auto expected_vad = fixture->require_f32_tensor("waveform_vad");
    require(input.shape.rank == 2 && input.shape.dims[0] == 1, "RNNoise waveform input shape mismatch");
    require(expected_output.shape.rank == 2 && expected_output.shape.dims[0] == 1, "RNNoise waveform output shape mismatch");
    require(expected_vad.shape.rank == 2 && expected_vad.shape.dims[0] == 1, "RNNoise waveform vad shape mismatch");
    // The checked-in fixtures were captured from the upstream reference, which
    // leaves the 20 ms synthesis delay in the output. Ask for that behaviour
    // explicitly so this stays a parity test; delay compensation is covered by
    // enhancement_alignment_test.
    engine::audio::RnnoiseProcessOptions reference_parity;
    reference_parity.compensate_output_delay = false;
    const auto actual = model.process_mono_48k(input.values, reference_parity);
    require(actual.sample_rate == 48000, "RNNoise sample rate mismatch");
    require_close(
        actual.samples,
        expected_output.values,
        2.0e-3f,
        2.0e-4,
        checkpoint + " waveform case " + std::to_string(case_index));
    require_close(
        actual.vad,
        expected_vad.values,
        2.5e-5f,
        2.0e-6,
        checkpoint + " waveform vad case " + std::to_string(case_index));
}

}  // namespace

int main() {
    try {
        for (const std::string checkpoint : {"rnnoise10Ga_12", "rnnoise10Gb_15"}) {
            const auto model = engine::audio::RnnoiseModel::load_from_safetensors(
                repo_path("assets/framework/audio_utilities/rnnoise/" + checkpoint + ".safetensors"),
                cuda_backend());
            run_case(model, checkpoint, 0);
            run_case(model, checkpoint, 1);
            run_waveform_case(model, checkpoint, 0);
            run_waveform_case(model, checkpoint, 1);
        }
        std::cout << "rnnoise_utility_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "rnnoise_utility_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
