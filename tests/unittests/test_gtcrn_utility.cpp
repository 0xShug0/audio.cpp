#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/gtcrn.h"

#include <algorithm>
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

engine::core::BackendConfig test_backend() {
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

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    const std::vector<float> & actual,
    const engine::assets::TensorDataF32 & expected,
    const size_t expected_size,
    float max_allowed,
    double mean_allowed,
    const std::string & label) {
    require(expected.values.size() == expected_size, label + " expected shape mismatch");
    require(actual.size() == expected.values.size(), label + " size mismatch");
    float max_diff = 0.0F;
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

void run_case(const std::string & model_name, int case_index) {
    const auto model = engine::audio::GTCRNModel::load_from_safetensors(
        repo_path("assets/framework/audio_utilities/gtcrn/" + model_name + ".safetensors"),
        test_backend());
    const auto fixture = engine::assets::open_tensor_source(
        asset_path("framework/audio_utilities/gtcrn/" + model_name + "_case" + std::to_string(case_index) + ".safetensors"));
    const auto input = fixture->require_f32_tensor("input");
    const auto expected = fixture->require_f32_tensor("output");
    const auto spec_frame0 = fixture->require_f32_tensor("spec_frame0");
    const auto output_frame0 = fixture->require_f32_tensor("output_frame0");
    require(spec_frame0.shape.rank == 3 && spec_frame0.shape.dims[0] == 1 && spec_frame0.shape.dims[1] == 257 && spec_frame0.shape.dims[2] == 2,
            model_name + " frame input shape mismatch");
    require(output_frame0.shape.rank == 3 && output_frame0.shape.dims[0] == 1 && output_frame0.shape.dims[1] == 257 && output_frame0.shape.dims[2] == 2,
            model_name + " frame output shape mismatch");
    auto session = model.create_streaming_session();
    std::vector<float> actual_frame(static_cast<size_t>(257 * 2), 0.0F);
    session->process_stft_frame(spec_frame0.values.data(), actual_frame.data());
    require_close(actual_frame, output_frame0, 257 * 2, 2.0e-3F, 2.0e-4, model_name + " frame case " + std::to_string(case_index));
    require(input.shape.rank == 2 && input.shape.dims[0] == 1, model_name + " input shape mismatch");
    const auto output = model.denoise_mono_16k(input.values);
    require(output.sample_rate == 16000, model_name + " sample rate mismatch");
    require_close(output.samples, expected, input.values.size(), 2.0e-3F, 2.0e-4, model_name + " case " + std::to_string(case_index));
}

}  // namespace

int main() {
    try {
        for (const std::string model_name : {"gtcrn_dns3", "gtcrn_vctk", "gtcrn_streaming"}) {
            run_case(model_name, 0);
            run_case(model_name, 1);
        }
        std::cout << "gtcrn_utility_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "gtcrn_utility_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
