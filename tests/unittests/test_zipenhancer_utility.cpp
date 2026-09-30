#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/utilities/zipenhancer.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <algorithm>
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

void run_case(const engine::audio::ZipEnhancerModel & model, int case_index) {
    const auto fixture = engine::assets::open_tensor_source(
        asset_path("framework/audio_utilities/zipenhancer/zipenhancer_case" + std::to_string(case_index) + ".safetensors"));
    const auto input = fixture->require_f32_tensor("input");
    const auto expected = fixture->require_f32_tensor("output");
    require(input.shape.rank == 2 && input.shape.dims[0] == 1, "ZipEnhancer input shape mismatch");
    const auto output = model.denoise_mono_16k(input.values);
    require(output.sample_rate == 16000, "ZipEnhancer sample rate mismatch");
    require_close(output.samples, expected, 3.0e-3f, 3.0e-4, "case " + std::to_string(case_index));
}

}  // namespace

int main() {
    try {
        const auto model = engine::audio::ZipEnhancerModel::load_from_directory(
            repo_path("assets/framework/audio_utilities/zipenhancer"),
            cuda_backend());
        const auto fixture_dir = asset_path("framework/audio_utilities/zipenhancer");
        std::vector<int> cases;
        for (const auto & entry : std::filesystem::directory_iterator(fixture_dir)) {
            const auto name = entry.path().filename().string();
            constexpr const char * prefix = "zipenhancer_case";
            constexpr const char * suffix = ".safetensors";
            if (name.rfind(prefix, 0) != 0 ||
                name.size() <= std::char_traits<char>::length(prefix) + std::char_traits<char>::length(suffix) ||
                name.substr(name.size() - std::char_traits<char>::length(suffix)) != suffix) {
                continue;
            }
            const auto index = name.substr(
                std::char_traits<char>::length(prefix),
                name.size() - std::char_traits<char>::length(prefix) - std::char_traits<char>::length(suffix));
            cases.push_back(std::stoi(index));
        }
        std::sort(cases.begin(), cases.end());
        require(cases.size() >= 2, "ZipEnhancer requires at least two reference fixtures");
        for (const int case_index : cases) {
            run_case(model, case_index);
        }
        std::cout << "zipenhancer_utility_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "zipenhancer_utility_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
