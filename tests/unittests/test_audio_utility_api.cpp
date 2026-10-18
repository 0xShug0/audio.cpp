#include "engine/framework/audio/utilities/utility_api.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef ENGINE_REPO_ROOT
#define ENGINE_REPO_ROOT "."
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

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<float> tone(int sample_rate, int samples, float base_hz) {
    constexpr float kPi = 3.14159265358979323846f;
    std::vector<float> output(static_cast<size_t>(samples), 0.0f);
    for (int i = 0; i < samples; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(sample_rate);
        output[static_cast<size_t>(i)] =
            0.18f * std::sin(2.0f * kPi * base_hz * t) +
            0.04f * std::sin(2.0f * kPi * (base_hz * 2.7f) * t + 0.31f);
    }
    return output;
}

void require_wav_shape(const std::filesystem::path & path, int sample_rate) {
    const auto wav = engine::audio::read_wav_f32(path);
    require(wav.sample_rate == sample_rate, "output sample rate mismatch: " + path.string());
    require(wav.channels == 1, "output channel count mismatch: " + path.string());
    require(!wav.samples.empty(), "output WAV is empty: " + path.string());
}

}  // namespace

int main() {
    try {
        const auto root = std::filesystem::temp_directory_path() / "audio_cpp_audio_utility_api_test";
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root);
        const engine::audio::AudioUtilityPaths paths{repo_path("assets/framework/audio_utilities"), cuda_backend()};

        const auto input48 = root / "input48.wav";
        engine::audio::write_pcm16_wav(input48, 48000, 1, tone(48000, 1920, 220.0f));
        const auto rnnoise_out = root / "rnnoise.wav";
        engine::audio::denoise_file(input48, rnnoise_out, "rnnoise", paths);
        require_wav_shape(rnnoise_out, 48000);

        const auto dfn_out = root / "deepfilternet2.wav";
        engine::audio::denoise_file(input48, dfn_out, "deepfilternet2", paths);
        require_wav_shape(dfn_out, 48000);

        const auto input16 = root / "input16.wav";
        engine::audio::write_pcm16_wav(input16, 16000, 1, tone(16000, 800, 330.0f));
        const auto zipenhancer_out = root / "zipenhancer.wav";
        engine::audio::denoise_file(input16, zipenhancer_out, "zipenhancer", paths);
        require_wav_shape(zipenhancer_out, 16000);

        const auto flashsr_out = root / "flashsr.wav";
        engine::audio::super_resolve_file(input16, flashsr_out, "flashsr", paths);
        require_wav_shape(flashsr_out, 48000);

        const auto input_dir = root / "batch_in";
        const auto output_dir = root / "batch_out";
        std::filesystem::create_directories(input_dir);
        engine::audio::write_pcm16_wav(input_dir / "one.wav", 48000, 1, tone(48000, 1440, 200.0f));
        engine::audio::write_pcm16_wav(input_dir / "two.wav", 48000, 1, tone(48000, 1440, 300.0f));
        const auto rnnoise_batch = engine::audio::denoise_directory(input_dir, output_dir / "rnnoise", "rnnoise", paths);
        require(rnnoise_batch.outputs.size() == 2, "RNNoise directory output count mismatch");
        for (const auto & output : rnnoise_batch.outputs) {
            require_wav_shape(output, 48000);
        }

        const auto zipenhancer_batch = engine::audio::denoise_directory(input_dir, output_dir / "zipenhancer", "zipenhancer", paths);
        require(zipenhancer_batch.outputs.size() == 2, "ZipEnhancer directory output count mismatch");
        for (const auto & output : zipenhancer_batch.outputs) {
            require_wav_shape(output, 16000);
        }

        const auto flashsr_batch = engine::audio::super_resolve_directory(input_dir, output_dir / "flashsr", "flashsr", paths);
        require(flashsr_batch.outputs.size() == 2, "FlashSR directory output count mismatch");
        for (const auto & output : flashsr_batch.outputs) {
            require_wav_shape(output, 48000);
        }

        std::cout << "audio_utility_api_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "audio_utility_api_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
