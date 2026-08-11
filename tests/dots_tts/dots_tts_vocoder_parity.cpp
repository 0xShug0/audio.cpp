#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/output.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/trace.h"
#include "engine/models/dots_tts/assets.h"
#include "engine/models/dots_tts/audio_vae.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

int int_arg(int argc, char ** argv, const std::string & name, int fallback) {
    return std::stoi(arg_value(argc, argv, name, std::to_string(fallback)));
}

double double_arg(int argc, char ** argv, const std::string & name, double fallback) {
    return std::stod(arg_value(argc, argv, name, std::to_string(fallback)));
}

engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cuda") {
        return engine::core::BackendType::Cuda;
    }
    if (value == "vulkan") {
        return engine::core::BackendType::Vulkan;
    }
    if (value == "best") {
        return engine::core::BackendType::BestAvailable;
    }
    throw std::runtime_error("dots_tts_vocoder_parity only supports cuda, vulkan, or best backends");
}

std::vector<float> read_f32_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open f32 file: " + path.string());
    }
    const auto bytes = input.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("f32 file size is not divisible by sizeof(float): " + path.string());
    }
    input.seekg(0, std::ios::beg);
    std::vector<float> values(static_cast<size_t>(bytes / static_cast<std::streamoff>(sizeof(float))));
    input.read(reinterpret_cast<char *>(values.data()), bytes);
    if (!input) {
        throw std::runtime_error("failed to read f32 file: " + path.string());
    }
    return values;
}

double rms(const std::vector<float> & values) {
    if (values.empty()) {
        return 0.0;
    }
    double sum_sq = 0.0;
    for (float value : values) {
        sum_sq += static_cast<double>(value) * static_cast<double>(value);
    }
    return std::sqrt(sum_sq / static_cast<double>(values.size()));
}

double peak_abs(const std::vector<float> & values) {
    double peak = 0.0;
    for (float value : values) {
        peak = std::max(peak, std::fabs(static_cast<double>(value)));
    }
    return peak;
}

double cosine_similarity(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    const size_t count = std::min(lhs.size(), rhs.size());
    double dot = 0.0;
    double lhs_norm = 0.0;
    double rhs_norm = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double a = lhs[i];
        const double b = rhs[i];
        dot += a * b;
        lhs_norm += a * a;
        rhs_norm += b * b;
    }
    if (lhs_norm == 0.0 || rhs_norm == 0.0) {
        return 0.0;
    }
    return dot / (std::sqrt(lhs_norm) * std::sqrt(rhs_norm));
}

double max_abs_diff(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    const size_t count = std::min(lhs.size(), rhs.size());
    double diff = 0.0;
    for (size_t i = 0; i < count; ++i) {
        diff = std::max(diff, std::fabs(static_cast<double>(lhs[i]) - static_cast<double>(rhs[i])));
    }
    return diff;
}

}  // namespace

int main(int argc, char ** argv) try {
    const std::filesystem::path model_path = arg_value(argc, argv, "--model", "");
    const std::filesystem::path latents_path = arg_value(argc, argv, "--latents", "");
    const std::filesystem::path reference_wav_path = arg_value(argc, argv, "--reference-wav", "");
    const std::filesystem::path output_wav_path = arg_value(argc, argv, "--out", "dots_tts_vocoder_cpp.wav");
    const std::filesystem::path stream_output_wav_path = arg_value(argc, argv, "--stream-out", "");
    const std::string trace_log_path = arg_value(argc, argv, "--trace-log", "");
    const int threads = int_arg(argc, argv, "--threads", 1);
    const int device = int_arg(argc, argv, "--device", 0);
    const int stream_chunk_frames = int_arg(argc, argv, "--stream-chunk-frames", 0);
    const double min_cosine = double_arg(argc, argv, "--min-cosine", 0.98);

    if (model_path.empty() || latents_path.empty() || reference_wav_path.empty()) {
        std::cerr
            << "usage: dots_tts_vocoder_parity --model <gguf-or-model-dir> "
            << "--latents <post-ar-denormalized-f32> --reference-wav <python-vocoder.wav> "
            << "[--backend cuda|vulkan|best] [--device 0] [--threads N] [--out cpp.wav] "
            << "[--stream-out stream.wav] [--stream-chunk-frames N] "
            << "[--min-cosine 0.98] [--trace-log trace.log]\n";
        return 2;
    }
    if (!trace_log_path.empty()) {
        engine::debug::configure_logging(engine::debug::LoggingConfig{true, trace_log_path});
    }

    const auto backend = parse_backend(arg_value(argc, argv, "--backend", "cuda"));
    const auto assets = engine::models::dots_tts::load_dots_assets(model_path);
    const int64_t latent_dim = assets->config.latent_dim;
    auto latents = read_f32_file(latents_path);
    if (latent_dim <= 0 || static_cast<int64_t>(latents.size()) % latent_dim != 0) {
        throw std::runtime_error("latent fixture size is not divisible by DotTTS latent_dim");
    }
    const int64_t frames = static_cast<int64_t>(latents.size()) / latent_dim;

    engine::core::BackendConfig backend_config;
    backend_config.type = backend;
    backend_config.device = device;
    backend_config.threads = threads;
    auto audio_vae = engine::models::dots_tts::DotsAudioVaeComponent::load_from_tensor_source(
        assets->vocoder_weights,
        backend_config,
        assets->config.vocoder,
        engine::assets::TensorStorageType::Native,
        engine::assets::TensorStorageType::Native);
    const auto decoded = audio_vae.decode_latents(latents, frames);
    std::vector<float> streamed_samples;
    if (!stream_output_wav_path.empty()) {
        const int64_t chunk_frames = stream_chunk_frames > 0 ? stream_chunk_frames : assets->config.patch_size;
        auto stream_state = audio_vae.create_stream_state(chunk_frames);
        for (int64_t start = 0; start < frames; start += chunk_frames) {
            const int64_t current = std::min<int64_t>(chunk_frames, frames - start);
            std::vector<float> chunk(static_cast<size_t>(current * latent_dim), 0.0F);
            std::copy(
                latents.begin() + static_cast<std::ptrdiff_t>(start * latent_dim),
                latents.begin() + static_cast<std::ptrdiff_t>((start + current) * latent_dim),
                chunk.begin());
            auto audio = audio_vae.stream_step(chunk, current, stream_state);
            streamed_samples.insert(streamed_samples.end(), audio.samples.begin(), audio.samples.end());
        }
        auto final_audio = audio_vae.flush_stream(stream_state);
        streamed_samples.insert(streamed_samples.end(), final_audio.samples.begin(), final_audio.samples.end());
        if (!stream_output_wav_path.parent_path().empty()) {
            std::filesystem::create_directories(stream_output_wav_path.parent_path());
        }
        engine::audio::WavPcm16Sink().write(
            stream_output_wav_path,
            engine::audio::AudioBuffer{static_cast<int>(decoded.sample_rate), 1, streamed_samples});
    }

    const auto reference = engine::audio::read_wav_f32(reference_wav_path);
    if (reference.channels != 1) {
        throw std::runtime_error("reference wav must be mono");
    }
    if (reference.sample_rate != decoded.sample_rate) {
        throw std::runtime_error("reference wav sample rate does not match decoded sample rate");
    }

    if (!output_wav_path.parent_path().empty()) {
        std::filesystem::create_directories(output_wav_path.parent_path());
    }
    engine::audio::WavPcm16Sink().write(
        output_wav_path,
        engine::audio::AudioBuffer{static_cast<int>(decoded.sample_rate), 1, decoded.samples});

    const double cosine = cosine_similarity(decoded.samples, reference.samples);
    const double max_diff = max_abs_diff(decoded.samples, reference.samples);
    std::cout << std::fixed << std::setprecision(6)
              << "frames=" << frames
              << " sample_rate=" << decoded.sample_rate
              << " cpp_samples=" << decoded.samples.size()
              << " reference_samples=" << reference.samples.size()
              << " cpp_rms=" << rms(decoded.samples)
              << " reference_rms=" << rms(reference.samples)
              << " cpp_peak=" << peak_abs(decoded.samples)
              << " reference_peak=" << peak_abs(reference.samples)
              << " waveform_cosine=" << cosine
              << " max_abs_diff=" << max_diff
              << " cpp_wav=" << output_wav_path.string();
    if (!stream_output_wav_path.empty()) {
        std::cout << " stream_samples=" << streamed_samples.size()
                  << " stream_full_cosine=" << cosine_similarity(streamed_samples, decoded.samples)
                  << " stream_full_max_abs_diff=" << max_abs_diff(streamed_samples, decoded.samples)
                  << " stream_wav=" << stream_output_wav_path.string();
    }
    std::cout << "\n";

    if (decoded.samples.size() != reference.samples.size()) {
        throw std::runtime_error("decoded waveform sample count does not match Python vocoder reference");
    }
    if (cosine < min_cosine) {
        throw std::runtime_error("decoded waveform cosine is below threshold");
    }
    return 0;
} catch (const std::exception & error) {
    std::cerr << "dots_tts_vocoder_parity failed: " << error.what() << '\n';
    return 1;
}
