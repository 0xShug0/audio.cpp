// Parity probe for the ported VibeASR VAE encoder.
//
// Without --reference-* the probe only checks that the graph runs and produces a
// sane feature block. With a reference dump from VibeASR.cpp's own vae_server
// (raw float32, frames * dim, row-major) it reports max abs error, mean abs
// error, and cosine similarity, and fails outside the tolerances below.
//
// Upstream: https://github.com/microsoft/VibeASR.cpp

#include "engine/community_models/vibeasr/vae_encoder.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/filesystem.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef ENGINE_REPO_ROOT
#define ENGINE_REPO_ROOT "."
#endif

namespace {

constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 125;

// Both encoders end in an I8_S matmul, so the whole feature block shares one
// scale: agreement is judged relative to that block's dynamic range rather than
// with an absolute epsilon.
//
// Bit-exactness is not reachable here and the tolerances reflect a measured
// noise floor rather than a guess. Every stage requantizes to int8, and the two
// implementations disagree in the last float bit of the per-tensor scale (this
// port stores amax/127 and multiplies, VibeASR.cpp stores 127/amax and divides),
// which flips a handful of values by one int8 step early on. Nudging a single
// input sample by one int8 step and re-running VibeASR.cpp against itself moves
// its own output by cosine 0.9959 (acoustic) / 0.9870 (semantic) -- i.e. the
// graph amplifies one LSB to about the same distance we see between the two
// implementations, so anything tighter would be testing rounding luck.
constexpr double kMaxMeanRelativeError = 0.02;
constexpr double kMinCosineSimilarity = 0.98;

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

std::vector<float> read_f32_dump(const std::filesystem::path & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open reference dump: " + path.string());
    }
    const auto bytes = static_cast<size_t>(file.tellg());
    if (bytes % sizeof(float) != 0) {
        throw std::runtime_error("reference dump is not a whole number of floats: " + path.string());
    }
    std::vector<float> values(bytes / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(bytes));
    if (!file) {
        throw std::runtime_error("short read on reference dump: " + path.string());
    }
    return values;
}

bool check_features(
    const char * branch,
    const engine::community_models::vibeasr::VaeEncoderFeatures & features,
    int64_t expected_dim,
    int64_t expected_frames) {
    if (features.frames != expected_frames || features.dim != expected_dim) {
        std::fprintf(
            stderr,
            "FAIL: %s features are [%lld frames, %lld dim], expected [%lld, %lld]\n",
            branch,
            static_cast<long long>(features.frames),
            static_cast<long long>(features.dim),
            static_cast<long long>(expected_frames),
            static_cast<long long>(expected_dim));
        return false;
    }

    double amax = 0.0;
    for (float value : features.values) {
        if (!std::isfinite(value)) {
            std::fprintf(stderr, "FAIL: %s features contain a non-finite value\n", branch);
            return false;
        }
        amax = std::max(amax, static_cast<double>(std::fabs(value)));
    }
    if (amax == 0.0) {
        std::fprintf(stderr, "FAIL: %s features are all zero\n", branch);
        return false;
    }
    std::printf("%s: %lld frames x %lld dim, amax %.6f\n",
                branch,
                static_cast<long long>(features.frames),
                static_cast<long long>(features.dim),
                amax);
    return true;
}

bool compare_reference(
    const char * branch,
    const engine::community_models::vibeasr::VaeEncoderFeatures & features,
    const std::filesystem::path & reference_path) {
    const auto reference = read_f32_dump(reference_path);
    if (reference.size() != features.values.size()) {
        std::fprintf(
            stderr,
            "FAIL: %s reference has %zu values, encoder produced %zu\n",
            branch,
            reference.size(),
            features.values.size());
        return false;
    }

    double max_abs = 0.0;
    double sum_abs = 0.0;
    double reference_amax = 0.0;
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double a = features.values[i];
        const double b = reference[i];
        const double diff = std::fabs(a - b);
        max_abs = std::max(max_abs, diff);
        sum_abs += diff;
        reference_amax = std::max(reference_amax, std::fabs(b));
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }
    const double mean_abs = sum_abs / static_cast<double>(reference.size());
    const double cosine = (norm_a > 0.0 && norm_b > 0.0) ? dot / std::sqrt(norm_a * norm_b) : 0.0;
    const double max_relative = reference_amax > 0.0 ? max_abs / reference_amax : max_abs;
    const double mean_relative = reference_amax > 0.0 ? mean_abs / reference_amax : mean_abs;

    std::printf(
        "%s vs reference: max abs %.6g (%.3g of range), mean abs %.6g (%.3g of range), cosine %.8f\n",
        branch, max_abs, max_relative, mean_abs, mean_relative, cosine);

    bool ok = true;
    if (mean_relative > kMaxMeanRelativeError) {
        std::fprintf(stderr, "FAIL: %s mean relative error %.6g exceeds %.6g\n",
                     branch, mean_relative, kMaxMeanRelativeError);
        ok = false;
    }
    if (cosine < kMinCosineSimilarity) {
        std::fprintf(stderr, "FAIL: %s cosine %.8f is below %.8f\n", branch, cosine, kMinCosineSimilarity);
        ok = false;
    }
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::filesystem::path model_path = arg_value(argc, argv, "--model", "");
    const std::filesystem::path audio_path = arg_value(argc, argv, "--audio", "");
    const std::filesystem::path acoustic_reference = arg_value(argc, argv, "--reference-acoustic", "");
    const std::filesystem::path semantic_reference = arg_value(argc, argv, "--reference-semantic", "");
    const int threads = std::atoi(arg_value(argc, argv, "--threads", "4").c_str());

    if (model_path.empty() || !engine::io::is_existing_file(model_path) ||
        audio_path.empty() || !engine::io::is_existing_file(audio_path)) {
        std::fprintf(
            stderr,
            "SKIP: test_vibeasr_vae_encoder needs --model <vibeasr-vae-encoder-i8_s.gguf> and --audio <wav>.\n"
            "      Convert a VibeASR.cpp checkpoint with tools/community_models/convert_vibeasr_gguf.py first.\n");
        return kExitSkip;
    }

    try {
        const auto wav = engine::audio::read_wav_f32(audio_path);
        if (wav.channels != 1) {
            std::fprintf(stderr, "SKIP: %s has %d channels, the encoder takes mono\n",
                         audio_path.string().c_str(), wav.channels);
            return kExitSkip;
        }
        // The encoder is a raw-waveform stack: it accepts whatever rate the clip
        // carries, and only the frame count and the reported RTF depend on it.
        if (wav.sample_rate <= 0) {
            std::fprintf(stderr, "SKIP: %s reports sample rate %d\n",
                         audio_path.string().c_str(), wav.sample_rate);
            return kExitSkip;
        }

        auto assets = engine::community_models::vibeasr::load_vibeasr_vae_assets(model_path);
        const auto & config = assets->config;
        std::printf(
            "acoustic: %zu stages, total stride %lld, latent %lld, connector %lld\n",
            config.acoustic.stages.size(),
            static_cast<long long>(config.acoustic.total_stride),
            static_cast<long long>(config.acoustic.latent_dim),
            static_cast<long long>(config.acoustic.connector_hidden));

        engine::core::BackendConfig backend_config;
        backend_config.type = engine::core::BackendType::Cpu;
        backend_config.threads = threads > 0 ? threads : 1;
        engine::core::ExecutionContext execution_context(backend_config);

        engine::community_models::vibeasr::VibeASRVaeEncoderRuntime runtime(assets, execution_context);

        const auto num_samples = static_cast<int64_t>(wav.samples.size());
        const double audio_seconds = static_cast<double>(num_samples) / static_cast<double>(wav.sample_rate);

        const auto acoustic_start = std::chrono::steady_clock::now();
        const auto acoustic = runtime.encode_acoustic(wav.samples);
        const auto semantic_start = std::chrono::steady_clock::now();
        const auto semantic = runtime.encode_semantic(wav.samples);
        const auto encode_end = std::chrono::steady_clock::now();

        const auto ms = [](auto from, auto to) {
            return std::chrono::duration<double, std::milli>(to - from).count();
        };
        const double acoustic_ms = ms(acoustic_start, semantic_start);
        const double semantic_ms = ms(semantic_start, encode_end);
        std::printf(
            "encode wall: acoustic %.1f ms, semantic %.1f ms, both %.1f ms for %.2f s of audio (RTF %.4f)\n",
            acoustic_ms, semantic_ms, acoustic_ms + semantic_ms, audio_seconds,
            (acoustic_ms + semantic_ms) / 1000.0 / audio_seconds);

        bool ok = true;
        ok &= check_features(
            "acoustic", acoustic, config.acoustic.connector_hidden,
            config.acoustic.frames_for_samples(num_samples));
        ok &= check_features(
            "semantic", semantic, config.semantic.connector_hidden,
            config.semantic.frames_for_samples(num_samples));

        if (!acoustic_reference.empty()) {
            ok &= compare_reference("acoustic", acoustic, acoustic_reference);
        }
        if (!semantic_reference.empty()) {
            ok &= compare_reference("semantic", semantic, semantic_reference);
        }
        if (acoustic_reference.empty() && semantic_reference.empty()) {
            std::printf("no reference dump given: shape and sanity checks only\n");
        }

        return ok ? kExitPass : kExitFail;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return kExitFail;
    }
}
