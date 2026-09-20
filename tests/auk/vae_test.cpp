#include "engine/community_models/auk/vae.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/sampling/torch_random.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <numeric>

int main(int argc, char ** argv) {
    if ((argc != 5 && argc != 6 && argc != 7) || std::string(argv[3]) != "--log" ||
        (argc == 6 && std::string(argv[5]) != "--encode-stats") ||
        (argc == 7 && std::string(argv[5]) != "--benchmark-runs" && std::string(argv[5]) != "--alternate-fixture" &&
         std::string(argv[5]) != "--latent-key")) {
        std::cerr << "Usage: auk_vae_test <vae.safetensors> <fixture-dir> --log <log-file> [--benchmark-runs N|--alternate-fixture DIR|--latent-key KEY|--encode-stats]\n";
        return 2;
    }
    engine::debug::configure_logging({true, std::string(argv[4])});
    try {
        const auto weights = engine::assets::open_tensor_source(argv[1]);
        const std::filesystem::path directory = argv[2];
        const auto reference = engine::assets::open_tensor_source(directory / "reference.safetensors");
        if (argc == 6) {
            const auto audio = reference->require_f32("vae.encoder_input");
            const auto expected = reference->require_f32("vae.encoder_stats");
            const auto shape = reference->require_metadata("vae.encoder_stats").shape;
            if (shape.size() != 3 || shape[0] != 1 || shape[1] != 128) {
                throw std::runtime_error("unexpected encoder statistics shape");
            }
            engine::core::ExecutionContext execution({engine::core::BackendType::Cuda, 0, 8});
            engine::models::auk::VaeEncoderRuntime encoder(execution, *weights, audio.size());
            const auto noise = reference->require_f32("vae.encoder_noise");
            std::vector<float> output;
            const auto normalized = encoder.encode(audio, noise, &output);
            if (output.size() != expected.size()) throw std::runtime_error("encoder statistics length mismatch");
            for (int part = 0; part < 2; ++part) {
                double dot = 0, norm = 0, expected_norm = 0, error = 0;
                const size_t count = expected.size() / 2;
                for (size_t i = part * count; i < (part + 1) * count; ++i) {
                    if (!std::isfinite(output[i])) throw std::runtime_error("non-finite encoder statistics");
                    dot += double(output[i]) * expected[i];
                    norm += double(output[i]) * output[i];
                    expected_norm += double(expected[i]) * expected[i];
                    error += std::pow(double(output[i]) - expected[i], 2);
                }
                const double cosine = dot / std::sqrt(norm * expected_norm);
                std::ostringstream message;
                message << std::setprecision(12) << "AuK encoder " << (part == 0 ? "mean" : "log_std")
                        << ": cosine=" << cosine << " RMSE=" << std::sqrt(error / count);
                engine::debug::log_message(message.str());
                std::cout << message.str() << '\n';
                if (!(cosine > 0.99999)) throw std::runtime_error("encoder statistics parity gate failed");
            }
            const auto expected_latents = reference->require_f32("vae.reference_latents");
            if (normalized.size() != expected_latents.size()) throw std::runtime_error("normalized reference shape mismatch");
            double ab = 0, aa = 0, bb = 0, squared_error = 0;
            for (size_t i = 0; i < normalized.size(); ++i) {
                if (!std::isfinite(normalized[i])) throw std::runtime_error("non-finite normalized reference");
                ab += double(normalized[i]) * expected_latents[i];
                aa += double(normalized[i]) * normalized[i];
                bb += double(expected_latents[i]) * expected_latents[i];
                squared_error += std::pow(double(normalized[i]) - expected_latents[i], 2);
            }
            const double similarity = ab / std::sqrt(aa * bb);
            std::ostringstream normalized_message;
            normalized_message << std::setprecision(12) << "AuK normalized reference with captured encoder noise: cosine="
                               << similarity << " RMSE=" << std::sqrt(squared_error / normalized.size());
            engine::debug::log_message(normalized_message.str());
            std::cout << normalized_message.str() << '\n';
            if (!(similarity > 0.99999)) throw std::runtime_error("normalized reference parity gate failed");
            const auto manifest = engine::io::json::parse_file(directory / "manifest.json");
            const auto policy = engine::sampling::resolve_torch_cuda_sampling_policy(
                engine::core::BackendType::Cuda, 0, "auk", "AuK");
            const auto native_noise = engine::sampling::generate_torch_cuda_tensor_iterator_randn(
                noise.size(), static_cast<uint64_t>(manifest.require("seed").as_i64()), 0, policy);
            double noise_error = 0;
            for (size_t i = 0; i < noise.size(); ++i) noise_error += std::pow(double(native_noise[i]) - noise[i], 2);
            const double noise_rmse = std::sqrt(noise_error / noise.size());
            engine::debug::trace_log_scalar("auk.vae.encoder.native_noise_rmse", noise_rmse);
            if (!(noise_rmse < 1.0e-6)) throw std::runtime_error("native encoder noise parity gate failed");
            const auto native_normalized = encoder.encode(audio, native_noise);
            double native_error = 0, native_dot = 0, native_norm = 0, noise_effect = 0;
            for (size_t i = 0; i < native_normalized.size(); ++i) {
                if (!std::isfinite(native_normalized[i])) throw std::runtime_error("non-finite native-noise reference");
                native_error += std::pow(double(native_normalized[i]) - expected_latents[i], 2);
                native_dot += double(native_normalized[i]) * expected_latents[i];
                native_norm += double(native_normalized[i]) * native_normalized[i];
                noise_effect += std::pow(double(native_normalized[i]) - normalized[i], 2);
            }
            const double native_rmse = std::sqrt(native_error / native_normalized.size());
            const double relative_l2 = std::sqrt(native_error / bb);
            const double native_cosine = native_dot / std::sqrt(native_norm * bb);
            const double noise_effect_rmse = std::sqrt(noise_effect / native_normalized.size());
            engine::debug::trace_log_scalar("auk.vae.encoder.native_normalized_rmse", native_rmse);
            engine::debug::trace_log_scalar("auk.vae.encoder.native_relative_l2", relative_l2);
            engine::debug::trace_log_scalar("auk.vae.encoder.native_cosine", native_cosine);
            engine::debug::trace_log_scalar("auk.vae.encoder.noise_effect_rmse", noise_effect_rmse);
            if (!(relative_l2 < 1.0e-4 && native_cosine > 0.99999999 && noise_effect_rmse < 1.0e-6)) {
                throw std::runtime_error("native-noise reference latent parity gate failed");
            }
            engine::debug::log_message("AuK reference encoder passes with native noise; no captured noise injected in this check");
            if (encoder.encode(audio, noise) != normalized) throw std::runtime_error("encoder repeated graph output differs");
            const std::vector<float> shorter(audio.begin(), audio.begin() + audio.size() / 2);
            int64_t shorter_frames = shorter.size();
            for (int rate : {2, 2, 2, 3, 4, 5}) shorter_frames = (shorter_frames - 2) / rate + 1;
            std::vector<float> shorter_output;
            for (int cycle = 0; cycle < 3; ++cycle) {
                encoder.prepare(shorter.size());
                const std::vector<float> shorter_noise(shorter_frames * 64, 0.0F);
                const auto actual = encoder.encode(shorter, shorter_noise);
                if (actual.size() != static_cast<size_t>(shorter_frames * 64)) {
                    throw std::runtime_error("shorter encoder output shape mismatch");
                }
                if (!std::all_of(actual.begin(), actual.end(), [](float value) { return std::isfinite(value); })) {
                    throw std::runtime_error("non-finite shorter encoder statistics");
                }
                if (cycle == 0) shorter_output = actual;
                else if (actual != shorter_output) throw std::runtime_error("shorter encoder output changed across cycles");
                encoder.prepare(shorter.size());
                if (encoder.encode(shorter, shorter_noise) != actual) throw std::runtime_error("same-shape encoder prepare changed output");
                encoder.prepare(audio.size());
                if (encoder.encode(audio, noise) != normalized) throw std::runtime_error("returning to original encoder shape changed output");
                const auto memory = execution.memory_snapshot();
                if (memory.available) engine::debug::trace_log_scalar("auk.vae.encoder.cycle_device_used_bytes", memory.used_bytes);
            }
            engine::debug::log_message("AuK encoder shape switching passed three round trips; shorter input is a lifecycle check only");
            return 0;
        }
        const std::string latent_key = argc == 7 && std::string(argv[5]) == "--latent-key" ? argv[6] : "fm.output";
        const auto shape = reference->require_metadata(latent_key).shape;
        if (shape.size() != 3 || shape[0] != 1 || shape[2] != 64) {
            throw std::runtime_error("unexpected FM fixture shape");
        }
        engine::core::ExecutionContext execution({engine::core::BackendType::Cuda, 0, 8});
        engine::models::auk::VaeDecoderRuntime decoder(execution, *weights, shape[1]);
        const auto latents = reference->require_f32(latent_key);
        const auto output = decoder.decode(latents);
        const auto expected = reference->require_f32("waveform");
        if (output.size() != expected.size()) {
            throw std::runtime_error("waveform length differs from Python");
        }
        engine::audio::write_pcm16_wav(directory / "native-vae.wav", 24000, 1, output);
        double dot = 0, norm = 0, expected_norm = 0, error = 0;
        for (size_t i = 0; i < output.size(); ++i) {
            if (!std::isfinite(output[i])) {
                throw std::runtime_error("non-finite VAE output");
            }
            dot += static_cast<double>(output[i]) * expected[i];
            norm += static_cast<double>(output[i]) * output[i];
            expected_norm += static_cast<double>(expected[i]) * expected[i];
            const double difference = static_cast<double>(output[i]) - expected[i];
            error += difference * difference;
        }
        const double cosine = dot / std::sqrt(norm * expected_norm);
        std::ostringstream message;
        message << std::setprecision(10) << "AuK teacher-forced VAE: samples=" << output.size()
                << " cosine=" << cosine << " RMSE=" << std::sqrt(error / output.size());
        engine::debug::log_message(message.str());
        std::cout << message.str() << '\n';
        if (decoder.decode(latents) != output) {
            throw std::runtime_error("VAE repeated graph output differs");
        }
        if (!(cosine > 0.99999)) {
            throw std::runtime_error("VAE waveform parity gate failed");
        }
        if (argc == 7 && std::string(argv[5]) == "--alternate-fixture") {
            const auto alternate = engine::assets::open_tensor_source(
                std::filesystem::path(argv[6]) / "reference.safetensors");
            const auto alternate_shape = alternate->require_metadata("fm.output").shape;
            if (alternate_shape.size() != 3 || alternate_shape[0] != 1 || alternate_shape[2] != 64 ||
                alternate_shape[1] == shape[1]) {
                throw std::runtime_error("alternate fixture must have a different valid frame count");
            }
            const auto alternate_latents = alternate->require_f32("fm.output");
            const auto alternate_expected = alternate->require_f32("waveform");
            for (int cycle = 0; cycle < 3; ++cycle) {
                decoder.prepare(alternate_shape[1]);
                const auto actual = decoder.decode(alternate_latents);
                if (actual.size() != alternate_expected.size()) throw std::runtime_error("alternate waveform length mismatch");
                const double ab = std::inner_product(actual.begin(), actual.end(), alternate_expected.begin(), 0.0,
                    std::plus<double>{}, [](float a, float b) { return double(a) * b; });
                const double aa = std::inner_product(actual.begin(), actual.end(), actual.begin(), 0.0,
                    std::plus<double>{}, [](float a, float b) { return double(a) * b; });
                const double bb = std::inner_product(alternate_expected.begin(), alternate_expected.end(), alternate_expected.begin(), 0.0,
                    std::plus<double>{}, [](float a, float b) { return double(a) * b; });
                const double similarity = ab / std::sqrt(aa * bb);
                engine::debug::trace_log_scalar("auk.vae.alternate_cosine", similarity);
                if (!(similarity > 0.99999)) throw std::runtime_error("alternate VAE parity gate failed");
                decoder.prepare(alternate_shape[1]);
                if (decoder.decode(alternate_latents) != actual) throw std::runtime_error("same-shape prepare changed output");
                decoder.prepare(shape[1]);
                if (decoder.decode(latents) != output) throw std::runtime_error("returning to original shape changed output");
                const auto memory = execution.memory_snapshot();
                if (memory.available) engine::debug::trace_log_scalar("auk.vae.cycle_device_used_bytes", memory.used_bytes);
            }
            engine::debug::log_message("AuK VAE shape switching passed three round trips");
        }
        if (argc == 7 && std::string(argv[5]) == "--benchmark-runs") {
            const int runs = std::stoi(argv[6]);
            if (runs <= 0) throw std::runtime_error("benchmark runs must be positive");
            std::vector<double> milliseconds;
            for (int run = 0; run < runs; ++run) {
                const auto start = std::chrono::steady_clock::now();
                const auto repeated = decoder.decode(latents);
                const double elapsed = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                if (repeated != output) throw std::runtime_error("benchmark VAE output changed");
                milliseconds.push_back(elapsed);
                engine::debug::timing_log_scalar("auk.vae.decode_ms", elapsed);
            }
            std::sort(milliseconds.begin(), milliseconds.end());
            const double median = (milliseconds[(runs - 1) / 2] + milliseconds[runs / 2]) * 0.5;
            engine::debug::log_message("AuK VAE median_ms=" + std::to_string(median) +
                " runs=" + std::to_string(runs));
            std::cout << "VAE median_ms=" << median << '\n';
        }
        return 0;
    } catch (const std::exception & error) {
        engine::debug::log_message(error.what());
        std::cerr << error.what() << '\n';
        return 1;
    }
}
