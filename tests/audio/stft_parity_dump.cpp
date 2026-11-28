#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/debug/trace.h"

#include <chrono>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void write_f32_file(const std::filesystem::path & path, const std::vector<float> & values) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path.string());
    }
    out.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!out) {
        throw std::runtime_error("failed to write output file: " + path.string());
    }
}

void write_i64_file(const std::filesystem::path & path, const std::vector<int64_t> & values) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path.string());
    }
    out.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(int64_t)));
    if (!out) {
        throw std::runtime_error("failed to write output file: " + path.string());
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::cerr << "Usage: stft_parity_dump <wav> <kind:stft> <out_f32> <out_shape_i64>"
                     " [--preemphasis VALUE] [--n-fft N] [--hop-length N] [--win-length N] [--pad-mode constant|reflect]\n";
        return 1;
    }

    const std::filesystem::path wav_path(argv[1]);
    const std::string kind(argv[2]);
    const std::filesystem::path out_f32(argv[3]);
    const std::filesystem::path out_shape(argv[4]);

    const auto started = std::chrono::steady_clock::now();
    bool periodic_window = false;
    float preemphasis = 0.0f;
    int64_t n_fft = 512;
    int64_t hop_length = 160;
    int64_t win_length = 400;
    engine::audio::STFTPadMode pad_mode = engine::audio::STFTPadMode::Constant;
    for (int i = 5; i < argc; ++i) {
        const std::string option(argv[i]);
        if (option == "--log") {
            engine::debug::configure_logging({true, std::nullopt});
            continue;
        }
        if (option == "--periodic-window") {
            periodic_window = true;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("missing value for trailing option");
        }
        const std::string value(argv[++i]);
        if (option == "--preemphasis") {
            preemphasis = std::strtof(value.c_str(), nullptr);
        } else if (option == "--n-fft") {
            n_fft = std::strtoll(value.c_str(), nullptr, 10);
        } else if (option == "--hop-length") {
            hop_length = std::strtoll(value.c_str(), nullptr, 10);
        } else if (option == "--win-length") {
            win_length = std::strtoll(value.c_str(), nullptr, 10);
        } else if (option == "--pad-mode") {
            if (value == "constant") {
                pad_mode = engine::audio::STFTPadMode::Constant;
            } else if (value == "reflect") {
                pad_mode = engine::audio::STFTPadMode::Reflect;
            } else {
                throw std::runtime_error("pad-mode must be 'constant' or 'reflect'");
            }
        } else {
            throw std::runtime_error("unknown option: " + option);
        }
    }

    auto wav = engine::audio::read_wav_f32(wav_path);
    if (wav.channels != 1) {
        throw std::runtime_error("stft_parity_dump requires mono WAV");
    }

    if (preemphasis != 0.0f && !wav.samples.empty()) {
        for (size_t i = wav.samples.size() - 1; i >= 1; --i) {
            wav.samples[i] = wav.samples[i] - preemphasis * wav.samples[i - 1];
        }
    }

    const engine::audio::STFTConfig config{
        n_fft,
        hop_length,
        win_length,
        true,
        pad_mode,
    };

    std::vector<float> window(static_cast<size_t>(config.win_length), 0.0f);
    if (config.win_length == 1) {
        window[0] = 1.0f;
    } else {
        constexpr long double kPi = 3.14159265358979323846264338327950288L;
        for (int64_t i = 0; i < config.win_length; ++i) {
            window[static_cast<size_t>(i)] =
                0.5f - 0.5f * std::cos(2.0L * kPi * static_cast<long double>(i) /
                                       static_cast<long double>(periodic_window ? config.win_length : config.win_length - 1));
        }
    }

    engine::audio::AudioTensor result;
    if (kind == "stft") {
        result = engine::audio::STFT().compute_complex(
            wav.samples,
            window,
            1,
            static_cast<int64_t>(wav.samples.size()),
            config);
    } else {
        throw std::runtime_error("kind must be 'stft'");
    }

    write_f32_file(out_f32, result.values);
    write_i64_file(out_shape, result.shape);
    std::cout << "wrote " << result.values.size() << " floats\n";
    std::cout << "preemphasis " << preemphasis << "\n";
    engine::debug::timing_log_scalar("session.wall_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
    return 0;
}
