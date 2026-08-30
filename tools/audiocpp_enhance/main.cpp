// audiocpp_enhance - command line front end for the framework audio utilities.
//
// The denoise, FlashSR super-resolution, and resampling helpers in
// engine::audio are library-only; this tool exposes them for shell use and for
// scripts such as scripts/omnivoice_studio.sh.

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/audio/utility_api.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

void print_usage(std::ostream & out) {
    out << "Usage: audiocpp_enhance --in <path> --out <path> [options]\n"
           "\n"
           "Options:\n"
           "  --in <path>            Input WAV file, or directory with --batch.\n"
           "  --out <path>           Output WAV file, or directory with --batch.\n"
           "  --denoise <model>      rnnoise (48k), deepfilternet2 (48k), zipenhancer (16k).\n"
           "  --flashsr              FlashSR super-resolution, 16k input to 48k output.\n"
           "  --resample <hz>        Resample the result, for example 44100.\n"
           "  --backend <name>       cpu or metal. Default cpu.\n"
           "  --threads <n>          Host worker threads. Default 4.\n"
           "  --assets <dir>         Utility weight directory.\n"
           "                         Default assets/framework/audio_utilities.\n"
           "  --batch                Treat --in and --out as directories.\n"
           "  -h, --help             Show this help.\n"
           "\n"
           "Utility models expect their native rate: rnnoise and deepfilternet2 want\n"
           "48 kHz input, zipenhancer and flashsr want 16 kHz input. Feed the right\n"
           "rate, or resample first with a separate --resample pass.\n";
}

engine::core::BackendType parse_backend(std::string_view name) {
    if (name == "cpu") {
        return engine::core::BackendType::Cpu;
    }
    if (name == "metal") {
        return engine::core::BackendType::Metal;
    }
    throw std::runtime_error("unsupported --backend value: " + std::string(name));
}

std::string require_value(int argc, char ** argv, int & index, const std::string & flag) {
    if (index + 1 >= argc) {
        throw std::runtime_error(flag + " requires a value");
    }
    ++index;
    return argv[index];
}

// Resample every channel independently and re-interleave. The previous
// implementation refused anything but mono, which made --resample unusable on
// the stereo output of the music models.
void resample_file(const std::filesystem::path & path, int target_rate) {
    const auto wav = engine::audio::read_wav_f32(path);
    if (wav.sample_rate == target_rate) {
        return;
    }
    if (wav.channels < 1) {
        throw std::runtime_error("resampling expects at least one channel: " + path.string());
    }

    engine::audio::SoxrResampleOptions options;
    options.warning_context = "audiocpp_enhance";

    std::vector<std::vector<float>> planes;
    planes.reserve(static_cast<size_t>(wav.channels));
    size_t resampled_frames = 0;
    for (int channel = 0; channel < wav.channels; ++channel) {
        const auto plane = wav.channels == 1
            ? wav.samples
            : engine::audio::extract_interleaved_channel(wav.samples, wav.channels, channel);
        // resample_mono_soxr_or_sinc uses libsoxr when it is loadable and the
        // in-tree windowed sinc otherwise. It never falls back to linear
        // interpolation, which on an integer rate ratio degenerates into plain
        // sample-dropping and passes the alias through unattenuated.
        planes.push_back(
            engine::audio::resample_mono_soxr_or_sinc(plane, wav.sample_rate, target_rate, options));
        resampled_frames = std::max(resampled_frames, planes.back().size());
    }
    for (auto & plane : planes) {
        plane.resize(resampled_frames, 0.0F);
    }

    if (wav.channels == 1) {
        engine::audio::write_pcm16_wav(path, target_rate, 1, planes.front());
        return;
    }

    // interleave_planar_channels takes one flat planar buffer: channel 0's
    // frames, then channel 1's, and so on.
    std::vector<float> planar;
    planar.reserve(resampled_frames * static_cast<size_t>(wav.channels));
    for (const auto & plane : planes) {
        planar.insert(planar.end(), plane.begin(), plane.end());
    }
    const auto interleaved = engine::audio::interleave_planar_channels(
        planar, wav.channels, static_cast<int64_t>(resampled_frames));
    engine::audio::write_pcm16_wav(path, target_rate, wav.channels, interleaved);
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        std::filesystem::path input;
        std::filesystem::path output;
        std::filesystem::path assets = "assets/framework/audio_utilities";
        std::string denoise_model;
        std::string backend_name = "cpu";
        int threads = 4;
        int resample_rate = 0;
        bool flashsr = false;
        bool batch = false;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "-h" || arg == "--help") {
                print_usage(std::cout);
                return 0;
            } else if (arg == "--in") {
                input = require_value(argc, argv, i, arg);
            } else if (arg == "--out") {
                output = require_value(argc, argv, i, arg);
            } else if (arg == "--denoise") {
                denoise_model = require_value(argc, argv, i, arg);
            } else if (arg == "--flashsr") {
                flashsr = true;
            } else if (arg == "--resample") {
                resample_rate = std::stoi(require_value(argc, argv, i, arg));
            } else if (arg == "--backend") {
                backend_name = require_value(argc, argv, i, arg);
            } else if (arg == "--threads") {
                threads = std::stoi(require_value(argc, argv, i, arg));
            } else if (arg == "--assets") {
                assets = require_value(argc, argv, i, arg);
            } else if (arg == "--batch") {
                batch = true;
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }

        if (input.empty() || output.empty()) {
            print_usage(std::cerr);
            throw std::runtime_error("--in and --out are required");
        }
        if (denoise_model.empty() && !flashsr && resample_rate <= 0) {
            throw std::runtime_error("nothing to do: pass --denoise, --flashsr, or --resample");
        }
        if (!denoise_model.empty() && flashsr) {
            throw std::runtime_error("--denoise and --flashsr write the same output; run them as separate passes");
        }
        if (!std::filesystem::exists(assets) && (!denoise_model.empty() || flashsr)) {
            throw std::runtime_error("utility asset directory not found: " + assets.string());
        }

        engine::core::BackendConfig backend;
        backend.type = parse_backend(backend_name);
        backend.device = 0;
        backend.threads = threads;
        const engine::audio::AudioUtilityPaths paths{assets, backend};

        std::vector<std::filesystem::path> written;
        if (batch) {
            std::filesystem::create_directories(output);
            if (!denoise_model.empty()) {
                written = engine::audio::denoise_directory(input, output, denoise_model, paths).outputs;
            } else if (flashsr) {
                written = engine::audio::super_resolve_directory(input, output, "flashsr", paths).outputs;
            } else {
                for (const auto & entry : std::filesystem::directory_iterator(input)) {
                    if (!entry.is_regular_file() || entry.path().extension() != ".wav") {
                        continue;
                    }
                    const auto destination = output / entry.path().filename();
                    std::filesystem::copy_file(
                        entry.path(), destination, std::filesystem::copy_options::overwrite_existing);
                    written.push_back(destination);
                }
            }
        } else {
            if (output.has_parent_path()) {
                std::filesystem::create_directories(output.parent_path());
            }
            if (!denoise_model.empty()) {
                engine::audio::denoise_file(input, output, denoise_model, paths);
            } else if (flashsr) {
                engine::audio::super_resolve_file(input, output, "flashsr", paths);
            } else {
                std::filesystem::copy_file(input, output, std::filesystem::copy_options::overwrite_existing);
            }
            written.push_back(output);
        }

        if (resample_rate > 0) {
            for (const auto & path : written) {
                resample_file(path, resample_rate);
            }
        }

        for (const auto & path : written) {
            const auto wav = engine::audio::read_wav_f32(path);
            std::cout << "wrote " << path.string() << " sample_rate=" << wav.sample_rate
                      << " channels=" << wav.channels << " samples=" << wav.samples.size() << "\n";
        }
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "audiocpp_enhance failed: " << ex.what() << "\n";
        return 1;
    }
}
