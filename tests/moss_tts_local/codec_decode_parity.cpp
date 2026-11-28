// Standalone verification for the MOSS-Audio-Tokenizer-v2 decoder (codes ->
// 48 kHz stereo waveform). It decodes a fixed, deterministic code matrix and
// prints per-channel statistics + writes a WAV, so the output can be diffed
// against the Python reference (model.decode with fp32 / autocast disabled).

#include "engine/framework/core/backend.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/codecs/moss_audio_tokenizer_codec_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
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

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::filesystem::path codec_dir = arg_value(argc, argv, "--codec", "");
        if (codec_dir.empty()) throw std::runtime_error("--codec is required");
        const int frames = int_arg(argc, argv, "--frames", 12);
        const int threads = int_arg(argc, argv, "--threads", 16);
        const int num_quantizers = int_arg(argc, argv, "--quantizers", 12);
        const std::string wav_path = arg_value(argc, argv, "--out", "moss_codec_decode_cpp.wav");

        if (frames <= 0 || num_quantizers <= 0 || num_quantizers > 12) {
            throw std::runtime_error("frames must be positive and quantizers must be in [1,12]");
        }

        constexpr size_t kWeightContextBytes = 256ull * 1024 * 1024;
        constexpr size_t kGraphArenaBytes = 1024ull * 1024 * 1024;

        std::vector<std::vector<int32_t>> codes(
            static_cast<size_t>(num_quantizers), std::vector<int32_t>(static_cast<size_t>(frames)));
        for (int q = 0; q < num_quantizers; ++q) {
            for (int t = 0; t < frames; ++t) {
                codes[static_cast<size_t>(q)][static_cast<size_t>(t)] = (q * 37 + t * 5) % 1024;
            }
        }

        engine::core::BackendConfig backend_config;
        backend_config.type = engine::core::BackendType::Cpu;
        backend_config.device = 0;
        backend_config.threads = threads;
        engine::core::ExecutionContext execution_context(backend_config);

        std::cout << "codec=" << codec_dir.string() << "\n";
        std::cout << "loading decoder weights...\n" << std::flush;
        auto codec_weights = engine::assets::open_tensor_source(codec_dir);
        engine::codecs::MossAudioTokenizerCodecRuntime decoder(
            codec_weights,
            execution_context,
            num_quantizers,
            engine::codecs::MossAudioTokenizerCodecRuntimeOptions{
                kWeightContextBytes,
                kGraphArenaBytes,
                kGraphArenaBytes,
                false,
            });

        std::cout << "decoding " << frames << " frames...\n" << std::flush;
        const auto decoded = decoder.decode(engine::codecs::MossAudioTokenizerCodes{frames, std::move(codes)});
        const auto & stereo = decoded.channels;
        if (decoded.sampling_rate != 48000 || stereo.size() != 2 ||
            stereo[0].size() != static_cast<size_t>(frames) * 3840 ||
            stereo[1].size() != stereo[0].size()) {
            throw std::runtime_error("v2 decoder returned incorrect format or length");
        }

        std::cout << "\n=== RESULT ===\n";
        std::cout << "channels=" << stereo.size() << " samples_per_channel=" << stereo.front().size() << "\n";
        for (size_t channel = 0; channel < stereo.size(); ++channel) {
            const auto & data = stereo[channel];
            double sum = 0.0;
            double sq = 0.0;
            for (float value : data) {
                if (!std::isfinite(value)) throw std::runtime_error("decoder produced non-finite samples");
                sum += value;
                sq += static_cast<double>(value) * value;
            }
            const double mean = sum / static_cast<double>(data.size());
            const double rms = std::sqrt(sq / static_cast<double>(data.size()));
            const double variance = sq / static_cast<double>(data.size()) - mean * mean;
            std::cout << "channel " << channel << ": mean=" << mean << " std="
                      << std::sqrt(std::max(0.0, variance)) << " rms=" << rms << "\n";
            std::cout << "  first16=[";
            for (int i = 0; i < 16 && i < static_cast<int>(data.size()); ++i) {
                std::cout << (i == 0 ? "" : ", ") << data[static_cast<size_t>(i)];
            }
            std::cout << "]\n";
        }

        std::vector<float> planar;
        planar.reserve(stereo[0].size() * 2);
        for (const auto & channel : stereo) planar.insert(planar.end(), channel.begin(), channel.end());
        engine::audio::write_pcm16_wav(
            wav_path, 48000, 2, engine::audio::interleave_planar_channels(planar, 2, static_cast<int64_t>(stereo[0].size())));
        std::cout << "wrote " << wav_path << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "codec_decode_parity failed: " << error.what() << "\n";
        return 1;
    }
}
