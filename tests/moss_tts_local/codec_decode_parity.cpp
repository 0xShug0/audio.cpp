// Standalone verification for the MOSS-Audio-Tokenizer-v2 decoder (codes ->
// 48 kHz stereo waveform). It decodes a fixed, deterministic code matrix and
// prints per-channel statistics + writes a WAV, so the output can be diffed
// against the Python reference (model.decode with fp32 / autocast disabled).

#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/models/moss/shared/audio_tokenizer_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
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

void write_pcm16_wav(
    const std::string & path,
    const std::vector<std::vector<float>> & channels,
    int32_t sample_rate) {
    const int32_t num_channels = static_cast<int32_t>(channels.size());
    const int64_t frames = channels.empty() ? 0 : static_cast<int64_t>(channels.front().size());
    const int32_t bits = 16;
    const int32_t block_align = num_channels * bits / 8;
    const int32_t byte_rate = sample_rate * block_align;
    const int32_t data_bytes = static_cast<int32_t>(frames * block_align);

    std::ofstream out(path, std::ios::binary);
    const auto put32 = [&](int32_t value) { out.write(reinterpret_cast<const char *>(&value), 4); };
    const auto put16 = [&](int16_t value) { out.write(reinterpret_cast<const char *>(&value), 2); };
    out.write("RIFF", 4);
    put32(36 + data_bytes);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    put32(16);
    put16(1);
    put16(static_cast<int16_t>(num_channels));
    put32(sample_rate);
    put32(byte_rate);
    put16(static_cast<int16_t>(block_align));
    put16(static_cast<int16_t>(bits));
    out.write("data", 4);
    put32(data_bytes);
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int32_t channel = 0; channel < num_channels; ++channel) {
            float sample = channels[static_cast<size_t>(channel)][static_cast<size_t>(frame)];
            sample = std::max(-1.0F, std::min(1.0F, sample));
            put16(static_cast<int16_t>(std::lround(sample * 32767.0F)));
        }
    }
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const std::string default_codec =
            "C:/Users/justi/.cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Audio-Tokenizer-v2/"
            "snapshots/f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169";
        const std::filesystem::path codec_dir = arg_value(argc, argv, "--codec", default_codec);
        const int frames = int_arg(argc, argv, "--frames", 12);
        const int threads = int_arg(argc, argv, "--threads", 16);
        const int num_quantizers = int_arg(argc, argv, "--quantizers", 12);
        const std::string wav_path = arg_value(argc, argv, "--out", "moss_codec_decode_cpp.wav");

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
        engine::models::moss::MossAudioTokenizerDecoder decoder(
            codec_dir, execution_context, num_quantizers, kWeightContextBytes, kGraphArenaBytes);

        std::cout << "decoding " << frames << " frames...\n" << std::flush;
        const auto stereo = decoder.decode(codes);

        std::cout << "\n=== RESULT ===\n";
        std::cout << "channels=" << stereo.size() << " samples_per_channel=" << stereo.front().size() << "\n";
        for (size_t channel = 0; channel < stereo.size(); ++channel) {
            const auto & data = stereo[channel];
            double sum = 0.0;
            double sq = 0.0;
            for (float value : data) {
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

        write_pcm16_wav(wav_path, stereo, 48000);
        std::cout << "wrote " << wav_path << "\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "codec_decode_parity failed: " << error.what() << "\n";
        return 1;
    }
}
