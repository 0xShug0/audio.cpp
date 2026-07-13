#include "engine/framework/audio/wav_reader.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
void write_le(std::ofstream & output, T value) {
    output.write(reinterpret_cast<const char *>(&value), sizeof(T));
    if (!output) {
        throw std::runtime_error("failed to write test WAV");
    }
}

void write_bytes(std::ofstream & output, const char * bytes, std::streamsize count) {
    output.write(bytes, count);
    if (!output) {
        throw std::runtime_error("failed to write test WAV");
    }
}

void write_pcm24_sample(std::ofstream & output, int32_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) & 0x00FFFFFFu;
    const char bytes[3] = {
        static_cast<char>(bits & 0xFFu),
        static_cast<char>((bits >> 8) & 0xFFu),
        static_cast<char>((bits >> 16) & 0xFFu),
    };
    write_bytes(output, bytes, 3);
}

void write_pcm24_wav(
    const std::filesystem::path & path,
    int sample_rate,
    int channels,
    const std::vector<int32_t> & samples) {
    const uint32_t data_bytes = static_cast<uint32_t>(samples.size() * 3);
    const uint16_t block_align = static_cast<uint16_t>(channels * 3);
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate * block_align);

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open test WAV: " + path.string());
    }
    write_bytes(output, "RIFF", 4);
    write_le<uint32_t>(output, 36u + data_bytes);
    write_bytes(output, "WAVE", 4);
    write_bytes(output, "fmt ", 4);
    write_le<uint32_t>(output, 16u);
    write_le<uint16_t>(output, 1u);
    write_le<uint16_t>(output, static_cast<uint16_t>(channels));
    write_le<uint32_t>(output, static_cast<uint32_t>(sample_rate));
    write_le<uint32_t>(output, byte_rate);
    write_le<uint16_t>(output, block_align);
    write_le<uint16_t>(output, 24u);
    write_bytes(output, "data", 4);
    write_le<uint32_t>(output, data_bytes);
    for (const int32_t sample : samples) {
        write_pcm24_sample(output, sample);
    }
}

void require_near(float actual, float expected, const std::string & label) {
    if (std::fabs(actual - expected) > 1.0e-7F) {
        throw std::runtime_error(label + " mismatch");
    }
}

}  // namespace

int main() {
    try {
        const auto root = std::filesystem::temp_directory_path() / "audio_cpp_wav_reader_test";
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root);
        const auto path = root / "pcm24_stereo.wav";
        write_pcm24_wav(
            path,
            48000,
            2,
            {
                0,
                0x007FFFFF,
                -0x00800000,
                -1,
            });

        const auto wav = engine::audio::read_wav_f32(path);
        require(wav.sample_rate == 48000, "PCM24 sample rate mismatch");
        require(wav.channels == 2, "PCM24 channel count mismatch");
        require(wav.samples.size() == 4, "PCM24 sample count mismatch");
        require_near(wav.samples[0], 0.0F, "PCM24 zero");
        require_near(wav.samples[1], 8388607.0F / 8388608.0F, "PCM24 max positive");
        require_near(wav.samples[2], -1.0F, "PCM24 min negative");
        require_near(wav.samples[3], -1.0F / 8388608.0F, "PCM24 negative one");

        std::cout << "wav_reader_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "wav_reader_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
