#include "engine/framework/audio/wav_reader.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
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

// Writes a fmt chunk of `format_tag`/`bits` plus a data chunk of raw bytes. When
// `extensible` is set the chunk is the 40-byte WAVEFORMATEXTENSIBLE layout and
// `format_tag` moves into the SubFormat GUID, exactly as encoders emit it for
// multichannel or channel-masked PCM.
void write_wav(
    const std::filesystem::path & path,
    uint16_t format_tag,
    uint16_t bits,
    int sample_rate,
    int channels,
    const std::vector<char> & payload,
    bool extensible = false) {
    const uint16_t block_align = static_cast<uint16_t>(channels * ((bits + 7) / 8));
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * block_align;
    const uint32_t data_bytes = static_cast<uint32_t>(payload.size());
    const uint32_t fmt_bytes = extensible ? 40u : 16u;

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open test WAV: " + path.string());
    }
    write_bytes(output, "RIFF", 4);
    write_le<uint32_t>(output, 20u + fmt_bytes + data_bytes);
    write_bytes(output, "WAVE", 4);
    write_bytes(output, "fmt ", 4);
    write_le<uint32_t>(output, fmt_bytes);
    write_le<uint16_t>(output, extensible ? uint16_t{0xFFFE} : format_tag);
    write_le<uint16_t>(output, static_cast<uint16_t>(channels));
    write_le<uint32_t>(output, static_cast<uint32_t>(sample_rate));
    write_le<uint32_t>(output, byte_rate);
    write_le<uint16_t>(output, block_align);
    write_le<uint16_t>(output, bits);
    if (extensible) {
        write_le<uint16_t>(output, uint16_t{22});      // cbSize
        write_le<uint16_t>(output, bits);              // wValidBitsPerSample
        write_le<uint32_t>(output, uint32_t{0x3});     // dwChannelMask
        write_le<uint16_t>(output, format_tag);        // SubFormat GUID, first field
        // Remainder of KSDATAFORMAT_SUBTYPE_*: 0000-0010-8000-00aa00389b71
        const char guid_tail[14] = {
            0x00, 0x00, 0x00, 0x00, 0x10, 0x00, static_cast<char>(0x80),
            0x00, 0x00, static_cast<char>(0xAA), 0x00, 0x38, static_cast<char>(0x9B),
            0x71,
        };
        write_bytes(output, guid_tail, 14);
    }
    write_bytes(output, "data", 4);
    write_le<uint32_t>(output, data_bytes);
    if (data_bytes > 0) {
        write_bytes(output, payload.data(), static_cast<std::streamsize>(data_bytes));
    }
}

template <typename T>
std::vector<char> to_bytes(const std::vector<T> & values) {
    std::vector<char> out(values.size() * sizeof(T));
    if (!values.empty()) {
        std::memcpy(out.data(), values.data(), out.size());
    }
    return out;
}

void require_near(float actual, float expected, float tolerance, const std::string & label) {
    if (!std::isfinite(actual)) {
        throw std::runtime_error(label + " is not finite");
    }
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(label + " mismatch");
    }
}

// Runs `body` and requires it to throw with `needle` in the message. A silent
// success here would mean the reader accepted something it cannot decode.
void require_throws_containing(
    const std::function<void()> & body, const std::string & needle, const std::string & label) {
    try {
        body();
    } catch (const std::exception & ex) {
        if (std::string(ex.what()).find(needle) == std::string::npos) {
            throw std::runtime_error(label + ": wrong message: " + ex.what());
        }
        return;
    }
    throw std::runtime_error(label + ": expected a throw, got none");
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

        // --- WAVEFORMATEXTENSIBLE wrapping ordinary PCM16 ---------------------
        // The case that actually bites: the payload is plain PCM16, but the
        // format tag says 0xFFFE and the real tag lives in the SubFormat GUID.
        // A reader that stops at the tag rejects a file it can decode.
        {
            const auto ext = root / "extensible_pcm16.wav";
            write_wav(ext, 0x0001, 16, 44100, 2,
                      to_bytes<int16_t>({0, 16384, -16384, -32768}), true);
            const auto wav = engine::audio::read_wav_f32(ext);
            require(wav.sample_rate == 44100, "EXTENSIBLE PCM16 sample rate mismatch");
            require(wav.channels == 2, "EXTENSIBLE PCM16 channel count mismatch");
            require(wav.samples.size() == 4, "EXTENSIBLE PCM16 sample count mismatch");
            require_near(wav.samples[0], 0.0F, 1.0e-7F, "EXTENSIBLE PCM16 zero");
            require_near(wav.samples[1], 0.5F, 1.0e-7F, "EXTENSIBLE PCM16 half");
            require_near(wav.samples[2], -0.5F, 1.0e-7F, "EXTENSIBLE PCM16 negative half");
            require_near(wav.samples[3], -1.0F, 1.0e-7F, "EXTENSIBLE PCM16 full negative");
        }

        // --- WAVEFORMATEXTENSIBLE wrapping float32 ---------------------------
        // Proves the GUID is actually read rather than assumed to be PCM.
        {
            const auto ext = root / "extensible_f32.wav";
            write_wav(ext, 0x0003, 32, 48000, 1,
                      to_bytes<float>({0.0F, 0.25F, -0.75F}), true);
            const auto wav = engine::audio::read_wav_f32(ext);
            require(wav.channels == 1, "EXTENSIBLE float32 channel count mismatch");
            require(wav.samples.size() == 3, "EXTENSIBLE float32 sample count mismatch");
            require_near(wav.samples[1], 0.25F, 1.0e-7F, "EXTENSIBLE float32 quarter");
            require_near(wav.samples[2], -0.75F, 1.0e-7F, "EXTENSIBLE float32 negative");
        }

        // --- PCM8 is unsigned, biased by 128 ---------------------------------
        // The sign convention differs from every other PCM width, so a decoder
        // that treats it as signed silently inverts the waveform.
        {
            const auto path8 = root / "pcm8.wav";
            write_wav(path8, 0x0001, 8, 8000, 1,
                      std::vector<char>{static_cast<char>(128), static_cast<char>(255),
                                        static_cast<char>(0), static_cast<char>(64)});
            const auto wav = engine::audio::read_wav_f32(path8);
            require(wav.samples.size() == 4, "PCM8 sample count mismatch");
            require_near(wav.samples[0], 0.0F, 1.0e-7F, "PCM8 midpoint is silence");
            require_near(wav.samples[1], 127.0F / 128.0F, 1.0e-7F, "PCM8 max positive");
            require_near(wav.samples[2], -1.0F, 1.0e-7F, "PCM8 min negative");
            require_near(wav.samples[3], -0.5F, 1.0e-7F, "PCM8 quarter scale");
        }

        // --- PCM32 -----------------------------------------------------------
        {
            const auto path32 = root / "pcm32.wav";
            write_wav(path32, 0x0001, 32, 96000, 1,
                      to_bytes<int32_t>({0, 1073741824, -2147483647 - 1}));
            const auto wav = engine::audio::read_wav_f32(path32);
            require(wav.sample_rate == 96000, "PCM32 sample rate mismatch");
            require(wav.samples.size() == 3, "PCM32 sample count mismatch");
            require_near(wav.samples[0], 0.0F, 1.0e-7F, "PCM32 zero");
            require_near(wav.samples[1], 0.5F, 1.0e-7F, "PCM32 half");
            require_near(wav.samples[2], -1.0F, 1.0e-7F, "PCM32 full negative");
        }

        // --- float64 ---------------------------------------------------------
        {
            const auto path64 = root / "float64.wav";
            write_wav(path64, 0x0003, 64, 44100, 2,
                      to_bytes<double>({0.0, 0.125, -0.875, 1.0}));
            const auto wav = engine::audio::read_wav_f32(path64);
            require(wav.channels == 2, "float64 channel count mismatch");
            require(wav.samples.size() == 4, "float64 sample count mismatch");
            require_near(wav.samples[1], 0.125F, 1.0e-7F, "float64 eighth");
            require_near(wav.samples[2], -0.875F, 1.0e-7F, "float64 negative");
            require_near(wav.samples[3], 1.0F, 1.0e-7F, "float64 unity");
        }

        // --- G.711 mu-law ----------------------------------------------------
        // Anchored on the published G.711 decode values, not on our own
        // implementation: 0x00 -> -32124, 0x80 -> +32124, and both 0x7F and 0xFF
        // -> 0. Pinning against a re-derivation of the same bit-twiddling would
        // prove nothing.
        {
            const auto path_mu = root / "mulaw.wav";
            write_wav(path_mu, 0x0007, 8, 8000, 1,
                      std::vector<char>{static_cast<char>(0xFF), static_cast<char>(0x7F),
                                        static_cast<char>(0x00), static_cast<char>(0x80)});
            const auto wav = engine::audio::read_wav_f32(path_mu);
            require(wav.samples.size() == 4, "mu-law sample count mismatch");
            require_near(wav.samples[0], 0.0F, 1.0e-7F, "mu-law 0xFF is silence");
            require_near(wav.samples[1], 0.0F, 1.0e-7F, "mu-law 0x7F is silence");
            require_near(wav.samples[2], -32124.0F / 32768.0F, 1.0e-7F, "mu-law 0x00 minimum");
            require_near(wav.samples[3], 32124.0F / 32768.0F, 1.0e-7F, "mu-law 0x80 maximum");
        }

        // --- G.711 A-law -----------------------------------------------------
        // Published anchors again: 0x55 -> +8, 0xD5 -> -8, 0x2A -> +32256,
        // 0xAA -> -32256. A-law has no exact zero, which is itself worth pinning.
        {
            const auto path_a = root / "alaw.wav";
            write_wav(path_a, 0x0006, 8, 8000, 1,
                      std::vector<char>{static_cast<char>(0x55), static_cast<char>(0xD5),
                                        static_cast<char>(0x2A), static_cast<char>(0xAA)});
            const auto wav = engine::audio::read_wav_f32(path_a);
            require(wav.samples.size() == 4, "A-law sample count mismatch");
            require_near(wav.samples[0], 8.0F / 32768.0F, 1.0e-7F, "A-law 0x55 smallest positive");
            require_near(wav.samples[1], -8.0F / 32768.0F, 1.0e-7F, "A-law 0xD5 smallest negative");
            require_near(wav.samples[2], 32256.0F / 32768.0F, 1.0e-7F, "A-law 0x2A maximum");
            require_near(wav.samples[3], -32256.0F / 32768.0F, 1.0e-7F, "A-law 0xAA minimum");
        }

        // --- Still rejects what it genuinely cannot decode -------------------
        // Widening the accepted set must not turn into accepting everything.
        {
            const auto path_bad = root / "unsupported.wav";
            write_wav(path_bad, 0x0011, 4, 8000, 1, std::vector<char>{0x01, 0x02});
            require_throws_containing(
                [&] { (void)engine::audio::read_wav_f32(path_bad); },
                "unsupported WAV encoding", "ADPCM rejection");

            const auto path_flac = root / "actually.flac";
            {
                std::ofstream output(path_flac, std::ios::binary);
                write_bytes(output, "fLaC\0\0\0\x22\0\0\0\0", 12);
            }
            require_throws_containing(
                [&] { (void)engine::audio::read_wav_f32(path_flac); },
                "FLAC", "FLAC container identification");
        }

        std::cout << "wav_reader_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "wav_reader_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
