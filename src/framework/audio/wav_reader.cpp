#include "engine/framework/audio/wav_reader.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <streambuf>

namespace engine::audio {
namespace {

class ReadOnlyMemoryStreamBuffer final : public std::streambuf {
public:
    explicit ReadOnlyMemoryStreamBuffer(std::string_view data) {
        // std::streambuf::setg takes char*, but this input stream only reads from the buffer.
        auto * begin = const_cast<char *>(data.data());
        setg(begin, begin, begin + data.size());
    }

protected:
    pos_type seekoff(off_type offset, std::ios_base::seekdir dir, std::ios_base::openmode which) override {
        if ((which & std::ios_base::in) == 0) {
            return pos_type(off_type(-1));
        }
        char * base = eback();
        char * next = gptr();
        char * end = egptr();
        char * target = nullptr;
        if (dir == std::ios_base::beg) {
            target = base + offset;
        } else if (dir == std::ios_base::cur) {
            target = next + offset;
        } else if (dir == std::ios_base::end) {
            target = end + offset;
        }
        if (target == nullptr || target < base || target > end) {
            return pos_type(off_type(-1));
        }
        setg(base, target, end);
        return pos_type(target - base);
    }

    pos_type seekpos(pos_type position, std::ios_base::openmode which) override {
        return seekoff(off_type(position), std::ios_base::beg, which);
    }
};

template <typename T>
T read_scalar(std::istream & input) {
    T value{};
    input.read(reinterpret_cast<char *>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error("failed to read WAV scalar");
    }
    return value;
}

void skip_bytes(std::istream & input, std::streamoff count) {
    input.seekg(count, std::ios::cur);
    if (!input) {
        throw std::runtime_error("failed to seek inside WAV file");
    }
}

}  // namespace

WavData read_wav_f32(std::istream & input) {
    if (!input) {
        throw std::runtime_error("could not open WAV input");
    }

    char riff[4];
    input.read(riff, 4);
    if (!input || std::string(riff, 4) != "RIFF") {
        throw std::runtime_error("invalid WAV RIFF header");
    }
    skip_bytes(input, 4);
    char wave[4];
    input.read(wave, 4);
    if (!input || std::string(wave, 4) != "WAVE") {
        throw std::runtime_error("invalid WAV WAVE header");
    }

    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    std::vector<char> data;

    while (input) {
        char chunk_id[4];
        input.read(chunk_id, 4);
        if (!input) {
            break;
        }
        const uint32_t chunk_size = read_scalar<uint32_t>(input);
        const std::string id(chunk_id, 4);
        if (id == "fmt ") {
            audio_format = read_scalar<uint16_t>(input);
            channels = read_scalar<uint16_t>(input);
            sample_rate = read_scalar<uint32_t>(input);
            skip_bytes(input, 6);
            bits_per_sample = read_scalar<uint16_t>(input);
            if (chunk_size > 16) {
                skip_bytes(input, static_cast<std::streamoff>(chunk_size - 16));
            }
        } else if (id == "data") {
            data.resize(chunk_size);
            input.read(data.data(), static_cast<std::streamsize>(chunk_size));
            if (!input) {
                throw std::runtime_error("failed to read WAV data chunk");
            }
        } else {
            skip_bytes(input, chunk_size);
        }
        if (chunk_size % 2 == 1) {
            skip_bytes(input, 1);
        }
    }

    if (channels == 0 || sample_rate == 0 || bits_per_sample == 0 || data.empty()) {
        throw std::runtime_error("incomplete WAV file");
    }

    WavData wav;
    wav.sample_rate = static_cast<int>(sample_rate);
    wav.channels = static_cast<int>(channels);

    if (audio_format == 1 && bits_per_sample == 16) {
        const size_t sample_count = data.size() / sizeof(int16_t);
        wav.samples.resize(sample_count);
        const auto * pcm = reinterpret_cast<const int16_t *>(data.data());
        for (size_t i = 0; i < sample_count; ++i) {
            wav.samples[i] = static_cast<float>(pcm[i]) / 32768.0F;
        }
        return wav;
    }

    if (audio_format == 1 && bits_per_sample == 24) {
        if (data.size() % 3 != 0) {
            throw std::runtime_error("malformed PCM24 WAV data chunk");
        }
        const size_t sample_count = data.size() / 3;
        wav.samples.resize(sample_count);
        const auto * pcm = reinterpret_cast<const uint8_t *>(data.data());
        for (size_t i = 0; i < sample_count; ++i) {
            const size_t offset = i * 3;
            int32_t value =
                static_cast<int32_t>(pcm[offset]) |
                (static_cast<int32_t>(pcm[offset + 1]) << 8) |
                (static_cast<int32_t>(pcm[offset + 2]) << 16);
            if ((value & 0x00800000) != 0) {
                value |= ~0x00FFFFFF;
            }
            wav.samples[i] = static_cast<float>(value) / 8388608.0F;
        }
        return wav;
    }

    if (audio_format == 3 && bits_per_sample == 32) {
        const size_t sample_count = data.size() / sizeof(float);
        wav.samples.resize(sample_count);
        const auto * pcm = reinterpret_cast<const float *>(data.data());
        for (size_t i = 0; i < sample_count; ++i) {
            wav.samples[i] = pcm[i];
        }
        return wav;
    }

    throw std::runtime_error("unsupported WAV encoding (need PCM16, PCM24, or float32)");
}

WavData read_wav_f32(std::string_view input) {
    ReadOnlyMemoryStreamBuffer buffer(input);
    std::istream stream(&buffer);
    return read_wav_f32(stream);
}

WavData read_wav_f32(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open WAV input: " + path.string());
    }

    return read_wav_f32(input);
}

}  // namespace engine::audio
