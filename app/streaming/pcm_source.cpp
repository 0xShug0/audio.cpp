#include "pcm_source.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <cstdio>
#endif

namespace minitts::app {
namespace {

// Decoded little-endian so a stream produced on one machine reads back the same on any host.
float decode_s16le(const unsigned char * bytes) {
    const auto raw = static_cast<uint16_t>(
        static_cast<uint16_t>(bytes[0]) | (static_cast<uint16_t>(bytes[1]) << 8));
    // Matches the scaling engine::audio::read_wav_f32 applies, so raw PCM and WAV input of the
    // same audio decode to identical samples.
    return static_cast<float>(static_cast<int16_t>(raw)) / 32768.0F;
}

float decode_f32le(const unsigned char * bytes) {
    const auto raw = static_cast<uint32_t>(
        static_cast<uint32_t>(bytes[0]) |
        (static_cast<uint32_t>(bytes[1]) << 8) |
        (static_cast<uint32_t>(bytes[2]) << 16) |
        (static_cast<uint32_t>(bytes[3]) << 24));
    float value = 0.0F;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

int pcm_sample_format_bytes(PcmSampleFormat format) {
    return format == PcmSampleFormat::S16LE ? 2 : 4;
}

// Required on Windows, where stdin's default text mode mangles PCM bytes; a no-op elsewhere.
void set_stdin_binary_mode() {
#ifdef _WIN32
    if (_setmode(_fileno(stdin), _O_BINARY) == -1) {
        throw std::runtime_error("failed to switch stdin to binary mode");
    }
#endif
}

}  // namespace

PcmSampleFormat parse_pcm_sample_format(std::string_view name) {
    if (name == "s16le") return PcmSampleFormat::S16LE;
    if (name == "f32le") return PcmSampleFormat::F32LE;
    throw std::runtime_error(
        "unsupported raw PCM sample format '" + std::string(name) + "' (expected s16le or f32le)");
}

std::string to_string(PcmSampleFormat format) {
    return format == PcmSampleFormat::S16LE ? "s16le" : "f32le";
}

AudioChunkStream make_pcm_chunk_stream(
    std::istream & input,
    AudioStreamFormat format,
    PcmSampleFormat sample_format) {
    if (format.sample_rate <= 0 || format.channels <= 0) {
        throw std::runtime_error("raw PCM input requires a positive sample rate and channel count");
    }

    const int sample_bytes = pcm_sample_format_bytes(sample_format);
    const size_t frame_bytes = static_cast<size_t>(sample_bytes) * static_cast<size_t>(format.channels);
    const auto decode = sample_format == PcmSampleFormat::S16LE ? &decode_s16le : &decode_f32le;

    AudioChunkStream stream;
    stream.format = format;
    stream.read = [source = &input, decode, sample_bytes, frame_bytes, bytes = std::vector<char>{}](
        int64_t max_samples, std::vector<float> & samples) mutable {
        if (max_samples <= 0) {
            throw std::runtime_error("raw PCM read requires a positive sample count");
        }
        bytes.resize(static_cast<size_t>(max_samples) * static_cast<size_t>(sample_bytes));
        source->read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        const auto received = static_cast<size_t>(source->gcount());

        // A short read means end of input; drop a trailing partial frame, which cannot be
        // interpreted, so the chunk always holds whole frames.
        const size_t usable = (received / frame_bytes) * frame_bytes;
        const size_t count = usable / static_cast<size_t>(sample_bytes);
        samples.resize(count);
        const auto * raw = reinterpret_cast<const unsigned char *>(bytes.data());
        for (size_t i = 0; i < count; ++i) {
            samples[i] = decode(raw + i * static_cast<size_t>(sample_bytes));
        }
        return received == bytes.size();
    };
    return stream;
}

AudioChunkStream make_stdin_pcm_stream(AudioStreamFormat format, PcmSampleFormat sample_format) {
    set_stdin_binary_mode();
    return make_pcm_chunk_stream(std::cin, format, sample_format);
}

}  // namespace minitts::app
