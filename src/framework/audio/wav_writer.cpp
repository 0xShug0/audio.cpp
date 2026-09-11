#include "engine/framework/audio/wav_writer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace engine::audio {
namespace {

constexpr uint16_t kFormatPcm = 1;
constexpr uint16_t kFormatFloat = 3;

// Deterministic dither source. splitmix64 is stateless apart from a 64-bit
// counter, which keeps the writer reentrant and the output reproducible.
class DitherGenerator {
public:
    explicit DitherGenerator(uint64_t seed)
        : state_(seed == 0 ? 0x9E3779B97F4A7C15ULL : seed) {}

    // Triangular PDF, 2 LSB peak to peak, in quantiser code units.
    float next_tpdf() {
        return next_unit() - next_unit();
    }

private:
    float next_unit() {
        state_ += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state_;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        return static_cast<float>(static_cast<double>(z >> 40) / 16777216.0);
    }

    uint64_t state_;
};

// Sliding-window minimum over [i - radius, i + radius], O(n) via a monotonic
// deque. This is the lookahead: the gain envelope has already reached its
// minimum by the time the peak that demanded it arrives.
std::vector<float> sliding_window_minimum(const std::vector<float> & values, int64_t radius) {
    const int64_t count = static_cast<int64_t>(values.size());
    std::vector<float> out(values.size(), 1.0F);
    std::deque<int64_t> window;
    int64_t next = 0;
    for (int64_t i = 0; i < count; ++i) {
        const int64_t limit = std::min(count - 1, i + radius);
        while (next <= limit) {
            while (!window.empty() && values[static_cast<size_t>(window.back())] >= values[static_cast<size_t>(next)]) {
                window.pop_back();
            }
            window.push_back(next);
            ++next;
        }
        while (!window.empty() && window.front() < i - radius) {
            window.pop_front();
        }
        out[static_cast<size_t>(i)] = values[static_cast<size_t>(window.front())];
    }
    return out;
}

// Box filter over [i - radius, i + radius]. Applied twice below, so the gain
// envelope is a triangular-kernel smooth of the running minimum rather than a
// staircase. Prefix sums keep it O(n).
std::vector<float> box_smooth(const std::vector<float> & values, int64_t radius) {
    if (radius <= 0) {
        return values;
    }
    const int64_t count = static_cast<int64_t>(values.size());
    std::vector<double> prefix(static_cast<size_t>(count) + 1, 0.0);
    for (int64_t i = 0; i < count; ++i) {
        prefix[static_cast<size_t>(i) + 1] = prefix[static_cast<size_t>(i)] + values[static_cast<size_t>(i)];
    }
    std::vector<float> out(values.size(), 0.0F);
    for (int64_t i = 0; i < count; ++i) {
        const int64_t begin = std::max<int64_t>(0, i - radius);
        const int64_t end = std::min<int64_t>(count, i + radius + 1);
        const double sum = prefix[static_cast<size_t>(end)] - prefix[static_cast<size_t>(begin)];
        out[static_cast<size_t>(i)] = static_cast<float>(sum / static_cast<double>(end - begin));
    }
    return out;
}

template <typename T>
void write_scalar(std::ofstream & out, const T & value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

void write_pcm24_sample(std::ofstream & out, int32_t value) {
    const char bytes[3] = {
        static_cast<char>(static_cast<uint8_t>(value & 0xFF)),
        static_cast<char>(static_cast<uint8_t>((value >> 8) & 0xFF)),
        static_cast<char>(static_cast<uint8_t>((value >> 16) & 0xFF)),
    };
    out.write(bytes, 3);
}

}  // namespace

float apply_lookahead_limiter_in_place(
    std::vector<float> & samples,
    int channel_count,
    int sample_rate,
    const LookaheadLimiterOptions & options) {
    if (channel_count <= 0) {
        throw std::runtime_error("limiter channel count must be positive");
    }
    if (sample_rate <= 0) {
        throw std::runtime_error("limiter sample rate must be positive");
    }
    if (!(options.ceiling > 0.0F)) {
        throw std::runtime_error("limiter ceiling must be positive");
    }
    if (samples.size() % static_cast<size_t>(channel_count) != 0) {
        throw std::runtime_error("limiter sample count must be divisible by channel count");
    }
    const int64_t frames = static_cast<int64_t>(samples.size() / static_cast<size_t>(channel_count));
    if (frames == 0) {
        return 0.0F;
    }

    // Gain is linked across channels from the per-frame peak, so a limited
    // stereo pair keeps its image instead of one side ducking alone.
    std::vector<float> required(static_cast<size_t>(frames), 1.0F);
    bool any_over = false;
    for (int64_t frame = 0; frame < frames; ++frame) {
        float peak = 0.0F;
        const size_t base = static_cast<size_t>(frame) * static_cast<size_t>(channel_count);
        for (int channel = 0; channel < channel_count; ++channel) {
            peak = std::max(peak, std::fabs(samples[base + static_cast<size_t>(channel)]));
        }
        if (peak > options.ceiling) {
            required[static_cast<size_t>(frame)] = options.ceiling / peak;
            any_over = true;
        }
    }
    if (!any_over) {
        // Nothing exceeded the ceiling, so leave the buffer bit-identical.
        return 0.0F;
    }

    const int64_t radius = std::max<int64_t>(
        1,
        static_cast<int64_t>(std::llround(options.window_seconds * static_cast<double>(sample_rate))));
    const int64_t smooth_radius = std::max<int64_t>(1, radius / 2);
    std::vector<float> gain =
        box_smooth(box_smooth(sliding_window_minimum(required, radius), smooth_radius), smooth_radius);

    float min_gain = 1.0F;
    for (int64_t frame = 0; frame < frames; ++frame) {
        // The smoothing can nudge the envelope back above what this frame
        // actually needs; the ceiling is a hard promise, so take the lower.
        float g = std::min(gain[static_cast<size_t>(frame)], required[static_cast<size_t>(frame)]);
        g = std::clamp(g, 0.0F, 1.0F);
        min_gain = std::min(min_gain, g);
        const size_t base = static_cast<size_t>(frame) * static_cast<size_t>(channel_count);
        for (int channel = 0; channel < channel_count; ++channel) {
            samples[base + static_cast<size_t>(channel)] *= g;
        }
    }
    if (!(min_gain > 0.0F)) {
        return std::numeric_limits<float>::infinity();
    }
    return -20.0F * std::log10(min_gain);
}

int wav_sample_format_bit_depth(WavSampleFormat format) {
    switch (format) {
        case WavSampleFormat::Pcm16:
            return 16;
        case WavSampleFormat::Pcm24:
            return 24;
        case WavSampleFormat::Float32:
            return 32;
    }
    throw std::runtime_error("unknown WAV sample format");
}

const char * wav_sample_format_name(WavSampleFormat format) {
    switch (format) {
        case WavSampleFormat::Pcm16:
            return "pcm16";
        case WavSampleFormat::Pcm24:
            return "pcm24";
        case WavSampleFormat::Float32:
            return "float32";
    }
    throw std::runtime_error("unknown WAV sample format");
}

void write_wav(
    const std::filesystem::path & path,
    int sample_rate,
    int channel_count,
    const std::vector<float> & audio,
    const WavWriteOptions & options) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("could not open WAV output: " + path.string());
    }
    if (sample_rate <= 0) {
        throw std::runtime_error("sample rate must be positive");
    }
    if (channel_count <= 0) {
        throw std::runtime_error("channel count must be positive");
    }
    if (audio.size() % static_cast<size_t>(channel_count) != 0) {
        throw std::runtime_error("audio sample count must be divisible by channel count");
    }

    const uint16_t channels = static_cast<uint16_t>(channel_count);
    const uint16_t bits_per_sample = static_cast<uint16_t>(wav_sample_format_bit_depth(options.format));
    const uint32_t bytes_per_sample = bits_per_sample / 8U;

    // RIFF sizes are 32-bit. At 48 kHz stereo 16-bit that is about 6.2 hours
    // before the header silently wraps and the file decodes as a fraction of
    // its real length; refuse instead.
    const uint64_t data_bytes64 = static_cast<uint64_t>(audio.size()) * static_cast<uint64_t>(bytes_per_sample);
    constexpr uint64_t kMaxDataBytes = 0xFFFFFFFFULL - 64ULL;
    if (data_bytes64 > kMaxDataBytes) {
        throw std::runtime_error(
            "WAV data chunk exceeds the 4 GiB RIFF limit (" + std::to_string(data_bytes64) + " bytes)");
    }

    const uint32_t data_bytes = static_cast<uint32_t>(data_bytes64);
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * channels * bytes_per_sample;
    const uint16_t block_align = static_cast<uint16_t>(channels * bytes_per_sample);
    const bool is_float = options.format == WavSampleFormat::Float32;
    // Non-PCM formats need a WAVEFORMATEX cbSize field and a fact chunk to be
    // read by strict decoders. The in-tree reader tolerates either, but files
    // written here also leave the process.
    const uint32_t fmt_size = is_float ? 18U : 16U;
    const uint32_t fact_bytes = is_float ? 12U : 0U;
    const uint32_t riff_size = 4U + (8U + fmt_size) + fact_bytes + (8U + data_bytes);

    out.write("RIFF", 4);
    write_scalar(out, riff_size);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    write_scalar(out, fmt_size);
    const uint16_t audio_format = is_float ? kFormatFloat : kFormatPcm;
    write_scalar(out, audio_format);
    write_scalar(out, channels);
    write_scalar(out, static_cast<uint32_t>(sample_rate));
    write_scalar(out, byte_rate);
    write_scalar(out, block_align);
    write_scalar(out, bits_per_sample);
    if (is_float) {
        const uint16_t cb_size = 0;
        write_scalar(out, cb_size);
        out.write("fact", 4);
        const uint32_t fact_size = 4;
        write_scalar(out, fact_size);
        const uint32_t frame_count = static_cast<uint32_t>(audio.size() / static_cast<size_t>(channel_count));
        write_scalar(out, frame_count);
    }
    out.write("data", 4);
    write_scalar(out, data_bytes);

    // The limiter needs the whole buffer to look ahead, so it runs once here
    // rather than per sample. When nothing exceeds the ceiling it returns 0 dB
    // and the copy is bit-identical to the input.
    std::vector<float> limited;
    if (options.peak_policy == WavPeakPolicy::LookaheadLimit) {
        limited = audio;
        apply_lookahead_limiter_in_place(limited, channel_count, sample_rate, options.limiter);
    }
    const std::vector<float> & source = options.peak_policy == WavPeakPolicy::LookaheadLimit ? limited : audio;

    // Float32 is the format whose point is that it has no ceiling, so the hard
    // clamp does not apply to it; an explicit limiter still does.
    const bool clamp_to_unit = !is_float;
    const bool dither = options.dither == WavDitherMode::TriangularPdf && !is_float;
    DitherGenerator generator(options.dither_seed);

    for (float sample : source) {
        if (clamp_to_unit) {
            sample = std::max(-1.0F, std::min(1.0F, sample));
        }
        switch (options.format) {
            case WavSampleFormat::Pcm16: {
                float scaled = sample * 32767.0F;
                if (dither) {
                    scaled += generator.next_tpdf();
                }
                const long rounded = std::lrint(scaled);
                const auto pcm = static_cast<int16_t>(std::clamp<long>(rounded, -32768L, 32767L));
                write_scalar(out, pcm);
                break;
            }
            case WavSampleFormat::Pcm24: {
                float scaled = sample * 8388607.0F;
                if (dither) {
                    scaled += generator.next_tpdf();
                }
                const long rounded = std::lrint(scaled);
                write_pcm24_sample(out, static_cast<int32_t>(std::clamp<long>(rounded, -8388608L, 8388607L)));
                break;
            }
            case WavSampleFormat::Float32: {
                write_scalar(out, sample);
                break;
            }
        }
    }
    if (!out) {
        throw std::runtime_error("failed to write WAV output: " + path.string());
    }
}

void write_pcm16_wav(
    const std::filesystem::path & path,
    int sample_rate,
    int channel_count,
    const std::vector<float> & audio) {
    write_wav(path, sample_rate, channel_count, audio, WavWriteOptions{});
}

}  // namespace engine::audio
