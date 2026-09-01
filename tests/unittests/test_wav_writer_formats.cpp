#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"

#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

std::filesystem::path scratch_dir() {
    const auto dir = std::filesystem::temp_directory_path() / "audiocpp_wav_writer_formats_test";
    std::filesystem::create_directories(dir);
    return dir;
}

std::vector<char> read_file_bytes(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("could not reopen written WAV: " + path.string());
    }
    return std::vector<char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

template <typename T>
T read_le(const std::vector<char> & bytes, size_t offset) {
    if (offset + sizeof(T) > bytes.size()) {
        throw std::runtime_error("WAV header is shorter than the field being read");
    }
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

std::string tag(const std::vector<char> & bytes, size_t offset) {
    if (offset + 4 > bytes.size()) {
        throw std::runtime_error("WAV header is shorter than the tag being read");
    }
    return std::string(bytes.data() + offset, 4);
}

// A mix of a tone and a ramp, so every code in the range is exercised rather
// than just the few a pure sine visits.
std::vector<float> make_test_signal(size_t frames, int channels) {
    std::vector<float> out(frames * static_cast<size_t>(channels), 0.0F);
    for (size_t frame = 0; frame < frames; ++frame) {
        const double t = static_cast<double>(frame) / static_cast<double>(frames);
        const double tone = 0.6 * std::sin(2.0 * kPi * 440.0 * static_cast<double>(frame) / 44100.0);
        const double ramp = 0.35 * (2.0 * t - 1.0);
        for (int channel = 0; channel < channels; ++channel) {
            const double sign = channel % 2 == 0 ? 1.0 : -1.0;
            out[frame * static_cast<size_t>(channels) + static_cast<size_t>(channel)] =
                static_cast<float>(sign * (tone + ramp));
        }
    }
    return out;
}

double max_abs_error(const std::vector<float> & a, const std::vector<float> & b) {
    engine::test::require_eq(a.size(), b.size(), "round trip sample count");
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return worst;
}

struct FormatExpectation {
    engine::audio::WavSampleFormat format;
    const char * label;
    uint16_t audio_format_tag;
    uint16_t bits_per_sample;
    uint32_t fmt_chunk_size;
    bool has_fact_chunk;
    // Error bound. For an integer format this is half an LSB of rounding plus
    // the scale disagreement between writer and reader: the writer maps 1.0 to
    // 32767 / 8388607 while the reader divides by 32768 / 8388608, which costs
    // a further |x| LSB. The worst case is therefore 1.5 LSB at full scale.
    double error_bound;
};

void check_format(const FormatExpectation & expectation, int sample_rate, int channels) {
    const auto dir = scratch_dir();
    const auto path = dir / (std::string("format_") + expectation.label + "_" +
                             std::to_string(channels) + "ch.wav");
    const size_t frames = 2048;
    const auto audio = make_test_signal(frames, channels);

    engine::audio::WavWriteOptions options;
    options.format = expectation.format;
    engine::audio::write_wav(path, sample_rate, channels, audio, options);

    const auto bytes = read_file_bytes(path);
    const std::string label = expectation.label;

    engine::test::require_eq(tag(bytes, 0), std::string("RIFF"), label + " RIFF tag");
    engine::test::require_eq(tag(bytes, 8), std::string("WAVE"), label + " WAVE tag");
    engine::test::require_eq(tag(bytes, 12), std::string("fmt "), label + " fmt tag");

    // RIFF size counts everything after the size field itself.
    engine::test::require_eq(
        static_cast<size_t>(read_le<uint32_t>(bytes, 4)) + 8U,
        bytes.size(),
        label + " RIFF chunk size");

    const uint32_t fmt_size = read_le<uint32_t>(bytes, 16);
    engine::test::require_eq(fmt_size, expectation.fmt_chunk_size, label + " fmt chunk size");
    engine::test::require_eq(
        read_le<uint16_t>(bytes, 20), expectation.audio_format_tag, label + " audio format tag");
    engine::test::require_eq(
        read_le<uint16_t>(bytes, 22), static_cast<uint16_t>(channels), label + " channel count");
    engine::test::require_eq(
        read_le<uint32_t>(bytes, 24), static_cast<uint32_t>(sample_rate), label + " sample rate");

    const uint32_t bytes_per_sample = expectation.bits_per_sample / 8U;
    const uint16_t expected_block_align = static_cast<uint16_t>(channels * bytes_per_sample);
    engine::test::require_eq(
        read_le<uint32_t>(bytes, 28),
        static_cast<uint32_t>(sample_rate) * expected_block_align,
        label + " byte rate");
    engine::test::require_eq(read_le<uint16_t>(bytes, 32), expected_block_align, label + " block align");
    engine::test::require_eq(
        read_le<uint16_t>(bytes, 34), expectation.bits_per_sample, label + " bits per sample");

    size_t cursor = 20 + fmt_size;
    if (expectation.has_fact_chunk) {
        // Non-PCM needs cbSize in fmt and a fact chunk carrying the frame count.
        engine::test::require_eq(read_le<uint16_t>(bytes, 36), static_cast<uint16_t>(0), label + " cbSize");
        engine::test::require_eq(tag(bytes, cursor), std::string("fact"), label + " fact tag");
        engine::test::require_eq(read_le<uint32_t>(bytes, cursor + 4), 4U, label + " fact chunk size");
        engine::test::require_eq(
            read_le<uint32_t>(bytes, cursor + 8), static_cast<uint32_t>(frames), label + " fact frame count");
        cursor += 12;
    }

    engine::test::require_eq(tag(bytes, cursor), std::string("data"), label + " data tag");
    const uint32_t data_bytes = read_le<uint32_t>(bytes, cursor + 4);
    engine::test::require_eq(
        static_cast<size_t>(data_bytes),
        audio.size() * bytes_per_sample,
        label + " data chunk size");
    engine::test::require_eq(bytes.size(), cursor + 8 + static_cast<size_t>(data_bytes), label + " file size");

    const auto decoded = engine::audio::read_wav_f32(path);
    engine::test::require_eq(decoded.sample_rate, sample_rate, label + " decoded sample rate");
    engine::test::require_eq(decoded.channels, channels, label + " decoded channel count");
    const double worst = max_abs_error(audio, decoded.samples);
    if (worst > expectation.error_bound) {
        throw std::runtime_error(
            label + " round trip max abs error " + std::to_string(worst) + " exceeds bound " +
            std::to_string(expectation.error_bound));
    }
    // Guard the bounds from the other side too: a 24-bit path that silently
    // truncated to 16 bits would still pass a loose upper bound.
    if (expectation.format == engine::audio::WavSampleFormat::Pcm24 && worst == 0.0) {
        throw std::runtime_error("pcm24 round trip was exact, which no 24-bit quantiser should be");
    }

    std::filesystem::remove(path);
}

void test_format_round_trips() {
    const FormatExpectation formats[] = {
        {engine::audio::WavSampleFormat::Pcm16, "pcm16", 1, 16, 16, false, 1.5 / 32768.0},
        {engine::audio::WavSampleFormat::Pcm24, "pcm24", 1, 24, 16, false, 1.5 / 8388608.0},
        {engine::audio::WavSampleFormat::Float32, "float32", 3, 32, 18, true, 0.0},
    };
    for (const auto & format : formats) {
        check_format(format, 44100, 1);
        check_format(format, 48000, 2);
    }
}

// 16-bit stays the default and stays byte-for-byte what it always was: 70 call
// sites depend on it.
void test_pcm16_default_is_unchanged() {
    const auto dir = scratch_dir();
    auto audio = make_test_signal(1024, 1);
    // Out-of-range samples so the clamp path is compared too.
    audio[10] = 1.9F;
    audio[11] = -2.4F;
    audio[12] = 1.0F;
    audio[13] = -1.0F;

    const auto legacy_path = dir / "default_via_pcm16_helper.wav";
    const auto explicit_path = dir / "default_via_write_wav.wav";
    engine::audio::write_pcm16_wav(legacy_path, 44100, 1, audio);
    engine::audio::write_wav(explicit_path, 44100, 1, audio, engine::audio::WavWriteOptions{});

    engine::test::require(
        read_file_bytes(legacy_path) == read_file_bytes(explicit_path),
        "write_pcm16_wav and default-option write_wav produced different bytes");

    // The canonical 44-byte header the previous writer emitted.
    const auto bytes = read_file_bytes(legacy_path);
    engine::test::require_eq(bytes.size(), 44U + audio.size() * 2U, "pcm16 file size");
    engine::test::require_eq(read_le<uint32_t>(bytes, 4), 36U + static_cast<uint32_t>(audio.size() * 2), "pcm16 RIFF size");

    const auto decoded = engine::audio::read_wav_f32(legacy_path);
    engine::test::require_close(decoded.samples[12], 1.0F, 4.6e-5F, "pcm16 clamps +1.0");
    engine::test::require_close(decoded.samples[10], 1.0F, 4.6e-5F, "pcm16 clamps above +1.0");
    engine::test::require_close(decoded.samples[11], -1.0F, 4.6e-5F, "pcm16 clamps below -1.0");

    std::filesystem::remove(legacy_path);
    std::filesystem::remove(explicit_path);
}

// F4.3. Dither is opt-in, deterministic, and bounded to the LSB it is supposed
// to live in. Applying it twice in a chain is the failure mode to avoid, which
// is why the default stays None.
void test_dither_is_opt_in_and_bounded() {
    const auto dir = scratch_dir();
    const auto plain_path = dir / "dither_off.wav";
    const auto dithered_path = dir / "dither_on.wav";
    const auto repeat_path = dir / "dither_on_repeat.wav";

    // -60 dBFS tone: quiet enough that the quantiser is the dominant error.
    std::vector<float> quiet(8192, 0.0F);
    for (size_t i = 0; i < quiet.size(); ++i) {
        quiet[i] = static_cast<float>(
            0.001 * std::sin(2.0 * kPi * 997.0 * static_cast<double>(i) / 44100.0));
    }

    engine::audio::WavWriteOptions plain;
    engine::audio::WavWriteOptions dithered;
    dithered.dither = engine::audio::WavDitherMode::TriangularPdf;

    engine::audio::write_wav(plain_path, 44100, 1, quiet, plain);
    engine::audio::write_wav(dithered_path, 44100, 1, quiet, dithered);
    engine::audio::write_wav(repeat_path, 44100, 1, quiet, dithered);

    engine::test::require(
        read_file_bytes(plain_path) != read_file_bytes(dithered_path),
        "TPDF dither did not change the written samples");
    engine::test::require(
        read_file_bytes(dithered_path) == read_file_bytes(repeat_path),
        "TPDF dither is not reproducible for a fixed seed");

    const auto plain_samples = engine::audio::read_wav_f32(plain_path).samples;
    const auto dithered_samples = engine::audio::read_wav_f32(dithered_path).samples;
    // TPDF at 2 LSB peak to peak can move a code by at most 1 either way, so
    // with rounding the total displacement never exceeds 2 LSB.
    const double worst = max_abs_error(plain_samples, dithered_samples);
    if (worst > 2.0 / 32768.0 + 1e-9) {
        throw std::runtime_error(
            "TPDF dither displaced a sample by " + std::to_string(worst * 32768.0) + " LSB");
    }
    engine::test::require(worst > 0.0, "TPDF dither displaced nothing at all");

    std::filesystem::remove(plain_path);
    std::filesystem::remove(dithered_path);
    std::filesystem::remove(repeat_path);
}

// F4.5. The limiter is opt-in, leaves in-range material untouched, and keeps
// overshoot off the rail instead of clamping it there.
void test_peak_policy() {
    const auto dir = scratch_dir();
    const size_t count = 8192;

    // A tone peaking 1 dB over full scale.
    std::vector<float> hot(count, 0.0F);
    for (size_t i = 0; i < count; ++i) {
        hot[i] = static_cast<float>(
            std::pow(10.0, 1.0 / 20.0) * std::sin(2.0 * kPi * 997.0 * static_cast<double>(i) / 44100.0));
    }

    const auto clipped_path = dir / "peak_hard_clip.wav";
    const auto limited_path = dir / "peak_limited.wav";
    engine::audio::WavWriteOptions clip_options;
    engine::audio::WavWriteOptions limit_options;
    limit_options.peak_policy = engine::audio::WavPeakPolicy::LookaheadLimit;
    engine::audio::write_wav(clipped_path, 44100, 1, hot, clip_options);
    engine::audio::write_wav(limited_path, 44100, 1, hot, limit_options);

    const auto clipped = engine::audio::read_wav_f32(clipped_path).samples;
    const auto limited = engine::audio::read_wav_f32(limited_path).samples;

    const auto count_on_rail = [](const std::vector<float> & samples) {
        size_t railed = 0;
        for (const float sample : samples) {
            if (std::abs(sample) >= 32766.0F / 32768.0F) {
                ++railed;
            }
        }
        return railed;
    };
    engine::test::require(
        count_on_rail(clipped) > count / 4,
        "hard clip did not put the expected share of a +1 dBFS tone on the rail");
    engine::test::require_eq(count_on_rail(limited), static_cast<size_t>(0), "limited samples on the rail");

    // Material already inside the ceiling must come back bit-identical, so
    // turning the limiter on is safe for the overwhelming majority of renders.
    const auto quiet_clip_path = dir / "peak_quiet_clip.wav";
    const auto quiet_limit_path = dir / "peak_quiet_limit.wav";
    const auto quiet = make_test_signal(count, 1);
    engine::audio::write_wav(quiet_clip_path, 44100, 1, quiet, clip_options);
    engine::audio::write_wav(quiet_limit_path, 44100, 1, quiet, limit_options);
    engine::test::require(
        read_file_bytes(quiet_clip_path) == read_file_bytes(quiet_limit_path),
        "the limiter altered a signal that never reached the ceiling");

    // And the standalone helper reports what it did.
    auto scratch = hot;
    const float reduction = engine::audio::apply_lookahead_limiter_in_place(scratch, 1, 44100);
    engine::test::require(
        reduction > 0.5F && reduction < 3.0F,
        "limiter reported " + std::to_string(reduction) + " dB of reduction for a +1 dBFS tone");
    float peak = 0.0F;
    for (const float sample : scratch) {
        peak = std::max(peak, std::abs(sample));
    }
    engine::test::require(peak <= 1.0F, "limiter left a sample above full scale");

    auto untouched = make_test_signal(count, 2);
    const auto before = untouched;
    engine::test::require_eq(
        engine::audio::apply_lookahead_limiter_in_place(untouched, 2, 48000),
        0.0F,
        "limiter reduction on in-range audio");
    engine::test::require(untouched == before, "limiter modified in-range audio");

    std::filesystem::remove(clipped_path);
    std::filesystem::remove(limited_path);
    std::filesystem::remove(quiet_clip_path);
    std::filesystem::remove(quiet_limit_path);
}

void test_format_metadata_helpers() {
    engine::test::require_eq(
        engine::audio::wav_sample_format_bit_depth(engine::audio::WavSampleFormat::Pcm16), 16, "pcm16 depth");
    engine::test::require_eq(
        engine::audio::wav_sample_format_bit_depth(engine::audio::WavSampleFormat::Pcm24), 24, "pcm24 depth");
    engine::test::require_eq(
        engine::audio::wav_sample_format_bit_depth(engine::audio::WavSampleFormat::Float32), 32, "float32 depth");
    engine::test::require_eq(
        std::string(engine::audio::wav_sample_format_name(engine::audio::WavSampleFormat::Pcm24)),
        std::string("pcm24"),
        "pcm24 name");
}

}  // namespace

int main() {
    try {
        test_format_round_trips();
        test_pcm16_default_is_unchanged();
        test_dither_is_opt_in_and_bounded();
        test_peak_policy();
        test_format_metadata_helpers();
        std::cout << "wav_writer_formats_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "wav_writer_formats_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
