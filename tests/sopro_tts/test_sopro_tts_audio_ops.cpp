// Host-side checks for the sopro reference level chain and the ISTFT head's
// band limit. Both are pure functions over plain buffers, so none of this needs
// the checkpoint; the weight-bound stages are covered by sopro_probe instead.
#include "engine/community_models/sopro_tts/assets.h"
#include "engine/community_models/sopro_tts/reference.h"
#include "engine/community_models/sopro_tts/vocoder.h"
#include "test_assert.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

namespace test = engine::test;
namespace sopro = engine::community_models::sopro_tts;
namespace audio_ops = sopro::audio_ops;

constexpr int kSampleRate = 24000;

// A constant-amplitude buffer makes speech_level_db exact: every 25 ms frame
// has the same RMS, so the 0.2-quantile gate keeps all of them and the median
// is the amplitude itself. Peak and speech level therefore agree, which keeps
// the expected values below plain arithmetic.
std::vector<float> flat(float amplitude, float seconds = 1.0F) {
    const auto samples = static_cast<size_t>(static_cast<float>(kSampleRate) * seconds);
    return std::vector<float>(samples, amplitude);
}

float peak_of(const std::vector<float> & wav) {
    float peak = 0.0F;
    for (const float value : wav) {
        peak = std::max(peak, std::fabs(value));
    }
    return peak;
}

void test_speech_level_db_reads_a_flat_buffer() {
    const auto level = audio_ops::speech_level_db(flat(0.1F), kSampleRate);
    test::require_close(level.level_db, -20.0F, 1.0e-3F, "flat 0.1 level");
    // 98 frames at a 10 ms hop; all of them survive the activity gate.
    test::require_close(level.active_seconds, 0.98F, 1.0e-3F, "flat 0.1 active seconds");

    // Shorter than one 25 ms window: whole-buffer RMS, and no active span.
    const auto tiny = audio_ops::speech_level_db(std::vector<float>(100, 0.1F), kSampleRate);
    test::require_close(tiny.level_db, -20.0F, 1.0e-3F, "short buffer level");
    test::require_close(tiny.active_seconds, 0.0F, 1.0e-6F, "short buffer active seconds");

    // Silence floors at 1e-6 rather than diverging.
    const auto silent = audio_ops::speech_level_db(std::vector<float>(100, 0.0F), kSampleRate);
    test::require_close(silent.level_db, -120.0F, 1.0e-3F, "silence level");
}

void test_quiet_reference_is_boosted_to_the_prompt_level() {
    const auto input = flat(0.01F);  // -40 dB, 20.2 dB below the prompt level
    const auto out = audio_ops::normalize_reference(input, kSampleRate);

    test::require_close(out.level_db, audio_ops::kPromptLevelDb, 1.0e-3F, "boosted level");
    const float expected_gain = std::pow(10.0F, 20.2F / 20.0F);
    test::require_close(out.wav.front(), 0.01F * expected_gain, 1.0e-6F, "boosted sample");
    // Well clear of the 0.95 ceiling, so the peak guard must not have bitten.
    test::require(peak_of(out.wav) < 0.95F, "boosted peak stays below the ceiling");
    test::require_eq(out.wav.size(), input.size(), "boosted length");
}

void test_hot_reference_is_left_alone() {
    // -6.02 dB, well above the prompt level. The pre-2.1 rule attenuated this
    // by 13.78 dB; boost-only must pass it through untouched.
    const auto input = flat(0.5F);
    const auto out = audio_ops::normalize_reference(input, kSampleRate);

    test::require_close(out.level_db, -6.0206F, 1.0e-3F, "hot level is unchanged");
    for (size_t i = 0; i < input.size(); i += 997) {
        test::require_eq(out.wav[i], input[i], "hot sample is unchanged");
    }
}

void test_peak_guard_caps_the_boost() {
    // An impulse train with a high crest factor: every 25 ms window holds
    // exactly six 0.9 spikes, so the frame RMS is a uniform 0.09 (-20.92 dB)
    // and the activity gate keeps all of it. A lone spike would not do — the
    // gate would keep only the frames containing it and read the level off
    // those. The buffer wants +1.12 dB but the peak leaves only +0.47 dB.
    std::vector<float> input(static_cast<size_t>(kSampleRate), 0.0F);
    for (size_t i = 0; i < input.size(); i += 100) {
        input[i] = 0.9F;
    }
    const auto out = audio_ops::normalize_reference(input, kSampleRate);

    const float capped_gain = 20.0F * std::log10(0.95F / 0.9F);
    test::require_close(out.level_db, -20.9151F + capped_gain, 1.0e-2F, "peak-guarded level");
    test::require_close(peak_of(out.wav), 0.95F, 1.0e-4F, "peak lands on the ceiling");
    test::require(out.level_db < audio_ops::kPromptLevelDb, "peak guard undershoots the target");
}

void test_boost_is_limited_to_thirty_db() {
    // -60 dB with 59.5 dB of peak headroom, so the 30 dB gain limit is what
    // binds rather than the ceiling.
    const auto out = audio_ops::normalize_reference(flat(0.001F), kSampleRate);
    test::require_close(out.level_db, -30.0F, 1.0e-2F, "gain-limited level");
}

void test_output_gain_tracks_the_reference_level() {
    test::require_close(
        audio_ops::output_gain(), std::pow(10.0F, -3.2F / 20.0F), 1.0e-6F, "default output gain");
    // A hotter reference has to be pulled down further to reach -23 dB.
    test::require(
        audio_ops::output_gain(-11.24F) < audio_ops::output_gain(),
        "a hot reference gets a smaller output gain");
    test::require_close(
        audio_ops::output_gain(-11.24F), std::pow(10.0F, -11.76F / 20.0F), 1.0e-6F,
        "hot reference output gain");
}

void test_match_gain_falls_back_to_the_reference_level() {
    // Under kMinActiveSeconds of measurable speech, so match_gain cannot level
    // off the audio itself and defers to the reference it was cloned from.
    const std::vector<float> too_short(100, 0.1F);
    test::require_close(
        audio_ops::match_gain(too_short, kSampleRate, audio_ops::kOutputLevelDb, -11.24F),
        audio_ops::output_gain(-11.24F), 1.0e-6F, "fallback uses the reference level");
    test::require(
        audio_ops::match_gain(too_short, kSampleRate, audio_ops::kOutputLevelDb, -11.24F) !=
            audio_ops::match_gain(too_short, kSampleRate),
        "fallback varies with the reference level");

    // With enough speech to measure, the reference level is irrelevant: the
    // gain comes from the audio actually produced.
    const auto measurable = flat(0.1F);  // -20 dB, 0.98 s active
    const float expected = std::pow(10.0F, -3.0F / 20.0F);
    test::require_close(
        audio_ops::match_gain(measurable, kSampleRate), expected, 1.0e-4F, "measured gain");
    test::require_close(
        audio_ops::match_gain(measurable, kSampleRate, audio_ops::kOutputLevelDb, -11.24F),
        expected, 1.0e-4F, "measured gain ignores the reference level");
}

void test_band_limit_bin() {
    sopro::SoproVocoderConfig config;  // 24 kHz, n_fft 1024, 10900 Hz
    // ceil(10900 * 1024 / 24000) = 466 of 513 bins, i.e. a 10921.9 Hz cut.
    test::require_eq(sopro::band_limit_bin(config), int64_t{466}, "default cut");

    config.band_limit_hz = 0.0F;
    test::require_eq(sopro::band_limit_bin(config), int64_t{513}, "zero keeps every bin");
    config.band_limit_hz = -1.0F;
    test::require_eq(sopro::band_limit_bin(config), int64_t{513}, "negative keeps every bin");

    // Nyquist itself is bin 512, and the cut is inclusive, so a 12 kHz limit
    // still drops that last bin — which is the one the unlimited head used to
    // synthesise with a bogus imaginary part.
    config.band_limit_hz = 12000.0F;
    test::require_eq(sopro::band_limit_bin(config), int64_t{512}, "Nyquist cut");
    config.band_limit_hz = 48000.0F;
    test::require_eq(sopro::band_limit_bin(config), int64_t{513}, "above Nyquist clamps");

    // The cut follows the transform size, not a hardcoded bin index.
    config.band_limit_hz = 10900.0F;
    config.n_fft = 2048;
    test::require_eq(sopro::band_limit_bin(config), int64_t{931}, "n_fft 2048 cut");
}

}  // namespace

int main() {
    try {
        test_speech_level_db_reads_a_flat_buffer();
        test_quiet_reference_is_boosted_to_the_prompt_level();
        test_hot_reference_is_left_alone();
        test_peak_guard_caps_the_boost();
        test_boost_is_limited_to_thirty_db();
        test_output_gain_tracks_the_reference_level();
        test_match_gain_falls_back_to_the_reference_level();
        test_band_limit_bin();
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << "PASS: sopro_tts audio ops checks\n";
    return 0;
}
