#include "engine/framework/audio/chunking.h"

#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using engine::audio::AudioChunkJoinMode;
using engine::audio::AudioChunkJoinSpec;
using engine::runtime::AudioBuffer;

constexpr int kSampleRate = 24000;

AudioBuffer make_buffer(int channels, std::vector<float> samples) {
    AudioBuffer buffer;
    buffer.sample_rate = kSampleRate;
    buffer.channels = channels;
    buffer.samples = std::move(samples);
    return buffer;
}

AudioChunkJoinSpec spec_for(AudioChunkJoinMode mode, float seconds = 0.010F) {
    AudioChunkJoinSpec spec;
    spec.mode = mode;
    spec.cross_fade_seconds = seconds;
    return spec;
}

// Peak of the first difference. A step discontinuity shows up here at its full
// height; a crossfade of length L spreads the same edge over L samples and drops
// the peak by roughly 20*log10(1/L).
float peak_first_difference(const std::vector<float> & samples) {
    float peak = 0.0F;
    for (size_t i = 1; i < samples.size(); ++i) {
        peak = std::max(peak, std::fabs(samples[i] - samples[i - 1]));
    }
    return peak;
}

float rms(const std::vector<float> & samples, size_t start, size_t end) {
    if (end <= start) {
        return 0.0F;
    }
    double sum = 0.0;
    for (size_t i = start; i < end; ++i) {
        sum += static_cast<double>(samples[i]) * static_cast<double>(samples[i]);
    }
    return static_cast<float>(std::sqrt(sum / static_cast<double>(end - start)));
}

float to_db(float ratio) {
    return 20.0F * std::log10(ratio);
}

std::vector<float> pseudo_random_noise(size_t count, uint32_t seed) {
    std::vector<float> out(count);
    uint32_t state = seed;
    for (auto & value : out) {
        state = state * 1664525U + 1013904223U;
        value = static_cast<float>(static_cast<double>(state >> 8U) / 8388608.0 - 1.0);
    }
    return out;
}

// ---------------------------------------------------------------------------
// F4.15 / F6.4 — the shared join point.
// ---------------------------------------------------------------------------

// The common case for the ~28 models that chunk text: the seam lands in the
// inter-sentence silence the text splitter produces. The default (Auto) join must
// leave that alone entirely — same length, same samples, no attenuation.
void test_silent_seam_join_is_a_bit_exact_splice() {
    std::vector<float> previous(4800, 0.0F);
    std::vector<float> next(4800, 0.0F);
    for (size_t i = 0; i < 2000; ++i) {
        previous[i] = std::sin(static_cast<float>(i) * 0.01F);
        next[next.size() - 1 - i] = std::sin(static_cast<float>(i) * 0.013F);
    }

    const AudioBuffer source = make_buffer(1, next);
    AudioBuffer joined = make_buffer(1, previous);
    engine::audio::append_audio_chunk(joined, source, AudioChunkJoinSpec{});

    engine::test::require_close(
        engine::audio::measure_chunk_seam_step(make_buffer(1, previous), source),
        0.0F,
        1e-9F,
        "silent seam step");
    engine::test::require_eq(
        joined.samples.size(),
        previous.size() + next.size(),
        "silent seam join keeps full length");
    for (size_t i = 0; i < previous.size(); ++i) {
        engine::test::require(
            joined.samples[i] == previous[i],
            "silent seam join left the first chunk untouched at sample " + std::to_string(i));
    }
    for (size_t i = 0; i < next.size(); ++i) {
        engine::test::require(
            joined.samples[previous.size() + i] == next[i],
            "silent seam join left the second chunk untouched at sample " + std::to_string(i));
    }
}

// A seam that is quiet but not digitally silent must also be left alone: 0.001 is
// -60 dBFS, below the -54 dBFS gate.
void test_near_silent_seam_is_still_spliced() {
    AudioBuffer joined = make_buffer(1, std::vector<float>(4800, 0.0005F));
    const AudioBuffer source = make_buffer(1, std::vector<float>(4800, -0.0005F));
    engine::audio::append_audio_chunk(joined, source, AudioChunkJoinSpec{});
    engine::test::require_eq(
        joined.samples.size(), static_cast<size_t>(9600), "near-silent seam join keeps full length");
    engine::test::require_close(joined.samples[4799], 0.0005F, 1e-9F, "near-silent tail untouched");
    engine::test::require_close(joined.samples[4800], -0.0005F, 1e-9F, "near-silent head untouched");
}

// Four ASR sessions use this same helper to accumulate contiguous streaming
// input (sense_asr, qwen3_asr, voxtral_realtime, nemotron_asr), and several
// vocoders append consecutive frames with it. There the seam step is just the
// signal's own slew — at 24 kHz a full-scale 1 kHz tone steps 0.26 per sample,
// far above the -54 dBFS silence gate — and crossfading it would both attenuate
// real audio and shorten the buffer, shifting every downstream timestamp. Auto
// must recognise a continuing waveform and splice it untouched.
void test_continuous_waveform_join_is_a_bit_exact_splice() {
    std::vector<float> tone(9600);
    for (size_t i = 0; i < tone.size(); ++i) {
        tone[i] = 0.9F * std::sin(6.2831853F * 1000.0F * static_cast<float>(i) /
            static_cast<float>(kSampleRate));
    }
    const std::vector<float> previous(tone.begin(), tone.begin() + 4800);
    const std::vector<float> next(tone.begin() + 4800, tone.end());

    const float step = engine::audio::measure_chunk_seam_step(
        make_buffer(1, previous), make_buffer(1, next));
    engine::test::require(
        step > 0.2F,
        "a continuous 1 kHz tone still steps well above the silence gate, measured " +
            std::to_string(step));
    engine::test::require(
        !engine::audio::chunk_seam_is_discontinuous(
            make_buffer(1, previous), make_buffer(1, next), AudioChunkJoinSpec{}),
        "a continuing waveform is not a discontinuity");

    AudioBuffer joined = make_buffer(1, previous);
    engine::audio::append_audio_chunk(joined, make_buffer(1, next), AudioChunkJoinSpec{});
    engine::test::require_eq(joined.samples.size(), tone.size(), "continuous join keeps full length");
    for (size_t i = 0; i < tone.size(); ++i) {
        engine::test::require(
            joined.samples[i] == tone[i],
            "continuous join left sample " + std::to_string(i) + " untouched");
    }
}

void test_edge_slew_measurement() {
    // A ramp of 0.01 per sample either side, with a 0.5 step at the join.
    std::vector<float> previous(128);
    std::vector<float> next(128);
    for (size_t i = 0; i < previous.size(); ++i) {
        previous[i] = 0.01F * static_cast<float>(i);
        next[i] = 0.5F + 0.01F * static_cast<float>(i);
    }
    const AudioBuffer previous_buffer = make_buffer(1, previous);
    const AudioBuffer next_buffer = make_buffer(1, next);
    engine::test::require_close(
        engine::audio::measure_chunk_edge_slew(previous_buffer, next_buffer, 64),
        0.01F,
        1e-5F,
        "edge slew excludes the seam itself");
    // Previous ends at 0.01 * 127 = 1.27, next starts at 0.5.
    engine::test::require_close(
        engine::audio::measure_chunk_seam_step(previous_buffer, next_buffer),
        0.77F,
        1e-4F,
        "ramp seam step");
    engine::test::require(
        engine::audio::chunk_seam_is_discontinuous(previous_buffer, next_buffer, AudioChunkJoinSpec{}),
        "a step 50x the local slew is a discontinuity");
    engine::test::require_close(
        engine::audio::measure_chunk_edge_slew(make_buffer(1, {0.5F}), make_buffer(1, {0.5F}), 64),
        0.0F,
        1e-9F,
        "single-frame chunks have no measurable slew");
}

// The pathological case F4.15 is about: a full-scale step at the seam. The
// crossfade must measurably reduce it, and the measurement is asserted as a
// number in dB, not as "smaller".
void test_discontinuous_seam_impulse_is_reduced_by_a_measured_amount() {
    const std::vector<float> previous(4800, 0.5F);
    const std::vector<float> next(4800, -0.5F);
    const AudioBuffer source = make_buffer(1, next);

    engine::test::require_close(
        engine::audio::measure_chunk_seam_step(make_buffer(1, previous), source),
        1.0F,
        1e-6F,
        "discontinuous seam step");

    AudioBuffer spliced = make_buffer(1, previous);
    engine::audio::append_audio_chunk(spliced, source, spec_for(AudioChunkJoinMode::Concat));
    const float spliced_peak = peak_first_difference(spliced.samples);
    engine::test::require_close(spliced_peak, 1.0F, 1e-6F, "raw splice keeps the full step");
    engine::test::require_eq(
        spliced.samples.size(), static_cast<size_t>(9600), "raw splice length");

    // Auto sees a 1.0 step, so it engages, and it engages with equal power.
    AudioBuffer faded = make_buffer(1, previous);
    engine::audio::append_audio_chunk(faded, source, AudioChunkJoinSpec{});
    const int64_t fade_frames = engine::audio::resolve_chunk_cross_fade_frames(
        4800, 4800, kSampleRate, AudioChunkJoinSpec{}.cross_fade_seconds);
    engine::test::require_eq(fade_frames, static_cast<int64_t>(240), "10 ms at 24 kHz is 240 frames");
    engine::test::require_eq(
        faded.samples.size(),
        static_cast<size_t>(9600 - 240),
        "equal-power join overlaps by the fade length");

    const float reduction_db = to_db(peak_first_difference(faded.samples) / spliced_peak);
    engine::test::require(
        reduction_db < -46.0F && reduction_db > -47.5F,
        "10 ms equal-power crossfade reduces the seam impulse by 46-47.5 dB, measured " +
            std::to_string(reduction_db));

    AudioBuffer cos_squared = make_buffer(1, previous);
    engine::audio::append_audio_chunk(cos_squared, source, spec_for(AudioChunkJoinMode::CosSquared));
    const float cos_squared_db = to_db(peak_first_difference(cos_squared.samples) / spliced_peak);
    engine::test::require(
        cos_squared_db < -43.0F && cos_squared_db > -44.5F,
        "10 ms cos-squared crossfade reduces the seam impulse by 43-44.5 dB, measured " +
            std::to_string(cos_squared_db));

    // The crossfade must not leave a residual step of its own where the blended
    // region meets the rest of the second chunk.
    engine::test::require_close(
        faded.samples[faded.samples.size() - 4561], -0.5F, 1e-5F, "blend ends on the second chunk");
}

// cos^2 + sin^2 == 1, so the two gains sum to exactly one and identical content
// passes through the overlap at unity.
void test_cos_squared_preserves_amplitude_for_correlated_input() {
    AudioBuffer joined = make_buffer(1, std::vector<float>(2400, 1.0F));
    const AudioBuffer source = make_buffer(1, std::vector<float>(2400, 1.0F));
    engine::audio::append_audio_chunk(joined, source, spec_for(AudioChunkJoinMode::CosSquared));
    engine::test::require_eq(
        joined.samples.size(), static_cast<size_t>(4560), "cos-squared join length");
    for (size_t i = 0; i < joined.samples.size(); ++i) {
        engine::test::require_close(
            joined.samples[i], 1.0F, 1e-5F, "cos-squared unity gain at sample " + std::to_string(i));
    }
}

// The equal-power pair is a constant-*power* fade, so identical content sums to
// sqrt(2) at the midpoint. That +3.0103 dB is the defining property of the shape,
// not a defect: it is the price of preserving RMS for the uncorrelated content a
// real chunk seam carries. Asserted so it cannot be "fixed" by accident.
void test_equal_power_correlated_midpoint_is_three_db() {
    AudioBuffer joined = make_buffer(1, std::vector<float>(2400, 1.0F));
    const AudioBuffer source = make_buffer(1, std::vector<float>(2400, 1.0F));
    engine::audio::append_audio_chunk(joined, source, spec_for(AudioChunkJoinMode::EqualPower));
    float peak = 0.0F;
    float trough = 2.0F;
    for (const float value : joined.samples) {
        peak = std::max(peak, value);
        trough = std::min(trough, value);
    }
    engine::test::require_close(peak, 1.41421356F, 1e-3F, "equal-power correlated peak is sqrt(2)");
    engine::test::require_close(to_db(peak), 3.0103F, 0.01F, "equal-power correlated peak in dB");
    engine::test::require_close(trough, 1.0F, 1e-5F, "equal-power never attenuates correlated input");
}

// The property that matters at a real seam: the two chunks are independent, and
// the fade must hold the level through the overlap.
void test_equal_power_preserves_level_for_uncorrelated_input() {
    const auto previous = pseudo_random_noise(48000, 12345U);
    const auto next = pseudo_random_noise(48000, 987654321U);
    const AudioBuffer source = make_buffer(1, next);

    AudioBuffer joined = make_buffer(1, previous);
    engine::audio::append_audio_chunk(joined, source, spec_for(AudioChunkJoinMode::EqualPower, 0.5F));
    const size_t fade = 12000;
    const size_t blend_start = previous.size() - fade;
    const float outside = rms(joined.samples, 0, blend_start);
    const float inside = rms(joined.samples, blend_start, blend_start + fade);
    const float level_db = to_db(inside / outside);
    engine::test::require(
        std::fabs(level_db) < 0.35F,
        "equal-power holds level through the overlap for uncorrelated input, measured " +
            std::to_string(level_db) + " dB");

    // The complementary result: cos-squared is amplitude-complementary, so it dips
    // for uncorrelated content. This is why Auto resolves to equal power.
    AudioBuffer cos_squared = make_buffer(1, previous);
    engine::audio::append_audio_chunk(
        cos_squared, source, spec_for(AudioChunkJoinMode::CosSquared, 0.5F));
    const float cos_squared_db =
        to_db(rms(cos_squared.samples, blend_start, blend_start + fade) / outside);
    engine::test::require(
        cos_squared_db < -0.7F && cos_squared_db > -1.7F,
        "cos-squared dips about 1.2 dB for uncorrelated input, measured " +
            std::to_string(cos_squared_db) + " dB");
}

void test_concat_mode_reproduces_the_historical_join() {
    const std::vector<float> previous = {0.1F, 0.2F, 0.3F};
    const std::vector<float> next = {-0.9F, 0.4F};
    AudioBuffer joined = make_buffer(1, previous);
    engine::audio::append_audio_chunk(joined, make_buffer(1, next), spec_for(AudioChunkJoinMode::Concat));
    engine::test::require_eq(joined.samples.size(), static_cast<size_t>(5), "concat length");
    const std::vector<float> expected = {0.1F, 0.2F, 0.3F, -0.9F, 0.4F};
    for (size_t i = 0; i < expected.size(); ++i) {
        engine::test::require(
            joined.samples[i] == expected[i], "concat sample " + std::to_string(i));
    }
}

void test_join_crossfades_each_channel_of_interleaved_audio() {
    // Stereo: left steps 0.6 -> -0.6, right holds 0.0. 4 frames per chunk, fade 2.
    AudioBuffer joined = make_buffer(2, {0.6F, 0.0F, 0.6F, 0.0F, 0.6F, 0.0F, 0.6F, 0.0F});
    const AudioBuffer source = make_buffer(2, {-0.6F, 0.0F, -0.6F, 0.0F, -0.6F, 0.0F, -0.6F, 0.0F});
    const float two_frames_seconds = 2.0F / static_cast<float>(kSampleRate);
    engine::audio::append_audio_chunk(
        joined, source, spec_for(AudioChunkJoinMode::EqualPower, two_frames_seconds));

    engine::test::require_eq(joined.samples.size(), static_cast<size_t>(12), "stereo join sample count");
    engine::test::require_eq(
        joined.samples.size() / 2, static_cast<size_t>(6), "stereo join frame count is 4 + 4 - 2");
    // Frame 2 is the first blend frame: fade_out 1, fade_in 0.
    engine::test::require_close(joined.samples[4], 0.6F, 1e-5F, "stereo blend starts on chunk one");
    // Frame 3 is the last blend frame: fade_out 0, fade_in 1.
    engine::test::require_close(joined.samples[6], -0.6F, 1e-5F, "stereo blend ends on chunk two");
    for (size_t frame = 0; frame < 6; ++frame) {
        engine::test::require_close(
            joined.samples[frame * 2 + 1], 0.0F, 1e-6F,
            "stereo right channel stays silent at frame " + std::to_string(frame));
    }
}

void test_fade_length_is_clamped_to_the_shorter_chunk() {
    engine::test::require_eq(
        engine::audio::resolve_chunk_cross_fade_frames(4800, 4800, kSampleRate, 0.010F),
        static_cast<int64_t>(240),
        "10 ms fade at 24 kHz");
    engine::test::require_eq(
        engine::audio::resolve_chunk_cross_fade_frames(100, 4800, kSampleRate, 0.010F),
        static_cast<int64_t>(100),
        "fade clamped to the shorter previous chunk");
    engine::test::require_eq(
        engine::audio::resolve_chunk_cross_fade_frames(4800, 40, kSampleRate, 0.010F),
        static_cast<int64_t>(40),
        "fade clamped to the shorter next chunk");
    engine::test::require_eq(
        engine::audio::resolve_chunk_cross_fade_frames(4800, 4800, kSampleRate, 0.0F),
        static_cast<int64_t>(0),
        "zero duration disables the fade");

    // A 20-sample chunk joined with a 10 ms request must not be consumed.
    AudioBuffer joined = make_buffer(1, std::vector<float>(20, 0.5F));
    engine::audio::append_audio_chunk(
        joined, make_buffer(1, std::vector<float>(4800, -0.5F)), AudioChunkJoinSpec{});
    engine::test::require_eq(
        joined.samples.size(), static_cast<size_t>(4800), "short chunk join length is 20 + 4800 - 20");
}

void test_single_frame_chunks_are_never_shortened() {
    AudioBuffer joined = make_buffer(1, {0.9F});
    engine::audio::append_audio_chunk(joined, make_buffer(1, {-0.9F}), AudioChunkJoinSpec{});
    engine::test::require_eq(
        joined.samples.size(), static_cast<size_t>(2), "a one-frame chunk is spliced, not consumed");
    engine::test::require(joined.samples[0] == 0.9F, "one-frame splice keeps the tail sample");
    engine::test::require(joined.samples[1] == -0.9F, "one-frame splice keeps the head sample");
}

void test_zero_duration_falls_back_to_a_splice() {
    AudioBuffer joined = make_buffer(1, std::vector<float>(480, 0.5F));
    engine::audio::append_audio_chunk(
        joined,
        make_buffer(1, std::vector<float>(480, -0.5F)),
        spec_for(AudioChunkJoinMode::EqualPower, 0.0F));
    engine::test::require_eq(
        joined.samples.size(), static_cast<size_t>(960), "zero-duration crossfade splices");
}

void test_join_adopts_and_validates_format() {
    AudioBuffer joined;
    engine::audio::append_audio_chunk(joined, make_buffer(2, {0.1F, 0.2F}), AudioChunkJoinSpec{});
    engine::test::require_eq(joined.sample_rate, kSampleRate, "empty destination adopts sample rate");
    engine::test::require_eq(joined.channels, 2, "empty destination adopts channel count");

    bool threw = false;
    try {
        AudioBuffer mismatched = make_buffer(1, {0.1F});
        AudioBuffer other = make_buffer(1, {0.2F});
        other.sample_rate = 48000;
        engine::audio::append_audio_chunk(mismatched, other, AudioChunkJoinSpec{});
    } catch (const std::exception &) {
        threw = true;
    }
    engine::test::require(threw, "sample-rate mismatch rejected");

    threw = false;
    try {
        AudioBuffer mismatched = make_buffer(1, {0.1F});
        engine::audio::append_audio_chunk(mismatched, make_buffer(2, {0.2F, 0.3F}), AudioChunkJoinSpec{});
    } catch (const std::exception &) {
        threw = true;
    }
    engine::test::require(threw, "channel-count mismatch rejected");

    threw = false;
    try {
        AudioBuffer mismatched = make_buffer(1, {0.1F});
        AudioBuffer invalid = make_buffer(1, {0.2F});
        invalid.sample_rate = 0;
        engine::audio::append_audio_chunk(mismatched, invalid, AudioChunkJoinSpec{});
    } catch (const std::exception &) {
        threw = true;
    }
    engine::test::require(threw, "invalid source format rejected");
}

void test_seam_step_measurement() {
    engine::test::require_close(
        engine::audio::measure_chunk_seam_step(make_buffer(1, {0.0F, 0.25F}), make_buffer(1, {-0.25F})),
        0.5F,
        1e-6F,
        "mono seam step");
    engine::test::require_close(
        engine::audio::measure_chunk_seam_step(
            make_buffer(2, {0.0F, 0.0F, 0.1F, 0.4F}), make_buffer(2, {0.1F, -0.4F})),
        0.8F,
        1e-6F,
        "stereo seam step takes the worst channel");
    engine::test::require_close(
        engine::audio::measure_chunk_seam_step(make_buffer(1, {}), make_buffer(1, {0.5F})),
        0.0F,
        1e-9F,
        "empty previous chunk has no seam");
}

// runtime::append_audio_buffer is the shared entry point all ~28 chunked models
// call. The two-argument form must take the Auto default, and the three-argument
// form must let a caller pin the historical splice.
void test_runtime_append_audio_buffer_routes_through_the_join() {
    AudioBuffer silent_seam = make_buffer(1, std::vector<float>(4800, 0.0F));
    engine::runtime::append_audio_buffer(silent_seam, make_buffer(1, std::vector<float>(4800, 0.0F)));
    engine::test::require_eq(
        silent_seam.samples.size(),
        static_cast<size_t>(9600),
        "two-argument append splices a silent seam");

    AudioBuffer discontinuous = make_buffer(1, std::vector<float>(4800, 0.5F));
    engine::runtime::append_audio_buffer(discontinuous, make_buffer(1, std::vector<float>(4800, -0.5F)));
    engine::test::require_eq(
        discontinuous.samples.size(),
        static_cast<size_t>(9360),
        "two-argument append crossfades a discontinuous seam");

    AudioBuffer pinned = make_buffer(1, std::vector<float>(4800, 0.5F));
    engine::runtime::append_audio_buffer(
        pinned, make_buffer(1, std::vector<float>(4800, -0.5F)), spec_for(AudioChunkJoinMode::Concat));
    engine::test::require_eq(
        pinned.samples.size(),
        static_cast<size_t>(9600),
        "three-argument append can pin the historical splice");
    engine::test::require(pinned.samples[4799] == 0.5F, "pinned splice keeps the tail sample");
    engine::test::require(pinned.samples[4800] == -0.5F, "pinned splice keeps the head sample");
}

void test_join_option_parsing() {
    engine::test::require(
        engine::audio::parse_audio_chunk_join_mode("auto") == AudioChunkJoinMode::Auto,
        "auto mode parsed");
    engine::test::require(
        engine::audio::parse_audio_chunk_join_mode("concat") == AudioChunkJoinMode::Concat,
        "concat mode parsed");
    engine::test::require(
        engine::audio::parse_audio_chunk_join_mode("none") == AudioChunkJoinMode::Concat,
        "none is an alias for concat");
    engine::test::require(
        engine::audio::parse_audio_chunk_join_mode("equal_power") == AudioChunkJoinMode::EqualPower,
        "equal_power mode parsed");
    engine::test::require(
        engine::audio::parse_audio_chunk_join_mode("cos-squared") == AudioChunkJoinMode::CosSquared,
        "cos-squared mode parsed");

    bool threw = false;
    try {
        (void)engine::audio::parse_audio_chunk_join_mode("triangle");
    } catch (const std::exception &) {
        threw = true;
    }
    engine::test::require(threw, "unknown join mode rejected");

    const AudioChunkJoinSpec defaults = engine::audio::parse_audio_chunk_join_spec({});
    engine::test::require(defaults.mode == AudioChunkJoinMode::Auto, "default join mode is auto");
    engine::test::require_close(defaults.cross_fade_seconds, 0.010F, 1e-9F, "default fade is 10 ms");
    engine::test::require_close(defaults.silent_seam_step, 0.002F, 1e-9F, "default silent seam gate");
    engine::test::require_close(defaults.seam_slew_ratio, 2.0F, 1e-9F, "default seam slew ratio");
    engine::test::require_eq(
        defaults.seam_slew_window_frames, static_cast<int64_t>(64), "default seam slew window");

    const std::unordered_map<std::string, std::string> options = {
        {"audio_chunk_join", "cos_squared"},
        {"cross_fade_duration_sec", "0.05"},
    };
    const AudioChunkJoinSpec parsed = engine::audio::parse_audio_chunk_join_spec(options);
    engine::test::require(parsed.mode == AudioChunkJoinMode::CosSquared, "join mode option honoured");
    engine::test::require_close(parsed.cross_fade_seconds, 0.05F, 1e-6F, "fade duration option honoured");
}

}  // namespace

int main() {
    try {
        test_silent_seam_join_is_a_bit_exact_splice();
        test_near_silent_seam_is_still_spliced();
        test_continuous_waveform_join_is_a_bit_exact_splice();
        test_edge_slew_measurement();
        test_discontinuous_seam_impulse_is_reduced_by_a_measured_amount();
        test_cos_squared_preserves_amplitude_for_correlated_input();
        test_equal_power_correlated_midpoint_is_three_db();
        test_equal_power_preserves_level_for_uncorrelated_input();
        test_concat_mode_reproduces_the_historical_join();
        test_join_crossfades_each_channel_of_interleaved_audio();
        test_fade_length_is_clamped_to_the_shorter_chunk();
        test_single_frame_chunks_are_never_shortened();
        test_zero_duration_falls_back_to_a_splice();
        test_join_adopts_and_validates_format();
        test_seam_step_measurement();
        test_runtime_append_audio_buffer_routes_through_the_join();
        test_join_option_parsing();
        std::cout << "audio_chunk_join_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "audio_chunk_join_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
