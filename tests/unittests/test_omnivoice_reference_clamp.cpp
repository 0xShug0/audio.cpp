// F6.1 -- OmniVoice's reference length clamp.
//
// The clamp used to sit behind `!options.has_reference_text`, a predicate no
// working voice clone can satisfy, so a 101 s reference was encoded whole at
// 75 Hz. These tests pin the arithmetic of the replacement on synthetic audio;
// no model weights and no backend are needed.

#include "engine/models/omnivoice/audio_tokenizer.h"

#include "test_assert.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::models::omnivoice::clip_reference_mono;
using engine::models::omnivoice::OmniVoiceReferenceClipOptions;

constexpr int kSampleRate = 24000;
// The analysis window is 20 ms, so every synthetic boundary below is placed on
// a 20 ms grid; that makes the expected cut positions exact rather than
// approximate.
constexpr int64_t kWindowMs = 20;

int64_t ms_to_samples(int64_t ms) {
    return ms * kSampleRate / 1000;
}

void append_silence(std::vector<float> & samples, int64_t ms) {
    samples.insert(samples.end(), static_cast<size_t>(ms_to_samples(ms)), 0.0F);
}

// A 220 Hz tone at a fixed amplitude. Deterministic, and far enough above the
// clip's own noise floor that the adaptive threshold classifies it as voiced.
void append_tone(std::vector<float> & samples, int64_t ms, float amplitude = 0.3F) {
    const int64_t count = ms_to_samples(ms);
    const double step = 2.0 * 3.14159265358979323846 * 220.0 / static_cast<double>(kSampleRate);
    const auto phase_base = static_cast<double>(samples.size());
    for (int64_t i = 0; i < count; ++i) {
        samples.push_back(
            amplitude * static_cast<float>(std::sin(step * (phase_base + static_cast<double>(i)))));
    }
}

double rms(const std::vector<float> & samples, size_t begin, size_t end) {
    if (begin >= end || end > samples.size()) {
        throw std::runtime_error("rms window out of range");
    }
    double sum = 0.0;
    for (size_t i = begin; i < end; ++i) {
        sum += static_cast<double>(samples[i]) * static_cast<double>(samples[i]);
    }
    return std::sqrt(sum / static_cast<double>(end - begin));
}

// A reference over the limit is cut at the START of a silence run, so the last
// word survives whole. Asserted as a number: the 250 ms after the cut are
// digital silence in the source, and the 100 ms before it are not.
void test_long_reference_cuts_at_a_pause() {
    // 500 ms lead silence, then 12 x (2000 ms speech + 300 ms pause).
    std::vector<float> source;
    append_silence(source, 500);
    for (int i = 0; i < 12; ++i) {
        append_tone(source, 2000);
        append_silence(source, 300);
    }

    OmniVoiceReferenceClipOptions options;
    options.max_seconds = 15.0F;
    options.pad_ms = 150;
    const auto result = clip_reference_mono(source, kSampleRate, options);

    engine::test::require(result.clamped, "long reference is clamped");
    engine::test::require(!result.no_pause_found, "long reference found a pause");
    engine::test::require_eq(result.pause_tier_ms, int64_t{250}, "pause tier");

    // Content starts at 500 ms. The budget ends at 15500 ms. Pause k starts at
    // 2500 + 2300k ms, so the last one inside the budget is k = 5 at 14000 ms.
    engine::test::require_eq(result.content_start_sample, ms_to_samples(500), "content start");
    engine::test::require_eq(result.cut_sample, ms_to_samples(14000), "cut sample");

    const auto cut = static_cast<size_t>(result.cut_sample);
    engine::test::require_close(
        static_cast<float>(rms(source, cut, cut + static_cast<size_t>(ms_to_samples(250)))),
        0.0F,
        1.0e-7F,
        "source is silent for a full pause after the cut");
    engine::test::require(
        rms(source, cut - static_cast<size_t>(ms_to_samples(100)), cut) > 0.1,
        "source carries speech right up to the cut, so no word was sliced");

    // Kept body is 500 ms .. 14000 ms, plus one pad at each end.
    const size_t expected =
        static_cast<size_t>(ms_to_samples(13500) + (2 * ms_to_samples(options.pad_ms)));
    engine::test::require_eq(result.samples.size(), expected, "clamped sample count");
    engine::test::require_close(
        static_cast<float>(result.output_seconds),
        13.8F,
        1.0e-3F,
        "clamped duration");
}

// Both ends of a clamped reference carry exact digital silence of the
// configured pad length -- not "quiet", zero.
void test_clamped_reference_pads_with_exact_silence() {
    std::vector<float> source;
    append_silence(source, 500);
    for (int i = 0; i < 12; ++i) {
        append_tone(source, 2000);
        append_silence(source, 300);
    }

    OmniVoiceReferenceClipOptions options;
    options.max_seconds = 15.0F;
    options.pad_ms = 150;
    const auto result = clip_reference_mono(source, kSampleRate, options);

    const auto pad = static_cast<size_t>(ms_to_samples(options.pad_ms));
    engine::test::require(result.samples.size() > 2 * pad, "clamped output is longer than its padding");
    engine::test::require_eq(rms(result.samples, 0, pad), 0.0, "head pad rms is exactly zero");
    engine::test::require_eq(
        rms(result.samples, result.samples.size() - pad, result.samples.size()),
        0.0,
        "tail pad rms is exactly zero");
    // The pad is exactly one sample longer than the silence at either end: the
    // sample just inside it is real signal, so nothing was over-trimmed.
    engine::test::require(
        std::fabs(result.samples[pad]) > 0.0F || std::fabs(result.samples[pad + 1]) > 0.0F,
        "signal resumes immediately after the head pad");
    engine::test::require(
        std::fabs(result.samples[result.samples.size() - pad - 1]) > 0.0F,
        "signal runs right up to the tail pad");

    // A different pad length is honoured exactly.
    OmniVoiceReferenceClipOptions wide = options;
    wide.pad_ms = 300;
    const auto wider = clip_reference_mono(source, kSampleRate, wide);
    const auto wide_pad = static_cast<size_t>(ms_to_samples(wide.pad_ms));
    engine::test::require_eq(
        wider.samples.size(),
        result.samples.size() + (2 * (wide_pad - pad)),
        "pad length is applied verbatim at both ends");
    engine::test::require_eq(rms(wider.samples, 0, wide_pad), 0.0, "wide head pad rms is exactly zero");
    engine::test::require_eq(
        rms(wider.samples, wider.samples.size() - wide_pad, wider.samples.size()),
        0.0,
        "wide tail pad rms is exactly zero");
}

// No pause anywhere inside the limit: keep the whole clip and take the warning
// path rather than slicing a word in half.
void test_reference_without_a_pause_is_kept_whole() {
    // 500 ms silence, 40 s of unbroken speech, 500 ms silence. The only silence
    // runs are the two edges: one starts at 0 (not after content start) and one
    // starts at 40500 ms (far past the 15500 ms budget).
    std::vector<float> source;
    append_silence(source, 500);
    append_tone(source, 40000);
    append_silence(source, 500);

    OmniVoiceReferenceClipOptions options;
    options.max_seconds = 15.0F;
    options.pad_ms = 150;
    const auto result = clip_reference_mono(source, kSampleRate, options);

    engine::test::require(result.no_pause_found, "warning path is taken");
    engine::test::require(!result.clamped, "no speech was dropped");
    engine::test::require_eq(result.pause_tier_ms, int64_t{0}, "no pause tier matched");
    engine::test::require_eq(result.content_start_sample, ms_to_samples(500), "content start");
    engine::test::require_eq(result.cut_sample, ms_to_samples(40500), "cut is the end of the content");

    const size_t expected =
        static_cast<size_t>(ms_to_samples(40000) + (2 * ms_to_samples(options.pad_ms)));
    engine::test::require_eq(result.samples.size(), expected, "whole clip is preserved");
    const auto pad = static_cast<size_t>(ms_to_samples(options.pad_ms));
    engine::test::require_eq(rms(result.samples, 0, pad), 0.0, "head pad rms is exactly zero");
    engine::test::require_eq(
        rms(result.samples, result.samples.size() - pad, result.samples.size()),
        0.0,
        "tail pad rms is exactly zero");
}

// A reference inside the limit is returned byte-for-byte, so the shipped demo
// voices (4.7-9.9 s) are untouched by the clamp.
void test_short_reference_is_passed_through_unchanged() {
    std::vector<float> source;
    append_silence(source, 160);
    append_tone(source, 6000);
    append_silence(source, 160);

    OmniVoiceReferenceClipOptions options;
    options.max_seconds = 15.0F;
    options.pad_ms = 150;
    const auto result = clip_reference_mono(source, kSampleRate, options);

    engine::test::require(!result.clamped, "short reference is not clamped");
    engine::test::require(!result.no_pause_found, "short reference takes no warning path");
    engine::test::require_eq(result.samples.size(), source.size(), "short reference keeps its length");
    for (size_t i = 0; i < source.size(); ++i) {
        if (result.samples[i] != source[i]) {
            throw std::runtime_error("short reference altered at sample " + std::to_string(i));
        }
    }
    engine::test::require_close(
        static_cast<float>(result.output_seconds),
        static_cast<float>(result.input_seconds),
        0.0F,
        "short reference duration");
}

// A larger limit turns the clamped case back into a pass-through, and a limit
// of zero disables the clamp entirely.
void test_limit_is_configurable() {
    std::vector<float> source;
    append_silence(source, 500);
    for (int i = 0; i < 12; ++i) {
        append_tone(source, 2000);
        append_silence(source, 300);
    }

    OmniVoiceReferenceClipOptions generous;
    generous.max_seconds = 60.0F;
    const auto kept = clip_reference_mono(source, kSampleRate, generous);
    engine::test::require(!kept.clamped, "a 60 s limit does not clamp a 28 s reference");
    engine::test::require_eq(kept.samples.size(), source.size(), "generous limit is a pass-through");

    OmniVoiceReferenceClipOptions disabled;
    disabled.max_seconds = 0.0F;
    const auto untouched = clip_reference_mono(source, kSampleRate, disabled);
    engine::test::require(!untouched.clamped, "max_seconds=0 disables the clamp");
    engine::test::require_eq(untouched.samples.size(), source.size(), "disabled limit is a pass-through");

    // A tighter limit lands on the earlier pause: budget ends at 8500 ms, so the
    // last qualifying pause starts at 2500 + 2300 * 2 = 7100 ms.
    OmniVoiceReferenceClipOptions tight;
    tight.max_seconds = 8.0F;
    tight.pad_ms = 150;
    const auto cut = clip_reference_mono(source, kSampleRate, tight);
    engine::test::require(cut.clamped, "an 8 s limit clamps");
    engine::test::require_eq(cut.cut_sample, ms_to_samples(7100), "tight cut sample");
}

// Silence entirely below the analysis threshold must not be mistaken for
// content, and a clip with no signal at all is left alone rather than emptied.
void test_silent_reference_is_left_alone() {
    std::vector<float> source(static_cast<size_t>(ms_to_samples(30000)), 0.0F);
    OmniVoiceReferenceClipOptions options;
    options.max_seconds = 15.0F;
    const auto result = clip_reference_mono(source, kSampleRate, options);
    engine::test::require(!result.clamped, "a silent reference is not clamped");
    engine::test::require_eq(result.samples.size(), source.size(), "a silent reference is untouched");
}

void test_window_grid_assumption() {
    // Every boundary used above is a whole number of analysis windows, which is
    // what makes the expected cut positions exact.
    engine::test::require_eq(500 % kWindowMs, int64_t{0}, "lead silence on the window grid");
    engine::test::require_eq(2000 % kWindowMs, int64_t{0}, "speech run on the window grid");
    engine::test::require_eq(300 % kWindowMs, int64_t{0}, "pause on the window grid");
    engine::test::require_eq(ms_to_samples(kWindowMs), int64_t{480}, "20 ms window at 24 kHz");
}

}  // namespace

int main() {
    try {
        test_window_grid_assumption();
        test_long_reference_cuts_at_a_pause();
        test_clamped_reference_pads_with_exact_silence();
        test_reference_without_a_pause_is_kept_whole();
        test_short_reference_is_passed_through_unchanged();
        test_limit_is_configurable();
        test_silent_reference_is_left_alone();
        std::cout << "omnivoice_reference_clamp_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "omnivoice_reference_clamp_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
