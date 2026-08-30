#include "engine/framework/audio/flashsr.h"
#include "engine/framework/audio/rnnoise.h"
#include "engine/framework/audio/zipenhancer.h"

#include "test_assert.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::test::require;
using engine::test::require_close;
using engine::test::require_eq;

int64_t ceil_div(int64_t value, int64_t divisor) {
    return (value + divisor - 1) / divisor;
}

float to_db(float ratio) {
    return 20.0f * std::log10(std::max(ratio, 1.0e-30f));
}

std::string with_length(const std::string & label, int64_t samples) {
    std::ostringstream oss;
    oss << label << " (n=" << samples << ")";
    return oss.str();
}

// ---------------------------------------------------------------------------
// F4.18 — RNNoise output is delayed 20 ms and never compensated
// ---------------------------------------------------------------------------

void test_rnnoise_alignment_plan_arithmetic() {
    constexpr int64_t frame = 480;
    require_eq(engine::audio::kRnnoiseOutputDelaySamples, static_cast<int64_t>(960), "RNNoise delay constant");

    const std::vector<int64_t> lengths = {1, 479, 480, 481, 960, 961, 4800, 5760, 48000, 48001, 123457};
    for (const int64_t samples : lengths) {
        const auto fixed = engine::audio::rnnoise_alignment_plan(samples, engine::audio::RnnoiseProcessOptions{});
        require_eq(fixed.crop_offset, static_cast<int64_t>(960), with_length("RNNoise crop offset", samples));
        require_eq(fixed.output_samples, samples, with_length("RNNoise output length", samples));
        require_eq(fixed.frames, ceil_div(samples + 960, frame), with_length("RNNoise frame count", samples));
        require_eq(fixed.padded_samples, fixed.frames * frame, with_length("RNNoise padded length", samples));
        require_eq(fixed.vad_frames, ceil_div(samples, frame), with_length("RNNoise vad frame count", samples));
        require(
            fixed.padded_samples >= fixed.crop_offset + fixed.output_samples,
            with_length("RNNoise padding must cover the delay-compensated crop", samples));

        // Legacy mode must reproduce the pre-fix arithmetic exactly: frames =
        // ceil(n / 480), no crop, output resized straight back to n.
        engine::audio::RnnoiseProcessOptions legacy;
        legacy.compensate_output_delay = false;
        const auto uncompensated = engine::audio::rnnoise_alignment_plan(samples, legacy);
        require_eq(uncompensated.crop_offset, static_cast<int64_t>(0), with_length("RNNoise legacy crop", samples));
        require_eq(uncompensated.frames, ceil_div(samples, frame), with_length("RNNoise legacy frames", samples));
        require_eq(
            uncompensated.padded_samples,
            uncompensated.frames * frame,
            with_length("RNNoise legacy padded length", samples));
        require_eq(uncompensated.output_samples, samples, with_length("RNNoise legacy output length", samples));
    }
}

// Model the analysis/synthesis pair as the pure 960-sample delay it is:
// raw_output[m] == padded_input[m - 960]. The crop the plan describes must undo
// it exactly, for an impulse anywhere in the input including the very last
// sample (which never reached the output before the fix).
void test_rnnoise_crop_recovers_input_alignment() {
    const std::vector<int64_t> lengths = {480, 1000, 4800, 5760, 20001};
    for (const int64_t samples : lengths) {
        for (const int64_t impulse : {static_cast<int64_t>(0), samples / 3, samples - 1}) {
            const auto plan = engine::audio::rnnoise_alignment_plan(samples, engine::audio::RnnoiseProcessOptions{});
            std::vector<float> padded_input(static_cast<size_t>(plan.padded_samples), 0.0f);
            padded_input[static_cast<size_t>(impulse)] = 1.0f;

            std::vector<float> raw_output(static_cast<size_t>(plan.padded_samples), 0.0f);
            for (int64_t m = engine::audio::kRnnoiseOutputDelaySamples; m < plan.padded_samples; ++m) {
                raw_output[static_cast<size_t>(m)] =
                    padded_input[static_cast<size_t>(m - engine::audio::kRnnoiseOutputDelaySamples)];
            }

            require(
                plan.crop_offset + plan.output_samples <= plan.padded_samples,
                with_length("RNNoise crop fits inside the synthesis buffer", samples));
            const std::vector<float> aligned(
                raw_output.begin() + static_cast<std::ptrdiff_t>(plan.crop_offset),
                raw_output.begin() + static_cast<std::ptrdiff_t>(plan.crop_offset + plan.output_samples));
            require_eq(
                static_cast<int64_t>(aligned.size()),
                samples,
                with_length("RNNoise aligned output length", samples));
            require_close(aligned[static_cast<size_t>(impulse)], 1.0f, 0.0f, "RNNoise impulse lands at its input index");

            double stray = 0.0;
            for (size_t i = 0; i < aligned.size(); ++i) {
                if (i != static_cast<size_t>(impulse)) {
                    stray += static_cast<double>(std::fabs(aligned[i]));
                }
            }
            require(stray == 0.0, with_length("RNNoise aligned output has no stray energy", samples));
        }
    }

    // Without compensation the same impulse comes back 960 samples late, and an
    // impulse in the final 960 samples of the input is lost entirely.
    engine::audio::RnnoiseProcessOptions legacy;
    legacy.compensate_output_delay = false;
    const int64_t samples = 4800;
    const auto plan = engine::audio::rnnoise_alignment_plan(samples, legacy);
    std::vector<float> padded_input(static_cast<size_t>(plan.padded_samples), 0.0f);
    padded_input[static_cast<size_t>(samples - 1)] = 1.0f;
    std::vector<float> raw_output(static_cast<size_t>(plan.padded_samples), 0.0f);
    for (int64_t m = engine::audio::kRnnoiseOutputDelaySamples; m < plan.padded_samples; ++m) {
        raw_output[static_cast<size_t>(m)] =
            padded_input[static_cast<size_t>(m - engine::audio::kRnnoiseOutputDelaySamples)];
    }
    const std::vector<float> uncompensated(
        raw_output.begin(),
        raw_output.begin() + static_cast<std::ptrdiff_t>(plan.output_samples));
    float peak = 0.0f;
    for (const float value : uncompensated) {
        peak = std::max(peak, std::fabs(value));
    }
    require(peak == 0.0f, "RNNoise legacy path drops the final 20 ms of input");
}

// ---------------------------------------------------------------------------
// F4.16 — ZipEnhancer hard-splices its output every 1.5 seconds
// ---------------------------------------------------------------------------

void test_zipenhancer_fade_envelope_is_complementary() {
    constexpr int64_t overlap = 8000;
    for (const int64_t fade : {static_cast<int64_t>(0), static_cast<int64_t>(80), static_cast<int64_t>(320),
                               static_cast<int64_t>(4000), overlap, overlap * 2}) {
        float previous = -1.0f;
        for (int64_t position = 0; position < overlap; ++position) {
            const float rising = engine::audio::zipenhancer_chunk_fade_weight(position, overlap, fade);
            const float falling = engine::audio::zipenhancer_chunk_fade_weight(overlap - 1 - position, overlap, fade);
            require(rising >= previous, "ZipEnhancer crossfade envelope must be non-decreasing");
            require(rising >= 0.0f && rising <= 1.0f, "ZipEnhancer crossfade envelope must stay in [0,1]");
            require_close(rising + falling, 1.0f, 1.0e-6f, "ZipEnhancer crossfade pair must sum to unity");
            previous = rising;
        }
    }

    // A fade of zero is exactly the legacy hard splice: full weight on the
    // earlier chunk for the first half of the overlap, then a one-sample step.
    require_close(engine::audio::zipenhancer_chunk_fade_weight(3999, 8000, 0), 0.0f, 0.0f, "legacy splice below step");
    require_close(engine::audio::zipenhancer_chunk_fade_weight(4000, 8000, 0), 1.0f, 0.0f, "legacy splice above step");
}

// A level mismatch of amplitude d across the join produces a single-sample
// discontinuity of d with a hard splice. Spreading it over a linear fade of F
// samples reduces the largest single-sample step to d / (F + 1), i.e. an
// attenuation of 20*log10(F + 1) dB.
void test_zipenhancer_fade_attenuates_the_seam_step() {
    constexpr int64_t overlap = 8000;
    struct FadeCase {
        int64_t fade;
        float min_attenuation_db;
    };
    // 20*log10(F + 1): 38.2 dB at 80 samples, 50.1 dB at 320, 78.1 dB at 8000.
    const FadeCase cases[] = {
        {80, 37.0f},      // 5 ms at 16 kHz
        {320, 49.0f},     // 20 ms
        {overlap, 77.0f}, // the shipped default: the whole 500 ms overlap
    };

    const float hard_splice_step =
        engine::audio::zipenhancer_chunk_fade_weight(4000, overlap, 0) -
        engine::audio::zipenhancer_chunk_fade_weight(3999, overlap, 0);
    require_close(hard_splice_step, 1.0f, 0.0f, "hard splice step is the full seam amplitude");

    for (const auto & fade_case : cases) {
        float max_step = 0.0f;
        for (int64_t position = 1; position < overlap; ++position) {
            const float step =
                engine::audio::zipenhancer_chunk_fade_weight(position, overlap, fade_case.fade) -
                engine::audio::zipenhancer_chunk_fade_weight(position - 1, overlap, fade_case.fade);
            max_step = std::max(max_step, std::fabs(step));
        }
        const float attenuation_db = -to_db(max_step / hard_splice_step);
        std::ostringstream oss;
        oss << "ZipEnhancer " << fade_case.fade << "-sample fade seam attenuation " << attenuation_db
            << " dB is below the required " << fade_case.min_attenuation_db << " dB";
        require(attenuation_db >= fade_case.min_attenuation_db, oss.str());
    }
}

// ---------------------------------------------------------------------------
// F4.17 — ZipEnhancer leaves up to 250 ms of silence on the end of the file
// ---------------------------------------------------------------------------

// Coverage the pre-fix code produced: chunk 0 wrote [0, window - give_up) and
// every later chunk wrote [start + give_up, start + window - give_up). The last
// give_up samples of the padded buffer were never written by anyone.
std::vector<float> legacy_hard_splice_coverage(const engine::audio::ZipEnhancerChunkPlan & plan) {
    const int64_t give_up = (plan.window_samples - plan.stride_samples) / 2;
    std::vector<float> covered(static_cast<size_t>(plan.padded_samples), 0.0f);
    for (int64_t current = 0; current + plan.window_samples <= plan.padded_samples; current += plan.stride_samples) {
        const int64_t begin = current == 0 ? 0 : current + give_up;
        const int64_t end = current + plan.window_samples - give_up;
        for (int64_t i = begin; i < end; ++i) {
            covered[static_cast<size_t>(i)] = 1.0f;
        }
    }
    return covered;
}

int64_t trailing_zero_run(const std::vector<float> & values, int64_t limit) {
    int64_t run = 0;
    for (int64_t i = limit - 1; i >= 0 && values[static_cast<size_t>(i)] == 0.0f; --i) {
        ++run;
    }
    return run;
}

void test_zipenhancer_segmented_coverage_has_no_tail_hole() {
    // 32000-sample window, 24000-sample stride: remainder == 0 whenever
    // n = 32000 + k * 24000, which is the case that used to leave a full 4000
    // zero samples (250 ms) on the end. Sweep it and its neighbours.
    std::vector<int64_t> lengths;
    for (int64_t k = 4; k <= 9; ++k) {
        const int64_t aligned = 32000 + k * 24000;
        for (const int64_t delta : {static_cast<int64_t>(-20001), static_cast<int64_t>(-3999),
                                    static_cast<int64_t>(-1), static_cast<int64_t>(0),
                                    static_cast<int64_t>(1), static_cast<int64_t>(3999),
                                    static_cast<int64_t>(20001)}) {
            lengths.push_back(aligned + delta);
        }
    }
    lengths.push_back(96001);
    lengths.push_back(2880000);  // 3 minutes

    bool saw_legacy_hole = false;
    for (const int64_t samples : lengths) {
        const engine::audio::ZipEnhancerOptions options;
        const auto plan = engine::audio::zipenhancer_chunk_plan(samples, options);
        require(plan.segmented, with_length("ZipEnhancer sweep length must be segmented", samples));
        require_eq(plan.output_samples, samples, with_length("ZipEnhancer output length", samples));
        require(
            plan.padded_samples >= samples,
            with_length("ZipEnhancer padded length must cover the input", samples));
        require_eq(
            (plan.padded_samples - plan.window_samples) % plan.stride_samples,
            static_cast<int64_t>(0),
            with_length("ZipEnhancer padded length must land on a stride boundary", samples));

        const auto weights = engine::audio::zipenhancer_chunk_weights(plan, options);
        require_eq(
            static_cast<int64_t>(weights.size()),
            plan.padded_samples,
            with_length("ZipEnhancer weight buffer length", samples));

        // Every returned sample must be covered, and the accumulated weight must
        // be exactly unity so the overlap-add neither dips nor bumps the level.
        float min_weight = weights.empty() ? 0.0f : weights[0];
        float max_deviation = 0.0f;
        for (int64_t i = 0; i < plan.output_samples; ++i) {
            const float weight = weights[static_cast<size_t>(i)];
            min_weight = std::min(min_weight, weight);
            max_deviation = std::max(max_deviation, std::fabs(weight - 1.0f));
        }
        require(min_weight > 0.0f, with_length("ZipEnhancer left an uncovered output sample", samples));
        require_close(max_deviation, 0.0f, 1.0e-6f, with_length("ZipEnhancer overlap-add weight is not unity", samples));
        require_eq(
            trailing_zero_run(weights, plan.output_samples),
            static_cast<int64_t>(0),
            with_length("ZipEnhancer trailing uncovered run", samples));

        // The legacy hard-splice geometry is the regression witness: on the
        // stride-aligned lengths it leaves 4000 uncovered samples (250 ms).
        const auto legacy_covered = legacy_hard_splice_coverage(plan);
        const int64_t legacy_hole = trailing_zero_run(legacy_covered, plan.output_samples);
        if (legacy_hole > 0) {
            saw_legacy_hole = true;
            require(
                legacy_hole <= 4000,
                with_length("legacy hole should never exceed the give-up region", samples));
        }

        // The legacy crossfade setting must still close the hole.
        engine::audio::ZipEnhancerOptions hard_splice;
        hard_splice.chunk_crossfade_samples = 0;
        const auto hard_weights = engine::audio::zipenhancer_chunk_weights(plan, hard_splice);
        for (int64_t i = 0; i < plan.output_samples; ++i) {
            require_close(
                hard_weights[static_cast<size_t>(i)],
                1.0f,
                1.0e-6f,
                with_length("ZipEnhancer hard-splice weight is not unity", samples));
        }
    }
    require(saw_legacy_hole, "sweep must include at least one length that used to lose its tail");
}

void test_zipenhancer_whole_file_lengths() {
    const engine::audio::ZipEnhancerOptions options;
    engine::audio::ZipEnhancerOptions legacy;
    legacy.match_input_length = false;
    legacy.chunk_crossfade_samples = 0;

    // Shorter than the 2 s window: zero-padded up to the window, then trimmed
    // back to the input length in both modes.
    for (const int64_t samples : {static_cast<int64_t>(1), static_cast<int64_t>(1601), static_cast<int64_t>(31999)}) {
        const auto plan = engine::audio::zipenhancer_chunk_plan(samples, options);
        require(!plan.segmented, with_length("short input must not segment", samples));
        require_eq(plan.padded_samples, static_cast<int64_t>(32000), with_length("short input padding", samples));
        require_eq(plan.output_samples, samples, with_length("short input output length", samples));
        require_eq(
            engine::audio::zipenhancer_chunk_plan(samples, legacy).output_samples,
            samples,
            with_length("short input legacy output length", samples));
    }

    // Between the window and the 6 s segmentation threshold the un-segmented
    // path used to return floor(n / 100) * 100 samples, up to 99 short.
    for (const int64_t samples : {static_cast<int64_t>(32000), static_cast<int64_t>(47831),
                                  static_cast<int64_t>(60099), static_cast<int64_t>(96000)}) {
        const auto plan = engine::audio::zipenhancer_chunk_plan(samples, options);
        require(!plan.segmented, with_length("mid-length input must not segment", samples));
        require_eq(plan.output_samples, samples, with_length("mid-length output length", samples));
        const auto legacy_plan = engine::audio::zipenhancer_chunk_plan(samples, legacy);
        require_eq(
            legacy_plan.output_samples,
            (samples / 100) * 100,
            with_length("legacy mid-length output length", samples));
        require(
            plan.output_samples - legacy_plan.output_samples <= 99,
            with_length("legacy shortfall bound", samples));
    }

    // The exact number the checked-in reference fixture carries.
    require_eq(
        engine::audio::zipenhancer_chunk_plan(47831, legacy).output_samples,
        static_cast<int64_t>(47800),
        "legacy path reproduces the reference fixture length");
}

// ---------------------------------------------------------------------------
// F4.19 — FlashSR peak-normalises the whole file to 0.999
// ---------------------------------------------------------------------------

void test_flashsr_gain_preserves_input_level() {
    const engine::audio::FlashSrOptions options;

    // A signal whose peak sits well below full scale must come back at its own
    // peak, not at 0.999.
    struct LevelCase {
        float input_peak;
        float output_peak;
    };
    const LevelCase cases[] = {
        {0.01f, 0.83f},   // -40 dBFS whisper
        {0.1f, 0.42f},
        {0.12f, 0.73f},
        {0.5f, 0.2f},
        {0.7071f, 0.9f},
    };
    for (const auto & level : cases) {
        const float gain = engine::audio::flashsr_output_gain(level.input_peak, level.output_peak, options);
        const float restored = level.output_peak * gain;
        std::ostringstream oss;
        oss << "FlashSR restored peak for input_peak=" << level.input_peak;
        require_close(restored, level.input_peak, 1.0e-6f, oss.str());
        require(restored <= options.peak_ceiling, "FlashSR restored peak must stay under the ceiling");

        // The legacy policy throws all of that away: every file lands on 0.999
        // regardless of what went in.
        engine::audio::FlashSrOptions legacy;
        legacy.preserve_input_level = false;
        const float legacy_gain = engine::audio::flashsr_output_gain(level.input_peak, level.output_peak, legacy);
        require_close(level.output_peak * legacy_gain, options.peak_ceiling, 1.0e-6f, "FlashSR legacy peak");
    }

    // Level must track the input: doubling the input peak doubles the output
    // peak. The legacy gain is completely blind to it.
    const float quiet = engine::audio::flashsr_output_gain(0.05f, 0.6f, options);
    const float loud = engine::audio::flashsr_output_gain(0.10f, 0.6f, options);
    require_close(loud / quiet, 2.0f, 1.0e-5f, "FlashSR gain must track the input level");
    const float gain_swing_db = to_db(engine::audio::flashsr_output_gain(0.01f, 0.6f, options)) -
                                to_db(engine::audio::flashsr_output_gain(0.7071f, 0.6f, options));
    require(
        std::fabs(gain_swing_db + 37.0f) < 1.0f,
        "FlashSR gain must follow a 37 dB input level swing rather than flatten it");
}

void test_flashsr_gain_safety_and_silence() {
    const engine::audio::FlashSrOptions options;

    // An input already at or above full scale is limited, never boosted past
    // the ceiling.
    for (const float input_peak : {1.0f, 1.5f, 12.0f}) {
        const float gain = engine::audio::flashsr_output_gain(input_peak, 0.9f, options);
        require_close(0.9f * gain, options.peak_ceiling, 1.0e-6f, "FlashSR ceiling clamp");
    }

    // Silence in, silence out — the old code threw here.
    require_close(engine::audio::flashsr_output_gain(0.4f, 0.0f, options), 0.0f, 0.0f, "FlashSR zero output peak");
    require_close(engine::audio::flashsr_output_gain(0.0f, 0.5f, options), 0.0f, 0.0f, "FlashSR silent input");

    // A whole synthetic waveform round-trip: a 0.12-peak sine handed to a model
    // that returned a 0.73-peak version of it comes back at 0.12.
    std::vector<float> model_output(4800, 0.0f);
    for (size_t i = 0; i < model_output.size(); ++i) {
        model_output[i] = 0.73f * std::sin(0.05f * static_cast<float>(i));
    }
    const float gain = engine::audio::flashsr_output_gain(0.12f, 0.73f, options);
    float peak = 0.0f;
    for (const float value : model_output) {
        peak = std::max(peak, std::fabs(value * gain));
    }
    require_close(peak, 0.12f, 1.0e-4f, "FlashSR waveform peak preserved");
}

}  // namespace

int main() {
    try {
        test_rnnoise_alignment_plan_arithmetic();
        test_rnnoise_crop_recovers_input_alignment();
        test_zipenhancer_fade_envelope_is_complementary();
        test_zipenhancer_fade_attenuates_the_seam_step();
        test_zipenhancer_segmented_coverage_has_no_tail_hole();
        test_zipenhancer_whole_file_lengths();
        test_flashsr_gain_preserves_input_level();
        test_flashsr_gain_safety_and_silence();
        std::cout << "enhancement_alignment_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "enhancement_alignment_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
