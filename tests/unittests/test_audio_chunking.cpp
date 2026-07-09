#include "engine/framework/audio/chunking.h"

#include "test_assert.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

engine::runtime::SpeechSegment speech(int64_t start, int64_t end) {
    engine::runtime::SpeechSegment segment;
    segment.span.start_sample = start;
    segment.span.end_sample = end;
    segment.confidence = 1.0F;
    return segment;
}

engine::runtime::WordTimestamp word(const std::string & text, int64_t start, int64_t end) {
    engine::runtime::WordTimestamp timestamp;
    timestamp.word = text;
    timestamp.span.start_sample = start;
    timestamp.span.end_sample = end;
    return timestamp;
}

class MockVadSession final : public engine::runtime::IOfflineVoiceTaskSession {
public:
    std::string family() const override {
        return "mock_vad";
    }

    engine::runtime::VoiceTaskKind task_kind() const override {
        return engine::runtime::VoiceTaskKind::Vad;
    }

    engine::runtime::RunMode run_mode() const override {
        return engine::runtime::RunMode::Offline;
    }

    void prepare(const engine::runtime::SessionPreparationRequest & request) override {
        engine::test::require(request.audio.has_value(), "mock VAD prepare audio contract");
        prepared = true;
    }

    engine::runtime::TaskResult run(const engine::runtime::TaskRequest & request) override {
        engine::test::require(prepared, "mock VAD run after prepare");
        engine::test::require(request.audio_input.has_value(), "mock VAD run audio input");
        ++runs;
        engine::runtime::TaskResult result;
        result.speech_segments = segments;
        return result;
    }

    bool prepared = false;
    int runs = 0;
    std::vector<engine::runtime::SpeechSegment> segments;
};

void require_span(
    const engine::runtime::TimeSpan & span,
    int64_t expected_start,
    int64_t expected_end,
    const std::string & label) {
    engine::test::require_eq(span.start_sample, expected_start, label + " start");
    engine::test::require_eq(span.end_sample, expected_end, label + " end");
}

void require_throws(const std::function<void()> & fn, const std::string & label) {
    bool threw = false;
    try {
        fn();
    } catch (const std::runtime_error &) {
        threw = true;
    }
    engine::test::require(threw, label + " did not throw");
}

void require_chunk_invariants(
    const std::vector<engine::runtime::TimeSpan> & chunks,
    int64_t audio_samples,
    int64_t max_chunk_samples,
    const std::string & label) {
    int64_t previous_end = 0;
    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto & chunk = chunks[i];
        engine::test::require(chunk.start_sample >= 0, label + " chunk start before zero");
        engine::test::require(chunk.end_sample <= audio_samples, label + " chunk end outside audio");
        engine::test::require(chunk.start_sample < chunk.end_sample, label + " empty chunk");
        engine::test::require(chunk.start_sample >= previous_end, label + " chunks overlap or are unsorted");
        engine::test::require(
            chunk.end_sample - chunk.start_sample <= max_chunk_samples,
            label + " chunk exceeds max length");
        previous_end = chunk.end_sample;
    }
}

void require_range_covered(
    const std::vector<engine::runtime::TimeSpan> & chunks,
    int64_t start,
    int64_t end,
    const std::string & label) {
    int64_t cursor = start;
    for (const auto & chunk : chunks) {
        if (chunk.end_sample <= cursor) {
            continue;
        }
        if (chunk.start_sample > cursor) {
            break;
        }
        cursor = std::max(cursor, std::min(chunk.end_sample, end));
        if (cursor >= end) {
            return;
        }
    }
    engine::test::require(false, label + " range is not fully covered");
}

void require_no_forbidden_gap_inside_chunks(
    const std::vector<engine::runtime::TimeSpan> & chunks,
    const std::vector<engine::runtime::TimeSpan> & speech_runs,
    int64_t allowed_gap,
    const std::string & label) {
    for (const auto & chunk : chunks) {
        for (size_t i = 1; i < speech_runs.size(); ++i) {
            const int64_t gap_start = speech_runs[i - 1].end_sample;
            const int64_t gap_end = speech_runs[i].start_sample;
            const int64_t gap = gap_end - gap_start;
            if (gap > allowed_gap &&
                chunk.start_sample < gap_start &&
                chunk.end_sample > gap_end) {
                engine::test::require(false, label + " chunk crosses forbidden silence gap");
            }
        }
    }
}

void test_fixed_chunks_plan_start_aligned_tail() {
    const engine::audio::AudioChunkSpec spec{
        4,
        4,
        engine::audio::AudioChunkPadMode::Zero,
        engine::audio::AudioChunkTailAlignment::Start,
        0,
    };
    const auto spans = engine::audio::plan_audio_chunks(10, spec);
    engine::test::require_eq(spans.size(), static_cast<size_t>(3), "fixed start-aligned span count");
    engine::test::require_eq(spans[0].index, 0, "span 0 index");
    engine::test::require_eq(spans[0].output_start_sample, 0, "span 0 output start");
    engine::test::require_eq(spans[0].copy_start_sample, 0, "span 0 copy start");
    engine::test::require_eq(spans[0].valid_start_in_chunk, 0, "span 0 valid start");
    engine::test::require_eq(spans[0].valid_samples, 4, "span 0 valid samples");
    engine::test::require_eq(spans[1].output_start_sample, 4, "span 1 output start");
    engine::test::require_eq(spans[1].valid_samples, 4, "span 1 valid samples");
    engine::test::require_eq(spans[2].output_start_sample, 8, "span 2 output start");
    engine::test::require_eq(spans[2].copy_start_sample, 8, "span 2 copy start");
    engine::test::require_eq(spans[2].valid_start_in_chunk, 0, "span 2 valid start");
    engine::test::require_eq(spans[2].valid_samples, 2, "span 2 valid samples");
}

void test_fixed_chunks_plan_overlapping_windows() {
    const engine::audio::AudioChunkSpec spec{
        4,
        2,
        engine::audio::AudioChunkPadMode::Zero,
        engine::audio::AudioChunkTailAlignment::Start,
        0,
    };
    const auto spans = engine::audio::plan_audio_chunks(9, spec);
    engine::test::require_eq(spans.size(), static_cast<size_t>(5), "overlap fixed span count");
    require_span({spans[0].output_start_sample, spans[0].output_start_sample + spans[0].valid_samples}, 0, 4, "overlap span 0 output");
    require_span({spans[1].output_start_sample, spans[1].output_start_sample + spans[1].valid_samples}, 2, 6, "overlap span 1 output");
    require_span({spans[2].output_start_sample, spans[2].output_start_sample + spans[2].valid_samples}, 4, 8, "overlap span 2 output");
    require_span({spans[3].output_start_sample, spans[3].output_start_sample + spans[3].valid_samples}, 6, 9, "overlap span 3 output");
    require_span({spans[4].output_start_sample, spans[4].output_start_sample + spans[4].valid_samples}, 8, 9, "overlap span 4 output");
    for (size_t i = 0; i < spans.size(); ++i) {
        engine::test::require_eq(spans[i].index, static_cast<int64_t>(i), "overlap span index " + std::to_string(i));
        engine::test::require(spans[i].valid_samples > 0, "overlap span has no valid samples");
        engine::test::require(spans[i].valid_start_in_chunk >= 0, "overlap span valid start before zero");
        engine::test::require(
            spans[i].valid_start_in_chunk + spans[i].valid_samples <= spec.chunk_samples,
            "overlap span valid range exceeds chunk");
    }
}

void test_fixed_chunks_plan_centered_tail_and_copy_zero_pad() {
    const engine::audio::AudioChunkSpec spec{
        4,
        4,
        engine::audio::AudioChunkPadMode::Zero,
        engine::audio::AudioChunkTailAlignment::Center,
        0,
    };
    const auto spans = engine::audio::plan_audio_chunks(10, spec);
    engine::test::require_eq(spans.size(), static_cast<size_t>(3), "fixed centered span count");
    engine::test::require_eq(spans[2].output_start_sample, 8, "centered tail output start");
    engine::test::require_eq(spans[2].copy_start_sample, 7, "centered tail copy start");
    engine::test::require_eq(spans[2].valid_start_in_chunk, 1, "centered tail valid start");
    engine::test::require_eq(spans[2].valid_samples, 2, "centered tail valid samples");

    const std::vector<float> input{
        0.0F, 10.0F,
        1.0F, 11.0F,
        2.0F, 12.0F,
        3.0F, 13.0F,
        4.0F, 14.0F,
        5.0F, 15.0F,
        6.0F, 16.0F,
        7.0F, 17.0F,
        8.0F, 18.0F,
        9.0F, 19.0F,
    };
    std::vector<float> output(8, -1.0F);
    engine::audio::copy_interleaved_chunk_to_planar(output, input, 2, 10, spans[2], spec);
    const std::vector<float> expected{
        7.0F, 8.0F, 9.0F, 0.0F,
        17.0F, 18.0F, 19.0F, 0.0F,
    };
    engine::test::require_eq(output.size(), expected.size(), "centered copy output size");
    for (size_t i = 0; i < output.size(); ++i) {
        engine::test::require_eq(output[i], expected[i], "centered copy sample " + std::to_string(i));
    }
}

void test_copy_planar_chunk_reflect_padding() {
    const engine::audio::AudioChunkSpec spec{
        5,
        5,
        engine::audio::AudioChunkPadMode::Reflect,
        engine::audio::AudioChunkTailAlignment::Start,
        0,
    };
    const engine::audio::AudioChunkSpan span{
        0,
        0,
        3,
        -2,
        2,
    };
    const std::vector<float> input{
        10.0F, 11.0F, 12.0F, 13.0F,
        20.0F, 21.0F, 22.0F, 23.0F,
    };
    std::vector<float> output(10, -1.0F);
    engine::audio::copy_planar_chunk(output, input, 2, 4, span, spec);
    const std::vector<float> expected{
        12.0F, 11.0F, 10.0F, 11.0F, 12.0F,
        22.0F, 21.0F, 20.0F, 21.0F, 22.0F,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        engine::test::require_eq(output[i], expected[i], "reflect planar sample " + std::to_string(i));
    }
}

void test_overlap_add_and_normalize_shared_counter() {
    std::vector<float> output(6, 0.0F);
    std::vector<float> weights(6, 0.0F);
    const std::vector<float> window{1.0F, 0.5F, 0.5F, 1.0F};
    const engine::audio::AudioChunkSpan first{
        0,
        0,
        4,
        0,
        0,
    };
    const engine::audio::AudioChunkSpan second{
        1,
        2,
        4,
        2,
        0,
    };
    engine::audio::overlap_add_planar_chunk(
        output,
        weights,
        std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F},
        1,
        6,
        first,
        window,
        engine::audio::AudioChunkCounterMode::SharedAcrossLanes);
    engine::audio::overlap_add_planar_chunk(
        output,
        weights,
        std::vector<float>{10.0F, 20.0F, 30.0F, 40.0F},
        1,
        6,
        second,
        window,
        engine::audio::AudioChunkCounterMode::SharedAcrossLanes);
    engine::audio::normalize_overlap_added_planar(
        output,
        weights,
        1,
        6,
        engine::audio::AudioChunkCounterMode::SharedAcrossLanes);
    const std::vector<float> expected{
        1.0F,
        2.0F,
        (3.0F * 0.5F + 10.0F * 1.0F) / 1.5F,
        (4.0F * 1.0F + 20.0F * 0.5F) / 1.5F,
        30.0F,
        40.0F,
    };
    for (size_t i = 0; i < expected.size(); ++i) {
        engine::test::require_close(output[i], expected[i], 1.0e-6F, "overlap-add sample " + std::to_string(i));
    }
}

void test_fixed_chunks_validate_inputs() {
    require_throws([]() {
        (void) engine::audio::plan_audio_chunks(0, {4, 4});
    }, "zero fixed chunk input samples");
    require_throws([]() {
        (void) engine::audio::plan_audio_chunks(10, {0, 4});
    }, "zero fixed chunk size");
    require_throws([]() {
        (void) engine::audio::plan_audio_chunks(10, {4, 0});
    }, "zero fixed hop size");
    require_throws([]() {
        (void) engine::audio::plan_audio_chunks(10, {4, 5});
    }, "fixed hop larger than chunk");
    require_throws([]() {
        const std::vector<float> input{1.0F, 2.0F, 3.0F};
        std::vector<float> output(4, 0.0F);
        engine::audio::copy_planar_chunk(output, input, 2, 2, {0, 0, 2, 0, 0}, {2, 2});
    }, "copy planar input size mismatch");
    require_throws([]() {
        std::vector<float> output(3, 0.0F);
        std::vector<float> weights(2, 0.0F);
        engine::audio::overlap_add_planar_chunk(
            output,
            weights,
            std::vector<float>{1.0F, 2.0F},
            1,
            3,
            {0, 2, 2, 0, 0},
            std::vector<float>{1.0F, 1.0F},
            engine::audio::AudioChunkCounterMode::SharedAcrossLanes);
    }, "overlap-add span past output");
}

void test_vad_chunks_sort_pad_and_merge() {
    const std::vector<engine::runtime::SpeechSegment> segments{
        speech(400, 500),
        speech(100, 200),
        speech(225, 260),
    };
    const engine::audio::VadAudioChunkOptions options{
        1000,
        40,
        10,
    };
    const auto chunks = engine::audio::plan_vad_audio_chunks(segments, 1000, options);
    engine::test::require_eq(chunks.size(), static_cast<size_t>(2), "VAD chunk count");
    require_chunk_invariants(chunks, 1000, options.max_chunk_samples, "sort-pad-merge");
    require_span(chunks[0], 90, 270, "first VAD chunk");
    require_span(chunks[1], 390, 510, "second VAD chunk");
    require_range_covered(chunks, 90, 210, "first padded segment");
    require_range_covered(chunks, 215, 270, "second padded segment");
    require_range_covered(chunks, 390, 510, "third padded segment");
}

void test_vad_chunks_clamp_padding_to_audio_bounds() {
    const std::vector<engine::runtime::SpeechSegment> segments{
        speech(5, 20),
        speech(985, 995),
    };
    const engine::audio::VadAudioChunkOptions options{
        200,
        0,
        20,
    };
    const auto chunks = engine::audio::plan_vad_audio_chunks(segments, 1000, options);
    engine::test::require_eq(chunks.size(), static_cast<size_t>(2), "clamped VAD chunk count");
    require_chunk_invariants(chunks, 1000, options.max_chunk_samples, "clamped-padding");
    require_span(chunks[0], 0, 40, "left-clamped VAD chunk");
    require_span(chunks[1], 965, 1000, "right-clamped VAD chunk");
}

void test_vad_chunks_respect_max_when_merging() {
    const std::vector<engine::runtime::SpeechSegment> segments{
        speech(0, 100),
        speech(150, 250),
        speech(300, 400),
    };
    const engine::audio::VadAudioChunkOptions options{
        260,
        100,
        0,
    };
    const auto chunks = engine::audio::plan_vad_audio_chunks(segments, 1000, options);
    engine::test::require_eq(chunks.size(), static_cast<size_t>(2), "max-bounded VAD chunk count");
    require_chunk_invariants(chunks, 1000, options.max_chunk_samples, "max-bounded");
    require_span(chunks[0], 0, 250, "max-bounded first chunk");
    require_span(chunks[1], 300, 400, "max-bounded second chunk");
}

void test_vad_chunks_merge_gap_boundary() {
    const std::vector<engine::runtime::SpeechSegment> segments{
        speech(0, 100),
        speech(150, 200),
        speech(251, 300),
    };
    const engine::audio::VadAudioChunkOptions options{
        500,
        50,
        0,
    };
    const auto chunks = engine::audio::plan_vad_audio_chunks(segments, 1000, options);
    engine::test::require_eq(chunks.size(), static_cast<size_t>(2), "gap-boundary VAD chunk count");
    require_chunk_invariants(chunks, 1000, options.max_chunk_samples, "gap-boundary");
    require_span(chunks[0], 0, 200, "gap equal to threshold should merge");
    require_span(chunks[1], 251, 300, "gap above threshold should not merge");
}

void test_vad_chunks_split_long_speech() {
    const std::vector<engine::runtime::SpeechSegment> segments{
        speech(0, 1000),
    };
    const engine::audio::VadAudioChunkOptions options{
        400,
        0,
        0,
    };
    const auto chunks = engine::audio::plan_vad_audio_chunks(segments, 1000, options);
    engine::test::require_eq(chunks.size(), static_cast<size_t>(3), "split VAD chunk count");
    require_chunk_invariants(chunks, 1000, options.max_chunk_samples, "split-long");
    require_span(chunks[0], 0, 400, "split first chunk");
    require_span(chunks[1], 400, 800, "split second chunk");
    require_span(chunks[2], 800, 1000, "split third chunk");
    require_range_covered(chunks, 0, 1000, "long speech");
}

void test_vad_chunks_do_not_overlap_when_max_blocks_merge() {
    const std::vector<engine::runtime::SpeechSegment> segments{
        speech(0, 300),
        speech(250, 500),
    };
    const engine::audio::VadAudioChunkOptions options{
        400,
        100,
        0,
    };
    const auto chunks = engine::audio::plan_vad_audio_chunks(segments, 1000, options);
    engine::test::require_eq(chunks.size(), static_cast<size_t>(2), "non-overlap chunk count");
    require_chunk_invariants(chunks, 1000, options.max_chunk_samples, "non-overlap");
    require_span(chunks[0], 0, 400, "non-overlap first chunk");
    require_span(chunks[1], 400, 500, "non-overlap second chunk");
    require_range_covered(chunks, 0, 500, "overlapping input speech");
}

void test_vad_chunks_nested_segments_keep_full_coverage() {
    const std::vector<engine::runtime::SpeechSegment> segments{
        speech(100, 700),
        speech(200, 300),
        speech(650, 900),
    };
    const engine::audio::VadAudioChunkOptions options{
        500,
        100,
        0,
    };
    const auto chunks = engine::audio::plan_vad_audio_chunks(segments, 1000, options);
    engine::test::require_eq(chunks.size(), static_cast<size_t>(2), "nested VAD chunk count");
    require_chunk_invariants(chunks, 1000, options.max_chunk_samples, "nested");
    require_span(chunks[0], 100, 600, "nested first chunk");
    require_span(chunks[1], 600, 900, "nested second chunk");
    require_range_covered(chunks, 100, 900, "nested and overlapping speech");
}

void test_vad_chunks_padding_overlap_merges_without_duplicate_coverage() {
    const std::vector<engine::runtime::SpeechSegment> segments{
        speech(100, 150),
        speech(175, 220),
        speech(245, 280),
    };
    const engine::audio::VadAudioChunkOptions options{
        500,
        0,
        20,
    };
    const auto chunks = engine::audio::plan_vad_audio_chunks(segments, 1000, options);
    engine::test::require_eq(chunks.size(), static_cast<size_t>(1), "padding-overlap chunk count");
    require_chunk_invariants(chunks, 1000, options.max_chunk_samples, "padding-overlap");
    require_span(chunks[0], 80, 300, "padding-overlap merged chunk");
    require_range_covered(chunks, 80, 300, "padding-overlap speech run");
}

void test_vad_chunks_keep_large_silence_gap_outside_chunks() {
    const std::vector<engine::runtime::SpeechSegment> segments{
        speech(100, 200),
        speech(500, 600),
    };
    const engine::audio::VadAudioChunkOptions options{
        1000,
        100,
        20,
    };
    const auto chunks = engine::audio::plan_vad_audio_chunks(segments, 1000, options);
    engine::test::require_eq(chunks.size(), static_cast<size_t>(2), "large-gap chunk count");
    require_chunk_invariants(chunks, 1000, options.max_chunk_samples, "large-gap");
    require_span(chunks[0], 80, 220, "large-gap first chunk");
    require_span(chunks[1], 480, 620, "large-gap second chunk");
    require_no_forbidden_gap_inside_chunks(chunks, {{80, 220}, {480, 620}}, options.merge_gap_samples, "large-gap");
}

void test_vad_chunks_close_run_split_at_max_without_losing_speech() {
    const std::vector<engine::runtime::SpeechSegment> segments{
        speech(0, 300),
        speech(350, 650),
        speech(700, 1000),
    };
    const engine::audio::VadAudioChunkOptions options{
        450,
        50,
        0,
    };
    const auto chunks = engine::audio::plan_vad_audio_chunks(segments, 1000, options);
    engine::test::require_eq(chunks.size(), static_cast<size_t>(3), "close-run split chunk count");
    require_chunk_invariants(chunks, 1000, options.max_chunk_samples, "close-run split");
    require_range_covered(chunks, 0, 300, "close-run first speech");
    require_range_covered(chunks, 350, 650, "close-run second speech");
    require_range_covered(chunks, 700, 1000, "close-run third speech");
    require_no_forbidden_gap_inside_chunks(chunks, {{0, 300}, {350, 650}, {700, 1000}}, options.merge_gap_samples, "close-run split");
}

void test_vad_chunks_stress_mixed_unsorted_segments() {
    const std::vector<engine::runtime::SpeechSegment> segments{
        speech(950, 990),
        speech(120, 200),
        speech(500, 900),
        speech(180, 260),
        speech(420, 520),
        speech(0, 30),
    };
    const engine::audio::VadAudioChunkOptions options{
        250,
        50,
        10,
    };
    const auto chunks = engine::audio::plan_vad_audio_chunks(segments, 1000, options);
    require_chunk_invariants(chunks, 1000, options.max_chunk_samples, "stress");
    require_range_covered(chunks, 0, 40, "stress first padded speech");
    require_range_covered(chunks, 110, 270, "stress merged early speech");
    require_range_covered(chunks, 410, 910, "stress long overlapping speech");
    require_range_covered(chunks, 940, 1000, "stress final padded speech");
}

void test_vad_chunks_validate_inputs() {
    const auto empty_chunks = engine::audio::plan_vad_audio_chunks({}, 1000, {400, 0, 0});
    engine::test::require(empty_chunks.empty(), "empty VAD segments should produce no chunks");

    require_throws([]() {
        const std::vector<engine::runtime::SpeechSegment> segments{
            speech(100, 100),
        };
        (void) engine::audio::plan_vad_audio_chunks(segments, 1000, {400, 0, 0});
    }, "invalid speech segment");

    require_throws([]() {
        const std::vector<engine::runtime::SpeechSegment> segments{
            speech(100, 200),
        };
        (void) engine::audio::plan_vad_audio_chunks(segments, 1000, {0, 0, 0});
    }, "invalid max chunk size");

    require_throws([]() {
        const std::vector<engine::runtime::SpeechSegment> segments{
            speech(100, 200),
        };
        (void) engine::audio::plan_vad_audio_chunks(segments, 1000, {400, -1, 0});
    }, "negative merge gap");

    require_throws([]() {
        const std::vector<engine::runtime::SpeechSegment> segments{
            speech(100, 200),
        };
        (void) engine::audio::plan_vad_audio_chunks(segments, 1000, {400, 0, -1});
    }, "negative padding");

    require_throws([]() {
        const std::vector<engine::runtime::SpeechSegment> segments{
            speech(900, 1001),
        };
        (void) engine::audio::plan_vad_audio_chunks(segments, 1000, {400, 0, 0});
    }, "speech segment past audio end");

    require_throws([]() {
        const std::vector<engine::runtime::SpeechSegment> segments{
            speech(-1, 10),
        };
        (void) engine::audio::plan_vad_audio_chunks(segments, 1000, {400, 0, 0});
    }, "speech segment before audio start");

    require_throws([]() {
        const std::vector<engine::runtime::SpeechSegment> segments{
            speech(100, 200),
        };
        (void) engine::audio::plan_vad_audio_chunks(segments, 0, {400, 0, 0});
    }, "invalid audio length");
}

void test_vad_session_planner_runs_vad_internally() {
    engine::runtime::AudioBuffer audio;
    audio.sample_rate = 100;
    audio.channels = 1;
    audio.samples.assign(1000, 0.0F);

    MockVadSession vad;
    vad.segments = {
        speech(100, 200),
        speech(230, 300),
        speech(700, 760),
    };

    const auto chunks = engine::audio::plan_vad_audio_chunks(audio, vad, {250, 50, 10});
    engine::test::require(vad.prepared, "mock VAD was prepared");
    engine::test::require_eq(vad.runs, 1, "mock VAD run count");
    engine::test::require_eq(chunks.size(), static_cast<size_t>(2), "mock VAD planned chunk count");
    require_span(chunks[0], 90, 310, "mock VAD first chunk");
    require_span(chunks[1], 690, 770, "mock VAD second chunk");
}

void test_audio_chunk_mode_parser() {
    std::unordered_map<std::string, std::string> options;
    engine::test::require(
        engine::audio::parse_audio_chunk_mode(options) == engine::audio::AudioChunkMode::Auto,
        "default audio chunk mode is auto");
    options["audio_chunk_mode"] = "fixed";
    engine::test::require(
        engine::audio::parse_audio_chunk_mode(options) == engine::audio::AudioChunkMode::Fixed,
        "fixed audio chunk mode");
    options["audio_chunk_mode"] = "vad";
    engine::test::require(
        engine::audio::parse_audio_chunk_mode(options) == engine::audio::AudioChunkMode::Vad,
        "vad audio chunk mode");
    options["audio_chunk_mode"] = "quiet_energy";
    engine::test::require(
        engine::audio::parse_audio_chunk_mode(options) == engine::audio::AudioChunkMode::QuietEnergy,
        "quiet-energy audio chunk mode");
    options["audio_chunk_mode"] = "none";
    engine::test::require(
        engine::audio::parse_audio_chunk_mode(options) == engine::audio::AudioChunkMode::None,
        "none audio chunk mode");
    options["audio_chunk_mode"] = "json";
    require_throws([&]() {
        (void) engine::audio::parse_audio_chunk_mode(options);
    }, "invalid audio chunk mode");
}

void test_slice_audio_buffer_preserves_channels() {
    engine::runtime::AudioBuffer audio;
    audio.sample_rate = 16000;
    audio.channels = 2;
    audio.samples = {
        0.0F, 0.1F,
        1.0F, 1.1F,
        2.0F, 2.1F,
        3.0F, 3.1F,
    };
    const auto slice = engine::audio::slice_audio_buffer(audio, {1, 3});
    engine::test::require_eq(slice.sample_rate, 16000, "slice sample rate");
    engine::test::require_eq(slice.channels, 2, "slice channels");
    engine::test::require_eq(slice.samples.size(), static_cast<size_t>(4), "slice sample count");
    engine::test::require_eq(slice.samples[0], 1.0F, "slice sample 0");
    engine::test::require_eq(slice.samples[1], 1.1F, "slice sample 1");
    engine::test::require_eq(slice.samples[2], 2.0F, "slice sample 2");
    engine::test::require_eq(slice.samples[3], 2.1F, "slice sample 3");
}

void test_slice_audio_buffer_validate_inputs() {
    engine::runtime::AudioBuffer audio;
    audio.sample_rate = 16000;
    audio.channels = 2;
    audio.samples = {
        0.0F, 0.1F,
        1.0F,
    };
    require_throws([&]() {
        (void) engine::audio::slice_audio_buffer(audio, {0, 1});
    }, "slice non-divisible interleaved audio");

    audio.samples = {
        0.0F, 0.1F,
        1.0F, 1.1F,
    };
    require_throws([&]() {
        (void) engine::audio::slice_audio_buffer(audio, {1, 1});
    }, "slice empty span");
    require_throws([&]() {
        (void) engine::audio::slice_audio_buffer(audio, {-1, 1});
    }, "slice negative start");
    require_throws([&]() {
        (void) engine::audio::slice_audio_buffer(audio, {0, 3});
    }, "slice past audio end");

    audio.sample_rate = 0;
    require_throws([&]() {
        (void) engine::audio::slice_audio_buffer(audio, {0, 1});
    }, "slice invalid sample rate");
}

void test_chunk_word_timestamp_merge_offsets_and_clips() {
    std::vector<engine::runtime::WordTimestamp> merged;
    const std::vector<engine::runtime::WordTimestamp> local{
        word("left", -10, 30),
        word("middle", 40, 80),
        word("right", 90, 130),
    };

    engine::audio::append_chunk_word_timestamps(
        merged,
        local,
        engine::runtime::TimeSpan{1000, 1100});

    engine::test::require_eq(merged.size(), static_cast<size_t>(3), "merged chunk word count");
    engine::test::require_eq(merged[0].word, std::string("left"), "left word kept");
    require_span(merged[0].span, 1000, 1030, "left word clipped");
    engine::test::require_eq(merged[1].word, std::string("middle"), "middle word kept");
    require_span(merged[1].span, 1040, 1080, "middle word offset");
    engine::test::require_eq(merged[2].word, std::string("right"), "right word kept");
    require_span(merged[2].span, 1090, 1100, "right word clipped");
}

void test_chunk_word_timestamp_merge_appends_multiple_chunks() {
    std::vector<engine::runtime::WordTimestamp> merged;
    engine::audio::append_chunk_word_timestamps(
        merged,
        {word("alpha", 5, 25), word("beta", 30, 50)},
        engine::runtime::TimeSpan{0, 100});
    engine::audio::append_chunk_word_timestamps(
        merged,
        {word("gamma", 0, 20)},
        engine::runtime::TimeSpan{100, 150});

    engine::test::require_eq(merged.size(), static_cast<size_t>(3), "multi-chunk merged word count");
    require_span(merged[0].span, 5, 25, "first chunk alpha");
    require_span(merged[1].span, 30, 50, "first chunk beta");
    require_span(merged[2].span, 100, 120, "second chunk gamma");
}

void test_chunk_word_timestamp_merge_keeps_non_overlapping_source_span() {
    std::vector<engine::runtime::WordTimestamp> merged;
    engine::audio::append_chunk_word_timestamps(
        merged,
        {
            word("left_context", 20, 60),
            word("kept_boundary", 100, 180),
            word("kept_inside", 260, 320),
            word("right_context", 400, 450),
        },
        engine::runtime::TimeSpan{900, 1400},
        engine::runtime::TimeSpan{1000, 1300});

    engine::test::require_eq(merged.size(), static_cast<size_t>(2), "keep span merged word count");
    engine::test::require_eq(merged[0].word, std::string("kept_boundary"), "boundary word kept");
    require_span(merged[0].span, 1000, 1080, "boundary word full source span");
    engine::test::require_eq(merged[1].word, std::string("kept_inside"), "inside word kept");
    require_span(merged[1].span, 1160, 1220, "inside word source span");
}

void test_chunk_word_timestamp_merge_rejects_invalid_spans() {
    require_throws(
        []() {
            std::vector<engine::runtime::WordTimestamp> merged;
            engine::audio::append_chunk_word_timestamps(
                merged,
                {word("bad", 20, 10)},
                engine::runtime::TimeSpan{0, 100});
        },
        "inverted word timestamp");

    require_throws(
        []() {
            std::vector<engine::runtime::WordTimestamp> merged;
            engine::audio::append_chunk_word_timestamps(
                merged,
                {word("bad", 0, 10)},
                engine::runtime::TimeSpan{100, 90});
        },
        "inverted chunk span");

    require_throws(
        []() {
            std::vector<engine::runtime::WordTimestamp> merged;
            engine::audio::append_chunk_word_timestamps(
                merged,
                {word("outside", 120, 140)},
                engine::runtime::TimeSpan{1000, 1100});
        },
        "word outside chunk");

    require_throws(
        []() {
            std::vector<engine::runtime::WordTimestamp> merged;
            engine::audio::append_chunk_word_timestamps(
                merged,
                {word("bad", 0, 10)},
                engine::runtime::TimeSpan{1000, 1100},
                engine::runtime::TimeSpan{900, 1100});
        },
        "keep span outside source");
}

}  // namespace

int main() {
    try {
        test_fixed_chunks_plan_start_aligned_tail();
        test_fixed_chunks_plan_overlapping_windows();
        test_fixed_chunks_plan_centered_tail_and_copy_zero_pad();
        test_copy_planar_chunk_reflect_padding();
        test_overlap_add_and_normalize_shared_counter();
        test_fixed_chunks_validate_inputs();
        test_vad_chunks_sort_pad_and_merge();
        test_vad_chunks_clamp_padding_to_audio_bounds();
        test_vad_chunks_respect_max_when_merging();
        test_vad_chunks_merge_gap_boundary();
        test_vad_chunks_split_long_speech();
        test_vad_chunks_do_not_overlap_when_max_blocks_merge();
        test_vad_chunks_nested_segments_keep_full_coverage();
        test_vad_chunks_padding_overlap_merges_without_duplicate_coverage();
        test_vad_chunks_keep_large_silence_gap_outside_chunks();
        test_vad_chunks_close_run_split_at_max_without_losing_speech();
        test_vad_chunks_stress_mixed_unsorted_segments();
        test_vad_chunks_validate_inputs();
        test_vad_session_planner_runs_vad_internally();
        test_audio_chunk_mode_parser();
        test_slice_audio_buffer_preserves_channels();
        test_slice_audio_buffer_validate_inputs();
        test_chunk_word_timestamp_merge_offsets_and_clips();
        test_chunk_word_timestamp_merge_appends_multiple_chunks();
        test_chunk_word_timestamp_merge_keeps_non_overlapping_source_span();
        test_chunk_word_timestamp_merge_rejects_invalid_spans();
        std::cout << "audio_chunking_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "audio_chunking_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
