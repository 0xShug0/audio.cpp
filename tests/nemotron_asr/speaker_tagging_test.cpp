// Speaker-tagged transcription helpers. The segment cases port NVIDIA-NeMo/Speech
// tests/collections/speaker_tasks/utils/test_spk_tagged_asr_utils.py (cf724ac).
#include "engine/models/nemotron_asr/masked_asr.h"
#include "engine/models/nemotron_asr/speaker_tagging.h"

#include "engine/models/nemotron_3_diar/streaming.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

using namespace engine::models::nemotron_asr;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

bool near(double a, double b) {
    return std::fabs(a - b) < 1.0e-6;
}

template <typename F>
bool throws(F && f) {
    try {
        f();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

using Update = std::tuple<std::string, std::vector<int64_t>, double>;

// NeMo run_parallel_hypothesis_updates: one speaker, uppercase_first_letter=False,
// decoded_length = max(timestamps) + 1.
SpeakerSegmentBuilder run_updates(const std::vector<Update> & updates, double gap_sec) {
    SpeakerSegmentBuilder builder(gap_sec, 10.0, false);
    for (const auto & [text, frames, offset] : updates) {
        const int64_t decoded = frames.empty() ? 0 : *std::max_element(frames.begin(), frames.end()) + 1;
        builder.update_hypothesis(0, text, frames, decoded, offset);
    }
    return builder;
}

std::string joined(const std::vector<SpeakerSegment> & segments) {
    std::string out;
    for (size_t i = 0; i < segments.size(); ++i) out += (i ? " " : "") + segments[i].words;
    return out;
}

void test_word_continuations_never_create_segment_boundaries() {
    require(joined(run_updates({{"Lind", {0, 1, 2, 3}, 0.0}, {"Linda", {0, 1, 2, 3, 30}, 2.0}}, 0.1).segments()) ==
                "Linda",
        "Lind -> Linda");
    require(joined(run_updates({{"Don", {0, 1, 2}, 0.0},
                                {"Don't start", {0, 1, 2, 30, 31, 32, 40, 41, 42, 43, 44}, 2.0}},
                               0.1).segments()) == "Don't start",
        "Don -> Don't start");
}

void test_continuation_timestamps_respect_inactive_gaps() {
    struct Case {
        std::vector<Update> updates;
        std::vector<std::tuple<std::string, double, double>> expected;
    };
    std::vector<int64_t> eleven;
    for (int64_t i = 0; i < 11; ++i) eleven.push_back(i);
    const std::vector<Case> cases = {
        {{{"hello", {0, 1, 2, 3, 4}, 0.0}, {"hello.", {0, 1, 2, 3, 4, 5}, 10.0}}, {{"hello.", 0.0, 0.4}}},
        {{{"Don", {0, 1, 2}, 0.0}, {"Don't start", eleven, 10.0}}, {{"Don't", 0.0, 0.24}, {"start", 10.0, 10.64}}},
        {{{"Lind", {0, 1, 2, 3}, 0.0}, {"Linda", {0, 1, 2, 3, 4}, 10.0}}, {{"Linda", 0.0, 0.32}}},
        {{{"Lind", {0, 1, 2, 3}, 0.0}, {"Linda", {0, 1, 2, 3, 4}, 0.32}}, {{"Linda", 0.0, 0.4}}},
    };
    for (size_t c = 0; c < cases.size(); ++c) {
        const auto segments = run_updates(cases[c].updates, 0.5).segments();
        require(segments.size() == cases[c].expected.size(), "gap case " + std::to_string(c) + " segment count");
        for (size_t i = 0; i < segments.size(); ++i) {
            const auto & [text, start, end] = cases[c].expected[i];
            require(segments[i].words == text && near(segments[i].start, start) && near(segments[i].end, end),
                "gap case " + std::to_string(c) + " segment " + std::to_string(i) + ": " + segments[i].words + " " +
                    std::to_string(segments[i].start) + " " + std::to_string(segments[i].end));
        }
    }
}

void test_clean_word_boundary_can_create_segment() {
    for (const std::string first : {"hello", "hello "}) {
        std::vector<int64_t> frames = {0, 1, 2, 3, 4};
        if (first.back() == ' ') frames.push_back(5);
        const auto segments = run_updates(
            {{first, frames, 0.0}, {"hello world", {0, 1, 2, 3, 4, 5, 30, 31, 32, 33, 34}, 2.0}}, 0.5).segments();
        require(segments.size() == 2 && segments[0].words.substr(0, 5) == "hello" && segments[0].words.size() <= 6 &&
                    segments[1].words == "world",  // NeMo keeps the trailing space of "hello "
            "clean boundary after '" + first + "'");
    }
}

void test_punctuation_and_repeated_prefixes_preserve_text() {
    require(joined(run_updates({{"hello", {0, 1, 2, 3, 4}, 0.0},
                                {"hello,", {0, 1, 2, 3, 4, 20}, 1.0},
                                {"hello, world", {0, 1, 2, 3, 4, 20, 21, 30, 31, 32, 33, 34}, 2.0}},
                               0.1).segments()) == "hello, world",
        "punctuation");
    require(joined(run_updates({{"go go", {0, 1, 2, 3, 4}, 0.0}, {"go go now", {0, 1, 2, 3, 4, 5, 20, 21, 22}, 1.0}},
                               0.1).segments()) == "go go now",
        "repeated prefix");
}

void test_word_stream_is_invariant_while_segment_layout_changes_with_threshold() {
    const std::vector<Update> updates = {
        {"Lind", {0, 1, 2, 3}, 0.0},
        {"Linda", {0, 1, 2, 3, 30}, 2.0},
        {"Linda next", {0, 1, 2, 3, 30, 60, 61, 62, 63, 64}, 4.08},
    };
    require(joined(run_updates(updates, 30.0).segments()) == "Linda next" &&
                run_updates(updates, 30.0).segments().size() == 1,
        "one segment at 30 s");
    const auto split = run_updates(updates, 0.0).segments();
    require(split.size() == 2 && split[0].words == "Linda" && split[1].words == "next", "two segments at 0 s");
}

void test_decoded_length_advances_when_chunk_has_no_new_text() {
    SpeakerSegmentBuilder builder(1.0, 10.0, false);
    builder.update_hypothesis(0, "", {}, 14, 0.0);
    builder.update_hypothesis(0, "hello", {15}, 28, 1.12);
    const auto segments = builder.segments();
    require(segments.size() == 1 && near(segments[0].start, 1.20) && near(segments[0].end, 1.28), "decoded length");
}

void test_capitalization_and_leading_punctuation() {
    SpeakerSegmentBuilder builder(1.0);
    builder.update_hypothesis(0, "yes", {0}, 14, 0.0);
    builder.update_hypothesis(0, "yes. okay", {0, 50, 51}, 70, 3.36);
    const auto segments = builder.segments();
    require(segments.size() == 2 && segments[0].words == "Yes." && segments[1].words == "Okay", "capitalize + punct");
}

void test_speakers_merge_by_start_time() {
    SpeakerSegmentBuilder builder(1.0);
    builder.append_word(1, "later", 2.0, 2.4);
    builder.append_word(0, "first", 0.0, 0.4);
    builder.append_word(0, " again", 5.0, 5.4);
    const auto segments = builder.segments();
    require(segments.size() == 3 && segments[0].speaker == 0 && segments[1].speaker == 1 &&
                segments[2].words == "Again",
        "merge order");
    const auto json = seglst_json(segments, "s");
    require(json.find("\"speaker\": \"speaker_1\", \"start_time\": 2.000") != std::string::npos, "seglst json");
    require(segments_to_lines(segments) == "speaker_0: First\nspeaker_1: Later\nspeaker_0: Again\n", "lines");
}

void test_events_close_on_gap_and_cap() {
    SpeakerSegmentBuilder builder(1.0, 2.0);
    builder.append_word(0, "one", 0.0, 0.4);
    builder.append_word(0, " two", 0.5, 0.9);
    require(builder.take_events(1.0, false).empty(), "open segment is not emitted");
    const auto paused = builder.take_events(2.5, false);
    require(paused.size() == 1 && paused[0].words == "One two", "pause closes the segment");
    for (int i = 0; i < 6; ++i) builder.append_word(0, " w" + std::to_string(i), 3.0 + i * 0.5, 3.4 + i * 0.5);
    const auto capped = builder.take_events(5.9, false);
    require(capped.size() == 1 && capped[0].words == "W0 w1 w2 w3 w4", "cap emits whole words: " +
        (capped.empty() ? std::string("none") : capped[0].words));
    const auto rest = builder.take_events(6.0, true);
    require(rest.size() == 1 && rest[0].words == "w5", "final flush");
    require(builder.take_events(7.0, true).empty(), "nothing twice");
}

void test_group_words_and_attribution() {
    auto token = [](const std::string & text, int64_t frame) {
        engine::runtime::WordTimestamp t;
        t.word = text;
        t.span.start_sample = frame * kSpeakerFrameSamples;
        t.span.end_sample = (frame + 1) * kSpeakerFrameSamples;
        return t;
    };
    const auto words = group_words({token("I", 23), token(" ", 25), token("gue", 25), token("ss", 26),
                                    token(",", 27), token(" we", 30), token(" ", 31)});
    require(words.size() == 3 && words[0].text == "I" && words[1].text == " guess," && words[1].first_frame == 25 &&
                words[1].last_frame == 27 && words[2].text == " we",
        "group_words");

    // NeMo TestWordDictContentOffline layout: speaker 2 active over 80 ms frames 2..4.
    const int64_t frames = 6;
    const int64_t speakers = 3;
    std::vector<float> native(static_cast<size_t>(frames * kSpeakerRowsPerFrame * speakers), 0.0F);
    for (int64_t row = 2 * kSpeakerRowsPerFrame; row < 5 * kSpeakerRowsPerFrame; ++row) {
        native[static_cast<size_t>(row * speakers + 2)] = 1.0F;
    }
    const auto probabilities = make_speaker_probabilities(native, frames * kSpeakerRowsPerFrame, speakers);
    require(attribute_speaker(probabilities, {"hello", 2, 4}) == 2, "attribution in the active span");
    require(attribute_speaker(probabilities, {"x", 0, 0}) == 0, "ties go to the lowest speaker");
    require(attribute_speaker(probabilities, {"late", 40, 41}) == 0, "past the frontier uses the last frame");
}

void test_probability_loading_and_validation() {
    std::vector<float> native(static_cast<size_t>(10 * 2), 0.0F);
    native[static_cast<size_t>(8 * 2 + 1)] = 1.0F;  // row 8 is the lone row of the partial second group
    const auto probabilities = make_speaker_probabilities(native, 10, 2);
    require(probabilities.frames == 2 && near(probabilities.frames80[3], 0.5), "partial group averages valid rows");
    require(throws([] { (void) make_speaker_probabilities({1.5F}, 1, 1); }), "values above 1 rejected");
    require(throws([] { (void) make_speaker_probabilities(std::vector<float>(9, 0.0F), 1, 9); }), "9 speakers rejected");
    require(throws([&] { require_speaker_probability_rows(probabilities, 1760); }), "row count mismatch rejected");
    require_speaker_probability_rows(probabilities, 1759);

    const auto bytes = engine::models::nemotron_3_diar::encode_speaker_probabilities_safetensors(
        native, 10, 2, {{"frame_hop_samples", "160"}, {"sample_rate", "16000"}, {"latency_profile", "asr_la6"},
                        {"chunk_len", "7"}, {"chunk_right_context", "0"}});
    const auto path = std::filesystem::temp_directory_path() / "nemotron_asr_speaker_tagging_test.safetensors";
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    const auto loaded = load_speaker_probabilities(path);
    std::filesystem::remove(path);
    require(loaded.frames80 == probabilities.frames80 && loaded.metadata.at("latency_profile") == "asr_la6",
        "load round trip");
}

void test_lookahead_resolution() {
    const std::vector<int64_t> supported = {0, 3, 6, 13};
    const std::unordered_map<std::string, std::string> la6 = {
        {"latency_profile", "asr_la6"}, {"chunk_len", "7"}, {"chunk_right_context", "0"}};
    const std::unordered_map<std::string, std::string> very_high = {
        {"latency_profile", "very_high"}, {"chunk_len", "340"}, {"chunk_right_context", "40"}};
    auto choice = resolve_masked_lookahead(la6, std::nullopt, supported);
    require(choice.lookahead == 6 && choice.warnings.empty(), "metadata lookahead");
    choice = resolve_masked_lookahead(la6, 6, supported);
    require(choice.lookahead == 6 && choice.warnings.empty(), "explicit equal");
    choice = resolve_masked_lookahead(la6, 13, supported);
    require(choice.lookahead == 13 && choice.warnings.size() == 1, "explicit wins with a warning");
    choice = resolve_masked_lookahead(very_high, std::nullopt, supported);
    require(choice.lookahead == 13 && choice.warnings.size() == 1, "unmatched geometry falls back to 13");
    choice = resolve_masked_lookahead({}, std::nullopt, supported);
    require(choice.lookahead == 13 && choice.warnings.size() == 1, "missing metadata falls back to 13");
}

void test_strip_language_tags() {
    require(nemo_hypothesis_text("Probably not. <en-US> I mean", true) == "Probably not. I mean", "strip tag");
    require(nemo_hypothesis_text("<en-US>hello", true) == "hello", "leading tag");
    require(nemo_hypothesis_text("And Monday ", true) == "And Monday", "trailing space stripped");
    require(nemo_hypothesis_text("not . <en-US> I", true) == "not. I", "space before punctuation");
    require(nemo_hypothesis_text("not . <en-US> I ", false) == "not. <en-US> I ", "tags kept");
}

// NeMo TestSpeakerTaggedASR.test_find_active_speakers_valid: speakers 0 (0.8) and 2 (0.9) are active.
void test_active_speakers() {
    const auto value = [](int64_t, int64_t speaker) { return speaker == 0 ? 0.8F : speaker == 2 ? 0.9F : 0.4F; };
    require(active_speakers(value, 4, 0, 10) == std::vector<int>{0, 2}, "active speakers");
    require(active_speakers(value, 4, 3, 3).empty(), "empty window");
}

// NeMo test_mask_features_valid: 12 frames at 80 ms mask 100 feature frames, so the mask is
// left-padded by 4; a mask longer than the chunk is truncated from the right.
void test_feature_mask() {
    std::vector<bool> groups(12, false);
    for (int i = 0; i < 5; ++i) groups[static_cast<size_t>(i)] = true;
    const auto mask = feature_mask(groups, 100);
    require(mask.size() == 100 && mask[3] == 0.0F && mask[4] == 1.0F && mask[43] == 1.0F && mask[44] == 0.0F,
        "left-padded mask");
    const auto truncated = feature_mask(std::vector<bool>(14, true), 106);
    require(truncated.size() == 106 && truncated.front() == 1.0F, "truncated mask");

    std::vector<float> features = {-3.0F, 0.0F, -2.0F, 5.0F};  // 2 frames x 2 bins
    auto nemo = features;
    apply_feature_mask(nemo, {0.0F, 1.0F}, 2);
    require(nemo[0] == 0.0F && near(nemo[1], -16.6355) && nemo[2] == -2.0F && nemo[3] == 5.0F, "nemo fill");
}

}  // namespace

int main() {
    try {
        test_word_continuations_never_create_segment_boundaries();
        test_continuation_timestamps_respect_inactive_gaps();
        test_clean_word_boundary_can_create_segment();
        test_punctuation_and_repeated_prefixes_preserve_text();
        test_word_stream_is_invariant_while_segment_layout_changes_with_threshold();
        test_decoded_length_advances_when_chunk_has_no_new_text();
        test_capitalization_and_leading_punctuation();
        test_speakers_merge_by_start_time();
        test_events_close_on_gap_and_cap();
        test_group_words_and_attribution();
        test_probability_loading_and_validation();
        test_lookahead_resolution();
        test_strip_language_tags();
        test_active_speakers();
        test_feature_mask();
    } catch (const std::exception & error) {
        std::cerr << "nemotron_asr_speaker_tagging_test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "nemotron_asr_speaker_tagging_test passed\n";
    return 0;
}
