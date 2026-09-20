// Grouping and timing of kokoro_tts's predicted durations.
//
// append_kokoro_word_timings is a pure function -- ids + durations + vocabulary + sample count in,
// spans out -- so all of it is reachable without a session, a package or a backend. The cases here
// are the ones that decide whether a caller can join its own words to the result: where a group
// starts and ends, what counts as a boundary, and what is not a word at all.

#include "engine/models/kokoro_tts/session.h"
#include "test_assert.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using engine::models::kokoro_tts::append_kokoro_word_timings;
using engine::runtime::WordTimestamp;

/// A miniature of Kokoro's own vocabulary: id 0 is the pad, 16 is the space, and the rest are a
/// few phonemes plus the punctuation the real one carries.
const std::unordered_map<std::string, int32_t> & vocab() {
    static const std::unordered_map<std::string, int32_t> values = {
        {"$", 0}, {" ", 16},
        {"h", 50}, {"i", 51}, {"k", 53}, {"æ", 43}, {"t", 62},
        {".", 4}, {",", 3}, {"“", 14}, {"”", 15},
    };
    return values;
}

/// [pad, h, i, space, k, æ, t, pad] -- "hi cat", two groups.
std::vector<int32_t> two_words() { return {0, 50, 51, 16, 53, 43, 62, 0}; }

void test_groups_are_cut_at_space_and_pad() {
    std::vector<WordTimestamp> out;
    // One frame per token, eight tokens, 800 samples -> 100 samples per frame.
    append_kokoro_word_timings(out, two_words(), {1, 1, 1, 1, 1, 1, 1, 1}, vocab(), 800, 0);

    engine::test::require(out.size() == 2, "two groups from two space-separated runs");
    engine::test::require(out[0].word == "hi", "first group carries its phonemes");
    engine::test::require(out[1].word == "kæt", "second group carries its phonemes");
    // The pad occupies frame 0, so the first group starts at frame 1.
    engine::test::require(out[0].span.start_sample == 100, "leading pad is not part of the first word");
    engine::test::require(out[0].span.end_sample == 300, "first group ends before the space");
    // The space occupies frame 3; the second group starts after it, not at the first group's end.
    engine::test::require(out[1].span.start_sample == 400, "the space is not part of either word");
    engine::test::require(out[1].span.end_sample == 700, "trailing pad is not part of the last word");
}

void test_durations_decide_the_spans() {
    std::vector<WordTimestamp> out;
    // "hi" is 1+1 frames and "kæt" is 3+3+3; 20 frames total over 2000 samples.
    append_kokoro_word_timings(out, two_words(), {1, 1, 1, 1, 3, 3, 3, 7}, vocab(), 2000, 0);

    engine::test::require(out.size() == 2, "two groups");
    engine::test::require(out[0].span.end_sample - out[0].span.start_sample == 200, "hi spans 2 of 20 frames");
    engine::test::require(out[1].span.end_sample - out[1].span.start_sample == 900, "kæt spans 9 of 20 frames");
}

void test_chunk_offset_accumulates() {
    std::vector<WordTimestamp> out;
    append_kokoro_word_timings(out, two_words(), {1, 1, 1, 1, 1, 1, 1, 1}, vocab(), 800, 0);
    // A second chunk appended after 800 samples of audio already merged.
    append_kokoro_word_timings(out, two_words(), {1, 1, 1, 1, 1, 1, 1, 1}, vocab(), 800, 800);

    engine::test::require(out.size() == 4, "both chunks reported");
    engine::test::require(out[2].span.start_sample == 900, "the second chunk is offset by the first");
    engine::test::require(out[2].span.start_sample >= out[1].span.end_sample,
                          "spans stay monotonic across the chunk seam");
}

void test_a_standalone_mark_is_not_a_word() {
    // ⚠ THE CASE THAT SHIFTS A CALLER'S WORD MAP. The G2P spaces a mark that followed a space in
    // the source, so `She said "hi" ...` puts the opening quote in a group of its own. Reporting it
    // as a word leaves every later word one group out for the rest of the chunk.
    // [pad, “, space, h, i, pad]
    const std::vector<int32_t> ids = {0, 14, 16, 50, 51, 0};
    std::vector<WordTimestamp> out;
    append_kokoro_word_timings(out, ids, {1, 1, 1, 1, 1, 1}, vocab(), 600, 0);

    engine::test::require(out.size() == 1, "the standalone quote is not reported as a word");
    engine::test::require(out[0].word == "hi", "the spoken group is the one reported");
    // Its frames are still consumed, so the word that follows is not pulled backwards onto them.
    engine::test::require(out[0].span.start_sample == 300, "the mark's frames are still accounted for");
}

void test_attached_punctuation_stays_with_its_word() {
    // The other half of the same rule: a mark the G2P did NOT space belongs to the word it is
    // attached to, and the group is still a word.
    // [pad, h, i, ., pad]
    const std::vector<int32_t> ids = {0, 50, 51, 4, 0};
    std::vector<WordTimestamp> out;
    append_kokoro_word_timings(out, ids, {1, 1, 1, 1, 1}, vocab(), 500, 0);

    engine::test::require(out.size() == 1, "one word");
    engine::test::require(out[0].word == "hi.", "the mark rides on the word it is attached to");
}

void test_a_count_mismatch_reports_nothing_rather_than_throwing() {
    // The audio is the product; an empty word list is a state every caller already handles, so a
    // broken invariant must not cost the render.
    std::vector<WordTimestamp> out;
    append_kokoro_word_timings(out, two_words(), {1, 1, 1}, vocab(), 800, 0);
    engine::test::require(out.empty(), "nothing reported when durations and tokens disagree");

    append_kokoro_word_timings(out, two_words(), {1, 1, 1, 1, 1, 1, 1, 1}, vocab(), 0, 0);
    engine::test::require(out.empty(), "nothing reported for a chunk with no audio");

    append_kokoro_word_timings(out, two_words(), {0, 0, 0, 0, 0, 0, 0, 0}, vocab(), 800, 0);
    engine::test::require(out.empty(), "nothing reported when the durations sum to zero");
}

}  // namespace

int main() {
    try {
        test_groups_are_cut_at_space_and_pad();
        test_durations_decide_the_spans();
        test_chunk_offset_accumulates();
        test_a_standalone_mark_is_not_a_word();
        test_attached_punctuation_stays_with_its_word();
        test_a_count_mismatch_reports_nothing_rather_than_throwing();
        std::cout << "kokoro_word_timings_test passed\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "kokoro_word_timings_test failed: " << ex.what() << "\n";
        return 1;
    }
}
