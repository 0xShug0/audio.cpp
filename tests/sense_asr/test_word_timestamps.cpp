#include "engine/community_models/sense_asr/word_timestamps.h"
#include "test_assert.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

using engine::community_models::sense_asr::SenseAsrWordFrames;
using engine::community_models::sense_asr::group_words;
using engine::community_models::sense_asr::word_timestamps_from_frames;
using engine::decoders::CtcTokenSpan;
using engine::test::require_eq;

constexpr int64_t kPrefix = 4; // language, event, emotion and ITN queries

// Token ids index this table; 0 is the CTC blank.
const std::vector<std::string> kVocab = {
    "<blank>",          // 0
    "<|zh|>",           // 1
    "<|NEUTRAL|>",      // 2
    "\xe5\xbc\x80",     // 3 开
    "\xe9\xa5\xad",     // 4 饭
    "\xe3\x80\x82",     // 5 。
    "\xe2\x96\x81I",    // 6 ▁I
    "'",                // 7
    "ve",               // 8
    "\xe2\x96\x81" "been", // 9 ▁been
    "\xe2\x96\x81",     // 10 ▁
    "lo",               // 11
    "OK",               // 12
    "\xe3\x81\x86",     // 13 う
    "\xef\xbc\x8c",     // 14 ，
    "4",                // 15
    ".",                // 16
    "5",                // 17
};

CtcTokenSpan span(int32_t id, int64_t start, int64_t end) {
  return CtcTokenSpan{id, start, end};
}

void require_word(const SenseAsrWordFrames &word, const std::string &text,
                  int64_t start, int64_t end, const std::string &label) {
  require_eq(word.text, text, label + " text");
  require_eq(word.start_frame, start, label + " start");
  require_eq(word.end_frame, end, label + " end");
}

void test_tags_are_skipped_and_frames_drop_the_query_prefix() {
  const auto words =
      group_words({span(1, 0, 1), span(2, 1, 2), span(3, 10, 11)}, kVocab, kPrefix);
  require_eq(words.size(), size_t{1}, "word count");
  require_word(words[0], "\xe5\xbc\x80", 6, 7, "first character");
}

void test_each_cjk_character_is_a_word_and_punctuation_joins_it() {
  const auto words = group_words(
      {span(3, 10, 11), span(4, 14, 16), span(5, 18, 19), span(13, 20, 21), span(14, 22, 23)},
      kVocab, kPrefix);
  require_eq(words.size(), size_t{3}, "word count");
  require_word(words[0], "\xe5\xbc\x80", 6, 7, "first character");
  // Punctuation adds text but not time.
  require_word(words[1], "\xe9\xa5\xad\xe3\x80\x82", 10, 12, "character with full stop");
  require_word(words[2], "\xe3\x81\x86\xef\xbc\x8c", 16, 17, "kana with comma");
}

void test_subwords_join_until_the_next_boundary() {
  const auto words = group_words(
      {span(6, 10, 11), span(7, 11, 12), span(8, 12, 14), span(9, 16, 17)}, kVocab, kPrefix);
  require_eq(words.size(), size_t{2}, "word count");
  require_word(words[0], "I've", 6, 10, "contraction");
  require_word(words[1], "been", 12, 13, "next word");
}

void test_numbers_keep_their_decimal_point() {
  const auto words =
      group_words({span(15, 10, 11), span(16, 11, 12), span(17, 12, 13)}, kVocab, kPrefix);
  require_eq(words.size(), size_t{1}, "word count");
  require_word(words[0], "4.5", 6, 9, "decimal number");
}

void test_a_lone_boundary_token_ends_the_word() {
  const auto words =
      group_words({span(9, 10, 11), span(10, 12, 13), span(11, 14, 15)}, kVocab, kPrefix);
  require_eq(words.size(), size_t{2}, "word count");
  require_word(words[0], "been", 6, 7, "word before the boundary");
  require_word(words[1], "lo", 10, 11, "word after the boundary");
}

void test_latin_after_cjk_starts_a_new_word() {
  const auto words = group_words({span(3, 10, 11), span(12, 12, 13)}, kVocab, kPrefix);
  require_eq(words.size(), size_t{2}, "word count");
  require_word(words[1], "OK", 8, 9, "latin word");
}

void test_leading_punctuation_and_unknown_ids_are_dropped() {
  const auto words =
      group_words({span(5, 8, 9), span(99, 9, 10), span(-1, 9, 10), span(3, 10, 11)}, kVocab, kPrefix);
  require_eq(words.size(), size_t{1}, "word count");
  require_word(words[0], "\xe5\xbc\x80", 6, 7, "character");
}

void test_spans_inside_the_prefix_clamp_to_the_first_frame() {
  const auto words = group_words({span(3, 2, 4)}, kVocab, kPrefix);
  require_eq(words.size(), size_t{1}, "word count");
  require_word(words[0], "\xe5\xbc\x80", 0, 1, "clamped character");
}

void test_frames_convert_to_input_rate_samples() {
  const std::vector<SenseAsrWordFrames> words = {{"a", 0, 1}, {"b", 10, 12}};
  const auto at16k = word_timestamps_from_frames(words, 60, 16000, 1000000);
  require_eq(at16k[1].span.start_sample, int64_t{9600}, "16 kHz start");
  require_eq(at16k[1].span.end_sample, int64_t{11520}, "16 kHz end");
  require_eq(at16k[1].word, std::string("b"), "word text");
  const auto at24k = word_timestamps_from_frames(words, 60, 24000, 1000000);
  require_eq(at24k[1].span.start_sample, int64_t{14400}, "24 kHz start");
  require_eq(at24k[1].span.end_sample, int64_t{17280}, "24 kHz end");
}

void test_samples_clamp_to_the_audio() {
  const std::vector<SenseAsrWordFrames> words = {{"a", 10, 12}};
  const auto clamped = word_timestamps_from_frames(words, 60, 16000, 10000);
  require_eq(clamped[0].span.start_sample, int64_t{9600}, "start inside");
  require_eq(clamped[0].span.end_sample, int64_t{10000}, "end clamped");
  const auto past = word_timestamps_from_frames(words, 60, 16000, 5000);
  require_eq(past[0].span.start_sample, int64_t{5000}, "start clamped");
  require_eq(past[0].span.end_sample, int64_t{5000}, "end not before start");
}

} // namespace

int main() {
  try {
    test_tags_are_skipped_and_frames_drop_the_query_prefix();
    test_each_cjk_character_is_a_word_and_punctuation_joins_it();
    test_subwords_join_until_the_next_boundary();
    test_numbers_keep_their_decimal_point();
    test_a_lone_boundary_token_ends_the_word();
    test_latin_after_cjk_starts_a_new_word();
    test_leading_punctuation_and_unknown_ids_are_dropped();
    test_spans_inside_the_prefix_clamp_to_the_first_frame();
    test_frames_convert_to_input_rate_samples();
    test_samples_clamp_to_the_audio();
    std::cout << "PASS: sense_asr word timestamps\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
