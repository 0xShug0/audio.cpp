#include "engine/framework/text/chunking.h"

#include "engine/framework/text/utf8.h"

#include "test_assert.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::text::TextChunkMode;

void require_chunks(
    const std::vector<std::string> & actual,
    const std::vector<std::string> & expected,
    const std::string & label) {
    engine::test::require_eq(actual.size(), expected.size(), label + " chunk count");
    for (size_t i = 0; i < expected.size(); ++i) {
        engine::test::require_eq(actual[i], expected[i], label + " chunk " + std::to_string(i));
    }
}

std::vector<uint32_t> to_codepoints(const std::string & text, const std::string & label) {
    std::vector<uint32_t> out;
    for (size_t pos = 0; pos < text.size();) {
        const auto lead = static_cast<unsigned char>(text[pos]);
        size_t width = 0;
        uint32_t value = 0;
        if (lead <= 0x7FU) {
            width = 1;
            value = lead;
        } else if ((lead & 0xE0U) == 0xC0U) {
            width = 2;
            value = lead & 0x1FU;
        } else if ((lead & 0xF0U) == 0xE0U) {
            width = 3;
            value = lead & 0x0FU;
        } else if ((lead & 0xF8U) == 0xF0U) {
            width = 4;
            value = lead & 0x07U;
        } else {
            throw std::runtime_error(label + " starts a codepoint on a continuation byte");
        }
        if (pos + width > text.size()) {
            throw std::runtime_error(label + " ends inside a multi-byte codepoint");
        }
        for (size_t i = 1; i < width; ++i) {
            const auto continuation = static_cast<unsigned char>(text[pos + i]);
            if (!engine::text::is_utf8_continuation(continuation)) {
                throw std::runtime_error(label + " has a torn multi-byte codepoint");
            }
            value = (value << 6U) | (continuation & 0x3FU);
        }
        out.push_back(value);
        pos += width;
    }
    return out;
}

// Every chunk must be standalone-valid UTF-8, and concatenating the chunks must
// reproduce the original codepoint sequence minus the ASCII whitespace the
// splitter is allowed to drop between chunks.
void require_lossless_codepoint_split(
    const std::string & source,
    const std::vector<std::string> & chunks,
    const std::string & label) {
    std::vector<uint32_t> rejoined;
    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto chunk = to_codepoints(chunks[i], label + " chunk " + std::to_string(i));
        engine::test::require(!chunk.empty(), label + " produced an empty chunk");
        rejoined.insert(rejoined.end(), chunk.begin(), chunk.end());
    }
    std::vector<uint32_t> expected;
    for (const uint32_t codepoint : to_codepoints(source, label + " source")) {
        if (codepoint == 0x20U || codepoint == 0x09U || codepoint == 0x0AU ||
            codepoint == 0x0DU || codepoint == 0x0BU || codepoint == 0x0CU) {
            continue;
        }
        expected.push_back(codepoint);
    }
    std::vector<uint32_t> observed;
    for (const uint32_t codepoint : rejoined) {
        if (codepoint == 0x20U || codepoint == 0x09U || codepoint == 0x0AU ||
            codepoint == 0x0DU || codepoint == 0x0BU || codepoint == 0x0CU) {
            continue;
        }
        observed.push_back(codepoint);
    }
    engine::test::require_eq(observed.size(), expected.size(), label + " codepoint count round-trip");
    for (size_t i = 0; i < expected.size(); ++i) {
        engine::test::require_eq(observed[i], expected[i], label + " codepoint " + std::to_string(i));
    }
}

// ---------------------------------------------------------------------------
// F6.3 regression guard.
//
// Every expectation below was captured from the splitter as it behaved before
// CJK-aware splitting was added, so any change to where Latin-script text splits
// fails here. ~28 models join their chunk audio at these boundaries; moving one
// changes their output.
// ---------------------------------------------------------------------------

void test_ascii_default_splits_are_unchanged() {
    require_chunks(
        engine::text::split_text_chunks("Hello world.", 100),
        {"Hello world."},
        "short ascii");

    require_chunks(
        engine::text::split_text_chunks(
            "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.", 50),
        {"The quick brown fox jumps over the lazy dog.", "Pack my box with five dozen liquor jugs."},
        "ascii sentence break");

    require_chunks(
        engine::text::split_text_chunks(
            "One two three, four five six, seven eight nine, ten eleven twelve", 20),
        {"One two three,", "four five six,", "seven eight nine,", "ten eleven twelve"},
        "ascii clause break");

    require_chunks(
        engine::text::split_text_chunks(
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike", 20),
        {"alpha bravo charlie", "delta echo foxtrot", "golf hotel india", "juliet kilo lima", "mike"},
        "ascii unpunctuated");

    // An over-long ASCII word keeps the historical whole-word escape hatch: it is
    // emitted verbatim rather than cut at the budget.
    require_chunks(
        engine::text::split_text_chunks(
            "supercalifragilisticexpialidociousandthensomeextraletters tail", 10),
        {"supercalifragilisticexpialidociousandthensomeextraletters", "tail"},
        "ascii over-long word");

    require_chunks(
        engine::text::split_text_chunks(
            "See https://example.com/a/very/long/path/that/never/ends/at/all now", 16),
        {"See", "https://example.com/a/very/long/path/that/never/ends/at/all", "now"},
        "ascii url");

    require_chunks(
        engine::text::split_text_chunks("a b c d e", 1),
        {"a", "b", "c", "d", "e"},
        "ascii budget one");

    require_chunks(
        engine::text::split_text_chunks("abc def ghi jkl", 3),
        {"abc", "def", "ghi", "jkl"},
        "ascii budget three");

    require_chunks(
        engine::text::split_text_chunks("Line one.\nLine two.\nLine three is a bit longer here.", 18),
        {"Line one.", "Line two.", "Line three is a", "bit longer here."},
        "ascii newlines");

    require_chunks(
        engine::text::split_text_chunks("one\ttwo\tthree\tfour\tfive\tsix", 9),
        {"one\ttwo", "three", "four\tfive", "six"},
        "ascii tabs");

    require_chunks(
        engine::text::split_text_chunks("one   two    three     four", 9),
        {"one   two", "three", "four"},
        "ascii runs of spaces");

    require_chunks(
        engine::text::split_text_chunks("   padded text with spaces around   ", 12),
        {"padded text", "with spaces", "around"},
        "ascii outer whitespace");

    require_chunks(
        engine::text::split_text_chunks("abcdefghij", 10),
        {"abcdefghij"},
        "ascii exactly at budget");

    require_chunks(
        engine::text::split_text_chunks("abcdefghijk", 10),
        {"abcdefghijk"},
        "ascii one over budget");

    require_chunks(
        engine::text::split_text_chunks("\"Hello,\" she said. \"How are you today?\" He shrugged.", 20),
        {"\"Hello,\" she said.", "\"How are you today?\"", "He shrugged."},
        "ascii dialogue");

    require_chunks(
        engine::text::split_text_chunks("first part; second part; third part; fourth part", 15),
        {"first part;", "second part;", "third part;", "fourth part"},
        "ascii semicolons");

    require_chunks(
        engine::text::split_text_chunks("1 22 333 4444 55555 666666 7777777 88888888", 12),
        {"1 22 333", "4444 55555", "666666", "7777777", "88888888"},
        "ascii numbers");

    require_chunks(
        engine::text::split_text_chunks("Wait... really? Yes! Absolutely, without a doubt.", 18),
        {"Wait... really?", "Yes! Absolutely,", "without a doubt."},
        "ascii mixed punctuation");

    require_chunks(engine::text::split_text_chunks("hello", 5), {"hello"}, "ascii single word");
    require_chunks(engine::text::split_text_chunks("", 10), {}, "ascii empty");
    require_chunks(engine::text::split_text_chunks("     ", 10), {}, "ascii whitespace only");

    // Latin-1 supplement and Cyrillic are multi-byte but space-delimited, so the
    // word splitter still applies and nothing about them changed either.
    require_chunks(
        engine::text::split_text_chunks("caf\xC3\xA9 na\xC3\xAF" "ve r\xC3\xA9sum\xC3\xA9 jalape\xC3\xB1o", 8),
        {"caf\xC3\xA9", "na\xC3\xAF" "ve", "r\xC3\xA9sum\xC3\xA9", "jalape\xC3\xB1o"},
        "latin accents");

    require_chunks(
        engine::text::split_text_chunks(
            "\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82 \xD0\xBC\xD0\xB8\xD1\x80 "
            "\xD0\xBA\xD0\xB0\xD0\xBA \xD0\xB4\xD0\xB5\xD0\xBB\xD0\xB0 "
            "\xD1\x81\xD0\xB5\xD0\xB3\xD0\xBE\xD0\xB4\xD0\xBD\xD1\x8F",
            8),
        {"\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82",
         "\xD0\xBC\xD0\xB8\xD1\x80 \xD0\xBA\xD0\xB0\xD0\xBA",
         "\xD0\xB4\xD0\xB5\xD0\xBB\xD0\xB0",
         "\xD1\x81\xD0\xB5\xD0\xB3\xD0\xBE\xD0\xB4\xD0\xBD\xD1\x8F"},
        "cyrillic");
}

void test_ascii_two_argument_and_default_mode_agree() {
    const char * text = "One two three, four five six, seven eight nine, ten eleven twelve";
    const auto implicit = engine::text::split_text_chunks(text, 20);
    const auto explicit_default = engine::text::split_text_chunks(text, 20, TextChunkMode::Default);
    require_chunks(explicit_default, implicit, "default mode overload agreement");
}

void test_tag_aware_and_endline_modes_are_unchanged() {
    require_chunks(
        engine::text::split_text_chunks(
            "[laugh] hello there my friend how are you [sigh] doing today",
            15,
            TextChunkMode::TagAware),
        {"[laugh] hello", "there my friend", "how are you", "[sigh] doing", "today"},
        "tag aware");

    require_chunks(
        engine::text::split_text_chunks(
            "First line.\nSecond line here.\nThird line is longer than the rest.",
            20,
            TextChunkMode::Endline),
        {"First line.", "Second line here.", "Third line is longer", "than the rest."},
        "endline");
}

// ---------------------------------------------------------------------------
// F6.3 fix.
// ---------------------------------------------------------------------------

void test_chinese_text_splits_at_sentence_punctuation() {
    // 这是第一句话。这是第二句话。这是第三句话。 — 21 codepoints, no ASCII space.
    const std::string text =
        "\xE8\xBF\x99\xE6\x98\xAF\xE7\xAC\xAC\xE4\xB8\x80\xE5\x8F\xA5\xE8\xAF\x9D\xE3\x80\x82"
        "\xE8\xBF\x99\xE6\x98\xAF\xE7\xAC\xAC\xE4\xBA\x8C\xE5\x8F\xA5\xE8\xAF\x9D\xE3\x80\x82"
        "\xE8\xBF\x99\xE6\x98\xAF\xE7\xAC\xAC\xE4\xB8\x89\xE5\x8F\xA5\xE8\xAF\x9D\xE3\x80\x82";
    const auto chunks = engine::text::split_text_chunks(text, 10);
    engine::test::require(chunks.size() > 1, "Chinese text must not collapse to one chunk");
    require_chunks(
        chunks,
        {"\xE8\xBF\x99\xE6\x98\xAF\xE7\xAC\xAC\xE4\xB8\x80\xE5\x8F\xA5\xE8\xAF\x9D\xE3\x80\x82",
         "\xE8\xBF\x99\xE6\x98\xAF\xE7\xAC\xAC\xE4\xBA\x8C\xE5\x8F\xA5\xE8\xAF\x9D\xE3\x80\x82",
         "\xE8\xBF\x99\xE6\x98\xAF\xE7\xAC\xAC\xE4\xB8\x89\xE5\x8F\xA5\xE8\xAF\x9D\xE3\x80\x82"},
        "chinese sentence split");
    require_lossless_codepoint_split(text, chunks, "chinese sentence split");
    for (const auto & chunk : chunks) {
        engine::test::require_eq(
            engine::text::utf8_codepoint_count(chunk, "chinese chunk"),
            static_cast<size_t>(7),
            "chinese chunk is one sentence");
    }
}

void test_chinese_text_falls_back_to_clause_punctuation() {
    // 一二三，四五六，七八九，十 — 13 codepoints, only clause breaks.
    const std::string text =
        "\xE4\xB8\x80\xE4\xBA\x8C\xE4\xB8\x89\xEF\xBC\x8C"
        "\xE5\x9B\x9B\xE4\xBA\x94\xE5\x85\xAD\xEF\xBC\x8C"
        "\xE4\xB8\x83\xE5\x85\xAB\xE4\xB9\x9D\xEF\xBC\x8C\xE5\x8D\x81";
    const auto chunks = engine::text::split_text_chunks(text, 6);
    require_chunks(
        chunks,
        {"\xE4\xB8\x80\xE4\xBA\x8C\xE4\xB8\x89\xEF\xBC\x8C",
         "\xE5\x9B\x9B\xE4\xBA\x94\xE5\x85\xAD\xEF\xBC\x8C",
         "\xE4\xB8\x83\xE5\x85\xAB\xE4\xB9\x9D\xEF\xBC\x8C\xE5\x8D\x81"},
        "chinese clause split");
    require_lossless_codepoint_split(text, chunks, "chinese clause split");
}

void test_unpunctuated_cjk_splits_at_the_budget() {
    // 一二三四五六七八九十一二三四五 — 15 codepoints, no punctuation at all.
    const std::string text =
        "\xE4\xB8\x80\xE4\xBA\x8C\xE4\xB8\x89\xE5\x9B\x9B\xE4\xBA\x94"
        "\xE5\x85\xAD\xE4\xB8\x83\xE5\x85\xAB\xE4\xB9\x9D\xE5\x8D\x81"
        "\xE4\xB8\x80\xE4\xBA\x8C\xE4\xB8\x89\xE5\x9B\x9B\xE4\xBA\x94";
    const auto chunks = engine::text::split_text_chunks(text, 6);
    engine::test::require_eq(chunks.size(), static_cast<size_t>(3), "unpunctuated CJK chunk count");
    engine::test::require_eq(
        engine::text::utf8_codepoint_count(chunks[0], "cjk chunk"),
        static_cast<size_t>(6),
        "first unpunctuated CJK chunk fills the budget");
    engine::test::require_eq(
        engine::text::utf8_codepoint_count(chunks[1], "cjk chunk"),
        static_cast<size_t>(6),
        "second unpunctuated CJK chunk fills the budget");
    engine::test::require_eq(
        engine::text::utf8_codepoint_count(chunks[2], "cjk chunk"),
        static_cast<size_t>(3),
        "final unpunctuated CJK chunk holds the remainder");
    require_lossless_codepoint_split(text, chunks, "unpunctuated CJK");
}

void test_mixed_ascii_and_cjk_only_splits_the_cjk_run() {
    // "Hello 世界世界世界世界世界 world" — the CJK run is the only over-long word.
    const std::string text =
        "Hello \xE4\xB8\x96\xE7\x95\x8C\xE4\xB8\x96\xE7\x95\x8C\xE4\xB8\x96\xE7\x95\x8C"
        "\xE4\xB8\x96\xE7\x95\x8C\xE4\xB8\x96\xE7\x95\x8C world";
    const auto chunks = engine::text::split_text_chunks(text, 6);
    require_chunks(
        chunks,
        {"Hello",
         "\xE4\xB8\x96\xE7\x95\x8C\xE4\xB8\x96\xE7\x95\x8C\xE4\xB8\x96\xE7\x95\x8C",
         "\xE4\xB8\x96\xE7\x95\x8C\xE4\xB8\x96\xE7\x95\x8C",
         "world"},
        "mixed ascii and cjk");
    require_lossless_codepoint_split(text, chunks, "mixed ascii and cjk");
}

void test_hangul_and_fullwidth_are_treated_as_space_free() {
    // 한국어문장입니다한국어문장입니다 — 16 Hangul syllables.
    const std::string hangul =
        "\xED\x95\x9C\xEA\xB5\xAD\xEC\x96\xB4\xEB\xAC\xB8\xEC\x9E\xA5"
        "\xEC\x9E\x85\xEB\x8B\x88\xEB\x8B\xA4"
        "\xED\x95\x9C\xEA\xB5\xAD\xEC\x96\xB4\xEB\xAC\xB8\xEC\x9E\xA5"
        "\xEC\x9E\x85\xEB\x8B\x88\xEB\x8B\xA4";
    const auto hangul_chunks = engine::text::split_text_chunks(hangul, 5);
    engine::test::require_eq(hangul_chunks.size(), static_cast<size_t>(4), "hangul chunk count");
    require_lossless_codepoint_split(hangul, hangul_chunks, "hangul");

    // Ｈｅｌｌｏ！Ｗｏｒｌｄ！ — fullwidth forms, including fullwidth '!'.
    const std::string fullwidth =
        "\xEF\xBC\xA8\xEF\xBD\x85\xEF\xBD\x8C\xEF\xBD\x8C\xEF\xBD\x8F\xEF\xBC\x81"
        "\xEF\xBC\xB7\xEF\xBD\x8F\xEF\xBD\x92\xEF\xBD\x8C\xEF\xBD\x84\xEF\xBC\x81";
    const auto fullwidth_chunks = engine::text::split_text_chunks(fullwidth, 5);
    engine::test::require_eq(fullwidth_chunks.size(), static_cast<size_t>(3), "fullwidth chunk count");
    require_lossless_codepoint_split(fullwidth, fullwidth_chunks, "fullwidth");
}

void test_four_byte_codepoints_are_never_split() {
    // U+20000..U+20007, CJK extension B — eight 4-byte codepoints, no punctuation.
    std::string text;
    for (unsigned char last = 0x80U; last < 0x88U; ++last) {
        text.push_back(static_cast<char>(0xF0));
        text.push_back(static_cast<char>(0xA0));
        text.push_back(static_cast<char>(0x80));
        text.push_back(static_cast<char>(last));
    }
    const auto chunks = engine::text::split_text_chunks(text, 3);
    engine::test::require_eq(chunks.size(), static_cast<size_t>(3), "extension B chunk count");
    engine::test::require_eq(chunks[0].size(), static_cast<size_t>(12), "extension B chunk 0 byte length");
    engine::test::require_eq(chunks[1].size(), static_cast<size_t>(12), "extension B chunk 1 byte length");
    engine::test::require_eq(chunks[2].size(), static_cast<size_t>(8), "extension B chunk 2 byte length");
    require_lossless_codepoint_split(text, chunks, "extension B");
}

void test_combining_kana_marks_stay_with_their_base() {
    // か+U+3099 き+U+3099 く+U+3099 け+U+3099 こ+U+3099 か+U+3099 き+U+3099
    // 14 codepoints, 7 grapheme clusters. A budget of 3 lands the raw boundary on
    // the combining mark, which must never start a chunk.
    const std::string text =
        "\xE3\x81\x8B\xE3\x82\x99\xE3\x81\x8D\xE3\x82\x99\xE3\x81\x8F\xE3\x82\x99"
        "\xE3\x81\x91\xE3\x82\x99\xE3\x81\x93\xE3\x82\x99\xE3\x81\x8B\xE3\x82\x99"
        "\xE3\x81\x8D\xE3\x82\x99";
    const std::vector<std::string> expected = {
        "\xE3\x81\x8B\xE3\x82\x99",
        "\xE3\x81\x8D\xE3\x82\x99",
        "\xE3\x81\x8F\xE3\x82\x99",
        "\xE3\x81\x91\xE3\x82\x99",
        "\xE3\x81\x93\xE3\x82\x99",
        "\xE3\x81\x8B\xE3\x82\x99",
        "\xE3\x81\x8D\xE3\x82\x99",
    };
    require_chunks(engine::text::split_text_chunks(text, 3), expected, "default mode kana clusters");
    require_chunks(
        engine::text::split_text_chunks(text, 3, TextChunkMode::Japanese),
        expected,
        "japanese mode kana clusters");
    for (const auto & chunk : expected) {
        engine::test::require(
            static_cast<unsigned char>(chunk[0]) != 0xE3U ||
                static_cast<unsigned char>(chunk[1]) != 0x82U ||
                static_cast<unsigned char>(chunk[2]) != 0x99U,
            "no chunk starts with a combining kana mark");
    }
}

void test_japanese_mode_output_is_unchanged() {
    // これはテストです。ありがとうございます。 — 20 codepoints.
    const std::string text =
        "\xE3\x81\x93\xE3\x82\x8C\xE3\x81\xAF\xE3\x83\x86\xE3\x82\xB9\xE3\x83\x88"
        "\xE3\x81\xA7\xE3\x81\x99\xE3\x80\x82"
        "\xE3\x81\x82\xE3\x82\x8A\xE3\x81\x8C\xE3\x81\xA8\xE3\x81\x86\xE3\x81\x94"
        "\xE3\x81\x96\xE3\x81\x84\xE3\x81\xBE\xE3\x81\x99\xE3\x80\x82";
    require_chunks(
        engine::text::split_text_chunks(text, 8, TextChunkMode::Japanese),
        {"\xE3\x81\x93\xE3\x82\x8C\xE3\x81\xAF\xE3\x83\x86\xE3\x82\xB9\xE3\x83\x88"
         "\xE3\x81\xA7\xE3\x81\x99",
         "\xE3\x80\x82\xE3\x81\x82\xE3\x82\x8A\xE3\x81\x8C\xE3\x81\xA8\xE3\x81\x86"
         "\xE3\x81\x94\xE3\x81\x96",
         "\xE3\x81\x84\xE3\x81\xBE\xE3\x81\x99\xE3\x80\x82"},
        "japanese mode");
}

void test_short_cjk_text_is_still_returned_whole() {
    const std::string text =
        "\xE4\xB8\x80\xE4\xBA\x8C\xE4\xB8\x89\xE5\x9B\x9B\xE4\xBA\x94";
    require_chunks(engine::text::split_text_chunks(text, 10), {text}, "short CJK text");
}

void test_budget_validation() {
    bool threw = false;
    try {
        (void)engine::text::split_text_chunks("some text", 0);
    } catch (const std::exception &) {
        threw = true;
    }
    engine::test::require(threw, "zero budget rejected");

    threw = false;
    try {
        (void)engine::text::split_text_chunks("some text", -4);
    } catch (const std::exception &) {
        threw = true;
    }
    engine::test::require(threw, "negative budget rejected");
}

}  // namespace

int main() {
    try {
        test_ascii_default_splits_are_unchanged();
        test_ascii_two_argument_and_default_mode_agree();
        test_tag_aware_and_endline_modes_are_unchanged();
        test_chinese_text_splits_at_sentence_punctuation();
        test_chinese_text_falls_back_to_clause_punctuation();
        test_unpunctuated_cjk_splits_at_the_budget();
        test_mixed_ascii_and_cjk_only_splits_the_cjk_run();
        test_hangul_and_fullwidth_are_treated_as_space_free();
        test_four_byte_codepoints_are_never_split();
        test_combining_kana_marks_stay_with_their_base();
        test_japanese_mode_output_is_unchanged();
        test_short_cjk_text_is_still_returned_whole();
        test_budget_validation();
        std::cout << "text_chunking_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "text_chunking_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
