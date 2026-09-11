#include "engine/framework/text/subtitle.h"

#include "test_assert.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

engine::runtime::WordTimestamp word(const std::string & text, int64_t start, int64_t end) {
    engine::runtime::WordTimestamp out;
    out.word = text;
    out.span.start_sample = start;
    out.span.end_sample = end;
    out.confidence = 1.0F;
    return out;
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

void test_formats_srt_timecodes_and_text() {
    engine::text::SubtitleFormatOptions options;
    options.sample_rate = 16000;
    options.max_chars_per_line = 80;
    options.max_lines = 2;
    options.max_block_seconds = 6.0;
    options.max_gap_seconds = 1.0;

    const std::vector<engine::runtime::WordTimestamp> words{
        word("hello", 0, 8000),
        word("world", 9600, 16000),
    };
    const auto srt = engine::text::format_srt(words, options);
    const std::string expected =
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "hello world\n"
        "\n";
    engine::test::require_eq(srt, expected, "srt output");
}

void test_splits_on_large_gap() {
    engine::text::SubtitleFormatOptions options;
    options.sample_rate = 1000;
    options.max_chars_per_line = 80;
    options.max_lines = 2;
    options.max_block_seconds = 10.0;
    options.max_gap_seconds = 0.5;

    const std::vector<engine::runtime::WordTimestamp> words{
        word("first", 0, 100),
        word("second", 200, 300),
        word("third", 1000, 1100),
    };
    const auto blocks = engine::text::build_subtitle_blocks(words, options);
    engine::test::require_eq(blocks.size(), size_t{2}, "gap split block count");
    engine::test::require_eq(blocks[0].lines[0], std::string("first second"), "first block text");
    engine::test::require_eq(blocks[1].lines[0], std::string("third"), "second block text");
}

void test_splits_on_duration() {
    engine::text::SubtitleFormatOptions options;
    options.sample_rate = 1000;
    options.max_chars_per_line = 80;
    options.max_lines = 2;
    options.max_block_seconds = 1.0;
    options.max_gap_seconds = 1.0;

    const std::vector<engine::runtime::WordTimestamp> words{
        word("one", 0, 200),
        word("two", 300, 600),
        word("three", 700, 1201),
    };
    const auto blocks = engine::text::build_subtitle_blocks(words, options);
    engine::test::require_eq(blocks.size(), size_t{2}, "duration split block count");
    engine::test::require_eq(blocks[0].lines[0], std::string("one two"), "duration first block");
    engine::test::require_eq(blocks[1].lines[0], std::string("three"), "duration second block");
}

void test_wraps_and_splits_by_line_budget() {
    engine::text::SubtitleFormatOptions options;
    options.sample_rate = 1000;
    options.max_chars_per_line = 10;
    options.max_lines = 2;
    options.max_block_seconds = 10.0;
    options.max_gap_seconds = 1.0;

    const std::vector<engine::runtime::WordTimestamp> words{
        word("alpha", 0, 100),
        word("beta", 100, 200),
        word("gamma", 200, 300),
        word("delta", 300, 400),
    };
    const auto blocks = engine::text::build_subtitle_blocks(words, options);
    engine::test::require_eq(blocks.size(), size_t{2}, "line budget block count");
    engine::test::require_eq(blocks[0].lines.size(), size_t{2}, "first line count");
    engine::test::require_eq(blocks[0].lines[0], std::string("alpha beta"), "first wrapped line");
    engine::test::require_eq(blocks[0].lines[1], std::string("gamma"), "second wrapped line");
    engine::test::require_eq(blocks[1].lines[0], std::string("delta"), "second block line");
}

void test_rejects_invalid_inputs() {
    engine::text::SubtitleFormatOptions options;
    options.sample_rate = 1000;

    require_throws(
        [&] {
            auto bad = options;
            bad.sample_rate = 0;
            engine::text::format_srt({word("hello", 0, 100)}, bad);
        },
        "sample_rate validation");
    require_throws(
        [&] {
            engine::text::format_srt({word("", 0, 100)}, options);
        },
        "empty word validation");
    require_throws(
        [&] {
            engine::text::format_srt({word("bad\nword", 0, 100)}, options);
        },
        "newline validation");
    require_throws(
        [&] {
            engine::text::format_srt({word("bad", 200, 100)}, options);
        },
        "span validation");
    require_throws(
        [&] {
            engine::text::format_srt({word("later", 200, 300), word("earlier", 100, 150)}, options);
        },
        "sort validation");
}

}  // namespace

int main() {
    try {
        test_formats_srt_timecodes_and_text();
        test_splits_on_large_gap();
        test_splits_on_duration();
        test_wraps_and_splits_by_line_budget();
        test_rejects_invalid_inputs();
        std::cout << "subtitle_formatter_test passed\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "subtitle_formatter_test failed: " << ex.what() << "\n";
        return 1;
    }
}
