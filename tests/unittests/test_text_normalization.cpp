#include "engine/framework/text/text_normalization.h"

#include "test_assert.h"

#include <iostream>
#include <string>

namespace {

void test_english_numbers() {
    engine::test::require_eq(
        engine::text::normalize_english_numbers("On March 3rd, 2026, uptime was 99.5%."),
        std::string("On March third twenty twenty six, uptime was ninety nine point five percent."),
        "English dates and percentages");
    engine::test::require_eq(
        engine::text::normalize_english_numbers("Use 1/2 now and 123456789012345678901 later."),
        std::string("Use one half now and one two three four five six seven eight nine zero one two three four five six seven eight nine zero one later."),
        "English fraction and overflow-safe long digit spelling");
    engine::test::require_eq(
        engine::text::normalize_english_numbers("Use DS4, R2D2, and 4K video."),
        std::string("Use DS four, R two D two, and four K video."),
        "English letter-digit token verbalization");
}

void test_english_index_tts_style() {
    engine::text::EnglishTextNormalizationOptions options;
    options.expand_common_contractions = true;
    options.index_tts_punctuation = true;
    options.uppercase_ascii = true;
    options.verbalize_symbols = true;
    engine::test::require_eq(
        engine::text::normalize_english_text("What's DS4, R2D2, C++_x=y, and 4K？", options),
        std::string("WHAT IS DS FOUR, R TWO D TWO, C PLUS PLUS UNDERSCORE X EQUAL SIGN Y, AND FOUR K?"),
        "IndexTTS-style English alphanumeric and symbol normalization");
}

void test_japanese_katakana() {
    engine::test::require_eq(
        engine::text::normalize_japanese_text("こんにちは　ﾊﾟﾋﾟﾌﾟﾍﾟﾎﾟ？！..."),
        std::string("コンニチハ パピプペポ?!…"),
        "Japanese hiragana, half-width katakana, punctuation, and ellipsis normalization");
    engine::test::require_eq(
        engine::text::normalize_japanese_text("音声クローニング"),
        std::string("音声クローニング"),
        "Japanese kanji passthrough without dictionary dependency");
}

}  // namespace

int main() {
    try {
        test_english_numbers();
        test_english_index_tts_style();
        test_japanese_katakana();
        std::cout << "text_normalization_test passed\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "text_normalization_test failed: " << ex.what() << "\n";
        return 1;
    }
}
