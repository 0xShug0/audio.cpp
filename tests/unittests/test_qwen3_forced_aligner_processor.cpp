#include "engine/models/qwen3_forced_aligner/processor.h"

#include "test_assert.h"

#include <iostream>
#include <string>

namespace {

void test_punctuation_only_text_has_no_alignable_words() {
    for (const std::string text : {".", "...", "?", " "}) {
        engine::test::require(
            !engine::models::qwen3_forced_aligner::has_alignable_words(text, "English"),
            "punctuation-only forced-aligner text should not be alignable");
    }
}

void test_word_text_has_alignable_words() {
    engine::test::require(
        engine::models::qwen3_forced_aligner::has_alignable_words("Some call me nature.", "English"),
        "English words should be alignable");
    engine::test::require(
        engine::models::qwen3_forced_aligner::has_alignable_words("你好。", "Chinese"),
        "Chinese words should be alignable");
}

}  // namespace

int main() {
    try {
        test_punctuation_only_text_has_no_alignable_words();
        test_word_text_has_alignable_words();
        std::cout << "qwen3_forced_aligner_processor_test passed\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "qwen3_forced_aligner_processor_test failed: " << ex.what() << "\n";
        return 1;
    }
}
