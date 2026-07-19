#include "engine/models/outetts/tokenizer.h"

#include "test_assert.h"

#include <iostream>

int main() try {
  using engine::models::outetts::estimate_text_generation_budget;

  const auto empty = estimate_text_generation_budget("");
  engine::test::require_eq(empty.words, int64_t{0}, "empty words");
  engine::test::require_eq(empty.non_whitespace_codepoints, int64_t{0},
                           "empty codepoints");
  engine::test::require_eq(empty.recommended_max_new_tokens, int64_t{384},
                           "empty minimum budget");

  const auto words =
      estimate_text_generation_budget("one two three four five six");
  engine::test::require_eq(words.words, int64_t{6}, "word count");
  engine::test::require_eq(words.non_whitespace_codepoints, int64_t{22},
                           "non-whitespace codepoints");
  engine::test::require_eq(words.recommended_max_new_tokens, int64_t{560},
                           "word-dominated budget");

  const auto characters = estimate_text_generation_budget(
      "supercalifragilisticexpialidocious");
  engine::test::require_eq(characters.words, int64_t{1},
                           "long word count");
  engine::test::require_eq(characters.non_whitespace_codepoints, int64_t{34},
                           "long word codepoints");
  engine::test::require_eq(characters.recommended_max_new_tokens, int64_t{536},
                           "character-dominated budget");

  const auto unicode = estimate_text_generation_budget(u8"Zażółć gęślą");
  engine::test::require_eq(unicode.words, int64_t{2}, "UTF-8 word count");
  engine::test::require_eq(unicode.non_whitespace_codepoints, int64_t{11},
                           "UTF-8 codepoints");
  engine::test::require_eq(unicode.recommended_max_new_tokens, int64_t{384},
                           "UTF-8 minimum budget");

  std::cout << "outetts_generation_budget_test passed\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "outetts_generation_budget_test failed: " << error.what()
            << "\n";
  return 1;
}
