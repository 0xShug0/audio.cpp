#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace engine::text {

// Returns true if the language tag indicates Cantonese/Yue. Covers
// "yue", "yue-Hant", "zh-yue", "cantonese", "zh-HK", "zh-MO" variants.
// Comparison is case-insensitive and accepts '-' / '_' separators.
bool is_cantonese_language(std::string_view language) noexcept;

// Returns true if the UTF-8 text contains at least one Traditional
// character that would be converted by convert_traditional_to_simplified().
bool contains_traditional_characters(std::string_view text);

// Convert Traditional Chinese characters to Simplified Chinese.
// Uses OpenCC TSCharacters.txt mapping (3222 single-char entries).
// Characters without a mapping are left unchanged. Invalid UTF-8
// throws std::runtime_error (label: "chinese variant text").
std::string convert_traditional_to_simplified(std::string_view text);

// Convert Traditional -> Simplified for Mandarin/Chinese language hints.
// Preserve Cantonese and explicit non-Chinese languages. With no language or
// "auto"/"none", preserve text containing kana; otherwise convert as before.
// Kanji-only Japanese needs an explicit Japanese language hint.
std::string maybe_convert_traditional_to_simplified(
    std::string_view text,
    std::string_view language);

// Missing language uses the same auto-detection policy as an empty hint.
inline std::string maybe_convert_traditional_to_simplified_opt(
    std::string_view text,
    const std::optional<std::string> & language) {
    return maybe_convert_traditional_to_simplified(
        text, language ? std::string_view(*language) : std::string_view());
}

}  // namespace engine::text
