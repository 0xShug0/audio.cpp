#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace engine::community_models::confucius4_r2t2 {

// Faithful C++ ports of the R2T2 text post-processing pipeline
// (r2t2/r2t2_asr.py and qwen_asr/inference/utils.py). All functions operate
// on UTF-8 text and mirror the Python string semantics so the streaming
// state machine reproduces the reference outputs token for token.

inline constexpr const char * kAsrTextTag = "<asr_text>";
inline constexpr const char * kLanguagePrefix = "language ";
inline constexpr const char * kReplacementChar = "\xEF\xBF\xBD";

struct R2T2ParsedOutput {
    std::string language;
    std::string text;
};

/// Replaces every maximal invalid UTF-8 subsequence with U+FFFD, mirroring
/// the replacement characters HuggingFace decode produces for truncated
/// multibyte tokens.
std::string sanitize_utf8_lossy(const std::string & text);

/// R2T2 punctuation normalization: each punctuation mark is rewritten to the
/// Chinese or ASCII variant based on the nearest preceding character.
std::string normalize_punct_by_context(const std::string & text);

/// qwen_asr detect_and_fix_repetitions: collapse degenerate character and
/// pattern repetition hallucinations (threshold counted in code points).
std::string detect_and_fix_repetitions(const std::string & text, int threshold = 20);

/// Removes whitespace runs between two Chinese characters.
std::string remove_spaces_between_chinese(const std::string & text);

/// First letter uppercase, remaining letters lowercase ("cHINese" -> "Chinese").
std::string normalize_language_name(const std::string & language);

/// Maps the ISO-639 style codes used by the model spec (and the UI) onto the
/// canonical language names the R2T2 prompt expects, e.g. "zh" -> "Chinese",
/// "en" -> "English". Empty input and "auto"/"Auto" mean automatic language
/// detection. Unrecognised values are passed through normalize_language_name so
/// callers can still validate them against the model's supported list.
std::string resolve_language(const std::string & language);

/// R2T2 parse_language_output: extracts the detected language from raw output
/// that is expected to carry "language X<asr_text>text".
R2T2ParsedOutput parse_language_output(const std::string & raw, const std::string & user_language);

/// qwen_asr parse_asr_output with repetition repair.
R2T2ParsedOutput parse_asr_output(const std::string & raw, const std::string & user_language);

/// text.split("|")[0]
std::string truncate_at_pipe(const std::string & text);

bool contains_asr_text_tag(const std::string & text);

/// Part before the <asr_text> tag (empty when the tag is absent).
std::string text_before_asr_tag(const std::string & text);

/// Part after the <asr_text> tag (empty when the tag is absent).
std::string text_after_asr_tag(const std::string & text);

bool ends_with_rollback_punctuation(const std::string & trimmed_text);

/// Number of Unicode code points (what Python's len() counts on str).
std::size_t utf8_codepoint_count(const std::string & text);

/// Returns the suffix starting at `start_codepoint` (never splits a code
/// point, unlike a byte-based substr). Used to reproduce the reference
/// integrator's `fixed_text[len(last_fixed_text):]` slice.
std::string utf8_slice_from_codepoint(const std::string & text, std::size_t start_codepoint);

}  // namespace engine::community_models::confucius4_r2t2
