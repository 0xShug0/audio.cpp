#include "engine/community_models/echo_tts/tokenizer.h"

#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace engine::models::echo_tts {
namespace {

// UTF-8 spellings of the codepoints upstream rewrites. Searching for these as
// byte substrings is safe: UTF-8 is self-synchronising, so a valid multi-byte
// sequence can never match across a character boundary.
constexpr std::string_view kEllipsis = "\xE2\x80\xA6";        // U+2026
constexpr std::string_view kRightSingleQuote = "\xE2\x80\x99"; // U+2019
constexpr std::string_view kRightDoubleQuote = "\xE2\x80\x9D"; // U+201D
constexpr std::string_view kEmDash = "\xE2\x80\x94";           // U+2014

void replace_all(std::string & text, std::string_view needle, std::string_view replacement) {
    if (needle.empty()) {
        return;
    }
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        text.replace(pos, needle.size(), replacement);
        pos += replacement.size();
    }
}

}  // namespace

std::string normalize_echo_text(const std::string & text) {
    std::string out = text;

    replace_all(out, kEllipsis, "...");
    replace_all(out, kRightSingleQuote, "'");
    // Upstream applies the right-double-quote rewrite twice and never rewrites
    // the *left* double quote (U+201C). Reproduced verbatim so token streams
    // match the reference implementation; see inference.py::tokenizer_encode.
    replace_all(out, kRightDoubleQuote, "\"");
    replace_all(out, kRightDoubleQuote, "\"");
    replace_all(out, "\n", " ");
    replace_all(out, ":", ",");
    replace_all(out, ";", ",");
    replace_all(out, kEmDash, ", ");

    const bool has_bracket = !out.empty() && (out.front() == '[' || out.front() == '(');
    const bool has_speaker_tag =
        out.find("S1") != std::string::npos || out.find("S2") != std::string::npos;
    if (!has_bracket && !has_speaker_tag) {
        out = "[S1] " + out;
    }
    return out;
}

EchoTokenizedText tokenize_echo_text(
    const std::string & text,
    int64_t max_length,
    bool normalize,
    bool pad_to_max) {
    if (max_length <= 0) {
        throw std::runtime_error("Echo-TTS tokenizer requires a positive max_length");
    }

    EchoTokenizedText out;
    out.normalized_text = normalize ? normalize_echo_text(text) : text;

    std::vector<int32_t> ids;
    ids.reserve(out.normalized_text.size() + 1);
    ids.push_back(0);  // BOS
    for (const char byte : out.normalized_text) {
        ids.push_back(static_cast<int32_t>(static_cast<unsigned char>(byte)));
    }

    const auto encoded_length = static_cast<int64_t>(ids.size());
    const int64_t length = std::min<int64_t>(encoded_length, max_length);
    out.truncated = encoded_length > max_length;

    const int64_t output_length = pad_to_max ? max_length : length;
    out.input_ids.assign(static_cast<size_t>(output_length), 0);
    out.mask.assign(static_cast<size_t>(output_length), 0.0F);
    for (int64_t i = 0; i < length; ++i) {
        out.input_ids[static_cast<size_t>(i)] = ids[static_cast<size_t>(i)];
        out.mask[static_cast<size_t>(i)] = 1.0F;
    }
    return out;
}

}  // namespace engine::models::echo_tts
