#include "engine/community_models/confucius4_r2t2/text_postprocess.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace engine::community_models::confucius4_r2t2 {
namespace {

std::vector<uint32_t> utf8_to_codepoints(const std::string & text) {
    std::vector<uint32_t> out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        const auto byte = static_cast<unsigned char>(text[i]);
        if (byte < 0x80) {
            out.push_back(byte);
            i += 1;
        } else if ((byte >> 5) == 0x6 && i + 1 < text.size() &&
                   (static_cast<unsigned char>(text[i + 1]) >> 6) == 0x2) {
            out.push_back(((byte & 0x1F) << 6) | (static_cast<unsigned char>(text[i + 1]) & 0x3F));
            i += 2;
        } else if ((byte >> 4) == 0xE && i + 2 < text.size() &&
                   (static_cast<unsigned char>(text[i + 1]) >> 6) == 0x2 &&
                   (static_cast<unsigned char>(text[i + 2]) >> 6) == 0x2) {
            out.push_back(((byte & 0x0F) << 12) | ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
                (static_cast<unsigned char>(text[i + 2]) & 0x3F));
            i += 3;
        } else if ((byte >> 3) == 0x1E && i + 3 < text.size() &&
                   (static_cast<unsigned char>(text[i + 1]) >> 6) == 0x2 &&
                   (static_cast<unsigned char>(text[i + 2]) >> 6) == 0x2 &&
                   (static_cast<unsigned char>(text[i + 3]) >> 6) == 0x2) {
            out.push_back(((byte & 0x07) << 18) | ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 12) |
                ((static_cast<unsigned char>(text[i + 2]) & 0x3F) << 6) |
                (static_cast<unsigned char>(text[i + 3]) & 0x3F));
            i += 4;
        } else {
            out.push_back(0xFFFD);
            i += 1;
        }
    }
    return out;
}

void append_utf8(std::string & out, uint32_t codepoint) {
    if (codepoint < 0x80) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
}

std::string codepoints_to_utf8(const std::vector<uint32_t> & codepoints) {
    std::string out;
    out.reserve(codepoints.size() * 3);
    for (const uint32_t codepoint : codepoints) {
        append_utf8(out, codepoint);
    }
    return out;
}

bool is_chinese_codepoint(uint32_t codepoint) {
    return codepoint >= 0x4E00 && codepoint <= 0x9FFF;
}

bool is_ascii_digit_or_letter(uint32_t codepoint) {
    return (codepoint >= '0' && codepoint <= '9') || (codepoint >= 'a' && codepoint <= 'z') ||
        (codepoint >= 'A' && codepoint <= 'Z');
}

bool is_python_space(uint32_t codepoint) {
    switch (codepoint) {
        case 0x20:  // space
        case 0x09:  // \t
        case 0x0A:  // \n
        case 0x0B:
        case 0x0C:
        case 0x0D:  // \r
        case 0x1C:
        case 0x1D:
        case 0x1E:
        case 0x1F:
        case 0x85:
        case 0xA0:
        case 0x1680:
        case 0x2000:
        case 0x2001:
        case 0x2002:
        case 0x2003:
        case 0x2004:
        case 0x2005:
        case 0x2006:
        case 0x2007:
        case 0x2008:
        case 0x2009:
        case 0x200A:
        case 0x2028:
        case 0x2029:
        case 0x202F:
        case 0x205F:
        case 0x3000:
            return true;
        default:
            return false;
    }
}

/// The punctuation set matched by _ALL_PUNCT_PAT in the reference code.
bool is_normalized_punctuation(uint32_t codepoint) {
    switch (codepoint) {
        case ',':
        case '.':
        case '!':
        case '?':
        case ';':
        case ':':
        case '(':
        case ')':
        case 0xFF0C:  // ，
        case 0x3002:  // 。
        case 0xFF01:  // ！
        case 0xFF1F:  // ？
        case 0xFF1B:  // ；
        case 0xFF1A:  // ：
        case 0xFF08:  // （
        case 0xFF09:  // ）
            return true;
        default:
            return false;
    }
}

uint32_t en_to_zh_punctuation(uint32_t codepoint) {
    switch (codepoint) {
        case ',':
            return 0xFF0C;
        case '.':
            return 0x3002;
        case '!':
            return 0xFF01;
        case '?':
            return 0xFF1F;
        case ';':
            return 0xFF1B;
        case ':':
            return 0xFF1A;
        case '(':
            return 0xFF08;
        case ')':
            return 0xFF09;
        default:
            return codepoint;
    }
}

uint32_t zh_to_en_punctuation(uint32_t codepoint) {
    switch (codepoint) {
        case 0xFF0C:
            return ',';
        case 0x3002:
            return '.';
        case 0xFF01:
            return '!';
        case 0xFF1F:
            return '?';
        case 0xFF1B:
            return ';';
        case 0xFF1A:
            return ':';
        case 0xFF08:
            return '(';
        case 0xFF09:
            return ')';
        default:
            return codepoint;
    }
}

bool is_ascii_alnum_or_quote(uint32_t codepoint) {
    return is_ascii_digit_or_letter(codepoint) || codepoint == '"' || codepoint == '\'';
}

std::vector<uint32_t> fix_char_repeats(const std::vector<uint32_t> & s, int threshold) {
    std::vector<uint32_t> out;
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        size_t count = 1;
        while (i + count < n && s[i + count] == s[i]) {
            ++count;
        }
        if (static_cast<int>(count) > threshold) {
            out.push_back(s[i]);
        } else {
            out.insert(out.end(), s.begin() + static_cast<std::ptrdiff_t>(i), s.begin() + static_cast<std::ptrdiff_t>(i + count));
        }
        i += count;
    }
    return out;
}

std::vector<uint32_t> fix_pattern_repeats(const std::vector<uint32_t> & s, int threshold, int max_len) {
    const size_t n = s.size();
    const size_t min_repeat_chars = static_cast<size_t>(threshold) * 2;
    if (n < min_repeat_chars) {
        return s;
    }
    std::vector<uint32_t> result;
    size_t i = 0;
    bool exhausted = false;
    while (i + min_repeat_chars <= n) {
        bool found = false;
        for (int k = 1; k <= max_len; ++k) {
            const size_t pattern_length = static_cast<size_t>(k);
            if (i + static_cast<size_t>(k) * static_cast<size_t>(threshold) > n) {
                break;
            }
            const auto pattern_begin = s.begin() + static_cast<std::ptrdiff_t>(i);
            const auto pattern = std::vector<uint32_t>(pattern_begin, pattern_begin + static_cast<std::ptrdiff_t>(pattern_length));
            bool valid = true;
            for (int rep = 1; rep < threshold; ++rep) {
                const size_t start_idx = i + static_cast<size_t>(rep) * pattern_length;
                if (std::vector<uint32_t>(s.begin() + static_cast<std::ptrdiff_t>(start_idx), s.begin() + static_cast<std::ptrdiff_t>(start_idx + pattern_length)) != pattern) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                size_t end_index = i + pattern_length * static_cast<size_t>(threshold);
                while (end_index + pattern_length <= n &&
                       std::vector<uint32_t>(s.begin() + static_cast<std::ptrdiff_t>(end_index), s.begin() + static_cast<std::ptrdiff_t>(end_index + pattern_length)) == pattern) {
                    end_index += pattern_length;
                }
                result.insert(result.end(), pattern.begin(), pattern.end());
                const auto rest = fix_pattern_repeats(
                    std::vector<uint32_t>(s.begin() + static_cast<std::ptrdiff_t>(end_index), s.end()), threshold, max_len);
                result.insert(result.end(), rest.begin(), rest.end());
                i = n;
                found = true;
                break;
            }
        }
        if (found) {
            exhausted = true;
            break;
        }
        result.push_back(s[i]);
        ++i;
    }
    if (!exhausted && i <= n) {
        result.insert(result.end(), s.begin() + static_cast<std::ptrdiff_t>(std::min(i, n)), s.end());
    }
    return result;
}

/// str.strip() semantics over the Python whitespace set.
std::string strip_python(const std::string & text) {
    const auto codepoints = utf8_to_codepoints(text);
    size_t begin = 0;
    size_t end = codepoints.size();
    while (begin < end && is_python_space(codepoints[begin])) {
        ++begin;
    }
    while (end > begin && is_python_space(codepoints[end - 1])) {
        --end;
    }
    return codepoints_to_utf8(std::vector<uint32_t>(codepoints.begin() + static_cast<std::ptrdiff_t>(begin), codepoints.begin() + static_cast<std::ptrdiff_t>(end)));
}

bool contains_ascii_needle(const std::string & text, const char * needle) {
    return text.find(needle) != std::string::npos;
}

bool starts_with_ascii(const std::string & text, const char * prefix) {
    const size_t length = std::char_traits<char>::length(prefix);
    if (text.size() < length) {
        return false;
    }
    return text.compare(0, length, prefix) == 0;
}

std::string to_lower_ascii(const std::string & text) {
    std::string out = text;
    for (char & ch : out) {
        if (ch >= 'A' && ch <= 'Z') {
            ch += 'a' - 'A';
        }
    }
    return out;
}

R2T2ParsedOutput parse_tagged_output(const std::string & raw, const std::string & user_language, bool apply_repetition_fix) {
    if (user_language.empty()) {
        std::string s = strip_python(raw);
        if (s.empty()) {
            return {};
        }
        if (apply_repetition_fix) {
            s = detect_and_fix_repetitions(s);
        }
        if (!contains_asr_text_tag(s)) {
            return {std::string(), strip_python(s)};
        }
        const std::string meta_part = text_before_asr_tag(s);
        const std::string text_part = text_after_asr_tag(s);
        const std::string meta_lower = to_lower_ascii(meta_part);
        if (meta_lower.find("language none") != std::string::npos) {
            const std::string t = strip_python(text_part);
            if (t.empty()) {
                return {};
            }
            return {std::string(), t};
        }
        std::string language;
        size_t line_begin = 0;
        while (line_begin <= meta_part.size()) {
            const size_t line_end = meta_part.find('\n', line_begin);
            const std::string line = strip_python(meta_part.substr(
                line_begin,
                line_end == std::string::npos ? std::string::npos : line_end - line_begin));
            if (!line.empty()) {
                if (starts_with_ascii(line, kLanguagePrefix)) {
                    const std::string value = strip_python(line.substr(std::char_traits<char>::length(kLanguagePrefix)));
                    if (!value.empty()) {
                        language = normalize_language_name(value);
                    }
                    break;
                }
            }
            if (line_end == std::string::npos) {
                break;
            }
            line_begin = line_end + 1;
        }
        return {language, strip_python(text_part)};
    }
    // Forced language: the model output is treated as pure transcription text.
    std::string s = strip_python(raw);
    if (!s.empty() && apply_repetition_fix) {
        s = detect_and_fix_repetitions(s);
    }
    return {user_language, s};
}

}  // namespace

std::string sanitize_utf8_lossy(const std::string & text) {
    std::string out;
    out.reserve(text.size() + 16);
    size_t i = 0;
    while (i < text.size()) {
        const auto byte = static_cast<unsigned char>(text[i]);
        const size_t remaining = text.size() - i;
        if (byte < 0x80) {
            out.push_back(text[i]);
            i += 1;
            continue;
        }
        size_t expected = 0;
        if ((byte >> 5) == 0x6) {
            expected = 2;
        } else if ((byte >> 4) == 0xE) {
            expected = 3;
        } else if ((byte >> 3) == 0x1E) {
            expected = 4;
        }
        bool valid = false;
        if (expected != 0 && remaining >= expected) {
            valid = true;
            for (size_t j = 1; j < expected; ++j) {
                if ((static_cast<unsigned char>(text[i + j]) >> 6) != 0x2) {
                    valid = false;
                    break;
                }
            }
        }
        if (valid) {
            out.append(text, i, expected);
            i += expected;
        } else {
            out.append(kReplacementChar);
            i += 1;
        }
    }
    return out;
}

std::string normalize_punct_by_context(const std::string & text) {
    const auto codepoints = utf8_to_codepoints(text);
    std::vector<uint32_t> out;
    out.reserve(codepoints.size());
    for (size_t pos = 0; pos < codepoints.size(); ++pos) {
        const uint32_t punct = codepoints[pos];
        if (!is_normalized_punctuation(punct)) {
            out.push_back(punct);
            continue;
        }
        // Find the closest preceding non-space character.
        uint32_t previous = 0;
        bool has_previous = false;
        for (size_t back = pos; back-- > 0;) {
            if (!is_python_space(codepoints[back])) {
                previous = codepoints[back];
                has_previous = true;
                break;
            }
        }
        if (!has_previous) {
            out.push_back(punct);
            continue;
        }
        if (is_chinese_codepoint(previous)) {
            out.push_back(en_to_zh_punctuation(punct));
        } else if (previous < 0x80 && is_ascii_alnum_or_quote(previous)) {
            out.push_back(zh_to_en_punctuation(punct));
        } else {
            out.push_back(punct);
        }
    }
    return codepoints_to_utf8(out);
}

std::string detect_and_fix_repetitions(const std::string & text, int threshold) {
    auto codepoints = utf8_to_codepoints(text);
    codepoints = fix_char_repeats(codepoints, threshold);
    codepoints = fix_pattern_repeats(codepoints, threshold, 20);
    return codepoints_to_utf8(codepoints);
}

std::string remove_spaces_between_chinese(const std::string & text) {
    const auto codepoints = utf8_to_codepoints(text);
    std::vector<uint32_t> out;
    out.reserve(codepoints.size());
    size_t i = 0;
    while (i < codepoints.size()) {
        if (is_python_space(codepoints[i])) {
            size_t j = i;
            while (j < codepoints.size() && is_python_space(codepoints[j])) {
                ++j;
            }
            const bool between_chinese = i > 0 && j < codepoints.size() &&
                is_chinese_codepoint(codepoints[i - 1]) && is_chinese_codepoint(codepoints[j]);
            if (!between_chinese) {
                out.insert(out.end(), codepoints.begin() + static_cast<std::ptrdiff_t>(i), codepoints.begin() + static_cast<std::ptrdiff_t>(j));
            }
            i = j;
            continue;
        }
        out.push_back(codepoints[i]);
        ++i;
    }
    return codepoints_to_utf8(out);
}

std::string normalize_language_name(const std::string & language) {
    const auto codepoints = utf8_to_codepoints(language);
    size_t begin = 0;
    size_t end = codepoints.size();
    while (begin < end && is_python_space(codepoints[begin])) {
        ++begin;
    }
    while (end > begin && is_python_space(codepoints[end - 1])) {
        --end;
    }
    if (begin >= end) {
        throw std::runtime_error("language is empty");
    }
    std::string out;
    for (size_t i = begin; i < end; ++i) {
        uint32_t codepoint = codepoints[i];
        if (i == begin) {
            if (codepoint >= 'a' && codepoint <= 'z') {
                codepoint -= 'a' - 'A';
            }
        } else if (codepoint >= 'A' && codepoint <= 'Z') {
            codepoint += 'a' - 'A';
        }
        append_utf8(out, codepoint);
    }
    return out;
}

std::string resolve_language(const std::string & language) {
    const std::string trimmed = strip_python(language);
    if (trimmed.empty()) {
        return {};
    }
    // The model spec and the WebUI speak ISO-639 style codes; the R2T2 prompt
    // wants the canonical names from config.json.
    static constexpr std::pair<const char *, const char *> kIsoToName[] = {
        {"zh", "Chinese"},      {"en", "English"},     {"yue", "Cantonese"},
        {"ar", "Arabic"},       {"de", "German"},      {"fr", "French"},
        {"es", "Spanish"},      {"pt", "Portuguese"},  {"id", "Indonesian"},
        {"it", "Italian"},      {"ko", "Korean"},      {"ru", "Russian"},
        {"th", "Thai"},         {"vi", "Vietnamese"},  {"ja", "Japanese"},
        {"tr", "Turkish"},      {"hi", "Hindi"},       {"ms", "Malay"},
        {"nl", "Dutch"},        {"sv", "Swedish"},     {"da", "Danish"},
        {"fi", "Finnish"},      {"pl", "Polish"},      {"cs", "Czech"},
        {"fil", "Filipino"},    {"fa", "Persian"},     {"el", "Greek"},
        {"hu", "Hungarian"},    {"mk", "Macedonian"},  {"ro", "Romanian"},
        {"auto", ""},           {"none", ""},
    };
    const std::string lowered = to_lower_ascii(trimmed);
    for (const auto & [code, name] : kIsoToName) {
        if (lowered == code) {
            return name;
        }
    }
    return normalize_language_name(trimmed);
}

R2T2ParsedOutput parse_language_output(const std::string & raw, const std::string & user_language) {    // The reference strips only the right side when the forced language is
    // exactly "English", and both sides otherwise.
    std::string s;
    if (user_language == "English") {
        auto codepoints = utf8_to_codepoints(raw);
        size_t end = codepoints.size();
        while (end > 0 && is_python_space(codepoints[end - 1])) {
            --end;
        }
        s = codepoints_to_utf8(std::vector<uint32_t>(codepoints.begin(), codepoints.begin() + static_cast<std::ptrdiff_t>(end)));
    } else {
        s = strip_python(raw);
    }
    if (s.empty()) {
        return {};
    }
    if (!user_language.empty()) {
        return {user_language, s};
    }
    return parse_tagged_output(s, std::string(), /*apply_repetition_fix=*/false);
}

R2T2ParsedOutput parse_asr_output(const std::string & raw, const std::string & user_language) {
    return parse_tagged_output(raw, user_language, /*apply_repetition_fix=*/true);
}

std::string truncate_at_pipe(const std::string & text) {
    const size_t pipe = text.find('|');
    return pipe == std::string::npos ? text : text.substr(0, pipe);
}

bool contains_asr_text_tag(const std::string & text) {
    return contains_ascii_needle(text, kAsrTextTag);
}

std::string text_before_asr_tag(const std::string & text) {
    const size_t tag = text.find(kAsrTextTag);
    return tag == std::string::npos ? std::string() : text.substr(0, tag);
}

std::string text_after_asr_tag(const std::string & text) {
    const size_t tag = text.find(kAsrTextTag);
    if (tag == std::string::npos) {
        return {};
    }
    return text.substr(tag + std::char_traits<char>::length(kAsrTextTag));
}

std::size_t utf8_codepoint_count(const std::string & text) {
    return utf8_to_codepoints(text).size();
}

std::string utf8_slice_from_codepoint(const std::string & text, std::size_t start_codepoint) {
    const auto codepoints = utf8_to_codepoints(text);
    if (start_codepoint >= codepoints.size()) {
        return {};
    }
    return codepoints_to_utf8(std::vector<uint32_t>(
        codepoints.begin() + static_cast<std::ptrdiff_t>(start_codepoint),
        codepoints.end()));
}

bool ends_with_rollback_punctuation(const std::string & trimmed_text) {
    static constexpr std::array<uint32_t, 12> kPunct = {
        0xFF0C, 0x3002, 0xFF01, 0xFF1F, 0x3001, 0xFF1B, 0xFF1A, ',', '.', '!', '?',
    };
    // The reference list also covers ':' and ';'.
    const auto codepoints = utf8_to_codepoints(trimmed_text);
    if (codepoints.empty()) {
        return false;
    }
    const uint32_t last = codepoints.back();
    for (const uint32_t punct : kPunct) {
        if (last == punct) {
            return true;
        }
    }
    return last == ':' || last == ';';
}

}  // namespace engine::community_models::confucius4_r2t2
