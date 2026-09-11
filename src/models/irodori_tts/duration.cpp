#include "engine/models/irodori_tts/duration.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::irodori_tts {
namespace {

bool starts_with(const std::string & text, size_t pos, const char * pattern) {
    const std::string value(pattern);
    return pos + value.size() <= text.size() && text.compare(pos, value.size(), value) == 0;
}

int utf8_codepoint(const std::string & text, size_t & pos) {
    if (pos >= text.size()) {
        return -1;
    }
    const unsigned char ch = static_cast<unsigned char>(text[pos]);
    int width = 1;
    int code = 0;
    if ((ch & 0x80U) == 0) {
        code = ch;
        width = 1;
    } else if ((ch & 0xE0U) == 0xC0U) {
        code = ch & 0x1FU;
        width = 2;
    } else if ((ch & 0xF0U) == 0xE0U) {
        code = ch & 0x0FU;
        width = 3;
    } else if ((ch & 0xF8U) == 0xF0U) {
        code = ch & 0x07U;
        width = 4;
    } else {
        throw std::runtime_error("Irodori-TTS duration feature text is invalid UTF-8");
    }
    if (pos + static_cast<size_t>(width) > text.size()) {
        throw std::runtime_error("Irodori-TTS duration feature text has truncated UTF-8");
    }
    for (int i = 1; i < width; ++i) {
        const unsigned char next = static_cast<unsigned char>(text[pos + static_cast<size_t>(i)]);
        if ((next & 0xC0U) != 0x80U) {
            throw std::runtime_error("Irodori-TTS duration feature text has invalid UTF-8 continuation");
        }
        code = (code << 6) | (next & 0x3FU);
    }
    pos += static_cast<size_t>(width);
    return code;
}

bool is_kana(int code) {
    return (0x3040 <= code && code <= 0x309F) || (0x30A0 <= code && code <= 0x30FF);
}

bool is_kanji(int code) {
    return (0x3400 <= code && code <= 0x4DBF) ||
        (0x4E00 <= code && code <= 0x9FFF) ||
        (0xF900 <= code && code <= 0xFAFF) ||
        (0x20000 <= code && code <= 0x2FA1F);
}

bool is_ascii_alnum(int code) {
    return ((code >= '0' && code <= '9') ||
            (code >= 'A' && code <= 'Z') ||
            (code >= 'a' && code <= 'z'));
}

float log1p_cap(int count, int cap) {
    const int bounded = std::min(std::max(count, 0), cap);
    return std::log1p(static_cast<float>(bounded)) / std::log1p(static_cast<float>(cap));
}

float log1p_cap_float(float value, float cap) {
    const float bounded = std::min(std::max(value, 0.0F), cap);
    return std::log1p(bounded) / std::log1p(cap);
}

int count_substring(const std::string & text, const std::string & needle) {
    if (needle.empty()) {
        return 0;
    }
    int count = 0;
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

int count_annotation_emojis(const std::string & text) {
    static constexpr const char * kEmojis[] = {
        "\xE2\x8F\xA9", "\xE2\x8F\xB1\xEF\xB8\x8F", "\xE2\x8F\xB8\xEF\xB8\x8F",
        "\xF0\x9F\x8C\xAC\xEF\xB8\x8F", "\xF0\x9F\x8D\xAD", "\xF0\x9F\x8E\x9B\xEF\xB8\x8F",
        "\xF0\x9F\x8E\xAD", "\xF0\x9F\x8E\xB5", "\xF0\x9F\x90\xA2", "\xF0\x9F\x90\xB1",
        "\xF0\x9F\x91\x82", "\xF0\x9F\x91\x83", "\xF0\x9F\x91\x85", "\xF0\x9F\x91\x8C",
        "\xF0\x9F\x91\x8F", "\xF0\x9F\x92\x8B", "\xF0\x9F\x92\xA5", "\xF0\x9F\x92\xA6",
        "\xF0\x9F\x92\xAA", "\xF0\x9F\x93\x84", "\xF0\x9F\x93\x9E", "\xF0\x9F\x93\xA2",
        "\xF0\x9F\x93\xA3", "\xF0\x9F\x98\x86", "\xF0\x9F\x98\x8A", "\xF0\x9F\x98\x8C",
        "\xF0\x9F\x98\x8E", "\xF0\x9F\x98\x8F", "\xF0\x9F\x98\x92", "\xF0\x9F\x98\x96",
        "\xF0\x9F\x98\x9F", "\xF0\x9F\x98\xA0", "\xF0\x9F\x98\xAA", "\xF0\x9F\x98\xAD",
        "\xF0\x9F\x98\xAE", "\xF0\x9F\x98\xAE\xE2\x80\x8D\xF0\x9F\x92\xA8", "\xF0\x9F\x98\xB0",
        "\xF0\x9F\x98\xB1", "\xF0\x9F\x98\xB2", "\xF0\x9F\x98\xB4", "\xF0\x9F\x99\x84",
        "\xF0\x9F\x99\x8F", "\xF0\x9F\xA4\x90", "\xF0\x9F\xA4\x94", "\xF0\x9F\xA4\xA2",
        "\xF0\x9F\xA4\xA7", "\xF0\x9F\xA4\xAD", "\xF0\x9F\xA5\xA4", "\xF0\x9F\xA5\xB1",
        "\xF0\x9F\xA5\xB4", "\xF0\x9F\xA5\xB5", "\xF0\x9F\xA5\xB9", "\xF0\x9F\xA5\xBA",
        "\xF0\x9F\xAB\xA3", "\xF0\x9F\xAB\xB6", "\xF0\x9F\x93\x96",
    };
    int count = 0;
    for (size_t pos = 0; pos < text.size();) {
        size_t best = 0;
        for (const char * emoji : kEmojis) {
            const size_t len = std::char_traits<char>::length(emoji);
            if (len > best && starts_with(text, pos, emoji)) {
                best = len;
            }
        }
        if (best > 0) {
            ++count;
            pos += best;
        } else {
            size_t next = pos;
            (void) utf8_codepoint(text, next);
            pos = next;
        }
    }
    return count;
}

}  // namespace

std::vector<float> build_irodori_duration_features(
    const std::string & text,
    int64_t token_count,
    int64_t max_text_len,
    bool has_speaker) {
    if (max_text_len <= 0) {
        throw std::runtime_error("Irodori-TTS duration max_text_len must be positive");
    }
    int64_t char_count = 0;
    int64_t kana_count = 0;
    int64_t kanji_count = 0;
    int64_t alnum_count = 0;
    for (size_t pos = 0; pos < text.size();) {
        const int code = utf8_codepoint(text, pos);
        ++char_count;
        kana_count += is_kana(code) ? 1 : 0;
        kanji_count += is_kanji(code) ? 1 : 0;
        alnum_count += is_ascii_alnum(code) ? 1 : 0;
    }
    char_count = std::max<int64_t>(char_count, 1);
    const int emoji_count = count_annotation_emojis(text);
    const int period_count = count_substring(text, "\xE3\x80\x82") + count_substring(text, ".");
    const int comma_count = count_substring(text, "\xE3\x80\x81") + count_substring(text, ",");
    const int long_vowel_count = count_substring(text, "\xE3\x83\xBC");
    const int ellipsis_count = count_substring(text, "\xE2\x80\xA6");
    const int exclamation_count = count_substring(text, "\xEF\xBC\x81") + count_substring(text, "!");
    const int question_count = count_substring(text, "\xEF\xBC\x9F") + count_substring(text, "?");

    return {
        std::min(std::max(static_cast<float>(token_count), 0.0F), static_cast<float>(max_text_len)) /
            static_cast<float>(max_text_len),
        log1p_cap_float(static_cast<float>(char_count), 512.0F),
        static_cast<float>(token_count) / static_cast<float>(char_count),
        log1p_cap(period_count, 8),
        log1p_cap(comma_count, 16),
        log1p_cap(long_vowel_count, 8),
        log1p_cap(ellipsis_count, 8),
        log1p_cap(exclamation_count, 8),
        log1p_cap(question_count, 8),
        log1p_cap(emoji_count, 8),
        static_cast<float>(kana_count) / static_cast<float>(char_count),
        static_cast<float>(kanji_count) / static_cast<float>(char_count),
        static_cast<float>(alnum_count) / static_cast<float>(char_count),
        has_speaker ? 1.0F : 0.0F,
    };
}

}  // namespace engine::models::irodori_tts
