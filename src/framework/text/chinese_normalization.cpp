#include "engine/framework/text/chinese_normalization.h"

#include "engine/framework/text/text_normalization.h"
#include "engine/framework/text/utf8.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::text {
namespace {

bool is_ascii_digit(char ch) {
    return ch >= '0' && ch <= '9';
}

bool is_ascii_alpha(char ch) {
    return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z');
}

bool starts_with_at(const std::string & text, size_t offset, const std::string & needle) {
    return offset + needle.size() <= text.size() && text.compare(offset, needle.size(), needle) == 0;
}

const std::string & chinese_digit_word(char digit, bool telephone_one = false) {
    static const std::string digits[] = {"零", "一", "二", "三", "四", "五", "六", "七", "八", "九"};
    static const std::string yao = "幺";
    if (telephone_one && digit == '1') {
        return yao;
    }
    return digits[static_cast<size_t>(digit - '0')];
}

std::string chinese_digits_individually(std::string_view digits, bool telephone_one = false) {
    std::string out;
    for (char digit : digits) {
        out += chinese_digit_word(digit, telephone_one);
    }
    return out;
}

std::string chinese_cardinal_under_10000(int value) {
    if (value == 0) {
        return "零";
    }

    static const std::string units[] = {"", "十", "百", "千"};
    std::string out;
    bool pending_zero = false;
    for (int unit_index = 3; unit_index >= 0; --unit_index) {
        int divisor = 1;
        for (int i = 0; i < unit_index; ++i) {
            divisor *= 10;
        }
        const int digit = value / divisor;
        value %= divisor;
        if (digit == 0) {
            if (!out.empty() && value > 0) {
                pending_zero = true;
            }
            continue;
        }
        if (pending_zero) {
            out += "零";
            pending_zero = false;
        }
        if (!(digit == 1 && unit_index == 1 && out.empty())) {
            if (digit == 2 && unit_index >= 2) {
                out += "两";
            } else {
                out += chinese_digit_word(static_cast<char>('0' + digit));
            }
        }
        out += units[static_cast<size_t>(unit_index)];
    }
    return out;
}

std::string chinese_cardinal_from_digits(const std::string & digits) {
    if (digits.empty()) {
        return "";
    }
    if (digits.size() > 4) {
        return chinese_digits_individually(digits, true);
    }
    return chinese_cardinal_under_10000(std::stoi(digits));
}

std::string uppercase_ascii_and_v(std::string value) {
    std::string out;
    out.reserve(value.size());
    for (size_t i = 0; i < value.size();) {
        const unsigned char ch = static_cast<unsigned char>(value[i]);
        if (i + 1 < value.size() && ch == 0xC3U &&
            (static_cast<unsigned char>(value[i + 1]) == 0xBCU || static_cast<unsigned char>(value[i + 1]) == 0x9CU)) {
            out.push_back('V');
            i += 2;
            continue;
        }
        out.push_back(static_cast<char>(std::toupper(ch)));
        ++i;
    }
    return out;
}

std::string correct_index_tts_pinyin(std::string value) {
    if (value.size() >= 3) {
        const char initial = static_cast<char>(std::tolower(static_cast<unsigned char>(value[0])));
        if (initial == 'j' || initial == 'q' || initial == 'x') {
            const char next = static_cast<char>(std::tolower(static_cast<unsigned char>(value[1])));
            if (next == 'u') {
                value[1] = 'v';
            } else if (value.size() >= 4 &&
                       static_cast<unsigned char>(value[1]) == 0xC3U &&
                       static_cast<unsigned char>(value[2]) == 0xBCU) {
                value.replace(1, 2, "v");
            }
            // The official correct_pinyin only uppercases the j/q/x ü->v cases;
            // other pinyin keeps its original casing.
            return uppercase_ascii_and_v(std::move(value));
        }
    }
    return value;
}

bool is_index_tts_pinyin_candidate(const std::string & candidate) {
    static const std::vector<std::string> initials = {
        "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
        "j", "q", "x", "z", "c", "s", "r", "y", "w",
    };
    static const std::vector<std::string> finals = {
        "a", "ai", "an", "ang", "ao", "e", "ei", "en", "eng", "er", "i", "ia", "ian", "iang",
        "iao", "ie", "in", "ing", "iong", "iu", "o", "ong", "ou", "u", "ua", "uai", "uan",
        "uang", "ue", "ui", "un", "uo", "v", "van", "ve", "vn", "ng",
    };
    if (candidate.size() < 2 || !is_ascii_digit(candidate.back()) || candidate.back() < '1' || candidate.back() > '5') {
        return false;
    }
    std::string body = candidate.substr(0, candidate.size() - 1);
    body = uppercase_ascii_and_v(std::move(body));
    std::transform(body.begin(), body.end(), body.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    for (const auto & initial : initials) {
        if (body.rfind(initial, 0) == 0) {
            const std::string final = body.substr(initial.size());
            return std::find(finals.begin(), finals.end(), final) != finals.end();
        }
    }
    return std::find(finals.begin(), finals.end(), body) != finals.end();
}

std::vector<std::pair<std::string, std::string>> save_regex_matches(
    std::string & text,
    const std::regex & pattern,
    std::string_view placeholder_prefix) {
    std::vector<std::string> matches;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), pattern); it != std::sregex_iterator(); ++it) {
        matches.push_back(it->str());
    }
    std::sort(matches.begin(), matches.end());
    matches.erase(std::unique(matches.begin(), matches.end()), matches.end());
    std::sort(matches.begin(), matches.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.size() > rhs.size();
    });

    std::vector<std::pair<std::string, std::string>> saved;
    for (size_t i = 0; i < matches.size(); ++i) {
        // Letter suffix like the official TextNormalizer: a digit suffix would be
        // rewritten by the number normalizer and the placeholder would leak.
        const std::string placeholder =
            "<" + std::string(placeholder_prefix) + "_" + static_cast<char>('a' + i % 26) + ">";
        text = replace_all(std::move(text), matches[i], placeholder);
        saved.emplace_back(placeholder, matches[i]);
    }
    return saved;
}

void protect_tech_terms(std::string & text) {
    const std::regex pattern(R"([A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+)");
    std::vector<std::string> matches;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), pattern); it != std::sregex_iterator(); ++it) {
        matches.push_back(it->str());
    }
    std::sort(matches.begin(), matches.end());
    matches.erase(std::unique(matches.begin(), matches.end()), matches.end());
    std::sort(matches.begin(), matches.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.size() > rhs.size();
    });
    for (const auto & term : matches) {
        text = replace_all(std::move(text), term, replace_all(term, "-", "<H>"));
    }
}

std::vector<std::pair<std::string, std::string>> save_pinyin_tones(std::string & text) {
    std::vector<std::string> matches;
    const std::regex pattern(R"([A-Za-z\xC3\x9C\xC3\xBCvV]+[1-5])");
    for (auto it = std::sregex_iterator(text.begin(), text.end(), pattern); it != std::sregex_iterator(); ++it) {
        const size_t pos = static_cast<size_t>(it->position());
        if (pos > 0 && is_ascii_alpha(text[pos - 1])) {
            continue;
        }
        const std::string value = it->str();
        if (is_index_tts_pinyin_candidate(value)) {
            matches.push_back(value);
        }
    }
    std::sort(matches.begin(), matches.end());
    matches.erase(std::unique(matches.begin(), matches.end()), matches.end());

    std::vector<std::pair<std::string, std::string>> saved;
    for (size_t i = 0; i < matches.size(); ++i) {
        // Letter suffix like the official TextNormalizer: a digit suffix would be
        // rewritten by the number normalizer and the placeholder would leak.
        const std::string placeholder = "<pinyin_" + std::string(1, static_cast<char>('a' + i % 26)) + ">";
        text = replace_all(std::move(text), matches[i], placeholder);
        saved.emplace_back(placeholder, correct_index_tts_pinyin(matches[i]));
    }
    return saved;
}

void restore_saved(std::string & text, const std::vector<std::pair<std::string, std::string>> & saved) {
    for (const auto & [placeholder, value] : saved) {
        text = replace_all(std::move(text), placeholder, value);
    }
}

std::string normalize_chinese_numbers(const std::string & text) {
    // Measure words after which a standalone "2" reads as 两 (wetext zh
    // cardinal rule): 两个, 两位, 两只, ...
    static const char * const kMeasureWords[] = {
        "个", "位", "只", "条", "张", "本", "件", "次", "台", "辆", "块", "颗",
        "杯", "瓶", "碗", "套", "把", "支", "根", "座", "间", "群", "双", "对",
        "种", "类", "项", "层", "栋", "扇", "面", "盏", "袋", "箱", "包", "桶",
    };
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        // "$50" -> "五十美元" (wetext zh currency rule); consumes the "$".
        if (text[i] == '$' && i + 1 < text.size() && is_ascii_digit(text[i + 1])) {
            size_t begin = ++i;
            while (i < text.size() && is_ascii_digit(text[i])) {
                ++i;
            }
            out += chinese_cardinal_from_digits(text.substr(begin, i - begin));
            if (i < text.size() && text[i] == '.' && i + 1 < text.size() && is_ascii_digit(text[i + 1])) {
                size_t decimal_end = i + 1;
                while (decimal_end < text.size() && is_ascii_digit(text[decimal_end])) {
                    ++decimal_end;
                }
                out += "点";
                out += chinese_digits_individually(text.substr(i + 1, decimal_end - i - 1));
                i = decimal_end;
            }
            out += "美元";
            continue;
        }
        if (!is_ascii_digit(text[i])) {
            out.push_back(text[i++]);
            continue;
        }

        const size_t begin = i;
        while (i < text.size() && is_ascii_digit(text[i])) {
            ++i;
        }
        const std::string digits = text.substr(begin, i - begin);
        // "100km/h" -> "每小时一百公里" (wetext zh measure rule).
        if (starts_with_at(text, i, "km/h")) {
            out += "每小时" + chinese_cardinal_from_digits(digits) + "公里";
            i += 4;
            continue;
        }
        if (i < text.size() && text[i] == '.' && i + 1 < text.size() && is_ascii_digit(text[i + 1])) {
            size_t decimal_end = i + 1;
            while (decimal_end < text.size() && is_ascii_digit(text[decimal_end])) {
                ++decimal_end;
            }
            out += chinese_cardinal_from_digits(digits);
            out += "点";
            out += chinese_digits_individually(text.substr(i + 1, decimal_end - i - 1));
            i = decimal_end;
            continue;
        }
        if (digits.size() == 4 && starts_with_at(text, i, "年")) {
            out += chinese_digits_individually(digits);
        } else if (digits == "2" && std::any_of(std::begin(kMeasureWords), std::end(kMeasureWords),
                                                [&](const char * word) { return starts_with_at(text, i, word); })) {
            out += "两";
        } else {
            out += chinese_cardinal_from_digits(digits);
        }
    }
    return out;
}

std::string normalize_date_separators(std::string text) {
    const std::regex date_pattern(R"((\d{4})/(\d{1,2})/(\d{1,2}))");
    return std::regex_replace(text, date_pattern, "$1年$2月$3日");
}

std::string normalize_percentages(std::string text) {
    const std::regex percent_pattern(R"((\d+(?:\.\d+)?)%)");
    for (std::smatch match; std::regex_search(text, match, percent_pattern);) {
        text.replace(
            static_cast<size_t>(match.position()),
            static_cast<size_t>(match.length()),
            "百分之" + normalize_chinese_numbers(match[1].str()));
    }
    return text;
}

std::string normalize_fractions(std::string text) {
    const std::regex fraction_pattern(R"((\d+)/(\d+))");
    for (std::smatch match; std::regex_search(text, match, fraction_pattern);) {
        text.replace(
            static_cast<size_t>(match.position()),
            static_cast<size_t>(match.length()),
            chinese_cardinal_from_digits(match[2].str()) + "分之" + chinese_cardinal_from_digits(match[1].str()));
    }
    return text;
}

std::string normalize_time_zero_minutes(std::string text) {
    const std::regex time_pattern(R"((\d{1,2}):00)");
    return std::regex_replace(text, time_pattern, "$1点");
}

std::string normalize_telephone_runs(std::string text) {
    const std::regex phone_pattern(R"((\d{3})-(\d{4})-(\d{4}))");
    for (std::smatch match; std::regex_search(text, match, phone_pattern);) {
        const std::string digits = match[1].str() + match[2].str() + match[3].str();
        text.replace(static_cast<size_t>(match.position()), static_cast<size_t>(match.length()),
            chinese_digits_individually(digits));
    }
    return text;
}

std::string apply_index_tts_punctuation_map(std::string text) {
    const std::vector<std::pair<std::string, std::string>> replacements = {
        {"$", "."}, {"：", ","}, {"；", ","}, {";", ","}, {"，", ","}, {"。", "."},
        {"！", "!"}, {"？", "?"}, {"\n", " "}, {"·", "-"}, {"、", ","}, {"...", "…"},
        {",,,", "…"}, {"，，，", "…"}, {"……", "…"}, {"“", "'"}, {"”", "'"}, {"\"", "'"},
        {"‘", "'"}, {"’", "'"}, {"（", "'"}, {"）", "'"}, {"(", "'"}, {")", "'"},
        {"《", "'"}, {"》", "'"}, {"【", "'"}, {"】", "'"}, {"[", "'"}, {"]", "'"},
        {"—", "-"}, {"～", "-"}, {"~", "-"}, {"「", "'"}, {"」", "'"}, {":", ","},
    };
    for (const auto & [from, to] : replacements) {
        text = replace_all(std::move(text), from, to);
    }
    return text;
}

std::string normalize_index_tts_chinese_text(std::string_view text) {
    std::string out = std::regex_replace(
        std::string(text),
        std::regex(R"((what|where|who|which|how|t?here|it|s?he|that|this)'s)", std::regex_constants::icase),
        "$1 is");

    const std::regex name_pattern("([\xE4-\xE9][\x80-\xBF][\x80-\xBF]+(?:[-·—][\xE4-\xE9][\x80-\xBF][\x80-\xBF]+){1,2})");
    protect_tech_terms(out);
    const auto pinyin_tones = save_pinyin_tones(out);
    const auto names = save_regex_matches(out, name_pattern, "name");

    out = normalize_telephone_runs(std::move(out));
    out = normalize_date_separators(std::move(out));
    out = normalize_time_zero_minutes(std::move(out));
    out = normalize_percentages(std::move(out));
    out = normalize_fractions(std::move(out));
    out = normalize_chinese_numbers(out);

    restore_saved(out, names);
    restore_saved(out, pinyin_tones);
    out = std::regex_replace(out, std::regex(R"(\s*<H>\s*)"), "-");
    return apply_index_tts_punctuation_map(std::move(out));
}

struct CodepointSpan {
    size_t start = 0;
    size_t end = 0;
    std::string_view text;
};

std::vector<CodepointSpan> split_codepoints(std::string_view text, std::string_view label) {
    std::vector<CodepointSpan> spans;
    spans.reserve(engine::text::utf8_codepoint_count(text, label));
    for (size_t pos = 0; pos < text.size();) {
        const auto ch = static_cast<unsigned char>(text[pos]);
        size_t width = 0;
        if (ch <= 0x7FU) {
            width = 1;
        } else if ((ch & 0xE0U) == 0xC0U) {
            width = 2;
        } else if ((ch & 0xF0U) == 0xE0U) {
            width = 3;
        } else if ((ch & 0xF8U) == 0xF0U) {
            width = 4;
        } else {
            throw std::runtime_error(std::string(label) + " contains invalid UTF-8");
        }
        spans.push_back({pos, pos + width, text.substr(pos, width)});
        pos += width;
    }
    return spans;
}

std::string substring_codepoints(const std::string & text, const std::vector<CodepointSpan> & spans, size_t start, size_t end) {
    if (start >= end) {
        return {};
    }
    return text.substr(spans[start].start, spans[end - 1].end - spans[start].start);
}

bool is_ascii_non_space(std::string_view ch) noexcept {
    return ch.size() == 1 && static_cast<unsigned char>(ch.front()) <= 0x7FU && ch.front() != ' ';
}

std::string remove_blank_between_chinese(const std::string & text) {
    const auto spans = split_codepoints(text, "Confucius4-TTS text");
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < spans.size(); ++i) {
        if (spans[i].text == " " && i > 0 && i + 1 < spans.size()) {
            if (is_ascii_non_space(spans[i - 1].text) && is_ascii_non_space(spans[i + 1].text)) {
                out.push_back(' ');
            }
            continue;
        }
        out.append(spans[i].text);
    }
    return out;
}

std::string normalize_confucius4_chinese_text(std::string_view text) {
    std::string out = remove_blank_between_chinese(std::string(text));
    out = replace_all(std::move(out), "²", "平方");
    out = replace_all(std::move(out), "³", "立方");
    out = replace_all(std::move(out), ".", "。");
    out = replace_all(std::move(out), " - ", "，");
    out = replace_all(std::move(out), "（", "");
    out = replace_all(std::move(out), "）", "");
    out = replace_all(std::move(out), "【", "");
    out = replace_all(std::move(out), "】", "");
    out = replace_all(std::move(out), "`", "");
    out = replace_all(std::move(out), "——", " ");

    const auto spans = split_codepoints(out, "Confucius4-TTS Chinese text");
    size_t trim_pos = spans.size();
    while (trim_pos > 0 &&
           (spans[trim_pos - 1].text == "，" || spans[trim_pos - 1].text == "," || spans[trim_pos - 1].text == "、")) {
        --trim_pos;
    }
    if (trim_pos != spans.size()) {
        return substring_codepoints(out, spans, 0, trim_pos) + "。";
    }
    return out;
}

}  // namespace

ChineseTextNormalizer::ChineseTextNormalizer(ChineseTextNormalizationTarget target)
    : target_(target) {}

std::string ChineseTextNormalizer::normalize(std::string_view text) const {
    switch (target_) {
    case ChineseTextNormalizationTarget::IndexTTS:
        return normalize_index_tts_chinese_text(text);
    case ChineseTextNormalizationTarget::Confucius4TTS:
        return normalize_confucius4_chinese_text(text);
    }
    throw std::runtime_error("unsupported Chinese text normalization target");
}

std::string normalize_chinese_text(std::string_view text, ChineseTextNormalizationTarget target) {
    return ChineseTextNormalizer(target).normalize(text);
}

}  // namespace engine::text
