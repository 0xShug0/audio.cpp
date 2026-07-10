#include "engine/models/supertonic/tokenizer_text.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace engine::models::supertonic {
namespace {

const std::unordered_set<std::string> kLanguages = {
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu",
    "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na",
};

std::shared_ptr<const SupertonicAssets> require_assets(std::shared_ptr<const SupertonicAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Supertonic tokenizer requires assets");
    }
    return assets;
}

void replace_all(std::string & text, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return;
    }
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::string trim_ascii_space(const std::string & text) {
    size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return text.substr(begin, end - begin);
}

bool ends_with_reference_punctuation(const std::string & text) {
    if (text.empty()) {
        return false;
    }
    const char last = text.back();
    switch (last) {
        case '.':
        case '!':
        case '?':
        case ';':
        case ':':
        case ',':
        case '\'':
        case '"':
        case ')':
        case ']':
        case '}':
            return true;
        default:
            break;
    }
    static constexpr const char * suffixes[] = {"…", "。", "」", "』", "】", "〉", "》", "›", "»"};
    for (const char * suffix : suffixes) {
        const std::string marker(suffix);
        if (text.size() >= marker.size() && text.compare(text.size() - marker.size(), marker.size(), marker) == 0) {
            return true;
        }
    }
    return false;
}

void append_hangul_decomposition(uint32_t codepoint, std::vector<uint32_t> & out) {
    constexpr uint32_t kSBase = 0xAC00;
    constexpr uint32_t kLBase = 0x1100;
    constexpr uint32_t kVBase = 0x1161;
    constexpr uint32_t kTBase = 0x11A7;
    constexpr uint32_t kLCount = 19;
    constexpr uint32_t kVCount = 21;
    constexpr uint32_t kTCount = 28;
    constexpr uint32_t kNCount = kVCount * kTCount;
    constexpr uint32_t kSCount = kLCount * kNCount;
    if (codepoint < kSBase || codepoint >= kSBase + kSCount) {
        out.push_back(codepoint);
        return;
    }
    const uint32_t s_index = codepoint - kSBase;
    const uint32_t l_index = s_index / kNCount;
    const uint32_t v_index = (s_index % kNCount) / kTCount;
    const uint32_t t_index = s_index % kTCount;
    out.push_back(kLBase + l_index);
    out.push_back(kVBase + v_index);
    if (t_index > 0) {
        out.push_back(kTBase + t_index);
    }
}

bool append_japanese_kana_decomposition(uint32_t codepoint, std::vector<uint32_t> & out) {
    constexpr uint32_t kDakuten = 0x3099;
    constexpr uint32_t kHandakuten = 0x309A;
    static constexpr std::pair<uint32_t, uint32_t> kDakutenKana[] = {
        {0x304C, 0x304B}, {0x304E, 0x304D}, {0x3050, 0x304F}, {0x3052, 0x3051}, {0x3054, 0x3053},
        {0x3056, 0x3055}, {0x3058, 0x3057}, {0x305A, 0x3059}, {0x305C, 0x305B}, {0x305E, 0x305D},
        {0x3060, 0x305F}, {0x3062, 0x3061}, {0x3065, 0x3064}, {0x3067, 0x3066}, {0x3069, 0x3068},
        {0x3070, 0x306F}, {0x3073, 0x3072}, {0x3076, 0x3075}, {0x3079, 0x3078}, {0x307C, 0x307B},
        {0x3094, 0x3046},
        {0x30AC, 0x30AB}, {0x30AE, 0x30AD}, {0x30B0, 0x30AF}, {0x30B2, 0x30B1}, {0x30B4, 0x30B3},
        {0x30B6, 0x30B5}, {0x30B8, 0x30B7}, {0x30BA, 0x30B9}, {0x30BC, 0x30BB}, {0x30BE, 0x30BD},
        {0x30C0, 0x30BF}, {0x30C2, 0x30C1}, {0x30C5, 0x30C4}, {0x30C7, 0x30C6}, {0x30C9, 0x30C8},
        {0x30D0, 0x30CF}, {0x30D3, 0x30D2}, {0x30D6, 0x30D5}, {0x30D9, 0x30D8}, {0x30DC, 0x30DB},
        {0x30F4, 0x30A6}, {0x30F7, 0x30EF}, {0x30F8, 0x30F0}, {0x30F9, 0x30F1}, {0x30FA, 0x30F2},
    };
    static constexpr std::pair<uint32_t, uint32_t> kHandakutenKana[] = {
        {0x3071, 0x306F}, {0x3074, 0x3072}, {0x3077, 0x3075}, {0x307A, 0x3078}, {0x307D, 0x307B},
        {0x30D1, 0x30CF}, {0x30D4, 0x30D2}, {0x30D7, 0x30D5}, {0x30DA, 0x30D8}, {0x30DD, 0x30DB},
    };

    for (const auto & [composed, base] : kDakutenKana) {
        if (codepoint == composed) {
            out.push_back(base);
            out.push_back(kDakuten);
            return true;
        }
    }
    for (const auto & [composed, base] : kHandakutenKana) {
        if (codepoint == composed) {
            out.push_back(base);
            out.push_back(kHandakuten);
            return true;
        }
    }
    return false;
}

std::vector<uint32_t> decompose_known_text_codepoints(const std::vector<uint32_t> & codepoints) {
    std::vector<uint32_t> out;
    out.reserve(codepoints.size() * 2);
    for (const uint32_t codepoint : codepoints) {
        if (append_japanese_kana_decomposition(codepoint, out)) {
            continue;
        }
        append_hangul_decomposition(codepoint, out);
    }
    return out;
}

}  // namespace

SupertonicTextTokenizer::SupertonicTextTokenizer(std::shared_ptr<const SupertonicAssets> assets)
    : assets_(require_assets(std::move(assets))) {}

SupertonicTextInputs SupertonicTextTokenizer::encode(const std::string & text, const std::string & language) const {
    const auto processed = preprocess(text, language);
    const auto codepoints = decompose_known_text_codepoints(utf8_to_codepoints(processed));
    SupertonicTextInputs out;
    out.length = static_cast<int64_t>(codepoints.size());
    out.ids.reserve(codepoints.size());
    out.mask.assign(codepoints.size(), 1.0F);
    for (const uint32_t codepoint : codepoints) {
        const auto it = assets_->unicode_indexer.find(codepoint);
        if (it == assets_->unicode_indexer.end()) {
            throw std::runtime_error("Supertonic unicode indexer has no entry for codepoint " + std::to_string(codepoint));
        }
        out.ids.push_back(it->second);
    }
    return out;
}

std::string SupertonicTextTokenizer::preprocess(const std::string & text, const std::string & language) const {
    if (kLanguages.find(language) == kLanguages.end()) {
        throw std::runtime_error("invalid Supertonic language: " + language);
    }
    std::string result = text;
    const std::pair<const char *, const char *> replacements[] = {
        {"–", "-"}, {"‑", "-"}, {"—", "-"}, {"_", " "}, {"“", "\""}, {"”", "\""}, {"‘", "'"}, {"’", "'"},
        {"´", "'"}, {"`", "'"}, {"[", " "}, {"]", " "}, {"|", " "}, {"/", " "}, {"#", " "}, {"→", " "},
        {"←", " "}, {"@", " at "}, {"e.g.,", "for example, "}, {"i.e.,", "that is, "},
    };
    for (const auto & replacement : replacements) {
        replace_all(result, replacement.first, replacement.second);
    }
    const char * remove_symbols[] = {"♥", "☆", "♡", "©", "\\"};
    for (const char * symbol : remove_symbols) {
        replace_all(result, symbol, "");
    }
    result = std::regex_replace(result, std::regex(" ,"), ",");
    result = std::regex_replace(result, std::regex(" \\."), ".");
    result = std::regex_replace(result, std::regex(" !"), "!");
    result = std::regex_replace(result, std::regex(" \\?"), "?");
    result = std::regex_replace(result, std::regex(" ;"), ";");
    result = std::regex_replace(result, std::regex(" :"), ":");
    result = std::regex_replace(result, std::regex(" '"), "'");
    while (result.find("\"\"") != std::string::npos) {
        replace_all(result, "\"\"", "\"");
    }
    while (result.find("''") != std::string::npos) {
        replace_all(result, "''", "'");
    }
    result = std::regex_replace(result, std::regex("\\s+"), " ");
    result = trim_ascii_space(result);
    if (!ends_with_reference_punctuation(result)) {
        result += ".";
    }
    return "<" + language + ">" + result + "</" + language + ">";
}

std::vector<uint32_t> SupertonicTextTokenizer::utf8_to_codepoints(const std::string & text) const {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < text.size();) {
        const auto c = static_cast<unsigned char>(text[i]);
        uint32_t codepoint = 0;
        size_t width = 0;
        if ((c & 0x80U) == 0U) {
            codepoint = c;
            width = 1;
        } else if ((c & 0xE0U) == 0xC0U && i + 1 < text.size()) {
            codepoint = static_cast<uint32_t>(c & 0x1FU) << 6U;
            codepoint |= static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3FU);
            width = 2;
        } else if ((c & 0xF0U) == 0xE0U && i + 2 < text.size()) {
            codepoint = static_cast<uint32_t>(c & 0x0FU) << 12U;
            codepoint |= static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3FU) << 6U;
            codepoint |= static_cast<uint32_t>(static_cast<unsigned char>(text[i + 2]) & 0x3FU);
            width = 3;
        } else if ((c & 0xF8U) == 0xF0U && i + 3 < text.size()) {
            codepoint = static_cast<uint32_t>(c & 0x07U) << 18U;
            codepoint |= static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3FU) << 12U;
            codepoint |= static_cast<uint32_t>(static_cast<unsigned char>(text[i + 2]) & 0x3FU) << 6U;
            codepoint |= static_cast<uint32_t>(static_cast<unsigned char>(text[i + 3]) & 0x3FU);
            width = 4;
        } else {
            throw std::runtime_error("Supertonic input contains invalid UTF-8");
        }
        out.push_back(codepoint);
        i += width;
    }
    return out;
}

}  // namespace engine::models::supertonic
