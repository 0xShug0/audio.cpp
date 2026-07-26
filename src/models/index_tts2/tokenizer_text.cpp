#include "engine/models/index_tts2/tokenizer_text.h"

#include "engine/framework/text/chinese_normalization.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace engine::models::index_tts2 {
namespace {

std::string replace_all(std::string text, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return text;
    }
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
    return text;
}

bool contains_token(
    const std::vector<std::string> & values,
    const std::vector<std::string> & needles) {
    for (const auto & value : values) {
        if (std::find(needles.begin(), needles.end(), value) != needles.end()) {
            return true;
        }
    }
    return false;
}

size_t utf8_codepoint_size(unsigned char byte) {
    if ((byte & 0x80U) == 0U) {
        return 1;
    }
    if ((byte & 0xE0U) == 0xC0U) {
        return 2;
    }
    if ((byte & 0xF0U) == 0xE0U) {
        return 3;
    }
    if ((byte & 0xF8U) == 0xF0U) {
        return 4;
    }
    return 1;
}

uint32_t decode_utf8_codepoint(const std::string & text, size_t offset, size_t size) {
    const auto byte = [&](size_t i) { return static_cast<unsigned char>(text[offset + i]); };
    if (size == 1) {
        return byte(0);
    }
    if (size == 2 && offset + 1 < text.size()) {
        return ((byte(0) & 0x1FU) << 6U) | (byte(1) & 0x3FU);
    }
    if (size == 3 && offset + 2 < text.size()) {
        return ((byte(0) & 0x0FU) << 12U) | ((byte(1) & 0x3FU) << 6U) | (byte(2) & 0x3FU);
    }
    if (size == 4 && offset + 3 < text.size()) {
        return ((byte(0) & 0x07U) << 18U) | ((byte(1) & 0x3FU) << 12U) | ((byte(2) & 0x3FU) << 6U) | (byte(3) & 0x3FU);
    }
    return byte(0);
}

bool is_han_codepoint(uint32_t cp) {
    return (cp >= 0x4E00U && cp <= 0x9FFFU);
}

bool is_cjk_codepoint(uint32_t cp) {
    return (cp >= 0x1100U && cp <= 0x11FFU)
        || (cp >= 0x2E80U && cp <= 0xA4CFU)
        || (cp >= 0xA840U && cp <= 0xD7AFU)
        || (cp >= 0xF900U && cp <= 0xFAFFU)
        || (cp >= 0xFE30U && cp <= 0xFE4FU)
        || (cp >= 0xFF65U && cp <= 0xFFDCU)
        || (cp >= 0x20000U && cp <= 0x2FFFFU);
}

bool contains_han(const std::string & text) {
    for (size_t i = 0; i < text.size();) {
        const size_t size = std::min(utf8_codepoint_size(static_cast<unsigned char>(text[i])), text.size() - i);
        if (is_han_codepoint(decode_utf8_codepoint(text, i, size))) {
            return true;
        }
        i += size;
    }
    return false;
}

std::string trim_ascii(const std::string & text) {
    size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
        --end;
    }
    return text.substr(begin, end - begin);
}

std::string uppercase_ascii(std::string text) {
    for (char & ch : text) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return text;
}

std::string tokenize_by_cjk_char(const std::string & text) {
    std::vector<std::string> tokens;
    std::string pending;
    auto flush_pending = [&]() {
        std::string trimmed = trim_ascii(pending);
        if (!trimmed.empty()) {
            tokens.push_back(uppercase_ascii(std::move(trimmed)));
        }
        pending.clear();
    };

    for (size_t i = 0; i < text.size();) {
        const size_t size = std::min(utf8_codepoint_size(static_cast<unsigned char>(text[i])), text.size() - i);
        const uint32_t cp = decode_utf8_codepoint(text, i, size);
        if (is_cjk_codepoint(cp)) {
            flush_pending();
            tokens.push_back(text.substr(i, size));
        } else {
            pending.append(text, i, size);
        }
        i += size;
    }
    flush_pending();

    std::string out;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i != 0) {
            out.push_back(' ');
        }
        out += tokens[i];
    }
    return out;
}

std::string english_cardinal_under_1000(int value) {
    static const std::string units[] = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen",
    };
    static const std::string tens[] = {
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    };

    if (value < 20) {
        return units[static_cast<size_t>(value)];
    }
    if (value < 100) {
        const int ten = value / 10;
        const int one = value % 10;
        return one == 0 ? tens[static_cast<size_t>(ten)]
                        : tens[static_cast<size_t>(ten)] + " " + units[static_cast<size_t>(one)];
    }
    const int hundred = value / 100;
    const int rest = value % 100;
    return rest == 0 ? units[static_cast<size_t>(hundred)] + " hundred"
                     : units[static_cast<size_t>(hundred)] + " hundred " + english_cardinal_under_1000(rest);
}

std::string english_cardinal(int64_t value) {
    if (value < 0) {
        return "minus " + english_cardinal(-value);
    }
    if (value < 1000) {
        return english_cardinal_under_1000(static_cast<int>(value));
    }
    if (value < 1000000) {
        const int64_t thousands = value / 1000;
        const int64_t rest = value % 1000;
        return rest == 0 ? english_cardinal(thousands) + " thousand"
                         : english_cardinal(thousands) + " thousand " + english_cardinal(rest);
    }
    return std::to_string(value);
}

std::string english_year(int value) {
    if (value >= 1900 && value <= 1999) {
        const int rest = value - 1900;
        return rest == 0 ? "nineteen hundred" : "nineteen " + english_cardinal_under_1000(rest);
    }
    if (value >= 2000 && value <= 2099) {
        const int rest = value - 2000;
        if (rest == 0) {
            return "two thousand";
        }
        if (rest < 10) {
            return "two thousand " + english_cardinal_under_1000(rest);
        }
        return "twenty " + english_cardinal_under_1000(rest);
    }
    return english_cardinal(value);
}

std::string english_ordinal(int64_t value) {
    static const std::string ordinals[] = {
        "zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
        "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth",
        "sixteenth", "seventeenth", "eighteenth", "nineteenth",
    };
    static const std::string tens_ordinals[] = {
        "", "", "twentieth", "thirtieth", "fortieth", "fiftieth", "sixtieth",
        "seventieth", "eightieth", "ninetieth",
    };
    if (value < 20) {
        return ordinals[static_cast<size_t>(value)];
    }
    if (value < 100) {
        const int64_t ten = value / 10;
        const int64_t one = value % 10;
        return one == 0 ? tens_ordinals[static_cast<size_t>(ten)]
                        : english_cardinal_under_1000(static_cast<int>(ten * 10)) + " " + english_ordinal(one);
    }
    if (value % 100 == 0) {
        return english_cardinal(value / 100) + " hundredth";
    }
    return english_cardinal(value - (value % 100)) + " " + english_ordinal(value % 100);
}

std::string english_fraction(int64_t numerator, int64_t denominator) {
    if (denominator == 2) {
        return numerator == 1 ? "one half" : english_cardinal(numerator) + " halves";
    }
    std::string out = english_cardinal(numerator) + " " + english_ordinal(denominator);
    if (numerator != 1) {
        out.push_back('s');
    }
    return out;
}

std::string english_digits_individually(std::string_view digits) {
    std::string out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i != 0) {
            out.push_back(' ');
        }
        out += english_cardinal_under_1000(digits[i] - '0');
    }
    return out;
}

std::string normalize_english_regex(
    std::string text,
    const std::regex & pattern,
    const std::function<std::string(const std::smatch &)> & replacement) {
    std::string out;
    auto begin = text.cbegin();
    auto end = text.cend();
    std::smatch match;
    while (std::regex_search(begin, end, match, pattern)) {
        out.append(begin, match[0].first);
        out += replacement(match);
        begin = match[0].second;
    }
    out.append(begin, end);
    return out;
}

std::string normalize_english_dates(std::string text) {
    const std::regex pattern(
        R"(\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?)",
        std::regex_constants::icase);
    return normalize_english_regex(std::move(text), pattern, [](const std::smatch & match) {
        std::string out = match[1].str() + " " + english_ordinal(std::stoll(match[2].str()));
        if (match[3].matched) {
            out += " " + english_year(static_cast<int>(std::stoll(match[3].str())));
        }
        return out;
    });
}

std::string normalize_english_numbers(std::string text) {
    text = normalize_english_dates(std::move(text));
    text = normalize_english_regex(
        std::move(text),
        std::regex(R"(\b(\d+(?:\.\d+)?)%)"),
        [](const std::smatch & match) {
            return normalize_english_regex(
                match[1].str(),
                std::regex(R"((\d+)\.(\d+))"),
                [](const std::smatch & decimal) {
                    return english_cardinal(std::stoll(decimal[1].str())) + " point " +
                        english_digits_individually(decimal[2].str());
                }) + " percent";
        });
    text = normalize_english_regex(
        std::move(text),
        std::regex(R"(\b(\d+)/(\d+)\b)"),
        [](const std::smatch & match) {
            return english_fraction(std::stoll(match[1].str()), std::stoll(match[2].str()));
        });
    text = normalize_english_regex(
        std::move(text),
        std::regex(R"(\b(\d+)\.(\d+)\b)"),
        [](const std::smatch & match) {
            return english_cardinal(std::stoll(match[1].str())) + " point " +
                english_digits_individually(match[2].str());
        });
    text = normalize_english_regex(
        std::move(text),
        std::regex(R"(\b(\d+)(st|nd|rd|th)\b)", std::regex_constants::icase),
        [](const std::smatch & match) {
            return english_ordinal(std::stoll(match[1].str()));
        });
    text = normalize_english_regex(
        std::move(text),
        std::regex(R"(\b(\d+)\b)"),
        [](const std::smatch & match) {
            return english_cardinal(std::stoll(match[1].str()));
        });
    return text;
}

std::vector<std::vector<std::string>> split_segments_by_token(
    const std::vector<std::string> & tokenized,
    const std::vector<std::string> & split_tokens,
    int max_text_tokens_per_segment) {
    if (tokenized.empty()) {
        return {};
    }
    std::vector<std::vector<std::string>> segments;
    std::vector<std::string> current_segment;
    int current_segment_tokens_len = 0;
    for (size_t i = 0; i < tokenized.size(); ++i) {
        const auto & token = tokenized[i];
        current_segment.push_back(token);
        ++current_segment_tokens_len;
        std::vector<std::vector<std::string>> sub_segments;
        bool should_flush_sub_segments = false;
        if (!contains_token(split_tokens, {",", "▁,"}) && contains_token(current_segment, {",", "▁,"})) {
            sub_segments = split_segments_by_token(current_segment, {",", "▁,"}, max_text_tokens_per_segment);
            should_flush_sub_segments = true;
        } else if (!contains_token(split_tokens, {"-"}) && contains_token(current_segment, {"-"})) {
            sub_segments = split_segments_by_token(current_segment, {"-"}, max_text_tokens_per_segment);
            should_flush_sub_segments = true;
        } else if (current_segment_tokens_len <= max_text_tokens_per_segment) {
            if (std::find(split_tokens.begin(), split_tokens.end(), token) != split_tokens.end() && current_segment_tokens_len > 2) {
                if (i + 1 < tokenized.size() && (tokenized[i + 1] == "'" || tokenized[i + 1] == "▁'")) {
                    current_segment.push_back(tokenized[i + 1]);
                }
                segments.push_back(current_segment);
                current_segment.clear();
                current_segment_tokens_len = 0;
            }
            continue;
        } else {
            for (size_t j = 0; j < current_segment.size(); j += static_cast<size_t>(max_text_tokens_per_segment)) {
                const size_t end = std::min(current_segment.size(), j + static_cast<size_t>(max_text_tokens_per_segment));
                sub_segments.emplace_back(current_segment.begin() + static_cast<std::ptrdiff_t>(j), current_segment.begin() + static_cast<std::ptrdiff_t>(end));
            }
            should_flush_sub_segments = true;
        }
        if (should_flush_sub_segments) {
            segments.insert(segments.end(), sub_segments.begin(), sub_segments.end());
            current_segment.clear();
            current_segment_tokens_len = 0;
        }
    }
    if (current_segment_tokens_len > 0) {
        if (current_segment_tokens_len > max_text_tokens_per_segment) {
            throw std::runtime_error("IndexTTS2 text segment exceeds max_text_tokens_per_segment");
        }
        segments.push_back(current_segment);
    }

    std::vector<std::vector<std::string>> merged_segments;
    int total_token = 0;
    for (const auto & segment : segments) {
        total_token += static_cast<int>(segment.size());
        if (segment.empty()) {
            continue;
        }
        if (merged_segments.empty()) {
            merged_segments.push_back(segment);
        } else if (static_cast<int>(merged_segments.back().size() + segment.size()) <= max_text_tokens_per_segment && total_token > 0) {
            merged_segments.back().insert(merged_segments.back().end(), segment.begin(), segment.end());
        } else if (static_cast<int>(merged_segments.back().size() + segment.size()) <= max_text_tokens_per_segment / 2) {
            merged_segments.back().insert(merged_segments.back().end(), segment.begin(), segment.end());
        } else {
            merged_segments.push_back(segment);
        }
    }
    return merged_segments;
}

}  // namespace

IndexTTS2TextTokenizer::IndexTTS2TextTokenizer(std::shared_ptr<const IndexTTS2Assets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("IndexTTS2 text tokenizer requires assets");
    }
    pieces_ = engine::tokenizers::load_sentencepiece_model(assets_->resources.require_file("bpe"));
    piece_to_id_.reserve(pieces_.size());
    for (const auto & piece : pieces_) {
        piece_to_id_.emplace(piece.text, static_cast<int32_t>(piece.id));
    }
}

std::string IndexTTS2TextTokenizer::normalize_english(const std::string & text) const {
    std::string out = std::regex_replace(
        text,
        std::regex(R"((what|where|who|which|how|t?here|it|s?he|that|this)'s)", std::regex_constants::icase),
        "$1 is");
    out = normalize_english_numbers(std::move(out));
    const std::vector<std::pair<std::string, std::string>> replacements = {
        {"：", ","},
        {"；", ","},
        {";", ","},
        {"，", ","},
        {"。", "."},
        {"！", "!"},
        {"？", "?"},
        {"\n", " "},
        {"·", "-"},
        {"、", ","},
        {"...", "…"},
        {",,,", "…"},
        {"，，，", "…"},
        {"……", "…"},
        {"“", "'"},
        {"”", "'"},
        {"\"", "'"},
        {"‘", "'"},
        {"’", "'"},
        {"（", "'"},
        {"）", "'"},
        {"(", "'"},
        {")", "'"},
        {"《", "'"},
        {"》", "'"},
        {"【", "'"},
        {"】", "'"},
        {"[", "'"},
        {"]", "'"},
        {"—", "-"},
        {"～", "-"},
        {"~", "-"},
        {"「", "'"},
        {"」", "'"},
        {":", ","},
    };
    for (const auto & [from, to] : replacements) {
        out = replace_all(std::move(out), from, to);
    }
    for (char & ch : out) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return out;
}

std::string IndexTTS2TextTokenizer::normalize_chinese(const std::string & text) const {
    return engine::text::normalize_chinese_text(
        text,
        engine::text::ChineseTextNormalizationTarget::IndexTTS);
}

std::string IndexTTS2TextTokenizer::normalize_text(const std::string & text) const {
    return contains_han(text) ? tokenize_by_cjk_char(normalize_chinese(text)) : normalize_english(text);
}

std::vector<int32_t> IndexTTS2TextTokenizer::encode(const std::string & text) const {
    return engine::tokenizers::tokenize_sentencepiece(pieces_, normalize_text(text));
}

std::vector<std::string> IndexTTS2TextTokenizer::tokenize_to_pieces(const std::string & text) const {
    const auto ids = encode(text);
    std::vector<std::string> out;
    out.reserve(ids.size());
    for (const int32_t id : ids) {
        out.push_back(id_to_piece(id));
    }
    return out;
}

IndexTTS2TextEncoding IndexTTS2TextTokenizer::encode_for_inference(
    const std::string & text,
    int max_text_tokens_per_segment) const {
    if (max_text_tokens_per_segment <= 0) {
        throw std::runtime_error("IndexTTS2 max_text_tokens_per_segment must be positive");
    }
    IndexTTS2TextEncoding encoding;
    encoding.normalized_text = normalize_text(text);
    encoding.token_ids = engine::tokenizers::tokenize_sentencepiece(pieces_, encoding.normalized_text);
    encoding.pieces.reserve(encoding.token_ids.size());
    for (const int32_t id : encoding.token_ids) {
        encoding.pieces.push_back(id_to_piece(id));
    }
    encoding.segments = split_segments(encoding.pieces, max_text_tokens_per_segment);
    encoding.segment_token_ids.reserve(encoding.segments.size());
    for (const auto & segment : encoding.segments) {
        std::vector<int32_t> ids;
        ids.reserve(segment.size());
        for (const auto & piece : segment) {
            ids.push_back(piece_to_id(piece));
        }
        encoding.segment_token_ids.push_back(std::move(ids));
    }
    return encoding;
}

int32_t IndexTTS2TextTokenizer::piece_to_id(const std::string & piece) const {
    const auto it = piece_to_id_.find(piece);
    if (it != piece_to_id_.end()) {
        return it->second;
    }
    throw std::runtime_error("IndexTTS2 tokenizer missing SentencePiece piece: " + piece);
}

std::string IndexTTS2TextTokenizer::id_to_piece(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= pieces_.size()) {
        throw std::runtime_error("IndexTTS2 tokenizer id out of range");
    }
    return pieces_[static_cast<size_t>(id)].text;
}

std::vector<std::vector<std::string>> IndexTTS2TextTokenizer::split_segments(
    const std::vector<std::string> & pieces,
    int max_text_tokens_per_segment) const {
    return split_segments_by_token(pieces, {".", "!", "?", "▁.", "▁?", "▁..."}, max_text_tokens_per_segment);
}

}  // namespace engine::models::index_tts2
