#include "engine/community_models/sopro_tts/text_tokenizer.h"

#include "sentencepiece_processor.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace engine::community_models::sopro_tts {
namespace {

bool is_ascii_space(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

// Python str.split() on whitespace followed by " ".join(...).
std::string collapse_whitespace(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    size_t index = 0;
    bool first = true;
    while (index < text.size()) {
        while (index < text.size() && is_ascii_space(text[index])) {
            ++index;
        }
        const size_t start = index;
        while (index < text.size() && !is_ascii_space(text[index])) {
            ++index;
        }
        if (index > start) {
            if (!first) {
                out.push_back(' ');
            }
            out.append(text, start, index - start);
            first = false;
        }
    }
    return out;
}

std::string trim(const std::string & text) {
    size_t begin = 0;
    size_t end = text.size();
    while (begin < end && is_ascii_space(text[begin])) {
        ++begin;
    }
    while (end > begin && is_ascii_space(text[end - 1])) {
        --end;
    }
    return text.substr(begin, end - begin);
}

size_t utf8_sequence_length(unsigned char lead) noexcept {
    if (lead < 0x80U) {
        return 1;
    }
    if ((lead & 0xE0U) == 0xC0U) {
        return 2;
    }
    if ((lead & 0xF0U) == 0xE0U) {
        return 3;
    }
    if ((lead & 0xF8U) == 0xF0U) {
        return 4;
    }
    return 1;  // lone continuation byte: treat as one unit so we never stall
}

size_t codepoint_length(const std::string & text) noexcept {
    size_t count = 0;
    size_t index = 0;
    while (index < text.size()) {
        index += std::min(utf8_sequence_length(static_cast<unsigned char>(text[index])),
                          text.size() - index);
        ++count;
    }
    return count;
}

uint32_t decode_utf8(const std::string & text, size_t offset, size_t & length) noexcept {
    const auto lead = static_cast<unsigned char>(text[offset]);
    length = std::min(utf8_sequence_length(lead), text.size() - offset);
    if (length == 1) {
        return lead;
    }
    static constexpr std::array<uint32_t, 5> kLeadMask = {0, 0x7FU, 0x1FU, 0x0FU, 0x07U};
    uint32_t value = lead & kLeadMask[length];
    for (size_t i = 1; i < length; ++i) {
        value = (value << 6U) | (static_cast<unsigned char>(text[offset + i]) & 0x3FU);
    }
    return value;
}

void append_utf8(std::string & out, uint32_t codepoint) {
    if (codepoint < 0x80U) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint < 0x800U) {
        out.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint < 0x10000U) {
        out.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
        out.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
}

// str.islower()/str.upper() for the Latin ranges the four supported languages
// use (ASCII, Latin-1 supplement, Latin Extended-A). Anything else is left as
// it is, which matches Python for scripts without case.
bool codepoint_is_lower(uint32_t cp) noexcept {
    if (cp >= 'a' && cp <= 'z') {
        return true;
    }
    if (cp == 0xDFU) {  // sharp s has no single-codepoint uppercase
        return false;
    }
    if (cp >= 0xE0U && cp <= 0xFEU && cp != 0xF7U) {
        return true;
    }
    if (cp >= 0x100U && cp <= 0x17FU) {
        return (cp % 2U) == 1U;  // Latin Extended-A alternates upper/lower
    }
    return false;
}

uint32_t codepoint_to_upper(uint32_t cp) noexcept {
    if (cp >= 'a' && cp <= 'z') {
        return cp - 32U;
    }
    if (cp >= 0xE0U && cp <= 0xFEU && cp != 0xF7U) {
        return cp - 32U;
    }
    if (cp >= 0x100U && cp <= 0x17FU && (cp % 2U) == 1U) {
        return cp - 1U;
    }
    return cp;
}

std::string capitalize_first(const std::string & text) {
    if (text.empty()) {
        return text;
    }
    size_t length = 0;
    const uint32_t cp = decode_utf8(text, 0, length);
    if (!codepoint_is_lower(cp)) {
        return text;
    }
    std::string out;
    out.reserve(text.size() + 1);
    append_utf8(out, codepoint_to_upper(cp));
    out.append(text, length, std::string::npos);
    return out;
}

void replace_all(std::string & text, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return;
    }
    size_t position = 0;
    while ((position = text.find(from, position)) != std::string::npos) {
        text.replace(position, from.size(), to);
        position += to.size();
    }
}

// sopro.text._TERMINALS
bool ends_with_terminal(const std::string & text) noexcept {
    if (text.empty()) {
        return false;
    }
    const char last = text.back();
    return last == '.' || last == '!' || last == '?' || last == '-' ||
           last == ',' || last == ';' || last == ':';
}

// sopro.text._pack: greedily join parts while they fit the codepoint budget.
std::vector<std::string> pack(const std::vector<std::string> & parts, int64_t max_chars) {
    std::vector<std::string> out;
    std::string current;
    for (const auto & part : parts) {
        if (current.empty()) {
            current = part;
        } else if (static_cast<int64_t>(codepoint_length(current) + 1 + codepoint_length(part)) <=
                   max_chars) {
            current += " ";
            current += part;
        } else {
            out.push_back(current);
            current = part;
        }
    }
    if (!current.empty()) {
        out.push_back(current);
    }
    return out;
}

// Split on runs of whitespace that follow one of `delimiters`, i.e. the
// Python lookbehind patterns (?<=[.!?…])\s+ and (?<=[,;:])\s+.
std::vector<std::string> split_after(
    const std::string & text,
    const std::vector<std::string> & delimiters) {
    std::vector<std::string> out;
    size_t start = 0;
    size_t index = 0;
    while (index < text.size()) {
        if (!is_ascii_space(text[index])) {
            ++index;
            continue;
        }
        bool preceded = false;
        for (const auto & delimiter : delimiters) {
            if (index >= delimiter.size() &&
                text.compare(index - delimiter.size(), delimiter.size(), delimiter) == 0) {
                preceded = true;
                break;
            }
        }
        size_t run_end = index;
        while (run_end < text.size() && is_ascii_space(text[run_end])) {
            ++run_end;
        }
        if (preceded) {
            out.push_back(text.substr(start, index - start));
            start = run_end;
        }
        index = run_end;
    }
    out.push_back(text.substr(start));
    return out;
}

std::vector<std::string> split_on_spaces(const std::string & text) {
    std::vector<std::string> out;
    size_t start = 0;
    for (size_t index = 0; index <= text.size(); ++index) {
        if (index == text.size() || text[index] == ' ') {
            out.push_back(text.substr(start, index - start));
            start = index + 1;
        }
    }
    return out;
}

// Matches ^((?:<\|[^|\s]+?\|>\s*)+)(.*)$ — a run of leading <|...|> markers.
size_t special_prefix_end(const std::string & text) noexcept {
    size_t index = 0;
    size_t last_match_end = 0;
    while (index + 2 <= text.size() && text.compare(index, 2, "<|") == 0) {
        const size_t close = text.find("|>", index + 2);
        if (close == std::string::npos) {
            break;
        }
        bool valid = close > index + 2;
        for (size_t i = index + 2; i < close && valid; ++i) {
            if (text[i] == '|' || is_ascii_space(text[i])) {
                valid = false;
            }
        }
        if (!valid) {
            break;
        }
        index = close + 2;
        last_match_end = index;
        while (index < text.size() && is_ascii_space(text[index])) {
            ++index;
        }
    }
    return last_match_end == 0 ? 0 : index;
}

}  // namespace

std::string language_tag(const std::string & language) {
    if (language.empty()) {
        return {};
    }
    std::string key = trim(language);
    std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (key.empty()) {
        return {};
    }
    if (key == "en" || key == "pt" || key == "fr" || key == "de") {
        return "<|lang_" + key + "|>";
    }
    throw std::runtime_error(
        "Sopro: unsupported language '" + language + "'; expected one of de, en, fr, pt");
}

std::string normalize_text(const std::string & text) {
    std::string body = trim(text);
    if (body.empty()) {
        return "You need to add some text for me to talk.";
    }
    const size_t prefix_end = special_prefix_end(body);
    if (prefix_end > 0) {
        const std::string prefix = collapse_whitespace(body.substr(0, prefix_end));
        const std::string rest = trim(body.substr(prefix_end));
        return rest.empty() ? prefix : prefix + " " + normalize_text(rest);
    }
    body = capitalize_first(body);
    body = collapse_whitespace(body);
    replace_all(body, "\xE2\x80\xA6", "...");  // U+2026 HORIZONTAL ELLIPSIS
    static const std::pair<const char *, const char *> kReplacements[] = {
        {" ,", ","}, {" .", "."}, {" !", "!"}, {" ?", "?"}, {" ;", ";"}, {" :", ":"},
        {"\xE2\x80\x9C", "\""},   // U+201C
        {"\xE2\x80\x9D", "\""},   // U+201D
        {"\xE2\x80\x98", "'"},    // U+2018
        {"\xE2\x80\x99", "'"},    // U+2019
    };
    for (const auto & [from, to] : kReplacements) {
        replace_all(body, from, to);
    }
    body = trim(collapse_whitespace(body));
    if (!ends_with_terminal(body)) {
        body += ".";
    }
    return body;
}

std::vector<std::string> split_text(const std::string & text, int64_t max_chars) {
    const std::string flat = collapse_whitespace(text);
    if (max_chars < 1) {
        throw std::runtime_error("Sopro max_segment_chars must be positive");
    }
    if (static_cast<int64_t>(codepoint_length(flat)) <= max_chars) {
        return flat.empty() ? std::vector<std::string>{} : std::vector<std::string>{flat};
    }
    static const std::vector<std::string> kSentenceEnd = {".", "!", "?", "\xE2\x80\xA6"};
    static const std::vector<std::string> kClauseEnd = {",", ";", ":"};
    std::vector<std::string> segments;
    for (const auto & sentence : pack(split_after(flat, kSentenceEnd), max_chars)) {
        if (static_cast<int64_t>(codepoint_length(sentence)) <= max_chars) {
            segments.push_back(sentence);
            continue;
        }
        for (const auto & clause : pack(split_after(sentence, kClauseEnd), max_chars)) {
            if (static_cast<int64_t>(codepoint_length(clause)) <= max_chars) {
                segments.push_back(clause);
            } else {
                for (auto & piece : pack(split_on_spaces(clause), max_chars)) {
                    segments.push_back(std::move(piece));
                }
            }
        }
    }
    return segments;
}

class SoproTextTokenizer::Impl {
public:
    Impl(const std::filesystem::path & model_path, int64_t max_length)
        : max_length_(max_length) {
        const auto status = processor_.Load(model_path.string());
        if (!status.ok()) {
            throw std::runtime_error(
                "Sopro: failed to load SentencePiece model '" + model_path.string() +
                "': " + status.ToString());
        }
        bos_id_ = processor_.bos_id() >= 0 ? processor_.bos_id() : 1;
        eos_id_ = processor_.eos_id() >= 0 ? processor_.eos_id() : 2;
        unk_id_ = processor_.unk_id() >= 0 ? processor_.unk_id() : 0;
        vocab_size_ = processor_.GetPieceSize();
        if (max_length_ < 2) {
            throw std::runtime_error("Sopro tokenizer max_length must be at least 2");
        }
    }

    std::vector<int32_t> encode(const std::string & text, const std::string & language) const {
        const std::string tag = language_tag(language);
        const std::string normalized = normalize_text(tag.empty() ? text : tag + " " + text);
        std::vector<int> pieces;
        const auto status = processor_.Encode(normalized, &pieces);
        if (!status.ok()) {
            throw std::runtime_error("Sopro: SentencePiece encode failed: " + status.ToString());
        }
        std::vector<int32_t> ids;
        ids.reserve(pieces.size() + 2);
        ids.push_back(bos_id_);
        for (const int piece : pieces) {
            ids.push_back(static_cast<int32_t>(piece));
        }
        ids.push_back(eos_id_);
        if (static_cast<int64_t>(ids.size()) > max_length_) {
            // Drop overflowing text ids, not the EOS marker the LM keys on.
            // The constructor guarantees max_length_ >= 2.
            ids.resize(static_cast<size_t>(max_length_ - 1));
            ids.push_back(eos_id_);
        }
        if (ids.empty()) {
            ids.push_back(unk_id_);
        }
        return ids;
    }

    int32_t bos_id_ = 1;
    int32_t eos_id_ = 2;
    int32_t unk_id_ = 0;
    int64_t vocab_size_ = 0;

private:
    sentencepiece::SentencePieceProcessor processor_;
    int64_t max_length_ = 512;
};

SoproTextTokenizer::SoproTextTokenizer(const std::filesystem::path & model_path, int64_t max_length)
    : impl_(std::make_unique<Impl>(model_path, max_length)) {}

SoproTextTokenizer::~SoproTextTokenizer() = default;

std::vector<int32_t> SoproTextTokenizer::encode(
    const std::string & text,
    const std::string & language) const {
    return impl_->encode(text, language);
}

int32_t SoproTextTokenizer::bos_id() const noexcept { return impl_->bos_id_; }
int32_t SoproTextTokenizer::eos_id() const noexcept { return impl_->eos_id_; }
int32_t SoproTextTokenizer::unk_id() const noexcept { return impl_->unk_id_; }
int64_t SoproTextTokenizer::vocab_size() const noexcept { return impl_->vocab_size_; }

}  // namespace engine::community_models::sopro_tts
