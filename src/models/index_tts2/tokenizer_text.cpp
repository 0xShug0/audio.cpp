#include "engine/models/index_tts2/tokenizer_text.h"

#include "engine/framework/text/chinese_normalization.h"
#include "engine/framework/text/text_normalization.h"

#include "bpe-core.h"
#include "unicode.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::index_tts2 {
namespace {

namespace vendor = llama_tokenizer_vendor;

// ---------------------------------------------------------------------------
// Shared UTF-8 / text helpers
// ---------------------------------------------------------------------------

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

uint32_t next_utf8_codepoint(const std::string & text, size_t & offset) {
    const size_t size = std::min(utf8_codepoint_size(static_cast<unsigned char>(text[offset])), text.size() - offset);
    const uint32_t cp = decode_utf8_codepoint(text, offset, size);
    offset += size;
    return cp;
}

bool is_han_codepoint(uint32_t cp) {
    return cp >= 0x4E00U && cp <= 0x9FFFU;
}

bool contains_han(const std::string & text) {
    for (size_t i = 0; i < text.size();) {
        if (is_han_codepoint(next_utf8_codepoint(text, i))) {
            return true;
        }
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

std::string lowercase_ascii(std::string text) {
    for (char & ch : text) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return text;
}

// ---------------------------------------------------------------------------
// v2 SentencePiece helpers
// ---------------------------------------------------------------------------

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

bool is_cjk_codepoint(uint32_t cp) {
    return (cp >= 0x1100U && cp <= 0x11FFU)
        || (cp >= 0x2E80U && cp <= 0xA4CFU)
        || (cp >= 0xA840U && cp <= 0xD7AFU)
        || (cp >= 0xF900U && cp <= 0xFAFFU)
        || (cp >= 0xFE30U && cp <= 0xFE4FU)
        || (cp >= 0xFF65U && cp <= 0xFFDCU)
        || (cp >= 0x20000U && cp <= 0x2FFFFU);
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

// ---------------------------------------------------------------------------
// v2.5 tiktoken helpers
// ---------------------------------------------------------------------------

// IndexTTS-2.5 pads every text segment with a trailing token id 1.
constexpr int32_t kSegmentPadTokenId = 1;

std::string decode_base64(const std::string & input) {
    static const std::array<int8_t, 256> table = [] {
        std::array<int8_t, 256> values{};
        values.fill(-1);
        for (int i = 0; i < 26; ++i) {
            values[static_cast<size_t>('A' + i)] = static_cast<int8_t>(i);
            values[static_cast<size_t>('a' + i)] = static_cast<int8_t>(26 + i);
        }
        for (int i = 0; i < 10; ++i) {
            values[static_cast<size_t>('0' + i)] = static_cast<int8_t>(52 + i);
        }
        values[static_cast<size_t>('+')] = 62;
        values[static_cast<size_t>('/')] = 63;
        return values;
    }();

    std::string out;
    int bits = 0;
    int value = 0;
    for (const unsigned char ch : input) {
        if (ch == '=') {
            break;
        }
        const int8_t digit = table[ch];
        if (digit < 0) {
            throw std::runtime_error("IndexTTS2 tiktoken vocabulary contains invalid base64 token bytes");
        }
        value = (value << 6) | digit;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((value >> bits) & 0xff));
        }
    }
    return out;
}

// The vendored llama BPE runtime works in the GPT-2 byte-to-unicode domain
// (e.g. space becomes U+0120) so that byte-level merges, including tokens that
// split a UTF-8 codepoint, are reproduced exactly. tiktoken ranks are keyed by
// raw bytes, so every token is mapped once at load time.
std::string map_token_bytes(const std::string & bytes) {
    std::string mapped;
    for (const unsigned char byte : bytes) {
        mapped += unicode_byte_to_utf8(byte);
    }
    return mapped;
}

std::string pair_key(const std::string & left, const std::string & right) {
    std::string key = left;
    key.push_back('\0');
    key += right;
    return key;
}

// Language codes in the LANGUAGES order of indextts/utils/tokenizer.py. The
// first 99 entries double as the <|lang|> special tokens below; the remaining
// codes (plus the fallback "common") only index the GPT lang_embedding table.
const std::array<const char *, 99> kLanguages = {
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
    "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
    "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
    "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
    "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
    "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
    "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
    "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
};
const std::array<const char *, 7> kEmbeddingOnlyLanguages = {
    "yue", "minnan", "wuyu", "dialect", "zh/en", "en/zh", "common",
};
constexpr int32_t kCommonLangId = 105;

void add_special_token(vendor::BpeVocabulary & vocab, const std::string & text, int32_t id) {
    vocab.token_to_id.emplace(text, id);
    vocab.id_to_token.emplace(id, vendor::TokenData{text, vendor::TOKEN_ATTR_CONTROL});
}

// Special token order must match indextts/utils/tokenizer.py exactly:
// ids are assigned sequentially starting right after the mergeable ranks.
void register_special_tokens(vendor::BpeVocabulary & vocab, int32_t base_id) {
    static const std::array<const char *, 11> kAudioEvents = {
        "ASR", "AED", "SER", "Speech", "/Speech", "BGM", "/BGM",
        "Laughter", "/Laughter", "Applause", "/Applause",
    };
    static const std::array<const char *, 4> kEmotions = {
        "HAPPY", "SAD", "ANGRY", "NEUTRAL",
    };
    static const std::array<const char *, 6> kTasks = {
        "translate", "transcribe", "startoflm", "startofprev", "nospeech", "notimestamps",
    };
    static const std::array<const char *, 7> kTtsVocal = {
        "TTS/B", "TTS/O", "TTS/Q", "TTS/A", "TTS/CO", "TTS/CL", "TTS/H",
    };

    int32_t id = base_id;
    add_special_token(vocab, "<|endoftext|>", id++);
    add_special_token(vocab, "<|startoftranscript|>", id++);
    for (const char * lang : kLanguages) {
        add_special_token(vocab, "<|" + std::string(lang) + "|>", id++);
    }
    for (const char * event : kAudioEvents) {
        add_special_token(vocab, "<|" + std::string(event) + "|>", id++);
    }
    for (const char * emotion : kEmotions) {
        add_special_token(vocab, "<|" + std::string(emotion) + "|>", id++);
    }
    for (const char * task : kTasks) {
        add_special_token(vocab, "<|" + std::string(task) + "|>", id++);
    }
    for (int i = 1; i <= 30; ++i) {
        add_special_token(vocab, "<|SPECIAL_TOKEN_" + std::to_string(i) + "|>", id++);
    }
    for (const char * vocal : kTtsVocal) {
        add_special_token(vocab, "<|" + std::string(vocal) + "|>", id++);
    }
    for (int i = 1; i <= 13; ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "<|TTS/SP%02d|>", i);
        add_special_token(vocab, name, id++);
    }
    // Timestamps <|0.00|> .. <|30.00|> in 0.02 steps; i * 0.02 == i / 50.
    for (int i = 0; i <= 1500; ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "<|%d.%02d|>", i / 50, (i * 2) % 100);
        add_special_token(vocab, name, id++);
    }
}

std::shared_ptr<vendor::BpeVocabulary> load_tiktoken_vocabulary(const std::filesystem::path & vocab_path) {
    std::ifstream input(vocab_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("IndexTTS2 failed to open tiktoken vocabulary: " + vocab_path.string());
    }

    auto vocab = std::make_shared<vendor::BpeVocabulary>();
    vocab->pre_type = vendor::PreTokenizerType::Gpt2;

    std::string line;
    int64_t mergeable_count = 0;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        std::istringstream parts(line);
        std::string token_base64;
        int64_t rank = -1;
        if (!(parts >> token_base64 >> rank) || rank < 0 || rank > INT32_MAX) {
            throw std::runtime_error("IndexTTS2 tiktoken vocabulary has an invalid line: " + line);
        }
        const std::string bytes = decode_base64(token_base64);
        const auto token_id = static_cast<int32_t>(rank);
        const std::string mapped = map_token_bytes(bytes);
        vocab->token_to_id.emplace(mapped, token_id);
        vocab->id_to_token.emplace(token_id, vendor::TokenData{mapped, 0});
        // tiktoken ranks double as merge priorities: an adjacent pair merges
        // iff its concatenation is a token, with that token's rank. Register
        // every split so find_bpe_rank(left, right) == rank(left + right).
        for (size_t split = 1; split < bytes.size(); ++split) {
            vocab->bpe_ranks.emplace(
                pair_key(map_token_bytes(bytes.substr(0, split)), map_token_bytes(bytes.substr(split))),
                token_id);
        }
        ++mergeable_count;
    }
    if (mergeable_count == 0) {
        throw std::runtime_error("IndexTTS2 tiktoken vocabulary is empty: " + vocab_path.string());
    }

    register_special_tokens(*vocab, static_cast<int32_t>(mergeable_count));
    vendor::rebuild_special_tokens_cache(*vocab);
    return vocab;
}

bool is_kana(const std::string & text) {
    if (text.empty()) {
        return false;
    }
    bool all_hiragana = true;
    bool all_katakana = true;
    for (size_t i = 0; i < text.size();) {
        const uint32_t cp = next_utf8_codepoint(text, i);
        if (cp < 0x3040U || cp > 0x309FU) {
            all_hiragana = false;
        }
        if (cp < 0x30A0U || cp > 0x30FFU) {
            all_katakana = false;
        }
    }
    return all_hiragana || all_katakana;
}

struct AnnotationMatch {
    size_t end = 0;  // one past the match; 0 when there is no match at pos
    size_t word_begin = 0;
    size_t word_end = 0;
    size_t pron_begin = 0;
    size_t pron_end = 0;
};

// Matches <([^|>\n]+)\|([^>\n]+)> anchored at pos.
AnnotationMatch match_pronunciation_annotation(const std::string & text, size_t pos) {
    AnnotationMatch match;
    if (text[pos] != '<') {
        return match;
    }
    size_t cursor = pos + 1;
    const size_t word_begin = cursor;
    while (cursor < text.size() && text[cursor] != '|' && text[cursor] != '>' && text[cursor] != '\n') {
        ++cursor;
    }
    if (cursor == word_begin || cursor >= text.size() || text[cursor] != '|') {
        return match;
    }
    match.word_begin = word_begin;
    match.word_end = cursor;
    const size_t pron_begin = ++cursor;
    while (cursor < text.size() && text[cursor] != '>' && text[cursor] != '\n') {
        ++cursor;
    }
    if (cursor == pron_begin || cursor >= text.size()) {
        return AnnotationMatch{};
    }
    match.pron_begin = pron_begin;
    match.pron_end = cursor;
    match.end = cursor + 1;
    return match;
}

// Base-26 spreadsheet-style index ("a".."z", "aa"..), mirroring the official
// TextNormalizer._protect_pronunciation_annotations placeholder naming.
std::string alpha_placeholder_index(size_t n) {
    std::string s;
    while (true) {
        s.insert(s.begin(), static_cast<char>('a' + (n % 26)));
        const size_t q = n / 26;
        if (q == 0) {
            break;
        }
        n = q - 1;
    }
    return s;
}

using PronunciationPlaceholders = std::vector<std::pair<std::string, std::string>>;

// Replaces <word|pronunciation> annotations with letter-only placeholders so
// text normalization cannot rewrite their digits/symbols (e.g. XING2).
std::pair<std::string, PronunciationPlaceholders> protect_pronunciation_annotations(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    PronunciationPlaceholders placeholders;
    size_t pos = 0;
    while (pos < text.size()) {
        const auto match = match_pronunciation_annotation(text, pos);
        if (match.end == 0) {
            out.push_back(text[pos++]);
            continue;
        }
        std::string key = "PRONPLACEHOLDER" + alpha_placeholder_index(placeholders.size()) + "PRONPLACEHOLDER";
        placeholders.emplace_back(key, text.substr(pos, match.end - pos));
        out += key;
        pos = match.end;
    }
    return {out, placeholders};
}

std::string restore_pronunciation_annotations(std::string text, const PronunciationPlaceholders & placeholders) {
    for (const auto & [key, original] : placeholders) {
        size_t at = 0;
        while ((at = text.find(key, at)) != std::string::npos) {
            text.replace(at, key.size(), original);
            at += original.size();
        }
    }
    return text;
}

// Expands <word|pronunciation> annotations (see infer_v2_5.py
// apply_pronunciation_annotations):
//   Chinese word -> <|SPECIAL_TOKEN_2|>PRON<|SPECIAL_TOKEN_2|>
//   other word   -> <|SPECIAL_TOKEN_1|>PRON<|SPECIAL_TOKEN_1|>
//   kana pron    -> inlined as " PRON "
std::string apply_pronunciation_annotations(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    size_t pos = 0;
    while (pos < text.size()) {
        const auto match = match_pronunciation_annotation(text, pos);
        if (match.end == 0) {
            out.push_back(text[pos++]);
            continue;
        }
        const std::string word = text.substr(match.word_begin, match.word_end - match.word_begin);
        const std::string pron = uppercase_ascii(text.substr(match.pron_begin, match.pron_end - match.pron_begin));
        if (is_kana(pron)) {
            out.push_back(' ');
            out += pron;
            out.push_back(' ');
        } else {
            const char * wrapper = contains_han(word) ? "<|SPECIAL_TOKEN_2|>" : "<|SPECIAL_TOKEN_1|>";
            out += wrapper;
            out += pron;
            out += wrapper;
        }
        pos = match.end;
    }
    return out;
}

// Uppercases the name inside <|...|> markers: re.sub(r'<\|([^|]+)\|>', upper).
std::string uppercase_special_token_names(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    size_t pos = 0;
    while (pos < text.size()) {
        if (text[pos] != '<' || pos + 1 >= text.size() || text[pos + 1] != '|') {
            out.push_back(text[pos++]);
            continue;
        }
        size_t cursor = pos + 2;
        while (cursor < text.size() && text[cursor] != '|') {
            ++cursor;
        }
        if (cursor == pos + 2 || cursor + 1 >= text.size() || text[cursor + 1] != '>') {
            out.push_back(text[pos++]);
            continue;
        }
        out += "<|";
        out += uppercase_ascii(text.substr(pos + 2, cursor - (pos + 2)));
        out += "|>";
        pos = cursor + 2;
    }
    return out;
}

bool is_segment_delimiter(uint32_t cp) {
    switch (cp) {
        case U',':
        case U'.':
        case U'!':
        case U'?':
        case U';':
        case U':':
        case U'\n':
        case 0xFF0CU:  // ，
        case 0x3002U:  // 。
        case 0xFF01U:  // ！
        case 0xFF1FU:  // ？
        case 0x3001U:  // 、
        case 0xFF1BU:  // ；
        case 0xFF1AU:  // ：
            return true;
        default:
            return false;
    }
}

// re.split(r'(?<=[，。！？、；：,\.!\?;:\n])', piece): split after each delimiter.
std::vector<std::string> split_after_delimiters(const std::string & piece) {
    std::vector<std::string> parts;
    std::string current;
    for (size_t i = 0; i < piece.size();) {
        const size_t begin = i;
        const uint32_t cp = next_utf8_codepoint(piece, i);
        current.append(piece, begin, i - begin);
        if (is_segment_delimiter(cp)) {
            parts.push_back(std::move(current));
            current.clear();
        }
    }
    if (!current.empty()) {
        parts.push_back(std::move(current));
    }
    return parts;
}

// Matches "<|SPECIAL_TOKEN_<digits>|>" at pos; returns the match length or 0.
size_t match_special_token_marker(const std::string & text, size_t pos) {
    static const std::string kPrefix = "<|SPECIAL_TOKEN_";
    if (text.compare(pos, kPrefix.size(), kPrefix) != 0) {
        return 0;
    }
    size_t cursor = pos + kPrefix.size();
    const size_t digits_begin = cursor;
    while (cursor < text.size() && std::isdigit(static_cast<unsigned char>(text[cursor])) != 0) {
        ++cursor;
    }
    if (cursor == digits_begin || cursor + 1 >= text.size() || text[cursor] != '|' || text[cursor + 1] != '>') {
        return 0;
    }
    return cursor + 2 - pos;
}

// SPLIT_PROTECTED_PATTERN spans (<|SPECIAL_TOKEN_n|>...<|SPECIAL_TOKEN_n|>)
// are kept atomic during segmentation.
std::vector<std::pair<std::string, bool>> split_atomic_pieces(const std::string & text) {
    std::vector<std::pair<std::string, bool>> pieces;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t opener = std::string::npos;
        size_t opener_len = 0;
        for (size_t i = pos; i < text.size(); ++i) {
            const size_t len = match_special_token_marker(text, i);
            if (len > 0) {
                opener = i;
                opener_len = len;
                break;
            }
        }
        if (opener == std::string::npos) {
            break;
        }
        size_t closer = std::string::npos;
        size_t closer_len = 0;
        for (size_t i = opener + opener_len; i < text.size(); ++i) {
            const size_t len = match_special_token_marker(text, i);
            if (len > 0) {
                closer = i;
                closer_len = len;
                break;
            }
        }
        if (closer == std::string::npos) {
            break;
        }
        if (opener > pos) {
            pieces.emplace_back(text.substr(pos, opener - pos), false);
        }
        pieces.emplace_back(text.substr(opener, closer + closer_len - opener), true);
        pos = closer + closer_len;
    }
    if (pos < text.size()) {
        pieces.emplace_back(text.substr(pos), false);
    }
    return pieces;
}

}  // namespace

IndexTTS2TextTokenizer::IndexTTS2TextTokenizer(std::shared_ptr<const IndexTTS2Assets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("IndexTTS2 text tokenizer requires assets");
    }
    variant_ = index_tts2_variant_from_version(assets_->config.version);
    if (variant_ == IndexTTS2Variant::kV2_5) {
        vocab_ = load_tiktoken_vocabulary(assets_->resources.require_file("tiktoken"));
    } else {
        pieces_ = engine::tokenizers::load_sentencepiece_model(assets_->resources.require_file("bpe"));
        piece_to_id_.reserve(pieces_.size());
        for (const auto & piece : pieces_) {
            piece_to_id_.emplace(piece.text, static_cast<int32_t>(piece.id));
        }
    }
}

std::string IndexTTS2TextTokenizer::normalize_english(const std::string & text) const {
    engine::text::EnglishTextNormalizationOptions options;
    options.expand_common_contractions = true;
    options.index_tts_punctuation = true;
    options.uppercase_ascii = variant_ == IndexTTS2Variant::kV2;
    options.verbalize_symbols = true;
    return engine::text::normalize_english_text(text, options);
}

std::string IndexTTS2TextTokenizer::normalize_chinese(const std::string & text) const {
    return engine::text::normalize_chinese_text(
        text,
        engine::text::ChineseTextNormalizationTarget::IndexTTS);
}

std::vector<int32_t> IndexTTS2TextTokenizer::encode(const std::string & text) const {
    if (variant_ == IndexTTS2Variant::kV2_5) {
        return vendor::tokenize_bpe(*vocab_, text, true);
    }
    return engine::tokenizers::tokenize_sentencepiece(pieces_, normalize_text(text));
}

std::string IndexTTS2TextTokenizer::normalize_text(const std::string & text) const {
    if (variant_ != IndexTTS2Variant::kV2) {
        throw std::runtime_error("IndexTTS2 normalize_text is only available for the v2 SentencePiece tokenizer");
    }
    return contains_han(text) ? tokenize_by_cjk_char(normalize_chinese(text)) : normalize_english(text);
}

std::vector<std::string> IndexTTS2TextTokenizer::tokenize_to_pieces(const std::string & text) const {
    if (variant_ != IndexTTS2Variant::kV2) {
        throw std::runtime_error("IndexTTS2 tokenize_to_pieces is only available for the v2 SentencePiece tokenizer");
    }
    const auto ids = encode(text);
    std::vector<std::string> out;
    out.reserve(ids.size());
    for (const int32_t id : ids) {
        out.push_back(id_to_piece(id));
    }
    return out;
}

int32_t IndexTTS2TextTokenizer::special_token_id(const std::string & token_text) const {
    if (variant_ != IndexTTS2Variant::kV2_5) {
        throw std::runtime_error("IndexTTS2 special_token_id is only available for the v2.5 tiktoken tokenizer");
    }
    const auto it = vocab_->token_to_id.find(token_text);
    return it == vocab_->token_to_id.end() ? -1 : it->second;
}

int32_t IndexTTS2TextTokenizer::lang_to_id(const std::string & lang) {
    const std::string normalized = lowercase_ascii(lang);
    for (size_t i = 0; i < kLanguages.size(); ++i) {
        if (normalized == kLanguages[i]) {
            return static_cast<int32_t>(i);
        }
    }
    for (size_t i = 0; i < kEmbeddingOnlyLanguages.size(); ++i) {
        if (normalized == kEmbeddingOnlyLanguages[i]) {
            return static_cast<int32_t>(kLanguages.size() + i);
        }
    }
    return kCommonLangId;
}

IndexTTS2TextEncoding IndexTTS2TextTokenizer::encode_for_inference(
    const std::string & text,
    int max_text_tokens_per_segment,
    const std::string & lang) const {
    if (max_text_tokens_per_segment <= 0) {
        throw std::runtime_error("IndexTTS2 max_text_tokens_per_segment must be positive");
    }
    if (variant_ == IndexTTS2Variant::kV2_5) {
        return encode_for_inference_v2_5(text, max_text_tokens_per_segment, lang);
    }
    return encode_for_inference_v2(text, max_text_tokens_per_segment);
}

IndexTTS2TextEncoding IndexTTS2TextTokenizer::encode_for_inference_v2(
    const std::string & text,
    int max_text_tokens_per_segment) const {
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

IndexTTS2TextEncoding IndexTTS2TextTokenizer::encode_for_inference_v2_5(
    const std::string & text,
    int max_text_tokens_per_segment,
    const std::string & lang) const {
    std::string resolved_lang = lowercase_ascii(lang);
    if (resolved_lang.empty()) {
        resolved_lang = contains_han(text) ? "zh" : "en";
    }

    std::string processed = text;
    if (resolved_lang == "zh" || resolved_lang == "en") {
        // Protect <word|pronunciation> annotations from the normalizer, as the
        // official TextNormalizer does inside normalize().
        auto protected_text = protect_pronunciation_annotations(processed);
        protected_text.first = resolved_lang == "zh"
            ? normalize_chinese(protected_text.first)
            : normalize_english(protected_text.first);
        processed = restore_pronunciation_annotations(std::move(protected_text.first), protected_text.second);
    } else {
        // ja/es/ar/...: no full TN, but the official front.normalize still
        // applies its punctuation map to every language (、→"," 。→"." ...).
        // Text with Han characters takes the zh path, which additionally maps
        // "$" → ".".
        const bool han = contains_han(processed);
        processed = engine::text::normalize_index_tts_punctuation(std::move(processed));
        if (han) {
            processed = engine::text::replace_all(std::move(processed), "$", ".");
        }
    }
    if (resolved_lang == "zh" || resolved_lang == "ja" || resolved_lang == "en") {
        processed = lowercase_ascii(std::move(processed));
    } else if (resolved_lang == "es") {
        processed = uppercase_ascii(std::move(processed));
    }
    processed = apply_pronunciation_annotations(processed);
    processed = uppercase_special_token_names(processed);

    const std::string lang_prefix = "<|" + resolved_lang + "|> ";
    const auto prefix_tokens = static_cast<int64_t>(encode(lang_prefix).size());
    const int64_t capacity = assets_->config.gpt.max_text_tokens;
    int64_t budget = std::min<int64_t>(max_text_tokens_per_segment, capacity - 2) - prefix_tokens;
    budget = std::max<int64_t>(budget, 1);

    std::vector<std::string> segments;
    const auto token_len = [this](const std::string & value) {
        return static_cast<int64_t>(encode(value).size());
    };
    if (token_len(processed) <= budget) {
        segments.push_back(processed);
    } else {
        std::vector<std::string> chunks;
        for (const auto & [piece, atomic] : split_atomic_pieces(processed)) {
            if (atomic) {
                chunks.push_back(piece);
                continue;
            }
            for (const auto & part : split_after_delimiters(piece)) {
                if (token_len(part) <= budget) {
                    chunks.push_back(part);
                    continue;
                }
                std::string current;
                for (size_t i = 0; i < part.size();) {
                    const size_t begin = i;
                    next_utf8_codepoint(part, i);
                    const std::string ch = part.substr(begin, i - begin);
                    if (!current.empty() && token_len(current + ch) > budget) {
                        chunks.push_back(std::move(current));
                        current = ch;
                    } else {
                        current += ch;
                    }
                }
                if (!current.empty()) {
                    chunks.push_back(std::move(current));
                }
            }
        }
        std::string current;
        for (const auto & chunk : chunks) {
            if (!current.empty() && token_len(current + chunk) > budget) {
                segments.push_back(std::move(current));
                current = chunk;
            } else {
                current += chunk;
            }
        }
        if (!current.empty()) {
            segments.push_back(std::move(current));
        }
        if (segments.empty()) {
            segments.push_back(processed);
        }
    }

    IndexTTS2TextEncoding encoding;
    encoding.lang = resolved_lang;
    encoding.normalized_text = processed;
    encoding.segments.reserve(segments.size());
    for (const auto & segment : segments) {
        encoding.segments.push_back({segment});
    }
    encoding.segment_token_ids.reserve(segments.size());
    for (const auto & segment : segments) {
        std::vector<int32_t> ids = encode(lang_prefix + segment);
        ids.push_back(kSegmentPadTokenId);
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
