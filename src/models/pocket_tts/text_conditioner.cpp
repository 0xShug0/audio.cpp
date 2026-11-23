#include "engine/models/pocket_tts/text_conditioner.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/models/pocket_tts/assets.h"
#include "graph_common.h"
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <unordered_set>

namespace engine::models::pocket_tts {
namespace {

struct CodePoint {
    uint32_t value = 0;
    size_t length = 0;
};

// Decodes the UTF-8 sequence starting at `start`. A malformed sequence is returned as a
// single opaque byte, so callers never read past it and never treat it as a letter.
CodePoint decode_utf8_at(const std::string & text, size_t start) {
    const auto lead = static_cast<unsigned char>(text[start]);
    if (lead < 0x80) {
        return {lead, 1};
    }
    size_t length = 0;
    uint32_t value = 0;
    if ((lead & 0xE0) == 0xC0) {
        length = 2;
        value = lead & 0x1F;
    } else if ((lead & 0xF0) == 0xE0) {
        length = 3;
        value = lead & 0x0F;
    } else if ((lead & 0xF8) == 0xF0) {
        length = 4;
        value = lead & 0x07;
    } else {
        return {0xFFFFFFFFU, 1};
    }
    if (start + length > text.size()) {
        return {0xFFFFFFFFU, 1};
    }
    for (size_t i = 1; i < length; ++i) {
        const auto next = static_cast<unsigned char>(text[start + i]);
        if ((next & 0xC0) != 0x80) {
            return {0xFFFFFFFFU, 1};
        }
        value = (value << 6) | (next & 0x3F);
    }
    return {value, length};
}

std::string encode_utf8(uint32_t value) {
    std::string out;
    if (value < 0x80) {
        out.push_back(static_cast<char>(value));
    } else if (value < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (value >> 6)));
        out.push_back(static_cast<char>(0x80 | (value & 0x3F)));
    } else if (value < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (value >> 12)));
        out.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (value & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (value >> 18)));
        out.push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (value & 0x3F)));
    }
    return out;
}

// The reference works on Python str, where "é".isalnum() is true. Testing the last BYTE of
// a UTF-8 string instead left every sentence ending in an accented letter without its
// final period ("C'est réglé"). Covers ASCII plus the Latin-1 Supplement and Latin
// Extended-A/B letters used by every Pocket TTS language; other scripts keep the byte rule.
bool is_letter_or_digit(uint32_t value) {
    if (value < 0x80) {
        return std::isalnum(static_cast<int>(value)) != 0;
    }
    return value >= 0x00C0 && value <= 0x024F && value != 0x00D7 && value != 0x00F7;
}

// Uppercase of a lowercase Latin letter, or 0 when it has no single-code-point uppercase:
// the reference capitalizes the first letter with str.upper(), which turns "état" into
// "État" where a byte-wise toupper() left it lowercase.
uint32_t latin_uppercase(uint32_t value) {
    if (value >= 'a' && value <= 'z') {
        return value - 0x20;
    }
    if (value >= 0x00E0 && value <= 0x00FE && value != 0x00F7) {
        return value - 0x20;
    }
    if (value == 0x00FF) {
        return 0x0178;
    }
    if (value == 0x0131) {
        return 'I';
    }
    const bool even_upper = (value >= 0x0100 && value <= 0x012F) || (value >= 0x0132 && value <= 0x0137) ||
        (value >= 0x014A && value <= 0x0177);
    if (even_upper && (value & 1U) == 1U) {
        return value - 1;
    }
    const bool odd_upper = (value >= 0x0139 && value <= 0x0148) || (value >= 0x0179 && value <= 0x017E);
    if (odd_upper && (value & 1U) == 0U) {
        return value - 1;
    }
    return 0;
}

std::string strip_ascii_whitespace(std::string text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
        text.erase(text.begin());
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.pop_back();
    }
    return text;
}

// The characters Python's str.split() separates on (str.isspace()), so the collapse below also
// folds the no-break spaces French text puts before ":" and inside guillemets.
bool is_python_whitespace(uint32_t value) {
    return (value >= 0x09 && value <= 0x0D) || (value >= 0x1C && value <= 0x20) || value == 0x85 ||
        value == 0xA0 || value == 0x1680 || (value >= 0x2000 && value <= 0x200A) || value == 0x2028 ||
        value == 0x2029 || value == 0x202F || value == 0x205F || value == 0x3000;
}

// The reference's replace_characters step (pocket_tts prepare_text_prompt):
//     text = " ".join(text.translate(str.maketrans(replace_characters)).split())
//     text = re.sub(r"([.!?…])\s*[,;:]", r"\1", text)
// Each character is replaced once, in a single pass. Deleted quotes leave '"Hi?", she said' as
// 'Hi?, she said', which reads as a sentence end followed by a stray comma; the second step keeps
// only the sentence mark.
std::string translate_replace_characters(
    const std::string & text,
    const std::vector<std::pair<std::string, std::string>> & replace_characters) {
    std::string translated;
    translated.reserve(text.size());
    for (size_t pos = 0; pos < text.size();) {
        const size_t length = decode_utf8_at(text, pos).length;
        const std::string character = text.substr(pos, length);
        const auto replacement = std::find_if(
            replace_characters.begin(), replace_characters.end(), [&](const auto & entry) {
                return entry.first == character;
            });
        translated += replacement == replace_characters.end() ? character : replacement->second;
        pos += length;
    }

    std::string collapsed;
    collapsed.reserve(translated.size());
    bool pending_space = false;
    for (size_t pos = 0; pos < translated.size();) {
        const CodePoint code_point = decode_utf8_at(translated, pos);
        if (is_python_whitespace(code_point.value)) {
            pending_space = !collapsed.empty();
        } else {
            if (pending_space) {
                collapsed.push_back(' ');
                pending_space = false;
            }
            collapsed.append(translated, pos, code_point.length);
        }
        pos += code_point.length;
    }

    std::string out;
    out.reserve(collapsed.size());
    for (size_t pos = 0; pos < collapsed.size();) {
        const CodePoint code_point = decode_utf8_at(collapsed, pos);
        out.append(collapsed, pos, code_point.length);
        pos += code_point.length;
        const uint32_t mark = code_point.value;
        if (mark == '.' || mark == '!' || mark == '?' || mark == 0x2026) {
            size_t next = pos;
            while (next < collapsed.size() && collapsed[next] == ' ') {
                ++next;
            }
            if (next < collapsed.size() &&
                (collapsed[next] == ',' || collapsed[next] == ';' || collapsed[next] == ':')) {
                pos = next + 1;
            }
        }
    }
    return out;
}

// Gradium Phonon: a final letter or digit gets a period, any other ending is kept.
void end_after_letter_or_digit(std::string & text) {
    size_t last_start = text.size() - 1;
    while (last_start > 0 && (static_cast<unsigned char>(text[last_start]) & 0xC0) == 0x80) {
        --last_start;
    }
    const CodePoint last = decode_utf8_at(text, last_start);
    if (last_start + last.length == text.size() && is_letter_or_digit(last.value)) {
        text.push_back('.');
    }
}

// The reference's _ensure_terminal_punctuation, as PocketTTS Small does it: past any closing
// quotes or brackets, a sentence mark is kept, a trailing ",;:-" or dash becomes a period, and
// anything else gets a period after the closers.
void end_like_pocket_tts(std::string & text) {
    size_t core_end = text.size();
    uint32_t terminal = 0;
    while (core_end > 0) {
        size_t start = core_end - 1;
        while (start > 0 && (static_cast<unsigned char>(text[start]) & 0xC0) == 0x80) {
            --start;
        }
        terminal = decode_utf8_at(text, start).value;
        if (terminal != '"' && terminal != '\'' && terminal != ')' && terminal != ']' && terminal != 0x201D &&
            terminal != 0x2019 && terminal != 0x00BB && terminal != ' ') {
            break;
        }
        core_end = start;
    }
    if (core_end > 0 && terminal != '.' && terminal != '!' && terminal != '?' && terminal != 0x2026) {
        size_t weak_end = core_end;
        while (weak_end > 0) {
            size_t start = weak_end - 1;
            while (start > 0 && (static_cast<unsigned char>(text[start]) & 0xC0) == 0x80) {
                --start;
            }
            const uint32_t value = decode_utf8_at(text, start).value;
            if (value != ',' && value != ';' && value != ':' && value != '-' && value != 0x2013 && value != 0x2014 &&
                value != ' ') {
                break;
            }
            weak_end = start;
        }
        if (weak_end < core_end) {
            text = text.substr(0, weak_end) + "." + strip_ascii_whitespace(text.substr(core_end));
        } else {
            text.push_back('.');
        }
    }
}

std::string prepare_pocket_tts_text(
    std::string text,
    bool pad_with_spaces_for_short_inputs,
    bool remove_semicolons,
    bool capitalize_first_letter = true,
    const std::vector<std::pair<std::string, std::string>> & replace_characters = {},
    PromptPunctuation punctuation = PromptPunctuation::PocketTTS) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
        text.erase(text.begin());
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.pop_back();
    }
    if (!replace_characters.empty()) {
        text = translate_replace_characters(text, replace_characters);
    }
    if (text.empty()) {
        throw std::runtime_error("PocketTTS text prompt cannot be empty");
    }

    for (char & ch : text) {
        if (ch == '\n' || ch == '\r') {
            ch = ' ';
        }
    }
    for (size_t i = 1; i < text.size();) {
        if (text[i] == ' ' && text[i - 1] == ' ') {
            text.erase(text.begin() + static_cast<ptrdiff_t>(i));
        } else {
            ++i;
        }
    }
    if (remove_semicolons) {
        std::replace(text.begin(), text.end(), ';', ',');
    }
    if (capitalize_first_letter) {
        // pocket-tts skips any opening quotes or brackets first ('"hello," she said').
        size_t start = 0;
        CodePoint first = decode_utf8_at(text, start);
        while (punctuation == PromptPunctuation::PocketTTS && start + first.length < text.size() &&
               (first.value == '"' || first.value == '\'' || first.value == 0x201C || first.value == 0x2018 ||
                first.value == '(' || first.value == '[' || first.value == 0x00AB)) {
            start += first.length;
            first = decode_utf8_at(text, start);
        }
        if (const uint32_t upper = latin_uppercase(first.value); upper != 0) {
            text.replace(start, first.length, encode_utf8(upper));
        }
    }
    if (punctuation == PromptPunctuation::PocketTTS) {
        end_like_pocket_tts(text);
    } else {
        end_after_letter_or_digit(text);
    }
    if (pad_with_spaces_for_short_inputs) {
        int words = 0;
        bool in_word = false;
        for (const char ch : text) {
            if (std::isspace(static_cast<unsigned char>(ch))) {
                if (in_word) {
                    ++words;
                    in_word = false;
                }
            } else {
                in_word = true;
            }
        }
        if (in_word) {
            ++words;
        }
        if (words < 5) {
            text = "        " + text;
        }
    }
    return text;
}

std::vector<float> gather_token_embeddings(
    const assets::TensorDataF32 & embedding_table,
    const std::vector<int32_t> & tokens,
    int64_t hidden_size) {
    if (embedding_table.shape.rank != 2 || embedding_table.shape.dims[1] != hidden_size) {
        throw std::runtime_error("PocketTTS embedding table must have shape [vocab, hidden_size]");
    }
    const int64_t vocab = embedding_table.shape.dims[0];
    const int64_t dim = embedding_table.shape.dims[1];

    std::vector<float> output(static_cast<size_t>(tokens.size() * dim));
    for (size_t token_index = 0; token_index < tokens.size(); ++token_index) {
        const int32_t token = tokens[token_index];
        if (token < 0 || token >= vocab) {
            throw std::runtime_error("PocketTTS token id is out of range for embedding table");
        }
        const size_t src_offset = static_cast<size_t>(token) * static_cast<size_t>(dim);
        const size_t dst_offset = token_index * static_cast<size_t>(dim);
        std::copy_n(
            embedding_table.values.begin() + static_cast<ptrdiff_t>(src_offset),
            static_cast<size_t>(dim),
            output.begin() + static_cast<ptrdiff_t>(dst_offset));
    }
    return output;
}

// Token ids of `symbols` minus the first one, exactly as the reference builds its boundary
// sets: SentencePiece prefixes the string with its word marker, and only the ids that
// follow it stand for the punctuation.
std::unordered_set<int32_t> boundary_token_ids(
    const std::vector<tokenizers::SentencePiecePiece> & pieces,
    const std::string & symbols) {
    const std::vector<int32_t> ids = tokenizers::tokenize_sentencepiece(pieces, symbols);
    std::unordered_set<int32_t> out;
    for (size_t i = 1; i < ids.size(); ++i) {
        out.insert(ids[i]);
    }
    return out;
}

std::string decode_range(
    const std::vector<tokenizers::SentencePiecePiece> & pieces,
    const std::vector<int32_t> & tokens,
    size_t begin,
    size_t end) {
    return tokenizers::decode_sentencepiece(
        pieces,
        std::vector<int32_t>(
            tokens.begin() + static_cast<ptrdiff_t>(begin),
            tokens.begin() + static_cast<ptrdiff_t>(end)));
}

// "3.5": a period between two digits does not end a sentence.
bool is_decimal_period_boundary(
    const std::vector<tokenizers::SentencePiecePiece> & pieces,
    const std::vector<int32_t> & tokens,
    size_t segment_start) {
    const std::string prefix = decode_range(pieces, tokens, 0, segment_start);
    const std::string suffix = decode_range(pieces, tokens, segment_start, tokens.size());
    return prefix.size() >= 2 && prefix.back() == '.' &&
        std::isdigit(static_cast<unsigned char>(prefix[prefix.size() - 2])) != 0 && !suffix.empty() &&
        std::isdigit(static_cast<unsigned char>(suffix.front())) != 0;
}

bool is_closer(uint32_t value) {
    return value == '"' || value == '\'' || value == 0x201D || value == 0x2019 || value == ')' || value == ']' ||
        value == 0x00BB;
}

// The reference's _skip_closers: where the segment that starts at `segment_start` really starts,
// after the closing quotes and brackets that directly follow the punctuation ("“Stop.”",
// "(quietly.)"), so that they stay with their sentence. A curly quote is several byte tokens
// that decode to U+FFFD on their own, so the start only moves past complete closers. A closer
// after a space opens the next sentence and is left alone.
size_t skip_closers(
    const std::vector<tokenizers::SentencePiecePiece> & pieces,
    const std::vector<int32_t> & tokens,
    size_t segment_start) {
    const std::string punctuation = decode_range(pieces, tokens, segment_start - 1, segment_start);
    size_t start = segment_start;
    const size_t stop = std::min(tokens.size(), segment_start + 8);
    for (size_t end = segment_start + 1; end <= stop; ++end) {
        const std::string window = decode_range(pieces, tokens, segment_start - 1, end);
        if (window.size() <= punctuation.size() || window.compare(0, punctuation.size(), punctuation) != 0) {
            break;
        }
        bool closers_only = true;
        bool partial = false;
        for (size_t pos = punctuation.size(); pos < window.size();) {
            const CodePoint code_point = decode_utf8_at(window, pos);
            if (code_point.value == 0xFFFD) {
                partial = true;
            } else if (!is_closer(code_point.value)) {
                closers_only = false;
                break;
            }
            pos += code_point.length;
        }
        if (!closers_only) {
            break;
        }
        if (!partial) {
            start = end;
        }
    }
    return start;
}

// A segment starts at the first token that is not a boundary token after a run of boundary
// tokens, or past the closers right after them when `closers_stay` is set. Returns
// [0, starts..., size], so consecutive pairs are the segments.
std::vector<size_t> find_boundary_indices(
    const std::vector<int32_t> & tokens,
    const std::unordered_set<int32_t> & boundary_ids,
    const std::vector<tokenizers::SentencePiecePiece> & pieces,
    bool check_decimal_period,
    bool closers_stay) {
    std::vector<size_t> indices{0};
    bool previous_was_boundary = false;
    for (size_t index = 0; index < tokens.size(); ++index) {
        if (boundary_ids.count(tokens[index]) != 0) {
            previous_was_boundary = true;
            continue;
        }
        if (previous_was_boundary &&
            !(check_decimal_period && is_decimal_period_boundary(pieces, tokens, index))) {
            const size_t start = closers_stay ? skip_closers(pieces, tokens, index) : index;
            if (start < tokens.size()) {
                indices.push_back(start);
            }
        }
        previous_was_boundary = false;
    }
    indices.push_back(tokens.size());
    return indices;
}

struct TextSegment {
    int64_t token_count = 0;
    std::string text;
};

std::vector<TextSegment> segments_from_boundaries(
    const std::vector<tokenizers::SentencePiecePiece> & pieces,
    const std::vector<int32_t> & tokens,
    const std::vector<size_t> & boundaries) {
    std::vector<TextSegment> segments;
    for (size_t i = 0; i + 1 < boundaries.size(); ++i) {
        segments.push_back(TextSegment{
            static_cast<int64_t>(boundaries[i + 1] - boundaries[i]),
            decode_range(pieces, tokens, boundaries[i], boundaries[i + 1]),
        });
    }
    return segments;
}

PromptPunctuation prompt_punctuation(const models::pocket_tts::PocketTTSAssets & manifest) {
    return manifest.impl != nullptr ? manifest.impl->prompt_punctuation : PromptPunctuation::PocketTTS;
}

}  // namespace

TextConditioner::TextConditioner(TextConditionerConfig config) : config_(config) {}

TextConditioningResult TextConditioner::prepare(
    const models::pocket_tts::PocketTTSAssets & manifest,
    const models::pocket_tts::PocketTTSHostWeights & weights,
    const std::string & text) const {
    TextConditioningResult result;
    const double prepare_ms = engine::debug::measure_ms([&]() {
        result.prepared_text = prepare_pocket_tts_text(
            text,
            manifest.model_config.pad_with_spaces_for_short_inputs,
            manifest.model_config.remove_semicolons,
            manifest.model_config.capitalize_first_letter,
            manifest.model_config.replace_characters,
            prompt_punctuation(manifest));
        if (manifest.tokenizer_pieces.empty()) {
            throw std::runtime_error("PocketTTS tokenizer pieces are not loaded");
        }
        result.tokens = tokenizers::tokenize_sentencepiece(manifest.tokenizer_pieces, result.prepared_text);
        const auto & embedding_table = weights.conditioner_embedding_table;
        result.text_embeddings = gather_token_embeddings(embedding_table, result.tokens, config_.hidden_size);
    });
    engine::debug::timing_log_scalar("pocket_tts.text.prepare_ms", prepare_ms);
    return result;
}

std::string TextConditioner::apply_replace_characters(
    const models::pocket_tts::PocketTTSAssets & manifest,
    const std::string & text) const {
    const auto & replace_characters = manifest.model_config.replace_characters;
    return replace_characters.empty() ? text : translate_replace_characters(text, replace_characters);
}

std::vector<std::string> TextConditioner::split_into_sentence_chunks(
    const models::pocket_tts::PocketTTSAssets & manifest,
    const std::string & text,
    int64_t max_tokens) const {
    if (manifest.tokenizer_pieces.empty()) {
        throw std::runtime_error("PocketTTS tokenizer pieces are not loaded");
    }
    if (max_tokens <= 0) {
        throw std::runtime_error("PocketTTS max_tokens must be positive");
    }
    const auto & pieces = manifest.tokenizer_pieces;
    // Same normalization as the prompt the model will see, then stripped: the reference drops
    // the short-input padding here and adds it back chunk by chunk in prepare().
    const std::string prepared = strip_ascii_whitespace(prepare_pocket_tts_text(
        text,
        manifest.model_config.pad_with_spaces_for_short_inputs,
        manifest.model_config.remove_semicolons,
        manifest.model_config.capitalize_first_letter,
        manifest.model_config.replace_characters,
        prompt_punctuation(manifest)));
    const bool closers_stay = prompt_punctuation(manifest) == PromptPunctuation::PocketTTS;
    const std::vector<int32_t> tokens = tokenizers::tokenize_sentencepiece(pieces, prepared);
    const std::vector<TextSegment> sentences = segments_from_boundaries(
        pieces,
        tokens,
        find_boundary_indices(tokens, boundary_token_ids(pieces, ".!...?"), pieces, true, closers_stay));

    // A sentence longer than the budget is split again on commas, semicolons and colons,
    // which keeps the model from skipping words at the end of an overlong chunk.
    const std::unordered_set<int32_t> fallback_ids = boundary_token_ids(pieces, ",;:");
    std::vector<TextSegment> refined;
    for (const TextSegment & sentence : sentences) {
        if (sentence.token_count <= max_tokens) {
            refined.push_back(sentence);
            continue;
        }
        const std::vector<int32_t> sub_tokens =
            tokenizers::tokenize_sentencepiece(pieces, strip_ascii_whitespace(sentence.text));
        std::vector<TextSegment> parts = segments_from_boundaries(
            pieces,
            sub_tokens,
            find_boundary_indices(sub_tokens, fallback_ids, pieces, false, closers_stay));
        if (parts.size() > 1) {
            refined.insert(refined.end(), parts.begin(), parts.end());
        } else {
            refined.push_back(sentence);
        }
    }

    std::vector<std::string> chunks;
    std::string current;
    int64_t current_tokens = 0;
    for (const TextSegment & segment : refined) {
        if (current.empty()) {
            current = segment.text;
            current_tokens = segment.token_count;
            continue;
        }
        if (current_tokens + segment.token_count > max_tokens) {
            chunks.push_back(strip_ascii_whitespace(current));
            current = segment.text;
            current_tokens = segment.token_count;
        } else {
            current += " " + segment.text;
            current_tokens += segment.token_count;
        }
    }
    if (!current.empty()) {
        chunks.push_back(strip_ascii_whitespace(current));
    }
    return chunks;
}

}  // namespace engine::models::pocket_tts
