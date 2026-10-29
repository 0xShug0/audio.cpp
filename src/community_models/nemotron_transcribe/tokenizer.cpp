#include "engine/community_models/nemotron_transcribe/tokenizer.h"

#include "engine/framework/io/json.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <utf8proc.h>

#include <algorithm>
#include <map>
#include <regex>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace engine::models::nemotron_transcribe {
namespace {

struct Character {
    size_t byte = 0;
    utf8proc_int32_t codepoint = 0;
    utf8proc_category_t category = UTF8PROC_CATEGORY_CN;
    bool letter() const { return category >= UTF8PROC_CATEGORY_LU && category <= UTF8PROC_CATEGORY_LO; }
    bool number() const { return category >= UTF8PROC_CATEGORY_ND && category <= UTF8PROC_CATEGORY_NO; }
    bool mark() const { return category >= UTF8PROC_CATEGORY_MN && category <= UTF8PROC_CATEGORY_ME; }
    bool upper_run() const {
        return category == UTF8PROC_CATEGORY_LU || category == UTF8PROC_CATEGORY_LT ||
            category == UTF8PROC_CATEGORY_LM || category == UTF8PROC_CATEGORY_LO || mark();
    }
    bool lower_run() const {
        return category == UTF8PROC_CATEGORY_LL || category == UTF8PROC_CATEGORY_LM ||
            category == UTF8PROC_CATEGORY_LO || mark();
    }
    bool newline() const { return codepoint == '\r' || codepoint == '\n'; }
    bool whitespace() const {
        return (codepoint >= 9 && codepoint <= 13) || codepoint == 0x85 ||
            category == UTF8PROC_CATEGORY_ZS || category == UTF8PROC_CATEGORY_ZL || category == UTF8PROC_CATEGORY_ZP;
    }
    bool prefix() const { return !newline() && !letter() && !number(); }
    bool punctuation_run() const { return !whitespace() && !letter() && !number(); }
};

// tokenizer.json's newer Tekken expression, in alternative order:
// [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|
// [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|
// \p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Exact categories matter for combining marks and non-ASCII case transitions.
std::vector<std::string> split_tekken(const std::string & text) {
    std::vector<Character> chars;
    for (size_t offset = 0; offset < text.size();) {
        utf8proc_int32_t codepoint = 0;
        const auto consumed = utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t *>(text.data() + offset),
            static_cast<utf8proc_ssize_t>(text.size() - offset), &codepoint);
        if (consumed <= 0) throw std::runtime_error("Nemotron Transcribe text contains invalid UTF-8");
        chars.push_back({offset, codepoint, utf8proc_category(codepoint)});
        offset += static_cast<size_t>(consumed);
    }
    const auto size = chars.size();
    auto lower_end = [&](size_t begin) {
        while (begin < size && chars[begin].lower_run()) ++begin;
        return begin;
    };
    auto letter_alternative = [&](size_t begin, bool first) -> size_t {
        size_t upper_end = begin;
        while (upper_end < size && chars[upper_end].upper_run()) ++upper_end;
        if (!first) return upper_end > begin ? lower_end(upper_end) : begin;
        const auto end = lower_end(upper_end);
        if (end > upper_end) return end;
        // The greedy upper run backtracks to the last place where lower+ can
        // match. Marks and caseless letters belong to both classes.
        while (upper_end > begin) {
            --upper_end;
            if (chars[upper_end].lower_run()) return lower_end(upper_end);
        }
        return begin;
    };
    std::vector<std::string> pieces;
    for (size_t pos = 0; pos < size;) {
        size_t end = pos;
        for (const bool first : {true, false}) {
            if (chars[pos].prefix() && pos + 1 < size) {
                const auto candidate = letter_alternative(pos + 1, first);
                if (candidate > pos + 1) end = candidate;
            }
            if (end == pos) end = letter_alternative(pos, first);
            if (end > pos) break;
        }
        if (end == pos && chars[pos].number()) end = pos + 1;
        if (end == pos) {
            size_t begin = pos + (chars[pos].codepoint == ' ' ? 1 : 0);
            end = begin;
            while (end < size && chars[end].punctuation_run()) ++end;
            if (end > begin) {
                while (end < size && (chars[end].newline() || chars[end].codepoint == '/')) ++end;
            } else end = pos;
        }
        if (end == pos && chars[pos].whitespace()) {
            size_t whitespace_end = pos;
            size_t last_newline = pos;
            while (whitespace_end < size && chars[whitespace_end].whitespace()) {
                if (chars[whitespace_end].newline()) last_newline = whitespace_end + 1;
                ++whitespace_end;
            }
            if (last_newline > pos) end = last_newline;
            else if (whitespace_end < size && whitespace_end - pos > 1) end = whitespace_end - 1;
            else end = whitespace_end;
        }
        if (end == pos) end = pos + 1;
        const auto end_byte = end < size ? chars[end].byte : text.size();
        pieces.push_back(text.substr(chars[pos].byte, end_byte - chars[pos].byte));
        pos = end;
    }
    return pieces;
}

tokenizers::LlamaBpeTokenizerSpec make_spec(const std::filesystem::path & tokenizer_json,
    const std::filesystem::path & tokenizer_config) {
    tokenizers::LlamaBpeTokenizerSpec spec{{}, {}, tokenizer_config, tokenizer_json,
        tokenizers::LlamaBpePreTokenizer::Tekken, {}, {}, true};
    spec.pre_tokenize = split_tekken;
    return spec;
}

void require_id(const tokenizers::LlamaBpeTokenizer & tokenizer, const std::string & text, int32_t expected) {
    const auto actual = tokenizer.find_token_id(text);
    if (!actual || *actual != expected || tokenizer.encode(text, true) != std::vector<int32_t>{expected}) {
        throw std::runtime_error("Nemotron Transcribe tokenizer has an unexpected special token: " + text);
    }
}

bool transcript_tag(const std::string & text) {
    // These are the checkpoint's own transcript tags and are valid in a guide
    // such as "Put speaker tags <spk:0> <spk:1> on the transcription."
    static const std::regex pattern(R"(<spk:\d+>|<ev:[a-z_]+>|<sty:(?:stt|end):[a-z_]+>)");
    return std::regex_match(text, pattern);
}

}  // namespace

struct TextTokenizer::Impl {
    Impl(const std::filesystem::path & tokenizer_json, const std::filesystem::path & tokenizer_config)
        : bpe(make_spec(tokenizer_json, tokenizer_config)) {
        require_id(bpe, "<|im_start|>", 10);
        require_id(bpe, "<|im_end|>", TextTokenizer::end_of_turn_id);
        require_id(bpe, "<think>", 12);
        require_id(bpe, "</think>", 13);
        require_id(bpe, "<|audio|>", TextTokenizer::audio_locator_id);
        // Use checkpoint-declared specials for timestamp transcript cleanup.
        // Numeric order makes the result deterministic across JSON parsers.
        std::map<int64_t, std::string> specials;
        const auto metadata = io::json::parse_file(tokenizer_json);
        if (const auto * added = metadata.find("added_tokens")) {
            for (const auto & token : added->as_array()) {
                if (io::json::optional_bool(token, "special", false)) {
                    specials[token.require("id").as_i64()] = token.require("content").as_string();
                }
            }
        }
        const auto configuration = io::json::parse_file(tokenizer_config);
        if (const auto * added = configuration.find("added_tokens_decoder")) {
            for (const auto & [id, token] : added->as_object()) {
                if (io::json::optional_bool(token, "special", false)) {
                    specials[std::stoll(id)] = token.require("content").as_string();
                }
            }
        }
        for (auto & entry : specials) {
            special_ids.insert(static_cast<int32_t>(entry.first));
            special_tokens.push_back(std::move(entry.second));
        }
    }

    std::string special_token_in(const std::string & text) const {
        for (const auto token : bpe.encode(text, true)) {
            if (!bpe.is_special_token_id(token)) continue;
            const auto spelling = bpe.decode({token}, false);
            if (!transcript_tag(spelling)) return spelling;
        }
        return {};
    }

    tokenizers::LlamaBpeTokenizer bpe;
    std::vector<std::string> special_tokens;
    std::unordered_set<int32_t> special_ids;
};

TextTokenizer::TextTokenizer(const std::filesystem::path & tokenizer_json_path,
    const std::filesystem::path & tokenizer_config_path)
    : impl_(std::make_shared<Impl>(tokenizer_json_path, tokenizer_config_path)) {}

std::vector<int32_t> TextTokenizer::encode(const std::string & text, bool parse_special) const {
    return impl_->bpe.encode(text, parse_special);
}

std::string TextTokenizer::decode(const std::vector<int32_t> & ids, bool skip_special_tokens) const {
    if (!skip_special_tokens) return impl_->bpe.decode(ids, false);
    std::vector<int32_t> kept;
    kept.reserve(ids.size());
    for (auto id : ids) if (impl_->special_ids.count(id) == 0) kept.push_back(id);
    return impl_->bpe.decode(kept, false);
}

PromptTokens TextTokenizer::build_prompt(const std::string & instruction, const std::string & system_prompt) const {
    for (const auto * text : {&instruction, &system_prompt}) {
        const auto token = special_token_in(*text);
        if (!token.empty()) {
            throw std::runtime_error("Nemotron Transcribe prompt text contains a template control or audio locator: " + token);
        }
    }
    PromptTokens out;
    out.before_audio = encode("<|im_start|>system\n" + system_prompt +
        "<|im_end|>\n<|im_start|>user\n" + instruction + " ", true);
    out.after_audio = encode("<|im_end|>\n<|im_start|>assistant\n<think></think>", true);
    for (const auto * ids : {&out.before_audio, &out.after_audio}) {
        if (std::any_of(ids->begin(), ids->end(), [](int32_t id) { return id < 0 || id >= embedding_rows; })) {
            throw std::runtime_error("Nemotron Transcribe prompt contains a token outside the embedding vocabulary");
        }
    }
    return out;
}

std::string TextTokenizer::special_token_in(const std::string & text) const {
    return impl_->special_token_in(text);
}

const std::vector<std::string> & TextTokenizer::alignment_special_tokens() const noexcept {
    return impl_->special_tokens;
}

}  // namespace engine::models::nemotron_transcribe
