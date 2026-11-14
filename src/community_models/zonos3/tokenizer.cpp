#include "engine/community_models/zonos3/tokenizer.h"

#include "engine/framework/io/json.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include "unicode.h"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace engine::models::zonos3 {
namespace {

struct CodePoint {
    uint32_t cp = 0;
    size_t begin = 0;
    size_t end = 0;
};

// ZONOS3's utf8_next: a malformed sequence decodes as U+FFFD covering the bytes consumed so far.
uint32_t utf8_next(const std::string & s, size_t & i) {
    const auto c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
        ++i;
        return c;
    }
    int len = 0;
    unsigned char lo = 0x80;
    unsigned char hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) len = 2;
    else if (c == 0xE0) len = 3, lo = 0xA0;
    else if (c == 0xED) len = 3, hi = 0x9F;
    else if (c >= 0xE1 && c <= 0xEF) len = 3;
    else if (c == 0xF0) len = 4, lo = 0x90;
    else if (c == 0xF4) len = 4, hi = 0x8F;
    else if (c >= 0xF1 && c <= 0xF3) len = 4;
    if (len == 0) {
        ++i;
        return 0xFFFD;
    }
    uint32_t cp = c & (0x7F >> len);
    for (int k = 1; k < len; ++k) {
        if (i + static_cast<size_t>(k) >= s.size()) {
            i += static_cast<size_t>(k);
            return 0xFFFD;
        }
        const auto d = static_cast<unsigned char>(s[i + static_cast<size_t>(k)]);
        if (d < (k == 1 ? lo : 0x80) || d > (k == 1 ? hi : 0xBF)) {
            i += static_cast<size_t>(k);
            return 0xFFFD;
        }
        cp = (cp << 6) | (d & 0x3F);
    }
    i += static_cast<size_t>(len);
    return cp;
}

bool is_letter(uint32_t cp) { return unicode_cpt_flags_from_cpt(cp).is_letter; }
bool is_number(uint32_t cp) { return unicode_cpt_flags_from_cpt(cp).is_number; }

// Unicode White_Space (Oniguruma \s), which unlike Python str.isspace() excludes U+001C..U+001F.
bool is_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
        (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
        cp == 0x3000;
}

bool is_other(uint32_t cp) { return !is_space(cp) && !is_letter(cp) && !is_number(cp); }

// GPT-2 bytes_to_unicode: printable bytes keep their code point, the rest map in order to U+0100 + k.
const std::vector<std::string> & byte_symbols() {
    static const std::vector<std::string> table = [] {
        std::vector<std::string> out(256);
        uint32_t extra = 0;
        for (int b = 0; b < 256; ++b) {
            const bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            out[static_cast<size_t>(b)] = unicode_cpt_to_utf8(keep ? static_cast<uint32_t>(b) : 256u + extra++);
        }
        return out;
    }();
    return table;
}

std::string byte_encode(const std::string & text) {
    const auto & symbols = byte_symbols();
    std::string out;
    out.reserve(text.size() * 2);
    for (const unsigned char byte : text) out += symbols[byte];
    return out;
}

std::vector<std::string> byte_encoded_split(const std::string & text) {
    auto pieces = Zonos3Tokenizer::split(text);
    for (auto & piece : pieces) piece = byte_encode(piece);
    return pieces;
}

tokenizers::LlamaBpeTokenizerSpec make_spec(const std::string & tokenizer_json) {
    namespace json = engine::io::json;
    const auto root = json::parse(tokenizer_json);
    const auto & model = root.require("model");
    if (model.require("type").as_string() != "BPE") throw std::runtime_error("zonos3: tokenizer model is not BPE");
    tokenizers::LlamaBpeGgufVocabulary vocabulary;
    const auto & vocab = model.require("vocab").as_object();
    vocabulary.tokens.resize(vocab.size());
    vocabulary.token_types.assign(vocab.size(), 1);
    for (const auto & [token, id_value] : vocab) {
        const auto id = id_value.as_i64();
        if (id < 0 || static_cast<size_t>(id) >= vocab.size() || !vocabulary.tokens[static_cast<size_t>(id)].empty()) {
            throw std::runtime_error("zonos3: tokenizer vocabulary ids are not 0..n-1");
        }
        vocabulary.tokens[static_cast<size_t>(id)] = token;
    }
    for (const auto & merge : model.require("merges").as_array()) {
        const auto & pair = merge.as_array();
        if (pair.size() != 2) throw std::runtime_error("zonos3: tokenizer merges must be [a,b] pairs");
        vocabulary.merges.push_back(pair[0].as_string() + " " + pair[1].as_string());
    }
    if (const auto * added = root.find("added_tokens"); added != nullptr && added->is_array()) {
        for (const auto & token : added->as_array()) {
            const auto id = token.require("id").as_i64();
            if (id < 0 || static_cast<size_t>(id) >= vocabulary.tokens.size() ||
                vocabulary.tokens[static_cast<size_t>(id)] != token.require("content").as_string()) {
                throw std::runtime_error("zonos3: tokenizer added token is not in the vocabulary");
            }
            vocabulary.token_types[static_cast<size_t>(id)] = 3;
        }
    }
    vocabulary.ignore_merges = model.find("ignore_merges") != nullptr && model.require("ignore_merges").is_bool() &&
        model.require("ignore_merges").as_bool();
    tokenizers::LlamaBpeTokenizerSpec spec;
    spec.gguf_vocabulary = std::move(vocabulary);
    spec.pre_tokenize = byte_encoded_split;
    return spec;
}

}  // namespace

Zonos3Tokenizer::Zonos3Tokenizer(const std::string & tokenizer_json)
    : bpe_(tokenizers::load_llama_bpe_tokenizer(make_spec(tokenizer_json))) {}

Zonos3Tokenizer::~Zonos3Tokenizer() = default;

std::vector<std::string> Zonos3Tokenizer::split(const std::string & utf8) {
    std::vector<CodePoint> cps;
    for (size_t i = 0; i < utf8.size();) {
        const size_t begin = i;
        const uint32_t cp = utf8_next(utf8, i);
        cps.push_back({cp, begin, i});
    }
    const size_t n = cps.size();
    // End of the regex match starting at i, or i when there is none.
    auto match_at = [&](size_t i) -> size_t {
        if (is_number(cps[i].cp)) {  // \p{N}{1,3}
            size_t j = i;
            while (j < n && j - i < 3 && is_number(cps[j].cp)) ++j;
            return j;
        }
        {  // ' '?[^\s\p{L}\p{N}]{2,}[\r\n/]*
            const size_t j = i + (cps[i].cp == 0x20 ? 1 : 0);
            size_t k = j;
            while (k < n && is_other(cps[k].cp)) ++k;
            if (k - j >= 2) {
                while (k < n && (cps[k].cp == '\r' || cps[k].cp == '\n' || cps[k].cp == '/')) ++k;
                return k;
            }
        }
        if (cps[i].cp == 0x20) {  // ' +(?!\S)'
            size_t e = i;
            while (e < n && cps[e].cp == 0x20) ++e;
            if (e == n || is_space(cps[e].cp)) return e;
            if (e - i >= 2) return e - 1;
        }
        return i;
    };
    auto piece = [&](size_t b, size_t e) { return utf8.substr(cps[b].begin, cps[e - 1].end - cps[b].begin); };
    std::vector<std::string> pieces;
    size_t gap = 0;
    for (size_t i = 0; i < n;) {
        const size_t e = match_at(i);
        if (e == i) {
            ++i;
            continue;
        }
        if (gap < i) pieces.push_back(piece(gap, i));
        pieces.push_back(piece(i, e));
        i = gap = e;
    }
    if (gap < n) pieces.push_back(piece(gap, n));
    return pieces;
}

std::vector<int32_t> Zonos3Tokenizer::text_ids(const std::string & utf8) const {
    if (utf8.find("\xEE\x80\x80") != std::string::npos || utf8.find("\xEE\x80\x81") != std::string::npos) {
        throw runtime::InvalidRequestError(
            "This checkpoint has no phoneme tokens: remove the U+E000 / U+E001 pronunciation groups");
    }
    std::vector<int32_t> out{2};
    for (const auto id : bpe_->encode(utf8, true)) out.push_back(id + 4);
    out.push_back(3);
    return out;
}

}  // namespace engine::models::zonos3
