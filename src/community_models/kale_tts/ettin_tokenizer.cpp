#include "engine/community_models/kale_tts/ettin_tokenizer.h"

#include "engine/framework/io/json.h"

#include <utf8proc.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace engine::models::kale_tts {
namespace {

std::u32string decode(const std::string & text) {
    std::u32string out;
    size_t at = 0;
    while (at < text.size()) {
        utf8proc_int32_t c = 0;
        const auto used = utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t *>(text.data() + at),
                                           static_cast<utf8proc_ssize_t>(text.size() - at), &c);
        if (used <= 0) {
            throw std::runtime_error("kale-tts: the text is not valid UTF-8");
        }
        out.push_back(static_cast<char32_t>(c));
        at += static_cast<size_t>(used);
    }
    return out;
}

std::string to_utf8(char32_t c) {
    utf8proc_uint8_t buffer[4];
    const auto n = utf8proc_encode_char(static_cast<utf8proc_int32_t>(c), buffer);
    return std::string(reinterpret_cast<const char *>(buffer), static_cast<size_t>(n));
}

std::string to_utf8(const std::u32string & text) {
    std::string out;
    for (const char32_t c : text) {
        out += to_utf8(c);
    }
    return out;
}

std::string nfc(const std::string & text) {
    utf8proc_uint8_t * mapped = utf8proc_NFC(reinterpret_cast<const utf8proc_uint8_t *>(text.c_str()));
    if (mapped == nullptr) {
        throw std::runtime_error("kale-tts: the text is not valid UTF-8");
    }
    std::string out(reinterpret_cast<const char *>(mapped));
    std::free(mapped);
    return out;
}

bool is_letter(char32_t c) {
    const auto k = utf8proc_category(static_cast<utf8proc_int32_t>(c));
    return k == UTF8PROC_CATEGORY_LU || k == UTF8PROC_CATEGORY_LL || k == UTF8PROC_CATEGORY_LT || k == UTF8PROC_CATEGORY_LM ||
           k == UTF8PROC_CATEGORY_LO;
}

bool is_number(char32_t c) {
    const auto k = utf8proc_category(static_cast<utf8proc_int32_t>(c));
    return k == UTF8PROC_CATEGORY_ND || k == UTF8PROC_CATEGORY_NL || k == UTF8PROC_CATEGORY_NO;
}

// JavaScript's \s
bool is_space(char32_t c) {
    if (c == U'\t' || c == U'\n' || c == 0x0B || c == 0x0C || c == U'\r' || c == U' ' || c == 0xA0 || c == 0xFEFF || c == 0x2028 ||
        c == 0x2029) {
        return true;
    }
    return utf8proc_category(static_cast<utf8proc_int32_t>(c)) == UTF8PROC_CATEGORY_ZS;
}

bool is_other(char32_t c) {
    return !is_space(c) && !is_letter(c) && !is_number(c);
}

// The GPT-2 pre-tokenizer: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
std::vector<std::pair<size_t, size_t>> gpt2_split(const std::u32string & s) {
    std::vector<std::pair<size_t, size_t>> out;
    size_t i = 0;
    const size_t n = s.size();
    auto run = [&](size_t from, bool (*keep)(char32_t)) {
        size_t j = from;
        while (j < n && keep(s[j])) {
            ++j;
        }
        return j;
    };
    while (i < n) {
        if (s[i] == U'\'') {
            static const std::u32string endings[] = {U"s", U"t", U"re", U"ve", U"m", U"ll", U"d"};
            bool found = false;
            for (const auto & e : endings) {
                if (s.compare(i + 1, e.size(), e) == 0) {
                    out.emplace_back(i, i + 1 + e.size());
                    i += 1 + e.size();
                    found = true;
                    break;
                }
            }
            if (found) {
                continue;
            }
        }
        const size_t start = i;
        const size_t body = (s[i] == U' ' && i + 1 < n) ? i + 1 : i;
        if (is_letter(s[body])) {
            i = run(body, is_letter);
        } else if (is_number(s[body])) {
            i = run(body, is_number);
        } else if (is_other(s[body])) {
            i = run(body, is_other);
        } else {
            const size_t end = run(i, is_space);
            i = (end < n && end - i > 1) ? end - 1 : end;
        }
        out.emplace_back(start, i);
    }
    return out;
}

const std::vector<std::string> & byte_symbols() {
    static const std::vector<std::string> table = [] {
        std::vector<int> bs;
        for (int b = 33; b <= 126; ++b) bs.push_back(b);
        for (int b = 161; b <= 172; ++b) bs.push_back(b);
        for (int b = 174; b <= 255; ++b) bs.push_back(b);
        std::vector<int> cs = bs;
        int extra = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + extra++);
            }
        }
        std::vector<std::string> out(256);
        for (size_t k = 0; k < bs.size(); ++k) {
            out[static_cast<size_t>(bs[k])] = to_utf8(static_cast<char32_t>(cs[k]));
        }
        return out;
    }();
    return table;
}

}  // namespace

EttinTokenizer::EttinTokenizer(const std::string & tokenizer_json) {
    const auto root = engine::io::json::parse(tokenizer_json);
    const auto & model = root.require("model");
    for (const auto & [token, id] : model.require("vocab").as_object()) {
        vocab_.emplace(token, static_cast<int32_t>(id.as_i64()));
    }
    int32_t rank = 0;
    for (const auto & merge : model.require("merges").as_array()) {
        std::string key;
        if (merge.is_string()) {
            key = merge.as_string();
        } else {
            key = merge.as_array().at(0).as_string() + " " + merge.as_array().at(1).as_string();
        }
        ranks_.emplace(key, rank++);
    }
    if (const auto * added = root.find("added_tokens"); added != nullptr) {
        for (const auto & token : added->as_array()) {
            const std::string & content = token.require("content").as_string();
            const auto id = static_cast<int32_t>(token.require("id").as_i64());
            if (content == "[CLS]") cls_ = id;
            if (content == "[SEP]") sep_ = id;
            if (!token.require("special").as_bool()) {
                added_.emplace_back(decode(content), id);
            }
        }
    }
    if (cls_ == 0 && vocab_.count("[CLS]")) cls_ = vocab_.at("[CLS]");
    if (sep_ == 0 && vocab_.count("[SEP]")) sep_ = vocab_.at("[SEP]");
    std::stable_sort(added_.begin(), added_.end(), [](const auto & a, const auto & b) { return a.first.size() > b.first.size(); });
}

std::vector<std::string> EttinTokenizer::bpe(const std::string & word) const {
    std::vector<std::string> parts;
    for (const char32_t c : decode(word)) {
        parts.push_back(to_utf8(c));
    }
    while (parts.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t best = parts.size();
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            const auto found = ranks_.find(parts[i] + " " + parts[i + 1]);
            if (found != ranks_.end() && found->second < best_rank) {
                best_rank = found->second;
                best = i;
            }
        }
        if (best == parts.size()) {
            break;
        }
        parts[best] += parts[best + 1];
        parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(best + 1));
    }
    return parts;
}

EttinTokenizer::Encoded EttinTokenizer::encode(const std::string & text) const {
    const std::u32string t = decode(nfc(text));
    Encoded out;
    out.ids.push_back(cls_);
    out.offsets.emplace_back(0, 0);
    const auto & bytes = byte_symbols();
    auto plain = [&](size_t from, size_t to) {
        const std::u32string segment = t.substr(from, to - from);
        for (const auto & [a, b] : gpt2_split(segment)) {
            std::string word;
            std::vector<size_t> byte_point;             // code point (within the piece) of every byte
            for (size_t k = a; k < b; ++k) {
                const std::string utf8 = to_utf8(segment[k]);
                for (const unsigned char byte : utf8) {
                    word += bytes[byte];
                    byte_point.push_back(k - a);
                }
            }
            size_t at = 0;
            for (const std::string & sub : bpe(word)) {
                const size_t count = decode(sub).size();
                const auto found = vocab_.find(sub);
                if (found != vocab_.end()) {
                    const auto start = static_cast<int64_t>(from + a + byte_point[at]);
                    const auto end = static_cast<int64_t>(from + a + byte_point[at + count - 1] + 1);
                    out.ids.push_back(found->second);
                    out.offsets.emplace_back(start, end);
                }
                at += count;
            }
        }
    };
    size_t last = 0;
    size_t i = 0;
    while (i < t.size()) {
        bool matched = false;
        for (const auto & [content, id] : added_) {
            if (!content.empty() && t.compare(i, content.size(), content) == 0) {
                if (i > last) plain(last, i);
                out.ids.push_back(id);
                out.offsets.emplace_back(static_cast<int64_t>(i), static_cast<int64_t>(i + content.size()));
                i += content.size();
                last = i;
                matched = true;
                break;
            }
        }
        if (!matched) ++i;
    }
    if (last < t.size()) plain(last, t.size());
    out.ids.push_back(sep_);
    out.offsets.emplace_back(0, 0);
    return out;
}

namespace {

std::vector<std::u32string> split_words(const std::u32string & text) {
    std::vector<std::u32string> words;
    std::u32string current;
    for (const char32_t c : text) {
        if (is_space(c)) {
            if (!current.empty()) words.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) words.push_back(current);
    return words;
}

std::u32string trim(const std::u32string & text) {
    size_t a = 0;
    size_t b = text.size();
    while (a < b && is_space(text[a])) ++a;
    while (b > a && is_space(text[b - 1])) --b;
    return text.substr(a, b - a);
}

}  // namespace

EttinInputs ettin_inputs(const EttinTokenizer & tokenizer, const std::string & text, const std::string & previous) {
    const auto words = split_words(decode(text));
    const std::u32string before = trim(decode(previous));
    const std::u32string prefix = before.empty() ? U"" : before + U" ";
    std::u32string full = prefix;
    for (size_t w = 0; w < words.size(); ++w) {
        full += (w ? U" " : U"") + words[w];
    }
    const auto encoded = tokenizer.encode(to_utf8(full));
    EttinInputs out;
    out.ids = encoded.ids;
    out.tokens = static_cast<int64_t>(encoded.ids.size());
    out.words = std::max<int64_t>(static_cast<int64_t>(words.size()), 1);
    out.wavg.assign(static_cast<size_t>(out.words * out.tokens), 0.0F);
    auto pos = static_cast<int64_t>(prefix.size());
    for (size_t w = 0; w < words.size(); ++w) {
        const int64_t a = pos;
        const int64_t b = pos + static_cast<int64_t>(words[w].size());
        pos = b + 1;
        std::vector<int64_t> hits;
        for (int64_t k = 0; k < out.tokens; ++k) {
            const auto [s, e] = encoded.offsets[static_cast<size_t>(k)];
            if (e > s && s < b && e > a) hits.push_back(k);
        }
        for (const int64_t k : hits) {
            out.wavg[static_cast<size_t>(static_cast<int64_t>(w) * out.tokens + k)] = 1.0F / static_cast<float>(std::max<size_t>(1, hits.size()));
        }
    }
    return out;
}

std::vector<int32_t> word_index(const std::vector<int32_t> & sentence_ids, const std::vector<std::vector<int32_t>> & word_ids,
                                int64_t words, int32_t space_id) {
    std::vector<int32_t> joined;
    for (size_t w = 0; w < word_ids.size(); ++w) {
        joined.insert(joined.end(), word_ids[w].begin(), word_ids[w].end());
        if (w + 1 < word_ids.size()) joined.push_back(space_id);
    }
    std::vector<int32_t> out;
    if (joined == sentence_ids) {
        for (size_t w = 0; w < word_ids.size(); ++w) {
            out.insert(out.end(), word_ids[w].size(), static_cast<int32_t>(w));
            if (w + 1 < word_ids.size()) out.push_back(static_cast<int32_t>(w));
        }
        return out;
    }
    const int64_t n = std::max<int64_t>(words, 1);
    int64_t spaces = 1;
    for (const int32_t x : sentence_ids) spaces += x == space_id ? 1 : 0;
    int64_t k = 0;
    for (const int32_t x : sentence_ids) {
        out.push_back(static_cast<int32_t>(std::min<int64_t>(n - 1, (k * words) / spaces)));
        if (x == space_id) ++k;
    }
    return out;
}

}  // namespace engine::models::kale_tts
