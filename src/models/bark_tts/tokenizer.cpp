#include "engine/models/bark_tts/tokenizer.h"

#include "engine/framework/io/json.h"

#include <cctype>
#include <stdexcept>

namespace engine::models::bark_tts {
namespace {

struct Character { uint32_t codepoint; std::string bytes; };

std::vector<Character> characters(const std::string & text) {
    std::vector<Character> out;
    for (size_t pos = 0; pos < text.size();) {
        const auto first = static_cast<unsigned char>(text[pos]);
        size_t width = 1;
        uint32_t value = first;
        if ((first & 0xe0U) == 0xc0U) { width = 2; value = first & 0x1fU; }
        else if ((first & 0xf0U) == 0xe0U) { width = 3; value = first & 0x0fU; }
        else if ((first & 0xf8U) == 0xf0U) { width = 4; value = first & 0x07U; }
        if (pos + width > text.size()) throw std::runtime_error("Bark text contains truncated UTF-8");
        for (size_t i = 1; i < width; ++i) {
            const auto next = static_cast<unsigned char>(text[pos + i]);
            if ((next & 0xc0U) != 0x80U) throw std::runtime_error("Bark text contains invalid UTF-8");
            value = (value << 6U) | (next & 0x3fU);
        }
        out.push_back({value, text.substr(pos, width)});
        pos += width;
    }
    return out;
}

bool is_cjk(uint32_t cp) {
    return (cp >= 0x3400 && cp <= 0x4dbf) || (cp >= 0x4e00 && cp <= 0x9fff) ||
           (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x20000 && cp <= 0x2fa1f);
}

bool is_punctuation(uint32_t cp) {
    if (cp < 128) return std::ispunct(static_cast<unsigned char>(cp)) != 0;
    return (cp >= 0x2000 && cp <= 0x206f) || (cp >= 0x3000 && cp <= 0x303f);
}

}  // namespace

BarkTokenizer::BarkTokenizer(const std::string & tokenizer_json) {
    const auto root = engine::io::json::parse(tokenizer_json);
    const auto & values = root.require("model").require("vocab").as_object();
    vocab_.reserve(values.size());
    for (const auto & [token, id] : values) vocab_.emplace(token, static_cast<int32_t>(id.as_i64()));
    if (const auto it = vocab_.find("[UNK]"); it != vocab_.end()) unknown_ = it->second;
}

std::vector<int32_t> BarkTokenizer::encode(const std::string & text) const {
    std::vector<std::string> words;
    std::string current;
    for (const auto & ch : characters(text)) {
        if (ch.codepoint <= 0x20 || ch.codepoint == 0x7f) {
            if (!current.empty()) { words.push_back(std::move(current)); current.clear(); }
        } else if (is_cjk(ch.codepoint) || is_punctuation(ch.codepoint)) {
            if (!current.empty()) { words.push_back(std::move(current)); current.clear(); }
            words.push_back(ch.bytes);
        } else {
            current += ch.bytes;
        }
    }
    if (!current.empty()) words.push_back(std::move(current));

    std::vector<int32_t> ids{101};
    for (const auto & word : words) {
        const auto chars = characters(word);
        if (chars.size() > 100) { ids.push_back(unknown_); continue; }
        size_t start = 0;
        std::vector<int32_t> pieces;
        while (start < chars.size()) {
            size_t end = chars.size();
            bool found = false;
            while (end > start) {
                std::string piece = start == 0 ? "" : "##";
                for (size_t i = start; i < end; ++i) piece += chars[i].bytes;
                if (const auto it = vocab_.find(piece); it != vocab_.end()) {
                    pieces.push_back(it->second); start = end; found = true; break;
                }
                --end;
            }
            if (!found) { pieces.assign(1, unknown_); break; }
        }
        ids.insert(ids.end(), pieces.begin(), pieces.end());
    }
    ids.push_back(102);
    return ids;
}

}  // namespace engine::models::bark_tts
