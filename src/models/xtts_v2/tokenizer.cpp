#include "engine/models/xtts_v2/tokenizer.h"

#include "engine/framework/io/json.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace engine::models::xtts_v2 {
namespace {

struct PairHash {
    size_t operator()(const std::pair<std::string, std::string> & pair) const noexcept {
        return std::hash<std::string>{}(pair.first) ^ (std::hash<std::string>{}(pair.second) << 1U);
    }
};

std::vector<std::string> utf8_symbols(std::string_view text) {
    std::vector<std::string> out;
    for (size_t pos = 0; pos < text.size();) {
        const auto lead = static_cast<unsigned char>(text[pos]);
        size_t width = 1;
        if ((lead & 0x80U) == 0) width = 1;
        else if ((lead & 0xE0U) == 0xC0U) width = 2;
        else if ((lead & 0xF0U) == 0xE0U) width = 3;
        else if ((lead & 0xF8U) == 0xF0U) width = 4;
        else throw std::runtime_error("XTTS v2 tokenizer received invalid UTF-8");
        if (pos + width > text.size()) throw std::runtime_error("XTTS v2 tokenizer received truncated UTF-8");
        out.emplace_back(text.substr(pos, width));
        pos += width;
    }
    return out;
}

size_t utf8_width_at(std::string_view text, size_t pos) {
    if (pos >= text.size()) throw std::runtime_error("XTTS v2 tokenizer UTF-8 offset is out of range");
    const auto lead = static_cast<unsigned char>(text[pos]);
    size_t width = 0;
    if ((lead & 0x80U) == 0) width = 1;
    else if ((lead & 0xE0U) == 0xC0U) width = 2;
    else if ((lead & 0xF0U) == 0xE0U) width = 3;
    else if ((lead & 0xF8U) == 0xF0U) width = 4;
    else throw std::runtime_error("XTTS v2 tokenizer received invalid UTF-8");
    if (pos + width > text.size()) throw std::runtime_error("XTTS v2 tokenizer received truncated UTF-8");
    return width;
}

std::string clean_text(const std::string & input) {
    std::string out;
    out.reserve(input.size());
    bool space = false;
    for (unsigned char ch : input) {
        if (ch == '"') continue;
        if (std::isspace(ch) != 0) {
            space = !out.empty();
            continue;
        }
        if (space) out.push_back(' ');
        space = false;
        out.push_back(ch < 0x80U ? static_cast<char>(std::tolower(ch)) : static_cast<char>(ch));
    }
    return out;
}

}  // namespace

struct XttsV2Tokenizer::Impl {
    std::unordered_map<std::string, int32_t> vocab;
    std::unordered_map<std::pair<std::string, std::string>, int32_t, PairHash> merges;
    std::vector<std::string> specials;
    int32_t unk = 1;
    int32_t start = 261;
    int32_t stop = 0;

    std::vector<int32_t> encode_piece(const std::string & text) const {
        auto symbols = utf8_symbols(text);
        while (symbols.size() > 1) {
            int32_t rank = std::numeric_limits<int32_t>::max();
            size_t selected = symbols.size();
            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                const auto found = merges.find({symbols[i], symbols[i + 1]});
                if (found != merges.end() && found->second < rank) {
                    rank = found->second;
                    selected = i;
                }
            }
            if (selected == symbols.size()) break;
            symbols[selected] += symbols[selected + 1];
            symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(selected + 1));
        }
        std::vector<int32_t> ids;
        ids.reserve(symbols.size());
        for (const auto & symbol : symbols) {
            const auto found = vocab.find(symbol);
            ids.push_back(found == vocab.end() ? unk : found->second);
        }
        return ids;
    }

    std::vector<int32_t> encode_marked(const std::string & text) const {
        std::vector<int32_t> ids;
        std::string ordinary;
        auto flush = [&] {
            if (ordinary.empty()) return;
            auto encoded = encode_piece(ordinary);
            ids.insert(ids.end(), encoded.begin(), encoded.end());
            ordinary.clear();
        };
        for (size_t pos = 0; pos < text.size();) {
            const auto special = std::find_if(specials.begin(), specials.end(), [&](const std::string & candidate) {
                return pos + candidate.size() <= text.size() && text.compare(pos, candidate.size(), candidate) == 0;
            });
            if (special != specials.end()) {
                flush();
                ids.push_back(vocab.at(*special));
                pos += special->size();
                continue;
            }
            const size_t width = utf8_width_at(text, pos);
            ordinary.append(text, pos, width);
            pos += width;
        }
        flush();
        return ids;
    }
};

XttsV2Tokenizer::XttsV2Tokenizer(const std::filesystem::path & path) : impl_(std::make_unique<Impl>()) {
    const auto root = engine::io::json::parse_file(path);
    const auto & model = root.require("model");
    if (model.require("type").as_string() != "BPE") throw std::runtime_error("XTTS v2 expects a BPE tokenizer");
    for (const auto & [piece, value] : model.require("vocab").as_object()) impl_->vocab.emplace(piece, static_cast<int32_t>(value.as_i64()));
    int32_t rank = 0;
    for (const auto & item : model.require("merges").as_array()) {
        const auto merge = item.as_string();
        const auto split = merge.find(' ');
        if (split == std::string::npos) throw std::runtime_error("XTTS v2 tokenizer has an invalid merge");
        impl_->merges.emplace(std::make_pair(merge.substr(0, split), merge.substr(split + 1)), rank++);
    }
    for (const auto & item : root.require("added_tokens").as_array()) {
        if (const auto * content = item.find("content"); content != nullptr && content->is_string()) impl_->specials.push_back(content->as_string());
    }
    std::sort(impl_->specials.begin(), impl_->specials.end(), [](const auto & a, const auto & b) { return a.size() > b.size(); });
    impl_->unk = impl_->vocab.at("[UNK]");
    impl_->start = impl_->vocab.at("[START]");
    impl_->stop = impl_->vocab.at("[STOP]");
}

XttsV2Tokenizer::~XttsV2Tokenizer() = default;
XttsV2Tokenizer::XttsV2Tokenizer(XttsV2Tokenizer &&) noexcept = default;
XttsV2Tokenizer & XttsV2Tokenizer::operator=(XttsV2Tokenizer &&) noexcept = default;

std::vector<int32_t> XttsV2Tokenizer::encode(const std::string & text, const std::string & language) const {
    std::string language_tag = language == "zh" ? "zh-cn" : language;
    std::string prepared = "[" + language_tag + "]" + clean_text(text);
    std::string marked;
    marked.reserve(prepared.size() + 16);
    for (const char ch : prepared) marked += ch == ' ' ? "[SPACE]" : std::string(1, ch);
    return impl_->encode_marked(marked);
}

int32_t XttsV2Tokenizer::start_token() const noexcept { return impl_->start; }
int32_t XttsV2Tokenizer::stop_token() const noexcept { return impl_->stop; }

}  // namespace engine::models::xtts_v2
