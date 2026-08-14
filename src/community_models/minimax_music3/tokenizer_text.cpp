#include "engine/community_models/minimax_music3/tokenizer_text.h"

#include "engine/community_models/minimax_music3/types.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace engine::models::minimax_music3 {
namespace {

std::vector<std::string> split_lines(const std::string & text) {
    std::vector<std::string> lines;
    std::string current;
    for (const char ch : text) {
        if (ch == '\n') {
            lines.push_back(current);
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    lines.push_back(current);
    return lines;
}

std::string join_lines(const std::vector<std::string> & lines) {
    std::string out;
    for (size_t index = 0; index < lines.size(); ++index) {
        if (index > 0) {
            out.push_back('\n');
        }
        out += lines[index];
    }
    return out;
}

std::string rstrip(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())) != 0) {
        value.pop_back();
    }
    return value;
}

std::string strip(std::string value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])) != 0) {
        ++start;
    }
    value.erase(0, start);
    return rstrip(std::move(value));
}

std::string replace_all(std::string text, const std::string & from, const std::string & to) {
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
    return text;
}

// Removes single-star emphasis `*text*` (with no adjacent stars), mirroring the reference
// `(?<!\*)\*([^*\n]+)\*(?!\*)` which std::regex cannot express (no lookbehind).
std::string remove_single_star_emphasis(const std::string & line) {
    std::string out;
    size_t index = 0;
    while (index < line.size()) {
        if (line[index] != '*' || (index > 0 && line[index - 1] == '*') ||
            (index + 1 < line.size() && line[index + 1] == '*')) {
            out.push_back(line[index]);
            ++index;
            continue;
        }
        const size_t close = line.find('*', index + 1);
        if (close == std::string::npos || close == index + 1 ||
            (close + 1 < line.size() && line[close + 1] == '*')) {
            out.push_back(line[index]);
            ++index;
            continue;
        }
        out += line.substr(index + 1, close - index - 1);
        index = close + 1;
    }
    return out;
}

}  // namespace

std::string MiniMaxMusic3TextTokenizer::clean_caption(const std::string & caption) {
    // Rewrite `<|key value|>` special tags to "key is value".
    static const std::regex special_tag("<\\|([^|]*)\\|>");
    std::string text;
    {
        std::sregex_iterator iter(caption.begin(), caption.end(), special_tag);
        const std::sregex_iterator end;
        size_t last = 0;
        for (; iter != end; ++iter) {
            text += caption.substr(last, static_cast<size_t>(iter->position()) - last);
            const std::string inner = strip((*iter)[1].str());
            const size_t space = inner.find_first_of(" \t\n\r\f\v");
            if (space != std::string::npos) {
                std::string key = inner.substr(0, space);
                size_t rest = inner.find_first_not_of(" \t\n\r\f\v", space);
                text += key + " is " + (rest == std::string::npos ? "" : inner.substr(rest));
            } else {
                text += inner;
            }
            last = static_cast<size_t>(iter->position() + iter->length());
        }
        text += caption.substr(last);
    }

    static const std::regex heading("^\\s{0,3}#{1,6}\\s+");
    static const std::regex bullet("^\\s*[*+-]\\s+");
    static const std::regex star_bullet("^\\s*\\*\\s+");
    static const std::regex bold("\\*\\*([^*]+)\\*\\*");
    auto lines = split_lines(text);
    for (auto & line : lines) {
        line = std::regex_replace(line, heading, "");
        line = std::regex_replace(line, bullet, "");
        line = std::regex_replace(line, star_bullet, "");
        while (line.find("**") != std::string::npos) {
            const std::string updated = std::regex_replace(line, bold, "$1");
            if (updated == line) {
                break;
            }
            line = updated;
        }
        line = remove_single_star_emphasis(line);
        line = rstrip(std::move(line));
    }
    text = join_lines(lines);

    static const std::regex horizontal_rule("^\\s*[-*_]{3,}\\s*$");
    lines = split_lines(text);
    for (auto & line : lines) {
        if (std::regex_match(line, horizontal_rule)) {
            line.clear();
        }
    }
    text = join_lines(lines);
    text = replace_all(std::move(text), "\xE2\x80\xA2 ", "");  // "• "
    text = replace_all(std::move(text), "    ", "");
    static const std::regex blank_lines("\n{2,}");
    return std::regex_replace(text, blank_lines, "\n");
}

std::string MiniMaxMusic3TextTokenizer::normalize_lyrics(const std::string & lyrics) {
    // Keep only consecutive structural tags at the start of a line; text on a tag line drops.
    static const std::regex leading_tags("^[ \t]*((?:\\[[^\\]]+\\][ \t]*)+)");
    auto lines = split_lines(lyrics);
    for (auto & line : lines) {
        std::smatch match;
        if (std::regex_search(line, match, leading_tags)) {
            line = strip(match[1].str());
        }
    }
    std::string text = join_lines(lines);
    text = replace_all(std::move(text), "] ", "]\n");
    text = replace_all(std::move(text), " [", "\n[");
    text = replace_all(std::move(text), " ^ ", "\n");

    // Lowercase tag contents.
    std::string out;
    out.reserve(text.size());
    bool in_tag = false;
    for (const char ch : text) {
        if (ch == '[') {
            in_tag = true;
            out.push_back(ch);
        } else if (ch == ']') {
            in_tag = false;
            out.push_back(ch);
        } else {
            out.push_back(in_tag ? static_cast<char>(std::tolower(static_cast<unsigned char>(ch))) : ch);
        }
    }
    return "[start]\n" + out;
}

struct MiniMaxMusic3TextTokenizer::Impl {
    std::shared_ptr<tokenizers::LlamaBpeTokenizer> tokenizer;
};

MiniMaxMusic3TextTokenizer::MiniMaxMusic3TextTokenizer(const assets::ResourceBundle & resources)
    : impl_(std::make_unique<Impl>()) {
    tokenizers::LlamaBpeTokenizerSpec spec;
    spec.tokenizer_json_path = resources.require_file("tokenizer_json");
    spec.tokenizer_config_path = resources.require_file("tokenizer_config");
    spec.pre_type = tokenizers::LlamaBpePreTokenizer::Qwen2;
    impl_->tokenizer = tokenizers::load_llama_bpe_tokenizer(spec);
}

MiniMaxMusic3TextTokenizer::~MiniMaxMusic3TextTokenizer() = default;

MiniMaxMusic3TextTokenizer::PromptIds MiniMaxMusic3TextTokenizer::encode_prompt(
    const std::string & caption,
    const std::string & lyrics) const {
    if (strip(caption).empty()) {
        throw std::runtime_error("MiniMax-Music3 caption must be a non-empty string");
    }
    if (strip(lyrics).empty()) {
        throw std::runtime_error("MiniMax-Music3 lyrics must be a non-empty string");
    }
    const std::string text =
        "<|im_start|><|caption_start|>" + clean_caption(caption) + "<|caption_end|>" +
        "<|lyrics_start|>" + normalize_lyrics(lyrics) + "<|lyrics_end|><|im_end|><|audio_start|>";
    PromptIds out;
    out.cond_ids = impl_->tokenizer->encode(text, /*parse_special=*/true);
    if (out.cond_ids.size() < 4) {
        throw std::runtime_error("MiniMax-Music3 assembled prompt is unexpectedly short");
    }
    if (static_cast<int64_t>(out.cond_ids.size()) > MiniMaxMusic3Contract::kMaxPromptTokens) {
        throw std::runtime_error(
            "MiniMax-Music3 assembled prompt has " + std::to_string(out.cond_ids.size()) +
            " tokens; the maximum is " + std::to_string(MiniMaxMusic3Contract::kMaxPromptTokens));
    }
    out.uncond_ids = out.cond_ids;
    for (size_t index = 1; index + 2 < out.uncond_ids.size(); ++index) {
        out.uncond_ids[index] = MiniMaxMusic3Contract::kAudioCfgTokenId;
    }
    return out;
}

}  // namespace engine::models::minimax_music3
