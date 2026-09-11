#include "engine/community_models/minimax_music3/prompt.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <stdexcept>
#include <utility>

namespace engine::models::minimax_music3 {
namespace {

constexpr const char * kImStart = "<|im_start|>";
constexpr const char * kImEnd = "<|im_end|>";
constexpr const char * kCaptionStart = "<|caption_start|>";
constexpr const char * kCaptionEnd = "<|caption_end|>";
constexpr const char * kLyricsStart = "<|lyrics_start|>";
constexpr const char * kLyricsEnd = "<|lyrics_end|>";
constexpr const char * kAudioStart = "<|audio_start|>";
constexpr int32_t kAudioCfgTokenId = 151654;

std::shared_ptr<const MiniMaxMusic3Assets> require_assets(
    std::shared_ptr<const MiniMaxMusic3Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("MiniMax Music 3 prompt builder requires assets");
    }
    return assets;
}

std::string strip_markdown_line(std::string line) {
    line = std::regex_replace(line, std::regex(R"(^\s{0,3}#{1,6}\s+)"), "");
    line = std::regex_replace(line, std::regex(R"(^\s*[*+-]\s+)"), "");
    line = std::regex_replace(line, std::regex(R"(^\s*\*\s+)"), "");
    for (;;) {
        auto updated = std::regex_replace(line, std::regex(R"(\*\*([^*]+)\*\*)"), "$1");
        if (updated == line) {
            break;
        }
        line = std::move(updated);
    }
    line = std::regex_replace(line, std::regex(R"(\*([^*\n]+)\*)"), "$1");
    while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) {
        line.pop_back();
    }
    return line;
}

std::string clean_caption(const std::string & caption) {
    const std::regex special_tag_re(R"(<\|([^|]*)\|>)");
    std::string text;
    std::sregex_iterator it(caption.begin(), caption.end(), special_tag_re);
    std::sregex_iterator end;
    size_t cursor = 0;
    for (; it != end; ++it) {
        const auto & match = *it;
        text.append(caption, cursor, static_cast<size_t>(match.position()) - cursor);
        std::string inner = match.str(1);
        while (!inner.empty() && std::isspace(static_cast<unsigned char>(inner.front()))) {
            inner.erase(inner.begin());
        }
        while (!inner.empty() && std::isspace(static_cast<unsigned char>(inner.back()))) {
            inner.pop_back();
        }
        const auto split = inner.find_first_of(" \t\n\r\f\v");
        if (split == std::string::npos) {
            text += inner;
        } else {
            size_t rest = split;
            while (rest < inner.size() && std::isspace(static_cast<unsigned char>(inner[rest]))) {
                ++rest;
            }
            text += inner.substr(0, split) + " is " + inner.substr(rest);
        }
        cursor = static_cast<size_t>(match.position() + match.length());
    }
    text.append(caption, cursor, std::string::npos);
    std::string out;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t end = text.find('\n', start);
        auto line = strip_markdown_line(text.substr(start, end == std::string::npos ? std::string::npos : end - start));
        if (!std::regex_match(line, std::regex(R"(^\s*[-*_]{3,}\s*$)"))) {
            if (!out.empty()) {
                out.push_back('\n');
            }
            out += line;
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    out = std::regex_replace(out, std::regex(u8"• "), "");
    out = std::regex_replace(out, std::regex("    "), "");
    return std::regex_replace(out, std::regex(R"(\n{2,})"), "\n");
}

std::string normalize_lyrics(const std::string & lyrics) {
    std::string out;
    size_t start = 0;
    const std::regex leading_tags(R"(^[ \t]*((?:\[[^\]]+\][ \t]*)+))");
    while (start <= lyrics.size()) {
        const size_t end = lyrics.find('\n', start);
        std::string line = lyrics.substr(start, end == std::string::npos ? std::string::npos : end - start);
        std::smatch match;
        if (std::regex_search(line, match, leading_tags)) {
            line = match.str(1);
            while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) {
                line.pop_back();
            }
        }
        if (!out.empty()) {
            out.push_back('\n');
        }
        out += line;
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    out = std::regex_replace(out, std::regex(R"(\] )"), "]\n");
    out = std::regex_replace(out, std::regex(R"( \[)"), "\n[");
    out = std::regex_replace(out, std::regex(R"( \^ )"), "\n");
    std::string lowered;
    lowered.reserve(out.size());
    for (size_t i = 0; i < out.size();) {
        if (out[i] == '[') {
            const size_t close = out.find(']', i + 1);
            if (close != std::string::npos) {
                lowered.push_back('[');
                for (size_t j = i + 1; j < close; ++j) {
                    lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(out[j]))));
                }
                lowered.push_back(']');
                i = close + 1;
                continue;
            }
        }
        lowered.push_back(out[i++]);
    }
    return "[start]\n" + lowered;
}

}  // namespace

struct MiniMaxMusic3PromptBuilder::Impl {
    explicit Impl(std::shared_ptr<const MiniMaxMusic3Assets> input_assets)
        : assets(require_assets(std::move(input_assets))) {
        engine::tokenizers::LlamaBpeTokenizerSpec spec;
        spec.tokenizer_json_path = assets->tokenizer_json_path;
        spec.tokenizer_config_path = assets->tokenizer_config_path;
        spec.pre_type = engine::tokenizers::LlamaBpePreTokenizer::Qwen2;
        tokenizer = engine::tokenizers::load_llama_bpe_tokenizer(spec);
    }

    std::shared_ptr<const MiniMaxMusic3Assets> assets;
    std::shared_ptr<engine::tokenizers::LlamaBpeTokenizer> tokenizer;
};

MiniMaxMusic3PromptBuilder::MiniMaxMusic3PromptBuilder(std::shared_ptr<const MiniMaxMusic3Assets> assets)
    : impl_(std::make_shared<Impl>(std::move(assets))) {}

MiniMaxMusic3Prompt MiniMaxMusic3PromptBuilder::build(
    const std::string & prompt,
    const std::string & lyrics) const {
    if (prompt.empty()) {
        throw std::runtime_error("MiniMax Music 3 requires non-empty prompt");
    }
    if (lyrics.empty()) {
        throw std::runtime_error("MiniMax Music 3 requires non-empty lyrics");
    }
    const std::string text =
        std::string(kImStart) + kCaptionStart + clean_caption(prompt) + kCaptionEnd +
        kLyricsStart + normalize_lyrics(lyrics) + kLyricsEnd + kImEnd + kAudioStart;
    MiniMaxMusic3Prompt out;
    out.conditional_ids = impl_->tokenizer->encode(text, true);
    if (static_cast<int64_t>(out.conditional_ids.size()) > impl_->assets->config.max_prompt_tokens) {
        throw std::runtime_error("MiniMax Music 3 prompt exceeds max_prompt_tokens");
    }
    out.unconditional_ids = out.conditional_ids;
    if (out.unconditional_ids.size() >= 3) {
        std::fill(out.unconditional_ids.begin() + 1, out.unconditional_ids.end() - 2, kAudioCfgTokenId);
    }
    return out;
}

}  // namespace engine::models::minimax_music3
