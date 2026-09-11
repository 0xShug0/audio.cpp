#include "engine/models/confucius4_tts/tokenizer_text.h"

#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chinese_normalization.h"
#include "engine/framework/text/chunking.h"
#include "engine/framework/text/text_normalization.h"
#include "engine/framework/text/utf8.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::models::confucius4_tts {
namespace {

using LanguageToken = std::pair<std::string_view, std::string_view>;

constexpr LanguageToken kLanguageTokens[] = {
    {"zh", "请用中文朗读接下来的文字"},
    {"ja", "请用日语朗读接下来的文字"},
    {"ko", "请用韩语朗读接下来的文字"},
    {"vi", "请用越南语朗读接下来的文字"},
    {"th", "请用泰语朗读接下来的文字"},
    {"id", "请用印尼语朗读接下来的文字"},
    {"ms", "请用马来语朗读接下来的文字"},
    {"tl", "请用菲律宾语朗读接下来的文字"},
    {"my", "请用缅甸语朗读接下来的文字"},
    {"km", "请用高棉语朗读接下来的文字"},
    {"lo", "请用老挝语朗读接下来的文字"},
    {"hi", "请用印地语朗读接下来的文字"},
    {"bn", "请用孟加拉语朗读接下来的文字"},
    {"ta", "请用泰米尔语朗读接下来的文字"},
    {"te", "请用泰卢固语朗读接下来的文字"},
    {"mr", "请用马拉地语朗读接下来的文字"},
    {"gu", "请用古吉拉特语朗读接下来的文字"},
    {"kn", "请用卡纳达语朗读接下来的文字"},
    {"ml", "请用马拉雅拉姆语朗读接下来的文字"},
    {"pa", "请用旁遮普语朗读接下来的文字"},
    {"ur", "请用乌尔都语朗读接下来的文字"},
    {"ne", "请用尼泊尔语朗读接下来的文字"},
    {"si", "请用僧伽罗语朗读接下来的文字"},
    {"en", "请用英文朗读接下来的文字"},
    {"de", "请用德语朗读接下来的文字"},
    {"nl", "请用荷兰语朗读接下来的文字"},
    {"sv", "请用瑞典语朗读接下来的文字"},
    {"da", "请用丹麦语朗读接下来的文字"},
    {"no", "请用挪威语朗读接下来的文字"},
    {"nb", "请用挪威语朗读接下来的文字"},
    {"nn", "请用挪威语朗读接下来的文字"},
    {"is", "请用冰岛语朗读接下来的文字"},
    {"af", "请用南非荷兰语朗读接下来的文字"},
    {"lb", "请用卢森堡语朗读接下来的文字"},
    {"fy", "请用弗里斯兰语朗读接下来的文字"},
    {"fr", "请用法语朗读接下来的文字"},
    {"es", "请用西班牙语朗读接下来的文字"},
    {"pt", "请用葡萄牙语朗读接下来的文字"},
    {"it", "请用意大利语朗读接下来的文字"},
    {"ro", "请用罗马尼亚语朗读接下来的文字"},
    {"ca", "请用加泰罗尼亚语朗读接下来的文字"},
    {"gl", "请用加利西亚语朗读接下来的文字"},
    {"oc", "请用奥克语朗读接下来的文字"},
    {"la", "请用拉丁语朗读接下来的文字"},
    {"ru", "请用俄语朗读接下来的文字"},
    {"uk", "请用乌克兰语朗读接下来的文字"},
    {"pl", "请用波兰语朗读接下来的文字"},
    {"cs", "请用捷克语朗读接下来的文字"},
    {"sk", "请用斯洛伐克语朗读接下来的文字"},
    {"bg", "请用保加利亚语朗读接下来的文字"},
    {"sr", "请用塞尔维亚语朗读接下来的文字"},
    {"hr", "请用克罗地亚语朗读接下来的文字"},
    {"sl", "请用斯洛文尼亚语朗读接下来的文字"},
    {"mk", "请用马其顿语朗读接下来的文字"},
    {"bs", "请用波斯尼亚语朗读接下来的文字"},
    {"be", "请用白俄罗斯语朗读接下来的文字"},
    {"lt", "请用立陶宛语朗读接下来的文字"},
    {"lv", "请用拉脱维亚语朗读接下来的文字"},
    {"fi", "请用芬兰语朗读接下来的文字"},
    {"et", "请用爱沙尼亚语朗读接下来的文字"},
    {"hu", "请用匈牙利语朗读接下来的文字"},
    {"ga", "请用爱尔兰语朗读接下来的文字"},
    {"cy", "请用威尔士语朗读接下来的文字"},
    {"gd", "请用苏格兰盖尔语朗读接下来的文字"},
    {"br", "请用布列塔尼语朗读接下来的文字"},
    {"el", "请用希腊语朗读接下来的文字"},
    {"sq", "请用阿尔巴尼亚语朗读接下来的文字"},
    {"eu", "请用巴斯克语朗读接下来的文字"},
    {"mt", "请用马耳他语朗读接下来的文字"},
    {"tr", "请用土耳其语朗读接下来的文字"},
    {"az", "请用阿塞拜疆语朗读接下来的文字"},
    {"kk", "请用哈萨克语朗读接下来的文字"},
    {"uz", "请用乌兹别克语朗读接下来的文字"},
    {"tk", "请用土库曼语朗读接下来的文字"},
    {"ky", "请用吉尔吉斯语朗读接下来的文字"},
    {"tt", "请用鞑靼语朗读接下来的文字"},
    {"ar", "请用阿拉伯语朗读接下来的文字"},
    {"he", "请用希伯来语朗读接下来的文字"},
    {"am", "请用阿姆哈拉语朗读接下来的文字"},
    {"fa", "请用波斯语朗读接下来的文字"},
    {"ps", "请用普什图语朗读接下来的文字"},
    {"ku", "请用库尔德语朗读接下来的文字"},
    {"tg", "请用塔吉克语朗读接下来的文字"},
    {"ka", "请用格鲁吉亚语朗读接下来的文字"},
    {"hy", "请用亚美尼亚语朗读接下来的文字"},
    {"sw", "请用斯瓦希里语朗读接下来的文字"},
    {"yo", "请用约鲁巴语朗读接下来的文字"},
    {"ha", "请用豪萨语朗读接下来的文字"},
    {"ig", "请用伊博语朗读接下来的文字"},
    {"zu", "请用祖鲁语朗读接下来的文字"},
    {"xh", "请用科萨语朗读接下来的文字"},
    {"mn", "请用蒙古语朗读接下来的文字"},
    {"eo", "请用世界语朗读接下来的文字"},
};

struct CodepointSpan {
    size_t start = 0;
    size_t end = 0;
    std::string_view text;
};

std::vector<CodepointSpan> split_codepoints(std::string_view text, std::string_view label) {
    std::vector<CodepointSpan> spans;
    spans.reserve(engine::text::utf8_codepoint_count(text, label));
    for (size_t pos = 0; pos < text.size();) {
        const auto ch = static_cast<unsigned char>(text[pos]);
        size_t width = 0;
        if (ch <= 0x7FU) {
            width = 1;
        } else if ((ch & 0xE0U) == 0xC0U) {
            width = 2;
        } else if ((ch & 0xF0U) == 0xE0U) {
            width = 3;
        } else if ((ch & 0xF8U) == 0xF0U) {
            width = 4;
        } else {
            throw std::runtime_error(std::string(label) + " contains invalid UTF-8");
        }
        if (pos + width > text.size()) {
            throw std::runtime_error(std::string(label) + " contains truncated UTF-8");
        }
        for (size_t i = 1; i < width; ++i) {
            if (!engine::text::is_utf8_continuation(static_cast<unsigned char>(text[pos + i]))) {
                throw std::runtime_error(std::string(label) + " contains invalid UTF-8 continuation byte");
            }
        }
        spans.push_back({pos, pos + width, text.substr(pos, width)});
        pos += width;
    }
    return spans;
}

std::string substring_codepoints(const std::string & text, const std::vector<CodepointSpan> & spans, size_t start, size_t end) {
    if (start >= end) {
        return {};
    }
    return text.substr(spans[start].start, spans[end - 1].end - spans[start].start);
}

bool is_segment_punctuation(std::string_view ch, const std::string & language) {
    if (language == "zh") {
        return ch == "。" || ch == "？" || ch == "！" || ch == "；" || ch == "：" ||
               ch == "." || ch == "?" || ch == "!" || ch == ";";
    }
    return ch == "." || ch == "?" || ch == "!" || ch == ";" || ch == ":";
}

bool is_quote_after_segment(std::string_view ch) noexcept {
    return ch == "\"" || ch == "”";
}

bool is_remove_tail_punctuation(std::string_view ch) noexcept {
    return ch == "。" || ch == "；" || ch == "：" || ch == "." || ch == ";";
}

bool is_only_segment_punctuation(const std::string & text) {
    const auto spans = split_codepoints(text, "Confucius4-TTS segment");
    if (spans.empty()) {
        return false;
    }
    for (const auto & span : spans) {
        if (span.text == " " || span.text == "\t" || span.text == "\n" || span.text == "\r") {
            continue;
        }
        if (!is_segment_punctuation(span.text, "zh") &&
            span.text != "," && span.text != "，" && span.text != "、" &&
            span.text != "\"" && span.text != "”" && span.text != "'" && span.text != "‘" &&
            span.text != "(" && span.text != ")" && span.text != "（" && span.text != "）" &&
            span.text != "[" && span.text != "]" && span.text != "【" && span.text != "】") {
            return false;
        }
    }
    return true;
}

int64_t official_segment_length(const ConfuciusTextTokenizer & tokenizer, const std::string & text, const std::string & language) {
    if (language == "zh") {
        return static_cast<int64_t>(engine::text::utf8_codepoint_count(text, "Confucius4-TTS Chinese segment"));
    }
    return static_cast<int64_t>(tokenizer.encode_without_special_tokens(text).size());
}

std::vector<std::string> official_segments(
    const ConfuciusTextTokenizer & tokenizer,
    std::string text,
    const std::string & language,
    int64_t max_tokens) {
    constexpr int64_t kMinTokens = 60;
    constexpr int64_t kMergeThreshold = 20;
    if (max_tokens <= 0) {
        throw std::runtime_error("Confucius4-TTS max text tokens per segment must be positive");
    }

    auto spans = split_codepoints(text, "Confucius4-TTS normalized text");
    if (!spans.empty() && !is_segment_punctuation(spans.back().text, language)) {
        text += language == "zh" ? "。" : ".";
        spans = split_codepoints(text, "Confucius4-TTS punctuated text");
    }

    std::vector<std::string> segments;
    size_t start = 0;
    for (size_t i = 0; i < spans.size(); ++i) {
        if (!is_segment_punctuation(spans[i].text, language)) {
            continue;
        }
        if (i <= start) {
            continue;
        }
        size_t end = i + 1;
        if (end < spans.size() && is_quote_after_segment(spans[end].text)) {
            ++end;
        }
        segments.push_back(substring_codepoints(text, spans, start, end));
        start = end;
    }

    if (segments.size() == 1 && official_segment_length(tokenizer, segments.front(), language) > max_tokens) {
        const std::string long_text = segments.front();
        const auto long_spans = split_codepoints(long_text, "Confucius4-TTS long segment");
        const size_t tail = long_spans.empty() ? 0 : long_spans.size() - 1;
        segments.clear();
        const size_t chunk = static_cast<size_t>(max_tokens);
        for (size_t offset = 0; offset < tail; offset += chunk) {
            segments.push_back(substring_codepoints(long_text, long_spans, offset, std::min(offset + chunk, tail)));
        }
    }

    std::vector<std::string> final_segments;
    std::string current;
    for (const std::string & segment : segments) {
        if (official_segment_length(tokenizer, current + segment, language) > max_tokens &&
            official_segment_length(tokenizer, current, language) > kMinTokens) {
            final_segments.push_back(std::move(current));
            current.clear();
        }
        current += segment;
    }
    if (!current.empty()) {
        if (official_segment_length(tokenizer, current, language) < kMergeThreshold && !final_segments.empty()) {
            final_segments.back() += current;
        } else {
            final_segments.push_back(std::move(current));
        }
    }

    if (language == "zh") {
        for (std::string & segment : final_segments) {
            const auto segment_spans = split_codepoints(segment, "Confucius4-TTS Chinese final segment");
            if (!segment_spans.empty() && is_remove_tail_punctuation(segment_spans.back().text)) {
                segment = substring_codepoints(segment, segment_spans, 0, segment_spans.size() - 1);
            }
        }
    }

    final_segments.erase(
        std::remove_if(final_segments.begin(), final_segments.end(), is_only_segment_punctuation),
        final_segments.end());
    return final_segments;
}

std::string language_token(const std::string & language) {
    const auto it = std::find_if(
        std::begin(kLanguageTokens),
        std::end(kLanguageTokens),
        [&](const LanguageToken & item) { return item.first == language; });
    if (it != std::end(kLanguageTokens)) {
        return std::string(it->second);
    }
    return "请用" + language + "朗读接下来的文字";
}

}  // namespace

ConfuciusTextTokenizer::ConfuciusTextTokenizer(std::shared_ptr<const ConfuciusAssets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Confucius4-TTS tokenizer requires assets");
    }
    pieces_ = engine::tokenizers::load_sentencepiece_model(assets_->resources.require_file("tokenizer_model"));
}

std::string ConfuciusTextTokenizer::normalize(const std::string & text, const std::string & language) const {
    if (language == "zh") {
        return engine::text::normalize_chinese_text(
            engine::text::collapse_ascii_whitespace(text),
            engine::text::ChineseTextNormalizationTarget::Confucius4TTS);
    }
    if (language == "ja") {
        return engine::text::collapse_ascii_whitespace(text);
    }
    if (language == "en") {
        return engine::text::normalize_english_text(text);
    }
    return engine::text::collapse_ascii_whitespace(text);
}

std::string ConfuciusTextTokenizer::format_prompt(const std::string & text, const std::string & language) const {
    return "You are a helpful assistant. " + language_token(language) + ":" + text;
}

std::vector<int32_t> ConfuciusTextTokenizer::encode(const std::string & text) const {
    auto ids = encode_without_special_tokens(text);
    ids.insert(ids.begin(), bos_id_);
    ids.push_back(eos_id_);
    return ids;
}

std::vector<int32_t> ConfuciusTextTokenizer::encode_without_special_tokens(const std::string & text) const {
    return engine::tokenizers::tokenize_sentencepiece(pieces_, text);
}

std::vector<int32_t> ConfuciusTextTokenizer::encode_prompt(const std::string & text, const std::string & language) const {
    return encode(format_prompt(text, language));
}

std::vector<ConfuciusTextSegment> ConfuciusTextTokenizer::segment_request(const ConfuciusRequest & request) const {
    const std::string normalized = normalize(request.text, request.language);
    const int64_t chunk_size = request.generation.max_text_tokens_per_segment;
    auto chunks = request.generation.text_chunk_mode == engine::text::TextChunkMode::Default
        ? official_segments(*this, normalized, request.language, chunk_size)
        : engine::text::split_text_chunks(normalized, chunk_size, request.generation.text_chunk_mode);
    if (chunks.empty() && !normalized.empty()) {
        chunks.push_back(normalized);
    }
    std::vector<ConfuciusTextSegment> out;
    out.reserve(chunks.size());
    for (const std::string & chunk : chunks) {
        out.push_back({chunk, encode_prompt(chunk, request.language)});
    }
    return out;
}

}  // namespace engine::models::confucius4_tts
