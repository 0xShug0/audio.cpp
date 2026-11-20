#include "engine/models/qwen3_asr/postprocess.h"

#include <utf8proc.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace engine::models::qwen3_asr {
namespace {

std::string trim(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }).base();
    if (first >= last) {
        return {};
    }
    return std::string(first, last);
}

bool starts_with_language(const std::string & value) {
    constexpr const char * kPrefix = "language ";
    if (value.size() < std::char_traits<char>::length(kPrefix)) {
        return false;
    }
    for (size_t i = 0; kPrefix[i] != '\0'; ++i) {
        if (static_cast<char>(std::tolower(static_cast<unsigned char>(value[i]))) != kPrefix[i]) {
            return false;
        }
    }
    return true;
}

}  // namespace

std::string trim_chunk_text(
    const std::string & text,
    const std::vector<runtime::WordTimestamp> & words,
    const runtime::WordTimestamp * kept,
    size_t kept_count,
    int64_t source_start_sample) {
    if (kept_count == 0) return {};

    // Compare canonical letter/number sequences, retaining byte offsets into
    // the original text. This handles stripped punctuation and Nagisa's NFKC
    // normalization without changing the returned text's spelling/punctuation.
    // utf8proc recommends four initial slots; grow only if a compatibility
    // decomposition requires more. Reuse the buffer for the text and all words.
    std::vector<int32_t> decomposed(4);
    const auto letters = [&](const std::string & value, std::vector<size_t> * offsets) {
        std::string normalized;
        for (size_t pos = 0; pos < value.size();) {
            int32_t cp;
            const auto width = utf8proc_iterate(
                reinterpret_cast<const uint8_t *>(value.data() + pos), value.size() - pos, &cp);
            if (width <= 0) throw std::runtime_error("Qwen3 ASR overlap text contains invalid UTF-8");
            constexpr auto options = static_cast<utf8proc_option_t>(
                UTF8PROC_STABLE | UTF8PROC_DECOMPOSE | UTF8PROC_COMPAT);
            auto count = utf8proc_decompose_char(cp, decomposed.data(), decomposed.size(), options, nullptr);
            if (count > static_cast<utf8proc_ssize_t>(decomposed.size())) {
                decomposed.resize(static_cast<size_t>(count));
                count = utf8proc_decompose_char(cp, decomposed.data(), decomposed.size(), options, nullptr);
            }
            if (count < 0) throw std::runtime_error("Qwen3 ASR overlap text normalization failed");
            for (utf8proc_ssize_t i = 0; i < count; ++i) {
                const auto part = decomposed[static_cast<size_t>(i)];
                const auto category = utf8proc_category(part);
                if ((category >= UTF8PROC_CATEGORY_LU && category <= UTF8PROC_CATEGORY_LO) ||
                    (category >= UTF8PROC_CATEGORY_ND && category <= UTF8PROC_CATEGORY_NO)) {
                    uint8_t encoded[4];
                    const auto bytes = utf8proc_encode_char(part, encoded);
                    normalized.append(reinterpret_cast<const char *>(encoded), bytes);
                    if (offsets) offsets->insert(offsets->end(), bytes, pos);
                }
            }
            pos += static_cast<size_t>(width);
        }
        return normalized;
    };
    std::vector<size_t> offsets;
    const auto normalized = letters(text, &offsets);
    size_t cursor = 0, selected = 0, begin = 0, end = text.size();
    bool finished = false;
    for (const auto & word : words) {
        const auto key = letters(word.word, nullptr);
        const auto position = key.empty() ? std::string::npos : normalized.find(key, cursor);
        if (position == std::string::npos) {
            throw std::runtime_error("Qwen3 ASR cannot map aligned word to overlap text: " + word.word);
        }
        cursor = position + key.size();
        const auto global_start = source_start_sample + std::max<int64_t>(0, word.span.start_sample);
        if (selected < kept_count && word.word == kept[selected].word &&
            global_start == kept[selected].span.start_sample) {
            if (selected == 0) begin = position == 0 ? 0 : offsets[position];
            ++selected;
            finished = selected == kept_count;
        } else if (finished) {
            end = offsets[position];
            break;
        }
    }
    if (selected != kept_count) throw std::runtime_error("Qwen3 ASR overlap word selection mismatch");
    return trim(text.substr(begin, end - begin));
}

Qwen3ASRPostprocessor::Qwen3ASRPostprocessor(const Qwen3ASRTextTokenizer & tokenizer)
    : tokenizer_(tokenizer) {}

Qwen3ASRResult Qwen3ASRPostprocessor::decode(
    const Qwen3ASRGeneratedTokens & tokens,
    const Qwen3ASRRequest & request) const {
    Qwen3ASRResult result;
    const std::string raw = trim(tokenizer_.decode(tokens.token_ids));
    result.language = request.language;
    if (!request.language.empty()) {
        result.text = raw;
        return result;
    }
    constexpr const char * kTag = "<asr_text>";
    const size_t tag = raw.find(kTag);
    if (tag == std::string::npos) {
        result.text = raw;
        return result;
    }
    const std::string meta = trim(raw.substr(0, tag));
    result.text = trim(raw.substr(tag + std::char_traits<char>::length(kTag)));
    if (starts_with_language(meta)) {
        result.language = trim(meta.substr(std::char_traits<char>::length("language ")));
        if (result.language == "None") {
            result.language.clear();
            if (result.text.empty()) {
                return result;
            }
        }
    }
    return result;
}

}  // namespace engine::models::qwen3_asr
