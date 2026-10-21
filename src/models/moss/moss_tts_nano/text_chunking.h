#pragma once

#include "engine/framework/text/utf8.h"

#include <cctype>
#include <cstdint>
#include <string_view>

namespace engine::models::moss_tts_nano::detail {

// The model's voice-cloning path is tuned for roughly 50-75 text tokens per
// chunk. CJK characters use substantially more tokens than Latin text, so
// CJK-dominant input gets the smaller framework chunk budget.
inline bool is_cjk_codepoint(uint32_t cp) noexcept {
    return (cp >= 0x3040U && cp <= 0x30FFU) || (cp >= 0x3400U && cp <= 0x4DBFU) ||
           (cp >= 0x4E00U && cp <= 0x9FFFU) || (cp >= 0xAC00U && cp <= 0xD7A3U) ||
           (cp >= 0xF900U && cp <= 0xFAFFU) || (cp >= 0x20000U && cp <= 0x2EE5FU) ||
           (cp >= 0x2F800U && cp <= 0x2FA1FU) || (cp >= 0x30000U && cp <= 0x323AFU);
}

inline int64_t default_text_chunk_size(std::string_view text) {
    constexpr int64_t kDefaultTextChunkSize = 256;
    constexpr int64_t kDefaultCjkTextChunkSize = 50;
    engine::text::utf8_codepoint_count(text, "MOSS-TTS-Nano text");

    size_t cjk = 0;
    size_t other = 0;
    for (size_t pos = 0; pos < text.size();) {
        const auto first = static_cast<unsigned char>(text[pos]);
        if (first < 0x80U) {
            other += std::isalnum(first) != 0;
            ++pos;
        } else if ((first & 0xE0U) == 0xC0U) {
            ++other;
            pos += 2;
        } else if ((first & 0xF0U) == 0xE0U) {
            const uint32_t cp = ((first & 0x0FU) << 12) |
                                ((static_cast<unsigned char>(text[pos + 1]) & 0x3FU) << 6) |
                                (static_cast<unsigned char>(text[pos + 2]) & 0x3FU);
            cjk += is_cjk_codepoint(cp);
            pos += 3;
        } else {
            const uint32_t cp = ((first & 0x07U) << 18) |
                                ((static_cast<unsigned char>(text[pos + 1]) & 0x3FU) << 12) |
                                ((static_cast<unsigned char>(text[pos + 2]) & 0x3FU) << 6) |
                                (static_cast<unsigned char>(text[pos + 3]) & 0x3FU);
            cjk += is_cjk_codepoint(cp);
            pos += 4;
        }
    }
    return cjk > other ? kDefaultCjkTextChunkSize : kDefaultTextChunkSize;
}

}  // namespace engine::models::moss_tts_nano::detail
