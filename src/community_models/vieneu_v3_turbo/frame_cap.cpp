#include "engine/community_models/vieneu_v3_turbo/frame_cap.h"

#include <algorithm>
#include <cmath>
#include <string_view>
#include <vector>

namespace engine::models::vieneu_v3_turbo {
namespace {

// Constants follow vieneu_utils/core_utils.py.
constexpr double kMaxFramesPerPhone = 2.0;
constexpr int64_t kFrameCapSlack = 24;          // lead-in / fixed cost
constexpr int64_t kSingleWordMaxFrames = 13;    // ~1 s at 12.5 frames/s
constexpr int64_t kSyllableCapPerExtra = 5;
constexpr int64_t kSyllableCapMaxSyl = 4;
constexpr int64_t kSingleWordMaxPhones = 24;

std::vector<char32_t> decode_utf8(std::string_view text) {
    std::vector<char32_t> out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        const auto byte = static_cast<unsigned char>(text[i]);
        char32_t cp = 0;
        size_t len = 1;
        if (byte < 0x80) {
            cp = byte;
        } else if ((byte >> 5) == 0x6) {
            cp = byte & 0x1F; len = 2;
        } else if ((byte >> 4) == 0xE) {
            cp = byte & 0x0F; len = 3;
        } else if ((byte >> 3) == 0x1E) {
            cp = byte & 0x07; len = 4;
        } else {
            ++i;
            continue;
        }
        if (i + len > text.size()) {
            break;
        }
        for (size_t k = 1; k < len; ++k) {
            cp = (cp << 6) | (static_cast<unsigned char>(text[i + k]) & 0x3F);
        }
        out.push_back(cp);
        i += len;
    }
    return out;
}

// Removes `<|emotion_N|>`, `<en>` and `</en>` (the Python `_FRAME_MARKUP_RE`).
std::vector<char32_t> strip_markup(const std::vector<char32_t> & cps) {
    std::vector<char32_t> out;
    out.reserve(cps.size());
    auto starts_with = [&](size_t at, std::u32string_view s) {
        return at + s.size() <= cps.size() && std::equal(s.begin(), s.end(), cps.begin() + static_cast<std::ptrdiff_t>(at));
    };
    for (size_t i = 0; i < cps.size();) {
        if (starts_with(i, U"<|emotion_")) {
            size_t j = i + 10;
            while (j < cps.size() && cps[j] >= U'0' && cps[j] <= U'9') ++j;
            if (starts_with(j, U"|>")) {
                i = j + 2;
                continue;
            }
        }
        if (starts_with(i, U"<en>")) { i += 4; continue; }
        if (starts_with(i, U"</en>")) { i += 5; continue; }
        out.push_back(cps[i]);
        ++i;
    }
    return out;
}

bool is_alpha(char32_t c) {
    // Python str.isalpha() over the characters SEA-G2P emits: ASCII letters, Latin-1 /
    // Latin Extended, IPA extensions, phonetic extensions and spacing modifiers that
    // are letters (ˈ ˌ ː are modifier letters in Unicode and count as alphabetic).
    return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z') ||
           (c >= 0x00C0 && c <= 0x024F && c != 0x00D7 && c != 0x00F7) ||
           (c >= 0x0250 && c <= 0x02AF) || (c >= 0x02B0 && c <= 0x02C1) ||
           (c >= 0x02C6 && c <= 0x02D1) || (c >= 0x02E0 && c <= 0x02E4) ||
           (c >= 0x1D00 && c <= 0x1DBF) || (c >= 0x1E00 && c <= 0x1EFF);
}

bool is_ipa_vowel(char32_t c) {
    static constexpr std::u32string_view kVowels = U"aeiouyæɐɑɒɔəɘɛɜɤɯɵøœʉʊʌɪɨɚɝᵻᵿ";
    return kVowels.find(c) != std::u32string_view::npos;
}

bool is_cue_only(const std::string & phonemes) {
    if (phonemes.find("<|emotion_") == std::string::npos) {
        return false;
    }
    const auto stripped = strip_markup(decode_utf8(phonemes));
    return std::none_of(stripped.begin(), stripped.end(), is_alpha);
}

}  // namespace

int64_t syllable_count(const std::string & phonemes) {
    const auto stripped = strip_markup(decode_utf8(phonemes));
    int64_t total = 0;
    size_t i = 0;
    while (i < stripped.size()) {
        while (i < stripped.size() && (stripped[i] == U' ' || stripped[i] == U'\t' || stripped[i] == U'\n')) ++i;
        const size_t start = i;
        while (i < stripped.size() && !(stripped[i] == U' ' || stripped[i] == U'\t' || stripped[i] == U'\n')) ++i;
        if (start == i) {
            break;
        }
        int64_t groups = 0;
        bool in_vowel = false;
        bool consonant_seen = true;
        bool has_alpha = false;
        for (size_t k = start; k < i; ++k) {
            const char32_t ch = stripped[k];
            has_alpha = has_alpha || is_alpha(ch);
            if (is_ipa_vowel(ch)) {
                if (!in_vowel && consonant_seen) {
                    ++groups;
                }
                in_vowel = true;
                consonant_seen = false;
            } else if (ch == U'ː' || ch == U'ˈ' || ch == U'ˌ' || (ch >= U'0' && ch <= U'9')) {
                // A stress mark after the token already has a vowel cluster starts a new
                // syllable (English: kɹiːˈeɪt -> 2). Vietnamese tokens carry exactly one
                // stress mark before their first vowel, so groups is still 0 there.
                if ((ch == U'ˈ' || ch == U'ˌ') && groups > 0) {
                    in_vowel = false;
                    consonant_seen = true;
                } else {
                    in_vowel = false;
                }
            } else {
                in_vowel = false;
                consonant_seen = true;
            }
        }
        if (has_alpha) {
            total += std::max<int64_t>(1, groups);
        }
    }
    return total;
}

int64_t max_expected_frames(const std::string & phonemes) {
    const auto stripped = strip_markup(decode_utf8(phonemes));
    const auto eff_len = static_cast<int64_t>(stripped.size());
    int64_t cap = kFrameCapSlack + static_cast<int64_t>(std::ceil(kMaxFramesPerPhone * static_cast<double>(eff_len)));
    if (is_cue_only(phonemes)) {
        return std::min(cap, kSingleWordMaxFrames);
    }
    if (phonemes.find("<|emotion_") == std::string::npos) {
        const int64_t syl = std::max<int64_t>(1, syllable_count(phonemes));
        if (syl <= kSyllableCapMaxSyl && eff_len <= kSingleWordMaxPhones * syl) {
            cap = std::min(cap, kSingleWordMaxFrames + kSyllableCapPerExtra * (syl - 1));
        }
    }
    return cap;
}

}  // namespace engine::models::vieneu_v3_turbo
