// Unicode data and the string methods that depend on it. The tables are read
// from Python (see tools/community_models/tiny_voca_tables.py), so every
// function here answers as the Python build that made them.
#include "text.h"

#include <algorithm>

namespace dizione {

namespace {

#include "unicode_tables.inc"

// `table` holds inclusive (first, last) pairs in rising order.
bool in_ranges(const uint32_t * table, size_t count, char32_t c) {
    size_t low = 0;
    size_t high = count / 2;
    while (low < high) {
        const size_t middle = (low + high) / 2;
        if (c > table[2 * middle + 1]) {
            low = middle + 1;
        } else if (c < table[2 * middle]) {
            high = middle;
        } else {
            return true;
        }
    }
    return false;
}

// `table` holds (key, value) pairs in rising order of key.
bool find_pair(const uint32_t * table, size_t count, char32_t c, uint32_t * value) {
    size_t low = 0;
    size_t high = count / 2;
    while (low < high) {
        const size_t middle = (low + high) / 2;
        const uint32_t key = table[2 * middle];
        if (c > key) {
            low = middle + 1;
        } else if (c < key) {
            high = middle;
        } else {
            *value = table[2 * middle + 1];
            return true;
        }
    }
    return false;
}

// `table` holds records (key, n, n values) in rising order of key. They are
// few, so a walk is enough.
const uint32_t * find_record(const uint32_t * table, size_t count, char32_t c, size_t * length) {
    size_t i = 0;
    while (i + 1 < count) {
        const uint32_t key = table[i];
        const size_t n = table[i + 1];
        if (key == c) {
            *length = n;
            return table + i + 2;
        }
        if (key > c) {
            break;
        }
        i += 2 + n;
    }
    return nullptr;
}

constexpr char32_t kSigma = 0x3A3;
constexpr char32_t kSmallSigma = 0x3C3;
constexpr char32_t kFinalSigma = 0x3C2;

// str.lower() writes a capital sigma as a final sigma after a cased letter and
// before none, skipping the letters Unicode calls case-ignorable.
char32_t lower_sigma(StrView text, size_t index) {
    long j = static_cast<long>(index) - 1;
    char32_t c = 0;
    for (; j >= 0; --j) {
        c = text[static_cast<size_t>(j)];
        if (!in_ranges(kCaseIgnorable, kCaseIgnorableCount, c)) {
            break;
        }
    }
    bool final_sigma = j >= 0 && in_ranges(kCasedNotIgnorable, kCasedNotIgnorableCount, c);
    if (final_sigma) {
        size_t k = index + 1;
        for (; k < text.size(); ++k) {
            c = text[k];
            if (!in_ranges(kCaseIgnorable, kCaseIgnorableCount, c)) {
                break;
            }
        }
        final_sigma = k == text.size() || !in_ranges(kCasedNotIgnorable, kCasedNotIgnorableCount, c);
    }
    return final_sigma ? kFinalSigma : kSmallSigma;
}

// Hangul syllables are composed and decomposed by arithmetic.
constexpr char32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr char32_t kLCount = 19, kVCount = 21, kTCount = 28;
constexpr char32_t kNCount = kVCount * kTCount, kSCount = kLCount * kNCount;

void decompose(char32_t c, Str & out) {
    if (c < 0xA0) {
        out.push_back(c);
        return;
    }
    if (c >= kSBase && c < kSBase + kSCount) {
        const char32_t index = c - kSBase;
        out.push_back(kLBase + index / kNCount);
        out.push_back(kVBase + (index % kNCount) / kTCount);
        if (index % kTCount != 0) {
            out.push_back(kTBase + index % kTCount);
        }
        return;
    }
    const uint32_t * end = kNfkdKeys + kNfkdKeysCount;
    const uint32_t * found = std::lower_bound(kNfkdKeys, end, static_cast<uint32_t>(c));
    if (found == end || *found != c) {
        out.push_back(c);
        return;
    }
    const size_t slot = static_cast<size_t>(found - kNfkdKeys);
    for (uint32_t k = kNfkdOffsets[slot]; k < kNfkdOffsets[slot + 1]; ++k) {
        out.push_back(kNfkdPool[k]);
    }
}

bool compose_pair(char32_t first, char32_t second, char32_t * composed) {
    if (first >= kLBase && first < kLBase + kLCount && second >= kVBase && second < kVBase + kVCount) {
        *composed = kSBase + ((first - kLBase) * kVCount + (second - kVBase)) * kTCount;
        return true;
    }
    if (first >= kSBase && first < kSBase + kSCount && (first - kSBase) % kTCount == 0 &&
        second > kTBase && second < kTBase + kTCount) {
        *composed = first + (second - kTBase);
        return true;
    }
    size_t low = 0;
    size_t high = kComposeCount / 3;
    while (low < high) {
        const size_t middle = (low + high) / 2;
        const uint32_t a = kCompose[3 * middle];
        const uint32_t b = kCompose[3 * middle + 1];
        if (a < first || (a == first && b < second)) {
            low = middle + 1;
        } else if (a == first && b == second) {
            *composed = kCompose[3 * middle + 2];
            return true;
        } else {
            high = middle;
        }
    }
    return false;
}

Str decomposed(StrView text) {
    Str out;
    out.reserve(text.size());
    for (const char32_t c : text) {
        decompose(c, out);
    }
    // Canonical order: marks that follow one another are sorted by class, stably.
    size_t i = 0;
    while (i < out.size()) {
        if (uni::combining(out[i]) == 0) {
            ++i;
            continue;
        }
        size_t end = i;
        while (end < out.size() && uni::combining(out[end]) != 0) {
            ++end;
        }
        std::stable_sort(out.begin() + static_cast<std::ptrdiff_t>(i), out.begin() + static_cast<std::ptrdiff_t>(end),
                         [](char32_t a, char32_t b) { return uni::combining(a) < uni::combining(b); });
        i = end;
    }
    return out;
}

}  // namespace

namespace uni {

bool is_alpha(char32_t c) {
    if (c < 0x80) {
        return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z');
    }
    return in_ranges(kAlpha, kAlphaCount, c);
}

bool is_decimal(char32_t c) {
    if (c < 0x80) {
        return c >= U'0' && c <= U'9';
    }
    return in_ranges(kDecimal, kDecimalCount, c);
}

bool is_digit(char32_t c) {
    if (c < 0x80) {
        return c >= U'0' && c <= U'9';
    }
    return in_ranges(kDigit, kDigitCount, c);
}

bool is_numeric(char32_t c) {
    if (c < 0x80) {
        return c >= U'0' && c <= U'9';
    }
    return in_ranges(kNumeric, kNumericCount, c);
}

bool is_alnum(char32_t c) {
    if (c < 0x80) {
        return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z') || (c >= U'0' && c <= U'9');
    }
    return is_alpha(c) || is_decimal(c) || is_digit(c) || is_numeric(c);
}

bool is_space(char32_t c) {
    if (c < 0x80) {
        return c == U' ' || (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x1F);
    }
    return in_ranges(kSpace, kSpaceCount, c);
}

bool is_upper(char32_t c) {
    if (c < 0x80) {
        return c >= U'A' && c <= U'Z';
    }
    return in_ranges(kUpper, kUpperCount, c);
}

bool is_lower(char32_t c) {
    if (c < 0x80) {
        return c >= U'a' && c <= U'z';
    }
    return in_ranges(kLower, kLowerCount, c);
}

bool is_title(char32_t c) {
    return c >= 0x80 && in_ranges(kTitle, kTitleCount, c);
}

int decimal_value(char32_t c) {
    if (c < 0x80) {
        return c >= U'0' && c <= U'9' ? static_cast<int>(c - U'0') : -1;
    }
    const uint32_t * end = kDecimalZeros + kDecimalZerosCount;
    const uint32_t * found = std::upper_bound(kDecimalZeros, end, static_cast<uint32_t>(c));
    if (found == kDecimalZeros) {
        return -1;
    }
    const uint32_t zero = *(found - 1);
    return c - zero <= 9 ? static_cast<int>(c - zero) : -1;
}

int combining(char32_t c) {
    if (c < 0x300) {
        return 0;
    }
    size_t low = 0;
    size_t high = kCombiningCount / 3;
    while (low < high) {
        const size_t middle = (low + high) / 2;
        if (c > kCombining[3 * middle + 1]) {
            low = middle + 1;
        } else if (c < kCombining[3 * middle]) {
            high = middle;
        } else {
            return static_cast<int>(kCombining[3 * middle + 2]);
        }
    }
    return 0;
}

char32_t simple_lower(char32_t c) {
    if (c < 0x80) {
        return c >= U'A' && c <= U'Z' ? c + 32 : c;
    }
    uint32_t value = 0;
    return find_pair(kSimpleLower, kSimpleLowerCount, c, &value) ? static_cast<char32_t>(value) : c;
}

bool sre_cased(char32_t c) {
    return in_ranges(kSreCased, kSreCasedCount, c);
}

const uint32_t * extra_cases(char32_t lower, size_t * count) {
    return find_record(kExtraCases, kExtraCasesCount, lower, count);
}

Str nfkd(StrView text) {
    return decomposed(text);
}

Str nfkc(StrView text) {
    bool plain = true;
    for (const char32_t c : text) {
        if (c >= 0xA0) {
            plain = false;
            break;
        }
    }
    if (plain) {
        return Str(text);
    }
    const Str source = decomposed(text);
    Str out;
    out.reserve(source.size());
    long starter = -1;               // where the last starter sits in `out`
    int last_class = -1;             // class of the last mark kept after it, -1 for none
    for (const char32_t c : source) {
        const int klass = combining(c);
        if (starter >= 0) {
            const bool blocked = last_class != -1 && last_class >= klass;
            char32_t composed = 0;
            if (!blocked && compose_pair(out[static_cast<size_t>(starter)], c, &composed)) {
                out[static_cast<size_t>(starter)] = composed;
                continue;
            }
        }
        if (klass == 0) {
            starter = static_cast<long>(out.size());
            last_class = -1;
        } else {
            last_class = klass;
        }
        out.push_back(c);
    }
    return out;
}

}  // namespace uni

namespace py {

Str lower(StrView text) {
    Str out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        const char32_t c = text[i];
        if (c < 0x80) {
            out.push_back(c >= U'A' && c <= U'Z' ? c + 32 : c);
            continue;
        }
        if (c == kSigma) {
            out.push_back(lower_sigma(text, i));
            continue;
        }
        size_t length = 0;
        if (const uint32_t * special = find_record(kSpecialLower, kSpecialLowerCount, c, &length)) {
            for (size_t k = 0; k < length; ++k) {
                out.push_back(special[k]);
            }
            continue;
        }
        out.push_back(uni::simple_lower(c));
    }
    return out;
}

Str upper(StrView text) {
    Str out;
    out.reserve(text.size());
    for (const char32_t c : text) {
        if (c < 0x80) {
            out.push_back(c >= U'a' && c <= U'z' ? c - 32 : c);
            continue;
        }
        size_t length = 0;
        if (const uint32_t * special = find_record(kSpecialUpper, kSpecialUpperCount, c, &length)) {
            for (size_t k = 0; k < length; ++k) {
                out.push_back(special[k]);
            }
            continue;
        }
        uint32_t value = 0;
        out.push_back(find_pair(kSimpleUpper, kSimpleUpperCount, c, &value) ? static_cast<char32_t>(value) : c);
    }
    return out;
}

}  // namespace py

}  // namespace dizione
