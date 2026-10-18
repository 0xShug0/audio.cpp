// Robust input normalisation for the MOSS-TTS family, ported from
// tts_robust_normalizer_single_script.py in the MOSS-TTS-v1.5 checkpoint.
//
// The reference is regex-heavy and leans on lookbehind, which std::regex does
// not support at all, and on Unicode classes that byte-oriented matching gets
// wrong. So this works on codepoints and scans by hand. It is longer than the
// Python but it is the same algorithm, stage for stage, and the upstream test
// vectors are kept as the contract (see tests/unittests).
//
// It is a no-op on already-clean text -- the five prompts used to bring the
// model up pass through unchanged, Chinese included. What it is for is text
// that arrives as markdown, with URLs, with CJK and Latin jammed together, or
// with punctuation repeated for emphasis.

#include "engine/framework/text/moss_tts_normalization.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace engine::text {
namespace {

using U32 = std::u32string;

U32 to_codepoints(const std::string & text) {
    U32 out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i]);
        uint32_t cp = lead;
        size_t extra = 0;
        if (lead >= 0xF0) { cp = lead & 0x07u; extra = 3; }
        else if (lead >= 0xE0) { cp = lead & 0x0Fu; extra = 2; }
        else if (lead >= 0xC0) { cp = lead & 0x1Fu; extra = 1; }
        if (i + extra >= text.size()) { extra = 0; cp = lead; }
        for (size_t k = 1; k <= extra; ++k) {
            cp = (cp << 6) | (static_cast<unsigned char>(text[i + k]) & 0x3Fu);
        }
        out.push_back(static_cast<char32_t>(cp));
        i += extra + 1;
    }
    return out;
}

std::string to_utf8(const U32 & text) {
    std::string out;
    out.reserve(text.size() * 2);
    for (const char32_t ch : text) {
        const auto cp = static_cast<uint32_t>(ch);
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        } else {
            out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        }
    }
    return out;
}

// Scripts that do not use spaces to separate words: Han plus kana.
bool is_cjk(char32_t c) {
    return (c >= 0x3400 && c <= 0x4DBF) || (c >= 0x4E00 && c <= 0x9FFF)
        || (c >= 0x3040 && c <= 0x30FF);
}

bool is_ascii_alpha(char32_t c) {
    return (c >= U'A' && c <= U'Z') || (c >= U'a' && c <= U'z');
}
bool is_ascii_digit(char32_t c) { return c >= U'0' && c <= U'9'; }
bool is_ascii_alnum(char32_t c) { return is_ascii_alpha(c) || is_ascii_digit(c); }
bool is_space(char32_t c) { return c == U' ' || c == U'\t' || c == U'\r' || c == U'\f' || c == U'\v'; }

// The character set a "latinish" token may contain: a run that carries at least
// one Latin letter, which is what earns a space against neighbouring CJK.
bool is_token_char(char32_t c) {
    return is_ascii_alnum(c) || c == U'.' || c == U'_' || c == U'/' || c == U'+'
        || c == U':' || c == U'-';
}

bool is_zero_width(char32_t c) {
    return (c >= 0x200B && c <= 0x200D) || c == 0xFEFF;
}

// Unicode general category C*, minus the whitespace the reference keeps.
bool is_dropped_control(char32_t c) {
    if (c == U'\n' || c == U'\t' || c == U' ') {
        return false;
    }
    return c < 0x20 || c == 0x7F || (c >= 0x80 && c <= 0x9F);
}

const U32 kCjkClosers = U"，。！？；：、”’」』】）》";
const U32 kCjkOpeners = U"（【「『《“‘";

bool contains(const U32 & set, char32_t c) { return set.find(c) != U32::npos; }

U32 base_cleanup(const U32 & text) {
    U32 out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        char32_t c = text[i];
        if (c == U'\r') {
            // CRLF and a bare CR both become one newline.
            if (i + 1 < text.size() && text[i + 1] == U'\n') continue;
            c = U'\n';
        }
        if (c == 0x3000) c = U' ';                 // ideographic space
        if (is_zero_width(c) || is_dropped_control(c)) continue;
        out.push_back(c);
    }
    return out;
}


// ---------------------------------------------------------------- markdown
// Markdown link: [text](http...) -> "text url". Then per line, strip heading,
// quote and list markers, drop blank lines, and join what is left with the
// ideographic full stop -- lines are sentences to the reference.
U32 normalize_markdown_and_lines(const U32 & text) {
    U32 linked;
    linked.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        if (text[i] == U'[') {
            const auto close = text.find(U']', i + 1);
            if (close != U32::npos && close + 1 < text.size() && text[close + 1] == U'('
                && text.compare(close + 2, 4, U"http") == 0) {
                const auto paren = text.find(U')', close + 2);
                if (paren != U32::npos) {
                    const auto label = text.substr(i + 1, close - i - 1);
                    const auto url = text.substr(close + 2, paren - close - 2);
                    if (label.find(U'[') == U32::npos && label.find(U']') == U32::npos
                        && url.find_first_of(U" \t\n") == U32::npos) {
                        linked += label;
                        linked += U' ';
                        linked += url;
                        i = paren + 1;
                        continue;
                    }
                }
            }
        }
        linked.push_back(text[i]);
        ++i;
    }

    std::vector<U32> lines;
    size_t start = 0;
    while (start <= linked.size()) {
        auto end = linked.find(U'\n', start);
        if (end == U32::npos) end = linked.size();
        U32 line = linked.substr(start, end - start);
        size_t a = 0;
        size_t b = line.size();
        while (a < b && (is_space(line[a]) || line[a] == U'\n')) ++a;
        while (b > a && (is_space(line[b - 1]) || line[b - 1] == U'\n')) --b;
        line = line.substr(a, b - a);
        start = end + 1;
        if (line.empty()) continue;

        size_t cut = 0;
        if (line[0] == U'#') {                                  // heading
            size_t h = 0;
            while (h < line.size() && h < 6 && line[h] == U'#') ++h;
            if (h < line.size() && is_space(line[h])) {
                cut = h;
                while (cut < line.size() && is_space(line[cut])) ++cut;
            }
        } else if (line[0] == U'>') {                           // quote
            if (line.size() > 1 && is_space(line[1])) {
                cut = 1;
                while (cut < line.size() && is_space(line[cut])) ++cut;
            }
        } else if (line[0] == U'-' || line[0] == U'*' || line[0] == U'+') {  // bullet
            if (line.size() > 1 && is_space(line[1])) {
                cut = 1;
                while (cut < line.size() && is_space(line[cut])) ++cut;
            }
        } else if (is_ascii_digit(line[0])) {                   // ordered list
            size_t d = 0;
            while (d < line.size() && is_ascii_digit(line[d])) ++d;
            if (d < line.size() && (line[d] == U'.' || line[d] == U')')
                && d + 1 < line.size() && is_space(line[d + 1])) {
                cut = d + 1;
                while (cut < line.size() && is_space(line[cut])) ++cut;
            }
        }
        lines.push_back(line.substr(cut));
    }

    U32 out;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i != 0) out += U'。';
        out += lines[i];
    }
    return out;
}


// ------------------------------------------------------------ protected spans
// High-risk runs that must survive the space and punctuation rules intact:
// URLs, emails, @mentions, u/ and r/ paths, hashtags, dot-tokens like ".env",
// and file-like tokens such as "app.js.map" or "v2.3.1". Each is lifted out and
// replaced by ___PROTn___ for the duration.

bool prev_is_wordish(const U32 & t, size_t i) {
    return i > 0 && (is_ascii_alnum(t[i - 1]) || t[i - 1] == U'_');
}

size_t scan_url(const U32 & t, size_t i) {
    if (t.compare(i, 7, U"http://") != 0 && t.compare(i, 8, U"https://") != 0) return 0;
    size_t j = i;
    while (j < t.size() && !is_space(t[j]) && t[j] != U'\n' && t[j] != 0x3000
           && !contains(U"，。！？；、）】》〉」』", t[j])) {
        ++j;
    }
    return j - i;
}

size_t scan_email(const U32 & t, size_t i) {
    if (prev_is_wordish(t, i) || (i > 0 && (t[i - 1] == U'.' || t[i - 1] == U'+' || t[i - 1] == U'-'))) return 0;
    size_t j = i;
    while (j < t.size() && (is_ascii_alnum(t[j]) || contains(U"._%+-", t[j]))) ++j;
    if (j == i || j >= t.size() || t[j] != U'@') return 0;
    size_t k = j + 1;
    size_t last_dot = U32::npos;
    while (k < t.size() && (is_ascii_alnum(t[k]) || t[k] == U'.' || t[k] == U'-')) {
        if (t[k] == U'.') last_dot = k;
        ++k;
    }
    if (last_dot == U32::npos || k - last_dot < 3) return 0;
    return k - i;
}

size_t scan_mention(const U32 & t, size_t i) {
    if (t[i] != U'@' || prev_is_wordish(t, i)) return 0;
    size_t j = i + 1;
    while (j < t.size() && j - i <= 32 && (is_ascii_alnum(t[j]) || t[j] == U'_')) ++j;
    return j > i + 1 ? j - i : 0;
}

size_t scan_reddit(const U32 & t, size_t i) {
    if (prev_is_wordish(t, i)) return 0;
    if (!(t[i] == U'u' || t[i] == U'r') || i + 1 >= t.size() || t[i + 1] != U'/') return 0;
    size_t j = i + 2;
    while (j < t.size() && (is_ascii_alnum(t[j]) || t[j] == U'_')) ++j;
    return j > i + 2 ? j - i : 0;
}

size_t scan_hashtag(const U32 & t, size_t i) {
    if (t[i] != U'#' || prev_is_wordish(t, i)) return 0;
    if (i + 1 >= t.size() || is_space(t[i + 1]) || t[i + 1] == U'\n') return 0;
    size_t j = i + 1;
    while (j < t.size() && !is_space(t[j]) && t[j] != U'\n' && t[j] != U'#') ++j;
    return j - i;
}

// ".env", ".gitignore" -- a leading dot that starts a token rather than ending
// a sentence.
size_t scan_dot_token(const U32 & t, size_t i) {
    if (t[i] != U'.' || prev_is_wordish(t, i)) return 0;
    size_t j = i + 1;
    bool saw_alnum = false;
    while (j < t.size() && (is_ascii_alnum(t[j]) || t[j] == U'.' || t[j] == U'_' || t[j] == U'-')) {
        if (is_ascii_alnum(t[j])) saw_alnum = true;
        ++j;
    }
    return saw_alnum ? j - i : 0;
}

// "app.js.map", "index.d.ts", "v2.3.1", "foo/bar-baz.py": a run that carries at
// least one letter and at least one separator, so it is not a plain word and
// not a bare number.
size_t scan_filelike(const U32 & t, size_t i) {
    if (prev_is_wordish(t, i) || !is_ascii_alnum(t[i])) return 0;
    size_t j = i;
    while (j < t.size() && is_token_char(t[j])) ++j;
    while (j > i && !is_ascii_alnum(t[j - 1])) --j;      // must end alphanumeric
    if (j <= i + 1) return 0;
    if (j < t.size() && (is_ascii_alnum(t[j]) || t[j] == U'_')) return 0;
    bool has_alpha = false;
    bool has_sep = false;
    for (size_t k = i; k < j; ++k) {
        if (is_ascii_alpha(t[k])) has_alpha = true;
        if (contains(U"._/+:-", t[k])) has_sep = true;
    }
    return (has_alpha && has_sep) ? j - i : 0;
}

U32 placeholder(size_t index) {
    U32 out = U"___PROT";
    const auto digits = std::to_string(index);
    for (const char d : digits) out.push_back(static_cast<char32_t>(d));
    out += U"___";
    return out;
}

U32 protect_spans(const U32 & text, std::vector<U32> & protectedSpans) {
    U32 out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        size_t len = 0;
        // Order matters and follows the reference: a URL wins over the file-like
        // rule that would otherwise bite off part of it.
        if ((len = scan_url(text, i)) == 0)
        if ((len = scan_email(text, i)) == 0)
        if ((len = scan_mention(text, i)) == 0)
        if ((len = scan_reddit(text, i)) == 0)
        if ((len = scan_hashtag(text, i)) == 0)
        if ((len = scan_dot_token(text, i)) == 0)
        len = scan_filelike(text, i);
        if (len > 0) {
            out += placeholder(protectedSpans.size());
            protectedSpans.push_back(text.substr(i, len));
            i += len;
            continue;
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}

U32 restore_spans(const U32 & text, const std::vector<U32> & protectedSpans) {
    // One left-to-right pass, never re-scanning what has been written. Replacing
    // token-by-token over the whole string instead loops forever when a
    // protected span contains placeholder text itself -- "https://x.com/___PROT0___"
    // is a URL, so it is protected whole, and restoring it puts the token back
    // for the next iteration to find. The string grows without bound and the
    // session thread never returns.
    U32 out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        if (text.compare(i, 7, U"___PROT") == 0) {
            size_t j = i + 7;
            size_t index = 0;
            bool digits = false;
            while (j < text.size() && is_ascii_digit(text[j])) {
                index = index * 10 + static_cast<size_t>(text[j] - U'0');
                digits = true;
                ++j;
            }
            if (digits && text.compare(j, 3, U"___") == 0 && index < protectedSpans.size()) {
                out += protectedSpans[index];
                i = j + 3;
                continue;
            }
        }
        out.push_back(text[i]);
        ++i;
    }
    return out;
}


// ------------------------------------------------------------------- spaces
// Latin runs keep single spaces; CJK runs lose them entirely; the boundary
// between the two gains exactly one, because a reader needs it and the model
// was trained with it.

bool is_protected_at(const U32 & t, size_t i, size_t & length) {
    if (t.compare(i, 7, U"___PROT") != 0) return false;
    size_t j = i + 7;
    while (j < t.size() && is_ascii_digit(t[j])) ++j;
    if (j == i + 7 || t.compare(j, 3, U"___") != 0) return false;
    length = j + 3 - i;
    return true;
}

// A token that counts as "latinish" for the boundary rule: a protected span, or
// a run of token characters carrying at least one Latin letter.
bool latinish_at(const U32 & t, size_t i, size_t & length) {
    if (is_protected_at(t, i, length)) return true;
    if (!is_ascii_alnum(t[i])) return false;
    size_t j = i;
    bool has_alpha = false;
    while (j < t.size() && is_token_char(t[j])) {
        if (is_ascii_alpha(t[j])) has_alpha = true;
        ++j;
    }
    if (!has_alpha) return false;
    length = j - i;
    return true;
}

U32 normalize_spaces(const U32 & text) {
    // Collapse horizontal whitespace runs to one space.
    U32 collapsed;
    collapsed.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        if (is_space(text[i])) {
            if (!collapsed.empty() && collapsed.back() == U' ') continue;
            collapsed.push_back(U' ');
            continue;
        }
        collapsed.push_back(text[i]);
    }

    // Drop spaces that sit between two CJK characters, or between CJK and a
    // bare digit, and insert one at a CJK/latinish boundary.
    U32 spaced;
    spaced.reserve(collapsed.size());
    // Only a token carrying a Latin letter earns a space against CJK. A bare
    // number does not: "2026 年" closes up, where "npm 包" does not. Tracking
    // this is what keeps the closing side of the boundary from re-inserting the
    // space the digit rule has just removed.
    bool prev_token_latinish = false;
    for (size_t i = 0; i < collapsed.size(); ++i) {
        const char32_t c = collapsed[i];
        if (c == U' ') {
            const char32_t prev = spaced.empty() ? U'\0' : spaced.back();
            size_t k = i;
            while (k < collapsed.size() && collapsed[k] == U' ') ++k;
            const char32_t next = k < collapsed.size() ? collapsed[k] : U'\0';
            if (is_cjk(prev) && (is_cjk(next) || is_ascii_digit(next))) continue;
            if (is_ascii_digit(prev) && is_cjk(next)) continue;
            spaced.push_back(U' ');
            continue;
        }
        size_t len = 0;
        if (!spaced.empty() && is_cjk(spaced.back()) && latinish_at(collapsed, i, len)) {
            spaced.push_back(U' ');
            spaced.append(collapsed, i, len);
            i += len - 1;
            prev_token_latinish = true;
            continue;
        }
        if (is_cjk(c) && prev_token_latinish) {
            if (!spaced.empty() && spaced.back() != U' ') {
                spaced.push_back(U' ');
            }
        }
        if (latinish_at(collapsed, i, len)) {
            spaced.append(collapsed, i, len);
            i += len - 1;
            prev_token_latinish = true;
            continue;
        }
        prev_token_latinish = false;
        spaced.push_back(c);
    }

    // CJK punctuation takes no space on either side; ASCII punctuation takes
    // none before it.
    U32 punctuated;
    punctuated.reserve(spaced.size());
    for (size_t i = 0; i < spaced.size(); ++i) {
        const char32_t c = spaced[i];
        if (c == U' ') {
            size_t k = i;
            while (k < spaced.size() && spaced[k] == U' ') ++k;
            const char32_t next = k < spaced.size() ? spaced[k] : U'\0';
            if (contains(kCjkClosers, next) || contains(U",.;!?", next)) continue;
            if (!punctuated.empty() && contains(kCjkOpeners, punctuated.back())) continue;
            if (!punctuated.empty() && contains(U"，。！？；：、", punctuated.back())) continue;
            punctuated.push_back(U' ');
            continue;
        }
        punctuated.push_back(c);
    }

    size_t a = 0;
    size_t b = punctuated.size();
    while (a < b && punctuated[a] == U' ') ++a;
    while (b > a && punctuated[b - 1] == U' ') --b;
    return punctuated.substr(a, b - a);
}


// ------------------------------------------------------- structural punctuation
// Bracketed labels at a sentence boundary become sentences; arrows and long
// dashes become punctuation the model can actually say.

bool at_sentence_boundary(const U32 & t, size_t i, size_t & lead) {
    lead = 0;
    if (i == 0) return true;
    size_t j = i;
    while (j > 0 && is_space(t[j - 1])) { --j; ++lead; }
    if (j == 0) return false;
    const char32_t prev = t[j - 1];
    return contains(U"。！？；", prev) || prev == U'!' || prev == U'?' || prev == U';';
}

U32 normalize_structural_punctuation(const U32 & text) {
    U32 work = text;
    // Two passes, because consecutive blocks only converge on the second.
    for (int pass = 0; pass < 2; ++pass) {
        U32 out;
        for (size_t i = 0; i < work.size();) {
            size_t lead = 0;
            const bool boundary = at_sentence_boundary(work, i, lead);
            if (boundary && contains(U"【〖『「", work[i])) {
                const char32_t close = work[i] == U'【' ? U'】' : work[i] == U'〖' ? U'〗'
                                     : work[i] == U'『' ? U'』' : U'」';
                const auto end = work.find(close, i + 1);
                if (end != U32::npos) {
                    out += work.substr(i + 1, end - i - 1);
                    out += U'。';
                    i = end + 1;
                    while (i < work.size() && is_space(work[i])) ++i;
                    continue;
                }
            }
            if (boundary && work[i] == U'《') {
                const auto end = work.find(U'》', i + 1);
                if (end != U32::npos) {
                    size_t after = end + 1;
                    while (after < work.size() && is_space(work[after])) ++after;
                    const bool standalone = after >= work.size()
                        || contains(U"。！？；，", work[after]) || work[after] == U'!'
                        || work[after] == U'?' || work[after] == U';' || work[after] == U','
                        || work[after] == U'—' || work[after] == U'–' || work[after] == U'―'
                        || work[after] == U'-' || work.compare(after, 7, U"___PROT") == 0;
                    if (standalone) {
                        out += work.substr(i + 1, end - i - 1);
                        i = end + 1;
                        continue;
                    }
                }
            }
            out.push_back(work[i]);
            ++i;
        }
        work = out;
    }

    // Arrows of every shape become the ideographic comma: the link survives,
    // the glyph does not reach the model.
    U32 arrows;
    for (size_t i = 0; i < work.size();) {
        size_t len = 0;
        if (contains(U"→←↔⇒⇐⇔⟶⟵⟷⟹⟸⟺↦↤↪↩", work[i])) {
            len = 1;
        } else if (work[i] == U'<' || work[i] == U'-' || work[i] == U'=') {
            size_t j = i;
            if (work[j] == U'<') ++j;
            size_t dashes = 0;
            while (j < work.size() && (work[j] == U'-' || work[j] == U'=')) { ++j; ++dashes; }
            const bool closes = j < work.size() && work[j] == U'>';
            if (dashes > 0 && (closes || work[i] == U'<')) {
                if (closes) ++j;
                len = j - i;
            }
        }
        if (len > 0) {
            while (!arrows.empty() && is_space(arrows.back())) arrows.pop_back();
            arrows += U'，';
            i += len;
            while (i < work.size() && is_space(work[i])) ++i;
            continue;
        }
        arrows.push_back(work[i]);
        ++i;
    }

    // Two or more dashes are a sentence break.
    U32 dashed;
    for (size_t i = 0; i < arrows.size();) {
        if (arrows[i] == U'—' || arrows[i] == U'–' || arrows[i] == U'―' || arrows[i] == U'-') {
            size_t j = i;
            while (j < arrows.size()
                   && (arrows[j] == U'—' || arrows[j] == U'–' || arrows[j] == U'―' || arrows[j] == U'-')) {
                ++j;
            }
            if (j - i >= 2) {
                while (!dashed.empty() && is_space(dashed.back())) dashed.pop_back();
                dashed += U'。';
                i = j;
                while (i < arrows.size() && is_space(arrows[i])) ++i;
                continue;
            }
        }
        dashed.push_back(arrows[i]);
        ++i;
    }
    return dashed;
}

// -------------------------------------------------------- repeated punctuation
U32 normalize_repeated_punctuation(const U32 & text) {
    U32 out;
    for (size_t i = 0; i < text.size();) {
        const char32_t c = text[i];
        // Ellipses of any spelling collapse to one full stop.
        if (c == U'.' || c == U'…') {
            size_t j = i;
            size_t dots = 0;
            size_t ellipses = 0;
            while (j < text.size() && (text[j] == U'.' || text[j] == U'…')) {
                if (text[j] == U'.') ++dots; else ++ellipses;
                ++j;
            }
            if (dots >= 3 || ellipses >= 2) {
                out += U'。';
                i = j;
                continue;
            }
        }
        // A run of one punctuation mark keeps one. Mixed ASCII/CJK spellings of
        // the same mark count as the same run.
        const auto same_run = [&](const U32 & set, char32_t keep) -> bool {
            if (!contains(set, c)) return false;
            size_t j = i;
            while (j < text.size() && contains(set, text[j])) ++j;
            if (j - i < 2) return false;
            out += keep;
            i = j;
            return true;
        };
        if (same_run(U"。．", U'。')) continue;
        if (same_run(U"，,", U'，')) continue;
        if (same_run(U"!！", U'！')) continue;
        if (same_run(U"?？", U'？')) continue;
        out.push_back(c);
        ++i;
    }
    return out;
}

}  // namespace

std::string normalize_moss_tts_text(const std::string & text) {
    U32 work = base_cleanup(to_codepoints(text));
    work = normalize_markdown_and_lines(work);

    std::vector<U32> spans;
    work = protect_spans(work, spans);

    work = normalize_spaces(work);
    work = normalize_structural_punctuation(work);
    work = normalize_repeated_punctuation(work);
    work = normalize_spaces(work);

    work = restore_spans(work, spans);

    size_t a = 0;
    size_t b = work.size();
    while (a < b && (is_space(work[a]) || work[a] == U'\n')) ++a;
    while (b > a && (is_space(work[b - 1]) || work[b - 1] == U'\n')) --b;
    return to_utf8(work.substr(a, b - a));
}

}  // namespace engine::text
