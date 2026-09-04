#include "engine/community_models/sanotts/frontend.h"

#include "engine/framework/io/dynamic_library.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace engine::models::sanotts {
namespace {

using InitializeFn = int (*)(int, int, const char *, int);
using SetVoiceFn = int (*)(const char *);
using TextToPhonemesFn = const char * (*)(const void **, int, int);
using TerminateFn = int (*)();

constexpr int kEspeakSynchronous = 2;
constexpr int kEspeakCharsUtf8 = 1;
// IPA output (0x02), tie flag (bit 7), and U+0361 COMBINING DOUBLE INVERTED
// BREVE in bits 8..23 as the tie character -- exactly the phonemes_mode
// phonemizer computes, so the E2M diphthong patterns ("a͡ɪ" -> "I") can match.
constexpr int kEspeakPhonemesIpaTie = 0x02 | (0x01 << 7) | (0x0361 << 8);
// The piperlite voices phonemize with tie=False: IPA with '_' as the phoneme
// separator in bits 8..23, exactly phonemizer's untied phonemes_mode.
constexpr int kEspeakPhonemesIpaUnderscore = 0x02 | ('_' << 8);

constexpr std::string_view kTieDefault = "͡";   // COMBINING DOUBLE INVERTED BREVE
constexpr std::string_view kTieMisaki = "^";
constexpr std::string_view kSyllabic = "̩";     // COMBINING VERTICAL LINE BELOW
constexpr std::string_view kNasal = "̃";        // COMBINING TILDE

/**
 * The frozen 62-symbol inventory, exactly as the trained package records it.
 *
 * Ids are positional and contiguous from zero; 0..2 are <pad>, <bos>, <eos>
 * and never appear in text.
 */
const std::unordered_map<std::string, int32_t> & vocabulary() {
    static const std::unordered_map<std::string, int32_t> table = [] {
        static constexpr std::array<const char *, 59> symbols = {
            " ", "!", "\"", "(", ")", ",", ".", ":", ";",
            "?", "A", "I", "O", "T", "W", "Y", "b",
            "d", "f", "h", "i", "j", "k", "l", "m",
            "n", "p", "s", "t", "u", "v", "w", "z",
            "æ", "ð", "ŋ", "ɐ", "ɑ",
            "ɔ", "ə", "ɛ", "ɜ", "ɡ",
            "ɪ", "ɹ", "ʃ", "ʊ", "ʌ",
            "ʒ", "ʤ", "ʧ", "ˈ", "ˌ",
            "θ", "ᵊ", "ᵻ", "—", "“",
            "”",
        };
        std::unordered_map<std::string, int32_t> out;
        int32_t id = 3;   // 0..2 are the specials
        for (const char * symbol : symbols) {
            out.emplace(symbol, id++);
        }
        if (out.size() + 3 != 62) {
            throw std::runtime_error("sanoTTS compiled symbol inventory is invalid");
        }
        return out;
    }();
    return table;
}

void replace_all(std::string & value, std::string_view from, std::string_view to) {
    if (from.empty()) {
        return;
    }
    size_t position = 0;
    while ((position = value.find(from, position)) != std::string::npos) {
        value.replace(position, from.size(), to);
        position += to.size();
    }
}

/**
 * misaki EspeakFallback's E2M rewrite, british=false.
 *
 * Order is load-bearing and matches the Python source, which sorts by
 * descending key length: "e͡ɪ" must be tried before the bare "e",
 * or every diphthong collapses to the wrong symbol.
 */
std::string apply_e2m(std::string ps) {
    static const std::array<std::pair<const char *, const char *>, 20> kE2M = {{
        {"ʔˌn̩", "ʔn"},
        {"ʔn̩", "ʔn"},
        {"a^ɪ", "I"}, {"a^ʊ", "W"}, {"d^ʒ", "ʤ"},
        {"e^ɪ", "A"}, {"t^ʃ", "ʧ"}, {"ɔ^ɪ", "Y"},
        {"ə^l", "ᵊl"},
        {"ʲo", "jo"}, {"ʲə", "jə"}, {"e", "A"}, {"ʲ", ""},
        {"ɚ", "əɹ"}, {"r", "ɹ"}, {"x", "k"}, {"ç", "k"},
        {"ɐ", "ə"}, {"ɬ", "l"}, {"̃", ""},
    }};
    // trim
    const auto not_space = [](unsigned char ch) { return std::isspace(ch) == 0; };
    ps.erase(ps.begin(), std::find_if(ps.begin(), ps.end(), not_space));
    ps.erase(std::find_if(ps.rbegin(), ps.rend(), not_space).base(), ps.end());

    for (const auto & [from, to] : kE2M) {
        replace_all(ps, from, to);
    }
    // re.sub(r'(\S)̩', r'ᵊ\1', ps) then drop any remaining syllabic
    size_t at = 0;
    while ((at = ps.find(kSyllabic, at)) != std::string::npos) {
        size_t start = at;
        while (start > 0 && (static_cast<unsigned char>(ps[start - 1]) & 0xC0U) == 0x80U) {
            --start;   // walk back over the UTF-8 continuation bytes
        }
        if (start == at || std::isspace(static_cast<unsigned char>(ps[start])) != 0) {
            ps.erase(at, kSyllabic.size());
            continue;
        }
        ps.erase(at, kSyllabic.size());
        ps.insert(start, "ᵊ");
        at = start + std::strlen("ᵊ");
    }
    replace_all(ps, "o^ʊ", "O");
    replace_all(ps, "ɜːɹ", "ɜɹ");
    replace_all(ps, "ɜː", "ɜɹ");
    replace_all(ps, "ɪə", "iə");
    replace_all(ps, "ː", "");
    replace_all(ps, "o", "ɔ");        // espeak < 1.52
    replace_all(ps, "ɾ", "T");        // version != '2.0'
    replace_all(ps, "ʔ", "t");
    replace_all(ps, "^", "");
    return ps;
}


// ---- phonemizer-fork punctuation preserve/restore ------------------------
//
// The reference front ends run eSpeak through phonemizer with
// preserve_punctuation=True: punctuation is cut out before phonemization and
// spliced back afterwards, so marks like "," and "." survive as tokens (the
// model was trained with them). This reproduces phonemizer's Punctuation
// class for the fixed marks and separator this model uses.

constexpr std::string_view kPunctuationMarks = "!'(),-.:;?\"";

bool is_punctuation_mark(char ch) {
    return kPunctuationMarks.find(ch) != std::string_view::npos;
}

bool is_ascii_space(char ch) {
    return std::isspace(static_cast<unsigned char>(ch)) != 0;
}

struct MarkIndex {
    std::string mark;
    char position = 'I';   // B(egin), E(nd), I(nside), A(lone)
};

/** Matches of phonemizer's (\s*[marks]+\s*)+ -- maximal runs of spaces and
 *  marks that contain at least one mark. */
std::vector<std::pair<size_t, size_t>> find_mark_runs(const std::string & line) {
    std::vector<std::pair<size_t, size_t>> runs;
    size_t i = 0;
    while (i < line.size()) {
        if (!is_ascii_space(line[i]) && !is_punctuation_mark(line[i])) {
            ++i;
            continue;
        }
        size_t end = i;
        bool has_mark = false;
        while (end < line.size() &&
               (is_ascii_space(line[end]) || is_punctuation_mark(line[end]))) {
            has_mark = has_mark || is_punctuation_mark(line[end]);
            ++end;
        }
        if (has_mark) {
            runs.emplace_back(i, end);
        }
        i = end;
    }
    return runs;
}

/** Punctuation._preserve_line: chunks without punctuation + ordered marks.
 *  Empty chunks are filtered, as Punctuation.preserve() does. */
std::pair<std::vector<std::string>, std::vector<MarkIndex>> preserve_punctuation(
    const std::string & line) {
    const auto runs = find_mark_runs(line);
    if (runs.empty()) {
        return {{line}, {}};
    }
    if (runs.size() == 1 && runs[0].first == 0 && runs[0].second == line.size()) {
        return {{}, {{line, 'A'}}};
    }
    std::vector<MarkIndex> marks;
    marks.reserve(runs.size());
    for (size_t index = 0; index < runs.size(); ++index) {
        const auto & run = runs[index];
        char position = 'I';
        if (index == 0 && run.first == 0) {
            position = 'B';
        } else if (index + 1 == runs.size() && run.second == line.size()) {
            position = 'E';
        }
        marks.push_back({line.substr(run.first, run.second - run.first), position});
    }
    // The find-first split dance, exactly as the Python does it.
    std::vector<std::string> chunks;
    std::string rest = line;
    for (const auto & mark : marks) {
        const size_t at = rest.find(mark.mark);
        if (at == std::string::npos) {
            chunks.push_back(rest);
            rest.clear();
            continue;
        }
        chunks.push_back(rest.substr(0, at));
        rest.erase(0, at + mark.mark.size());
    }
    chunks.push_back(rest);
    chunks.erase(
        std::remove_if(chunks.begin(), chunks.end(),
                       [](const std::string & chunk) { return chunk.empty(); }),
        chunks.end());
    return {std::move(chunks), std::move(marks)};
}

/** Punctuation.restore for a single line, sep.word = " ", strip = False. */
std::string restore_punctuation(
    std::vector<std::string> chunk_phonemes,
    std::vector<MarkIndex> marks) {
    std::deque<std::string> text(chunk_phonemes.begin(), chunk_phonemes.end());
    std::deque<MarkIndex> pending(marks.begin(), marks.end());
    std::vector<std::string> out;
    size_t pos = 0;
    while (!text.empty() || !pending.empty()) {
        if (pending.empty()) {
            for (auto & line : text) {
                if (line.empty() || line.back() != ' ') {
                    line.push_back(' ');
                }
                out.push_back(std::move(line));
            }
            text.clear();
        } else if (text.empty()) {
            std::string joined;
            for (const auto & mark : pending) {
                joined += mark.mark;
            }
            out.push_back(std::move(joined));
            pending.clear();
        } else if (pos == 0) {   // single line: every mark carries index 0
            const auto current = pending.front();
            pending.pop_front();
            if (!text.front().empty() && text.front().back() == ' ') {
                text.front().pop_back();
            }
            const bool mark_ends_with_sep =
                !current.mark.empty() && current.mark.back() == ' ';
            if (current.position == 'B') {
                text.front() = current.mark + text.front();
            } else if (current.position == 'E') {
                out.push_back(text.front() + current.mark + (mark_ends_with_sep ? "" : " "));
                text.pop_front();
                ++pos;
            } else if (current.position == 'A') {
                out.push_back(current.mark + (mark_ends_with_sep ? "" : " "));
                ++pos;
            } else {   // 'I'
                if (text.size() == 1) {
                    text.front() += current.mark;
                } else {
                    auto first = std::move(text.front());
                    text.pop_front();
                    text.front() = first + current.mark + text.front();
                }
            }
        } else {
            auto & line = text.front();
            if (line.empty() || line.back() != ' ') {
                line.push_back(' ');
            }
            out.push_back(std::move(line));
            text.pop_front();
            ++pos;
        }
    }
    // phonemizer would return these as separate lines and the reference
    // takes the first; a single input line produces one in practice.
    std::string result;
    for (const auto & line : out) {
        result += line;
    }
    return result;
}

/** phonemizer EspeakBackend._postprocess_line with tie enabled,
 *  with_stress=True, strip=False, word separator " ", phone separator "". */
std::string postprocess_espeak_line(std::string line) {
    const auto not_space = [](unsigned char ch) { return std::isspace(ch) == 0; };
    line.erase(line.begin(), std::find_if(line.begin(), line.end(), not_space));
    line.erase(std::find_if(line.rbegin(), line.rend(), not_space).base(), line.end());
    std::replace(line.begin(), line.end(), '\n', ' ');
    replace_all(line, "  ", " ");
    // espeak-ng#694: stray '_' separators at word ends
    std::string squeezed;
    squeezed.reserve(line.size());
    for (const char ch : line) {
        if (ch == '_' && !squeezed.empty() && squeezed.back() == '_') {
            continue;
        }
        squeezed.push_back(ch);
    }
    line = std::move(squeezed);
    replace_all(line, "_ ", " ");
    // language_switch="remove-flags": strip espeak's (lang) switch flags
    if (line.find('(') != std::string::npos) {
        std::string unflagged;
        size_t at = 0;
        while (at < line.size()) {
            if (line[at] == '(') {
                const size_t close = line.find(')', at + 1);
                if (close != std::string::npos) {
                    at = close + 1;
                    continue;
                }
            }
            unflagged.push_back(line[at++]);
        }
        line = std::move(unflagged);
    }
    if (line.empty()) {
        return line;
    }
    // per word: strip, drop in-word '_' (phone separator is empty), append " "
    std::string out;
    size_t start = 0;
    while (start <= line.size()) {
        size_t end = line.find(' ', start);
        if (end == std::string::npos) {
            end = line.size();
        }
        std::string word = line.substr(start, end - start);
        word.erase(std::remove(word.begin(), word.end(), '_'), word.end());
        out += word;
        out.push_back(' ');
        if (end == line.size()) {
            break;
        }
        start = end + 1;
    }
    return out;
}

/** The reference front end's own line post-processing: rewrite eSpeak's tie
 *  to '^' per word so the E2M diphthong patterns can match. */
std::string rewrite_ties_per_word(const std::string & line_in) {
    std::string line = line_in;
    const auto not_space = [](unsigned char ch) { return std::isspace(ch) == 0; };
    line.erase(line.begin(), std::find_if(line.begin(), line.end(), not_space));
    line.erase(std::find_if(line.rbegin(), line.rend(), not_space).base(), line.end());
    std::replace(line.begin(), line.end(), '\n', ' ');
    replace_all(line, "  ", " ");
    if (line.empty()) {
        return line;
    }
    std::string out;
    size_t start = 0;
    while (start <= line.size()) {
        size_t end = line.find(' ', start);
        if (end == std::string::npos) {
            end = line.size();
        }
        std::string word = line.substr(start, end - start);
        replace_all(word, kTieDefault, kTieMisaki);
        out += word;
        out.push_back(' ');
        if (end == line.size()) {
            break;
        }
        start = end + 1;
    }
    return out;
}


// ---- NFD decomposition for the Piper codepoint mapping -------------------
//
// piper normalizes the phonemized string with NFD before mapping each
// codepoint through phoneme_id_map. eSpeak-ng's IPA output for the shipped
// languages is already NFD-normal (probed over en/vi/id corpora), so this
// table only has to cover precomposed Latin letters that could slip through;
// characters outside it pass through unchanged, and an unmapped codepoint is
// skipped exactly as piper skips it. The framework's NFKD normalizer is NOT
// usable here: compatibility decomposition rewrites IPA modifier letters.

// Canonical (NFD) decompositions for Latin-1/Extended and Vietnamese
// precomposed letters -- everything eSpeak-ng can plausibly emit in IPA
// mode. Generated from Python unicodedata (Unicode 15); canonical
// decompositions are stable across Unicode versions by policy.
struct NfdEntry { uint32_t composed; uint32_t parts[3]; };
constexpr NfdEntry kNfdEntries[] = {
    {0x00C0, {0x0041, 0x0300, 0x0000}},
    {0x00C1, {0x0041, 0x0301, 0x0000}},
    {0x00C2, {0x0041, 0x0302, 0x0000}},
    {0x00C3, {0x0041, 0x0303, 0x0000}},
    {0x00C4, {0x0041, 0x0308, 0x0000}},
    {0x00C5, {0x0041, 0x030A, 0x0000}},
    {0x00C7, {0x0043, 0x0327, 0x0000}},
    {0x00C8, {0x0045, 0x0300, 0x0000}},
    {0x00C9, {0x0045, 0x0301, 0x0000}},
    {0x00CA, {0x0045, 0x0302, 0x0000}},
    {0x00CB, {0x0045, 0x0308, 0x0000}},
    {0x00CC, {0x0049, 0x0300, 0x0000}},
    {0x00CD, {0x0049, 0x0301, 0x0000}},
    {0x00CE, {0x0049, 0x0302, 0x0000}},
    {0x00CF, {0x0049, 0x0308, 0x0000}},
    {0x00D1, {0x004E, 0x0303, 0x0000}},
    {0x00D2, {0x004F, 0x0300, 0x0000}},
    {0x00D3, {0x004F, 0x0301, 0x0000}},
    {0x00D4, {0x004F, 0x0302, 0x0000}},
    {0x00D5, {0x004F, 0x0303, 0x0000}},
    {0x00D6, {0x004F, 0x0308, 0x0000}},
    {0x00D9, {0x0055, 0x0300, 0x0000}},
    {0x00DA, {0x0055, 0x0301, 0x0000}},
    {0x00DB, {0x0055, 0x0302, 0x0000}},
    {0x00DC, {0x0055, 0x0308, 0x0000}},
    {0x00DD, {0x0059, 0x0301, 0x0000}},
    {0x00E0, {0x0061, 0x0300, 0x0000}},
    {0x00E1, {0x0061, 0x0301, 0x0000}},
    {0x00E2, {0x0061, 0x0302, 0x0000}},
    {0x00E3, {0x0061, 0x0303, 0x0000}},
    {0x00E4, {0x0061, 0x0308, 0x0000}},
    {0x00E5, {0x0061, 0x030A, 0x0000}},
    {0x00E7, {0x0063, 0x0327, 0x0000}},
    {0x00E8, {0x0065, 0x0300, 0x0000}},
    {0x00E9, {0x0065, 0x0301, 0x0000}},
    {0x00EA, {0x0065, 0x0302, 0x0000}},
    {0x00EB, {0x0065, 0x0308, 0x0000}},
    {0x00EC, {0x0069, 0x0300, 0x0000}},
    {0x00ED, {0x0069, 0x0301, 0x0000}},
    {0x00EE, {0x0069, 0x0302, 0x0000}},
    {0x00EF, {0x0069, 0x0308, 0x0000}},
    {0x00F1, {0x006E, 0x0303, 0x0000}},
    {0x00F2, {0x006F, 0x0300, 0x0000}},
    {0x00F3, {0x006F, 0x0301, 0x0000}},
    {0x00F4, {0x006F, 0x0302, 0x0000}},
    {0x00F5, {0x006F, 0x0303, 0x0000}},
    {0x00F6, {0x006F, 0x0308, 0x0000}},
    {0x00F9, {0x0075, 0x0300, 0x0000}},
    {0x00FA, {0x0075, 0x0301, 0x0000}},
    {0x00FB, {0x0075, 0x0302, 0x0000}},
    {0x00FC, {0x0075, 0x0308, 0x0000}},
    {0x00FD, {0x0079, 0x0301, 0x0000}},
    {0x00FF, {0x0079, 0x0308, 0x0000}},
    {0x0100, {0x0041, 0x0304, 0x0000}},
    {0x0101, {0x0061, 0x0304, 0x0000}},
    {0x0102, {0x0041, 0x0306, 0x0000}},
    {0x0103, {0x0061, 0x0306, 0x0000}},
    {0x0104, {0x0041, 0x0328, 0x0000}},
    {0x0105, {0x0061, 0x0328, 0x0000}},
    {0x0106, {0x0043, 0x0301, 0x0000}},
    {0x0107, {0x0063, 0x0301, 0x0000}},
    {0x0108, {0x0043, 0x0302, 0x0000}},
    {0x0109, {0x0063, 0x0302, 0x0000}},
    {0x010A, {0x0043, 0x0307, 0x0000}},
    {0x010B, {0x0063, 0x0307, 0x0000}},
    {0x010C, {0x0043, 0x030C, 0x0000}},
    {0x010D, {0x0063, 0x030C, 0x0000}},
    {0x010E, {0x0044, 0x030C, 0x0000}},
    {0x010F, {0x0064, 0x030C, 0x0000}},
    {0x0112, {0x0045, 0x0304, 0x0000}},
    {0x0113, {0x0065, 0x0304, 0x0000}},
    {0x0114, {0x0045, 0x0306, 0x0000}},
    {0x0115, {0x0065, 0x0306, 0x0000}},
    {0x0116, {0x0045, 0x0307, 0x0000}},
    {0x0117, {0x0065, 0x0307, 0x0000}},
    {0x0118, {0x0045, 0x0328, 0x0000}},
    {0x0119, {0x0065, 0x0328, 0x0000}},
    {0x011A, {0x0045, 0x030C, 0x0000}},
    {0x011B, {0x0065, 0x030C, 0x0000}},
    {0x011C, {0x0047, 0x0302, 0x0000}},
    {0x011D, {0x0067, 0x0302, 0x0000}},
    {0x011E, {0x0047, 0x0306, 0x0000}},
    {0x011F, {0x0067, 0x0306, 0x0000}},
    {0x0120, {0x0047, 0x0307, 0x0000}},
    {0x0121, {0x0067, 0x0307, 0x0000}},
    {0x0122, {0x0047, 0x0327, 0x0000}},
    {0x0123, {0x0067, 0x0327, 0x0000}},
    {0x0124, {0x0048, 0x0302, 0x0000}},
    {0x0125, {0x0068, 0x0302, 0x0000}},
    {0x0128, {0x0049, 0x0303, 0x0000}},
    {0x0129, {0x0069, 0x0303, 0x0000}},
    {0x012A, {0x0049, 0x0304, 0x0000}},
    {0x012B, {0x0069, 0x0304, 0x0000}},
    {0x012C, {0x0049, 0x0306, 0x0000}},
    {0x012D, {0x0069, 0x0306, 0x0000}},
    {0x012E, {0x0049, 0x0328, 0x0000}},
    {0x012F, {0x0069, 0x0328, 0x0000}},
    {0x0130, {0x0049, 0x0307, 0x0000}},
    {0x0134, {0x004A, 0x0302, 0x0000}},
    {0x0135, {0x006A, 0x0302, 0x0000}},
    {0x0136, {0x004B, 0x0327, 0x0000}},
    {0x0137, {0x006B, 0x0327, 0x0000}},
    {0x0139, {0x004C, 0x0301, 0x0000}},
    {0x013A, {0x006C, 0x0301, 0x0000}},
    {0x013B, {0x004C, 0x0327, 0x0000}},
    {0x013C, {0x006C, 0x0327, 0x0000}},
    {0x013D, {0x004C, 0x030C, 0x0000}},
    {0x013E, {0x006C, 0x030C, 0x0000}},
    {0x0143, {0x004E, 0x0301, 0x0000}},
    {0x0144, {0x006E, 0x0301, 0x0000}},
    {0x0145, {0x004E, 0x0327, 0x0000}},
    {0x0146, {0x006E, 0x0327, 0x0000}},
    {0x0147, {0x004E, 0x030C, 0x0000}},
    {0x0148, {0x006E, 0x030C, 0x0000}},
    {0x014C, {0x004F, 0x0304, 0x0000}},
    {0x014D, {0x006F, 0x0304, 0x0000}},
    {0x014E, {0x004F, 0x0306, 0x0000}},
    {0x014F, {0x006F, 0x0306, 0x0000}},
    {0x0150, {0x004F, 0x030B, 0x0000}},
    {0x0151, {0x006F, 0x030B, 0x0000}},
    {0x0154, {0x0052, 0x0301, 0x0000}},
    {0x0155, {0x0072, 0x0301, 0x0000}},
    {0x0156, {0x0052, 0x0327, 0x0000}},
    {0x0157, {0x0072, 0x0327, 0x0000}},
    {0x0158, {0x0052, 0x030C, 0x0000}},
    {0x0159, {0x0072, 0x030C, 0x0000}},
    {0x015A, {0x0053, 0x0301, 0x0000}},
    {0x015B, {0x0073, 0x0301, 0x0000}},
    {0x015C, {0x0053, 0x0302, 0x0000}},
    {0x015D, {0x0073, 0x0302, 0x0000}},
    {0x015E, {0x0053, 0x0327, 0x0000}},
    {0x015F, {0x0073, 0x0327, 0x0000}},
    {0x0160, {0x0053, 0x030C, 0x0000}},
    {0x0161, {0x0073, 0x030C, 0x0000}},
    {0x0162, {0x0054, 0x0327, 0x0000}},
    {0x0163, {0x0074, 0x0327, 0x0000}},
    {0x0164, {0x0054, 0x030C, 0x0000}},
    {0x0165, {0x0074, 0x030C, 0x0000}},
    {0x0168, {0x0055, 0x0303, 0x0000}},
    {0x0169, {0x0075, 0x0303, 0x0000}},
    {0x016A, {0x0055, 0x0304, 0x0000}},
    {0x016B, {0x0075, 0x0304, 0x0000}},
    {0x016C, {0x0055, 0x0306, 0x0000}},
    {0x016D, {0x0075, 0x0306, 0x0000}},
    {0x016E, {0x0055, 0x030A, 0x0000}},
    {0x016F, {0x0075, 0x030A, 0x0000}},
    {0x0170, {0x0055, 0x030B, 0x0000}},
    {0x0171, {0x0075, 0x030B, 0x0000}},
    {0x0172, {0x0055, 0x0328, 0x0000}},
    {0x0173, {0x0075, 0x0328, 0x0000}},
    {0x0174, {0x0057, 0x0302, 0x0000}},
    {0x0175, {0x0077, 0x0302, 0x0000}},
    {0x0176, {0x0059, 0x0302, 0x0000}},
    {0x0177, {0x0079, 0x0302, 0x0000}},
    {0x0178, {0x0059, 0x0308, 0x0000}},
    {0x0179, {0x005A, 0x0301, 0x0000}},
    {0x017A, {0x007A, 0x0301, 0x0000}},
    {0x017B, {0x005A, 0x0307, 0x0000}},
    {0x017C, {0x007A, 0x0307, 0x0000}},
    {0x017D, {0x005A, 0x030C, 0x0000}},
    {0x017E, {0x007A, 0x030C, 0x0000}},
    {0x1E00, {0x0041, 0x0325, 0x0000}},
    {0x1E01, {0x0061, 0x0325, 0x0000}},
    {0x1E02, {0x0042, 0x0307, 0x0000}},
    {0x1E03, {0x0062, 0x0307, 0x0000}},
    {0x1E04, {0x0042, 0x0323, 0x0000}},
    {0x1E05, {0x0062, 0x0323, 0x0000}},
    {0x1E06, {0x0042, 0x0331, 0x0000}},
    {0x1E07, {0x0062, 0x0331, 0x0000}},
    {0x1E08, {0x0043, 0x0327, 0x0301}},
    {0x1E09, {0x0063, 0x0327, 0x0301}},
    {0x1E0A, {0x0044, 0x0307, 0x0000}},
    {0x1E0B, {0x0064, 0x0307, 0x0000}},
    {0x1E0C, {0x0044, 0x0323, 0x0000}},
    {0x1E0D, {0x0064, 0x0323, 0x0000}},
    {0x1E0E, {0x0044, 0x0331, 0x0000}},
    {0x1E0F, {0x0064, 0x0331, 0x0000}},
    {0x1E10, {0x0044, 0x0327, 0x0000}},
    {0x1E11, {0x0064, 0x0327, 0x0000}},
    {0x1E12, {0x0044, 0x032D, 0x0000}},
    {0x1E13, {0x0064, 0x032D, 0x0000}},
    {0x1E14, {0x0045, 0x0304, 0x0300}},
    {0x1E15, {0x0065, 0x0304, 0x0300}},
    {0x1E16, {0x0045, 0x0304, 0x0301}},
    {0x1E17, {0x0065, 0x0304, 0x0301}},
    {0x1E18, {0x0045, 0x032D, 0x0000}},
    {0x1E19, {0x0065, 0x032D, 0x0000}},
    {0x1E1A, {0x0045, 0x0330, 0x0000}},
    {0x1E1B, {0x0065, 0x0330, 0x0000}},
    {0x1E1C, {0x0045, 0x0327, 0x0306}},
    {0x1E1D, {0x0065, 0x0327, 0x0306}},
    {0x1E1E, {0x0046, 0x0307, 0x0000}},
    {0x1E1F, {0x0066, 0x0307, 0x0000}},
    {0x1E20, {0x0047, 0x0304, 0x0000}},
    {0x1E21, {0x0067, 0x0304, 0x0000}},
    {0x1E22, {0x0048, 0x0307, 0x0000}},
    {0x1E23, {0x0068, 0x0307, 0x0000}},
    {0x1E24, {0x0048, 0x0323, 0x0000}},
    {0x1E25, {0x0068, 0x0323, 0x0000}},
    {0x1E26, {0x0048, 0x0308, 0x0000}},
    {0x1E27, {0x0068, 0x0308, 0x0000}},
    {0x1E28, {0x0048, 0x0327, 0x0000}},
    {0x1E29, {0x0068, 0x0327, 0x0000}},
    {0x1E2A, {0x0048, 0x032E, 0x0000}},
    {0x1E2B, {0x0068, 0x032E, 0x0000}},
    {0x1E2C, {0x0049, 0x0330, 0x0000}},
    {0x1E2D, {0x0069, 0x0330, 0x0000}},
    {0x1E2E, {0x0049, 0x0308, 0x0301}},
    {0x1E2F, {0x0069, 0x0308, 0x0301}},
    {0x1E30, {0x004B, 0x0301, 0x0000}},
    {0x1E31, {0x006B, 0x0301, 0x0000}},
    {0x1E32, {0x004B, 0x0323, 0x0000}},
    {0x1E33, {0x006B, 0x0323, 0x0000}},
    {0x1E34, {0x004B, 0x0331, 0x0000}},
    {0x1E35, {0x006B, 0x0331, 0x0000}},
    {0x1E36, {0x004C, 0x0323, 0x0000}},
    {0x1E37, {0x006C, 0x0323, 0x0000}},
    {0x1E38, {0x004C, 0x0323, 0x0304}},
    {0x1E39, {0x006C, 0x0323, 0x0304}},
    {0x1E3A, {0x004C, 0x0331, 0x0000}},
    {0x1E3B, {0x006C, 0x0331, 0x0000}},
    {0x1E3C, {0x004C, 0x032D, 0x0000}},
    {0x1E3D, {0x006C, 0x032D, 0x0000}},
    {0x1E3E, {0x004D, 0x0301, 0x0000}},
    {0x1E3F, {0x006D, 0x0301, 0x0000}},
    {0x1E40, {0x004D, 0x0307, 0x0000}},
    {0x1E41, {0x006D, 0x0307, 0x0000}},
    {0x1E42, {0x004D, 0x0323, 0x0000}},
    {0x1E43, {0x006D, 0x0323, 0x0000}},
    {0x1E44, {0x004E, 0x0307, 0x0000}},
    {0x1E45, {0x006E, 0x0307, 0x0000}},
    {0x1E46, {0x004E, 0x0323, 0x0000}},
    {0x1E47, {0x006E, 0x0323, 0x0000}},
    {0x1E48, {0x004E, 0x0331, 0x0000}},
    {0x1E49, {0x006E, 0x0331, 0x0000}},
    {0x1E4A, {0x004E, 0x032D, 0x0000}},
    {0x1E4B, {0x006E, 0x032D, 0x0000}},
    {0x1E4C, {0x004F, 0x0303, 0x0301}},
    {0x1E4D, {0x006F, 0x0303, 0x0301}},
    {0x1E4E, {0x004F, 0x0303, 0x0308}},
    {0x1E4F, {0x006F, 0x0303, 0x0308}},
    {0x1E50, {0x004F, 0x0304, 0x0300}},
    {0x1E51, {0x006F, 0x0304, 0x0300}},
    {0x1E52, {0x004F, 0x0304, 0x0301}},
    {0x1E53, {0x006F, 0x0304, 0x0301}},
    {0x1E54, {0x0050, 0x0301, 0x0000}},
    {0x1E55, {0x0070, 0x0301, 0x0000}},
    {0x1E56, {0x0050, 0x0307, 0x0000}},
    {0x1E57, {0x0070, 0x0307, 0x0000}},
    {0x1E58, {0x0052, 0x0307, 0x0000}},
    {0x1E59, {0x0072, 0x0307, 0x0000}},
    {0x1E5A, {0x0052, 0x0323, 0x0000}},
    {0x1E5B, {0x0072, 0x0323, 0x0000}},
    {0x1E5C, {0x0052, 0x0323, 0x0304}},
    {0x1E5D, {0x0072, 0x0323, 0x0304}},
    {0x1E5E, {0x0052, 0x0331, 0x0000}},
    {0x1E5F, {0x0072, 0x0331, 0x0000}},
    {0x1E60, {0x0053, 0x0307, 0x0000}},
    {0x1E61, {0x0073, 0x0307, 0x0000}},
    {0x1E62, {0x0053, 0x0323, 0x0000}},
    {0x1E63, {0x0073, 0x0323, 0x0000}},
    {0x1E64, {0x0053, 0x0301, 0x0307}},
    {0x1E65, {0x0073, 0x0301, 0x0307}},
    {0x1E66, {0x0053, 0x030C, 0x0307}},
    {0x1E67, {0x0073, 0x030C, 0x0307}},
    {0x1E68, {0x0053, 0x0323, 0x0307}},
    {0x1E69, {0x0073, 0x0323, 0x0307}},
    {0x1E6A, {0x0054, 0x0307, 0x0000}},
    {0x1E6B, {0x0074, 0x0307, 0x0000}},
    {0x1E6C, {0x0054, 0x0323, 0x0000}},
    {0x1E6D, {0x0074, 0x0323, 0x0000}},
    {0x1E6E, {0x0054, 0x0331, 0x0000}},
    {0x1E6F, {0x0074, 0x0331, 0x0000}},
    {0x1E70, {0x0054, 0x032D, 0x0000}},
    {0x1E71, {0x0074, 0x032D, 0x0000}},
    {0x1E72, {0x0055, 0x0324, 0x0000}},
    {0x1E73, {0x0075, 0x0324, 0x0000}},
    {0x1E74, {0x0055, 0x0330, 0x0000}},
    {0x1E75, {0x0075, 0x0330, 0x0000}},
    {0x1E76, {0x0055, 0x032D, 0x0000}},
    {0x1E77, {0x0075, 0x032D, 0x0000}},
    {0x1E78, {0x0055, 0x0303, 0x0301}},
    {0x1E79, {0x0075, 0x0303, 0x0301}},
    {0x1E7A, {0x0055, 0x0304, 0x0308}},
    {0x1E7B, {0x0075, 0x0304, 0x0308}},
    {0x1E7C, {0x0056, 0x0303, 0x0000}},
    {0x1E7D, {0x0076, 0x0303, 0x0000}},
    {0x1E7E, {0x0056, 0x0323, 0x0000}},
    {0x1E7F, {0x0076, 0x0323, 0x0000}},
    {0x1E80, {0x0057, 0x0300, 0x0000}},
    {0x1E81, {0x0077, 0x0300, 0x0000}},
    {0x1E82, {0x0057, 0x0301, 0x0000}},
    {0x1E83, {0x0077, 0x0301, 0x0000}},
    {0x1E84, {0x0057, 0x0308, 0x0000}},
    {0x1E85, {0x0077, 0x0308, 0x0000}},
    {0x1E86, {0x0057, 0x0307, 0x0000}},
    {0x1E87, {0x0077, 0x0307, 0x0000}},
    {0x1E88, {0x0057, 0x0323, 0x0000}},
    {0x1E89, {0x0077, 0x0323, 0x0000}},
    {0x1E8A, {0x0058, 0x0307, 0x0000}},
    {0x1E8B, {0x0078, 0x0307, 0x0000}},
    {0x1E8C, {0x0058, 0x0308, 0x0000}},
    {0x1E8D, {0x0078, 0x0308, 0x0000}},
    {0x1E8E, {0x0059, 0x0307, 0x0000}},
    {0x1E8F, {0x0079, 0x0307, 0x0000}},
    {0x1E90, {0x005A, 0x0302, 0x0000}},
    {0x1E91, {0x007A, 0x0302, 0x0000}},
    {0x1E92, {0x005A, 0x0323, 0x0000}},
    {0x1E93, {0x007A, 0x0323, 0x0000}},
    {0x1E94, {0x005A, 0x0331, 0x0000}},
    {0x1E95, {0x007A, 0x0331, 0x0000}},
    {0x1E96, {0x0068, 0x0331, 0x0000}},
    {0x1E97, {0x0074, 0x0308, 0x0000}},
    {0x1E98, {0x0077, 0x030A, 0x0000}},
    {0x1E99, {0x0079, 0x030A, 0x0000}},
    {0x1E9B, {0x017F, 0x0307, 0x0000}},
    {0x1EA0, {0x0041, 0x0323, 0x0000}},
    {0x1EA1, {0x0061, 0x0323, 0x0000}},
    {0x1EA2, {0x0041, 0x0309, 0x0000}},
    {0x1EA3, {0x0061, 0x0309, 0x0000}},
    {0x1EA4, {0x0041, 0x0302, 0x0301}},
    {0x1EA5, {0x0061, 0x0302, 0x0301}},
    {0x1EA6, {0x0041, 0x0302, 0x0300}},
    {0x1EA7, {0x0061, 0x0302, 0x0300}},
    {0x1EA8, {0x0041, 0x0302, 0x0309}},
    {0x1EA9, {0x0061, 0x0302, 0x0309}},
    {0x1EAA, {0x0041, 0x0302, 0x0303}},
    {0x1EAB, {0x0061, 0x0302, 0x0303}},
    {0x1EAC, {0x0041, 0x0323, 0x0302}},
    {0x1EAD, {0x0061, 0x0323, 0x0302}},
    {0x1EAE, {0x0041, 0x0306, 0x0301}},
    {0x1EAF, {0x0061, 0x0306, 0x0301}},
    {0x1EB0, {0x0041, 0x0306, 0x0300}},
    {0x1EB1, {0x0061, 0x0306, 0x0300}},
    {0x1EB2, {0x0041, 0x0306, 0x0309}},
    {0x1EB3, {0x0061, 0x0306, 0x0309}},
    {0x1EB4, {0x0041, 0x0306, 0x0303}},
    {0x1EB5, {0x0061, 0x0306, 0x0303}},
    {0x1EB6, {0x0041, 0x0323, 0x0306}},
    {0x1EB7, {0x0061, 0x0323, 0x0306}},
    {0x1EB8, {0x0045, 0x0323, 0x0000}},
    {0x1EB9, {0x0065, 0x0323, 0x0000}},
    {0x1EBA, {0x0045, 0x0309, 0x0000}},
    {0x1EBB, {0x0065, 0x0309, 0x0000}},
    {0x1EBC, {0x0045, 0x0303, 0x0000}},
    {0x1EBD, {0x0065, 0x0303, 0x0000}},
    {0x1EBE, {0x0045, 0x0302, 0x0301}},
    {0x1EBF, {0x0065, 0x0302, 0x0301}},
    {0x1EC0, {0x0045, 0x0302, 0x0300}},
    {0x1EC1, {0x0065, 0x0302, 0x0300}},
    {0x1EC2, {0x0045, 0x0302, 0x0309}},
    {0x1EC3, {0x0065, 0x0302, 0x0309}},
    {0x1EC4, {0x0045, 0x0302, 0x0303}},
    {0x1EC5, {0x0065, 0x0302, 0x0303}},
    {0x1EC6, {0x0045, 0x0323, 0x0302}},
    {0x1EC7, {0x0065, 0x0323, 0x0302}},
    {0x1EC8, {0x0049, 0x0309, 0x0000}},
    {0x1EC9, {0x0069, 0x0309, 0x0000}},
    {0x1ECA, {0x0049, 0x0323, 0x0000}},
    {0x1ECB, {0x0069, 0x0323, 0x0000}},
    {0x1ECC, {0x004F, 0x0323, 0x0000}},
    {0x1ECD, {0x006F, 0x0323, 0x0000}},
    {0x1ECE, {0x004F, 0x0309, 0x0000}},
    {0x1ECF, {0x006F, 0x0309, 0x0000}},
    {0x1ED0, {0x004F, 0x0302, 0x0301}},
    {0x1ED1, {0x006F, 0x0302, 0x0301}},
    {0x1ED2, {0x004F, 0x0302, 0x0300}},
    {0x1ED3, {0x006F, 0x0302, 0x0300}},
    {0x1ED4, {0x004F, 0x0302, 0x0309}},
    {0x1ED5, {0x006F, 0x0302, 0x0309}},
    {0x1ED6, {0x004F, 0x0302, 0x0303}},
    {0x1ED7, {0x006F, 0x0302, 0x0303}},
    {0x1ED8, {0x004F, 0x0323, 0x0302}},
    {0x1ED9, {0x006F, 0x0323, 0x0302}},
    {0x1EDA, {0x004F, 0x031B, 0x0301}},
    {0x1EDB, {0x006F, 0x031B, 0x0301}},
    {0x1EDC, {0x004F, 0x031B, 0x0300}},
    {0x1EDD, {0x006F, 0x031B, 0x0300}},
    {0x1EDE, {0x004F, 0x031B, 0x0309}},
    {0x1EDF, {0x006F, 0x031B, 0x0309}},
    {0x1EE0, {0x004F, 0x031B, 0x0303}},
    {0x1EE1, {0x006F, 0x031B, 0x0303}},
    {0x1EE2, {0x004F, 0x031B, 0x0323}},
    {0x1EE3, {0x006F, 0x031B, 0x0323}},
    {0x1EE4, {0x0055, 0x0323, 0x0000}},
    {0x1EE5, {0x0075, 0x0323, 0x0000}},
    {0x1EE6, {0x0055, 0x0309, 0x0000}},
    {0x1EE7, {0x0075, 0x0309, 0x0000}},
    {0x1EE8, {0x0055, 0x031B, 0x0301}},
    {0x1EE9, {0x0075, 0x031B, 0x0301}},
    {0x1EEA, {0x0055, 0x031B, 0x0300}},
    {0x1EEB, {0x0075, 0x031B, 0x0300}},
    {0x1EEC, {0x0055, 0x031B, 0x0309}},
    {0x1EED, {0x0075, 0x031B, 0x0309}},
    {0x1EEE, {0x0055, 0x031B, 0x0303}},
    {0x1EEF, {0x0075, 0x031B, 0x0303}},
    {0x1EF0, {0x0055, 0x031B, 0x0323}},
    {0x1EF1, {0x0075, 0x031B, 0x0323}},
    {0x1EF2, {0x0059, 0x0300, 0x0000}},
    {0x1EF3, {0x0079, 0x0300, 0x0000}},
    {0x1EF4, {0x0059, 0x0323, 0x0000}},
    {0x1EF5, {0x0079, 0x0323, 0x0000}},
    {0x1EF6, {0x0059, 0x0309, 0x0000}},
    {0x1EF7, {0x0079, 0x0309, 0x0000}},
    {0x1EF8, {0x0059, 0x0303, 0x0000}},
    {0x1EF9, {0x0079, 0x0303, 0x0000}},
};


void append_codepoint_utf8(uint32_t codepoint, std::string & out) {
    if (codepoint < 0x80U) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint < 0x800U) {
        out.push_back(static_cast<char>(0xC0U | (codepoint >> 6)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint < 0x10000U) {
        out.push_back(static_cast<char>(0xE0U | (codepoint >> 12)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
        out.push_back(static_cast<char>(0xF0U | (codepoint >> 18)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 12) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
}

/** Iterates whole UTF-8 codepoints; invalid lead bytes pass through as one. */
size_t utf8_sequence_length(const std::string & text, size_t at) {
    const auto lead = static_cast<unsigned char>(text[at]);
    size_t len = 1;
    if ((lead & 0xF8U) == 0xF0U) { len = 4; }
    else if ((lead & 0xF0U) == 0xE0U) { len = 3; }
    else if ((lead & 0xE0U) == 0xC0U) { len = 2; }
    return std::min(len, text.size() - at);
}

uint32_t decode_codepoint_utf8(const std::string & text, size_t at, size_t len) {
    const auto lead = static_cast<unsigned char>(text[at]);
    if (len == 1) {
        return lead;
    }
    uint32_t value = lead & (0x7FU >> len);
    for (size_t i = 1; i < len; ++i) {
        value = (value << 6) | (static_cast<unsigned char>(text[at + i]) & 0x3FU);
    }
    return value;
}

std::string nfd_decompose(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        const size_t len = utf8_sequence_length(text, i);
        const uint32_t codepoint = decode_codepoint_utf8(text, i, len);
        const auto * entry = std::lower_bound(
            std::begin(kNfdEntries),
            std::end(kNfdEntries),
            codepoint,
            [](const NfdEntry & candidate, uint32_t value) {
                return candidate.composed < value;
            });
        if (entry != std::end(kNfdEntries) && entry->composed == codepoint) {
            for (const uint32_t part : entry->parts) {
                if (part != 0) {
                    append_codepoint_utf8(part, out);
                }
            }
        } else {
            out.append(text, i, len);
        }
        i += len;
    }
    return out;
}

struct EspeakApi {
    io::DynamicLibraryHandle library = nullptr;
    InitializeFn initialize = nullptr;
    SetVoiceFn set_voice = nullptr;
    TextToPhonemesFn text_to_phonemes = nullptr;
    TerminateFn terminate = nullptr;
    mutable std::mutex call_mutex;

    EspeakApi(const std::filesystem::path & requested_library,
              const std::filesystem::path & requested_data,
              const std::string & voice) {
        if (!requested_library.empty() &&
            !std::filesystem::is_regular_file(requested_library)) {
            throw std::runtime_error(
                "sanoTTS eSpeak-ng library does not exist: " + requested_library.string());
        }
        if (!requested_data.empty() &&
            (!std::filesystem::is_directory(requested_data) ||
             !std::filesystem::is_regular_file(requested_data / "phontab"))) {
            throw std::runtime_error(
                "sanoTTS eSpeak-ng data path is invalid; expected the espeak-ng-data "
                "directory containing phontab: " + requested_data.string());
        }
        if (!requested_library.empty()) {
            library = io::open_dynamic_library(requested_library.string());
        } else {
            library = io::open_dynamic_library({
#ifdef _WIN32
                "espeak-ng.dll", "libespeak-ng.dll",
#elif defined(__APPLE__)
                "libespeak-ng.dylib", "libespeak-ng.1.dylib",
#else
                "libespeak-ng.so.1", "libespeak-ng.so",
#endif
            });
        }
        if (library == nullptr) {
            throw std::runtime_error(
                "sanoTTS could not load eSpeak-ng. Install it (apt install espeak-ng, "
                "brew install espeak-ng) or pass "
                "--session-option sanotts.espeak_library_path=/path/to/libespeak-ng.so");
        }
        initialize = reinterpret_cast<InitializeFn>(
            io::dynamic_library_symbol(library, "espeak_Initialize"));
        set_voice = reinterpret_cast<SetVoiceFn>(
            io::dynamic_library_symbol(library, "espeak_SetVoiceByName"));
        text_to_phonemes = reinterpret_cast<TextToPhonemesFn>(
            io::dynamic_library_symbol(library, "espeak_TextToPhonemes"));
        terminate = reinterpret_cast<TerminateFn>(
            io::dynamic_library_symbol(library, "espeak_Terminate"));
        if (initialize == nullptr || set_voice == nullptr || text_to_phonemes == nullptr) {
            throw std::runtime_error("sanoTTS eSpeak-ng is missing required symbols");
        }
        // espeak appends "/espeak-ng-data" to the path it is given, so the
        // PARENT of the data directory is what it wants. Handing it the data
        // directory itself makes it fall back to its compiled-in default.
        const std::string data =
            requested_data.empty() ? std::string() : requested_data.parent_path().string();
        if (initialize(kEspeakSynchronous, 0, data.empty() ? nullptr : data.c_str(), 0) <= 0) {
            throw std::runtime_error(
                "sanoTTS eSpeak-ng failed to initialize; pass "
                "--session-option sanotts.espeak_data_path=/path/to/espeak-ng-data");
        }
        // Some packages name a bare language code ("en"). phonemizer, which
        // the reference front end drives, rejects bare codes on every
        // espeak-ng >= 1.49 and falls back to the regional variant, even
        // though espeak_SetVoiceByName itself would accept "en" (and select
        // a different accent). Prefer the regional variants first so both
        // stacks phonemize identically; a code with no variant (vi, id)
        // falls through to itself.
        std::vector<std::string> candidates;
        if (voice.find('-') == std::string::npos) {
            candidates.push_back(voice + "-us");
            candidates.push_back(voice + "-gb");
        }
        candidates.push_back(voice);
        bool selected = false;
        for (const auto & candidate : candidates) {
            if (set_voice(candidate.c_str()) == 0) {
                selected = true;
                break;
            }
        }
        if (!selected) {
            throw std::runtime_error("sanoTTS eSpeak-ng has no voice matching '" + voice + "'");
        }
    }

    ~EspeakApi() {
        if (terminate != nullptr) {
            terminate();
        }
        if (library != nullptr) {
            io::close_dynamic_library(library);
        }
    }

    [[nodiscard]] std::string phonemize(const std::string & text, int phonemes_mode) const {
        const std::lock_guard<std::mutex> guard(call_mutex);
        std::string out;
        const char * cursor = text.c_str();
        const void * position = cursor;
        // espeak consumes one clause per call and advances the pointer; it
        // returns null when the input is spent.
        while (position != nullptr) {
            const char * clause =
                text_to_phonemes(&position, kEspeakCharsUtf8, phonemes_mode);
            if (clause == nullptr) {
                break;
            }
            if (!out.empty()) {
                out.push_back(' ');
            }
            out.append(clause);
        }
        return out;
    }
};

}  // namespace

struct SanoTtsFrontend::Impl {
    EspeakApi espeak;
    Impl(const std::filesystem::path & library, const std::filesystem::path & data)
        : espeak(library, data, "en-us") {}
};

SanoTtsFrontend::SanoTtsFrontend(
    std::filesystem::path espeak_library_path,
    std::filesystem::path espeak_data_path,
    int64_t max_tokens)
    : impl_(std::make_unique<Impl>(espeak_library_path, espeak_data_path)),
      max_tokens_(max_tokens > 2 ? max_tokens : 207) {}

SanoTtsFrontend::~SanoTtsFrontend() = default;

SanoTtsEncoded SanoTtsFrontend::encode(const std::string & text) const {
    auto [chunks, marks] = preserve_punctuation(text);
    std::vector<std::string> chunk_phonemes;
    chunk_phonemes.reserve(chunks.size());
    for (const auto & chunk : chunks) {
        chunk_phonemes.push_back(postprocess_espeak_line(
            impl_->espeak.phonemize(chunk, kEspeakPhonemesIpaTie)));
    }
    const std::string restored =
        restore_punctuation(std::move(chunk_phonemes), std::move(marks));
    // Ties become '^' BEFORE E2M runs: every diphthong pattern in the table
    // ("o^ʊ" -> "O", "t^ʃ" -> "ʧ", ...) matches on the rewritten form.
    std::string ipa = apply_e2m(rewrite_ties_per_word(restored));

    const auto & vocab = vocabulary();
    SanoTtsEncoded out;
    out.token_ids.push_back(1);   // <bos>
    // Iterate whole UTF-8 codepoints: every vocabulary symbol is one
    // codepoint, so anything else can only be dropped -- which is what the
    // reference front ends do with unknown symbols.
    for (size_t i = 0; i < ipa.size();) {
        size_t len = 1;
        const auto lead = static_cast<unsigned char>(ipa[i]);
        if ((lead & 0xF8U) == 0xF0U) { len = 4; }
        else if ((lead & 0xF0U) == 0xE0U) { len = 3; }
        else if ((lead & 0xE0U) == 0xC0U) { len = 2; }
        len = std::min(len, ipa.size() - i);
        const std::string symbol = ipa.substr(i, len);
        i += len;
        const auto found = vocab.find(symbol);
        if (found != vocab.end()) {
            out.token_ids.push_back(found->second);
        } else {
            out.dropped.append(symbol);
        }
    }
    if (out.token_ids.size() == 1) {
        throw std::runtime_error(
            "sanoTTS phonemization produced no symbols in the packaged vocabulary");
    }
    out.token_ids.push_back(2);   // <eos>
    if (static_cast<int64_t>(out.token_ids.size()) > max_tokens_) {
        throw SanoTtsTooLongError(
            "sanoTTS phoneme sequence has " + std::to_string(out.token_ids.size()) +
            " tokens including BOS/EOS; the duration model was trained for at most " +
            std::to_string(max_tokens_) + ".");
    }
    return out;
}

std::vector<std::string> SanoTtsFrontend::split_text(
    const std::string & text,
    int64_t max_codepoints) {
    const size_t budget = max_codepoints > 0 ? static_cast<size_t>(max_codepoints) : 280U;
    std::vector<std::string> chunks;
    std::string current;
    size_t codepoints = 0;
    for (size_t i = 0; i < text.size();) {
        size_t len = 1;
        const auto lead = static_cast<unsigned char>(text[i]);
        if ((lead & 0xF8U) == 0xF0U) { len = 4; }
        else if ((lead & 0xF0U) == 0xE0U) { len = 3; }
        else if ((lead & 0xE0U) == 0xC0U) { len = 2; }
        len = std::min(len, text.size() - i);
        current.append(text, i, len);
        i += len;
        ++codepoints;
        const bool sentence_end = len == 1 && (text[i - 1] == '.' || text[i - 1] == '!' ||
                                               text[i - 1] == '?');
        if ((sentence_end && codepoints >= budget / 4) || codepoints >= budget) {
            chunks.push_back(current);
            current.clear();
            codepoints = 0;
        }
    }
    if (!current.empty()) {
        chunks.push_back(current);
    }
    if (chunks.empty()) {
        chunks.push_back(text);
    }
    return chunks;
}

double SanoTtsFrontend::boundary_pause_seconds(const std::string & chunk) {
    for (auto it = chunk.rbegin(); it != chunk.rend(); ++it) {
        if (std::isspace(static_cast<unsigned char>(*it)) != 0) {
            continue;
        }
        return (*it == '.' || *it == '!' || *it == '?') ? 0.20 : 0.08;
    }
    return 0.08;
}

// ---- piperlite front end -------------------------------------------------

struct SanoTtsPiperFrontend::Impl {
    EspeakApi espeak;
    Impl(const std::filesystem::path & library,
         const std::filesystem::path & data,
         const std::string & voice)
        : espeak(library, data, voice) {}
};

SanoTtsPiperFrontend::SanoTtsPiperFrontend(
    std::filesystem::path espeak_library_path,
    std::filesystem::path espeak_data_path,
    std::string espeak_voice,
    std::unordered_map<std::string, int32_t> phoneme_id_map,
    int64_t max_tokens)
    : impl_(std::make_unique<Impl>(espeak_library_path, espeak_data_path, espeak_voice)),
      id_map_(std::move(phoneme_id_map)),
      max_tokens_(max_tokens > 3 ? max_tokens : 3) {}

SanoTtsPiperFrontend::~SanoTtsPiperFrontend() = default;

SanoTtsEncoded SanoTtsPiperFrontend::encode(const std::string & text) const {
    auto [chunks, marks] = preserve_punctuation(text);
    std::vector<std::string> chunk_phonemes;
    chunk_phonemes.reserve(chunks.size());
    for (const auto & chunk : chunks) {
        chunk_phonemes.push_back(postprocess_espeak_line(
            impl_->espeak.phonemize(chunk, kEspeakPhonemesIpaUnderscore)));
    }
    std::string restored =
        restore_punctuation(std::move(chunk_phonemes), std::move(marks));
    // phonemizer leaves a trailing word separator that piper's own bridge
    // does not emit at true end of input; the reference rstrips before NFD.
    while (!restored.empty() &&
           std::isspace(static_cast<unsigned char>(restored.back())) != 0) {
        restored.pop_back();
    }
    const std::string decomposed = nfd_decompose(restored);

    // piper.phoneme_ids.phonemes_to_ids: bos, pad, then (id, pad) per
    // codepoint, then eos. The exporter validated the framing symbols
    // ('_' -> 0, '^' -> 1, '$' -> 2) against the voice's own map.
    SanoTtsEncoded out;
    out.token_ids.push_back(1);   // <bos>
    out.token_ids.push_back(0);   // <pad>
    for (size_t i = 0; i < decomposed.size();) {
        const size_t len = utf8_sequence_length(decomposed, i);
        const std::string symbol = decomposed.substr(i, len);
        i += len;
        const auto found = id_map_.find(symbol);
        if (found != id_map_.end()) {
            out.token_ids.push_back(found->second);
            out.token_ids.push_back(0);
        } else {
            // skipped with a note, exactly as piper skips unmapped phonemes
            out.dropped.append(symbol);
        }
    }
    if (out.token_ids.size() <= 2) {
        throw std::runtime_error(
            "sanoTTS phonemization produced no symbols in the voice's phoneme_id_map");
    }
    out.token_ids.push_back(2);   // <eos>
    if (static_cast<int64_t>(out.token_ids.size()) > max_tokens_) {
        throw SanoTtsTooLongError(
            "sanoTTS phoneme sequence has " + std::to_string(out.token_ids.size()) +
            " ids including framing; the duration model was trained for at most " +
            std::to_string(max_tokens_) + ".");
    }
    return out;
}

}  // namespace engine::models::sanotts
