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

struct EspeakApi {
    io::DynamicLibraryHandle library = nullptr;
    InitializeFn initialize = nullptr;
    SetVoiceFn set_voice = nullptr;
    TextToPhonemesFn text_to_phonemes = nullptr;
    TerminateFn terminate = nullptr;
    mutable std::mutex call_mutex;

    EspeakApi(const std::filesystem::path & requested_library,
              const std::filesystem::path & requested_data) {
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
        if (set_voice("en-us") != 0) {
            throw std::runtime_error("sanoTTS eSpeak-ng has no en-us voice");
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

    [[nodiscard]] std::string phonemize(const std::string & text) const {
        const std::lock_guard<std::mutex> guard(call_mutex);
        std::string out;
        const char * cursor = text.c_str();
        const void * position = cursor;
        // espeak consumes one clause per call and advances the pointer; it
        // returns null when the input is spent.
        while (position != nullptr) {
            const char * clause =
                text_to_phonemes(&position, kEspeakCharsUtf8, kEspeakPhonemesIpaTie);
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
        : espeak(library, data) {}
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
        chunk_phonemes.push_back(postprocess_espeak_line(impl_->espeak.phonemize(chunk)));
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

}  // namespace engine::models::sanotts
