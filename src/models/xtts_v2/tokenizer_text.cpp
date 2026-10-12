#include "engine/models/xtts_v2/tokenizer_text.h"
#include "engine/models/xtts_v2/assets.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <climits>
#include <unordered_map>
#include <vector>
#include <cstdio>

#include "gguf.h"

namespace engine::models::xtts_v2 {
namespace {

static const char* k_languages[] = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar",
    "zh-cn", "hu", "ko", "ja", "hi"
};
static const int k_n_languages = sizeof(k_languages) / sizeof(k_languages[0]);

int xtts_v2_language_index_impl(const char* language) {
    for (int i = 0; i < k_n_languages; i++) {
        if (std::strcmp(k_languages[i], language) == 0) return i;
    }
    return -1;
}

// ---------- abbreviation expansion ----------
// Reference: _abbreviations[lang] - regex \b(abbrev)\. -> replacement (word-boundary
// before the abbreviation, literal dot after). Order matters for overlapping patterns
// (e.g. "mrs." before "mr."), so we use an ordered vector of (word, replacement).
static bool is_word_char(unsigned char c) { return std::isalnum(c) || c == '_'; }

// Word-boundary abbreviation expansion: match `word` + '.' only when the word
// starts at a word boundary (preceded by non-word char or string start).
static std::string expand_abbr_list(const std::string& text,
                                    const std::vector<std::pair<std::string, std::string>>& abbrs) {
    std::string result = text;
    for (const auto& ab : abbrs) {
        const std::string& key = ab.first;
        const std::string& val = ab.second;
        // regex: \b key \.  ->  val   (the dot is consumed)
        const std::string pat = key + ".";
        std::string out;
        size_t pos = 0;
        while (pos < result.size()) {
            size_t found = result.find(pat, pos);
            if (found == std::string::npos) { out += result.substr(pos); break; }
            // check word boundary before the match
            bool boundary = (found == 0) || !is_word_char((unsigned char)result[found - 1]);
            if (boundary) {
                out += result.substr(pos, found - pos);
                out += val;
                pos = found + pat.size();
            } else {
                // not at a boundary: copy through the match start + 1 char and retry
                out += result.substr(pos, found - pos + 1);
                pos = found + 1;
            }
        }
        result = out;
    }
    return result;
}

static std::string expand_abbreviations(const std::string& text, int lang_idx) {
    // Lists mirror TTS/tts/layers/xtts/tokenizer.py _abbreviations (order preserved).
    static const std::vector<std::pair<std::string, std::string>> en = {
        {"mrs", "misess"}, {"mr", "mister"}, {"dr", "doctor"}, {"st", "saint"},
        {"co", "company"}, {"jr", "junior"}, {"maj", "major"}, {"gen", "general"},
        {"drs", "doctors"}, {"rev", "reverend"}, {"lt", "lieutenant"}, {"hon", "honorable"},
        {"sgt", "sergeant"}, {"capt", "captain"}, {"esq", "esquire"}, {"ltd", "limited"},
        {"col", "colonel"}, {"ft", "fort"},
    };
    static const std::vector<std::pair<std::string, std::string>> es = {
        {"sra", "senora"}, {"sr", "senor"}, {"dr", "doctor"}, {"dra", "doctora"},
        {"st", "santo"}, {"co", "company"}, {"jr", "junior"}, {"ltd", "limitada"},
    };
    static const std::vector<std::pair<std::string, std::string>> fr = {
        {"mme", "madame"}, {"mr", "monsieur"}, {"dr", "docteur"}, {"st", "saint"},
        {"co", "compagnie"}, {"jr", "junior"}, {"ltd", "limitee"},
    };
    static const std::vector<std::pair<std::string, std::string>> de = {
        {"fr", "frau"}, {"dr", "doktor"}, {"st", "sankt"}, {"co", "firma"}, {"jr", "junior"},
    };
    static const std::vector<std::pair<std::string, std::string>> it = {
        {"sig.ra", "signora"}, {"sig", "signore"}, {"dr", "dottore"}, {"st", "santo"},
        {"co", "compagnia"}, {"jr", "junior"}, {"ltd", "limitata"},
    };
    static const std::vector<std::pair<std::string, std::string>> pt = {
        {"sra", "senhora"}, {"sr", "senhor"}, {"dr", "doutor"}, {"dra", "doutora"},
        {"st", "santo"}, {"co", "companhia"}, {"jr", "junior"}, {"ltd", "limitada"},
    };
    static const std::vector<std::pair<std::string, std::string>> pl = {
        {"p", "pani"}, {"m", "pan"}, {"dr", "doktor"}, {"sw", "swiety"}, {"jr", "junior"},
    };
    static const std::vector<std::pair<std::string, std::string>> cs = {
        {"dr", "doktor"}, {"ing", "inzenyr"}, {"p", "pan"},
    };
    static const std::vector<std::pair<std::string, std::string>> nl = {
        {"dhr", "de heer"}, {"mevr", "mevrouw"}, {"dr", "dokter"}, {"jhr", "jonkheer"},
    };
    static const std::vector<std::pair<std::string, std::string>> tr = {
        {"b", "bay"}, {"byk", "buyuk"}, {"dr", "doktor"},
    };
    static const std::vector<std::pair<std::string, std::string>> hu = {
        {"dr", "doktor"}, {"b", "bacsi"}, {"nov", "nover"},
    };
    static const std::vector<std::pair<std::string, std::string>> ru = {
        {"\xD0\xB3-\xD0\xB6\xD0\xB0", "gospozha"}, {"\xD0\xB3-\xD0\xBD", "gospodin"},
        {"\xD0\xB4-\xD1\x80", "doktor"},
    };
    switch (lang_idx) {
        case 0: case -1: return expand_abbr_list(text, en);
        case 1:          return expand_abbr_list(text, es);
        case 2:          return expand_abbr_list(text, fr);
        case 3:          return expand_abbr_list(text, de);
        case 4:          return expand_abbr_list(text, it);
        case 5:          return expand_abbr_list(text, pt);
        case 6:          return expand_abbr_list(text, pl);
        case 10:         return expand_abbr_list(text, cs);
        case 8:          return expand_abbr_list(text, ru);
        case 9:          return expand_abbr_list(text, nl);
        case 11:         return expand_abbr_list(text, tr);
        case 13:         return expand_abbr_list(text, hu);
        default:         return text;
    }
}

// ---------- symbol expansion ----------
// Reference: _symbols_multilingual - literal replacement; replacements carry
// their own surrounding spaces, a "  " -> " " collapse runs after each symbol
// substitution and the result is stripped at the end.
static std::string expand_symbols(const std::string& text, int lang_idx) {
    // English list, reference order
    static const std::vector<std::pair<std::string, std::string>> en = {
        {"&", " and "}, {"@", " at "}, {"%", " percent "}, {"#", " hash "},
        {"$", " dollar "}, {"\xC2\xA3", " pound "}, {"\xC2\xB0", " degree "},
    };
    const std::vector<std::pair<std::string, std::string>>* syms =
        (lang_idx == 0 || lang_idx == -1) ? &en : nullptr;
    if (!syms) return text;  // other languages: no symbol expansion (rarely used)

    std::string result = text;
    for (const auto& sym : *syms) {
        const std::string& key = sym.first;
        const std::string& val = sym.second;
        std::string out;
        size_t pos = 0;
        while (pos < result.size()) {
            size_t found = result.find(key, pos);
            if (found == std::string::npos) { out += result.substr(pos); break; }
            out += result.substr(pos, found - pos);
            out += val;
            pos = found + key.size();
        }
        // reference collapses double spaces after each symbol substitution
        std::string collapsed;
        for (size_t k = 0; k < out.size(); k++) {
            if (k + 1 < out.size() && out[k] == ' ' && out[k + 1] == ' ') continue;
            collapsed += out[k];
        }
        result = collapsed;
    }
    // .strip() equivalent
    while (!result.empty() && std::isspace((unsigned char)result.front())) result.erase(result.begin());
    while (!result.empty() && std::isspace((unsigned char)result.back())) result.pop_back();
    return result;
}

// ---------- helpers ----------
static std::string lowercase(const std::string& text) {
    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

static std::string collapse_whitespace(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    bool last_ws = false;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!last_ws) { result += ' '; last_ws = true; }
        } else { result += c; last_ws = false; }
    }
    while (!result.empty() && std::isspace(static_cast<unsigned char>(result.front())))
        result.erase(result.begin());
    while (!result.empty() && std::isspace(static_cast<unsigned char>(result.back())))
        result.pop_back();
    return result;
}

// ---------- GGUF metadata reader (minimal, no external dep) ----------
static uint32_t gguf_kv_u32(gguf_context* gctx, const char* key, uint32_t def) {
    int ki = gguf_find_key(gctx, key);
    if (ki < 0) return def;
    if (gguf_get_kv_type(gctx, ki) != GGUF_TYPE_UINT32) return def;
    const auto* data = static_cast<const uint32_t*>(gguf_get_arr_data(gctx, ki));
    return data[0];
}

static std::vector<std::string> gguf_kv_str_array(gguf_context* gctx, const char* key) {
    int ki = gguf_find_key(gctx, key);
    if (ki < 0) return {};
    size_t n = gguf_get_arr_n(gctx, ki);
    std::vector<std::string> out;
    out.reserve(n);
    for (size_t i = 0; i < n; i++) out.push_back(gguf_get_arr_str(gctx, ki, i));
    return out;
}

// ---------- BPE encode ----------
static std::vector<int32_t> bpe_encode(const XttSv2BpeVocab* tok, const std::string& text,
                                        const std::unordered_map<std::string, int32_t>& vocab_cache,
                                        const std::unordered_map<std::string, int>& merge_rank) {
    if (!tok->id_to_token.empty()) {
        // Build a segment list: special tokens kept atomic.
        struct seg { bool is_special; int32_t sid; std::string text; };
        std::vector<seg> segments;

        size_t pos = 0;
        while (pos < text.size()) {
            bool matched = false;
            for (auto& [sp_str, sp_id] : tok->special_tokens) {
                if (pos + sp_str.size() <= text.size() &&
                    text.compare(pos, sp_str.size(), sp_str) == 0) {
                    segments.push_back({true, sp_id, {}});
                    pos += sp_str.size();
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                size_t start = pos;
                while (pos < text.size()) {
                    bool found_sp = false;
                    for (auto& [sp_str, _] : tok->special_tokens) {
                        if (pos + sp_str.size() <= text.size() &&
                            text.compare(pos, sp_str.size(), sp_str) == 0) {
                            found_sp = true; break;
                        }
                    }
                    if (found_sp) break;
                    pos++;
                }
                if (pos > start) segments.push_back({false, 0, text.substr(start, pos - start)});
            }
        }

        // BPE merge on regular segments
        auto make_key = [](const std::string& a, const std::string& b) {
            std::string key = a; key += (char)0x01; key += b; return key;
        };

        std::vector<int32_t> result;
        for (auto& seg : segments) {
            if (seg.is_special) { result.push_back(seg.sid); continue; }

            // linked-list symbols
            struct sym { std::string text; int next = -1; };
            std::vector<sym> symbols;
            symbols.reserve(seg.text.size());
            for (size_t i = 0; i < seg.text.size(); i++) {
                symbols.push_back({std::string(1, seg.text[i]),
                    i + 1 < seg.text.size() ? static_cast<int>(i + 1) : -1});
            }

            bool changed = true;
            while (changed) {
                changed = false;
                int best_idx = -1, best_rank = INT32_MAX;
                for (int i = 0; i >= 0; i = symbols[i].next) {
                    int nx = symbols[i].next;
                    if (nx < 0) break;
                    auto it = merge_rank.find(make_key(symbols[i].text, symbols[nx].text));
                    if (it != merge_rank.end() && it->second < best_rank) {
                        best_rank = it->second; best_idx = i;
                    }
                }
                if (best_idx >= 0) {
                    int nx = symbols[best_idx].next;
                    symbols[best_idx].text += symbols[nx].text;
                    symbols[best_idx].next = symbols[nx].next;
                    changed = true;
                }
            }

            for (int i = 0; i >= 0; i = symbols[i].next) {
                auto it = vocab_cache.find(symbols[i].text);
                result.push_back(it != vocab_cache.end() ? it->second : tok->unk_token);
            }
        }
        return result;
    }
    return {};
}

} // namespace

bool load_xtts_v2_tokenizer(XttSv2BpeVocab* tok, const std::filesystem::path& gguf_path) {
    gguf_context* g = gguf_init_from_file(gguf_path.string().c_str(), {true, 0});
    if (!g) {
        fprintf(stderr, "xtts_v2: failed to open GGUF for tokenizer: %s\n", gguf_path.string().c_str());
        return false;
    }

    auto tokens = gguf_kv_str_array(g, "tokenizer.ggml.tokens");
    if (tokens.empty()) {
        gguf_free(g);
        return false;
    }
    tok->id_to_token = std::move(tokens);

    auto merge_lines = gguf_kv_str_array(g, "tokenizer.ggml.bpe.merges");
    for (auto& line : merge_lines) {
        size_t sp = line.find(' ');
        if (sp != std::string::npos)
            tok->merges.push_back({line.substr(0, sp), line.substr(sp + 1)});
    }

    // Scores
    { int ki = gguf_find_key(g, "tokenizer.ggml.scores");
      if (ki >= 0 && gguf_get_kv_type(g, ki) == GGML_TYPE_F32) {
          int n = gguf_get_arr_n(g, ki);
          const float* sd = static_cast<const float*>(gguf_get_arr_data(g, ki));
          tok->scores.assign(sd, sd + n);
      }
    }

    // Language tokens
    tok->lang_tokens.clear();
    for (int i = 0; i < k_n_languages; i++) {
        std::string lt = "[" + std::string(k_languages[i]) + "]";
        bool found = false;
        for (int j = 0; j < (int)tok->id_to_token.size(); j++) {
            if (tok->id_to_token[j] == lt) { tok->lang_tokens.push_back(j); found = true; break; }
        }
        if (!found) tok->lang_tokens.push_back(-1);
    }

    // Special tokens
    {
        struct { const char* name; int32_t fallback; } spec[] = {
            {"[STOP]", 0}, {"[UNK]", 1}, {"[SPACE]", 2}, {"[START]", 261},
            {"[zh-cn]", 5023}, {"[ar]", 5022}, {"[cs]", 293}, {"[de]", 260},
            {"[en]", 259}, {"[es]", 284}, {"[fr]", 262}, {"[hi]", 6680},
            {"[hu]", 5753}, {"[it]", 285}, {"[ja]", 5412}, {"[ko]", 6152},
            {"[nl]", 297}, {"[pl]", 294}, {"[pt]", 286}, {"[ru]", 267}, {"[tr]", 295}
        };
        for (auto& [name, fallback] : spec) {
            for (int i = 0; i < (int)tok->id_to_token.size(); i++) {
                if (tok->id_to_token[i] == name) {
                    tok->special_tokens.push_back({name, i}); break;
                }
            }
        }
        std::sort(tok->special_tokens.begin(), tok->special_tokens.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    }

    gguf_free(g);
    fprintf(stderr, "xtts_v2: tokenizer loaded %zu tokens, %zu merges, %zu special\n",
            tok->id_to_token.size(), tok->merges.size(), tok->special_tokens.size());
    if (getenv("XTTS_DEBUG")) {
        for (auto& [str, id] : tok->special_tokens)
            fprintf(stderr, "xtts_dbg: special '%s' = %d\n", str.c_str(), id);
        for (int i = 0; i < 25 && i < (int)tok->id_to_token.size(); i++)
            fprintf(stderr, "xtts_dbg: vocab[%d] = '%s'\n", i, tok->id_to_token[i].c_str());
        for (int i = 6660; i < 6681 && i < (int)tok->id_to_token.size(); i++)
            fprintf(stderr, "xtts_dbg: vocab[%d] = '%s'\n", i, tok->id_to_token[i].c_str());
    }
    return true;
}

std::vector<int32_t> tokenize_xtts_v2(const XttSv2BpeVocab* tok, const std::string& text, const char* language) {
    std::string lang_code = language && language[0] ? language : "en";
    int lang_idx = xtts_v2_language_index_impl(lang_code.c_str());
    if (lang_idx < 0) { lang_idx = 0; lang_code = "en"; }

    // Preprocess: prepend [{lang}], replace spaces with [SPACE]
    std::string processed = "[" + lang_code + "]";
    for (char c : text) {
        if (c == ' ') processed += "[SPACE]";
        else processed += c;
    }

    // Build vocab cache
    std::unordered_map<std::string, int32_t> vocab_cache;
    for (int i = 0; i < (int)tok->id_to_token.size(); i++)
        vocab_cache[tok->id_to_token[i]] = i;

    // Build merge rank map
    std::unordered_map<std::string, int> merge_rank;
    for (int i = 0; i < (int)tok->merges.size(); i++) {
        std::string key = tok->merges[i].first; key += (char)0x01; key += tok->merges[i].second;
        merge_rank[key] = i;
    }

    return bpe_encode(tok, processed, vocab_cache, merge_rank);
}

// ---------- number expansion (reference: num2words(lang='en')) ----------
// English cardinal words for 0..999,999,999,999 and decimals with "point".
static const char* k_ones[] = {"zero", "one", "two", "three", "four", "five", "six", "seven",
                               "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
                               "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
static const char* k_tens[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
                               "eighty", "ninety"};

static std::string three_digits_to_words(int n) {
    // 0 < n < 1000
    std::string out;
    int hundreds = n / 100, rest = n % 100;
    if (hundreds) {
        out += k_ones[hundreds];
        out += " hundred";
        if (rest) out += " and ";  // num2words(en): "one hundred and five"
    }
    if (rest) {
        if (rest < 20) out += k_ones[rest];
        else {
            out += k_tens[rest / 10];
            if (rest % 10) { out += "-"; out += k_ones[rest % 10]; }
        }
    }
    return out;
}

static std::string int_to_words(long long n) {
    if (n == 0) return "zero";
    if (n < 0) return "minus " + int_to_words(-n);
    std::string out;
    struct { long long value; const char* name; } scales[] = {
        {1000000000000LL, " trillion"}, {1000000000LL, " billion"},
        {1000000LL, " million"}, {1000LL, " thousand"},
    };
    bool first = true;
    for (auto& s : scales) {
        if (n >= s.value) {
            long long q = n / s.value;
            if (!first) out += " ";
            out += three_digits_to_words((int)q) + s.name;
            n %= s.value;
            first = false;
        }
    }
    if (n > 0) {
        if (!first) out += " ";
        out += three_digits_to_words((int)n);
    }
    return out;
}

static std::string ordinal_to_words(long long n) {
    // num2words(en, ordinal=True): "first", "second", ... "twenty-first", "one hundred and first"
    static const char* ord_special[] = {"zeroth", "first", "second", "third", "fourth", "fifth",
                                        "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh",
                                        "twelfth", "thirteenth", "fourteenth", "fifteenth",
                                        "sixteenth", "seventeenth", "eighteenth", "nineteenth"};
    if (n < 20) return ord_special[n];
    if (n < 100) {
        int t = (int)(n / 10), r = (int)(n % 10);
        if (r == 0) {
            static const char* tens_ord[] = {"", "", "twentieth", "thirtieth", "fortieth", "fiftieth",
                                             "sixtieth", "seventieth", "eightieth", "ninetieth"};
            return tens_ord[t];
        }
        return std::string(k_tens[t]) + "-" + ord_special[r];
    }
    // >= 100: cardinal minus last word, plus ordinal of the last part
    std::string card = int_to_words(n);
    // find last word group: if it ends with a hyphenated part or a simple word
    size_t sp = card.find_last_of(' ');
    size_t hp = card.find_last_of('-');
    size_t cut = std::max(sp, hp);
    std::string head = card.substr(0, cut);
    std::string tail = card.substr(cut + 1);
    // ordinalize tail
    std::string ord;
    if (tail == "one") ord = "first";
    else if (tail == "two") ord = "second";
    else if (tail == "three") ord = "third";
    else if (tail == "five") ord = "fifth";
    else if (tail == "eight") ord = "eighth";
    else if (tail == "nine") ord = "ninth";
    else if (tail == "twelve") ord = "twelfth";
    else if (tail == "twenty") ord = "twentieth";
    else if (tail == "thirty") ord = "thirtieth";
    else if (tail == "forty") ord = "fortieth";
    else if (tail == "fifty") ord = "fiftieth";
    else if (tail == "sixty") ord = "sixtieth";
    else if (tail == "seventy") ord = "seventieth";
    else if (tail == "eighty") ord = "eightieth";
    else if (tail == "ninety") ord = "ninetieth";
    else ord = tail + "th";
    return head + " " + ord;
}

// Replace numbers in text. Reference order: comma-grouped numbers -> currency ->
// decimal ("3.5" -> "three point five") -> ordinals ("1st" -> "first") -> integers.
static std::string expand_numbers_en(const std::string& text) {
    std::string result;
    result.reserve(text.size() + 16);
    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = (unsigned char)text[i];
        if (!std::isdigit(c)) { result += (char)c; i++; continue; }
        // parse number: digits with optional , groups and/or .decimal
        size_t j = i;
        while (j < text.size() && (std::isdigit((unsigned char)text[j]) || text[j] == ',' || text[j] == '.')) {
            // stop if ',' / '.' is not followed by a digit (sentence period etc.)
            if ((text[j] == ',' || text[j] == '.') && (j + 1 >= text.size() || !std::isdigit((unsigned char)text[j + 1])))
                break;
            j++;
        }
        std::string num = text.substr(i, j - i);
        // ordinal suffix?
        bool ordinal = false;
        std::string suffix;
        if (j + 1 < text.size() || j + 2 < text.size()) {
            for (const char* sfx : {"st", "nd", "rd", "th"}) {
                if (text.compare(j, 2, sfx) == 0) { ordinal = true; suffix = sfx; break; }
            }
        }
        // strip commas (thousand separators)
        std::string clean;
        for (char ch : num) if (ch != ',') clean += ch;
        size_t dot = clean.find('.');
        if (!clean.empty() && clean.find_first_not_of("0123456789.") == std::string::npos) {
            if (ordinal && dot == std::string::npos) {
                long long n = strtoll(clean.c_str(), nullptr, 10);
                result += ordinal_to_words(n);
                i = j + 2;  // consume suffix
                continue;
            } else if (dot != std::string::npos) {
                // decimal: "12.5" -> "twelve point five"
                std::string ip = clean.substr(0, dot), fp = clean.substr(dot + 1);
                result += int_to_words(strtoll(ip.c_str(), nullptr, 10));
                result += " point";
                for (char d : fp) result += std::string(" ") + k_ones[d - '0'];
                i = j;
                continue;
            } else {
                long long n = strtoll(clean.c_str(), nullptr, 10);
                result += int_to_words(n);
                i = j;
                continue;
            }
        }
        result += text[i];  // not a number after all
        i++;
    }
    return result;
}

std::string normalize_xtts_v2_text(const std::string& text, const char* language) {
    int lang_idx = (language && language[0]) ? xtts_v2_language_index_impl(language) : 0;
    if (lang_idx < 0) lang_idx = 0;

    std::string result = text;
    result = lowercase(result);
    if (lang_idx == 0 || lang_idx == -1) result = expand_numbers_en(result);
    result = expand_abbreviations(result, lang_idx);
    result = expand_symbols(result, lang_idx);
    result = collapse_whitespace(result);
    return result;
}

int xtts_v2_language_index(const char* language) {
    return xtts_v2_language_index_impl(language);
}

} // namespace engine::models::xtts_v2

