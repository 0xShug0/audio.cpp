#include "engine/models/kokoro_tts/g2p_en.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace kokoro_ggml::g2p_en {
namespace {

constexpr char kPrimaryStress[] = "ˈ";
constexpr char kSecondaryStress[] = "ˌ";

const std::string kDiphthongs = "AIOQWYʤʧ";
const std::string kVowels = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
const std::string kConsonants = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
const std::string kUsTaus = "AIOWYiuæɑəɛɪɹʊʌ";

const std::unordered_set<std::string> kPunctTags = {".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP"};
const std::unordered_map<std::string, std::string> kPunctTagPhonemes = {
    {"-LRB-", "("}, {"-RRB-", ")"}, {"``", "\xE2\x80\x9C"}, {"\"\"", "\xE2\x80\x9D"}, {"''", "\xE2\x80\x9D"},
};
const std::unordered_map<char, std::string> kAddSymbols = {{'.', "dot"}, {'/', "slash"}};
const std::unordered_map<char, std::string> kSymbols = {{'%', "percent"}, {'&', "and"}, {'+', "plus"}, {'@', "at"}};
const std::unordered_map<std::string, std::pair<std::string, std::string>> kCurrencies = {
    {"$", {"dollar", "cent"}},
    {"£", {"pound", "pence"}},
    {"€", {"euro", "cent"}},
};
const std::unordered_set<std::string> kOrdinals = {"st", "nd", "rd", "th"};
const std::unordered_set<char> kSubtokenJunk = {'\'', ',', '-', '.', '_', '/', static_cast<char>(0xe2)};
const std::string kPuncts = ";:,.!?—…\"“”";
const std::string kNonQuotePuncts = ";:,.!?—…";

const std::unordered_set<std::string> kDeterminers = {"a", "an", "the", "this", "that", "these", "those"};
const std::unordered_set<std::string> kPronouns = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"};
const std::unordered_set<std::string> kPrepositions = {
    "to", "in", "on", "at", "by", "for", "from", "with", "without", "within", "into", "onto", "over",
    "under", "before", "after", "through", "across", "between", "among", "versus", "vs", "of"
};
const std::unordered_set<std::string> kConjunctions = {"and", "or", "but", "nor", "so", "yet"};
const std::unordered_set<std::string> kAdverbs = {"not", "never", "always", "often", "rarely", "here", "there", "why", "how", "when"};

struct NumberWords {
    std::string cardinal;
    std::string ordinal;
};

template <typename Container, typename Key>
bool contains_key(const Container & container, const Key & key) {
    return container.find(key) != container.end();
}

bool contains_codepoint(const std::string & text, const std::string & symbol) {
    return text.find(symbol) != std::string::npos;
}

bool starts_with(const std::string & text, const std::string & prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string & text, const std::string & suffix) {
    return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool ends_with_any(const std::string & text, const std::initializer_list<const char *> & suffixes) {
    for (const char * suffix : suffixes) {
        if (ends_with(text, suffix)) {
            return true;
        }
    }
    return false;
}

bool is_ascii_alpha(char ch) {
    return std::isalpha(static_cast<unsigned char>(ch)) != 0;
}

bool is_all_ascii_alpha(const std::string & text) {
    return !text.empty() && std::all_of(text.begin(), text.end(), is_ascii_alpha);
}

bool is_all_upper_ascii(const std::string & text) {
    return !text.empty() && std::all_of(text.begin(), text.end(), [](char ch) {
        return !std::isalpha(static_cast<unsigned char>(ch)) || std::isupper(static_cast<unsigned char>(ch)) != 0;
    });
}

bool is_capitalized_ascii(const std::string & text) {
    return !text.empty() &&
           std::isupper(static_cast<unsigned char>(text.front())) != 0 &&
           std::all_of(text.begin() + 1, text.end(), [](char ch) {
               return !std::isalpha(static_cast<unsigned char>(ch)) || std::islower(static_cast<unsigned char>(ch)) != 0;
           });
}

std::string lower_ascii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

std::string upper_ascii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::toupper(ch));
    });
    return text;
}

std::string capitalize_ascii(std::string text) {
    if (!text.empty()) {
        text[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(text[0])));
        for (size_t i = 1; i < text.size(); ++i) {
            text[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(text[i])));
        }
    }
    return text;
}

std::vector<std::string> split_tab(const std::string & line) {
    std::vector<std::string> fields;
    size_t begin = 0;
    while (true) {
        const size_t tab = line.find('\t', begin);
        if (tab == std::string::npos) {
            fields.push_back(line.substr(begin));
            return fields;
        }
        fields.push_back(line.substr(begin, tab - begin));
        begin = tab + 1;
    }
}

std::optional<float> normalize_stress(const std::optional<float> & stress, const std::string & phonemes) {
    if (!stress.has_value()) {
        return stress;
    }
    if (*stress == 0.0f && contains_codepoint(phonemes, kPrimaryStress)) {
        return -1.0f;
    }
    return stress;
}

int stress_weight(const std::string & phonemes) {
    int sum = 0;
    for (char ch : phonemes) {
        sum += kDiphthongs.find(ch) != std::string::npos ? 2 : 1;
    }
    return sum;
}

bool is_digit_text(const std::string & text) {
    return !text.empty() && std::all_of(text.begin(), text.end(), [](char ch) {
        return std::isdigit(static_cast<unsigned char>(ch)) != 0;
    });
}

bool is_number_token_impl(const std::string & word, bool is_head) {
    if (std::none_of(word.begin(), word.end(), [](char ch) { return std::isdigit(static_cast<unsigned char>(ch)) != 0; })) {
        return false;
    }
    std::string trimmed = word;
    for (const char * suffix : {"ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"}) {
        if (ends_with(trimmed, suffix)) {
            trimmed = trimmed.substr(0, trimmed.size() - std::char_traits<char>::length(suffix));
            break;
        }
    }
    for (size_t i = 0; i < trimmed.size(); ++i) {
        const char ch = trimmed[i];
        if (std::isdigit(static_cast<unsigned char>(ch)) != 0 || ch == ',' || ch == '.') {
            continue;
        }
        if (is_head && i == 0 && ch == '-') {
            continue;
        }
        return false;
    }
    return !trimmed.empty();
}

bool contains_whitespace(const std::string & text) {
    return std::any_of(text.begin(), text.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
}

int english_g2p_char_class(char ch) {
    if (std::isalpha(static_cast<unsigned char>(ch)) != 0) {
        return 0;
    }
    if (is_digit_text(std::string(1, ch))) {
        return 1;
    }
    return 2;
}

size_t count_words(const std::string & text) {
    size_t count = 0;
    bool in_word = false;
    for (unsigned char ch : text) {
        if (std::isspace(ch) != 0) {
            in_word = false;
            continue;
        }
        if (!in_word) {
            ++count;
            in_word = true;
        }
    }
    return count;
}

bool all_subtoken_junk(const std::string & text) {
    for (char ch : text) {
        if (ch == '\'' || ch == ',' || ch == '-' || ch == '.' || ch == '_' || ch == '/') {
            continue;
        }
        return false;
    }
    return !text.empty();
}

size_t utf8_codepoint_width(const std::string & text, size_t offset) {
    const unsigned char lead = static_cast<unsigned char>(text[offset]);
    size_t width = 0;
    if ((lead & 0x80u) == 0) {
        width = 1;
    } else if ((lead & 0xE0u) == 0xC0u) {
        width = 2;
    } else if ((lead & 0xF0u) == 0xE0u) {
        width = 3;
    } else if ((lead & 0xF8u) == 0xF0u) {
        width = 4;
    } else {
        throw std::runtime_error("invalid UTF-8 lead byte in English G2P phoneme string");
    }
    if (offset + width > text.size()) {
        throw std::runtime_error("truncated UTF-8 codepoint in English G2P phoneme string");
    }
    for (size_t j = 1; j < width; ++j) {
        const unsigned char byte = static_cast<unsigned char>(text[offset + j]);
        if ((byte & 0xC0u) != 0x80u) {
            throw std::runtime_error("invalid UTF-8 continuation byte in English G2P phoneme string");
        }
    }
    return width;
}

bool utf8_inventory_contains(
    const std::string & inventory,
    const char * symbol,
    size_t symbol_width) {
    for (size_t i = 0; i < inventory.size();) {
        const size_t width = utf8_codepoint_width(inventory, i);
        if (width == symbol_width && std::memcmp(inventory.data() + i, symbol, symbol_width) == 0) {
            return true;
        }
        i += width;
    }
    return false;
}

std::string join_phoneme_pieces(const std::vector<std::string> & pieces) {
    std::string out;
    for (size_t i = 0; i < pieces.size(); ++i) {
        if (i > 0) {
            out.push_back(' ');
        }
        out += pieces[i];
    }
    return out;
}

std::optional<LexiconResult> compound_fallback_impl(
    const Lexicon & lexicon,
    const std::string & word,
    std::unordered_map<std::string, std::optional<LexiconResult>> & memo) {
    if (const auto found = memo.find(word); found != memo.end()) {
        return found->second;
    }
    if (word.size() < 4 || !is_all_ascii_alpha(word)) {
        memo[word] = std::nullopt;
        return std::nullopt;
    }

    const TokenContext ctx{};
    if (auto direct = lexicon.lookup(word, std::nullopt, std::nullopt, ctx)) {
        memo[word] = direct;
        return direct;
    }

    for (size_t split = 2; split + 2 <= word.size(); ++split) {
        const std::string left = word.substr(0, split);
        const std::string right = word.substr(split);
        auto left_result = lexicon.lookup(left, std::nullopt, std::nullopt, ctx);
        if (!left_result.has_value()) {
            continue;
        }
        auto right_result = compound_fallback_impl(lexicon, right, memo);
        if (!right_result.has_value()) {
            right_result = lexicon.lookup(right, std::nullopt, std::nullopt, ctx);
        }
        if (!right_result.has_value()) {
            continue;
        }
        const int rating = std::min(left_result->rating, right_result->rating);
        memo[word] = LexiconResult{left_result->phonemes + " " + right_result->phonemes, rating};
        return memo[word];
    }

    memo[word] = std::nullopt;
    return std::nullopt;
}

std::optional<LexiconResult> compound_fallback(
    const Lexicon & lexicon,
    const std::string & text,
    const std::optional<float> & stress) {
    const std::string word = lower_ascii(text);
    std::unordered_map<std::string, std::optional<LexiconResult>> memo;
    auto result = compound_fallback_impl(lexicon, word, memo);
    if (!result.has_value()) {
        return std::nullopt;
    }
    result->phonemes = Lexicon::apply_stress(result->phonemes, stress);
    return result;
}

std::optional<LexiconResult> spell_out_fallback(
    const Lexicon & lexicon,
    const std::string & text,
    const std::optional<float> & stress) {
    std::vector<std::string> pieces;
    for (char ch : text) {
        if (std::isspace(static_cast<unsigned char>(ch)) != 0) {
            continue;
        }
        MToken token;
        token.text = std::string(1, is_ascii_alpha(ch) ? static_cast<char>(std::toupper(static_cast<unsigned char>(ch))) : ch);
        token.tag = std::isdigit(static_cast<unsigned char>(ch)) != 0 ? "CD" : "NNP";
        token.meta.is_head = true;
        if (auto result = lexicon(token, TokenContext{})) {
            pieces.push_back(result->phonemes);
        }
    }
    if (pieces.empty()) {
        return std::nullopt;
    }
    return LexiconResult{Lexicon::apply_stress(join_phoneme_pieces(pieces), stress), 1};
}

LexiconResult resolve_with_safe_fallback(
    const Lexicon & lexicon,
    const MToken & token) {
    if (auto compound = compound_fallback(lexicon, token.text, token.meta.stress)) {
        return *compound;
    }
    if (auto spelled = spell_out_fallback(lexicon, token.text, token.meta.stress)) {
        return *spelled;
    }
    return LexiconResult{Lexicon::apply_stress("ə", token.meta.stress), 1};
}

std::string trim_copy(const std::string & text) {
    size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
        --end;
    }
    return text.substr(begin, end - begin);
}

void replace_all(std::string & text, const std::string & from, const std::string & to) {
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != std::string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
}

std::vector<std::string> split_words(const std::string & text) {
    std::vector<std::string> words;
    std::string current;
    for (char ch : text) {
        if (std::isspace(static_cast<unsigned char>(ch)) != 0) {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
            continue;
        }
        current.push_back(ch);
    }
    if (!current.empty()) {
        words.push_back(current);
    }
    return words;
}

std::vector<std::string> split_non_alpha(const std::string & text) {
    std::vector<std::string> out;
    std::string current;
    for (char ch : text) {
        if (std::isalpha(static_cast<unsigned char>(ch)) != 0) {
            current.push_back(ch);
            continue;
        }
        if (!current.empty()) {
            out.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) {
        out.push_back(current);
    }
    return out;
}

std::filesystem::path resolve_default_lexicon_dir() {
    const std::array<std::filesystem::path, 4> candidates = {
        std::filesystem::path("models/kokoro-82m-v1_0-ggml/misaki_en"),
        std::filesystem::path("models/misaki_en"),
        std::filesystem::current_path() / "models/kokoro-82m-v1_0-ggml/misaki_en",
        std::filesystem::current_path() / "models/misaki_en",
    };
    for (const auto & candidate : candidates) {
        if (std::filesystem::exists(candidate / "us_gold.tsv") || std::filesystem::exists(candidate / "gb_gold.tsv")) {
            return candidate;
        }
    }
    throw std::runtime_error("failed to locate Kokoro misaki_en lexicon assets");
}

std::vector<std::string> subtokenize(const std::string & word) {
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < word.size()) {
        const unsigned char ch = static_cast<unsigned char>(word[i]);
        if (word[i] == '\'') {
            size_t j = i + 1;
            while (j < word.size() && word[j] == '\'') {
                ++j;
            }
            tokens.push_back(word.substr(i, j - i));
            i = j;
            continue;
        }
        if (std::isdigit(ch) != 0 || ((word[i] == '-' || word[i] == '+') && i + 1 < word.size() && std::isdigit(static_cast<unsigned char>(word[i + 1])) != 0)) {
            size_t j = i + 1;
            while (j < word.size()) {
                const unsigned char cj = static_cast<unsigned char>(word[j]);
                if (std::isdigit(cj) != 0) {
                    ++j;
                    continue;
                }
                if ((word[j] == ',' || word[j] == '.') &&
                    j + 1 < word.size() &&
                    std::isdigit(static_cast<unsigned char>(word[j + 1])) != 0) {
                    ++j;
                    continue;
                }
                break;
            }
            tokens.push_back(word.substr(i, j - i));
            i = j;
            continue;
        }
        if (std::isalpha(ch) != 0) {
            size_t j = i + 1;
            while (j < word.size()) {
                const unsigned char pj = static_cast<unsigned char>(word[j]);
                if (std::isalpha(pj) != 0 || word[j] == '\'') {
                    if (j + 1 < word.size() && std::islower(pj) != 0 && std::isupper(static_cast<unsigned char>(word[j + 1])) != 0) {
                        ++j;
                        break;
                    }
                    ++j;
                    continue;
                }
                break;
            }
            tokens.push_back(word.substr(i, j - i));
            i = j;
            continue;
        }
        if (word[i] == '-' || word[i] == '_') {
            size_t j = i + 1;
            while (j < word.size() && (word[j] == '-' || word[j] == '_')) {
                ++j;
            }
            tokens.push_back(word.substr(i, j - i));
            i = j;
            continue;
        }
        tokens.push_back(word.substr(i, 1));
        ++i;
    }
    return tokens;
}

std::string guess_tag(const std::string & token) {
    if (token.empty()) {
        return "NN";
    }
    if (token == "$" || token == "£" || token == "€") {
        return "$";
    }
    if (token == "(") return "-LRB-";
    if (token == ")") return "-RRB-";
    if (token == "\"" || token == "“") return "``";
    if (token == "”") return "''";
    if (token == ":" || token == ";" || token == "-" || token == "–" || token == "—") return ":";
    if (token == "," || token == "." || token == "!" || token == "?" || token == "…") return std::string(1, token[0]);

    const std::string lowered = lower_ascii(token);
    if (contains_key(kDeterminers, lowered)) return "DT";
    if (contains_key(kPronouns, lowered)) return "PRP";
    if (contains_key(kPrepositions, lowered)) return lowered == "to" ? "TO" : "IN";
    if (contains_key(kConjunctions, lowered)) return "CC";
    if (contains_key(kAdverbs, lowered) || ends_with(lowered, "ly")) return "RB";
    if (is_number_token_impl(token, true)) return "CD";
    if (is_all_upper_ascii(token) || is_capitalized_ascii(token)) return "NNP";
    return "NN";
}

std::string join_words(const std::vector<std::string> & words, const std::string & delimiter) {
    std::ostringstream out;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i > 0) {
            out << delimiter;
        }
        out << words[i];
    }
    return out.str();
}

std::string two_digit_cardinal(int value) {
    static const std::array<const char *, 20> below_20 = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"
    };
    static const std::array<const char *, 10> tens = {
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
    };
    if (value < 20) {
        return below_20[static_cast<size_t>(value)];
    }
    const int ten = value / 10;
    const int one = value % 10;
    if (one == 0) {
        return tens[static_cast<size_t>(ten)];
    }
    return std::string(tens[static_cast<size_t>(ten)]) + "-" + below_20[static_cast<size_t>(one)];
}

std::string integer_to_cardinal(int64_t value) {
    if (value == 0) {
        return "zero";
    }
    if (value < 0) {
        return "minus " + integer_to_cardinal(-value);
    }
    static const std::array<const char *, 7> scales = {
        "", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion"
    };
    std::vector<std::string> chunks;
    size_t scale = 0;
    while (value > 0) {
        const int part = static_cast<int>(value % 1000);
        if (part != 0) {
            std::string chunk;
            const int hundreds = part / 100;
            const int rest = part % 100;
            if (hundreds > 0) {
                chunk += two_digit_cardinal(hundreds) + " hundred";
                if (rest != 0) {
                    chunk += " ";
                }
            }
            if (rest != 0) {
                chunk += two_digit_cardinal(rest);
            }
            if (scale > 0) {
                chunk += " ";
                chunk += scales[scale];
            }
            chunks.push_back(chunk);
        }
        value /= 1000;
        ++scale;
    }
    std::reverse(chunks.begin(), chunks.end());
    return join_words(chunks, " ");
}

std::string integer_to_ordinal(int64_t value) {
    static const std::unordered_map<std::string, std::string> replacements = {
        {"one", "first"},
        {"two", "second"},
        {"three", "third"},
        {"five", "fifth"},
        {"eight", "eighth"},
        {"nine", "ninth"},
        {"twelve", "twelfth"},
        {"twenty", "twentieth"},
        {"thirty", "thirtieth"},
        {"forty", "fortieth"},
        {"fifty", "fiftieth"},
        {"sixty", "sixtieth"},
        {"seventy", "seventieth"},
        {"eighty", "eightieth"},
        {"ninety", "ninetieth"},
        {"hundred", "hundredth"},
        {"thousand", "thousandth"},
        {"million", "millionth"},
        {"billion", "billionth"},
        {"trillion", "trillionth"},
    };
    std::string cardinal = integer_to_cardinal(value);
    size_t pos = cardinal.find_last_of(" -");
    const size_t begin = pos == std::string::npos ? 0 : pos + 1;
    std::string tail = cardinal.substr(begin);
    const auto it = replacements.find(tail);
    if (it != replacements.end()) {
        return cardinal.substr(0, begin) + it->second;
    }
    if (ends_with(tail, "y")) {
        return cardinal.substr(0, cardinal.size() - 1) + "ieth";
    }
    return cardinal + "th";
}

std::string integer_to_year_words(int64_t value) {
    if (value < 1000 || value > 9999) {
        return integer_to_cardinal(value);
    }
    if (value >= 2000 && value <= 2009) {
        return "two thousand " + integer_to_cardinal(value % 1000);
    }
    const int first = static_cast<int>(value / 100);
    const int second = static_cast<int>(value % 100);
    if (second == 0) {
        return integer_to_cardinal(first) + " hundred";
    }
    if (second < 10) {
        return integer_to_cardinal(first) + " oh " + integer_to_cardinal(second);
    }
    return integer_to_cardinal(first) + " " + integer_to_cardinal(second);
}

std::vector<std::string> spell_digit_string(const std::string & digits) {
    static const std::array<const char *, 10> digit_words = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
    };
    std::vector<std::string> out;
    for (char ch : digits) {
        if (std::isdigit(static_cast<unsigned char>(ch)) != 0) {
            out.emplace_back(digit_words[static_cast<size_t>(ch - '0')]);
        }
    }
    return out;
}

}  // namespace

Lexicon::Lexicon(bool british)
    : Lexicon(
          resolve_default_lexicon_dir() / (british ? "gb_gold.tsv" : "us_gold.tsv"),
          resolve_default_lexicon_dir() / (british ? "gb_silver.tsv" : "us_silver.tsv"),
          british) {}

Lexicon::Lexicon(std::filesystem::path gold_tsv, std::filesystem::path silver_tsv, bool british)
    : british_(british) {
    load_tsv(gold_tsv, golds_);
    load_tsv(silver_tsv, silvers_);
}

void Lexicon::load_tsv(
    const std::filesystem::path & path,
    std::unordered_map<std::string, LexiconEntry> & target) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open misaki lexicon TSV: " + path.string());
    }
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto fields = split_tab(line);
        if (fields.size() != 3) {
            throw std::runtime_error("invalid misaki lexicon TSV line: " + line);
        }
        auto & entry = target[fields[0]];
        if (fields[1] == "*") {
            entry.has_default = true;
            entry.default_phonemes = fields[2];
        } else {
            entry.by_tag[fields[1]] = fields[2] == "~" ? std::optional<std::string>() : std::optional<std::string>(fields[2]);
        }
        if (fields[0].size() >= 2) {
            if (fields[0] == lower_ascii(fields[0])) {
                const auto capitalized = capitalize_ascii(fields[0]);
                if (capitalized != fields[0] && target.find(capitalized) == target.end()) {
                    target[capitalized] = entry;
                }
            } else if (fields[0] == capitalize_ascii(fields[0])) {
                const auto lowered = lower_ascii(fields[0]);
                if (target.find(lowered) == target.end()) {
                    target[lowered] = entry;
                }
            }
        }
    }
}

std::optional<std::string> Lexicon::parent_tag(const std::optional<std::string> & tag) {
    if (!tag.has_value()) {
        return std::nullopt;
    }
    if (tag->rfind("VB", 0) == 0) return std::string("VERB");
    if (tag->rfind("NN", 0) == 0) return std::string("NOUN");
    if (tag->rfind("ADV", 0) == 0 || tag->rfind("RB", 0) == 0) return std::string("ADV");
    if (tag->rfind("ADJ", 0) == 0 || tag->rfind("JJ", 0) == 0) return std::string("ADJ");
    return tag;
}

std::string Lexicon::apply_stress(const std::string & phonemes, const std::optional<float> & stress) {
    if (phonemes.empty() || !stress.has_value()) {
        return phonemes;
    }
    std::string out = phonemes;
    if (*stress < -1.0f) {
        while (true) {
            size_t pos = out.find(kPrimaryStress);
            if (pos == std::string::npos) break;
            out.erase(pos, std::char_traits<char>::length(kPrimaryStress));
        }
        while (true) {
            size_t pos = out.find(kSecondaryStress);
            if (pos == std::string::npos) break;
            out.erase(pos, std::char_traits<char>::length(kSecondaryStress));
        }
        return out;
    }
    if (*stress == -1.0f || ((*stress == 0.0f || *stress == -0.5f) && contains_codepoint(out, kPrimaryStress))) {
        replace_all(out, kSecondaryStress, "");
        replace_all(out, kPrimaryStress, kSecondaryStress);
        return out;
    }
    if ((*stress == 0.0f || *stress == 0.5f || *stress == 1.0f) &&
        !contains_codepoint(out, kPrimaryStress) && !contains_codepoint(out, kSecondaryStress)) {
        for (char ch : out) {
            if (kVowels.find(ch) != std::string::npos) {
                return std::string(kSecondaryStress) + out;
            }
        }
        return out;
    }
    if (*stress >= 1.0f && !contains_codepoint(out, kPrimaryStress) && contains_codepoint(out, kSecondaryStress)) {
        replace_all(out, kSecondaryStress, kPrimaryStress);
        return out;
    }
    if (*stress > 1.0f && !contains_codepoint(out, kPrimaryStress) && !contains_codepoint(out, kSecondaryStress)) {
        for (char ch : out) {
            if (kVowels.find(ch) != std::string::npos) {
                return std::string(kPrimaryStress) + out;
            }
        }
    }
    return out;
}

std::optional<LexiconResult> Lexicon::get_nnp(const std::string & word) const {
    std::string phonemes;
    for (char ch : word) {
        if (!is_ascii_alpha(ch)) {
            continue;
        }
        const std::string key(1, static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
        const auto it = golds_.find(key);
        if (it == golds_.end() || !it->second.has_default) {
            return std::nullopt;
        }
        phonemes += it->second.default_phonemes;
    }
    if (phonemes.empty()) {
        return std::nullopt;
    }
    phonemes = apply_stress(phonemes, 0.0f);
    const size_t split = phonemes.rfind(kSecondaryStress);
    if (split != std::string::npos) {
        phonemes.replace(split, std::char_traits<char>::length(kSecondaryStress), kPrimaryStress);
    }
    return LexiconResult{phonemes, 3};
}

bool Lexicon::is_known(const std::string & word) const {
    if (contains_key(golds_, word) || contains_key(silvers_, word) || (word.size() == 1 && contains_key(kSymbols, word[0]))) {
        return true;
    }
    if (!is_all_ascii_alpha(word)) {
        return false;
    }
    if (word.size() == 1) {
        return true;
    }
    if (is_all_upper_ascii(word) && contains_key(golds_, lower_ascii(word))) {
        return true;
    }
    return word.size() > 1 && std::all_of(word.begin() + 1, word.end(), [](char ch) {
        return std::isupper(static_cast<unsigned char>(ch)) != 0;
    });
}

std::optional<LexiconResult> Lexicon::lookup_raw(
    const std::string & word,
    const std::optional<std::string> & tag,
    const std::optional<float> & stress) const {
    std::optional<std::string> selected;
    int rating = 4;
    auto it = golds_.find(word);
    if (it != golds_.end()) {
        const auto & entry = it->second;
        if (!entry.by_tag.empty()) {
            std::optional<std::string> query = tag;
            if (!query.has_value()) {
                query = "DEFAULT";
            } else if (!contains_key(entry.by_tag, *query)) {
                query = parent_tag(query);
            }
            auto jt = entry.by_tag.find(query.value_or("DEFAULT"));
            if (jt == entry.by_tag.end()) {
                jt = entry.by_tag.find("DEFAULT");
            }
            if (jt != entry.by_tag.end()) {
                if (!jt->second.has_value()) {
                    return std::nullopt;
                }
                selected = *jt->second;
            }
        } else if (entry.has_default) {
            selected = entry.default_phonemes;
        }
    }
    if (!selected.has_value()) {
        auto st = silvers_.find(word);
        if (st != silvers_.end() && st->second.has_default) {
            selected = st->second.default_phonemes;
            rating = 3;
        }
    }
    if (!selected.has_value() && is_all_upper_ascii(word)) {
        return get_nnp(word);
    }
    if (!selected.has_value()) {
        return std::nullopt;
    }
    return LexiconResult{apply_stress(*selected, normalize_stress(stress, *selected)), rating};
}

std::optional<std::string> Lexicon::s_suffix(const std::string & stem) const {
    if (stem.empty()) return std::nullopt;
    if (ends_with_any(stem, {"p", "t", "k", "f", "θ"})) return stem + "s";
    if (ends_with_any(stem, {"s", "z", "ʃ", "ʒ", "ʧ", "ʤ"})) {
        return stem + (british_ ? "ɪz" : "ᵻz");
    }
    return stem + "z";
}

std::optional<std::string> Lexicon::ed_suffix(const std::string & stem) const {
    if (stem.empty()) return std::nullopt;
    if (ends_with_any(stem, {"p", "k", "f", "θ", "ʃ", "s", "ʧ"})) return stem + "t";
    if (ends_with(stem, "d")) return stem + (british_ ? "ɪd" : "ᵻd");
    if (!ends_with(stem, "t")) return stem + "d";
    if (british_ || stem.size() < 2) return stem + "ɪd";
    if (kUsTaus.find(stem[stem.size() - 2]) != std::string::npos) return stem.substr(0, stem.size() - 1) + "ɾᵻd";
    return stem + "ᵻd";
}

std::optional<std::string> Lexicon::ing_suffix(const std::string & stem) const {
    if (stem.empty()) return std::nullopt;
    if (british_) {
        if (ends_with_any(stem, {"ə", "ː"})) return std::nullopt;
    } else if (stem.size() > 1 && stem.back() == 't' && kUsTaus.find(stem[stem.size() - 2]) != std::string::npos) {
        return stem.substr(0, stem.size() - 1) + "ɾɪŋ";
    }
    return stem + "ɪŋ";
}

std::optional<LexiconResult> Lexicon::stem_s(
    const std::string & word,
    const std::optional<std::string> & tag,
    const std::optional<float> & stress,
    const TokenContext & ctx) const {
    if (word.size() < 3 || word.back() != 's') return std::nullopt;
    std::string stem;
    if (!(word.size() >= 2 && word.substr(word.size() - 2) == "ss") && is_known(word.substr(0, word.size() - 1))) {
        stem = word.substr(0, word.size() - 1);
    } else if (((word.size() >= 2 && word.substr(word.size() - 2) == "'s") ||
                (word.size() > 4 && word.size() >= 2 && word.substr(word.size() - 2) == "es" &&
                 word.substr(word.size() - 3) != "ies")) &&
               is_known(word.substr(0, word.size() - 2))) {
        stem = word.substr(0, word.size() - 2);
    } else if (word.size() > 4 && word.substr(word.size() - 3) == "ies" && is_known(word.substr(0, word.size() - 3) + "y")) {
        stem = word.substr(0, word.size() - 3) + "y";
    } else {
        return std::nullopt;
    }
    auto base = lookup(stem, tag, stress, ctx);
    if (!base.has_value()) return std::nullopt;
    auto phonemes = s_suffix(base->phonemes);
    if (!phonemes.has_value()) return std::nullopt;
    return LexiconResult{*phonemes, base->rating};
}

std::optional<LexiconResult> Lexicon::stem_ed(
    const std::string & word,
    const std::optional<std::string> & tag,
    const std::optional<float> & stress,
    const TokenContext & ctx) const {
    if (word.size() < 4 || word.back() != 'd') return std::nullopt;
    std::string stem;
    if (!(word.size() >= 2 && word.substr(word.size() - 2) == "dd") && is_known(word.substr(0, word.size() - 1))) {
        stem = word.substr(0, word.size() - 1);
    } else if (word.size() > 4 && word.substr(word.size() - 2) == "ed" &&
               word.substr(word.size() - 3) != "eed" && is_known(word.substr(0, word.size() - 2))) {
        stem = word.substr(0, word.size() - 2);
    } else {
        return std::nullopt;
    }
    auto base = lookup(stem, tag, stress, ctx);
    if (!base.has_value()) return std::nullopt;
    auto phonemes = ed_suffix(base->phonemes);
    if (!phonemes.has_value()) return std::nullopt;
    return LexiconResult{*phonemes, base->rating};
}

std::optional<LexiconResult> Lexicon::stem_ing(
    const std::string & word,
    const std::optional<std::string> & tag,
    const std::optional<float> & stress,
    const TokenContext & ctx) const {
    if (word.size() < 5 || word.substr(word.size() - 3) != "ing") return std::nullopt;
    std::string stem;
    if (word.size() > 5 && is_known(word.substr(0, word.size() - 3))) {
        stem = word.substr(0, word.size() - 3);
    } else if (is_known(word.substr(0, word.size() - 3) + "e")) {
        stem = word.substr(0, word.size() - 3) + "e";
    } else if (word.size() > 5) {
        const std::string doubled = word.substr(word.size() - 6);
        if ((doubled.size() >= 6 &&
             ((doubled[0] == doubled[1] && std::string("bcdgklmnprstvxz").find(doubled[0]) != std::string::npos) ||
              doubled.rfind("cking") != std::string::npos)) &&
            is_known(word.substr(0, word.size() - 4))) {
            stem = word.substr(0, word.size() - 4);
        } else {
            return std::nullopt;
        }
    } else {
        return std::nullopt;
    }
    auto base = lookup(stem, tag, stress.has_value() ? stress : std::optional<float>(0.5f), ctx);
    if (!base.has_value()) return std::nullopt;
    auto phonemes = ing_suffix(base->phonemes);
    if (!phonemes.has_value()) return std::nullopt;
    return LexiconResult{*phonemes, base->rating};
}

std::optional<LexiconResult> Lexicon::get_special_case(
    const std::string & word,
    const std::optional<std::string> & tag,
    const std::optional<float> & stress,
    const TokenContext & ctx) const {
    (void) stress;
    if (tag.has_value() && *tag == "ADD" && word.size() == 1 && contains_key(kAddSymbols, word[0])) {
        return lookup(kAddSymbols.at(word[0]), std::nullopt, -0.5f, ctx);
    }
    if (word.size() == 1 && contains_key(kSymbols, word[0])) {
        return lookup(kSymbols.at(word[0]), std::nullopt, std::nullopt, ctx);
    }
    if ((word == "a" || word == "A")) {
        if (tag.has_value() && *tag == "DT") return LexiconResult{"ɐ", 4};
        return LexiconResult{"ˈA", 4};
    }
    if (word == "am" || word == "Am" || word == "AM") {
        if (tag.has_value() && tag->rfind("NN", 0) == 0) {
            return get_nnp(word);
        }
        if (!ctx.future_vowel.has_value() || word != "am" || (stress.has_value() && *stress > 0.0f)) {
            return lookup_raw("am", std::nullopt, stress);
        }
        return LexiconResult{"ɐm", 4};
    }
    if (word == "an" || word == "An" || word == "AN") {
        if (word == "AN" && tag.has_value() && tag->rfind("NN", 0) == 0) {
            return get_nnp(word);
        }
        return LexiconResult{"ɐn", 4};
    }
    if (word == "I" && tag.has_value() && *tag == "PRP") {
        return LexiconResult{std::string(kSecondaryStress) + "I", 4};
    }
    if ((word == "by" || word == "By" || word == "BY") && parent_tag(tag) == std::optional<std::string>("ADV")) {
        return LexiconResult{"bˈI", 4};
    }
    if ((word == "to" || word == "To") || (word == "TO" && tag.has_value() && (*tag == "TO" || *tag == "IN"))) {
        if (!ctx.future_vowel.has_value()) return lookup_raw("to", std::nullopt, stress);
        return LexiconResult{*ctx.future_vowel ? "tʊ" : "tə", 4};
    }
    if ((word == "in" || word == "In") || (word == "IN" && (!tag.has_value() || *tag != "NNP"))) {
        return LexiconResult{(ctx.future_vowel.has_value() && tag == std::optional<std::string>("IN")) ? "ɪn" : std::string(kPrimaryStress) + "ɪn", 4};
    }
    if ((word == "the" || word == "The") || (word == "THE" && tag.has_value() && *tag == "DT")) {
        return LexiconResult{ctx.future_vowel == true ? "ði" : "ðə", 4};
    }
    if (tag == std::optional<std::string>("IN")) {
        const std::string lowered = lower_ascii(word);
        if (lowered == "vs" || lowered == "vs.") {
            return lookup("versus", std::nullopt, std::nullopt, ctx);
        }
    }
    if (word == "used" || word == "Used" || word == "USED") {
        if (tag == std::optional<std::string>("VBD") && ctx.future_to) {
            return lookup_raw("used", std::optional<std::string>("VBD"), stress);
        }
        return lookup_raw("used", std::nullopt, stress);
    }
    if (word.find('.') != std::string::npos) {
        std::string stripped = word;
        stripped.erase(std::remove(stripped.begin(), stripped.end(), '.'), stripped.end());
        if (!stripped.empty() && is_all_ascii_alpha(stripped)) {
            const auto parts = split_non_alpha(word);
            size_t longest = 0;
            for (const auto & part : parts) longest = std::max(longest, part.size());
            if (longest < 3) {
                return get_nnp(word);
            }
        }
    }
    return std::nullopt;
}

std::optional<LexiconResult> Lexicon::lookup(
    const std::string & word,
    const std::optional<std::string> & tag,
    const std::optional<float> & stress,
    const TokenContext & ctx) const {
    if (const auto special = get_special_case(word, tag, stress, ctx)) {
        return special;
    }
    std::string lookup_word = word;
    const std::string lowered = lower_ascii(word);
    if (word.size() > 1 &&
        std::all_of(word.begin(), word.end(), [](unsigned char ch) { return std::isalpha(ch) != 0 || ch == '\''; }) &&
        word != lowered &&
        (!tag.has_value() || *tag != "NNP" || word.size() > 7) &&
        !contains_key(golds_, word) && !contains_key(silvers_, word) &&
        (is_all_upper_ascii(word) || is_capitalized_ascii(word)) &&
        (contains_key(golds_, lowered) || contains_key(silvers_, lowered))) {
        lookup_word = lowered;
    }
    if (is_known(lookup_word)) {
        if (const auto direct = lookup_raw(lookup_word, tag, stress)) {
            return direct;
        }
    }
    if (lookup_word.size() >= 2 && lookup_word.substr(lookup_word.size() - 2) == "s'") {
        if (const auto direct = lookup_raw(lookup_word.substr(0, lookup_word.size() - 2) + "'s", tag, stress)) {
            return direct;
        }
    }
    if (!lookup_word.empty() && lookup_word.back() == '\'') {
        if (const auto direct = lookup_raw(lookup_word.substr(0, lookup_word.size() - 1), tag, stress)) {
            return direct;
        }
    }
    if (const auto stem = stem_s(lookup_word, tag, stress, ctx)) return stem;
    if (const auto stem = stem_ed(lookup_word, tag, stress, ctx)) return stem;
    if (const auto stem = stem_ing(lookup_word, tag, stress, ctx)) return stem;
    return std::nullopt;
}

std::optional<LexiconResult> Lexicon::get_word(
    const std::string & word,
    const std::optional<std::string> & tag,
    const std::optional<float> & stress,
    const TokenContext & ctx) const {
    std::string maybe_lower = word;
    const std::string lowered = lower_ascii(word);
    if (word.size() > 1 && word.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'") == std::string::npos &&
        word != lowered &&
        (!tag.has_value() || *tag != "NNP" || word.size() > 7) &&
        !contains_key(golds_, word) && !contains_key(silvers_, word) &&
        (is_all_upper_ascii(word) || is_capitalized_ascii(word)) &&
        (contains_key(golds_, lowered) || contains_key(silvers_, lowered) ||
         stem_s(lowered, tag, stress, ctx).has_value() ||
         stem_ed(lowered, tag, stress, ctx).has_value() ||
         stem_ing(lowered, tag, stress, ctx).has_value())) {
        maybe_lower = lowered;
    }
    if (is_known(maybe_lower)) {
        return lookup(maybe_lower, tag, stress, ctx);
    }
    if (ends_with(maybe_lower, "s'") && is_known(maybe_lower.substr(0, maybe_lower.size() - 2) + "'s")) {
        return lookup(maybe_lower.substr(0, maybe_lower.size() - 2) + "'s", tag, stress, ctx);
    }
    if (ends_with(maybe_lower, "'") && is_known(maybe_lower.substr(0, maybe_lower.size() - 1))) {
        return lookup(maybe_lower.substr(0, maybe_lower.size() - 1), tag, stress, ctx);
    }
    if (const auto stem = stem_s(maybe_lower, tag, stress, ctx)) return stem;
    if (const auto stem = stem_ed(maybe_lower, tag, stress, ctx)) return stem;
    if (const auto stem = stem_ing(maybe_lower, tag, stress.has_value() ? stress : std::optional<float>(0.5f), ctx)) return stem;
    return std::nullopt;
}

bool Lexicon::is_currency_shape(const std::string & word) {
    const size_t dot_count = static_cast<size_t>(std::count(word.begin(), word.end(), '.'));
    if (dot_count == 0) {
        return true;
    }
    if (dot_count > 1) {
        return false;
    }
    const size_t dot = word.find('.');
    const std::string cents = word.substr(dot + 1);
    if (cents.size() < 3) {
        return true;
    }
    return std::all_of(cents.begin(), cents.end(), [](char ch) { return ch == '0'; });
}

bool Lexicon::is_number_token(const std::string & word, bool is_head) {
    return is_number_token_impl(word, is_head);
}

std::optional<LexiconResult> Lexicon::get_number(
    const std::string & word,
    const std::optional<std::string> & currency,
    bool is_head) const {
    std::string suffix;
    std::string raw = word;
    size_t suffix_begin = raw.size();
    while (suffix_begin > 0 && std::isalpha(static_cast<unsigned char>(raw[suffix_begin - 1])) != 0) {
        --suffix_begin;
    }
    suffix = raw.substr(suffix_begin);
    raw = raw.substr(0, suffix_begin);

    std::vector<std::pair<std::string, int>> pieces;
    auto append_lookup = [&](const std::string & token, std::optional<float> stress = std::nullopt) {
        auto found = lookup(token, std::nullopt, stress, TokenContext{});
        if (!found.has_value()) {
            throw std::runtime_error("english g2p number expansion missing lexicon entry for: " + token);
        }
        pieces.emplace_back(found->phonemes, found->rating);
    };
    auto append_phrase = [&](const std::string & phrase) {
        for (const std::string & chunk : split_words(phrase)) {
            const auto parts = split_non_alpha(chunk);
            if (parts.empty()) {
                continue;
            }
            for (const std::string & part : parts) {
                if (part.empty()) continue;
                append_lookup(part, part == "point" ? std::optional<float>(-2.0f) : std::nullopt);
            }
        }
    };

    if (!raw.empty() && raw.front() == '-') {
        append_lookup("minus");
        raw.erase(raw.begin());
    }

    const std::string no_commas = [&]() {
        std::string s = raw;
        s.erase(std::remove(s.begin(), s.end(), ','), s.end());
        return s;
    }();

    if (currency.has_value() && contains_key(kCurrencies, *currency) && is_currency_shape(no_commas)) {
        const auto [major_name, minor_name] = kCurrencies.at(*currency);
        const size_t dot = no_commas.find('.');
        const int64_t major = dot == std::string::npos || no_commas.substr(0, dot).empty() ? 0 : std::stoll(no_commas.substr(0, dot));
        std::string minor_text = dot == std::string::npos ? "" : no_commas.substr(dot + 1);
        if (minor_text.size() > 2) {
            minor_text = minor_text.substr(0, 2);
        }
        while (!minor_text.empty() && minor_text.size() < 2) {
            minor_text.push_back('0');
        }
        const int64_t minor = minor_text.empty() ? 0 : std::stoll(minor_text);
        if (!(major == 0 && minor > 0)) {
            append_phrase(integer_to_cardinal(major));
            append_phrase(major == 1 ? major_name : major_name + "s");
        }
        if (minor > 0) {
            if (major > 0) {
                append_lookup("and");
            }
            append_phrase(integer_to_cardinal(minor));
            append_phrase((minor == 1 || minor_name == "pence") ? minor_name : minor_name + "s");
        }
    } else if (is_digit_text(no_commas) && contains_key(kOrdinals, lower_ascii(suffix))) {
        append_phrase(integer_to_ordinal(std::stoll(no_commas)));
    } else if (currency.has_value() == false && no_commas.find('.') == std::string::npos && no_commas.size() == 4 && is_digit_text(no_commas)) {
        append_phrase(integer_to_year_words(std::stoll(no_commas)));
    } else if (!is_head && no_commas.find('.') == std::string::npos) {
        if ((no_commas.size() > 1 && no_commas.front() == '0') || no_commas.size() > 3) {
            for (const std::string & digit_word : spell_digit_string(no_commas)) {
                append_lookup(digit_word);
            }
        } else if (no_commas.size() == 3 && no_commas.substr(1) != "00") {
            append_phrase(integer_to_cardinal(no_commas[0] - '0'));
            if (no_commas[1] == '0') {
                append_lookup("O", -2.0f);
                append_phrase(integer_to_cardinal(no_commas[2] - '0'));
            } else {
                append_phrase(integer_to_cardinal(std::stoi(no_commas.substr(1))));
            }
        } else {
            append_phrase(integer_to_cardinal(std::stoll(no_commas)));
        }
    } else if (std::count(no_commas.begin(), no_commas.end(), '.') > 1 || !is_head) {
        std::stringstream ss(no_commas);
        std::string segment;
        while (std::getline(ss, segment, '.')) {
            if (segment.empty()) continue;
            if ((segment.size() > 1 && segment.front() == '0') || (segment.size() != 2 && std::any_of(segment.begin() + 1, segment.end(), [](char ch) { return ch != '0'; }))) {
                for (const std::string & digit_word : spell_digit_string(segment)) {
                    append_lookup(digit_word);
                }
            } else {
                append_phrase(integer_to_cardinal(std::stoll(segment)));
            }
        }
    } else if (no_commas.find('.') != std::string::npos) {
        const size_t dot = no_commas.find('.');
        const std::string left = no_commas.substr(0, dot);
        const std::string right = no_commas.substr(dot + 1);
        if (left.empty()) {
            append_lookup("point");
        } else {
            append_phrase(integer_to_cardinal(std::stoll(left)));
            append_lookup("point");
        }
        for (const std::string & digit_word : spell_digit_string(right)) {
            append_lookup(digit_word);
        }
    } else {
        append_phrase(integer_to_cardinal(std::stoll(no_commas)));
    }

    if (pieces.empty()) {
        return std::nullopt;
    }

    std::string phonemes;
    int rating = pieces.front().second;
    for (size_t i = 0; i < pieces.size(); ++i) {
        if (i > 0) phonemes.push_back(' ');
        phonemes += pieces[i].first;
        rating = std::min(rating, pieces[i].second);
    }

    const std::string lowered_suffix = lower_ascii(suffix);
    if (lowered_suffix == "s" || lowered_suffix == "'s") {
        auto updated = s_suffix(phonemes);
        if (updated.has_value()) phonemes = *updated;
    } else if (lowered_suffix == "ed" || lowered_suffix == "'d") {
        auto updated = ed_suffix(phonemes);
        if (updated.has_value()) phonemes = *updated;
    } else if (lowered_suffix == "ing") {
        auto updated = ing_suffix(phonemes);
        if (updated.has_value()) phonemes = *updated;
    }
    return LexiconResult{phonemes, rating};
}

std::optional<LexiconResult> Lexicon::operator()(const MToken & token, const TokenContext & ctx) const {
    std::string word = token.meta.alias.value_or(token.text);
    std::replace(word.begin(), word.end(), static_cast<char>(0x91), '\'');
    std::replace(word.begin(), word.end(), static_cast<char>(0x92), '\'');

    std::optional<float> stress;
    if (word != lower_ascii(word)) {
        stress = word == upper_ascii(word) ? 2.0f : 0.5f;
    }
    if (const auto result = get_word(word, token.tag, stress, ctx)) {
        std::string phonemes = result->phonemes;
        if (token.meta.currency.has_value()) {
            const auto it = kCurrencies.find(*token.meta.currency);
            if (it != kCurrencies.end()) {
                auto unit = get_word(it->second.first + "s", std::nullopt, std::nullopt, TokenContext{});
                if (unit.has_value()) {
                    phonemes += " " + unit->phonemes;
                }
            }
        }
        return LexiconResult{apply_stress(phonemes, token.meta.stress), result->rating};
    }
    if (is_number_token(word, token.meta.is_head)) {
        if (const auto result = get_number(word, token.meta.currency, token.meta.is_head)) {
            return LexiconResult{apply_stress(result->phonemes, token.meta.stress), result->rating};
        }
    }
    if (!std::all_of(word.begin(), word.end(), [](unsigned char ch) { return std::isalpha(ch) != 0; })) {
        return std::nullopt;
    }
    return std::nullopt;
}

EnglishG2P::EnglishG2P(bool british)
    : lexicon_(british) {}

EnglishG2P::EnglishG2P(std::filesystem::path lexicon_dir, bool british)
    : lexicon_(
          lexicon_dir / (british ? "gb_gold.tsv" : "us_gold.tsv"),
          lexicon_dir / (british ? "gb_silver.tsv" : "us_silver.tsv"),
          british) {}

PreprocessResult EnglishG2P::preprocess(const std::string & text) {
    PreprocessResult result;
    result.text = trim_copy(text);
    std::string rebuilt;
    size_t token_index = 0;
    for (size_t i = 0; i < result.text.size();) {
        if (result.text[i] == '[') {
            const size_t close_text = result.text.find(']', i + 1);
            const size_t open_feat = close_text == std::string::npos ? std::string::npos : result.text.find('(', close_text + 1);
            const size_t close_feat = open_feat == std::string::npos ? std::string::npos : result.text.find(')', open_feat + 1);
            if (close_text != std::string::npos && open_feat == close_text + 1 && close_feat != std::string::npos) {
                const std::string visible = result.text.substr(i + 1, close_text - i - 1);
                const std::string feature = result.text.substr(open_feat + 1, close_feat - open_feat - 1);
                rebuilt += visible;
                const size_t emitted_count = count_words(visible);
                FeatureValue value;
                if (!feature.empty()) {
                    int64_t as_int = 0;
                    const auto int_result = std::from_chars(feature.data(), feature.data() + feature.size(), as_int);
                    if (int_result.ec == std::errc() && int_result.ptr == feature.data() + feature.size()) {
                        value = as_int;
                    } else if (feature == "0.5" || feature == "+0.5") {
                        value = 0.5f;
                    } else if (feature == "-0.5") {
                        value = -0.5f;
                    } else if (feature.size() > 1 && feature.front() == '/' && feature.back() == '/') {
                        value = feature.substr(0, feature.size() - 1);
                    } else if (feature.size() > 1 && feature.front() == '#' && feature.back() == '#') {
                        value = feature.substr(0, feature.size() - 1);
                    } else {
                        value = std::string();
                    }
                    if (emitted_count != 0) {
                        result.features[token_index] = value;
                    }
                }
                token_index += emitted_count;
                i = close_feat + 1;
                continue;
            }
        }
        rebuilt.push_back(result.text[i]);
        ++i;
    }
    result.text = rebuilt;
    return result;
}

std::vector<MToken> EnglishG2P::tokenize(
    const std::string & text,
    const std::unordered_map<size_t, FeatureValue> & features) const {
    std::vector<MToken> tokens;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) != 0) {
            ++i;
        }
        if (i >= text.size()) {
            break;
        }
        size_t begin = i;
        if (std::isalpha(static_cast<unsigned char>(text[i])) != 0 || std::isdigit(static_cast<unsigned char>(text[i])) != 0) {
            while (i < text.size() && (std::isalnum(static_cast<unsigned char>(text[i])) != 0 || text[i] == '\'' || text[i] == '.' || text[i] == ',' || text[i] == '-' || text[i] == '/' || text[i] == '_' || static_cast<unsigned char>(text[i]) >= 0x80)) {
                ++i;
            }
        } else {
            ++i;
        }
        size_t end = i;
        size_t ws_begin = i;
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) != 0) {
            ++i;
        }
        MToken token;
        token.text = text.substr(begin, end - begin);
        token.tag = guess_tag(token.text);
        token.whitespace = text.substr(ws_begin, i - ws_begin);
        token.meta.num_flags = "";
        tokens.push_back(std::move(token));
    }
    for (const auto & [index, value] : features) {
        if (index >= tokens.size()) {
            continue;
        }
        auto & token = tokens[index];
        if (const auto * as_int = std::get_if<int64_t>(&value)) {
            token.meta.stress = static_cast<float>(*as_int);
        } else if (const auto * as_float = std::get_if<float>(&value)) {
            token.meta.stress = *as_float;
        } else if (const auto * as_string = std::get_if<std::string>(&value)) {
            if (starts_with(*as_string, "/")) {
                token.phonemes = as_string->substr(1);
                token.meta.rating = 5;
            } else if (starts_with(*as_string, "#")) {
                token.meta.num_flags = as_string->substr(1);
            }
        }
    }
    for (size_t idx = 0; idx < tokens.size(); ++idx) {
        auto & token = tokens[idx];
        const std::string lowered = lower_ascii(token.text);
        const std::string next = idx + 1 < tokens.size() ? lower_ascii(tokens[idx + 1].text) : std::string();
        if (lowered == "used" && next == "to") {
            token.tag = "VBD";
        } else if (lowered == "by" && next != "the" && next != "way") {
            token.tag = "RB";
        } else if (lowered == "read" && idx > 0 && lower_ascii(tokens[idx - 1].text) == "to") {
            token.tag = "VB";
        }
    }
    return tokens;
}

MToken EnglishG2P::merge_token_pair(
    const MToken & left,
    const MToken & right,
    const std::optional<std::string> & unk) {
    MToken merged;
    merged.text.reserve(left.text.size() + left.whitespace.size() + right.text.size());
    merged.text += left.text;
    merged.text += left.whitespace;
    merged.text += right.text;
    merged.tag = left.tag;
    merged.whitespace = right.whitespace;
    merged.meta.is_head = left.meta.is_head;
    merged.meta.prespace = left.meta.prespace;
    merged.meta.num_flags = left.meta.num_flags;
    if (unk.has_value()) {
        std::string phonemes;
        phonemes += left.phonemes.value_or(*unk);
        if (right.meta.prespace && !phonemes.empty() &&
            !std::isspace(static_cast<unsigned char>(phonemes.back())) &&
            right.phonemes.has_value()) {
            phonemes.push_back(' ');
        }
        phonemes += right.phonemes.value_or(*unk);
        merged.phonemes = std::move(phonemes);
    }
    if (left.meta.rating.has_value() && right.meta.rating.has_value()) {
        merged.meta.rating = std::min(*left.meta.rating, *right.meta.rating);
    } else if (left.meta.rating.has_value()) {
        merged.meta.rating = left.meta.rating;
    } else {
        merged.meta.rating = right.meta.rating;
    }
    if (left.meta.stress.has_value() && right.meta.stress.has_value()) {
        if (*left.meta.stress == *right.meta.stress) {
            merged.meta.stress = left.meta.stress;
        }
    } else if (left.meta.stress.has_value()) {
        merged.meta.stress = left.meta.stress;
    } else {
        merged.meta.stress = right.meta.stress;
    }
    merged.meta.currency = right.meta.currency.has_value() ? right.meta.currency : left.meta.currency;
    return merged;
}

MToken EnglishG2P::merge_tokens(
    const std::vector<MToken> & tokens,
    size_t begin,
    size_t end,
    const std::optional<std::string> & unk) {
    MToken merged;
    merged.text.reserve((end - begin) * 4);
    for (size_t i = begin; i < end; ++i) {
        merged.text += tokens[i].text;
        if (i + 1 != end) {
            merged.text += tokens[i].whitespace;
        }
    }
    merged.tag = tokens[begin].tag;
    merged.whitespace = tokens[end - 1].whitespace;
    merged.meta.is_head = tokens[begin].meta.is_head;
    merged.meta.prespace = tokens[begin].meta.prespace;
    merged.meta.num_flags = tokens[begin].meta.num_flags;
    std::optional<int> rating;
    std::optional<float> stress;
    std::optional<std::string> currency;
    if (unk.has_value()) {
        std::string phonemes;
        for (size_t i = begin; i < end; ++i) {
            const auto & token = tokens[i];
            if (token.meta.prespace && !phonemes.empty() && !std::isspace(static_cast<unsigned char>(phonemes.back())) && token.phonemes.has_value()) {
                phonemes.push_back(' ');
            }
            phonemes += token.phonemes.value_or(*unk);
        }
        merged.phonemes = phonemes;
    }
    for (size_t i = begin; i < end; ++i) {
        const auto & token = tokens[i];
        if (!rating.has_value()) {
            rating = token.meta.rating;
        } else if (token.meta.rating.has_value()) {
            rating = std::min(*rating, *token.meta.rating);
        }
        if (token.meta.stress.has_value()) {
            if (stress.has_value() && *stress != *token.meta.stress) {
                stress = std::nullopt;
            } else if (!stress.has_value()) {
                stress = token.meta.stress;
            }
        }
        if (token.meta.currency.has_value()) {
            currency = token.meta.currency;
        }
    }
    merged.meta.rating = rating;
    merged.meta.stress = stress;
    merged.meta.currency = currency;
    return merged;
}

std::vector<MToken> EnglishG2P::fold_left(const std::vector<MToken> & tokens) {
    std::vector<MToken> result;
    result.reserve(tokens.size());
    for (const auto & token : tokens) {
        if (!result.empty() && !token.meta.is_head) {
            result.back() = merge_token_pair(result.back(), token, std::string());
        } else {
            result.push_back(token);
        }
    }
    return result;
}

std::vector<std::variant<MToken, std::vector<MToken>>> EnglishG2P::retokenize(const std::vector<MToken> & tokens) {
    std::vector<std::variant<MToken, std::vector<MToken>>> words;
    std::optional<std::string> currency;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto & token = tokens[i];
        std::vector<MToken> subtokens;
        if (!token.meta.alias.has_value() && !token.phonemes.has_value()) {
            const auto parts = subtokenize(token.text);
            subtokens.reserve(parts.size());
            for (const auto & part : parts) {
                MToken sub = token;
                sub.text = part;
                sub.tag = part == token.text ? token.tag : guess_tag(part);
                sub.whitespace.clear();
                sub.meta.is_head = true;
                sub.meta.prespace = false;
                subtokens.push_back(std::move(sub));
            }
        } else {
            subtokens.push_back(token);
        }
        if (!subtokens.empty()) {
            subtokens.back().whitespace = token.whitespace;
        }
        for (size_t j = 0; j < subtokens.size(); ++j) {
            auto & sub = subtokens[j];
            if (sub.meta.alias.has_value() || sub.phonemes.has_value()) {
            } else if (sub.tag == "$" && contains_key(kCurrencies, sub.text)) {
                currency = sub.text;
                sub.phonemes = "";
                sub.meta.rating = 4;
            } else if (sub.tag == ":" && (sub.text == "-" || sub.text == "–")) {
                sub.phonemes = "—";
                sub.meta.rating = 3;
            } else if (contains_key(kPunctTags, sub.tag) &&
                       !std::all_of(sub.text.begin(), sub.text.end(), [](unsigned char ch) { return std::isalpha(ch) != 0; })) {
                const auto it = kPunctTagPhonemes.find(sub.tag);
                if (it != kPunctTagPhonemes.end()) {
                    sub.phonemes = it->second;
                } else {
                    std::string kept;
                    for (char ch : sub.text) {
                        if (kPuncts.find(ch) != std::string::npos) {
                            kept.push_back(ch);
                        }
                    }
                    sub.phonemes = kept;
                }
                sub.meta.rating = 4;
            } else if (currency.has_value()) {
                if (sub.tag != "CD") {
                    currency.reset();
                } else if (j + 1 == subtokens.size() && (i + 1 == tokens.size() || tokens[i + 1].tag != "CD")) {
                    sub.meta.currency = currency;
                }
            } else if (0 < j && j + 1 < subtokens.size() && sub.text == "2" &&
                       !subtokens[j - 1].text.empty() && !subtokens[j + 1].text.empty() &&
                       std::isalpha(static_cast<unsigned char>(subtokens[j - 1].text.back())) != 0 &&
                       std::isalpha(static_cast<unsigned char>(subtokens[j + 1].text.front())) != 0) {
                sub.meta.alias = "to";
            }

            if (sub.meta.alias.has_value() || sub.phonemes.has_value()) {
                words.push_back(sub);
            } else if (!words.empty() && std::holds_alternative<std::vector<MToken>>(words.back()) &&
                       std::get<std::vector<MToken>>(words.back()).back().whitespace.empty()) {
                sub.meta.is_head = false;
                std::get<std::vector<MToken>>(words.back()).push_back(sub);
            } else {
                if (sub.whitespace.empty()) {
                    words.push_back(std::vector<MToken>{sub});
                } else {
                    words.push_back(sub);
                }
            }
        }
    }
    for (auto & word : words) {
        if (std::holds_alternative<std::vector<MToken>>(word)) {
            auto & group = std::get<std::vector<MToken>>(word);
            if (group.size() == 1) {
                word = group.front();
            }
        }
    }
    return words;
}

void EnglishG2P::resolve_tokens(std::vector<MToken> & tokens) {
    std::string text;
    for (size_t i = 0; i < tokens.size(); ++i) {
        text += tokens[i].text;
        if (i + 1 != tokens.size()) {
            text += tokens[i].whitespace;
        }
    }
    bool saw_alpha = false;
    bool saw_digit = false;
    bool saw_other = false;
    for (char ch : text) {
        if (contains_key(kSubtokenJunk, ch)) {
            continue;
        }
        const int klass = english_g2p_char_class(ch);
        saw_alpha = saw_alpha || klass == 0;
        saw_digit = saw_digit || klass == 1;
        saw_other = saw_other || klass == 2;
    }
    const bool prespace =
        contains_whitespace(text) ||
        text.find('/') != std::string::npos ||
        static_cast<int>(saw_alpha) + static_cast<int>(saw_digit) + static_cast<int>(saw_other) > 1;
    for (size_t i = 0; i < tokens.size(); ++i) {
        auto & token = tokens[i];
        if (!token.phonemes.has_value()) {
            if (i + 1 == tokens.size() && token.text.size() == 1 && kNonQuotePuncts.find(token.text[0]) != std::string::npos) {
                token.phonemes = token.text;
                token.meta.rating = 3;
            } else if (all_subtoken_junk(token.text)) {
                token.phonemes = "";
                token.meta.rating = 3;
            }
        } else if (i > 0) {
            token.meta.prespace = prespace;
        }
    }
    if (prespace) {
        return;
    }
    struct WeightedIndex {
        bool primary = false;
        int weight = 0;
        size_t index = 0;
    };
    std::vector<WeightedIndex> indices;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i].phonemes.has_value() && !tokens[i].phonemes->empty()) {
            indices.push_back({tokens[i].phonemes->find(kPrimaryStress) != std::string::npos, stress_weight(*tokens[i].phonemes), i});
        }
    }
    int primary_count = 0;
    for (const auto & entry : indices) primary_count += entry.primary ? 1 : 0;
    if (indices.size() == 2 && tokens[indices[0].index].text.size() == 1) {
        tokens[indices[1].index].phonemes = Lexicon::apply_stress(*tokens[indices[1].index].phonemes, -0.5f);
        return;
    }
    if (indices.size() < 2 || primary_count <= static_cast<int>((indices.size() + 1) / 2)) {
        return;
    }
    std::sort(indices.begin(), indices.end(), [](const WeightedIndex & lhs, const WeightedIndex & rhs) {
        if (lhs.primary != rhs.primary) return lhs.primary < rhs.primary;
        if (lhs.weight != rhs.weight) return lhs.weight < rhs.weight;
        return lhs.index < rhs.index;
    });
    indices.resize(indices.size() / 2);
    for (const auto & entry : indices) {
        tokens[entry.index].phonemes = Lexicon::apply_stress(*tokens[entry.index].phonemes, -0.5f);
    }
}

TokenContext EnglishG2P::token_context(const TokenContext & ctx, const std::optional<std::string> & phonemes, const MToken & token) {
    std::optional<bool> vowel = ctx.future_vowel;
    if (phonemes.has_value()) {
        for (size_t i = 0; i < phonemes->size();) {
            const size_t width = utf8_codepoint_width(*phonemes, i);
            const char * symbol = phonemes->data() + i;
            if (utf8_inventory_contains(kVowels, symbol, width)) {
                vowel = true;
                break;
            }
            if (utf8_inventory_contains(kConsonants, symbol, width) ||
                (width == 1 && kNonQuotePuncts.find(*symbol) != std::string::npos)) {
                vowel = false;
                break;
            }
            i += width;
        }
    }
    const bool future_to = token.text == "to" || token.text == "To" || (token.text == "TO" && (token.tag == "TO" || token.tag == "IN"));
    return TokenContext{vowel, future_to};
}

std::string EnglishG2P::tokens_to_phonemes(const std::vector<MToken> & tokens) {
    std::string result;
    size_t reserve = 0;
    for (const auto & token : tokens) {
        if (token.phonemes.has_value()) {
            reserve += token.phonemes->size();
            if (!token.whitespace.empty()) {
                ++reserve;
            }
        }
    }
    result.reserve(reserve);
    bool pending_space = false;
    for (const auto & token : tokens) {
        if (!token.phonemes.has_value()) {
            continue;
        }
        if (pending_space && !result.empty()) {
            result.push_back(' ');
        }
        result += *token.phonemes;
        pending_space = !token.whitespace.empty();
    }
    return result;
}

std::pair<std::string, std::vector<MToken>> EnglishG2P::operator()(const std::string & text, bool enable_preprocess) const {
    const PreprocessResult pre = enable_preprocess ? preprocess(text) : PreprocessResult{text, {}};
    auto tokens = tokenize(pre.text, pre.features);
    tokens = fold_left(tokens);
    auto words = retokenize(tokens);
    TokenContext ctx;
    std::vector<MToken> resolved;
    resolved.reserve(words.size());
    for (auto it = words.rbegin(); it != words.rend(); ++it) {
        if (std::holds_alternative<MToken>(*it)) {
            MToken token = std::move(std::get<MToken>(*it));
            if (!token.phonemes.has_value()) {
                auto result = lexicon_(token, ctx);
                if (!result.has_value()) {
                    result = resolve_with_safe_fallback(lexicon_, token);
                }
                token.phonemes = result->phonemes;
                token.meta.rating = result->rating;
            }
            ctx = token_context(ctx, token.phonemes, token);
            resolved.push_back(std::move(token));
            continue;
        }

        auto group = std::move(std::get<std::vector<MToken>>(*it));
        bool should_fallback = false;
        size_t left = 0;
        size_t right = group.size();
        while (left < right) {
            bool blocked = false;
            for (size_t i = left; i < right; ++i) {
                if (group[i].meta.alias.has_value() || group[i].phonemes.has_value()) {
                    blocked = true;
                    break;
                }
            }
            if (!blocked) {
                MToken merged = merge_tokens(group, left, right, std::nullopt);
                if (auto result = lexicon_(merged, ctx)) {
                    group[left].phonemes = result->phonemes;
                    group[left].meta.rating = result->rating;
                    for (size_t i = left + 1; i < right; ++i) {
                        group[i].phonemes = "";
                        group[i].meta.rating = result->rating;
                    }
                    ctx = token_context(ctx, group[left].phonemes, merged);
                    right = left;
                    left = 0;
                    continue;
                }
            }
            if (left + 1 < right) {
                ++left;
                continue;
            }
            --right;
            if (!group[right].phonemes.has_value()) {
                if (all_subtoken_junk(group[right].text)) {
                    group[right].phonemes = "";
                    group[right].meta.rating = 3;
                } else {
                    should_fallback = true;
                    break;
                }
            }
            left = 0;
        }
        if (should_fallback) {
            MToken merged = merge_tokens(group, 0, group.size(), std::nullopt);
            const LexiconResult fallback_result = resolve_with_safe_fallback(lexicon_, merged);
            group[0].phonemes = fallback_result.phonemes;
            group[0].meta.rating = fallback_result.rating;
            for (size_t i = 1; i < group.size(); ++i) {
                group[i].phonemes = "";
                group[i].meta.rating = fallback_result.rating;
            }
        } else {
            resolve_tokens(group);
        }
        for (auto rit = group.rbegin(); rit != group.rend(); ++rit) {
            ctx = token_context(ctx, rit->phonemes, *rit);
        }
        std::reverse(group.begin(), group.end());
        for (auto & token : group) {
            resolved.push_back(std::move(token));
        }
    }
    std::reverse(resolved.begin(), resolved.end());
    for (auto & token : resolved) {
        if (token.phonemes.has_value()) {
            replace_all(*token.phonemes, "ɾ", "T");
            replace_all(*token.phonemes, "ʔ", "t");
        }
    }
    return {tokens_to_phonemes(resolved), resolved};
}

}  // namespace kokoro_ggml::g2p_en
