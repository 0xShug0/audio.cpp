#include "engine/community_models/kale_tts/frontend.h"

#include <utf8proc.h>

#include <algorithm>
#include <cctype>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace engine::models::kale_tts {
namespace {

constexpr int kIpaWithUnderscoreSeparator = ('_' << 8) | 0x02;
constexpr int32_t kSpaceId = 16;

std::vector<std::string> code_points(const std::string & text) {
    std::vector<std::string> out;
    size_t at = 0;
    while (at < text.size()) {
        utf8proc_int32_t c = 0;
        const auto used = utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t *>(text.data() + at),
                                           static_cast<utf8proc_ssize_t>(text.size() - at), &c);
        if (used <= 0) {
            throw std::runtime_error("kale-tts: the text is not valid UTF-8");
        }
        out.push_back(text.substr(at, static_cast<size_t>(used)));
        at += static_cast<size_t>(used);
    }
    return out;
}

// word<TAB>IPA lines; '#' starts a comment; "word#v" (the form before a vowel) is not used here
void read_table(const std::string & text, std::unordered_map<std::string, std::string> & out) {
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.empty() || line[0] == '#') continue;
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string key = line.substr(0, tab);
        std::string value = line.substr(tab + 1);
        while (!value.empty() && (value.back() == '\r' || value.back() == ' ')) value.pop_back();
        if (key.empty() || value.empty() || key.find('#') != std::string::npos) continue;
        out[key] = value;
    }
}

std::string lower(std::string text) {
    for (char & c : text) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return text;
}

bool is_word_char(unsigned char c) {
    return std::isalnum(c) != 0 || c >= 0x80;
}

// Technical terms: exact written form -> words to say, whole words only.
std::string apply_terms(const std::string & text, const std::unordered_map<std::string, std::string> & terms) {
    if (terms.empty()) return text;
    std::vector<std::pair<std::string, std::string>> sorted(terms.begin(), terms.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto & a, const auto & b) { return a.first.size() > b.first.size(); });
    std::string out;
    size_t i = 0;
    while (i < text.size()) {
        bool done = false;
        const bool start = i == 0 || !is_word_char(static_cast<unsigned char>(text[i - 1]));
        for (const auto & [written, spoken] : sorted) {
            if (written[0] != '.' && !start) continue;
            if (text.compare(i, written.size(), written) != 0) continue;
            const size_t end = i + written.size();
            if (end < text.size() && is_word_char(static_cast<unsigned char>(text[end]))) continue;
            out += (written[0] == '.' && !start) ? " " + spoken : spoken;
            i = end;
            done = true;
            break;
        }
        if (!done) out += text[i++];
    }
    return out;
}

const std::set<std::string> & abbreviations() {
    static const std::set<std::string> value = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "no", "inc", "ltd", "co", "e.g", "i.e"};
    return value;
}

std::vector<std::string> split_sentences(const std::string & text) {
    std::vector<std::string> out;
    std::string current;
    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        current += c;
        const bool end = c == '.' || c == '!' || c == '?';
        const bool next_space = i + 1 < text.size() && std::isspace(static_cast<unsigned char>(text[i + 1])) != 0;
        if (end && next_space) {
            size_t w = current.size() - 1;
            while (w > 0 && is_word_char(static_cast<unsigned char>(current[w - 1]))) --w;
            const std::string last = lower(current.substr(w, current.size() - 1 - w));
            const bool initial = last.size() == 1 && std::isalpha(static_cast<unsigned char>(last[0])) != 0;
            if (c == '.' && (abbreviations().count(last) || initial)) continue;
            out.push_back(current);
            current.clear();
        }
    }
    out.push_back(current);
    std::vector<std::string> kept;
    for (auto & s : out) {
        const size_t a = s.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        const size_t b = s.find_last_not_of(" \t\r\n");
        kept.push_back(s.substr(a, b - a + 1));
    }
    return kept;
}

std::vector<std::string> split_whitespace(const std::string & text) {
    std::vector<std::string> out;
    std::istringstream in(text);
    std::string word;
    while (in >> word) out.push_back(word);
    return out;
}

}  // namespace

std::vector<std::string> kokoro_native(const std::vector<std::string> & chars, bool british) {
    static const std::set<std::string> vowels = [] {
        std::set<std::string> v;
        for (const auto & c : code_points("aeiouɑɐɒæəɘɚɛɜɝɞɪɨʊʉʌɔøœɤɯʏyAIWOYᵊᵻ")) v.insert(c);
        return v;
    }();
    std::vector<std::pair<std::string, std::string>> pairs = {{"eɪ", "A"}, {"aɪ", "I"}, {"aʊ", "W"}, {"ɔɪ", "Y"}, {"tʃ", "ʧ"}, {"dʒ", "ʤ"}};
    pairs.emplace_back(british ? std::make_pair(std::string("əʊ"), std::string("Q")) : std::make_pair(std::string("oʊ"), std::string("O")));
    std::vector<std::string> out;
    for (size_t i = 0; i < chars.size();) {
        if (i + 1 < chars.size()) {
            const std::string two = chars[i] + chars[i + 1];
            const auto found = std::find_if(pairs.begin(), pairs.end(), [&](const auto & p) { return p.first == two; });
            if (found != pairs.end()) {
                out.push_back(found->second);
                i += 2;
                continue;
            }
        }
        const std::string & c = chars[i++];
        if (british) {
            out.push_back(c);
            continue;
        }
        if (c == "ː") continue;
        if (c == "ɚ") {
            out.insert(out.end(), {"ə", "ɹ"});
        } else if (c == "ɝ") {
            out.insert(out.end(), {"ɜ", "ɹ"});
        } else if (c == "ɾ") {
            out.push_back("T");
        } else {
            out.push_back(c);
        }
    }
    static const std::set<std::string> ends = {" ", ".", ",", "!", "?", ";", ":"};
    for (size_t k = 1; k + 1 < out.size(); ++k) {
        if (out[k] == "ə" && (out[k + 1] == "l" || out[k + 1] == "n" || out[k + 1] == "m") && (k + 2 == out.size() || ends.count(out[k + 2])) &&
            !vowels.count(out[k - 1]) && out[k - 1] != "ˈ" && out[k - 1] != "ˌ" && out[k - 1] != " ") {
            out[k] = "ᵊ";
        }
    }
    return out;
}

KaleFrontend::KaleFrontend(const KaleAssets & assets, std::filesystem::path espeak_library, std::filesystem::path espeak_data)
    : assets_(assets) {
    const auto & r = assets.resources;
    read_table(r.read_text("words_lexicon"), words_us_);
    read_table(r.read_text("words_override"), words_us_);
    read_table(r.read_text("words_terms"), terms_);
    if (r.has_file("words_gb_lexicon")) {
        read_table(r.read_text("words_gb_lexicon"), words_gb_);
        read_table(r.read_text("words_gb_override"), words_gb_);
    }
    if (assets.config.ettin) {
        ettin_ = std::make_unique<EttinTokenizer>(r.read_text("ettin_tokenizer"));
    }
    // eSpeak-ng reads only what the word lists don't have (numbers, rare words); load it when first needed.
    espeak_library_ = std::move(espeak_library);
    espeak_data_ = std::move(espeak_data);
}

KaleFrontend::~KaleFrontend() = default;

std::string KaleFrontend::word_ipa(const std::string & word, bool british) const {
    // a word = letters with punctuation around it; the list has the letters
    size_t a = 0;
    size_t b = word.size();
    while (a < b && !is_word_char(static_cast<unsigned char>(word[a]))) ++a;
    while (b > a && !is_word_char(static_cast<unsigned char>(word[b - 1]))) --b;
    std::string before = word.substr(0, a);
    std::string after = word.substr(b);
    const std::string core = word.substr(a, b - a);
    std::string keep;
    for (const char c : after) {
        if (c == '.' || c == ',' || c == '!' || c == '?' || c == ';' || c == ':') keep += c;
    }
    if (core.empty()) return keep;
    // numbers go to eSpeak-ng whole, with their signs and units ($1,299.99  -12  0.5%  2½)
    const bool number = std::any_of(word.begin(), word.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; });
    const std::string spoken = number ? word.substr(0, word.size() - keep.size()) : core;
    const auto & table = (british && !words_gb_.empty()) ? words_gb_ : words_us_;
    std::string key = lower(core);
    if (key.size() > 2 && key.compare(key.size() - 2, 2, "’s") == 0) key.replace(key.size() - 2, 2, "'s");
    if (const auto found = table.find(key); !number && found != table.end()) return found->second + keep;
    // hyphenated words: every part from the list
    if (!number && key.find('-') != std::string::npos) {
        std::string out;
        std::istringstream parts(key);
        std::string part;
        bool all = true;
        while (std::getline(parts, part, '-')) {
            const auto found = table.find(part);
            if (part.empty() || found == table.end()) {
                all = false;
                break;
            }
            out += (out.empty() ? "" : " ") + found->second;
        }
        if (all && !out.empty()) return out + keep;
    }
    auto & espeak = british ? espeak_gb_ : espeak_us_;
    if (!espeak) {
        espeak = std::make_unique<engine::audio::EspeakPhonemizer>(espeak_library_, espeak_data_,
                                                                   british ? std::vector<std::string>{"en-gb", "en"} : std::vector<std::string>{"en-us"});
    }
    std::string ipa = espeak->phonemize(spoken, kIpaWithUnderscoreSeparator);
    ipa.erase(std::remove(ipa.begin(), ipa.end(), '_'), ipa.end());
    while (!ipa.empty() && (ipa.back() == ' ' || ipa.back() == '\n')) ipa.pop_back();
    size_t start = ipa.find_first_not_of(" \n");
    return (start == std::string::npos ? std::string() : ipa.substr(start)) + keep;
}

std::vector<int32_t> KaleFrontend::ids(const std::string & ipa, bool british) const {
    std::vector<std::string> chars;
    for (const auto & c : code_points(ipa)) {
        if (assets_.config.vocab.count(c) || c == "ɚ" || c == "ɝ" || c == "ɾ" || c == "ː") chars.push_back(c);
    }
    if (assets_.config.kokoro_native) chars = kokoro_native(chars, british);
    std::vector<int32_t> out;
    for (const auto & c : chars) {
        if (const auto found = assets_.config.vocab.find(c); found != assets_.config.vocab.end()) out.push_back(found->second);
    }
    return out;
}

std::vector<KaleFrontend::Sentence> KaleFrontend::sentences(const std::string & text, const std::string & accent) const {
    const bool british = accent == "en-gb";
    std::vector<Sentence> out;
    for (const std::string & raw : split_sentences(apply_terms(text, terms_))) {
        Sentence sentence;
        sentence.text = raw;
        for (const std::string & word : split_whitespace(raw)) {
            sentence.word_ids.push_back(ids(word_ipa(word, british), british));
        }
        for (size_t w = 0; w < sentence.word_ids.size(); ++w) {
            if (w) sentence.ids.push_back(kSpaceId);
            sentence.ids.insert(sentence.ids.end(), sentence.word_ids[w].begin(), sentence.word_ids[w].end());
        }
        if (!sentence.ids.empty()) out.push_back(std::move(sentence));
    }
    return out;
}

std::vector<std::string> KaleFrontend::sentence_texts(const std::string & text) const {
    return split_sentences(apply_terms(text, terms_));
}

std::vector<int32_t> KaleFrontend::phoneme_ids(const std::string & phonemes) const {
    std::vector<int32_t> out;
    for (const auto & c : code_points(phonemes)) {
        const auto found = assets_.config.vocab.find(c);
        if (found == assets_.config.vocab.end()) {
            throw std::runtime_error("kale-tts: phoneme '" + c + "' is not in the model's symbols (use Kokoro's own symbols)");
        }
        out.push_back(found->second);
    }
    return out;
}

}  // namespace engine::models::kale_tts
