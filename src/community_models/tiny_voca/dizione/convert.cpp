#include "convert.h"

#include "align.h"

#include <initializer_list>
#include <unordered_map>
#include <unordered_set>

namespace dizione {

namespace {

using Symbols = std::initializer_list<std::string_view>;

bool one_of(std::string_view value, Symbols options) {
    for (const std::string_view option : options) {
        if (value == option) {
            return true;
        }
    }
    return false;
}

bool one_of(StrView value, std::initializer_list<StrView> options) {
    for (const StrView option : options) {
        if (value == option) {
            return true;
        }
    }
    return false;
}

const std::unordered_set<std::string> & arpa_vowels() {
    static const std::unordered_set<std::string> value = {
        "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
    };
    return value;
}

const std::unordered_map<std::string, std::string> & consonants() {
    static const std::unordered_map<std::string, std::string> value = {
        {"B", "b"}, {"CH", "tʃ"}, {"D", "d"}, {"DH", "ð"}, {"F", "f"}, {"G", "ɡ"}, {"HH", "h"}, {"JH", "dʒ"},
        {"K", "k"}, {"L", "l"}, {"M", "m"}, {"N", "n"}, {"NG", "ŋ"}, {"P", "p"}, {"R", "ɹ"}, {"S", "s"},
        {"SH", "ʃ"}, {"T", "t"}, {"TH", "θ"}, {"V", "v"}, {"W", "w"}, {"Y", "j"}, {"Z", "z"}, {"ZH", "ʒ"},
    };
    return value;
}

const std::unordered_map<std::string, std::string> & full_vowels() {
    static const std::unordered_map<std::string, std::string> value = {
        {"AA", "ɑː"}, {"AE", "æ"}, {"AH", "ʌ"}, {"AO", "ɔː"}, {"AW", "aʊ"}, {"AY", "aɪ"}, {"EH", "ɛ"}, {"ER", "ɜː"},
        {"EY", "eɪ"}, {"IH", "ɪ"}, {"IY", "iː"}, {"OW", "oʊ"}, {"OY", "ɔɪ"}, {"UH", "ʊ"}, {"UW", "uː"},
    };
    return value;
}

const std::unordered_set<std::string> & arpabet() {
    static const std::unordered_set<std::string> value = [] {
        std::unordered_set<std::string> out;
        for (const auto & entry : consonants()) {
            out.insert(entry.first);
        }
        for (const auto & vowel : arpa_vowels()) {
            for (const char digit : {'0', '1', '2'}) {
                out.insert(vowel + digit);
            }
        }
        return out;
    }();
    return value;
}

bool is_sibilant(std::string_view base) {
    return one_of(base, {"S", "Z", "SH", "ZH", "CH", "JH"});
}

bool vowel_letter(char32_t c) {
    return c == U'a' || c == U'e' || c == U'i' || c == U'o' || c == U'u' || c == U'y';
}

Str vowel_letters(StrView run) {
    Str out;
    for (const char32_t c : run) {
        if (vowel_letter(c)) {
            out.push_back(c);
        }
    }
    return out;
}

constexpr const char * kSyllabicN = "n\xCC\xA9";
constexpr int kNoStress = -1;           // a symbol without a stress digit

// The aligned pronunciation of one word and the questions the rules ask of it.
struct Word {
    Str word;
    std::vector<std::string> bases;
    std::vector<int> stress;
    bool aligned = false;
    std::vector<Str> runs;
    std::vector<long> starts;
    long count = 0;

    Word(StrView word_in, const std::vector<std::string> & arpabet_in, const std::optional<std::vector<Str>> & runs_in)
        : word(word_in) {
        for (const auto & symbol : arpabet_in) {
            bases.push_back(arpabet_base(symbol));
            if (symbol.empty()) {
                throw IndexError("string index out of range");
            }
            const char last = symbol.back();
            stress.push_back(last == '0' || last == '1' || last == '2' ? last - '0' : kNoStress);
        }
        for (size_t k = 0; k < bases.size(); ++k) {
            if (consonants().count(bases[k]) == 0 && arpa_vowels().count(bases[k]) == 0) {
                throw ValueError("'" + arpabet_in[k] + "' is not an ARPABET symbol");
            }
        }
        aligned = runs_in.has_value();
        runs = aligned ? *runs_in : std::vector<Str>(arpabet_in.size());
        long position = 0;
        for (const Str & run : runs) {
            starts.push_back(position);
            position += static_cast<long>(run.size());
        }
        count = static_cast<long>(bases.size());
    }

    // The symbol at `i`, empty outside the word (Python: None).
    std::string_view base(long i) const {
        return i >= 0 && i < count ? std::string_view(bases[static_cast<size_t>(i)]) : std::string_view();
    }

    bool exists(long i) const { return i >= 0 && i < count; }

    bool vowel(long i) const {
        return exists(i) && arpa_vowels().count(bases[static_cast<size_t>(i)]) != 0;
    }

    bool stressed(long i) const { return stress[static_cast<size_t>(i)] > 0; }

    bool unstressed(long i) const { return vowel(i) && !stressed(i); }

    const Str & run(long i) const { return runs.at(static_cast<size_t>(i)); }

    Str letters(long i) const { return vowel_letters(run(i)); }

    // Spelling that precedes the letters of segment `i`.
    Str before(long i) const {
        return aligned ? py::slice(word, 0, starts.at(static_cast<size_t>(i))) : Str();
    }

    // Spelling that follows the letters of segment `i`.
    Str after(long i) const {
        return aligned ? py::slice(word, starts.at(static_cast<size_t>(i)) + static_cast<long>(run(i).size())) : Str();
    }

    // True when exactly `symbols` follow segment `i` to the end of the word.
    bool ends_with(long i, Symbols symbols) const {
        if (count - (i + 1) != static_cast<long>(symbols.size())) {
            return false;
        }
        long k = i + 1;
        for (const std::string_view symbol : symbols) {
            if (bases[static_cast<size_t>(k++)] != symbol) {
                return false;
            }
        }
        return true;
    }

    bool first_vowel(long i) const {
        for (long k = 0; k < i && k < count; ++k) {
            if (arpa_vowels().count(bases[static_cast<size_t>(k)]) != 0) {
                return false;
            }
        }
        return true;
    }

    // Stress of the next vowel, kNoStress when there is none.
    int next_vowel_stress(long i) const {
        for (long j = i + 1; j < count; ++j) {
            if (arpa_vowels().count(bases[static_cast<size_t>(j)]) != 0) {
                return stress[static_cast<size_t>(j)];
            }
        }
        return kNoStress;
    }

    // Segment `i` is the "e" of an unstressed be-, de-, re- or pre-.
    bool weak_prefix(long i) const {
        return aligned && first_vowel(i) && letters(i) == U"e" && py::startswith_any(word, {U"be", U"de", U"re", U"pre"}) &&
            one_of(StrView(before(i)), {U"b", U"d", U"r", U"pr"});
    }
};

// AO + R: "oː" for the FORCE set (more, four, door, port, story), "ɔː" for
// NORTH (for, short, war). General American merges them; the model has both
// tokens, so the choice follows the spelling.
std::string north_or_force(const Word & w, long i, bool before_vowel) {
    if (!w.aligned) {
        return "ɔː";
    }
    const Str spelled = w.letters(i);
    if (one_of(StrView(spelled), {U"oa", U"oo", U"ou"})) {
        return "oː";
    }
    if (spelled != U"o") {
        return "ɔː";
    }
    const Str & run = w.run(i);
    const Str tail = py::slice(run, py::find(run, U"o") + 1) + w.after(i);     // the letters after the "o", normally starting with "r"
    const Str rest = py::slice(tail, 1);
    if (!before_vowel) {
        if (py::startswith(rest, U"e")) {            // more, store, before, ignored
            return "oː";
        }
        if (py::endswith(w.before(i), U"p") && py::startswith(rest, U"t")) {    // port, support, important
            return "oː";
        }
        // Words of the FORCE set written "or" before a consonant: they keep the
        // closer vowel that "more" and "four" have.
        for (const StrView spelling : {StrView(U"forc"), StrView(U"ford"), StrView(U"sword"), StrView(U"worn"), StrView(U"torn"),
                                       StrView(U"sworn"), StrView(U"borne"), StrView(U"shorn"), StrView(U"pork"), StrView(U"divorc")}) {
            if (py::contains(w.word, spelling)) {
                return "oː";
            }
        }
        return "ɔː";
    }
    if (py::startswith(rest, U"r")) {                // horror, torrent
        return "ɔː";
    }
    if (one_of(StrView(rest), {U"y", U"ies", U"ied", U"a", U"as", U"ah", U"er", U"ers"}) ||
        py::startswith_any(rest, {U"ia", U"io", U"iu", U"ing", U"age"})) {
        return "oː";                                 // story, flora, memorial, boring, storage
    }
    return "ɔː";
}

// A reduced vowel written "i": the barred i of "possible", "quality" and
// "terrify", the happy vowel of "beautiful", otherwise ɪ.
std::string weak_i(const Word & w, long i) {
    const std::string_view nxt = w.base(i + 1);
    const std::string_view nxt2 = w.base(i + 2);
    if (nxt == "F" && nxt2 == "AH" && w.base(i + 3) == "L") {
        return "i";
    }
    if ((nxt == "T" && nxt2 == "IY") || (nxt == "B" && one_of(nxt2, {"AH", "L"})) || (nxt == "F" && nxt2 == "AY")) {
        return "ᵻ";
    }
    if (nxt == "T" && w.ends_with(i, {"T"}) && one_of(w.base(i - 1), {"R", "ER"}) && py::endswith(w.word, U"ite")) {
        return "ᵻ";                                  // favorite
    }
    return "ɪ";
}

// A reduced vowel written "e" outside a weak prefix.
std::string weak_e(const Word & w, long i) {
    const std::string_view nxt = w.base(i + 1);
    const std::string_view prev = w.base(i - 1);
    if (nxt == "T" && (w.ends_with(i, {"T"}) || w.ends_with(i, {"T", "S"}))) {
        return prev == "L" ? "ə" : prev == "R" ? "ᵻ" : "ɪ";       // bracelet; secret; pocket, basket
    }
    if (nxt == "S" && (w.ends_with(i, {"S", "T"}) || w.ends_with(i, {"S", "T", "L", "IY"}))) {
        return "ɪ";                                  // nearest, honest, honestly
    }
    if (nxt == "JH") {
        return "ɪ";                                  // college, knowledge
    }
    if (nxt == "K" && w.base(i + 2) == "T") {
        return "ɛ";                                  // perfect, subject
    }
    if (nxt == "D" && w.ends_with(i, {"D"})) {
        return "ɪ";                                  // naked, wicked
    }
    return "ə";
}

// The vowel of "-ed" after t/d and of "-es" after a sibilant.
bool inflection(const Word & w, long i) {
    if (i + 1 >= w.count || (w.aligned && !py::contains(w.run(i) + w.run(i + 1), U'e'))) {
        return false;
    }
    const std::string_view nxt = w.base(i + 1);
    const std::string_view prev = w.base(i - 1);
    if (nxt == "D" && one_of(prev, {"T", "D"}) && (w.ends_with(i, {"D"}) || w.ends_with(i, {"D", "L", "IY"}))) {
        return true;
    }
    return nxt == "Z" && is_sibilant(prev) && w.ends_with(i, {"Z"});
}

// An unstressed first vowel right before the stressed syllable: the weak
// prefixes of "before", "return", "expect", "enjoy". Empty if no rule applies.
std::string first_syllable(const Word & w, long i) {
    if (!w.aligned || !w.first_vowel(i) || w.next_vowel_stress(i) <= 0) {
        return std::string();
    }
    const Str & word = w.word;
    if (i == 0 && py::startswith(word, U"e")) {
        if (py::startswith(word, U"ex")) {
            return "ɛ";
        }
        const Str second = py::slice(word, 1, 2);
        if (one_of(StrView(second), {U"n", U"m", U"s"}) && !py::slice(word, 2, 3).empty() && !vowel_letter(py::at(word, 2)) &&
            py::at(word, 2) != py::at(word, 1)) {
            return "ɛ";                              // enjoy, embrace, escape
        }
        return "ɪ";                                  // eleven, effect, enough
    }
    if (w.weak_prefix(i)) {
        if (py::startswith(word, U"pre") || (py::startswith(word, U"be") && one_of(StrView(py::slice(word, 2, 3)), {U"c", U"g"}))) {
            return "ɪ";                              // prevent; because, began
        }
        return "ᵻ";                                  // before, return, decide
    }
    if (w.letters(i) == U"i" && py::startswith(word, U"di") && word.size() > 3 && !vowel_letter(py::at(word, 2)) &&
        py::at(word, 2) != U's' && vowel_letter(py::at(word, 3))) {
        return "ᵻ";                                  // divide, direct
    }
    return std::string();
}

// Quality of an unstressed AH or IH.
std::string reduced(const Word & w, long i) {
    const std::string & name = w.bases[static_cast<size_t>(i)];
    const Str spelled = w.letters(i);
    const std::string_view nxt = w.base(i + 1);
    const std::string_view prev = w.base(i - 1);
    if (name == "IH" && nxt == "NG") {
        return "ɪ";
    }
    if (inflection(w, i)) {
        return "ᵻ";
    }
    if (name == "AH") {
        if (prev == "Y" && py::contains(w.run(i - 1) + w.run(i), U'u')) {
            return "ʊ";                              // popular, document
        }
        if (i == 0 && py::startswith(w.word, U"a")) {
            return "ɐ";                              // about, again
        }
        if (i == 0 && nxt == "N" && py::startswith(w.word, U"un")) {
            return "ʌ";                              // until, unhappy
        }
        if (spelled == U"a" && one_of(StrView(w.before(i)), {U"dis", U"mis", U"un", U"in", U"re", U"non", U"over", U"pre"}) &&
            py::startswith(w.run(i), U"a")) {
            return "ɐ";                              // disappoint, inability
        }
    }
    const std::string prefix = first_syllable(w, i);
    if (!prefix.empty()) {
        return prefix;
    }
    if (spelled == U"e" && one_of(nxt, {"K", "G"}) && py::startswith(w.run(i + 1), U"x") && w.next_vowel_stress(i) > 0) {
        return "ɛ";                                  // unexpected, inexperienced
    }
    if (one_of(StrView(spelled), {U"i", U"ai", U"ui", U"y"})) {
        return weak_i(w, i);
    }
    if (spelled == U"e") {
        return weak_e(w, i);
    }
    if (spelled == U"a") {
        if (nxt == "JH" || (nxt == "N" && w.base(i + 2) == "JH")) {
            return "ɪ";                              // village, orange
        }
        if (nxt == "S" && w.ends_with(i, {"S"}) && py::endswith(w.word, U"ace")) {
            return "ɪ";                              // palace, surface
        }
        if (name == "IH" && !(nxt == "T" && w.ends_with(i, {"T"}))) {
            return "ɪ";                              // character; but private
        }
        return "ə";
    }
    if (name == "AH" && nxt == "M" && w.ends_with(i, {"M"}) && py::endswith(w.word, U"ome")) {
        return "ʌ";                                  // handsome, welcome
    }
    return name == "IH" && !one_of(StrView(spelled), {U"o", U"u", U"ou"}) ? "ɪ" : "ə";
}

Phone unstressed_iy(const Word & w, long i) {
    const std::string_view nxt = w.base(i + 1);
    if (!w.exists(i + 1) || w.ends_with(i, {"Z"}) || w.ends_with(i, {"D"})) {
        return {"i", 0};                             // happy, stories, married
    }
    if (w.vowel(i + 1)) {
        return {nxt == "IH" && w.base(i + 2) == "NG" ? "i" : "ɪ", 0};      // carrying; video, earlier
    }
    if (w.weak_prefix(i)) {
        return w.next_vowel_stress(i) == 1 ? Phone{"ᵻ", 0} : Phone{"iː", 2};   // remind; reconsider
    }
    return {"ɪ", 0};                                 // anything, happiness
}

// t between a vowel (or r) and an unstressed vowel: water, city, party.
void flap(Phones & out) {
    for (size_t k = 1; k + 1 < out.size(); ++k) {
        if (out[k].phone != "t") {
            continue;
        }
        const std::string & before = out[k - 1].phone;
        const std::string & after = out[k + 1].phone;
        const int stress = out[k + 1].stress;
        if ((!is_vowel(before) && before != "ɹ") || before == kSyllabicN || before == "əl" || !is_vowel(after) || after == kSyllabicN) {
            continue;
        }
        // A word-final "o" is weak whatever stress digit CMUdict gives it: potato, photo.
        const bool final_o = after == "oʊ" && (k + 2 == out.size() || (k + 3 == out.size() && out[k + 2].phone == "z"));
        if (stress != 0 && !final_o) {
            continue;
        }
        if ((after == "ə" || after == "ɚ") && k + 2 < out.size() && out[k + 2].phone == "n") {
            continue;                                // skeleton, important, pattern
        }
        out[k].phone = "ɾ";
    }
}

// One primary stress per word, and no secondary stress on a syllable next to
// the primary one (weekend, notebook, important).
void settle(Phones & out) {
    std::vector<size_t> vowels;
    for (size_t k = 0; k < out.size(); ++k) {
        if (is_vowel(out[k].phone)) {
            vowels.push_back(k);
        }
    }
    std::vector<size_t> primaries;
    for (const size_t k : vowels) {
        if (out[k].stress == 1) {
            primaries.push_back(k);
        }
    }
    for (size_t p = 0; p + 1 < primaries.size(); ++p) {
        out[primaries[p]].stress = 2;
    }
    for (size_t position = 0; position < vowels.size(); ++position) {
        const size_t k = vowels[position];
        if (out[k].stress != 2) {
            continue;
        }
        const bool before = position > 0 && out[vowels[position - 1]].stress == 1;
        const bool after = position + 1 < vowels.size() && out[vowels[position + 1]].stress == 1;
        if (before || after) {
            out[k].stress = 0;
        }
    }
}

}  // namespace

bool is_arpabet(const std::vector<std::string> & symbols) {
    if (symbols.empty()) {
        return false;
    }
    for (const auto & symbol : symbols) {
        if (arpabet().count(symbol) == 0) {
            return false;
        }
    }
    return true;
}

std::string to_text(const Phones & phones) {
    std::string out;
    for (size_t k = 0; k < phones.size(); ++k) {
        if (k != 0) {
            out.push_back(' ');
        }
        if (phones[k].stress == 1) {
            out += "ˈ";
        } else if (phones[k].stress == 2) {
            out += "ˌ";
        }
        out += phones[k].phone;
    }
    return out;
}

Phones from_text(std::string_view text) {
    Phones phones;
    for (const Str & unit : py::split(decode(text))) {
        const int stress = unit[0] == 0x2C8 ? 1 : unit[0] == 0x2CC ? 2 : 0;
        const std::string phone = encode(stress != 0 ? StrView(unit).substr(1) : StrView(unit));
        if (!is_phone(phone)) {
            throw ValueError("'" + phone + "' is not a phone of the inventory");
        }
        if (stress != 0 && !is_vowel(phone)) {
            throw ValueError("stress mark on the consonant '" + phone + "'");
        }
        phones.push_back(Phone{phone, stress});
    }
    return phones;
}

Phones convert(StrView word, const std::vector<std::string> & arpabet_in, const std::optional<std::vector<Str>> & runs) {
    const Word w(word, arpabet_in, runs);
    Phones out;
    auto emit = [&](const std::string & phone, int stress = 0) { out.push_back(Phone{phone, stress}); };
    long i = 0;
    while (i < w.count) {
        const std::string & name = w.bases[static_cast<size_t>(i)];
        const int stress = w.stressed(i) ? w.stress[static_cast<size_t>(i)] : 0;
        const std::string_view nxt = w.base(i + 1);
        if (arpa_vowels().count(name) == 0) {
            if (name == "Y" && nxt == "AH" && w.unstressed(i + 1) && one_of(w.base(i - 1), {"N", "L"}) && w.base(i + 2) != "L" &&
                (!w.aligned || py::contains(w.run(i) + w.run(i + 1), U'i'))) {
                emit("iə");                          // opinion, million, brilliant
                i += 2;
                continue;
            }
            emit(consonants().at(name));
            i += 1;
            continue;
        }
        const bool before_vowel = w.vowel(i + 2);    // only meaningful when the next segment is R
        if (name == "ER") {
            emit(stress != 0 ? "ɜː" : "ɚ", stress);
            if (nxt == "R") {
                i += 1;
            }
            if (w.vowel(i + 1)) {
                emit("ɹ");                           // around, history
            }
        } else if (nxt == "R" && name == "AW") {
            emit("aʊ", stress);                      // hour, flour
            emit("ɚ");
            if (before_vowel) {
                emit("ɹ");
            }
            i += 1;
        } else if (nxt == "R" && one_of(name, {"AA", "AO", "IH", "IY", "EH", "UH", "UW", "AY"})) {
            std::string quality;
            std::string single;
            if (name == "AA") {
                quality = "ɑː";
                single = "ɑːɹ";
            } else if (name == "AO") {
                quality = north_or_force(w, i, before_vowel);
                single = quality + "ɹ";
            } else if (name == "IH" || name == "IY") {
                quality = stress != 0 && one_of(StrView(w.letters(i)), {U"e", U"eo"}) ? "iə" : "ɪ";      // zero, period; hearing, spirit
                single = "ɪɹ";
            } else if (name == "EH") {
                const bool doubled = py::startswith(w.run(i + 1) + w.after(i + 1), U"rr") ||
                    (py::endswith(w.run(i), U"r") && py::startswith(w.run(i + 1), U"r"));
                quality = w.letters(i) == U"a" && doubled ? "æ" : "ɛ";      // marry; very
                single = "ɛɹ";
            } else if (name == "UH" || name == "UW") {
                quality = single = "ʊɹ";             // "during" keeps the r-coloured vowel and adds ɹ
            } else {
                static const std::initializer_list<StrView> ire_endings = {U"ing", U"ings", U"ingly", U"able", U"y", U"er", U"ers", U"est"};
                quality = !w.aligned || one_of(StrView(w.after(i + 1)), ire_endings) ? "aɪɚ" : "aɪ";    // hiring; pirate
                single = "aɪɚ";
            }
            if (before_vowel) {
                emit(quality, stress);
                emit("ɹ");
            } else {
                emit(single, stress);
            }
            i += 1;
        } else if (name == "AY" && nxt == "ER" && w.unstressed(i + 1) && w.run(i + 1) == U"er" && w.run(i) != U"i") {
            emit("aɪ", stress);                      // higher, buyer, flyer: a word plus "-er"
        } else if (name == "AY" && nxt == "ER" && w.unstressed(i + 1)) {
            emit("aɪɚ", stress);                     // fire, tired, iron
            i += 1;
            if (w.vowel(i + 1)) {
                emit("ɹ");
            }
        } else if (name == "AY" && nxt == "AH" && w.unstressed(i + 1)) {
            emit("aɪə", stress);                     // quiet, society
            i += 1;
        } else if (name == "IY" && nxt == "AH" && w.unstressed(i + 1) && !w.weak_prefix(i)) {
            const std::string_view after = w.base(i + 2);
            if (stress != 1 && (one_of(after, {"L", "D", "T"}) || (after == "N" && w.base(i + 3) == "S"))) {
                emit("ɪ");                           // material, period, immediate, experience
            } else {
                emit("iə", stress);                  // idea, various, museum
                i += 1;
            }
        } else if (stress == 0 && name == "AH" && nxt == "N" && w.base(i - 1) == "T" && w.base(i + 2) != "T" && out.size() >= 2 &&
                   out[out.size() - 1].phone == "t" && is_vowel(out[out.size() - 2].phone) && out[out.size() - 2].stress != 0) {
            out[out.size() - 1].phone = "ʔ";         // button, certain, frightened
            emit(kSyllabicN);
            i += 1;
        } else if (stress == 0 && name == "AH" && nxt == "W" && w.base(i + 2) == "AH" && w.unstressed(i + 2) &&
                   py::contains(w.run(i) + w.run(i + 1), U'u')) {
            emit("uː");                              // usual, individual: the "u" of "-ual"
            i += 1;
        } else if (stress == 0 && name == "AH" && nxt == "L" && i > 0 && !(w.vowel(i + 2) && w.stressed(i + 2))) {
            if (w.letters(i) == U"i" && w.vowel(i + 2)) {
                emit("ɪ");                           // family, easily
            } else if (w.base(i - 1) == "Y" && py::contains(w.run(i - 1) + w.run(i), U'u')) {
                emit("ʊ");                           // popular, regular
            } else {
                emit("əl");                          // little, finally
                i += 1;
            }
        } else if (stress == 0 && (name == "AH" || name == "IH")) {
            emit(reduced(w, i));
        } else if (name == "IY" && (stress == 0 || (stress == 2 && !w.exists(i + 1)))) {
            const Phone phone = unstressed_iy(w, i);
            emit(phone.phone, phone.stress);
        } else if (stress == 0 && name == "AE" && i == 0 && py::startswith_any(w.word, {U"ad", U"acc"}) && one_of(nxt, {"D", "K"})) {
            emit("ɐ");                               // advice, accept
        } else if (stress == 0 && name == "ER") {
            emit("ɚ");
        } else if ((name == "AA" || name == "AO") && w.letters(i) == U"a" && py::endswith_any(w.before(i), {U"w", U"qu"}) && nxt == "N") {
            emit("ɔ", stress);                       // want, quantity
        } else if (name == "AO" && w.letters(i) == U"o") {
            emit("ɔ", stress);                       // long, off, lost
        } else {
            emit(full_vowels().at(name), stress);
        }
        i += 1;
    }
    flap(out);
    for (size_t k = 0; k + 1 < out.size(); ++k) {
        if (out[k].phone == "n" && (out[k + 1].phone == "k" || out[k + 1].phone == "ɡ")) {
            out[k].phone = "ŋ";                      // income, engaged
        }
    }
    settle(out);
    return out;
}

Phones settle_stress(const Phones & phones) {
    Phones out = phones;
    settle(out);
    return out;
}

}  // namespace dizione
