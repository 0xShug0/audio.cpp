#include "morphology.h"

#include "pyre.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace dizione::morphology {

namespace {

bool one_of(const std::string & value, std::initializer_list<const char *> options) {
    for (const char * option : options) {
        if (value == option) {
            return true;
        }
    }
    return false;
}

bool sibilant(const std::string & phone) { return one_of(phone, {"s", "z", "ʃ", "ʒ", "tʃ", "dʒ"}); }
bool voiceless(const std::string & phone) { return one_of(phone, {"p", "t", "k", "f", "θ", "ʔ"}); }
bool rhotic(const std::string & phone) { return one_of(phone, {"ɚ", "ɜː", "aɪɚ", "ʊɹ"}); }

// An r-coloured vowel without its r, empty when the phone is not one.
const char * split_vowel(const std::string & phone) {
    if (phone == "ɑːɹ") return "ɑː";
    if (phone == "ɔːɹ") return "ɔː";
    if (phone == "oːɹ") return "oː";
    if (phone == "ɪɹ") return "ɪ";
    if (phone == "ɛɹ") return "ɛ";
    return "";
}

bool consonant_letter(char32_t c) {
    return c < 128 && std::string_view("bcdfghjklmnpqrstvwxz").find(static_cast<char>(c)) != std::string_view::npos;
}

// Phones without stress.
Phones P(std::initializer_list<const char *> phones) {
    Phones out;
    for (const char * phone : phones) {
        out.push_back(Phone{phone, 0});
    }
    return out;
}

struct Ending {
    Str spelling;
    Phones phones;
};

// Endings that leave the stem's stress alone.
const std::vector<Ending> & endings() {
    static const std::vector<Ending> value = {
        {U"fully", P({"f", "əl", "i"})}, {U"lessly", P({"l", "ə", "s", "l", "i"})}, {U"ness", P({"n", "ə", "s"})},
        {U"less", P({"l", "ə", "s"})}, {U"ments", P({"m", "ə", "n", "t", "s"})}, {U"ment", P({"m", "ə", "n", "t"})},
        {U"ful", P({"f", "əl"})}, {U"ship", {{"ʃ", 0}, {"ɪ", 2}, {"p", 0}}}, {U"hood", {{"h", 0}, {"ʊ", 2}, {"d", 0}}},
        {U"like", {{"l", 0}, {"aɪ", 2}, {"k", 0}}}, {U"dom", P({"d", "ə", "m"})}, {U"ish", P({"ɪ", "ʃ"})},
        {U"able", P({"ə", "b", "əl"})}, {U"ism", P({"ɪ", "z", "ə", "m"})}, {U"ist", P({"ɪ", "s", "t"})},
        {U"wards", P({"w", "ɚ", "d", "z"})}, {U"ward", P({"w", "ɚ", "d"})}, {U"age", P({"ɪ", "dʒ"})},
        {U"ers", P({"ɚ", "z"})}, {U"er", P({"ɚ"})}, {U"est", P({"ɪ", "s", "t"})}, {U"y", P({"i"})},
    };
    return value;
}

struct Prefix {
    Str spelling;
    Phones plain;           // before a stem stressed on its first syllable
    Phones stressed;        // otherwise
};

const std::vector<Prefix> & prefixes() {
    static const std::vector<Prefix> value = [] {
        const Phones under = {{"ʌ", 2}, {"n", 0}, {"d", 0}, {"ɚ", 0}};
        const Phones over = {{"oʊ", 2}, {"v", 0}, {"ɚ", 0}};
        const Phones anti = {{"æ", 2}, {"n", 0}, {"t", 0}, {"i", 0}};
        const Phones semi = {{"s", 0}, {"ɛ", 2}, {"m", 0}, {"i", 0}};
        const Phones non = {{"n", 0}, {"ɑː", 2}, {"n", 0}};
        const Phones out_ = {{"aʊ", 2}, {"t", 0}};
        const Phones pre = {{"p", 0}, {"ɹ", 0}, {"iː", 2}};
        return std::vector<Prefix>{
            {U"under", under, under}, {U"over", over, over}, {U"anti", anti, anti}, {U"semi", semi, semi},
            {U"non", non, non}, {U"out", out_, out_},
            {U"dis", P({"d", "ɪ", "s"}), {{"d", 0}, {"ɪ", 2}, {"s", 0}}},
            {U"mis", P({"m", "ɪ", "s"}), {{"m", 0}, {"ɪ", 2}, {"s", 0}}},
            {U"un", P({"ʌ", "n"}), {{"ʌ", 2}, {"n", 0}}}, {U"in", P({"ɪ", "n"}), {{"ɪ", 2}, {"n", 0}}},
            {U"re", P({"ɹ", "ᵻ"}), {{"ɹ", 0}, {"iː", 2}}}, {U"pre", pre, pre},
        };
    }();
    return value;
}

const std::vector<std::pair<Str, Str>> & british() {
    static const std::vector<std::pair<Str, Str>> value = {
        {U"isation", U"ization"}, {U"ising", U"izing"}, {U"ised", U"ized"}, {U"ises", U"izes"}, {U"ise", U"ize"},
        {U"yse", U"yze"}, {U"our", U"or"}, {U"ourable", U"orable"}, {U"ours", U"ors"}, {U"oured", U"ored"},
        {U"ouring", U"oring"}, {U"tre", U"ter"}, {U"tres", U"ters"}, {U"bre", U"ber"}, {U"ence", U"ense"},
        {U"ogue", U"og"}, {U"mme", U"m"}, {U"lled", U"led"}, {U"lling", U"ling"}, {U"ller", U"ler"},
        {U"oul", U"ol"}, {U"ae", U"e"}, {U"enqui", U"inqui"},
    };
    return value;
}

// Spellings the stem may have had before an ending starting with a vowel.
std::vector<Str> stems(const Str & body) {
    static const pyre::Regex short_stem(R"re((?:^|[^aeiou])[aeiou](?:[bcdfgklmnprstvz]|th)$)re");
    std::vector<Str> out = {body};
    const size_t size = body.size();
    if (size > 2 && body[size - 1] == body[size - 2] && consonant_letter(body[size - 1])) {
        out.push_back(body.substr(0, size - 1));                             // stopp-ed
    }
    if (size > 0 && (consonant_letter(body[size - 1]) || body[size - 1] == U'u')) {
        // A short vowel before a single consonant would have doubled it ("hopped"),
        // so "hoped" and "clothed" come from "hope" and "clothe".
        const bool single = static_cast<bool>(short_stem.search(body));
        out.insert(single ? out.begin() : out.end(), body + U"e");            // lov-ed, spong-ing
    }
    if (py::endswith(body, U"i") && size > 2) {
        out.push_back(body.substr(0, size - 1) + U"y");                      // stormi-er
    }
    return out;
}

// Demote the primary stress of the second part of a compound.
Phones secondary(const Phones & phones) {
    Phones out = phones;
    for (Phone & phone : out) {
        if (phone.stress == 1) {
            phone.stress = 2;
        }
    }
    return out;
}

bool truthy(const std::optional<Phones> & phones) {
    return phones.has_value() && !phones->empty();
}

}  // namespace

Phones join(const Phones & stem_in, const Phones & ending) {
    Phones stem = stem_in;
    if (stem.empty() || ending.empty()) {
        stem.insert(stem.end(), ending.begin(), ending.end());
        return stem;
    }
    const std::string last = stem.back().phone;
    const int stress = stem.back().stress;
    const std::string & first = ending[0].phone;
    const int first_stress = ending[0].stress;
    if (is_vowel(first)) {
        if (*split_vowel(last) != '\0') {
            stem.back() = Phone{split_vowel(last), stress};                  // star + ing
            stem.push_back(Phone{"ɹ", 0});
        } else if (rhotic(last)) {
            stem.push_back(Phone{"ɹ", 0});                                   // wander + ing, sure + er
        } else if (last == "t" && stem.size() > 1 && (is_vowel(stem[stem.size() - 2].phone) || stem[stem.size() - 2].phone == "ɹ") &&
                   first_stress == 0 && !(first == "ə" && ending.size() > 1 && ending[1].phone == "n")) {
            stem.back() = Phone{"ɾ", 0};                                     // wait + ed
        } else if (last == "i") {
            stem.back() = first != "ɪ" ? Phone{"ɪ", stress} : Phone{"i", stress};    // storm·y + er; carry + ing
        }
    } else if (last == "i" && one_of(first, {"n", "l", "f", "m"})) {
        stem.back() = Phone{"ɪ", stress};                                    // happi + ness, happi + ly
    } else if (last == "n" && one_of(first, {"k", "ɡ"})) {
        stem.back() = Phone{"ŋ", 0};
    }
    stem.insert(stem.end(), ending.begin(), ending.end());
    return stem;
}

Phones plural(const Phones & stem) {
    if (stem.empty()) {
        throw IndexError("tuple index out of range");
    }
    const std::string & last = stem.back().phone;
    return join(stem, sibilant(last) ? P({"ᵻ", "z"}) : voiceless(last) ? P({"s"}) : P({"z"}));
}

Phones past(const Phones & stem) {
    if (stem.empty()) {
        throw IndexError("tuple index out of range");
    }
    const std::string & last = stem.back().phone;
    if (one_of(last, {"t", "d", "ɾ"})) {
        return join(stem, P({"ᵻ", "d"}));
    }
    return join(stem, voiceless(last) || one_of(last, {"s", "ʃ", "tʃ"}) ? P({"t"}) : P({"d"}));
}

Phones adverb(const Phones & stem) {
    if (stem.empty()) {
        throw IndexError("tuple index out of range");
    }
    const std::string & last = stem.back().phone;
    return join(stem, one_of(last, {"əl", "l"}) ? P({"i"}) : P({"l", "i"}));
}

Phones compound(const Phones & first, const Phones & second) {
    return join(first, secondary(second));
}

std::optional<Phones> derive(const Str & word, const Lookup & lookup, const Lookup & part, int depth) {
    using Build = std::function<std::optional<Phones>(const Phones &)>;
    const long size = static_cast<long>(word.size());

    auto stem = [&](const Str & spelling) -> std::optional<Phones> {
        if (spelling.size() < 2) {
            return std::nullopt;
        }
        std::optional<Phones> found = lookup(spelling);
        if (truthy(found)) {
            return found;
        }
        return depth < 2 ? derive(spelling, lookup, part, depth + 1) : std::nullopt;
    };

    // The first spelling that has a pronunciation decides; `build` may still give nothing.
    auto first = [&](const std::vector<Str> & candidates, const Build & build) -> std::optional<Phones> {
        for (const Str & spelling : candidates) {
            const std::optional<Phones> found = stem(spelling);
            if (truthy(found)) {
                return build(*found);
            }
        }
        return std::nullopt;
    };

    const Build as_plural = [](const Phones & found) { return std::optional<Phones>(plural(found)); };
    const Build as_past = [](const Phones & found) { return std::optional<Phones>(past(found)); };
    const Build as_adverb = [](const Phones & found) { return std::optional<Phones>(adverb(found)); };

    if (py::endswith(word, U"'s")) {
        return first({py::slice(word, 0, -2)}, as_plural);
    }
    if (py::endswith(word, U"s'")) {
        return first({py::slice(word, 0, -1)}, [](const Phones & found) { return std::optional<Phones>(found); });
    }
    std::optional<Phones> found;
    if (py::endswith(word, U"ies") && size > 4) {
        found = first({py::slice(word, 0, -3) + U"y"}, as_plural);
        if (truthy(found)) return found;
    }
    if (py::endswith(word, U"ves") && size > 4) {
        found = first({py::slice(word, 0, -3) + U"f", py::slice(word, 0, -3) + U"fe"}, [](const Phones & f) -> std::optional<Phones> {
            if (f.back().phone != "f") {
                return std::nullopt;
            }
            Phones voiced(f.begin(), f.end() - 1);
            voiced.push_back(Phone{"v", 0});
            return join(voiced, P({"z"}));
        });
        if (truthy(found)) return found;
    }
    if (py::endswith(word, U"es") && size > 3) {
        const char32_t before = py::at(word, -3);
        const bool hiss = before == U's' || before == U'x' || before == U'z' || before == U'h';
        found = first(hiss ? std::vector<Str>{py::slice(word, 0, -2)} : std::vector<Str>{}, as_plural);
        if (truthy(found)) return found;
    }
    if (py::endswith(word, U"s") && !py::endswith_any(word, {U"ss", U"us", U"is"}) && size > 3) {
        found = first({py::slice(word, 0, -1)}, as_plural);
        if (truthy(found)) return found;
    }
    if (py::endswith(word, U"ied") && size > 4) {
        found = first({py::slice(word, 0, -3) + U"y"}, as_past);
        if (truthy(found)) return found;
    }
    if (py::endswith(word, U"ed") && size > 3) {
        found = first(stems(py::slice(word, 0, -2)), as_past);
        if (truthy(found)) return found;
    }
    if (py::endswith(word, U"ing") && size > 4) {
        found = first(stems(py::slice(word, 0, -3)), [](const Phones & f) { return std::optional<Phones>(join(f, P({"ɪ", "ŋ"}))); });
        if (truthy(found)) return found;
    }
    if (py::endswith(word, U"ily") && size > 4) {
        found = first({py::slice(word, 0, -3) + U"y"}, as_adverb);
        if (truthy(found)) return found;
    }
    if (py::endswith(word, U"ly") && size > 3) {
        const Str body = py::slice(word, 0, -2);
        found = first(py::endswith(word, U"bly") || py::endswith(word, U"ply")
                          ? std::vector<Str>{body, body + U"le", py::slice(word, 0, -1) + U"e"}
                          : std::vector<Str>{body},
                      as_adverb);
        if (truthy(found)) return found;
    }
    for (const Ending & ending : endings()) {
        const long length = static_cast<long>(ending.spelling.size());
        if (!py::endswith(word, ending.spelling) || size - length <= 2) {
            continue;
        }
        const Str body = py::slice(word, 0, -length);
        if (ending.spelling == U"y" && !consonant_letter(body.back())) {
            continue;                                                        // "copay" is not "copa" + y
        }
        const bool vowel_first = is_vowel(ending.phones[0].phone);
        const std::vector<Str> candidates = vowel_first ? stems(body)
            : py::endswith(body, U"i") ? std::vector<Str>{py::slice(body, 0, -1) + U"y", body}
            : std::vector<Str>{body};
        found = first(candidates, [&](const Phones & f) { return std::optional<Phones>(join(f, ending.phones)); });
        if (truthy(found)) return found;
    }
    for (const Prefix & prefix : prefixes()) {
        const long length = static_cast<long>(prefix.spelling.size());
        if (!py::startswith(word, prefix.spelling) || size - length <= 2) {
            continue;
        }
        found = stem(py::slice(word, length));
        if (truthy(found)) {
            int first_vowel_stress = -1;
            for (const Phone & phone : *found) {
                if (is_vowel(phone.phone)) {
                    first_vowel_stress = phone.stress;
                    break;
                }
            }
            return join(first_vowel_stress == 1 ? prefix.plain : prefix.stressed, *found);
        }
    }
    for (const auto & spelling : british()) {
        if (py::contains(word, spelling.first)) {
            found = stem(py::replace(word, spelling.first, spelling.second, 1));
            if (truthy(found)) return found;
        }
    }
    if (depth == 0) {
        // Two dictionary words of four letters or more: "sandpiper", "houseplant".
        // Shorter halves turn up by chance too often ("couplet" is not "coup" + "let").
        long best = -1;
        Phones best_left, best_right;
        for (long cut = 4; cut < size - 3; ++cut) {
            const std::optional<Phones> left = part(py::slice(word, 0, cut));
            const std::optional<Phones> right = part(py::slice(word, cut));
            if (truthy(left) && truthy(right)) {
                const long score = std::min(cut, size - cut);
                if (best < 0 || score > best) {
                    best = score;
                    best_left = *left;
                    best_right = *right;
                }
            }
        }
        if (best >= 0) {
            return compound(best_left, best_right);
        }
    }
    return std::nullopt;
}

}  // namespace dizione::morphology
