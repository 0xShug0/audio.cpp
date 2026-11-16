// Numbers as words, and the small questions the figure rules ask of a text.
#include "normalize.h"

#include <algorithm>

namespace dizione::normalize {

namespace {

const char32_t * const kOnes[] = {
    U"zero", U"one", U"two", U"three", U"four", U"five", U"six", U"seven", U"eight", U"nine", U"ten", U"eleven", U"twelve",
    U"thirteen", U"fourteen", U"fifteen", U"sixteen", U"seventeen", U"eighteen", U"nineteen",
};
const char32_t * const kTens[] = {U"twenty", U"thirty", U"forty", U"fifty", U"sixty", U"seventy", U"eighty", U"ninety"};
const char32_t * const kScales[] = {U"thousand", U"million", U"billion", U"trillion", U"quadrillion", U"quintillion"};
constexpr int kScaleCount = 6;

Str ordinal_of(const Str & cardinal_words) {
    std::vector<Str> words = py::split(cardinal_words);
    Str & last = words.back();
    static const std::pair<const char32_t *, const char32_t *> irregular[] = {
        {U"one", U"first"}, {U"two", U"second"}, {U"three", U"third"}, {U"five", U"fifth"}, {U"eight", U"eighth"},
        {U"nine", U"ninth"}, {U"twelve", U"twelfth"},
    };
    bool done = false;
    for (const auto & entry : irregular) {
        if (last == entry.first) {
            last = entry.second;
            done = true;
            break;
        }
    }
    if (!done) {
        last = py::endswith(last, U"y") ? py::slice(last, 0, -1) + U"ieth" : last + U"th";
    }
    return py::join(U" ", words);
}

}  // namespace

long long Number::value() const {
    long long out = 0;
    for (const char c : digits) {
        out = out * 10 + (c - '0');
    }
    return out;
}

Number number_of(StrView text) {
    const Str trimmed = py::strip(text);
    Number out;
    bool previous_digit = false;
    size_t i = 0;
    if (i < trimmed.size() && trimmed[i] == U'+') {
        ++i;
    }
    for (; i < trimmed.size(); ++i) {
        const char32_t c = trimmed[i];
        if (c == U'_' && previous_digit) {
            previous_digit = false;
            continue;
        }
        const int digit = uni::decimal_value(c);
        if (digit < 0) {
            throw ValueError("invalid literal for int() with base 10: '" + encode(text) + "'");
        }
        if (!out.digits.empty() || digit != 0) {
            out.digits.push_back(static_cast<char>('0' + digit));
        }
        previous_digit = true;
    }
    if (!previous_digit) {
        throw ValueError("invalid literal for int() with base 10: '" + encode(text) + "'");
    }
    if (out.digits.empty()) {
        out.digits = "0";
    }
    return out;
}

Str cardinal(long long number) {
    if (number < 0) {
        return U"minus " + cardinal(-number);
    }
    if (number < 20) {
        return kOnes[number];
    }
    if (number < 100) {
        const long long tens = number / 10;
        const long long ones = number % 10;
        return Str(kTens[tens - 2]) + (ones != 0 ? U" " + Str(kOnes[ones]) : Str());
    }
    if (number < 1000) {
        const long long hundreds = number / 100;
        const long long rest = number % 100;
        return Str(kOnes[hundreds]) + U" hundred" + (rest != 0 ? U" " + cardinal(rest) : Str());
    }
    std::vector<Str> parts;
    long long scale = 1000000000000000000LL;            // 1000 ** 6
    for (int power = kScaleCount; power >= 1; --power) {
        const long long count = number / scale;
        number %= scale;
        if (count != 0) {
            parts.push_back(cardinal(count) + U" " + kScales[power - 1]);
        }
        scale /= 1000;
    }
    if (number != 0) {
        parts.push_back(cardinal(number));
    }
    return py::join(U" ", parts);
}

Str cardinal(const Number & number) {
    if (number.fits()) {
        return cardinal(number.value());
    }
    if (number.digits.size() > 21) {                    // 1000 ** 7 and more: digit by digit
        return digits(S(number.digits));
    }
    const std::string padded = std::string(21 - number.digits.size(), '0') + number.digits;
    std::vector<Str> parts;
    for (int group = 0; group < 7; ++group) {
        const long long count = std::stoll(padded.substr(static_cast<size_t>(3 * group), 3));
        if (count == 0) {
            continue;
        }
        parts.push_back(group < 6 ? cardinal(count) + U" " + kScales[5 - group] : cardinal(count));
    }
    return py::join(U" ", parts);
}

Str ordinal(long long number) {
    return ordinal_of(cardinal(number));
}

Str ordinal(const Number & number) {
    return ordinal_of(cardinal(number));
}

Str digits(StrView text, StrView zero) {
    std::vector<Str> out;
    for (const char32_t c : text) {
        if (!uni::is_digit(c)) {
            continue;
        }
        if (c == U'0') {
            out.emplace_back(zero);
            continue;
        }
        const int value = uni::decimal_value(c);
        if (value < 0) {
            throw ValueError("invalid literal for int() with base 10: '" + encode(StrView(&c, 1)) + "'");
        }
        out.emplace_back(kOnes[value]);
    }
    return py::join(U" ", out);
}

Str year(long long number) {
    if (number < 1000 || number > 2999) {
        return cardinal(number);
    }
    const long long high = number / 100;
    const long long low = number % 100;
    if (low == 0) {
        return high % 10 == 0 ? cardinal(number) : cardinal(high) + U" hundred";
    }
    if (high == 20 && low < 10) {
        return U"two thousand " + Str(kOnes[low]);
    }
    return cardinal(high) + (low < 10 ? U" oh " + Str(kOnes[low]) : U" " + cardinal(low));
}

Str pairs(long long number) {
    const long long high = number / 100;
    const long long low = number % 100;
    if (number < 100 || number > 9999 || (low == 0 && (high < 10 || high % 10 == 0))) {
        return cardinal(number);
    }
    if (low == 0) {
        return cardinal(high) + U" hundred";
    }
    return cardinal(high) + (low < 10 ? U" oh " + Str(kOnes[low]) : U" " + cardinal(low));
}

Str number_words(StrView text) {
    const py::Parts parts = py::partition(py::replace(text, U",", U""), U".");
    const Str & whole = parts.head;
    const Str & fraction = parts.tail;
    if (!fraction.empty()) {
        return (!whole.empty() ? cardinal(number_of(whole)) : Str()) + U" point " + digits(fraction);
    }
    if ((whole.size() > 1 && py::startswith(whole, U"0")) || whole.size() > 18) {
        return digits(whole);
    }
    return cardinal(number_of(whole));
}

Str duration(long long hours, long long minutes, long long seconds) {
    std::vector<Str> parts;
    const std::pair<long long, const char32_t *> counts[] = {{hours, U"hour"}, {minutes, U"minute"}, {seconds, U"second"}};
    for (const auto & entry : counts) {
        if (entry.first != 0) {
            parts.push_back(cardinal(entry.first) + U" " + entry.second + (entry.first == 1 ? U"" : U"s"));
        }
    }
    if (parts.size() < 2) {
        return parts.empty() ? Str(U"zero seconds") : parts[0];
    }
    const Str last = parts.back();
    parts.pop_back();
    return py::join(U", ", parts) + U" and " + last;
}

Str plural_of(StrView words) {
    const py::Parts parts = py::rpartition(words, U" ");
    const Str last = py::endswith(parts.tail, U"y") ? py::slice(parts.tail, 0, -1) + U"ies" : parts.tail + U"s";
    return (!parts.head.empty() ? parts.head + U" " : Str()) + last;
}

Str fraction_words(long long top, long long bottom, bool article) {
    Str one, many;
    if (bottom == 2) {
        one = U"half";
        many = U"halves";
    } else if (bottom == 4) {
        one = U"quarter";
        many = U"quarters";
    } else {
        one = ordinal(bottom);
        many = one + U"s";
    }
    if (top != 1) {
        return cardinal(top) + U" " + many;
    }
    const char32_t first = py::at(one, 0);
    const bool vowel = first == U'a' || first == U'e' || first == U'i' || first == U'o' || first == U'u';
    return (article ? (vowel ? Str(U"an ") : Str(U"a ")) : Str(U"one ")) + one;
}

// ---- shared helpers -------------------------------------------------------------

Token Token::word(Str text, Str raw, bool spell, bool attach) {
    Token token;
    token.text = std::move(text);
    token.raw = std::move(raw);
    token.spell = spell;
    token.attach = attach;
    return token;
}

Token Token::punctuation(Str mark) {
    Token token;
    token.punct = true;
    token.mark = std::move(mark);
    return token;
}

Str spelled(StrView letters) {
    Str out(1, kSpell);
    out.append(letters);
    out.push_back(kEnd);
    return out;
}

std::unordered_set<Str> word_set(StrView words) {
    std::unordered_set<Str> out;
    for (Str & word : py::split(words)) {
        out.insert(std::move(word));
    }
    return out;
}

// A number in a code that is said as a number: one or two digits, or a round one (B-12, KX-300).
bool short_number(StrView text) {
    if (py::startswith(text, U"0")) {
        return false;
    }
    if (text.size() <= 2) {
        return true;
    }
    for (size_t i = 1; i < text.size(); ++i) {
        if (text[i] != U'0') {
            return false;
        }
    }
    return true;
}

// The words just before `position` in the same clause, in lower case.
std::vector<Str> words_before(StrView text, size_t position, size_t count) {
    const size_t start = position > 80 ? position - 80 : 0;
    const std::vector<Str> clauses = RX(R"re([.!?;…]\s|[()\"])re").split(text.substr(start, position - start));
    std::vector<Str> words;
    for (const Str & word : RX(R"re([A-Za-z]+)re").findall(clauses.back())) {
        words.push_back(py::lower(word));
    }
    if (words.size() > count) {
        words.erase(words.begin(), words.end() - static_cast<std::ptrdiff_t>(count));
    }
    return words;
}

std::string identifier(StrView text, size_t position) {
    // Words that say what the number after them is. A telephone number, a PIN or a
    // reference is read digit by digit; an amount is not.
    static const std::unordered_set<Str> phone_words = word_set(U"call dial ring phone telephone tel mobile cell fax hotline text sms whatsapp");
    static const std::unordered_set<Str> phone_links = word_set(U"to on at or us me him her them back");
    static const std::unordered_set<Str> pin_words = word_set(U"code pin extension ext passcode password zip postcode cvv cvc otp");
    static const std::unordered_set<Str> strong_words = word_set(U"number id reference ref serial account tracking booking confirmation");
    static const std::unordered_set<Str> weak_words = word_set(
        U"order ticket invoice case claim policy receipt transaction parcel shipment container payment "
        U"reservation membership customer batch lot item job issue bug request");
    static const std::unordered_set<Str> link_words = word_set(U"is was are were ends ending in with starts starting begins beginning");
    const std::vector<Str> words = words_before(text, position);
    if (words.empty()) {
        return std::string();
    }
    const Str & last = words.back();
    if (phone_words.count(last) != 0) {
        return "phone";
    }
    if (phone_links.count(last) != 0) {
        // words[-3:-1]
        const long size = static_cast<long>(words.size());
        for (long k = std::max(0L, size - 3); k < size - 1; ++k) {
            if (phone_words.count(words[static_cast<size_t>(k)]) != 0) {
                return "phone";
            }
        }
    }
    size_t index = words.size() - 1;
    while (index > 0 && link_words.count(words[index]) != 0) {
        --index;
    }
    if (phone_words.count(words[index]) != 0) return "phone";
    if (pin_words.count(words[index]) != 0) return "pin";
    if (strong_words.count(words[index]) != 0) return "strong";
    if (weak_words.count(words[index]) != 0) return "weak";
    return last == U"enter" ? "enter" : std::string();
}

const std::unordered_set<Str> & sentence_starters() {
    static const std::unordered_set<Str> value = word_set(
        U"the he she it i we they this that there but and in on at a an if when what why how who my our "
        U"you then so now after before however");
    return value;
}

const std::unordered_set<Str> & capital_words() {
    static const std::unordered_set<Str> value = word_set(
        U"IN ON AT TO OF BY OR IS AS IF NO SO WE HE MY ME DO GO UP AN AM PM BE IT US THE AND FOR BUT NOT ALL");
    return value;
}

}  // namespace dizione::normalize
