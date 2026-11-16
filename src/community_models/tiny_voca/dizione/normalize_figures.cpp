// The parts of a text that are not words: money, times, dates, telephone
// numbers, versions, codes, units, fractions, web addresses and file names,
// abbreviations. Each function follows the method of the same name in
// dizione's normalize.py.
#include "normalize.h"

#include "convert.h"
#include "tables.h"

#include <algorithm>

namespace dizione::normalize {

namespace {

#define DZ_CURRENCY_CODES "USD|EUR|GBP|JPY|INR|CHF|CAD|AUD|CNY"
#define DZ_ZONES "ET|CT|MT|PT|EST|EDT|CST|CDT|MST|MDT|PST|PDT|GMT|UTC|BST|CET|CEST|IST|AEST|AEDT|JST"
#define DZ_LENGTHS "mm|cm|km|m|ft|in|inch|inches|feet|foot|metres?|meters?|yards?|yd|px|pixels?"
// Before a number these mean a version: its parts are read as whole numbers ("three point twelve").
#define DZ_VERSIONED \
    "version|ver|release|firmware|patch|kernel|build|python|node|nodejs|ios|ipados|macos|watchos|tvos|android|" \
    "chrome|firefox|safari|java|php|ruby|rust|swift|kotlin|golang|django|rails|react|angular|vue|docker|" \
    "kubernetes|ubuntu|debian|fedora|windows|linux|postgres|postgresql|mysql|redis|numpy|pytorch|tensorflow|cuda|" \
    "xcode|gcc|clang|unity|minecraft|typescript|bluetooth|usb|hdmi|opengl|vulkan|directx|" \
    "upgrade[ds]?|update[ds]?|downgrade[ds]?"
#define DZ_HALF_DAY R"re((?:\s?([AaPp])\.[Mm]\.?|\s?([AaPp])[Mm]\b))re"
#define DZ_AMOUNT R"re((?P<whole>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<fraction>\d+))?)re"
// A currency written after the amount: "20 €", "20€", "12.50 EUR".
#define DZ_SIGN_AFTER R"re([$£€¥₹](?![\w$£€¥₹])|(?:)re" DZ_CURRENCY_CODES R"re()(?![A-Za-z]))re"
#define DZ_CLOCK R"re(\d{1,2}:[0-5]\d(?:\s?[AaPp]\.?[Mm]\.?)?|\b\d{1,2}\s?[AaPp]\.?[Mm]\.?)re"

const char32_t * const kMonths[] = {
    U"January", U"February", U"March", U"April", U"May", U"June", U"July", U"August", U"September", U"October", U"November", U"December",
};

// The number of the month whose name starts with these three letters, 0 for none.
long long month_number(StrView three) {
    for (long long index = 0; index < 12; ++index) {
        if (py::slice(py::lower(kMonths[index]), 0, 3) == three) {
            return index + 1;
        }
    }
    return 0;
}

long long month_of(StrView name) {
    const Str key = py::slice(py::lower(name), 0, 3);
    const long long number = month_number(key);
    if (number == 0) {
        throw KeyError(encode(key));
    }
    return number;
}

bool full_month(StrView name) {
    for (const char32_t * month : kMonths) {
        if (name == month) {
            return true;
        }
    }
    return false;
}

bool any_in(const std::vector<Str> & words, const std::unordered_set<Str> & set) {
    for (const Str & word : words) {
        if (set.count(word) != 0) {
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

Str spaced(const Str & words) {
    return U" " + words + U" ";
}

struct Currency {
    const char32_t * symbol;
    const char32_t * one;
    const char32_t * many;
    const char32_t * cent;
    const char32_t * cents;
};

const Currency & currency(StrView symbol) {
    static const Currency table[] = {
        {U"$", U"dollar", U"dollars", U"cent", U"cents"},
        {U"\U000000A3", U"pound", U"pounds", U"penny", U"pence"},
        {U"\U000020AC", U"euro", U"euros", U"cent", U"cents"},
        {U"\U000000A5", U"yen", U"yen", U"", U""},
        {U"\U000020B9", U"rupee", U"rupees", U"paisa", U"paise"},
        {U"USD", U"dollar", U"dollars", U"cent", U"cents"},
        {U"EUR", U"euro", U"euros", U"cent", U"cents"},
        {U"GBP", U"pound", U"pounds", U"penny", U"pence"},
        {U"JPY", U"yen", U"yen", U"", U""},
        {U"INR", U"rupee", U"rupees", U"paisa", U"paise"},
        {U"CHF", U"Swiss franc", U"Swiss francs", U"centime", U"centimes"},
        {U"CAD", U"Canadian dollar", U"Canadian dollars", U"cent", U"cents"},
        {U"AUD", U"Australian dollar", U"Australian dollars", U"cent", U"cents"},
        {U"CNY", U"yuan", U"yuan", U"", U""},
    };
    for (const Currency & entry : table) {
        if (symbol == entry.symbol) {
            return entry;
        }
    }
    throw KeyError(encode(symbol));
}

// "1,5" before a unit is one number: a comma with one or two digits after
// it does not part thousands. "1,250" is left as it is.
Str decimal_comma(const Str & text) {
    return RX(R"re(\d+,\d{1,2})re").fullmatch(text) ? py::replace(text, U",", U".") : text;
}

// Digits inside a file name or an address: 3, 2026, 0412.
Str address_digits(const Str & text) {
    if (text.size() <= 2 && !(text.size() == 2 && text[0] == U'0')) {
        return cardinal(py::to_int(text));
    }
    if (text.size() == 4) {
        const long long value = py::to_int(text);
        if (value >= 1900 && value <= 2099) {
            return year(value);
        }
    }
    return digits(text);
}

// One part of a version number: "4" four, "12" twelve, "03" zero three, "129" one two nine.
Str version_part(const Str & part) {
    return part.size() > 2 ? digits(part) : number_words(part);
}

// The part after the point of a version: "12" twelve, "04" oh four.
Str minor_part(const Str & part) {
    return py::startswith(part, U"0") ? digits(part, U"oh") : cardinal(py::to_int(part));
}

// " A M" or " P M" as letters. A full stop that closes "p.m." is kept as a
// sentence end only when a new sentence follows.
Str half_day(const pyre::Match & match, const Str & dotted, const Str & plain) {
    if (dotted.empty() && plain.empty()) {
        return Str();
    }
    const Str following = py::lstrip(match.string().substr(static_cast<size_t>(match.end())));
    const pyre::Match first = RX(R"re([A-Za-z']+)re").match(following);
    const bool closes = !dotted.empty() && py::endswith(match.group(0), U".") &&
        (following.empty() || (uni::is_upper(following[0]) && first && sentence_starters().count(py::lower(first.group(0))) != 0));
    Str out(1, kSpell);
    out += py::upper(!dotted.empty() ? dotted : plain);
    out += U"M";
    out.push_back(kEnd);
    out += closes ? U" . " : U" ";
    return out;
}

// A group that mixes letters and digits: the letters are named, the digits read one by one.
Str alnum(const Str & group, StrView zero = U"zero") {
    std::vector<Str> out;
    for (const Str & run : RX(R"re([A-Za-z]+|\d+)re").findall(group)) {
        out.push_back(py::isdigit(run) ? digits(run, zero) : spelled(run));
    }
    return py::join(U" ", out);
}

// Groups of digits with a short break between them. Small numbers stay numbers: 9-1-1.
Str digit_groups(const std::vector<Str> & groups, StrView zero = U"zero") {
    bool all_short = true;
    for (const Str & group : groups) {
        all_short = all_short && short_number(group) && group.size() <= 2;
    }
    std::vector<Str> out;
    for (const Str & group : groups) {
        out.push_back(all_short ? cardinal(py::to_int(group)) : digits(group, zero));
    }
    return py::join(U" , ", out);
}

const std::unordered_set<Str> & duration_words() {
    static const std::unordered_set<Str> value = word_set(U"lasted took duration runtime recording video clip track lap pace length long");
    return value;
}

// The first run of letters after `position`, in lower case; empty when the text goes on otherwise.
Str word_after(StrView text, size_t position) {
    const pyre::Match after = RX(R"re(\s*([A-Za-z]+))re").match(text.substr(position));
    return after ? py::lower(after.group(1)) : Str();
}

}  // namespace

Normalizer::Normalizer(IsWord is_word) : is_word_(is_word ? std::move(is_word) : IsWord([](const Str &) { return true; })) {
    for (const tables::Row & row : tables::rows("abbreviations.txt")) {
        abbreviations_[row.columns.at(0)] = {row.columns.at(1), row.columns.at(2)};
    }
    // A unit marked "exact" is written in one letter case only: "m" is a meter, "M" is not.
    for (const tables::Row & row : tables::rows("units.txt")) {
        const auto & columns = row.columns;
        if (columns.size() > 3 && columns[3] == U"exact") {
            exact_units_[columns.at(0)] = {columns.at(1), columns.at(2)};
        } else {
            units_[py::lower(columns.at(0))] = {columns.at(1), columns.at(2)};
        }
    }
    // A chat spelling marked "alone" is read that way only as a reply by itself ("k", "k, thanks").
    for (const tables::Row & row : tables::rows("chat.txt")) {
        const auto & columns = row.columns;
        if (columns.size() > 2 && !(columns.size() == 3 && columns[2] == U"alone")) {
            throw ValueError("chat.txt line " + std::to_string(row.number) + ": after the words to say there can only be \"alone\"");
        }
        chat_[columns.at(0)] = columns.size() > 1 ? columns[1] : Str(U"-");
        if (columns.size() > 2) {
            replies_.insert(columns[0]);
        }
    }
    endings_ = tables::mapping("endings.txt");
    // The unit names as one pattern, the longest first.
    std::vector<Str> names;
    {
        std::unordered_set<Str> seen;
        for (const auto * table : {&units_, &exact_units_}) {
            for (const auto & entry : *table) {
                Str escaped = decode(pyre::escape(entry.first));
                if (seen.insert(escaped).second) {
                    names.push_back(std::move(escaped));
                }
            }
        }
    }
    std::sort(names.begin(), names.end(), [](const Str & a, const Str & b) {
        return a.size() != b.size() ? a.size() > b.size() : a < b;
    });
    unit_names_ = encode(py::join(U"|", names));
    unit_.emplace(R"re((?<![\w.])(\d[\d,]*(?:\.\d+)?)(?:[-–](\d[\d,]*(?:\.\d+)?))?\s?()re" + unit_names_ +
                      R"re()(?![A-Za-z])(\.(?=\s+(?-i:[a-z])))?)re",
                  pyre::IGNORECASE);
    unit_after_.emplace(R"re(\s?(?:)re" + unit_names_ + R"re()(?![A-Za-z]))re", pyre::IGNORECASE);
    unit_or_scale_after_.emplace(R"re(\s?(?:%|percent|million|billion|)re" + unit_names_ + R"re()(?![A-Za-z]))re", pyre::IGNORECASE);
}

// ---- pieces of the text that are not words -----------------------------------

Str Normalizer::amount(const Str & symbol, const Str & whole, const Str & fraction, const Str & scale_in) const {
    const Currency & names = currency(symbol);
    const Str cent = names.cent;
    if (!scale_in.empty()) {
        Str scale = py::lower(scale_in);
        if (scale == U"k") scale = U"thousand";
        else if (scale == U"m") scale = U"million";
        else if (scale == U"b" || scale == U"bn") scale = U"billion";
        else if (scale == U"t") scale = U"trillion";
        return number_words(whole + (!fraction.empty() ? U"." + fraction : Str())) + U" " + scale + U" " + names.many;
    }
    if (!fraction.empty() && (fraction.size() > 2 || cent.empty())) {
        return number_words(whole + U"." + fraction) + U" " + names.many;
    }
    const Number count = number_of(whole);
    const long long change = py::to_int(py::ljust(!fraction.empty() ? fraction : Str(U"0"), 2, U'0'));
    std::vector<Str> parts;
    if (!count.is(0) || change == 0) {
        parts.push_back(cardinal(count) + U" " + (count.is(1) ? names.one : names.many));
    }
    if (change != 0 && !cent.empty()) {
        parts.push_back(cardinal(change) + U" " + (change == 1 ? names.cent : names.cents));
    }
    return py::join(U" and ", parts);
}

Str Normalizer::money(const Match & match) const {
    const Str whole = py::replace(py::replace(match.group("whole"), U",", U""), U" ", U"");
    const Str words = amount(match.group("symbol"), whole, match.group("fraction"), match.group("scale"));
    return (!match.group("minus").empty() || !match.group("inner").empty() ? U" minus " : U" ") + words + U" ";
}

Str Normalizer::time(const Match & match) const {
    const long long hour = py::to_int(match.group(1));
    const long long minute = py::to_int(match.group(2));
    const bool has_second = match.has(3);
    const Str second = match.group(3);
    const Str half = !match.group(4).empty() ? match.group(4) : match.group(5);
    const StrView text = match.string();
    const std::vector<Str> before = words_before(text, static_cast<size_t>(match.start()), 3);
    if (has_second && half.empty() && !(!before.empty() && before.back() == U"at")) {
        return spaced(duration(hour, minute, py::to_int(second)));              // 2:03:45 is a length of time
    }
    if (hour > 24 && !has_second && half.empty()) {
        const Str after = word_after(text, static_cast<size_t>(match.end()));
        if (any_in(before, duration_words()) || (!after.empty() && duration_words().count(after) != 0)) {
            return spaced(duration(0, hour, minute));                           // 48:12 long
        }
        return U" " + cardinal(hour) + U" " + (minute < 10 ? U"oh " : U"") + cardinal(minute) + U" ";
    }
    if (hour > 24) {
        return match.group(0);
    }
    Str words = cardinal(hour);
    if (minute == 0) {
        if (!half.empty()) {
        } else if (hour == 0) {
            words = U"midnight";
        } else if (hour > 12) {
            words += U" hundred";                                               // 20:00
        } else {
            words += U" o'clock";
        }
    } else {
        words += (minute < 10 ? U" oh " : U" ") + cardinal(minute);
    }
    if (!second.empty()) {
        const long long seconds = py::to_int(second);
        words += U" and " + cardinal(seconds) + (seconds == 1 ? U" second" : U" seconds");
    }
    return spaced(words) + half_day(match, match.group(4), match.group(5));
}

// "9.30 to 11.30" where the words before say it is a time of day.
Str Normalizer::dot_times(const Match & match) const {
    static const std::unordered_set<Str> clock_words = word_set(
        U"window slot open opens opening hours shift session served service available booked appointment "
        U"delivery arrive arrives arriving");
    const StrView text = match.string();
    const StrView following = text.substr(static_cast<size_t>(match.end()));
    if (py::to_int(match.group(1)) > 23 || py::to_int(match.group(4)) > 23 ||
        !any_in(words_before(text, static_cast<size_t>(match.start())), clock_words) || unit_or_scale_after_->match(following)) {
        return match.group(0);
    }
    return match.group(1) + U":" + match.group(2) + match.group(3) + match.group(4) + U":" + match.group(5);
}

// 16:9, 3:1, 1:25,000. Two digits after the colon are a time unless the words say "ratio".
Str Normalizer::ratio(const Match & match) const {
    static const std::unordered_set<Str> ratio_words = word_set(U"ratio ratios scale odds aspect mix split proportion");
    const Str first = match.group(1);
    const Str second = match.group(2);
    if (first.size() <= 2 && second.size() == 2 && second[0] >= U'0' && second[0] <= U'5') {
        const StrView text = match.string();
        const Str after = word_after(text, static_cast<size_t>(match.end()));
        if (!(any_in(words_before(text, static_cast<size_t>(match.start()), 3), ratio_words) ||
              (!after.empty() && ratio_words.count(after) != 0))) {
            return match.group(0);
        }
    }
    return U" " + number_words(first) + U" to " + number_words(second) + U" ";
}

std::optional<Str> Normalizer::date(long long month, long long day, const Str * year_text, bool the) const {
    if (month < 1 || month > 12 || day < 1 || day > 31) {
        return std::nullopt;
    }
    Str words = the ? U"the " + ordinal(day) + U" of " + kMonths[month - 1] : Str(kMonths[month - 1]) + U" " + ordinal(day);
    if (year_text != nullptr && !year_text->empty()) {
        const long long value = py::to_int(*year_text);
        words += U" " + (year_text->size() == 4 ? year(value) : (value < 10 ? U"oh " : U"") + cardinal(value));
    }
    return spaced(words);
}

Str Normalizer::slash_date(const Match & match) const {
    long long first = py::to_int(match.group(1));
    long long second = py::to_int(match.group(2));
    if ((first > 12 && 12 >= second) || (day_first_ && second <= 12)) {
        std::swap(first, second);                                               // day first
    }
    const Str year_text = match.group(3);
    const std::optional<Str> spoken = date(first, second, &year_text);
    return spoken ? *spoken : match.group(0);
}

// 15.11.2026: dates written with full stops put the day first.
Str Normalizer::dot_date(const Match & match) const {
    long long day = py::to_int(match.group(1));
    long long month = py::to_int(match.group(2));
    if (month > 12 && 12 >= day) {
        std::swap(day, month);
    }
    const Str year_text = match.group(3);
    const std::optional<Str> spoken = date(month, day, &year_text);
    return spoken ? *spoken : match.group(0);
}

// "22 February", "5 Jan. 2026". The stop after a short month name is dropped; any other one ends the sentence.
Str Normalizer::day_month(const Match & match) const {
    const Str month = match.group(2);
    const Str year_text = match.group(4);
    const std::optional<Str> spoken = date(month_of(month), py::to_int(match.group(1)), &year_text, true);
    if (!spoken) {
        return match.group(0);
    }
    return *spoken + (!match.group(3).empty() && full_month(month) ? U". " : U"");
}

// "3-17 November" and "November 3-17".
Str Normalizer::days_of_month(const Match & match) const {
    const bool month_first = uni::is_alpha(py::at(match.group(1), 0));
    const Str month = month_first ? match.group(1) : match.group(3);
    const long long first = py::to_int(month_first ? match.group(2) : match.group(1));
    const long long second = py::to_int(month_first ? match.group(3) : match.group(2));
    if (!(first >= 1 && first <= 31 && second >= 1 && second <= 31)) {
        return match.group(0);
    }
    const Str name = kMonths[month_of(month) - 1];
    const Str words = month_first ? name + U" " + ordinal(first) + U" to " + ordinal(second)
                                  : U"the " + ordinal(first) + U" to the " + ordinal(second) + U" of " + name;
    return U" " + words + (!match.group(4).empty() ? U" " + year(py::to_int(match.group(4))) : Str()) + U" ";
}

// "Mon.-Fri.", "Jan-Mar": from one day or month to another.
Str Normalizer::day_range(const Match & match) const {
    static const std::unordered_map<Str, Str> day_names = {
        {U"mon", U"Monday"}, {U"tue", U"Tuesday"}, {U"wed", U"Wednesday"}, {U"thu", U"Thursday"},
        {U"fri", U"Friday"}, {U"sat", U"Saturday"}, {U"sun", U"Sunday"},
    };
    std::vector<Str> names;
    for (const int group : {1, 3}) {
        const Str name = match.group(group);
        const Str key = py::lower(name);
        const auto abbreviation = abbreviations_.find(key);
        if (abbreviation != abbreviations_.end()) {
            names.push_back(abbreviation->second.second);
        } else {
            const auto day = day_names.find(key);
            names.push_back(day != day_names.end() ? day->second : name);
        }
    }
    const Str following = py::lstrip(match.string().substr(static_cast<size_t>(match.end())));
    const pyre::Match first = RX(R"re([A-Za-z']+)re").match(following);
    const bool ends = !match.group(4).empty() &&
        (following.empty() || (uni::is_upper(following[0]) && first && sentence_starters().count(py::lower(first.group(0))) != 0));
    return names[0] + U" to " + names[1] + (ends ? U" ." : U"");
}

// "14/03", "12/10" as a day of the year, or nothing when it is a fraction.
std::optional<Str> Normalizer::short_date(const Match & match) const {
    static const std::unordered_set<Str> date_before = word_set(
        U"on dated due until till by since from before after for shipped delivered arriving arrives arrive "
        U"arrived expires expired ends ending starts starting begins beginning monday tuesday wednesday "
        U"thursday friday saturday sunday mon tue tues wed thu thur thurs fri sat sun");
    const Str first_text = match.group(1);
    const Str second_text = match.group(2);
    if (first_text.size() > 2 || second_text.size() > 2) {
        return std::nullopt;
    }
    long long first = py::to_int(first_text);
    long long second = py::to_int(second_text);
    const std::vector<Str> before = words_before(match.string(), static_cast<size_t>(match.start()));
    if (first >= 1 && first <= 12 && second >= 20 && second <= 49 && std::find(before.begin(), before.end(), U"card") != before.end()) {
        return U" " + Str(kMonths[first - 1]) + U" " + year(2000 + second) + U" ";     // a card that expires 08/27
    }
    const Str last = before.empty() ? Str() : before.back();
    bool classic_bottom = false;
    for (const long long bottom : {2, 3, 4, 5, 6, 8, 10, 16, 32, 64, 100}) {
        classic_bottom = classic_bottom || second == bottom;
    }
    const bool classic = first < second && classic_bottom;                              // 1/2, 3/4, 7/10
    const bool zero = (first_text.size() == 2 && first_text[0] == U'0') || (second_text.size() == 2 && second_text[0] == U'0');
    if (!(zero || (first > 12 && first <= 31 && second <= 12 && !(first == 24 && second == 7)) ||
          (date_before.count(last) != 0 && (!classic || last == U"on")) || (dated_ && !classic && first != second))) {
        return std::nullopt;
    }
    if (first > 12 || (day_first_ && second <= 12)) {
        std::swap(first, second);                                               // day first
    }
    return date(first, second);
}

Str Normalizer::fraction(const Match & match) const {
    const std::optional<Str> spoken = short_date(match);
    if (spoken && !spoken->empty()) {
        return *spoken;
    }
    const Number top = number_of(match.group(1));
    const Number bottom = number_of(match.group(2));
    if ((top.is(24) && bottom.is(7)) || top == bottom) {
        return U" " + cardinal(top) + U" " + cardinal(bottom) + U" ";           // twenty four seven, fifty fifty
    }
    const bool top_fits = top.fits() && top.value() > 0 && top.value() < 100;
    const bool bottom_fits = bottom.fits() && bottom.value() > 1 && bottom.value() <= 100;
    if (!top_fits || !bottom_fits) {
        return U" " + cardinal(top) + U" over " + cardinal(bottom) + U" ";
    }
    return spaced(fraction_words(top.value(), bottom.value()));
}

// "2 1/2": two and a half.
Str Normalizer::mixed_fraction(const Match & match) const {
    const long long top = py::to_int(match.group(2));
    const long long bottom = py::to_int(match.group(3));
    if (!(top > 0 && top < bottom)) {
        return match.group(0);
    }
    return U" " + number_words(match.group(1)) + U" and " + fraction_words(top, bottom, true) + U" ";
}

// "½" and "1½". Runs before the text is normalised, which would turn "1½" into "11/2".
Str Normalizer::vulgar(const Match & match) const {
    const std::vector<Str> parts = py::split(uni::nfkc(match.group(2)), Str(1, char32_t{0x2044}));
    if (parts.size() != 2) {
        throw ValueError("not enough values to unpack");
    }
    const long long top = py::to_int(parts[0]);
    const long long bottom = py::to_int(parts[1]);
    if (!match.has(1)) {
        return U" " + py::from_int(top) + U"/" + py::from_int(bottom) + U" ";
    }
    return U" " + cardinal(number_of(match.group(1))) + U" and " + fraction_words(top, bottom, true) + U" ";
}

Str Normalizer::phone(const Match & match) const {
    std::vector<Str> out;
    for (int group = 1; group <= match.groups(); ++group) {
        const Str digits_ = match.group(group);
        if (!digits_.empty()) {
            out.push_back(digits_ == U"800" ? Str(U"eight hundred") : digits(digits_, U"oh"));
        }
    }
    return U" " + py::join(U", ", out) + U" ";
}

Str Normalizer::measure(const Match & match) const {
    const Str unit = match.group(3);
    const Names * names = nullptr;
    const auto exact = exact_units_.find(unit);
    if (exact != exact_units_.end()) {
        names = &exact->second;
    } else {
        const auto any = units_.find(py::lower(unit));
        if (any != units_.end()) {
            names = &any->second;
        }
    }
    const StrView text = match.string();
    const auto unit_end = static_cast<size_t>(match.end(3));
    const Str after = Str(text.substr(std::min(unit_end, text.size()), 1));
    if (names == nullptr || (unit.size() == 1 && one_of(StrView(after), {U"-", U"'", U"/"}))) {
        return match.group(0);
    }
    const Str value = decimal_comma(match.group(1));
    const Str upper = !match.group(2).empty() ? decimal_comma(match.group(2)) : Str();
    if (!upper.empty()) {
        return U" " + number_words(value) + U" to " + number_words(upper) + U" " + names->second + U" ";
    }
    return U" " + number_words(value) + U" " + (value == U"1" || value == U"1.0" ? names->first : names->second) + U" ";
}

// "120 m²", "3 m3": square and cubic measures.
Str Normalizer::square(const Match & match) const {
    const Str unit = match.group(2);
    Names names;
    if (unit == U"in") {
        names = {U"inch", U"inches"};
    } else {
        const auto exact = exact_units_.find(unit);
        if (exact != exact_units_.end()) {
            names = exact->second;
        } else {
            const auto any = units_.find(unit);
            if (any == units_.end()) {
                throw KeyError(encode(unit));
            }
            names = any->second;
        }
    }
    const Str power = match.group(3);
    const Str kind = power == U"\U000000B2" || power == U"2" ? U"square" : U"cubic";
    const Str value = decimal_comma(match.group(1));
    return U" " + number_words(value) + U" " + kind + U" " + (value == U"1" || value == U"1.0" ? names.first : names.second) + U" ";
}

// True when a plural noun follows a number: it is then an amount, not a reference.
bool Normalizer::counted(StrView text, size_t end) const {
    static const std::unordered_set<Str> not_counted = word_set(
        U"is was has does goes ships arrives departs leaves shows needs blocks remains includes contains "
        U"says reads starts ends closes opens runs takes comes this his its us as yes plus minus less");
    const Str word = word_after(text, end);
    if (word.empty()) {
        return false;
    }
    if (one_of(StrView(word), {U"people", U"men", U"women", U"children", U"feet", U"miles"})) {
        return true;
    }
    return word.size() > 3 && py::endswith(word, U"s") && !py::endswith(word, U"ss") && not_counted.count(word) == 0 &&
        is_word_(py::slice(word, 0, -1));
}

Str Normalizer::number(const Match & match, StrView text) const {
    static const std::unordered_set<Str> year_context = word_set(
        U"in since by from until till before after during around circa of year early late mid summer winter "
        U"spring fall autumn class");
    const Str value = match.group(0);
    const Str plain = py::replace(value, U",", U"");
    const bool whole = !py::contains(plain, U'.') && !py::contains(value, U',');
    const auto start = static_cast<size_t>(match.start());
    const auto end = static_cast<size_t>(match.end());
    if (whole && !py::startswith(plain, U"0") && plain.size() > 2) {
        const std::string kind = identifier(text, start);
        const size_t limit = kind == "phone" || kind == "pin" ? 3 : kind == "strong" || kind == "enter" ? 4 : kind == "weak" ? 5 : 7;
        if (plain.size() >= limit && !(kind.empty() && py::endswith(plain, U"000")) &&
            !(kind == "enter" && plain.size() == 4 && py::to_int(plain) >= 1900 && py::to_int(plain) <= 2099) &&
            !((kind.empty() || kind == "phone" || kind == "weak") && counted(text, end))) {
            return spaced(digits(plain, kind == "phone" ? U"oh" : U"zero"));    // a reference, not an amount
        }
    }
    if (whole && plain.size() == 4 && !py::startswith(plain, U"0")) {
        const long long number = py::to_int(plain);
        const StrView head = text.substr(0, start);
        const std::vector<Str> before = RX(R"re([A-Za-z]+)re").findall(py::slice(head, -24));
        const Str after = word_after(text, end);
        const bool counted_ = !after.empty() && RX(R"re((?:[^s]s|people|men|women|children|feet|miles)$)re").search(after);
        bool dated = false;
        if (!before.empty()) {
            const Str last = py::lower(before.back());
            dated = year_context.count(last) != 0 || month_number(py::slice(last, 0, 3)) != 0;
        }
        const pyre::Match model = RX(R"re((?<![A-Za-z])([A-Z]{2,5}) $)re").search(head);
        if (model && capital_words().count(model.group(1)) == 0 && !counted_) {
            return spaced(pairs(number));                                       // RTX 4070, ISO 9001
        }
        if ((number >= 1100 && number <= 1999) || (number >= 2000 && number <= 2099 && (dated || !counted_))) {
            return spaced(year(number));
        }
    }
    return spaced(number_words(value));
}

// ---- codes and reference numbers ----------------------------------------------

// Letters inside a code that are a word to say, not letters to name: SAVE-20.
bool Normalizer::reads_as_word(const Str & letters) const {
    const Str lower = py::lower(letters);
    if (lower.size() < 3 || !is_word_(lower)) {
        return false;
    }
    if (lower.size() > 3) {
        return true;
    }
    for (const char32_t * ending : {U"s", U"es", U"ed", U"ing"}) {
        if (is_word_(lower + ending)) {
            return true;
        }
    }
    return false;
}

std::optional<Str> Normalizer::code_words(const std::vector<Str> & groups, const Match & match) const {
    const Str token = py::join(U"-", groups);
    bool any_digit = false;
    bool any_upper = false;
    for (const char32_t c : token) {
        any_digit = any_digit || uni::is_digit(c);
        any_upper = any_upper || uni::is_upper(c);
    }
    if (!any_digit) {
        return std::nullopt;                                                    // well-known, Wi-Fi
    }
    std::string kinds;
    for (const Str & g : groups) {
        bool has_digit = false;
        for (const char32_t c : g) {
            has_digit = has_digit || uni::is_digit(c);
        }
        kinds.push_back(py::isdigit(g) ? 'd' : has_digit ? 'm' : py::isupper(g) ? 'u' : 'l');
    }
    if (kinds.find_first_not_of('d') == std::string::npos) {
        if (groups.size() == 2) {
            bool years = true;
            bool leading_zero = false;
            size_t shortest = groups[0].size();
            for (const Str & g : groups) {
                years = years && g.size() == 4 && py::to_int(g) >= 1000 && py::to_int(g) <= 2100;
                leading_zero = leading_zero || (g.size() > 1 && g[0] == U'0');
                shortest = std::min(shortest, g.size());
            }
            if (!(!match.group(1).empty() || leading_zero || (shortest >= 4 && !years))) {
                const std::string kind = identifier(match.string(), static_cast<size_t>(match.start()));
                if (!(kind == "strong" || kind == "pin")) {
                    return std::nullopt;                                        // 10-15 and 2019-2021 are ranges
                }
            }
        }
        return digit_groups(groups);
    }
    if (groups.size() == 2 && kinds[1] == 'd' && (kinds[0] == 'u' || kinds[0] == 'l')) {
        const Str & letters = groups[0];
        const Str & number_text = groups[1];
        if ((kinds[0] == 'l' && letters.size() > 1) || reads_as_word(letters)) {
            return std::nullopt;                                                // pre-2020, COVID-19, SAVE-20
        }
        if (short_number(number_text)) {
            return spelled(letters) + U" " + cardinal(number_of(number_text));  // B-12, KX-300
        }
        return spelled(letters) + U" , " + digits(number_text);
    }
    if (groups.size() == 2 && kinds[0] == 'd' &&
        (kinds[1] == 'l' || groups[1].size() == 1 || RX(R"re(\d+[A-Za-z]+)re").fullmatch(groups[1]))) {
        return std::nullopt;                                                    // 5-year, 3-D, 5-10MB
    }
    for (size_t k = 0; k < groups.size(); ++k) {
        if ((kinds[k] == 'l' && groups[k].size() > 2) || (kinds[k] == 'm' && RX(R"re(\d+(?:s|st|nd|rd|th))re").fullmatch(groups[k]))) {
            return std::nullopt;                                                // 20-year-old, F-16s
        }
    }
    if (!any_upper && (kinds.find('l') != std::string::npos || kinds.find('m') != std::string::npos)) {
        return std::nullopt;                                                    // 3-in-1, 50s-60s
    }
    std::vector<Str> parts;
    for (size_t k = 0; k < groups.size(); ++k) {
        const Str & group = groups[k];
        if (kinds[k] == 'd') {
            parts.push_back(digits(group));
        } else if (kinds[k] == 'm') {
            parts.push_back(alnum(group));
        } else if (group.size() > 3 && is_word_(py::lower(group))) {
            parts.push_back(group);
        } else {
            parts.push_back(group.size() <= 5 ? spelled(group) : group);
        }
    }
    return py::join(U" , ", parts);
}

// Codes written with hyphens: AB-4471-Z, 91-220-0017, #A7-4419.
Str Normalizer::code(const Match & match) const {
    const std::optional<Str> words = code_words(py::split(match.group(2), U"-"), match);
    if (!words) {
        return match.group(0);
    }
    return (!match.group(1).empty() ? U" number " : U" ") + *words + U" ";
}

// Digits written in groups with spaces: a telephone number, a card or
// tracking number, or a large number with spaces between the thousands.
Str Normalizer::grouped(const Match & match) const {
    const StrView text = match.string();
    const Str plus = match.group("plus");
    Str lead = match.group("lead");
    Str trail = match.group("trail");
    const std::vector<Str> core = py::split(match.group("core"), U" ");
    Str head, tail;
    if (capital_words().count(lead) != 0) {                                     // "IN 2019 150 ..." in capitals
        head = lead + U" ";
        tail = !trail.empty() ? U" " + trail : Str();
        lead.clear();
        trail.clear();
    }
    const auto start = static_cast<size_t>(match.start());
    const auto end = static_cast<size_t>(match.end());
    const std::string kind = identifier(text, start);
    bool pure = true;
    size_t total = 0;
    bool long_part = false;
    bool thousands = true;
    for (size_t k = 0; k < core.size(); ++k) {
        pure = pure && py::isdigit(core[k]);
        total += core[k].size();
        long_part = long_part || core[k].size() > 2;
        thousands = thousands && (k == 0 || core[k].size() == 3);
    }
    const char32_t first_digit = py::at(core.at(0), 0);
    if (pure && plus.empty() && lead.empty()) {
        const bool amount_ = counted(text, end) || unit_after_->match(text.substr(end));
        if (kind.empty() && first_digit != U'0' && core[0].size() <= 3 && thousands &&
            (core.back() == U"000" || (amount_ && core.size() > 2))) {
            return head + py::join(U",", core) + tail;                          // 1 000 000: the number rules read it
        }
        if (!(!kind.empty() || (core[0].size() > 1 && first_digit == U'0') || (core.size() > 2 && long_part))) {
            return match.group(0);
        }
    } else if (!pure && lead.empty() && plus.empty() && kind.empty()) {
        return match.group(0);
    }
    const StrView zero = !plus.empty() || kind == "phone" || (pure && first_digit == U'0' && total >= 10) ? U"oh" : U"zero";
    Str spoken;
    if (pure) {
        spoken = digit_groups(core, zero);
    } else {
        std::vector<Str> parts;
        for (const Str & part : core) {
            parts.push_back(py::isdigit(part) ? digits(part, zero) : alnum(part, zero));
        }
        spoken = py::join(U" , ", parts);
    }
    if (!lead.empty()) {
        spoken = spelled(lead) + U" , " + spoken;
    }
    if (!trail.empty()) {
        spoken = spoken + U" , " + spelled(trail);
    }
    return head + (!plus.empty() ? U" plus " : U" ") + spoken + U" " + tail;
}

// Letters and digits written together: B12, 14C, X7K, PS5. A single letter is
// always named; it is never the article "a".
Str Normalizer::glued(const Match & match) const {
    std::vector<Str> out;
    for (const Str & run : RX(R"re([A-Za-z]+|\d+)re").findall(match.group(0))) {
        if (py::isdigit(run)) {
            out.push_back(short_number(run) ? cardinal(number_of(run)) : digits(run));
        } else if (run.size() == 1 || ((py::isupper(run) || run.size() <= 3) && !reads_as_word(run))) {
            out.push_back(spelled(run));
        } else {
            out.push_back(run);
        }
    }
    return U" " + py::join(U" ", out) + U" ";
}

// ---- web addresses and file names -----------------------------------------------

// One run of letters inside an address: a word, or letters to name.
Str Normalizer::address_part(const Str & part) const {
    static const std::unordered_set<Str> short_words = word_set(U"my to in on of go up no do me we us it is at by be so or an as if he hi ok");
    const Str lower = py::lower(part);
    if (part.size() > 1 && is_word_(lower) && (part.size() > 2 || short_words.count(lower) != 0)) {
        return part;
    }
    return part.size() <= 3 ? spelled(part) : part;
}

Str Normalizer::spoken_address(const Str & core) const {
    static const std::unordered_map<Str, Str> separators = {
        {U".", U"dot"}, {U"/", U"slash"}, {U"@", U"at"}, {U"-", U"dash"}, {U"_", U"underscore"}, {U":", U"colon"},
        {U"=", U"equals"}, {U"&", U"and"}, {U"+", U"plus"},
    };
    const std::vector<Str> pieces = RX(R"re([A-Za-z]+|\d+|.)re").findall(core);
    const bool mail = py::contains(core, U'@');
    bool host = false;
    std::vector<Str> out;
    for (size_t index = 0; index < pieces.size(); ++index) {
        const Str & piece = pieces[index];
        const Str following = index + 1 < pieces.size() ? pieces[index + 1] : Str();
        if (piece == U"@") {
            host = true;
        }
        const Str lower = py::lower(piece);
        if (py::isdigit(piece)) {
            out.push_back(address_digits(piece));
        } else if (!py::isalpha(piece)) {
            const auto found = separators.find(piece);
            out.push_back(found != separators.end() ? found->second : Str());
        } else if (index != 0 && pieces[index - 1] == U"." && (host || !mail) &&
                   (one_of(StrView(following), {U"", U"/", U":", U"?", U"#"}) ||
                    (piece.size() <= 3 && (endings_.count(lower) != 0 || !is_word_(lower))))) {
            // the ending of a file name or an address: a word if the table says so, otherwise letters
            const auto ending = endings_.find(lower);
            out.push_back(ending != endings_.end() && ending->second == U"word" ? lower : spelled(piece));
        } else {
            Str letters;
            for (const Str & part : RX(R"re([A-Z]?[a-z]+|[A-Z]+(?![a-z]))re").findall(piece)) {
                const Str spoken = address_part(part);
                if (!spoken.empty() && spoken[0] == kSpell) {
                    letters += part;
                    continue;
                }
                if (!letters.empty()) {
                    out.push_back(spelled(letters));
                    letters.clear();
                }
                out.push_back(spoken);
            }
            if (!letters.empty()) {
                out.push_back(spelled(letters));
            }
        }
    }
    std::vector<Str> kept;
    for (Str & item : out) {
        if (!item.empty()) {
            kept.push_back(std::move(item));
        }
    }
    return py::join(U" ", kept);
}

// Web and mail addresses, file names and paths: the dots, slashes,
// underscores and @ are spoken. The ending is a word when the endings table
// says so ("dot com") and letters otherwise ("dot P D F").
Str Normalizer::address(const Match & match) const {
    const Str whole = match.group(0);
    const Str text = py::rstrip(whole, U".,;:!?)");                             // punctuation of the sentence, not of the address
    const Str tail = whole.substr(text.size());
    if (!match.group("file").empty()) {
        const Str ending = match.group("ending");
        const bool known = endings_.count(py::lower(ending)) != 0;
        const bool marked = !match.group("folder").empty() || !match.group("path").empty() || py::contains(match.group("stem"), U'_');
        if (!(known || (marked && ending.size() <= 5 && (py::islower(ending) || py::isupper(ending)) && !is_word_(py::lower(ending))))) {
            return whole;                                                       // "the end.Then": a missing space, not a file
        }
    }
    const Str core = py::rstrip(RX(R"re(^(?:https?://)?(?:www\.)?)re").sub(U"", text), U"/");
    return U" " + spoken_address(core) + U" " + tail;
}

// Chat spellings that contain a digit or a slash: gr8, b4, 2day.
Str Normalizer::chat_form(const Match & match) const {
    const auto said = chat_.find(py::lower(match.group(0)));
    if (said == chat_.end()) {
        return match.group(0);
    }
    if (said->second == U"-") {
        return U" ";
    }
    std::vector<Str> parts;
    for (const Str & part : py::split(said->second)) {
        parts.push_back(py::startswith(part, U"*") ? spelled(StrView(part).substr(1)) : part);
    }
    return U" " + py::join(U" ", parts) + U" ";
}

// {...}: a pronunciation written by the user. Braces that hold anything else
// are no error: they stay and are read like brackets ("{name}", "{1, 2, 3}").
// So are single letters, which are phones only by chance ("x_{i}", "{n}").
Str Normalizer::inline_phones(const Match & match) {
    const std::vector<Str> symbols = py::split(match.group(2));
    bool letters = true;
    for (const Str & symbol : symbols) {
        letters = letters && symbol.size() == 1 && py::is_ascii(symbol);
    }
    if (symbols.empty() || letters) {
        return match.group(0);
    }
    Phones phones;
    try {
        std::vector<std::string> names;
        for (const Str & symbol : symbols) {
            names.push_back(encode(symbol));
        }
        phones = is_arpabet(names) ? convert(U"", names) : from_text(encode(py::join(U" ", symbols)));
    } catch (const ValueError &) {
        return match.group(0);
    }
    inline_.push_back(Inline{match.group(1), std::move(phones)});
    Str out = U" ";
    out.push_back(kInline);
    out.push_back(kInlineBase + static_cast<char32_t>(inline_.size() - 1));
    out.push_back(kInline);
    out += U" ";
    return out;
}

// U.S., e.g., p.m.: letters with full stops.
Str Normalizer::dotted(const Match & match, StrView text) const {
    const Str letters = py::replace(match.group(0), U".", U"");
    const Str key = py::lower(letters);
    const Str following = py::lstrip(text.substr(static_cast<size_t>(match.end())));
    Str words;
    if (key == U"eg") {
        words = U"for example";
    } else if (key == U"ie") {
        words = U"that is";
    } else {
        words = spelled(py::upper(letters));
    }
    const pyre::Match first = RX(R"re([A-Za-z']+)re").match(following);
    const bool ends_sentence = following.empty() ||
        (uni::is_upper(following[0]) && first && sentence_starters().count(py::lower(first.group(0))) != 0);
    return U" " + words + (ends_sentence && py::endswith(match.group(0), U".") ? U" ." : U"") + U" ";
}

Str Normalizer::abbreviation(const Match & match, StrView text) const {
    static const std::unordered_set<Str> numbered = word_set(
        U"ref reference serial account order invoice tracking part model phone tel fax id room flight case "
        U"ticket policy licence license booking customer vat tax reg registration");
    const Str token = match.group(1);
    const bool stop = !match.group(2).empty();
    const Str key = py::lower(token);
    const StrView following = text.substr(static_cast<size_t>(match.end()));
    const pyre::Match next_word = RX(R"re(\s*([A-Za-z]+|\d))re").match(following);
    const Str next = next_word ? next_word.group(1) : Str();
    const bool capital_next = next_word && uni::is_upper(next[0]);
    const StrView head = text.substr(0, static_cast<size_t>(match.start()));
    const bool capital_before = static_cast<bool>(RX(R"re([A-Z][a-z]+\s+$)re").search(head));
    const bool rest_empty = py::strip(following).empty();
    const bool token_capital = uni::is_upper(py::at(token, 0));
    if ((key == U"st" || key == U"dr") && token_capital) {
        if (capital_next && (stop || key == U"dr")) {
            return key == U"st" ? U" saint " : U" doctor ";
        }
        if (capital_before && stop) {
            return Str(key == U"st" ? U" street" : U" drive") + (rest_empty ? U" ." : U" ");
        }
        return match.group(0);
    }
    const auto found = abbreviations_.find(key);
    if (found == abbreviations_.end()) {
        return match.group(0);
    }
    const Str & when = found->second.first;
    Str words = found->second.second;
    const bool number_next = next_word && py::isdigit(next);
    if (when == U"title") {
        if (!(token_capital && (capital_next || stop))) {
            return match.group(0);
        }
    } else if (when == U"number") {
        const pyre::Match previous = RX(R"re(([A-Za-z]+)\.?\s+$)re").search(head);
        if (!(stop && (number_next || (key == U"no" && previous && numbered.count(py::lower(previous.group(1))) != 0)))) {
            return match.group(0);
        }
    } else if (!stop || (when == U"small" && py::isupper(token))) {
        return match.group(0);
    } else if (one_of(StrView(key), {U"mar", U"sun", U"sat", U"jan", U"dec", U"max", U"min", U"est", U"no"}) && !token_capital &&
               !number_next) {
        // also words; "max. load" and "est. time" are the abbreviation, "he sat." is not
        if (!(one_of(StrView(key), {U"max", U"min", U"est"}) && next_word && uni::is_lower(next[0]))) {
            return match.group(0);
        }
    }
    if (key == U"est" && !number_next) {
        words = U"estimated";                                                   // "Est. arrival"; "est. 1885" is "established"
    }
    const bool ends_sentence = when != U"title" && (rest_empty || (capital_next && sentence_starters().count(py::lower(next)) != 0));
    return U" " + words + (ends_sentence ? U" ." : U"") + U" ";
}

// The passes that need a digit: money, times, dates, telephone numbers,
// versions, codes, units, fractions, ranges.
Str Normalizer::figures(Str text) {
    // "$x = 2$", "$3 + 5 = 8$": the sign after the last figure closes a formula, it is
    // not a currency written after an amount. The closing sign becomes a mark no
    // rule knows, so the rest is read as it was before.
    text = RX(R"re((?<![\w$])(?<!\d\s)\$(?:(?=[^\s\d$.,;:!?)])[^$\n]{0,80})re"
              R"re(|(?=\d)[^$\n]{0,80}[=^_\\{}+*/<>][^$\n]{0,80})re"
              R"re(|(?=\s)[^$\n]{0,80}[=^_\\{}+*<>][^$\n]{0,80})\d\s?\$(?![\w$]))re")
               .sub_with([](const Match & m) { return py::slice(m.group(0), 0, -1) + Str(1, kFormulaSign); }, text);
    // continental figures become American ones, so every rule below sees one notation
    text = RX(R"re((?<![\w.,])([1-9]\d{0,2}(?:\.\d{3})+),(\d{1,2})(?!\d|[.,]\d))re").sub_with(
        [](const Match & m) {
            // "1.299,50" as "1,299.50". Four groups that could be an internet address are left alone.
            const std::vector<Str> groups = py::split(m.group(1), U".");
            bool address = groups.size() == 4;
            for (const Str & group : groups) {
                address = address && py::to_int(group) <= 255;
            }
            return address ? m.group(0) : py::join(U",", groups) + U"." + m.group(2);
        },
        text);
    text = RX(R"re(((?:[$£€¥₹]|(?<![A-Za-z])(?:)re" DZ_CURRENCY_CODES R"re())\s?-?\d+),(\d{1,2})(?!\d|[.,]\d))re").sub(U"\\1.\\2", text);
    text = RX(R"re((?<![\w.,])(\d+),(\d{1,2})(?=\s?(?:%|)re" DZ_SIGN_AFTER R"re()))re").sub(U"\\1.\\2", text);
    text = RX(R"re((?:(?<![\w.])(?P<minus>-)\s?)?(?P<symbol>[$£€¥₹]|(?<![A-Za-z])(?:)re" DZ_CURRENCY_CODES R"re()(?![A-Za-z]))\s?)re"
              R"re((?P<inner>-)?)re" DZ_AMOUNT R"re((?:\s?(?P<scale>thousand|million|billion|trillion|bn|[kKmMbB])(?![A-Za-z]))?)re")
               .sub_with([this](const Match & m) { return money(m); }, text);
    text = RX(R"re((?<![\w.,-])(\d[\d,]*(?:\.\d+)?)-(?=\d[\d,]*(?:\.\d+)?\s?(?:)re" DZ_SIGN_AFTER R"re()))re").sub(U"\\1 to ", text);    // 5-10 €
    // Those who write the currency after the amount often part the thousands with spaces: "750 000 $".
    text = RX(R"re((?:(?<![\w.])(?P<minus>-)\s?)?(?<![\w.,^]))re"
              R"re((?P<whole>\d{1,3}(?:,\d{3})+|[1-9]\d{0,2}(?: \d{3})+|\d+)(?:\.(?P<fraction>\d+))?)re"
              R"re(\s?(?P<symbol>)re" DZ_SIGN_AFTER R"re())re")
               .sub_with([this](const Match & m) { return money(m); }, text);
    text = RX(R"re((?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?¢)re").sub_with(
        [](const Match & m) { return U" " + number_words(m.group(1)) + U" " + (m.group(1) == U"1" ? U"cent" : U"cents") + U" "; }, text);

    // times of day, lengths of time, ratios
    text = RX(R"re(()re" DZ_CLOCK R"re()\s()re" DZ_ZONES R"re()\b)re").sub_with(
        [](const Match & m) { return m.group(1) + U" " + spelled(m.group(2)); }, text);
    text = RX(R"re((?<![\d.,:])(\d{1,2})\.([0-5]\d)(?=\s?[AaPp]\.?[Mm]\b))re").sub(U"\\1:\\2", text);        // 7.30 pm
    text = RX(R"re((?<![\d.,:])(\d{1,2})\.(00|15|30|45)(\s?(?:to|until|till|and|[-—])\s?)(\d{1,2})\.(00|15|30|45)(?!\d|\.\d))re")
               .sub_with([this](const Match & m) { return dot_times(m); }, text);
    text = RX(R"re(()re" DZ_CLOCK R"re()\s?[-—]\s?(?=\d{1,2}(?::[0-5]\d|\s?[AaPp]\.?[Mm]\b)))re").sub(U"\\1 to ", text);
    text = RX(R"re((?<![\d:.,])(\d{1,3}):(\d{1,3}(?:,\d{3})+|\d+)(?![\d:]|[.,]\d))re").sub_with(
        [this](const Match & m) { return ratio(m); }, text);
    text = RX(R"re((?<![\d:])(\d{1,2}):([0-5]\d)(?::([0-5]\d))?)re" DZ_HALF_DAY R"re(?(?![\d:]))re").sub_with(
        [this](const Match & m) { return time(m); }, text);
    text = RX(R"re((?<![\d.:,])(\d{1,2}))re" DZ_HALF_DAY R"re((?![A-Za-z]))re").sub_with(
        [](const Match & m) { return U" " + cardinal(py::to_int(m.group(1))) + U" " + half_day(m, m.group(2), m.group(3)); }, text);

    // dates
    bool day_first = false;
    bool month_first = false;
    for (const Match & m : RX(R"re((?<![\d/.])(\d{1,2})/(\d{1,2})(?:/(?:\d{4}|\d{2}))?(?![\d/]))re").finditer(text)) {
        const long long a = py::to_int(m.group(1));
        const long long b = py::to_int(m.group(2));
        if (a == 24 && b == 7) {
            continue;
        }
        day_first = day_first || (a > 12 && 12 >= b);
        month_first = month_first || (b > 12 && 12 >= a);
    }
    day_first_ = day_first && !month_first;
    dated_ = static_cast<bool>(RX(R"re((?<![\d/])\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})(?![\d/]))re").search(text));
    auto year_first = [this](const Match & m) {
        const Str year_text = m.group(1);
        const std::optional<Str> spoken = date(py::to_int(m.group(2)), py::to_int(m.group(3)), &year_text);
        return spoken ? *spoken : m.group(0);
    };
    text = RX(R"re((?<![\d-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d-]))re").sub_with(year_first, text);
    text = RX(R"re((?<![\d/])(\d{4})/(\d{1,2})/(\d{1,2})(?![\d/]))re").sub_with(year_first, text);
    text = RX(R"re((?<![\d.])(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d|\.\d))re").sub_with([this](const Match & m) { return dot_date(m); }, text);
    text = RX(R"re((?<![\d/])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\d/]))re").sub_with([this](const Match & m) { return slash_date(m); }, text);
    text = RX(R"re(\b()re" DZ_MONTHS R"re()\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s?[-—]\s?(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?(?![\d:]))re")
               .sub_with([this](const Match & m) { return days_of_month(m); }, text);
    text = RX(R"re((?<![\d.,/:-])(\d{1,2})(?:st|nd|rd|th)?\s?[-—]\s?(\d{1,2})(?:st|nd|rd|th)?\s+()re" DZ_MONTHS R"re()\b(?:,?\s+(\d{4}))?)re")
               .sub_with([this](const Match & m) { return days_of_month(m); }, text);
    text = RX(R"re(\b()re" DZ_MONTHS R"re()\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?(?![\d:]))re").sub_with(
        [this](const Match & m) {
            const Str year_text = m.group(3);
            const std::optional<Str> spoken = date(month_of(m.group(1)), py::to_int(m.group(2)), &year_text);
            return spoken ? *spoken : m.group(0);
        },
        text);
    text = RX(R"re((?<![\d.,/:-])(\d{1,2})(?:st|nd|rd|th)?\s+()re" DZ_MONTHS R"re()\b(\.(?=\s+[a-z\d]))?(?:,?\s+(\d{4}))?)re")
               .sub_with([this](const Match & m) { return day_month(m); }, text);

    // telephone numbers, versions, codes
    text = RX(R"re((?<![\d-])(?:\+?(1)[-.\s])?\(?(\d{3})\)?[-.\s](\d{3})[-.\s](\d{4})(?![\d-]))re").sub_with(
        [this](const Match & m) { return phone(m); }, text);
    text = RX(R"re((?<![\w.,-])(\d{3})-(\d{4})(?![\w-]))re").sub_with([this](const Match & m) { return phone(m); }, text);
    text = RX(R"re((?<![\w.])([vV])?(\d+(?:\.\d+){2,})(?:-(\d+)(?![\w-]|\.\d))?(?![\w]))re").sub_with(
        [](const Match & m) {
            std::vector<Str> parts;
            for (const Str & part : py::split(m.group(2), U".")) {
                parts.push_back(version_part(part));
            }
            return U" " + (!m.group(1).empty() ? spelled(m.group(1)) + U" " : Str()) + py::join(U" point ", parts) +
                (!m.group(3).empty() ? U" dash " + number_words(m.group(3)) : Str()) + U" ";
        },
        text);
    text = RX(R"re((?<![\w.])([vV])(\d+)\.(\d{2,3})(?!\d|\.\d|\w))re").sub_with(
        [](const Match & m) { return U" " + spelled(m.group(1)) + U" " + cardinal(number_of(m.group(2))) + U" point " + minor_part(m.group(3)) + U" "; },
        text);
    text = RX(R"re(\b()re" DZ_VERSIONED
              R"re()(\.?\s(?:(?:from|to)\s)?)(\d+)\.(\d{2,3})(?:(\s(?:to|or|and)\s)(\d+)\.(\d{2,3}))?(?!\d|\.\d|\s?%|\s(?:percent|million|billion)\b))re",
              pyre::IGNORECASE)
               .sub_with(
                   [](const Match & m) {
                       return m.group(1) + m.group(2) + cardinal(number_of(m.group(3))) + U" point " + minor_part(m.group(4)) +
                           (!m.group(5).empty() ? m.group(5) + cardinal(number_of(m.group(6))) + U" point " + minor_part(m.group(7)) : Str()) + U" ";
                   },
                   text);
    text = RX(R"re((?<![\w.,/-])(#)?([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)(?![\w-]|[.,]\d))re").sub_with(
        [this](const Match & m) { return code(m); }, text);
    text = RX(R"re((?<![\w.,/-])(\d+(?:/\d+){2,})(?![\w/]|[.,]\d))re").sub_with(
        [](const Match & m) { return U" " + digit_groups(py::split(m.group(1), U"/")) + U" "; }, text);
    text = RX(R"re((?<![\w.,:/#+-])(?P<plus>\+)?(?:(?P<lead>[A-Z]{2}) )?)re"
              R"re((?P<core>[A-Z0-9]*\d[A-Z0-9]*(?: [A-Z0-9]*\d[A-Z0-9]*)+))re"
              R"re((?(lead)(?: (?P<trail>[A-Z]{2})(?![\w]))?))re"
              R"re((?![\w]|[.,:/-]\d|\s?[%°]))re")
               .sub_with([this](const Match & m) { return grouped(m); }, text);

    text = RX(R"re((?<![\w.])-(?=\d))re").sub(U" minus ", text);
    text = RX(R"re((?:(?<=\d)\s?[xX×]|×)\s?(?=\d[\d.,]*\s?(?:)re" DZ_LENGTHS R"re()\b))re").sub(U" by ", text);    // 3 × 4 metres
    text = RX(R"re((?<=\d)\s[xX×]\s(?=\d))re").sub(U" times ", text);
    text = RX(R"re((?<![\w/'])[A-Za-z0-9/]*[0-9/][A-Za-z0-9/]*(?![\w/']))re").sub_with([this](const Match & m) { return chat_form(m); }, text);
    text = RX(R"re((?<![\w.])'?(\d{2}|\d{4})'?s\b)re").sub_with(
        [](const Match & m) {
            const long long value = py::to_int(m.group(1));
            return U" " + plural_of(m.group(1).size() == 4 ? year(value) : cardinal(value)) + U" ";
        },
        text);
    text = RX(R"re((?<![\w.])(\d+)(?:st|nd|rd|th)\b)re").sub_with([](const Match & m) { return U" " + ordinal(number_of(m.group(1))) + U" "; }, text);
    text = RX(R"re((?<![\w.])(144|240|360|480|540|576|720|900|1080|1440|2160|4320)([pi])\b)re").sub_with(
        [](const Match & m) { return U" " + pairs(py::to_int(m.group(1))) + U" " + spelled(m.group(2)) + U" "; }, text);     // 1080p
    text = RX(R"re((\d[\d,]*(?:\.\d+)?)\s?%)re").sub_with([](const Match & m) { return U" " + number_words(m.group(1)) + U" percent "; }, text);
    text = RX(R"re((?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?(km|cm|mm|m|ft)([23])(?![\w]))re").sub_with([this](const Match & m) { return square(m); }, text);
    text = unit_->sub_with([this](const Match & m) { return measure(m); }, text);
    text = RX(R"re((?<![\d/.,-])(\d+) (\d)/(2|3|4|5|6|8|10|16)(?![\d/]|\.\d))re").sub_with([this](const Match & m) { return mixed_fraction(m); }, text);
    text = RX(R"re((?<![\d/.])(\d+)/(\d+)(?![\d/]|\.\d))re").sub_with([this](const Match & m) { return fraction(m); }, text);
    text = RX(R"re((?<![\w.,-])(\d{4})[-–](\d{4})(?![\w.,-]))re").sub_with(
        [](const Match & m) { return U" " + year(py::to_int(m.group(1))) + U" to " + year(py::to_int(m.group(2))) + U" "; }, text);
    text = RX(R"re((?<![\w.,-])(\d+)[-–](\d+)(?![\w-]|[.,]\d))re").sub_with(
        [](const Match & m) { return U" " + number_words(m.group(1)) + U" to " + number_words(m.group(2)) + U" "; }, text);
    text = RX(R"re((?<![\w.])(\d+)x\b)re").sub_with([](const Match & m) { return U" " + number_words(m.group(1)) + U" times "; }, text);
    text = RX(R"re((?<=\d)\^(?=\d))re").sub(U" to the power of ", text);
    return text;
}

}  // namespace dizione::normalize
