// Text -> the words to say and the punctuation marks between them.
//
// Everything that is not a plain word is turned into words here, by rule:
// numbers, ordinals, money, dates, times, telephone numbers, units,
// abbreviations, initialisms, symbols, chat spellings, version numbers, codes
// and reference numbers, web addresses and file names. Emoticons and emoji are
// dropped. A pronunciation can be given inline, "{HH AH0 L OW1}", alone or
// right after the word it replaces.
//
// Part of the dizione C++ port (normalize.py). The patterns are the Python
// ones, word for word. Standard library only.
#pragma once

#include "pyre.h"
#include "symbols.h"
#include "text.h"

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dizione::normalize {

// A word to say, or a punctuation mark.
struct Token {
    bool punct = false;
    Str mark;                           // the mark, for punctuation
    Str text;                           // letters and apostrophes, lower case
    bool attach = false;                // continues the previous word (hyphenated compound, parts of one token)
    Str raw;                            // as written
    bool spell = false;                 // read letter by letter
    std::optional<Phones> phones;       // pronunciation given inline by the user

    static Token word(Str text, Str raw = Str(), bool spell = false, bool attach = false);
    static Token punctuation(Str mark);
};

// ---- numbers as words (normalize_numbers.cpp) -------------------------------

// A whole number as Python has it after int() of digits: of any length.
struct Number {
    std::string digits;                 // no leading zeros; "0" for zero

    bool fits() const { return digits.size() <= 18; }       // in a long long
    long long value() const;            // when fits()
    bool is(long long other) const { return fits() && value() == other; }
    bool operator==(const Number & other) const { return digits == other.digits; }
};
// int(text) for digits; raises ValueError for anything else.
Number number_of(StrView text);

Str cardinal(long long number);         // 1234 -> "one thousand two hundred thirty four"
Str cardinal(const Number & number);
Str ordinal(long long number);
Str ordinal(const Number & number);
Str digits(StrView text, StrView zero = U"zero");
Str year(long long number);             // the way a year is said: 1984, 1905, 1900, 2005, 2026
Str pairs(long long number);            // two digits at a time, as model numbers are: 4070, 9001, 720
Str number_words(StrView text);         // an unsigned number as written: 1,234  3.14  007
Str duration(long long hours, long long minutes, long long seconds);
Str plural_of(StrView words);           // plural of a spelled number, for decades
Str fraction_words(long long top, long long bottom, bool article = false);

// ---- the normaliser -----------------------------------------------------------

class Normalizer {
public:
    // `is_word(spelling)` tells whether the dictionary knows a spelling.
    using IsWord = std::function<bool(const Str &)>;
    explicit Normalizer(IsWord is_word);

    // Words and punctuation marks of `text`.
    std::vector<Token> tokens(const Str & text);
    // The text with every non-word written out, markers included.
    Str rewrite(const Str & text);

private:
    using Match = pyre::Match;
    using Names = std::pair<Str, Str>;          // singular, plural

    // pieces of the text that are not words (normalize_figures.cpp)
    Str amount(const Str & symbol, const Str & whole, const Str & fraction, const Str & scale) const;
    Str money(const Match & match) const;
    Str time(const Match & match) const;
    Str dot_times(const Match & match) const;
    Str ratio(const Match & match) const;
    std::optional<Str> date(long long month, long long day, const Str * year_text = nullptr, bool the = false) const;
    Str slash_date(const Match & match) const;
    Str dot_date(const Match & match) const;
    Str day_month(const Match & match) const;
    Str days_of_month(const Match & match) const;
    Str day_range(const Match & match) const;
    std::optional<Str> short_date(const Match & match) const;
    Str fraction(const Match & match) const;
    Str mixed_fraction(const Match & match) const;
    Str vulgar(const Match & match) const;
    Str phone(const Match & match) const;
    Str measure(const Match & match) const;
    Str square(const Match & match) const;
    bool counted(StrView text, size_t end) const;
    Str number(const Match & match, StrView text) const;
    bool reads_as_word(const Str & letters) const;
    std::optional<Str> code_words(const std::vector<Str> & groups, const Match & match) const;
    Str code(const Match & match) const;
    Str grouped(const Match & match) const;
    Str glued(const Match & match) const;
    Str address_part(const Str & part) const;
    Str spoken_address(const Str & core) const;
    Str address(const Match & match) const;
    Str chat_form(const Match & match) const;
    Str inline_phones(const Match & match);
    Str dotted(const Match & match, StrView text) const;
    Str abbreviation(const Match & match, StrView text) const;
    Str figures(Str text);

    // signs and words (normalize.cpp)
    Str joined_comparisons(const Str & text) const;
    Str signs(Str text) const;
    std::vector<Token> chat_words(const Str & chat, const Str & raw) const;
    std::vector<Token> word_tokens(const Str & raw, bool all_caps, const Str & previous, bool following, bool fresh) const;

    IsWord is_word_;
    std::unordered_map<Str, std::pair<Str, Str>> abbreviations_;        // when, words
    std::unordered_map<Str, Names> units_;
    std::unordered_map<Str, Names> exact_units_;
    std::unordered_map<Str, Str> chat_;
    std::unordered_set<Str> replies_;
    std::unordered_map<Str, Str> endings_;
    std::string unit_names_;                                            // a pattern: the unit names, longest first
    std::optional<pyre::Regex> unit_;
    std::optional<pyre::Regex> unit_after_;                             // a unit right after a number
    std::optional<pyre::Regex> unit_or_scale_after_;
    struct Inline {
        Str written;
        Phones phones;
    };
    std::vector<Inline> inline_;
    bool day_first_ = false;
    bool dated_ = false;
};

// ---- shared by the normaliser's files ---------------------------------------

// Private-use characters that carry decisions through the rewriting of the
// text: letters to be named sit between kSpell and kEnd, and an inline
// pronunciation is a single character (its number added to kInlineBase)
// between two kInline. kIs marks a comparison that is said with "is".
constexpr char32_t kSpell = 0xE000;
constexpr char32_t kEnd = 0xE001;
constexpr char32_t kInline = 0xE003;
constexpr char32_t kFormulaSign = 0xE004;
constexpr char32_t kAngleMark = 0xE005;
constexpr char32_t kIs = 0xE006;
constexpr char32_t kInlineBase = 0xE100;

// The same characters, as they are written inside a pattern.
#define DZ_SPELL "\\U0000E000"
#define DZ_END "\\U0000E001"
#define DZ_INLINE "\\U0000E003"
#define DZ_IS "\\U0000E006"

// Names of months and days, as alternatives of a pattern.
#define DZ_MONTHS \
    "January|February|March|April|May|June|July|August|September|October|November|December" \
    "|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec"
#define DZ_DAYS "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Thurs|Thur|Tues|Mon|Tue|Wed|Thu|Fri|Sat|Sun"

// A pattern compiled once, where it is used: RX(R"re(\d+)re").
#define RX(...) \
    ([]() -> const ::dizione::pyre::Regex & { \
        static const ::dizione::pyre::Regex value(__VA_ARGS__); \
        return value; \
    }())

Str spelled(StrView letters);                                           // letters to be named
std::vector<Str> words_before(StrView text, size_t position, size_t count = 4);
// What the words before a number say it is: "phone", "pin", "strong" (a
// number, an ID, a reference), "weak" (an order, a ticket), "enter", or "".
std::string identifier(StrView text, size_t position);
std::unordered_set<Str> word_set(StrView words);
bool short_number(StrView text);
Str without_angles(Str text);
const std::unordered_set<Str> & sentence_starters();
const std::unordered_set<Str> & capital_words();

}  // namespace dizione::normalize
