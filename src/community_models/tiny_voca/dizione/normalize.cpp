// The order of the passes over a text, the signs, and the words themselves.
// Each function follows the one of the same name in dizione's normalize.py.
#include "normalize.h"

#include <algorithm>
#include <set>

namespace dizione::normalize {

namespace {

#define DZ_SIZES "XXXL|XXL|XXS|XL|XS|S|M|L"
// "A", "a" and the small letters chat writes for a word ("k", "u", "r") are read
// as words. Beside a sign or alone in brackets they stand for something and are
// named: "a=10", "b>a", "k = k + 1", "(a)". After "=" or a spaced sign the letter
// must not open a phrase ("mode = a set of notes").
#define DZ_SIGN R"re([<>+*/^≤≥≠≈×÷±])re"
#define DZ_SPACED_SIGN R"re( [<>+*≤≥≠≈×÷±] )re"
#define DZ_NO_PHRASE R"re((?![\w'’-]|\s+(?!(?:and|or|then|if|when|where|while|for|is|are|was|but|so|in|to)\b)[A-Za-z]))re"

bool one_of(StrView value, std::initializer_list<StrView> options) {
    for (const StrView option : options) {
        if (value == option) {
            return true;
        }
    }
    return false;
}

bool is_mark_of(const Token & token, StrView marks) {
    return token.punct && token.mark.size() == 1 && marks.find(token.mark[0]) != StrView::npos;
}

// An accented Latin letter without its accent ("café" is in the dictionary
// as "cafe"). Other characters are left alone.
Str plain_letter(char32_t c) {
    if (c < 0x80 || !uni::is_alpha(c)) {
        return Str(1, c);
    }
    switch (c) {
        case 0xDF: return U"ss";            // sharp s
        case 0xE6: return U"ae";
        case 0xC6: return U"Ae";
        case 0x153: return U"oe";
        case 0x152: return U"Oe";
        case 0xF8: return U"o";
        case 0xD8: return U"O";
        case 0x111: return U"d";
        case 0x142: return U"l";
        case 0x141: return U"L";
        default: break;
    }
    Str base;
    for (const char32_t part : uni::nfkd(StrView(&c, 1))) {
        if (uni::combining(part) == 0) {
            base.push_back(part);
        }
    }
    return py::is_ascii(base) && py::isalpha(base) ? base : Str(1, c);
}

// Words whose accent is the only sign of how they are read: without it "résumé" is "resume".
const char * accent_decides(const Str & word) {
    static const std::pair<const char32_t *, const char *> table[] = {
        {U"résumé", "R EH1 Z AH0 M EY2"}, {U"resumé", "R EH1 Z AH0 M EY2"}, {U"exposé", "EH2 K S P OW0 Z EY1"},
        {U"rosé", "R OW0 Z EY1"},         {U"pâté", "P AA0 T EY1"},         {U"maté", "M AA1 T EY2"},
        {U"saké", "S AA1 K EY2"},         {U"lamé", "L AA0 M EY1"},
    };
    for (const auto & entry : table) {
        if (word == entry.first) {
            return entry.second;
        }
    }
    throw KeyError(encode(word));
}

const pyre::Regex & angles() {
    // "<b>", "</p>", "<your name>", "Vec<u64>", "map<int, int>": angle brackets that
    // enclose something. The "<" and ">" that are left may compare two things.
    return RX(R"re((?<![\w))re" DZ_END DZ_IS R"re(])<[A-Za-z0-9/!?][^<>(){}+*;|\n]*>)re"
              R"re(|(?<=[\w)re" DZ_END DZ_IS R"re(])<[^<>(){}+*;|\n]{1,80}>(?![\w$]))re"
              R"re(|</?[A-Za-z][\w-]+/?>)re");
}

// A comparison between two values is said with "is": "67 > 66" is "sixty seven
// is greater than sixty six", "see if b>a" is "see if B is greater than A". A
// value is a figure, a single letter or a closing bracket, with the same space
// (or none) on both sides of the sign. "Is" is left out where the comparison
// only describes the value: "for n > 4", "any q > 0", "an integer k ≥ 2". The
// mark is put after the first value and stays there while the figures around
// it are turned into words.
Str mark_comparisons(const Str & text) {
    static const std::unordered_set<Str> clause_openers = word_set(
        U"if elif while when whenever where whether that because since as and or but so then think thinks "
        U"thought know knows say says said suppose assume means implies hence thus therefore although though "
        U"unless until once else also clearly obviously indeed here now again first next finally similarly "
        U"otherwise however moreover");
    const Str plain = without_angles(text);
    return RX(R"re((?P<gap>[ \t]?)(?P<sign>>=|<=|!=|[<>≥≤≠])(?P=gap))re").sub_with(
        [&](const pyre::Match & match) {
            const Str before = plain.substr(0, static_cast<size_t>(match.start()));
            const Str after = plain.substr(static_cast<size_t>(match.end()));
            const pyre::Match value = RX(R"re((?:(?<![\w.,])\d[\d.,]*%?|(?<![\w'’.-])[A-Za-z]|[)\]])$)re").search(before);
            if (py::at(plain, match.start("sign")) != py::at(match.group("sign"), 0) || !value ||
                !RX(R"re((?:[-+]?[$£€¥₹]?\d[\d.,]*%?(?!\w)|\.\d|[A-Za-z](?![\w'’])|[(\[]))re").match(after)) {
                return match.group(0);
            }
            long start = value.start();
            const Str found = value.group(0);
            if (found == U")" || found == U"]") {               // "f(x) > 0", "a[i] < b": the whole of it is the value
                long depth = 0;
                while (start >= 0) {
                    const char32_t c = before[static_cast<size_t>(start)];
                    depth += (c == U')' || c == U']' ? 1 : 0) - (c == U'(' || c == U'[' ? 1 : 0);
                    if (depth == 0) {
                        break;
                    }
                    --start;
                }
                while (start > 0) {
                    const char32_t c = before[static_cast<size_t>(start - 1)];
                    if (!(uni::is_alnum(c) || c == U'_' || c == U'.')) {
                        break;
                    }
                    --start;
                }
            }
            const Str head = py::rstrip(py::slice(before, 0, std::max(start, 0L)));
            const pyre::Match word = RX(R"re([A-Za-z']+$)re").search(head);
            if (word && clause_openers.count(py::lower(word.group(0))) == 0) {
                return match.group(0);
            }
            return Str(1, kIs) + match.group(0);
        },
        text);
}

// Mark the letters that stand for something, to be named: "a=10", "b>a", "(k)".
Str named_letters(const Str & text) {
    bool any = false;
    for (const char32_t sign : StrView(U"=<>+*/^()≤≥≠≈×÷±")) {
        any = any || py::contains(text, sign);
    }
    if (!any) {
        return text;
    }
    const Str plain = without_angles(text);
    static const std::vector<pyre::Regex> letter_places = {
        pyre::Regex(R"re((?<![\w'’.-])(?P<letter>[Aakur])(?=)re" DZ_IS R"re(?\s?[<>!=]?=))re"),                      // a=10, a = 10, a >= b
        pyre::Regex(R"re([<>!=]?=\s?(?P<letter>[Aakur]))re" DZ_NO_PHRASE),                                          // x=a, c = a + b
        pyre::Regex(R"re((?<![\w'’.-])(?P<letter>[Aakur])(?=)re" DZ_IS "?" DZ_SIGN R"re((?:[\w($]|\.\d)))re"),       // a>b, a+b, a/b, a^2
        pyre::Regex(R"re((?<=[\w))re" DZ_IS "]" DZ_SIGN R"re()(?<![Ww]/)(?P<letter>[Aakur])(?![\w'’-]))re"),         // b>a, 1/a; not "w/a friend"
        pyre::Regex(R"re((?<![\w'’.-])(?P<letter>[Aakur])(?=)re" DZ_IS "?" DZ_SPACED_SIGN R"re((?:[A-Za-z]|\d[\d.,]*)(?![\w'’])))re"),   // a > b
        pyre::Regex(R"re((?<![\w'’.])(?:[A-Za-z]|\d[\d.,]*))re" DZ_IS "?" DZ_SPACED_SIGN R"re((?P<letter>[Aakur]))re" DZ_NO_PHRASE),    // b > a
        pyre::Regex(R"re((?<=\()(?P<letter>[Aakur])(?=\)|, ))re"),                                                  // (a), f(a, b)
        pyre::Regex(R"re((?<=, )(?P<letter>[Aakur])(?=\)))re"),                                                     // f(b, a)
        pyre::Regex(R"re((?:(?<=\s)|^)(?P<letter>[akur])(?=\)\s))re"),                                              // a) the first point
    };
    std::set<long> places;
    for (const pyre::Regex & pattern : letter_places) {
        for (const pyre::Match & match : pattern.finditer(plain)) {
            places.insert(match.start("letter"));
        }
    }
    if (places.empty()) {
        return text;
    }
    Str out;
    for (size_t index = 0; index < text.size(); ++index) {
        if (places.count(static_cast<long>(index)) != 0) {
            out += spelled(py::upper(text.substr(index, 1)));
        } else {
            out.push_back(text[index]);
        }
    }
    return out;
}

// `words` for a comparison sign, after "is" where the mark for it stands before the sign.
Str compared(const pyre::Match & match, StrView words) {
    const Str whole = match.group(0);
    const Str head = py::rstrip(match.string().substr(0, static_cast<size_t>(match.start())), U" ");
    const bool marked = (!whole.empty() && whole[0] == kIs) || (!head.empty() && head.back() == kIs);
    return Str(U" ") + (marked ? U"is " : U"") + Str(words) + U" ";
}

// Spellings a word stretched for emphasis may stand for. A letter typed three
// times or more is cut to one or to two: at the end of a word one is the usual
// spelling ("sooo", "heyyy") unless the letter is one that doubles there
// ("alll", "offf"); inside a word two ("goood", "cooool"). "mmm" is tried with
// two first: it is the hum "mm", not the letter M.
std::vector<Str> unstretched(const Str & word) {
    if (!RX(R"re((.)\1\1)re").search(word)) {
        return {};
    }
    auto cut = [&](bool double_inside, bool double_final) {
        return RX(R"re((.)\1{2,})re").sub_with(
            [&](const pyre::Match & match) {
                const bool final = static_cast<size_t>(match.end()) == word.size();
                return py::repeat(match.group(1), (final ? double_final : double_inside) ? 2 : 1);
            },
            word);
    };
    bool only_m = !word.empty();
    for (const char32_t c : word) {
        only_m = only_m && c == U'm';
    }
    const bool liked = RX(R"re(([lfsz])\1\1+$)re").search(word) || only_m || RX(R"re((.)\1{2,}$)re").sub(U"\\1", word) == U"to";
    std::vector<Str> out;
    for (const Str & candidate : {cut(true, liked), cut(true, !liked), cut(false, liked), cut(false, !liked)}) {
        if (std::find(out.begin(), out.end(), candidate) == out.end()) {
            out.push_back(candidate);
        }
    }
    return out;
}

// True when the word at `place` is a clause by itself and can be a reply: "k",
// "k, thanks", "see you there, k?". Between two commas, or with a single letter
// next to it across a comma, it is one of a list: "i, j, k".
bool reply(const std::vector<Token> & out, size_t place) {
    const long size = static_cast<long>(out.size());
    std::vector<Str> marks;
    for (const long step : {-1L, 1L}) {
        long index = static_cast<long>(place) + step;
        while (index >= 0 && index < size && is_mark_of(out[static_cast<size_t>(index)], U"\"'()")) {
            index += step;                                      // quotation marks and brackets part nothing
        }
        if (!(index >= 0 && index < size)) {
            marks.emplace_back();                               // the text starts or ends here
        } else if (!out[static_cast<size_t>(index)].punct) {
            return false;                                       // a word in the same clause: "the value of k"
        } else {
            const Token & mark = out[static_cast<size_t>(index)];
            marks.push_back(mark.mark);
            const long beyond = index + step;
            if (mark.mark == U"," && beyond >= 0 && beyond < size && !out[static_cast<size_t>(beyond)].punct &&
                out[static_cast<size_t>(beyond)].text.size() == 1 && out[static_cast<size_t>(beyond)].text != U"i") {
                return false;
            }
        }
    }
    return !(marks.size() == 2 && marks[0] == U"," && marks[1] == U",");
}

}  // namespace

// `text` with the angle brackets that enclose something replaced by a mark,
// one for one, the inner ones first.
Str without_angles(Str text) {
    while (py::contains(text, U'<') && py::contains(text, U'>')) {
        Str marked = angles().sub_with(
            [](const pyre::Match & match) {
                const Str whole = match.group(0);
                return Str(1, kAngleMark) + whole.substr(1, whole.size() - 2) + Str(1, kAngleMark);
            },
            text);
        if (marked == text) {
            break;
        }
        text = std::move(marked);
    }
    return text;
}

Str Normalizer::rewrite(const Str & input) {
    Str text = RX(R"re((?<![\d.,/])(?:(\d+)\s?)?([½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅐⅛⅜⅝⅞⅑⅒]))re").sub_with(
        [this](const Match & m) { return vulgar(m); }, input);          // before "1½" becomes "11/2" and "m²" becomes "m2"
    text = RX(R"re((?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?(km|cm|mm|m|ft|mi|yd|in)([²³]))re").sub_with([this](const Match & m) { return square(m); }, text);
    text = py::replace(py::replace(text, U"\U000000B2", U" squared "), U"\U000000B3", U" cubed ");
    text = py::replace(py::replace(uni::nfkc(text), U"\U00002044", U"/"), U"\U00002212", U" minus ");
    if (!py::is_ascii(text)) {
        text = RX(R"re(\b(résumé|resumé|exposé|rosé|pâté|maté|saké|lamé)(s?)\b)re", pyre::IGNORECASE).sub_with(
            [](const Match & m) { return U"{" + S(accent_decides(py::lower(m.group(1)))) + (!m.group(2).empty() ? U" Z" : U"") + U"}"; }, text);
        Str plain;
        for (const char32_t c : text) {
            plain += plain_letter(c);
        }
        text = std::move(plain);
    }
    text = RX(R"re((?<=\d)–(?=\d))re").sub(U"-", text);                 // 10–15 is a range, not a pause
    text = RX(R"re(\b()re" DZ_DAYS "|" DZ_MONTHS R"re()\b(\.?)\s?[-–]\s?()re" DZ_DAYS "|" DZ_MONTHS R"re()\b(\.?))re").sub_with(
        [this](const Match & m) { return day_range(m); }, text);
    // quotation marks, dashes and brackets, each kind written one way
    static const std::pair<char32_t, char32_t> same[] = {
        {0x2018, U'\''}, {0x2019, U'\''}, {0x201A, U'\''}, {0x201B, U'\''}, {U'`', U'\''}, {0xB4, U'\''},
        {0x201C, U'"'},  {0x201D, U'"'},  {0x201E, U'"'},  {0x201F, U'"'},  {0xAB, U'"'},  {0xBB, U'"'},
        {0x2013, 0x2014}, {0x2015, 0x2014}, {U'[', U'('}, {U']', U')'},
    };
    for (char32_t & c : text) {
        for (const auto & pair : same) {
            if (c == pair.first) {
                c = pair.second;
                break;
            }
        }
    }
    inline_.clear();
    text = RX(R"re(([A-Za-z']+)?\{([^{}]*)\})re").sub_with([this](const Match & m) { return inline_phones(m); }, text);
    for (char32_t & c : text) {
        if (c == U'{') {
            c = U'(';
        } else if (c == U'}') {
            c = U')';
        }
    }
    text = RX(R"re((?P<url>(?:https?://|www\.)[^\s<>\"']+))re"
              R"re(|(?P<mail>[\w.+-]+@[\w-]+(?:\.[\w-]+)+))re"
              R"re(|(?P<file>(?<![\w@./-])(?P<folder>/?(?:[\w~-]+/)+)?(?P<stem>[\w-]+(?:\.[\w-]+)*)\.(?P<ending>[A-Za-z][A-Za-z0-9]{0,9}))re"
              R"re((?![\w-])(?P<path>/[^\s<>\"']*)?))re")
               .sub_with([this](const Match & m) { return address(m); }, text);
    text = py::replace(text, U"\U00002192", U" to ");
    // emoji and emoticons are dropped: never read, never a pause
    text = RX(R"re([\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002190-\U000021FF\U00002B00-\U00002BFF\U0000FE0F\U0000200D\U000020E3]+)re")
               .sub(U" ", text);
    text = RX(R"re((?<![A-Za-z0-9])(?:[:;=]['^-]?[DPpOo3|/\\*\]\[}{><]+|[)(\]\[][-^']?[:;=]|<3+|</3|\^_?\^|-_-|[oO]_[oO]|>_<|T_T|\\o/|:'[()])(?![A-Za-z0-9]))re"
              R"re(|[:;]['^-]?[)(]+)re"
              R"re(|(?<![A-Za-z0-9)])=['^-]?[)(]+|=['^-]?[)(]+(?![A-Za-z0-9(.$-]))re"       // "=(" is a face, but not in "y=(a+b)"
              R"re(|(?<=[A-Za-z]{2})<3+(?![.,]?\d))re")                                     // a heart joined to a word: "you<3"
               .sub(U" ", text);
    text = RX(R"re(\.{3,})re").sub(U"\U00002026", text);
    text = RX(R"re((?<=\d)\s-\s(?=\d[\d.,]*\s?=))re").sub(U" minus ", text);
    text = RX(R"re((?<![\d.,/-])(\d[\d.,]*)-(?=\d[\d.,]*=))re").sub(U"\\1 minus ", text);      // 10-3=7, not the date 2024-11-15=
    text = RX(R"re(-{2,}|\s-\s)re").sub(U"\U00002014", text);
    text = RX(R"re(\bw/o\b)re", pyre::IGNORECASE).sub(U" without ", text);
    text = RX(R"re(\bw/(?=\s|$))re", pyre::IGNORECASE).sub(U" with ", text);
    text = RX(R"re(\bn/a\b)re", pyre::IGNORECASE).sub(U" not applicable ", text);
    text = py::replace(text, U"'n'", U" 'n ");
    if (RX(R"re([<>≥≤≠]|!=)re").search(text)) {
        text = mark_comparisons(text);
    }
    if (RX(R"re(\d)re").search(text)) {                                 // most text has no digit and skips these passes
        text = figures(std::move(text));
    }
    return signs(std::move(text));
}

// "b>a", "x<5", "p<.05": the words for "<" and ">" written with no spaces. It is
// read as a comparison when a figure stands on one side, or a single letter or
// a bracket on both. "File>Open" and "M>Train" are not comparisons.
Str Normalizer::joined_comparisons(const Str & text) const {
    const Str plain = without_angles(text);
    return RX(R"re((?<=[\w))re" DZ_END DZ_IS R"re(])[<>](?=[\w()re" DZ_SPELL R"re(]|\.\d| minus \d))re").sub_with(
        [&](const Match & match) {
            const Str before = py::rstrip(text.substr(0, static_cast<size_t>(match.start())), Str(1, kIs));
            const Str after = text.substr(static_cast<size_t>(match.end()));
            const Str sign = match.group(0);
            if (py::at(plain, match.start()) != sign[0]) {              // the bracket of a tag
                return sign;
            }
            if (!(RX(R"re((?<![\w.,])\d[\d.,]*%?$)re").search(before) || RX(R"re((?:\d[\d.,]*%?(?!\w)|\.\d| minus \d))re").match(after) ||
                  (RX(R"re((?:(?<![\w'’.])[A-Za-z]|[))re" DZ_END R"re(])$)re").search(before) &&
                   RX(R"re((?:[A-Za-z](?![\w'’])|[()re" DZ_SPELL R"re(]))re").match(after)))) {
                return sign;
            }
            return compared(match, sign == U"<" ? U"less than" : U"greater than");
        },
        text);
}

// The passes that follow: signs, abbreviations, and the numbers that are left.
Str Normalizer::signs(Str text) const {
    text = named_letters(text);
    if (py::contains(text, U'<') || py::contains(text, U'>')) {
        text = joined_comparisons(text);
    }
    text = RX(R"re((?<![\w-])()re" DZ_SIZES R"re()-()re" DZ_SIZES R"re()(?![\w-]))re").sub_with(
        [](const Match & m) { return U" " + spelled(m.group(1)) + U" to " + spelled(m.group(2)) + U" "; }, text);
    text = RX(R"re(#(?=\d))re").sub(U" number ", text);
    text = RX(R"re(#(?=[A-Z]{1,4}-?\d))re").sub(U" number ", text);     // Order #A7
    text = RX(R"re(#(?=[A-Za-z]))re").sub(U" hashtag ", text);
    text = RX(R"re(~(?=\s?\d))re").sub(U" about ", text);
    text = RX(R"re((?<=[\w\s)re" DZ_END DZ_IS R"re(])(>=|<=|!=)(?=[\w\s)re" DZ_SPELL R"re(]))re").sub_with(
        [](const Match & m) {
            const Str sign = m.group(1);
            return compared(m, sign == U">=" ? U"greater than or equal to" : sign == U"<=" ? U"less than or equal to" : U"not equal to");
        },
        text);
    text = RX(R"re((?<=\d) Ti\b)re").sub(U" " + spelled(U"TI"), text);  // RTX 4070 Ti
    text = RX(R"re(\b([Rr])ef\.\s?[Nn]o\.(?=\s))re").sub(U"\\1eference number", text);
    if (py::contains(text, U'.')) {
        const Str whole = text;
        text = RX(R"re(\b((?:[A-Za-z]\.){2,}))re").sub_with([&](const Match & m) { return dotted(m, whole); }, whole);
    }
    {
        const Str whole = text;
        text = RX(R"re(\b([A-Za-z]{1,6})\b(\.?)(?![A-Za-z'’]))re").sub_with([&](const Match & m) { return abbreviation(m, whole); }, whole);
    }
    if (RX(R"re(\d)re").search(text)) {
        text = RX(R"re((?<![\w'.])(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]+(?![\w']|\.\d))re").sub_with(
            [this](const Match & m) { return glued(m); }, text);
        const Str whole = text;
        text = RX(R"re((?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|(?<![\w.])\.\d+)re").sub_with(
            [&](const Match & m) { return number(m, whole); }, whole);
    }
    text = RX(R"re((?<=[A-Za-z])/(?=[A-Za-z]))re").sub(U" ", text);
    text = RX(R"re((?<=\d))re" DZ_IS R"re(?\s?[<>]\s?(?=\d)|(?<=\s)[<>](?=\s))re").sub_with(
        [](const Match & m) { return compared(m, py::contains(m.group(0), U'<') ? U"less than" : U"greater than"); }, text);
    text = RX(R"re([\&@\+=%×÷±©§°≥≤≠≈∞√π])re").sub_with(
        [](const Match & m) {
            static const std::pair<char32_t, const char32_t *> words[] = {
                {U'&', U"and"}, {U'@', U"at"}, {U'+', U"plus"}, {U'=', U"equals"}, {U'%', U"percent"},
                {0xD7, U"times"}, {0xF7, U"divided by"}, {0xB1, U"plus or minus"}, {0xA9, U"copyright"}, {0xA7, U"section"},
                {0xB0, U"degrees"}, {0x2265, U"greater than or equal to"}, {0x2264, U"less than or equal to"},
                {0x2260, U"not equal to"}, {0x2248, U"approximately"}, {0x221E, U"infinity"}, {0x221A, U"square root of"},
                {0x3C0, U"pi"},
            };
            const char32_t sign = m.group(0)[0];
            for (const auto & entry : words) {
                if (entry.first == sign) {
                    if (sign == 0x2265 || sign == 0x2264 || sign == 0x2260) {
                        return compared(m, entry.second);
                    }
                    return U" " + Str(entry.second) + U" ";
                }
            }
            throw KeyError(encode(m.group(0)));
        },
        text);
    return py::replace(text, Str(1, kIs), U"");
}

// ---- words ---------------------------------------------------------------------

// Tokens for one row of the chat table: words, letters to name, or nothing.
std::vector<Token> Normalizer::chat_words(const Str & chat, const Str & raw) const {
    std::vector<Token> out;
    if (chat == U"-") {
        return out;
    }
    for (const Str & part : py::split(chat)) {
        if (py::startswith(part, U"*")) {
            out.push_back(Token::word(py::lower(StrView(part).substr(1)), raw, true));
        } else {
            out.push_back(Token::word(py::lower(part), part));
        }
    }
    return out;
}

// Tokens for one run of letters and apostrophes as written. `previous` is the
// word before it, `following` whether a word follows directly, `fresh` whether
// a new sentence starts here.
std::vector<Token> Normalizer::word_tokens(const Str & raw, bool all_caps, const Str & previous, bool following, bool fresh) const {
    // Capitals read letter by letter although they also spell a word.
    static const std::unordered_set<Str> initialisms = word_set(U"US IT AI ID UN EU UK LA AD BC AM PM OS PE PA MA LED SAT RIP ER");
    static const std::unordered_set<Str> cardinal_after = word_set(
        U"war chapter part book volume act scene section phase type class level stage title article appendix world");
    static const std::unordered_set<Str> labelled = word_set(
        U"plan vitamin grade type option class section part team group exhibit appendix point model series hepatitis level");
    static const std::unordered_map<Str, long long> roman = {
        {U"II", 2}, {U"III", 3}, {U"IV", 4}, {U"VI", 6}, {U"VII", 7}, {U"VIII", 8}, {U"IX", 9}, {U"XI", 11}, {U"XII", 12},
        {U"XIII", 13}, {U"XIV", 14}, {U"XV", 15}, {U"XVI", 16}, {U"XVIII", 18}, {U"XX", 20},
    };
    const Str lower = py::lower(raw);
    const auto chat = chat_.find(lower);
    if (chat != chat_.end() && (raw.size() > 1 || py::islower(raw)) && replies_.count(lower) == 0) {
        return chat_words(chat->second, raw);
    }
    const auto numeral = roman.find(raw);
    if (numeral != roman.end() && !previous.empty() && uni::is_upper(previous[0])) {
        const Str words = cardinal_after.count(py::lower(previous)) != 0 ? cardinal(numeral->second) : U"the " + ordinal(numeral->second);
        std::vector<Token> out;
        for (const Str & word : py::split(words)) {
            out.push_back(Token::word(word, raw));
        }
        return out;
    }
    const Str letters = py::replace(raw, U"'", U"");
    const bool possessive = py::endswith(lower, U"'s");
    const Str stem = possessive ? py::slice(raw, 0, -2) : raw;
    if (stem.size() == 1 && py::isalpha(stem)) {
        // "A" is the article unless it labels something: "Plan A.", "vitamin A"
        const bool label = stem == U"A" && !previous.empty() && (!following || (!fresh && labelled.count(py::lower(previous)) != 0));
        if (one_of(StrView(stem), {U"a", U"A", U"I", U"i"}) && !possessive && !label) {
            return {Token::word(py::lower(stem), raw)};
        }
        return {Token::word(lower, raw, true)};
    }
    bool one_letter = !lower.empty();
    for (const char32_t c : lower) {
        one_letter = one_letter && c == lower[0];
    }
    if (stem.size() > 2 && one_of(StrView(stem).substr(0, 1), {U"S", U"s", U"Z", U"z"}) && one_letter && (all_caps || !py::isupper(stem))) {
        Token token = Token::word(lower, raw);                          // "Sss", "Zzzz": a hiss, a snore
        token.phones = Phones{Phone{encode(lower.substr(0, 1)), 0}};
        return {token};
    }
    if (py::isupper(stem) && py::isalpha(stem) && stem.size() >= 2 && stem.size() <= 5 && (!all_caps || !is_word_(py::lower(stem)))) {
        if (initialisms.count(stem) != 0 || !is_word_(py::lower(stem))) {
            return {Token::word(lower, raw, true)};
        }
    }
    if (!py::isalpha(letters)) {
        return {};
    }
    // iOS, macOS, eSIM: small letters, then capitals that are named. The dictionary's "ios" is another word.
    const bool capitals = static_cast<bool>(RX(R"re([a-z]+(?:[A-Z][a-z]+)*[A-Z]{2,}(?:'s)?)re").fullmatch(raw));
    if (capitals || !is_word_(lower)) {
        if (!capitals) {
            for (const Str & candidate : unstretched(lower)) {           // "soooo" -> "so", "goooood" -> "good"
                if (is_word_(candidate)) {
                    return {Token::word(candidate, raw)};
                }
            }
        }
        const std::vector<Str> parts = RX(R"re([A-Z]?[a-z']+|[A-Z]+(?![a-z]))re").findall(raw);
        if (parts.size() > 1 && py::join(U"", parts) == raw) {           // iPhone, YouTube, JavaScript
            std::vector<Token> out;
            for (const Str & part : parts) {
                const bool named = capitals && py::isupper(part) ? initialisms.count(part) != 0 : false;
                const bool spell = part.size() == 1 || named || (py::isupper(part) && !is_word_(py::lower(part)));
                if (capitals && !out.empty() && out.back().spell && (spell || part == U"'s")) {
                    out.back().text += py::lower(part);                  // i + OS
                    out.back().raw += part;
                } else {
                    out.push_back(Token::word(py::lower(part), part, spell, !out.empty()));
                }
            }
            return out;
        }
    }
    return {Token::word(lower, raw)};
}

std::vector<Token> Normalizer::tokens(const Str & input) {
    static const std::unordered_set<Str> leading_apostrophe = {U"'em", U"'tis", U"'twas", U"'til", U"'cause", U"'bout", U"'round", U"'n"};
    // Sounds written without a vowel. "Ss" among them, or alone, is a hiss ("Sh! Ss! Tsk!").
    static const std::unordered_set<Str> sounds = word_set(U"sh shh ss zz hm hmm mm tsk psst pfft hmph psh pssh");
    const Str text = rewrite(input);
    const std::vector<Str> runs = RX(R"re([A-Za-z]{2,})re").findall(text);
    size_t upper_runs = 0;
    for (const Str & run : runs) {
        upper_runs += py::isupper(run) ? 1 : 0;
    }
    const bool all_caps = runs.size() >= 3 && static_cast<double>(upper_runs) >= 0.6 * static_cast<double>(runs.size());
    std::vector<Token> out;
    static const pyre::Regex pattern(DZ_SPELL R"re(([A-Za-z ]+))re" DZ_END "|" DZ_INLINE "(.)" DZ_INLINE
                                     R"re(|('?[A-Za-z]+(?:'[A-Za-z]+)*)|(\d+)|(-)|(\S))re");
    bool attach = false;
    Str previous;
    bool fresh = true;                          // a sentence starts here: "a plan. A dog" is the article
    std::vector<size_t> short_places;           // where "Ss" or "Zz" stands apart: perhaps a hiss, a snore
    std::vector<size_t> replies;                // where a chat spelling stands that is read only as a reply
    const size_t size = text.size();
    for (const Match & match : pattern.finditer(text)) {
        const auto start = static_cast<size_t>(match.start());
        const auto end = static_cast<size_t>(match.end());
        const char32_t before = start != 0 ? text[start - 1] : U' ';
        if (!match.group(5).empty()) {          // a hyphen
            attach = !out.empty() && !out.back().punct && !uni::is_space(before) && end < size && uni::is_alnum(text[end]);
            continue;
        }
        std::vector<Token> fresh_tokens;
        if (match.has(1)) {
            const Str letters = match.group(1);
            fresh_tokens = {Token::word(py::lower(py::replace(letters, U" ", U"")), letters, true)};
        } else if (match.has(2)) {
            long index = static_cast<long>(match.group(2)[0]) - static_cast<long>(kInlineBase);
            if (index < 0) {
                index += static_cast<long>(inline_.size());
            }
            if (index < 0 || index >= static_cast<long>(inline_.size())) {
                throw IndexError("list index out of range");
            }
            const Inline & given = inline_[static_cast<size_t>(index)];
            Token token = Token::word(py::lower(given.written), given.written);
            token.phones = given.phones;
            fresh_tokens = {token};
        } else if (match.has(3)) {
            Str word = match.group(3);
            if (py::startswith(word, U"'") && leading_apostrophe.count(py::lower(word)) == 0) {
                out.push_back(Token::punctuation(U"'"));
                word = word.substr(1);
            } else if (py::startswith(word, U"'") && !is_word_(py::lower(word))) {
                word = word.substr(1);
            }
            const StrView rest = StrView(text).substr(end);
            fresh_tokens = word_tokens(word, all_caps, previous, static_cast<bool>(RX(R"re(\s*[A-Za-z])re").match(rest)), fresh);
            previous = word;
            const bool apart = uni::is_space(before) || before == U'"' || before == U'\'';
            if (one_of(StrView(word), {U"Ss", U"ss", U"Zz", U"zz"}) && fresh_tokens.size() == 1 && (apart || before == U'(') &&
                RX(R"re([.!?…,;:—\"')]*(?:\s|$))re").match(rest)) {
                short_places.push_back(out.size());
            }
            // a reply stands free: not "(k)", "/k/", "[:k]", and not before a colon or a mid-text "!" ("k: king", "k! = mk")
            if (replies_.count(py::lower(word)) != 0 && (word.size() > 1 || py::islower(word)) && fresh_tokens.size() == 1 && apart &&
                RX(R"re((?:[.?…,;—]*[\"']?(?:\s|$)|!+[\"']?\s*$))re").match(rest)) {
                replies.push_back(out.size());
            }
        } else if (match.has(4)) {
            const Str number_text = match.group(4);
            for (const Str & part : py::split(number_words(number_text))) {
                fresh_tokens.push_back(Token::word(part, number_text));
            }
        } else if (match.has(6) && is_punctuation(encode(match.group(6)))) {
            const Str other = match.group(6);
            if (!(other == U"'" && uni::is_alnum(before) && end < size && uni::is_alnum(text[end]))) {
                out.push_back(Token::punctuation(other));
            }
            if (one_of(StrView(other), {U".", U"!", U"?", U";", U"\U00002026"})) {
                fresh = true;
            }
            attach = false;
            continue;
        }
        if (!fresh_tokens.empty() && attach) {
            fresh_tokens[0].attach = true;
        }
        const bool none = fresh_tokens.empty();
        out.insert(out.end(), fresh_tokens.begin(), fresh_tokens.end());
        attach = false;
        fresh = fresh && none;
    }
    // "Ss!" and "Sh, ss, tsk" are a hiss. Beside an ordinary word the two letters
    // stay letters, as in "the ss", "mm:ss" and a variable named ss. Quotation
    // marks and brackets do not part them from that word: "written as 'ss'".
    for (const size_t place : short_places) {
        size_t first = place;
        size_t last = place + 1;
        while (first != 0 && (!out[first - 1].punct || is_mark_of(out[first - 1], U"\"'()"))) {
            --first;
        }
        while (last < out.size() && (!out[last].punct || is_mark_of(out[last], U"\"'()"))) {
            ++last;
        }
        bool all_sounds = true;
        for (size_t k = first; k < last; ++k) {
            const Token & token = out[k];
            if (token.punct) {
                continue;
            }
            bool one_letter = !token.text.empty();
            for (const char32_t c : token.text) {
                one_letter = one_letter && c == token.text[0];
            }
            all_sounds = all_sounds && (sounds.count(token.text) != 0 || (token.phones.has_value() && one_letter));
        }
        if (all_sounds) {
            out[place].phones = Phones{Phone{encode(out[place].text.substr(0, 1)), 0}};
        }
    }
    // "k" is "okay" as a reply by itself; beside a word or a figure it is the letter.
    for (size_t k = replies.size(); k-- > 0;) {
        const size_t place = replies[k];
        if (reply(out, place)) {
            const Str raw = out[place].raw;
            const std::vector<Token> said = chat_words(chat_.at(py::lower(raw)), raw);
            out.erase(out.begin() + static_cast<std::ptrdiff_t>(place));
            out.insert(out.begin() + static_cast<std::ptrdiff_t>(place), said.begin(), said.end());
        }
    }
    return out;
}

}  // namespace dizione::normalize
