#include "pyre.h"

#include <algorithm>
#include <mutex>

namespace dizione::pyre {

namespace {

// ---- character sets ---------------------------------------------------------

enum ClassBit : unsigned {
    kDigit = 1,
    kNotDigit = 2,
    kWord = 4,
    kNotWord = 8,
    kSpace = 16,
    kNotSpace = 32,
};

bool ascii_space(char32_t c) {
    return c == U' ' || (c >= 0x09 && c <= 0x0D);
}

bool ascii_word(char32_t c) {
    return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z') || (c >= U'0' && c <= U'9') || c == U'_';
}

char32_t ascii_lower(char32_t c) {
    return c >= U'A' && c <= U'Z' ? c + 32 : c;
}

bool is_word_char(char32_t c, bool ascii) {
    return ascii ? ascii_word(c) : uni::is_word(c);
}

struct CharSet {
    bool negate = false;
    bool fold = false;          // compare the lower-case form of the character
    bool ascii = false;         // the ASCII flag: classes and case stop at 127
    uint64_t low[2] = {0, 0};   // members below 128
    std::vector<std::pair<char32_t, char32_t>> ranges;      // the other members
    unsigned classes = 0;

    void add(char32_t c) {
        if (c < 128) {
            low[c >> 6] |= uint64_t{1} << (c & 63);
        } else {
            ranges.emplace_back(c, c);
        }
    }

    void add(char32_t first, char32_t last) {
        for (char32_t c = first; c <= last && c < 128; ++c) {
            low[c >> 6] |= uint64_t{1} << (c & 63);
        }
        if (last >= 128) {
            ranges.emplace_back(std::max<char32_t>(first, 128), last);
        }
    }

    void finish() {
        std::sort(ranges.begin(), ranges.end());
        std::vector<std::pair<char32_t, char32_t>> merged;
        for (const auto & range : ranges) {
            if (!merged.empty() && range.first <= merged.back().second + 1) {
                merged.back().second = std::max(merged.back().second, range.second);
            } else {
                merged.push_back(range);
            }
        }
        ranges = std::move(merged);
    }

    bool member(char32_t c) const {
        if (c < 128) {
            if ((low[c >> 6] >> (c & 63)) & 1) {
                return true;
            }
        } else if (!ranges.empty()) {
            size_t first = 0;
            size_t last = ranges.size();
            while (first < last) {
                const size_t middle = (first + last) / 2;
                if (c > ranges[middle].second) {
                    first = middle + 1;
                } else if (c < ranges[middle].first) {
                    last = middle;
                } else {
                    return true;
                }
            }
        }
        if (classes == 0) {
            return false;
        }
        if (classes & (kDigit | kNotDigit)) {
            const bool digit = ascii ? (c >= U'0' && c <= U'9') : uni::is_decimal(c);
            if (digit ? (classes & kDigit) : (classes & kNotDigit)) {
                return true;
            }
        }
        if (classes & (kWord | kNotWord)) {
            const bool word = is_word_char(c, ascii);
            if (word ? (classes & kWord) : (classes & kNotWord)) {
                return true;
            }
        }
        if (classes & (kSpace | kNotSpace)) {
            const bool space = ascii ? ascii_space(c) : uni::is_space(c);
            if (space ? (classes & kSpace) : (classes & kNotSpace)) {
                return true;
            }
        }
        return false;
    }

    bool test(char32_t c) const {
        if (fold) {
            c = ascii ? ascii_lower(c) : uni::simple_lower(c);
        }
        return member(c) != negate;
    }
};

// ---- the compiled program ---------------------------------------------------

enum Op : uint8_t {
    OpChar,             // x: the character
    OpCharFold,         // x: lower-case character; the text is folded before comparing
    OpCharFoldAscii,
    OpAny,              // any character but a newline
    OpSet,              // x: set number
    OpSplit,            // try x, then y
    OpJump,             // x
    OpSave,             // x: slot
    OpBegin,            // ^ and \A
    OpEndLine,          // $
    OpEnd,              // \Z
    OpWordBoundary,     // x: 1 for the ASCII flag
    OpNotWordBoundary,
    OpBackRef,          // x: group
    OpBackRefFold,      // x: group, y: 1 for the ASCII flag
    OpAssert,           // x: bit 0 negative, bit 1 behind; y: where to go on; z: width
    OpAssertEnd,
    OpCond,             // x: group, y: where the "no" branch starts
    OpSetReg,           // x: register, set to -1
    OpMarkReg,          // x: register, set to the position
    OpLoopCheck,        // x: register, y: where to go when nothing was consumed
    OpRepeatOne,        // the next instruction (one character) x to y times, as many as possible first; y -1: no limit
    OpMatch,
};

struct Inst {
    Op op;
    int32_t x = 0;
    int32_t y = 0;
    int32_t z = 0;
};

// The characters a branch can start with. A split whose first branch cannot
// start with the next character goes straight to its second one.
struct Guard {
    uint64_t low[2] = {0, 0};
    bool other = false;

    bool allows(char32_t c) const {
        return c < 128 ? ((low[c >> 6] >> (c & 63)) & 1) != 0 : other;
    }
};

}  // namespace

struct Program {
    std::string pattern;
    int flags = 0;
    std::vector<Inst> code;
    std::vector<CharSet> sets;
    std::vector<Guard> guards;
    int groups = 0;
    int registers = 0;
    std::vector<std::pair<std::string, int>> names;
    bool begins_at_start = false;       // the pattern can only match at position 0
    bool has_first = false;             // the characters a match can start with are known
    uint64_t first_low[2] = {0, 0};
    bool first_other = false;           // a match can start with a character above 127

    int group_index(std::string_view name) const {
        for (const auto & entry : names) {
            if (entry.first == name) {
                return entry.second;
            }
        }
        return -1;
    }
};

namespace {

// ---- parsing ---------------------------------------------------------------

struct Node;
using NodePtr = std::unique_ptr<Node>;

struct Node {
    enum Kind {
        Empty, Char, CharFold, CharFoldAscii, Any, Set, Cat, Alt, Group, Repeat,
        Begin, EndLine, End, WordBoundary, NotWordBoundary, BackRef, Look, Cond,
    };
    Kind kind = Empty;
    char32_t ch = 0;
    int index = -1;         // Set: set number; Group: group number or -1; BackRef, Cond: group number
    long min = 0;
    long max = 0;           // -1: no upper bound
    bool lazy = false;
    bool behind = false;
    bool negative = false;
    bool fold = false;
    bool ascii = false;
    std::vector<NodePtr> kids;
};

NodePtr make(Node::Kind kind) {
    auto node = std::make_unique<Node>();
    node->kind = kind;
    return node;
}

bool is_digit(char32_t c) { return c >= U'0' && c <= U'9'; }
bool is_octal(char32_t c) { return c >= U'0' && c <= U'7'; }
bool is_ascii_letter(char32_t c) { return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z'); }

int hex_value(char32_t c) {
    if (c >= U'0' && c <= U'9') return static_cast<int>(c - U'0');
    if (c >= U'a' && c <= U'f') return static_cast<int>(c - U'a') + 10;
    if (c >= U'A' && c <= U'F') return static_cast<int>(c - U'A') + 10;
    return -1;
}

class Parser {
public:
    Parser(Str pattern, int flags, Program & program)
        : p_(std::move(pattern)), program_(program), fold_((flags & IGNORECASE) != 0), ascii_((flags & ASCII) != 0) {}

    NodePtr parse() {
        NodePtr root = alternation();
        if (i_ < p_.size()) {
            fail(p_[i_] == U')' ? "unbalanced parenthesis" : "unexpected character");
        }
        for (const auto & pending : conditions_) {
            const int group = pending.second.empty() ? pending.first->index : name_index(pending.second);
            if (group <= 0 || group > program_.groups) {
                fail("unknown group in a conditional");
            }
            pending.first->index = group;
        }
        return root;
    }

private:
    [[noreturn]] void fail(const std::string & what) const {
        throw PatternError(what + " at position " + std::to_string(i_) + " of pattern " + program_.pattern);
    }

    bool more() const { return i_ < p_.size(); }
    char32_t peek(size_t ahead = 0) const { return i_ + ahead < p_.size() ? p_[i_ + ahead] : 0; }

    bool eat(char32_t c) {
        if (more() && p_[i_] == c) {
            ++i_;
            return true;
        }
        return false;
    }

    int name_index(const Str & name) const {
        return program_.group_index(encode(name));
    }

    NodePtr alternation() {
        std::vector<NodePtr> branches;
        branches.push_back(sequence());
        while (eat(U'|')) {
            branches.push_back(sequence());
        }
        if (branches.size() == 1) {
            return std::move(branches[0]);
        }
        NodePtr node = make(Node::Alt);
        node->kids = std::move(branches);
        return node;
    }

    NodePtr sequence() {
        NodePtr node = make(Node::Cat);
        while (more() && peek() != U'|' && peek() != U')') {
            NodePtr item = atom();
            if (item) {
                node->kids.push_back(quantified(std::move(item)));
            }
        }
        if (node->kids.size() == 1) {
            return std::move(node->kids[0]);
        }
        if (node->kids.empty()) {
            return make(Node::Empty);
        }
        return node;
    }

    NodePtr quantified(NodePtr item) {
        if (!more()) {
            return item;
        }
        long min = 0;
        long max = -1;
        const char32_t c = peek();
        if (c == U'*') {
            ++i_;
        } else if (c == U'+') {
            min = 1;
            ++i_;
        } else if (c == U'?') {
            max = 1;
            ++i_;
        } else if (c == U'{') {
            // {n}, {n,}, {,m}, {n,m}; anything else is a plain brace
            size_t k = i_ + 1;
            if (k < p_.size() && p_[k] == U'}') {
                return item;
            }
            Str low, high;
            while (k < p_.size() && is_digit(p_[k])) low.push_back(p_[k++]);
            if (k < p_.size() && p_[k] == U',') {
                ++k;
                while (k < p_.size() && is_digit(p_[k])) high.push_back(p_[k++]);
            } else {
                high = low;
            }
            if (k >= p_.size() || p_[k] != U'}') {
                return item;
            }
            i_ = k + 1;
            min = low.empty() ? 0 : static_cast<long>(py::to_int(low));
            max = high.empty() ? -1 : static_cast<long>(py::to_int(high));
            if (max != -1 && max < min) {
                fail("min repeat greater than max repeat");
            }
        } else {
            return item;
        }
        NodePtr node = make(Node::Repeat);
        node->min = min;
        node->max = max;
        if (eat(U'?')) {
            node->lazy = true;
        } else if (peek() == U'+') {
            fail("possessive quantifiers are not supported");
        }
        if (more() && (peek() == U'*' || peek() == U'+' || peek() == U'?')) {
            fail("multiple repeat");
        }
        node->kids.push_back(std::move(item));
        return node;
    }

    NodePtr literal(char32_t c) {
        if (!fold_) {
            NodePtr node = make(Node::Char);
            node->ch = c;
            return node;
        }
        if (ascii_) {
            NodePtr node = make(is_ascii_letter(c) ? Node::CharFoldAscii : Node::Char);
            node->ch = is_ascii_letter(c) ? ascii_lower(c) : c;
            return node;
        }
        if (!uni::sre_cased(c)) {
            NodePtr node = make(Node::Char);
            node->ch = c;
            return node;
        }
        const char32_t lower = uni::simple_lower(c);
        size_t count = 0;
        const uint32_t * extra = uni::extra_cases(lower, &count);
        if (extra == nullptr) {
            NodePtr node = make(Node::CharFold);
            node->ch = lower;
            return node;
        }
        CharSet set;
        set.fold = true;
        set.add(lower);
        for (size_t k = 0; k < count; ++k) {
            set.add(extra[k]);
        }
        return set_node(std::move(set));
    }

    NodePtr set_node(CharSet set) {
        set.finish();
        NodePtr node = make(Node::Set);
        node->index = static_cast<int>(program_.sets.size());
        program_.sets.push_back(std::move(set));
        return node;
    }

    NodePtr class_node(unsigned bits) {
        CharSet set;
        set.classes = bits;
        set.ascii = ascii_;
        return set_node(std::move(set));
    }

    // \xNN, \uNNNN, \UNNNNNNNN after the letter has been read.
    char32_t hex_escape(int digits) {
        char32_t value = 0;
        for (int k = 0; k < digits; ++k) {
            const int digit = more() ? hex_value(peek()) : -1;
            if (digit < 0) {
                fail("incomplete escape");
            }
            value = value * 16 + static_cast<char32_t>(digit);
            ++i_;
        }
        if (value > 0x10FFFF) {
            fail("bad escape");
        }
        return value;
    }

    // An escape that stands for one character, inside or outside a set. Returns
    // false when `c` (the character after the backslash) is not one.
    bool simple_escape(char32_t c, bool in_set, char32_t * out) {
        switch (c) {
            case U'a': *out = 0x07; return true;
            case U'f': *out = 0x0C; return true;
            case U'n': *out = U'\n'; return true;
            case U'r': *out = U'\r'; return true;
            case U't': *out = U'\t'; return true;
            case U'v': *out = 0x0B; return true;
            case U'\\': *out = U'\\'; return true;
            case U'b':
                if (in_set) {
                    *out = 0x08;
                    return true;
                }
                return false;
            case U'x': *out = hex_escape(2); return true;
            case U'u': *out = hex_escape(4); return true;
            case U'U': *out = hex_escape(8); return true;
            default: return false;
        }
    }

    static unsigned class_bits(char32_t c) {
        switch (c) {
            case U'd': return kDigit;
            case U'D': return kNotDigit;
            case U'w': return kWord;
            case U'W': return kNotWord;
            case U's': return kSpace;
            case U'S': return kNotSpace;
            default: return 0;
        }
    }

    NodePtr escape() {
        if (!more()) {
            fail("bad escape (end of pattern)");
        }
        const char32_t c = p_[i_++];
        if (const unsigned bits = class_bits(c)) {
            return class_node(bits);
        }
        if (c == U'A') return make(Node::Begin);
        if (c == U'Z') return make(Node::End);
        if (c == U'b' || c == U'B') {
            NodePtr node = make(c == U'b' ? Node::WordBoundary : Node::NotWordBoundary);
            node->ascii = ascii_;
            return node;
        }
        char32_t value = 0;
        if (simple_escape(c, false, &value)) {
            return literal(value);
        }
        if (c == U'0') {
            value = 0;
            for (int k = 0; k < 2 && more() && is_octal(peek()); ++k) {
                value = value * 8 + (p_[i_++] - U'0');
            }
            return literal(value);
        }
        if (is_digit(c)) {
            long group = static_cast<long>(c - U'0');
            if (more() && is_digit(peek())) {
                if (is_octal(c) && is_octal(peek()) && is_octal(peek(1))) {
                    fail("octal escapes are not supported");
                }
                group = group * 10 + static_cast<long>(p_[i_++] - U'0');
            }
            if (group > program_.groups) {
                fail("invalid group reference");
            }
            NodePtr node = make(Node::BackRef);
            node->index = static_cast<int>(group);
            node->fold = fold_;
            node->ascii = ascii_;
            return node;
        }
        if (is_ascii_letter(c)) {
            fail("bad escape");
        }
        return literal(c);
    }

    NodePtr set() {
        CharSet set;
        set.ascii = ascii_;
        std::vector<char32_t> singles;
        std::vector<std::pair<char32_t, char32_t>> spans;
        set.negate = eat(U'^');
        bool any = false;
        for (;;) {
            if (!more()) {
                fail("unterminated character set");
            }
            char32_t c = p_[i_++];
            if (c == U']' && any) {
                break;
            }
            any = true;
            bool is_class = false;
            char32_t first = c;
            if (c == U'\\') {
                if (!more()) {
                    fail("bad escape (end of pattern)");
                }
                const char32_t e = p_[i_++];
                if (const unsigned bits = class_bits(e)) {
                    set.classes |= bits;
                    is_class = true;
                } else if (!simple_escape(e, true, &first)) {
                    if (is_octal(e)) {
                        first = e - U'0';
                        for (int k = 0; k < 2 && more() && is_octal(peek()); ++k) {
                            first = first * 8 + (p_[i_++] - U'0');
                        }
                    } else if (is_ascii_letter(e) || is_digit(e)) {
                        fail("bad escape in a character set");
                    } else {
                        first = e;
                    }
                }
            }
            if (peek() == U'-') {
                ++i_;
                if (!more()) {
                    fail("unterminated character set");
                }
                if (peek() == U']') {
                    ++i_;
                    if (!is_class) {
                        singles.push_back(first);
                    }
                    singles.push_back(U'-');
                    break;
                }
                char32_t last = p_[i_++];
                bool last_class = false;
                if (last == U'\\') {
                    if (!more()) {
                        fail("bad escape (end of pattern)");
                    }
                    const char32_t e = p_[i_++];
                    if (class_bits(e)) {
                        last_class = true;
                    } else if (!simple_escape(e, true, &last)) {
                        if (is_ascii_letter(e) || is_digit(e)) {
                            fail("bad escape in a character set");
                        }
                        last = e;
                    }
                }
                if (is_class || last_class || last < first) {
                    fail("bad character range");
                }
                spans.emplace_back(first, last);
                continue;
            }
            if (!is_class) {
                singles.push_back(first);
            }
        }
        bool fold = false;
        if (fold_) {
            // As the re module does: the set holds lower-case members and the text
            // is folded before the test, but only when a member has a case at all.
            for (const char32_t c : singles) {
                fold = fold || (ascii_ ? is_ascii_letter(c) : uni::sre_cased(c));
            }
            for (const auto & span : spans) {
                for (char32_t c = span.first; c <= span.second && !fold; ++c) {
                    fold = ascii_ ? is_ascii_letter(c) : uni::sre_cased(c);
                }
            }
        }
        if (!fold) {
            for (const char32_t c : singles) set.add(c);
            for (const auto & span : spans) set.add(span.first, span.second);
            return set_node(std::move(set));
        }
        set.fold = true;
        auto add_folded = [&](char32_t c) {
            if (ascii_) {
                set.add(ascii_lower(c));
                return;
            }
            const char32_t lower = uni::simple_lower(c);
            set.add(lower);
            size_t count = 0;
            if (const uint32_t * extra = uni::extra_cases(lower, &count)) {
                for (size_t k = 0; k < count; ++k) {
                    set.add(extra[k]);
                }
            }
        };
        for (const char32_t c : singles) add_folded(c);
        for (const auto & span : spans) {
            for (char32_t c = span.first; c <= span.second; ++c) add_folded(c);
        }
        return set_node(std::move(set));
    }

    Str until(char32_t stop) {
        Str out;
        while (more() && peek() != stop) {
            out.push_back(p_[i_++]);
        }
        if (!eat(stop)) {
            fail("missing delimiter");
        }
        return out;
    }

    NodePtr close_group(NodePtr inner, int capture) {
        if (!eat(U')')) {
            fail("missing ), unterminated subpattern");
        }
        NodePtr node = make(Node::Group);
        node->index = capture;
        node->kids.push_back(std::move(inner));
        return node;
    }

    NodePtr group() {
        if (!eat(U'?')) {
            const int capture = ++program_.groups;
            return close_group(alternation(), capture);
        }
        if (!more()) {
            fail("unexpected end of pattern");
        }
        const char32_t c = p_[i_++];
        if (c == U':') {
            return close_group(alternation(), -1);
        }
        if (c == U'#') {
            until(U')');
            return nullptr;
        }
        if (c == U'P') {
            if (eat(U'<')) {
                const Str name = until(U'>');
                const int capture = ++program_.groups;
                program_.names.emplace_back(encode(name), capture);
                return close_group(alternation(), capture);
            }
            if (eat(U'=')) {
                const Str name = until(U')');
                const int group = name_index(name);
                if (group < 0) {
                    fail("unknown group name");
                }
                NodePtr node = make(Node::BackRef);
                node->index = group;
                node->fold = fold_;
                node->ascii = ascii_;
                return node;
            }
            fail("unknown extension ?P");
        }
        if (c == U'=' || c == U'!' || c == U'<') {
            NodePtr node = make(Node::Look);
            char32_t kind = c;
            if (c == U'<') {
                if (!more() || (peek() != U'=' && peek() != U'!')) {
                    fail("unknown extension ?<");
                }
                node->behind = true;
                kind = p_[i_++];
            }
            node->negative = kind == U'!';
            node->kids.push_back(alternation());
            if (!eat(U')')) {
                fail("missing ), unterminated subpattern");
            }
            return node;
        }
        if (c == U'(') {
            const Str name = until(U')');
            NodePtr node = make(Node::Cond);
            bool number = !name.empty();
            for (const char32_t d : name) {
                number = number && is_digit(d);
            }
            if (number) {
                node->index = static_cast<int>(py::to_int(name));
                conditions_.emplace_back(node.get(), Str());
            } else {
                conditions_.emplace_back(node.get(), name);
            }
            node->kids.push_back(sequence());
            node->kids.push_back(eat(U'|') ? sequence() : make(Node::Empty));
            if (peek() == U'|') {
                fail("conditional backref with more than two branches");
            }
            if (!eat(U')')) {
                fail("missing ), unterminated subpattern");
            }
            return node;
        }
        // flags: (?i), (?a), (?-i:...), (?i:...)
        --i_;
        bool on = true;
        bool fold = fold_;
        bool ascii = ascii_;
        for (;;) {
            if (!more()) {
                fail("missing ), unterminated subpattern");
            }
            const char32_t f = p_[i_++];
            if (f == U'-') {
                on = false;
            } else if (f == U'i') {
                fold = on;
            } else if (f == U'a') {
                ascii = on;
            } else if (f == U'u') {
                ascii = !on;
            } else if (f == U')') {
                fold_ = fold;
                ascii_ = ascii;
                return nullptr;
            } else if (f == U':') {
                const bool old_fold = fold_;
                const bool old_ascii = ascii_;
                fold_ = fold;
                ascii_ = ascii;
                NodePtr node = close_group(alternation(), -1);
                fold_ = old_fold;
                ascii_ = old_ascii;
                return node;
            } else {
                fail("unsupported flag or extension");
            }
        }
    }

    NodePtr atom() {
        const char32_t c = p_[i_++];
        switch (c) {
            case U'(': return group();
            case U'[': return set();
            case U'.': return make(Node::Any);
            case U'^': return make(Node::Begin);
            case U'$': return make(Node::EndLine);
            case U'\\': return escape();
            case U'*':
            case U'+':
            case U'?':
                fail("nothing to repeat");
            default: return literal(c);
        }
    }

    Str p_;
    size_t i_ = 0;
    Program & program_;
    bool fold_;
    bool ascii_;
    std::vector<std::pair<Node *, Str>> conditions_;
};

// ---- analysis -------------------------------------------------------------

bool nullable(const Node & node) {
    switch (node.kind) {
        case Node::Char:
        case Node::CharFold:
        case Node::CharFoldAscii:
        case Node::Any:
        case Node::Set:
            return false;
        case Node::Cat:
            for (const auto & kid : node.kids) {
                if (!nullable(*kid)) return false;
            }
            return true;
        case Node::Alt:
        case Node::Cond:
            for (const auto & kid : node.kids) {
                if (nullable(*kid)) return true;
            }
            return false;
        case Node::Group:
            return nullable(*node.kids[0]);
        case Node::Repeat:
            return node.min == 0 || nullable(*node.kids[0]);
        default:
            return true;
    }
}

// The number of characters a look-behind steps back. Python needs it fixed.
long width(const Node & node, const Program & program) {
    switch (node.kind) {
        case Node::Char:
        case Node::CharFold:
        case Node::CharFoldAscii:
        case Node::Any:
        case Node::Set:
            return 1;
        case Node::Cat: {
            long total = 0;
            for (const auto & kid : node.kids) total += width(*kid, program);
            return total;
        }
        case Node::Alt:
        case Node::Cond: {
            const long first = width(*node.kids[0], program);
            for (const auto & kid : node.kids) {
                if (width(*kid, program) != first) {
                    throw PatternError("look-behind requires fixed-width pattern: " + program.pattern);
                }
            }
            return first;
        }
        case Node::Group:
            return width(*node.kids[0], program);
        case Node::Repeat:
            if (node.min != node.max) {
                throw PatternError("look-behind requires fixed-width pattern: " + program.pattern);
            }
            return node.min * width(*node.kids[0], program);
        case Node::BackRef:
            throw PatternError("look-behind requires fixed-width pattern: " + program.pattern);
        default:
            return 0;
    }
}

struct First {
    uint64_t low[2] = {0, 0};
    bool other = false;
    bool unknown = false;

    void add(char32_t c) {
        if (c < 128) {
            low[c >> 6] |= uint64_t{1} << (c & 63);
        } else {
            other = true;
        }
    }
};

// Collects the characters a match of `node` can start with. Returns true when
// the node can match without consuming anything.
bool first_of(const Node & node, const Program & program, First & first) {
    switch (node.kind) {
        case Node::Char:
            first.add(node.ch);
            return false;
        case Node::CharFold:
        case Node::CharFoldAscii:
            first.add(node.ch);
            if (node.ch >= U'a' && node.ch <= U'z') {
                first.add(node.ch - 32);
            }
            if (node.kind == Node::CharFold) {
                first.other = true;         // letters above 127 fold to ASCII ones (the Kelvin sign)
            }
            return false;
        case Node::Any:
            first.unknown = true;
            return false;
        case Node::Set: {
            const CharSet & set = program.sets[static_cast<size_t>(node.index)];
            for (char32_t c = 0; c < 128; ++c) {
                if (set.test(c)) first.add(c);
            }
            first.other = first.other || set.negate || set.fold || set.classes != 0 || !set.ranges.empty();
            return false;
        }
        case Node::Cat:
            for (const auto & kid : node.kids) {
                if (!first_of(*kid, program, first)) return false;
            }
            return true;
        case Node::Alt:
        case Node::Cond: {
            bool empty = false;
            for (const auto & kid : node.kids) {
                empty = first_of(*kid, program, first) || empty;
            }
            return empty;
        }
        case Node::Group:
            return first_of(*node.kids[0], program, first);
        case Node::Repeat: {
            const bool empty = first_of(*node.kids[0], program, first);
            return empty || node.min == 0;
        }
        case Node::BackRef:
            first.unknown = true;
            return true;
        default:
            return true;
    }
}

bool begins_at_start(const Node & node) {
    if (node.kind == Node::Begin) return true;
    if (node.kind == Node::Cat && !node.kids.empty()) return begins_at_start(*node.kids[0]);
    if (node.kind == Node::Group) return begins_at_start(*node.kids[0]);
    return false;
}

// ---- code generation --------------------------------------------------------

class Compiler {
public:
    explicit Compiler(Program & program) : program_(program) {}

    void compile(const Node & root) {
        emit(root);
        add(OpMatch);
    }

private:
    int add(Op op, int32_t x = 0, int32_t y = 0, int32_t z = 0) {
        program_.code.push_back(Inst{op, x, y, z});
        return static_cast<int>(program_.code.size()) - 1;
    }

    int here() const { return static_cast<int>(program_.code.size()); }

    // The number (from 1) of the guard for a branch, 0 when the branch can match
    // nothing at all or its first character is not known.
    int32_t guard(const Node & branch) {
        First first;
        if (first_of(branch, program_, first) || first.unknown) {
            return 0;
        }
        Guard made;
        made.low[0] = first.low[0];
        made.low[1] = first.low[1];
        made.other = first.other;
        program_.guards.push_back(made);
        return static_cast<int32_t>(program_.guards.size());
    }

    void emit(const Node & node) {
        switch (node.kind) {
            case Node::Empty:
                break;
            case Node::Char:
                add(OpChar, static_cast<int32_t>(node.ch));
                break;
            case Node::CharFold:
                add(OpCharFold, static_cast<int32_t>(node.ch));
                break;
            case Node::CharFoldAscii:
                add(OpCharFoldAscii, static_cast<int32_t>(node.ch));
                break;
            case Node::Any:
                add(OpAny);
                break;
            case Node::Set:
                add(OpSet, node.index);
                break;
            case Node::Cat:
                for (const auto & kid : node.kids) emit(*kid);
                break;
            case Node::Alt: {
                std::vector<int> exits;
                for (size_t k = 0; k < node.kids.size(); ++k) {
                    if (k + 1 == node.kids.size()) {
                        emit(*node.kids[k]);
                        break;
                    }
                    const int split = add(OpSplit, 0, 0, guard(*node.kids[k]));
                    program_.code[static_cast<size_t>(split)].x = here();
                    emit(*node.kids[k]);
                    exits.push_back(add(OpJump));
                    program_.code[static_cast<size_t>(split)].y = here();
                }
                for (const int jump : exits) {
                    program_.code[static_cast<size_t>(jump)].x = here();
                }
                break;
            }
            case Node::Group:
                if (node.index >= 0) add(OpSave, 2 * node.index);
                emit(*node.kids[0]);
                if (node.index >= 0) add(OpSave, 2 * node.index + 1);
                break;
            case Node::Repeat:
                repeat(node);
                break;
            case Node::Begin:
                add(OpBegin);
                break;
            case Node::EndLine:
                add(OpEndLine);
                break;
            case Node::End:
                add(OpEnd);
                break;
            case Node::WordBoundary:
                add(OpWordBoundary, node.ascii ? 1 : 0);
                break;
            case Node::NotWordBoundary:
                add(OpNotWordBoundary, node.ascii ? 1 : 0);
                break;
            case Node::BackRef:
                add(node.fold ? OpBackRefFold : OpBackRef, node.index, node.ascii ? 1 : 0);
                break;
            case Node::Look: {
                const long back = node.behind ? width(*node.kids[0], program_) : 0;
                const int lookaround = add(OpAssert, (node.negative ? 1 : 0) | (node.behind ? 2 : 0), 0, static_cast<int32_t>(back));
                emit(*node.kids[0]);
                // Only an assertion that holds a group can change what the groups hold.
                for (int k = lookaround + 1; k < here(); ++k) {
                    if (program_.code[static_cast<size_t>(k)].op == OpSave) {
                        program_.code[static_cast<size_t>(lookaround)].x |= 4;
                        break;
                    }
                }
                add(OpAssertEnd);
                program_.code[static_cast<size_t>(lookaround)].y = here();
                break;
            }
            case Node::Cond: {
                const int cond = add(OpCond, node.index);
                emit(*node.kids[0]);
                const int jump = add(OpJump);
                program_.code[static_cast<size_t>(cond)].y = here();
                emit(*node.kids[1]);
                program_.code[static_cast<size_t>(jump)].x = here();
                break;
            }
        }
    }

    void repeat(const Node & node) {
        const Node & body = *node.kids[0];
        const bool one = body.kind == Node::Char || body.kind == Node::CharFold || body.kind == Node::CharFoldAscii ||
            body.kind == Node::Any || body.kind == Node::Set;
        if (one && !node.lazy) {
            // One character repeated: counted in a loop, and given back one at a
            // time, which is what the general form below does in more steps.
            add(OpRepeatOne, static_cast<int32_t>(node.min), static_cast<int32_t>(node.max));
            emit(body);
            return;
        }
        for (long k = 0; k < node.min; ++k) {
            emit(body);
        }
        if (node.max == -1) {
            // body* : loop, leaving when an iteration consumed nothing (as Python does)
            const bool guard = nullable(body);
            int reg = -1;
            if (guard) {
                reg = program_.registers++;
                add(OpSetReg, reg);
            }
            const int top = here();
            int check = -1;
            if (guard) {
                check = add(OpLoopCheck, reg);
            }
            const int fork = add(OpSplit, 0, 0, node.lazy ? 0 : this->guard(body));
            const int enter = here();
            if (guard) {
                add(OpMarkReg, reg);
            }
            emit(body);
            add(OpJump, top);
            const int leave = here();
            program_.code[static_cast<size_t>(fork)].x = node.lazy ? leave : enter;
            program_.code[static_cast<size_t>(fork)].y = node.lazy ? enter : leave;
            if (check >= 0) {
                program_.code[static_cast<size_t>(check)].y = leave;
            }
            return;
        }
        // up to (max - min) more: body? nested, each giving up to the common end
        std::vector<int> forks;
        for (long k = node.min; k < node.max; ++k) {
            forks.push_back(add(OpSplit, 0, 0, node.lazy ? 0 : this->guard(body)));
            const int enter = here();
            emit(body);
            program_.code[static_cast<size_t>(forks.back())].x = enter;
        }
        const int leave = here();
        for (const int fork : forks) {
            Inst & inst = program_.code[static_cast<size_t>(fork)];
            if (node.lazy) {
                inst.y = inst.x;
                inst.x = leave;
            } else {
                inst.y = leave;
            }
        }
    }

    Program & program_;
};

// ---- matching -----------------------------------------------------------------

enum FrameKind : int32_t { FrameBranch, FrameCapture, FrameRegister, FrameSnapshot, FrameRepeat };

struct Frame {
    int32_t kind;
    int32_t index;
    long value;
    long least = 0;         // FrameRepeat: the position it may not go back past
};

// Does the one-character instruction `inst` accept `c`?
inline bool accepts(const Program & program, const Inst & inst, char32_t c) {
    switch (inst.op) {
        case OpChar: return c == static_cast<char32_t>(inst.x);
        case OpCharFold: return uni::simple_lower(c) == static_cast<char32_t>(inst.x);
        case OpCharFoldAscii: return ascii_lower(c) == static_cast<char32_t>(inst.x);
        case OpAny: return c != U'\n';
        case OpSet: return program.sets[static_cast<size_t>(inst.x)].test(c);
        default: return false;
    }
}

struct State {
    StrView text;
    std::vector<long> caps;
    std::vector<long> regs;
    std::vector<Frame> stack;
    std::vector<std::vector<long>> snapshots;
    long steps = 0;
    long start = 0;
    bool must_advance = false;
    bool match_all = false;
};

bool boundary(const State & state, long position, bool ascii) {
    const long size = static_cast<long>(state.text.size());
    if (size == 0) {
        return false;
    }
    const bool before = position > 0 && is_word_char(state.text[static_cast<size_t>(position - 1)], ascii);
    const bool after = position < size && is_word_char(state.text[static_cast<size_t>(position)], ascii);
    return before != after;
}

// Runs the program from `pc` at `position`. Returns true at OpMatch (top
// level) or OpAssertEnd (inside an assertion); the frames above `base` are the
// caller's to drop. On failure everything above `base` has been undone.
bool run(const Program & program, State & state, int pc, long position, size_t base, bool top, long * end) {
    const StrView text = state.text;
    const long size = static_cast<long>(text.size());
    const Inst * code = program.code.data();
    for (;;) {
        if (++state.steps > kStepLimit) {
            throw StepLimit();
        }
        const Inst & inst = code[pc];
        bool ok = true;
        switch (inst.op) {
            case OpChar:
                if (position < size && text[static_cast<size_t>(position)] == static_cast<char32_t>(inst.x)) {
                    ++position;
                    ++pc;
                } else {
                    ok = false;
                }
                break;
            case OpCharFold:
                if (position < size && uni::simple_lower(text[static_cast<size_t>(position)]) == static_cast<char32_t>(inst.x)) {
                    ++position;
                    ++pc;
                } else {
                    ok = false;
                }
                break;
            case OpCharFoldAscii:
                if (position < size && ascii_lower(text[static_cast<size_t>(position)]) == static_cast<char32_t>(inst.x)) {
                    ++position;
                    ++pc;
                } else {
                    ok = false;
                }
                break;
            case OpAny:
                if (position < size && text[static_cast<size_t>(position)] != U'\n') {
                    ++position;
                    ++pc;
                } else {
                    ok = false;
                }
                break;
            case OpSet:
                if (position < size && program.sets[static_cast<size_t>(inst.x)].test(text[static_cast<size_t>(position)])) {
                    ++position;
                    ++pc;
                } else {
                    ok = false;
                }
                break;
            case OpSplit:
                if (inst.z != 0 && (position >= size || !program.guards[static_cast<size_t>(inst.z - 1)].allows(text[static_cast<size_t>(position)]))) {
                    pc = inst.y;
                    break;
                }
                state.stack.push_back(Frame{FrameBranch, inst.y, position});
                pc = inst.x;
                break;
            case OpJump:
                pc = inst.x;
                break;
            case OpSave:
                state.stack.push_back(Frame{FrameCapture, inst.x, state.caps[static_cast<size_t>(inst.x)]});
                state.caps[static_cast<size_t>(inst.x)] = position;
                ++pc;
                break;
            case OpBegin:
                ok = position == 0;
                ++pc;
                break;
            case OpEndLine:
                ok = position == size || (position == size - 1 && text[static_cast<size_t>(position)] == U'\n');
                ++pc;
                break;
            case OpEnd:
                ok = position == size;
                ++pc;
                break;
            case OpWordBoundary:
                ok = boundary(state, position, inst.x != 0);
                ++pc;
                break;
            case OpNotWordBoundary:
                ok = !boundary(state, position, inst.x != 0);
                ++pc;
                break;
            case OpBackRef:
            case OpBackRefFold: {
                const long from = state.caps[static_cast<size_t>(2 * inst.x)];
                const long to = state.caps[static_cast<size_t>(2 * inst.x + 1)];
                if (from < 0 || to < from || position + (to - from) > size) {
                    ok = false;
                    break;
                }
                for (long k = 0; k < to - from && ok; ++k) {
                    char32_t a = text[static_cast<size_t>(from + k)];
                    char32_t b = text[static_cast<size_t>(position + k)];
                    if (inst.op == OpBackRefFold) {
                        a = inst.y ? ascii_lower(a) : uni::simple_lower(a);
                        b = inst.y ? ascii_lower(b) : uni::simple_lower(b);
                    }
                    ok = a == b;
                }
                if (ok) {
                    position += to - from;
                    ++pc;
                }
                break;
            }
            case OpAssert: {
                const bool negative = (inst.x & 1) != 0;
                const bool behind = (inst.x & 2) != 0;
                bool found = false;
                if (!behind || position >= inst.z) {
                    const size_t mark = state.stack.size();
                    const bool captures = (inst.x & 4) != 0;
                    std::vector<long> saved;
                    if (captures) {
                        saved = state.caps;
                    }
                    long unused = 0;
                    found = run(program, state, pc + 1, behind ? position - inst.z : position, mark, false, &unused);
                    // An assertion is not entered again: what it left to try is dropped.
                    state.stack.resize(mark);
                    if (captures && found) {
                        if (negative) {
                            state.caps = std::move(saved);
                        } else if (state.caps != saved) {
                            state.snapshots.push_back(std::move(saved));
                            state.stack.push_back(Frame{FrameSnapshot, static_cast<int32_t>(state.snapshots.size()) - 1, 0});
                        }
                    }
                }
                ok = found != negative;
                pc = inst.y;
                break;
            }
            case OpAssertEnd:
                *end = position;
                return true;
            case OpCond: {
                const long from = state.caps[static_cast<size_t>(2 * inst.x)];
                const long to = state.caps[static_cast<size_t>(2 * inst.x + 1)];
                pc = (from >= 0 && to >= from) ? pc + 1 : inst.y;
                break;
            }
            case OpSetReg:
                state.stack.push_back(Frame{FrameRegister, inst.x, state.regs[static_cast<size_t>(inst.x)]});
                state.regs[static_cast<size_t>(inst.x)] = -1;
                ++pc;
                break;
            case OpMarkReg:
                state.stack.push_back(Frame{FrameRegister, inst.x, state.regs[static_cast<size_t>(inst.x)]});
                state.regs[static_cast<size_t>(inst.x)] = position;
                ++pc;
                break;
            case OpLoopCheck:
                pc = state.regs[static_cast<size_t>(inst.x)] == position ? inst.y : pc + 1;
                break;
            case OpRepeatOne: {
                const Inst & atom = code[pc + 1];
                const long first = position;
                const long most = inst.y < 0 ? size : std::min(size, first + inst.y);
                while (position < most && accepts(program, atom, text[static_cast<size_t>(position)])) {
                    ++position;
                }
                if (position - first < inst.x) {
                    ok = false;
                    break;
                }
                if (position - first > inst.x) {
                    state.stack.push_back(Frame{FrameRepeat, pc + 2, position, first + inst.x});
                }
                pc += 2;
                break;
            }
            case OpMatch:
                if (top &&((state.match_all && position != size) || (state.must_advance && position == state.start))) {
                    ok = false;
                    break;
                }
                *end = position;
                return true;
        }
        if (ok) {
            continue;
        }
        // Back to the last branch not yet tried, undoing what was set since.
        bool resumed = false;
        while (state.stack.size() > base) {
            const Frame frame = state.stack.back();
            state.stack.pop_back();
            if (frame.kind == FrameBranch) {
                pc = frame.index;
                position = frame.value;
                resumed = true;
                break;
            }
            if (frame.kind == FrameRepeat) {
                // Give characters back, one at a time. When a plain character must
                // follow, only the places where it stands are worth going on from.
                long back = frame.value - 1;
                const Inst & next = code[frame.index];
                if (next.op == OpChar) {
                    while (back >= frame.least && !(back < size && text[static_cast<size_t>(back)] == static_cast<char32_t>(next.x))) {
                        --back;
                    }
                    if (back < frame.least) {
                        continue;
                    }
                }
                position = back;
                pc = frame.index;
                if (position > frame.least) {
                    state.stack.push_back(Frame{FrameRepeat, frame.index, position, frame.least});
                }
                resumed = true;
                break;
            }
            if (frame.kind == FrameCapture) {
                state.caps[static_cast<size_t>(frame.index)] = frame.value;
            } else if (frame.kind == FrameRegister) {
                state.regs[static_cast<size_t>(frame.index)] = frame.value;
            } else {
                state.caps = state.snapshots[static_cast<size_t>(frame.index)];
            }
        }
        if (!resumed) {
            return false;
        }
    }
}

std::mutex & record_mutex() {
    static std::mutex mutex;
    return mutex;
}

bool & recording() {
    static bool on = false;
    return on;
}

std::vector<std::pair<std::string, int>> & recorded() {
    static std::vector<std::pair<std::string, int>> patterns;
    return patterns;
}

}  // namespace

// ---- Regex ----------------------------------------------------------------------

Regex::Regex(std::string_view pattern, int flags) {
    auto program = std::make_shared<Program>();
    program->pattern = std::string(pattern);
    program->flags = flags;
    Parser parser(decode(pattern), flags, *program);
    const NodePtr root = parser.parse();
    Compiler(*program).compile(*root);
    First first;
    const bool empty = first_of(*root, *program, first);
    if (!empty && !first.unknown) {
        program->has_first = true;
        program->first_low[0] = first.low[0];
        program->first_low[1] = first.low[1];
        program->first_other = first.other;
    }
    program->begins_at_start = begins_at_start(*root);
    program_ = std::move(program);
    {
        const std::lock_guard<std::mutex> lock(record_mutex());
        if (recording()) {
            recorded().emplace_back(program_->pattern, flags);
        }
    }
}

const std::string & Regex::pattern() const { return program_->pattern; }
int Regex::flags() const { return program_->flags; }
int Regex::groups() const { return program_->groups; }

// The backtracking stack is kept between calls on the same thread, so that a
// search does not start by growing it again.
struct StackLoan {
    explicit StackLoan(State & state_in) : state(state_in) {
        state.stack.swap(spare());
        state.stack.clear();
    }
    ~StackLoan() { state.stack.swap(spare()); }
    static std::vector<Frame> & spare() {
        thread_local std::vector<Frame> value;
        return value;
    }
    State & state;
};

Match Regex::find(StrView text, size_t start, bool must_advance, bool anchored) const {
    const Program & program = *program_;
    State state;
    const StackLoan loan(state);
    state.text = text;
    state.caps.assign(static_cast<size_t>(2 * (program.groups + 1)), -1);
    state.regs.assign(static_cast<size_t>(program.registers), -1);
    const size_t size = text.size();
    for (size_t at = start; at <= size; ++at) {
        const bool first_try = at == start;
        if (program.has_first) {
            if (at == size) {
                break;
            }
            const char32_t c = text[at];
            const bool possible = c < 128 ? ((program.first_low[c >> 6] >> (c & 63)) & 1) != 0 : program.first_other;
            if (!possible) {
                if (anchored) {
                    break;
                }
                continue;
            }
        }
        if (!first_try) {
            std::fill(state.caps.begin(), state.caps.end(), -1);
            std::fill(state.regs.begin(), state.regs.end(), -1);
            state.stack.clear();
            state.snapshots.clear();
        }
        state.steps = 0;
        state.start = static_cast<long>(at);
        state.must_advance = must_advance && first_try;
        long end = 0;
        if (run(program, state, 0, static_cast<long>(at), 0, true, &end)) {
            Match found;
            found.ok_ = true;
            found.text_ = text;
            found.program_ = program_;
            found.spans_ = std::move(state.caps);
            found.spans_[0] = static_cast<long>(at);
            found.spans_[1] = end;
            return found;
        }
        if (anchored || program.begins_at_start) {
            break;
        }
    }
    return Match();
}

Match Regex::search(StrView text, size_t position) const {
    return find(text, position, false, false);
}

Match Regex::match(StrView text, size_t position) const {
    return find(text, position, false, true);
}

Match Regex::fullmatch(StrView text) const {
    const Program & program = *program_;
    State state;
    const StackLoan loan(state);
    state.text = text;
    state.caps.assign(static_cast<size_t>(2 * (program.groups + 1)), -1);
    state.regs.assign(static_cast<size_t>(program.registers), -1);
    state.match_all = true;
    long end = 0;
    if (!run(program, state, 0, 0, 0, true, &end)) {
        return Match();
    }
    Match found;
    found.ok_ = true;
    found.text_ = text;
    found.program_ = program_;
    found.spans_ = std::move(state.caps);
    found.spans_[0] = 0;
    found.spans_[1] = end;
    return found;
}

std::vector<Match> Regex::finditer(StrView text) const {
    std::vector<Match> out;
    size_t start = 0;
    bool must_advance = false;
    for (;;) {
        Match found = find(text, start, must_advance, false);
        if (!found) {
            break;
        }
        must_advance = found.end() == found.start();
        start = static_cast<size_t>(found.end());
        out.push_back(std::move(found));
    }
    return out;
}

std::vector<Str> Regex::findall(StrView text) const {
    if (program_->groups > 1) {
        throw PatternError("findall with several groups: use finditer");
    }
    std::vector<Str> out;
    for (const Match & found : finditer(text)) {
        out.push_back(found.group(program_->groups));
    }
    return out;
}

std::vector<Str> Regex::split(StrView text) const {
    std::vector<Str> out;
    size_t last = 0;
    size_t start = 0;
    bool must_advance = false;
    for (;;) {
        const Match found = find(text, start, must_advance, false);
        if (!found) {
            break;
        }
        out.emplace_back(text.substr(last, static_cast<size_t>(found.start()) - last));
        for (int group = 1; group <= program_->groups; ++group) {
            out.push_back(found.group(group));
        }
        must_advance = found.end() == found.start();
        last = start = static_cast<size_t>(found.end());
    }
    out.emplace_back(text.substr(last));
    return out;
}

Str Regex::sub(StrView replacement, StrView text) const {
    // The template, as Python reads it: \1, \g<1>, \g<name>, and the usual escapes.
    struct Piece {
        Str text;
        int group = -1;
    };
    std::vector<Piece> pieces;
    Str literal;
    auto flush = [&]() {
        if (!literal.empty()) {
            pieces.push_back(Piece{literal, -1});
            literal.clear();
        }
    };
    for (size_t i = 0; i < replacement.size(); ++i) {
        const char32_t c = replacement[i];
        if (c != U'\\' || i + 1 >= replacement.size()) {
            literal.push_back(c);
            continue;
        }
        const char32_t next = replacement[++i];
        if (next == U'g') {
            if (i + 1 >= replacement.size() || replacement[i + 1] != U'<') {
                throw PatternError("missing < in a replacement");
            }
            const size_t close = replacement.find(U'>', i + 2);
            if (close == StrView::npos) {
                throw PatternError("missing > in a replacement");
            }
            const Str name(replacement.substr(i + 2, close - i - 2));
            bool number = !name.empty();
            for (const char32_t d : name) {
                number = number && is_digit(d);
            }
            const int group = number ? static_cast<int>(py::to_int(name)) : program_->group_index(encode(name));
            if (group < 0 || group > program_->groups) {
                throw PatternError("unknown group in a replacement");
            }
            flush();
            pieces.push_back(Piece{Str(), group});
            i = close;
        } else if (is_digit(next)) {
            int group = static_cast<int>(next - U'0');
            if (next != U'0' && i + 1 < replacement.size() && is_digit(replacement[i + 1])) {
                group = group * 10 + static_cast<int>(replacement[++i] - U'0');
            }
            if (next == U'0' || group > program_->groups) {
                throw PatternError("invalid group reference in a replacement");
            }
            flush();
            pieces.push_back(Piece{Str(), group});
        } else {
            switch (next) {
                case U'n': literal.push_back(U'\n'); break;
                case U't': literal.push_back(U'\t'); break;
                case U'r': literal.push_back(U'\r'); break;
                case U'f': literal.push_back(0x0C); break;
                case U'v': literal.push_back(0x0B); break;
                case U'a': literal.push_back(0x07); break;
                case U'b': literal.push_back(0x08); break;
                case U'\\': literal.push_back(U'\\'); break;
                default:
                    if (is_ascii_letter(next)) {
                        throw PatternError("bad escape in a replacement");
                    }
                    literal.push_back(U'\\');
                    literal.push_back(next);
            }
        }
    }
    flush();
    return sub_with(
        [&](const Match & found) {
            Str out;
            for (const Piece & piece : pieces) {
                if (piece.group < 0) {
                    out.append(piece.text);
                } else {
                    out.append(found.group(piece.group));
                }
            }
            return out;
        },
        text);
}

// ---- Match ----------------------------------------------------------------------

bool Match::has(int group) const {
    if (!ok_ || group < 0 || static_cast<size_t>(2 * group + 1) >= spans_.size()) {
        return false;
    }
    return spans_[static_cast<size_t>(2 * group)] >= 0 && spans_[static_cast<size_t>(2 * group + 1)] >= spans_[static_cast<size_t>(2 * group)];
}

bool Match::has(std::string_view name) const {
    return program_ != nullptr && has(program_->group_index(name));
}

Str Match::group(int group) const {
    if (!has(group)) {
        return Str();
    }
    const auto from = static_cast<size_t>(spans_[static_cast<size_t>(2 * group)]);
    const auto to = static_cast<size_t>(spans_[static_cast<size_t>(2 * group + 1)]);
    return Str(text_.substr(from, to - from));
}

Str Match::group(std::string_view name) const {
    return program_ == nullptr ? Str() : group(program_->group_index(name));
}

long Match::start(int group) const {
    return has(group) ? spans_[static_cast<size_t>(2 * group)] : -1;
}

long Match::end(int group) const {
    return has(group) ? spans_[static_cast<size_t>(2 * group + 1)] : -1;
}

long Match::start(std::string_view name) const {
    return program_ == nullptr ? -1 : start(program_->group_index(name));
}

long Match::end(std::string_view name) const {
    return program_ == nullptr ? -1 : end(program_->group_index(name));
}

int Match::groups() const {
    return program_ == nullptr ? 0 : program_->groups;
}

// ---- helpers ----------------------------------------------------------------------

std::string escape(StrView text) {
    static const StrView special = U"()[]{}?*+-|^$\\.&~# \t\n\r\v\f";
    std::string out;
    for (const char32_t c : text) {
        if (special.find(c) != StrView::npos) {
            out.push_back('\\');
        }
        append_utf8(out, c);
    }
    return out;
}

void record_patterns(bool on) {
    const std::lock_guard<std::mutex> lock(record_mutex());
    recording() = on;
}

std::vector<std::pair<std::string, int>> recorded_patterns() {
    const std::lock_guard<std::mutex> lock(record_mutex());
    return recorded();
}

}  // namespace dizione::pyre
