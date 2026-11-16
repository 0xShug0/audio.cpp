// A small regular-expression matcher that behaves like Python's `re` module
// for the constructs dizione uses, so its patterns can be kept word for word:
// look-ahead and fixed-width look-behind, named groups, back-references,
// conditional groups, lazy quantifiers, scoped flags, and the Unicode-aware
// classes \w \d \s \b. Text is matched by code point, as Python does.
//
// Not supported, and reported when a pattern is compiled: MULTILINE, DOTALL,
// VERBOSE, possessive quantifiers and atomic groups.
//
// A match attempt that takes more than `kStepLimit` steps stops with
// `StepLimit`, so no input can keep a caller busy for long. Python has no such
// limit; ordinary text stays far below it.
//
// Part of the dizione C++ port. Standard library only.
#pragma once

#include "text.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dizione::pyre {

constexpr int IGNORECASE = 2;
constexpr int ASCII = 256;
constexpr long kStepLimit = 5000000;

struct PatternError : Error {
    explicit PatternError(const std::string & message) : Error("PatternError", message) {}
};
struct StepLimit : Error {
    StepLimit() : Error("RegexStepLimit", "a pattern needed too many steps on this text") {}
};

struct Program;

// The result of one match. It refers to the text that was searched, which must
// outlive it. A match that failed converts to false.
class Match {
public:
    Match() = default;

    explicit operator bool() const noexcept { return ok_; }

    // True when the group took part in the match (Python: group(n) is not None).
    bool has(int group) const;
    bool has(std::string_view name) const;
    // The text of a group, empty when it took no part.
    Str group(int group = 0) const;
    Str group(std::string_view name) const;
    // Positions in the text, -1 when the group took no part.
    long start(int group = 0) const;
    long end(int group = 0) const;
    long start(std::string_view name) const;
    long end(std::string_view name) const;
    // match.string
    StrView string() const noexcept { return text_; }
    // The number of groups of the pattern.
    int groups() const;

private:
    friend class Regex;
    bool ok_ = false;
    StrView text_;
    std::vector<long> spans_;
    std::shared_ptr<const Program> program_;
};

class Regex {
public:
    // `pattern` is UTF-8. Besides Python's syntax, \UXXXXXXXX names a code point.
    explicit Regex(std::string_view pattern, int flags = 0);

    Match search(StrView text, size_t position = 0) const;
    Match match(StrView text, size_t position = 0) const;
    Match fullmatch(StrView text) const;
    std::vector<Match> finditer(StrView text) const;
    // re.findall for a pattern with no group or one: the matches, or group 1.
    std::vector<Str> findall(StrView text) const;
    std::vector<Str> split(StrView text) const;
    // re.sub with a template (\1, \g<name>).
    Str sub(StrView replacement, StrView text) const;
    // re.sub with a function of the match.
    template <typename Function>
    Str sub_with(Function && function, StrView text) const {
        Str out;
        size_t copied = 0;
        size_t start = 0;
        bool must_advance = false;
        for (;;) {
            const Match found = find(text, start, must_advance, false);
            if (!found) {
                break;
            }
            const auto begin = static_cast<size_t>(found.start());
            const auto end = static_cast<size_t>(found.end());
            out.append(text.substr(copied, begin - copied));
            out.append(function(found));
            copied = end;
            must_advance = end == begin;
            start = end;
        }
        out.append(text.substr(copied));
        return out;
    }

    const std::string & pattern() const;
    int flags() const;
    int groups() const;

private:
    Match find(StrView text, size_t start, bool must_advance, bool anchored) const;

    std::shared_ptr<const Program> program_;
};

// re.escape
std::string escape(StrView text);

// For the tests: keep every pattern compiled from now on, and list them.
void record_patterns(bool on);
std::vector<std::pair<std::string, int>> recorded_patterns();

}  // namespace dizione::pyre
