// Strings as Python has them: sequences of code points, with the methods the
// port of dizione needs. Every function here follows the Python method of the
// same name, so the C++ rules can be read side by side with the Python ones.
//
// Part of the dizione C++ port. Standard library only.
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace dizione {

using Str = std::u32string;
using StrView = std::u32string_view;

// The Python exceptions the rules can raise. `kind` is the Python class name.
struct Error : std::runtime_error {
    Error(const char * kind_in, const std::string & message)
        : std::runtime_error(message), kind(kind_in) {}
    const char * kind;
};
struct ValueError : Error {
    explicit ValueError(const std::string & message) : Error("ValueError", message) {}
};
struct KeyError : Error {
    explicit KeyError(const std::string & message) : Error("KeyError", message) {}
};
struct IndexError : Error {
    explicit IndexError(const std::string & message) : Error("IndexError", message) {}
};

// UTF-8. A byte that is not valid UTF-8 becomes U+FFFD.
Str decode(std::string_view utf8);
std::string encode(StrView text);
void append_utf8(std::string & out, char32_t code);

namespace uni {

bool is_alpha(char32_t c);
bool is_decimal(char32_t c);
bool is_digit(char32_t c);
bool is_numeric(char32_t c);
bool is_alnum(char32_t c);
bool is_space(char32_t c);
bool is_upper(char32_t c);
bool is_lower(char32_t c);
bool is_title(char32_t c);
// The classes of the re module: \w and \d are is_word and is_decimal, \s is is_space.
inline bool is_word(char32_t c) { return c == U'_' || is_alnum(c); }
// Value of a decimal digit, -1 for any other character.
int decimal_value(char32_t c);
int combining(char32_t c);
// Case folding as the re module does it for IGNORECASE.
char32_t simple_lower(char32_t c);
bool sre_cased(char32_t c);
// Other lower-case characters that match `lower` when case is ignored.
const uint32_t * extra_cases(char32_t lower, size_t * count);
Str nfkc(StrView text);
Str nfkd(StrView text);

}  // namespace uni

namespace py {

constexpr long kNone = 0x7fffffffL;      // "no bound" in a slice

Str lower(StrView text);
Str upper(StrView text);
bool isalpha(StrView text);
bool isdigit(StrView text);
bool isalnum(StrView text);
bool isupper(StrView text);
bool islower(StrView text);
bool isspace(StrView text);
bool is_ascii(StrView text);        // str.isascii(); "isascii" itself is a macro in some C libraries

// text[index], with negative indices; raises IndexError like Python.
char32_t at(StrView text, long index);
// text[start:stop]
Str slice(StrView text, long start, long stop = kNone);

Str strip(StrView text);
Str lstrip(StrView text);
Str rstrip(StrView text);
Str strip(StrView text, StrView chars);
Str lstrip(StrView text, StrView chars);
Str rstrip(StrView text, StrView chars);

// text.split(): on runs of white space, no empty pieces.
std::vector<Str> split(StrView text);
// text.split(sep[, maxsplit])
std::vector<Str> split(StrView text, StrView sep, long maxsplit = -1);
std::vector<Str> rsplit(StrView text, StrView sep, long maxsplit);
// text.splitlines()
std::vector<Str> splitlines(StrView text);
Str join(StrView sep, const std::vector<Str> & parts);

bool startswith(StrView text, StrView prefix);
bool endswith(StrView text, StrView suffix);
bool startswith_any(StrView text, std::initializer_list<StrView> prefixes);
bool endswith_any(StrView text, std::initializer_list<StrView> suffixes);
bool contains(StrView text, StrView part);
inline bool contains(StrView text, char32_t c) { return text.find(c) != StrView::npos; }
long find(StrView text, StrView part);
Str replace(StrView text, StrView old_part, StrView new_part, long count = -1);
Str ljust(StrView text, size_t width, char32_t fill);
Str repeat(StrView text, long count);

struct Parts {
    Str head, sep, tail;
};
Parts partition(StrView text, StrView sep);
Parts rpartition(StrView text, StrView sep);

// int(text) for a non-negative decimal number that fits 63 bits; raises
// ValueError for anything Python's int() rejects. Longer numbers are handled
// by the callers that can meet them (see normalize.cpp).
long long to_int(StrView text);
Str from_int(long long value);

}  // namespace py

// ASCII text as a Str, for literals and table keys.
Str S(std::string_view ascii_or_utf8);

}  // namespace dizione
