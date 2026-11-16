// The hand-written tables that ship with dizione (data/*.txt in the Python
// package). Here they are compiled in (tables_data.inc), read once when a
// phonemizer is made.
//
// Plain text, one entry per line, columns separated by tabs or by two or more
// spaces; "#" starts a comment line. A pronunciation is either phones of the
// inventory with stress marks ("h ə l ˈoʊ") or CMUdict notation
// ("HH AH0 L OW1"); the reader tells them apart.
//
// Part of the dizione C++ port (tables.py). Standard library only.
#pragma once

#include "symbols.h"
#include "text.h"

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dizione::tables {

struct Row {
    long number = 0;                // line number, from 1
    std::vector<Str> columns;
};

// The text of a built-in table ("units.txt"); raises KeyError for an unknown name.
Str text(std::string_view name);
// Rows of a table given as text.
std::vector<Row> rows_of(StrView table_text);
// Rows of a built-in table.
std::vector<Row> rows(std::string_view name);

struct FunctionWord {
    std::string kind;               // stress class
    Phones form;
    std::optional<Phones> final_form;
};
// word -> class, form, form at the end of a clause
std::unordered_map<Str, FunctionWord> function_words();

struct Phrase {
    std::string kind;
    std::vector<Phones> forms;      // one when the words are joined, one per word when the boundaries stay
    std::optional<Phones> final_form;
};
std::map<std::vector<Str>, Phrase> phrases();

struct Pronunciation {
    bool arpabet = false;
    std::vector<std::string> symbols;       // CMUdict notation
    Phones phones;                          // or phones as written
};

// A table of words and pronunciations: a built-in table, the text of one, or pairs.
struct Source {
    std::string name;                               // "exceptions.txt"
    Str table_text;
    std::vector<std::pair<Str, Str>> pairs;
    enum Kind { BuiltIn, Text, Pairs } kind = BuiltIn;

    static Source built_in(std::string_view name);
    static Source from_text(StrView table_text);
    static Source from_pairs(std::vector<std::pair<Str, Str>> pairs);
};

// word -> pronunciation, for the built-in exception list and for the user's
// own words. Rows whose first column has several words are names: see `names`.
std::unordered_map<Str, Pronunciation> pronunciations(const Source & source);
// (word, word, ...) -> one pronunciation per word, for a name of several words.
std::map<std::vector<Str>, std::vector<Pronunciation>> names(const Source & source);
// A table of two columns as a map (the second column may be missing).
std::unordered_map<Str, Str> mapping(std::string_view name);

}  // namespace dizione::tables
