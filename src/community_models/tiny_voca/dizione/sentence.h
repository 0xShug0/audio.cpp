// Sentence rules: what a word's pronunciation owes to its neighbours. They
// work on one clause (the words between two punctuation marks):
//
// * sentence stress: function words lose their stress, or keep half of it,
//   unless they close the clause (function_words.txt says which);
// * "the" and "to" before a vowel;
// * linking r: "for" + "all", "after" + "it";
// * the flap at the end of "it" and "at" before a vowel.
//
// Part of the dizione C++ port (sentence.py). Standard library only.
#pragma once

#include "symbols.h"
#include "text.h"

#include <optional>
#include <string>
#include <vector>

namespace dizione {

struct Unit {
    Str word;                           // spelling in lower case; several words for a phrase
    Phones phones;
    std::string kind = "c";             // stress class of function_words.txt; "c" = stress as given
    std::optional<Phones> final_form;   // form to use when the word is promoted
    bool attach = false;                // no word boundary before this unit (hyphenated compound)
    std::string source;                 // where the pronunciation came from (for reports)

    Str last_word() const;
    Str first_word() const;
};

namespace sentence {

Phones unstress(const Phones & phones);
// Give the word a primary stress: on its marked vowel, else on the first vowel.
Phones promote(const Phones & phones);
// Resolve the stress class of every unit of a clause, right to left.
void stress_clause(std::vector<Unit> & units);
// Adjust the end of each word to the beginning of the next.
void join_clause(std::vector<Unit> & units);
// Units of a clause as spoken words: an attached unit continues the one before.
std::vector<Phones> groups(const std::vector<Unit> & units);

}  // namespace sentence

}  // namespace dizione
