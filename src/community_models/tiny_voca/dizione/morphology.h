// Words the dictionary lacks but can be built from words it has: plurals,
// possessives, -ed, -ing, -ly and other regular endings, a few prefixes,
// compounds of two dictionary words and British spellings. The stem's
// pronunciation comes from the dictionary; the ending is added by rule, with
// the adjustments English makes at the seam (the flap of "waited", the r of
// "starring", the voicing of "dogs" against "cats").
//
// Part of the dizione C++ port (morphology.py). Standard library only.
#pragma once

#include "symbols.h"
#include "text.h"

#include <functional>
#include <optional>

namespace dizione::morphology {

// A dictionary: the pronunciation of a spelling, or nothing.
using Lookup = std::function<std::optional<Phones>(const Str &)>;

// Stem + ending, adjusted at the seam.
Phones join(const Phones & stem, const Phones & ending);
Phones plural(const Phones & stem);
Phones past(const Phones & stem);
Phones adverb(const Phones & stem);
// Two words said as one: the second keeps only a lesser stress ("sandpiper").
Phones compound(const Phones & first, const Phones & second);
// Pronunciation of `word` from a shorter dictionary word, or nothing. `part`
// looks up a word that may be half of a compound.
std::optional<Phones> derive(const Str & word, const Lookup & lookup, const Lookup & part, int depth = 0);

}  // namespace dizione::morphology
