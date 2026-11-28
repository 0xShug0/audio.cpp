// One word: CMUdict pronunciation -> phones of the inventory.
//
// CMUdict writes a broad transcription (39 phonemes, stress digits on the
// vowels). The model was trained on a narrower one: flaps, a glottal stop
// before syllabic n, r-coloured vowels as single phones, syllabic l, several
// reduced vowels. The rules here add what CMUdict lacks. Many depend on how a
// vowel is written, so they work on the pronunciation aligned to the spelling.
//
// Part of the dizione C++ port (convert.py). Standard library only.
#pragma once

#include "symbols.h"
#include "text.h"

#include <optional>
#include <string>
#include <vector>

namespace dizione {

// True when every symbol is ARPABET (vowels with their stress digit) and there is one at least.
bool is_arpabet(const std::vector<std::string> & symbols);
// "h ə l ˈoʊ" <-> phones. from_text raises ValueError for a symbol outside the inventory.
std::string to_text(const Phones & phones);
Phones from_text(std::string_view text);
// Citation form of `word`. `arpabet` is a CMUdict pronunciation and `runs` the
// letters aligned to each symbol, when there is an alignment.
Phones convert(StrView word, const std::vector<std::string> & arpabet, const std::optional<std::vector<Str>> & runs = std::nullopt);
// One primary stress per word, and no secondary stress next to the primary one.
Phones settle_stress(const Phones & phones);

}  // namespace dizione
