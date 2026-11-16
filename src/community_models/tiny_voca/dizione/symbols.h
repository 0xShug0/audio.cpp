// The phone inventory and the token sequence built from it: 59 phones of
// American English, 12 punctuation marks and 5 special tokens. Stress is a
// feature beside each token (0 none, 1 primary, 2 secondary), not a symbol.
// The order is fixed: models trained on these ids depend on it.
//
// Part of the dizione C++ port (symbols.py). Standard library only.
#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace dizione {

struct Phone {
    std::string phone;      // UTF-8
    int stress = 0;

    bool operator==(const Phone & other) const { return stress == other.stress && phone == other.phone; }
    bool operator!=(const Phone & other) const { return !(*this == other); }
};
using Phones = std::vector<Phone>;

inline constexpr const char * kPad = "<pad>";
inline constexpr const char * kBos = "<bos>";
inline constexpr const char * kEos = "<eos>";
inline constexpr const char * kSp = "<sp>";
inline constexpr const char * kUnk = "<unk>";

// All 76 symbols, in id order.
const std::vector<std::string> & symbols();
// The id of a symbol, -1 when it is not one.
int symbol_id(std::string_view symbol);
bool is_phone(std::string_view symbol);
bool is_vowel(std::string_view symbol);
bool is_punctuation(std::string_view symbol);

struct Phonemized {
    std::vector<std::string> symbols;
    std::vector<int> ids;
    std::vector<int> stress;
    std::vector<std::string> unknown;       // phones outside the inventory (the built-in rules give none)
};

// One stretch of a text: a punctuation mark, or the words between two marks.
struct Piece {
    bool punctuation = false;
    std::string mark;
    std::vector<Phones> words;
};

// The token sequence: <bos>, the words parted by <sp>, the marks, <eos>.
// Raises ValueError when there is nothing to pronounce.
Phonemized tokenize(const std::vector<Piece> & pieces);

}  // namespace dizione
