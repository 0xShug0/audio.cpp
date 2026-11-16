// dizione: English text to phonemes.
//
// A dictionary built from CMUdict, a small model for words the dictionary
// lacks, homograph rules, and text rules for numbers, dates, money, units and
// chat forms. This is the C++ port of https://github.com/n4ze3m/dizione and
// gives the same phoneme ids as the Python package for the same data files.
//
// Nothing in this folder depends on anything but the C++ standard library.
// One Phonemizer serves one thread at a time.
#pragma once

#include "g2p.h"
#include "homographs.h"
#include "align.h"
#include "normalize.h"
#include "sentence.h"
#include "symbols.h"
#include "tables.h"
#include "text.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dizione {

// The token-convention version recorded by models trained on the output.
constexpr int kVersion = 4;

// The built data files, as their contents.
struct Data {
    std::string lexicon;            // lexicon.tsv
    std::string align;              // align.json
    std::string g2p;                // g2p.json; empty for no unknown-word model
    Tensors g2p_weights;            // the arrays of g2p.npz
    std::string homographs;         // homographs.json; empty for none
};

// Reads lexicon.tsv, align.json, g2p.json, g2p.npz and homographs.json from a
// directory. Raises ValueError when the dictionary is missing.
Data load_directory(const std::string & directory);

// The dictionary: the lines of lexicon.tsv ("word<TAB>pronunciation"), found
// by binary search. The index is four bytes a word, where a hash map of the
// same lines takes about sixty.
class Lexicon {
public:
    Lexicon() = default;
    explicit Lexicon(std::string text);

    bool empty() const { return lines_.empty(); }
    bool contains(std::string_view word) const { return find(word).has_value(); }
    // The pronunciation of a word, as it is written in the file.
    std::optional<std::string_view> find(std::string_view word) const;

private:
    std::string_view word_at(uint32_t line) const;

    std::string text_;
    std::vector<uint32_t> lines_;       // where each line starts, ordered by its word
};

class Phonemizer {
public:
    explicit Phonemizer(Data data, bool join_phrases = true);

    // Pronunciations that win over everything else: a table as text ("word,
    // two spaces, pronunciation" per line) or pairs. A pronunciation is
    // CMUdict notation or phones with stress marks; a name of several words
    // takes one pronunciation per word, separated by "|".
    void add_overrides(const tables::Source & source);

    // Text -> symbols, ids and stress. Raises ValueError for an empty text and
    // for one with nothing to pronounce.
    Phonemized phonemize(std::string_view utf8_text);

    // The steps, for checks and reports.
    struct Clause {
        bool punctuation = false;
        Str mark;
        std::vector<Unit> units;
    };
    std::vector<Clause> analyse(const Str & text);
    std::vector<Piece> pieces(const Str & text);
    // (phones, source) for one spelling in lower case.
    std::pair<Phones, std::string> word(const Str & word);
    bool is_word(const Str & word) const;
    normalize::Normalizer & normalizer() { return *normalizer_; }

private:
    using Token = normalize::Token;

    Phones entry(const Str & word, const tables::Pronunciation & pronunciation) const;
    std::optional<Phones> known(const Str & word) const;
    std::optional<Phones> ordinary(const Str & word) const;
    Phones guess(const Str & word) const;
    Phones spell(const Str & letters) const;
    std::optional<std::pair<Phones, std::string>> from_parts(const Str & word);
    std::vector<Unit> units(const std::vector<Token> & words, const std::vector<size_t> & places, const std::vector<Str> & written);
    size_t name_size(const std::vector<Token> & words, size_t index) const;
    size_t phrase_size(const std::vector<Token> & words, size_t index) const;
    void refresh_names();

    Lexicon lexicon_;
    std::optional<Aligner> aligner_;
    std::optional<G2p> model_;
    std::optional<Homographs> homographs_;
    std::unordered_map<Str, tables::FunctionWord> functions_;
    std::map<std::vector<Str>, tables::Phrase> phrases_;
    size_t phrase_length_ = 1;
    std::unordered_map<Str, tables::Pronunciation> exceptions_;
    std::map<std::vector<Str>, std::vector<tables::Pronunciation>> names_;
    std::unordered_set<Str> name_starts_;
    size_t name_length_ = 1;
    std::map<std::vector<Str>, bool> user_names_;
    std::unordered_map<Str, tables::Pronunciation> overrides_;
    std::unordered_map<Str, Phones> letter_names_;
    std::unique_ptr<normalize::Normalizer> normalizer_;
    std::unordered_map<Str, std::pair<Phones, std::string>> cache_;
};

}  // namespace dizione
