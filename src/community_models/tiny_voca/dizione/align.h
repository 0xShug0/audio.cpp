// Which letters spell which phoneme: every phoneme of a pronunciation gets a
// run of zero to four letters of the word, in order ("thought" TH AO T is
// th|ough|t). The pronunciation rules need this to tell an unstressed vowel
// written "i" from one written "a".
//
// Part of the dizione C++ port (align.py; the table is estimated in Python
// when the data is built). Standard library only.
#pragma once

#include "json.h"
#include "text.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace dizione {

// An ARPABET symbol without its stress digit.
std::string arpabet_base(std::string_view symbol);

class Aligner {
public:
    // `table` is align.json: {phoneme: {letters: log probability}}.
    explicit Aligner(const json::Value & table);

    double score(const std::string & phone, StrView run) const;
    // One run of letters per phoneme, or nothing when the word is too long for
    // the pronunciation (an abbreviation such as "dept").
    std::optional<std::vector<Str>> align(StrView word, const std::vector<std::string> & phones) const;

private:
    std::unordered_map<std::string, std::unordered_map<std::string, double>> table_;
};

}  // namespace dizione
