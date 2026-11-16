// Words spelled alike and pronounced differently: which reading is meant
// ("I read it yesterday" / "to read", "a live show" / "they live here"). For
// each homograph a small linear classifier looks at the neighbouring words;
// a few hand-written rules come first. The weights are trained in Python
// (homographs.json); this adds them up over the same features.
//
// Part of the dizione C++ port (homographs.py). Standard library only.
#pragma once

#include "json.h"
#include "text.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dizione {

class Homographs {
public:
    // `data` is homographs.json. The hand rules are the built-in table.
    explicit Homographs(const json::Value & data);

    // True for a homograph and for the regular -s form of a noun-or-verb one.
    bool covers(const Str & word) const;

    struct Spoken {
        std::string phones;         // as text: "ɹ ˈɛ d"
        bool plural = false;        // the -s ending has to be added
    };
    // The reading of tokens[index]. `joined` says a hyphen joins the word to the one before it.
    Spoken pronounce(const Str & word, const std::vector<Str> & tokens, size_t index, bool joined) const;

private:
    struct Sense {
        std::string id;
        std::string role;
        std::string phones;
    };
    using Weights = std::unordered_map<std::string, double>;
    struct Entry {
        std::vector<Sense> senses;
        bool noun_verb = false;
        double bias = 0.0;                  // noun-or-verb model
        Weights weights;
        std::vector<double> biases;         // one per sense otherwise
        std::vector<Weights> sense_weights;
    };
    struct Rule {
        std::string sense;
        std::vector<std::pair<std::string, std::unordered_set<Str>>> conditions;
    };

    const Sense & choose(const Str & word, const std::vector<Str> & tokens, size_t index, bool joined) const;
    std::vector<double> scores(const Entry & entry, const std::vector<Str> & tokens, size_t index) const;

    std::unordered_map<Str, Entry> words_;
    double pooled_bias_ = 0.0;
    Weights pooled_weights_;
    std::unordered_set<Str> plural_bases_;
    std::unordered_map<Str, std::vector<Rule>> rules_;
};

// sum() of Python 3.12 over floats: the first value, then compensated addition.
double python_sum(const std::vector<double> & values);

}  // namespace dizione
