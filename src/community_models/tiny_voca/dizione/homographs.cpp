#include "homographs.h"

#include "tables.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace dizione {

namespace {

constexpr long kWindow = 6;

// Coarse grammatical classes of the neighbouring words, from closed lists.
const std::unordered_map<Str, std::string> & word_classes() {
    static const std::unordered_map<Str, std::string> value = [] {
        const std::vector<std::pair<const char *, const char32_t *>> classes = {
            {"TO", U"to"},
            {"DET", U"the a an this that these those every each no some any another such what which whose"},
            {"POSS", U"my your his her its our their"},
            {"SUBJ", U"i you he she it we they who"},
            {"OBJ", U"me him us them"},
            {"MODAL", U"will would shall should can could may might must 'll 'd won't can't cannot couldn't shouldn't wouldn't let's please"},
            {"DO", U"do does did don't doesn't didn't"},
            {"BE", U"am is are was were be been being isn't aren't wasn't weren't i'm you're we're they're he's she's it's that's there's"},
            {"HAVE", U"have has had having haven't hasn't hadn't i've you've we've they've"},
            {"NEG", U"not never"},
            {"PREP", U"of in on at by for from with into onto over under about after before between through during without within "
                     U"against among around across behind beside near upon toward towards as than like"},
            {"CONJ", U"and or but nor so yet if because although though while when where unless until since whether"},
            {"ADV", U"very too quite rather more most less least also just only even still always often sometimes usually again "
                    U"now then here there soon already almost"},
        };
        std::unordered_map<Str, std::string> out;
        for (const auto & entry : classes) {
            for (const Str & word : py::split(entry.second)) {
                out[word] = entry.first;        // a word listed twice keeps its last class, as in Python
            }
        }
        return out;
    }();
    return value;
}

bool sentence_end(const Str & token) {
    return token == U"." || token == U"?" || token == U"!";
}

Str tag(const Str & token) {
    if (token == U"<s>" || token == U"</s>") {
        return token;
    }
    const char32_t first = py::at(token, 0);
    if (!uni::is_alpha(first)) {
        return uni::is_digit(first) ? U"NUM" : U"PUNCT";
    }
    const Str lower = py::lower(token);
    const auto found = word_classes().find(lower);
    if (found != word_classes().end()) {
        return S(found->second);
    }
    const std::pair<const char32_t *, const char32_t *> endings[] = {{U"'s", U"POS"}, {U"ly", U"LY"}, {U"ing", U"ING"}, {U"ed", U"ED"}};
    for (const auto & ending : endings) {
        const StrView suffix(ending.first);
        if (py::endswith(lower, suffix) && lower.size() > suffix.size() + 1) {
            return ending.second;
        }
    }
    if (py::endswith(lower, U"s") && !py::endswith(lower, U"ss") && lower.size() > 3) {
        return U"S";
    }
    return uni::is_upper(first) ? U"CAP" : U"W";
}

// Names of the features that are on for tokens[index]. `tokens` is one
// sentence: words as written, punctuation marks as their own tokens.
std::vector<std::string> features(const std::vector<Str> & tokens, long index) {
    const long size = static_cast<long>(tokens.size());
    auto at = [&](long offset) -> Str {
        const long position = index + offset;
        if (position >= 0 && position < size) {
            return tokens[static_cast<size_t>(position)];
        }
        return offset < 0 ? U"<s>" : U"</s>";
    };
    const Str p2 = at(-2), p1 = at(-1), n1 = at(1), n2 = at(2);
    const std::string tp2 = encode(tag(p2)), tp1 = encode(tag(p1)), tn1 = encode(tag(n1)), tn2 = encode(tag(n2));
    std::vector<std::string> out = {
        "tp1=" + tp1, "tp2=" + tp2, "tn1=" + tn1, "tn2=" + tn2, "tp2p1=" + tp2 + "+" + tp1, "tp1n1=" + tp1 + "+" + tn1,
        "tn1n2=" + tn1 + "+" + tn2,
    };
    const Str & self = tokens.at(static_cast<size_t>(index));
    if (index == 0) {
        out.push_back("first");
    } else if (uni::is_upper(py::at(self, 0))) {
        out.push_back("cap");
    }
    out.push_back("p1=" + encode(py::lower(p1)));
    out.push_back("p2=" + encode(py::lower(p2)));
    out.push_back("n1=" + encode(py::lower(n1)));
    out.push_back("n2=" + encode(py::lower(n2)));
    const Str target = py::lower(self);
    std::set<Str> nearby;                   // sorted by code point, as Python sorts strings
    for (long k = std::max(0L, index - kWindow); k < std::min(size, index + kWindow + 1); ++k) {
        const Str & token = tokens[static_cast<size_t>(k)];
        if (uni::is_alpha(py::at(token, 0))) {
            nearby.insert(py::lower(token));
        }
    }
    nearby.erase(target);
    for (const Str & word : nearby) {
        out.push_back("w=" + encode(word));
    }
    return out;
}

// The sentence around tokens[index] and the index inside it.
std::pair<std::vector<Str>, long> sentence_of(const std::vector<Str> & tokens, long index) {
    long start = index;
    while (start > 0 && !sentence_end(tokens[static_cast<size_t>(start - 1)])) {
        --start;
    }
    long end = index;
    const long size = static_cast<long>(tokens.size());
    while (end < size && !sentence_end(tokens[static_cast<size_t>(end)])) {
        ++end;
    }
    const long stop = std::min(size, end + 1);
    return {std::vector<Str>(tokens.begin() + start, tokens.begin() + stop), index - start};
}

bool skipped(const Str & klass, bool for_aux) {
    if (klass == U"ADV" || klass == U"NEG" || klass == U"LY") {
        return true;
    }
    return for_aux && (klass == U"SUBJ" || klass == U"OBJ" || klass == U"DET" || klass == U"POSS");
}

// The nearest token to the left that is not of a skipped class, in lower
// case, with "'ll" and "'ve" read as "will" and "have".
Str left(const std::vector<Str> & tokens, long index, bool for_aux) {
    for (long position = index - 1; position > std::max(index - 6, -1L); --position) {
        const Str & token = tokens[static_cast<size_t>(position)];
        if (skipped(tag(token), for_aux)) {
            continue;
        }
        const Str lower = py::lower(token);
        return py::endswith(lower, U"'ll") ? Str(U"will") : py::endswith(lower, U"'ve") ? Str(U"have") : lower;
    }
    return U"<s>";
}

bool holds(const std::string & condition, const std::unordered_set<Str> & words, const std::vector<Str> & tokens, long index, bool joined) {
    const long size = static_cast<long>(tokens.size());
    if (condition == "first") {
        return index == 0;
    }
    if (condition == "joined") {
        return joined && index > 0 && words.count(py::lower(tokens[static_cast<size_t>(index - 1)])) != 0;
    }
    if (condition == "aux") {
        return words.count(left(tokens, index, true)) != 0;
    }
    if (condition == "before") {
        return words.count(left(tokens, index, false)) != 0;
    }
    if (condition == "next") {
        return words.count(index + 1 < size ? py::lower(tokens[static_cast<size_t>(index + 1)]) : Str(U"</s>")) != 0;
    }
    for (long k = std::max(0L, index - kWindow); k < std::min(size, index + kWindow + 1); ++k) {
        if (words.count(py::lower(tokens[static_cast<size_t>(k)])) != 0) {
            return true;
        }
    }
    return false;
}

std::unordered_map<std::string, double> weights_of(const json::Value & value) {
    std::unordered_map<std::string, double> out;
    for (const auto & member : value.members) {
        out.emplace(member.first, member.second.number);
    }
    return out;
}

}  // namespace

double python_sum(const std::vector<double> & values) {
    if (values.empty()) {
        return 0.0;
    }
    double total = 0.0 + values[0];
    double carry = 0.0;
    for (size_t k = 1; k < values.size(); ++k) {
        // Neumaier's variant of Kahan summation, as CPython 3.12 has it
        const double x = values[k];
        const double t = total + x;
        if (std::fabs(total) >= std::fabs(x)) {
            carry += (total - t) + x;
        } else {
            carry += (x - t) + total;
        }
        total = t;
    }
    if (carry != 0.0 && std::isfinite(carry)) {
        total += carry;
    }
    return total;
}

Homographs::Homographs(const json::Value & data) {
    for (const auto & member : data.at("words").members) {
        Entry entry;
        for (const json::Value & sense : member.second.at("senses").items) {
            entry.senses.push_back(Sense{sense.at("id").string, sense.at("role").string, sense.at("phones").string});
        }
        const json::Value & model = member.second.at("model");
        entry.noun_verb = model.at("kind").string == "noun-verb";
        if (entry.noun_verb) {
            entry.bias = model.at("bias").number;
            entry.weights = weights_of(model.at("weights"));
        } else {
            for (const json::Value & bias : model.at("bias").items) {
                entry.biases.push_back(bias.number);
            }
            for (const json::Value & weights : model.at("weights").items) {
                entry.sense_weights.push_back(weights_of(weights));
            }
        }
        words_.emplace(decode(member.first), std::move(entry));
    }
    const json::Value & pooled = data.at("pooled");
    pooled_bias_ = pooled.at("bias").number;
    pooled_weights_ = weights_of(pooled.at("weights"));
    if (const json::Value * bases = data.find("plural_bases")) {
        for (const json::Value & base : bases->items) {
            plural_bases_.insert(decode(base.string));
        }
    }
    // The hand-written rules: word, sense, conditions separated by ";".
    for (const tables::Row & row : tables::rows("homograph_rules.txt")) {
        if (row.columns.size() != 3) {
            throw ValueError("homograph_rules.txt line " + std::to_string(row.number) + ": expected word, sense, conditions");
        }
        Rule rule;
        rule.sense = encode(row.columns[1]);
        for (const Str & part : py::split(row.columns[2], U";")) {
            const py::Parts parts = py::partition(py::strip(part), U":");
            const std::string name = encode(parts.head);
            const bool known = name == "aux" || name == "before" || name == "next" || name == "near" || name == "first" || name == "joined";
            if (!known || (name == "first") == !py::strip(parts.tail).empty()) {
                throw ValueError("homograph_rules.txt line " + std::to_string(row.number) + ": bad condition '" + encode(py::strip(part)) + "'");
            }
            std::unordered_set<Str> words;
            for (const Str & word : py::split(parts.tail)) {
                words.insert(word);
            }
            rule.conditions.emplace_back(name, std::move(words));
        }
        rules_[row.columns[0]].push_back(std::move(rule));
    }
    for (const auto & entry : rules_) {
        const auto known = words_.find(entry.first);
        for (const Rule & rule : entry.second) {
            bool found = false;
            if (known != words_.end()) {
                for (const Sense & sense : known->second.senses) {
                    found = found || sense.id == rule.sense;
                }
            }
            if (!found) {
                throw ValueError("homograph rule for '" + encode(entry.first) + "' names the unknown sense '" + rule.sense + "'");
            }
        }
    }
}

bool Homographs::covers(const Str & word) const {
    return words_.count(word) != 0 || (py::endswith(word, U"s") && plural_bases_.count(py::slice(word, 0, -1)) != 0);
}

Homographs::Spoken Homographs::pronounce(const Str & word, const std::vector<Str> & tokens, size_t index, bool joined) const {
    if (words_.count(word) != 0) {
        return {choose(word, tokens, index, joined).phones, false};
    }
    const Str base = py::slice(word, 0, -1);
    const auto found = words_.find(base);
    if (found == words_.end()) {
        throw KeyError(encode(base));
    }
    const Entry & entry = found->second;
    const Sense * noun = nullptr;
    for (const Sense & sense : entry.senses) {
        if (sense.role == "n") {
            noun = &sense;
            break;
        }
    }
    if (noun != nullptr && noun->id == "adj") {             // "separates" can only be the verb
        for (const Sense & sense : entry.senses) {
            if (sense.role == "v") {
                return {sense.phones, true};
            }
        }
        throw Error("StopIteration", "no verb sense");
    }
    return {choose(base, tokens, index, joined).phones, true};
}

std::vector<double> Homographs::scores(const Entry & entry, const std::vector<Str> & tokens, size_t index) const {
    const auto around = sentence_of(tokens, static_cast<long>(index));
    const std::vector<std::string> names = features(around.first, around.second);
    auto weight = [](const Weights & weights, const std::string & name) {
        const auto found = weights.find(name);
        return found == weights.end() ? 0.0 : found->second;
    };
    std::vector<double> out;
    if (entry.noun_verb) {
        std::vector<double> shared, own;
        for (const std::string & name : names) {
            if (name[0] == 't' || name == "first" || name == "cap") {
                shared.push_back(weight(pooled_weights_, name));
            }
            own.push_back(weight(entry.weights, name));
        }
        double verb = pooled_bias_ + python_sum(shared);
        verb += entry.bias + python_sum(own);
        for (const Sense & sense : entry.senses) {
            out.push_back(sense.role == "v" ? verb : -verb);
        }
        return out;
    }
    const size_t count = std::min(entry.biases.size(), entry.sense_weights.size());
    for (size_t k = 0; k < count; ++k) {
        std::vector<double> values;
        for (const std::string & name : names) {
            values.push_back(weight(entry.sense_weights[k], name));
        }
        out.push_back(entry.biases[k] + python_sum(values));
    }
    return out;
}

const Homographs::Sense & Homographs::choose(const Str & word, const std::vector<Str> & tokens, size_t index, bool joined) const {
    const Entry & entry = words_.at(word);
    const auto around = sentence_of(tokens, static_cast<long>(index));
    const auto rules = rules_.find(word);
    if (rules != rules_.end()) {
        for (const Rule & rule : rules->second) {
            bool all = true;
            for (const auto & condition : rule.conditions) {
                if (!holds(condition.first, condition.second, around.first, around.second, joined)) {
                    all = false;
                    break;
                }
            }
            if (all) {
                for (const Sense & sense : entry.senses) {
                    if (sense.id == rule.sense) {
                        return sense;
                    }
                }
            }
        }
    }
    const std::vector<double> values = scores(entry, tokens, index);
    size_t best = 0;
    for (size_t k = 1; k < values.size(); ++k) {
        if (values[k] > values[best]) {
            best = k;
        }
    }
    return entry.senses.at(best);
}

}  // namespace dizione
