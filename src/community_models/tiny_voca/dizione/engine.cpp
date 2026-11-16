// The phonemizer: text -> spoken words of phones, clause by clause.
//
// Order of authority for one word:
//   1. pronunciation written inline in the text
//   2. the user's override list
//   3. the built-in exception list; a name of several words listed there or
//      by the user is spoken with the stress written
//   4. function words and function-word groups (sentence level)
//   5. the dictionary built from CMUdict
//   6. regular endings, prefixes and compounds of dictionary words
//   7. names in "-leigh" and words in "-virus", put together from their parts
//   8. the unknown-word model
#include "dizione.h"

#include "convert.h"
#include "morphology.h"
#include "pyre.h"

#include <algorithm>

namespace dizione {

namespace {

const pyre::Regex & leigh() {
    static const pyre::Regex value(R"re((?<=[a-z]{2})leigh(?=s?$))re");
    return value;
}

bool ends_in_stress_digit(const std::string & symbol) {
    return !symbol.empty() && (symbol.back() == '0' || symbol.back() == '1' || symbol.back() == '2');
}

std::vector<Str> texts(const std::vector<normalize::Token> & words, size_t index, size_t size) {
    std::vector<Str> key;
    for (size_t k = index; k < index + size; ++k) {
        key.push_back(words[k].text);
    }
    return key;
}

}  // namespace

Lexicon::Lexicon(std::string text) : text_(std::move(text)) {
    if (text_.size() > 0xFFFFFFFFULL) {
        throw ValueError("the dictionary (lexicon.tsv) is too large");
    }
    const std::string_view all(text_);
    size_t start = 0;
    while (start < all.size()) {
        size_t end = all.find('\n', start);
        if (end == std::string_view::npos) {
            end = all.size();
        }
        if (all.substr(start, end - start).find('\t') != std::string_view::npos) {
            lines_.push_back(static_cast<uint32_t>(start));
        }
        start = end + 1;
    }
    // Ordered by word; of two lines for one word the later in the file wins,
    // as it does when the lines fill a table one after another.
    std::stable_sort(lines_.begin(), lines_.end(), [this](uint32_t a, uint32_t b) { return word_at(a) < word_at(b); });
    size_t kept = 0;
    for (size_t i = 0; i < lines_.size(); ++i) {
        if (i + 1 < lines_.size() && word_at(lines_[i + 1]) == word_at(lines_[i])) {
            continue;
        }
        lines_[kept++] = lines_[i];
    }
    lines_.resize(kept);
    lines_.shrink_to_fit();
}

std::string_view Lexicon::word_at(uint32_t line) const {
    const std::string_view rest = std::string_view(text_).substr(line);
    return rest.substr(0, rest.find('\t'));
}

std::optional<std::string_view> Lexicon::find(std::string_view word) const {
    const auto found = std::lower_bound(
        lines_.begin(), lines_.end(), word, [this](uint32_t line, std::string_view key) { return word_at(line) < key; });
    if (found == lines_.end() || word_at(*found) != word) {
        return std::nullopt;
    }
    std::string_view value = std::string_view(text_).substr(*found + word.size() + 1);
    value = value.substr(0, value.find('\n'));
    if (!value.empty() && value.back() == '\r') {
        value.remove_suffix(1);
    }
    return value;
}

Phonemizer::Phonemizer(Data data, bool join_phrases) : lexicon_(std::move(data.lexicon)) {
    if (lexicon_.empty()) {
        throw ValueError("no phonemizer data: the dictionary (lexicon.tsv) is missing");
    }
    aligner_.emplace(json::parse(data.align));
    if (!data.g2p.empty()) {
        model_.emplace(json::parse(data.g2p), std::move(data.g2p_weights));
    }
    if (!data.homographs.empty()) {
        homographs_.emplace(json::parse(data.homographs));
    }
    functions_ = tables::function_words();
    if (join_phrases) {
        phrases_ = tables::phrases();
    }
    for (const auto & entry : phrases_) {
        phrase_length_ = std::max(phrase_length_, entry.first.size());
    }
    const tables::Source exceptions = tables::Source::built_in("exceptions.txt");
    exceptions_ = tables::pronunciations(exceptions);
    names_ = tables::names(exceptions);                 // names of several words, with their own rhythm
    refresh_names();
    for (const auto & entry : tables::mapping("letters.txt")) {
        letter_names_[entry.first] = from_text(encode(entry.second));
    }
    normalizer_ = std::make_unique<normalize::Normalizer>([this](const Str & word) { return is_word(word); });
}

void Phonemizer::refresh_names() {
    name_starts_.clear();
    name_length_ = 1;
    for (const auto & entry : names_) {
        name_starts_.insert(entry.first[0]);
        name_length_ = std::max(name_length_, entry.first.size());
    }
}

void Phonemizer::add_overrides(const tables::Source & source) {
    for (auto & entry : tables::pronunciations(source)) {
        overrides_[entry.first] = std::move(entry.second);
    }
    for (auto & entry : tables::names(source)) {
        user_names_[entry.first] = true;
        names_[entry.first] = std::move(entry.second);
    }
    refresh_names();
    cache_.clear();
}

// ---- single words -----------------------------------------------------------

Phones Phonemizer::entry(const Str & word, const tables::Pronunciation & pronunciation) const {
    if (!pronunciation.arpabet) {
        return pronunciation.phones;
    }
    return convert(word, pronunciation.symbols, aligner_->align(word, pronunciation.symbols));
}

// Pronunciation from the override list, the exceptions or the dictionary.
std::optional<Phones> Phonemizer::known(const Str & word) const {
    const auto override_ = overrides_.find(word);
    if (override_ != overrides_.end()) {
        return entry(word, override_->second);
    }
    const auto exception = exceptions_.find(word);
    if (exception != exceptions_.end()) {
        return entry(word, exception->second);
    }
    const auto found = lexicon_.find(encode(word));
    if (!found.has_value()) {
        return std::nullopt;
    }
    return from_text(*found);
}

// Pronunciation of a dictionary word that may be half of a compound. CMUdict
// is full of surnames; a word that is also listed with a regular ending
// ("webs", "screens") is an ordinary one.
std::optional<Phones> Phonemizer::ordinary(const Str & word) const {
    if (word.size() < 3) {
        return std::nullopt;
    }
    const std::string key = encode(word);
    bool listed = false;
    for (const char * ending : {"s", "es", "ed", "ing"}) {
        listed = listed || lexicon_.contains(key + ending);
    }
    return listed ? known(word) : std::nullopt;
}

// The unknown-word model, then the conversion rules, on one spelling.
Phones Phonemizer::guess(const Str & word) const {
    if (!model_) {
        return spell(word);
    }
    const Str spelling = model_->spellable(word);
    std::vector<std::string> symbols = model_->predict(spelling);
    // The model emits each phoneme on its own, and between two possible vowels before
    // an r it sometimes emits neither ("okafor" -> ... F R). An r with no vowel beside
    // it is a syllable: write it as one.
    for (size_t k = 0; k < symbols.size(); ++k) {
        if (symbols[k] != "R") {
            continue;
        }
        const bool before = k > 0 && ends_in_stress_digit(symbols[k - 1]);
        const bool after = k + 1 < symbols.size() && ends_in_stress_digit(symbols[k + 1]);
        if (!before && !after) {
            symbols[k] = "ER0";
        }
    }
    bool vowel = false;
    for (const std::string & symbol : symbols) {
        vowel = vowel || ends_in_stress_digit(symbol);
    }
    if (!vowel) {
        return spell(word);
    }
    // Every word spoken alone has a main stress, even when the model marked none.
    return sentence::promote(convert(spelling, symbols, aligner_->align(spelling, symbols)));
}

// Letter names, joined: the last letter carries the main stress.
Phones Phonemizer::spell(const Str & letters) const {
    const bool possessive = py::endswith(py::lower(letters), U"'s");
    std::vector<const Phones *> names;
    for (const char32_t c : py::lower(possessive ? py::slice(letters, 0, -2) : letters)) {
        const auto found = letter_names_.find(Str(1, c));
        if (found != letter_names_.end()) {
            names.push_back(&found->second);
        }
    }
    Phones out;
    for (size_t index = 0; index < names.size(); ++index) {
        for (const Phone & phone : *names[index]) {
            out.push_back(Phone{phone.phone, phone.stress != 0 && index + 1 < names.size() ? 2 : phone.stress});
        }
    }
    return possessive && !out.empty() ? morphology::plural(out) : out;
}

bool Phonemizer::is_word(const Str & word) const {
    return lexicon_.contains(encode(word)) || functions_.count(word) != 0 || exceptions_.count(word) != 0 ||
        overrides_.count(word) != 0;
}

// (phones, source) for two kinds of new word the model reads badly, or
// nothing. A name in "-leigh" is said like the same name in "-ley" (Kayleigh,
// Ryleigh). A word in "-virus" is its first part, from the dictionary or the
// model, and then "virus" (coronavirus).
std::optional<std::pair<Phones, std::string>> Phonemizer::from_parts(const Str & spelling) {
    if (py::endswith(spelling, U"'s") && spelling.size() > 3) {
        auto found = from_parts(py::slice(spelling, 0, -2));
        if (!found) {
            return std::nullopt;
        }
        return std::make_pair(morphology::plural(found->first), found->second);
    }
    if (leigh().search(spelling)) {
        return word(leigh().sub(U"ley", spelling));
    }
    for (const StrView tail : {StrView(U"viruses"), StrView(U"virus")}) {
        if (!(py::endswith(spelling, tail) && spelling.size() > tail.size() + 2)) {
            continue;
        }
        const std::optional<Phones> second = known(Str(tail));
        if (second) {
            const auto first = word(spelling.substr(0, spelling.size() - tail.size()));
            Phones phones = sentence::promote(settle_stress(morphology::compound(first.first, *second)));
            return std::make_pair(std::move(phones), std::string(first.second == "model" ? "model" : "ending"));
        }
    }
    return std::nullopt;
}

std::pair<Phones, std::string> Phonemizer::word(const Str & spelling) {
    const auto cached = cache_.find(spelling);
    if (cached != cache_.end()) {
        return cached->second;
    }
    std::optional<Phones> phones = known(spelling);
    std::string source = "dictionary";
    if (!phones) {
        phones = morphology::derive(
            spelling, [this](const Str & w) { return known(w); }, [this](const Str & w) { return ordinary(w); });
        source = "ending";
        if (phones) {
            phones = settle_stress(*phones);
        }
    }
    if (!phones) {
        if (auto found = from_parts(spelling)) {
            phones = std::move(found->first);
            source = found->second;
        }
    }
    if (!phones && py::endswith(spelling, U"'s") && spelling.size() > 3) {
        phones = morphology::plural(guess(py::slice(spelling, 0, -2)));
        source = "model";
    }
    if (!phones) {
        phones = guess(spelling);
        source = "model";
    }
    auto result = std::make_pair(std::move(*phones), source);
    cache_[spelling] = result;
    return result;
}

// ---- sentences ----------------------------------------------------------------

size_t Phonemizer::name_size(const std::vector<Token> & words, size_t index) const {
    for (size_t size = std::min(name_length_, words.size() - index); size > 1; --size) {
        const std::vector<Str> key = texts(words, index, size);
        if (names_.count(key) == 0) {
            continue;
        }
        bool plain = true;
        bool overridden = false;
        for (size_t k = index; k < index + size; ++k) {
            plain = plain && !words[k].phones.has_value() && !words[k].spell && (k == index || !words[k].attach);
            overridden = overridden || overrides_.count(words[k].text) != 0;
        }
        if (plain && (user_names_.count(key) != 0 || !overridden)) {
            return size;                                // the user's own word list wins over a built-in name
        }
    }
    return 0;
}

size_t Phonemizer::phrase_size(const std::vector<Token> & words, size_t index) const {
    for (size_t size = std::min(phrase_length_, words.size() - index); size > 1; --size) {
        bool plain = !words[index].phones.has_value() && !words[index].spell;
        bool overridden = false;
        for (size_t k = index; k < index + size; ++k) {
            if (k != index) {
                plain = plain && !words[k].phones.has_value() && !words[k].spell && !words[k].attach;
            }
            overridden = overridden || overrides_.count(words[k].text) != 0;
        }
        if (plain && !overridden && phrases_.count(texts(words, index, size)) != 0) {
            return size;
        }
    }
    return 0;
}

// Units of one clause from its tokens. `written` is the whole text as tokens
// and `places` where each word sits in it, for the homograph rules.
std::vector<Unit> Phonemizer::units(const std::vector<Token> & words, const std::vector<size_t> & places, const std::vector<Str> & written) {
    std::vector<Unit> out;
    auto unit = [](Str word, Phones phones, std::string kind, bool attach, std::string source) {
        Unit made;
        made.word = std::move(word);
        made.phones = std::move(phones);
        made.kind = std::move(kind);
        made.attach = attach;
        made.source = std::move(source);
        return made;
    };
    size_t index = 0;
    while (index < words.size()) {
        const Token & token = words[index];
        if (token.phones.has_value()) {
            out.push_back(unit(token.text, *token.phones, "p", token.attach, "inline"));
        } else if (token.spell) {
            out.push_back(unit(token.text, spell(token.text), "p", token.attach, "letters"));
        } else {
            size_t size = name_starts_.count(token.text) != 0 ? name_size(words, index) : 0;
            if (size != 0) {                            // a taught name: every word as written in the table
                const std::vector<Str> key = texts(words, index, size);
                const auto & spoken = names_.at(key);
                for (size_t position = 0; position < key.size() && position < spoken.size(); ++position) {
                    out.push_back(unit(key[position], entry(key[position], spoken[position]), "p", token.attach && position == 0, "name"));
                }
                index += size;
                continue;
            }
            size = phrase_size(words, index);
            if (size != 0) {
                const std::vector<Str> key = texts(words, index, size);
                const tables::Phrase & phrase = phrases_.at(key);
                if (phrase.forms.size() == 1) {
                    Unit made = unit(py::join(U" ", key), phrase.forms[0], phrase.kind, token.attach, "phrase");
                    made.final_form = phrase.final_form;
                    out.push_back(std::move(made));
                } else {                                // the boundaries stay, only the stress is fixed
                    for (size_t position = 0; position < key.size() && position < phrase.forms.size(); ++position) {
                        out.push_back(unit(key[position], phrase.forms[position], phrase.kind, token.attach && position == 0, "phrase"));
                    }
                }
                index += size;
                continue;
            }
            const auto function = functions_.find(token.text);
            if (overrides_.count(token.text) != 0) {
                out.push_back(unit(token.text, *known(token.text), "c", token.attach, "override"));
            } else if (homographs_ && !written.empty() && homographs_->covers(token.text) && exceptions_.count(token.text) == 0) {
                const Homographs::Spoken spoken = homographs_->pronounce(token.text, written, places[index], token.attach);
                Phones phones = from_text(spoken.phones);
                if (spoken.plural) {
                    phones = morphology::plural(phones);
                }
                out.push_back(unit(token.text, std::move(phones), "c", token.attach, "homograph"));
            } else if (function != functions_.end()) {
                Unit made = unit(token.text, function->second.form, function->second.kind, token.attach, "function");
                made.final_form = function->second.final_form;
                out.push_back(std::move(made));
            } else {
                auto found = word(token.text);
                out.push_back(unit(token.text, std::move(found.first), "c", token.attach, found.second));
            }
        }
        ++index;
    }
    return out;
}

std::vector<Phonemizer::Clause> Phonemizer::analyse(const Str & text) {
    const std::vector<Token> stream = normalizer_->tokens(text);
    // The text as the homograph rules see it; a word given only as phones stands as "x".
    std::vector<Str> written;
    for (const Token & token : stream) {
        written.push_back(token.punct ? token.mark : !token.raw.empty() ? token.raw : !token.text.empty() ? token.text : Str(U"x"));
    }
    std::vector<Clause> out;
    std::vector<Token> words;
    std::vector<size_t> places;
    auto flush = [&]() {
        if (words.empty()) {
            return;
        }
        Clause clause;
        clause.units = units(words, places, written);
        sentence::stress_clause(clause.units);
        sentence::join_clause(clause.units);
        out.push_back(std::move(clause));
        words.clear();
        places.clear();
    };
    for (size_t place = 0; place < stream.size(); ++place) {
        if (stream[place].punct) {
            flush();
            Clause mark;
            mark.punctuation = true;
            mark.mark = stream[place].mark;
            out.push_back(std::move(mark));
        } else {
            words.push_back(stream[place]);
            places.push_back(place);
        }
    }
    flush();
    return out;
}

std::vector<Piece> Phonemizer::pieces(const Str & text) {
    std::vector<Piece> out;
    for (Clause & clause : analyse(text)) {
        Piece piece;
        piece.punctuation = clause.punctuation;
        if (clause.punctuation) {
            piece.mark = encode(clause.mark);
        } else {
            piece.words = sentence::groups(clause.units);
        }
        out.push_back(std::move(piece));
    }
    return out;
}

Phonemized Phonemizer::phonemize(std::string_view utf8_text) {
    const Str text = decode(utf8_text);
    if (py::strip(text).empty()) {
        throw ValueError("text must not be empty");
    }
    return tokenize(pieces(text));
}

}  // namespace dizione
