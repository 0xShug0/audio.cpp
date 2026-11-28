#include "symbols.h"

#include "text.h"

#include <unordered_map>
#include <unordered_set>

namespace dizione {

namespace {

// "n" with the syllabic mark below it (U+0329), written in bytes so that the
// mark cannot be lost in an editor.
constexpr const char * kSyllabicN = "n\xCC\xA9";

struct Inventory {
    std::vector<std::string> all;
    std::unordered_map<std::string, int> ids;
    std::unordered_set<std::string> phones;
    std::unordered_set<std::string> vowels;
    std::unordered_set<std::string> punctuation;

    Inventory() {
        const std::vector<std::string> specials = {kPad, kBos, kEos, kSp, kUnk};
        const std::vector<std::string> marks = {",", ".", "?", "!", ";", ":", "\"", "'", "(", ")", "—", "…"};
        const std::vector<std::string> consonants = {
            "p", "b", "t", "d", "k", "ɡ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
            "m", "n", "ŋ", "l", "ɹ", "j", "w", "tʃ", "dʒ", "ɾ", "ʔ",
        };
        const std::vector<std::string> vowel_phones = {
            "ə", "ɐ", "ɚ", "ᵻ", "ɪ", "i", "iː", "ɛ", "æ", "ʌ", "ɑː", "ɔ", "ɔː", "ʊ", "uː", "ɜː",
            "eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ", "iə", "oː",
            "ɑːɹ", "ɔːɹ", "oːɹ", "ɪɹ", "ɛɹ", "ʊɹ", "aɪɚ", "aɪə", "əl", kSyllabicN,
        };
        for (const auto * group : {&specials, &marks, &consonants, &vowel_phones}) {
            all.insert(all.end(), group->begin(), group->end());
        }
        for (size_t i = 0; i < all.size(); ++i) {
            ids.emplace(all[i], static_cast<int>(i));
        }
        phones.insert(consonants.begin(), consonants.end());
        phones.insert(vowel_phones.begin(), vowel_phones.end());
        vowels.insert(vowel_phones.begin(), vowel_phones.end());
        punctuation.insert(marks.begin(), marks.end());
    }
};

const Inventory & inventory() {
    static const Inventory value;
    return value;
}

}  // namespace

const std::vector<std::string> & symbols() {
    return inventory().all;
}

int symbol_id(std::string_view symbol) {
    const auto & ids = inventory().ids;
    const auto found = ids.find(std::string(symbol));
    return found == ids.end() ? -1 : found->second;
}

bool is_phone(std::string_view symbol) {
    return inventory().phones.count(std::string(symbol)) != 0;
}

bool is_vowel(std::string_view symbol) {
    return inventory().vowels.count(std::string(symbol)) != 0;
}

bool is_punctuation(std::string_view symbol) {
    return inventory().punctuation.count(std::string(symbol)) != 0;
}

Phonemized tokenize(const std::vector<Piece> & pieces) {
    Phonemized out;
    out.symbols.push_back(kBos);
    out.stress.push_back(0);
    for (const Piece & piece : pieces) {
        if (piece.punctuation) {
            if (out.symbols.back() != piece.mark) {     // repeated punctuation is collapsed
                out.symbols.push_back(piece.mark);
                out.stress.push_back(0);
            }
            continue;
        }
        for (const Phones & word : piece.words) {
            if (is_phone(out.symbols.back()) || out.symbols.back() == kUnk) {
                out.symbols.push_back(kSp);
                out.stress.push_back(0);
            }
            for (const Phone & phone : word) {
                if (is_phone(phone.phone)) {
                    out.symbols.push_back(phone.phone);
                    out.stress.push_back(phone.stress);
                } else {
                    out.unknown.push_back(phone.phone);
                    out.symbols.push_back(kUnk);
                    out.stress.push_back(0);
                }
            }
        }
    }
    out.symbols.push_back(kEos);
    out.stress.push_back(0);
    bool spoken = false;
    for (const auto & symbol : out.symbols) {
        spoken = spoken || is_phone(symbol) || symbol == kUnk;
    }
    if (!spoken) {
        throw ValueError("the text has nothing to pronounce");
    }
    out.ids.reserve(out.symbols.size());
    for (const auto & symbol : out.symbols) {
        out.ids.push_back(symbol_id(symbol));
    }
    return out;
}

}  // namespace dizione
