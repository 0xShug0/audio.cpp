#include "sentence.h"

namespace dizione {

Str Unit::last_word() const {
    const size_t space = word.rfind(U' ');
    return space == Str::npos ? word : word.substr(space + 1);
}

Str Unit::first_word() const {
    const size_t space = word.find(U' ');
    return space == Str::npos ? word : word.substr(0, space);
}

namespace sentence {

namespace {

// An r-coloured vowel without its r, empty when the phone is not one.
const char * split_vowel(const std::string & phone) {
    if (phone == "ɑːɹ") return "ɑː";
    if (phone == "ɔːɹ") return "ɔː";
    if (phone == "oːɹ") return "oː";
    if (phone == "ɪɹ") return "ɪ";
    if (phone == "ɛɹ") return "ɛ";
    if (phone == "ʊɹ") return "ʊ";
    return "";
}

bool rhotic(const std::string & phone) {
    return phone == "ɚ" || phone == "ɜː" || phone == "aɪɚ";
}

bool has_stress(const Phones & phones, int level) {
    for (const Phone & phone : phones) {
        if (phone.stress == level) {
            return true;
        }
    }
    return false;
}

}  // namespace

Phones unstress(const Phones & phones) {
    Phones out = phones;
    for (Phone & phone : out) {
        phone.stress = 0;
    }
    return out;
}

Phones promote(const Phones & phones) {
    if (has_stress(phones, 1)) {
        return phones;
    }
    long target = -1;
    if (has_stress(phones, 2)) {
        for (size_t i = 0; i < phones.size(); ++i) {
            if (phones[i].stress == 2) {
                target = static_cast<long>(i);          // the last one
            }
        }
    } else {
        for (size_t i = 0; i < phones.size(); ++i) {
            if (is_vowel(phones[i].phone)) {
                target = static_cast<long>(i);
                break;
            }
        }
    }
    Phones out = phones;
    if (target >= 0) {
        out[static_cast<size_t>(target)].stress = 1;
    }
    return out;
}

void stress_clause(std::vector<Unit> & units) {
    bool primary_follows = false;
    for (size_t index = units.size(); index-- > 0;) {
        Unit & unit = units[index];
        const bool last = index + 1 == units.size();
        auto promoted = [&]() { return unit.final_form.has_value() ? *unit.final_form : promote(unit.phones); };
        Phones phones;
        if (unit.kind == "u") {
            phones = unstress(unit.phones);
        } else if (unit.kind == "uf") {
            phones = last ? promoted() : unstress(unit.phones);
        } else if (unit.kind == "ur") {
            phones = primary_follows ? unstress(unit.phones) : promoted();
        } else if (unit.kind == "s") {
            phones = primary_follows ? unit.phones : promoted();
        } else if (unit.kind == "si") {
            phones = !primary_follows ? promoted() : index == 0 ? unit.phones : unstress(unit.phones);
        } else if (unit.kind == "pf") {
            phones = last ? promoted() : unit.phones;
        } else {                                        // "c", "p", "ss": as written
            phones = unit.phones;
        }
        unit.phones = std::move(phones);
        primary_follows = primary_follows || has_stress(unit.phones, 1);
    }
    bool any_stress = false;
    for (const Unit & unit : units) {
        for (const Phone & phone : unit.phones) {
            any_stress = any_stress || phone.stress != 0;
        }
    }
    if (!units.empty() && !any_stress) {
        for (size_t index = units.size(); index-- > 0;) {        // a clause is never without any stress
            Unit & unit = units[index];
            bool vowel = false;
            for (const Phone & phone : unit.phones) {
                vowel = vowel || is_vowel(phone.phone);
            }
            if (vowel) {
                unit.phones = promote(unit.phones);
                break;
            }
        }
    }
}

void join_clause(std::vector<Unit> & units) {
    for (size_t k = 0; k + 1 < units.size(); ++k) {
        Unit & unit = units[k];
        const Unit & following = units[k + 1];
        if (unit.phones.empty() || following.phones.empty() || !is_vowel(following.phones[0].phone)) {
            continue;
        }
        const std::string last = unit.phones.back().phone;
        const int stress = unit.phones.back().stress;
        const Str next_word = following.first_word();
        if (unit.last_word() == U"the" && last == "ə") {
            unit.phones.back() = Phone{"ɪ", stress};
        } else if (unit.last_word() == U"to" && last == "ə") {
            unit.phones.back() = Phone{"ʊ", stress};
        } else if (next_word == U"and" || next_word == U"or" || next_word == U"if") {
            continue;                                   // a slight break precedes these: nothing links or flaps into them
        } else if (*split_vowel(last) != '\0') {
            unit.phones.back() = Phone{split_vowel(last), stress};
            unit.phones.push_back(Phone{"ɹ", 0});
        } else if (rhotic(last)) {
            unit.phones.push_back(Phone{"ɹ", 0});
        } else if (last == "t" && (unit.word == U"it" || unit.word == U"at") && unit.phones.size() > 1 &&
                   is_vowel(unit.phones[unit.phones.size() - 2].phone)) {
            unit.phones.back() = Phone{"ɾ", 0};
        }
    }
}

std::vector<Phones> groups(const std::vector<Unit> & units) {
    std::vector<Phones> out;
    for (const Unit & unit : units) {
        if (unit.attach && !out.empty()) {
            out.back().insert(out.back().end(), unit.phones.begin(), unit.phones.end());
        } else {
            out.push_back(unit.phones);
        }
    }
    std::vector<Phones> kept;
    for (Phones & group : out) {
        if (!group.empty()) {
            kept.push_back(std::move(group));
        }
    }
    return kept;
}

}  // namespace sentence

}  // namespace dizione
