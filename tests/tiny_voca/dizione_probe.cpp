// The C++ side of the checks for the dizione port. Each command reads what
// reference.py wrote and compares it with what the C++ code gives, or writes
// the C++ answer in the same form for a plain file comparison.
//
//   dizione_probe unicode CASES
//   dizione_probe regex CASES [--patterns]
//
// It needs nothing but the dizione sources.
#include "dizione.h"
#include "json.h"
#include "pyre.h"
#include "text.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace dizione;

namespace {

std::vector<std::string> fields(const std::string & line) {
    std::vector<std::string> out;
    size_t start = 0;
    for (;;) {
        const size_t tab = line.find('\t', start);
        if (tab == std::string::npos) {
            out.push_back(line.substr(start));
            return out;
        }
        out.push_back(line.substr(start, tab - start));
        start = tab + 1;
    }
}

Str from_codes(const std::string & text) {
    Str out;
    std::istringstream stream(text);
    std::string item;
    while (stream >> item) {
        out.push_back(static_cast<char32_t>(std::stoul(item, nullptr, 16)));
    }
    return out;
}

std::string to_codes(StrView text) {
    std::string out;
    char buffer[16];
    for (size_t i = 0; i < text.size(); ++i) {
        std::snprintf(buffer, sizeof(buffer), i == 0 ? "%X" : " %X", static_cast<unsigned>(text[i]));
        out += buffer;
    }
    return out;
}

int unicode(const char * path) {
    std::ifstream in(path);
    std::string line;
    long cases = 0;
    long wrong = 0;
    while (std::getline(in, line)) {
        const auto f = fields(line);
        if (f.size() < 3) {
            continue;
        }
        ++cases;
        std::string got;
        const Str text = from_codes(f[1]);
        switch (f[0][0]) {
            case 'P': {
                const char32_t c = text[0];
                const Str one(1, c);
                const unsigned bits = (uni::is_alpha(c) ? 1u : 0u) | (uni::is_decimal(c) ? 2u : 0u) | (uni::is_digit(c) ? 4u : 0u) |
                    (uni::is_numeric(c) ? 8u : 0u) | (uni::is_space(c) ? 16u : 0u) | (py::isupper(one) ? 32u : 0u) |
                    (py::islower(one) ? 64u : 0u) | (uni::is_alnum(c) ? 128u : 0u) | (uni::is_word(c) ? 256u : 0u) |
                    (uni::is_decimal(c) ? 512u : 0u) | (uni::is_space(c) ? 1024u : 0u);
                got = std::to_string(bits);
                if (f.size() > 3 && std::to_string(uni::combining(c)) != f[3]) {
                    got += " combining " + std::to_string(uni::combining(c));
                }
                break;
            }
            case 'L': got = to_codes(py::lower(text)); break;
            case 'U': got = to_codes(py::upper(text)); break;
            case 'C': got = to_codes(uni::nfkc(text)); break;
            case 'D': got = to_codes(uni::nfkd(text)); break;
            case 'u': got = py::isupper(text) ? "1" : "0"; break;
            case 'l': got = py::islower(text) ? "1" : "0"; break;
            case 'T': got = to_codes(py::strip(text)); break;
            case 'S': {
                const auto parts = py::split(text);
                for (size_t i = 0; i < parts.size(); ++i) {
                    got += (i ? "|" : "") + to_codes(parts[i]);
                }
                break;
            }
            case 'I':
                try {
                    got = std::to_string(py::to_int(text));
                } catch (const Error & error) {
                    got = error.kind;
                }
                break;
            default: continue;
        }
        if (got != f[2]) {
            if (++wrong <= 40) {
                std::cout << f[0] << "\t" << f[1] << "\texpected " << f[2] << "\tgot " << got << "\n";
            }
        }
    }
    std::cout << cases << " cases, " << wrong << " wrong\n";
    return wrong == 0 ? 0 : 1;
}

json::Value spans_of(const pyre::Match & found) {
    json::Value out;
    if (!found) {
        return out;
    }
    out.kind = json::Value::Array;
    for (int group = 0; group <= found.groups(); ++group) {
        json::Value span;
        span.kind = json::Value::Array;
        for (const long value : {found.start(group), found.end(group)}) {
            json::Value number;
            number.kind = json::Value::Int;
            number.integer = value;
            span.items.push_back(number);
        }
        out.items.push_back(span);
    }
    return out;
}

bool same(const json::Value & a, const json::Value & b) {
    if (a.kind != b.kind) {
        return false;
    }
    switch (a.kind) {
        case json::Value::Int: return a.integer == b.integer;
        case json::Value::String: return a.string == b.string;
        case json::Value::Array:
            if (a.items.size() != b.items.size()) {
                return false;
            }
            for (size_t i = 0; i < a.items.size(); ++i) {
                if (!same(a.items[i], b.items[i])) {
                    return false;
                }
            }
            return true;
        default: return true;
    }
}

std::string show(const json::Value & value) {
    switch (value.kind) {
        case json::Value::Null: return "null";
        case json::Value::Int: return std::to_string(value.integer);
        case json::Value::String: return "\"" + value.string + "\"";
        case json::Value::Array: {
            std::string out = "[";
            for (size_t i = 0; i < value.items.size(); ++i) {
                out += (i ? "," : "") + show(value.items[i]);
            }
            return out + "]";
        }
        default: return "?";
    }
}

int regex(const char * path) {
    std::ifstream in(path);
    std::string line;
    std::map<std::string, pyre::Regex> compiled;
    std::map<std::string, std::string> broken;
    long cases = 0;
    long wrong = 0;
    while (std::getline(in, line)) {
        const json::Value item = json::parse(line);
        const std::string & pattern = item.at("p").string;
        const int flags = static_cast<int>(item.at("f").integer);
        const std::string key = std::to_string(flags) + ":" + pattern;
        ++cases;
        if (broken.count(key)) {
            ++wrong;
            continue;
        }
        auto found = compiled.find(key);
        if (found == compiled.end()) {
            try {
                found = compiled.emplace(key, pyre::Regex(pattern, flags)).first;
            } catch (const Error & error) {
                broken[key] = error.what();
                std::cout << "COMPILE\t" << pattern << "\t" << error.what() << "\n";
                ++wrong;
                continue;
            }
        }
        const pyre::Regex & re = found->second;
        const Str text = decode(item.at("s").string);
        std::string what;
        json::Value got, want;
        try {
            json::Value all;
            all.kind = json::Value::Array;
            for (const auto & m : re.finditer(text)) {
                all.items.push_back(spans_of(m));
            }
            if (!same(all, item.at("all"))) {
                what = "finditer";
                got = all;
                want = item.at("all");
            }
            if (what.empty()) {
                const json::Value first = spans_of(re.match(text));
                if (!same(first, item.at("match"))) {
                    what = "match";
                    got = first;
                    want = item.at("match");
                }
            }
            if (what.empty()) {
                const json::Value whole = spans_of(re.fullmatch(text));
                if (!same(whole, item.at("full"))) {
                    what = "fullmatch";
                    got = whole;
                    want = item.at("full");
                }
            }
            if (what.empty() && item.at("split").kind == json::Value::Array) {
                json::Value pieces;
                pieces.kind = json::Value::Array;
                for (const Str & piece : re.split(text)) {
                    json::Value s;
                    s.kind = json::Value::String;
                    s.string = encode(piece);
                    pieces.items.push_back(s);
                }
                if (!same(pieces, item.at("split"))) {
                    what = "split";
                    got = pieces;
                    want = item.at("split");
                }
            }
        } catch (const Error & error) {
            what = std::string("error ") + error.kind;
        }
        if (!what.empty()) {
            if (++wrong <= 60) {
                std::cout << what << "\t" << pattern << "\tflags " << flags << "\ttext " << item.at("s").string << "\n\texpected "
                          << show(want) << "\n\tgot      " << show(got) << "\n";
            }
        }
    }
    std::cout << cases << " cases, " << compiled.size() + broken.size() << " patterns, " << broken.size() << " not compiled, "
              << wrong << " wrong\n";
    return wrong == 0 ? 0 : 1;
}

std::string show_phones(const Phones & phones) {
    std::string out;
    for (size_t i = 0; i < phones.size(); ++i) {
        out += (i ? " " : "") + phones[i].phone + "/" + std::to_string(phones[i].stress);
    }
    return out;
}

std::string join_numbers(const std::vector<int> & values) {
    std::string out;
    for (size_t i = 0; i < values.size(); ++i) {
        out += (i ? " " : "") + std::to_string(values[i]);
    }
    return out;
}

std::vector<std::string> lines_of(const char * path) {
    std::ifstream in(path);
    std::vector<std::string> out;
    std::string line;
    while (std::getline(in, line)) {
        out.push_back(line);
    }
    return out;
}

// One line of output for one line of input, in the form reference.py writes.
std::string answer(Phonemizer & phonemizer, const std::string & command, const std::string & line) {
    try {
        if (command == "ids") {
            const Phonemized result = phonemizer.phonemize(line);
            return join_numbers(result.ids) + "\t" + join_numbers(result.stress);
        }
        if (command == "rewrite") {
            return encode(phonemizer.normalizer().rewrite(decode(line)));
        }
        if (command == "words") {
            const auto found = phonemizer.word(decode(line));
            return found.second + "\t" + show_phones(found.first);
        }
        std::string out;
        if (command == "tokens") {
            bool first = true;
            for (const normalize::Token & token : phonemizer.normalizer().tokens(decode(line))) {
                out += first ? "" : "\x1f";
                first = false;
                if (token.punct) {
                    out += "P:" + encode(token.mark);
                } else {
                    out += "W:" + encode(token.text) + "|" + (token.attach ? "1" : "0") + "|" + encode(token.raw) + "|" +
                        (token.spell ? "1" : "0") + "|" + (token.phones ? show_phones(*token.phones) : std::string("-"));
                }
            }
            return out;
        }
        bool first = true;
        for (const Phonemizer::Clause & clause : phonemizer.analyse(decode(line))) {
            if (clause.punctuation) {
                out += std::string(first ? "" : "\x1f") + "P:" + encode(clause.mark);
                first = false;
                continue;
            }
            for (const Unit & unit : clause.units) {
                out += std::string(first ? "" : "\x1f") + "U:" + encode(unit.word) + "|" + unit.kind + "|" + (unit.attach ? "1" : "0") + "|" +
                    unit.source + "|" + show_phones(unit.phones);
                first = false;
            }
        }
        return out;
    } catch (const Error & error) {
        return std::string("ERROR ") + error.kind;
    } catch (const std::exception & error) {
        return std::string("ERROR std ") + error.what();
    }
}

int phonemes(const std::string & command, const char * data, const char * texts, const char * out_path, const char * patterns_path) {
    if (patterns_path != nullptr) {
        pyre::record_patterns(true);
    }
    Phonemizer phonemizer(load_directory(data));
    std::ofstream out(out_path);
    for (const std::string & line : lines_of(texts)) {
        out << answer(phonemizer, command, line) << "\n";
    }
    if (patterns_path != nullptr) {
        std::ofstream listing(patterns_path);
        for (const auto & pattern : pyre::recorded_patterns()) {
            listing << pattern.second << "\t" << pattern.first << "\n";
        }
    }
    return 0;
}

int bench(const char * data, const char * texts, int rounds) {
    using Clock = std::chrono::steady_clock;
    const auto seconds = [](Clock::time_point from) { return std::chrono::duration<double>(Clock::now() - from).count(); };
    auto start = Clock::now();
    Phonemizer phonemizer(load_directory(data));
    std::cout << "load " << seconds(start) << " s\n";
    const std::vector<std::string> lines = lines_of(texts);
    size_t characters = 0;
    for (const std::string & line : lines) {
        characters += decode(line).size();
    }
    for (int round = 0; round < rounds; ++round) {
        start = Clock::now();
        size_t tokens = 0;
        for (const std::string & line : lines) {
            try {
                tokens += phonemizer.phonemize(line).ids.size();
            } catch (const Error &) {
            }
        }
        const double took = seconds(start);
        std::cout << "round " << round << ": " << lines.size() << " texts, " << characters << " characters, " << tokens << " tokens in "
                  << took << " s: " << static_cast<double>(lines.size()) / took << " texts/s\n";
    }
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::cerr << "usage: dizione_probe unicode|regex CASES\n"
                     "       dizione_probe ids|tokens|units|words|rewrite DATA TEXTS OUT [PATTERNS]\n"
                     "       dizione_probe bench DATA TEXTS [ROUNDS]\n";
        return 2;
    }
    const std::string command = argv[1];
    try {
        if (command == "unicode") {
            return unicode(argv[2]);
        }
        if (command == "regex") {
            return regex(argv[2]);
        }
        if (command == "bench" && argc >= 4) {
            return bench(argv[2], argv[3], argc > 4 ? std::stoi(argv[4]) : 3);
        }
        if ((command == "ids" || command == "tokens" || command == "units" || command == "words" || command == "rewrite") && argc >= 5) {
            return phonemes(command, argv[2], argv[3], argv[4], argc > 5 ? argv[5] : nullptr);
        }
    } catch (const std::exception & error) {
        std::cerr << "dizione_probe: " << error.what() << "\n";
        return 3;
    }
    std::cerr << "unknown command " << command << "\n";
    return 2;
}
