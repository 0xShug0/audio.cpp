#include "tables.h"

#include "convert.h"
#include "pyre.h"

#include <cstddef>

namespace dizione::tables {

namespace {

struct EmbeddedTable {
    const char * name;
    const char * text;
    size_t size;
};

#include "tables_data.inc"

bool stress_class(StrView kind) {
    for (const StrView known : {StrView(U"u"), StrView(U"uf"), StrView(U"ur"), StrView(U"s"), StrView(U"ss"), StrView(U"si"),
                                StrView(U"pf"), StrView(U"p")}) {
        if (kind == known) {
            return true;
        }
    }
    return false;
}

std::string where(std::string_view name, long number) {
    return std::string(name) + " line " + std::to_string(number);
}

Pronunciation pronunciation(StrView written) {
    Pronunciation out;
    for (const Str & symbol : py::split(written)) {
        out.symbols.push_back(encode(symbol));
    }
    if (is_arpabet(out.symbols)) {
        out.arpabet = true;
        return out;
    }
    out.symbols.clear();
    out.phones = dizione::from_text(encode(written));
    return out;
}

// The rows of a source, each with the place to name in an error.
std::vector<std::pair<std::string, std::vector<Str>>> entries(const Source & source) {
    std::vector<std::pair<std::string, std::vector<Str>>> out;
    if (source.kind == Source::Pairs) {
        for (const auto & pair : source.pairs) {
            out.emplace_back(encode(pair.first), std::vector<Str>{pair.first, pair.second});
        }
        return out;
    }
    const std::vector<Row> found = source.kind == Source::BuiltIn ? rows(source.name) : rows_of(source.table_text);
    for (const Row & row : found) {
        out.emplace_back(std::to_string(row.number), row.columns);
    }
    return out;
}

}  // namespace

Str text(std::string_view name) {
    for (const EmbeddedTable & table : kEmbeddedTables) {
        if (name == table.name) {
            return decode(std::string_view(table.text, table.size));
        }
    }
    throw KeyError(std::string(name));
}

std::vector<Row> rows_of(StrView table_text) {
    static const pyre::Regex columns(R"re(\t+| {2,})re");
    std::vector<Row> out;
    long number = 0;
    for (const Str & raw : py::splitlines(table_text)) {
        ++number;
        const Str line = py::strip(raw);
        if (line.empty() || py::startswith(line, U"#")) {
            continue;
        }
        Row row;
        row.number = number;
        for (const Str & column : columns.split(line)) {
            row.columns.push_back(py::strip(column));
        }
        out.push_back(std::move(row));
    }
    return out;
}

std::vector<Row> rows(std::string_view name) {
    return rows_of(text(name));
}

std::unordered_map<Str, FunctionWord> function_words() {
    std::unordered_map<Str, FunctionWord> table;
    for (const Row & row : rows("function_words.txt")) {
        const auto & columns = row.columns;
        if ((columns.size() != 3 && columns.size() != 4) || !stress_class(columns[1])) {
            throw ValueError(where("function_words.txt", row.number) + ": expected word, class, form[, final form]");
        }
        FunctionWord word;
        word.kind = encode(columns[1]);
        word.form = dizione::from_text(encode(columns[2]));
        if (columns.size() == 4) {
            word.final_form = dizione::from_text(encode(columns[3]));
        }
        table[columns[0]] = std::move(word);
    }
    return table;
}

std::map<std::vector<Str>, Phrase> phrases() {
    std::map<std::vector<Str>, Phrase> table;
    for (const Row & row : rows("phrases.txt")) {
        const auto & columns = row.columns;
        if ((columns.size() != 3 && columns.size() != 4) || !stress_class(columns[1])) {
            throw ValueError(where("phrases.txt", row.number) + ": expected words, class, form[, final form]");
        }
        Phrase phrase;
        phrase.kind = encode(columns[1]);
        for (const Str & part : py::split(columns[2], U"|")) {
            phrase.forms.push_back(dizione::from_text(encode(part)));
        }
        const std::vector<Str> words = py::split(columns[0]);
        if (phrase.forms.size() > 1 && (phrase.kind != "p" || phrase.forms.size() != words.size())) {
            throw ValueError(where("phrases.txt", row.number) + ": a form with boundaries needs class p and one part per word");
        }
        if (columns.size() == 4) {
            phrase.final_form = dizione::from_text(encode(columns[3]));
        }
        table[words] = std::move(phrase);
    }
    return table;
}

Source Source::built_in(std::string_view name) {
    Source source;
    source.kind = BuiltIn;
    source.name = std::string(name);
    return source;
}

Source Source::from_text(StrView table_text) {
    Source source;
    source.kind = Text;
    source.table_text = Str(table_text);
    return source;
}

Source Source::from_pairs(std::vector<std::pair<Str, Str>> pairs) {
    Source source;
    source.kind = Pairs;
    source.pairs = std::move(pairs);
    return source;
}

std::unordered_map<Str, Pronunciation> pronunciations(const Source & source) {
    std::unordered_map<Str, Pronunciation> table;
    for (const auto & entry : entries(source)) {
        const auto & columns = entry.second;
        if (columns.size() != 2) {
            throw ValueError(entry.first + ": expected a word and its pronunciation, separated by a tab or two spaces");
        }
        if (py::split(columns[0]).size() == 1) {
            table[py::lower(columns[0])] = pronunciation(columns[1]);
        }
    }
    return table;
}

std::map<std::vector<Str>, std::vector<Pronunciation>> names(const Source & source) {
    std::map<std::vector<Str>, std::vector<Pronunciation>> table;
    for (const auto & entry : entries(source)) {
        const auto & columns = entry.second;
        const std::vector<Str> words = py::split(py::lower(columns.at(0)));
        if (columns.size() != 2 || words.size() < 2) {
            continue;
        }
        std::vector<Str> parts;
        for (const Str & part : py::split(columns[1], U"|")) {
            parts.push_back(py::strip(part));
        }
        if (parts.size() != words.size()) {
            throw ValueError(entry.first + ": '" + encode(columns[0]) + "' has " + std::to_string(words.size()) + " words and " +
                             std::to_string(parts.size()) + " pronunciations; write one per word, separated by |");
        }
        std::vector<Pronunciation> spoken;
        for (const Str & part : parts) {
            spoken.push_back(pronunciation(part));
        }
        table[words] = std::move(spoken);
    }
    return table;
}

std::unordered_map<Str, Str> mapping(std::string_view name) {
    std::unordered_map<Str, Str> table;
    for (const Row & row : rows(name)) {
        table[row.columns[0]] = row.columns.size() > 1 ? row.columns[1] : Str();
    }
    return table;
}

}  // namespace dizione::tables
