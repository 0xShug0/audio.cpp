// Word-timestamp helpers for sanoTTS: word labels come from the chunk text,
// phoneme groups end at spaces and word-end marks, and words whose counts
// don't match the groups are aligned by their own phonemes. Uses a fake
// symbol table and a fake per-word phonemizer, so no model files are needed.

#include "word_timestamps.h"

#include "test_assert.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

using engine::models::sanotts::word_timestamps::label_ranges;
using engine::models::sanotts::word_timestamps::LabeledRange;
using engine::models::sanotts::word_timestamps::PhonemeGroup;
using engine::models::sanotts::word_timestamps::phoneme_word_ranges;
using engine::models::sanotts::word_timestamps::text_words;
using engine::models::sanotts::word_timestamps::WordPhonemizer;

// Each symbol is one token of 2 frames. Token id 0 is a special with no symbol.
struct Tokens {
    std::map<int32_t, std::string> symbols;
    std::vector<int32_t> ids;
    std::vector<int64_t> frames;

    explicit Tokens(const std::vector<std::string> & syms) {
        int32_t next = 1;
        for (const auto & s : syms) {
            symbols[next] = s;
            ids.push_back(next++);
            frames.push_back(2);
        }
    }
    std::string lookup(int32_t id) const {
        const auto found = symbols.find(id);
        return found == symbols.end() ? std::string{} : found->second;
    }
};

// Splits a stream written as "wʌzə_sʌn" into one symbol per code point,
// with "_" meaning a space.
std::vector<std::string> symbols_of(const std::string & stream) {
    std::vector<std::string> out;
    for (size_t i = 0; i < stream.size();) {
        const auto lead = static_cast<unsigned char>(stream[i]);
        const size_t len = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
        std::string ch = stream.substr(i, len);
        out.push_back(ch == "_" ? " " : ch);
        i += len;
    }
    return out;
}

std::vector<PhonemeGroup> groups_of(const Tokens & tokens) {
    return phoneme_word_ranges(
        tokens.ids, tokens.frames, [&tokens](int32_t id) { return tokens.lookup(id); });
}

// Phonemizer that knows a fixed table; unknown words give nothing.
WordPhonemizer table(std::map<std::string, std::string> entries) {
    return [entries](const std::string & word) {
        const auto found = entries.find(word);
        return found == entries.end() ? std::string{} : found->second;
    };
}

std::vector<std::string> labels_of(const std::vector<LabeledRange> & labeled) {
    std::vector<std::string> out;
    for (const auto & item : labeled) {
        out.push_back(item.label);
    }
    return out;
}

void expect_labels(const std::string & name, const std::string & text,
                   const std::vector<std::string> & syms, const std::vector<std::string> & expect) {
    Tokens tokens(syms);
    const auto got = labels_of(label_ranges(groups_of(tokens), text_words(text), table({})));
    engine::test::require(got == expect, name + ": label mismatch");
    std::cout << "PASS " << name << "\n";
}

}  // namespace

int main() try {
    // Words split on punctuation with or without a space after it.
    expect_labels("comma and space", "Hello, world!",
                  {"h", "e", "l", "o", ",", " ", "w", "o", "r", "l", "d", "!"},
                  {"Hello", "world"});
    expect_labels("comma without space", "Hello,world!",
                  {"h", "e", "l", "o", ",", "w", "o", "r", "l", "d", "!"},
                  {"Hello", "world"});

    // Apostrophes and hyphens stay inside a word.
    expect_labels("apostrophe and hyphen", "don't well-known",
                  {"d", "o", "n", "'", "t", " ", "w", "e", "l", "-", "n", "o", "w", "n"},
                  {"don't", "well-known"});

    // Curly quotes and the em dash end words.
    expect_labels("curly quotes and dash", "“Hi”—there",
                  {"h", "i", "”", "—", "t", "h", "e", "r", "e"},
                  {"Hi", "there"});

    // Espeak merged "was a" into one group (no space). The two words share the
    // group's time, split by phoneme count: "was" gets 3 of its 4 chars.
    {
        Tokens tokens(symbols_of("helo_wʌzə_sʌn"));
        const auto groups = groups_of(tokens);
        engine::test::require(groups.size() == 3, "merged: expected three groups");
        const auto labeled = label_ranges(
            groups, text_words("Hello was a sun"),
            table({{"Hello", "helo"}, {"was", "wʌz"}, {"a", "ə"}, {"sun", "sʌn"}}));
        engine::test::require(labels_of(labeled) == std::vector<std::string>({"Hello", "was", "a", "sun"}),
                              "merged: labels");
        const auto & was = labeled[1].frames;
        const auto & a = labeled[2].frames;
        engine::test::require(was.start == groups[1].frames.start, "merged: was starts at group");
        engine::test::require(was.end == a.start, "merged: was ends where a starts");
        engine::test::require(a.end == groups[1].frames.end, "merged: a ends at group end");
        std::cout << "PASS merged groups split by phonemes\n";
    }

    // Split word: "$5" is pronounced as two groups ("five dollars"). It gets
    // one span covering both.
    {
        Tokens tokens(symbols_of("pe_fIv_dɑləɹz_naʊ"));
        const auto groups = groups_of(tokens);
        engine::test::require(groups.size() == 4, "split: expected four groups");
        const auto labeled = label_ranges(
            groups, text_words("pay $5 now"),
            table({{"pay", "pe"}, {"$5", "fIvdɑləɹz"}, {"now", "naʊ"}}));
        engine::test::require(labels_of(labeled) == std::vector<std::string>({"pay", "$5", "now"}),
                              "split: labels");
        engine::test::require(labeled[1].frames.start == groups[1].frames.start,
                              "split: $5 starts at its first group");
        engine::test::require(labeled[1].frames.end == groups[2].frames.end,
                              "split: $5 ends at its last group");
        std::cout << "PASS split word spans two groups\n";
    }

    // Poor match: nothing agrees, so one span labeled with the whole text.
    {
        Tokens tokens(symbols_of("helo_wʌzə_sʌn"));
        const auto groups = groups_of(tokens);
        const auto labeled = label_ranges(
            groups, text_words("Hello was a sun today"),
            table({{"Hello", "xxxxx"}, {"was", "yyy"}, {"a", "zz"}, {"sun", "qqq"}, {"today", "wwwww"}}));
        engine::test::require(labeled.size() == 1, "poor: expected one span");
        engine::test::require(labeled[0].label == "Hello was a sun today", "poor: joined label");
        std::cout << "PASS poor match falls back to one span\n";
    }

    // Nothing spoken, nothing emitted.
    expect_labels("only punctuation", "...", {".", ","}, {});

    // Unknown ids (specials) end a word and produce no label.
    {
        Tokens tokens({"h", "i"});
        tokens.ids.insert(tokens.ids.begin() + 1, 0);
        tokens.frames.insert(tokens.frames.begin() + 1, 2);
        engine::test::require(groups_of(tokens).size() == 2, "special id: expected two groups");
        std::cout << "PASS special id splits groups\n";
    }

    // Frame ranges follow the duration model: each token covers its own frames.
    {
        Tokens tokens({"a", " ", "b"});
        tokens.frames = {3, 5, 4};
        const auto groups = groups_of(tokens);
        engine::test::require(groups.size() == 2, "frames: expected two groups");
        engine::test::require(groups[0].frames.start == 0 && groups[0].frames.end == 3,
                              "frames: first group");
        engine::test::require(groups[1].frames.start == 8 && groups[1].frames.end == 12,
                              "frames: second group");
        std::cout << "PASS frame ranges\n";
    }

    std::cout << "ALL PASSED\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "FAILED: " << error.what() << "\n";
    return 1;
}
