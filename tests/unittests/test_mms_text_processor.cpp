#include "engine/community_models/mms_forced_aligner/text_processor.h"
#include "test_assert.h"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using engine::test::require;
using engine::test::require_eq;

engine::community_models::mms_forced_aligner::MmsVocabulary reference_vocab() {
    engine::community_models::mms_forced_aligner::MmsVocabulary vocab;
    const char * tokens[] = {"<blank>", "<pad>", "</s>", "<unk>", "a", "i", "e", "n", "o", "u",
                             "t", "s", "r", "m", "k", "l", "d", "g", "h", "y", "b", "p", "w", "c",
                             "v", "j", "z", "f", "'", "q", "x"};
    for (int32_t id = 0; id < 31; ++id) {
        vocab.token_to_id.emplace(tokens[id], id);
        vocab.id_to_token.push_back(tokens[id]);
    }
    vocab.blank_id = 0;
    return vocab;
}

using engine::community_models::mms_forced_aligner::MmsStarFrequency;
using engine::community_models::mms_forced_aligner::MmsTextNormalization;
using engine::community_models::mms_forced_aligner::MmsTextProcessorOptions;
using engine::community_models::mms_forced_aligner::prepare_mms_text;

const MmsTextProcessorOptions kLatinSegment{MmsTextNormalization::Latin, MmsStarFrequency::Segment};

void test_language_canonicalization() {
    require_eq(
        engine::community_models::mms_forced_aligner::mms_canonical_language("nl"),
        std::string("nld"),
        "nl");
    require_eq(
        engine::community_models::mms_forced_aligner::mms_canonical_language("nld"),
        std::string("nld"),
        "nld");
    require_eq(
        engine::community_models::mms_forced_aligner::mms_canonical_language("en"),
        std::string("eng"),
        "en");
    require_eq(
        engine::community_models::mms_forced_aligner::mms_canonical_language("eng"),
        std::string("eng"),
        "eng");
    bool threw = false;
    try {
        (void) engine::community_models::mms_forced_aligner::mms_canonical_language("de");
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "unsupported language must be rejected");
    threw = false;
    try {
        (void) engine::community_models::mms_forced_aligner::mms_canonical_language("");
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "empty language must be rejected");
}

void test_dutch_punctuation_and_case() {
    const auto prepared = prepare_mms_text(reference_vocab(), "De groep keert terug naar Reverdin.", "nld", kLatinSegment);
    require_eq(static_cast<int64_t>(prepared.original_words.size()), int64_t{6}, "word count");
    require_eq(prepared.original_words[0], std::string("De"), "original word preserved");
    require_eq(prepared.normalized_words[0], std::string("de"), "normalized word");
    require_eq(prepared.normalized_words[5], std::string("reverdin"), "trailing punct dropped");
    require_eq(prepared.target_to_word[0], int32_t{-1}, "leading star");
    require_eq(prepared.target_ids[0], int32_t{31}, "star id");
    require_eq(prepared.target_ids[1], int32_t{16}, "d id");
    require_eq(prepared.target_ids[2], int32_t{6}, "e id");
    // "De" -> star, d, e ; target_to_word of d/e is 0
    require_eq(prepared.target_to_word[1], int32_t{0}, "d maps to word 0");
    // star before every word
    size_t star_count = 0;
    for (const int32_t id : prepared.target_ids) {
        if (id == 31) {
            ++star_count;
        }
    }
    require_eq(static_cast<int64_t>(star_count), int64_t{6}, "one star per word");
    require_eq(prepared.canonical_language, std::string("nld"), "canonical language");
}

void test_diacritics() {
    const auto prepared = prepare_mms_text(reference_vocab(), "café déjà-vu", "en", kLatinSegment);
    require_eq(prepared.original_words.size(), size_t{2}, "two words");
    require_eq(prepared.normalized_words[0], std::string("cafe"), "e acute decomposed");
    require_eq(prepared.normalized_words[1], std::string("dejavu"), "hyphen is a separator");
}

void test_apostrophe() {
    const auto prepared = prepare_mms_text(reference_vocab(), "l'été", "en", kLatinSegment);
    require_eq(prepared.normalized_words[0], std::string("l'ete"), "apostrophe kept");
    const auto star = prepared.target_ids[0];
    require_eq(star, int32_t{31}, "star first");
    require_eq(prepared.target_ids[1], int32_t{15}, "l id");
    require_eq(prepared.target_ids[2], int32_t{28}, "apostrophe id");
}

void test_numbers() {
    const auto prepared = prepare_mms_text(reference_vocab(), "4.5 billion years, 22 thousand", "en", kLatinSegment);
    require_eq(prepared.original_words.size(), size_t{3}, "digit-only words dropped");
    require_eq(prepared.original_words[0], std::string("billion"), "first surviving word");
    require_eq(prepared.normalized_words[0], std::string("billion"), "billion");
    require_eq(prepared.original_words[1], std::string("years,"), "original punctuation preserved");
    require_eq(prepared.normalized_words[1], std::string("years"), "years");
    require_eq(prepared.normalized_words[2], std::string("thousand"), "thousand");
}

void test_pre_romanized() {
    const MmsTextProcessorOptions options{MmsTextNormalization::PreRomanized, MmsStarFrequency::Segment};
    const auto prepared = prepare_mms_text(reference_vocab(), "Sayonara sekai", "ja", options);
    require_eq(prepared.original_words.size(), size_t{2}, "romanized words");
    require_eq(prepared.normalized_words[0], std::string("sayonara"), "lowercased");
    require_eq(prepared.canonical_language, std::string("ja"), "pre-romanized language passthrough");
    bool threw = false;
    try {
        (void) prepare_mms_text(reference_vocab(), "sayonara", "ja", kLatinSegment);
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "latin mode rejects unsupported languages");
    threw = false;
    try {
        const MmsTextProcessorOptions empty_language{MmsTextNormalization::PreRomanized, MmsStarFrequency::Segment};
        (void) prepare_mms_text(reference_vocab(), "sayonara", "", empty_language);
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "pre_romanized mode rejects empty language");
}

void test_non_latin_rejected() {
    bool threw = false;
    try {
        (void) prepare_mms_text(reference_vocab(), "Привет мир", "en", kLatinSegment);
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "Cyrillic must be rejected in latin mode");
    threw = false;
    try {
        const MmsTextProcessorOptions options{MmsTextNormalization::PreRomanized, MmsStarFrequency::Segment};
        (void) prepare_mms_text(reference_vocab(), "Привет", "en", options);
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "Cyrillic must be rejected in pre_romanized mode");
}

void test_empty_and_punctuation_only() {
    bool threw = false;
    try {
        (void) prepare_mms_text(reference_vocab(), "   ", "en", kLatinSegment);
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "empty transcript must be rejected");
    threw = false;
    try {
        (void) prepare_mms_text(reference_vocab(), "!!! --- ...", "en", kLatinSegment);
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "punctuation-only transcript must be rejected");
    threw = false;
    try {
        (void) prepare_mms_text(reference_vocab(), "42 7 2024", "en", kLatinSegment);
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "digit-only transcript must be rejected");
}

void test_edges_star_placement() {
    const MmsTextProcessorOptions options{MmsTextNormalization::Latin, MmsStarFrequency::Edges};
    const auto prepared = prepare_mms_text(reference_vocab(), "hello world", "en", options);
    require_eq(prepared.target_ids[0], int32_t{31}, "leading star");
    require_eq(prepared.target_to_word.back(), int32_t{-1}, "trailing star");
    const int32_t star = static_cast<int32_t>(prepared.target_ids.size()) - 1;
    require_eq(prepared.target_ids[star], int32_t{31}, "trailing star id");
    size_t star_count = 0;
    for (const int32_t id : prepared.target_ids) {
        if (id == 31) {
            ++star_count;
        }
    }
    require_eq(static_cast<int64_t>(star_count), int64_t{2}, "exactly two edge stars");
}

}  // namespace

int main() {
    try {
        test_language_canonicalization();
        test_dutch_punctuation_and_case();
        test_diacritics();
        test_apostrophe();
        test_numbers();
        test_pre_romanized();
        test_non_latin_rejected();
        test_empty_and_punctuation_only();
        test_edges_star_placement();
        std::cout << "mms_text_processor_test passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "mms_text_processor_test: %s\n", error.what());
        return 1;
    }
}
