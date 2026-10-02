#include "engine/community_models/lfm2_audio/tokenizer.h"
#include "test_assert.h"

#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using engine::community_models::lfm2_audio::Lfm2TextTokenizer;
using engine::community_models::lfm2_audio::Lfm2TextVocabulary;
using engine::test::require;
using engine::test::require_eq;

// tokenizer.ggml.token_type values.
constexpr int32_t kNormal = 1;
constexpr int32_t kControl = 3;
constexpr int32_t kUserDefined = 4;
constexpr int32_t kUnused = 5;

// A byte-level vocabulary in the GGUF layout: "Ġ" is the space byte and "Ċ"
// the newline byte.
Lfm2TextVocabulary vocabulary() {
    Lfm2TextVocabulary vocab;
    vocab.pre_tokenizer = "lfm2";
    const std::pair<const char *, int32_t> tokens[] = {
        {"<|startoftext|>", kControl},  // 0
        {"<|im_start|>", kControl},     // 1
        {"<|im_end|>", kControl},       // 2
        {"python", kUserDefined},       // 3
        {"h", kNormal},                 // 4
        {"e", kNormal},                 // 5
        {"l", kNormal},                 // 6
        {"o", kNormal},                 // 7
        {"Ġ", kNormal},                 // 8
        {"w", kNormal},                 // 9
        {"r", kNormal},                 // 10
        {"d", kNormal},                 // 11
        {"u", kNormal},                 // 12
        {"s", kNormal},                 // 13
        {"Ċ", kNormal},                 // 14
        {"he", kNormal},                // 15
        {"ll", kNormal},                // 16
        {"hell", kNormal},              // 17
        {"hello", kNormal},             // 18
        {"Ġw", kNormal},                // 19
        {"or", kNormal},                // 20
        {"Ġwor", kNormal},              // 21
        {"ld", kNormal},                // 22
        {"Ġworld", kNormal},            // 23
        {"1", kNormal},                 // 24
        {"2", kNormal},                 // 25
        {"3", kNormal},                 // 26
        {"4", kNormal},                 // 27
        {"5", kNormal},                 // 28
        {"12", kNormal},                // 29
        {"34", kNormal},                // 30
        {"123", kNormal},               // 31
        {"45", kNormal},                // 32
        {"I", kNormal},                 // 33
        {"'", kNormal},                 // 34
        {"T", kNormal},                 // 35
        {"'T", kNormal},                // 36
        {"[PAD37]", kUnused},           // 37, converter filler past the trained vocabulary
    };
    for (const auto & [text, type] : tokens) {
        vocab.tokens.emplace_back(text);
        vocab.token_types.push_back(type);
    }

    vocab.merges = {
        "h e", "l l", "he ll", "hell o", "Ġ w", "o r", "Ġw or", "l d", "Ġwor ld",
        // "3 4" outranks "12 3", so only a split into "123" and "45" can
        // produce "123".
        "1 2", "3 4", "4 5", "12 3",
        "' T",
    };
    return vocab;
}

std::string ids(const std::vector<int32_t> & values) {
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); ++i) {
        out << (i == 0 ? "" : ",") << values[i];
    }

    return out.str();
}

void require_ids(const std::vector<int32_t> & actual, const std::vector<int32_t> & expected, const std::string & label) {
    require_eq(ids(actual), ids(expected), label);
}

void test_encode() {
    const Lfm2TextTokenizer tokenizer(vocabulary());
    require_ids(tokenizer.encode("hello world"), {18, 23}, "hello world");
    require_ids(tokenizer.encode("<|im_start|>user\n"), {1, 12, 13, 5, 10, 14}, "control token prefix");
    require_ids(tokenizer.encode("hello<|im_end|>"), {18, 2}, "control token suffix");
    // User-defined tokens are matched whole too, as in the HF tokenizer.
    require_ids(tokenizer.encode("hello python"), {18, 8, 3}, "user-defined token");
    require(tokenizer.encode("").empty(), "empty text must encode to no tokens");
}

// llama.cpp tokenizes pre = "lfm2" with its Llama 3 split: digits go in groups
// of up to three, and contractions match in any case. GPT-2's split would give
// 12,34,5 and I,',T here.
void test_llama3_split() {
    const Lfm2TextTokenizer tokenizer(vocabulary());
    require_ids(tokenizer.encode("12345"), {31, 32}, "digit groups");
    require_ids(tokenizer.encode("I'T"), {33, 36}, "uppercase contraction");
}

void test_decode() {
    const Lfm2TextTokenizer tokenizer(vocabulary());
    require_eq(tokenizer.decode({18, 23}), std::string("hello world"), "byte-level decode");
    require_eq(tokenizer.decode({0, 1, 18, 2}), std::string("hello"), "control tokens are dropped");
    require_eq(tokenizer.decode({18, 8, 3}), std::string("hello python"), "user-defined tokens are kept");
    require_eq(tokenizer.decode({12, 13, 5, 10, 14}), std::string("user\n"), "newline byte");
    require_eq(tokenizer.decode({18, 37}), std::string("hello"), "unused tokens are dropped");
}

void test_token_lookup() {
    const Lfm2TextTokenizer tokenizer(vocabulary());
    require_eq(tokenizer.require_token_id("<|im_end|>"), 2, "<|im_end|> id");
    require(tokenizer.is_control_token(2), "<|im_end|> is a control token");
    require(!tokenizer.is_control_token(3), "python is not a control token");
    require(!tokenizer.is_control_token(18), "hello is not a control token");
    require(!tokenizer.is_control_token(1000), "an unknown id is not a control token");
}

template <typename Fn>
void require_throws(Fn && fn, const std::string & label) {
    bool threw = false;
    try {
        fn();
    } catch (const std::runtime_error &) {
        threw = true;
    }

    require(threw, label + " must throw");
}

void test_rejects_bad_vocabulary() {
    const Lfm2TextTokenizer tokenizer(vocabulary());
    require_throws([&] { (void)tokenizer.require_token_id("<|audio_start|>"); }, "a missing token");
    require_throws([&] { (void)tokenizer.decode({1000}); }, "decoding an unknown id");

    auto other_pre = vocabulary();
    other_pre.pre_tokenizer = "llama-bpe";
    require_throws([&] { (void)Lfm2TextTokenizer{other_pre}; }, "another pre-tokenizer");

    auto no_pre = vocabulary();
    no_pre.pre_tokenizer.clear();
    require_throws([&] { (void)Lfm2TextTokenizer{no_pre}; }, "a missing pre-tokenizer");

    auto bad_merge = vocabulary();
    bad_merge.merges.emplace_back("hello");
    require_throws([&] { (void)Lfm2TextTokenizer{bad_merge}; }, "a merge without a separator");
}

}  // namespace

int main() {
    try {
        test_encode();
        test_llama3_split();
        test_decode();
        test_token_lookup();
        test_rejects_bad_vocabulary();
        std::cout << "lfm2_audio_tokenizer_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "lfm2_audio_tokenizer_test: " << error.what() << '\n';
        return 1;
    }
}
