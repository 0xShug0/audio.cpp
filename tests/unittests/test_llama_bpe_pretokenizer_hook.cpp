#include "engine/framework/tokenizers/llama_bpe.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::tokenizers::LlamaBpeGgufVocabulary;
using engine::tokenizers::LlamaBpeTokenizer;
using engine::tokenizers::LlamaBpeTokenizerSpec;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

LlamaBpeTokenizerSpec minimal_spec() {
    // ByteLevel maps the two UTF-8 bytes of e-acute to U+00C3, U+00A9.
    // This vocabulary is entirely self-generated and requires no model files.
    const std::string first_byte = "\xC3\x83";
    const std::string second_byte = "\xC2\xA9";
    LlamaBpeGgufVocabulary vocabulary;
    vocabulary.tokens = {"a", "b", "c", "ab", "bc", first_byte, second_byte,
        first_byte + second_byte, "<SPECIAL>"};
    vocabulary.token_types = {1, 1, 1, 1, 1, 1, 1, 1, 3};
    vocabulary.merges = {"a b", "b c", first_byte + " " + second_byte};
    LlamaBpeTokenizerSpec spec;
    spec.gguf_vocabulary = std::move(vocabulary);
    return spec;
}

void test_hook_is_opt_in_and_instance_local() {
    const LlamaBpeTokenizer default_tokenizer(minimal_spec());
    auto empty_spec = minimal_spec();
    empty_spec.pre_tokenize = {};
    const LlamaBpeTokenizer empty_hook_tokenizer(empty_spec);
    const std::vector<int32_t> default_ids{3, 2};
    require(default_tokenizer.encode("abc", false) == default_ids, "default BPE merge order");
    require(empty_hook_tokenizer.encode("abc", false) == default_ids, "empty hook must keep default splitting");

    auto custom_spec = minimal_spec();
    custom_spec.pre_tokenize = [](const std::string & text) {
        return std::vector<std::string>{text.substr(0, 1), text.substr(1)};
    };
    const LlamaBpeTokenizer custom(custom_spec);
    require(custom.encode("abc", false) == std::vector<int32_t>{0, 4}, "opted-in hook controls merge boundaries");
    require(custom.decode({0, 4}) == "abc", "custom pre-tokenization preserves decoded bytes");
    require(default_tokenizer.encode("abc", false) == default_ids, "another instance must retain default splitting");
    require(empty_hook_tokenizer.encode("abc", false) == default_ids, "explicit empty hook remains unchanged");
}

void test_utf8_bytes_and_special_token_fragments() {
    const std::string acute = "\xC3\xA9";
    const LlamaBpeTokenizer baseline(minimal_spec());
    require(baseline.encode(acute, false) == std::vector<int32_t>{7}, "baseline merges UTF-8 byte pieces");

    auto split_bytes_spec = minimal_spec();
    split_bytes_spec.pre_tokenize = [](const std::string & text) {
        return std::vector<std::string>{text.substr(0, 1), text.substr(1)};
    };
    const LlamaBpeTokenizer split_bytes(split_bytes_spec);
    const auto split_ids = split_bytes.encode(acute, false);
    require(split_ids == std::vector<int32_t>{5, 6}, "hook fragments are byte encoded exactly once");
    require(split_bytes.decode(split_ids) == acute, "split UTF-8 bytes must concatenate without corruption");

    auto fragments = std::make_shared<std::vector<std::string>>();
    auto fragment_spec = minimal_spec();
    fragment_spec.pre_tokenize = [fragments](const std::string & text) {
        fragments->push_back(text);
        return std::vector<std::string>{text};
    };
    const LlamaBpeTokenizer tokenizer(fragment_spec);
    const auto ids = tokenizer.encode(acute + "<SPECIAL>ab", true);
    require(ids == std::vector<int32_t>{7, 8, 3}, "added token matching must precede the hook");
    require(*fragments == std::vector<std::string>{acute, "ab"}, "hook sees only raw UTF-8 text fragments");
    require(tokenizer.decode(ids) == acute + "<SPECIAL>ab", "special-token boundaries preserve decoded bytes");
    require(tokenizer.decode(ids, true) == acute + "ab", "special skipping remains independent of the hook");
}

}  // namespace

int main() {
    try {
        test_hook_is_opt_in_and_instance_local();
        test_utf8_bytes_and_special_token_fragments();
    } catch (const std::exception & error) {
        std::cerr << "llama BPE pretokenizer hook test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "llama BPE pretokenizer hook test passed\n";
    return 0;
}
