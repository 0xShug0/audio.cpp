// The multilingual Granite tokenizer is a SentencePiece-style Sequence decoder:
// replace ▁ with a space, then expand <0xHH> byte-fallback pieces. The English
// Granite tokenizer is a top-level ByteLevel decoder and must keep decoding Ġ
// as a leading space. A leading ▁ stays a leading space so word grouping can
// see the boundary; the session trims the finished transcript.

#include "engine/framework/tokenizers/hf_tokenizer_json.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::filesystem::path write_tokenizer(const std::string & json) {
    static int counter = 0;
    const auto path = std::filesystem::temp_directory_path() /
                      ("hf_sequence_decode_" + std::to_string(counter++) + ".json");
    std::ofstream out(path);
    require(out.good(), "failed to open temp tokenizer file");
    out << json;
    require(out.good(), "failed to write temp tokenizer file");
    return path;
}

void test_sequence_metaspace_and_byte_fallback() {
    const auto path = write_tokenizer(R"({
        "model": {"vocab": {"<unk>": 0, "▁hello": 1, "world": 2, "<0x61>": 3, "▁world": 4}},
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
                {"type": "Strip", "content": " ", "start": 1, "stop": 0}
            ]
        }
    })");
    const auto tokenizer = engine::tokenizers::load_huggingface_tokenizer_json(path);
    std::filesystem::remove(path);
    require(tokenizer->decode_ids({1}) == " hello",
            "a leading ▁ stays a leading space, got: [" + tokenizer->decode_ids({1}) + "]");
    require(tokenizer->decode_ids({1, 2}) == " helloworld",
            "a piece without ▁ continues the word, got: [" + tokenizer->decode_ids({1, 2}) + "]");
    require(tokenizer->decode_ids({1, 4}) == " hello world",
            "a ▁ piece starts a new word, got: [" + tokenizer->decode_ids({1, 4}) + "]");
    require(tokenizer->decode_ids({3}) == "a",
            "byte fallback <0x61> decodes to a, got: [" + tokenizer->decode_ids({3}) + "]");
    require(tokenizer->decode_ids({1, 3}) == " helloa",
            "byte fallback joins the current word, got: [" + tokenizer->decode_ids({1, 3}) + "]");
}

void test_byte_level_decoder_is_unchanged() {
    const auto path = write_tokenizer(
        "{\"model\":{\"vocab\":{\"<|blank|>\":0,\"\\u0120hello\":1,\"world\":2}},"
        "\"decoder\":{\"type\":\"ByteLevel\",\"add_prefix_space\":true,\"trim_offsets\":true}}");
    const auto tokenizer = engine::tokenizers::load_huggingface_tokenizer_json(path);
    std::filesystem::remove(path);
    require(tokenizer->decode_ids({1, 2}) == " helloworld",
            "ByteLevel Ġ still decodes as a space, got: [" + tokenizer->decode_ids({1, 2}) + "]");
}

}  // namespace

int main() {
    try {
        test_sequence_metaspace_and_byte_fallback();
        test_byte_level_decoder_is_unchanged();
    } catch (const std::exception & error) {
        std::cerr << "hf tokenizer sequence decode test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "hf tokenizer sequence decode test passed\n";
    return 0;
}
